// =============================================================================
// gff_synth.sv
//
// Synthesizable GFF (Global Feature Fusion) Module.
// Replaces behavioral stubs with proper synthesizable implementations:
//   - AvgPool: Hybrid tree-fold (log2(N/G)+2 cycles instead of N cycles)
//   - Project branch: Fused pipeline (Linear→LN→Linear→GELU→Linear) with
//     skid buffers to eliminate inter-layer overhead
//   - l2norm: Newton-Raphson based (l2norm_nr), L2_ITER=3 (default)
//   - GELU: ROM-based piecewise-linear interpolation (gelu_rom)
//   - LayerNorm: Pipelined with LUT-based operations (layerNorm_fixed)
//   - Linear: ROM-based weights (linear_layer)
//
// Architecture:
//   Input (N × 4C)
//     ├─► AvgPool3D (tree-fold) ─► gff_fused_proj ─► l2norm ─► UnPooling
//     │   (log2(N/G)+2 cy)        (8C+27 cy)         (9 cy)     (1 cy)
//     │                                                                │
//     └─► l2norm ──────────────────────────────────────────────────── Cat
//                                                                      │
//                                                               Linear (2C → C)
//                                                                      │
//                                                                    GELU
//                                                                      │
//                                                                   Output
//
// BUG FIX (GFF-AVGPOOL-001): AvgPool now uses tree-fold reduction
// instead of sequential accumulation that had an early-termination bug.
//
// Artix-7 100T Optimization:
//   - USE_DSP defaults to 0 (LUT-based) for non-critical paths
//   - Saves DSP slices for systolic array MAC operations
//   - GELU uses ROM (BRAM) + LUT multiplier
//   - LayerNorm uses LUT-based operations
//   - l2norm uses Newton-Raphson iterations (3 iterations sufficient for 16-bit)
//   - AvgPool uses tree-fold reduction (log2(N/GROUP)+2 cycles vs N cycles)
//
// Latency (with defaults, C=32):
//   - AvgPool: log2(N/GROUP_SIZE) + 2 cycles (e.g., 8 for N=1024, GROUP=16)
//   - Fused Project (gff_fused_proj): 8C + 27 cycles (283 for C=32)
//   - l2norm (proj branch): 3 + 2*L2_ITER cycles (9 with L2_ITER=3)
//   - UnPool: 1 cycle
//   - l2norm (input branch): 3 + 2*L2_ITER cycles (9 with L2_ITER=3)
//   - Cat: 1 cycle (with sync buffer)
//   - Linear 2C→C: IN_CH + 3 cycles (67 for C=32)
//   - GELU (output): 4 cycles
// =============================================================================
// =============================================================================

`timescale 1ns / 1ps

import pkg_config::*;
import pkg_module_config::*;

module gff_synth #(
    parameter int C           = pkg_config::MAX_C,
    parameter int N           = 1024,
    parameter int DATA_W = pkg_module_config::GFF_DATA_WIDTH,
    parameter int FRAC_W      = pkg_module_config::GFF_DATA_WIDTH / 2,
    parameter int ACC_W       = pkg_config::ACC_WIDTH,
    parameter int L2_ITER     = pkg_config::L2_ITER,
    parameter bit  USE_DSP    = 0,
    parameter bit  GELU_DSP   = 0,
    parameter int AVG_GROUP_SIZE = pkg_config::AVG_GROUP_SIZE,
    parameter int TIME_MUX    = pkg_config::TIME_MUX,
    parameter string ROM_DIR = "rtl/mem" // ROM file directory
)(
    input  logic clk,
    input  logic rst_n,

    // Input feature map: N points × 4C channels
    input  logic signed [DATA_W-1:0] feat_in   [0:N-1][0:4*C-1],
    input  logic                      feat_valid,

    // Output feature map: N points × C channels
    output logic signed [DATA_W-1:0] feat_out  [0:N-1][0:C-1],
    output logic                      feat_out_valid,

    // LayerNorm learned parameters (runtime configurable)
    input  logic signed [DATA_W-1:0] proj_ln_gamma [0:3*C-1],
    input  logic signed [DATA_W-1:0] proj_ln_beta  [0:3*C-1],

    // FIXED: P0-1 — Weight loading ports for runtime weight updates
    input  logic                      weight_valid,
    input  logic [31:0]               weight_addr,
    input  logic signed [DATA_W-1:0]  weight_data
);

    // =========================================================================
    // Internal wires between pipeline stages
    // =========================================================================

    // After AvgPool3D+Unroll: 1 × 4C  (global vector, broadcast to N)
    logic signed [DATA_W-1:0] pooled      [0:4*C-1];
    logic                     pooled_valid;

    // After Fused Project Pipeline (AvgPool → Linear1 → LN → Linear2 → GELU → Linear3)
    logic signed [DATA_W-1:0] proj_out    [0:C-1];
    logic                     proj_out_v;

    // After l2norm on projected branch (C)
    logic signed [DATA_W-1:0] proj_l2norm [0:C-1];
    logic                     proj_l2norm_v;

    // After UnPooling: broadcast back to N × C
    logic signed [DATA_W-1:0] unpooled    [0:N-1][0:C-1];
    logic                     unpooled_v;

    // l2norm on the *input* branch (N × 4C → N × C using first C channels)
    logic signed [DATA_W-1:0] in_l2norm   [0:N-1][0:C-1];
    logic                     in_l2norm_v;

    // After Cat (N × 2C)
    logic signed [DATA_W-1:0] cat_out     [0:N-1][0:2*C-1];
    logic                     cat_v;

    // After Cat Linear (2C → C)
    logic signed [DATA_W-1:0] cat_lin     [0:N-1][0:C-1];
    logic                     cat_lin_v;

    // =========================================================================
    // Stage 0: AvgPool3D — Tree-Fold Reduction
    // =========================================================================
    avgpool3d_unroll #(
        .C               (C),
        .N               (N),
        .DWIDTH          (DATA_W),
        .TREE_GROUP_SIZE (AVG_GROUP_SIZE)
    ) u_avgpool (
        .clk        (clk),
        .rst_n      (rst_n),
        .feat_in    (feat_in),
        .valid_in   (feat_valid),
        .feat_out   (pooled),
        .valid_out  (pooled_valid)
    );

    // =========================================================================
    // Stages 1-5: Fused Project Pipeline
    //   Linear1(4C→3C) → LayerNorm(3C) → Linear2(3C→C) → GELU(C) → Linear3(C→C)
    //   Connected with 1-cycle skid buffers instead of 3-cycle handshake overhead.
    //   Saves ~12 cycles compared to separate instantiations.
    // =========================================================================
    gff_fused_proj #(
        .C       (C),
        .DATA_W  (DATA_W),
        .FRAC_W  (FRAC_W),
        .ACC_W   (ACC_W),
        .USE_DSP (USE_DSP),
        .GELU_DSP(GELU_DSP),
        .ROM_DIR (ROM_DIR)
    ) u_fused_proj (
        .clk           (clk),
        .rst_n         (rst_n),
        .pooled        (pooled),
        .pooled_valid  (pooled_valid),
        .proj_out      (proj_out),
        .proj_out_v    (proj_out_v),
        .ln_gamma      (proj_ln_gamma),
        .ln_beta       (proj_ln_beta),
        .weight_valid  (weight_valid),
        .weight_addr   (weight_addr),
        .weight_data   (weight_data)
    );

    // =========================================================================
    // Stage 6: l2norm on projected global vector (Newton-Raphson)
    // =========================================================================
    // Pack/unpack for l2norm_nr (packed port — Verilator compatible)
    logic signed [C*DATA_W-1:0] proj_out_flat, proj_l2norm_flat;
    always_comb begin
        for (int c = 0; c < C; c++)
            proj_out_flat[c*DATA_W +: DATA_W] = proj_out[c];
    end
    always_comb begin
        for (int c = 0; c < C; c++)
            proj_l2norm[c] = proj_l2norm_flat[c*DATA_W +: DATA_W];
    end

    l2norm_nr #(
        .CH         (C),
        .DATA_W     (DATA_W),
        .FRAC_W     (FRAC_W),
        .ITERATIONS (L2_ITER),
        .ACC_W      (ACC_W),
        .USE_DSP    (USE_DSP)
    ) u_proj_l2norm (
        .clk          (clk),
        .rst_n        (rst_n),
        .data_in_flat (proj_out_flat),
        .valid_in     (proj_out_v),
        .ready_out    (),            // free-running here
        .data_out_flat(proj_l2norm_flat),
        .valid_out    (proj_l2norm_v),
        .out_ready    (1'b1)         // FIXED: was unconnected (floating) — tie high
    );

    // =========================================================================
    // Stage 7: UnPooling — broadcast global C-dim vector to N × C
    // =========================================================================
    unpooling #(
        .C      (C),
        .N      (N),
        .DWIDTH (DATA_W)
    ) u_unpool (
        .clk        (clk),
        .rst_n      (rst_n),
        .data_in    (proj_l2norm),
        .valid_in   (proj_l2norm_v),
        .data_out   (unpooled),
        .valid_out  (unpooled_v)
    );

    // =========================================================================
    // Parallel Branch: l2norm on raw input (N × 4C → first C channels)
    // TIME_MUX=1: full parallel (N instances), TIME_MUX>1: time-multiplexed
    // =========================================================================
    generate
        if (TIME_MUX > 1) begin : gen_in_l2norm_tm
            l2norm_nd_nr_tm #(
                .N        (N),
                .CH       (C),
                .DWIDTH   (DATA_W),
                .FRAC_W   (FRAC_W),
                .ITER     (L2_ITER),
                .USE_DSP  (USE_DSP),
                .TIME_MUX (TIME_MUX)
            ) u_in_l2norm (
                .clk        (clk),
                .rst_n      (rst_n),
                .feat_in    (feat_in),
                .valid_in   (feat_valid),
                .feat_out   (in_l2norm),
                .valid_out  (in_l2norm_v)
            );
        end else begin : gen_in_l2norm_full
            l2norm_nd_nr #(
                .N      (N),
                .CH     (C),
                .DWIDTH (DATA_W),
                .FRAC_W (FRAC_W),
                .ITER   (L2_ITER),
                .USE_DSP(USE_DSP)
            ) u_in_l2norm (
                .clk        (clk),
                .rst_n      (rst_n),
                .feat_in    (feat_in),
                .valid_in   (feat_valid),
                .feat_out   (in_l2norm),
                .valid_out  (in_l2norm_v)
            );
        end
    endgenerate

    // =========================================================================
    // Stage 8: Cat Layer — concatenate unpooled (N×C) and in_l2norm (N×C)
    // =========================================================================
    cat_layer #(
        .N      (N),
        .C      (C),
        .DWIDTH (DATA_W)
    ) u_cat (
        .clk         (clk),
        .rst_n       (rst_n),
        .branch_a    (unpooled),
        .valid_a     (unpooled_v),
        .branch_b    (in_l2norm),
        .valid_b     (in_l2norm_v),
        .cat_out     (cat_out),
        .valid_out   (cat_v)
    );

    // =========================================================================
    // Stage 9: Linear 2C → C (per point)
    // TIME_MUX=1: full parallel (N FSMs), TIME_MUX>1: time-multiplexed
    // =========================================================================
    generate
        if (TIME_MUX > 1) begin : gen_cat_lin_tm
            linear_nd_tm #(
                .N            (N),
                .IN_CH        (2*C),
                .OUT_CH       (C),
                .DWIDTH       (DATA_W),
                .STUB_BEHAVIOR(1'b0),
                .WEIGHT_FILE  ({ROM_DIR, "/gff_cat_weight.mem"}),
                .BIAS_FILE    ({ROM_DIR, "/gff_cat_bias.mem"}),
                .TIME_MUX    (TIME_MUX)
            ) u_cat_lin (
                .clk        (clk),
                .rst_n      (rst_n),
                .feat_in    (cat_out),
                .valid_in   (cat_v),
                .feat_out   (cat_lin),
                .valid_out  (cat_lin_v)
            );
        end else begin : gen_cat_lin_full
            linear_nd #(
                .N            (N),
                .IN_CH        (2*C),
                .OUT_CH       (C),
                .DWIDTH       (DATA_W),
                .STUB_BEHAVIOR(1'b0),
                .WEIGHT_FILE  ({ROM_DIR, "/gff_cat_weight.mem"}),
                .BIAS_FILE    ({ROM_DIR, "/gff_cat_bias.mem"})
            ) u_cat_lin (
                .clk        (clk),
                .rst_n      (rst_n),
                .feat_in    (cat_out),
                .valid_in   (cat_v),
                .feat_out   (cat_lin),
                .valid_out  (cat_lin_v)
            );
        end
    endgenerate

    // =========================================================================
    // Stage 10: GELU (output activation, N × C) - ROM-based
    // TIME_MUX=1: full parallel (N*CH instances), TIME_MUX>1: time-multiplexed
    // =========================================================================
    generate
        if (TIME_MUX > 1) begin : gen_out_gelu_tm
            gelu_nd_rom_tm #(
                .N          (N),
                .CH         (C),
                .DATA_W     (DATA_W),
                .FRAC_W     (FRAC_W),
                .USE_DSP    (GELU_DSP),
                .ROM_DIR    (ROM_DIR),
                .TIME_MUX   (TIME_MUX)
            ) u_out_gelu (
                .clk        (clk),
                .rst_n      (rst_n),
                .feat_in    (cat_lin),
                .valid_in   (cat_lin_v),
                .feat_out   (feat_out),
                .valid_out  (feat_out_valid)
            );
        end else begin : gen_out_gelu_full
            gelu_nd_rom #(
                .N          (N),
                .CH         (C),
                .DATA_W     (DATA_W),
                .FRAC_W     (FRAC_W),
                .USE_DSP    (GELU_DSP),
                .ROM_DIR    (ROM_DIR)
            ) u_out_gelu (
                .clk        (clk),
                .rst_n      (rst_n),
                .feat_in    (cat_lin),
                .valid_in   (cat_lin_v),
                .feat_out   (feat_out),
                .valid_out  (feat_out_valid)
            );
        end
    endgenerate

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    GFF_CHANNEL_POWER: assert property (
        @(posedge clk) disable iff (!rst_n)
        (C & (C - 1)) == 0
    ) else $error("gff_synth: C must be a power of 2");

    GFF_ITER_RANGE: assert property (
        @(posedge clk) disable iff (!rst_n)
        L2_ITER >= 2 && L2_ITER <= 8
    ) else $error("gff_synth: L2_ITER must be between 2 and 8");
    // synthesis translate_on

endmodule


// =============================================================================
// l2norm_nd_nr.sv
//
// N-Dimensional L2 Normalization using Newton-Raphson.
// Normalizes first CH channels of the 4C input per spatial point.
// =============================================================================

module l2norm_nd_nr #(
    parameter int N      = 1024,
    parameter int CH     = 64,
    parameter int DWIDTH = 16,
    parameter int FRAC_W = 8,
    parameter int ITER   = 4,
    parameter int USE_DSP = 0
)(
    input  logic clk,
    input  logic rst_n,
    input  logic signed [DWIDTH-1:0] feat_in  [0:N-1][0:4*CH-1],
    input  logic                     valid_in,
    output logic signed [DWIDTH-1:0] feat_out [0:N-1][0:CH-1],
    output logic                     valid_out
);
    // Process each spatial point independently using l2norm_nr
    // This is a parallel instantiation over N points

    localparam int LATENCY = 3 + 2 * ITER;

    // Valid pipeline
    logic [LATENCY-1:0] valid_pipe;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_pipe <= '0;
        else        valid_pipe <= {valid_pipe[LATENCY-2:0], valid_in};
    end
    assign valid_out = valid_pipe[LATENCY-1];

    // L2Norm for each spatial point
    generate
        for (genvar n = 0; n < N; n++) begin : gen_l2norm_points
            // Extract first CH channels from 4C input
            logic signed [DWIDTH-1:0] point_in [0:CH-1];
            logic signed [DWIDTH-1:0] point_out [0:CH-1];

            always_comb begin
                for (int c = 0; c < CH; c++)
                    point_in[c] = feat_in[n][c];
            end

            // Pack/unpack for l2norm_nr (packed port — Verilator compatible)
            logic signed [CH*DWIDTH-1:0] point_in_flat, point_out_flat;
            always_comb begin
                for (int c = 0; c < CH; c++)
                    point_in_flat[c*DWIDTH +: DWIDTH] = point_in[c];
            end

            l2norm_nr #(
                .CH         (CH),
                .DATA_W     (DWIDTH),
                .FRAC_W     (FRAC_W),
                .ITERATIONS (ITER),
                .USE_DSP    (USE_DSP)
            ) u_l2norm (
                .clk          (clk),
                .rst_n        (rst_n),
                .data_in_flat (point_in_flat),
                .valid_in     (valid_in),
                .ready_out    (),
                .data_out_flat(point_out_flat),
                .valid_out    (),  // Handled by valid_pipe
                .out_ready    (1'b1)  // FIXED: was unconnected (floating) — tie high
            );

            // Unpack output
            always_comb begin
                for (int c = 0; c < CH; c++)
                    point_out[c] = point_out_flat[c*DWIDTH +: DWIDTH];
            end

            // Output assignment
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int c = 0; c < CH; c++)
                        feat_out[n][c] <= '0;
                end else begin
                    for (int c = 0; c < CH; c++)
                        feat_out[n][c] <= point_out[c];
                end
            end
        end
    endgenerate

endmodule


// =============================================================================
// gelu_nd_rom.sv
//
// N-Dimensional GELU using ROM-based activation.
// Applies GELU to each element of N × CH tensor.
// =============================================================================

module gelu_nd_rom #(
    parameter int N       = 1024,
    parameter int CH      = 64,
    parameter int DATA_W  = 16,
    parameter int FRAC_W  = 8,
    parameter bit USE_DSP = 0,
    parameter string ROM_DIR = "rtl/mem"
)(
    input  logic clk,
    input  logic rst_n,
    input  logic signed [DATA_W-1:0] feat_in  [0:N-1][0:CH-1],
    input  logic                     valid_in,
    output logic signed [DATA_W-1:0] feat_out [0:N-1][0:CH-1],
    output logic                     valid_out
);
    // ROM-based GELU for each element.
    // All N×CH channels share a single ROM copy via gelu_rom_shared.
    localparam int LATENCY = 4;
    localparam int INDEX_BITS = pkg_config::INDEX_BITS;
    localparam int NUM_GELU_PORTS = N * CH;
    localparam int GELU_FRAC_W = DATA_W / 2;

    logic [LATENCY-1:0] valid_pipe;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_pipe <= '0;
        else        valid_pipe <= {valid_pipe[LATENCY-2:0], valid_in};
    end
    assign valid_out = valid_pipe[LATENCY-1];

    // ── Shared GELU ROM (1 copy instead of N copies) ────────────────────────
    logic [INDEX_BITS-1:0]      shared_rd_addr [NUM_GELU_PORTS-1:0];
    logic signed [DATA_W-1:0]   shared_y_base  [NUM_GELU_PORTS-1:0];
    logic signed [GELU_FRAC_W-1:0] shared_slope [NUM_GELU_PORTS-1:0];

    gelu_rom_shared #(
        .DATA_W     (DATA_W),
        .FRAC_W     (GELU_FRAC_W),
        .INDEX_BITS (INDEX_BITS),
        .NUM_PORTS  (NUM_GELU_PORTS),
        .ROM_DIR    (ROM_DIR)
    ) u_shared_gelu_rom (
        .clk        (clk),
        .rst_n      (rst_n),
        .rd_addr    (shared_rd_addr),
        .rd_en      (1'b1),
        .y_base_out (shared_y_base),
        .slope_out  (shared_slope)
    );

    // Instantiate one gelu_rom per spatial point, SHARE_ROM=1
    generate
        for (genvar n = 0; n < N; n++) begin : gen_gelu_n
            // Per-instance shared ROM interface wires
            logic [INDEX_BITS-1:0]      inst_index_o    [0:CH-1];
            logic signed [DATA_W-1:0]   inst_y_base_i   [0:CH-1];
            logic signed [GELU_FRAC_W-1:0] inst_slope_i [0:CH-1];

            gelu_rom #(
                .DATA_W     (DATA_W),
                .FRAC_W     (GELU_FRAC_W),
                .INDEX_BITS (INDEX_BITS),
                .CHANNELS   (CH),
                .USE_DSP    (USE_DSP),
                .SHARE_ROM  (1),
                .ROM_DIR    (ROM_DIR)
            ) u_gelu (
                .clk          (clk),
                .rst_n        (rst_n),
                .x_in         (feat_in[n]),
                .valid_in     (valid_in),
                .y_out        (feat_out[n]),
                .valid_out    (),
                .gelu_en      (1'b1),
                .index_o      (inst_index_o),
                .rom_y_base_i (inst_y_base_i),
                .rom_slope_i  (inst_slope_i)
            );

            // Wire per-instance ports to shared ROM flat arrays
            for (genvar c = 0; c < CH; c++) begin : gen_shared_wire
                assign shared_rd_addr[n*CH + c] = inst_index_o[c];
                assign inst_y_base_i[c]         = shared_y_base[n*CH + c];
                assign inst_slope_i[c]          = shared_slope[n*CH + c];
            end
        end
    endgenerate

endmodule


// =============================================================================
// linear_nd.sv (Enhanced version with ROM support)
//
// N-Dimensional Linear Layer.
// Applies the same linear transform to each of N spatial points.
// =============================================================================

module linear_nd #(
    parameter int N            = 1024,
    parameter int IN_CH        = 128,
    parameter int OUT_CH      = 64,
    parameter int DWIDTH      = 16,
    parameter bit STUB_BEHAVIOR = 1'b0,
    parameter string WEIGHT_FILE = "",
    parameter string BIAS_FILE   = ""
)(
    input  logic clk,
    input  logic rst_n,
    input  logic signed [DWIDTH-1:0] feat_in  [0:N-1][0:IN_CH-1],
    input  logic                     valid_in,
    output logic signed [DWIDTH-1:0] feat_out [0:N-1][0:OUT_CH-1],
    output logic                     valid_out
);

    localparam int ACC_WIDTH = DWIDTH * 2 + $clog2(IN_CH);
    localparam int CNT_WIDTH = $clog2(IN_CH);
    localparam int WEIGHT_SIZE = OUT_CH * IN_CH;
    localparam int FRAC_W = (DWIDTH > 1) ? DWIDTH / 2 : 1;
    localparam int LATENCY = IN_CH + 3;

    // Weight and Bias ROMs
    (* rom_style = "block" *) logic signed [DWIDTH-1:0] weight_rom [0:WEIGHT_SIZE-1];
    (* rom_style = "block" *) logic signed [DWIDTH-1:0] bias_rom   [0:OUT_CH-1];

    initial begin
        if (STUB_BEHAVIOR) begin
            for (int o = 0; o < OUT_CH; o++) begin
                for (int i = 0; i < IN_CH; i++)
                    weight_rom[o * IN_CH + i] = {DWIDTH{1'b1}};
                bias_rom[o] = '0;
            end
        end else begin
            if (WEIGHT_FILE != "") $readmemh(WEIGHT_FILE, weight_rom);
            else for (int i = 0; i < WEIGHT_SIZE; i++) weight_rom[i] = '0;
            if (BIAS_FILE != "") $readmemh(BIAS_FILE, bias_rom);
            else for (int i = 0; i < OUT_CH; i++) bias_rom[i] = '0;
        end
    end

    // Valid pipeline
    logic [LATENCY-1:0] valid_pipe;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_pipe <= '0;
        else        valid_pipe <= {valid_pipe[LATENCY-2:0], valid_in};
    end
    assign valid_out = valid_pipe[LATENCY-1];

    // Process each spatial point
    generate
        for (genvar n = 0; n < N; n++) begin : gen_linear_n
            // Input register
            logic signed [DWIDTH-1:0] data_in_reg [0:IN_CH-1];

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int i = 0; i < IN_CH; i++) data_in_reg[i] <= '0;
                end else if (valid_in) begin
                    data_in_reg <= feat_in[n];
                end
            end

            // Sequential MAC
            logic [CNT_WIDTH-1:0] cnt;
            logic running;
            logic signed [ACC_WIDTH-1:0] acc [0:OUT_CH-1];

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    running <= 1'b0;
                    cnt <= '0;
                    for (int o = 0; o < OUT_CH; o++) acc[o] <= '0;
                end else begin
                    if (valid_in && !running) begin
                        running <= 1'b1;
                        cnt <= '0;
                        for (int o = 0; o < OUT_CH; o++)
                            acc[o] <= ACC_WIDTH'(bias_rom[o]) <<< FRAC_W;
                    end else if (running) begin
                        if (cnt == IN_CH - 1) begin
                            running <= 1'b0;
                            cnt <= '0;
                        end else begin
                            cnt <= cnt + 1'b1;
                        end
                        // Accumulate
                        for (int o = 0; o < OUT_CH; o++) begin
                            acc[o] <= acc[o] + weight_rom[o * IN_CH + cnt] * data_in_reg[cnt];
                        end
                    end
                end
            end

            // Output — extract from correct Q-format position with saturation
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int o = 0; o < OUT_CH; o++)
                        feat_out[n][o] <= '0;
                end else begin
                    for (int o = 0; o < OUT_CH; o++) begin
                        // Saturation: check full-width accumulator before truncation
                        localparam logic signed [ACC_WIDTH-1:0] POS_SAT = (1 <<< (FRAC_W + DWIDTH - 1)) - 1;
                        localparam logic signed [ACC_WIDTH-1:0] NEG_SAT = -(1 <<< (FRAC_W + DWIDTH - 1));

                        if (acc[o] > POS_SAT) begin
                            feat_out[n][o] <= {DWIDTH{1'b1}} - 1;  // +MAX
                        end else if (acc[o] < NEG_SAT) begin
                            feat_out[n][o] <= {(DWIDTH){1'b1}};   // -MAX-1 (min signed)
                        end else begin
                            feat_out[n][o] <= acc[o][FRAC_W+DWIDTH-1:FRAC_W];
                        end
                    end
                end
            end
        end
    endgenerate

endmodule
