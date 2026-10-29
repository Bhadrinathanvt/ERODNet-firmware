// =============================================================================
// gff_synth_tm.sv
//
// Time-multiplexed variants of GFF parallel modules for Arty A7-100T.
// When TIME_MUX=1, behavior is identical to the original parallel versions.
// When TIME_MUX=4, resource usage drops by ~4x at the cost of TIME_MUXx
// longer processing time for these stages.
//
// Modules:
//   l2norm_nd_nr_tm  — Time-multiplexed N-dimensional L2 normalization
//   linear_nd_tm     — Time-multiplexed N-dimensional linear layer
//   gelu_nd_rom_tm   — Time-multiplexed N-dimensional GELU activation
//
// Since GFF is frame-at-a-time (single valid_in pulse), time-multiplexing
// simply stretches the processing time per frame. The valid_out pulse is
// delayed accordingly.
// =============================================================================

`timescale 1ns / 1ps

import pkg_config::*;
import pkg_module_config::*;

// =============================================================================
// l2norm_nd_nr_tm
//
// Time-multiplexed N-dimensional L2 normalization.
// Instantiates N/TIME_MUX parallel l2norm_nr modules instead of N.
// Processes TIME_MUX groups sequentially per frame.
//
// Latency: TIME_MUX * (3 + 2*ITER) cycles
// Resource: N/TIME_MUX l2norm_nr instances + round-robin logic
// =============================================================================

module l2norm_nd_nr_tm #(
    parameter int N        = 1024,
    parameter int CH      = pkg_config::MAX_C,
    parameter int DWIDTH = pkg_module_config::GFF_DATA_WIDTH,
    parameter int FRAC_W   = pkg_module_config::GFF_DATA_WIDTH / 2,
    parameter int ITER     = pkg_config::L2_ITER,
    parameter int USE_DSP  = 0,
    parameter int TIME_MUX = pkg_config::TIME_MUX
)(
    input  logic clk,
    input  logic rst_n,
    input  logic signed [DWIDTH-1:0] feat_in  [0:N-1][0:4*CH-1],
    input  logic                     valid_in,
    output logic signed [DWIDTH-1:0] feat_out [0:N-1][0:CH-1],
    output logic                     valid_out
);

    localparam int PAR_N      = N / TIME_MUX;           // Parallel instances per group
    localparam int GROUP_W    = $clog2(TIME_MUX);       // Group index width
    localparam int LAT_PER_GRP = 3 + 2 * ITER;          // Cycles per group
    localparam int TOTAL_LAT  = TIME_MUX * LAT_PER_GRP; // Total cycles
    localparam int CNT_W      = $clog2(TOTAL_LAT + 1);

    // -------------------------------------------------------------------------
    // State machine: cycle through TIME_MUX groups
    // -------------------------------------------------------------------------
    logic [GROUP_W-1:0]  group_idx;
    logic [CNT_W-1:0]    cycle_cnt;
    logic                processing;
    logic [CNT_W-1:0]    valid_delay;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            group_idx  <= '0;
            cycle_cnt  <= '0;
            processing <= 1'b0;
        end else begin
            if (valid_in && !processing) begin
                // Start processing: begin with group 0
                group_idx  <= '0;
                cycle_cnt  <= '0;
                processing <= 1'b1;
            end else if (processing) begin
                cycle_cnt <= cycle_cnt + 1'b1;
                if (cycle_cnt == LAT_PER_GRP - 1) begin
                    // End of current group
                    if (group_idx == TIME_MUX - 1) begin
                        // All groups done
                        processing <= 1'b0;
                        group_idx  <= '0;
                        cycle_cnt  <= '0;
                    end else begin
                        // Move to next group
                        group_idx <= group_idx + 1'b1;
                        cycle_cnt <= '0;
                    end
                end
            end
        end
    end

    // Valid output: fires TOTAL_LAT cycles after valid_in
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_delay <= '0;
        end else begin
            valid_delay <= {valid_delay[TOTAL_LAT-2:0], valid_in};
        end
    end
    assign valid_out = valid_delay[TOTAL_LAT-1];

    // -------------------------------------------------------------------------
    // Input mux: select current group's PAR_N points from feat_in
    // -------------------------------------------------------------------------
    logic signed [DWIDTH-1:0] group_feat_in [0:PAR_N-1][0:4*CH-1];

    always_comb begin
        for (int p = 0; p < PAR_N; p++) begin
            for (int c = 0; c < 4*CH; c++)
                group_feat_in[p][c] = feat_in[group_idx * PAR_N + p][c];
        end
    end

    // -------------------------------------------------------------------------
    // PAR_N parallel l2norm_nr instances (reused across TIME_MUX groups)
    // -------------------------------------------------------------------------
    logic group_valid_in;
    // Assert valid for the first cycle of each group processing
    assign group_valid_in = processing && (cycle_cnt == 0);

    generate
        for (genvar p = 0; p < PAR_N; p++) begin : gen_l2norm
            // Extract first CH channels from 4C input
            logic signed [DWIDTH-1:0] point_in [0:CH-1];
            logic signed [DWIDTH-1:0] point_out [0:CH-1];

            always_comb begin
                for (int c = 0; c < CH; c++)
                    point_in[c] = group_feat_in[p][c];
            end

            // Pack/unpack for l2norm_nr
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
                .valid_in     (group_valid_in),
                .ready_out    (),
                .data_out_flat(point_out_flat),
                .valid_out    (),  // Handled by valid_delay
                .out_ready    (1'b1)  // FIXED: was unconnected (floating) — tie high
            );

            // Unpack output
            always_comb begin
                for (int c = 0; c < CH; c++)
                    point_out[c] = point_out_flat[c*DWIDTH +: DWIDTH];
            end

            // Output register — capture each group's results
            // l2norm_nr has LAT_PER_GRP cycle latency, so output arrives
            // LAT_PER_GRP cycles after group_valid_in. We need to know which
            // group produced each output to write it to the correct feat_out indices.
            // Use a tracking counter that mirrors the group_idx at the time the
            // l2norm_nr input was registered.
            logic [GROUP_W-1:0] out_group;
            logic               out_group_valid;

            // Track which group's output is emerging from the pipeline
            // LAT_PER_GRP cycle delay from group_valid_in to output
            logic [GROUP_W:0] group_pipe [0:LAT_PER_GRP-1];

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int s = 0; s < LAT_PER_GRP; s++)
                        group_pipe[s] <= '0;
                    out_group_valid <= 1'b0;
                end else begin
                    group_pipe[0] <= {1'b0, group_valid_in ? group_idx : '0};
                    group_pipe[0][GROUP_W] <= group_valid_in;
                    for (int s = 1; s < LAT_PER_GRP; s++)
                        group_pipe[s] <= group_pipe[s-1];
                    out_group       <= group_pipe[LAT_PER_GRP-1][GROUP_W-1:0];
                    out_group_valid <= group_pipe[LAT_PER_GRP-1][GROUP_W];
                end
            end

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int c = 0; c < CH; c++)
                        feat_out[p][c] <= '0;
                end else if (out_group_valid) begin
                    // Write to the correct output group
                    for (int c = 0; c < CH; c++)
                        feat_out[out_group * PAR_N + p][c] <= point_out[c];
                end
            end
        end
    endgenerate

    // Assertion
    // synthesis translate_off
    initial begin
        if (N % TIME_MUX != 0)
            $error("l2norm_nd_nr_tm: N must be divisible by TIME_MUX (N=%0d, TIME_MUX=%0d)", N, TIME_MUX);
        if ((TIME_MUX & (TIME_MUX - 1)) != 0)
            $error("l2norm_nd_nr_tm: TIME_MUX must be power of 2 (got %0d)", TIME_MUX);
    end
    // synthesis translate_on

endmodule


// =============================================================================
// linear_nd_tm
//
// Time-multiplexed N-dimensional linear layer.
// Instantiates N/TIME_MUX parallel MAC FSMs instead of N.
// Processes TIME_MUX groups sequentially per frame.
//
// Latency: TIME_MUX * (IN_CH + 3) cycles
// Resource: N/TIME_MUX FSM engines + shared weight ROM
// =============================================================================

module linear_nd_tm #(
    parameter int N            = 1024,
    parameter int IN_CH        = pkg_config::MAX_C * 2,
    parameter int OUT_CH      = pkg_config::MAX_C,
    parameter int DWIDTH = pkg_module_config::GFF_DATA_WIDTH,
    parameter bit STUB_BEHAVIOR = 1'b0,
    parameter string WEIGHT_FILE = "",
    parameter string BIAS_FILE   = "",
    parameter int TIME_MUX     = pkg_config::TIME_MUX
)(
    input  logic clk,
    input  logic rst_n,
    input  logic signed [DWIDTH-1:0] feat_in  [0:N-1][0:IN_CH-1],
    input  logic                     valid_in,
    output logic signed [DWIDTH-1:0] feat_out [0:N-1][0:OUT_CH-1],
    output logic                     valid_out
);

    localparam int PAR_N       = N / TIME_MUX;
    localparam int GROUP_W     = $clog2(TIME_MUX);
    localparam int ACC_WIDTH   = DWIDTH * 2 + $clog2(IN_CH);
    localparam int CNT_WIDTH   = $clog2(IN_CH);
    localparam int WEIGHT_SIZE = OUT_CH * IN_CH;
    localparam int FRAC_W      = (DWIDTH > 1) ? DWIDTH / 2 : 1;
    localparam int LAT_PER_GRP = IN_CH + 3;
    localparam int TOTAL_LAT   = TIME_MUX * LAT_PER_GRP;
    localparam int TOTAL_CNT_W = $clog2(TOTAL_LAT + 1);

    // Weight and Bias ROMs (shared across all PAR_N engines)
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

    // -------------------------------------------------------------------------
    // State machine: cycle through TIME_MUX groups
    // -------------------------------------------------------------------------
    logic [GROUP_W-1:0]    group_idx;
    logic [TOTAL_CNT_W-1:0] cycle_cnt;
    logic                  processing;
    logic                  group_start;   // Pulse at start of each group

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            group_idx  <= '0;
            cycle_cnt  <= '0;
            processing <= 1'b0;
            group_start <= 1'b0;
        end else begin
            group_start <= 1'b0;

            if (valid_in && !processing) begin
                group_idx  <= '0;
                cycle_cnt  <= '0;
                processing <= 1'b1;
                group_start <= 1'b1;
            end else if (processing) begin
                cycle_cnt <= cycle_cnt + 1'b1;
                if (cycle_cnt == LAT_PER_GRP - 1) begin
                    if (group_idx == TIME_MUX - 1) begin
                        processing <= 1'b0;
                        group_idx  <= '0;
                        cycle_cnt  <= '0;
                    end else begin
                        group_idx <= group_idx + 1'b1;
                        cycle_cnt <= '0;
                        group_start <= 1'b1;
                    end
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Input mux: select current group's PAR_N points
    // -------------------------------------------------------------------------
    logic signed [DWIDTH-1:0] group_feat_in [0:PAR_N-1][0:IN_CH-1];

    always_comb begin
        for (int p = 0; p < PAR_N; p++)
            for (int i = 0; i < IN_CH; i++)
                group_feat_in[p][i] = feat_in[group_idx * PAR_N + p][i];
    end

    // -------------------------------------------------------------------------
    // Valid output
    // -------------------------------------------------------------------------
    logic [TOTAL_LAT-1:0] valid_pipe;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_pipe <= '0;
        else        valid_pipe <= {valid_pipe[TOTAL_LAT-2:0], valid_in};
    end
    assign valid_out = valid_pipe[TOTAL_LAT-1];

    // -------------------------------------------------------------------------
    // PAR_N parallel MAC engines (reused across TIME_MUX groups)
    // -------------------------------------------------------------------------
    // Track which output group is emerging from the pipeline
    logic [GROUP_W:0] grp_pipe [0:TOTAL_LAT-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int s = 0; s < TOTAL_LAT; s++)
                grp_pipe[s] <= '0;
        end else begin
            grp_pipe[0] <= group_start ? {1'b1, group_idx} : '0;
            for (int s = 1; s < TOTAL_LAT; s++)
                grp_pipe[s] <= grp_pipe[s-1];
        end
    end

    logic [GROUP_W-1:0] out_group;
    logic               out_group_valid;
    assign out_group       = grp_pipe[TOTAL_LAT-1][GROUP_W-1:0];
    assign out_group_valid = grp_pipe[TOTAL_LAT-1][GROUP_W];

    generate
        for (genvar p = 0; p < PAR_N; p++) begin : gen_linear_n
            // Input register
            logic signed [DWIDTH-1:0] data_in_reg [0:IN_CH-1];

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int i = 0; i < IN_CH; i++) data_in_reg[i] <= '0;
                end else if (group_start) begin
                    data_in_reg <= group_feat_in[p];
                end
            end

            // Sequential MAC FSM
            logic [CNT_WIDTH-1:0] cnt;
            logic running;
            logic signed [ACC_WIDTH-1:0] acc [0:OUT_CH-1];

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    running <= 1'b0;
                    cnt <= '0;
                    for (int o = 0; o < OUT_CH; o++) acc[o] <= '0;
                end else begin
                    if (group_start && !running) begin
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
                        for (int o = 0; o < OUT_CH; o++) begin
                            acc[o] <= acc[o] + weight_rom[o * IN_CH + cnt] * data_in_reg[cnt];
                        end
                    end
                end
            end

            // Output: write to correct group in feat_out
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int o = 0; o < OUT_CH; o++)
                        feat_out[p][o] <= '0;
                end else if (out_group_valid) begin
                    for (int o = 0; o < OUT_CH; o++)
                        feat_out[out_group * PAR_N + p][o] <= acc[o][FRAC_W+DWIDTH-1:FRAC_W];
                end
            end
        end
    endgenerate

    // Assertion
    // synthesis translate_off
    initial begin
        if (N % TIME_MUX != 0)
            $error("linear_nd_tm: N must be divisible by TIME_MUX (N=%0d, TIME_MUX=%0d)", N, TIME_MUX);
        if ((TIME_MUX & (TIME_MUX - 1)) != 0)
            $error("linear_nd_tm: TIME_MUX must be power of 2 (got %0d)", TIME_MUX);
    end
    // synthesis translate_on

endmodule


// =============================================================================
// gelu_nd_rom_tm
//
// Time-multiplexed N-dimensional GELU activation.
// Instantiates N/TIME_MUX * CH gelu_rom instances instead of N * CH.
// Processes TIME_MUX groups of spatial points sequentially.
//
// Latency: TIME_MUX * 4 cycles
// Resource: (N/TIME_MUX) * CH gelu_rom instances
// =============================================================================

module gelu_nd_rom_tm #(
    parameter int N        = 1024,
    parameter int CH      = pkg_config::MAX_C,
    parameter int DATA_W = pkg_module_config::GFF_DATA_WIDTH,
    parameter int FRAC_W  = DATA_W / 2,
    parameter bit USE_DSP = 0,
    parameter string ROM_DIR = "rtl/mem",
    parameter int TIME_MUX = pkg_config::TIME_MUX
)(
    input  logic clk,
    input  logic rst_n,
    input  logic signed [DATA_W-1:0] feat_in  [0:N-1][0:CH-1],
    input  logic                     valid_in,
    output logic signed [DATA_W-1:0] feat_out [0:N-1][0:CH-1],
    output logic                     valid_out
);

    localparam int PAR_N      = N / TIME_MUX;
    localparam int GROUP_W    = $clog2(TIME_MUX);
    localparam int GELU_LAT   = 4;
    localparam int TOTAL_LAT  = TIME_MUX * GELU_LAT;
    localparam int INDEX_BITS = pkg_config::INDEX_BITS;
    localparam int NUM_GELU_PORTS = PAR_N * CH;
    localparam int GELU_FRAC_W = DATA_W / 2;

    // -------------------------------------------------------------------------
    // State machine: cycle through TIME_MUX groups
    // -------------------------------------------------------------------------
    logic [GROUP_W-1:0] group_idx;
    logic [2:0]         cycle_cnt;    // 0..GELU_LAT-1 per group
    logic               processing;
    logic               group_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            group_idx  <= '0;
            cycle_cnt  <= 3'b0;
            processing <= 1'b0;
        end else begin
            if (valid_in && !processing) begin
                group_idx  <= '0;
                cycle_cnt  <= 3'b0;
                processing <= 1'b1;
            end else if (processing) begin
                if (cycle_cnt == GELU_LAT - 1) begin
                    if (group_idx == TIME_MUX - 1) begin
                        processing <= 1'b0;
                        group_idx  <= '0;
                        cycle_cnt  <= 3'b0;
                    end else begin
                        group_idx <= group_idx + 1'b1;
                        cycle_cnt <= 3'b0;
                    end
                end else begin
                    cycle_cnt <= cycle_cnt + 1'b1;
                end
            end
        end
    end

    // group_valid: pulse at the start of each group's GELU processing
    assign group_valid = processing && (cycle_cnt == 0);

    // -------------------------------------------------------------------------
    // Input mux: select current group's PAR_N points
    // -------------------------------------------------------------------------
    logic signed [DATA_W-1:0] group_feat_in [0:PAR_N-1][0:CH-1];

    always_comb begin
        for (int p = 0; p < PAR_N; p++)
            for (int c = 0; c < CH; c++)
                group_feat_in[p][c] = feat_in[group_idx * PAR_N + p][c];
    end

    // -------------------------------------------------------------------------
    // Valid output
    // -------------------------------------------------------------------------
    logic [TOTAL_LAT-1:0] valid_pipe;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_pipe <= '0;
        else        valid_pipe <= {valid_pipe[TOTAL_LAT-2:0], valid_in};
    end
    assign valid_out = valid_pipe[TOTAL_LAT-1];

    // -------------------------------------------------------------------------
    // Track which group's output is emerging from the pipeline
    // -------------------------------------------------------------------------
    logic [GROUP_W:0] grp_pipe [0:GELU_LAT-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int s = 0; s < GELU_LAT; s++)
                grp_pipe[s] <= '0;
        end else begin
            grp_pipe[0] <= group_valid ? {1'b1, group_idx} : '0;
            for (int s = 1; s < GELU_LAT; s++)
                grp_pipe[s] <= grp_pipe[s-1];
        end
    end

    logic [GROUP_W-1:0] out_group;
    logic               out_group_valid;
    assign out_group       = grp_pipe[GELU_LAT-1][GROUP_W-1:0];
    assign out_group_valid = grp_pipe[GELU_LAT-1][GROUP_W];

    // -------------------------------------------------------------------------
    // Shared GELU ROM (1 copy instead of PAR_N copies)
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // PAR_N gelu_rom instances (reused across TIME_MUX groups), SHARE_ROM=1.
    // Each instance has CHANNELS=CH, so all CH channels within a spatial point
    // share one ROM copy. Total ROM copies: 1 (shared) instead of PAR_N.
    // -------------------------------------------------------------------------
    generate
        for (genvar p = 0; p < PAR_N; p++) begin : gen_gelu_n
            logic signed [DATA_W-1:0] gelu_out [0:CH-1];

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
                .x_in         (group_feat_in[p]),
                .valid_in     (group_valid),
                .y_out        (gelu_out),
                .valid_out    (),
                .gelu_en      (1'b1),
                .index_o      (inst_index_o),
                .rom_y_base_i (inst_y_base_i),
                .rom_slope_i  (inst_slope_i)
            );

            // Wire per-instance ports to shared ROM flat arrays
            for (genvar c = 0; c < CH; c++) begin : gen_shared_wire
                assign shared_rd_addr[p*CH + c] = inst_index_o[c];
                assign inst_y_base_i[c]         = shared_y_base[p*CH + c];
                assign inst_slope_i[c]          = shared_slope[p*CH + c];
            end

            // Output register: write to correct group
            for (genvar c = 0; c < CH; c++) begin : gen_out_reg
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n)
                        feat_out[p][c] <= '0;
                    else if (out_group_valid)
                        feat_out[out_group * PAR_N + p][c] <= gelu_out[c];
                end
            end
        end
    endgenerate

    // Assertion
    // synthesis translate_off
    initial begin
        if (N % TIME_MUX != 0)
            $error("gelu_nd_rom_tm: N must be divisible by TIME_MUX (N=%0d, TIME_MUX=%0d)", N, TIME_MUX);
        if ((TIME_MUX & (TIME_MUX - 1)) != 0)
            $error("gelu_nd_rom_tm: TIME_MUX must be power of 2 (got %0d)", TIME_MUX);
    end
    // synthesis translate_on

endmodule
