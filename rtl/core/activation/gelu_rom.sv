// =============================================================================
// gelu_rom.sv
//
// ROM-based GELU Activation Function with Piecewise-Linear Interpolation.
// Computes: GELU(x) = x * Φ(x) where Φ(x) is the standard normal CDF.
// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
//
// Implementation:
//   Uses ROM lookup table with piecewise-linear interpolation.
//   ROM stores base values and slopes for each segment.
//   y = y_base[index] + slope[index] * (x - x_base[index])
//
// Pipeline:
//   Stage 1: Input registration, index extraction
//   Stage 2: ROM read (y_base and slope)
//   Stage 3: Multiply (slope × delta)
//   Stage 4: Add (y_base + product)
//
// ROM Format:
//   - INDEX_BITS from pkg_config (default 8 = 256 segments)
//   - INPUT_RANGE: [-4σ, +4σ] mapped to [0, 2^INDEX_BITS-1]
//   - y_base.mem:  32-bit Q16.16 values
//   - slope.mem:   16-bit Q0.16 fractional slope
//
// ROM Sharing:
//   When SHARE_ROM=1, the ROM arrays are NOT instantiated inside this module.
//   Instead, the external ports rom_y_base_i and rom_slope_i are used to read
//   from a shared ROM instantiated once at a higher level. This eliminates the
//   N×C ROM duplication problem: instead of N×C ROM copies, only 1 ROM is used.
//   The index computation and pipeline registers remain per-channel.
//
//   With SHARE_ROM=0 (default), the ROM is instantiated inside this module
//   and the external ROM ports are unused (tied off).
//
// Latency: 4 cycles
// Throughput: 1 result/cycle
// =============================================================================

`timescale 1ns / 1ps

import pkg_config::*;

module gelu_rom #(
    parameter int DATA_W     = pkg_config::ROM_WIDTH,   // Data width (Q16.16)
    parameter int FRAC_W     = pkg_config::FRAC_WIDTH,  // Fractional bits
    parameter int INDEX_BITS = pkg_config::INDEX_BITS,  // ROM address width
    parameter int CHANNELS   = 1,        // Number of parallel channels
    parameter bit  USE_DSP   = 1,        // Use DSP slices for multiply
    parameter bit  ROM_BLOCK = 1,         // Use block RAM (vs distributed)
    parameter bit  SHARE_ROM = 0,         // 1 = use external shared ROM, 0 = local ROM
    parameter string ROM_DIR  = "rtl/mem"  // ROM file directory
)(
    input  logic clk,
    input  logic rst_n,

    // Input
    input  logic signed [DATA_W-1:0] x_in    [CHANNELS-1:0],
    input  logic                        valid_in,

    // Output
    output logic signed [DATA_W-1:0] y_out    [CHANNELS-1:0],
    output logic                        valid_out,

    // Enable (for clock gating)
    input  logic                        gelu_en,

    // Shared ROM interface (used only when SHARE_ROM=1)
    // index_o: computed index for shared ROM to read (1 cycle after valid_in)
    // rom_y_base_i / rom_slope_i: shared ROM outputs (2 cycles after valid_in)
    output logic [INDEX_BITS-1:0]       index_o      [CHANNELS-1:0],
    input  logic signed [DATA_W-1:0]   rom_y_base_i [CHANNELS-1:0],
    input  logic signed [FRAC_W-1:0]   rom_slope_i  [CHANNELS-1:0]
);

    // =========================================================================
    // ROM Parameters
    // =========================================================================
    localparam int ROM_DEPTH = 1 << INDEX_BITS;  // 256 entries

    // GELU input clip range: x is clamped to [-RANGE/2, +RANGE/2] before
    // segment indexing. RANGE comes from pkg_config (config.yaml::activation_rom).
    localparam int GELU_RANGE = pkg_config::GELU_INPUT_RANGE;
    localparam signed [DATA_W-1:0] X_MIN = -(GELU_RANGE/2) * (1 << FRAC_W);
    localparam signed [DATA_W-1:0] X_MAX =  (GELU_RANGE/2) * (1 << FRAC_W);
    localparam signed [DATA_W-1:0] X_RANGE = GELU_RANGE * (1 << FRAC_W);
    localparam int SEGMENT_SHIFT = FRAC_W + 3 - INDEX_BITS;
    localparam int DELTA_SCALE = FRAC_W - SEGMENT_SHIFT;

    // =========================================================================
    // ROM Declaration (only when SHARE_ROM=0)
    // =========================================================================
    // Intermediate wires to bridge generate-block scope for Verilator compatibility
    logic signed [DATA_W-1:0]   local_rom_y_base [0:ROM_DEPTH-1];
    logic signed [FRAC_W-1:0]   local_rom_slope  [0:ROM_DEPTH-1];
    logic local_rom_valid;

    generate
        if (!SHARE_ROM) begin : gen_local_rom
            (* rom_style = "block" *)
            logic signed [DATA_W-1:0] rom_y_base [0:ROM_DEPTH-1];

            (* rom_style = "block" *)
            logic signed [FRAC_W-1:0] rom_slope [0:ROM_DEPTH-1];

            initial begin
                $readmemh({ROM_DIR, "/gelu_y_base.mem"}, rom_y_base);
                $readmemh({ROM_DIR, "/gelu_m_slope.mem"}, rom_slope);
            end

            // Bridge to module-level wires
            for (genvar i = 0; i < ROM_DEPTH; i++) begin : gen_bridge
                assign local_rom_y_base[i] = rom_y_base[i];
                assign local_rom_slope[i]   = rom_slope[i];
            end
            assign local_rom_valid = 1'b1;
        end else begin : gen_no_local_rom
            assign local_rom_valid = 1'b0;
        end
    endgenerate

    // =========================================================================
    // Pipeline Stage 1: Input Registration & Index Extraction
    // =========================================================================
    logic signed [DATA_W-1:0] x_d1 [CHANNELS-1:0];
    logic [INDEX_BITS-1:0]    index [CHANNELS-1:0];
    logic signed [FRAC_W-1:0] delta [CHANNELS-1:0];
    logic valid_d1;

    // Combinational index/delta computation
    logic [INDEX_BITS-1:0]    index_next [CHANNELS-1:0];
    logic signed [FRAC_W-1:0] delta_next [CHANNELS-1:0];

    generate
        for (genvar c = 0; c < CHANNELS; c++) begin : gen_stage1_comb
            // x_offset declared at always_comb scope with a default — declaring it
            // inside the `else if` branch made it (implicitly static) only-
            // conditionally-assigned, which Verilator flagged as an inferred latch.
            logic signed [DATA_W-1:0] x_offset;
            always_comb begin
                // Default: clamp to last entry (covers out-of-range and unused elements)
                index_next[c] = ROM_DEPTH[INDEX_BITS-1:0] - 1'b1;
                delta_next[c] = '0;
                x_offset      = '0;
                if (x_in[c] < X_MIN) begin
                    index_next[c] = '0;
                    delta_next[c] = '0;
                end else if (x_in[c] < X_MAX) begin
                    x_offset = x_in[c] - X_MIN;
                    index_next[c] = x_offset[SEGMENT_SHIFT + INDEX_BITS - 1 -: INDEX_BITS];
                    delta_next[c] = x_offset[SEGMENT_SHIFT-1:0] << DELTA_SCALE;
                end
            end
        end
    endgenerate

    generate
        for (genvar c = 0; c < CHANNELS; c++) begin : gen_stage1
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    x_d1[c] <= '0;
                    index[c] <= '0;
                    delta[c] <= '0;
                end else if (valid_in && gelu_en) begin
                    x_d1[c] <= x_in[c];
                    index[c] <= index_next[c];
                    delta[c] <= delta_next[c];
                end
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_d1 <= 1'b0;
        else        valid_d1 <= valid_in && gelu_en;
    end

    // Expose computed index for shared ROM (available 1 cycle after valid_in)
    generate
        for (genvar c = 0; c < CHANNELS; c++) begin : gen_index_out
            assign index_o[c] = index[c];
        end
    endgenerate

    // =========================================================================
    // Pipeline Stage 2: ROM Read
    // =========================================================================
    logic signed [DATA_W-1:0] y_base [CHANNELS-1:0];
    logic signed [FRAC_W-1:0] slope  [CHANNELS-1:0];
    logic signed [DATA_W-1:0] x_d2 [CHANNELS-1:0];
    logic signed [FRAC_W-1:0] delta_d2 [CHANNELS-1:0];
    logic valid_d2;

    generate
        for (genvar c = 0; c < CHANNELS; c++) begin : gen_stage2
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    y_base[c] <= '0;
                    slope[c] <= '0;
                    x_d2[c] <= '0;
                    delta_d2[c] <= '0;
                end else if (valid_d1) begin
                    if (SHARE_ROM) begin
                        // Use shared ROM inputs from parent module
                        y_base[c] <= rom_y_base_i[c];
                        slope[c] <= rom_slope_i[c];
                    end else begin
                        // Use local ROM (via module-level bridge wires)
                        y_base[c] <= local_rom_y_base[index[c]];
                        slope[c] <= local_rom_slope[index[c]];
                    end
                    x_d2[c] <= x_d1[c];
                    delta_d2[c] <= delta[c];
                end
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_d2 <= 1'b0;
        else        valid_d2 <= valid_d1;
    end

    // =========================================================================
    // Pipeline Stage 3: Multiply (slope × delta)
    // =========================================================================
    logic signed [DATA_W+FRAC_W-1:0] mult_result [CHANNELS-1:0];
    logic signed [DATA_W-1:0] y_base_d3 [CHANNELS-1:0];
    logic signed [DATA_W-1:0] x_d3 [CHANNELS-1:0];
    logic valid_d3;

    generate
        for (genvar c = 0; c < CHANNELS; c++) begin : gen_stage3
            (* use_dsp = "yes" *)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    mult_result[c] <= '0;
                    y_base_d3[c] <= '0;
                    x_d3[c] <= '0;
                end else if (valid_d2) begin
                    mult_result[c] <= slope[c] * delta_d2[c];
                    y_base_d3[c] <= y_base[c];
                    x_d3[c] <= x_d2[c];
                end
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_d3 <= 1'b0;
        else        valid_d3 <= valid_d2;
    end

    // =========================================================================
    // Pipeline Stage 4: Add & Output
    // =========================================================================
    logic valid_d4;

    generate
        for (genvar c = 0; c < CHANNELS; c++) begin : gen_stage4
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    y_out[c] <= '0;
                end else if (valid_d3) begin
                    if (x_d3[c] < X_MIN) begin
                        y_out[c] <= '0;
                    end else if (x_d3[c] > X_MAX) begin
                        y_out[c] <= x_d3[c];
                    end else begin
                        y_out[c] <= y_base_d3[c] + (mult_result[c] >>> FRAC_W);
                    end
                end
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_out <= 1'b0;
        else        valid_out <= valid_d3;
    end

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    GELU_CHANNELS: assert property (
        @(posedge clk) disable iff (!rst_n)
        CHANNELS >= 1 && CHANNELS <= 256
    ) else $error("gelu_rom: Channel count out of range");

    GELU_INDEX_BITS: assert property (
        @(posedge clk) disable iff (!rst_n)
        INDEX_BITS >= 6 && INDEX_BITS <= 10
    ) else $error("gelu_rom: INDEX_BITS out of range (6-10)");
    // synthesis translate_on

endmodule


// =============================================================================
// gelu_rom_shared.sv
//
// Shared ROM wrapper for GELU activation.
// Instantiates a single ROM pair (y_base + slope) and provides per-channel
// read ports. This module is instantiated once at a higher level (e.g., in
// gelu_nd_rom or gelu_nd_rom_tm) and the ROM outputs are wired to all
// gelu_rom instances with SHARE_ROM=1.
//
// Resource savings:
//   Without sharing: N × C ROM copies = N × C × (256×32 + 256×16) bits
//   With sharing:    1 ROM copy = 256×32 + 256×16 = 12,288 bits = ~0.4 BRAM36
//
// For N=256, C=8 (C=32, TIME_MUX=4): saves 2048 - 1 = 2047 ROM copies,
// reducing BRAM from ~2048 to ~3 (the shared ROM + pipeline registers).
// =============================================================================

module gelu_rom_shared #(
    parameter int DATA_W     = pkg_config::ROM_WIDTH,
    parameter int FRAC_W     = pkg_config::FRAC_WIDTH,
    parameter int INDEX_BITS = pkg_config::INDEX_BITS,
    parameter int NUM_PORTS  = 1,        // Number of read ports (max channels per cycle)
    parameter bit  ROM_BLOCK = 1,
    parameter string ROM_DIR  = "rtl/mem"
)(
    input  logic clk,
    input  logic rst_n,

    // Index inputs (one per read port)
    input  logic [INDEX_BITS-1:0]    rd_addr [NUM_PORTS-1:0],
    input  logic                     rd_en,

    // ROM outputs (one per read port)
    output logic signed [DATA_W-1:0] y_base_out [NUM_PORTS-1:0],
    output logic signed [FRAC_W-1:0] slope_out  [NUM_PORTS-1:0]
);

    localparam int ROM_DEPTH = 1 << INDEX_BITS;

    // =========================================================================
    // Single shared ROM
    // =========================================================================
    (* rom_style = "block" *)
    logic signed [DATA_W-1:0] rom_y_base [0:ROM_DEPTH-1];

    (* rom_style = "block" *)
    logic signed [FRAC_W-1:0] rom_slope [0:ROM_DEPTH-1];

    initial begin
        $readmemh({ROM_DIR, "/gelu_y_base.mem"}, rom_y_base);
        $readmemh({ROM_DIR, "/gelu_m_slope.mem"}, rom_slope);
    end

    // =========================================================================
    // Read ports: combinational (async) read from shared ROM.
    // Multiple ports can read different addresses simultaneously.
    // The consumer (gelu_rom stage 2) registers these outputs.
    //
    // Note: Async reads prevent block RAM inference in some synthesis tools.
    // If block RAM is required, use registered reads and add a 1-cycle
    // pipeline stage in gelu_rom when SHARE_ROM=1 (not implemented here).
    // =========================================================================
    generate
        for (genvar p = 0; p < NUM_PORTS; p++) begin : gen_rd_port
            assign y_base_out[p] = rom_y_base[rd_addr[p]];
            assign slope_out[p]  = rom_slope[rd_addr[p]];
        end
    endgenerate

endmodule
