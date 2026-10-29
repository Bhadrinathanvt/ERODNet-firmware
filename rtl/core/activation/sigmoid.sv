// =============================================================================
// Activation_Sigmoid_Interpolated.sv
//
// Sigmoid activation using ROM-based piecewise-linear interpolation.
// σ(x) = 1 / (1 + e^(-x))
//
// Index extraction:
//   The ROM uses INDEX_START_BIT=12 in a 32-bit Q16.16 representation.
//   When DATA_WIDTH < ROM_WIDTH, the narrow input is sign-extended and
//   shifted left by FRAC_WIDTH before index/delta extraction.
//
// Pipeline: 4 cycles
//   Stage 0: Scale x_in to ROM_WIDTH (Q16.16), extract index + delta
//   Stage 1: Register ROM data + delta
//   Stage 2: Multiply (slope × delta) — registered product
//   Stage 3: Round product + align y_base
//   Stage 4: Add (y_base + product), truncate to DATA_WIDTH
//
// Artix-7 100T Optimization:
//   - USE_DSP defaults to 0 (LUT-based interpolation)
//   - ROM uses BRAM, interpolation multiply uses LUTs
//   - DSP savings: 1 DSP per channel
// =============================================================================
import pkg_config::*;
import pkg_module_config::*;

module Activation_Sigmoid_Interpolated #(
    parameter int DATA_WIDTH = pkg_module_config::UNIFIED_ACTIVATION_DATA_WIDTH,
    parameter int FRAC_WIDTH = pkg_config::FRAC_WIDTH,
    parameter int INDEX_BITS = pkg_config::INDEX_BITS,
    parameter bit  USE_DSP = 0       // Default: LUT-based for Artix-7 100T
)(
    input  logic clk,
    input  logic rst_n,
    input  logic signed [DATA_WIDTH-1:0] x_in,
    input  logic signed [pkg_config::ROM_WIDTH-1:0] y_base_in,   // from shared ROM (Q16.16)
    input  logic signed [pkg_config::ROM_WIDTH-1:0] slope_in,    // from shared ROM (Q16.16)
    output logic signed [DATA_WIDTH-1:0] y_out
);

    // =========================================================================
    // Input Scaling: convert narrow DATA_WIDTH input to ROM_WIDTH (Q16.16)
    // =========================================================================
    // ROM_WIDTH and INDEX_START_BIT come from pkg_config (config.yaml ::
    // activation_rom). They are tied to the .mem file geometry — regenerate
    // ROMs if they change (scripts/generate_roms.py).
    localparam int ROM_WIDTH       = pkg_config::ROM_WIDTH;
    localparam int INDEX_START_BIT = pkg_config::INDEX_START_BIT;
    localparam int DELTA_BITS     = ROM_WIDTH - INDEX_START_BIT - INDEX_BITS;

    // Sign-extend x_in to ROM_WIDTH, then shift left by FRAC_WIDTH
    // to convert from INT (or narrow Q-format) to Q16.16 representation.
    logic signed [ROM_WIDTH-1:0] x_in_scaled;
    assign x_in_scaled = { {(ROM_WIDTH-DATA_WIDTH){x_in[DATA_WIDTH-1]}}, x_in } <<< FRAC_WIDTH;

    // Extract index and delta from scaled Q16.16 representation
    logic [INDEX_BITS-1:0]  index;
    logic signed [DELTA_BITS-1:0] delta_x;

    assign index   = x_in_scaled[INDEX_START_BIT+INDEX_BITS-1 : INDEX_START_BIT];
    assign delta_x = x_in_scaled[INDEX_START_BIT-1 : 0];

    // Pipeline registers for delta
    logic signed [DELTA_BITS-1:0] delta_x_reg;

    // =========================================================================
    // Stage 1: Register external ROM data + delta_x
    // =========================================================================
    logic signed [ROM_WIDTH-1:0] y_base_reg;
    logic signed [ROM_WIDTH-1:0] slope_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_base_reg  <= '0;
            slope_reg   <= '0;
            delta_x_reg <= '0;
        end else begin
            y_base_reg  <= y_base_in;
            slope_reg   <= slope_in;
            delta_x_reg <= delta_x;
        end
    end

    // =========================================================================
    // Stage 2: Interpolation Multiply (LUT-based or DSP)
    // =========================================================================
    // Product of slope (Q16.16) × delta (Q0.DELTA_BITS) yields
    // Q(16+DELTA_BITS).16.  Shift right by FRAC_WIDTH to align.
    logic signed [ROM_WIDTH-1:0]      dsp_mult;
    logic signed [2*ROM_WIDTH-1:0]   mult_full;
    logic signed [ROM_WIDTH-1:0]      y_base_reg2;
    logic signed [ROM_WIDTH-1:0]      y_base_reg3;

    generate
        if (USE_DSP) begin : gen_interp_dsp
            (* use_dsp = "yes" *)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    mult_full  <= '0;
                    dsp_mult   <= '0;
                    y_base_reg2 <= '0;
                    y_base_reg3 <= '0;
                end else begin
                    mult_full  <= slope_reg * delta_x_reg;
                    dsp_mult   <= (mult_full + (1 <<< (FRAC_WIDTH-1))) >>> FRAC_WIDTH;
                    y_base_reg2 <= y_base_reg;
                    y_base_reg3 <= y_base_reg2;
                end
            end
        end else begin : gen_interp_lut
            (* use_dsp = "no" *)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    mult_full  <= '0;
                    dsp_mult   <= '0;
                    y_base_reg2 <= '0;
                    y_base_reg3 <= '0;
                end else begin
                    // LUT-based multiply (saves DSP)
                    mult_full  <= slope_reg * delta_x_reg;
                    dsp_mult   <= (mult_full + (1 <<< (FRAC_WIDTH-1))) >>> FRAC_WIDTH;
                    y_base_reg2 <= y_base_reg;
                    y_base_reg3 <= y_base_reg2;
                end
            end
        end
    endgenerate

    // =========================================================================
    // Stage 3: Final Add + Truncate to DATA_WIDTH
    // =========================================================================
    // Result in Q16.16: shift right by FRAC_WIDTH to get integer part,
    // then truncate to DATA_WIDTH with saturation.
    logic signed [ROM_WIDTH-1:0] result_q16;
    logic signed [DATA_WIDTH-1:0] result_sat;

    // Combinational saturation (extracted from always_ff for Verilator compat)
    always_comb begin
        automatic logic signed [ROM_WIDTH-1:0] shifted;
        result_q16 = y_base_reg3 + dsp_mult;
        shifted = result_q16 >>> FRAC_WIDTH;
        if (shifted > ((1 <<< (DATA_WIDTH-1)) - 1))
            result_sat = {1'b0, {(DATA_WIDTH-1){1'b1}}};
        else if (shifted < -(1 <<< (DATA_WIDTH-1)))
            result_sat = {1'b1, {(DATA_WIDTH-1){1'b0}}};
        else
            result_sat = shifted[DATA_WIDTH-1:0];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_out <= '0;
        end else begin
            y_out <= result_sat;
        end
    end

    // Synthesis info
    // synthesis translate_off
    initial begin
        if (!USE_DSP)
            $info("Activation_Sigmoid_Interpolated: Using LUT-based multiplier (USE_DSP=0). DSP saved: 1");
    end
    // synthesis translate_on

endmodule