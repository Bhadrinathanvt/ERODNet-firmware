// =============================================================================
// Activation_GELU_Interpolated.sv
//
// Piecewise-linear interpolation of GELU:
//   GELU(x) = x * Φ(x) where Φ(x) is the standard normal CDF
//
// Approximation: GELU(x) ≈ y_base[index] + slope[index] * (x - x_base[index])
//
// Index extraction:
//   The ROM uses INDEX_START_BIT=12 in a 32-bit Q16.16 representation.
//   When DATA_WIDTH < ROM_WIDTH, the narrow input is sign-extended and
//   shifted left by FRAC_WIDTH before index/delta extraction.
//
// Artix-7 100T Optimization:
//   - USE_DSP defaults to 0 (LUT-based interpolation)
//   - ROM lookup uses BRAM, multiplier uses LUTs
//   - Interpolation multiply is small (16-bit × 16-bit)
//   - DSP savings: 1 DSP per channel
//
// Pipeline: 3 cycles
//   Stage 1: ROM read (y_base and slope)
//   Stage 2: Multiply (slope × delta) - LUT-based
//   Stage 3: Add (y_base + product), truncate to DATA_WIDTH
// =============================================================================
import pkg_config::*;
import pkg_module_config::*;

module Activation_GELU_Interpolated #(
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
    // ROM_WIDTH/INDEX_START_BIT from pkg_config (config.yaml::activation_rom).
    // Tied to .mem file geometry — regenerate via scripts/generate_roms.py if changed.
    localparam int ROM_WIDTH       = pkg_config::ROM_WIDTH;
    localparam int INDEX_START_BIT = pkg_config::INDEX_START_BIT;
    localparam int DELTA_BITS     = ROM_WIDTH - INDEX_START_BIT - INDEX_BITS;

    // Sign-extend x_in to ROM_WIDTH, then shift left by FRAC_WIDTH
    // to convert from INT (or narrow Q-format) to Q16.16 representation.
    logic signed [ROM_WIDTH-1:0] x_in_scaled;
    assign x_in_scaled = { {(ROM_WIDTH-DATA_WIDTH){x_in[DATA_WIDTH-1]}}, x_in } <<< FRAC_WIDTH;

    // Index and delta extraction from scaled Q16.16 representation
    logic [INDEX_BITS-1:0]  index;
    logic signed [DELTA_BITS-1:0] delta_x;

    assign index   = x_in_scaled[INDEX_START_BIT+INDEX_BITS-1 : INDEX_START_BIT];
    assign delta_x = x_in_scaled[INDEX_START_BIT-1 : 0];

    // =========================================================================
    // Stage 1: Register external ROM data + delta_x
    // =========================================================================
    logic signed [ROM_WIDTH-1:0] y_base_d1;
    logic signed [ROM_WIDTH-1:0] slope_d1;
    logic signed [DELTA_BITS-1:0] delta_d1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_base_d1 <= '0;
            slope_d1  <= '0;
            delta_d1  <= '0;
        end else begin
            y_base_d1 <= y_base_in;
            slope_d1  <= slope_in;
            delta_d1  <= delta_x;
        end
    end

    // =========================================================================
    // Stage 2: Interpolation Multiply (LUT-based)
    // =========================================================================
    logic signed [2*ROM_WIDTH-1:0] prod_d2;
    logic signed [ROM_WIDTH-1:0] y_base_d2;

    generate
        if (USE_DSP) begin : gen_interp_dsp
            (* use_dsp = "yes" *)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    prod_d2 <= '0;
                    y_base_d2 <= '0;
                end else begin
                    prod_d2 <= slope_d1 * delta_d1;
                    y_base_d2 <= y_base_d1;
                end
            end
        end else begin : gen_interp_lut
            (* use_dsp = "no" *)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    prod_d2 <= '0;
                    y_base_d2 <= '0;
                end else begin
                    // LUT-based multiply (saves DSP)
                    prod_d2 <= slope_d1 * delta_d1;
                    y_base_d2 <= y_base_d1;
                end
            end
        end
    endgenerate

    // =========================================================================
    // Stage 3: Final Add + Truncate to DATA_WIDTH
    // =========================================================================
    // Rounding constant for fractional alignment (ROM_WIDTH arithmetic)
    localparam logic [2*ROM_WIDTH-1:0] ROUND_CONST =
        {{(2*ROM_WIDTH-FRAC_WIDTH){1'b0}}, 1'b1, {(FRAC_WIDTH-1){1'b0}}};

    logic signed [2*ROM_WIDTH-1:0] prod_rounded;
    logic signed [ROM_WIDTH-1:0]    corr;
    logic signed [ROM_WIDTH-1:0]    result_q16;
    logic signed [DATA_WIDTH-1:0]   result_sat;

    assign prod_rounded = prod_d2 + ROUND_CONST;
    assign corr = prod_rounded[ROM_WIDTH+FRAC_WIDTH-1:FRAC_WIDTH];

    always_ff @(posedge clk or negedge rst_n) begin
        logic signed [ROM_WIDTH-1:0] shifted;
        if (!rst_n) begin
            y_out <= '0;
        end else begin
            result_q16 = y_base_d2 + corr;
            // Arithmetic right-shift by FRAC_WIDTH to extract integer part
            // then saturate to DATA_WIDTH range.
            shifted = result_q16 >>> FRAC_WIDTH;
            if (shifted > ((1 <<< (DATA_WIDTH-1)) - 1))
                result_sat = {1'b0, {(DATA_WIDTH-1){1'b1}}};   // +MAX
            else if (shifted < -(1 <<< (DATA_WIDTH-1)))
                result_sat = {1'b1, {(DATA_WIDTH-1){1'b0}}};   // -MAX-1
            else
                result_sat = shifted[DATA_WIDTH-1:0];
            y_out <= result_sat;
        end
    end

    // Synthesis info
    // synthesis translate_off
    initial begin
        if (!USE_DSP)
            $info("Activation_GELU_Interpolated: Using LUT-based multiplier (USE_DSP=0). DSP saved: 1");
    end
    // synthesis translate_on

endmodule