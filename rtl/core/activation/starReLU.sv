// =============================================================================
// Activation_StarReLU.sv
//
// Equation: StarReLU(x) = scale · (ReLU(x))² + bias
//
// Pipeline (3 cycles, x_in → y_out):
//   Stage 1: ReLU & Square
//   Stage 2: Scale Multiply
//   Stage 3: Bias Addition
//
// Artix-7 100T Optimization:
//   - USE_DSP defaults to 0 (LUT-based multiplier)
//   - Square and scale operations use LUTs
//   - Only ~32 LUTs per instance for INT8 operations
//   - DSP savings: 2 DSPs per channel
//
// Synthesis Directives:
//   - (* use_dsp = "no" *) by default
//   - Optional DSP for larger FPGAs
// =============================================================================
import pkg_config::*;
import pkg_module_config::*;

module Activation_StarReLU #(
    parameter int DATA_WIDTH = pkg_module_config::UNIFIED_ACTIVATION_DATA_WIDTH,
    parameter int FRAC_WIDTH = pkg_config::FRAC_WIDTH,
    parameter bit  USE_DSP = 0    // Default: LUT-based for Artix-7 100T
)(
    input  logic clk,
    input  logic rst_n,

    // Learnable parameters
    input  logic signed [DATA_WIDTH-1:0] scale_in,
    input  logic signed [DATA_WIDTH-1:0] bias_in,

    input  logic signed [DATA_WIDTH-1:0] x_in,
    output logic signed [DATA_WIDTH-1:0] y_out
);

    localparam logic signed [2*DATA_WIDTH-1:0] ROUND_CONST =
        {{(DATA_WIDTH){1'b0}}, 1'b1, {(DATA_WIDTH-1){1'b0}}};

    // =========================================================================
    // Stage 1: ReLU and Square
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] relu_x;
    logic signed [2*DATA_WIDTH-1:0] sq_raw;
    logic signed [2*DATA_WIDTH-1:0] sq_rounded;
    logic signed [DATA_WIDTH-1:0] sq_d1;
    logic signed [DATA_WIDTH-1:0] scale_d1, bias_d1;

    // Combinational ReLU
    assign relu_x = (x_in[DATA_WIDTH-1] == 1'b1) ? '0 : x_in;

    // Square operation - LUT-based by default
    generate
        if (USE_DSP) begin : gen_sq_dsp
            (* use_dsp = "yes" *)
            always_comb begin
                sq_raw = relu_x * relu_x;
            end
        end else begin : gen_sq_lut
            (* use_dsp = "no" *)
            always_comb begin
                sq_raw = relu_x * relu_x;
            end
        end
    endgenerate

    assign sq_rounded = sq_raw + ROUND_CONST;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sq_d1    <= '0;
            scale_d1 <= '0;
            bias_d1  <= '0;
        end else begin
            sq_d1    <= sq_rounded[2*DATA_WIDTH-1 : DATA_WIDTH];
            scale_d1 <= scale_in;
            bias_d1  <= bias_in;
        end
    end

    // =========================================================================
    // Stage 2: Scale Multiply
    // =========================================================================
    logic signed [2*DATA_WIDTH-1:0] scaled_raw;
    logic signed [2*DATA_WIDTH-1:0] scaled_rounded;
    logic signed [DATA_WIDTH-1:0] scaled_d2;
    logic signed [DATA_WIDTH-1:0] bias_d2;

    generate
        if (USE_DSP) begin : gen_scale_dsp
            (* use_dsp = "yes" *)
            always_comb begin
                scaled_raw = sq_d1 * scale_d1;
            end
        end else begin : gen_scale_lut
            (* use_dsp = "no" *)
            always_comb begin
                scaled_raw = sq_d1 * scale_d1;
            end
        end
    endgenerate

    assign scaled_rounded = scaled_raw + ROUND_CONST;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scaled_d2 <= '0;
            bias_d2   <= '0;
        end else begin
            scaled_d2 <= scaled_rounded[2*DATA_WIDTH-1 : DATA_WIDTH];
            bias_d2   <= bias_d1;
        end
    end

    // =========================================================================
    // Stage 3: Bias Addition (no DSP needed - adder)
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_out <= '0;
        end else begin
            y_out <= scaled_d2 + bias_d2;
        end
    end

    // Synthesis info message
    // synthesis translate_off
    initial begin
        if (!USE_DSP)
            $info("Activation_StarReLU: Using LUT-based multipliers (USE_DSP=0). DSP saved: 2");
    end
    // synthesis translate_on

endmodule