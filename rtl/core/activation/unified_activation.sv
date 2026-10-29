// =============================================================================
//  unified_activation_lane.sv
//
//  Single activation lane for one element of the output matrix.
//
//  Pipeline structure (4-cycle latency, input → registered y_out):
//    Stage 0  : x_in presented; sigmoid/GELU/StarReLU lookup/math starts
//    Stage 1  : x_d[0] = x_in;  submodule pipe 1
//    Stage 2  : x_d[1] = x_in;  submodule pipe 2
//    Stage 3  : x_d[2] = x_in;  submodule pipe 3 → outputs valid
//    Stage 4  : y_out registered (mux from combinational activations on x_d[2]
//               or registered submodule outputs)
//
//  Combinational activations (ReLU, LeakyReLU, ReLU6) operate on x_d[2] and
//  are captured into y_out at stage 4.  Sigmoid, GELU, and StarReLU have
//  3-cycle submodule pipelines whose registered outputs are captured at stage 4.
//
//  act_sel encoding (3 bits):
//    3'b000 = ReLU
//    3'b001 = Leaky ReLU
//    3'b010 = ReLU6
//    3'b011 = Sigmoid    (3-cycle interpolated ROM)
//    3'b100 = GELU       (3-cycle interpolated ROM, clean pipeline)
//    3'b101 = Star ReLU  (3-cycle DSP math pipeline)
//    3'b111 = Reserved   (bypass)
//
//  Synthesis Directives:
//    - DSP inference in submodule pipelines (sigmoid, GELU, StarReLU)
//    - Block RAM for activation LUTs
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;

module unified_activation_lane #(
    parameter int DATA_WIDTH = pkg_module_config::UNIFIED_ACTIVATION_DATA_WIDTH,
    parameter int FRAC_WIDTH = pkg_config::FRAC_WIDTH,
    // ROM_WIDTH from pkg_config (config.yaml::activation_rom). The shared
    // sigmoid/GELU ROMs emit y_base/slope at this width.
    parameter int ROM_WIDTH  = pkg_config::ROM_WIDTH,
    // ── Hardware Generation Flags ──
    parameter bit EN_SIGMOID   = 1,
    parameter bit EN_GELU      = 1,
    parameter bit EN_STAR_RELU = 1,
    parameter bit USE_DSP      = 0   // Default: LUT-based for Artix-7 100T
)(
    input  logic clk,
    input  logic rst_n,

    // ── Handshake ────────────────────────────────────────────────────────────
    input  logic        in_valid,
    output logic        ready_in,       // Can accept new input (backpressure)
    output logic        out_valid,
    input  logic        out_ready,       // Consumer ready for output

    // ── Control ──────────────────────────────────────────────────────────────
    input  logic        act_enable,
    input  logic [2:0]  act_sel,        
    input  logic [3:0]  leak_shift,
    input  logic signed [DATA_WIDTH-1:0] saturation_in,
    input  logic signed [DATA_WIDTH-1:0] scale_in, 
    input  logic signed [DATA_WIDTH-1:0] bias_in, 

    // ── Data ─────────────────────────────────────────────────────────────────
    input  logic signed [DATA_WIDTH-1:0] x_in,
    output logic signed [DATA_WIDTH-1:0] y_out,

    // ── Shared ROM inputs (from shared_activation_roms) ──────────────────────
    // ROM values are in Q16.16 format. Activation submodules interpolate at
    // ROM_WIDTH (parameter, defaults to pkg_config::ROM_WIDTH) and truncate
    // to DATA_WIDTH at the output.
    input  logic signed [ROM_WIDTH-1:0] sigmoid_y_base_in,
    input  logic signed [ROM_WIDTH-1:0] sigmoid_slope_in,
    input  logic signed [ROM_WIDTH-1:0] gelu_y_base_in,
    input  logic signed [ROM_WIDTH-1:0] gelu_slope_in
);

    // =========================================================================
    //  Sigmoid submodule (Conditional Synthesis)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] sigmoid_out;

    generate
        if (EN_SIGMOID) begin : gen_sigmoid
            Activation_Sigmoid_Interpolated #(
                .DATA_WIDTH (DATA_WIDTH),
                .FRAC_WIDTH (FRAC_WIDTH),
                .INDEX_BITS (pkg_config::INDEX_BITS)
            ) u_sigmoid (
                .clk        (clk),
                .rst_n      (rst_n),
                .x_in       (x_in),
                .y_base_in  (sigmoid_y_base_in),
                .slope_in   (sigmoid_slope_in),
                .y_out      (sigmoid_out)
            );
        end else begin : gen_no_sigmoid
            assign sigmoid_out = '0;
        end
    endgenerate

    // =========================================================================
    //  GELU submodule (Conditional Synthesis)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] gelu_out;

    generate
        if (EN_GELU) begin : gen_gelu
            Activation_GELU_Interpolated #(
                .DATA_WIDTH (DATA_WIDTH),
                .FRAC_WIDTH (FRAC_WIDTH),
                .INDEX_BITS (pkg_config::INDEX_BITS)
            ) u_gelu (
                .clk        (clk),
                .rst_n      (rst_n),
                .x_in       (x_in),
                .y_base_in  (gelu_y_base_in),
                .slope_in   (gelu_slope_in),
                .y_out      (gelu_out)
            );
        end else begin : gen_no_gelu
            assign gelu_out = '0;
        end
    endgenerate

    // =========================================================================
    //  Star ReLU submodule (Conditional Synthesis)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] star_relu_out;

    generate
        if (EN_STAR_RELU) begin : gen_star_relu
            Activation_StarReLU #(
                .DATA_WIDTH (DATA_WIDTH),
                .FRAC_WIDTH (FRAC_WIDTH)
            ) u_star_relu (
                .clk      (clk),
                .rst_n    (rst_n),
                .scale_in (scale_in),
                .bias_in  (bias_in),
                .x_in     (x_in),
                .y_out    (star_relu_out)
            );
        end else begin : gen_no_star_relu
            assign star_relu_out = '0;
        end
    endgenerate

    // =========================================================================
    //  Backpressure / Stall Logic
    // =========================================================================
    logic stall;
    assign stall    = out_valid && !out_ready;
    assign ready_in = !stall;

    // =========================================================================
    //  3-stage data + control pipeline
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] x_d [2:0];
    logic [3:0]  v_d;
    logic [2:0]  en_d;
    logic [2:0]  sel_d [2:0];
    logic [3:0]  leak_shift_d [2:0];
    logic signed [DATA_WIDTH-1:0] saturation_in_d [2:0];
    logic signed [DATA_WIDTH-1:0] scale_in_reg, bias_in_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_d[0]   <= '0; x_d[1]   <= '0; x_d[2]   <= '0;
            v_d      <= '0;
            en_d     <= '0;
            sel_d[0] <= '0; sel_d[1] <= '0; sel_d[2] <= '0;
            leak_shift_d[0] <= '0; leak_shift_d[1] <= '0; leak_shift_d[2] <= '0;
            saturation_in_d[0] <= '0; saturation_in_d[1] <= '0; saturation_in_d[2] <= '0;
            scale_in_reg <= '0;
            bias_in_reg <= '0;
            y_out    <= '0;
        end else if (!stall) begin
            // ── Data shift ───────────────────────────────────────────────
            x_d[0] <= x_in;
            x_d[1] <= x_d[0];
            x_d[2] <= x_d[1];

            // ── Valid shift (4 stages, matches y_out latency) ─────────
            v_d <= {v_d[2:0], in_valid};

            // ── Control shift (3 stages, aligned with x_d[2]) ────────
            // Latch control signals only when valid data enters the pipeline.
            // Hold previous values during idle cycles so CSR changes mid-stream
            // don't corrupt the activation selection for in-flight tokens.
            en_d     <= {en_d[1:0],  in_valid ? act_enable : en_d[0]};
            sel_d[0] <= in_valid ? act_sel : sel_d[0];
            sel_d[1] <= sel_d[0];
            sel_d[2] <= sel_d[1];
            leak_shift_d[0] <= in_valid ? leak_shift : leak_shift_d[0];
            leak_shift_d[1] <= leak_shift_d[0];
            leak_shift_d[2] <= leak_shift_d[1];
            saturation_in_d[0] <= in_valid ? saturation_in : saturation_in_d[0];
            saturation_in_d[1] <= saturation_in_d[0];
            saturation_in_d[2] <= saturation_in_d[1];
            scale_in_reg <= in_valid ? scale_in : scale_in_reg;
            bias_in_reg  <= in_valid ? bias_in  : bias_in_reg;

            // ── Output mux (stage-4 register) ────────────────────────
            if (!en_d[2]) begin
                y_out <= x_d[2];            // bypass
            end else begin
                case (sel_d[2])
                    3'b000:   y_out <= relu_comb;      // ReLU
                    3'b001:   y_out <= leaky_comb;     // Leaky ReLU
                    3'b010:   y_out <= relu6_comb;     // ReLU6
                    3'b011:   y_out <= sigmoid_out;    // Sigmoid
                    3'b100:   y_out <= gelu_out;       // GELU
                    3'b101:   y_out <= star_relu_out;  // Star ReLU
                    default:  y_out <= x_d[2];         // reserved → bypass
                endcase
            end
        end
        // else: stall — hold all pipeline registers stable
    end

    // out_valid tracks y_out latency (4 cycles after in_valid)
    assign out_valid = v_d[3];

    // =========================================================================
    //  Combinational activation functions on x_d[2]
    // =========================================================================

    // ReLU: max(0, x)
    logic signed [DATA_WIDTH-1:0] relu_comb;
    assign relu_comb = x_d[2][DATA_WIDTH-1] ? '0 : x_d[2];

    // Leaky ReLU: x if x≥0, else x >>> leak_shift
    // Bounded case instead of barrel shifter — synthesis optimizes each constant shift.
    // Shifts ≥ DATA_WIDTH produce sign-extension (all bits = sign).
    logic signed [DATA_WIDTH-1:0] leaky_comb;
    logic signed [DATA_WIDTH-1:0] leaky_neg;
    always_comb begin
        case (leak_shift_d[2])
            4'd0: leaky_neg = x_d[2] >>> 0;
            4'd1: leaky_neg = x_d[2] >>> 1;
            4'd2: leaky_neg = x_d[2] >>> 2;
            4'd3: leaky_neg = x_d[2] >>> 3;
            4'd4: leaky_neg = x_d[2] >>> 4;
            4'd5: leaky_neg = x_d[2] >>> 5;
            4'd6: leaky_neg = x_d[2] >>> 6;
            4'd7: leaky_neg = x_d[2] >>> 7;
            default: leaky_neg = {DATA_WIDTH{x_d[2][DATA_WIDTH-1]}};
        endcase
    end
    assign leaky_comb = x_d[2][DATA_WIDTH-1] ? leaky_neg : x_d[2];

    // ReLU6: clamp to [0, saturation_in]
    logic signed [DATA_WIDTH-1:0] relu6_comb;
    assign relu6_comb = x_d[2][DATA_WIDTH-1]     ? '0
                      : (x_d[2] > saturation_in_d[2]) ? saturation_in_d[2]
                      :                             x_d[2];

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    ACT_VALID_STABLE: assert property(@(posedge clk) disable iff (!rst_n)
        (out_valid && !out_ready) |=> out_valid)
        else $error("unified_activation_lane: out_valid deasserted before out_ready");

    ACT_DATA_STABLE: assert property(@(posedge clk) disable iff (!rst_n)
        (out_valid && !out_ready) |=> $stable(y_out))
        else $error("unified_activation_lane: y_out changed while out_valid held without out_ready");
    // synthesis translate_on

endmodule
