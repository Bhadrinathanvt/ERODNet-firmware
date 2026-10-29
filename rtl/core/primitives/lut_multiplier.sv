// =============================================================================
// lut_multiplier.sv
//
// LUT-based multiplier for resource-constrained FPGA implementations.
// Uses distributed RAM (LUTs) instead of DSP48 slices.
//
// Use cases:
//   - LayerNorm/BatchNorm scale operations (not critical throughput path)
//   - Activation function interpolation
//   - Low-precision multiplications (INT8-INT16)
//   - When DSP budget is exhausted
//
// Trade-offs:
//   - LUT cost: ~4 LUTs per bit of operand width
//   - Latency: Same as DSP (1 cycle) for registered output
//   - Throughput: 1 result per cycle
//   - Power: Lower than DSP for small bit-widths
//
// Resource Usage (estimated):
//   - INT8 × INT8:   ~32 LUTs, 0 DSP
//   - INT16 × INT16: ~128 LUTs, 0 DSP
//   - INT32 × INT32: ~512 LUTs, 0 DSP
// =============================================================================

package lut_multiplier_pkg;
    // LUT-based multiplier style selection
    typedef enum logic [1:0] {
        LUT_AUTO = 2'b00,    // Auto-select based on width
        LUT_FORCE = 2'b01,   // Force LUT implementation
        DSP_FORCE = 2'b10    // Force DSP implementation
    } mult_style_e;
endpackage

import pkg_config::*;

/* verilator lint_off DECLFILENAME */

// =============================================================================
// lut_mult_signed - Signed LUT-based multiplier
// =============================================================================
module lut_mult_signed #(
    parameter int A_W = 16,
    parameter int B_W = 16,
    parameter int OUT_W = A_W + B_W,
    parameter bit  REG_OUT = 1,
    parameter bit  USE_DSP = 0      // Override: 1=force DSP, 0=use LUT
)(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  logic signed [A_W-1:0] a,
    input  logic signed [B_W-1:0] b,
    output logic signed [OUT_W-1:0] result,
    output logic valid_out
);

    // Pipeline delay
    localparam int PIPELINE_STAGES = REG_OUT ? 1 : 0;

    generate
        if (USE_DSP) begin : gen_dsp_mult
            // Force DSP implementation for comparison or fallback
            (* use_dsp = "yes" *)
            logic signed [OUT_W-1:0] mult_result;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    mult_result <= '0;
                end else if (valid_in) begin
                    mult_result <= a * b;
                end
            end

            assign result = mult_result;

        end else begin : gen_lut_mult
            // LUT-based multiplier using carry-chain
            (* use_dsp = "no" *)
            logic signed [OUT_W-1:0] mult_result;

            logic signed [A_W-1:0] a_abs;
            logic signed [B_W-1:0] b_abs;
            logic result_sign;

            // Get absolute values and sign
            assign a_abs = a[A_W-1] ? -a : a;
            assign b_abs = b[B_W-1] ? -b : b;
            assign result_sign = a[A_W-1] ^ b[B_W-1];

            // Unsigned multiplication on absolute values
            (* use_dsp = "no" *)
            logic unsigned [A_W+B_W-1:0] mult_unsigned;

            // Generate multiplication using LUTs
            always_comb begin
                mult_unsigned = '0;
                for (int i = 0; i < B_W; i++) begin
                    if (b_abs[i]) begin
                        mult_unsigned = mult_unsigned + ((A_W+B_W)'(unsigned'(a_abs)) << i);
                    end
                end
            end

            // Apply sign
            assign mult_result = result_sign ? -signed'(OUT_W'(mult_unsigned)) : signed'(OUT_W'(mult_unsigned));

            if (REG_OUT) begin : gen_reg
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n)
                        result <= '0;
                    else if (valid_in)
                        result <= mult_result;
                end
            end else begin : gen_no_reg
                assign result = mult_result;
            end
        end
    endgenerate

    // Valid pipeline
    logic [PIPELINE_STAGES:0] valid_pipe;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_pipe <= '0;
        else
            valid_pipe <= {valid_pipe[PIPELINE_STAGES-1:0], valid_in};
    end

    assign valid_out = valid_pipe[PIPELINE_STAGES];

endmodule

// =============================================================================
// lut_mult_accum - LUT-based multiply-accumulate (no DSP)
// =============================================================================
module lut_mult_accum #(
    parameter int A_W = 16,
    parameter int B_W = 16,
    parameter int ACC_W = 32,
    parameter bit  USE_DSP = 0
)(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  logic signed [A_W-1:0] a,
    input  logic signed [B_W-1:0] b,
    input  logic signed [ACC_W-1:0] accum_in,
    input  logic clear_acc,
    output logic signed [ACC_W-1:0] accum_out,
    output logic valid_out
);

    // Pipeline stages
    logic valid_d1;

    // Stage 1: Multiply
    logic signed [A_W+B_W-1:0] mult_result;

    generate
        if (USE_DSP) begin : gen_dsp_mac
            (* use_dsp = "yes" *)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    mult_result <= '0;
                else if (valid_in)
                    mult_result <= a * b;
            end
        end else begin : gen_lut_mac
            (* use_dsp = "no" *)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    mult_result <= '0;
                else if (valid_in)
                    mult_result <= a * b;
            end
        end
    endgenerate

    // Stage 2: Accumulate
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum_out <= '0;
            valid_out <= '0;
        end else begin
            valid_out <= valid_d1;
            if (clear_acc)
                accum_out <= ACC_W'(mult_result);
            else if (valid_d1)
                accum_out <= accum_in + ACC_W'(mult_result);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_d1 <= '0;
        else
            valid_d1 <= valid_in;
    end

endmodule

// =============================================================================
// lut_dot_product - LUT-based dot product (vector × vector)
// =============================================================================
module lut_dot_product #(
    parameter int NUM_ELEMENTS = 8,
    parameter int DATA_W = 8,
    parameter int ACC_W = 32,
    parameter bit  USE_DSP = 0   // Use DSP only if set
)(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  logic signed [DATA_W-1:0] vec_a [NUM_ELEMENTS-1:0],
    input  logic signed [DATA_W-1:0] vec_b [NUM_ELEMENTS-1:0],
    output logic signed [ACC_W-1:0] result,
    output logic valid_out
);

    localparam int TREE_STAGES = $clog2(NUM_ELEMENTS);
    localparam int MULT_W = 2 * DATA_W;

    // Stage 1: Parallel multiplications
    logic signed [MULT_W-1:0] mult_results [NUM_ELEMENTS-1:0];
    logic valid_d1;

    generate
        for (genvar i = 0; i < NUM_ELEMENTS; i++) begin : gen_mult
            (* use_dsp = USE_DSP ? "yes" : "no" *)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    mult_results[i] <= '0;
                else if (valid_in)
                    mult_results[i] <= vec_a[i] * vec_b[i];
            end
        end
    endgenerate

    // Stage 2+: Tree reduction (adder tree)
    logic signed [ACC_W-1:0] tree [TREE_STAGES+1][NUM_ELEMENTS-1:0];
    logic [TREE_STAGES:0] valid_tree;

    // Initialize tree with multiplication results
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < NUM_ELEMENTS; i++)
                tree[0][i] <= '0;
            valid_tree[0] <= '0;
        end else begin
            for (int i = 0; i < NUM_ELEMENTS; i++)
                tree[0][i] <= ACC_W'(mult_results[i]);
            valid_tree[0] <= valid_d1;
        end
    end

    // Tree reduction stages
    generate
        for (genvar s = 0; s < TREE_STAGES; s++) begin : gen_tree_stage
            localparam int WIDTH_S = NUM_ELEMENTS / (2 ** s);
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int i = 0; i < WIDTH_S/2; i++)
                        tree[s+1][i] <= '0;
                    valid_tree[s+1] <= '0;
                end else begin
                    for (int i = 0; i < WIDTH_S/2; i++)
                        tree[s+1][i] <= tree[s][2*i] + tree[s][2*i+1];
                    valid_tree[s+1] <= valid_tree[s];
                end
            end
        end
    endgenerate

    assign result = tree[TREE_STAGES][0];
    assign valid_out = valid_tree[TREE_STAGES];

    // Valid pipeline for stage 1
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_d1 <= '0;
        else
            valid_d1 <= valid_in;
    end

endmodule

// =============================================================================
// Resource Usage Estimator
// =============================================================================
function automatic int estimate_lut_multiplier_resources(
    input int a_width,
    input int b_width,
    input bit use_dsp
);
    int lut_count;
    int dsp_count;
    begin
        if (use_dsp) begin
            // DSP usage
            lut_count = 0;
            // Estimate DSP count based on bit-width
            if (a_width <= 25 && b_width <= 18)
                dsp_count = 1;
            else
                dsp_count = ((a_width + 24) / 25) * ((b_width + 17) / 18);
        end else begin
            // LUT usage (approximately 4 LUTs per bit pair for signed)
            lut_count = (a_width * b_width) / 2;  // Rough estimate
            dsp_count = 0;
        end
        estimate_lut_multiplier_resources = lut_count;
    end
endfunction

// =============================================================================
// Artix-7 100T Resource Budget Check
// =============================================================================
module resource_budget_check #(
    parameter int DSP_COUNT = 0,
    parameter int LUT_COUNT = 0,
    parameter int BRAM_COUNT = 0
)(
    output logic budget_ok
);

    // Arty A7-100T constraints
    localparam int MAX_DSP = 240;
    localparam int MAX_LUT = 63400;
    localparam int MAX_BRAM = 135;
    localparam int MARGIN_PCT = 10;

    localparam int DSP_BUDGET = MAX_DSP * (100 - MARGIN_PCT) / 100;  // 216
    localparam int LUT_BUDGET = MAX_LUT * (100 - MARGIN_PCT) / 100;  // 57060
    localparam int BRAM_BUDGET = MAX_BRAM * (100 - MARGIN_PCT) / 100;  // 121

    function automatic bit check_budget();
        bit ok;
        ok = (DSP_COUNT <= DSP_BUDGET) &&
             (LUT_COUNT <= LUT_BUDGET) &&
             (BRAM_COUNT <= BRAM_BUDGET);
        return ok;
    endfunction

    // Synthesis-time check
    initial begin
        if (!check_budget())
            $warning("Resource budget exceeded: DSP=%0d/%0d, LUT=%0d/%0d, BRAM=%0d/%0d",
                      DSP_COUNT, DSP_BUDGET, LUT_COUNT, LUT_BUDGET, BRAM_COUNT, BRAM_BUDGET);
    end

    assign budget_ok = check_budget();

endmodule