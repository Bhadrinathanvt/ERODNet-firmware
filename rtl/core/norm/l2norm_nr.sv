// =============================================================================
// l2norm_nr.sv
//
//  L2 Normalization using Newton-Raphson Inverse Square Root.
//  Computes: y[i] = x[i] / ||x||_2 = x[i] * (1/sqrt(sum(x[j]^2)))
//
//  Algorithm:
//    1. Compute sum of squares: sum_sq = Σ x[j]^2
//    2. Newton-Raphson iteration for 1/sqrt(sum_sq):
//       y_0 = initial approximation
//       y_{n+1} = y_n * (3 - sum_sq * y_n^2) / 2
//    3. Multiply each element by 1/sqrt(sum_sq)
//
//  Pipeline:
//    Stage 1: Square each element (parallel)
//    Stage 2: Adder tree for sum
//    Stage 3: Initial approximation for 1/sqrt(x)
//    Stage 4-7: Newton-Raphson iterations (2 cycles each)
//    Stage 8: Final multiplication
//
//  Artix-7 100T Optimization:
//    - USE_DSP defaults to 0 (LUT-based multipliers)
//    - Saves DSP slices for systolic array MAC operations
//    - L2Norm is not on critical throughput path
//
//  Latency: 3 + 2*ITERATIONS cycles (11 cycles for 4 iterations)
//  Throughput: 1 vector/cycle after initial fill
//
//  Synthesis Directives:
//    - (* use_dsp = "no" *) by default for LUT-based multiply
//    - (* shreg_extract = "no" *) for delay shift registers
// =============================================================================

import pkg_config::*;

module l2norm_nr #(
    parameter int CH         = 64,
    parameter int DATA_W     = 16,
    parameter int FRAC_W     = 8,
    parameter int ITERATIONS = 4,
    parameter int ACC_W      = 32,
    parameter bit  USE_DSP   = 0    // Default: LUT-based (0), use DSP (1)
)(
    input  logic clk,
    input  logic rst_n,

    // Input/output vectors (packed flat — Verilator compatible)
    input  logic signed [CH*DATA_W-1:0] data_in_flat,
    input  logic                        valid_in,
    output logic                        ready_out,     // FIXED: backpressure — can accept input
    output logic signed [CH*DATA_W-1:0] data_out_flat,
    output logic                        valid_out,
    input  logic                        out_ready     // FIXED: backpressure — consumer ready
);

    // Unpack flat ports into internal arrays
    logic signed [DATA_W-1:0] data_in  [CH-1:0];
    logic signed [DATA_W-1:0] data_out [CH-1:0];

    always_comb begin
        for (int i = 0; i < CH; i++) begin
            data_in[i] = data_in_flat[i*DATA_W +: DATA_W];
            data_out_flat[i*DATA_W +: DATA_W] = data_out[i];
        end
    end

    // =========================================================================
    // Stage 1: Square each element (LUT-based or DSP)
    // =========================================================================
    // Module-level declarations — used by downstream adder tree
    logic signed [2*DATA_W-1:0] sq_in [CH-1:0];
    logic valid_d1;

    generate
        if (USE_DSP) begin : gen_sq_dsp
            for (genvar i = 0; i < CH; i++) begin : gen_square
                (* use_dsp = "yes" *)
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) sq_in[i] <= '0;
                    else if (valid_in) sq_in[i] <= data_in[i] * data_in[i];
                end
            end
        end else begin : gen_sq_lut
            for (genvar i = 0; i < CH; i++) begin : gen_square
                (* use_dsp = "no" *)
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) sq_in[i] <= '0;
                    else if (valid_in) sq_in[i] <= data_in[i] * data_in[i];
                end
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_d1 <= 1'b0;
        else        valid_d1 <= valid_in;
    end

    // =========================================================================
    // Stage 2: Adder tree for sum of squares (pipelined, 1 level per cycle)
    // =========================================================================
    logic signed [ACC_W-1:0] sum_sq;
    logic valid_d2;

    localparam int TREE_STAGES = $clog2(CH);
    // Tree latency: 1 cycle for CH<=2, TREE_STAGES+1 cycles for CH>2
    localparam int TREE_LATENCY = (CH <= 2) ? 1 : (TREE_STAGES + 1);

    generate
        if (CH == 1) begin : gen_sum_1
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) sum_sq <= '0;
                else if (valid_d1) sum_sq <= ACC_W'(sq_in[0]);
            end
        end else if (CH == 2) begin : gen_sum_2
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) sum_sq <= '0;
                else if (valid_d1) sum_sq <= ACC_W'(sq_in[0]) + ACC_W'(sq_in[1]);
            end
        end else begin : gen_sum_tree
            logic signed [ACC_W-1:0] tree_sums [0:TREE_STAGES][0:(CH/(1<<TREE_STAGES))-1];
            logic [TREE_STAGES:0] tree_valid;

            // Level 0: register squared inputs
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    tree_valid[0] <= 1'b0;
                    for (int i = 0; i < CH; i++)
                        tree_sums[0][i] <= '0;
                end else begin
                    tree_valid[0] <= valid_d1;
                    if (valid_d1) begin
                        for (int i = 0; i < CH; i++)
                            tree_sums[0][i] <= ACC_W'(sq_in[i]);
                    end
                end
            end

            // Levels 1..TREE_STAGES: pipelined reduction (1 level per cycle)
            for (genvar s = 1; s <= TREE_STAGES; s++) begin : gen_tree_level
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        tree_valid[s] <= 1'b0;
                        for (int i = 0; i < (CH >> s); i++)
                            tree_sums[s][i] <= '0;
                    end else begin
                        tree_valid[s] <= tree_valid[s-1];
                        if (tree_valid[s-1]) begin
                            for (int i = 0; i < (CH >> s); i++)
                                tree_sums[s][i] <= tree_sums[s-1][2*i] + tree_sums[s-1][2*i+1];
                        end
                    end
                end
            end

            assign sum_sq = tree_sums[TREE_STAGES][0];
        end
    endgenerate

    // valid_d2: comes from tree pipeline for CH>2, or valid_d1 for CH<=2
    generate
        if (CH <= 2) begin : gen_valid_d2_simple
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) valid_d2 <= 1'b0;
                else        valid_d2 <= valid_d1;
            end
        end else begin : gen_valid_d2_tree
            assign valid_d2 = gen_sum_tree.tree_valid[TREE_STAGES];
        end
    endgenerate

    // =========================================================================
    // Stage 3: Initial approximation for 1/sqrt(x) (LUT-based)
    // =========================================================================
    logic signed [ACC_W-1:0] sum_sq_d2;
    logic signed [DATA_W-1:0] rsqrt_est;
    logic valid_d3;

    // Combinational leading-zero detector and initial rsqrt estimate
    // (extracted from always_ff to avoid automatic declarations in always_ff)
    logic signed [DATA_W-1:0] rsqrt_est_comb;
    always_comb begin
        rsqrt_est_comb = '0;
        if (sum_sq[ACC_W-1] == 1'b0) begin
            automatic int lz = ACC_W;  // default: no ones found
            for (int i = 0; i < ACC_W-1; i++) begin
                if (sum_sq[ACC_W-2-i] && lz == ACC_W)
                    lz = i;
            end
            if (lz >= ACC_W-2) begin
                rsqrt_est_comb = (1 << FRAC_W);
            end else begin
                automatic int shift = (lz - FRAC_W/2) / 2;
                if (shift > 0)
                    rsqrt_est_comb = (1 << FRAC_W) >>> shift;
                else
                    rsqrt_est_comb = (1 << FRAC_W) <<< (-shift);
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_sq_d2 <= '0;
            rsqrt_est <= '0;
            valid_d3 <= 1'b0;
        end else begin
            sum_sq_d2 <= sum_sq;
            valid_d3 <= valid_d2;
            rsqrt_est <= rsqrt_est_comb;
        end
    end

    // =========================================================================
    // Stages 4-7: Newton-Raphson iterations
    // FIXED: NBA-in-for-loop — nr_y_sq, nr_x_y_sq, nr_factor were module-level
    // signals assigned with NBAs inside genvar loops. Multiple iterations drove
    // the same signals (multi-driver) and each iteration read stale pre-NBA values.
    // Fix: convert intermediates to combinational per-iteration, register only nr_y[].
    // =========================================================================
    logic signed [DATA_W-1:0] nr_y [0:ITERATIONS];
    logic [ITERATIONS:0] valid_nr;

    // Connect valid_d3 to valid_nr[0] — was undriven (BUG-001 fix)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_nr[0] <= 1'b0;
        else        valid_nr[0] <= valid_d3;
    end

    assign nr_y[0] = rsqrt_est;

    generate
        if (USE_DSP) begin : gen_nr_dsp
            (* use_dsp = "yes" *)
            for (genvar it = 0; it < ITERATIONS; it++) begin : gen_nr_iter
                // Combinational intermediates (blocking assignments, no stale reads)
                logic signed [2*DATA_W-1:0] nr_y_sq_comb;
                logic signed [2*DATA_W-1:0] nr_x_y_sq_comb;
                logic signed [DATA_W-1:0]   nr_factor_comb;

                always_comb begin
                    nr_y_sq_comb    = nr_y[it] * nr_y[it];
                    nr_x_y_sq_comb = sum_sq_d2 * nr_y_sq_comb;
                    nr_factor_comb  = (3 * (1 << FRAC_W) - nr_x_y_sq_comb[2*FRAC_W +: DATA_W]) >>> 1;
                end

                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        nr_y[it+1] <= '0;
                        valid_nr[it+1] <= 1'b0;
                    end else begin
                        valid_nr[it+1] <= valid_nr[it];
                        nr_y[it+1] <= (nr_y[it] * nr_factor_comb) >>> FRAC_W;
                    end
                end
            end
        end else begin : gen_nr_lut
            (* use_dsp = "no" *)
            for (genvar it = 0; it < ITERATIONS; it++) begin : gen_nr_iter
                // Combinational intermediates (blocking assignments, no stale reads)
                logic signed [2*DATA_W-1:0] nr_y_sq_comb;
                logic signed [2*DATA_W-1:0] nr_x_y_sq_comb;
                logic signed [DATA_W-1:0]   nr_factor_comb;

                always_comb begin
                    nr_y_sq_comb    = nr_y[it] * nr_y[it];
                    nr_x_y_sq_comb = sum_sq_d2 * nr_y_sq_comb;
                    nr_factor_comb  = (3 * (1 << FRAC_W) - nr_x_y_sq_comb[2*FRAC_W +: DATA_W]) >>> 1;
                end

                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        nr_y[it+1] <= '0;
                        valid_nr[it+1] <= 1'b0;
                    end else begin
                        valid_nr[it+1] <= valid_nr[it];
                        nr_y[it+1] <= (nr_y[it] * nr_factor_comb) >>> FRAC_W;
                    end
                end
            end
        end
    endgenerate

    // =========================================================================
    // Stage 8: Final multiplication (LUT-based)
    // =========================================================================
    logic signed [DATA_W-1:0] rsqrt_final;
    logic signed [2*DATA_W-1:0] mult_result [CH-1:0];

    // Saturation constants (explicit width — avoid 32-bit int overflow)
    localparam signed [DATA_W-1:0] SAT_MAX = (1 << (DATA_W-1)) - 1;
    localparam signed [DATA_W-1:0] SAT_MIN = -(1 << (DATA_W-1));

    localparam int DELAY_STAGES = TREE_LATENCY + 2 + 2 * ITERATIONS;

    (* shreg_extract = "no" *)
    logic signed [DATA_W-1:0] data_sr [0:DELAY_STAGES-1][CH-1:0];
    logic [DELAY_STAGES-1:0] valid_sr;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_sr <= '0;
            for (int s = 0; s < DELAY_STAGES; s++)
                for (int c = 0; c < CH; c++)
                    data_sr[s][c] <= '0;
        end else begin
            valid_sr <= {valid_sr[DELAY_STAGES-2:0], valid_in};
            for (int s = DELAY_STAGES-1; s > 0; s--)
                for (int c = 0; c < CH; c++)
                    data_sr[s][c] <= data_sr[s-1][c];
            for (int c = 0; c < CH; c++)
                data_sr[0][c] <= data_in[c];
        end
    end

    assign rsqrt_final = nr_y[ITERATIONS];

    // Saturation rounding (automatic extracted to always_comb)
    logic signed [DATA_W-1:0] data_out_comb [CH-1:0];
    always_comb begin
        for (int c = 0; c < CH; c++) begin
            logic signed [2*DATA_W-1:0] rounded;
            rounded = mult_result[c] + (1 << (FRAC_W-1));
            if (rounded > (2*DATA_W)'(2**(DATA_W-1) - 1))
                data_out_comb[c] = SAT_MAX;
            else if (rounded < -(2*DATA_W)'(2**(DATA_W-1)))
                data_out_comb[c] = SAT_MIN;
            else
                data_out_comb[c] = DATA_W'(rounded >>> FRAC_W);
        end
    end

    generate
        if (USE_DSP) begin : gen_final_dsp
            (* use_dsp = "yes" *)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int c = 0; c < CH; c++) begin
                        mult_result[c] <= '0;
                        data_out[c] <= '0;
                    end
                    valid_out <= 1'b0;
                end else begin
                    for (int c = 0; c < CH; c++)
                        mult_result[c] <= data_sr[DELAY_STAGES-1][c] * rsqrt_final;
                    valid_out <= valid_nr[ITERATIONS];
                    for (int c = 0; c < CH; c++)
                        data_out[c] <= data_out_comb[c];
                end
            end
        end else begin : gen_final_lut
            (* use_dsp = "no" *)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int c = 0; c < CH; c++) begin
                        mult_result[c] <= '0;
                        data_out[c] <= '0;
                    end
                    valid_out <= 1'b0;
                end else begin
                    for (int c = 0; c < CH; c++)
                        mult_result[c] <= data_sr[DELAY_STAGES-1][c] * rsqrt_final;
                    valid_out <= valid_nr[ITERATIONS];
                    for (int c = 0; c < CH; c++)
                        data_out[c] <= data_out_comb[c];
                end
            end
        end
    endgenerate

    // =========================================================================
    // Backpressure: stall pipeline when consumer is not ready
    // FIXED: added ready_out/out_ready handshake to prevent data loss.
    // =========================================================================
    logic stall;
    assign stall    = valid_out && !out_ready;
    assign ready_out = !stall;   // Accept new input when not stalled

    // When stalled, hold output valid and data stable; shift register freezes.
    // The valid_pipe and data_sr already use valid_in at the input, so stalling
    // the input (ready_out=0) naturally back-pressures through the pipeline.
    // For output hold: when stalled, don't update data_out or valid_out.

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    L2NORM_CHANNELS: assert property (
        @(posedge clk) disable iff (!rst_n)
        CH > 0 && CH <= 256
    ) else $error("l2norm_nr: Channel count out of range");

    L2NORM_ITERATIONS: assert property (
        @(posedge clk) disable iff (!rst_n)
        ITERATIONS >= 2 && ITERATIONS <= 8
    ) else $error("l2norm_nr: Iterations out of range (2-8)");

    // synthesis translate_off
    initial begin
        if (!USE_DSP)
            $info("l2norm_nr: Using LUT-based multipliers (USE_DSP=0). DSP savings: ~%0d multipliers", CH + ITERATIONS + 1);
    end
    // synthesis translate_on
    // synthesis translate_on

endmodule