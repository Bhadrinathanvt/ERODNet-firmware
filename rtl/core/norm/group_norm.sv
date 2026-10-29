// =============================================================================
// group_norm.sv
//
// Group Normalization (per-token, per-group statistics).
//
// For each input token of MAX_C channels split into GROUPS groups of
// (MAX_C/GROUPS) channels each, compute per-group mean and variance,
// then apply (x - mean) / sqrt(var + eps) * gamma + beta per channel.
//
// Differs from LayerNorm by computing stats over a *subset* of channels
// (one group) instead of all channels, and from BatchNorm by computing
// stats per-token rather than per-batch.
//
// Pipeline: 8 stages. Reuses the inverse-sqrt LUT used by layerNorm_fixed
// (rtl/ln_inv_sqrt_base.mem / ln_inv_sqrt_slope.mem).
//
// Synthesis Directives:
//   - (* rom_style = "block" *) for inverse sqrt LUT
//   - (* use_dsp = "no" *) by default — preserves DSP budget for the
//     systolic array on Artix-7-100T (240 DSP48E1).
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;

module group_norm #(
    parameter int DATA_WIDTH = pkg_module_config::GROUP_NORM_DATA_WIDTH,
    parameter int FRAC_WIDTH = pkg_config::FRAC_WIDTH,
    parameter int MAX_C      = 64,
    parameter int GROUPS     = 8,    // MAX_C must be divisible by GROUPS
    parameter bit  USE_DSP   = 0
)(
    input  logic clk,
    input  logic rst_n,

    input  logic [15:0] cfg_c,        // runtime channel count (<= MAX_C)

    input  logic                              valid_in,
    output logic                              ready_in,
    output logic                              valid_out,
    input  logic                              out_ready,

    input  logic signed [DATA_WIDTH-1:0]      gamma [0:MAX_C-1],
    input  logic signed [DATA_WIDTH-1:0]      beta  [0:MAX_C-1],

    input  logic signed [DATA_WIDTH-1:0]      x_in  [0:MAX_C-1],
    output logic signed [DATA_WIDTH-1:0]      y_out [0:MAX_C-1]
);

    // =========================================================================
    // Constants
    // =========================================================================
    localparam int CHAN_PER_GROUP = MAX_C / GROUPS;
    localparam real EPSILON       = 1.0 / 256.0;
    localparam int  EPSILON_Q     = int'(EPSILON * (1 << FRAC_WIDTH));
    localparam int  SUM_WIDTH     = DATA_WIDTH + $clog2(CHAN_PER_GROUP) + 1;
    localparam int  SQ_SUM_WIDTH  = 2*DATA_WIDTH + $clog2(CHAN_PER_GROUP) + 1;
    localparam int  PIPELINE_STAGES = 8;

    // =========================================================================
    // Inverse-sqrt LUT (shared with layerNorm)
    // =========================================================================
    localparam int LUT_SIZE       = pkg_config::LUT_SIZE;
    localparam int LUT_INDEX_BITS = pkg_config::LUT_INDEX_BITS;

    (* rom_style = "block" *)
    logic signed [DATA_WIDTH-1:0] inv_sqrt_lut   [0:LUT_SIZE-1];
    (* rom_style = "block" *)
    logic signed [DATA_WIDTH-1:0] inv_sqrt_slope [0:LUT_SIZE-1];

    initial begin
        $readmemh("rtl/ln_inv_sqrt_base.mem",  inv_sqrt_lut);
        $readmemh("rtl/ln_inv_sqrt_slope.mem", inv_sqrt_slope);
    end

    // =========================================================================
    // Backpressure / stall
    // =========================================================================
    logic stall;
    assign stall    = valid_out && !out_ready;
    assign ready_in = !stall;

    logic [PIPELINE_STAGES-1:0] valid_pipe;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)            valid_pipe <= '0;
        else if (!stall)       valid_pipe <= {valid_pipe[PIPELINE_STAGES-2:0], valid_in};
    end
    assign valid_out = valid_pipe[PIPELINE_STAGES-1];

    // =========================================================================
    // Stage 0: Input register + forward gamma/beta down the pipeline
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] x_s0 [0:MAX_C-1];
    logic signed [DATA_WIDTH-1:0] gamma_s0 [0:MAX_C-1];
    logic signed [DATA_WIDTH-1:0] beta_s0  [0:MAX_C-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < MAX_C; i++) begin
                x_s0[i]     <= '0;
                gamma_s0[i] <= '0;
                beta_s0[i]  <= '0;
            end
        end else if (valid_in && !stall) begin
            for (int i = 0; i < MAX_C; i++) begin
                x_s0[i]     <= x_in[i];
                gamma_s0[i] <= gamma[i];
                beta_s0[i]  <= beta[i];
            end
        end
    end

    // =========================================================================
    // Stage 1: per-group sum (parallel across groups, sequential within)
    // =========================================================================
    logic signed [SUM_WIDTH-1:0] sum_s1 [0:GROUPS-1];
    logic signed [DATA_WIDTH-1:0] x_s1 [0:MAX_C-1];
    logic signed [SUM_WIDTH-1:0] sum_s1_comb [0:GROUPS-1];

    always_comb begin
        for (int g = 0; g < GROUPS; g++) begin
            automatic logic signed [SUM_WIDTH-1:0] s;
            s = '0;
            for (int k = 0; k < CHAN_PER_GROUP; k++)
                s = s + SUM_WIDTH'(x_s0[g*CHAN_PER_GROUP + k]);
            sum_s1_comb[g] = s;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int g = 0; g < GROUPS; g++) sum_s1[g] <= '0;
            for (int i = 0; i < MAX_C; i++)  x_s1[i]   <= '0;
        end else if (!stall) begin
            for (int g = 0; g < GROUPS; g++) sum_s1[g] <= sum_s1_comb[g];
            for (int i = 0; i < MAX_C; i++) x_s1[i] <= x_s0[i];
        end
    end

    // =========================================================================
    // Stage 2: per-group mean (divide by CHAN_PER_GROUP — power-of-two only)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] mean_s2 [0:GROUPS-1];
    logic signed [DATA_WIDTH-1:0] x_s2 [0:MAX_C-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int g = 0; g < GROUPS; g++) mean_s2[g] <= '0;
            for (int i = 0; i < MAX_C; i++)  x_s2[i]    <= '0;
        end else if (!stall) begin
            for (int g = 0; g < GROUPS; g++)
                mean_s2[g] <= DATA_WIDTH'(sum_s1[g] >>> $clog2(CHAN_PER_GROUP));
            for (int i = 0; i < MAX_C; i++) x_s2[i] <= x_s1[i];
        end
    end

    // =========================================================================
    // Stage 3: centered = x - mean (per group)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] cent_s3 [0:MAX_C-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < MAX_C; i++) cent_s3[i] <= '0;
        end else if (!stall) begin
            for (int g = 0; g < GROUPS; g++)
                for (int k = 0; k < CHAN_PER_GROUP; k++)
                    cent_s3[g*CHAN_PER_GROUP + k] <= x_s2[g*CHAN_PER_GROUP + k] - mean_s2[g];
        end
    end

    // =========================================================================
    // Stage 4: per-group sum of squares
    // =========================================================================
    logic signed [SQ_SUM_WIDTH-1:0] ssq_s4 [0:GROUPS-1];
    logic signed [DATA_WIDTH-1:0] cent_s4 [0:MAX_C-1];
    logic signed [SQ_SUM_WIDTH-1:0] ssq_s4_comb [0:GROUPS-1];

    always_comb begin
        for (int g = 0; g < GROUPS; g++) begin
            automatic logic signed [SQ_SUM_WIDTH-1:0] s;
            s = '0;
            for (int k = 0; k < CHAN_PER_GROUP; k++) begin
                automatic logic signed [2*DATA_WIDTH-1:0] sq;
                if (USE_DSP) begin
                    (* use_dsp = "yes" *)
                    sq = cent_s3[g*CHAN_PER_GROUP + k] * cent_s3[g*CHAN_PER_GROUP + k];
                end else begin
                    (* use_dsp = "no" *)
                    sq = cent_s3[g*CHAN_PER_GROUP + k] * cent_s3[g*CHAN_PER_GROUP + k];
                end
                s = s + SQ_SUM_WIDTH'(sq);
            end
            ssq_s4_comb[g] = s;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int g = 0; g < GROUPS; g++) ssq_s4[g]  <= '0;
            for (int i = 0; i < MAX_C; i++)  cent_s4[i] <= '0;
        end else if (!stall) begin
            for (int g = 0; g < GROUPS; g++) ssq_s4[g] <= ssq_s4_comb[g];
            for (int i = 0; i < MAX_C; i++) cent_s4[i] <= cent_s3[i];
        end
    end

    // =========================================================================
    // Stage 5: per-group variance (mean of squares, divide by CHAN_PER_GROUP),
    //          add EPSILON, prepare LUT index
    // =========================================================================
    logic [LUT_INDEX_BITS-1:0]    inv_idx_s5 [0:GROUPS-1];
    logic signed [DATA_WIDTH-1:0] cent_s5    [0:MAX_C-1];

    // Combinational LUT index computation (Verilator: no declarations inside always_ff)
    logic [LUT_INDEX_BITS-1:0] inv_idx_comb [0:GROUPS-1];
    always_comb begin
        for (int g = 0; g < GROUPS; g++) begin
            automatic logic signed [DATA_WIDTH-1:0] var_q;
            automatic logic signed [DATA_WIDTH-1:0] var_int;
            var_q = DATA_WIDTH'((ssq_s4[g] >>> $clog2(CHAN_PER_GROUP)) + EPSILON_Q);
            // Saturating index extraction from integer part of variance.
            // Use arithmetic shift to avoid hardcoded bit slices that go
            // out of range when DATA_WIDTH <= FRAC_WIDTH + LUT_INDEX_BITS.
            var_int = var_q >>> FRAC_WIDTH;
            if (var_q[DATA_WIDTH-1])
                inv_idx_comb[g] = '0;
            else if (var_int >= LUT_SIZE)
                inv_idx_comb[g] = LUT_INDEX_BITS'(LUT_SIZE - 1);
            else
                inv_idx_comb[g] = LUT_INDEX_BITS'(var_int);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int g = 0; g < GROUPS; g++) inv_idx_s5[g] <= '0;
            for (int i = 0; i < MAX_C; i++)  cent_s5[i]    <= '0;
        end else if (!stall) begin
            for (int g = 0; g < GROUPS; g++) inv_idx_s5[g] <= inv_idx_comb[g];
            for (int i = 0; i < MAX_C; i++)  cent_s5[i]   <= cent_s4[i];
        end
    end

    // =========================================================================
    // Stage 6: 1/sqrt(var) LUT lookup (one read port per group)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] invs_s6 [0:GROUPS-1];
    logic signed [DATA_WIDTH-1:0] cent_s6 [0:MAX_C-1];
    logic signed [DATA_WIDTH-1:0] gamma_s6 [0:MAX_C-1];
    logic signed [DATA_WIDTH-1:0] beta_s6  [0:MAX_C-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int g = 0; g < GROUPS; g++) invs_s6[g] <= '0;
            for (int i = 0; i < MAX_C; i++) begin
                cent_s6[i]  <= '0;
                gamma_s6[i] <= '0;
                beta_s6[i]  <= '0;
            end
        end else if (!stall) begin
            for (int g = 0; g < GROUPS; g++) invs_s6[g] <= inv_sqrt_lut[inv_idx_s5[g]];
            for (int i = 0; i < MAX_C; i++) begin
                cent_s6[i]  <= cent_s5[i];
                gamma_s6[i] <= gamma_s0[i];
                beta_s6[i]  <= beta_s0[i];
            end
        end
    end

    // =========================================================================
    // Stage 7: scale (centered * inv_sqrt * gamma) + beta
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] y_out_comb [0:MAX_C-1];

    always_comb begin
        for (int g = 0; g < GROUPS; g++) begin
            for (int k = 0; k < CHAN_PER_GROUP; k++) begin
                automatic int idx = g*CHAN_PER_GROUP + k;
                if (idx < cfg_c) begin
                    automatic logic signed [2*DATA_WIDTH-1:0] m1;
                    automatic logic signed [3*DATA_WIDTH-1:0] m2;
                    if (USE_DSP) begin
                        (* use_dsp = "yes" *)
                        m1 = cent_s6[idx] * invs_s6[g];
                        (* use_dsp = "yes" *)
                        m2 = (m1 >>> FRAC_WIDTH) * gamma_s6[idx];
                    end else begin
                        (* use_dsp = "no" *)
                        m1 = cent_s6[idx] * invs_s6[g];
                        (* use_dsp = "no" *)
                        m2 = (m1 >>> FRAC_WIDTH) * gamma_s6[idx];
                    end
                    y_out_comb[idx] = DATA_WIDTH'((m2 >>> FRAC_WIDTH) + beta_s6[idx]);
                end else begin
                    y_out_comb[idx] = '0;
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < MAX_C; i++) y_out[i] <= '0;
        end else if (!stall) begin
            for (int i = 0; i < MAX_C; i++) y_out[i] <= y_out_comb[i];
        end
    end

    // cfg_c is now actively used for channel gating in Stage 7

endmodule
