import pkg_config::*;
import pkg_module_config::*;

module batch_norm_3d #(
    parameter int DATA_WIDTH = pkg_module_config::BATCH_NORM_3D_DATA_WIDTH,
    parameter int FRAC_WIDTH = pkg_config::FRAC_WIDTH,
    parameter int CHANNELS   = 4        // C – one lane per channel
)(
    input  logic clk,
    input  logic rst_n,
    input  logic in_valid,
    output logic out_valid,

    // Per-channel learned parameters (packed flat vectors — Verilator compatible)
    input  logic signed [CHANNELS*DATA_WIDTH-1:0] gamma_flat,
    input  logic signed [CHANNELS*DATA_WIDTH-1:0] beta_flat,
    // Per-channel running statistics from training
    input  logic signed [CHANNELS*DATA_WIDTH-1:0] run_mean_flat,
    // inv_std = 1 / sqrt(run_var + eps), pre-computed, Q-format
    input  logic signed [CHANNELS*DATA_WIDTH-1:0] inv_std_flat,

    input  logic signed [CHANNELS*DATA_WIDTH-1:0] x_in_flat,
    output logic signed [CHANNELS*DATA_WIDTH-1:0] y_out_flat
);

    // Unpack flat ports into internal per-channel arrays
    logic signed [DATA_WIDTH-1:0] gamma    [CHANNELS-1:0];
    logic signed [DATA_WIDTH-1:0] beta     [CHANNELS-1:0];
    logic signed [DATA_WIDTH-1:0] run_mean [CHANNELS-1:0];
    logic signed [DATA_WIDTH-1:0] inv_std  [CHANNELS-1:0];
    logic signed [DATA_WIDTH-1:0] x_in     [CHANNELS-1:0];
    logic signed [DATA_WIDTH-1:0] y_out    [CHANNELS-1:0];

    always_comb begin
        for (int c = 0; c < CHANNELS; c++) begin
            gamma[c]    = gamma_flat[c*DATA_WIDTH +: DATA_WIDTH];
            beta[c]     = beta_flat[c*DATA_WIDTH +: DATA_WIDTH];
            run_mean[c] = run_mean_flat[c*DATA_WIDTH +: DATA_WIDTH];
            inv_std[c]  = inv_std_flat[c*DATA_WIDTH +: DATA_WIDTH];
            x_in[c]     = x_in_flat[c*DATA_WIDTH +: DATA_WIDTH];
            y_out_flat[c*DATA_WIDTH +: DATA_WIDTH] = y_out[c];
        end
    end

    localparam int PIPE = 4;

    // valid shift register
    logic [PIPE-1:0] valid_sr;
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) valid_sr <= '0;
        else        valid_sr <= {valid_sr[PIPE-2:0], in_valid};
    assign out_valid = valid_sr[PIPE-1];

    // Pipeline registers
    logic signed [DATA_WIDTH-1:0]   s1_xm  [CHANNELS-1:0]; // x - mean
    logic signed [2*DATA_WIDTH-1:0] s2_norm[CHANNELS-1:0]; // (x-mean)*inv_std
    logic signed [3*DATA_WIDTH-1:0] s3_scl [CHANNELS-1:0]; // (s2_norm>>>FRAC_WIDTH)*gamma
    logic signed [DATA_WIDTH-1:0]   s4_out [CHANNELS-1:0]; // + beta

    generate
        for (genvar c = 0; c < CHANNELS; c++) begin : gen_bn_chan

            // ── Stage 1: subtract running mean ───────────────────────────────
            always_ff @(posedge clk or negedge rst_n)
                if (!rst_n) s1_xm[c] <= '0;
                else        s1_xm[c] <= x_in[c] - run_mean[c];

            // ── Stage 2: normalise  (x-mean) * inv_std  ──────────────────────
            always_ff @(posedge clk or negedge rst_n)
                if (!rst_n) s2_norm[c] <= '0;
                else        s2_norm[c] <= s1_xm[c] * inv_std[c];

            // ── Stage 3: scale by gamma ───────────────────────────────────────
            // Shift right by FRAC_WIDTH to keep Q-format after multiply
            always_ff @(posedge clk or negedge rst_n)
                if (!rst_n) s3_scl[c] <= '0;
                else        s3_scl[c] <=
                    (s2_norm[c] >>> FRAC_WIDTH) * gamma[c];

            // ── Stage 4: add beta, truncate back to DATA_WIDTH ────────────────
            always_ff @(posedge clk or negedge rst_n)
                if (!rst_n) s4_out[c] <= '0;
                else        s4_out[c] <=
                    DATA_WIDTH'((s3_scl[c] >>> FRAC_WIDTH) + beta[c]);

            assign y_out[c] = s4_out[c];
        end
    endgenerate

endmodule