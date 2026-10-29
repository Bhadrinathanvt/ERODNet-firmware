// =============================================================================
// gff.sv
//
// GFF (Global Feature Fusion) streaming interface wrapper.
//
// Wraps gff_stream_core (token-streaming compute engine) with a dual-buffered
// frame accumulator. The design philosophy:
//
//   gff_synth (OLD): batch-parallel — processes one N×4C frame as a single unit.
//                    Compute time ≈ 382 cycles << frame fill time ≈ 16384 cycles.
//                    Dual-buffer gives near-zero benefit (compute finishes long
//                    before the next frame's fill completes).
//
//   gff_stream_core (NEW): token-streaming — processes one cell per cycle through
//                    a pipeline. Compute and fill overlap naturally since each
//                    cell's compute is much shorter than the frame fill time.
//
// DUAL-BUFFER ARCHITECTURE (STUB_BEHAVIOR=0):
//   The outer wrapper holds two frame buffers (A and B). While gff_stream_core
//   streams cells from buffer A (compute), buffer B fills with incoming tokens.
//   When gff_stream_core finishes with A, it switches to B while A drains.
//   This is standard double-buffering at the frame level.
//   True 2× throughput comes from compute/drain overlapping with next-frame fill.
//
// BUFFER PHILOSOPHY:
//   Input BRAM (dual-buffer): holds raw N×4C frame tokens
//   gff_stream_core: streams cells through compute pipeline
//   Output: cells stream out one per cycle (no output buffering needed)
//
// STUB_BEHAVIOR=1: wire-through with 1-cycle delay.
// =============================================================================

`timescale 1ns / 1ps

import pkg_config::*;
import pkg_module_config::*;

module gff #(
    parameter int DATA_WIDTH     = pkg_module_config::GFF_DATA_WIDTH,
    parameter int FRAC_WIDTH    = pkg_module_config::GFF_DATA_WIDTH / 2,
    parameter int CHANNELS      = pkg_config::MAX_C,     // HIDDEN_CHANNELS
    parameter bit STUB_BEHAVIOR = pkg_config::STUB_BEHAVIOR,  // 1=wire-through, 0=streaming GFF
    parameter int WEIGHT_LOAD_MODE = pkg_config::WEIGHT_LOAD_MODE,
    parameter string WEIGHT_FILE    = "rtl/mem/gff_cat_weight.mem",
    parameter string BIAS_FILE      = "rtl/mem/gff_cat_bias.mem",
    parameter int FEATURE_H     = 128,      // Frame height
    parameter int FEATURE_W     = 128,      // Frame width
    parameter int TIME_MUX      = pkg_config::TIME_MUX,    // Time-multiplexing factor
    parameter int L2_ITER      = pkg_config::L2_ITER,     // Newton-Raphson iterations
    parameter int AVG_GROUP_SIZE = pkg_config::AVG_GROUP_SIZE  // AvgPool tree-fold group size
)(
    input  logic clk,
    input  logic rst_n,
    input  logic cfg_enable,

    // ── Streaming input ───────────────────────────────────────────────────────
    input  logic                          valid_in,
    input  logic                          valid_in_last,  // Last token of frame
    input  logic signed [DATA_WIDTH-1:0]  data_in [0:CHANNELS-1],
    output logic                          ready_in,

    // ── Streaming output ───────────────────────────────────────────────────────
    output logic                          valid_out,
    output logic                          valid_out_last,
    output logic signed [DATA_WIDTH-1:0]  data_out [0:CHANNELS-1],
    input  logic                          out_ready,

    // ── Weight loading ─────────────────────────────────────────────────────────
    input  logic                          weight_valid,
    input  logic [31:0]                   weight_addr,
    input  logic signed [DATA_WIDTH-1:0]  weight_data,
    input  logic                          bias_valid,
    input  logic [31:0]                   bias_addr,
    input  logic signed [DATA_WIDTH-1:0]  bias_data,
    input  logic signed [DATA_WIDTH-1:0]  l2norm_scale,

    // ── Performance counter ───────────────────────────────────────────────────
    output logic [31:0]                   perf_cycle_count
);

    // Suppress unused warnings
    /* verilator lint_off UNUSED */
    logic _unused;
    assign _unused = &{bias_valid, bias_addr, bias_data, l2norm_scale};
    /* verilator lint_on UNUSED */

    // =========================================================================
    // STUB PATH (STUB_BEHAVIOR=1): single-cycle registered wire-through.
    // =========================================================================
    generate
        if (STUB_BEHAVIOR) begin : gen_stub

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    valid_out       <= 1'b0;
                    valid_out_last  <= 1'b0;
                    for (int c = 0; c < CHANNELS; c++)
                        data_out[c] <= '0;
                end else if (cfg_enable) begin
                    valid_out       <= valid_in;
                    valid_out_last  <= valid_in_last;
                    for (int c = 0; c < CHANNELS; c++)
                        data_out[c] <= data_in[c];
                end else begin
                    valid_out       <= 1'b0;
                    valid_out_last  <= 1'b0;
                end
            end

            assign ready_in = out_ready || !valid_out;

        end else begin : gen_full
            // ---------------------------------------------------------------
            // FULL PATH: Dual-buffered frame accumulator + streaming compute.
            // Input: one token per cycle (one spatial cell's feature vector).
            // Output: one token per cycle (processed cell).
            // ---------------------------------------------------------------

            localparam int FRAME_CELLS  = FEATURE_H * FEATURE_W;
            localparam int CELL_ADDR_W  = $clog2(FRAME_CELLS);
            localparam int C            = CHANNELS;
            localparam int D            = 8*C + 28;  // Concat delay alignment

            // ── Input dual-buffer BRAMs ────────────────────────────────────────
            (* ram_style = "block" *)
            logic signed [DATA_WIDTH-1:0] accum_buf_a [0:FRAME_CELLS-1][0:CHANNELS-1];
            (* ram_style = "block" *)
            logic signed [DATA_WIDTH-1:0] accum_buf_b [0:FRAME_CELLS-1][0:CHANNELS-1];

            // ── Buffer control ──────────────────────────────────────────────────
            typedef enum logic [2:0] {
                S_IDLE,
                S_FILL_A,
                S_RUN_A,
                S_DRAIN_A,
                S_FILL_B,
                S_RUN_B,
                S_DRAIN_B
            } state_t;

            state_t state;
            logic buf_a_full, buf_b_full;
            logic buf_a_draining, buf_b_draining;

            logic [CELL_ADDR_W-1:0] wr_ptr;
            logic [CELL_ADDR_W-1:0] rd_ptr;
            logic gff_started;
            logic draining;

            // LayerNorm parameters (default)
            logic signed [DATA_WIDTH-1:0] ln_gamma [0:CHANNELS*3-1];
            logic signed [DATA_WIDTH-1:0] ln_beta  [0:CHANNELS*3-1];
            initial begin
                for (int i = 0; i < CHANNELS*3; i++) begin
                    ln_gamma[i] = DATA_WIDTH'(1 << (DATA_WIDTH/2));
                    ln_beta[i] = '0;
                end
            end

            // ── gff_stream_core streaming interface ────────────────────────────────
            // We feed one cell per cycle from the buffer being processed.
            // gff_stream_core handles the full streaming pipeline internally.

            logic                          gff_feat_valid;
            logic                          gff_feat_last;
            logic signed [DATA_WIDTH-1:0]  gff_feat_in [0:4*CHANNELS-1];
            logic signed [DATA_WIDTH-1:0]  gff_feat_out [0:CHANNELS-1];
            logic                          gff_feat_out_valid;
            logic                          gff_feat_out_last;
            logic                          gff_ready;

            // Cell counter for streaming into gff_stream_core
            logic [CELL_ADDR_W-1:0] stream_cnt;
            logic streaming;

            // ── Streaming into gff_stream_core ─────────────────────────────────
            // Select active buffer and stream cells one per cycle.
            // When S_RUN_A: stream from accum_buf_a at 1 cell/cycle.
            // When S_RUN_B: stream from accum_buf_b at 1 cell/cycle.

            always_comb begin
                for (int ch = 0; ch < CHANNELS; ch++)
                    gff_feat_in[ch] = (state == S_RUN_B || state == S_DRAIN_B)
                                      ? accum_buf_b[stream_cnt][ch]
                                      : accum_buf_a[stream_cnt][ch];
                for (int ch = CHANNELS; ch < 4*CHANNELS; ch++)
                    gff_feat_in[ch] = '0;  // Zero-pad upper channels
            end

            // gff_stream_core instantiation
            gff_stream_core #(
                .C              (C),
                .N              (FRAME_CELLS),
                .DATA_W         (DATA_WIDTH),
                .FRAC_W         (FRAC_WIDTH),
                .ACC_W          (DATA_WIDTH + 8),
                .L2_ITER        (L2_ITER),
                .AVG_GROUP_SIZE (AVG_GROUP_SIZE),
                .TIME_MUX       (TIME_MUX),
                .ROM_DIR        ("rtl/mem"),
                .WEIGHT_FILE    (WEIGHT_FILE),
                .BIAS_FILE      (BIAS_FILE)
            ) u_gff_stream (
                .clk             (clk),
                .rst_n           (rst_n),
                .feat_in         (gff_feat_in),
                .feat_valid      (gff_feat_valid),
                .feat_last       (gff_feat_last),
                .feat_ready      (gff_ready),
                .feat_out        (gff_feat_out),
                .feat_out_valid  (gff_feat_out_valid),
                .feat_out_last   (gff_feat_out_last),
                .proj_ln_gamma   (ln_gamma),
                .proj_ln_beta    (ln_beta),
                .weight_valid    (weight_valid),
                .weight_addr     (weight_addr),
                .weight_data     (weight_data)
            );

            // ── Buffer fill logic ────────────────────────────────────────────────
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    state         <= S_IDLE;
                    wr_ptr        <= '0;
                    buf_a_full    <= 1'b0;
                    buf_b_full    <= 1'b0;
                    buf_a_draining<= 1'b0;
                    buf_b_draining<= 1'b0;
                    valid_out     <= 1'b0;
                    valid_out_last <= 1'b0;
                    draining      <= 1'b0;
                    for (int c = 0; c < CHANNELS; c++)
                        data_out[c] <= '0;
                end else begin
                    // Default: clear single-cycle pulses
                    valid_out      <= 1'b0;
                    valid_out_last <= 1'b0;
                    // NOTE: gff_feat_valid/gff_feat_last are driven exclusively by the
                    // streaming control block (below) — do NOT clear them here.

                    case (state)
                        // ── Idle: wait for first frame ─────────────────────
                        S_IDLE: begin
                            if (valid_in && cfg_enable) begin
                                state  <= S_FILL_A;
                                wr_ptr <= '0;
                            end
                        end

                        // ── Fill buffer A ────────────────────────────────────
                        S_FILL_A: begin
                            if (valid_in && cfg_enable) begin
                                for (int c = 0; c < CHANNELS; c++)
                                    accum_buf_a[wr_ptr][c] <= data_in[c];
                                wr_ptr <= wr_ptr + 1'b1;
                                if (valid_in_last) begin
                                    buf_a_full   <= 1'b1;
                                    wr_ptr      <= '0;
                                    state       <= S_RUN_A;
                                    // gff_started reset handled by streaming control block
                                end
                            end
                        end

                        // ── Stream buffer A to gff_stream_core ─────────────────
                        // FIXED: gff_state_priority — drain transition must take
                        // priority over fill transition to avoid losing gff output.
                        // FIXED: capture first output cell on transition cycle
                        // (previously the first gff_feat_out_valid was only used for
                        // the state transition, data was silently dropped).
                        S_RUN_A: begin
                            if (gff_feat_out_valid) begin
                                valid_out     <= 1'b1;
                                for (int c = 0; c < CHANNELS; c++)
                                    data_out[c] <= gff_feat_out[c];
                                buf_a_draining <= 1'b1;
                                draining      <= 1'b1;
                                rd_ptr        <= '0;
                                if (gff_feat_out_last) begin
                                    valid_out_last <= 1'b1;
                                    buf_a_full     <= 1'b0;
                                    buf_a_draining <= 1'b0;
                                    draining       <= 1'b0;
                                    if (buf_b_full)
                                        state <= S_RUN_B;
                                    else
                                        state <= S_FILL_B;
                                end else begin
                                    state <= S_DRAIN_A;
                                end
                            end else if (valid_in && cfg_enable && !buf_b_full && gff_started) begin
                                // Only transition to fill B after gff_stream_core has
                                // started processing buffer A (gff_started=1).
                                state <= S_FILL_B;
                            end
                        end

                        // ── Drain A while filling B ────────────────────────────
                        S_DRAIN_A: begin
                            valid_out <= gff_feat_out_valid;
                            if (gff_feat_out_valid) begin
                                for (int c = 0; c < CHANNELS; c++)
                                    data_out[c] <= gff_feat_out[c];
                                if (gff_feat_out_last) begin
                                    valid_out_last   <= 1'b1;
                                    buf_a_full       <= 1'b0;
                                    buf_a_draining   <= 1'b0;
                                    draining         <= 1'b0;
                                    if (buf_b_full)
                                        state <= S_RUN_B;
                                    else
                                        state <= S_FILL_B;
                                end else begin
                                    rd_ptr <= rd_ptr + 1'b1;
                                end
                            end
                        end

                        // ── Fill buffer B ────────────────────────────────────
                        S_FILL_B: begin
                            if (valid_in && cfg_enable) begin
                                for (int c = 0; c < CHANNELS; c++)
                                    accum_buf_b[wr_ptr][c] <= data_in[c];
                                wr_ptr <= wr_ptr + 1'b1;
                                if (valid_in_last) begin
                                    buf_b_full   <= 1'b1;
                                    wr_ptr      <= '0;
                                    if (buf_a_draining)
                                        state <= S_DRAIN_A;
                                    else
                                        state <= S_RUN_B;
                                    // gff_started reset handled by streaming control block
                                end
                            end
                        end

                        // ── Stream buffer B to gff_stream_core ─────────────────
                        // FIXED: drain priority over fill (same as S_RUN_A fix).
                        // FIXED: capture first output cell on transition cycle
                        // (same fix as S_RUN_A — first cell was silently dropped).
                        S_RUN_B: begin
                            if (gff_feat_out_valid) begin
                                valid_out     <= 1'b1;
                                for (int c = 0; c < CHANNELS; c++)
                                    data_out[c] <= gff_feat_out[c];
                                buf_b_draining <= 1'b1;
                                draining      <= 1'b1;
                                rd_ptr        <= '0;
                                if (gff_feat_out_last) begin
                                    valid_out_last <= 1'b1;
                                    buf_b_full     <= 1'b0;
                                    buf_b_draining <= 1'b0;
                                    draining       <= 1'b0;
                                    if (buf_a_full)
                                        state <= S_RUN_A;
                                    else
                                        state <= S_FILL_A;
                                end else begin
                                    state <= S_DRAIN_B;
                                end
                            end else if (valid_in && cfg_enable && !buf_a_full && gff_started) begin
                                state <= S_FILL_A;
                            end
                        end

                        // ── Drain B while filling A ────────────────────────────
                        S_DRAIN_B: begin
                            valid_out <= gff_feat_out_valid;
                            if (gff_feat_out_valid) begin
                                for (int c = 0; c < CHANNELS; c++)
                                    data_out[c] <= gff_feat_out[c];
                                if (gff_feat_out_last) begin
                                    valid_out_last   <= 1'b1;
                                    buf_b_full       <= 1'b0;
                                    buf_b_draining   <= 1'b0;
                                    draining         <= 1'b0;
                                    if (buf_a_full)
                                        state <= S_RUN_A;
                                    else
                                        state <= S_FILL_A;
                                end else begin
                                    rd_ptr <= rd_ptr + 1'b1;
                                end
                            end
                        end

                        default: state <= S_IDLE;
                    endcase
                end
            end

            // ── Streaming control: send one cell per cycle to gff_stream_core ───
            // gff_started is set when streaming begins, cleared on state exit to
            // prevent re-triggering after stream_cnt wraps.
            logic [2:0] stream_state_d1;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    stream_cnt      <= '0;
                    streaming       <= 1'b0;
                    gff_started     <= 1'b0;
                    gff_feat_valid  <= 1'b0;
                    gff_feat_last   <= 1'b0;
                    stream_state_d1 <= S_IDLE;
                end else begin
                    stream_state_d1 <= state;

                    // Clear gff_started when leaving S_RUN_A or S_RUN_B
                    // (edge detection — fires once per RUN→DRAIN or RUN→FILL transition)
                    if ((stream_state_d1 == S_RUN_A && state != S_RUN_A) ||
                        (stream_state_d1 == S_RUN_B && state != S_RUN_B)) begin
                        gff_started <= 1'b0;
                    end

                    if (state == S_RUN_A && !gff_started) begin
                        streaming   <= 1'b1;
                        stream_cnt  <= '0;
                        gff_started <= 1'b1;
                    end else if (state == S_RUN_B && !gff_started) begin
                        streaming   <= 1'b1;
                        stream_cnt  <= '0;
                        gff_started <= 1'b1;
                    end else if (streaming) begin
                        gff_feat_valid <= 1'b1;
                        if (stream_cnt == CELL_ADDR_W'(FRAME_CELLS - 1)) begin
                            gff_feat_last <= 1'b1;
                            streaming     <= 1'b0;
                            stream_cnt    <= '0;
                            // gff_started stays high — cleared by state exit edge detect
                        end else begin
                            stream_cnt <= stream_cnt + 1'b1;
                        end
                    end else begin
                        gff_feat_valid <= 1'b0;
                        gff_feat_last  <= 1'b0;
                        stream_cnt     <= '0;
                    end
                end
            end

            // ── ready_in: can accept input when a buffer is available ─────────────
            assign ready_in = cfg_enable &&
                              ((!buf_a_full && state != S_RUN_A && state != S_DRAIN_A) ||
                               (!buf_b_full && state != S_RUN_B && state != S_DRAIN_B));

            // FIXED: gff_feat_valid, gff_feat_last, gff_started, stream_cnt, and
            // streaming are driven exclusively by the streaming control always_ff
            // above (not by the main FSM block) to avoid multi-driver issues.

        end
    endgenerate

    // ── Performance counter ─────────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                      perf_cycle_count <= '0;
        else if (valid_in && cfg_enable)  perf_cycle_count <= perf_cycle_count + 1'b1;
    end

endmodule
