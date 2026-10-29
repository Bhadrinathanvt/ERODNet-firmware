// =============================================================================
// temporal_inter_stage_buf.sv
//
// 2-deep circular buffer decoupling front-end (QKV+softmax) from back-end
// (aggregate+output) in temporal_attention_enhanced_v2.
//
// FE writes softmax results when S_FE_SOFTMAX_NORM completes.
// BE reads when S_BE_IDLE sees valid data and transitions to S_BE_AGGREGATE.
// 2-slot depth prevents FE stall unless both slots are occupied.
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;

module temporal_inter_stage_buf #(
    parameter int SLOTS        = 2,
    parameter int NUM_HEADS    = pkg_config::NUM_HEADS,
    parameter int HEAD_DIM     = pkg_config::HEAD_DIM,
    parameter int HISTORY_DEPTH = pkg_config::HISTORY_DEPTH,
    parameter int CHANNELS     = MAX_C,
    parameter int DATA_W       = pkg_module_config::TEMPORAL_ATTENTION_ENHANCED_DATA_WIDTH,
    parameter int ACC_W        = pkg_config::ACC_WIDTH
)(
    input  logic clk,
    input  logic rst_n,

    // FE write port
    input  logic                          fe_wr_valid,
    output logic                          fe_wr_ready,
    input  logic signed [ACC_W-1:0]      fe_exp_scores  [0:NUM_HEADS-1][0:HISTORY_DEPTH-1],
    input  logic signed [ACC_W-1:0]      fe_sm_recip    [0:NUM_HEADS-1],
    input  logic signed [DATA_W-1:0]     fe_v_proj      [0:HISTORY_DEPTH-1][0:NUM_HEADS-1][0:HEAD_DIM-1],
    input  logic signed [DATA_W-1:0]     fe_query_reg   [0:CHANNELS-1],
    input  logic [HISTORY_DEPTH-1:0]      fe_mask_reg,

    // BE read port
    output logic                          be_rd_valid,
    input  logic                          be_rd_ready,
    output logic signed [ACC_W-1:0]      be_exp_scores  [0:NUM_HEADS-1][0:HISTORY_DEPTH-1],
    output logic signed [ACC_W-1:0]      be_sm_recip    [0:NUM_HEADS-1],
    output logic signed [DATA_W-1:0]     be_v_proj      [0:HISTORY_DEPTH-1][0:NUM_HEADS-1][0:HEAD_DIM-1],
    output logic signed [DATA_W-1:0]     be_query_reg   [0:CHANNELS-1],
    output logic [HISTORY_DEPTH-1:0]      be_mask_reg,

    // Debug
    output logic [$clog2(SLOTS+1)-1:0]    buf_count
);

    // -------------------------------------------------------------------------
    // Storage — 2 slots of all FE→BE data
    // -------------------------------------------------------------------------
    logic signed [ACC_W-1:0]    slot_exp_scores  [0:SLOTS-1][0:NUM_HEADS-1][0:HISTORY_DEPTH-1];
    logic signed [ACC_W-1:0]    slot_sm_recip    [0:SLOTS-1][0:NUM_HEADS-1];
    logic signed [DATA_W-1:0]   slot_v_proj      [0:SLOTS-1][0:HISTORY_DEPTH-1][0:NUM_HEADS-1][0:HEAD_DIM-1];
    logic signed [DATA_W-1:0]   slot_query_reg   [0:SLOTS-1][0:CHANNELS-1];
    logic [HISTORY_DEPTH-1:0]   slot_mask_reg    [0:SLOTS-1];
    logic [SLOTS-1:0]           slot_valid;

    // -------------------------------------------------------------------------
    // Pointers
    // -------------------------------------------------------------------------
    logic [$clog2(SLOTS)-1:0] wr_ptr, rd_ptr;

    assign fe_wr_ready = (buf_count < SLOTS);
    assign be_rd_valid = (buf_count > 0);

    // -------------------------------------------------------------------------
    // Count tracking
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buf_count <= '0;
            wr_ptr    <= '0;
            rd_ptr    <= '0;
            slot_valid <= '0;
        end else begin
            // FE write
            if (fe_wr_valid && fe_wr_ready) begin
                for (int h = 0; h < NUM_HEADS; h++)
                    for (int t = 0; t < HISTORY_DEPTH; t++)
                        slot_exp_scores[wr_ptr][h][t] <= fe_exp_scores[h][t];
                for (int h = 0; h < NUM_HEADS; h++)
                    slot_sm_recip[wr_ptr][h] <= fe_sm_recip[h];
                for (int t = 0; t < HISTORY_DEPTH; t++)
                    for (int h = 0; h < NUM_HEADS; h++)
                        for (int d = 0; d < HEAD_DIM; d++)
                            slot_v_proj[wr_ptr][t][h][d] <= fe_v_proj[t][h][d];
                for (int c = 0; c < CHANNELS; c++)
                    slot_query_reg[wr_ptr][c] <= fe_query_reg[c];
                slot_mask_reg[wr_ptr] <= fe_mask_reg;
                slot_valid[wr_ptr] <= 1'b1;
                wr_ptr <= wr_ptr + 1'b1;
            end

            // BE read
            if (be_rd_ready && be_rd_valid) begin
                slot_valid[rd_ptr] <= 1'b0;
                rd_ptr <= rd_ptr + 1'b1;
            end

            // Count update
            if (fe_wr_valid && fe_wr_ready && !(be_rd_ready && be_rd_valid))
                buf_count <= buf_count + 1'b1;
            else if (!(fe_wr_valid && fe_wr_ready) && be_rd_ready && be_rd_valid)
                buf_count <= buf_count - 1'b1;
        end
    end

    // -------------------------------------------------------------------------
    // BE output mux — always presents the rd_ptr slot
    // -------------------------------------------------------------------------
    always_comb begin
        for (int h = 0; h < NUM_HEADS; h++) begin
            for (int t = 0; t < HISTORY_DEPTH; t++)
                be_exp_scores[h][t] = slot_exp_scores[rd_ptr][h][t];
            be_sm_recip[h] = slot_sm_recip[rd_ptr][h];
        end
        for (int t = 0; t < HISTORY_DEPTH; t++)
            for (int h = 0; h < NUM_HEADS; h++)
                for (int d = 0; d < HEAD_DIM; d++)
                    be_v_proj[t][h][d] = slot_v_proj[rd_ptr][t][h][d];
        for (int c = 0; c < CHANNELS; c++)
            be_query_reg[c] = slot_query_reg[rd_ptr][c];
        be_mask_reg = slot_mask_reg[rd_ptr];
    end

    // -------------------------------------------------------------------------
    // Assertions
    // -------------------------------------------------------------------------
    // synthesis translate_off
    INTER_BUF_NO_OVERFLOW: assert property (
        @(posedge clk) disable iff (!rst_n)
        !(fe_wr_valid && !fe_wr_ready)
    ) else $error("temporal_inter_stage_buf: FE write when full");

    INTER_BUF_NO_UNDERFLOW: assert property (
        @(posedge clk) disable iff (!rst_n)
        !(be_rd_ready && !be_rd_valid)
    ) else $error("temporal_inter_stage_buf: BE read when empty");

    INTER_BUF_COUNT_VALID: assert property (
        @(posedge clk) disable iff (!rst_n)
        buf_count <= SLOTS
    ) else $error("temporal_inter_stage_buf: count exceeded SLOTS");
    // synthesis translate_on

endmodule