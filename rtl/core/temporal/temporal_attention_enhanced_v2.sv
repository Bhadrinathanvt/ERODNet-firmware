// =============================================================================
// temporal_attention_enhanced_v2.sv
//
// Pipelined Multi-Head Temporal Attention — dual-FSM architecture.
// Splits the v1 monolithic FSM into concurrent Front-End (QKV+softmax) and
// Back-End (aggregate+output) stages with a 2-deep inter-stage buffer.
//
// Latency improvement (C=64, HEADS=4, HDIM=16, HIST=8):
//   v1 serial:  ~4653 cycles/query
//   v2 pipelined: max(FE=520, BE=81) = 520 cycles/query (8.9x speedup)
//
// Key changes from v1:
//   1. Dual-FSM: FE and BE operate concurrently on different queries
//   2. 2-deep inter-stage buffer decouples FE from BE
//   3. 2-deep input buffer prevents upstream stall during processing
//   4. Output skid buffer prevents downstream backpressure from blocking BE
//   5. Combinational FSM transitions eliminate IDLE-cycle gaps
//   6. Config parameters increased: MAC_UNROLL=8, OUT_MAC_UNROLL=4, etc.
//
// Port interface: identical to temporal_attention_enhanced (v1).
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;

module temporal_attention_enhanced_v2 #(
    parameter int HISTORY_DEPTH   = pkg_config::HISTORY_DEPTH,
    parameter int CHANNELS        = MAX_C,
    parameter int NUM_HEADS       = pkg_config::NUM_HEADS,
    parameter int HEAD_DIM        = pkg_config::HEAD_DIM,
    parameter int DATA_W = pkg_module_config::TEMPORAL_ATTENTION_ENHANCED_DATA_WIDTH,
    parameter int FRAC_W          = pkg_module_config::TEMPORAL_ATTENTION_ENHANCED_FRAC_WIDTH,
    parameter int ACC_W           = pkg_config::ACC_WIDTH,
    parameter int WEIGHT_LOAD_MODE = pkg_config::WEIGHT_LOAD_MODE,
    parameter string WEIGHT_FILE   = "",
    parameter int MAC_UNROLL      = pkg_config::MAC_UNROLL,
    parameter int OUT_MAC_UNROLL  = pkg_config::OUT_MAC_UNROLL,
    parameter int ATTN_HEADS_PER_CYCLE = pkg_config::ATTN_HEADS_PER_CYCLE,
    parameter int HD_UNROLL           = pkg_config::HD_UNROLL,
    parameter int CH_OUT_UNROLL       = pkg_config::CH_OUT_UNROLL,
    parameter int SCORE_HIST_UNROLL   = pkg_config::SCORE_HIST_UNROLL,
    parameter bit  ENABLE_RESIDUAL    = pkg_config::ENABLE_RESIDUAL,
    parameter int  EXP_UNROLL         = pkg_config::EXP_UNROLL,
    parameter bit  ENABLE_TEMPERATURE = pkg_config::ENABLE_TEMPERATURE,
    parameter bit  ENABLE_MASK        = pkg_config::ENABLE_MASK,
    parameter bit  QK_WEIGHT_SHARING  = pkg_config::QK_WEIGHT_SHARING,
    // V2-specific
    parameter int INTER_BUF_DEPTH     = 2
)(
    input  logic clk,
    input  logic rst_n,

    // Configuration
    input  logic [2:0]  cfg_attention_mode,
    input  logic [3:0]  cfg_history_depth,
    input  logic [3:0]  cfg_local_window,
    input  logic        cfg_enable,
    input  logic signed [ACC_W-1:0] cfg_temperature,

    // Current frame (query)
    input  logic                        query_valid_in,
    input  logic signed [DATA_W-1:0] query_data [0:CHANNELS-1],
    output logic                        query_ready_in,

    // History frames (keys and values)
    input  logic                        history_valid_in,
    input  logic [HISTORY_DEPTH-1:0]   history_valid_mask,
    input  logic signed [DATA_W-1:0] history_data [0:HISTORY_DEPTH-1][0:CHANNELS-1],

    // Weight loading
    input  logic                        weight_valid,
    input  logic [15:0]                 weight_addr,
    input  logic signed [DATA_W-1:0] weight_data,

    // Output
    output logic                        attended_valid_out,
    output logic signed [DATA_W-1:0] attended_data_out [0:CHANNELS-1],
    input  logic                        attended_ready_in
);

    localparam bit [2:0] MODE_FULL    = 3'b000;
    localparam bit [2:0] MODE_LOCAL   = 3'b001;
    localparam bit [2:0] MODE_STRIDED = 3'b010;

    // =========================================================================
    // Weight Storage (shared between FE and BE, separate BRAM arrays)
    // =========================================================================
    localparam int QKV_WEIGHTS = 3 * CHANNELS * NUM_HEADS * HEAD_DIM;
    localparam int OUT_WEIGHTS = NUM_HEADS * HEAD_DIM * CHANNELS;
    localparam int PARALLEL_DIV = pkg_config::PARALLEL_DIV;
    localparam int PARALLEL_HIST = pkg_config::PARALLEL_HIST;

    (* ram_style = "block" *)
    logic signed [DATA_W-1:0] q_weights [0:CHANNELS*NUM_HEADS*HEAD_DIM-1];
    (* ram_style = "block" *)
    logic signed [DATA_W-1:0] k_weights [0:CHANNELS*NUM_HEADS*HEAD_DIM-1];
    (* ram_style = "block" *)
    logic signed [DATA_W-1:0] v_weights [0:CHANNELS*NUM_HEADS*HEAD_DIM-1];
    (* ram_style = "block" *)
    logic signed [DATA_W-1:0] out_weights [0:OUT_WEIGHTS-1];

    // =========================================================================
    // Input Buffer (2-deep register buffer)
    // =========================================================================
    localparam int IN_BUF_DEPTH = 2;

    logic signed [DATA_W-1:0] in_buf_query    [0:IN_BUF_DEPTH-1][0:CHANNELS-1];
    logic signed [DATA_W-1:0] in_buf_history  [0:IN_BUF_DEPTH-1][0:HISTORY_DEPTH-1][0:CHANNELS-1];
    logic [HISTORY_DEPTH-1:0] in_buf_mask     [0:IN_BUF_DEPTH-1];
    logic [IN_BUF_DEPTH-1:0]  in_buf_valid;
    logic [$clog2(IN_BUF_DEPTH)-1:0] in_wr_ptr, in_rd_ptr;

    wire in_buf_full  = (&in_buf_valid);
    wire in_buf_empty = (in_buf_valid == '0);

    assign query_ready_in = !in_buf_full;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_buf_valid <= '0;
            in_wr_ptr   <= '0;
            in_rd_ptr   <= '0;
        end else begin
            // Input write
            if (query_valid_in && history_valid_in && cfg_enable && !in_buf_full) begin
                for (int c = 0; c < CHANNELS; c++)
                    in_buf_query[in_wr_ptr][c] <= query_data[c];
                for (int t = 0; t < HISTORY_DEPTH; t++)
                    for (int c = 0; c < CHANNELS; c++)
                        in_buf_history[in_wr_ptr][t][c] <= history_data[t][c];
                in_buf_mask[in_wr_ptr] <= ENABLE_MASK ? history_valid_mask : {(HISTORY_DEPTH){1'b1}};
                in_buf_valid[in_wr_ptr] <= 1'b1;
                in_wr_ptr <= in_wr_ptr + 1'b1;
            end

            // FE read
            if (fe_consume_in_buf) begin
                in_buf_valid[in_rd_ptr] <= 1'b0;
                in_rd_ptr <= in_rd_ptr + 1'b1;
            end
        end
    end

    // =========================================================================
    // Front-End FSM (QKV Projection → Attention Score → Softmax)
    // =========================================================================
    localparam int SQRT_HEAD_DIM_SHIFT = $clog2(HEAD_DIM) / 2;
    localparam int ATTN_PASSES = (NUM_HEADS + ATTN_HEADS_PER_CYCLE - 1) / ATTN_HEADS_PER_CYCLE;

    typedef enum logic [2:0] {
        S_FE_IDLE,
        S_FE_PROJ_MAC,
        S_FE_ATTN_SCORE,
        S_FE_SOFTMAX_EXP,
        S_FE_SOFTMAX_NORM
    } fe_state_t;

    fe_state_t fe_state;

    // FE counters
    logic [$clog2(CHANNELS):0]            mac_idx;
    logic [$clog2(NUM_HEADS*HEAD_DIM):0]  hd_idx;
    logic [$clog2(HISTORY_DEPTH):0]       t_idx;
    logic [$clog2(HISTORY_DEPTH):0]      score_t_idx;
    logic [$clog2(ATTN_PASSES):0]         attn_pass;
    logic [$clog2(NUM_HEADS):0]           head_idx;
    logic [$clog2(NUM_HEADS*HISTORY_DEPTH):0] exp_idx;

    // FE accumulators
    logic signed [ACC_W-1:0] q_accum   [0:NUM_HEADS-1][0:HEAD_DIM-1];
    logic signed [ACC_W-1:0] k_accum   [0:HISTORY_DEPTH-1][0:NUM_HEADS-1][0:HEAD_DIM-1];
    logic signed [ACC_W-1:0] v_accum   [0:HISTORY_DEPTH-1][0:NUM_HEADS-1][0:HEAD_DIM-1];

    // FE projections
    logic signed [DATA_W-1:0] q_proj [0:NUM_HEADS-1][0:HEAD_DIM-1];
    logic signed [DATA_W-1:0] k_proj [0:HISTORY_DEPTH-1][0:NUM_HEADS-1][0:HEAD_DIM-1];
    logic signed [DATA_W-1:0] v_proj [0:HISTORY_DEPTH-1][0:NUM_HEADS-1][0:HEAD_DIM-1];

    // FE attention + softmax
    logic signed [ACC_W-1:0] attention_scores [0:NUM_HEADS-1][0:HISTORY_DEPTH-1];
    logic signed [ACC_W-1:0] exp_scores [0:NUM_HEADS-1][0:HISTORY_DEPTH-1];
    logic signed [ACC_W-1:0] exp_sum_partial [0:NUM_HEADS-1];
    logic signed [ACC_W-1:0] exp_sum_reg [0:NUM_HEADS-1];
    logic signed [ACC_W-1:0] sm_recip_reg [0:NUM_HEADS-1];

    // FE latched inputs
    logic signed [DATA_W-1:0] query_reg [0:CHANNELS-1];
    logic signed [DATA_W-1:0] history_reg [0:HISTORY_DEPTH-1][0:CHANNELS-1];
    logic [HISTORY_DEPTH-1:0] mask_reg;

    // FE input buffer consume signal
    logic fe_consume_in_buf;

    // FE→inter-stage buffer commit signal
    logic fe_commit_valid;
    logic fe_norm_done;

    // =========================================================================
    // Back-End FSM (Aggregate → Output MAC → Residual)
    // =========================================================================
    typedef enum logic [1:0] {
        S_BE_AGGREGATE,
        S_BE_OUT_MAC,
        S_BE_RESIDUAL,
        S_BE_WAIT          // Waiting for inter_buf data (replaces old S_BE_IDLE)
    } be_state_t;

    be_state_t be_state;

    // BE counters
    logic [$clog2(NUM_HEADS*HEAD_DIM):0] be_hd_idx;
    logic [$clog2(NUM_HEADS*HEAD_DIM):0] out_mac_idx;
    logic [$clog2(CHANNELS):0]            ch_out_idx;

    // BE data (read from inter-stage buffer)
    logic signed [ACC_W-1:0]    be_exp_scores_r  [0:NUM_HEADS-1][0:HISTORY_DEPTH-1];
    logic signed [ACC_W-1:0]    be_sm_recip_r     [0:NUM_HEADS-1];
    logic signed [DATA_W-1:0]  be_v_proj_r       [0:HISTORY_DEPTH-1][0:NUM_HEADS-1][0:HEAD_DIM-1];
    logic signed [DATA_W-1:0]  be_query_reg_r    [0:CHANNELS-1];
    logic [HISTORY_DEPTH-1:0]  be_mask_reg_r;

    // BE computation
    logic signed [DATA_W-1:0] attended_per_head [0:NUM_HEADS-1][0:HEAD_DIM-1];
    logic signed [ACC_W-1:0]  out_accum [0:CHANNELS-1];
    logic signed [DATA_W-1:0] output_proj [0:CHANNELS-1];

    // BE→inter-stage buffer consume signal
    logic be_consume_inter_buf;

    // =========================================================================
    // Output Skid Buffer
    // =========================================================================
    localparam int OUT_PACKED_W = CHANNELS * DATA_W;
    logic [OUT_PACKED_W-1:0] skid_data_in, skid_data_out;
    logic be_output_valid, be_output_ready;

    // BE computed output (registered, feeds into skid buffer)
    logic signed [DATA_W-1:0] be_attended_data [0:CHANNELS-1];

    always_comb begin
        for (int c = 0; c < CHANNELS; c++)
            skid_data_in[c*DATA_W +: DATA_W] = be_attended_data[c];
    end

    skid_buffer #(
        .DATA_WIDTH(OUT_PACKED_W),
        .INCLUDE_LAST(0)
    ) u_output_skid (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(skid_data_in),
        .valid_in(be_output_valid),
        .ready_in(be_output_ready),
        .last_in(1'b0),
        .data_out(skid_data_out),
        .valid_out(attended_valid_out),
        .out_ready(attended_ready_in),
        .last_out()
    );

    always_comb begin
        for (int c = 0; c < CHANNELS; c++)
            attended_data_out[c] = skid_data_out[c*DATA_W +: DATA_W];
    end

    // =========================================================================
    // Inter-Stage Buffer
    // =========================================================================
    logic inter_buf_fe_wr_valid;
    logic inter_buf_fe_wr_ready;
    logic inter_buf_be_rd_valid;
    logic inter_buf_be_rd_ready;
    logic [$clog2(INTER_BUF_DEPTH+1)-1:0] inter_buf_count;

    // BE reads from inter-stage buffer outputs (combinational)
    logic signed [ACC_W-1:0]    ib_exp_scores  [0:NUM_HEADS-1][0:HISTORY_DEPTH-1];
    logic signed [ACC_W-1:0]    ib_sm_recip    [0:NUM_HEADS-1];
    logic signed [DATA_W-1:0]   ib_v_proj      [0:HISTORY_DEPTH-1][0:NUM_HEADS-1][0:HEAD_DIM-1];
    logic signed [DATA_W-1:0]   ib_query_reg   [0:CHANNELS-1];
    logic [HISTORY_DEPTH-1:0]   ib_mask_reg;

    temporal_inter_stage_buf #(
        .SLOTS(INTER_BUF_DEPTH),
        .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .HISTORY_DEPTH(HISTORY_DEPTH),
        .CHANNELS(CHANNELS),
        .DATA_W(DATA_W),
        .ACC_W(ACC_W)
    ) u_inter_buf (
        .clk(clk),
        .rst_n(rst_n),
        .fe_wr_valid(inter_buf_fe_wr_valid),
        .fe_wr_ready(inter_buf_fe_wr_ready),
        .fe_exp_scores(exp_scores),
        .fe_sm_recip(sm_recip_reg),
        .fe_v_proj(v_proj),
        .fe_query_reg(query_reg),
        .fe_mask_reg(mask_reg),
        .be_rd_valid(inter_buf_be_rd_valid),
        .be_rd_ready(inter_buf_be_rd_ready),
        .be_exp_scores(ib_exp_scores),
        .be_sm_recip(ib_sm_recip),
        .be_v_proj(ib_v_proj),
        .be_query_reg(ib_query_reg),
        .be_mask_reg(ib_mask_reg),
        .buf_count(inter_buf_count)
    );

    assign inter_buf_be_rd_ready = be_consume_inter_buf;

    // =========================================================================
    // Arithmetic Functions (same as v1)
    // =========================================================================
    function automatic logic signed [ACC_W-1:0] nr_reciprocal(
        input logic signed [ACC_W-1:0] x
    );
        automatic logic signed [ACC_W-1:0] one;
        automatic logic signed [ACC_W-1:0] x0, x1, x2;

        one = (1 << FRAC_W);
        if (x <= 0) begin
            x2 = one;
        end else begin
            x0 = one + ((one - x) >>> 1);
            if (x0 <= 0) x0 = 1;
            x1 = (x0 * (2 * one - ((x * x0) >>> FRAC_W))) >>> FRAC_W;
            if (x1 <= 0) x1 = x0;
            x2 = (x1 * (2 * one - ((x * x1) >>> FRAC_W))) >>> FRAC_W;
            if (x2 <= 0) x2 = one;
        end
        return x2;
    endfunction

    function automatic logic signed [ACC_W-1:0] approx_exp(
        input logic signed [ACC_W-1:0] x
    );
        automatic logic signed [ACC_W-1:0] result;
        automatic logic signed [ACC_W-1:0] abs_x;
        automatic logic signed [ACC_W-1:0] x_sq_half;
        automatic logic signed [ACC_W-1:0] max_val = (1 << FRAC_W);

        abs_x = (x < 0) ? -x : x;
        x_sq_half = (abs_x * abs_x) >>> (FRAC_W + 1);

        if (x >= 0) begin
            if (x > max_val) begin
                result = max_val;
            end else begin
                result = max_val + x + x_sq_half;
                if (result > max_val || result < 0) result = max_val;
            end
        end else begin
            if (abs_x >= (4 << FRAC_W))
                result = 0;
            else begin
                automatic logic signed [ACC_W-1:0] denom;
                denom = max_val + (abs_x >>> FRAC_W) + (x_sq_half >>> FRAC_W);
                // Bounded shift: denom ranges from max_val to max_val+4+eps,
                // so only bits FRAC_W-1 through FRAC_W+2 can be the MSB.
                // Replace 32-bit barrel shifter with case over 5 values.
                case (1'b1)
                    denom[FRAC_W+2]: result = max_val >>> (FRAC_W+2);
                    denom[FRAC_W+1]: result = max_val >>> (FRAC_W+1);
                    denom[FRAC_W]:   result = max_val >>> FRAC_W;
                    denom[FRAC_W-1]: result = max_val >>> (FRAC_W-1);
                    default:         result = max_val >>> (FRAC_W-2);
                endcase
            end
        end
        return result;
    endfunction

    // =========================================================================
    // Front-End FSM
    // =========================================================================
    assign fe_consume_in_buf = (fe_state == S_FE_IDLE) && !in_buf_empty && cfg_enable;
    assign fe_commit_valid   = fe_norm_done && inter_buf_fe_wr_ready;
    assign fe_norm_done     = (fe_state == S_FE_SOFTMAX_NORM) && ((head_idx + PARALLEL_DIV) >= NUM_HEADS);
    assign inter_buf_fe_wr_valid = fe_norm_done && inter_buf_fe_wr_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fe_state <= S_FE_IDLE;
            mac_idx <= '0;
            hd_idx <= '0;
            t_idx <= '0;
            score_t_idx <= '0;
            attn_pass <= '0;
            head_idx <= '0;
            exp_idx <= '0;
            for (int h = 0; h < NUM_HEADS; h++) begin
                for (int d = 0; d < HEAD_DIM; d++) begin
                    q_proj[h][d] <= '0;
                    q_accum[h][d] <= '0;
                end
                for (int t = 0; t < HISTORY_DEPTH; t++) begin
                    attention_scores[h][t] <= '0;
                    exp_scores[h][t] <= '0;
                    exp_sum_partial[h] <= '0;
                    exp_sum_reg[h] <= '0;
                    sm_recip_reg[h] <= '0;
                    for (int d = 0; d < HEAD_DIM; d++) begin
                        k_proj[t][h][d] <= '0;
                        v_proj[t][h][d] <= '0;
                        k_accum[t][h][d] <= '0;
                        v_accum[t][h][d] <= '0;
                    end
                end
            end
            for (int c = 0; c < CHANNELS; c++) query_reg[c] <= '0;
            for (int t = 0; t < HISTORY_DEPTH; t++)
                for (int c = 0; c < CHANNELS; c++)
                    history_reg[t][c] <= '0;
            mask_reg <= '0;
        end else begin
            case (fe_state)
                S_FE_IDLE: begin
                    // Combinational transition: start immediately if input available
                    if (!in_buf_empty && cfg_enable) begin
                        for (int c = 0; c < CHANNELS; c++)
                            query_reg[c] <= in_buf_query[in_rd_ptr][c];
                        for (int t = 0; t < HISTORY_DEPTH; t++)
                            for (int c = 0; c < CHANNELS; c++)
                                history_reg[t][c] <= in_buf_history[in_rd_ptr][t][c];
                        mask_reg <= in_buf_mask[in_rd_ptr];

                        // Clear accumulators
                        for (int h = 0; h < NUM_HEADS; h++)
                            for (int d = 0; d < HEAD_DIM; d++)
                                q_accum[h][d] <= '0;
                        for (int t = 0; t < HISTORY_DEPTH; t++)
                            for (int h = 0; h < NUM_HEADS; h++)
                                for (int d = 0; d < HEAD_DIM; d++) begin
                                    k_accum[t][h][d] <= '0;
                                    v_accum[t][h][d] <= '0;
                                end

                        mac_idx <= '0;
                        hd_idx <= '0;
                        t_idx <= '0;
                        fe_state <= S_FE_PROJ_MAC;
                    end
                end

                S_FE_PROJ_MAC: begin
                    // Chunk validity check
                    automatic logic chunk_active = 1'b0;
                    for (int i = 0; i < PARALLEL_HIST; i++) begin
                        if ((t_idx + i) < HISTORY_DEPTH && (!ENABLE_MASK || mask_reg[t_idx + i]))
                            chunk_active = 1'b1;
                    end

                    if (!chunk_active) begin
                        // Skip invalid chunk
                        for (int i = 0; i < PARALLEL_HIST; i++) begin
                            if ((t_idx + i) < HISTORY_DEPTH) begin
                                for (int h = 0; h < NUM_HEADS; h++)
                                    for (int d = 0; d < HEAD_DIM; d++) begin
                                        k_proj[t_idx + i][h][d] <= '0;
                                        v_proj[t_idx + i][h][d] <= '0;
                                    end
                            end
                        end

                        if ((t_idx + PARALLEL_HIST) >= HISTORY_DEPTH) begin
                            t_idx <= '0;
                            mac_idx <= '0;
                            hd_idx <= '0;
                            score_t_idx <= '0;
                            attn_pass <= '0;
                            fe_state <= S_FE_ATTN_SCORE;
                        end else begin
                            t_idx <= t_idx + PARALLEL_HIST;
                        end

                    end else begin
                        // MAC loop (identical to v1)
                        for (int i = 0; i < HD_UNROLL; i++) begin
                            automatic int hd_pair = hd_idx + i;
                            if (hd_pair < NUM_HEADS * HEAD_DIM) begin
                                automatic int h = hd_pair / HEAD_DIM;
                                automatic int d = hd_pair % HEAD_DIM;

                                automatic logic signed [ACC_W-1:0] q_acc_next = q_accum[h][d];
                                automatic logic signed [ACC_W-1:0] k_acc_next [PARALLEL_HIST-1:0];
                                automatic logic signed [ACC_W-1:0] v_acc_next [PARALLEL_HIST-1:0];
                                for (int ii = 0; ii < PARALLEL_HIST; ii++) begin
                                    k_acc_next[ii] = k_accum[t_idx + ii][h][d];
                                    v_acc_next[ii] = v_accum[t_idx + ii][h][d];
                                end
                                for (int u = 0; u < MAC_UNROLL; u++) begin
                                    automatic logic [$clog2(CHANNELS)-1:0] ch_idx;
                                    ch_idx = mac_idx[$clog2(CHANNELS)-1:0] + u;
                                    if (ch_idx < CHANNELS) begin
                                        if (t_idx == 0)
                                            q_acc_next = q_acc_next +
                                                (query_reg[ch_idx] *
                                                 q_weights[h * HEAD_DIM * CHANNELS + d * CHANNELS + ch_idx]);

                                        for (int ii = 0; ii < PARALLEL_HIST; ii++) begin
                                            if ((t_idx + ii) < HISTORY_DEPTH && (!ENABLE_MASK || mask_reg[t_idx + ii])) begin
                                                k_acc_next[ii] = k_acc_next[ii] +
                                                    (history_reg[t_idx + ii][ch_idx] *
                                                     (QK_WEIGHT_SHARING ? q_weights[h * HEAD_DIM * CHANNELS + d * CHANNELS + ch_idx]
                                                                        : k_weights[h * HEAD_DIM * CHANNELS + d * CHANNELS + ch_idx]));
                                                v_acc_next[ii] = v_acc_next[ii] +
                                                    (history_reg[t_idx + ii][ch_idx] *
                                                     v_weights[h * HEAD_DIM * CHANNELS + d * CHANNELS + ch_idx]);
                                            end
                                        end
                                    end
                                end
                                q_accum[h][d] <= q_acc_next;
                                for (int ii = 0; ii < PARALLEL_HIST; ii++) begin
                                    k_accum[t_idx + ii][h][d] <= k_acc_next[ii];
                                    v_accum[t_idx + ii][h][d] <= v_acc_next[ii];
                                end
                            end
                        end

                        // (h,d) group advancement
                        if (hd_idx + HD_UNROLL >= NUM_HEADS * HEAD_DIM) begin
                            hd_idx <= '0;
                            if (mac_idx + MAC_UNROLL >= CHANNELS) begin
                                mac_idx <= '0;
                                if (t_idx == 0) begin
                                    for (int h = 0; h < NUM_HEADS; h++)
                                        for (int d = 0; d < HEAD_DIM; d++)
                                            q_proj[h][d] <= DATA_W'(q_accum[h][d] >>> FRAC_W);
                                end
                                for (int i = 0; i < PARALLEL_HIST; i++) begin
                                    if ((t_idx + i) < HISTORY_DEPTH) begin
                                        if (!ENABLE_MASK || mask_reg[t_idx + i]) begin
                                            for (int h = 0; h < NUM_HEADS; h++)
                                                for (int d = 0; d < HEAD_DIM; d++) begin
                                                    k_proj[t_idx + i][h][d] <= DATA_W'(k_accum[t_idx + i][h][d] >>> FRAC_W);
                                                    v_proj[t_idx + i][h][d] <= DATA_W'(v_accum[t_idx + i][h][d] >>> FRAC_W);
                                                end
                                        end else begin
                                            for (int h = 0; h < NUM_HEADS; h++)
                                                for (int d = 0; d < HEAD_DIM; d++) begin
                                                    k_proj[t_idx + i][h][d] <= '0;
                                                    v_proj[t_idx + i][h][d] <= '0;
                                                end
                                        end
                                    end
                                end
                                if ((t_idx + PARALLEL_HIST) >= HISTORY_DEPTH) begin
                                    t_idx <= '0;
                                    score_t_idx <= '0;
                                    attn_pass <= '0;
                                    fe_state <= S_FE_ATTN_SCORE;
                                end else begin
                                    t_idx <= t_idx + PARALLEL_HIST;
                                end
                            end else begin
                                mac_idx <= mac_idx + MAC_UNROLL;
                            end
                        end else begin
                            hd_idx <= hd_idx + HD_UNROLL;
                        end
                    end
                end

                S_FE_ATTN_SCORE: begin
                    for (int hp = 0; hp < ATTN_HEADS_PER_CYCLE; hp++) begin
                        automatic int h = attn_pass * ATTN_HEADS_PER_CYCLE + hp;
                        if (h < NUM_HEADS) begin
                            for (int to = 0; to < SCORE_HIST_UNROLL; to++) begin
                                automatic int t = score_t_idx + to;
                                if (t < HISTORY_DEPTH) begin
                                    if (!ENABLE_MASK || mask_reg[t]) begin
                                        if (cfg_attention_mode == MODE_FULL ||
                                            (cfg_attention_mode == MODE_LOCAL && t >= cfg_history_depth - cfg_local_window) ||
                                            (cfg_attention_mode == MODE_STRIDED && (t & (NUM_HEADS - 1)) == h)) begin
                                            automatic logic signed [ACC_W-1:0] score = '0;
                                            for (int d = 0; d < HEAD_DIM; d++)
                                                score = score + (q_proj[h][d] * k_proj[t][h][d]);

                                            if (ENABLE_TEMPERATURE) begin
                                                // Bounded shift for temperature:
                                                // cfg_temperature MSB is near FRAC_W (0.5x-4x range).
                                                // Replace highest_bit_pos + barrel shifter with
                                                // case (1'b1) over 5 candidate MSB positions.
                                                automatic logic signed [ACC_W-1:0] temp_scaled;
                                                if (cfg_temperature == '0) begin
                                                    temp_scaled = score >>> (FRAC_W + SQRT_HEAD_DIM_SHIFT);
                                                end else begin
                                                    case (1'b1)
                                                        cfg_temperature[FRAC_W+2]:
                                                            temp_scaled = (score >>> (FRAC_W+2)) >>> SQRT_HEAD_DIM_SHIFT;
                                                        cfg_temperature[FRAC_W+1]:
                                                            temp_scaled = (score >>> (FRAC_W+1)) >>> SQRT_HEAD_DIM_SHIFT;
                                                        cfg_temperature[FRAC_W]:
                                                            temp_scaled = (score >>> FRAC_W) >>> SQRT_HEAD_DIM_SHIFT;
                                                        cfg_temperature[FRAC_W-1]:
                                                            temp_scaled = (score >>> (FRAC_W-1)) >>> SQRT_HEAD_DIM_SHIFT;
                                                        default:
                                                            temp_scaled = (score >>> (FRAC_W-2)) >>> SQRT_HEAD_DIM_SHIFT;
                                                    endcase
                                                end
                                                attention_scores[h][t] <= temp_scaled;
                                            end else begin
                                                attention_scores[h][t] <= (score >>> (FRAC_W + SQRT_HEAD_DIM_SHIFT));
                                            end
                                        end else begin
                                            attention_scores[h][t] <= '0;
                                        end
                                    end else begin
                                        attention_scores[h][t] <= '0;
                                    end
                                end
                            end
                        end
                    end

                    if (score_t_idx + SCORE_HIST_UNROLL >= HISTORY_DEPTH) begin
                        score_t_idx <= '0;
                        if (attn_pass + 1 >= ATTN_PASSES) begin
                            exp_idx <= '0;
                            fe_state <= S_FE_SOFTMAX_EXP;
                        end else begin
                            attn_pass <= attn_pass + 1'b1;
                        end
                    end else begin
                        score_t_idx <= score_t_idx + SCORE_HIST_UNROLL;
                    end
                end

                S_FE_SOFTMAX_EXP: begin
                    automatic logic signed [ACC_W-1:0] acc [0:NUM_HEADS-1];
                    for (int h = 0; h < NUM_HEADS; h++)
                        acc[h] = (exp_idx == 0) ? '0 : exp_sum_partial[h];

                    for (int i = 0; i < EXP_UNROLL; i++) begin
                        automatic int flat_idx = exp_idx + i;
                        if (flat_idx < NUM_HEADS * HISTORY_DEPTH) begin
                            automatic int h = flat_idx / HISTORY_DEPTH;
                            automatic int t = flat_idx % HISTORY_DEPTH;
                            automatic logic signed [ACC_W-1:0] exp_val;
                            if ((!ENABLE_MASK || mask_reg[t]) && attention_scores[h][t] != '0) begin
                                exp_val = approx_exp(attention_scores[h][t]);
                                exp_scores[h][t] <= exp_val;
                                acc[h] = acc[h] + exp_val;
                            end else begin
                                exp_scores[h][t] <= '0;
                            end
                        end
                    end

                    for (int h = 0; h < NUM_HEADS; h++)
                        exp_sum_partial[h] <= acc[h];

                    if (exp_idx + EXP_UNROLL >= NUM_HEADS * HISTORY_DEPTH) begin
                        exp_idx <= '0;
                        head_idx <= '0;
                        for (int h = 0; h < NUM_HEADS; h++)
                            exp_sum_reg[h] <= acc[h];
                        fe_state <= S_FE_SOFTMAX_NORM;
                    end else begin
                        exp_idx <= exp_idx + EXP_UNROLL;
                    end
                end

                S_FE_SOFTMAX_NORM: begin
                    for (int i = 0; i < PARALLEL_DIV; i++) begin
                        if ((head_idx + i) < NUM_HEADS) begin
                            sm_recip_reg[head_idx + i] <= nr_reciprocal(exp_sum_reg[head_idx + i]);
                        end
                    end

                    if ((head_idx + PARALLEL_DIV) >= NUM_HEADS) begin
                        head_idx <= '0;
                        // Check if inter-stage buffer has space
                        if (inter_buf_fe_wr_ready) begin
                            // Commit to inter-stage buffer and return to idle
                            fe_state <= S_FE_IDLE;
                        end
                        // If buffer is full, hold here (stall FE until space opens)
                    end else begin
                        head_idx <= head_idx + PARALLEL_DIV;
                    end
                end

                default: fe_state <= S_FE_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Back-End FSM
    // =========================================================================
    // BE consumes inter-stage buffer entry when transitioning to AGGREGATE.
    // This fires when: (1) in WAIT state with data available, or (2) in RESIDUAL
    // completing with data available (direct transition, skipping WAIT).
    assign be_consume_inter_buf =
        ((be_state == S_BE_WAIT) && inter_buf_be_rd_valid) ||
        ((be_state == S_BE_RESIDUAL) && be_output_ready && inter_buf_be_rd_valid);

    // Latch inter-stage buffer data into BE registers when consuming.
    // S_BE_IDLE has been merged: transitioning to S_BE_AGGREGATE latches data
    // in the same cycle, and aggregation starts on the *next* cycle (the
    // latched registers are available after the clock edge). This saves
    // 1 cycle per query compared to the old IDLE→AGGREGATE two-step path.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            be_state <= S_BE_WAIT;
            be_hd_idx <= '0;
            out_mac_idx <= '0;
            ch_out_idx <= '0;
            be_output_valid <= 1'b0;
            for (int h = 0; h < NUM_HEADS; h++)
                for (int d = 0; d < HEAD_DIM; d++)
                    attended_per_head[h][d] <= '0;
            for (int c = 0; c < CHANNELS; c++) begin
                out_accum[c] <= '0;
                output_proj[c] <= '0;
                be_attended_data[c] <= '0;
            end
        end else begin
            be_output_valid <= 1'b0;

            case (be_state)
                S_BE_WAIT: begin
                    // Waiting for inter-stage buffer data.
                    // When data is available, latch it and go to AGGREGATE.
                    if (inter_buf_be_rd_valid) begin
                        for (int h = 0; h < NUM_HEADS; h++) begin
                            for (int t = 0; t < HISTORY_DEPTH; t++)
                                be_exp_scores_r[h][t] <= ib_exp_scores[h][t];
                            be_sm_recip_r[h] <= ib_sm_recip[h];
                        end
                        for (int t = 0; t < HISTORY_DEPTH; t++)
                            for (int h = 0; h < NUM_HEADS; h++)
                                for (int d = 0; d < HEAD_DIM; d++)
                                    be_v_proj_r[t][h][d] <= ib_v_proj[t][h][d];
                        for (int c = 0; c < CHANNELS; c++)
                            be_query_reg_r[c] <= ib_query_reg[c];
                        be_mask_reg_r <= ib_mask_reg;
                        be_hd_idx <= '0;
                        for (int c = 0; c < CHANNELS; c++)
                            out_accum[c] <= '0;
                        be_state <= S_BE_AGGREGATE;
                    end
                end

                S_BE_AGGREGATE: begin
                    for (int i = 0; i < HD_UNROLL; i++) begin
                        automatic int hd_pair = be_hd_idx + i;
                        if (hd_pair < NUM_HEADS * HEAD_DIM) begin
                            automatic int h = hd_pair / HEAD_DIM;
                            automatic int d = hd_pair % HEAD_DIM;
                            automatic logic signed [ACC_W-1:0] recip = be_sm_recip_r[h];
                            automatic logic signed [ACC_W-1:0] agg = '0;
                            for (int t = 0; t < HISTORY_DEPTH; t++) begin
                                if (!ENABLE_MASK || be_mask_reg_r[t]) begin
                                    automatic logic signed [ACC_W-1:0] w;
                                    w = (be_exp_scores_r[h][t] * recip) >>> FRAC_W;
                                    agg = agg + (w * be_v_proj_r[t][h][d]);
                                end
                            end
                            attended_per_head[h][d] <= DATA_W'(agg >>> FRAC_W);
                        end
                    end

                    if (be_hd_idx + HD_UNROLL >= NUM_HEADS * HEAD_DIM) begin
                        be_hd_idx <= '0;
                        out_mac_idx <= '0;
                        ch_out_idx <= '0;
                        for (int c = 0; c < CHANNELS; c++)
                            out_accum[c] <= '0;
                        be_state <= S_BE_OUT_MAC;
                    end else begin
                        be_hd_idx <= be_hd_idx + HD_UNROLL;
                    end
                end

                S_BE_OUT_MAC: begin
                    for (int co = 0; co < CH_OUT_UNROLL; co++) begin
                        automatic int c = ch_out_idx + co;
                        if (c < CHANNELS) begin
                            automatic logic signed [ACC_W-1:0] out_acc_next = out_accum[c];
                            for (int u = 0; u < OUT_MAC_UNROLL; u++) begin
                                automatic logic [$clog2(NUM_HEADS*HEAD_DIM)-1:0] idx;
                                idx = out_mac_idx[$clog2(NUM_HEADS*HEAD_DIM)-1:0] + u;
                                if (idx < NUM_HEADS * HEAD_DIM) begin
                                    out_acc_next = out_acc_next +
                                        (attended_per_head[idx / HEAD_DIM]
                                                          [idx & (HEAD_DIM - 1)] *
                                         out_weights[c * NUM_HEADS * HEAD_DIM + idx]);
                                end
                            end
                            out_accum[c] <= out_acc_next;
                        end
                    end

                    if (out_mac_idx + OUT_MAC_UNROLL >= NUM_HEADS * HEAD_DIM) begin
                        out_mac_idx <= '0;
                        if (ch_out_idx + CH_OUT_UNROLL >= CHANNELS) begin
                            ch_out_idx <= '0;
                            for (int c = 0; c < CHANNELS; c++)
                                output_proj[c] <= DATA_W'(out_accum[c] >>> FRAC_W);
                            be_state <= S_BE_RESIDUAL;
                        end else begin
                            ch_out_idx <= ch_out_idx + CH_OUT_UNROLL;
                        end
                    end else begin
                        out_mac_idx <= out_mac_idx + OUT_MAC_UNROLL;
                    end
                end

                S_BE_RESIDUAL: begin
                    // Compute residual and push to skid buffer
                    for (int c = 0; c < CHANNELS; c++) begin
                        if (ENABLE_RESIDUAL) begin
                            automatic logic signed [ACC_W-1:0] res =
                                ACC_W'(output_proj[c]) + ACC_W'(be_query_reg_r[c]);
                            if (res > (1 <<< (DATA_W - 1)) - 1)
                                be_attended_data[c] <= (1 <<< (DATA_W - 1)) - 1;
                            else if (res < -(1 <<< (DATA_W - 1)))
                                be_attended_data[c] <= -(1 <<< (DATA_W - 1));
                            else
                                be_attended_data[c] <= DATA_W'(res);
                        end else begin
                            be_attended_data[c] <= output_proj[c];
                        end
                    end
                    be_output_valid <= 1'b1;
                    // Advance once skid buffer accepts.
                    // If inter_buf has data, go directly to AGGREGATE (latch on
                    // this cycle, aggregate starts next cycle — skips WAIT).
                    // Otherwise go to WAIT.
                    if (be_output_ready) begin
                        if (inter_buf_be_rd_valid) begin
                            // Direct transition: latch new data, skip WAIT
                            for (int h = 0; h < NUM_HEADS; h++) begin
                                for (int t = 0; t < HISTORY_DEPTH; t++)
                                    be_exp_scores_r[h][t] <= ib_exp_scores[h][t];
                                be_sm_recip_r[h] <= ib_sm_recip[h];
                            end
                            for (int t = 0; t < HISTORY_DEPTH; t++)
                                for (int h = 0; h < NUM_HEADS; h++)
                                    for (int d = 0; d < HEAD_DIM; d++)
                                        be_v_proj_r[t][h][d] <= ib_v_proj[t][h][d];
                            for (int c = 0; c < CHANNELS; c++)
                                be_query_reg_r[c] <= ib_query_reg[c];
                            be_mask_reg_r <= ib_mask_reg;
                            be_hd_idx <= '0;
                            for (int c = 0; c < CHANNELS; c++)
                                out_accum[c] <= '0;
                            be_state <= S_BE_AGGREGATE;
                        end else begin
                            be_state <= S_BE_WAIT;
                        end
                    end
                end

                default: be_state <= S_BE_WAIT;
            endcase
        end
    end

    // =========================================================================
    // Weight Loading (shared between FE and BE reads, no port conflict)
    // =========================================================================
    generate
    if (WEIGHT_LOAD_MODE == 1) begin : gen_mem_weights
        initial begin
            if (WEIGHT_FILE != "")
                $readmemh(WEIGHT_FILE, q_weights);
        end
    end else begin : gen_runtime_weights
        always_ff @(posedge clk) begin
            if (weight_valid) begin
                if (QK_WEIGHT_SHARING) begin
                    if (weight_addr < CHANNELS * NUM_HEADS * HEAD_DIM) begin
                        q_weights[weight_addr] <= weight_data;
                    end else if (weight_addr < 2 * CHANNELS * NUM_HEADS * HEAD_DIM) begin
                        v_weights[weight_addr - CHANNELS * NUM_HEADS * HEAD_DIM] <= weight_data;
                    end else begin
                        out_weights[weight_addr - 2 * CHANNELS * NUM_HEADS * HEAD_DIM] <= weight_data;
                    end
                end else begin
                    if (weight_addr < CHANNELS * NUM_HEADS * HEAD_DIM) begin
                        q_weights[weight_addr] <= weight_data;
                    end else if (weight_addr < 2 * CHANNELS * NUM_HEADS * HEAD_DIM) begin
                        k_weights[weight_addr - CHANNELS * NUM_HEADS * HEAD_DIM] <= weight_data;
                    end else if (weight_addr < 3 * CHANNELS * NUM_HEADS * HEAD_DIM) begin
                        v_weights[weight_addr - 2 * CHANNELS * NUM_HEADS * HEAD_DIM] <= weight_data;
                    end else begin
                        out_weights[weight_addr - 3 * CHANNELS * NUM_HEADS * HEAD_DIM] <= weight_data;
                    end
                end
            end
        end
    end
    endgenerate

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    TEMPORAL_ATTN_HEADS: assert property (
        @(posedge clk) disable iff (!rst_n)
        NUM_HEADS > 0 && HEAD_DIM > 0
    ) else $error("temporal_attention_enhanced_v2: Invalid head configuration");

    TEMPORAL_ATTN_HISTORY: assert property (
        @(posedge clk) disable iff (!rst_n)
        cfg_history_depth > 0 && cfg_history_depth <= HISTORY_DEPTH
    ) else $error("temporal_attention_enhanced_v2: Invalid history depth");

    // Pipeline invariants
    BE_START_IMPLIES_DATA: assert property (
        @(posedge clk) disable iff (!rst_n)
        (be_state == S_BE_AGGREGATE && $past(be_state == S_BE_WAIT || be_state == S_BE_RESIDUAL)) |-> $past(inter_buf_be_rd_valid)
    ) else $error("v2: BE started without inter_buf data");

    BE_OUTPUT_STABLE: assert property (
        @(posedge clk) disable iff (!rst_n)
        (be_output_valid && !be_output_ready) |=> be_output_valid
    ) else $error("v2: BE output deasserted before skid accepted");

    // Cover FE/BE overlap
    FE_BE_OVERLAP: cover property (
        @(posedge clk) disable iff (!rst_n)
        fe_state inside {S_FE_PROJ_MAC, S_FE_ATTN_SCORE, S_FE_SOFTMAX_EXP, S_FE_SOFTMAX_NORM}
        && be_state inside {S_BE_AGGREGATE, S_BE_OUT_MAC, S_BE_RESIDUAL}
    );
    // synthesis translate_on

endmodule