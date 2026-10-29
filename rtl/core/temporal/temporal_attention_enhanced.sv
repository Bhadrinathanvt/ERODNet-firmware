// =============================================================================
// temporal_attention_enhanced.sv
//
// Enhanced Multi-Head Temporal Attention for Radar Frame Aggregation.
//
// FIXED: P0-2 — Major rework:
//   1. Multi-cycle MAC for Q/K/V projections (65536→512 multiplies/cycle)
//   2. Multi-cycle MAC for output projection (4096→64 multiplies/cycle)
//   3. Newton-Raphson reciprocal replaces 32 combinational softmax dividers
//   4. Pipeline latency increased but timing closure achieved
//
// Pipeline: INPUT → Q_MAC → KV_MAC → ATTN_SCORE → SOFTMAX_EXP → SOFTMAX_NORM →
//           AGGREGATE → OUT_MAC → RESIDUAL → OUTPUT
// OPTIMIZATION: S_SOFTMAX_APPLY removed — softmax weight is applied directly
// in S_AGGREGATE using the NR reciprocal computed in S_SOFTMAX_NORM (1 cycle saved per frame).
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;

module temporal_attention_enhanced #(
    parameter int HISTORY_DEPTH   = pkg_config::HISTORY_DEPTH,
    parameter int CHANNELS        = MAX_C,
    parameter int NUM_HEADS       = pkg_config::NUM_HEADS,
    parameter int HEAD_DIM        = pkg_config::HEAD_DIM,
    parameter int DATA_W = pkg_module_config::TEMPORAL_ATTENTION_ENHANCED_DATA_WIDTH,
    parameter int FRAC_W          = pkg_module_config::TEMPORAL_ATTENTION_ENHANCED_FRAC_WIDTH,
    parameter int ACC_W           = pkg_config::ACC_WIDTH,
    parameter int WEIGHT_LOAD_MODE = pkg_config::WEIGHT_LOAD_MODE,
    parameter string WEIGHT_FILE   = "",
    // Synthesizable-time parallelism — configurable via config.yaml
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
    parameter bit  QK_WEIGHT_SHARING  = pkg_config::QK_WEIGHT_SHARING
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
    // Weight Storage
    // =========================================================================
    localparam int QKV_WEIGHTS = 3 * CHANNELS * NUM_HEADS * HEAD_DIM;
    localparam int OUT_WEIGHTS = NUM_HEADS * HEAD_DIM * CHANNELS;
    localparam int PARALLEL_DIV = pkg_config::PARALLEL_DIV; // Number of physical NR dividers to synthesize
    localparam int PARALLEL_HIST = pkg_config::PARALLEL_HIST; // Number of history frames to process concurrently

    (* ram_style = "block" *)
    logic signed [DATA_W-1:0] q_weights [0:CHANNELS*NUM_HEADS*HEAD_DIM-1];
    (* ram_style = "block" *)
    logic signed [DATA_W-1:0] k_weights [0:CHANNELS*NUM_HEADS*HEAD_DIM-1];
    (* ram_style = "block" *)
    logic signed [DATA_W-1:0] v_weights [0:CHANNELS*NUM_HEADS*HEAD_DIM-1];
    (* ram_style = "block" *)
    logic signed [DATA_W-1:0] out_weights [0:OUT_WEIGHTS-1];

    // =========================================================================
    // FSM State
    // =========================================================================
    typedef enum logic [3:0] {
        S_IDLE,           // Waiting for valid input
        S_PROJ_MAC,         // Multi-cycle K/V projection MAC
        S_ATTN_SCORE,     // Scaled dot-product attention (1 cycle)
        S_SOFTMAX_EXP,    // Compute exp and sum (1 cycle)
        S_SOFTMAX_NORM,   // Compute Newton-Raphson reciprocal (1 cycle)
        S_AGGREGATE,      // Softmax apply + weighted value aggregation (1 cycle, merged)
        S_OUT_MAC,        // Multi-cycle output projection MAC
        S_RESIDUAL        // Residual connection and output (1 cycle)
    } state_t;

    state_t state;

    // =========================================================================
    // MAC counters and accumulators
    // =========================================================================
    logic [$clog2(CHANNELS):0] mac_idx;        // Channel index for Q/KV MAC
    logic [$clog2(NUM_HEADS*HEAD_DIM):0] out_mac_idx;  // Index for output MAC

    logic signed [ACC_W-1:0] q_accum [0:NUM_HEADS-1][0:HEAD_DIM-1];
    logic signed [ACC_W-1:0] k_accum [0:HISTORY_DEPTH-1][0:NUM_HEADS-1][0:HEAD_DIM-1];
    logic signed [ACC_W-1:0] v_accum [0:HISTORY_DEPTH-1][0:NUM_HEADS-1][0:HEAD_DIM-1];

    logic signed [DATA_W-1:0] q_proj [0:NUM_HEADS-1][0:HEAD_DIM-1];
    logic signed [DATA_W-1:0] k_proj [0:HISTORY_DEPTH-1][0:NUM_HEADS-1][0:HEAD_DIM-1];
    logic signed [DATA_W-1:0] v_proj [0:HISTORY_DEPTH-1][0:NUM_HEADS-1][0:HEAD_DIM-1];

    // Attention scores
    logic signed [ACC_W-1:0] attention_scores [0:NUM_HEADS-1][0:HISTORY_DEPTH-1];

    // Softmax intermediates
    logic signed [ACC_W-1:0] exp_scores [0:NUM_HEADS-1][0:HISTORY_DEPTH-1];
    logic signed [ACC_W-1:0] exp_sum_reg [0:NUM_HEADS-1];
    logic signed [ACC_W-1:0] sm_recip_reg [0:NUM_HEADS-1];  // NR reciprocal (registered in S_SOFTMAX_NORM, consumed in S_AGGREGATE)

    // Aggregation
    logic signed [DATA_W-1:0] attended_per_head [0:NUM_HEADS-1][0:HEAD_DIM-1];

    // Output projection
    logic signed [ACC_W-1:0] out_accum [0:CHANNELS-1];
    logic signed [DATA_W-1:0] output_proj [0:CHANNELS-1];

    // Registered inputs (held stable during multi-cycle MAC)
    logic signed [DATA_W-1:0] query_reg [0:CHANNELS-1];
    logic signed [DATA_W-1:0] history_reg [0:HISTORY_DEPTH-1][0:CHANNELS-1];
    logic [HISTORY_DEPTH-1:0] mask_reg;

    // Attention score pass counter (for ATTN_HEADS_PER_CYCLE < NUM_HEADS)
    localparam int ATTN_PASSES = (NUM_HEADS + ATTN_HEADS_PER_CYCLE - 1) / ATTN_HEADS_PER_CYCLE;
    logic [$clog2(ATTN_PASSES):0] attn_pass;

    // Newton-Raphson reciprocal function
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

    // Piecewise exponential approximation (Optimized for 32-bit math)
    function automatic logic signed [ACC_W-1:0] approx_exp(
        input logic signed [ACC_W-1:0] x
    );
        automatic logic signed [ACC_W-1:0] result;
        automatic logic signed [ACC_W-1:0] abs_x;
        automatic logic signed [ACC_W-1:0] x_sq_half;
        automatic logic signed [ACC_W-1:0] max_val = (1 << FRAC_W); // 1.0 in Q format

        abs_x = (x < 0) ? -x : x;
        x_sq_half = (abs_x * abs_x) >>> (FRAC_W + 1);

        if (x >= 0) begin
            // Clamp before addition to avoid 64-bit promotion and overflow
            if (x > max_val) begin
                result = max_val;
            end else begin
                result = max_val + x + x_sq_half;
                // Safe guard against 32-bit wrap-around
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
    // Main FSM
    // =========================================================================
    localparam int SQRT_HEAD_DIM_SHIFT = $clog2(HEAD_DIM) / 2;
    logic [$clog2(HISTORY_DEPTH):0] t_idx; // History frame counter
    logic [$clog2(NUM_HEADS):0] head_idx; // Head counter for sharing logic
    logic [$clog2(NUM_HEADS*HEAD_DIM):0] hd_idx; // (h,d) counter for HD_UNROLL time-sharing
    logic [$clog2(HISTORY_DEPTH):0] score_t_idx; // History-frame counter for SCORE_HIST_UNROLL
    logic [$clog2(CHANNELS):0] ch_out_idx; // Channel-group counter for CH_OUT_UNROLL
    logic [$clog2(NUM_HEADS*HISTORY_DEPTH):0] exp_idx; // (h,t) pair counter for EXP_UNROLL
    logic signed [ACC_W-1:0] exp_sum_partial [0:NUM_HEADS-1]; // Per-head exp partial sums

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            mac_idx <= '0;
            out_mac_idx <= '0;
            attended_valid_out <= 1'b0;
            t_idx <= '0;
            head_idx <= '0;
            hd_idx <= '0;
            score_t_idx <= '0;
            ch_out_idx <= '0;
            exp_idx <= '0;
            attn_pass <= '0;
            for (int c = 0; c < CHANNELS; c++) begin
                query_reg[c] <= '0;
                attended_data_out[c] <= '0;
                output_proj[c] <= '0;
                out_accum[c] <= '0;
            end
            for (int t = 0; t < HISTORY_DEPTH; t++)
                for (int c = 0; c < CHANNELS; c++)
                    history_reg[t][c] <= '0;
            mask_reg <= '0;
            for (int h = 0; h < NUM_HEADS; h++) begin
                exp_sum_partial[h] <= '0;
                exp_sum_reg[h] <= '0;
                sm_recip_reg[h] <= '0;
                for (int d = 0; d < HEAD_DIM; d++) begin
                    q_proj[h][d] <= '0;
                    q_accum[h][d] <= '0;
                    attended_per_head[h][d] <= '0;
                end
                for (int t = 0; t < HISTORY_DEPTH; t++) begin
                    attention_scores[h][t] <= '0;
                    exp_scores[h][t] <= '0;
                    for (int d = 0; d < HEAD_DIM; d++) begin
                        k_proj[t][h][d] <= '0;
                        v_proj[t][h][d] <= '0;
                        k_accum[t][h][d] <= '0;
                        v_accum[t][h][d] <= '0;
                    end
                end
            end
        end else begin
            attended_valid_out <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (query_valid_in && history_valid_in && cfg_enable) begin
                        // Latch inputs
                        for (int c = 0; c < CHANNELS; c++)
                            query_reg[c] <= query_data[c];
                        for (int t = 0; t < HISTORY_DEPTH; t++)
                            for (int c = 0; c < CHANNELS; c++)
                                history_reg[t][c] <= history_data[t][c];
                        mask_reg <= ENABLE_MASK ? history_valid_mask : {(HISTORY_DEPTH){1'b1}};

                        // Clear Q accumulators
                        for (int h = 0; h < NUM_HEADS; h++)
                            for (int d = 0; d < HEAD_DIM; d++)
                                q_accum[h][d] <= '0;

                        // Clear K/V accumulators for first chunk
                        for (int t = 0; t < HISTORY_DEPTH; t++)
                            for (int h = 0; h < NUM_HEADS; h++)
                                for (int d = 0; d < HEAD_DIM; d++) begin
                                    k_accum[t][h][d] <= '0;
                                    v_accum[t][h][d] <= '0;
                                end

                        mac_idx <= '0;
                        state <= S_PROJ_MAC;
                    end
                end

                S_PROJ_MAC: begin
                    // -----------------------------------------------------------------
                    // EVALUATE CHUNK: Check if any frame in the current chunk is valid
                    // -----------------------------------------------------------------
                    automatic logic chunk_active = 1'b0;
                    for (int i = 0; i < PARALLEL_HIST; i++) begin
                        if ((t_idx + i) < HISTORY_DEPTH && (!ENABLE_MASK || mask_reg[t_idx + i])) begin
                            chunk_active = 1'b1;
                        end
                    end

                    if (!chunk_active) begin
                        // -----------------------------------------------------------------
                        // EARLY EXIT: Entire chunk is invalid, skip the 64-cycle MAC
                        // -----------------------------------------------------------------
                        for (int i = 0; i < PARALLEL_HIST; i++) begin
                            if ((t_idx + i) < HISTORY_DEPTH) begin
                                for (int h = 0; h < NUM_HEADS; h++) begin
                                    for (int d = 0; d < HEAD_DIM; d++) begin
                                        k_proj[t_idx + i][h][d] <= '0;
                                        v_proj[t_idx + i][h][d] <= '0;
                                    end
                                end
                            end
                        end

                        if ((t_idx + PARALLEL_HIST) >= HISTORY_DEPTH) begin
                            t_idx <= '0;
                            mac_idx <= '0;
                            hd_idx <= '0;
                            score_t_idx <= '0;
                            attn_pass <= '0;
                            state <= S_ATTN_SCORE;
                        end else begin
                            t_idx <= t_idx + PARALLEL_HIST;
                        end

                    end else begin
                        // -----------------------------------------------------------------
                        // PROCESS CHUNK: MAC_UNROLL channels per cycle
                        // FIXED: NBA-in-for-loop — accumulators used blocking assignments
                        // via automatic variables to avoid stale-read bug. Only the final
                        // accumulated values are written with NBAs.
                        // -----------------------------------------------------------------
                        // Time-shared (h,d) MAC: process HD_UNROLL pairs per cycle
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

                                        for (int i = 0; i < PARALLEL_HIST; i++) begin
                                            if ((t_idx + i) < HISTORY_DEPTH && (!ENABLE_MASK || mask_reg[t_idx + i])) begin
                                                k_acc_next[i] = k_acc_next[i] +
                                                    (history_reg[t_idx + i][ch_idx] *
                                                     (QK_WEIGHT_SHARING ? q_weights[h * HEAD_DIM * CHANNELS + d * CHANNELS + ch_idx]
                                                                        : k_weights[h * HEAD_DIM * CHANNELS + d * CHANNELS + ch_idx]));
                                                v_acc_next[i] = v_acc_next[i] +
                                                    (history_reg[t_idx + i][ch_idx] *
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

                            // All (h,d) done for this channel — advance channel
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
                                    state <= S_ATTN_SCORE;
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

                S_ATTN_SCORE: begin
                    // Scaled dot-product: Q·K^T / temperature / sqrt(d)
                    // ATTN_HEADS_PER_CYCLE heads × SCORE_HIST_UNROLL history frames per cycle
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
                                                // same case pattern as approx_exp,
                                                // avoids 32-bit barrel shifter.
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
                            state <= S_SOFTMAX_EXP;
                        end else begin
                            attn_pass <= attn_pass + 1'b1;
                        end
                    end else begin
                        score_t_idx <= score_t_idx + SCORE_HIST_UNROLL;
                    end
                end

                S_SOFTMAX_EXP: begin
                    // Time-shared exp: EXP_UNROLL (h,t) pairs per cycle.
                    // On first cycle (exp_idx==0), partial sums start from 0.
                    // Accumulate across cycles, then copy to exp_sum_reg when done.
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
                        for (int h = 0; h < NUM_HEADS; h++)
                            exp_sum_reg[h] <= acc[h];
                        state <= S_SOFTMAX_NORM;
                    end else begin
                        exp_idx <= exp_idx + EXP_UNROLL;
                    end
                end

                S_SOFTMAX_NORM: begin
                    // -----------------------------------------------------------------
                    // PARTIAL UNROLLING: 4 Parallel Newton-Raphson Dividers
                    // Processes 4 heads per clock cycle. 
                    // If NUM_HEADS = 4, takes 1 cycle. If NUM_HEADS = 8, takes 2 cycles.
                    // -----------------------------------------------------------------
                    for (int i = 0; i < PARALLEL_DIV; i++) begin
                        // Guard condition in case NUM_HEADS is not a perfect multiple of 4
                        if ((head_idx + i) < NUM_HEADS) begin
                            sm_recip_reg[head_idx + i] <= nr_reciprocal(exp_sum_reg[head_idx + i]);
                        end
                    end
                    
                    // Check if we have processed all heads
                    if ((head_idx + PARALLEL_DIV) >= NUM_HEADS) begin
                        head_idx <= '0;
                        hd_idx <= '0;
                        state <= S_AGGREGATE;
                    end else begin
                        // Advance the counter by the number of parallel hardware blocks
                        head_idx <= head_idx + PARALLEL_DIV;
                    end
                end

                S_AGGREGATE: begin
                    // HD_UNROLL time-shared softmax apply + weighted value aggregation
                    // Computes HD_UNROLL (h,d) pairs per cycle:
                    //   agg[h][d] = sum_t( (exp_score[h][t] * recip[h]) * v_proj[t][h][d] )
                    for (int i = 0; i < HD_UNROLL; i++) begin
                        automatic int hd_pair = hd_idx + i;
                        if (hd_pair < NUM_HEADS * HEAD_DIM) begin
                            automatic int h = hd_pair / HEAD_DIM;
                            automatic int d = hd_pair % HEAD_DIM;
                            automatic logic signed [ACC_W-1:0] recip = sm_recip_reg[h];
                            automatic logic signed [ACC_W-1:0] agg = '0;
                            for (int t = 0; t < HISTORY_DEPTH; t++) begin
                                if (!ENABLE_MASK || mask_reg[t]) begin
                                    automatic logic signed [ACC_W-1:0] w;
                                    w = (exp_scores[h][t] * recip) >>> FRAC_W;
                                    agg = agg + (w * v_proj[t][h][d]);
                                end
                            end
                            attended_per_head[h][d] <= DATA_W'(agg >>> FRAC_W);
                        end
                    end

                    if (hd_idx + HD_UNROLL >= NUM_HEADS * HEAD_DIM) begin
                        hd_idx <= '0;
                        for (int c = 0; c < CHANNELS; c++)
                            out_accum[c] <= '0;
                        out_mac_idx <= '0;
                        state <= S_OUT_MAC;
                    end else begin
                        hd_idx <= hd_idx + HD_UNROLL;
                    end
                end

                S_OUT_MAC: begin
                    // Channel-time-shared output projection MAC.
                    // Processes CH_OUT_UNROLL channels × OUT_MAC_UNROLL (h,d) indices per cycle.
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

                    // Two-level advancement: first (h,d) indices, then channel groups
                    if (out_mac_idx + OUT_MAC_UNROLL >= NUM_HEADS * HEAD_DIM) begin
                        out_mac_idx <= '0;
                        if (ch_out_idx + CH_OUT_UNROLL >= CHANNELS) begin
                            ch_out_idx <= '0;
                            for (int c = 0; c < CHANNELS; c++)
                                output_proj[c] <= DATA_W'(out_accum[c] >>> FRAC_W);
                            state <= S_RESIDUAL;
                        end else begin
                            ch_out_idx <= ch_out_idx + CH_OUT_UNROLL;
                        end
                    end else begin
                        out_mac_idx <= out_mac_idx + OUT_MAC_UNROLL;
                    end
                end

                S_RESIDUAL: begin
                    // Residual connection (configurable via ENABLE_RESIDUAL).
                    // When disabled, output_proj passes through directly, saving
                    // CHANNELS adders + saturation logic LUTs.
                    attended_valid_out <= 1'b1;
                    for (int c = 0; c < CHANNELS; c++) begin
                        if (ENABLE_RESIDUAL) begin
                            automatic logic signed [ACC_W-1:0] res =
                                ACC_W'(output_proj[c]) + ACC_W'(query_reg[c]);
                            if (res > (1 <<< (DATA_W - 1)) - 1)
                                attended_data_out[c] <= (1 <<< (DATA_W - 1)) - 1;
                            else if (res < -(1 <<< (DATA_W - 1)))
                                attended_data_out[c] <= -(1 <<< (DATA_W - 1));
                            else
                                attended_data_out[c] <= DATA_W'(res);
                        end else begin
                            attended_data_out[c] <= output_proj[c];
                        end
                    end
                    if (attended_ready_in) begin
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // Backpressure
    // FIXED: query_ready_in should only accept new input in IDLE.
    // Previously (state == S_IDLE) || attended_ready_in would allow new
    // queries while still processing, overwriting query_reg.
    assign query_ready_in = (state == S_IDLE);

    // =========================================================================
    // Weight Loading
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
                    // [0, N) → Q (K aliased to same storage), [N, 2N) → V, [2N, ...) → out
                    if (weight_addr < CHANNELS * NUM_HEADS * HEAD_DIM) begin
                        q_weights[weight_addr] <= weight_data;
                    end else if (weight_addr < 2 * CHANNELS * NUM_HEADS * HEAD_DIM) begin
                        v_weights[weight_addr - CHANNELS * NUM_HEADS * HEAD_DIM] <= weight_data;
                    end else begin
                        out_weights[weight_addr - 2 * CHANNELS * NUM_HEADS * HEAD_DIM] <= weight_data;
                    end
                end else begin
                    // [0, N) → Q, [N, 2N) → K, [2N, 3N) → V, [3N, ...) → out
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
    ) else $error("temporal_attention_enhanced: Invalid head configuration");

    TEMPORAL_ATTN_HISTORY: assert property (
        @(posedge clk) disable iff (!rst_n)
        cfg_history_depth > 0 && cfg_history_depth <= HISTORY_DEPTH
    ) else $error("temporal_attention_enhanced: Invalid history depth");
    // synthesis translate_on

endmodule