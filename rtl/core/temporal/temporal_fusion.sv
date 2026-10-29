// =============================================================================
// temporal_fusion.sv
//
// Attention-based temporal fusion for multi-frame feature aggregation.
// Computes learned attention weights across T frames and produces a fused
// output that emphasizes temporally relevant features.
//
// Architecture:
//   1. Query: Current frame features (frame_t)
//   2. Key: All history frame features (frame_{t-1}, ..., frame_{t-T})
//   3. Attention: Softmax(Q·K^T / sqrt(d))
//   4. Value: Weighted sum of history frames
//
// Modes:
//   - ATTENTION: Learned attention weights (configurable)
//   - AVERAGE: Simple averaging across frames
//   - MAX: Max pooling across frames
//   - LEARNED: Fixed learnable weights from CSR
//
// Latency: 8 cycles (attention computation + softmax)
// Throughput: 1 frame/cycle
//
// BUG-FIX history (BUG-TF-*):
//   BUG-TF-001: Weighted sum used 2*DATA_W product shifted by FRAC_WIDTH (16),
//     which always produced zero for DATA_WIDTH=8. Fixed: ACC_W-wide product,
//     shift by DATA_W-1 (Q0.7 weight format).
//   BUG-TF-002: Final output shifted fused_acc by FRAC_WIDTH (16), also always
//     zero for INT8. Fixed: no shift (fused_acc already integer after BUG-TF-001 fix).
//   BUG-TF-003: function automatic misbehaves in Verilator. Replaced approx_exp
//     and avg_weight_lut with always_comb lookup arrays.
//   BUG-TF-004: Input registration accepted data regardless of backpressure,
//     corrupting in-flight data. Fixed: gate on query_ready_in.
//   BUG-TF-005: cfg_fusion_mode not pipelined — mode change mid-pipeline
//     corrupts processing. Fixed: cfg_mode_d* pipeline registers.
//   BUG-TF-006: cfg_enable declared but never used — no bypass path.
//     Fixed: when cfg_enable=0, output = query_reg (passthrough).
// =============================================================================

import pkg_config::*;

module temporal_fusion #(
    parameter int HISTORY_DEPTH = 8,
    parameter int CHANNELS      = MAX_C,
    parameter int DATA_W        = DATA_WIDTH,
    parameter int ACC_W         = ACC_WIDTH,
    // Time-multiplexed channel parallelism (configurable via config.yaml)
    parameter int TM_CH         = pkg_config::TM_CH,         // Channels per dot-product cycle
    parameter int DOT_PRODUCT_CH_PER_CYCLE = TM_CH,          // Alias for backward compatibility
    parameter int WEIGHTED_SUM_FRAMES_PER_CYCLE = HISTORY_DEPTH  // History frames per weighted-sum cycle
)(
    input  logic clk,
    input  logic rst_n,

    // Configuration
    input  logic [2:0]  cfg_fusion_mode,      // 0=ATTENTION, 1=AVERAGE, 2=MAX, 3=LEARNED
    input  logic [3:0]  cfg_history_depth,    // Number of frames to consider
    input  logic         cfg_enable,          // Enable temporal fusion

    // Current frame (query)
    input  logic                        query_valid_in,
    input  logic signed [DATA_W-1:0]   query_data [CHANNELS-1:0],
    output logic                        query_ready_in,

    // History frames (keys and values)
    input  logic                        history_valid_in,
    input  logic [HISTORY_DEPTH-1:0]   history_valid_mask,
    input  logic signed [DATA_W-1:0]   history_data [HISTORY_DEPTH-1:0][CHANNELS-1:0],

    // Learned attention weights (for LEARNED mode)
    input  logic signed [DATA_W-1:0]   attention_weights [HISTORY_DEPTH-1:0],

    // Output (fused features)
    output logic                        fused_valid_out,
    output logic signed [DATA_W-1:0]   fused_data_out [CHANNELS-1:0],
    input  logic                        fused_ready_in,

    // Debug: attention scores
    output logic signed [ACC_W-1:0]    attention_scores [HISTORY_DEPTH-1:0]
);

    // =========================================================================
    // Fusion Mode Encoding
    // =========================================================================
    localparam bit [2:0] MODE_ATTENTION = 3'b000;
    localparam bit [2:0] MODE_AVERAGE   = 3'b001;
    localparam bit [2:0] MODE_MAX       = 3'b010;
    localparam bit [2:0] MODE_LEARNED   = 3'b011;

    // =========================================================================
    // Stage 1: Input Registration (BUG-TF-004: gate on query_ready_in)
    // =========================================================================
    logic signed [DATA_W-1:0] query_reg [CHANNELS-1:0];
    logic signed [DATA_W-1:0] history_reg [HISTORY_DEPTH-1:0][CHANNELS-1:0];
    logic [HISTORY_DEPTH-1:0] mask_reg;
    logic valid_d1;

    // Pipeline cfg_fusion_mode and cfg_enable (BUG-TF-005)
    logic [2:0] cfg_mode_d1, cfg_mode_d2, cfg_mode_d3, cfg_mode_d4,
                cfg_mode_d5, cfg_mode_d6, cfg_mode_d7;
    logic       cfg_en_d1, cfg_en_d2, cfg_en_d3, cfg_en_d4,
                cfg_en_d5, cfg_en_d6, cfg_en_d7;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d1 <= 1'b0;
            mask_reg <= '0;
            cfg_mode_d1 <= 3'b0; cfg_mode_d2 <= 3'b0; cfg_mode_d3 <= 3'b0;
            cfg_mode_d4 <= 3'b0; cfg_mode_d5 <= 3'b0; cfg_mode_d6 <= 3'b0;
            cfg_mode_d7 <= 3'b0;
            cfg_en_d1 <= 1'b0; cfg_en_d2 <= 1'b0; cfg_en_d3 <= 1'b0;
            cfg_en_d4 <= 1'b0; cfg_en_d5 <= 1'b0; cfg_en_d6 <= 1'b0;
            cfg_en_d7 <= 1'b0;
            for (int c = 0; c < CHANNELS; c++)
                query_reg[c] <= '0;
            for (int t = 0; t < HISTORY_DEPTH; t++)
                for (int c = 0; c < CHANNELS; c++)
                    history_reg[t][c] <= '0;
        end else begin
            // Pipeline cfg alongside data
            cfg_mode_d1 <= cfg_fusion_mode;
            cfg_mode_d2 <= cfg_mode_d1;
            cfg_mode_d3 <= cfg_mode_d2;
            cfg_mode_d4 <= cfg_mode_d3;
            cfg_mode_d5 <= cfg_mode_d4;
            cfg_mode_d6 <= cfg_mode_d5;
            cfg_mode_d7 <= cfg_mode_d6;
            cfg_en_d1 <= cfg_enable;
            cfg_en_d2 <= cfg_en_d1;
            cfg_en_d3 <= cfg_en_d2;
            cfg_en_d4 <= cfg_en_d3;
            cfg_en_d5 <= cfg_en_d4;
            cfg_en_d6 <= cfg_en_d5;
            cfg_en_d7 <= cfg_en_d6;

            if (query_valid_in && history_valid_in && query_ready_in) begin
                valid_d1 <= 1'b1;
                mask_reg <= history_valid_mask;
                for (int c = 0; c < CHANNELS; c++)
                    query_reg[c] <= query_data[c];
                for (int t = 0; t < HISTORY_DEPTH; t++)
                    for (int c = 0; c < CHANNELS; c++)
                        history_reg[t][c] <= history_data[t][c];
            end else begin
                valid_d1 <= 1'b0;
            end
        end
    end

    // =========================================================================
    // Stage 2: Similarity Computation (Query · Key^T)
    // =========================================================================
    // For attention mode: compute similarity between query and each history frame
    // Similarity = sum(query[c] * history[t][c]) for all channels

    logic signed [ACC_W-1:0] similarity [HISTORY_DEPTH-1:0];
    logic valid_d2;

    // Combinational dot product (extracted from always_ff for Verilator compat)
    logic signed [ACC_W-1:0] similarity_comb [HISTORY_DEPTH-1:0];
    always_comb begin
        for (int t = 0; t < HISTORY_DEPTH; t++) begin
            if (mask_reg[t]) begin
                automatic logic signed [ACC_W-1:0] acc = '0;
                for (int c = 0; c < CHANNELS; c++) begin
                    acc = acc + (ACC_W'(query_reg[c]) * ACC_W'(history_reg[t][c]));
                end
                similarity_comb[t] = acc;
            end else begin
                similarity_comb[t] = '0;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d2 <= 1'b0;
            for (int t = 0; t < HISTORY_DEPTH; t++)
                similarity[t] <= '0;
        end else begin
            valid_d2 <= valid_d1;
            for (int t = 0; t < HISTORY_DEPTH; t++)
                similarity[t] <= similarity_comb[t];
        end
    end

    // =========================================================================
    // Stage 3-4: Softmax (Attention Mode) or Direct Weighting
    // =========================================================================
    // For attention mode: softmax(similarity / sqrt(d))
    // For other modes: direct weights

    localparam int SCALE_SHIFT = $clog2(CHANNELS);  // sqrt(d) approximation

    logic signed [ACC_W-1:0] exp_similarity [HISTORY_DEPTH-1:0];
    logic signed [ACC_W-1:0] exp_sum;
    logic signed [ACC_W-1:0] exp_sum_comb;  // Combinational reduction for softmax denominator
    logic signed [DATA_W-1:0] softmax_weights [HISTORY_DEPTH-1:0];
    logic valid_d3, valid_d4;

    // MSB position of exp_sum_comb — used for shift-based softmax normalization (no division).
    // Bounded range: exp_sum_comb is sum of up to HISTORY_DEPTH LUT-valued exponentials,
    // each ≤ (1 << FRAC_W), so MSB is in range [0, FRAC_W+$clog2(HISTORY_DEPTH)].
    // Replace full ACC_W-bit priority encoder with bounded case over ~20 values.
    logic [5:0] norm_shift;
    always_comb begin
        norm_shift = 6'd0;
        case (1'b1)
            exp_sum_comb[17]: norm_shift = 6'd17;
            exp_sum_comb[16]: norm_shift = 6'd16;
            exp_sum_comb[15]: norm_shift = 6'd15;
            exp_sum_comb[14]: norm_shift = 6'd14;
            exp_sum_comb[13]: norm_shift = 6'd13;
            exp_sum_comb[12]: norm_shift = 6'd12;
            exp_sum_comb[11]: norm_shift = 6'd11;
            exp_sum_comb[10]: norm_shift = 6'd10;
            exp_sum_comb[9]:  norm_shift = 6'd9;
            exp_sum_comb[8]:  norm_shift = 6'd8;
            exp_sum_comb[7]:  norm_shift = 6'd7;
            exp_sum_comb[6]:  norm_shift = 6'd6;
            exp_sum_comb[5]:  norm_shift = 6'd5;
            exp_sum_comb[4]:  norm_shift = 6'd4;
            exp_sum_comb[3]:  norm_shift = 6'd3;
            exp_sum_comb[2]:  norm_shift = 6'd2;
            exp_sum_comb[1]:  norm_shift = 6'd1;
            exp_sum_comb[0]:  norm_shift = 6'd0;
            default:          norm_shift = 6'd0;
        endcase
    end

    // Combinational sum of exp_similarity for softmax denominator
    // (Avoids NBA accumulation bug where only last iteration's assignment survives)
    always_comb begin
        exp_sum_comb = '0;
        for (int t = 0; t < HISTORY_DEPTH; t++) begin
            if (mask_reg[t])
                exp_sum_comb = exp_sum_comb + exp_similarity[t];
        end
    end

    // Combinational valid count for MODE_AVERAGE (extracted from always_ff for Verilator compat)
    logic [3:0] valid_count_comb;
    always_comb begin
        valid_count_comb = '0;
        for (int b = 0; b < HISTORY_DEPTH; b++)
            if (mask_reg[b]) valid_count_comb = valid_count_comb + 1'b1;
    end

    // Exponential approximation with LUT for negative inputs.
    // After max-subtraction in softmax, non-max scores are negative.
    // Returning 0 for negatives produced hard-max; the LUT gives proper
    // soft-attention distributions by returning small-but-nonzero exp(-k).
    //
    // BUG-TF-003: Replaced function automatic with always_comb — Verilator
    // mishandles function automatic return values.

    logic signed [ACC_W-1:0] exp_result_comb [HISTORY_DEPTH-1:0];
    always_comb begin
        for (int t = 0; t < HISTORY_DEPTH; t++) begin
            automatic logic signed [ACC_W-1:0] shifted_sim;
            automatic logic [ACC_W-1:0]        abs_idx;
            automatic logic [3:0]              neg_idx;
            shifted_sim = similarity[t] >>> SCALE_SHIFT;
            if (shifted_sim[ACC_W-1]) begin
                // BUG-EROD-TF-001 (FIXED): `neg_idx` was a 4-bit register, so the
                //   right-shift result was *truncated* to 4 bits before the
                //   `if (neg_idx > 4'd15)` clamp ever ran — making the clamp dead
                //   code (Verilator: CMPCONST) and silently wrapping any value
                //   ≥16 down into bins 0..15.  Clamp the *wide* value first, then
                //   narrow.
                abs_idx = ((~shifted_sim) + 1'b1) >>> FRAC_WIDTH;
                neg_idx = (abs_idx > {{(ACC_W-4){1'b0}}, 4'd15}) ? 4'd15 : abs_idx[3:0];
                case (neg_idx)
                    4'd0:  exp_result_comb[t] = ACC_W'(65536);
                    4'd1:  exp_result_comb[t] = ACC_W'(24100);
                    4'd2:  exp_result_comb[t] = ACC_W'(8873);
                    4'd3:  exp_result_comb[t] = ACC_W'(3265);
                    4'd4:  exp_result_comb[t] = ACC_W'(1201);
                    4'd5:  exp_result_comb[t] = ACC_W'(441);
                    4'd6:  exp_result_comb[t] = ACC_W'(162);
                    4'd7:  exp_result_comb[t] = ACC_W'(60);
                    4'd8:  exp_result_comb[t] = ACC_W'(22);
                    4'd9:  exp_result_comb[t] = ACC_W'(8);
                    4'd10: exp_result_comb[t] = ACC_W'(3);
                    4'd11: exp_result_comb[t] = ACC_W'(1);
                    4'd12: exp_result_comb[t] = ACC_W'(1);
                    4'd13: exp_result_comb[t] = ACC_W'(1);
                    4'd14: exp_result_comb[t] = ACC_W'(1);
                    4'd15: exp_result_comb[t] = ACC_W'(1);
                    default: exp_result_comb[t] = ACC_W'(1);
                endcase
            end else begin
                exp_result_comb[t] = shifted_sim + (1 <<< FRAC_WIDTH);
            end
        end
    end

    // Returns 1/cnt scaled to DATA_W signed fixed-point where 1.0 = 2^(DATA_W-1).
    // BUG-TF-003: Replaced function automatic with always_comb.
    logic signed [DATA_W-1:0] avg_weight_comb;
    always_comb begin
        case (valid_count_comb)
            4'd1:    avg_weight_comb = DATA_W'(signed'('d127));
            4'd2:    avg_weight_comb = DATA_W'(signed'('d64));
            4'd3:    avg_weight_comb = DATA_W'(signed'('d43));
            4'd4:    avg_weight_comb = DATA_W'(signed'('d32));
            4'd5:    avg_weight_comb = DATA_W'(signed'('d26));
            4'd6:    avg_weight_comb = DATA_W'(signed'('d21));
            4'd7:    avg_weight_comb = DATA_W'(signed'('d18));
            4'd8:    avg_weight_comb = DATA_W'(signed'('d16));
            default: avg_weight_comb = DATA_W'(signed'('d16));
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d3 <= 1'b0;
            valid_d4 <= 1'b0;
            exp_sum <= '0;
            for (int t = 0; t < HISTORY_DEPTH; t++) begin
                exp_similarity[t] <= '0;
                softmax_weights[t] <= '0;
            end
        end else begin
            valid_d3 <= valid_d2;
            valid_d4 <= valid_d3;

            if (valid_d2 && cfg_mode_d2 == MODE_ATTENTION) begin
                // Compute approximate softmax (BUG-TF-003: use always_comb result)
                for (int t = 0; t < HISTORY_DEPTH; t++) begin
                    if (mask_reg[t]) begin
                        exp_similarity[t] <= exp_result_comb[t];
                    end else begin
                        exp_similarity[t] <= '0;
                    end
                end
            end else if (valid_d2 && cfg_mode_d2 == MODE_LEARNED) begin
                // Use learned weights directly
                for (int t = 0; t < HISTORY_DEPTH; t++) begin
                    if (mask_reg[t])
                        softmax_weights[t] <= attention_weights[t];
                    else
                        softmax_weights[t] <= '0;
                end
            end else if (valid_d2 && cfg_mode_d2 == MODE_AVERAGE) begin
                // Uniform weights for valid frames (BUG-TF-003: use always_comb result)
                for (int t = 0; t < HISTORY_DEPTH; t++) begin
                    if (mask_reg[t])
                        softmax_weights[t] <= avg_weight_comb;
                    else
                        softmax_weights[t] <= '0;
                end
            end
            // MODE_MAX: weights computed in final stage
        end
    end

    // =========================================================================
    // Stage 5: Normalize Weights (for attention mode)
    // =========================================================================
    logic signed [DATA_W-1:0] normalized_weights [HISTORY_DEPTH-1:0];
    logic valid_d5;

    // Combinational normalization (extracted from always_ff for Verilator compat)
    logic signed [DATA_W-1:0] norm_w_comb [HISTORY_DEPTH-1:0];
    always_comb begin
        for (int t = 0; t < HISTORY_DEPTH; t++) begin
            if (exp_sum_comb > 0 && mask_reg[t]) begin
                automatic logic [5:0] rs = (norm_shift >= DATA_W - 1) ?
                     norm_shift - (DATA_W - 2) : 6'd1;
                automatic logic signed [ACC_W-1:0] nw_tmp = exp_similarity[t] >>> rs;
                if (nw_tmp > (1 <<< (DATA_W-1)) - 1)
                    norm_w_comb[t] = (1 <<< (DATA_W-1)) - 1;
                else
                    norm_w_comb[t] = DATA_W'(nw_tmp);
            end else begin
                norm_w_comb[t] = '0;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d5 <= 1'b0;
            for (int t = 0; t < HISTORY_DEPTH; t++)
                normalized_weights[t] <= '0;
        end else begin
            valid_d5 <= valid_d4;
            if (valid_d4 && cfg_mode_d4 == MODE_ATTENTION) begin
                for (int t = 0; t < HISTORY_DEPTH; t++)
                    normalized_weights[t] <= norm_w_comb[t];
            end else begin
                for (int t = 0; t < HISTORY_DEPTH; t++)
                    normalized_weights[t] <= softmax_weights[t];
            end
        end
    end

    // =========================================================================
    // Stage 6-7: Weighted Sum (Value aggregation)
    // =========================================================================
    logic signed [ACC_W-1:0] fused_acc [CHANNELS-1:0];
    logic signed [DATA_W-1:0] max_val [CHANNELS-1:0];
    logic valid_d6, valid_d7;

    // Combinational MODE_MAX (extracted from always_ff for Verilator compat)
    logic signed [DATA_W-1:0] max_val_comb [CHANNELS-1:0];
    always_comb begin
        for (int c = 0; c < CHANNELS; c++) begin
            automatic logic signed [DATA_W-1:0] m = history_reg[0][c];
            for (int t = 1; t < HISTORY_DEPTH; t++) begin
                if (mask_reg[t] && history_reg[t][c] > m)
                    m = history_reg[t][c];
            end
            max_val_comb[c] = m;
        end
    end

    // Combinational weighted sum (extracted from always_ff for Verilator compat)
    // BUG-TF-001: Product was 2*DATA_W wide shifted by FRAC_WIDTH (16) — always zero
    // for DATA_WIDTH=8. Fixed: ACC_W-wide product, shift by DATA_W-1 (Q0.(DATA_W-1)
    // weight format: weights are signed DATA_W values where 1.0 ≈ 2^(DATA_W-1)).
    logic signed [ACC_W-1:0] fused_acc_comb [CHANNELS-1:0];
    always_comb begin
        for (int c = 0; c < CHANNELS; c++) begin
            automatic logic signed [ACC_W-1:0] acc = '0;
            for (int t = 0; t < HISTORY_DEPTH; t++) begin
                if (mask_reg[t]) begin
                    automatic logic signed [ACC_W-1:0] prod_full =
                        (ACC_W)'(normalized_weights[t]) *
                        (ACC_W)'(history_reg[t][c]);
                    automatic logic signed [ACC_W:0] prod_aligned =
                        (ACC_W+1)'(prod_full >>> (DATA_W - 1));
                    if (prod_aligned > (1 <<< (ACC_W-1)) - 1)
                        acc = acc + (1 <<< (ACC_W-1)) - 1;
                    else if (prod_aligned < -(1 <<< (ACC_W-1)))
                        acc = acc + (-(1 <<< (ACC_W-1)));
                    else
                        acc = acc + ACC_W'(prod_aligned);
                end
            end
            fused_acc_comb[c] = acc;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d6 <= 1'b0;
            valid_d7 <= 1'b0;
            for (int c = 0; c < CHANNELS; c++) begin
                fused_acc[c] <= '0;
                max_val[c] <= '0;
            end
        end else begin
            valid_d6 <= valid_d5;
            valid_d7 <= valid_d6;

            if (valid_d5) begin
                if (cfg_mode_d5 == MODE_MAX) begin
                    for (int c = 0; c < CHANNELS; c++)
                        max_val[c] <= max_val_comb[c];
                end else begin
                    for (int c = 0; c < CHANNELS; c++)
                        fused_acc[c] <= fused_acc_comb[c];
                end
            end
        end
    end

    // =========================================================================
    // Stage 8: Output (with residual connection)
    // =========================================================================
    // Add query (current frame) to fused history for residual connection
    logic valid_d8;

    // Combinational fused output (extracted from always_ff for Verilator compat)
    // BUG-TF-002: Was shifting fused_acc by FRAC_WIDTH (16), always zero for INT8.
    //   fused_acc is already in integer format after BUG-TF-001 fix.
    // BUG-TF-006: cfg_enable bypass — when disabled, pass query through unchanged.
    logic signed [DATA_W-1:0] fused_out_comb [CHANNELS-1:0];
    always_comb begin
        for (int c = 0; c < CHANNELS; c++) begin
            if (!cfg_en_d7) begin
                // Bypass: pass query through without fusion
                fused_out_comb[c] = query_reg[c];
            end else if (cfg_mode_d7 == MODE_MAX) begin
                fused_out_comb[c] = max_val[c];
            end else begin
                // fused_acc is in integer format (Q0.0) after BUG-TF-001 fix
                automatic logic signed [ACC_W-1:0] r = fused_acc[c] + ACC_W'(query_reg[c]);
                if (r > (1 <<< (DATA_W - 1)) - 1)
                    fused_out_comb[c] = (1 <<< (DATA_W - 1)) - 1;
                else if (r < -(1 <<< (DATA_W - 1)))
                    fused_out_comb[c] = -(1 <<< (DATA_W - 1));
                else
                    fused_out_comb[c] = DATA_W'(r);
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d8 <= 1'b0;
            fused_valid_out <= 1'b0;
            for (int c = 0; c < CHANNELS; c++)
                fused_data_out[c] <= '0;
        end else begin
            valid_d8 <= valid_d7;

            if (valid_d7) begin
                fused_valid_out <= 1'b1;
                for (int c = 0; c < CHANNELS; c++)
                    fused_data_out[c] <= fused_out_comb[c];
            end else if (fused_valid_out && fused_ready_in) begin
                fused_valid_out <= 1'b0;
            end
        end
    end

    // Backpressure
    assign query_ready_in = fused_ready_in || !fused_valid_out;

    // Output attention scores for debugging
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int t = 0; t < HISTORY_DEPTH; t++) begin
                attention_scores[t] <= '0;
            end
        end else begin
            for (int t = 0; t < HISTORY_DEPTH; t++) begin
                attention_scores[t] <= normalized_weights[t];
            end
        end
    end

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    TEMPORAL_FUSION_MODE: assert property (
        @(posedge clk) disable iff (!rst_n)
        cfg_fusion_mode inside {MODE_ATTENTION, MODE_AVERAGE, MODE_MAX, MODE_LEARNED}
    ) else $error("temporal_fusion: Invalid fusion mode %0b", cfg_fusion_mode);

    TEMPORAL_FUSION_HISTORY: assert property (
        @(posedge clk) disable iff (!rst_n)
        cfg_history_depth > 0 && cfg_history_depth <= HISTORY_DEPTH
    ) else $error("temporal_fusion: Invalid history depth %0d", cfg_history_depth);

    TEMPORAL_FUSION_BACKPRESSURE: assert property (
        @(posedge clk) disable iff (!rst_n)
        !(query_valid_in && history_valid_in && !query_ready_in)
    ) else $error("temporal_fusion: input accepted while not ready — data overwritten");
    // synthesis translate_on

endmodule