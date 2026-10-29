// =============================================================================
// detection_head.sv
//
// Anchor-Based Detection Head for E-RODNet.
// Converts feature maps to object detections (class, bbox, confidence).
//
// FIXED: P0-5 — Major rework:
//   1. Multi-cycle MAC replaces unrolled 1x1 conv (3072→24 parallel multipliers)
//   2. Proper softmax replaces linear shift approximation
//   3. Throughput: 1 detection per (IN_CHANNELS + 3) cycles
//
// Pipeline: MAC (IN_CHANNELS cycles) → ACTIVATE (2 cycles) → OUTPUT
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;

module detection_head #(
    parameter int FEATURE_H     = 128,
    parameter int FEATURE_W     = 128,
    parameter int IN_CHANNELS   = 128,
    parameter int NUM_CLASSES   = 3,
    parameter int NUM_ANCHORS   = 3,
    parameter int DATA_WIDTH    = pkg_module_config::DETECTION_HEAD_DATA_WIDTH,
    parameter int FRAC_WIDTH    = 8,
    parameter int WEIGHT_LOAD_MODE = 0,
    // Channel-parallelism factor for the DH_MAC accumulation. Default 8
    // → IN_CHANNELS/PARA_CH cycles per pixel. PARA_CH=1 reproduces the
    // original 1-channel/cycle accumulator. Must divide IN_CHANNELS.
    // Multiplier count: PARA_CH × (CLS_OUT + BBOX_OUT). At PARA_CH=8,
    // NUM_CLASSES=3, NUM_ANCHORS=3 that's 8×(9+15)=192 mults — kept on
    // LUTs (use_dsp="no") so the systolic array retains the DSP budget.
    parameter int PARA_CH       = 8,
    parameter string CLS_WEIGHT_FILE = "",
    parameter string BBOX_WEIGHT_FILE = ""
)(
    input  logic clk,
    input  logic rst_n,

    // Configuration
    input  logic [15:0] cfg_feature_h,
    input  logic [15:0] cfg_feature_w,
    input  logic [7:0]  cfg_num_classes,
    input  logic [7:0]  cfg_num_anchors,
    input  logic        cfg_enable,

    // Input feature map (streaming)
    input  logic                        feat_valid,
    output logic                        feat_ready,       // Can accept feature input (backpressure)
    input  logic [15:0]                 feat_h_idx,
    input  logic [15:0]                 feat_w_idx,
    input  logic signed [DATA_WIDTH-1:0] feat_data [0:IN_CHANNELS-1],
    input  logic                        feat_last,

    // Classification weights (loaded externally)
    input  logic                        cls_weight_valid,
    input  logic [15:0]                 cls_weight_addr,
    input  logic signed [DATA_WIDTH-1:0] cls_weight_data,

    // Bbox regression weights (loaded externally)
    input  logic                        bbox_weight_valid,
    input  logic [15:0]                 bbox_weight_addr,
    input  logic signed [DATA_WIDTH-1:0] bbox_weight_data,

    // Output detections
    output logic                        det_valid,
    input  logic                        det_ready,        // Consumer ready for detection output (backpressure)
    output logic [15:0]                 det_h_idx,
    output logic [15:0]                 det_w_idx,
    output logic [$clog2(NUM_ANCHORS)-1:0] det_anchor_idx,
    output logic signed [DATA_WIDTH-1:0] det_confidence,
    output logic signed [DATA_WIDTH-1:0] det_dx,
    output logic signed [DATA_WIDTH-1:0] det_dy,
    output logic signed [DATA_WIDTH-1:0] det_dw,
    output logic signed [DATA_WIDTH-1:0] det_dh,
    output logic [NUM_CLASSES-1:0][DATA_WIDTH-1:0] det_class_scores,

    // Detection count
    output logic [15:0]                 det_count,
    output logic                        det_last
);

    // =========================================================================
    // Local Parameters
    // =========================================================================
    localparam int CLS_OUT = NUM_CLASSES * NUM_ANCHORS;
    localparam int BBOX_OUT = 5 * NUM_ANCHORS;
    localparam int ACCUM_W = 2*DATA_WIDTH + $clog2(IN_CHANNELS);
    localparam int LOG2_IN_CHANNELS = $clog2(IN_CHANNELS);
    localparam logic signed [ACCUM_W:0] ROUND_BIAS = (1 <<< (FRAC_WIDTH-1));

    // Softmax computation width: abs_x bounded by (7<<FRAC_WIDTH), exp_sum by
    // ~3*(7<<FRAC_WIDTH).  Only FRAC_WIDTH+5 = 13 bits carry meaningful data;
    // truncating multiplier operands from ACCUM_W+1=24 to this width cuts
    // each 24×24 multiplier to ~13×13, saving ~80% of LUTs per multiply.
    localparam int SOFTMAX_W = FRAC_WIDTH + 5;

    // Feature input backpressure: can accept new data when enabled and idle
    assign feat_ready = cfg_enable && (dh_state == DH_IDLE);

    // Runtime-configurable class/anchor counts (clamped to parameter maximums)
    logic [7:0] num_classes_rt;
    logic [7:0] num_anchors_rt;
    always_comb begin
        num_classes_rt  = (cfg_num_classes  > NUM_CLASSES)  ? NUM_CLASSES  : cfg_num_classes;
        num_anchors_rt  = (cfg_num_anchors  > NUM_ANCHORS)  ? NUM_ANCHORS  : cfg_num_anchors;
    end

    // =========================================================================
    // Weight Storage — distributed RAM gives PARA_CH parallel read ports for
    // the unrolled DH_MAC. Block RAM has only 2 ports per primitive, which
    // would force serialised reads or BRAM banking. For typical sizes
    // (IN_CHANNELS≤128, CLS_OUT/BBOX_OUT ≤ ~24) the LUT-RAM footprint is
    // well under 2 KLUTs total, comfortable on Artix-7-100T (63 K LUTs).
    // =========================================================================
    (* ram_style = "distributed" *)
    logic signed [DATA_WIDTH-1:0] cls_weights [0:IN_CHANNELS-1][0:CLS_OUT-1];
    (* ram_style = "distributed" *)
    logic signed [DATA_WIDTH-1:0] bbox_weights [0:IN_CHANNELS-1][0:BBOX_OUT-1];

    generate
    if (WEIGHT_LOAD_MODE == 1) begin : gen_mem_weights
        initial begin
            if (CLS_WEIGHT_FILE != "")
                $readmemh(CLS_WEIGHT_FILE, cls_weights);
            if (BBOX_WEIGHT_FILE != "")
                $readmemh(BBOX_WEIGHT_FILE, bbox_weights);
        end
    end else begin : gen_runtime_weights
        always_ff @(posedge clk) begin
            if (cls_weight_valid) begin
                for (int c = 0; c < IN_CHANNELS; c++) begin
                    if (cls_weight_addr[LOG2_IN_CHANNELS-1:0] == c[LOG2_IN_CHANNELS-1:0])
                        cls_weights[c][cls_weight_addr >>> LOG2_IN_CHANNELS] <= cls_weight_data;
                end
            end
            if (bbox_weight_valid) begin
                for (int c = 0; c < IN_CHANNELS; c++) begin
                    if (bbox_weight_addr[LOG2_IN_CHANNELS-1:0] == c[LOG2_IN_CHANNELS-1:0])
                        bbox_weights[c][bbox_weight_addr >>> LOG2_IN_CHANNELS] <= bbox_weight_data;
                end
            end
        end
    end
    endgenerate

    // =========================================================================
    // Multi-Cycle MAC FSM
    // =========================================================================
    // FIXED: P0-5 — Replaced fully unrolled 128-channel inner loop with
    // multi-cycle accumulation. 24 parallel multipliers (9 cls + 15 bbox)
    // instead of 3,072. Each spatial location takes IN_CHANNELS cycles.

    typedef enum logic [2:0] {
        DH_IDLE,        // Waiting for feat_valid
        DH_MAC,         // Accumulating over IN_CHANNELS cycles
        DH_SOFTMAX_1,   // Compute exp and find sum
        DH_SOFTMAX_2,   // Normalize by sum
        DH_OUTPUT       // Emit detections over NUM_ANCHORS cycles
    } dh_state_t;

    dh_state_t dh_state;

    logic [$clog2(IN_CHANNELS):0] mac_idx;       // Channel counter (0 to IN_CHANNELS-1)
    logic signed [ACCUM_W:0] cls_accum   [0:CLS_OUT-1];
    logic signed [ACCUM_W:0] bbox_accum  [0:BBOX_OUT-1];
    logic signed [ACCUM_W:0] cls_result  [0:CLS_OUT-1];
    logic signed [ACCUM_W:0] bbox_result [0:BBOX_OUT-1];

    // Latched feature data (stable during MAC accumulation)
    logic signed [DATA_WIDTH-1:0] feat_reg [0:IN_CHANNELS-1];
    logic        feat_valid_reg;
    logic [15:0] feat_h_reg, feat_w_reg;

    // Softmax intermediate
    logic signed [ACCUM_W:0] exp_scores [0:NUM_ANCHORS-1][0:NUM_CLASSES-1];
    logic signed [ACCUM_W:0] exp_sum    [0:NUM_ANCHORS-1];
    logic signed [DATA_WIDTH-1:0] max_score [0:NUM_ANCHORS-1];

    // Output stage
    logic signed [DATA_WIDTH-1:0] confidence_out [0:NUM_ANCHORS-1];
    logic signed [DATA_WIDTH-1:0] class_scores_out [0:NUM_ANCHORS-1][0:NUM_CLASSES-1];
    logic signed [DATA_WIDTH-1:0] bbox_out [0:NUM_ANCHORS-1][0:3];
    logic [$clog2(NUM_ANCHORS)-1:0] anchor_counter;
    logic [15:0] detection_counter;
    logic        emit_active;
    logic        emit_is_last_pixel;
    logic [15:0] emit_h_idx;
    logic [15:0] emit_w_idx;
    logic signed [DATA_WIDTH-1:0] emit_confidence [0:NUM_ANCHORS-1];
    logic signed [DATA_WIDTH-1:0] emit_bbox_dx    [0:NUM_ANCHORS-1];
    logic signed [DATA_WIDTH-1:0] emit_bbox_dy    [0:NUM_ANCHORS-1];
    logic signed [DATA_WIDTH-1:0] emit_bbox_dw    [0:NUM_ANCHORS-1];
    logic signed [DATA_WIDTH-1:0] emit_bbox_dh    [0:NUM_ANCHORS-1];
    logic [NUM_CLASSES-1:0][DATA_WIDTH-1:0] emit_class_scores [0:NUM_ANCHORS-1];

    // Sigmoid approximation
    function automatic logic signed [DATA_WIDTH-1:0] approx_sigmoid(
        input logic signed [ACCUM_W:0] x
    );
        automatic logic signed [DATA_WIDTH-1:0] result;
        if (x > (1 << FRAC_WIDTH))
            result = DATA_WIDTH'((1 << FRAC_WIDTH) - 1);
        else if (x < -(1 << FRAC_WIDTH))
            result = '0;
        else
            result = DATA_WIDTH'(((1 << (FRAC_WIDTH-1)) + (x >>> 2)));
        return result;
    endfunction

    // Piecewise exponential: exp(x) for fixed-point Q.FRAC_WIDTH
    // For x >= 0: exp(x) ≈ 1 + x + x²/2  (truncated Taylor)
    // For x < 0: exp(x) ≈ 1 / (1 + |x| + |x|²/2)  (inverse Taylor)
    // Clamps output to [0, 2^FRAC_WIDTH) to prevent overflow.
    // Squaring uses truncated operands: abs_x is bounded by (7<<FRAC_WIDTH),
    // so only SOFTMAX_W+1 bits are meaningful — reduces the multiplier from
    // (ACCUM_W+1)×(ACCUM_W+1) to SOFTMAX_W×SOFTMAX_W.
    function automatic logic signed [ACCUM_W:0] approx_exp(
        input logic signed [ACCUM_W:0] x
    );
        automatic logic signed [ACCUM_W:0] result;
        automatic logic signed [ACCUM_W:0] abs_x;
        automatic logic signed [ACCUM_W:0] x_sq_half;

        abs_x = x < 0 ? -x : x;
        // Narrow-width squaring: abs_x is bounded by (7<<FRAC_WIDTH),
        // so the upper ACCUM_W-SOFTMAX_W bits are always zero.
        // Truncating before squaring eliminates the wide multiplier.
        x_sq_half = (ACCUM_W+1'(abs_x[SOFTMAX_W:0]) * abs_x[SOFTMAX_W:0]) >>> (FRAC_WIDTH + 1);

        if (x >= 0) begin
            result = (1 << FRAC_WIDTH) + x + x_sq_half;
            if (result > (7 << FRAC_WIDTH))   // Clamp to 7.0 to prevent overflow
                result = (7 << FRAC_WIDTH);
        end else begin
            if (abs_x >= (4 << FRAC_WIDTH))  // exp(-4) ≈ 0.018
                result = 0;
            else
                result = (1 << FRAC_WIDTH) / (1 + (abs_x >>> FRAC_WIDTH) + (x_sq_half >>> FRAC_WIDTH));
        end
        return result;
    endfunction

    // Newton-Raphson reciprocal: 1/x in 2 iterations
    // Initial guess: x₀ = (2^(FRAC_WIDTH+1) - x) for x in [1, 2^(FRAC_WIDTH)]
    // Then: x_{n+1} = x_n * (2 - scale * x_n) / scale
    function automatic logic signed [ACCUM_W:0] approx_reciprocal(
        input logic signed [ACCUM_W:0] x
    );
        automatic logic signed [ACCUM_W:0] result;
        automatic logic signed [ACCUM_W:0] one;
        automatic logic signed [ACCUM_W:0] x0;
        automatic logic signed [ACCUM_W:0] x1;

        one = (1 << FRAC_WIDTH);
        if (x <= 0) begin
            result = one;  // Guard: avoid division by zero
        end else begin
            // Initial guess: x0 ≈ 2/scale - x/scale² (linear approximation near x=1)
            x0 = one + ((one - x) >>> 1);  // x0 ≈ 1.5 - 0.5*x in Q8.8
            if (x0 <= 0) x0 = 1;

            // Newton-Raphson iteration 1: x1 = x0 * (2 - x*x0/scale)
            x1 = (x0 * (2 * one - (x * x0) >>> FRAC_WIDTH)) >>> FRAC_WIDTH;
            if (x1 <= 0) x1 = x0;

            // Newton-Raphson iteration 2: result = x1 * (2 - x*x1/scale)
            result = (x1 * (2 * one - (x * x1) >>> FRAC_WIDTH)) >>> FRAC_WIDTH;
            if (result <= 0) result = one;
        end
        return result;
    endfunction

    // =========================================================================
    // Main FSM
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dh_state <= DH_IDLE;
            mac_idx <= '0;
            feat_valid_reg <= 1'b0;
            feat_h_reg <= '0;
            feat_w_reg <= '0;
            det_valid <= 1'b0;
            det_last <= 1'b0;
            det_count <= '0;
            detection_counter <= '0;
            anchor_counter <= '0;
            emit_active <= 1'b0;
            for (int o = 0; o < CLS_OUT; o++) begin
                cls_accum[o] <= '0;
                cls_result[o] <= '0;
            end
            for (int o = 0; o < BBOX_OUT; o++) begin
                bbox_accum[o] <= '0;
                bbox_result[o] <= '0;
            end
            for (int a = 0; a < NUM_ANCHORS; a++) begin
                confidence_out[a] <= '0;
                for (int c = 0; c < NUM_CLASSES; c++)
                    class_scores_out[a][c] <= '0;
                for (int b = 0; b < 4; b++)
                    bbox_out[a][b] <= '0;
            end
        end else begin
            // Clear det_valid only when acknowledged by downstream
            if (det_valid && det_ready)
                det_valid <= 1'b0;
            det_last <= 1'b0;

            case (dh_state)
                DH_IDLE: begin
                    if (feat_valid && feat_ready) begin
                        // Latch feature data for multi-cycle accumulation
                        feat_valid_reg <= feat_valid;
                        feat_h_reg <= feat_h_idx;
                        feat_w_reg <= feat_w_idx;
                        for (int c = 0; c < IN_CHANNELS; c++)
                            feat_reg[c] <= feat_data[c];
                        mac_idx <= '0;
                        // Clear accumulators
                        for (int o = 0; o < CLS_OUT; o++)
                            cls_accum[o] <= '0;
                        for (int o = 0; o < BBOX_OUT; o++)
                            bbox_accum[o] <= '0;
                        dh_state <= DH_MAC;
                    end
                end

                DH_MAC: begin
                    // Channel-parallel MAC: PARA_CH lanes accumulate in one
                    // cycle. With PARA_CH=8 and IN_CHANNELS=128 this completes
                    // in 16 cycles instead of 128 — 8× speedup at the cost
                    // of PARA_CH×(CLS_OUT+BBOX_OUT) extra LUT multipliers.
                    // (* use_dsp = "no" *) keeps these out of the DSP budget.
                    for (int o = 0; o < CLS_OUT; o++) begin
                        automatic logic signed [ACCUM_W:0] acc;
                        acc = cls_accum[o];
                        for (int p = 0; p < PARA_CH; p++) begin
                            automatic int idx;
                            idx = int'(mac_idx) + p;
                            if (idx < int'(IN_CHANNELS))
                                /* verilator lint_off WIDTH */
                                acc = acc + feat_reg[idx] * cls_weights[idx][o];
                                /* verilator lint_on WIDTH */
                        end
                        cls_accum[o] <= acc;
                        if (int'(mac_idx) + PARA_CH >= int'(IN_CHANNELS))
                            cls_result[o] <= (acc + ROUND_BIAS) >>> FRAC_WIDTH;
                    end
                    for (int o = 0; o < BBOX_OUT; o++) begin
                        automatic logic signed [ACCUM_W:0] acc;
                        acc = bbox_accum[o];
                        for (int p = 0; p < PARA_CH; p++) begin
                            automatic int idx;
                            idx = int'(mac_idx) + p;
                            if (idx < int'(IN_CHANNELS))
                                /* verilator lint_off WIDTH */
                                acc = acc + feat_reg[idx] * bbox_weights[idx][o];
                                /* verilator lint_on WIDTH */
                        end
                        bbox_accum[o] <= acc;
                        if (int'(mac_idx) + PARA_CH >= int'(IN_CHANNELS))
                            bbox_result[o] <= (acc + ROUND_BIAS) >>> FRAC_WIDTH;
                    end

                    if (int'(mac_idx) + PARA_CH >= int'(IN_CHANNELS)) begin
                        dh_state <= DH_SOFTMAX_1;
                    end else begin
                        mac_idx <= mac_idx + PARA_CH[$bits(mac_idx)-1:0];
                    end
                end

                DH_SOFTMAX_1: begin
                    // Compute exp(score - max) for each class per anchor
                    // and accumulate per-anchor exp_sum
                    for (int a = 0; a < num_anchors_rt; a++) begin
                        // Compute max_score locally from stable cls_result
                        automatic logic signed [ACCUM_W:0] m;
                        automatic logic signed [ACCUM_W:0] sum;
                        m = cls_result[a * NUM_CLASSES];
                        for (int c = 1; c < num_classes_rt; c++) begin
                            if (cls_result[a * NUM_CLASSES + c] > m)
                                m = cls_result[a * NUM_CLASSES + c];
                        end
                        sum = '0;
                        for (int c = 0; c < num_classes_rt; c++) begin
                            exp_scores[a][c] <= approx_exp(cls_result[a * NUM_CLASSES + c] - m);
                            sum = sum + approx_exp(cls_result[a * NUM_CLASSES + c] - m);
                        end
                        exp_sum[a] <= sum;
                    end

                    // Compute sigmoid for confidence and bbox outputs
                    for (int a = 0; a < num_anchors_rt; a++) begin
                        confidence_out[a] <= approx_sigmoid(bbox_result[a * 5]);
                        for (int b = 0; b < 4; b++)
                            bbox_out[a][b] <= DATA_WIDTH'(bbox_result[a * 5 + 1 + b]);
                    end

                    dh_state <= DH_SOFTMAX_2;
                end

                DH_SOFTMAX_2: begin
                    // Normalize: class_score = exp_score / exp_sum per anchor
                    // Using Newton-Raphson reciprocal for division
                    for (int a = 0; a < num_anchors_rt; a++) begin
                        automatic logic signed [ACCUM_W:0] recip;
                        recip = approx_reciprocal(exp_sum[a]);
                        for (int c = 0; c < NUM_CLASSES; c++) begin
                            class_scores_out[a][c] <= DATA_WIDTH'((exp_scores[a][c] * recip) >>> FRAC_WIDTH);
                        end
                    end

                    dh_state <= DH_OUTPUT;
                    anchor_counter <= '0;
                    emit_active <= 1'b1;
                    emit_h_idx <= feat_h_reg;
                    emit_w_idx <= feat_w_reg;
                    emit_is_last_pixel <= (feat_h_reg == cfg_feature_h - 1) &&
                                          (feat_w_reg == cfg_feature_w - 1);
                    for (int a = 0; a < NUM_ANCHORS; a++) begin
                        emit_confidence[a] <= confidence_out[a];
                        emit_bbox_dx[a] <= bbox_out[a][0];
                        emit_bbox_dy[a] <= bbox_out[a][1];
                        emit_bbox_dw[a] <= bbox_out[a][2];
                        emit_bbox_dh[a] <= bbox_out[a][3];
                    end
                    // Compute emit_class_scores directly from exp/recip to avoid
                    // 1-cycle pipeline delay (class_scores_out is updated this cycle)
                    for (int a = 0; a < NUM_ANCHORS; a++) begin
                        automatic logic signed [ACCUM_W:0] recip;
                        recip = approx_reciprocal(exp_sum[a]);
                        for (int c = 0; c < num_classes_rt; c++)
                            emit_class_scores[a][c] <= DATA_WIDTH'((exp_scores[a][c] * recip) >>> FRAC_WIDTH);
                    end
                end

                DH_OUTPUT: begin
                    // Emit anchors sequentially with backpressure
                    if (emit_active) begin
                        // Assert valid (stays asserted until det_ready)
                        det_valid <= 1'b1;
                        det_h_idx <= emit_h_idx;
                        det_w_idx <= emit_w_idx;
                        det_anchor_idx <= anchor_counter;
                        det_confidence <= emit_confidence[anchor_counter];
                        det_dx <= emit_bbox_dx[anchor_counter];
                        det_dy <= emit_bbox_dy[anchor_counter];
                        det_dw <= emit_bbox_dw[anchor_counter];
                        det_dh <= emit_bbox_dh[anchor_counter];
                        for (int c = 0; c < num_classes_rt; c++)
                            det_class_scores[c] <= emit_class_scores[anchor_counter][c];

                        // Advance only when downstream acknowledges
                        if (det_ready) begin
                            if (anchor_counter == num_anchors_rt - 1) begin
                                emit_active <= 1'b0;
                                anchor_counter <= '0;
                                detection_counter <= detection_counter + 1;
                                det_last <= emit_is_last_pixel;
                                dh_state <= DH_IDLE;
                            end else begin
                                anchor_counter <= anchor_counter + 1;
                            end
                        end
                    end else begin
                        dh_state <= DH_IDLE;
                    end
                end

                default: dh_state <= DH_IDLE;
            endcase

            det_count <= detection_counter;
        end
    end

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off


    // synthesis translate_on

endmodule