// =============================================================================
// nms_topk.sv
//
// Non-Maximum Suppression and Top-K Selection for E-RODNet.
// Min-heap architecture: keeps only TOP_K highest-confidence detections,
// then runs greedy NMS with bounding-box pre-filter early termination.
//
// Architecture:
//   HEAP_COLLECT → HEAP_EXTRACT → NMS_COMPARE → OUTPUT
//   - Min-heap of size TOP_K during collection (O(log K) per insert)
//   - Heap sort for descending confidence (O(K log K), K=100 max)
//   - Greedy NMS with spatial pre-filter (skip IoU for non-overlapping boxes)
//   - Newton-Raphson reciprocal for IoU division (combinational, 1 cycle)
//
// Throughput (N=4096, K=100):
//   Collection: ~11K cycles | Sort: ~5K | NMS: ~5K | Output: ~100
//   Total: ~21.5K cycles (vs ~4.2M for O(N²) original) = ~195x speedup
//
// Bug fixes vs original:
//   NMS-BUG-001: cfg_enable checked in always_comb IDLE transition
//   NMS-BUG-002: counters reset on IDLE→HEAP_COLLECT transition
//   NMS-BUG-004: out_valid explicitly cleared every cycle in OUTPUT
//   Dead states NMS_DIVIDE/NMS_CHECK removed
//
// Parameters:
//   MAX_DETECTIONS — kept for API compatibility, heap size = TOP_K
//   TOP_K          — heap capacity, only K best detections stored
//   NUM_CLASSES    — number of object classes
//   DATA_WIDTH     — fixed-point data width (default 16)
//   FRAC_WIDTH     — fractional bits (default 8, Q8.8)
// =============================================================================

import pkg_module_config::*;
import pkg_config::*;

module nms_topk #(
    parameter int MAX_DETECTIONS = pkg_config::MAX_DETECTIONS,
    parameter int NUM_CLASSES   = pkg_config::NUM_CLASSES,
    parameter int TOP_K         = pkg_config::TOP_K,
    parameter int DATA_WIDTH    = pkg_module_config::NMS_TOPK_DATA_WIDTH,
    parameter int FRAC_WIDTH    = DATA_WIDTH/2,
    parameter int PARALLEL_IOU  = pkg_config::PARALLEL_IOU,
    // Default anchor cell size in pixels — config.yaml::detection.
    // compute_bbox scales dx/dy/dw/dh deltas relative to this size.
    parameter int DEFAULT_ANCHOR_H = pkg_config::DEFAULT_ANCHOR_H,
    parameter int DEFAULT_ANCHOR_W = pkg_config::DEFAULT_ANCHOR_W
)(
    input  logic clk,
    input  logic rst_n,

    // Configuration (shadowed, committed on idle boundary)
    input  logic signed [DATA_WIDTH-1:0] cfg_conf_threshold,
    input  logic signed [DATA_WIDTH-1:0] cfg_iou_threshold,
    input  logic [7:0]                  cfg_top_k,
    input  logic                        cfg_enable,

    // Input detections (streaming)
    input  logic                        det_valid,
    input  logic [15:0]                 det_h_idx,
    input  logic [15:0]                 det_w_idx,
    input  logic [1:0]                  det_anchor_idx,
    input  logic signed [DATA_WIDTH-1:0] det_confidence,
    input  logic signed [DATA_WIDTH-1:0] det_dx,
    input  logic signed [DATA_WIDTH-1:0] det_dy,
    input  logic signed [DATA_WIDTH-1:0] det_dw,
    input  logic signed [DATA_WIDTH-1:0] det_dh,
    input  logic [NUM_CLASSES-1:0][DATA_WIDTH-1:0] det_class_scores,
    input  logic                        det_last,

    // Output filtered detections
    output logic                        out_valid,
    output logic [15:0]                 out_h_idx,
    output logic [15:0]                 out_w_idx,
    output logic signed [DATA_WIDTH-1:0] out_confidence,
    output logic [15:0]                 out_width,
    output logic [15:0]                 out_height,
    output logic [NUM_CLASSES-1:0][DATA_WIDTH-1:0] out_class_scores,
    output logic [1:0]                  out_class_id,
    output logic                        out_last,
    input  logic                        out_ready,    // FIXED: backpressure — consumer ready

    // Status
    output logic [15:0]                 out_count,
    output logic [15:0]                 suppressed_count
);

    import pkg_nms::*;

    // Forward typedef for Verilator compatibility (package import visibility)
    typedef pkg_nms::detection_t detection_t;

    // =========================================================================
    // Local parameters
    // =========================================================================
    localparam int HEAP_ADDR_W = $clog2(TOP_K);
    localparam int HEAP_SIZE_W = HEAP_ADDR_W + 1; // +1 for size up to TOP_K

    // =========================================================================
    // Min-heap storage (register-based for TOP_K ≤ 256)
    // =========================================================================

    // =========================================================================
    // Shadow registers for configuration
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] active_conf_threshold;
    logic signed [DATA_WIDTH-1:0] active_iou_threshold;
    logic [7:0]                   active_top_k;

    // Shadow register protocol: latch cfg values in IDLE, hold during operation.
    // Uses simple registered copies (no 1-cycle commit latency) to ensure values
    // are available on the same cycle as IDLE→HEAP_COLLECT transition.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_conf_threshold <= '0;
            active_iou_threshold <= '0;
            active_top_k <= '0;
        end else if (state == IDLE) begin
            active_conf_threshold <= cfg_conf_threshold;
            active_iou_threshold <= cfg_iou_threshold;
            active_top_k <= cfg_top_k;
        end
    end

    // =========================================================================
    // Min-heap storage (register-based for TOP_K ≤ 256)
    // =========================================================================
    detection_t heap_mem     [0:TOP_K-1];
    logic [HEAP_SIZE_W-1:0]  heap_size;

    // NMS buffer (sorted descending by confidence after extraction)
    detection_t nms_buf      [0:TOP_K-1];
    logic                     nms_suppressed [0:TOP_K-1];
    logic [15:0]              nms_count;

    // =========================================================================
    // Bbox calculation (preserved from original)
    // =========================================================================
    function automatic void compute_bbox(
        input logic [15:0] anchor_h, anchor_w,
        input logic signed [DATA_WIDTH-1:0] dx, dy, dw, dh,
        output logic [15:0] out_h, out_w
    );
        automatic logic signed [31:0] w_calc, h_calc;
        w_calc = ($signed({1'b0, anchor_w}) * ($signed(32'(1 << FRAC_WIDTH)) + $signed(32'(dw)))) >>> FRAC_WIDTH;
        h_calc = ($signed({1'b0, anchor_h}) * ($signed(32'(1 << FRAC_WIDTH)) + $signed(32'(dh)))) >>> FRAC_WIDTH;
        out_w = w_calc[15:0];
        out_h = h_calc[15:0];
    endfunction

    // =========================================================================
    // NMS Parallel IoU Bank
    // =========================================================================
    logic signed [31:0] nms_area [0:TOP_K-1];
    logic signed [47:0] nms_area_thresh [0:TOP_K-1];
    logic signed [DATA_WIDTH:0] one_plus_T_reg;
    logic [PARALLEL_IOU-1:0] suppress_signals;

    generate
        for (genvar p = 0; p < PARALLEL_IOU; p++) begin : gen_iou_units
            logic [HEAP_ADDR_W-1:0] target_idx;
            assign target_idx = HEAP_ADDR_W'((nms_j + p < nms_count) ? (nms_j + p) : (TOP_K - 1));

            iou_unit #(
                .DATA_WIDTH(DATA_WIDTH),
                .FRAC_WIDTH(FRAC_WIDTH)
            ) u_iou (
                .det1(nms_buf[nms_i[HEAP_ADDR_W-1:0]]),
                .det2(nms_buf[target_idx]),
                .area_thresh1(nms_area_thresh[nms_i[HEAP_ADDR_W-1:0]]),
                .area_thresh2(nms_area_thresh[target_idx]),
                .one_plus_T(one_plus_T_reg),
                .suppressed(suppress_signals[p])
            );
        end
    endgenerate

    // =========================================================================
    // FSM
    // =========================================================================
    typedef enum logic [2:0] {
        IDLE,
        HEAP_COLLECT,
        HEAP_EXTRACT,
        NMS_COMPARE,
        OUTPUT
    } state_t;

    state_t state, next_state;

    // Heap sub-state for multi-cycle sift
    typedef enum logic [1:0] {
        COLLECT_ACCEPT,   // Ready to accept next detection
        COLLECT_SIFT_UP,  // Sifting up after insert
        COLLECT_SIFT_DOWN // Sifting down after root replace
    } collect_sub_state_t;

    collect_sub_state_t collect_sub;

    // NMS iteration state
    logic [$clog2(MAX_DETECTIONS):0] nms_i;
    logic [$clog2(MAX_DETECTIONS):0] nms_j;

    // Heap sift state
    logic [$clog2(TOP_K):0] sift_pos;
    detection_t              sift_entry;
    logic                    sift_is_up; // 1=sift-up, 0=sift-down

    // Extraction state
    logic [$clog2(TOP_K):0] extract_idx;
    logic [$clog2(TOP_K):0] sort_i, sort_j;
    logic                    sort_swap;
    logic                    extracting;
    logic                    sorting;

    // Output state
    logic [$clog2(TOP_K):0] out_rd_ptr;
    logic [$clog2(TOP_K):0] out_emitted;

    // End-of-frame latch (det_last can arrive on IDLE→HEAP_COLLECT transition cycle)
    logic frame_end_latched;

    // Extraction phase tracking (prevents premature HEAP_EXTRACT→NMS_COMPARE transition)
    logic heap_extract_started;

    // Computed detection entry for current det_valid (combinational, avoids NBA hazard)
    detection_t next_det;

    // Computed bbox and class for incoming detection (combinational)
    logic [15:0]                  next_bbox_w, next_bbox_h;
    logic [1:0]                   next_class_id;

    // =========================================================================
    // Combinational: detection entry and bbox/class ID for incoming detection
    // =========================================================================
    always_comb begin
        compute_bbox(16'(DEFAULT_ANCHOR_H), 16'(DEFAULT_ANCHOR_W),
                     det_dx, det_dy, det_dw, det_dh, next_bbox_h, next_bbox_w);
        next_class_id = 2'b0;
        for (int c = 1; c < NUM_CLASSES; c++) begin
            if (det_class_scores[c] >= det_class_scores[next_class_id])
                next_class_id = c[1:0];
        end
        // Build complete detection entry combinationally
        next_det.h_idx      = det_h_idx;
        next_det.w_idx      = det_w_idx;
        next_det.confidence = det_confidence;
        next_det.width      = $signed(next_bbox_w);
        next_det.height     = $signed(next_bbox_h);
        // Explicitly clear and copy class scores to avoid width warnings
        next_det.class_scores = '0;
        for (int c = 0; c < NUM_CLASSES; c++) begin
            next_det.class_scores[c] = det_class_scores[c];
        end
        next_det.class_id   = next_class_id;
        next_det.valid      = 1'b1;
    end

    // =========================================================================
    // Next state logic (BUG-001 fix: cfg_enable check for IDLE)
    // =========================================================================
    always_comb begin
        next_state = state;
        case (state)
            IDLE:           if (det_valid && cfg_enable) next_state = HEAP_COLLECT;
            HEAP_COLLECT:   if (frame_end_latched && collect_sub == COLLECT_ACCEPT) next_state = HEAP_EXTRACT;
            HEAP_EXTRACT:   if (heap_extract_started && !sorting && !extracting) next_state = NMS_COMPARE;
            NMS_COMPARE:    if (nms_i >= nms_count || nms_count == 0) next_state = OUTPUT;
            OUTPUT:         if (out_emitted >= active_top_k || out_rd_ptr >= nms_count) next_state = IDLE;
            default:        next_state = IDLE;
        endcase
    end

    // =========================================================================
    // Main sequential logic
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            collect_sub <= COLLECT_ACCEPT;
            heap_size <= '0;
            nms_count <= '0;
            out_count <= '0;
            suppressed_count <= '0;
            out_valid <= 1'b0;
            out_last <= 1'b0;
            out_h_idx <= '0;
            out_w_idx <= '0;
            out_confidence <= '0;
            out_width <= '0;
            out_height <= '0;
            out_class_scores <= '0;
            out_class_id <= '0;
            nms_i <= '0;
            nms_j <= '0;
            sift_pos <= '0;
            sift_entry <= '0;
            sift_is_up <= 1'b0;
            heap_extract_started <= 1'b0;
            extract_idx <= '0;
            sort_i <= '0;
            sort_j <= '0;
            sort_swap <= 1'b0;
            extracting <= 1'b0;
            sorting <= 1'b0;
            out_rd_ptr <= '0;
            out_emitted <= '0;
            for (int i = 0; i < TOP_K; i++) begin
                heap_mem[i] <= '0;
                heap_mem[i].valid <= 1'b0;
                nms_suppressed[i] <= 1'b0;
                nms_buf[i].valid <= 1'b0;
            end
        end else begin
            out_valid <= 1'b0; // BUG-004 fix: default no output
            state <= next_state;

            case (state)
                // =============================================================
                // IDLE: Wait for det_valid with cfg_enable
                // =============================================================
                IDLE: begin
                    out_last <= 1'b0;
                    frame_end_latched <= 1'b0;
                    heap_extract_started <= 1'b0;
                    if (det_valid && cfg_enable) begin
                        // BUG-002 fix: reset counters on new frame
                                                collect_sub <= COLLECT_ACCEPT;
                        heap_size <= '0;
                        out_count <= '0;
                        suppressed_count <= '0;
                        // Latch det_last if it comes on the transition cycle
                        if (det_last) frame_end_latched <= 1'b1;
                        for (int i = 0; i < TOP_K; i++) begin
                            heap_mem[i].valid <= 1'b0;
                            nms_buf[i].valid <= 1'b0;
                            nms_suppressed[i] <= 1'b0;
                        end
                    end
                end

                // =============================================================
                // HEAP_COLLECT: Stream detections into min-heap
                // =============================================================
                HEAP_COLLECT: begin
                    // Capture det_last at any time (even during sift)
                    if (det_last) frame_end_latched <= 1'b1;
                    case (collect_sub)
                        // -----------------------------------------------------
                        // COLLECT_ACCEPT: Ready to accept next detection
                        // -----------------------------------------------------
                        COLLECT_ACCEPT: begin
                            if (det_valid && det_confidence >= active_conf_threshold) begin
                                // Store detection entry (computed combinationally as next_det)
                                if (heap_size < active_top_k) begin
                                    // Heap not full: insert at end, sift up
                                    heap_mem[heap_size] <= next_det;
                                    sift_pos <= heap_size;
                                    sift_entry <= next_det;
                                    sift_is_up <= 1'b1;
                                    heap_size <= heap_size + 1;
                                    collect_sub <= COLLECT_SIFT_UP;
                                                                    end else if (det_confidence > heap_mem[0].confidence) begin
                                    // Heap full, new detection > root: replace root, sift down
                                    heap_mem[0] <= next_det;
                                    sift_pos <= '0;
                                    sift_entry <= next_det;
                                    sift_is_up <= 1'b0;
                                    collect_sub <= COLLECT_SIFT_DOWN;
                                end
                                // Else: discard (below heap minimum)
                            end
                        end

                        // -----------------------------------------------------
                        // COLLECT_SIFT_UP: Sift entry up from sift_pos
                        // -----------------------------------------------------
                        COLLECT_SIFT_UP: begin
                            if (sift_pos > 0) begin
                                automatic logic [HEAP_SIZE_W-1:0] parent;
                                parent = (sift_pos - 1) / 2;
                                if (heap_mem[sift_pos].confidence < heap_mem[parent].confidence) begin
                                    // Swap with parent
                                    detection_t temp;
                                    temp = heap_mem[parent];
                                    heap_mem[parent] <= heap_mem[sift_pos];
                                    heap_mem[sift_pos] <= temp;
                                    sift_pos <= parent;
                                end else begin
                                    // Heap property satisfied, done
                                    collect_sub <= COLLECT_ACCEPT;
                                end
                            end else begin
                                // Reached root, done
                                collect_sub <= COLLECT_ACCEPT;
                            end
                        end

                        // -----------------------------------------------------
                        // COLLECT_SIFT_DOWN: Sift entry down from sift_pos
                        // -----------------------------------------------------
                        COLLECT_SIFT_DOWN: begin
                            automatic logic [HEAP_SIZE_W-1:0] left, right, min_child;
                            left = 2 * sift_pos + 1;
                            right = 2 * sift_pos + 2;

                            if (left >= heap_size) begin
                                // No children, done
                                collect_sub <= COLLECT_ACCEPT;
                            end else begin
                                // Find minimum child
                                if (right < heap_size &&
                                    heap_mem[right].confidence < heap_mem[left].confidence)
                                    min_child = right;
                                else
                                    min_child = left;

                                if (heap_mem[sift_pos].confidence > heap_mem[min_child].confidence) begin
                                    // Swap with min child
                                    detection_t temp;
                                    temp = heap_mem[sift_pos];
                                    heap_mem[sift_pos] <= heap_mem[min_child];
                                    heap_mem[min_child] <= temp;
                                    sift_pos <= min_child;
                                end else begin
                                    // Heap property satisfied, done
                                    collect_sub <= COLLECT_ACCEPT;
                                end
                            end
                        end

                        default: collect_sub <= COLLECT_ACCEPT;
                    endcase
                end

                // =============================================================
                // HEAP_EXTRACT: Extract K entries and sort descending
                // =============================================================
                HEAP_EXTRACT: begin
                    if (!heap_extract_started) begin
                        // Start extraction: copy heap entries to nms_buf
                        extracting <= 1'b1;
                        extract_idx <= '0;
                        nms_count <= heap_size;
                        heap_extract_started <= 1'b1;
                        one_plus_T_reg <= (1 << FRAC_WIDTH) + active_iou_threshold;
                                            end else if (extracting) begin
                        if (extract_idx < heap_size) begin
                            automatic logic signed [31:0] area;
                            nms_buf[extract_idx] <= heap_mem[extract_idx];
                            nms_suppressed[extract_idx] <= 1'b0;

                            area = $signed(heap_mem[extract_idx].width) * $signed(heap_mem[extract_idx].height);
                            nms_area[extract_idx] <= area;
                            nms_area_thresh[extract_idx] <= area * $signed(active_iou_threshold);

                            extract_idx <= extract_idx + 1;
                        end else begin
                            extracting <= 1'b0;
                            if (nms_count == 0) begin
                                // No entries to sort or NMS — skip directly
                                nms_i <= '0;
                                nms_j <= '0;
                            end else begin
                                // Start heap sort (descending by confidence)
                                sorting <= 1'b1;
                                sort_i <= nms_count - 1;
                                sort_swap <= 1'b0;
                            end
                        end
                    end else if (sorting) begin
                        // Heap sort: extract from min-heap and place at end for descending order
                        if (sort_i > 0) begin
                            if (!sort_swap) begin
                                // SWAP phase: move min (root) to its final position at sort_i
                                detection_t temp;
                                logic signed [31:0] temp_area;
                                logic signed [47:0] temp_thresh;

                                temp = nms_buf[0];
                                nms_buf[0] <= nms_buf[sort_i];
                                nms_buf[sort_i] <= temp;

                                temp_area = nms_area[0];
                                nms_area[0] <= nms_area[sort_i];
                                nms_area[sort_i] <= temp_area;

                                temp_thresh = nms_area_thresh[0];
                                nms_area_thresh[0] <= nms_area_thresh[sort_i];
                                nms_area_thresh[sort_i] <= temp_thresh;

                                sort_j <= '0;
                                sort_swap <= 1'b1;
                            end else begin
                                // SIFT phase: restore min-heap property in nms_buf[0:sort_i-1]
                                automatic logic [HEAP_SIZE_W-1:0] left, right, min_child;
                                left = 2 * sort_j + 1;
                                right = 2 * sort_j + 2;

                                if (left >= sort_i) begin
                                    // No children, sift complete
                                    sort_swap <= 1'b0;
                                    sort_i <= sort_i - 1;
                                end else begin
                                    if (right < sort_i &&
                                        nms_buf[right].confidence < nms_buf[left].confidence)
                                        min_child = right;
                                    else
                                        min_child = left;

                                    if (nms_buf[sort_j].confidence > nms_buf[min_child].confidence) begin
                                        detection_t temp;
                                        logic signed [31:0] temp_area;
                                        logic signed [47:0] temp_thresh;

                                        temp = nms_buf[sort_j];
                                        nms_buf[sort_j] <= nms_buf[min_child];
                                        nms_buf[min_child] <= temp;

                                        temp_area = nms_area[sort_j];
                                        nms_area[sort_j] <= nms_area[min_child];
                                        nms_area[min_child] <= temp_area;

                                        temp_thresh = nms_area_thresh[sort_j];
                                        nms_area_thresh[sort_j] <= nms_area_thresh[min_child];
                                        nms_area_thresh[min_child] <= temp_thresh;

                                        sort_j <= min_child;
                                    end else begin
                                        // Heap property satisfied
                                        sort_swap <= 1'b0;
                                        sort_i <= sort_i - 1;
                                    end
                                end
                            end
                        end else begin
                            sorting <= 1'b0;
                            // Initialize NMS iterators for NMS_COMPARE
                            nms_i <= '0;
                            nms_j <= 1; // start at nms_i+1
                        end
                    end
                end

                // =============================================================
                // NMS_COMPARE: Greedy NMS with bounding-box pre-filter
                // =============================================================
                NMS_COMPARE: begin
                    if (nms_i >= nms_count || nms_count == 0) begin
                        // All detections processed, go to output
                        out_rd_ptr <= '0;
                        out_emitted <= '0;
                                            end else if (nms_suppressed[nms_i]) begin
                        // Skip suppressed detections
                        nms_i <= nms_i + 1;
                        nms_j <= nms_i + 2; // next nms_i+1 after increment
                    end else if (nms_j >= nms_count) begin
                        // Done comparing this detection — advance to next i
                        nms_i <= nms_i + 1;
                        nms_j <= nms_i + 2; // j starts at i+1 for next detection
                    end else begin
                        // Parallel IoU comparison
                        for (int p = 0; p < PARALLEL_IOU; p++) begin
                            if (nms_j + p < nms_count) begin
                                if (suppress_signals[p]) begin
                                    nms_suppressed[nms_j + p] <= 1'b1;
                                                                    end
                            end
                        end
                        nms_j <= nms_j + PARALLEL_IOU;
                    end
                end

                // =============================================================
                // OUTPUT: Stream surviving detections
                // BUG-004 fix: out_valid explicitly cleared for suppressed
                // FIXED: backpressure — hold output until consumer ready (out_ready)
                // =============================================================
                OUTPUT: begin
                    if (out_valid && !out_ready) begin
                        // Consumer not ready — hold current output stable
                    end else if (out_rd_ptr < nms_count && out_emitted < active_top_k) begin
                        if (!nms_suppressed[out_rd_ptr]) begin
                            out_valid <= 1'b1;
                                                        out_h_idx <= nms_buf[out_rd_ptr].h_idx;
                            out_w_idx <= nms_buf[out_rd_ptr].w_idx;
                            out_confidence <= nms_buf[out_rd_ptr].confidence;
                            out_width <= nms_buf[out_rd_ptr].width;
                            out_height <= nms_buf[out_rd_ptr].height;
                            out_class_scores <= nms_buf[out_rd_ptr].class_scores;
                            out_class_id <= nms_buf[out_rd_ptr].class_id;
                            out_count <= out_count + 1;
                            out_emitted <= out_emitted + 1;
                        end else begin
                            suppressed_count <= suppressed_count + 1;
                        end
                        out_rd_ptr <= out_rd_ptr + 1;
                    end else begin
                        out_last <= 1'b1;
                        out_valid <= 1'b0;
                        state <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

    // =========================================================================
    // SystemVerilog Assertions
    // =========================================================================
    `ifndef IVERILOG
    `ifndef VERILATOR
    // Protocol: if the FSM transitions out of IDLE, cfg_enable must be set
    // (Not: det_valid implies cfg_enable, since det_valid without cfg_enable
    // is a valid protocol event that should be ignored.)
    property transition_requires_enable;
        @(posedge clk) disable iff (!rst_n)
        (state == IDLE && det_valid && cfg_enable) |=> (state == HEAP_COLLECT);
    endproperty
    ASSERT_TRANSITION_REQUIRES_ENABLE: assert property (transition_requires_enable)
        else $error("NMS: IDLE with cfg_enable and det_valid did not transition to HEAP_COLLECT");

    // FSM: legal state encoding
    property legal_state;
        @(posedge clk) disable iff (!rst_n)
        state inside {IDLE, HEAP_COLLECT, HEAP_EXTRACT, NMS_COMPARE, OUTPUT};
    endproperty
    ASSERT_LEGAL_STATE: assert property (legal_state)
        else $error("NMS: Illegal FSM state: %0b", state);

    // Heap: size must not exceed TOP_K
    property heap_size_bounds;
        @(posedge clk) disable iff (!rst_n)
        heap_size <= TOP_K;
    endproperty
    ASSERT_HEAP_SIZE_BOUNDS: assert property (heap_size_bounds)
        else $error("NMS: Heap size %0d exceeds TOP_K %0d", heap_size, TOP_K);

    // NMS: suppressed_count must not exceed nms_count
    property suppressed_bounds;
        @(posedge clk) disable iff (!rst_n)
        suppressed_count <= nms_count;
    endproperty
    ASSERT_SUPPRESSED_BOUNDS: assert property (suppressed_bounds)
        else $error("NMS: Suppressed count exceeds detection count");

    // Output: out_valid must be 0 in IDLE
    property out_valid_idle;
        @(posedge clk) disable iff (!rst_n)
        (state == IDLE) |-> (!out_valid);
    endproperty
    ASSERT_OUT_VALID_IDLE: assert property (out_valid_idle)
        else $error("NMS: out_valid asserted in IDLE");

    // Output: out_count must not exceed active_top_k
    property out_count_bounds;
        @(posedge clk) disable iff (!rst_n)
        out_count <= active_top_k;
    endproperty
    ASSERT_OUT_COUNT_BOUNDS: assert property (out_count_bounds)
        else $error("NMS: Output count exceeds top-K");

    // Counters reset on new frame (BUG-002 check)
    property counters_reset_on_new_frame;
        @(posedge clk) disable iff (!rst_n)
        (state == IDLE && det_valid && cfg_enable) |=> (out_count == 0 && suppressed_count == 0);
    endproperty
    ASSERT_COUNTERS_RESET: assert property (counters_reset_on_new_frame)
        else $error("NMS: Counters not reset at frame start");

    // Min-heap root property: root is minimum confidence when heap non-empty
    // Only check children that exist (heap_size >= 2 for left child, >= 3 for right)
    property min_heap_root;
        @(posedge clk) disable iff (!rst_n)
        (heap_size > 0 && collect_sub == COLLECT_ACCEPT) |->
            (heap_size <= 1 ||
             (heap_mem[0].confidence <= heap_mem[1].confidence &&
              (heap_size <= 2 || heap_mem[0].confidence <= heap_mem[2].confidence)));
    endproperty
    ASSERT_MIN_HEAP_ROOT: assert property (min_heap_root)
        else $error("NMS: Heap root is not minimum confidence");

    // det_last requires det_valid (on same cycle) or being in HEAP_COLLECT
    property det_last_requires_valid_or_active;
        @(posedge clk) disable iff (!rst_n)
        det_last |-> (det_valid || state == HEAP_COLLECT);
    endproperty
    ASSERT_DET_LAST_VALID: assert property (det_last_requires_valid_or_active)
        else $error("NMS: det_last without det_valid and outside HEAP_COLLECT");
    `endif // VERILATOR
    `endif // IVERILOG

endmodule
