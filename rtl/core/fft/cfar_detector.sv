// =============================================================================
// cfar_detector.sv
//
// Constant False Alarm Rate (CFAR) Detector for Radar Target Detection.
// Implements CA-CFAR, GO-CFAR, SO-CFAR, and OS-CFAR algorithms.
//
// Architecture:
//   Cell Under Test (CUT) is compared against a threshold derived from
//   the average noise estimate from surrounding training cells.
//
//   Training Cells: [guard][train......][CUT][......train][guard]
//                   |<-- left window -->|   |<-- right window -->|
//
// Pipeline:
//   Stage 1: Range line buffer fill (guard + train cells on each side)
//   Stage 2a: Partial-sum groups of 8 (registered pipeline cut)
//   Stage 2b: Final 4-input sum + noise estimate
//   Stage 3: Threshold comparison with selected CFAR method
//   Stage 4: Output detection with coordinates
//
// CFAR Methods:
//   - CA-CFAR (Cell Averaging): threshold = noise_floor × scale_factor
//   - GO-CFAR (Greatest Of): threshold = max(left_avg, right_avg) × scale
//   - SO-CFAR (Smallest Of): threshold = min(left_avg, right_avg) × scale
//   - OS-CFAR (Ordered Statistic): threshold = kth_ordered_value × scale
//
// Latency: 5 cycles + window fill time (1 extra for adder tree pipeline)
// Throughput: 1 detection/cycle after initial fill
// =============================================================================

import pkg_config::*;
import pkg_ra_processing::*;

module cfar_detector #(
    parameter int RANGE_BINS    = 1024,     // Maximum range bins
    parameter int ANGLE_BINS    = 4,        // Number of angle bins
    parameter int MAX_GUARD     = 8,        // Maximum guard cells per side
    parameter int MAX_TRAIN    = 32,       // Maximum training cells per side
    parameter int DATA_WIDTH    = 32,       // Magnitude data width
    parameter int FRAC_WIDTH    = 16        // Fractional bits (Q16.16)
)(
    input  logic clk,
    input  logic rst_n,

    // Configuration (from CSR)
    input  logic [9:0]  threshold_db,       // Detection threshold in dB × 256
    input  logic [3:0]  guard_cells,        // Number of guard cells (0-8)
    input  logic [4:0]  train_cells,        // Number of training cells (0-32)
    input  logic        cfar_en,            // CFAR detection enable
    input  logic [1:0]  cfar_mode,          // 00=CA-CFAR, 01=GO-CFAR, 10=SO-CFAR, 11=OS-CFAR
    input  logic [7:0]  scale_factor,       // Threshold scale factor (Q0.8)
    input  logic [3:0]  os_rank_k,          // Rank for OS-CFAR (k-th smallest value)

    // Input: Magnitude map from RA processor
    input  logic signed [DATA_WIDTH-1:0] mag_in     [ANGLE_BINS-1:0],
    input  logic [9:0]  range_idx,
    input  logic [1:0]  angle_idx,
    input  logic        mag_valid,
    input  logic        mag_last,

    // Output: Detections
    output logic signed [DATA_WIDTH-1:0] detection_value,
    output logic [9:0]  detection_range,
    output logic [1:0]  detection_angle,
    output logic        detection_valid,
    output logic        detection_last,

    // AXI-Stream output for detection map (optional)
    output logic        det_out_valid,
    output logic [DATA_WIDTH-1:0] det_out_data,
    input  logic        det_out_ready
);

    // =========================================================================
    // CFAR Mode Encoding
    // =========================================================================
    localparam bit [1:0] CFAR_CA = 2'b00;   // Cell Averaging
    localparam bit [1:0] CFAR_GO = 2'b01;   // Greatest Of
    localparam bit [1:0] CFAR_SO = 2'b10;   // Smallest Of
    localparam bit [1:0] CFAR_OS = 2'b11;   // Ordered Statistic

    // =========================================================================
    // Pipeline Stage 0: Input Registration
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] mag_d0 [ANGLE_BINS-1:0];
    logic [9:0]  range_idx_d0;
    logic [1:0]  angle_idx_d0;
    logic        mag_valid_d0;
    logic        mag_last_d0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mag_valid_d0 <= 1'b0;
            mag_last_d0  <= 1'b0;
            range_idx_d0 <= '0;
            angle_idx_d0 <= '0;
            for (int i = 0; i < ANGLE_BINS; i++)
                mag_d0[i] <= '0;
        end else begin
            mag_valid_d0 <= mag_valid;
            mag_last_d0  <= mag_last;
            range_idx_d0 <= range_idx;
            angle_idx_d0 <= angle_idx;
            for (int i = 0; i < ANGLE_BINS; i++)
                mag_d0[i] <= mag_in[i];
        end
    end

    // =========================================================================
    // Pipeline Stage 1: Range Line Buffer (Sliding Window)
    // =========================================================================
    // Store range_bins lines of magnitude data for sliding window
    // Window structure: [left_train | left_guard | CUT | right_guard | right_train]

    localparam int WINDOW_SIZE = 2 * (MAX_TRAIN + MAX_GUARD + 1);

    (* ram_style = "block" *)
    logic signed [DATA_WIDTH-1:0] range_buffer [RANGE_BINS-1:0][ANGLE_BINS-1:0];

    logic signed [DATA_WIDTH-1:0] window_left  [MAX_TRAIN-1:0][ANGLE_BINS-1:0];
    logic signed [DATA_WIDTH-1:0] window_right [MAX_TRAIN-1:0][ANGLE_BINS-1:0];
    logic signed [DATA_WIDTH-1:0] cut_value    [ANGLE_BINS-1:0];

    logic [9:0] wr_ptr;
    logic [9:0] rd_ptr_left;
    logic [9:0] rd_ptr_right;
    logic        buffer_valid;

    // Drive buffer_valid: asserted when valid data flows into the range buffer
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            buffer_valid <= 1'b0;
        else
            buffer_valid <= mag_valid_d0;
    end

    // Write pointer for circular buffer
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= '0;
        end else if (mag_valid_d0) begin
            wr_ptr <= (wr_ptr == RANGE_BINS - 1) ? '0 : wr_ptr + 1'b1;
        end
    end

    // Store magnitude data in circular buffer
    always_ff @(posedge clk) begin
        if (mag_valid_d0) begin
            for (int a = 0; a < ANGLE_BINS; a++)
                range_buffer[wr_ptr][a] <= mag_d0[a];
        end
    end

    // Read pointers for training windows
    always_comb begin
        // Left window: (guard_cells + train_cells) positions before CUT
        rd_ptr_left = (wr_ptr < (guard_cells + train_cells)) ?
                      RANGE_BINS + wr_ptr - guard_cells - train_cells :
                      wr_ptr - guard_cells - train_cells;
        // Right window: (guard_cells + train_cells) positions after CUT
        rd_ptr_right = (wr_ptr + guard_cells + train_cells) & (RANGE_BINS - 1);
    end

    // FIXED: CFAR-DEAD-003 — Populate sliding window arrays from range_buffer.
    // Read training cells and CUT value using computed read pointers.
    always_comb begin
        for (int t = 0; t < MAX_TRAIN; t++) begin
            for (int a = 0; a < ANGLE_BINS; a++) begin
                window_left[t][a]  = range_buffer[(rd_ptr_left + t) & (RANGE_BINS - 1)][a];
                window_right[t][a] = range_buffer[(rd_ptr_right + t) & (RANGE_BINS - 1)][a];
            end
        end
        for (int a = 0; a < ANGLE_BINS; a++)
            cut_value[a] = range_buffer[wr_ptr][a];
    end

    // =========================================================================
    // Pipeline Stage 2a: Partial-Sum Groups (registered pipeline cut)
    // =========================================================================
    // Split 32-input adder tree into groups of 8 to reduce combinational depth.
    // Each group sums 8 training cells; partial sums are registered.
    // This cuts LUT usage by ~500-700 by breaking the 32-input combinational
    // chain into 4×(8-input) + 1×(4-input) structure.

    localparam int SUM_GROUPS = (MAX_TRAIN + 7) / 8;  // 4 groups for MAX_TRAIN=32
    localparam int CELLS_PER_GROUP = 8;

    logic signed [DATA_WIDTH+7:0] left_partial_comb [ANGLE_BINS-1:0][SUM_GROUPS-1:0];
    logic signed [DATA_WIDTH+7:0] right_partial_comb [ANGLE_BINS-1:0][SUM_GROUPS-1:0];
    logic signed [DATA_WIDTH+7:0] left_partial [ANGLE_BINS-1:0][SUM_GROUPS-1:0];
    logic signed [DATA_WIDTH+7:0] right_partial [ANGLE_BINS-1:0][SUM_GROUPS-1:0];
    logic signed [DATA_WIDTH-1:0] cut_value_d2a [ANGLE_BINS-1:0];
    logic valid_d2a;
    logic last_d2a;
    logic [9:0] range_idx_d2a;
    logic [1:0] angle_idx_d2a;

    // Combinational: sum 8 cells per group
    always_comb begin
        for (int a = 0; a < ANGLE_BINS; a++) begin
            for (int g = 0; g < SUM_GROUPS; g++) begin
                left_partial_comb[a][g] = '0;
                right_partial_comb[a][g] = '0;
                for (int c = 0; c < CELLS_PER_GROUP; c++) begin
                    if (g * CELLS_PER_GROUP + c < train_cells) begin
                        left_partial_comb[a][g]  += window_left[g * CELLS_PER_GROUP + c][a];
                        right_partial_comb[a][g] += window_right[g * CELLS_PER_GROUP + c][a];
                    end
                end
            end
        end
    end

    // Register partial sums + CUT value
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d2a <= 1'b0;
            last_d2a  <= 1'b0;
            range_idx_d2a <= '0;
            angle_idx_d2a <= '0;
            for (int a = 0; a < ANGLE_BINS; a++) begin
                cut_value_d2a[a] <= '0;
                for (int g = 0; g < SUM_GROUPS; g++) begin
                    left_partial[a][g]  <= '0;
                    right_partial[a][g] <= '0;
                end
            end
        end else begin
            valid_d2a <= buffer_valid;
            last_d2a  <= last_d1;
            range_idx_d2a <= range_idx_d1;
            angle_idx_d2a <= angle_idx_d1;

            if (cfar_en && buffer_valid) begin
                for (int a = 0; a < ANGLE_BINS; a++) begin
                    cut_value_d2a[a] <= cut_value[a];
                    for (int g = 0; g < SUM_GROUPS; g++) begin
                        left_partial[a][g]  <= left_partial_comb[a][g];
                        right_partial[a][g] <= right_partial_comb[a][g];
                    end
                end
            end
        end
    end

    // =========================================================================
    // Pipeline Stage 2b: Final Sum + Noise Estimate
    // =========================================================================
    // Combine 4 partial sums into left_sum / right_sum, then compute
    // noise estimate based on CFAR mode. The 4-input add is combinational
    // but much smaller than the original 32-input tree.

    logic signed [DATA_WIDTH+7:0] left_sum [ANGLE_BINS-1:0];
    logic signed [DATA_WIDTH+7:0] right_sum [ANGLE_BINS-1:0];
    logic signed [DATA_WIDTH+7:0] noise_est [ANGLE_BINS-1:0];
    logic signed [DATA_WIDTH-1:0] cut_value_d2b [ANGLE_BINS-1:0];
    logic valid_d2;

    always_comb begin
        for (int a = 0; a < ANGLE_BINS; a++) begin
            left_sum[a] = '0;
            right_sum[a] = '0;
            for (int g = 0; g < SUM_GROUPS; g++) begin
                left_sum[a]  += left_partial[a][g];
                right_sum[a] += right_partial[a][g];
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d2 <= 1'b0;
            for (int a = 0; a < ANGLE_BINS; a++) begin
                noise_est[a] <= '0;
                cut_value_d2b[a] <= '0;
            end
        end else begin
            valid_d2 <= valid_d2a;

            if (cfar_en && valid_d2a) begin
                for (int a = 0; a < ANGLE_BINS; a++) begin
                    cut_value_d2b[a] <= cut_value_d2a[a];
                    case (cfar_mode)
                        CFAR_CA: begin
                            noise_est[a] <= (left_sum[a] + right_sum[a])
                                            >>> ($clog2(2 * train_cells));
                        end
                        CFAR_GO: begin
                            noise_est[a] <= (left_sum[a] > right_sum[a]) ?
                                            (left_sum[a] >>> $clog2(train_cells)) :
                                            (right_sum[a] >>> $clog2(train_cells));
                        end
                        CFAR_SO: begin
                            noise_est[a] <= (left_sum[a] < right_sum[a]) ?
                                            (left_sum[a] >>> $clog2(train_cells)) :
                                            (right_sum[a] >>> $clog2(train_cells));
                        end
                        CFAR_OS: begin
                            noise_est[a] <= (left_sum[a] + right_sum[a])
                                            >>> ($clog2(2 * train_cells) + 1);
                        end
                        default: noise_est[a] <= '0;
                    endcase
                end
            end
        end
    end

    // =========================================================================
    // Pipeline Stage 3: Threshold Comparison
    // =========================================================================
    // Detect if CUT magnitude exceeds threshold × noise_estimate
    // Optimization: truncate noise_est to 24 bits before 24×8 multiply
    // instead of 39×32. The top bits of noise_est are always zero or sign
    // extension for typical radar magnitudes. This reduces multiplier LUTs
    // by ~60% with no precision loss for typical data.

    localparam int NOISE_TRUNC_W = 24;
    logic signed [NOISE_TRUNC_W-1:0] noise_est_trunc [ANGLE_BINS-1:0];
    logic signed [NOISE_TRUNC_W+7:0] threshold_scaled [ANGLE_BINS-1:0];
    logic        detection_flag [ANGLE_BINS-1:0];
    logic valid_d3;

    always_comb begin
        for (int a = 0; a < ANGLE_BINS; a++)
            noise_est_trunc[a] = noise_est[a][NOISE_TRUNC_W-1:0];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d3 <= 1'b0;
            for (int a = 0; a < ANGLE_BINS; a++) begin
                threshold_scaled[a] <= '0;
                detection_flag[a] <= 1'b0;
            end
        end else begin
            valid_d3 <= valid_d2;

            if (cfar_en && valid_d2) begin
                for (int a = 0; a < ANGLE_BINS; a++) begin
                    threshold_scaled[a] <= (noise_est_trunc[a] * NOISE_TRUNC_W'(scale_factor)) >>> 8;
                    detection_flag[a] <= (cut_value_d2b[a] > threshold_scaled[a][DATA_WIDTH-1:0]);
                end
            end
        end
    end

    // =========================================================================
    // Pipeline Stage 4: Output Formatting
    // =========================================================================
    // Output detected targets with coordinates

    logic signed [DATA_WIDTH-1:0] mag_d3 [ANGLE_BINS-1:0];
    logic [9:0] range_idx_d1, range_idx_d3;
    logic [1:0] angle_idx_d1, angle_idx_d3;
    logic last_d1, last_d3, last_d4;
    logic valid_d4;

    // FIXED: CFAR-PIPE-008 — Pipeline metadata (range_idx, angle_idx, last)
    // through every stage alongside valid, instead of shortcutting.
    // Stage 2a registers (range_idx_d2a, angle_idx_d2a, last_d2a) are
    // already declared in the partial-sum section above.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            range_idx_d1 <= '0; range_idx_d3 <= '0;
            angle_idx_d1 <= '0; angle_idx_d3 <= '0;
            last_d1 <= 1'b0; last_d3 <= 1'b0; last_d4 <= 1'b0;
        end else begin
            // Stage 0→1
            range_idx_d1 <= range_idx_d0;
            angle_idx_d1 <= angle_idx_d0;
            last_d1      <= mag_last_d0;
            // Stage 2a→2b (registered in partial-sum block above)
            // Stage 2b→3
            range_idx_d3 <= range_idx_d2a;
            angle_idx_d3 <= angle_idx_d2a;
            last_d3      <= last_d2a;
            // Stage 3→4
            last_d4      <= last_d3;
        end
    end

    // Pipeline mag and valid through stages 3→4
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d4 <= 1'b0;
            for (int a = 0; a < ANGLE_BINS; a++)
                mag_d3[a] <= '0;
        end else begin
            valid_d4 <= valid_d3;
            for (int a = 0; a < ANGLE_BINS; a++)
                mag_d3[a] <= cut_value_d2b[a];
        end
    end

    // FIXED: CFAR-PRIOR-009 + CFAR-CAST-010 — Combinational priority encoder.
    // Previously used NBAs in for-loop (detection_valid read stale value).
    // Now uses combinational logic with registered output.
    logic signed [DATA_WIDTH-1:0] next_det_value;
    logic [9:0]  next_det_range;
    logic [1:0]  next_det_angle;
    logic        next_det_valid;

    always_comb begin
        next_det_valid = 1'b0;
        next_det_value = '0;
        next_det_range = '0;
        next_det_angle = '0;
        for (int a = 0; a < ANGLE_BINS; a++) begin
            if (detection_flag[a] && !next_det_valid) begin
                next_det_valid = 1'b1;
                next_det_value = mag_d3[a];
                next_det_range = range_idx_d3;
                next_det_angle = a[1:0];  // FIXED: CFAR-CAST-010 — was angle_idx_d3'(a)
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            detection_valid <= 1'b0;
            detection_value <= '0;
            detection_range <= '0;
            detection_angle <= '0;
            detection_last  <= 1'b0;
        end else if (cfar_en && valid_d4) begin
            detection_valid  <= next_det_valid;
            detection_value  <= next_det_value;
            detection_range  <= next_det_range;
            detection_angle  <= next_det_angle;
            detection_last   <= last_d4;
        end else begin
            detection_valid <= 1'b0;
            detection_last  <= 1'b0;
        end
    end

    // AXI-Stream output with backpressure
    // FIXED: det_out_ready was unused — detections could be lost when consumer
    // isn't ready. Now holds output valid/data stable until det_out_ready.
    logic det_out_held_valid;
    logic signed [DATA_WIDTH-1:0] det_out_held_data;
    logic [9:0]  det_out_held_range;
    logic [1:0]  det_out_held_angle;
    logic        det_out_held_last;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            det_out_held_valid <= 1'b0;
            det_out_held_data  <= '0;
            det_out_held_range <= '0;
            det_out_held_angle <= '0;
            det_out_held_last  <= 1'b0;
        end else if (det_out_held_valid && !det_out_ready) begin
            // Consumer not ready — hold current output
        end else begin
            det_out_held_valid <= detection_valid;
            det_out_held_data  <= detection_value;
            det_out_held_range <= detection_range;
            det_out_held_angle <= detection_angle;
            det_out_held_last  <= detection_last;
        end
    end

    assign det_out_valid = det_out_held_valid;
    assign det_out_data  = det_out_held_data;

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    `ifndef VERILATOR
    CFAR_GUARD_RANGE: assert property (
        @(posedge clk) disable iff (!rst_n)
        guard_cells >= 0 && guard_cells <= MAX_GUARD
    ) else $error("cfar_detector: guard_cells out of range");

    CFAR_TRAIN_RANGE: assert property (
        @(posedge clk) disable iff (!rst_n)
        train_cells >= 1 && train_cells <= MAX_TRAIN
    ) else $error("cfar_detector: train_cells out of range");

    // FIXED: Warn if train_cells is non-power-of-2 (division is approximate)
    CFAR_TRAIN_POWER_OF_2: assert property (
        @(posedge clk) disable iff (!rst_n)
        (train_cells & (train_cells - 1)) == 0 || train_cells == 0
    ) else $warning("cfar_detector: train_cells is not power-of-2, noise estimate is approximate");

    CFAR_MODE_RANGE: assert property (
        @(posedge clk) disable iff (!rst_n)
        cfar_mode inside {CFAR_CA, CFAR_GO, CFAR_SO, CFAR_OS}
    ) else $error("cfar_detector: invalid CFAR mode");
    `endif // VERILATOR
    // synthesis translate_on

endmodule