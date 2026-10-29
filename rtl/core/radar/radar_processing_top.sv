// =============================================================================
// radar_processing_top.sv
//
// Top-level integration module for the radar processing pipeline.
// Chains RA Map Processor (windowing + FFT + beamforming + magnitude)
// into CFAR Detector for automatic target detection.
//
// Pipeline:
//   ADC → Windowing → Range FFT → Beamforming → Angle FFT →
//   Magnitude → CFAR Detection → Detection Output
//
// The RA Map Processor produces a range-angle magnitude map, and the
// CFAR Detector applies adaptive thresholding to identify targets.
//
// Latency: ra_map_processor (~20 cycles + FFT) + cfar_detector (~4 cycles + fill)
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;

module radar_processing_top #(
    parameter int NUM_ANTENNAS    = 4,
    parameter int NUM_BEAMS       = 5,
    parameter int MAX_RANGE_BINS  = 1024,
    parameter int MAX_ANGLE_BINS  = 4,
    parameter int DATA_WIDTH      = pkg_module_config::RADAR_PROCESSING_TOP_DATA_WIDTH,
    parameter int FRAC_WIDTH      = 16,
    parameter int MAX_GUARD       = 8,
    parameter int MAX_TRAIN       = 32
)(
    input  logic clk,
    input  logic rst_n,

    // ── RA Configuration ──────────────────────────────────────────────
    input  logic [11:0] range_fft_size,
    input  logic [2:0]  range_fft_radix,
    input  logic [1:0]  window_type,
    input  logic        window_enable,
    input  logic        beamforming_enable,
    input  logic        angle_fft_enable,
    input  logic        magnitude_enable,
    input  logic        log_output,

    // ── CFAR Configuration ─────────────────────────────────────────────
    input  logic [9:0]  cfar_threshold_db,
    input  logic [3:0]  cfar_guard_cells,
    input  logic [4:0]  cfar_train_cells,
    input  logic        cfar_en,
    input  logic [1:0]  cfar_mode,
    input  logic [7:0]  cfar_scale_factor,
    input  logic [3:0]  cfar_os_rank_k,

    // ── ADC Input ──────────────────────────────────────────────────────
    input  logic signed [DATA_WIDTH-1:0] adc_re [NUM_ANTENNAS-1:0],
    input  logic signed [DATA_WIDTH-1:0] adc_im [NUM_ANTENNAS-1:0],
    input  logic        adc_valid,
    input  logic        adc_last,
    input  logic [11:0] adc_sample_idx,
    output logic        adc_ready,

    // ── FFT Accelerator Interface ──────────────────────────────────────
    output logic        fft_start,
    output logic [11:0] fft_size,
    output logic [2:0]  fft_radix_out,
    output logic        fft_inverse,
    input  logic        fft_done,
    input  logic        fft_busy,

    // ── Detection Output ───────────────────────────────────────────────
    output logic signed [DATA_WIDTH-1:0] detection_value,
    output logic [9:0]  detection_range,
    output logic [1:0]  detection_angle,
    output logic        detection_valid,
    output logic        detection_last,
    input  logic        detection_ready,      // FIXED: CFAR-DETECT-NOREADY-001

    // ── RA Map Output (bypass CFAR) ───────────────────────────────────
    output logic signed [DATA_WIDTH-1:0] ra_mag [MAX_ANGLE_BINS-1:0],
    output logic [9:0]  ra_range_idx,
    output logic [1:0]  ra_angle_idx,
    output logic        ra_valid,
    output logic        ra_last,
    input  logic        ra_ready,       // FIXED: RA-READY-DRIVE-001 — removed illegal assign

    // ── AXI-Stream Detection Output ────────────────────────────────────
    output logic        det_out_valid,
    output logic [DATA_WIDTH-1:0] det_out_data,
    input  logic        det_out_ready
);

    // =========================================================================
    // RA Map Processor
    // =========================================================================
    ra_map_processor #(
        .NUM_ANTENNAS   (NUM_ANTENNAS),
        .NUM_BEAMS      (NUM_BEAMS),
        .MAX_RANGE_BINS (MAX_RANGE_BINS),
        .MAX_ANGLE_BINS (MAX_ANGLE_BINS),
        .DATA_WIDTH     (DATA_WIDTH),
        .FRAC_WIDTH     (FRAC_WIDTH)
    ) u_ra_map (
        .clk                (clk),
        .rst_n              (rst_n),
        .range_fft_size     (range_fft_size),
        .range_fft_radix    (range_fft_radix),
        .window_type        (window_type),
        .window_enable      (window_enable),
        .beamforming_enable (beamforming_enable),
        .angle_fft_enable   (angle_fft_enable),
        .magnitude_enable   (magnitude_enable),
        .log_output         (log_output),

        .adc_re           (adc_re),
        .adc_im           (adc_im),
        .adc_valid        (adc_valid),
        .adc_last         (adc_last),
        .adc_sample_idx   (adc_sample_idx),
        .adc_ready        (adc_ready),

        .fft_start        (fft_start),
        .fft_size         (fft_size),
        .fft_radix        (fft_radix_out),
        .fft_inverse      (fft_inverse),
        .fft_done         (fft_done),
        .fft_busy         (fft_busy),

        .ra_mag          (ra_mag),
        .ra_range_idx    (ra_range_idx),
        .ra_angle_idx    (ra_angle_idx),
        .ra_valid        (ra_valid),
        .ra_last         (ra_last),
        .ra_ready        (ra_ready_internal)  // FIXED: CFAR-BACKPRESS-001 — gate with downstream
    );

    // =========================================================================
    // CFAR Detector
    // =========================================================================
    cfar_detector #(
        .RANGE_BINS  (MAX_RANGE_BINS),
        .ANGLE_BINS  (MAX_ANGLE_BINS),
        .MAX_GUARD   (MAX_GUARD),
        .MAX_TRAIN   (MAX_TRAIN),
        .DATA_WIDTH  (DATA_WIDTH),
        .FRAC_WIDTH  (FRAC_WIDTH)
    ) u_cfar (
        .clk            (clk),
        .rst_n          (rst_n),

        .threshold_db   (cfar_threshold_db),
        .guard_cells    (cfar_guard_cells),
        .train_cells    (cfar_train_cells),
        .cfar_en        (cfar_en),
        .cfar_mode      (cfar_mode),
        .scale_factor   (cfar_scale_factor),
        .os_rank_k       (cfar_os_rank_k),

        .mag_in          (ra_mag),
        .range_idx      (ra_range_idx),
        .angle_idx      (ra_angle_idx),
        .mag_valid       (ra_valid),
        .mag_last        (ra_last),

        .detection_value  (detection_value),
        .detection_range  (detection_range),
        .detection_angle  (detection_angle),
        .detection_valid   (detection_valid),
        .detection_last    (detection_last),

        .det_out_valid   (det_out_valid),
        .det_out_data    (det_out_data),
        .det_out_ready   (det_out_ready)
    );

    // =========================================================================
    // Backpressure Logic (CFAR-BACKPRESS-001)
    // =========================================================================
    // RA Map output fans out to external port and CFAR. Both consumers must
    // be ready before we can assert backpressure to the RA Map processor.
    // FIXED: RA-READY-DRIVE-001 — ra_ready is now correctly an input from the
    // external consumer. Combined backpressure gates RA Map processor.
    logic ra_ready_internal;
    assign ra_ready_internal = ra_ready & detection_ready;

endmodule