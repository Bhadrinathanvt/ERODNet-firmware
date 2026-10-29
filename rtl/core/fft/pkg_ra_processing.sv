// =============================================================================
// pkg_ra_processing.sv
//
// Package containing Range-Angle Map Processing types, CSR register map,
// pre-computed constants, and utility functions.
//
// Numeric parameters (antennas, FFT sizes, CFAR, data widths, latencies)
// are imported from pkg_ra_config (auto-generated from config.yaml).
// Do not duplicate those localparams here — edit config.yaml and regenerate.
// =============================================================================

package pkg_ra_processing;

    import pkg_ra_config::*;

    // =========================================================================
    // Windowing Types
    // =========================================================================

    typedef enum logic [1:0] {
        WIN_NONE     = 2'b00,    // Rectangular (no window)
        WIN_HAMMING  = 2'b01,    // Hamming window
        WIN_HANNING  = 2'b10,    // Hanning window
        WIN_BLACKMAN = 2'b11     // Blackman window
    } window_type_e;

    // =========================================================================
    // Beamforming Modes
    // =========================================================================

    typedef enum logic [2:0] {
        BEAM_5_DIR    = 3'b000,   // 5 beams: -60°, -30°, 0°, +30°, +60°
        BEAM_7_DIR    = 3'b001,   // 7 beams: -60°, -40°, -20°, 0°, +20°, +40°, +60°
        BEAM_9_DIR    = 3'b010,   // 9 beams: finer resolution
        BEAM_MIMO     = 3'b011,   // MIMO: 12 virtual antennas
        BEAM_CUSTOM   = 3'b100    // Custom phase weights
    } beam_mode_e;

    // =========================================================================
    // RA Map Processing Modes
    // =========================================================================

    typedef enum logic [2:0] {
        RA_RANGE_ANGLE   = 3'b000,   // Range-Angle map (Doppler collapsed)
        RA_RANGE_DOPPLER = 3'b001,   // Range-Doppler map (Angle collapsed)
        RA_RANGE_ONLY    = 3'b010,   // Range only (angle FFT bypassed)
        RA_FULL_3D       = 3'b011,   // Full 3D: Range × Doppler × Angle
        RA_CFAR_OUTPUT   = 3'b100    // CFAR detection output
    } ra_mode_e;

    // =========================================================================
    // Register Map
    // =========================================================================

    // RA Processor CSR Addresses (12-bit)
    localparam int CSR_RA_CONFIG      = 12'h000;  // RW: Mode, window, beamform enable
    localparam int CSR_RA_RANGE_SIZE  = 12'h004;  // RW: Range FFT size
    localparam int CSR_RA_DOPPLER_SIZE= 12'h008;  // RW: Doppler FFT size (for 3D)
    localparam int CSR_RA_WINDOW_TYPE = 12'h00C;  // RW: Windowing type
    localparam int CSR_RA_BEAM_MODE   = 12'h010;  // RW: Beamforming mode
    localparam int CSR_RA_CFAR_THRESH  = 12'h014;  // RW: CFAR threshold
    localparam int CSR_RA_STATUS      = 12'h018;  // RO: Status flags
    localparam int CSR_RA_FRAME_COUNT = 12'h01C;  // RO: Frames processed

    // RA_CONFIG register bit fields
    localparam int RA_WINDOW_EN_BIT    = 0;
    localparam int RA_BEAMFORM_EN_BIT  = 1;
    localparam int RA_ANGLE_FFT_EN_BIT = 2;
    localparam int RA_MAG_EN_BIT       = 3;
    localparam int RA_LOG_OUT_BIT      = 4;
    localparam int RA_CFAR_EN_BIT      = 5;

    // RA_STATUS register bit fields
    localparam int RA_BUSY_BIT         = 0;
    localparam int RA_DONE_BIT         = 1;
    localparam int RA_ERROR_BIT        = 2;
    localparam int RA_OVERFLOW_BIT     = 3;

    // =========================================================================
    // Pre-computed Beam Directions (sin(θ) in Q0.16)
    // =========================================================================

    localparam signed [15:0] SIN_THETA [0:4] = '{
        16'sh9000,   // sin(-60°) ≈ -0.866
        16'shC000,   // sin(-30°) ≈ -0.5
        16'sh0000,   // sin(0°)   =  0
        16'sh4000,   // sin(+30°) ≈ +0.5
        16'sh7000    // sin(+60°) ≈ +0.866
    };

    // =========================================================================
    // Useful Functions
    // =========================================================================

    // Calculate number of range bins from FFT size
    function automatic int range_bins_from_fft(input int fft_size);
        return fft_size / 2;  // Only positive frequencies
    endfunction

    // Calculate range resolution (meters)
    // range_res = c / (2 * B), where B = bandwidth
    // For AWR1843AOP: 76-81 GHz, B = 4 GHz typical
    function automatic real range_resolution(input int fft_size, input real bandwidth_hz);
        real c = 299792458.0;  // Speed of light
        return c / (2.0 * bandwidth_hz);
    endfunction

    // Calculate angle resolution (degrees)
    // angle_res = arcsin(λ / (N * d)), where d = λ/2
    // For 4 antennas: angle_res ≈ 28.6°
    function automatic real angle_resolution(input int num_antennas);
        real lambda = 1.0;  // Normalized wavelength
        return $asin(lambda / (real'(num_antennas) * 0.5)) * 180.0 / 3.14159265359;
    endfunction

    // Total RA processing latency (excluding Range FFT)
    function automatic int ra_total_latency();
        return WINDOW_LATENCY + BEAMFORM_LATENCY + ANGLE_FFT_LATENCY + MAGNITUDE_LATENCY;
    endfunction

endpackage