// ==========================================
// AUTO-GENERATED FILE. DO NOT EDIT MANUALLY.
// Generated from config.yaml — radar section
// ==========================================

package pkg_ra_config;

    // ── Antenna Configuration ──
    localparam int NUM_TX_ANTENNAS    = 3;
    localparam int NUM_RX_ANTENNAS    = 4;
    localparam int NUM_VIRTUAL_ANTENNAS = 3 * 4;  // 12 virtual (MIMO)

    // ── Beam Configuration ──
    localparam int NUM_BEAMS          = 5;
    localparam int MAX_BEAMS          = 8;

    // ── FFT Configuration ──
    localparam int MAX_RANGE_BINS     = 1024;
    localparam int MIN_RANGE_BINS     = 64;
    localparam int DEFAULT_RANGE_BINS = 256;
    localparam int MAX_DOPPLER_BINS   = 256;
    localparam int MIN_DOPPLER_BINS   = 16;
    localparam int DEFAULT_DOPPLER_BINS = 64;
    localparam int ANGLE_FFT_SIZE     = 4;

    // ── Data Widths ──
    localparam int ADC_DATA_WIDTH      = 12;
    localparam int FFT_DATA_WIDTH      = 32;
    localparam int TWIDDLE_WIDTH       = 18;

    // ── CORDIC Configuration ──
    localparam bit USE_CORDIC          = 1'b0;
    localparam int CORDIC_STAGES       = 12;

    // ── CFAR Configuration ──
    localparam int CFAR_GUARD_CELLS   = 4;
    localparam int CFAR_TRAIN_CELLS   = 8;
    localparam int CFAR_THRESHOLD_DB  = 10;

    // ── Pipeline Latencies ──
    localparam int WINDOW_LATENCY     = 3;
    localparam int BEAMFORM_LATENCY   = 5;
    localparam int ANGLE_FFT_LATENCY  = 3;
    localparam int MAGNITUDE_LATENCY  = 4;

endpackage