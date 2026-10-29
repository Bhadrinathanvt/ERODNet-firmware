// =============================================================================
// ra_map_processor.sv
//
// Range-Angle Map Processor for AWR1843AOP mmWave Radar.
// Integrates Range FFT, Beamforming, Angle FFT, and Magnitude computation.
//
// Pipeline:
//   ADC Data → Windowing → Range FFT → Beamforming → Angle FFT → Magnitude → RA Map
//
// AWR1843AOP Configuration:
//   - 4 RX Antennas (MIMO possible with 3 TX)
//   - Typical Range FFT: 256-1024 points
//   - Angle FFT: 4 points (one per RX antenna)
//   - Output: Range bins × Angle bins magnitude map
//
// Latency:
//   - Windowing: 3 cycles
//   - Range FFT: N*log2(N) cycles (using external FFT accelerator)
//   - Beamforming: 5 cycles
//   - Angle FFT: 3 cycles
//   - Magnitude: 4 cycles
//   Total: ~20 cycles + Range FFT time
//
// Data Flow:
//   1. Input: 4 complex samples (one per RX antenna)
//   2. Windowing applied per antenna
//   3. Range FFT (external, uses fft_accelerator)
//   4. Digital beamforming across antennas
//   5. 4-point Angle FFT
//   6. Magnitude computation
//   7. Output: RA map pixel
// =============================================================================

import pkg_config::*;

module ra_map_processor #(
    parameter int NUM_ANTENNAS    = 4,       // AWR1843AOP: 4 RX
    parameter int NUM_BEAMS       = 5,       // Beam directions
    parameter int MAX_RANGE_BINS  = 1024,    // Maximum range FFT size
    parameter int MAX_ANGLE_BINS  = 4,       // = NUM_ANTENNAS
    parameter int DATA_WIDTH      = 32,
    parameter int FRAC_WIDTH     = 16,
    parameter int PHASE_WIDTH    = 18
)(
    input  logic clk,
    input  logic rst_n,

    // Configuration
    input  logic [11:0] range_fft_size,        // Range FFT size (256, 512, 1024)
    input  logic [2:0]  range_fft_radix,       // Range FFT radix (2, 4, 8)
    input  logic [1:0]  window_type,           // Windowing type
    input  logic        window_enable,          // Enable windowing
    input  logic        beamforming_enable,    // Enable beamforming
    input  logic        angle_fft_enable,      // Enable angle FFT
    input  logic        magnitude_enable,      // Enable magnitude output
    input  logic        log_output,            // Output log10(magnitude) in dB

    // Input: Raw ADC data (complex samples from each antenna)
    input  logic signed [DATA_WIDTH-1:0] adc_re [NUM_ANTENNAS-1:0],
    input  logic signed [DATA_WIDTH-1:0] adc_im [NUM_ANTENNAS-1:0],
    input  logic        adc_valid,
    input  logic        adc_last,              // Last sample of chirp
    input  logic [11:0] adc_sample_idx,        // Sample index in chirp
    output logic        adc_ready,

    // FFT Accelerator Interface (for Range FFT)
    output logic        fft_start,
    output logic [11:0] fft_size,
    output logic [2:0]  fft_radix,
    output logic        fft_inverse,
    input  logic        fft_done,
    input  logic        fft_busy,

    // Output: RA Map
    output logic signed [DATA_WIDTH-1:0] ra_mag [MAX_ANGLE_BINS-1:0],  // Magnitude per angle bin
    output logic [9:0]  ra_range_idx,         // Range bin index
    output logic [1:0]  ra_angle_idx,         // Angle bin index
    output logic        ra_valid,
    output logic        ra_last,               // Last pixel of frame
    input  logic        ra_ready
);

    // =========================================================================
    // Internal Signals
    // =========================================================================

    // Stage 1: Windowing Output
    logic signed [DATA_WIDTH-1:0] win_re [NUM_ANTENNAS-1:0];
    logic signed [DATA_WIDTH-1:0] win_im [NUM_ANTENNAS-1:0];
    logic win_valid, win_last;

    // Stage 2: Range FFT Output (complex)
    // Note: Range FFT is done by external fft_accelerator
    // This module receives FFT results via streaming interface
    logic signed [DATA_WIDTH-1:0] range_re [NUM_ANTENNAS-1:0];
    logic signed [DATA_WIDTH-1:0] range_im [NUM_ANTENNAS-1:0];
    logic range_valid, range_last;
    logic [9:0] range_bin_idx;

    // Stage 3: Beamforming Output
    logic signed [DATA_WIDTH-1:0] beam_re [NUM_BEAMS-1:0];
    logic signed [DATA_WIDTH-1:0] beam_im [NUM_BEAMS-1:0];
    logic beam_valid, beam_last;

    // Stage 4: Angle FFT Output
    logic signed [DATA_WIDTH-1:0] angle_re [MAX_ANGLE_BINS-1:0];
    logic signed [DATA_WIDTH-1:0] angle_im [MAX_ANGLE_BINS-1:0];
    logic angle_valid, angle_last;

    // Stage 5: Magnitude Output
    logic signed [DATA_WIDTH-1:0] mag_out [MAX_ANGLE_BINS-1:0];
    logic mag_valid, mag_last;

    // =========================================================================
    // Stage 1: Windowing
    // =========================================================================
    // Apply window function to each antenna's data separately

    generate
        for (genvar ant = 0; ant < NUM_ANTENNAS; ant++) begin : gen_windowing
            windowing_unit #(
                .DATA_WIDTH(DATA_WIDTH),
                .FRAC_WIDTH(FRAC_WIDTH),
                .MAX_FFT_SIZE(MAX_RANGE_BINS)
            ) u_windowing (
                .clk(clk),
                .rst_n(rst_n),
                .window_type(window_type),
                .fft_size(range_fft_size),
                .window_enable(window_enable),
                .data_re_in(adc_re[ant]),
                .data_im_in(adc_im[ant]),
                .valid_in(adc_valid),
                .last_in(adc_last),
                .sample_idx(adc_sample_idx),
                .data_re_out(win_re[ant]),
                .data_im_out(win_im[ant]),
                .valid_out(win_valid),
                .last_out(win_last)
            );
        end
    endgenerate

    // =========================================================================
    // Stage 2: Range FFT (External)
    // =========================================================================
    // The Range FFT is performed by the external fft_accelerator
    // This module controls the FFT and receives results

    // FFT Control State Machine
    typedef enum logic [1:0] {
        IDLE,
        RANGE_FFT_WAIT,
        RANGE_FFT_DONE
    } state_t;

    state_t state;
    logic [9:0] chirp_cnt;
    logic [11:0] fft_sample_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            fft_start <= 1'b0;
            fft_size <= '0;
            fft_radix <= '0;
            fft_inverse <= 1'b0;  // Forward FFT
            chirp_cnt <= '0;
            fft_sample_cnt <= '0;
        end else begin
            case (state)
                IDLE: begin
                    fft_start <= 1'b0;
                    if (adc_valid && adc_last) begin
                        // Start Range FFT for this chirp — assert fft_start
                        // for one cycle and go directly to WAIT
                        state <= RANGE_FFT_WAIT;
                        fft_start <= 1'b1;
                        fft_size <= range_fft_size;
                        fft_radix <= range_fft_radix;
                    end
                end

                RANGE_FFT_WAIT: begin
                    fft_start <= 1'b0;
                    if (fft_done) begin
                        state <= RANGE_FFT_DONE;
                    end
                end

                RANGE_FFT_DONE: begin
                    if (fft_sample_cnt >= range_fft_size - 1) begin
                        state <= IDLE;
                        fft_sample_cnt <= '0;
                        chirp_cnt <= chirp_cnt + 1;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

    // =========================================================================
    // Stage 3: Digital Beamforming
    // =========================================================================
    logic beam_ready;
    logic signed [PHASE_WIDTH-1:0] default_phases [NUM_ANTENNAS-1:0][NUM_BEAMS-1:0];
    always_comb begin
        for (int a = 0; a < NUM_ANTENNAS; a++)
            for (int b = 0; b < NUM_BEAMS; b++)
                default_phases[a][b] = '0;
    end

    digital_beamforming #(
        .NUM_ANTENNAS(NUM_ANTENNAS),
        .NUM_BEAMS(NUM_BEAMS),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH)
    ) u_beamforming (
        .clk(clk),
        .rst_n(rst_n),
        .beamforming_enable(beamforming_enable),
        .beam_mode('0),  // Default 5-beam mode
        .custom_phases(default_phases),
        .antenna_re(range_re),
        .antenna_im(range_im),
        .valid_in(range_valid),
        .ready_in(beam_ready),
        .last_in(range_last),
        .beam_re(beam_re),
        .beam_im(beam_im),
        .valid_out(beam_valid),
        .out_ready(beamforming_out_ready),  // Backpressure from output pipeline
        .last_out(beam_last)
    );

    // =========================================================================
    // Stage 4: Angle FFT (4-point)
    // =========================================================================
    angle_fft_4pt #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH)
    ) u_angle_fft (
        .clk(clk),
        .rst_n(rst_n),
        .x_re(beam_re[0 +: 4]),  // First 4 beams for 4-point FFT
        .x_im(beam_im[0 +: 4]),
        .valid_in(beam_valid),
        .last_in(beam_last),
        .X_re(angle_re),
        .X_im(angle_im),
        .valid_out(angle_valid),
        .last_out(angle_last)
    );

    // =========================================================================
    // Stage 5: Magnitude Computation
    // =========================================================================
    // FIXED: CFAR-MULTIDRIVER-002 — Per-bin valid/last arrays to avoid
    // multi-driver conflict on scalar mag_valid/mag_last.
    logic mag_valid_arr [MAX_ANGLE_BINS-1:0];
    logic mag_last_arr  [MAX_ANGLE_BINS-1:0];

    generate
        for (genvar bin = 0; bin < MAX_ANGLE_BINS; bin++) begin : gen_magnitude
            magnitude_unit #(
                .DATA_WIDTH(DATA_WIDTH),
                .FRAC_WIDTH(FRAC_WIDTH),
                .OUTPUT_WIDTH(DATA_WIDTH),
                .USE_CORDIC(1'b0)  // Use mult-based (faster)
            ) u_magnitude (
                .clk(clk),
                .rst_n(rst_n),
                .magnitude_enable(magnitude_enable),
                .output_squared(1'b0),  // Output magnitude, not squared
                .log_output(log_output),
                .data_re_in(angle_re[bin]),
                .data_im_in(angle_im[bin]),
                .valid_in(angle_valid),
                .last_in(angle_last && (bin == MAX_ANGLE_BINS-1)),
                .magnitude_out(mag_out[bin]),
                .valid_out(mag_valid_arr[bin]),
                .last_out(mag_last_arr[bin])
            );
        end
    endgenerate

    // Combine per-bin valid/last: all bins must be valid, last from highest bin
    logic mag_valid_all;
    always_comb begin
        mag_valid_all = 1'b1;
        for (int bin = 0; bin < MAX_ANGLE_BINS; bin++)
            mag_valid_all = mag_valid_all & mag_valid_arr[bin];
    end
    assign mag_valid = mag_valid_all;
    assign mag_last  = mag_last_arr[MAX_ANGLE_BINS-1];

    // =========================================================================
    // Pipeline stall / backpressure propagation
    // =========================================================================
    // When ra_ready is low, stall the output pipeline to prevent data loss.
    // Backpressure propagates to beamforming via beamforming_out_ready.
    // angle_fft_4pt and magnitude_unit are streaming-only (no ready ports),
    // so the stall gates the output register stage; the intermediate modules
    // recompute every cycle (combinational path from beamforming).
    wire pipeline_stall = ra_valid && !ra_ready;
    wire beamforming_out_ready = !pipeline_stall;

    // =========================================================================
    // Output Interface
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < MAX_ANGLE_BINS; i++) begin
                ra_mag[i] <= '0;
            end
            ra_range_idx <= '0;
            ra_angle_idx <= '0;
            ra_valid <= 1'b0;
            ra_last <= 1'b0;
        end else if (!pipeline_stall) begin
            ra_valid <= mag_valid;
            ra_last <= mag_last;

            // Output magnitude values
            for (int i = 0; i < MAX_ANGLE_BINS; i++) begin
                ra_mag[i] <= mag_out[i];
            end

            // Index counters
            ra_range_idx <= range_bin_idx;
            ra_angle_idx <= '0;  // Angle bin index handled by parallel output
        end
    end

    // Backpressure handling
    assign adc_ready = !fft_busy && (state == IDLE);

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    `ifndef VERILATOR
    RA_RANGE_SIZE: assert property (
        @(posedge clk) disable iff (!rst_n)
        range_fft_size >= 64 && range_fft_size <= MAX_RANGE_BINS
    ) else $error("ra_map_processor: Range FFT size out of range");

    RA_ANGLE_SIZE: assert property (
        @(posedge clk) disable iff (!rst_n)
        MAX_ANGLE_BINS == NUM_ANTENNAS
    ) else $error("ra_map_processor: Angle bins must equal antenna count");

    RA_PIPELINE: assert property (
        @(posedge clk) disable iff (!rst_n)
        adc_valid && adc_last |-> ##20 mag_valid
    ) else $error("ra_map_processor: Pipeline latency mismatch");
    `endif // VERILATOR
    // synthesis translate_on

endmodule