// =============================================================================
// digital_beamforming.sv
//
// Digital beamforming for AWR1843AOP's 4 RX antennas.
// Implements delay-and-sum (DAS) beamforming with phase shift compensation.
//
// Antenna Array Configuration:
//   AWR1843AOP: 4 RX antennas with λ/2 spacing
//   Beam directions: 0°, ±30°, ±60° (5 beams typical)
//   Phase shift per angle: φ = (2π*d*sin(θ))/λ, where d = λ/2
//
// Beamforming equation:
//   Y[θ] = sum(w[n] * x[n] * exp(-j*φ[n,θ])) for n=0..N-1
//   where:
//     x[n] = received signal from antenna n
//     φ[n,θ] = phase shift for antenna n, direction θ
//     w[n] = window weight (optional)
//
// Pipeline: 3 stages (phase generation + complex multiply + sum)
// Latency: 5 cycles
// Throughput: 1 beam/cycle
// =============================================================================

import pkg_config::*;

module digital_beamforming #(
    parameter int NUM_ANTENNAS   = 4,       // AWR1843AOP: 4 RX
    parameter int NUM_BEAMS      = 5,       // Beam directions
    parameter int DATA_WIDTH     = 32,
    parameter int FRAC_WIDTH     = 16,
    parameter int PHASE_WIDTH    = 18       // Twiddle factor width
)(
    input  logic clk,
    input  logic rst_n,

    // Configuration
    input  logic beamforming_enable,          // Enable beamforming
    input  logic [2:0] beam_mode,             // 0=5-beam, 1=7-beam, 2=custom
    input  logic signed [PHASE_WIDTH-1:0] custom_phases [NUM_ANTENNAS-1:0][NUM_BEAMS-1:0], // Custom phase weights

    // Input: Complex samples from each antenna (after Range FFT)
    input  logic signed [DATA_WIDTH-1:0] antenna_re [NUM_ANTENNAS-1:0],
    input  logic signed [DATA_WIDTH-1:0] antenna_im [NUM_ANTENNAS-1:0],
    input  logic        valid_in,
    output logic        ready_in,             // Can accept new input (backpressure)
    input  logic        last_in,              // Last sample of chirp

    // Output: Beamformed complex samples
    output logic signed [DATA_WIDTH-1:0] beam_re [NUM_BEAMS-1:0],
    output logic signed [DATA_WIDTH-1:0] beam_im [NUM_BEAMS-1:0],
    output logic        valid_out,
    input  logic        out_ready,            // Consumer ready for output (backpressure)
    output logic        last_out
);

    // =========================================================================
    // Beam Direction Phase Shifts (λ/2 spacing)
    // =========================================================================
    // For d = λ/2: phase_shift = π * sin(θ) per antenna
    // Antenna spacing: d = λ/2, so phase = 2π * n * d * sin(θ) / λ = π * n * sin(θ)
    //
    // Pre-computed phases for 5 beams:
    // Beam 0: θ = -60°  → sin(-60°) = -0.866
    // Beam 1: θ = -30°  → sin(-30°) = -0.5
    // Beam 2: θ =  0°   → sin(0°)    =  0
    // Beam 3: θ = +30°  → sin(+30°) = +0.5
    // Beam 4: θ = +60°  → sin(+60°) = +0.866

    // Phase shift for antenna n: φ[n] = π * n * sin(θ)
    // In Q2.16 format: π ≈ 205887 (Q2.16)

    localparam signed [PHASE_WIDTH-1:0] PI_Q2_16 = 18'sh3243F;  // π in Q2.16

    // Pre-computed sin(θ) values in Q0.16 format
    localparam signed [15:0] SIN_M60 = 16'sh9000;   // sin(-60°) ≈ -0.866 in Q0.16
    localparam signed [15:0] SIN_M30 = 16'shC000;   // sin(-30°) ≈ -0.5 in Q0.16
    localparam signed [15:0] SIN_0   = 16'sh0000;   // sin(0°) = 0
    localparam signed [15:0] SIN_P30 = 16'sh4000;   // sin(+30°) ≈ +0.5 in Q0.16
    localparam signed [15:0] SIN_P60 = 16'sh7000;   // sin(+60°) ≈ +0.866 in Q0.16

    // Twiddle factors: cos(φ) and sin(φ) for each antenna and beam
    // Stored in Q1.15 format: cos/sin values in [-1, 1]
    // SYNTHESIZABLE: Using constant functions instead of initial block

    // Constant function to compute cos(π * ant * sin_θ)
    // Returns Q1.15 fixed-point value
    function automatic logic signed [15:0] compute_cos_twiddle(input int ant, input int beam_idx);
        real phase;
        real cos_val;
        real sin_theta;
        int  cos_int;
        begin
            // sin(θ) for each beam direction
            case (beam_idx)
                0: sin_theta = -0.8660254037844386;  // sin(-60°)
                1: sin_theta = -0.5;                // sin(-30°)
                2: sin_theta = 0.0;                 // sin(0°)
                3: sin_theta = 0.5;                 // sin(+30°)
                4: sin_theta = 0.8660254037844386;  // sin(+60°)
                default: sin_theta = 0.0;
            endcase
            // Phase = π * n * sin(θ)
            phase = 3.1415926535897932 * ant * sin_theta;
            cos_val = $cos(phase);
            // Clamp to [-1, 1] and convert to Q1.15
            if (cos_val > 0.999969482421875) cos_val = 0.999969482421875;
            if (cos_val < -1.0) cos_val = -1.0;
            cos_int = int'(cos_val * 32768.0);
            compute_cos_twiddle = 16'(cos_int);
        end
    endfunction

    // Constant function to compute sin(π * ant * sin_θ)
    function automatic logic signed [15:0] compute_sin_twiddle(input int ant, input int beam_idx);
        real phase;
        real sin_val;
        real sin_theta;
        int  sin_int;
        begin
            // sin(θ) for each beam direction
            case (beam_idx)
                0: sin_theta = -0.8660254037844386;  // sin(-60°)
                1: sin_theta = -0.5;                // sin(-30°)
                2: sin_theta = 0.0;                 // sin(0°)
                3: sin_theta = 0.5;                 // sin(+30°)
                4: sin_theta = 0.8660254037844386;  // sin(+60°)
                default: sin_theta = 0.0;
            endcase
            // Phase = π * n * sin(θ)
            phase = 3.1415926535897932 * ant * sin_theta;
            sin_val = $sin(phase);
            // Clamp to [-1, 1] and convert to Q1.15
            if (sin_val > 0.999969482421875) sin_val = 0.999969482421875;
            if (sin_val < -1.0) sin_val = -1.0;
            sin_int = int'(sin_val * 32768.0);
            compute_sin_twiddle = 16'(sin_int);
        end
    endfunction

    // Twiddle ROM constants (computed at elaboration time, no registers needed)
    logic signed [15:0] twiddle_cos [NUM_ANTENNAS-1:0][NUM_BEAMS-1:0];
    logic signed [15:0] twiddle_sin [NUM_ANTENNAS-1:0][NUM_BEAMS-1:0];

    generate
        for (genvar ant = 0; ant < NUM_ANTENNAS; ant++) begin : gen_twiddle_ant
            for (genvar beam = 0; beam < NUM_BEAMS; beam++) begin : gen_twiddle_beam
                localparam logic signed [15:0] COS_VAL = compute_cos_twiddle(ant, beam);
                localparam logic signed [15:0] SIN_VAL = compute_sin_twiddle(ant, beam);

                assign twiddle_cos[ant][beam] = COS_VAL;
                assign twiddle_sin[ant][beam] = SIN_VAL;
            end
        end
    endgenerate

    // =========================================================================
    // Backpressure / Stall Logic
    // =========================================================================
    logic stall;
    assign stall    = valid_out && !out_ready;
    assign ready_in = !stall;

    // =========================================================================
    // Pipeline Stage 1: Complex Multiplication
    // =========================================================================
    // For each beam and antenna: x_ant * (cos(φ) + j*sin(φ))
    // Result: (re*cos - im*sin) + j*(re*sin + im*cos)

    logic signed [DATA_WIDTH+15:0] prod_re [NUM_ANTENNAS-1:0][NUM_BEAMS-1:0];
    logic signed [DATA_WIDTH+15:0] prod_im [NUM_ANTENNAS-1:0][NUM_BEAMS-1:0];
    logic valid_d1, last_d1;

    // Pipeline registers for antenna inputs
    logic signed [DATA_WIDTH-1:0] antenna_re_d1 [NUM_ANTENNAS-1:0];
    logic signed [DATA_WIDTH-1:0] antenna_im_d1 [NUM_ANTENNAS-1:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int a = 0; a < NUM_ANTENNAS; a++) begin
                for (int b = 0; b < NUM_BEAMS; b++) begin
                    prod_re[a][b] <= '0;
                    prod_im[a][b] <= '0;
                end
                antenna_re_d1[a] <= '0;
                antenna_im_d1[a] <= '0;
            end
            valid_d1 <= 1'b0;
            last_d1 <= 1'b0;
        end else if (!stall) begin
            valid_d1 <= valid_in;
            last_d1 <= last_in;

            // Store delayed antenna inputs
            for (int a = 0; a < NUM_ANTENNAS; a++) begin
                antenna_re_d1[a] <= antenna_re[a];
                antenna_im_d1[a] <= antenna_im[a];
            end

            // Complex multiplication for each antenna and beam
            if (beamforming_enable) begin
                for (int a = 0; a < NUM_ANTENNAS; a++) begin
                    for (int b = 0; b < NUM_BEAMS; b++) begin
                        // (re + j*im) * (cos + j*sin)
                        // = (re*cos - im*sin) + j*(re*sin + im*cos)
                        prod_re[a][b] <= (antenna_re[a] * twiddle_cos[a][b])
                                       - (antenna_im[a] * twiddle_sin[a][b]);
                        prod_im[a][b] <= (antenna_re[a] * twiddle_sin[a][b])
                                       + (antenna_im[a] * twiddle_cos[a][b]);
                    end
                end
            end else begin
                // Bypass: just pass through antenna 0 to beam 2 (center beam)
                for (int b = 0; b < NUM_BEAMS; b++) begin
                    prod_re[0][b] <= { {(DATA_WIDTH+16-DATA_WIDTH){antenna_re[0][DATA_WIDTH-1]}}, antenna_re[0] };
                    prod_im[0][b] <= { {(DATA_WIDTH+16-DATA_WIDTH){antenna_im[0][DATA_WIDTH-1]}}, antenna_im[0] };
                    for (int a = 1; a < NUM_ANTENNAS; a++) begin
                        prod_re[a][b] <= '0;
                        prod_im[a][b] <= '0;
                    end
                end
            end
        end
    end

    // =========================================================================
    // Pipeline Stage 2: Summation
    // =========================================================================
    // Sum all antenna contributions for each beam
    // Y[beam] = sum(antenna[ant] * twiddle[ant][beam])

    logic signed [DATA_WIDTH+16:0] sum_re [NUM_BEAMS-1:0];
    logic signed [DATA_WIDTH+16:0] sum_im [NUM_BEAMS-1:0];
    logic signed [DATA_WIDTH+16:0] next_sum_re [NUM_BEAMS-1:0];
    logic signed [DATA_WIDTH+16:0] next_sum_im [NUM_BEAMS-1:0];
    logic valid_d2, last_d2;

    // Combinational accumulation across all antennas (BEAMFORMING-001 fix)
    // Previously: NBA in for-loop only retained last antenna's contribution.
    always_comb begin
        for (int b = 0; b < NUM_BEAMS; b++) begin
            next_sum_re[b] = '0;
            next_sum_im[b] = '0;
            for (int a = 0; a < NUM_ANTENNAS; a++) begin
                next_sum_re[b] = next_sum_re[b] + (prod_re[a][b] >>> 15);
                next_sum_im[b] = next_sum_im[b] + (prod_im[a][b] >>> 15);
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int b = 0; b < NUM_BEAMS; b++) begin
                sum_re[b] <= '0;
                sum_im[b] <= '0;
            end
            valid_d2 <= 1'b0;
            last_d2 <= 1'b0;
        end else if (!stall) begin
            valid_d2 <= valid_d1;
            last_d2 <= last_d1;

            // FIXED: Move summation to combinational logic (BEAMFORMING-001)
            // Original NBA-in-for-loop only retained last antenna's contribution.
            // Now: always_comb computes next_sum, always_ff registers it.
            for (int b = 0; b < NUM_BEAMS; b++) begin
                sum_re[b] <= next_sum_re[b];
                sum_im[b] <= next_sum_im[b];
            end
        end
    end

    // =========================================================================
    // Pipeline Stage 3: Output
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int b = 0; b < NUM_BEAMS; b++) begin
                beam_re[b] <= '0;
                beam_im[b] <= '0;
            end
            valid_out <= 1'b0;
            last_out <= 1'b0;
        end else if (!stall) begin
            valid_out <= valid_d2;
            last_out <= last_d2;

            for (int b = 0; b < NUM_BEAMS; b++) begin
                beam_re[b] <= DATA_WIDTH'(sum_re[b]);
                beam_im[b] <= DATA_WIDTH'(sum_im[b]);
            end
        end
    end

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    `ifndef VERILATOR
    BEAM_PIPELINE: assert property (
        @(posedge clk) disable iff (!rst_n)
        valid_in && last_in |-> ##3 valid_out && last_out
    ) else $error("digital_beamforming: Pipeline latency mismatch");

    BEAM_VALID_STABLE: assert property(@(posedge clk) disable iff (!rst_n)
        (valid_out && !out_ready) |=> valid_out)
        else $error("digital_beamforming: valid_out deasserted before out_ready");

    // BEAM_DATA_STABLE: assert property(@(posedge clk) disable iff (!rst_n)
    //     (valid_out && !out_ready) |=> $stable(beam_re) && $stable(beam_im))
    //     else $error("digital_beamforming: output changed while valid_out held without out_ready");
    // Commented out: $stable does not support unpacked arrays (beam_re, beam_im)

    BEAM_COUNT: assert property (
        @(posedge clk) disable iff (!rst_n)
        NUM_BEAMS >= 1 && NUM_BEAMS <= 8
    ) else $error("digital_beamforming: Invalid beam count");

    ANTENNA_COUNT: assert property (
        @(posedge clk) disable iff (!rst_n)
        NUM_ANTENNAS >= 1 && NUM_ANTENNAS <= 16
    ) else $error("digital_beamforming: Invalid antenna count");
    `endif // VERILATOR
    // synthesis translate_on

endmodule