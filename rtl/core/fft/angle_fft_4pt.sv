// =============================================================================
// angle_fft_4pt.sv
//
// Hardcoded 4-point DFT butterfly for Angle FFT.
// Optimized for AWR1843AOP's 4 RX antennas.
//
// 4-point DFT: X[k] = sum(x[n] * exp(-j*2*pi*k*n/4)) for n=0..3
//
// Twiddle factors:
//   W4^0 = 1          =  1 + j0
//   W4^1 = -j         =  0 - j1
//   W4^2 = -1         = -1 + j0
//   W4^3 = j          =  0 + j1
//
// Simplified butterfly:
//   X[0] = x[0] + x[1] + x[2] + x[3]
//   X[1] = x[0] - j*x[1] - x[2] + j*x[3]
//   X[2] = x[0] - x[1] + x[2] - x[3]
//   X[3] = x[0] + j*x[1] - x[2] - j*x[3]
//
// Pipeline: 2 stages (adder tree + output registration)
// Latency: 3 cycles
// Throughput: 1 output/cycle after initial fill
// =============================================================================

import pkg_config::*;

module angle_fft_4pt #(
    parameter int DATA_WIDTH = 32,
    parameter int FRAC_WIDTH = 16
)(
    input  logic clk,
    input  logic rst_n,

    // Input: 4 complex samples (from Range FFT)
    input  logic signed [DATA_WIDTH-1:0] x_re [3:0],  // Real parts
    input  logic signed [DATA_WIDTH-1:0] x_im [3:0],  // Imaginary parts
    input  logic        valid_in,
    input  logic        last_in,      // Last sample of chirp

    // Output: 4 complex frequency bins
    output logic signed [DATA_WIDTH-1:0] X_re [3:0],  // Real parts
    output logic signed [DATA_WIDTH-1:0] X_im [3:0],  // Imaginary parts
    output logic        valid_out,
    output logic        last_out
);

    // =========================================================================
    // Internal Signals
    // =========================================================================

    // Stage 1: Partial sums
    logic signed [DATA_WIDTH:0] s0_re, s0_im;  // x[0] + x[2]
    logic signed [DATA_WIDTH:0] s1_re, s1_im;  // x[0] - x[2]
    logic signed [DATA_WIDTH:0] s2_re, s2_im;  // x[1] + x[3]
    logic signed [DATA_WIDTH:0] s3_re, s3_im;  // x[1] - x[3]

    logic signed [DATA_WIDTH:0] d0_re, d0_im;  // x[0] - x[2]
    logic signed [DATA_WIDTH:0] d1_re, d1_im;  // x[1] - x[3]

    // Stage 1 registered
    logic signed [DATA_WIDTH:0] s0_re_d1, s0_im_d1;
    logic signed [DATA_WIDTH:0] s1_re_d1, s1_im_d1;
    logic signed [DATA_WIDTH:0] s2_re_d1, s2_im_d1;
    logic signed [DATA_WIDTH:0] s3_re_d1, s3_im_d1;
    logic valid_d1, last_d1;

    // Stage 2: Final outputs
    // X[0] = s0 + s2 = (x[0]+x[2]) + (x[1]+x[3])
    // X[1] = s1 - j*s3 = (x[0]-x[2]) - j*(x[1]-x[3]) => Re=s1, Im=-s3
    // X[2] = s0 - s2 = (x[0]+x[2]) - (x[1]+x[3])
    // X[3] = s1 + j*s3 = (x[0]-x[2]) + j*(x[1]-x[3]) => Re=s1, Im=s3

    // =========================================================================
    // Stage 1: Adder Tree
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s0_re_d1 <= '0;
            s0_im_d1 <= '0;
            s1_re_d1 <= '0;
            s1_im_d1 <= '0;
            s2_re_d1 <= '0;
            s2_im_d1 <= '0;
            s3_re_d1 <= '0;
            s3_im_d1 <= '0;
            valid_d1 <= 1'b0;
            last_d1 <= 1'b0;
        end else begin
            // s0 = x[0] + x[2]
            s0_re_d1 <= x_re[0] + x_re[2];
            s0_im_d1 <= x_im[0] + x_im[2];

            // s1 = x[0] - x[2]
            s1_re_d1 <= x_re[0] - x_re[2];
            s1_im_d1 <= x_im[0] - x_im[2];

            // s2 = x[1] + x[3]
            s2_re_d1 <= x_re[1] + x_re[3];
            s2_im_d1 <= x_im[1] + x_im[3];

            // s3 = x[1] - x[3]
            s3_re_d1 <= x_re[1] - x_re[3];
            s3_im_d1 <= x_im[1] - x_im[3];

            valid_d1 <= valid_in;
            last_d1 <= last_in;
        end
    end

    // =========================================================================
    // Stage 2: Final Butterfly Outputs
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 4; i++) begin
                X_re[i] <= '0;
                X_im[i] <= '0;
            end
            valid_out <= 1'b0;
            last_out <= 1'b0;
        end else begin
            // X[0] = s0 + s2 (all real additions)
            X_re[0] <= DATA_WIDTH'(s0_re_d1 + s2_re_d1);
            X_im[0] <= DATA_WIDTH'(s0_im_d1 + s2_im_d1);

            // X[1] = s1 - j*s3 => Re = s1_re, Im = -s3_im
            // Note: s1 is already complex, we need (s1_re + j*s1_im) - j*(s3_re + j*s3_im)
            //     = s1_re + j*s1_im - j*s3_re + s3_im
            //     = (s1_re + s3_im) + j*(s1_im - s3_re)
            X_re[1] <= DATA_WIDTH'(s1_re_d1 + s3_im_d1);  // s1_re + s3_im
            X_im[1] <= DATA_WIDTH'(s1_im_d1 - s3_re_d1);  // s1_im - s3_re

            // X[2] = s0 - s2
            X_re[2] <= DATA_WIDTH'(s0_re_d1 - s2_re_d1);
            X_im[2] <= DATA_WIDTH'(s0_im_d1 - s2_im_d1);

            // X[3] = s1 + j*s3 => Re = s1_re - s3_im, Im = s1_im + s3_re
            // (s1_re + j*s1_im) + j*(s3_re + j*s3_im)
            // = s1_re + j*s1_im + j*s3_re - s3_im
            // = (s1_re - s3_im) + j*(s1_im + s3_re)
            X_re[3] <= DATA_WIDTH'(s1_re_d1 - s3_im_d1);  // s1_re - s3_im
            X_im[3] <= DATA_WIDTH'(s1_im_d1 + s3_re_d1);  // s1_im + s3_re

            valid_out <= valid_d1;
            last_out <= last_d1;
        end
    end

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    `ifndef VERILATOR
    ANGLE_FFT_VALID: assert property (
        @(posedge clk) disable iff (!rst_n)
        valid_in |-> ##2 valid_out
    ) else $error("angle_fft_4pt: Pipeline latency mismatch");

    ANGLE_FFT_LAST: assert property (
        @(posedge clk) disable iff (!rst_n)
        valid_in && last_in |-> ##2 valid_out && last_out
    ) else $error("angle_fft_4pt: Last signal not propagated correctly");
    `endif // VERILATOR
    // synthesis translate_on

endmodule