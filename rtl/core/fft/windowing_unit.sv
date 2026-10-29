// =============================================================================
// windowing_unit.sv
//
// Windowing functions for FFT input to reduce spectral leakage.
// Supports: Rectangular, Hamming, Hanning, Blackman windows.
//
// Window functions (N-point):
//   Rectangular: w[n] = 1
//   Hamming:     w[n] = 0.54 - 0.46*cos(2*pi*n/(N-1))
//   Hanning:     w[n] = 0.5 * (1 - cos(2*pi*n/(N-1)))
//   Blackman:    w[n] = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
//
// Pipeline: 2 stages (LUT lookup + multiplication)
// Latency: 3 cycles
// Throughput: 1 sample/cycle
// =============================================================================

import pkg_config::*;

module windowing_unit #(
    parameter int DATA_WIDTH    = 32,
    parameter int FRAC_WIDTH    = 16,
    parameter int MAX_FFT_SIZE  = 1024,
    parameter int WINDOW_ADDR_W = 10  // log2(MAX_FFT_SIZE)
)(
    input  logic clk,
    input  logic rst_n,

    // Configuration
    input  logic [1:0] window_type,      // 0=None, 1=Hamming, 2=Hanning, 3=Blackman
    input  logic [11:0] fft_size,         // Current FFT size (64, 128, 256, 512, 1024)
    input  logic        window_enable,    // Enable windowing

    // Input
    input  logic signed [DATA_WIDTH-1:0] data_re_in,
    input  logic signed [DATA_WIDTH-1:0] data_im_in,
    input  logic        valid_in,
    input  logic        last_in,          // Last sample of frame
    input  logic [11:0] sample_idx,       // Sample index in frame

    // Output
    output logic signed [DATA_WIDTH-1:0] data_re_out,
    output logic signed [DATA_WIDTH-1:0] data_im_out,
    output logic        valid_out,
    output logic        last_out
);

    // =========================================================================
    // Window Type Encoding
    // =========================================================================
    localparam bit [1:0] WIN_NONE     = 2'b00;
    localparam bit [1:0] WIN_HAMMING   = 2'b01;
    localparam bit [1:0] WIN_HANNING   = 2'b10;
    localparam bit [1:0] WIN_BLACKMAN  = 2'b11;

    // =========================================================================
    // Window Coefficient LUT (Q16.16 format — matches .mem file width)
    // =========================================================================
    // Pre-computed coefficients for each window type
    // Stored as: [window_type][sample_index]
    // Max 1024 samples, 4 window types

    (* rom_style = "block" *)
    logic signed [DATA_WIDTH-1:0] hamming_lut [0:MAX_FFT_SIZE-1];

    (* rom_style = "block" *)
    logic signed [DATA_WIDTH-1:0] hanning_lut [0:MAX_FFT_SIZE-1];

    (* rom_style = "block" *)
    logic signed [DATA_WIDTH-1:0] blackman_lut [0:MAX_FFT_SIZE-1];

    // Initialize LUTs using constant functions (latent: overflow at coeff=1.0 clamped)
    function automatic logic signed [DATA_WIDTH-1:0] calc_hamming(int n, int N);
        real coeff;
        begin
            if (N <= 1)
                coeff = 0.54;
            else
                coeff = 0.54 - 0.46 * $cos(2.0 * 3.14159265359 * real'(n) / real'(N-1));
            coeff = coeff > 1.0 ? 1.0 : coeff;
            calc_hamming = DATA_WIDTH'(coeff * real'(1 << FRAC_WIDTH));
        end
    endfunction

    function automatic logic signed [DATA_WIDTH-1:0] calc_hanning(int n, int N);
        real coeff;
        begin
            if (N <= 1)
                coeff = 0.5;
            else
                coeff = 0.5 * (1.0 - $cos(2.0 * 3.14159265359 * real'(n) / real'(N-1)));
            coeff = coeff > 1.0 ? 1.0 : coeff;
            calc_hanning = DATA_WIDTH'(coeff * real'(1 << FRAC_WIDTH));
        end
    endfunction

    function automatic logic signed [DATA_WIDTH-1:0] calc_blackman(int n, int N);
        real coeff;
        begin
            if (N <= 1)
                coeff = 0.42;
            else
                coeff = 0.42 - 0.5 * $cos(2.0 * 3.14159265359 * real'(n) / real'(N-1))
                            + 0.08 * $cos(4.0 * 3.14159265359 * real'(n) / real'(N-1));
            coeff = coeff > 1.0 ? 1.0 : coeff;
            calc_blackman = DATA_WIDTH'(coeff * real'(1 << FRAC_WIDTH));
        end
    endfunction

    // Initialize LUTs from precomputed ROM files (synthesizable via $readmemh)
    initial begin
        $readmemh("rtl/hamming_window.mem", hamming_lut);
        $readmemh("rtl/hanning_window.mem", hanning_lut);
        $readmemh("rtl/blackman_window.mem", blackman_lut);
    end

    // =========================================================================
    // Pipeline Stage 1: LUT Lookup
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] window_coeff_d1;
    logic signed [DATA_WIDTH-1:0] data_re_d1;
    logic signed [DATA_WIDTH-1:0] data_im_d1;
    logic valid_d1, last_d1;
    logic [1:0] window_type_d1;
    logic window_enable_d1;

    // Address calculation: scale index to MAX_FFT_SIZE
    // Intent: lut_addr = sample_idx * MAX_FFT_SIZE / fft_size
    // Since fft_size is always a power of 2, use shift: sample_idx << (log2(MAX) - log2(fft_size))
    logic [WINDOW_ADDR_W-1:0] lut_addr;
    logic [3:0] addr_shift;
    always_comb begin
        case (fft_size)
            12'd64:    addr_shift = 4'd4;
            12'd128:   addr_shift = 4'd3;
            12'd256:   addr_shift = 4'd2;
            12'd512:   addr_shift = 4'd1;
            12'd1024:  addr_shift = 4'd0;
            default:   addr_shift = 4'd0;
        endcase
    end
    assign lut_addr = WINDOW_ADDR_W'(sample_idx << addr_shift);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            window_coeff_d1 <= DATA_WIDTH'((1 << FRAC_WIDTH));  // 1.0 in Q16.16
            data_re_d1 <= '0;
            data_im_d1 <= '0;
            valid_d1 <= 1'b0;
            last_d1 <= 1'b0;
            window_type_d1 <= WIN_NONE;
            window_enable_d1 <= 1'b0;
        end else begin
            valid_d1 <= valid_in;
            last_d1 <= last_in;
            window_type_d1 <= window_type;
            window_enable_d1 <= window_enable;
            data_re_d1 <= data_re_in;
            data_im_d1 <= data_im_in;

            // Select window coefficient
            if (!window_enable) begin
                window_coeff_d1 <= DATA_WIDTH'((1 << FRAC_WIDTH));  // 1.0 (no window)
            end else begin
                case (window_type)
                    WIN_HAMMING:  window_coeff_d1 <= hamming_lut[lut_addr];
                    WIN_HANNING:  window_coeff_d1 <= hanning_lut[lut_addr];
                    WIN_BLACKMAN: window_coeff_d1 <= blackman_lut[lut_addr];
                    default:      window_coeff_d1 <= DATA_WIDTH'((1 << FRAC_WIDTH));  // 1.0
                endcase
            end
        end
    end

    // =========================================================================
    // Pipeline Stage 2: Multiplication
    // =========================================================================
    // Multiply data by window coefficient
    // data_out = data_in * window_coeff (Q16.16 * Q16.16 = Q32.32)

    logic signed [2*DATA_WIDTH-1:0] prod_re, prod_im;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_re_out <= '0;
            data_im_out <= '0;
            valid_out <= 1'b0;
            last_out <= 1'b0;
        end else begin
            // Signed multiplication
            prod_re = data_re_d1 * window_coeff_d1;
            prod_im = data_im_d1 * window_coeff_d1;

            // Shift back to original Q format
            // Q16.16 * Q16.16 = Q32.32, shift right by FRAC_WIDTH to get Q16.16
            data_re_out <= DATA_WIDTH'(prod_re >>> FRAC_WIDTH);
            data_im_out <= DATA_WIDTH'(prod_im >>> FRAC_WIDTH);

            valid_out <= valid_d1;
            last_out <= last_d1;
        end
    end

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    `ifndef VERILATOR
    WINDOW_SIZE: assert property (
        @(posedge clk) disable iff (!rst_n)
        valid_in |-> (fft_size >= 64 && fft_size <= MAX_FFT_SIZE)
    ) else $error("windowing_unit: FFT size out of range");

    WINDOW_PIPELINE: assert property (
        @(posedge clk) disable iff (!rst_n)
        valid_in && last_in |-> ##2 valid_out && last_out
    ) else $error("windowing_unit: Pipeline latency mismatch");
    `endif // VERILATOR
    // synthesis translate_on

endmodule