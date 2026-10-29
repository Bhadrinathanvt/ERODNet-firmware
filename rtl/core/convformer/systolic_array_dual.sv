// =============================================================================
//  systolic_array_dual.sv  —  Weight-Stationary Systolic Array with
//                               DSP48 INT8 Dual-MAC Packing
//
//  Computes two independent matrix products simultaneously:
//    C_lo[M×N] = A[M×K] · B_lo[K×N]   (weight_lo path)
//    C_hi[M×N] = A[M×K] · B_hi[K×N]   (weight_hi path)
//
//  Each PE packs two 8-bit weights into one DSP48, sharing a single
//  activation stream. This doubles throughput per DSP48 slice.
//
//  Data flow (same topology as systolic_array):
//    A (activations)   flows left → right  across each row
//    B_lo/B_hi (weights) flow top → bottom down each column
//    C_lo, C_hi         accumulate in-place at each PE
//
//  Timing: identical to systolic_array (3-stage PE pipeline + skew).
//    result_lo/hi[i][j] valid after  (i + j + K + 2)  clock cycles
//
//  Control protocol: same as systolic_array (en, clear_acc, load_psum).
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;

module systolic_array_dual #(
    parameter int M          = 4,
    parameter int N          = 4,
    parameter int DATA_WIDTH = pkg_module_config::SYSTOLIC_ARRAY_DATA_WIDTH,
    parameter int ACC_WIDTH  = 4 * DATA_WIDTH,
    parameter bit  LOAD_PSUM_EN = 1'b0   // Enable partial-sum load path
)(
    input  logic clk,
    input  logic rst_n,

    // Control (broadcast to all PEs)
    input  logic en,
    input  logic clear_acc,
    input  logic load_psum,

    // A matrix row inputs: a_in[i] carries row i, one element per cycle
    input  logic signed [DATA_WIDTH-1:0] a_in [M-1:0],

    // B matrix column inputs: two weight streams per column
    input  logic signed [DATA_WIDTH-1:0] b_lo_in [N-1:0],
    input  logic signed [DATA_WIDTH-1:0] b_hi_in [N-1:0],

    // Partial Sum inputs (two independent paths)
    input  logic signed [ACC_WIDTH-1:0]  psum_in_lo [M-1:0][N-1:0],
    input  logic signed [ACC_WIDTH-1:0]  psum_in_hi [M-1:0][N-1:0],

    // Results: two independent M×N output matrices
    output logic signed [ACC_WIDTH-1:0]  result_lo [M-1:0][N-1:0],
    output logic signed [ACC_WIDTH-1:0]  result_hi [M-1:0][N-1:0],

    output logic result_rdy
);

    // =========================================================================
    //  INPUT SKEW NETWORK (identical topology to systolic_array)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] a_skewed [M-1:0];
    logic signed [DATA_WIDTH-1:0] b_lo_skewed [N-1:0];
    logic signed [DATA_WIDTH-1:0] b_hi_skewed [N-1:0];
    logic load_psum_skewed [M-1:0][N-1:0];
    logic signed [ACC_WIDTH-1:0] psum_lo_skewed [M-1:0][N-1:0];
    logic signed [ACC_WIDTH-1:0] psum_hi_skewed [M-1:0][N-1:0];

    assign a_skewed[0]    = a_in[0];
    assign b_lo_skewed[0] = b_lo_in[0];
    assign b_hi_skewed[0] = b_hi_in[0];

    generate
        // A row skew (same as original)
        for (genvar i = 1; i < M; i++) begin : gen_a_skew
            logic signed [DATA_WIDTH-1:0] sr [0:i-1];

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) sr[0] <= '0;
                else        sr[0] <= a_in[i];
            end

            for (genvar s = 1; s < i; s++) begin : gen_a_sr
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) sr[s] <= '0;
                    else        sr[s] <= sr[s-1];
                end
            end

            assign a_skewed[i] = sr[i-1];
        end

        // B_lo column skew (same delay as original b_in)
        for (genvar j = 1; j < N; j++) begin : gen_b_lo_skew
            logic signed [DATA_WIDTH-1:0] sr [0:j-1];

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) sr[0] <= '0;
                else        sr[0] <= b_lo_in[j];
            end

            for (genvar s = 1; s < j; s++) begin : gen_b_lo_sr
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) sr[s] <= '0;
                    else        sr[s] <= sr[s-1];
                end
            end

            assign b_lo_skewed[j] = sr[j-1];
        end

        // B_hi column skew (same delay as B_lo — same column, same delay)
        for (genvar j = 1; j < N; j++) begin : gen_b_hi_skew
            logic signed [DATA_WIDTH-1:0] sr [0:j-1];

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) sr[0] <= '0;
                else        sr[0] <= b_hi_in[j];
            end

            for (genvar s = 1; s < j; s++) begin : gen_b_hi_sr
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) sr[s] <= '0;
                    else        sr[s] <= sr[s-1];
                end
            end

            assign b_hi_skewed[j] = sr[j-1];
        end

        // Psum skew (duplicated for lo and hi paths) — eliminated when LOAD_PSUM_EN=0
        if (LOAD_PSUM_EN) begin : gen_psum_skew
            for (genvar i = 0; i < M; i++) begin : gen_lp_row
                for (genvar j = 0; j < N; j++) begin : gen_lp_col
                    localparam int DELAY = i + j;

                    if (DELAY == 0) begin : gen_lp_nodelay
                        assign load_psum_skewed[i][j] = load_psum;
                        assign psum_lo_skewed[i][j]   = psum_in_lo[i][j];
                        assign psum_hi_skewed[i][j]    = psum_in_hi[i][j];

                    end else begin : gen_lp_delay
                        logic [DELAY-1:0] lp_sr;
                        logic signed [ACC_WIDTH-1:0] psum_lo_sr [0:DELAY-1];
                        logic signed [ACC_WIDTH-1:0] psum_hi_sr [0:DELAY-1];

                        always_ff @(posedge clk or negedge rst_n) begin
                            if (!rst_n) begin
                                lp_sr[0]     <= 1'b0;
                                psum_lo_sr[0] <= '0;
                                psum_hi_sr[0] <= '0;
                            end else begin
                                lp_sr[0]     <= load_psum;
                                psum_lo_sr[0] <= psum_in_lo[i][j];
                                psum_hi_sr[0] <= psum_in_hi[i][j];
                            end
                        end

                        for (genvar s = 1; s < DELAY; s++) begin : gen_lp_sr
                            always_ff @(posedge clk or negedge rst_n) begin
                                if (!rst_n) begin
                                    lp_sr[s]     <= 1'b0;
                                    psum_lo_sr[s] <= '0;
                                    psum_hi_sr[s] <= '0;
                                end else begin
                                    psum_lo_sr[s] <= psum_lo_sr[s-1];
                                    psum_hi_sr[s] <= psum_hi_sr[s-1];
                                    lp_sr[s]     <= lp_sr[s-1];
                                end
                            end
                        end

                        assign load_psum_skewed[i][j] = lp_sr[DELAY-1];
                        assign psum_lo_skewed[i][j]   = psum_lo_sr[DELAY-1];
                        assign psum_hi_skewed[i][j]    = psum_hi_sr[DELAY-1];
                    end
                end
            end
        end else begin : gen_no_psum_skew
            for (genvar i = 0; i < M; i++) begin : gen_lp_zero_row
                for (genvar j = 0; j < N; j++) begin : gen_lp_zero_col
                    assign load_psum_skewed[i][j] = 1'b0;
                    assign psum_lo_skewed[i][j]   = '0;
                    assign psum_hi_skewed[i][j]    = '0;
                end
            end
        end
    endgenerate

    // =========================================================================
    //  INTER-PE DATA BUSES
    //
    //  a_h[i][j]  : activation data, flows left → right (same as original)
    //  b_lo_v[i][j], b_hi_v[i][j] : two weight streams, flow top → bottom
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] a_h      [M-1:0][N:0];
    logic signed [DATA_WIDTH-1:0] b_lo_v   [M:0][N-1:0];
    logic signed [DATA_WIDTH-1:0] b_hi_v   [M:0][N-1:0];

    generate
        for (genvar i = 0; i < M; i++) begin : gen_a_h_init
            assign a_h[i][0] = a_skewed[i];
        end
        for (genvar j = 0; j < N; j++) begin : gen_b_v_init
            assign b_lo_v[0][j] = b_lo_skewed[j];
            assign b_hi_v[0][j] = b_hi_skewed[j];
        end
    endgenerate

    // =========================================================================
    //  PE ARRAY  —  M×N grid of PE_mac_pipelined_dual instances
    //
    //  Each PE processes two weights (lo and hi) per cycle against the
    //  shared activation, producing two independent accumulation results.
    // =========================================================================
    generate
        for (genvar i = 0; i < M; i++) begin : gen_pe_row
            for (genvar j = 0; j < N; j++) begin : gen_pe_col

                // Pass-through FFs for activation and both weight streams
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        a_h[i][j+1]    <= '0;
                        b_lo_v[i+1][j] <= '0;
                        b_hi_v[i+1][j] <= '0;
                    end else begin
                        a_h[i][j+1]    <= a_h[i][j];
                        b_lo_v[i+1][j] <= b_lo_v[i][j];
                        b_hi_v[i+1][j] <= b_hi_v[i][j];
                    end
                end

                // PE instantiation — dual-MAC
                PE_mac_pipelined_dual #(
                    .DATA_WIDTH    (DATA_WIDTH),
                    .IN_WIDTH      (DATA_WIDTH),
                    .ACC_WIDTH     (ACC_WIDTH),
                    .LOAD_PSUM_EN  (LOAD_PSUM_EN)
                ) u_pe_dual (
                    .clk       (clk),
                    .rst_n     (rst_n),
                    .en        (en),
                    .clear_acc (clear_acc),
                    .weight_lo (b_lo_v[i][j]),      // B_lo — flows top→down
                    .weight_hi (b_hi_v[i][j]),      // B_hi — flows top→down
                    .act       (a_h[i][j]),         // A    — flows left→right
                    .load_psum (load_psum_skewed[i][j]),
                    .psum_in_lo(psum_lo_skewed[i][j]),
                    .psum_in_hi(psum_hi_skewed[i][j]),
                    .accum_lo  (result_lo[i][j]),
                    .accum_hi  (result_hi[i][j])
                );

            end
        end
    endgenerate

    // =========================================================================
    //  RESULT READY GENERATION (identical timing to systolic_array)
    // =========================================================================
    localparam int FLUSH_DELAY = (M - 1) + (N - 1) + 2;

    logic en_d;
    logic [FLUSH_DELAY-1:0] flush_sr;
    logic en_falling;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            en_d       <= 1'b0;
            flush_sr   <= '0;
            result_rdy <= 1'b0;
        end else begin
            // Blocking: captures pre-NBA en_d before update
            // (Verilator processes NBAs in declaration order)
            en_falling = en_d & ~en;
            en_d <= en;
            flush_sr <= {flush_sr[FLUSH_DELAY-2:0], en_falling};
            result_rdy <= flush_sr[FLUSH_DELAY-1];
        end
    end

endmodule
