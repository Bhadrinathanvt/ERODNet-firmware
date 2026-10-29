// =============================================================================
//  systolic_array_sparse.sv
//
//  Sparsity-aware weight-stationary systolic array.
//
//  Architecture:
//    - M×N grid of PE_mac_pipelined instances (same as systolic_array)
//    - Per-PE zero-detection: gate PE `en` when either operand is zero,
//      skipping the MAC accumulation cycle
//    - Per-PE MAC skip counter for performance monitoring
//    - Dense-mode backward compatibility: when sparse_en_cfg=0, the array
//      produces bit-identical results to the original systolic_array
//
//  Sparse mode (sparse_en_cfg=1):
//    The original K-cycle en protocol is preserved. The skew network aligns
//    data exactly as in the dense array. At each PE, when both operands are
//    non-zero, the PE accumulates normally. When either operand is zero, the
//    PE's en is gated off — the multiplier and accumulator are idle, saving
//    dynamic power and tracking the skip count.
//
//    clear_acc always passes through ungated so the accumulator initializes
//    correctly even on the first cycle (which may have a zero operand).
//
//  Future extension (SPARSE_STREAM_EN parameter, when 1):
//    Activates the NZ encoder + intersection unit data path for compressed
//    streams. Requires separate skew alignment logic. Currently unused.
//
//  Control:
//    en/clear_acc follow the same protocol as systolic_array.sv.
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;
import pkg_sparse_config::*;

module systolic_array_sparse #(
    parameter int M          = 4,
    parameter int N          = 4,
    parameter int DATA_WIDTH = pkg_module_config::SYSTOLIC_ARRAY_DATA_WIDTH,
    parameter int ACC_WIDTH  = 4 * DATA_WIDTH,
    parameter bit  CTRL_SKEW_EN = 1'b1,
    parameter bit  SPARSE_EN    = pkg_config::EN_SPARSE_SYSTOLIC,  // Enable sparsity-aware mode
    parameter int  ELASTIC_FIFO_DEPTH = pkg_config::ELASTIC_FIFO_DEPTH,
    parameter bit  SPARSE_STREAM_EN = 1'b0,  // Future: compressed-stream path
    parameter bit  LOAD_PSUM_EN = 1'b0        // Enable partial-sum load path (area: +66 FFs/PE)
)(
    input  logic clk,
    input  logic rst_n,

    // ── Control (broadcast) ────────────────────────────────────────────────
    input  logic en,
    input  logic clear_acc,
    input  logic load_psum,

    // ── A matrix row inputs (dense) ─────────────────────────────────────────
    input  logic signed [DATA_WIDTH-1:0] a_in [M-1:0],

    // ── B matrix column inputs (dense) ──────────────────────────────────────
    input  logic signed [DATA_WIDTH-1:0] b_in [N-1:0],

    // ── Partial sum inputs ──────────────────────────────────────────────────
    input  logic signed [ACC_WIDTH-1:0]  psum_in [M-1:0][N-1:0],

    // ── Results ──────────────────────────────────────────────────────────────
    output logic signed [ACC_WIDTH-1:0]  result [M-1:0][N-1:0],
    output logic                          result_valid_pe [M-1:0][N-1:0],
    output logic                          result_rdy,
    input  logic                          result_ready,    // FIXED: consumer backpressure

    // ── Sparse configuration ─────────────────────────────────────────────────
    input  logic        sparse_en_cfg,      // Runtime enable/disable
    input  logic [1:0]  weight_mode_cfg,    // Future: weight encoding mode
    input  logic [1:0]  activ_mode_cfg,     // Future: activation encoding mode

    // ── Performance counters ─────────────────────────────────────────────────
    output logic [31:0] mac_skip_total,     // Total MAC cycles skipped
    output logic [31:0] mac_active_total    // Total MAC cycles executed
);

    // =========================================================================
    //  Input Skew Network (identical to systolic_array.sv)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] a_skewed [M-1:0];
    logic signed [DATA_WIDTH-1:0] b_skewed [N-1:0];
    logic en_skewed        [M-1:0][N-1:0];
    logic clear_acc_skewed [M-1:0][N-1:0];
    logic load_psum_skewed [M-1:0][N-1:0];
    logic signed [ACC_WIDTH-1:0] psum_skewed [M-1:0][N-1:0];

    assign a_skewed[0] = a_in[0];
    assign b_skewed[0] = b_in[0];

    generate
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

        for (genvar j = 1; j < N; j++) begin : gen_b_skew
            logic signed [DATA_WIDTH-1:0] sr [0:j-1];
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) sr[0] <= '0;
                else        sr[0] <= b_in[j];
            end
            for (genvar s = 1; s < j; s++) begin : gen_b_sr
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) sr[s] <= '0;
                    else        sr[s] <= sr[s-1];
                end
            end
            assign b_skewed[j] = sr[j-1];
        end

        // Control skew (same as original) + psum skew (conditional on LOAD_PSUM_EN)
        for (genvar i = 0; i < M; i++) begin : gen_ctrl_row
            for (genvar j = 0; j < N; j++) begin : gen_ctrl_col
                localparam int DELAY_CTRL = i + j;

                if (!CTRL_SKEW_EN || DELAY_CTRL == 0) begin : gen_ctrl_nodelay
                    assign en_skewed[i][j]        = en;
                    assign clear_acc_skewed[i][j] = clear_acc;
                    if (LOAD_PSUM_EN) begin : gen_lp_nodelay
                        assign load_psum_skewed[i][j] = load_psum;
                        assign psum_skewed[i][j]      = psum_in[i][j];
                    end else begin : gen_lp_nodelay_zero
                        assign load_psum_skewed[i][j] = 1'b0;
                        assign psum_skewed[i][j]       = '0;
                    end
                end else if (DELAY_CTRL == 1) begin : gen_ctrl_delay1
                    // Single-stage register: avoids the [DELAY_CTRL-2:0]=[-1:0]
                    // out-of-range slice when DELAY_CTRL collapses to 1.
                    logic en_sr;
                    logic clr_sr;
                    always_ff @(posedge clk or negedge rst_n) begin
                        if (!rst_n) begin
                            en_sr   <= 1'b0;
                            clr_sr  <= 1'b0;
                        end else begin
                            en_sr  <= en;
                            clr_sr <= clear_acc;
                        end
                    end
                    assign en_skewed[i][j]        = en_sr;
                    assign clear_acc_skewed[i][j] = clr_sr;

                    if (LOAD_PSUM_EN) begin : gen_lp_delay1
                        logic lp_sr;
                        logic signed [ACC_WIDTH-1:0] psum_sr;
                        always_ff @(posedge clk or negedge rst_n) begin
                            if (!rst_n) begin
                                lp_sr   <= 1'b0;
                                psum_sr <= '0;
                            end else begin
                                lp_sr   <= load_psum;
                                psum_sr <= psum_in[i][j];
                            end
                        end
                        assign load_psum_skewed[i][j] = lp_sr;
                        assign psum_skewed[i][j]      = psum_sr;
                    end else begin : gen_lp_delay1_zero
                        assign load_psum_skewed[i][j] = 1'b0;
                        assign psum_skewed[i][j]      = '0;
                    end
                end else begin : gen_ctrl_delay
                    logic [DELAY_CTRL-1:0] en_sr;
                    logic [DELAY_CTRL-1:0] clr_sr;
                    always_ff @(posedge clk or negedge rst_n) begin
                        if (!rst_n) begin
                            en_sr   <= '0;
                            clr_sr  <= '0;
                        end else begin
                            en_sr  <= {en_sr[DELAY_CTRL-2:0], en};
                            clr_sr <= {clr_sr[DELAY_CTRL-2:0], clear_acc};
                        end
                    end
                    assign en_skewed[i][j]        = en_sr[DELAY_CTRL-1];
                    assign clear_acc_skewed[i][j] = clr_sr[DELAY_CTRL-1];

                    if (LOAD_PSUM_EN) begin : gen_lp_delay
                        logic [DELAY_CTRL-1:0] lp_sr;
                        logic signed [ACC_WIDTH-1:0] psum_sr [0:DELAY_CTRL-1];
                        always_ff @(posedge clk or negedge rst_n) begin
                            if (!rst_n) begin
                                lp_sr   <= '0;
                                for (int s = 0; s < DELAY_CTRL; s++)
                                    psum_sr[s] <= '0;
                            end else begin
                                lp_sr  <= {lp_sr[DELAY_CTRL-2:0], load_psum};
                                psum_sr[0] <= psum_in[i][j];
                                for (int s = 1; s < DELAY_CTRL; s++)
                                    psum_sr[s] <= psum_sr[s-1];
                            end
                        end
                        assign load_psum_skewed[i][j] = lp_sr[DELAY_CTRL-1];
                        assign psum_skewed[i][j]      = psum_sr[DELAY_CTRL-1];
                    end else begin : gen_lp_delay_zero
                        assign load_psum_skewed[i][j] = 1'b0;
                        assign psum_skewed[i][j]       = '0;
                    end
                end
            end
        end
    endgenerate

    // =========================================================================
    //  Inter-PE Data Buses (same as original)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] a_h [M-1:0][N:0];
    logic signed [DATA_WIDTH-1:0] b_v [M:0][N-1:0];

    generate
        for (genvar i = 0; i < M; i++) begin : gen_a_h_init
            assign a_h[i][0] = a_skewed[i];
        end
        for (genvar j = 0; j < N; j++) begin : gen_b_v_init
            assign b_v[0][j] = b_skewed[j];
        end
    endgenerate

    // =========================================================================
    //  PE Array — Per-PE Zero-Detection + En Gating
    //
    //  When SPARSE_EN=1 and sparse_en_cfg=1:
    //    - Detect if both operands are non-zero (op_active)
    //    - Gate PE en: pe_en = en_skewed && op_active
    //    - clear_acc always passes through (PE initialization must work)
    //    - Track per-PE skip counts for performance monitoring
    //
    //  When SPARSE_EN=0 or sparse_en_cfg=0:
    //    - Operates identically to original systolic_array
    //    - Zero-detect still gates multiplier inputs (power saving)
    //    - No MAC skipping (en always passes through)
    // =========================================================================
    logic [31:0] mac_skip_pe [M-1:0][N-1:0];
    logic [31:0] mac_active_pe [M-1:0][N-1:0];

    generate
        for (genvar i = 0; i < M; i++) begin : gen_pe_row
            for (genvar j = 0; j < N; j++) begin : gen_pe_col

                // ── Pass-through FFs (1 cycle per hop) ────────────────────
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        a_h[i][j+1] <= '0;
                        b_v[i+1][j] <= '0;
                    end else begin
                        a_h[i][j+1] <= a_h[i][j];
                        b_v[i+1][j] <= b_v[i][j];
                    end
                end

                // ── Zero detection ──────────────────────────────────────────
                logic op_active;
                assign op_active = (a_h[i][j] != '0) && (b_v[i][j] != '0);

                // ── PE en gating ────────────────────────────────────────────
                // In sparse mode: gate en to skip MAC when either operand is zero
                // In dense mode: en passes through unchanged
                logic pe_en;
                if (SPARSE_EN) begin : gen_sparse_gate
                    // Sparse: gate en when either operand is zero.
                    // clear_acc has priority inside PE, so it still initializes
                    // the accumulator even when pe_en=0 on the first cycle.
                    assign pe_en = sparse_en_cfg ? (en_skewed[i][j] && op_active)
                                                 : en_skewed[i][j];
                end else begin : gen_dense_gate
                    assign pe_en = en_skewed[i][j];
                end

                // ── Operand gating (power saving, same as original) ────────
                logic signed [DATA_WIDTH-1:0] act_gated, w_gated;
                assign act_gated = op_active ? a_h[i][j] : '0;
                assign w_gated   = op_active ? b_v[i][j] : '0;

                // ── PE instantiation ────────────────────────────────────────
                PE_mac_pipelined #(
                    .DATA_W        (DATA_WIDTH),
                    .IN_W          (DATA_WIDTH),
                    .ACC_W         (ACC_WIDTH),
                    .LOAD_PSUM_EN (LOAD_PSUM_EN)
                ) u_pe (
                    .clk       (clk),
                    .rst_n     (rst_n),
                    .en        (pe_en),
                    .clear_acc (clear_acc_skewed[i][j]),
                    .weight    (w_gated),
                    .act       (act_gated),
                    .psum_in   (psum_skewed[i][j]),
                    .load_psum (load_psum_skewed[i][j]),
                    .accum     (result[i][j])
                );

                // ── Per-PE result_valid ─────────────────────────────────────
                logic en_skew_d;
                logic [1:0] rvsr;
                logic pe_falling;
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        en_skew_d <= 1'b0;
                        rvsr      <= 2'b00;
                    end else begin
                        // Blocking: captures pre-NBA en_skew_d before update
                        // (Verilator processes NBAs in declaration order)
                        pe_falling = en_skew_d & ~pe_en;
                        en_skew_d <= pe_en;
                        rvsr <= {rvsr[0], pe_falling};
                    end
                end
                assign result_valid_pe[i][j] = rvsr[1];

                // ── Per-PE performance counters ─────────────────────────────
                if (SPARSE_EN) begin : gen_perf_cnt
                    always_ff @(posedge clk or negedge rst_n) begin
                        if (!rst_n) begin
                            mac_skip_pe[i][j]   <= '0;
                            mac_active_pe[i][j] <= '0;
                        end else if (en_skewed[i][j]) begin
                            if (op_active)
                                mac_active_pe[i][j] <= mac_active_pe[i][j] + 1'b1;
                            else if (sparse_en_cfg)
                                mac_skip_pe[i][j] <= mac_skip_pe[i][j] + 1'b1;
                        end
                    end
                end else begin : gen_no_perf_cnt
                    assign mac_skip_pe[i][j]   = '0;
                    assign mac_active_pe[i][j] = '0;
                end

            end // gen_pe_col
        end // gen_pe_row
    endgenerate

    // =========================================================================
    //  Aggregate Performance Counters
    // =========================================================================
    always_comb begin
        mac_skip_total   = '0;
        mac_active_total = '0;
        for (int i = 0; i < M; i++) begin
            for (int j = 0; j < N; j++) begin
                mac_skip_total   += mac_skip_pe[i][j];
                mac_active_total += mac_active_pe[i][j];
            end
        end
    end

    // =========================================================================
    //  Result Ready Generation (same as original)
    // =========================================================================
    localparam int FLUSH_DELAY = (M - 1) + (N - 1) + 2;

    logic en_d;
    logic [FLUSH_DELAY-1:0] flush_sr;
    logic result_rdy_internal;
    logic en_falling;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            en_d      <= 1'b0;
            flush_sr  <= '0;
            result_rdy_internal <= 1'b0;
        end else begin
            // Blocking: captures pre-NBA en_d before Verilator NBA ordering
            // would clear it.
            en_falling = en_d & ~en;
            en_d <= en;
            flush_sr <= {flush_sr[FLUSH_DELAY-2:0], en_falling};
            result_rdy_internal <= flush_sr[FLUSH_DELAY-1];
        end
    end

    // FIXED: consumer backpressure — hold result_rdy asserted until consumer
    // accepts (result_ready=1). Without this, result_rdy pulses for one cycle
    // and the consumer may miss it.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            result_rdy <= 1'b0;
        else if (result_ready)
            result_rdy <= result_rdy_internal;
        else if (result_rdy_internal)
            result_rdy <= 1'b1;  // Latch until consumer accepts
    end

endmodule