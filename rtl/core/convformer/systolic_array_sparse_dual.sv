// =============================================================================
//  systolic_array_sparse_dual.sv
//
//  Sparsity-aware weight-stationary systolic array with dual-MAC DSP48 packing.
//
//  Combines:
//    - Dual-weight topology from systolic_array_dual (b_lo_in, b_hi_in)
//    - Per-PE zero-detect with per-accumulator gating from systolic_array_sparse
//    - PE_mac_pipelined_dual_sparse for independent lo/hi accumulation gating
//
//  Computes two independent matrix products with zero-skipping:
//    C_lo[M×N] = A[M×K] · B_lo[K×N]  (per-accumulator skip when weight_lo=0 or act=0)
//    C_hi[M×N] = A[M×K] · B_hi[K×N]  (per-accumulator skip when weight_hi=0 or act=0)
//
//  Sparse mode (sparse_en_cfg=1):
//    - PE-level en gating when BOTH weight_lo=0 AND weight_hi=0 (or act=0)
//    - Per-accumulator gating when only one weight is zero
//    - Clear_acc always passes through (PE initialization works correctly)
//    - Performance counters for skip/active per-PE
//
//  Dense mode (sparse_en_cfg=0):
//    - Bit-identical to systolic_array_dual (no zero-skipping)
//
//  Timing: identical to systolic_array_dual
//    result_lo/hi valid after (i + j + K + 2) clock cycles
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;
import pkg_sparse_config::*;

module systolic_array_sparse_dual #(
    parameter int M          = 4,
    parameter int N          = 4,
    parameter int DATA_WIDTH = pkg_module_config::SYSTOLIC_ARRAY_DATA_WIDTH,
    parameter int ACC_WIDTH  = 4 * DATA_WIDTH,
    parameter bit  LOAD_PSUM_EN = 1'b0,
    parameter bit  SPARSE_EN    = pkg_config::EN_SPARSE_SYSTOLIC
)(
    input  logic clk,
    input  logic rst_n,

    // ── Control (broadcast) ────────────────────────────────────────────────
    input  logic en,
    input  logic clear_acc,
    input  logic load_psum,

    // ── A matrix row inputs ────────────────────────────────────────────────
    input  logic signed [DATA_WIDTH-1:0] a_in [M-1:0],

    // ── B matrix column inputs (dual weight streams) ───────────────────────
    input  logic signed [DATA_WIDTH-1:0] b_lo_in [N-1:0],
    input  logic signed [DATA_WIDTH-1:0] b_hi_in [N-1:0],

    // ── Partial sum inputs (dual paths) ─────────────────────────────────────
    input  logic signed [ACC_WIDTH-1:0]  psum_in_lo [M-1:0][N-1:0],
    input  logic signed [ACC_WIDTH-1:0]  psum_in_hi [M-1:0][N-1:0],

    // ── Results (dual output matrices) ──────────────────────────────────────
    output logic signed [ACC_WIDTH-1:0]  result_lo [M-1:0][N-1:0],
    output logic signed [ACC_WIDTH-1:0]  result_hi [M-1:0][N-1:0],
    output logic                          result_valid_pe [M-1:0][N-1:0],
    output logic                          result_rdy,
    input  logic                          result_ready,

    // ── Sparse configuration ────────────────────────────────────────────────
    input  logic        sparse_en_cfg,
    input  logic [1:0]  weight_mode_cfg,
    input  logic [1:0]  activ_mode_cfg,

    // ── Performance counters ─────────────────────────────────────────────────
    output logic [31:0] mac_skip_total,
    output logic [31:0] mac_active_total
);

    // =========================================================================
    //  Input Skew Network (identical topology to systolic_array_dual)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] a_skewed [M-1:0];
    logic signed [DATA_WIDTH-1:0] b_lo_skewed [N-1:0];
    logic signed [DATA_WIDTH-1:0] b_hi_skewed [N-1:0];
    logic en_skewed        [M-1:0][N-1:0];
    logic clear_acc_skewed [M-1:0][N-1:0];
    logic load_psum_skewed [M-1:0][N-1:0];
    logic signed [ACC_WIDTH-1:0] psum_lo_skewed [M-1:0][N-1:0];
    logic signed [ACC_WIDTH-1:0] psum_hi_skewed [M-1:0][N-1:0];

    assign a_skewed[0]    = a_in[0];
    assign b_lo_skewed[0] = b_lo_in[0];
    assign b_hi_skewed[0] = b_hi_in[0];

    generate
        // A row skew
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

        // B_lo column skew
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

        // B_hi column skew
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

        // Control + psum skew
        for (genvar i = 0; i < M; i++) begin : gen_ctrl_row
            for (genvar j = 0; j < N; j++) begin : gen_ctrl_col
                localparam int DELAY_CTRL = i + j;

                if (DELAY_CTRL == 0) begin : gen_ctrl_nodelay
                    assign en_skewed[i][j]        = en;
                    assign clear_acc_skewed[i][j] = clear_acc;
                    assign load_psum_skewed[i][j] = (LOAD_PSUM_EN) ? load_psum : 1'b0;
                    assign psum_lo_skewed[i][j]   = (LOAD_PSUM_EN) ? psum_in_lo[i][j] : '0;
                    assign psum_hi_skewed[i][j]   = (LOAD_PSUM_EN) ? psum_in_hi[i][j] : '0;
                end else begin : gen_ctrl_delay
                    logic [DELAY_CTRL-1:0] en_sr, clr_sr;
                    always_ff @(posedge clk or negedge rst_n) begin
                        if (!rst_n) begin
                            en_sr  <= '0;
                            clr_sr <= '0;
                        end else begin
                            en_sr  <= {en_sr[DELAY_CTRL-2:0], en};
                            clr_sr <= {clr_sr[DELAY_CTRL-2:0], clear_acc};
                        end
                    end
                    assign en_skewed[i][j]        = en_sr[DELAY_CTRL-1];
                    assign clear_acc_skewed[i][j] = clr_sr[DELAY_CTRL-1];

                    if (LOAD_PSUM_EN) begin : gen_psum_delay
                        logic [DELAY_CTRL-1:0] lp_sr;
                        logic signed [ACC_WIDTH-1:0] psum_lo_sr [0:DELAY_CTRL-1];
                        logic signed [ACC_WIDTH-1:0] psum_hi_sr [0:DELAY_CTRL-1];
                        always_ff @(posedge clk or negedge rst_n) begin
                            if (!rst_n) begin
                                lp_sr <= '0;
                                for (int s = 0; s < DELAY_CTRL; s++) begin
                                    psum_lo_sr[s] <= '0;
                                    psum_hi_sr[s] <= '0;
                                end
                            end else begin
                                lp_sr <= {lp_sr[DELAY_CTRL-2:0], load_psum};
                                psum_lo_sr[0] <= psum_in_lo[i][j];
                                psum_hi_sr[0] <= psum_in_hi[i][j];
                                for (int s = 1; s < DELAY_CTRL; s++) begin
                                    psum_lo_sr[s] <= psum_lo_sr[s-1];
                                    psum_hi_sr[s] <= psum_hi_sr[s-1];
                                end
                            end
                        end
                        assign load_psum_skewed[i][j] = lp_sr[DELAY_CTRL-1];
                        assign psum_lo_skewed[i][j]   = psum_lo_sr[DELAY_CTRL-1];
                        assign psum_hi_skewed[i][j]   = psum_hi_sr[DELAY_CTRL-1];
                    end else begin : gen_no_psum_delay
                        assign load_psum_skewed[i][j] = 1'b0;
                        assign psum_lo_skewed[i][j]   = '0;
                        assign psum_hi_skewed[i][j]   = '0;
                    end
                end
            end
        end
    endgenerate

    // =========================================================================
    //  Inter-PE Data Buses
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] a_h    [M-1:0][N:0];
    logic signed [DATA_WIDTH-1:0] b_lo_v [M:0][N-1:0];
    logic signed [DATA_WIDTH-1:0] b_hi_v [M:0][N-1:0];

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
    //  Per-PE Performance Counter Storage
    // =========================================================================
    logic [31:0] mac_skip_pe   [M-1:0][N-1:0];
    logic [31:0] mac_active_pe [M-1:0][N-1:0];

    // =========================================================================
    //  PE Array — M×N grid
    // =========================================================================
    generate
        for (genvar i = 0; i < M; i++) begin : gen_pe_row
            for (genvar j = 0; j < N; j++) begin : gen_pe_col

                // ── Pass-through FFs (1 cycle per hop) ──────────────────────
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

                // ── Zero detection (dual-path) ──────────────────────────────
                // op_active_lo = act != 0 && weight_lo != 0
                // op_active_hi = act != 0 && weight_hi != 0
                logic op_active_lo, op_active_hi;
                logic any_active;
                assign op_active_lo = (a_h[i][j] != '0) && (b_lo_v[i][j] != '0);
                assign op_active_hi = (a_h[i][j] != '0) && (b_hi_v[i][j] != '0);
                assign any_active   = op_active_lo || op_active_hi;

                // ── PE en gating ────────────────────────────────────────────
                // Gate PE en when BOTH paths are inactive (all operands zero)
                // clear_acc always passes through (PE initialization)
                logic pe_en;
                if (SPARSE_EN) begin : gen_sparse_gate
                    assign pe_en = sparse_en_cfg ? (en_skewed[i][j] && any_active)
                                                 : en_skewed[i][j];
                end else begin : gen_dense_gate
                    assign pe_en = en_skewed[i][j];
                end

                // ── Per-accumulator enables ─────────────────────────────────
                // Gate each accumulator independently when its operand is zero
                logic acc_lo_en, acc_hi_en;
                assign acc_lo_en = sparse_en_cfg ? op_active_lo : 1'b1;
                assign acc_hi_en = sparse_en_cfg ? op_active_hi : 1'b1;

                // ── Operand gating (power saving) ──────────────────────────
                logic signed [DATA_WIDTH-1:0] act_gated;
                logic signed [DATA_WIDTH-1:0] w_lo_gated, w_hi_gated;
                assign act_gated  = any_active ? a_h[i][j]    : '0;
                assign w_lo_gated = any_active ? b_lo_v[i][j] : '0;
                assign w_hi_gated = any_active ? b_hi_v[i][j] : '0;

                // ── PE instantiation (dual-MAC with per-accumulator sparse) ─
                PE_mac_pipelined_dual_sparse #(
                    .DATA_WIDTH   (DATA_WIDTH),
                    .IN_WIDTH     (DATA_WIDTH),
                    .ACC_WIDTH    (ACC_WIDTH),
                    .LOAD_PSUM_EN (LOAD_PSUM_EN)
                ) u_pe (
                    .clk       (clk),
                    .rst_n     (rst_n),
                    .en        (pe_en),
                    .clear_acc (clear_acc_skewed[i][j]),
                    .load_psum (load_psum_skewed[i][j]),
                    .sparse_en (sparse_en_cfg),
                    .acc_lo_en (acc_lo_en),
                    .acc_hi_en (acc_hi_en),
                    .weight_lo (w_lo_gated),
                    .weight_hi (w_hi_gated),
                    .act       (act_gated),
                    .psum_in_lo(psum_lo_skewed[i][j]),
                    .psum_in_hi(psum_hi_skewed[i][j]),
                    .accum_lo  (result_lo[i][j]),
                    .accum_hi  (result_hi[i][j])
                );

                // ── Per-PE result_valid (same as sparse array) ──────────────
                logic en_skew_d;
                logic [1:0] rvsr;
                logic pe_falling;
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        en_skew_d <= 1'b0;
                        rvsr      <= 2'b00;
                    end else begin
                        // Blocking: captures pre-NBA en_skew_d before update
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
                            if (any_active)
                                mac_active_pe[i][j] <= mac_active_pe[i][j] + 1'b1;
                            else if (sparse_en_cfg)
                                mac_skip_pe[i][j] <= mac_skip_pe[i][j] + 1'b1;
                        end
                    end
                end else begin : gen_no_perf_cnt
                    assign mac_skip_pe[i][j]   = '0;
                    assign mac_active_pe[i][j] = '0;
                end

            end
        end
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
    //  Result Ready Generation (same timing as systolic_array_dual)
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
            // Blocking: captures pre-NBA en_d before update
            // (Verilator processes NBAs in declaration order)
            en_falling = en_d & ~en;
            en_d <= en;
            flush_sr <= {flush_sr[FLUSH_DELAY-2:0], en_falling};
            result_rdy_internal <= flush_sr[FLUSH_DELAY-1];
        end
    end

    // Backpressure: hold result_rdy asserted until consumer accepts
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            result_rdy <= 1'b0;
        else if (result_ready)
            result_rdy <= result_rdy_internal;
        else if (result_rdy_internal)
            result_rdy <= 1'b1;
    end

endmodule
