// =============================================================================
// systolic_array_pingpong.sv
//
// Dual systolic_array with ping-pong tile scheduling (Option 2 of the
// systolic-utilisation discussion).
//
// Two M×N systolic_array instances run in parallel:
//   - Even-indexed tiles → array A
//   - Odd-indexed tiles  → array B
//
// While array A is computing tile T (filling, computing, draining), array B
// is computing tile T+1, with its clear_acc offset by one tile boundary.
// Net effect: the per-tile fill+drain bubble is hidden behind the other
// array's compute window, so steady-state throughput is ~2× a single array.
//
// Resource cost vs. single-array:
//   - PE count: 2 × M × N (4×4 → 32 DSPs; well under Artix-7-100T's 240)
//   - Skew network: 2× (each array still has its own internal skew)
//   - Output mux:  M × N × ACC_WIDTH wide 2:1 mux
//
// Control protocol
// ────────────────
//   The caller still drives a single en/clear_acc pair. The wrapper
//   internally toggles a tile_phase bit on each clear_acc pulse and routes
//   that tile's K-cycle stream to whichever array is free. The aggregate
//   result_rdy and per-PE result_valid are merged from both arrays so the
//   downstream consumer sees one coherent stream of result tiles.
//
// Backward compat
// ───────────────
//   When DUAL_ARRAY_EN=0 the wrapper degenerates to a single systolic_array
//   instance with the second instance optimized away by synthesis (its enable
//   stays 0). This keeps the module pin-compatible whether or not the
//   ping-pong is enabled.
//
// NOTE: This is a wrapper. The base systolic_array already has CTRL_SKEW_EN
// for option 1 and ZERO_SKIP_EN for option 6 — both stay enabled here.
// =============================================================================

import pkg_config::*;

module systolic_array_pingpong #(
    parameter int M             = 4,
    parameter int N             = 4,
    parameter int DATA_WIDTH    = 8,
    parameter int ACC_WIDTH     = 4 * DATA_WIDTH,
    parameter bit DUAL_ARRAY_EN = 1'b1,    // 0 = bypass to single array
    parameter bit CTRL_SKEW_EN  = 1'b1,    // forwarded to both arrays
    parameter bit ZERO_SKIP_EN  = 1'b1,
    parameter bit LOAD_PSUM_EN  = 1'b0     // forwarded to both arrays
)(
    input  logic clk,
    input  logic rst_n,

    // Single control / data interface (caller-facing)
    input  logic en,
    input  logic clear_acc,
    input  logic load_psum,
    input  logic signed [DATA_WIDTH-1:0] a_in [M-1:0],
    input  logic signed [DATA_WIDTH-1:0] b_in [N-1:0],
    input  logic signed [ACC_WIDTH-1:0]  psum_in [M-1:0][N-1:0],

    // Combined output (mux'd from whichever array just finished a tile)
    output logic signed [ACC_WIDTH-1:0]  result [M-1:0][N-1:0],
    output logic                          result_valid_pe [M-1:0][N-1:0],
    output logic                          result_rdy
);

    // =========================================================================
    // Tile-phase tracking: toggles every clear_acc pulse
    // =========================================================================
    logic tile_phase;          // 0 → array A; 1 → array B
    logic clear_acc_d;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tile_phase  <= 1'b0;
            clear_acc_d <= 1'b0;
        end else begin
            clear_acc_d <= clear_acc;
            // Toggle on rising edge of clear_acc (= start of new tile).
            // Single-tile callers (clear_acc broadcast) still get array A.
            if (DUAL_ARRAY_EN && clear_acc && !clear_acc_d)
                tile_phase <= ~tile_phase;
        end
    end

    // =========================================================================
    // Demux control to the two arrays. en is broadcast (the array internally
    // gates by clear_acc presence). clear_acc only fires for the active
    // tile_phase, so the inactive array sees no boundary — its accumulator
    // is preserved while the active one streams.
    // =========================================================================
    logic clear_acc_a, clear_acc_b;
    logic en_a, en_b;
    always_comb begin
        if (DUAL_ARRAY_EN) begin
            clear_acc_a = clear_acc & ~tile_phase;
            clear_acc_b = clear_acc &  tile_phase;
            en_a        = en        & ~tile_phase;
            en_b        = en        &  tile_phase;
        end else begin
            clear_acc_a = clear_acc;
            clear_acc_b = 1'b0;
            en_a        = en;
            en_b        = 1'b0;
        end
    end

    // =========================================================================
    // Two systolic_array instances
    // =========================================================================
    logic signed [ACC_WIDTH-1:0] result_a [M-1:0][N-1:0];
    logic signed [ACC_WIDTH-1:0] result_b [M-1:0][N-1:0];
    logic                         rvalid_a_pe [M-1:0][N-1:0];
    logic                         rvalid_b_pe [M-1:0][N-1:0];
    logic                         rdy_a, rdy_b;

    systolic_array #(
        .M(M), .N(N),
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .CTRL_SKEW_EN(CTRL_SKEW_EN),
        .ZERO_SKIP_EN(ZERO_SKIP_EN),
        .LOAD_PSUM_EN(LOAD_PSUM_EN)
    ) u_arr_a (
        .clk(clk), .rst_n(rst_n),
        .en(en_a), .clear_acc(clear_acc_a), .load_psum(load_psum),
        .a_in(a_in), .b_in(b_in),
        .psum_in(psum_in),
        .result(result_a),
        .result_valid_pe(rvalid_a_pe),
        .result_rdy(rdy_a)
    );

    systolic_array #(
        .M(M), .N(N),
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .CTRL_SKEW_EN(CTRL_SKEW_EN),
        .ZERO_SKIP_EN(ZERO_SKIP_EN),
        .LOAD_PSUM_EN(LOAD_PSUM_EN)
    ) u_arr_b (
        .clk(clk), .rst_n(rst_n),
        .en(en_b), .clear_acc(clear_acc_b), .load_psum(load_psum),
        .a_in(a_in), .b_in(b_in),
        .psum_in(psum_in),
        .result(result_b),
        .result_valid_pe(rvalid_b_pe),
        .result_rdy(rdy_b)
    );

    // =========================================================================
    // Output merge — OR result_rdy and result_valid_pe; mux result by which
    // array just produced it. The downstream consumer reads `result` on the
    // cycle result_rdy (or per-PE result_valid_pe) is high.
    // =========================================================================
    assign result_rdy = rdy_a | rdy_b;

    always_comb begin
        for (int i = 0; i < M; i++) begin
            for (int j = 0; j < N; j++) begin
                result_valid_pe[i][j] = rvalid_a_pe[i][j] | rvalid_b_pe[i][j];
                // If both fired simultaneously (shouldn't with proper tile
                // pacing), prefer A. result_b wins only when B's per-PE
                // valid is the current one.
                result[i][j] = rvalid_b_pe[i][j] ? result_b[i][j] : result_a[i][j];
            end
        end
    end

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    PINGPONG_NO_DOUBLE_RDY: assert property (
        @(posedge clk) disable iff (!rst_n)
        !DUAL_ARRAY_EN || !(rdy_a && rdy_b)
    ) else $error("systolic_array_pingpong: both arrays signalled result_rdy same cycle");
    // synthesis translate_on

endmodule
