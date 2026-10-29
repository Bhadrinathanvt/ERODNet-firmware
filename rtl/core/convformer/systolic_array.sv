// =============================================================================
//  systolic_array.sv  —  Weight-Stationary Output-Stationary Systolic Array
//  Computes  C[M×N] = A[M×K] · B[K×N],  streaming one element per lane per cycle.
//
//  Data flow
//  ─────────
//    A (activations)  flows left → right  across each row
//    B (weights)      flows top  → bottom down each column
//    C (result)       accumulates in-place at each PE; no data exits the array
//
//  Input skew network (built-in)
//  ─────────────────────────────
//    a_in[i] is delayed i cycles before entering the array.
//    b_in[j] is delayed j cycles before entering the array.
//    This "wavefront skew" guarantees that A[i,k] and B[k,j] arrive at
//    PE[i][j] on the exact same clock cycle for every valid (i,j,k) triple.
//
//  Timing from first valid input arriving at port
//  ───────────────────────────────────────────────
//    result[i][j] valid after  (i + j + K + 2)  clock cycles
//    where K = inner dimension, 2 = PE_mac_pipelined's internal pipeline depth.
//    Corner PE[0][0] is first; PE[M-1][N-1] is last (+M+N-2 extra cycles).
//
//  Control protocol for one computation (K-element dot product)
//  ─────────────────────────────────────────────────────────────
//    Cycle  0      : assert clear_acc=1, en=1  — pipeline starts
//    Cycles 1..K-1 : assert clear_acc=0, en=1  — stream remaining elements
//    Cycle  K+     : de-assert en
//
//    en and clear_acc are broadcast to ALL PEs.  Because each PE receives
//    zero-valued data before its skewed input arrives, early accumulation of
//    0×0=0 is harmless and the final result is always correct.
//
//    For back-to-back multiplications: re-assert clear_acc at least 2 cycles
//    before the first valid element of the next batch enters the array.
//
//  Synthesis Directives
//  ─────────────────────
//    - DSP inference forced via PE_mac_pipelined (* use_dsp = "yes" *)
//    - Skew registers use SRL inference for area efficiency
//    - Array PEs are fully pipelined for timing closure
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;

module systolic_array #(
    parameter int M          = 4,               // Output rows  /  A rows
    parameter int N          = 4,               // Output cols  /  B cols
    parameter int DATA_WIDTH = pkg_module_config::SYSTOLIC_ARRAY_DATA_WIDTH,
    parameter int ACC_WIDTH  = 4 * DATA_WIDTH,  // Must match PE_mac_pipelined
    // Option 1: skew en/clear_acc per PE so back-to-back tiles run without
    // a fill+drain bubble. Default enabled — costs Σ(i+j)≈M*N adders worth
    // of LUT/FF skew registers (24 FFs at M=N=4) and gives ~1.6× speedup
    // for small K. Set to 0 to restore broadcast control.
    parameter bit  CTRL_SKEW_EN = 1'b1,
    // Option 6: gate the multiplier when either operand is zero. Saves
    // dynamic switching power on the DSP/LUT multiplier and is a no-op
    // for correctness (mult by 0 is 0). Free at synth time.
    parameter bit  ZERO_SKIP_EN = 1'b1,
    // Option 7: partial-sum load path. When 0 (default), the psum_in port and
    // load_psum skew network are eliminated, saving ACC_WIDTH*Σ(i+j) FFs
    // per instance plus 2 FFs per PE. Set to 1 only if external partial sums
    // are needed (currently unused in convformer — always tied to 0).
    parameter bit  LOAD_PSUM_EN = 1'b0
)(
    input  logic clk,
    input  logic rst_n,

    // ── Control (broadcast — internally skewed when CTRL_SKEW_EN=1) ──────
    input  logic en,          // Hold high for K cycles per dot-product
    input  logic clear_acc,   // Pulse high 1 cycle at the start of each computation
    input  logic load_psum,    // Command to load a partial sum (active only when LOAD_PSUM_EN=1)


    // ── A matrix row inputs: a_in[i] carries row i, one element per cycle ───
    input  logic signed [DATA_WIDTH-1:0] a_in [M-1:0],

    // ── B matrix column inputs: b_in[j] carries column j, one element/cycle ─
    input  logic signed [DATA_WIDTH-1:0] b_in [N-1:0],

    // ── Partial Sum inputs (active only when LOAD_PSUM_EN=1) ────────────────
    input  logic signed [ACC_WIDTH-1:0]  psum_in [M-1:0][N-1:0],

    // ── Results: result[i][j] = dot(A row i, B col j) ───────────────────────
    output logic signed [ACC_WIDTH-1:0]  result [M-1:0][N-1:0],

    // ── Option 3: per-PE result_valid pulse (one cycle, fires when PE[i][j]
    //   completes its current K-element dot product). Lets a downstream
    //   consumer drain results as they finish instead of waiting on the
    //   aggregate result_rdy. Fires at cycle (i+j) + K + 2 after clear_acc
    //   when CTRL_SKEW_EN=1.
    output logic                          result_valid_pe [M-1:0][N-1:0],

    output logic result_rdy
);

    // =========================================================================
    //  INPUT SKEW NETWORK
    //
    //  Without skewing, a_in[0] and b_in[0] would both reach PE[0][0] at
    //  cycle 0, but a_in[1] would reach PE[1][1] 2 cycles late vs b_in[1]
    //  (1 row of propagation for A, 1 col of propagation for B → mismatch).
    //
    //  By delaying a_in[i] by i cycles and b_in[j] by j cycles, both streams
    //  accumulate exactly (i + j) cycles of delay on their way to PE[i][j]:
    //    A arrives at PE[i][j] at cycle:  k + i + j   (k = element index)
    //    B arrives at PE[i][j] at cycle:  k + j + i   ← same!
    // =========================================================================

    logic signed [DATA_WIDTH-1:0] a_skewed [M-1:0];
    logic signed [DATA_WIDTH-1:0] b_skewed [N-1:0];
    // Psum skew network — eliminated when LOAD_PSUM_EN=0 (saves ACC_WIDTH*Σ(i+j) FFs)
    logic load_psum_skewed [M-1:0][N-1:0];
    logic signed [ACC_WIDTH-1:0] psum_skewed [M-1:0][N-1:0];

    // Row 0 and col 0 need no delay
    assign a_skewed[0] = a_in[0];
    assign b_skewed[0] = b_in[0];

    generate
        // ── A row skew ───────────────────────────────────────────────────────
        for (genvar i = 1; i < M; i++) begin : gen_a_skew
            // i-stage shift register
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

        // ── B column skew ────────────────────────────────────────────────────
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

        // ── Psum skew network (Option 7) ──────────────────────────────────
        // When LOAD_PSUM_EN=0, the entire psum/load_psum skew network is
        // eliminated. load_psum_skewed defaults to 0 and psum_skewed to 0,
        // which synthesis will optimize away since the PE psum path is also
        // gated off by LOAD_PSUM_EN=0 in PE_mac_pipelined.
        // Savings: ACC_WIDTH * Σ(i+j for all PE positions) FFs + control FFs
        if (LOAD_PSUM_EN) begin : gen_psum_skew
            for (genvar i = 0; i < M; i++) begin : gen_lp_row
                for (genvar j = 0; j < N; j++) begin : gen_lp_col
                    localparam int DELAY = i + j;

                    if (DELAY == 0) begin : gen_lp_nodelay
                        assign load_psum_skewed[i][j] = load_psum;
                        assign psum_skewed[i][j]      = psum_in[i][j];
                    end else begin : gen_lp_delay
                        logic lp_sr [0:DELAY-1];
                        logic signed [ACC_WIDTH-1:0] psum_sr [0:DELAY-1];

                        always_ff @(posedge clk or negedge rst_n) begin
                            if (!rst_n) begin
                                lp_sr[0] <= 1'b0;
                                psum_sr[0] <= '0;
                            end else begin
                                lp_sr[0] <= load_psum;
                                psum_sr[0] <= psum_in[i][j];
                            end
                        end

                        for (genvar s = 1; s < DELAY; s++) begin : gen_lp_sr
                            always_ff @(posedge clk or negedge rst_n) begin
                                if (!rst_n) begin
                                    lp_sr[s] <= 1'b0;
                                    psum_sr[s] <= '0;
                                end else begin
                                    psum_sr[s] <= psum_sr[s-1];
                                    lp_sr[s] <= lp_sr[s-1];
                                end
                            end
                        end

                        assign load_psum_skewed[i][j] = lp_sr[DELAY-1];
                        assign psum_skewed[i][j]      = psum_sr[DELAY-1];
                    end
                end
            end
        end else begin : gen_no_psum_skew
            // Psum path eliminated — tie to zero, synthesis will optimize away
            for (genvar i = 0; i < M; i++) begin : gen_lp_zero_row
                for (genvar j = 0; j < N; j++) begin : gen_lp_zero_col
                    assign load_psum_skewed[i][j] = 1'b0;
                    assign psum_skewed[i][j]       = '0;
                end
            end
        end
    endgenerate

    // =========================================================================
    //  INTER-PE DATA BUSES
    //
    //  a_h[i][j] : activation data arriving at PE[i][j] from the left
    //    a_h[i][0]   = a_skewed[i]        ← left boundary (array input)
    //    a_h[i][j+1] = a_h[i][j] delayed 1 cycle  ← pass-through FF
    //
    //  b_v[i][j] : weight data arriving at PE[i][j] from above
    //    b_v[0][j]   = b_skewed[j]        ← top boundary (array input)
    //    b_v[i+1][j] = b_v[i][j] delayed 1 cycle  ← pass-through FF
    //
    //  One pass-through FF per inter-PE hop is what makes this array systolic:
    //  data ripples across the grid like a wave, one cell per clock.
    // =========================================================================

    // Size [N:0] / [M:0] to include both array boundary (index 0/M/N) and
    // the [N] / [M] drain slot that is driven but not read.
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
    //  PE ARRAY  —  M×N grid of PE_mac_pipelined instances
    //
    //  Each generate cell instantiates:
    //    ① 1 pass-through FF for a_h[i][j+1]  (propagates A rightward)
    //    ② 1 pass-through FF for b_v[i+1][j]  (propagates B downward)
    //    ③ 1 PE_mac_pipelined computing weight × act and accumulating
    //
    //  Mapping of PE ports:
    //    weight  ← b_v[i][j]  (B matrix flows as weights)
    //    act     ← a_h[i][j]  (A matrix flows as activations)
    //    accum   → result[i][j]
    // =========================================================================

    // =========================================================================
    //  CONTROL SKEW NETWORK  (Option 1)
    //
    //  When CTRL_SKEW_EN=1, en/clear_acc are skewed by (i+j) cycles per PE
    //  to match the data wavefront. PE[i][j] then runs an independent
    //  K-element dot product whose validity boundary is (i+j) cycles after
    //  the controller-level boundary. This lets the controller retrigger
    //  clear_acc every K cycles back-to-back without the M+N+2 fill/drain
    //  bubble — ~1.6× speedup at K=16, ~1.15× at K=64.
    //
    //  When CTRL_SKEW_EN=0, signals are broadcast (legacy behaviour).
    // =========================================================================
    logic en_skewed        [M-1:0][N-1:0];
    logic clear_acc_skewed [M-1:0][N-1:0];

    generate
        for (genvar i = 0; i < M; i++) begin : gen_ctrl_row
            for (genvar j = 0; j < N; j++) begin : gen_ctrl_col
                localparam int DELAY_CTRL = i + j;

                if (!CTRL_SKEW_EN || DELAY_CTRL == 0) begin : gen_ctrl_nodelay
                    assign en_skewed[i][j]        = en;
                    assign clear_acc_skewed[i][j] = clear_acc;
                end else if (DELAY_CTRL == 1) begin : gen_ctrl_delay1
                    // 1-bit skew: single-FF delay. {en_sr[-1:0],en} is illegal in
                    // Vivado (negative part-select); Verilator tolerated it.
                    logic en_sr;
                    logic clr_sr;
                    always_ff @(posedge clk or negedge rst_n) begin
                        if (!rst_n) begin
                            en_sr  <= 1'b0;
                            clr_sr <= 1'b0;
                        end else begin
                            en_sr  <= en;
                            clr_sr <= clear_acc;
                        end
                    end
                    assign en_skewed[i][j]        = en_sr;
                    assign clear_acc_skewed[i][j] = clr_sr;
                end else begin : gen_ctrl_delay
                    logic [DELAY_CTRL-1:0] en_sr;
                    logic [DELAY_CTRL-1:0] clr_sr;
                    always_ff @(posedge clk or negedge rst_n) begin
                        if (!rst_n) begin
                            en_sr  <= '0;
                            clr_sr <= '0;
                        end else begin
                            en_sr  <= {en_sr [DELAY_CTRL-2:0], en};
                            clr_sr <= {clr_sr[DELAY_CTRL-2:0], clear_acc};
                        end
                    end
                    assign en_skewed[i][j]        = en_sr [DELAY_CTRL-1];
                    assign clear_acc_skewed[i][j] = clr_sr[DELAY_CTRL-1];
                end
            end
        end
    endgenerate

    generate
        for (genvar i = 0; i < M; i++) begin : gen_pe_row
            for (genvar j = 0; j < N; j++) begin : gen_pe_col

                // ① Pass-through FFs (1 cycle per hop)
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        a_h[i][j+1] <= '0;   // A flows right
                        b_v[i+1][j] <= '0;   // B flows down
                    end else begin
                        a_h[i][j+1] <= a_h[i][j];
                        b_v[i+1][j] <= b_v[i][j];
                    end
                end

                // ② Operand gating (Option 6)
                //
                // Hold the multiplier inputs at zero whenever either operand
                // is zero. Saves dynamic power on the multiplier (no toggling
                // when activations are post-ReLU sparse) and propagates a
                // clean 0 down the accumulator. Free correctness — multiply
                // by zero already gives zero.
                logic signed [DATA_WIDTH-1:0] act_gated, w_gated;
                if (ZERO_SKIP_EN) begin : gen_zero_skip
                    wire op_active = (a_h[i][j] != '0) && (b_v[i][j] != '0);
                    assign act_gated = op_active ? a_h[i][j] : '0;
                    assign w_gated   = op_active ? b_v[i][j] : '0;
                end else begin : gen_no_skip
                    assign act_gated = a_h[i][j];
                    assign w_gated   = b_v[i][j];
                end

                // ③ PE instantiation
                PE_mac_pipelined #(
                    .DATA_W        (DATA_WIDTH),
                    .IN_W          (DATA_WIDTH),
                    .ACC_W         (ACC_WIDTH),
                    .LOAD_PSUM_EN  (LOAD_PSUM_EN)
                ) u_pe (
                    .clk       (clk),
                    .rst_n     (rst_n),
                    .en        (en_skewed[i][j]),
                    .clear_acc (clear_acc_skewed[i][j]),
                    .weight    (w_gated),      // B — flows top→down (gated)
                    .act       (act_gated),    // A — flows left→right (gated)
                    .psum_in   (psum_skewed[i][j]), // Incoming partial sum
                    .load_psum (load_psum_skewed[i][j]),
                    .accum     (result[i][j])
                );

                // ④ Per-PE result_valid (Option 3)
                //
                // Pulses 2 cycles after the per-PE en falling edge. The 2
                // cycles align with PE_mac_pipelined's internal stage 1+2
                // delay so result[i][j] is settled when result_valid_pe
                // pulses high.
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
                        pe_falling = en_skew_d & ~en_skewed[i][j];
                        en_skew_d <= en_skewed[i][j];
                        rvsr <= {rvsr[0], pe_falling};
                    end
                end
                assign result_valid_pe[i][j] = rvsr[1];

            end // gen_pe_col
        end // gen_pe_row
    endgenerate

    // =========================================================================
    //  RESULT READY GENERATION
    //
    //  When 'en' goes low, the first PE[0][0] finishes its K-element dot product.
    //  However, the data wavefront still needs to propagate to the furthest PE.
    //  The delay required is:
    //    (M - 1) cycles for A matrix row skew
    //    (N - 1) cycles for B matrix column skew
    //    + 2     cycles for the internal pipeline of PE_mac_pipelined
    // =========================================================================

    localparam int FLUSH_DELAY = (M - 1) + (N - 1) + 2;

    logic en_d;
    logic [FLUSH_DELAY-1:0] flush_sr;
    logic en_falling;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            en_d            <= 1'b0;
            flush_sr        <= '0;
            result_rdy      <= 1'b0;
        end else begin
            // Blocking assignment captures pre-NBA en_d before update.
            // Without this, en_d <= en would update en_d first (NBAs are
            // sequential in some simulators) and the edge detection
            // en_d & ~en would read the new (already-cleared) value.
            en_falling = en_d & ~en;
            en_d <= en;

            // Shift register to delay the "done" token across the array
            flush_sr <= {flush_sr[FLUSH_DELAY-2:0], en_falling};

            // Assert result_rdy when the token pops out
            result_rdy <= flush_sr[FLUSH_DELAY-1];
        end
    end
endmodule
