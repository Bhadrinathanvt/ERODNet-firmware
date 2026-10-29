// =============================================================================
// layerNorm_fixed.sv
//
// Layer Normalization, fully streaming, fixed-latency pipeline (PIPELINE_STAGES
// = 13 cycles input->output) with synthesizable ROM/LUT initialization.
//
// Pipeline (one token advances one stage per non-stalled cycle):
//   Stage  0 : input register + capture gamma/beta (per-frame weights)
//   Stage  1 : sum, group 0           (adder-tree, split for timing)
//   Stage  2 : sum, rest of first half
//   Stage  3 : sum, second half  -> sum_full
//   Stage  4 : mean = sum_full * (1/cfg_c)   (reciprocal LUT multiply, no DSP)
//   Stage  5 : x - mean
//   Stage  6 : sum-of-squares, group 0
//   Stage  7 : sum-of-squares, rest of first half
//   Stage  8 : sum-of-squares, second half -> sq_sum
//   Stage  9 : variance = sq_sum * (1/cfg_c) + epsilon  -> variance_reg
//   Stage 10 : 1/sqrt LUT index/fraction (saturating)   [reads variance_reg]
//   Stage 11 : 1/sqrt = base + slope*frac    (LUT lookup + linear interp)
//   Stage 12 : y = gamma * (x-mean) * (1/sqrt) + beta  ;  valid_out asserted
//
// Backpressure: when valid_out & !out_ready, the *entire* pipeline freezes
//   (every register gated by !stall); ready_in = !stall.
//
// CONTRACT: cfg_c, gamma[] and beta[] must be held constant for every token
//   that is in flight (i.e. per inference frame).  cfg_c_d3 is only forwarded
//   3 stages but reused at stages 6-12, and gamma/beta are sampled once at
//   stage 0; mixing parameters between in-flight tokens corrupts results.
//   Enforced in simulation by the LN_CFG_*_STABLE assertions below.
//
// Synthesis Directives:
//   - (* rom_style = "block" *) for the inverse-sqrt LUT
//   - (* use_dsp = "no" *) by default - LUT-based multipliers for Artix-7 100T
//
// Artix-7 100T Optimization:
//   - LUT-based multipliers + reciprocal/1-sqrt LUTs save DSP slices
//   - Scale path is not the critical throughput path
//   - DSP budget reserved for the systolic-array MACs
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;

module layerNorm_fixed #(
    parameter int DATA_WIDTH = pkg_module_config::LAYER_NORM_FIXED_DATA_WIDTH,
    parameter int FRAC_WIDTH = pkg_config::FRAC_WIDTH,
    parameter int MAX_C = 64,
    parameter bit  USE_DSP = 0,    // Default: LUT-based (0), use DSP (1)
    parameter int ADDER_GROUP_SIZE = 8    // Adder tree pipeline group width
)(
    input  logic clk,
    input  logic rst_n,

    // Config
    input  logic [15:0] cfg_c,

    // Handshake
    input  logic valid_in,
    output logic ready_in,       // Can accept new input (backpressure)
    output logic valid_out,
    input  logic out_ready,       // Consumer ready for output

    // Learned parameters
    input  logic signed [DATA_WIDTH-1:0] gamma [0:MAX_C-1],
    input  logic signed [DATA_WIDTH-1:0] beta  [0:MAX_C-1],

    // Data
    input  logic signed [DATA_WIDTH-1:0] x_in  [0:MAX_C-1],
    output logic signed [DATA_WIDTH-1:0] y_out [0:MAX_C-1]
);

    localparam real EPSILON   = 1.0 / 256.0;
    // Q-format integer view of EPSILON for the variance bias-add. Computed
    // at elaboration so synthesis sees a constant. Width matches FRAC_WIDTH.
    localparam int  EPSILON_Q = int'(EPSILON * (1 << FRAC_WIDTH));
    localparam int SUM_WIDTH = DATA_WIDTH + $clog2(MAX_C);
    localparam int SQ_SUM_WIDTH = 2 * DATA_WIDTH + $clog2(MAX_C);
    localparam int GROUP_SIZE   = ADDER_GROUP_SIZE;    // Adder tree pipeline group width
    localparam int PIPELINE_STAGES = 13;

    // Parameterized LUT bit-slice positions.  variance_reg is 2*DATA_WIDTH bits
    // and holds (variance + eps) in Q(.).(2*FRAC); the 1/sqrt LUT is indexed by
    // floor((variance+eps)*64), so the index sits just above bit 2*FRAC_WIDTH-6.
    //
    // BUG-EROD-LN-007 (FIXED): when layerNorm_fixed is instantiated with a small
    //   DATA_WIDTH relative to FRAC_WIDTH (e.g. the ConvFormer block passes
    //   DATA_WIDTH=8 with the system FRAC_WIDTH=16), the raw slice positions go
    //   beyond 2*DATA_WIDTH-1 — or LUT_IDX_FULL_W = 2*DATA_WIDTH-LUT_IDX_LSB goes
    //   negative — and elaboration fails (zero/negative-size cast).  Clamp every
    //   position into [0, 2*DATA_WIDTH-1] so the module always elaborates; at
    //   such params the 1/sqrt result is coarse (use FRAC_WIDTH <= DATA_WIDTH/2,
    //   à la the GFF Q-format fix, for full precision).  No effect at the normal
    //   config (DATA_WIDTH=32, FRAC_WIDTH=16): LUT_IDX_LSB=26, LUT_IDX_FULL_W=38.
    localparam int VR_W         = 2*DATA_WIDTH;                       // variance_reg width
    localparam int LUT_IDX_LSB  = (2*FRAC_WIDTH - 6 < 0)        ? 0
                                : (2*FRAC_WIDTH - 6 > VR_W - 1) ? (VR_W - 1)
                                :  2*FRAC_WIDTH - 6;
    // Interpolation fraction = top (up to) FRAC_WIDTH bits of the sub-index remainder.
    localparam int LUT_FRAC_MSB = (2*FRAC_WIDTH - 7 < 0)        ? 0
                                : (2*FRAC_WIDTH - 7 > VR_W - 1) ? (VR_W - 1)
                                :  2*FRAC_WIDTH - 7;
    localparam int LUT_FRAC_LSB = (FRAC_WIDTH - 6 < 0)            ? 0
                                : (FRAC_WIDTH - 6 > LUT_FRAC_MSB) ? LUT_FRAC_MSB
                                :  FRAC_WIDTH - 6;

    // =========================================================================
    // 1/sqrt LUT from ROM files (covers [0, 4.0) with 256 entries)
    // =========================================================================
    localparam int LUT_SIZE = pkg_config::LUT_SIZE;
    localparam int LUT_INDEX_BITS = pkg_config::LUT_INDEX_BITS;

    // ROM inference
    (* rom_style = "block" *)
    logic signed [DATA_WIDTH-1:0] inv_sqrt_lut   [0:LUT_SIZE-1];
    (* rom_style = "block" *)
    logic signed [DATA_WIDTH-1:0] inv_sqrt_slope  [0:LUT_SIZE-1];

    // Initialize LUT from precomputed ROM file (synthesizable via $readmemh)
    initial begin
        $readmemh("rtl/ln_inv_sqrt_base.mem", inv_sqrt_lut);
        $readmemh("rtl/ln_inv_sqrt_slope.mem", inv_sqrt_slope);
    end

    // =========================================================================
    // Reciprocal LUT for runtime channel-count division (mean / variance)
    // =========================================================================
    // cfg_c can be any value from 1 to MAX_C.  Instead of a constant shift
    // by $clog2(MAX_C) we multiply by 1/cfg_c in Q(FRAC_WIDTH).FRAC_WIDTH.
    //
    // BUG-EROD-LN-004 (FIXED): the table was unconditionally initialized for
    //   indices 0..64 but the array was sized [0:MAX_C].  For any build with
    //   MAX_C < 64 (e.g. the DV harness uses MAX_C=4) the writes to
    //   recip_lut[MAX_C+1 .. 64] are out-of-bounds — undefined behaviour in
    //   simulation (Verilator clobbers adjacent state, corrupting mean / 1-sqrt
    //   and producing garbage outputs).  Fix: generate the table at elaboration
    //   with a loop bounded by MAX_C — recip_lut[k] = round((1<<FRAC_WIDTH)/k).
    //
    // BUG-EROD-LN-005 (FIXED): the first cut of that loop used a block-local
    //   `int recip_i = int'(...)` temp plus an `if` clamp inside the unrolled
    //   `initial` for-loop; Verilator's loop unroller dropped the writes and
    //   left recip_lut all-zero → mean & variance both came out 0 → 1/sqrt(ε)
    //   leaked into every output.  Fix: no block-local temp / no $cast in the
    //   loop body — compute and saturate inline (only k==1 can overflow, and
    //   cfg_c==1 ⇒ x−mean==0 so that entry is don't-care anyway).
    localparam int RECIP_W = FRAC_WIDTH + 1;
    localparam logic signed [RECIP_W-1:0] RECIP_MAX = (1 <<< (RECIP_W-1)) - 1;
    logic signed [RECIP_W-1:0] recip_lut [0:MAX_C];
    initial begin
        recip_lut[0] = '0;
        for (int k = 1; k <= MAX_C; k++) begin
            recip_lut[k] = (k == 1) ? RECIP_MAX
                                    : RECIP_W'(((1 <<< FRAC_WIDTH) + (k >>> 1)) / k);
        end
    end

    // =========================================================================
    // Backpressure / Stall Logic
    // =========================================================================
    // When the output is stalled (valid_out asserted but out_ready low),
    // freeze the entire pipeline to hold all data stable.
    logic stall;
    assign stall    = valid_out && !out_ready;
    assign ready_in = !stall;

    // =========================================================================
    // Valid Pipeline (frozen when stalled)
    // =========================================================================
    logic [PIPELINE_STAGES-1:0] valid_pipe;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_pipe <= '0;
        else if (!stall) valid_pipe <= {valid_pipe[PIPELINE_STAGES-2:0], valid_in};
    end
    assign valid_out = valid_pipe[PIPELINE_STAGES-1];

    // =========================================================================
    // Stage 0: Input Register
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] x_reg [0:MAX_C-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < MAX_C; i++) x_reg[i] <= '0;
        end else if (valid_in && !stall) begin
            for (int i = 0; i < MAX_C; i++) x_reg[i] <= x_in[i];
        end
    end

    // ── Pipeline forwarding (BUG-EROD-LN-002 fix) ──────────────────────
    // x_reg shared between Stage 0, 1, 2, and 4. Back-to-back valid_in
    // overwrites x_reg before Stages 2 and 4 read it.
    // x_minus_mean shared between Stage 4 and Stage 9.
    // gamma/beta are layer weights (constant per-frame, not per-token).
    // Instead of forwarding through 7 stages (~7,200 FFs), register once
    // on valid_in and hold for the entire pipeline.

    // x_in forwarding: Stage 0 → Stages 2,3,5 (5 hops for pipelined adder)
    logic signed [DATA_WIDTH-1:0] x_in_fwd [0:4][0:MAX_C-1];
    // gamma/beta: single-register hold (saves 7× forwarding stages)
    logic signed [DATA_WIDTH-1:0] gamma_reg [0:MAX_C-1];
    logic signed [DATA_WIDTH-1:0] beta_reg  [0:MAX_C-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int s = 0; s < 5; s++)
                for (int i = 0; i < MAX_C; i++) x_in_fwd[s][i] <= '0;
            for (int i = 0; i < MAX_C; i++) begin
                gamma_reg[i] <= '0;
                beta_reg[i]  <= '0;
            end
        end else if (!stall) begin
            // x_in forwarding
            if (valid_in)    for (int i = 0; i < MAX_C; i++) x_in_fwd[0][i] <= x_in[i];
            if (valid_pipe[0]) x_in_fwd[1] <= x_in_fwd[0];
            if (valid_pipe[1]) x_in_fwd[2] <= x_in_fwd[1];
            if (valid_pipe[2]) x_in_fwd[3] <= x_in_fwd[2];
            if (valid_pipe[3]) x_in_fwd[4] <= x_in_fwd[3];
            // gamma/beta: sample once per accepted token
            if (valid_in) begin
                gamma_reg <= gamma;
                beta_reg  <= beta;
            end
        end
    end

    // =========================================================================
    // Stage 1-3: Sum Computation (pipelined adder tree)
    // =========================================================================
    localparam int HALF_C = MAX_C / 2;
    logic signed [SUM_WIDTH-1:0] sum_partial;
    logic signed [SUM_WIDTH-1:0] sum_full;
    logic [15:0] cfg_c_d1, cfg_c_d2, cfg_c_d3;

    // Stage 1: First group of first half sum (GROUP_SIZE terms)
    logic signed [SUM_WIDTH-1:0] sum_group0;
    always_comb begin
        sum_group0 = '0;
        for (int i = 0; i < GROUP_SIZE; i++) begin
            if (i < cfg_c && i < HALF_C) sum_group0 = sum_group0 + SUM_WIDTH'(x_reg[i]);
        end
    end

    // Registered first group partial sum
    logic signed [SUM_WIDTH-1:0] sum_group0_d1;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_group0_d1 <= '0;
            cfg_c_d1 <= '0;
        end else if (valid_pipe[0] && !stall) begin
            sum_group0_d1 <= sum_group0;
            cfg_c_d1 <= cfg_c;
        end
    end

    // Stage 2: Remaining groups of first half → sum_partial
    // (automatic variable extracted to always_comb)
    logic signed [SUM_WIDTH-1:0] sum_partial_comb;
    always_comb begin
        sum_partial_comb = sum_group0_d1;
        for (int i = GROUP_SIZE; i < HALF_C; i++) begin
            if (i < cfg_c_d1) sum_partial_comb = sum_partial_comb + SUM_WIDTH'(x_in_fwd[1][i]);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_partial <= '0;
            cfg_c_d2 <= '0;
        end else if (valid_pipe[1] && !stall) begin
            sum_partial <= sum_partial_comb;
            cfg_c_d2 <= cfg_c_d1;
        end
    end

    // Stage 3: Complete sum (adder tree - uses forwarded x_in for hazard safety)
    // (automatic variable extracted to always_comb)
    logic signed [SUM_WIDTH-1:0] sum_full_comb;
    always_comb begin
        sum_full_comb = sum_partial;
        for (int i = HALF_C; i < MAX_C; i++) begin
            if (i < cfg_c_d2) sum_full_comb = sum_full_comb + SUM_WIDTH'(x_in_fwd[2][i]);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_full <= '0;
            cfg_c_d3 <= '0;
        end else if (valid_pipe[2] && !stall) begin
            sum_full <= sum_full_comb;
            cfg_c_d3 <= cfg_c_d2;
        end
    end

    // =========================================================================
    // Stage 4: Mean (reciprocal LUT multiply, no DSP)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] mean_reg;
    logic signed [SUM_WIDTH+RECIP_W-1:0] mean_product;

    always_comb begin
        // cfg_c is static per-layer, so cfg_c_d3 is valid for all pipeline stages
        mean_product = sum_full * recip_lut[cfg_c_d3[$clog2(MAX_C+1)-1:0]];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) mean_reg <= '0;
        else if (valid_pipe[3] && !stall) mean_reg <= DATA_WIDTH'(mean_product >>> FRAC_WIDTH);
    end

    // =========================================================================
    // Stage 5: x - mean (uses forwarded x_in, not shared x_reg)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] x_minus_mean [0:MAX_C-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < MAX_C; i++) x_minus_mean[i] <= '0;
        end else if (valid_pipe[4] && !stall) begin
            for (int i = 0; i < MAX_C; i++) begin
                x_minus_mean[i] <= x_in_fwd[4][i] - mean_reg;
            end
        end
    end

    // ── x_minus_mean forwarding: Stage 5 → Stage 11 (6 hops) ──
    logic signed [DATA_WIDTH-1:0] x_mm_fwd [0:5][0:MAX_C-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int s = 0; s < 6; s++)
                for (int i = 0; i < MAX_C; i++) x_mm_fwd[s][i] <= '0;
        end else if (!stall) begin
            if (valid_pipe[5]) x_mm_fwd[0] <= x_minus_mean;
            if (valid_pipe[6]) x_mm_fwd[1] <= x_mm_fwd[0];
            if (valid_pipe[7]) x_mm_fwd[2] <= x_mm_fwd[1];
            if (valid_pipe[8]) x_mm_fwd[3] <= x_mm_fwd[2];
            if (valid_pipe[9]) x_mm_fwd[4] <= x_mm_fwd[3];
            if (valid_pipe[10]) x_mm_fwd[5] <= x_mm_fwd[4];
        end
    end

    // =========================================================================
    // Stage 6-8: Sum of Squares (pipelined LUT-based multiplier)
    // =========================================================================
    // Square operations: can use LUT-based multiplier (not critical path)
    // Set use_dsp based on parameter
    logic signed [SQ_SUM_WIDTH-1:0] sq_partial;
    logic signed [SQ_SUM_WIDTH-1:0] sq_sum;

    // Stage 6: First group of first half squares (GROUP_SIZE terms)
    logic signed [SQ_SUM_WIDTH-1:0] sq_group0;
    generate
        if (USE_DSP) begin : gen_sq_g0_dsp
            (* use_dsp = "yes" *)
            always_comb begin
                sq_group0 = '0;
                for (int i = 0; i < GROUP_SIZE; i++) begin
                    if (i < cfg_c_d3 && i < HALF_C)
                        sq_group0 = sq_group0 + SQ_SUM_WIDTH'(x_minus_mean[i]) * x_minus_mean[i];
                end
            end
        end else begin : gen_sq_g0_lut
            (* use_dsp = "no" *)
            always_comb begin
                sq_group0 = '0;
                for (int i = 0; i < GROUP_SIZE; i++) begin
                    if (i < cfg_c_d3 && i < HALF_C)
                        sq_group0 = sq_group0 + SQ_SUM_WIDTH'(x_minus_mean[i]) * x_minus_mean[i];
                end
            end
        end
    endgenerate

    // Registered first group partial sum
    logic signed [SQ_SUM_WIDTH-1:0] sq_group0_d1;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) sq_group0_d1 <= '0;
        else if (valid_pipe[5] && !stall) sq_group0_d1 <= sq_group0;
    end

    // Stage 7: Remaining groups of first half → sq_partial
    // (automatic variable extracted to always_comb)
    logic signed [SQ_SUM_WIDTH-1:0] sq_partial_comb;
    always_comb begin
        sq_partial_comb = sq_group0_d1;
        for (int i = GROUP_SIZE; i < HALF_C; i++) begin
            if (i < cfg_c_d3) sq_partial_comb = sq_partial_comb + SQ_SUM_WIDTH'(x_mm_fwd[0][i]) * x_mm_fwd[0][i];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) sq_partial <= '0;
        else if (valid_pipe[6] && !stall) sq_partial <= sq_partial_comb;
    end

    // Stage 8: Complete sum of squares
    // (automatic variable extracted to always_comb)
    logic signed [SQ_SUM_WIDTH-1:0] sq_sum_comb;
    always_comb begin
        sq_sum_comb = sq_partial;
        for (int i = HALF_C; i < MAX_C; i++) begin
            if (i < cfg_c_d3) sq_sum_comb = sq_sum_comb + SQ_SUM_WIDTH'(x_mm_fwd[1][i]) * x_mm_fwd[1][i];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) sq_sum <= '0;
        else if (valid_pipe[7] && !stall) sq_sum <= sq_sum_comb;
    end

    // =========================================================================
    // Stage 9: Variance (reciprocal LUT multiply, no DSP)
    // =========================================================================
    logic signed [2*DATA_WIDTH-1:0] variance_reg;
    logic signed [SQ_SUM_WIDTH+RECIP_W-1:0] var_product;

    always_comb begin
        var_product = sq_sum * recip_lut[cfg_c_d3[$clog2(MAX_C+1)-1:0]];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) variance_reg <= '0;
        else if (valid_pipe[8] && !stall) begin
            // EPSILON_Q is the Q.FRAC_WIDTH integer; widen before the shift
            // so we don't lose MSBs when promoting Q.16 → Q.32 (variance is
            // stored in 2*DATA_WIDTH bits → 64-bit on Q16.16 default).
            variance_reg <= (2*DATA_WIDTH)'((var_product >>> FRAC_WIDTH) +
                           ((SQ_SUM_WIDTH)'(EPSILON_Q) << FRAC_WIDTH));
        end
    end

    // =========================================================================
    // Stage 10: 1/sqrt (LUT lookup - no DSP)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] inv_sqrt_reg;
    logic [LUT_INDEX_BITS-1:0] lut_idx_reg;
    logic [FRAC_WIDTH-1:0] lut_frac_reg;  // Unsigned: fraction in [0,1) as Q0.16

    // BUG-EROD-LN-006 (FIXED): the index slice variance_reg[LUT_IDX_MSB:LUT_IDX_LSB]
    //   is only LUT_INDEX_BITS wide, so for variance >= LUT-range (default 4.0)
    //   floor(variance*64) overflows and *wraps* mod LUT_SIZE — a silently wrong
    //   1/sqrt — and the ">= LUT_SIZE" clamp downstream was unreachable (it can
    //   never see a value >= LUT_SIZE).  Fix: take the *full* high slice
    //   variance_reg[2*DATA_WIDTH-1:LUT_IDX_LSB] (= floor(variance*64), unbounded,
    //   including the sign bit so a "negative"/overflowed variance also saturates)
    //   and saturate to LUT_SIZE-1 before storing the LUT_INDEX_BITS-wide index.
    localparam int LUT_IDX_FULL_W = 2*DATA_WIDTH - LUT_IDX_LSB;  // 38 on the default config
    logic [LUT_IDX_FULL_W-1:0] lut_idx_full_comb;
    always_comb lut_idx_full_comb = variance_reg[2*DATA_WIDTH-1 : LUT_IDX_LSB];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lut_idx_reg <= '0;
            lut_frac_reg <= '0;
        end else if (valid_pipe[9] && !stall) begin
            // LUT index for [0, 4.0) range in Q(2*DATA_WIDTH).(2*FRAC_WIDTH),
            // saturated so out-of-range variance maps to the last LUT entry
            // (smallest 1/sqrt) instead of aliasing.
            lut_idx_reg  <= (lut_idx_full_comb >= LUT_IDX_FULL_W'(LUT_SIZE))
                          ? LUT_INDEX_BITS'(LUT_SIZE-1)
                          : LUT_INDEX_BITS'(lut_idx_full_comb);
            // Fractional offset in Q0.FRAC_WIDTH for interpolation
            lut_frac_reg <= variance_reg[LUT_FRAC_MSB:LUT_FRAC_LSB];
        end
    end

    // LUT lookup (automatic extracted to always_comb)
    logic signed [DATA_WIDTH-1:0] inv_sqrt_base_comb;
    logic signed [DATA_WIDTH-1:0] inv_sqrt_slope_comb;
    logic signed [2*DATA_WIDTH-1:0] inv_sqrt_product_comb;

    always_comb begin
        if ({1'b0, lut_idx_reg} >= (LUT_INDEX_BITS+1)'(LUT_SIZE)) begin
            inv_sqrt_base_comb  = inv_sqrt_lut[LUT_SIZE-1];
            inv_sqrt_slope_comb = inv_sqrt_slope[LUT_SIZE-1];
        end else begin
            inv_sqrt_base_comb  = inv_sqrt_lut[lut_idx_reg];
            inv_sqrt_slope_comb = inv_sqrt_slope[lut_idx_reg];
        end
        // slope (Q16.16) * frac (Q0.16) = Q16.32; truncate to Q16.16
        inv_sqrt_product_comb = inv_sqrt_slope_comb *
            $signed({{(DATA_WIDTH > FRAC_WIDTH ? DATA_WIDTH - FRAC_WIDTH : 0){1'b0}}, lut_frac_reg});
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) inv_sqrt_reg <= '0;
        else if (valid_pipe[10] && !stall)
            inv_sqrt_reg <= inv_sqrt_base_comb +
                           DATA_WIDTH'(inv_sqrt_product_comb >>> FRAC_WIDTH);
    end

    // =========================================================================
    // Stage 11: Final Scale and Shift (single combinational compute + reg out)
    // =========================================================================
    // Two multiply operations per channel:
    //   1. normalized = x_minus_mean * inv_sqrt_reg
    //   2. scaled = gamma * normalized
    // Intermediates are local loop variables — synthesis need not preserve
    // module-level array signals, reducing routing/mux overhead.  Only the
    // y_out vector is registered.  Bulk array assignment in always_ff avoids
    // per-channel cfg_c comparators.

    logic signed [DATA_WIDTH-1:0] y_out_comb [0:MAX_C-1];

    // Q-format: x_mm_fwd (Q16.16) × inv_sqrt_reg (Q16.16) = Q32.32 → norm  (64-bit)
    //           gamma_reg (Q16.16) × norm (Q32.32)            = Q48.48 → scaled (96-bit)
    //           scaled >>> (2*FRAC_WIDTH) extracts Q16.16 from the upper bits
    localparam int NORM_W   = 2 * DATA_WIDTH;
    localparam int SCALED_W = 3 * DATA_WIDTH;

    generate
        if (USE_DSP) begin : gen_scale_dsp
            (* use_dsp = "yes" *)
            for (genvar gi = 0; gi < MAX_C; gi++) begin : gen_ch
                logic signed [NORM_W-1:0]   norm;
                logic signed [SCALED_W-1:0] scaled;
                always_comb begin
                    norm   = (NORM_W)'(x_mm_fwd[5][gi]) * (NORM_W)'(inv_sqrt_reg);
                    scaled = (SCALED_W)'(gamma_reg[gi]) * (SCALED_W)'(norm);
                    y_out_comb[gi] = (gi < cfg_c_d3)
                        ? DATA_WIDTH'(scaled >>> (2*FRAC_WIDTH)) + beta_reg[gi]
                        : '0;
                end
            end
        end else begin : gen_scale_lut
            (* use_dsp = "no" *)
            for (genvar gi = 0; gi < MAX_C; gi++) begin : gen_ch
                logic signed [NORM_W-1:0]   norm;
                logic signed [SCALED_W-1:0] scaled;
                always_comb begin
                    norm   = (NORM_W)'(x_mm_fwd[5][gi]) * (NORM_W)'(inv_sqrt_reg);
                    scaled = (SCALED_W)'(gamma_reg[gi]) * (SCALED_W)'(norm);
                    y_out_comb[gi] = (gi < cfg_c_d3)
                        ? DATA_WIDTH'(scaled >>> (2*FRAC_WIDTH)) + beta_reg[gi]
                        : '0;
                end
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < MAX_C; i++) y_out[i] <= '0;
        end else if (valid_pipe[11] && !stall) begin
            y_out <= y_out_comb;
        end
    end

    // =========================================================================
    // Assertions — Verilator 5.046 doesn't support $stable() on unpacked
    // arrays (y_out is logic[DATA_WIDTH-1:0][0:MAX_C-1]). Gate behind
    // VERILATOR ifndef so iverilog/Questa still see them.
    // =========================================================================
    // synthesis translate_off
    LAYERNORM_CHANNEL_BOUNDS: assert property (
        @(posedge clk) disable iff (!rst_n)
        valid_in |-> cfg_c <= MAX_C
    ) else $error("layerNorm_fixed: cfg_c exceeds MAX_C");

    // Per-frame-config contract (see header): cfg_c / gamma / beta must not
    // change while any token is in flight, otherwise stage 6-12 reuse of
    // cfg_c_d3 and the once-sampled gamma_reg/beta_reg corrupts results.
    LN_CFG_C_STABLE: assert property(@(posedge clk) disable iff (!rst_n)
        (|valid_pipe) |-> $stable(cfg_c))
        else $error("layerNorm_fixed: cfg_c changed while a token is in flight");

    LN_GAMMA_STABLE: assert property(@(posedge clk) disable iff (!rst_n)
        (|valid_pipe) |-> $stable(gamma[0]))
        else $error("layerNorm_fixed: gamma[] changed while a token is in flight");

    LN_BETA_STABLE: assert property(@(posedge clk) disable iff (!rst_n)
        (|valid_pipe) |-> $stable(beta[0]))
        else $error("layerNorm_fixed: beta[] changed while a token is in flight");

    LN_VALID_STABLE: assert property(@(posedge clk) disable iff (!rst_n)
        (valid_out && !out_ready) |=> valid_out)
        else $error("layerNorm_fixed: valid_out deasserted before out_ready");

`ifndef VERILATOR
    LN_DATA_STABLE: assert property(@(posedge clk) disable iff (!rst_n)
        (valid_out && !out_ready) |=> $stable(y_out))
        else $error("layerNorm_fixed: y_out changed while valid_out held without out_ready");

    LN_STALL_FREEZES_PIPE: assert property(@(posedge clk) disable iff (!rst_n)
        (valid_out && !out_ready) |=> $stable(valid_pipe))
        else $error("layerNorm_fixed: valid_pipe shifted while pipeline stalled");
`endif

    generate
        if (!USE_DSP) begin : gen_lut_mult
            initial begin
                $info("layerNorm_fixed: Using LUT-based multipliers (USE_DSP=0). Saves ~128 DSP slices for MAX_C=64.");
            end
        end
    endgenerate
    // synthesis translate_on

endmodule
