// =============================================================================
//  PE_mac_pipelined_dual.sv
//
//  Dual-MAC Processing Element using DSP48 INT8 packing.
//
//  Packs two 8-bit signed weights into a single DSP48E1 multiply:
//    A[24:16] = sign-extended weight_hi (9 bits)
//    A[15:0]  = sign-extended weight_lo (16 bits, acts as guard)
//    B[17:0]  = sign-extended activation (18 bits)
//
//  A = weight_hi × 2^16 + weight_lo (both correctly sign-extended)
//
//  Product extraction from P = A × B:
//    P[15:0]  = weight_lo × activation  (sign-extend to ACC_WIDTH)
//    P[31:16] = weight_hi × activation + carry from P[15:0] − sign correction
//
//  When weight_lo < 0, the 16-bit sign extension in A[15:0] adds 2^16 to
//  weight_lo's contribution, causing P[31:16] to be off by +act. The correction
//  subtracts act from product_hi when weight_lo was negative.
//  The carry from P[15] compensates for the borrow when weight_lo × act < 0.
//
//  Pipeline Stages (same 3-stage latency as PE_mac_pipelined):
//    Stage 1: Input registration + DSP48 A/B port packing
//    Stage 2: DSP48 multiplication (registered P output)
//    Stage 3: Product extraction + sign correction + dual accumulation
//
//  Resource Usage (per instance):
//    1 DSP48 slice (packed dual-MAC)
//    2 × 32-bit LUT accumulators + pipeline registers
// =============================================================================

import pkg_config::*;

module PE_mac_pipelined_dual #(
    parameter DATA_WIDTH = 8,
    parameter IN_WIDTH   = DATA_WIDTH,
    parameter ACC_WIDTH  = 4 * DATA_WIDTH,
    parameter bit USE_DSP = 1,      // Force DSP inference for packed multiply
    parameter bit LOAD_PSUM_EN = 0  // Enable partial-sum load path
)(
    input  logic clk,
    input  logic rst_n,

    // Control signals (same protocol as PE_mac_pipelined)
    input  logic en,
    input  logic clear_acc,
    input  logic load_psum,

    // Data inputs — two weights, one shared activation
    input  logic signed [IN_WIDTH-1:0]  weight_lo,
    input  logic signed [IN_WIDTH-1:0]  weight_hi,
    input  logic signed [IN_WIDTH-1:0]  act,

    // Partial sum inputs — two independent accumulation paths
    input  logic signed [ACC_WIDTH-1:0] psum_in_lo,
    input  logic signed [ACC_WIDTH-1:0] psum_in_hi,

    // Outputs — two independent accumulator results
    output logic signed [ACC_WIDTH-1:0] accum_lo,
    output logic signed [ACC_WIDTH-1:0] accum_hi
);

    // =========================================================================
    //  STAGE 1: Input Registration + DSP48 Port Packing
    // =========================================================================
    logic signed [24:0]          dsp_a_reg;  // Packed A port (25-bit)
    logic signed [17:0]          dsp_b_reg;  // Sign-extended B port (18-bit)

    // Partial sum pipeline (2 stages to match multiply latency)
    logic signed [ACC_WIDTH-1:0] psum_lo_reg1, psum_hi_reg1;
    logic signed [ACC_WIDTH-1:0] psum_lo_reg2, psum_hi_reg2;

    // Sign correction pipeline (2 stages, aligned with product extraction)
    logic weight_lo_neg_reg1, weight_lo_neg_reg2;
    logic signed [IN_WIDTH-1:0] act_corr_reg1, act_corr_reg2;

    // Control signal delay line (2 stages)
    logic clear_acc_reg1, clear_acc_reg2;
    logic en_reg1, en_reg2;
    logic load_psum_reg1, load_psum_reg2;

    // =========================================================================
    //  STAGE 2: DSP48 Multiplication (registered P output)
    // =========================================================================
    (* use_dsp = "yes" *)
    logic signed [47:0] dsp_p;

    // =========================================================================
    //  Combinational Product Extraction from registered DSP48 result
    //
    //  When weight_lo < 0, the 16-bit sign extension in A[15:0] adds 2^16
    //  to weight_lo's contribution, causing P[31:16] to be inflated by act.
    //  The correction subtracts act from product_hi when weight_lo was negative.
    //  The carry P[15] compensates for the borrow when weight_lo × act < 0.
    // =========================================================================
    logic signed [ACC_WIDTH-1:0] product_lo;
    logic signed [ACC_WIDTH-1:0] product_hi;
    logic signed [ACC_WIDTH-1:0] sign_correction;

    assign sign_correction = weight_lo_neg_reg2
                           ? ACC_WIDTH'(signed'(act_corr_reg2))
                           : ACC_WIDTH'(0);

    assign product_lo = ACC_WIDTH'(signed'(dsp_p[15:0]));
    assign product_hi = ACC_WIDTH'(signed'(dsp_p[31:16]))
                     + ACC_WIDTH'(dsp_p[15])              // carry from lo product
                     - sign_correction;                   // sign correction

    // =========================================================================
    //  PIPELINE LOGIC
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Stage 1 resets
            dsp_a_reg      <= '0;
            dsp_b_reg      <= '0;
            psum_lo_reg1   <= '0;
            psum_hi_reg1   <= '0;
            clear_acc_reg1 <= 1'b0;
            en_reg1        <= 1'b0;
            load_psum_reg1 <= 1'b0;
            weight_lo_neg_reg1 <= 1'b0;
            act_corr_reg1     <= '0;

            // Stage 2 resets
            dsp_p          <= '0;
            psum_lo_reg2   <= '0;
            psum_hi_reg2   <= '0;
            clear_acc_reg2 <= 1'b0;
            en_reg2        <= 1'b0;
            load_psum_reg2 <= 1'b0;
            weight_lo_neg_reg2 <= 1'b0;
            act_corr_reg2     <= '0;

            // Stage 3 resets
            accum_lo       <= '0;
            accum_hi       <= '0;
        end else begin
            // -------------------------------------------------------------
            // STAGE 3: Dual accumulation (reads pre-update values — must precede
            // Stage 2 NBAs for Verilator sequential-NBA compatibility)
            // -------------------------------------------------------------
            if (clear_acc_reg2) begin
                accum_lo <= product_lo;
                accum_hi <= product_hi;
            end else if (LOAD_PSUM_EN && load_psum_reg2) begin
                accum_lo <= psum_lo_reg2 + product_lo;
                accum_hi <= psum_hi_reg2 + product_hi;
            end else if (en_reg2) begin
                accum_lo <= accum_lo + product_lo;
                accum_hi <= accum_hi + product_hi;
            end

            // -------------------------------------------------------------
            // STAGE 2: DSP48 multiply (registered result)
            // -------------------------------------------------------------
            dsp_p <= dsp_a_reg * dsp_b_reg;

            // Pipeline partial sums and control to match multiply latency
            psum_lo_reg2   <= psum_lo_reg1;
            psum_hi_reg2   <= psum_hi_reg1;
            clear_acc_reg2 <= clear_acc_reg1;
            en_reg2        <= en_reg1;
            load_psum_reg2 <= load_psum_reg1;
            weight_lo_neg_reg2 <= weight_lo_neg_reg1;
            act_corr_reg2     <= act_corr_reg1;

            // -------------------------------------------------------------
            // STAGE 1: Register inputs and pack DSP48 ports
            // -------------------------------------------------------------

            // Pack A port: sign_ext(weight_hi, 9) in upper 9 bits,
            //              sign_ext(weight_lo, 16) in lower 16 bits.
            // A = weight_hi × 2^16 + weight_lo (both sign-extended)
            dsp_a_reg <= { weight_hi[7], weight_hi[7:0],
                           {8{weight_lo[7]}}, weight_lo[7:0] };

            // Pack B port: sign-extend activation to 18 bits
            dsp_b_reg <= { {10{act[7]}}, act[7:0] };

            psum_lo_reg1   <= (LOAD_PSUM_EN) ? psum_in_lo   : '0;
            psum_hi_reg1   <= (LOAD_PSUM_EN) ? psum_in_hi   : '0;
            clear_acc_reg1 <= clear_acc;
            en_reg1        <= en;
            load_psum_reg1 <= (LOAD_PSUM_EN) ? load_psum    : 1'b0;

            // Pipeline sign correction: weight_lo sign bit and act value
            weight_lo_neg_reg1 <= weight_lo[IN_WIDTH-1];
            act_corr_reg1      <= act;
        end
    end

endmodule
