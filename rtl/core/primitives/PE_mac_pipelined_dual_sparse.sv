// =============================================================================
//  PE_mac_pipelined_dual_sparse.sv
//
//  Dual-MAC Processing Element with per-accumulator zero-skip gating.
//
//  Combines the DSP48 INT8 dual-packing from PE_mac_pipelined_dual with
//  sparsity-aware accumulation gating from the sparse array.
//
//  DSP packing (same as PE_mac_pipelined_dual):
//    A[24:16] = sign-extended weight_hi (9 bits)
//    A[15:0]  = sign-extended weight_lo (16 bits)
//    B[17:0]  = sign-extended activation (18 bits)
//    P[15:0]  = weight_lo × act   (sign-extended to ACC_WIDTH)
//    P[31:16] = weight_hi × act + carry - sign correction
//
//  Sparse gating (new):
//    acc_lo_en / acc_hi_en control independent accumulation enables.
//    When sparse_en=1 and acc_lo_en=0, accum_lo skips the product add
//    (saving dynamic power). Same for accum_hi.
//    When sparse_en=0, both accumulators always update (normal mode).
//
//  Pipeline Stages (same 3-stage latency):
//    Stage 1: Input registration + DSP packing + zero-detect pipeline
//    Stage 2: DSP48 multiply + pipeline delay for gating signals
//    Stage 3: Product extraction + sign correction + gated dual accumulation
//
//  Resource Usage:
//    1 DSP48 slice + 2 × 32-bit LUT accumulators + sparse gating FFs
// =============================================================================

import pkg_config::*;

module PE_mac_pipelined_dual_sparse #(
    parameter DATA_WIDTH = 8,
    parameter IN_WIDTH   = DATA_WIDTH,
    parameter ACC_WIDTH  = 4 * DATA_WIDTH,
    parameter bit USE_DSP = 1,
    parameter bit LOAD_PSUM_EN = 0
)(
    input  logic clk,
    input  logic rst_n,

    // Control signals
    input  logic en,
    input  logic clear_acc,
    input  logic load_psum,

    // Sparse gating enables (per-accumulator, from array-level zero-detect)
    input  logic sparse_en,      // 1 = sparse gating active
    input  logic acc_lo_en,      // 0 = gate accum_lo (skip MAC)
    input  logic acc_hi_en,      // 0 = gate accum_hi (skip MAC)

    // Data inputs — two weights, one shared activation
    input  logic signed [IN_WIDTH-1:0]  weight_lo,
    input  logic signed [IN_WIDTH-1:0]  weight_hi,
    input  logic signed [IN_WIDTH-1:0]  act,

    // Partial sum inputs
    input  logic signed [ACC_WIDTH-1:0] psum_in_lo,
    input  logic signed [ACC_WIDTH-1:0] psum_in_hi,

    // Outputs — two independent accumulator results
    output logic signed [ACC_WIDTH-1:0] accum_lo,
    output logic signed [ACC_WIDTH-1:0] accum_hi
);

    // =========================================================================
    //  STAGE 1: Input Registration + DSP48 Port Packing
    // =========================================================================
    logic signed [24:0]          dsp_a_reg;
    logic signed [17:0]          dsp_b_reg;

    logic signed [ACC_WIDTH-1:0] psum_lo_reg1, psum_hi_reg1;
    logic signed [ACC_WIDTH-1:0] psum_lo_reg2, psum_hi_reg2;

    // Sign correction pipeline
    logic weight_lo_neg_reg1, weight_lo_neg_reg2;
    logic signed [IN_WIDTH-1:0] act_corr_reg1, act_corr_reg2;

    // Control signal delay line
    logic clear_acc_reg1, clear_acc_reg2;
    logic en_reg1, en_reg2;
    logic load_psum_reg1, load_psum_reg2;

    // Sparse gating pipeline (2-stage to align with Stage 3 accumulation)
    logic sparse_en_reg1, sparse_en_reg2;
    logic acc_lo_en_reg1, acc_lo_en_reg2;
    logic acc_hi_en_reg1, acc_hi_en_reg2;

    // =========================================================================
    //  STAGE 2: DSP48 Multiplication (registered P output)
    // =========================================================================
    (* use_dsp = "yes" *)
    logic signed [47:0] dsp_p;

    // =========================================================================
    //  Combinational Product Extraction (same as PE_mac_pipelined_dual)
    // =========================================================================
    logic signed [ACC_WIDTH-1:0] product_lo;
    logic signed [ACC_WIDTH-1:0] product_hi;
    logic signed [ACC_WIDTH-1:0] sign_correction;

    assign sign_correction = weight_lo_neg_reg2
                           ? ACC_WIDTH'(signed'(act_corr_reg2))
                           : ACC_WIDTH'(0);

    assign product_lo = ACC_WIDTH'(signed'(dsp_p[15:0]));
    assign product_hi = ACC_WIDTH'(signed'(dsp_p[31:16]))
                      + ACC_WIDTH'(dsp_p[15])
                      - sign_correction;

    // =========================================================================
    //  PIPELINE LOGIC (3 stages, same latency as PE_mac_pipelined_dual)
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Stage 1 resets
            dsp_a_reg         <= '0;
            dsp_b_reg         <= '0;
            psum_lo_reg1      <= '0;
            psum_hi_reg1      <= '0;
            clear_acc_reg1    <= 1'b0;
            en_reg1           <= 1'b0;
            load_psum_reg1    <= 1'b0;
            weight_lo_neg_reg1 <= 1'b0;
            act_corr_reg1     <= '0;
            sparse_en_reg1    <= 1'b0;
            acc_lo_en_reg1    <= 1'b0;
            acc_hi_en_reg1    <= 1'b0;

            // Stage 2 resets
            dsp_p             <= '0;
            psum_lo_reg2      <= '0;
            psum_hi_reg2      <= '0;
            clear_acc_reg2    <= 1'b0;
            en_reg2           <= 1'b0;
            load_psum_reg2    <= 1'b0;
            weight_lo_neg_reg2 <= 1'b0;
            act_corr_reg2     <= '0;
            sparse_en_reg2    <= 1'b0;
            acc_lo_en_reg2    <= 1'b0;
            acc_hi_en_reg2    <= 1'b0;

            // Stage 3 resets
            accum_lo          <= '0;
            accum_hi          <= '0;
        end else begin
            // ----------------------------------------------------------------
            // STAGE 1: Register inputs and pack DSP48 ports
            // ----------------------------------------------------------------
            dsp_a_reg <= { weight_hi[7], weight_hi[7:0],
                           {8{weight_lo[7]}}, weight_lo[7:0] };
            dsp_b_reg <= { {10{act[7]}}, act[7:0] };

            psum_lo_reg1   <= (LOAD_PSUM_EN) ? psum_in_lo   : '0;
            psum_hi_reg1   <= (LOAD_PSUM_EN) ? psum_in_hi   : '0;
            clear_acc_reg1 <= clear_acc;
            en_reg1        <= en;
            load_psum_reg1 <= (LOAD_PSUM_EN) ? load_psum    : 1'b0;

            weight_lo_neg_reg1 <= weight_lo[IN_WIDTH-1];
            act_corr_reg1      <= act;

            // Pipeline sparse gating signals
            sparse_en_reg1 <= sparse_en;
            acc_lo_en_reg1 <= acc_lo_en;
            acc_hi_en_reg1 <= acc_hi_en;

            // ----------------------------------------------------------------
            // STAGE 2: DSP48 multiply
            // ----------------------------------------------------------------
            dsp_p <= dsp_a_reg * dsp_b_reg;

            psum_lo_reg2   <= psum_lo_reg1;
            psum_hi_reg2   <= psum_hi_reg1;
            clear_acc_reg2 <= clear_acc_reg1;
            en_reg2        <= en_reg1;
            load_psum_reg2 <= load_psum_reg1;
            weight_lo_neg_reg2 <= weight_lo_neg_reg1;
            act_corr_reg2     <= act_corr_reg1;

            // Pipeline sparse gating (align with Stage 3)
            sparse_en_reg2 <= sparse_en_reg1;
            acc_lo_en_reg2 <= acc_lo_en_reg1;
            acc_hi_en_reg2 <= acc_hi_en_reg1;

            // ----------------------------------------------------------------
            //  STAGE 3: Gated dual accumulation
            // ----------------------------------------------------------------
            if (clear_acc_reg2) begin
                accum_lo <= product_lo;
                accum_hi <= product_hi;
            end else if (LOAD_PSUM_EN && load_psum_reg2) begin
                accum_lo <= psum_lo_reg2 + product_lo;
                accum_hi <= psum_hi_reg2 + product_hi;
            end else if (en_reg2) begin
                // Sparse gating: skip accumulator update when enable is low
                if (!sparse_en_reg2 || acc_lo_en_reg2)
                    accum_lo <= accum_lo + product_lo;
                if (!sparse_en_reg2 || acc_hi_en_reg2)
                    accum_hi <= accum_hi + product_hi;
            end
        end
    end

endmodule
