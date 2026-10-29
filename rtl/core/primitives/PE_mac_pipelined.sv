(* use_dsp = "yes" *)

// =============================================================================
//  PE_mac_pipelined.sv
//
//  Pipelined MAC (Multiply-Accumulate) Processing Element for Systolic Arrays.
//
//  Pipeline Stages:
//   Stage 1: Input registration (weight, act)
//   Stage 2: Multiplication (weight × act)
//   Stage 3: Accumulation (result + mult or continue accumulation)
//
//  When LOAD_PSUM_EN=0 (default), the partial-sum load path is eliminated,
//  saving 2×ACC_W pipeline registers and 2 control FFs per PE. This is the
//  common case since the systolic array never loads external partial sums.
//
//  Synthesis Directives:
//   - (* use_dsp = "yes" *) forces DSP48 inference for multiply-accumulate
//   - Pipeline registers aligned with DSP48 internal registers (M, P, Pcascade)
//   - Async reset for clear operation, synchronous for accumulation
//
//  Resource Usage (LOAD_PSUM_EN=0, per instance):
//   - 1 DSP48 slice (multiplier + accumulator)
//   - 68 FFs (weight_reg, act_reg, mult_reg, accum_internal, control)
//  Resource Usage (LOAD_PSUM_EN=1, per instance):
//   - 1 DSP48 slice + 134 FFs (includes psum pipeline)
// =============================================================================

module PE_mac_pipelined #(
    parameter int DATA_W = 8,
    parameter int IN_W   = DATA_W,
    parameter int ACC_W  = 32,
    parameter bit USE_DSP = 1,        // Force DSP inference
    parameter bit LOAD_PSUM_EN = 0    // Enable partial-sum load path (area: +66 FFs)
)(
    input  logic clk,
    input  logic rst_n,

    // Control signals
    input  logic en,
    input  logic clear_acc,
    input  logic load_psum,  // Command to load a partial sum (active only when LOAD_PSUM_EN=1)

    // Data inputs
    input  logic signed [IN_W-1:0]  weight,
    input  logic signed [IN_W-1:0]  act,
    input  logic signed [ACC_W-1:0] psum_in, // Partial sum (active only when LOAD_PSUM_EN=1)

    // Output
    output logic signed [ACC_W-1:0] accum
);

    // ==========================================
    // DSP Pipeline Registers
    // ==========================================
    logic signed [IN_W-1:0]       weight_reg;
    logic signed [IN_W-1:0]       act_reg;
    logic signed [(2*IN_W)-1:0]   mult_reg;

    // ==========================================
    // Control Signal Delay Line
    // ==========================================
    logic clear_acc_reg1, clear_acc_reg2;
    logic en_reg1, en_reg2;

    // Use synthesis attribute to ensure optimization doesn't remove these
    (* keep = "true" *) logic signed [ACC_W-1:0] accum_internal;

    generate
        if (LOAD_PSUM_EN) begin : gen_psum_path
            // Full psum pipeline — active when LOAD_PSUM_EN=1
            logic signed [ACC_W-1:0] psum_reg1, psum_reg2;
            logic load_psum_reg1, load_psum_reg2;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    weight_reg     <= '0;
                    act_reg        <= '0;
                    mult_reg       <= '0;
                    accum_internal <= '0;
                    psum_reg1      <= '0;
                    psum_reg2      <= '0;
                    clear_acc_reg1 <= 1'b0;
                    clear_acc_reg2 <= 1'b0;
                    en_reg1        <= 1'b0;
                    en_reg2        <= 1'b0;
                    load_psum_reg1 <= 1'b0;
                    load_psum_reg2 <= 1'b0;
                end else begin
                    // STAGE 3: Accumulation (reads pre-update values — must precede
                    // Stage 2 NBAs for Verilator sequential-NBA compatibility)
                    if (clear_acc_reg2) begin
                        accum_internal <= ACC_W'(mult_reg);
                    end else if (load_psum_reg2) begin
                        accum_internal <= psum_reg2 + ACC_W'(mult_reg);
                    end else if (en_reg2) begin
                        accum_internal <= accum_internal + ACC_W'(mult_reg);
                    end

                    // STAGE 2: Multiplication
                    mult_reg       <= weight_reg * act_reg;
                    psum_reg2      <= psum_reg1;
                    clear_acc_reg2 <= clear_acc_reg1;
                    en_reg2        <= en_reg1;
                    load_psum_reg2 <= load_psum_reg1;

                    // STAGE 1: Clock the inputs
                    weight_reg     <= weight;
                    act_reg        <= act;
                    psum_reg1      <= psum_in;
                    clear_acc_reg1 <= clear_acc;
                    en_reg1        <= en;
                    load_psum_reg1 <= load_psum;
                end
            end
        end else begin : gen_no_psum
            // Psum path eliminated — saves 2×ACC_W + 2 FFs per PE
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    weight_reg     <= '0;
                    act_reg        <= '0;
                    mult_reg       <= '0;
                    accum_internal <= '0;
                    clear_acc_reg1 <= 1'b0;
                    clear_acc_reg2 <= 1'b0;
                    en_reg1        <= 1'b0;
                    en_reg2        <= 1'b0;
                end else begin
                    // STAGE 3: Accumulation (reads pre-update values — must precede
                    // Stage 2 NBAs for Verilator sequential-NBA compatibility)
                    if (clear_acc_reg2) begin
                        accum_internal <= ACC_W'(mult_reg);
                    end else if (en_reg2) begin
                        accum_internal <= accum_internal + ACC_W'(mult_reg);
                    end

                    // STAGE 2: Multiplication
                    mult_reg       <= weight_reg * act_reg;
                    clear_acc_reg2 <= clear_acc_reg1;
                    en_reg2        <= en_reg1;

                    // STAGE 1: Clock the inputs
                    weight_reg     <= weight;
                    act_reg        <= act;
                    clear_acc_reg1 <= clear_acc;
                    en_reg1        <= en;
                end
            end
        end
    endgenerate

    assign accum = accum_internal;

    // Suppress unused param warnings
    // verilator lint_off UNUSEDPARAM
    localparam bit _unused_dsp  = USE_DSP;
    localparam bit _unused_psum = LOAD_PSUM_EN;
    // verilator lint_on UNUSEDPARAM

endmodule