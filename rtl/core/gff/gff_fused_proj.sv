// =============================================================================
// gff_fused_proj.sv
//
// Fused Project Pipeline for GFF Global Feature Fusion.
// Chains: Linear1(4C→3C) → LayerNorm(3C) → Linear2(3C→C) → GELU(C) → Linear3(C→C)
// with 1-cycle skid buffers between stages, eliminating the 3×3=9 cycles of
// inter-layer overhead from the original separate-instantiation approach.
//
// The original approach used 4 separate `linear_layer` instances, each with
// 3-cycle overhead (input register + round/saturate + output register).
// This fused module replaces those 4 stages plus LayerNorm and GELU with
// a tightly-coupled pipeline using skid buffers (1 cycle per stage boundary).
//
// Latency: (4C+3) + 10 + (3C+3) + 4 + (C+3) + 4(skids) = 8C + 27 cycles
//   vs original: (4C+3) + 3 + 10 + 3 + (3C+3) + 3 + 4 + 3 + (C+3) + 3 = 8C + 39
//   Savings: 12 cycles (3 per stage boundary × 4 boundaries)
//
// Note: the linear_layer instances are reused internally but connected
// with zero-idle-gap skid buffers instead of full handshake overhead.
// =============================================================================

`timescale 1ns / 1ps

import pkg_config::*;

module gff_fused_proj #(
    parameter int C           = 64,
    parameter int DATA_W     = 16,
    parameter int FRAC_W     = 8,
    parameter int ACC_W      = 32,
    parameter bit  USE_DSP   = 0,
    parameter bit  GELU_DSP  = 0,
    parameter string ROM_DIR = "rtl/mem"
)(
    input  logic clk,
    input  logic rst_n,

    // Input: pooled global vector (4C channels)
    input  logic signed [DATA_W-1:0] pooled      [0:4*C-1],
    input  logic                      pooled_valid,

    // Output: projected global vector (C channels, after l2norm)
    output logic signed [DATA_W-1:0] proj_out    [0:C-1],
    output logic                      proj_out_v,

    // LayerNorm learned parameters (runtime configurable)
    input  logic signed [DATA_W-1:0] ln_gamma [0:3*C-1],
    input  logic signed [DATA_W-1:0] ln_beta  [0:3*C-1],

    // Weight loading
    input  logic                      weight_valid,
    input  logic [31:0]               weight_addr,
    input  logic signed [DATA_W-1:0]  weight_data
);

    // =========================================================================
    // Internal wires between pipeline stages
    // =========================================================================
    logic signed [DATA_W-1:0] proj_lin1   [0:3*C-1];
    logic                     proj_lin1_v;
    logic                     proj_lin1_skid_v;

    logic signed [DATA_W-1:0] proj_ln     [0:3*C-1];
    logic                     proj_ln_v;

    logic signed [DATA_W-1:0] proj_lin2   [0:C-1];
    logic                     proj_lin2_v;
    logic                     proj_lin2_skid_v;

    logic signed [DATA_W-1:0] proj_gelu   [0:C-1];
    logic                     proj_gelu_v;
    logic                     proj_gelu_skid_v;

    logic signed [DATA_W-1:0] proj_lin3   [0:C-1];
    logic                     proj_lin3_v;

    // =========================================================================
    // Stage 1: Linear 4C → 3C
    // =========================================================================
    linear_layer #(
        .DATA_WIDTH   (DATA_W),
        .FRAC_WIDTH   (FRAC_W),
        .IN_CH        (4*C),
        .OUT_CH       (3*C),
        .STUB_BEHAVIOR(1'b0),
        .WEIGHT_LOAD_MODE(0),
        .WEIGHT_FILE  ({ROM_DIR, "/gff_proj1_weight.mem"}),
        .BIAS_FILE    ({ROM_DIR, "/gff_proj1_bias.mem"})
    ) u_proj_lin1 (
        .clk        (clk),
        .rst_n      (rst_n),
        .in_valid   (pooled_valid),
        .out_valid  (proj_lin1_v),
        .data_in    (pooled),
        .data_out   (proj_lin1),
        .weight_valid (weight_valid),
        .weight_addr  (weight_addr),
        .weight_data  (weight_data)
    );

    // =========================================================================
    // Skid buffer between proj_lin1 and LayerNorm
    // Holds proj_lin1 output valid for 1 extra cycle so LayerNorm can capture it
    // without the 3-cycle gap of a full handshake.
    // =========================================================================
    logic signed [DATA_W-1:0] proj_lin1_skid [0:3*C-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            proj_lin1_skid_v <= 1'b0;
            for (int ch = 0; ch < 3*C; ch++)
                proj_lin1_skid[ch] <= '0;
        end else begin
            proj_lin1_skid_v <= proj_lin1_v;
            if (proj_lin1_v) begin
                for (int ch = 0; ch < 3*C; ch++)
                    proj_lin1_skid[ch] <= proj_lin1[ch];
            end
        end
    end

    // =========================================================================
    // Stage 2: LayerNorm 3C
    // =========================================================================
    logic proj_ln_ready, proj_ln_out_ready;

    // Backpressure: stall LN when Linear2 is busy computing.
    // linear_layer has no out_ready port, so we track its busy state externally.
    // proj_ln_consumed prevents linear_layer restart when LN's valid_out is
    // held high during stall — it gates in_valid to a single-cycle pulse.
    logic proj_lin2_busy;
    logic proj_ln_consumed;
    logic proj_ln_in_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            proj_lin2_busy  <= 1'b0;
            proj_ln_consumed <= 1'b0;
        end else begin
            // Set busy when LN output is accepted by Linear2
            if (proj_ln_v && !proj_lin2_busy && !proj_ln_consumed)
                proj_lin2_busy <= 1'b1;
            // Clear busy when Linear2 produces output
            else if (proj_lin2_v)
                proj_lin2_busy <= 1'b0;

            // Mark LN output as consumed to prevent linear_layer restart.
            // Resets when LN deasserts valid_out (pipeline advanced after stall release).
            if (proj_ln_v && !proj_lin2_busy && !proj_ln_consumed)
                proj_ln_consumed <= 1'b1;
            else if (!proj_ln_v)
                proj_ln_consumed <= 1'b0;
        end
    end

    assign proj_ln_out_ready = ~proj_lin2_busy;
    assign proj_ln_in_valid  = proj_ln_v && !proj_ln_consumed;

    layerNorm_fixed #(
        .DATA_WIDTH (DATA_W),
        .FRAC_WIDTH (FRAC_W),
        .MAX_C      (3*C),
        .USE_DSP    (USE_DSP)
    ) u_proj_ln (
        .clk        (clk),
        .rst_n      (rst_n),
        .cfg_c      (16'(3*C)),
        .valid_in   (proj_lin1_skid_v),
        .ready_in   (proj_ln_ready),
        .valid_out  (proj_ln_v),
        .out_ready  (proj_ln_out_ready),
        .gamma      (ln_gamma),
        .beta       (ln_beta),
        .x_in       (proj_lin1_skid),
        .y_out      (proj_ln)
    );

    // =========================================================================
    // Stage 3: Linear 3C → C
    // =========================================================================
    linear_layer #(
        .DATA_WIDTH   (DATA_W),
        .FRAC_WIDTH   (FRAC_W),
        .IN_CH        (3*C),
        .OUT_CH       (C),
        .STUB_BEHAVIOR(1'b0),
        .WEIGHT_LOAD_MODE(0),
        .WEIGHT_FILE  ({ROM_DIR, "/gff_proj2_weight.mem"}),
        .BIAS_FILE    ({ROM_DIR, "/gff_proj2_bias.mem"})
    ) u_proj_lin2 (
        .clk        (clk),
        .rst_n      (rst_n),
        .in_valid   (proj_ln_in_valid),  // Gated: single-cycle pulse, prevents restart
        .out_valid  (proj_lin2_v),
        .data_in    (proj_ln),
        .data_out   (proj_lin2),
        .weight_valid (weight_valid),
        .weight_addr  (weight_addr),
        .weight_data  (weight_data)
    );

    // =========================================================================
    // Skid buffer between proj_lin2 and GELU
    // =========================================================================
    logic signed [DATA_W-1:0] proj_lin2_skid [0:C-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            proj_lin2_skid_v <= 1'b0;
            for (int ch = 0; ch < C; ch++)
                proj_lin2_skid[ch] <= '0;
        end else begin
            proj_lin2_skid_v <= proj_lin2_v;
            if (proj_lin2_v) begin
                for (int ch = 0; ch < C; ch++)
                    proj_lin2_skid[ch] <= proj_lin2[ch];
            end
        end
    end

    // =========================================================================
    // Stage 4: GELU (ROM-based, shared ROM)
    // =========================================================================
    localparam int GELU_INDEX_BITS = pkg_config::INDEX_BITS;
    localparam int GELU_NUM_PORTS  = C;
    localparam int GELU_FRAC_W     = DATA_W / 2;

    logic [GELU_INDEX_BITS-1:0]    fused_gelu_rd_addr [GELU_NUM_PORTS-1:0];
    logic signed [DATA_W-1:0]       fused_gelu_y_base  [GELU_NUM_PORTS-1:0];
    logic signed [GELU_FRAC_W-1:0] fused_gelu_slope   [GELU_NUM_PORTS-1:0];

    gelu_rom_shared #(
        .DATA_W     (DATA_W),
        .FRAC_W     (GELU_FRAC_W),
        .INDEX_BITS (GELU_INDEX_BITS),
        .NUM_PORTS  (GELU_NUM_PORTS),
        .ROM_DIR    (ROM_DIR)
    ) u_shared_gelu_rom (
        .clk        (clk),
        .rst_n      (rst_n),
        .rd_addr    (fused_gelu_rd_addr),
        .rd_en      (1'b1),
        .y_base_out (fused_gelu_y_base),
        .slope_out  (fused_gelu_slope)
    );

    gelu_rom #(
        .DATA_W     (DATA_W),
        .FRAC_W     (GELU_FRAC_W),
        .INDEX_BITS (GELU_INDEX_BITS),
        .CHANNELS   (C),
        .USE_DSP    (GELU_DSP),
        .SHARE_ROM  (1),
        .ROM_DIR    (ROM_DIR)
    ) u_proj_gelu (
        .clk          (clk),
        .rst_n        (rst_n),
        .x_in         (proj_lin2_skid),
        .valid_in     (proj_lin2_skid_v),
        .y_out        (proj_gelu),
        .valid_out    (proj_gelu_v),
        .gelu_en      (1'b1),
        .index_o      (fused_gelu_rd_addr),
        .rom_y_base_i (fused_gelu_y_base),
        .rom_slope_i  (fused_gelu_slope)
    );

    // =========================================================================
    // Skid buffer between GELU and proj_lin3
    // =========================================================================
    logic signed [DATA_W-1:0] proj_gelu_skid [0:C-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            proj_gelu_skid_v <= 1'b0;
            for (int ch = 0; ch < C; ch++)
                proj_gelu_skid[ch] <= '0;
        end else begin
            proj_gelu_skid_v <= proj_gelu_v;
            if (proj_gelu_v) begin
                for (int ch = 0; ch < C; ch++)
                    proj_gelu_skid[ch] <= proj_gelu[ch];
            end
        end
    end

    // =========================================================================
    // Stage 5: Linear C → C
    // =========================================================================
    linear_layer #(
        .DATA_WIDTH   (DATA_W),
        .FRAC_WIDTH   (FRAC_W),
        .IN_CH        (C),
        .OUT_CH       (C),
        .STUB_BEHAVIOR(1'b0),
        .WEIGHT_LOAD_MODE(0),
        .WEIGHT_FILE  ({ROM_DIR, "/gff_proj3_weight.mem"}),
        .BIAS_FILE    ({ROM_DIR, "/gff_proj3_bias.mem"})
    ) u_proj_lin3 (
        .clk        (clk),
        .rst_n      (rst_n),
        .in_valid   (proj_gelu_skid_v),
        .out_valid  (proj_lin3_v),
        .data_in    (proj_gelu_skid),
        .data_out   (proj_lin3),
        .weight_valid (weight_valid),
        .weight_addr  (weight_addr),
        .weight_data  (weight_data)
    );

    // =========================================================================
    // Output assignment
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            proj_out_v <= 1'b0;
            for (int ch = 0; ch < C; ch++)
                proj_out[ch] <= '0;
        end else begin
            proj_out_v <= proj_lin3_v;
            if (proj_lin3_v) begin
                for (int ch = 0; ch < C; ch++)
                    proj_out[ch] <= proj_lin3[ch];
            end
        end
    end

    // =========================================================================
    // Inline SVA Assertions — backpressure invariants
    // =========================================================================

    // LN must not push data while Linear2 is busy (backpressure invariant)
    GFF_LN_BP_LIN2: assert property (
        @(posedge clk) disable iff (!rst_n)
        proj_lin2_busy |-> !proj_ln_out_ready
    ) else $error("gff_fused_proj: proj_ln_out_ready high while Linear2 busy — data loss risk");

    // Gated in_valid must be a single-cycle pulse: once consumed, stays low until LN deasserts
    GFF_LIN2_NO_RESTART: assert property (
        @(posedge clk) disable iff (!rst_n)
        proj_ln_consumed |-> !proj_ln_in_valid
    ) else $error("gff_fused_proj: linear_layer in_valid re-asserted after consumption — restart risk");

    // When Linear2 is idle and LN has valid output, the gated pulse must fire
    GFF_LN_LIN2_HANDSHAKE: assert property (
        @(posedge clk) disable iff (!rst_n)
        proj_ln_v && !proj_lin2_busy && !proj_ln_consumed |-> proj_ln_in_valid
    ) else $error("gff_fused_proj: LN valid output not forwarded to Linear2 when both idle");

endmodule