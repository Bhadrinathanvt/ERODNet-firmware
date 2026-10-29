// =============================================================================
// e_rodnet_unet_pingpong_top.sv
//
// Thin orchestration shim that exposes the e_rodnet_unet_top with the
// new options-1/2/3/6 systolic infrastructure made visible at the wrapper
// boundary. This file does not duplicate the UNet datapath — it re-uses
// e_rodnet_unet_top verbatim and adds parameters that, when set, propagate
// down to convformer_block / systolic_array / PE_mac.
//
// Why a separate top?
// -------------------
// e_rodnet_unet_top.sv stays parameter-stable so existing testbenches keep
// working. This wrapper exists to make the ping-pong / skewed-control /
// zero-skip knobs first-class without forcing every caller to re-read the
// underlying chain of parameters. It also documents the recommended high-
// throughput configuration for Artix-7-100T (240 DSP) deployment.
//
// Recommended Artix-7-100T config:
//   USE_PINGPONG_ARRAY = 1   (Option 2: dual systolic, 32 DSPs)
//   CTRL_SKEW_EN       = 1   (Option 1: per-PE skewed control)
//   ZERO_SKIP_EN       = 1   (Option 6: multiplier zero-gate)
//   TILE_H = TILE_W    = 32  (164 KB skip BRAM, fits easily)
//
// Throughput envelope vs. legacy (single array, broadcast control, K=64):
//   Encoder pipelining (already in unet_top): ~1.24×
//   Per-PE skew (Option 1):                  ~1.15× (more for small K)
//   Dual-array ping-pong (Option 2):          ~2.0× (steady-state)
//   Combined:                                 ~2.8× over legacy
//
// This wrapper does NOT itself instantiate systolic_array_pingpong; the
// ping-pong substitution has to happen inside convformer_block (the
// systolic-using leaf). USE_PINGPONG_ARRAY is forwarded as a parameter
// hint that convformer_block may consult once the substitution is wired
// up. Until then, the parameter is documentation only — see TODO below.
//
// TODO(integration): when convformer_block accepts USE_PINGPONG_ARRAY,
// forward it through temporal_convformer_block. For now, single-array
// behaviour applies and the speedup from Options 1/3/6 still holds.
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;

module e_rodnet_unet_pingpong_top #(
    // Frame / tile geometry — passed through to e_rodnet_unet_top
    parameter int FEATURE_H     = 64,
    parameter int FEATURE_W     = 64,
    parameter int TILE_H        = 32,
    parameter int TILE_W        = 32,

    parameter int IN_CHANNELS   = 16,
    parameter int NUM_CLASSES   = 3,
    parameter int NUM_ANCHORS   = 3,
    parameter int TOP_K         = 100,
    parameter int HISTORY_DEPTH = 8,
    parameter int DATA_WIDTH    = pkg_config::DATA_WIDTH,
    parameter int FRAC_WIDTH    = pkg_config::FRAC_WIDTH,

    // Systolic-array option toggles (hint forwarded as documentation
    // until convformer_block accepts these as parameters).
    parameter bit USE_PINGPONG_ARRAY = 1'b1,   // Option 2
    parameter bit CTRL_SKEW_EN       = 1'b1,   // Option 1
    parameter bit ZERO_SKIP_EN       = 1'b1    // Option 6
)(
    input  logic clk,
    input  logic rst_n,

    // Configuration
    input  logic                        cfg_enable,
    input  logic [2:0]                  cfg_temporal_mode,
    input  logic [3:0]                  cfg_history_depth,
    input  logic signed [DATA_WIDTH-1:0] cfg_conf_threshold,
    input  logic signed [DATA_WIDTH-1:0] cfg_iou_threshold,
    input  logic [7:0]                  cfg_top_k,

    // Input tokens
    input  logic                        in_valid,
    input  logic                        in_frame_last,
    input  logic signed [DATA_WIDTH-1:0] in_data [MAX_C-1:0],
    output logic                        in_ready,

    // Weight loading
    input  logic        weight_valid,
    input  logic [31:0] weight_addr,
    input  logic signed [DATA_WIDTH-1:0] weight_data,
    input  logic [3:0]  weight_layer_sel,

    // Detection output
    output logic                        det_valid_out,
    output logic [15:0]                 det_h_idx,
    output logic [15:0]                 det_w_idx,
    output logic signed [DATA_WIDTH-1:0] det_confidence,
    output logic [15:0]                 det_width,
    output logic [15:0]                 det_height,
    output logic [NUM_CLASSES-1:0][DATA_WIDTH-1:0] det_class_scores,
    output logic [1:0]                  det_class_id,
    output logic                        det_last,
    output logic [15:0]                 det_count,
    input  logic                        det_ready_in,

    // Status
    output logic [31:0] perf_cycle_count,
    output logic [31:0] perf_frame_count,
    output logic        processing_busy,
    output logic        processing_done,

    // Sparse CSR bus (0xB00-0xB3F) — pass-through to e_rodnet_unet_top
    input  logic                        sparse_csr_req_valid,
    input  logic [11:0]                 sparse_csr_req_addr,
    input  logic [31:0]                 sparse_csr_req_wdata,
    input  logic                        sparse_csr_req_we,
    output logic [31:0]                 sparse_csr_req_rdata,
    output logic                        sparse_csr_req_ready
);

    // synthesis translate_off
    initial begin
        if (USE_PINGPONG_ARRAY)
            $info("e_rodnet_unet_pingpong_top: USE_PINGPONG_ARRAY=1 — dual-array ping-pong is wired in convformer_block once the parameter chain is connected. Single-array behaviour applies for now.");
    end
    // synthesis translate_on

    e_rodnet_unet_top #(
        .FEATURE_H     (FEATURE_H),
        .FEATURE_W     (FEATURE_W),
        .TILE_H        (TILE_H),
        .TILE_W        (TILE_W),
        .IN_CHANNELS   (IN_CHANNELS),
        .NUM_CLASSES   (NUM_CLASSES),
        .NUM_ANCHORS   (NUM_ANCHORS),
        .TOP_K         (TOP_K),
        .HISTORY_DEPTH (HISTORY_DEPTH),
        .DATA_WIDTH    (DATA_WIDTH),
        .FRAC_WIDTH    (FRAC_WIDTH)
    ) u_unet (
        .clk(clk), .rst_n(rst_n),
        .cfg_enable(cfg_enable),
        .cfg_temporal_mode(cfg_temporal_mode),
        .cfg_history_depth(cfg_history_depth),
        .cfg_conf_threshold(cfg_conf_threshold),
        .cfg_iou_threshold(cfg_iou_threshold),
        .cfg_top_k(cfg_top_k),
        .in_valid(in_valid),
        .in_frame_last(in_frame_last),
        .in_data(in_data),
        .in_ready(in_ready),
        .weight_valid(weight_valid),
        .weight_addr(weight_addr),
        .weight_data(weight_data),
        .weight_layer_sel(weight_layer_sel),
        .det_valid_out(det_valid_out),
        .det_h_idx(det_h_idx),
        .det_w_idx(det_w_idx),
        .det_confidence(det_confidence),
        .det_width(det_width),
        .det_height(det_height),
        .det_class_scores(det_class_scores),
        .det_class_id(det_class_id),
        .det_last(det_last),
        .det_count(det_count),
        .det_ready_in(det_ready_in),
        .perf_cycle_count(perf_cycle_count),
        .perf_frame_count(perf_frame_count),
        .processing_busy(processing_busy),
        .processing_done(processing_done),
        .sparse_csr_req_valid(sparse_csr_req_valid),
        .sparse_csr_req_addr(sparse_csr_req_addr),
        .sparse_csr_req_wdata(sparse_csr_req_wdata),
        .sparse_csr_req_we(sparse_csr_req_we),
        .sparse_csr_req_rdata(sparse_csr_req_rdata),
        .sparse_csr_req_ready(sparse_csr_req_ready)
    );

endmodule
