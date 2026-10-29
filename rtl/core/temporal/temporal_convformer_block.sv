// =============================================================================
// temporal_convformer_block.sv
//
// Complete temporal ConvFormer block with multi-frame aggregation.
// Extends the standard ConvFormer with temporal processing for radar
// object detection (RODNet architecture).
//
// Architecture:
//   ┌────────────────────────────────────────────────────────────────────┐
//   │                    Temporal ConvFormer Block                        │
//   │                                                                    │
//   │  Input ──▶ [Temporal Buffer] ──▶ [Temporal Fusion] ──▶           │
//   │                                        │                          │
//   │                                        ▼                          │
//   │                              [ConvFormer Block]                   │
//   │                                        │                          │
//   │                                        ▼                          │
//   │                              [Multi-Frame Agg] ──▶ Output        │
//   │                                                                    │
//   └────────────────────────────────────────────────────────────────────┘
//
// Pipeline Stages:
//   1. Temporal Buffer: Store T frames of history
//   2. Temporal Fusion: Attention-based fusion of history frames
//   3. ConvFormer: Spatial-channel processing (depthwise + pointwise)
//   4. Multi-Frame Aggregation: Combine temporal and spatial features
//
// Modes (cfg_temporal_mode):
//   0: No temporal processing (standard ConvFormer)
//   1: Temporal fusion only
//   2: Multi-frame aggregation only
//   3: Both fusion + aggregation
//   4: ConvLSTM recurrent mode
//
// Latency: ~40-50 cycles (depends on mode)
// Throughput: 1 token/cycle after fill
// =============================================================================

import pkg_config::*;

module temporal_convformer_block #(
    parameter int HISTORY_DEPTH = 8,
    parameter int TOTAL_LATENCY = pkg_config::TOTAL_LATENCY
)(
    input  logic clk,
    input  logic rst_n,

    // Configuration
    input  logic [2:0]  cfg_temporal_mode,     // Temporal processing mode
    input  logic [3:0]  cfg_history_depth,     // Number of frames (T)
    input  logic [3:0]  cfg_kernel_t,          // Temporal kernel size
    input  logic [15:0] cfg_c,                 // Active channels
    input  logic [15:0] cfg_kernel_size,       // Spatial kernel size
    input  logic [2:0]  cfg_act_sel,           // Activation select
    input  logic signed [31:0] cfg_scale_in,   // StarReLU scale
    input  logic signed [31:0] cfg_bias_in,    // StarReLU bias
    input  logic         cfg_clk_gate,
    input  logic         cfg_enable_temporal,   // Enable temporal processing

    // Input stream
    input  logic                        valid_in,
    input  logic                        frame_last_in,      // Frame boundary marker
    input  logic signed [DATA_WIDTH-1:0] token_in [MAX_C-1:0],
    output logic                        ready_in,

    // Output stream
    output logic                        valid_out,
    output logic                        frame_last_out,
    output logic signed [DATA_WIDTH-1:0] token_out [MAX_C-1:0],
    input  logic                        out_ready,

    // Weight loading (ConvFormer)
    input  logic             pw_wr_en,
    input  logic [1:0]       pw_wr_bank,
    input  logic [13:0]      pw_wr_addr,
    input  logic signed [DATA_WIDTH-1:0] pw_wr_data,

    input  logic             dw_wr_en,
    input  logic [15:0]      dw_wr_ch,
    input  logic [15:0]      dw_wr_k,
    input  logic signed [DATA_WIDTH-1:0] dw_wr_data,

    // Temporal weights (for temporal fusion)
    input  logic signed [DATA_WIDTH-1:0] temporal_weights [7:0][MAX_C-1:0],

    // Performance counters
    output logic [31:0]      perf_cycle_count,
    output logic [31:0]      perf_stall_count,
    output logic [31:0]      perf_token_count,

    // ── Sparse CSR Interface (0xB00–0xB3F) ────────────────────────────────────
    input  logic        sparse_csr_req_valid,
    input  logic [11:0] sparse_csr_req_addr,
    input  logic [31:0] sparse_csr_req_wdata,
    input  logic        sparse_csr_req_we,
    output logic [31:0] sparse_csr_req_rdata,
    output logic        sparse_csr_req_ready
);

    localparam int EXPANDED_C = MAX_C * EXPANSION_FACTOR;

    // =========================================================================
    // Temporal Modes
    // =========================================================================
    localparam bit [2:0] MODE_NONE      = 3'b000;  // Standard ConvFormer
    localparam bit [2:0] MODE_FUSION    = 3'b001;  // Temporal fusion only
    localparam bit [2:0] MODE_AGG       = 3'b010;  // Multi-frame aggregation only
    localparam bit [2:0] MODE_BOTH      = 3'b011;  // Fusion + Aggregation
    localparam bit [2:0] MODE_CONVLSTM  = 3'b100;  // ConvLSTM recurrent

    // =========================================================================
    // Internal Signals
    // =========================================================================
    // Stage 1: Temporal Buffer
    logic                        buffer_valid_out;
    logic [HISTORY_DEPTH-1:0]   buffer_valid_mask;
    logic signed [DATA_WIDTH-1:0] buffer_data_out [HISTORY_DEPTH-1:0][MAX_C-1:0];
    logic                        buffer_ready_in;

    // Stage 2: Temporal Fusion
    logic                        fusion_valid_out;
    logic signed [DATA_WIDTH-1:0] fusion_data_out [MAX_C-1:0];
    logic                        fusion_ready_in;
    logic signed [ACC_WIDTH-1:0] fusion_attention [HISTORY_DEPTH-1:0];

    // Stage 3: ConvFormer
    logic                        conv_valid_out;
    logic                        conv_last_out;
    logic signed [DATA_WIDTH-1:0] conv_data_out [MAX_C-1:0];
    logic                        conv_ready_in;

    // Stage 4: Multi-Frame Aggregation
    logic                        agg_valid_out;
    logic signed [DATA_WIDTH-1:0] agg_data_out [MAX_C-1:0];
    logic                        agg_ready_in;
    logic signed [DATA_WIDTH-1:0] hidden_state [MAX_C-1:0];

    // Attention weight adapter: temporal_weights is [7:0][MAX_C-1:0] (2-D),
    // but temporal_fusion expects [HISTORY_DEPTH-1:0] (1-D scalar per frame).
    // Extract first channel of each history slot as frame-level attention weight.
    logic signed [DATA_WIDTH-1:0] temporal_attn_weights [HISTORY_DEPTH-1:0];

    // Bypass paths
    logic                        bypass_temporal_valid;
    logic signed [DATA_WIDTH-1:0] bypass_temporal_data [MAX_C-1:0];

    // =========================================================================
    // Mode Selection Mux
    // =========================================================================
    logic use_temporal_buffer;
    logic use_temporal_fusion;
    logic use_multi_frame_agg;

    always_comb begin
        use_temporal_buffer = cfg_enable_temporal && (cfg_temporal_mode != MODE_NONE);
        use_temporal_fusion = cfg_enable_temporal &&
                             (cfg_temporal_mode == MODE_FUSION ||
                              cfg_temporal_mode == MODE_BOTH);
        use_multi_frame_agg = cfg_enable_temporal &&
                             (cfg_temporal_mode == MODE_AGG ||
                              cfg_temporal_mode == MODE_BOTH ||
                              cfg_temporal_mode == MODE_CONVLSTM);
    end

    // =========================================================================
    // Attention Weight Adapter (TEMP-PORT-001)
    // =========================================================================
    always_comb begin
        for (int t = 0; t < HISTORY_DEPTH; t++)
            temporal_attn_weights[t] = temporal_weights[t][0];
    end

    // =========================================================================
    // Stage 1: Temporal Buffer
    // =========================================================================
    generate
        if (HISTORY_DEPTH > 1) begin : gen_temporal_buffer
            temporal_buffer #(
                .HISTORY_DEPTH(HISTORY_DEPTH),
                .CHANNELS(MAX_C),
                .DATA_W(DATA_WIDTH)
            ) u_temporal_buffer (
                .clk(clk),
                .rst_n(rst_n),
                .cfg_history_depth(cfg_history_depth),
                .cfg_enable(use_temporal_buffer),
                .frame_valid_in(valid_in),
                .frame_last(frame_last_in),
                .frame_data_in(token_in),
                .frame_ready_in(buffer_ready_in),
                .history_valid_out(buffer_valid_out),
                .history_valid_mask(buffer_valid_mask),
                .history_data(buffer_data_out),
                .history_ready_in(fusion_ready_in),
                .frame_start(frame_last_in),
                .buffer_full(),
                .frame_count()
            );
        end else begin : gen_no_temporal_buffer
            // Bypass when history depth is 1
            assign buffer_valid_out = valid_in;
            assign buffer_valid_mask = 1'b1;
            for (genvar t = 0; t < HISTORY_DEPTH; t++)
                for (genvar c = 0; c < MAX_C; c++)
                    assign buffer_data_out[t][c] = token_in[c];
            assign buffer_ready_in = fusion_ready_in;
        end
    endgenerate

    // =========================================================================
    // Stage 2: Temporal Fusion
    // =========================================================================
    temporal_fusion #(
        .HISTORY_DEPTH(HISTORY_DEPTH),
        .CHANNELS(MAX_C),
        .DATA_W(DATA_WIDTH),
        .ACC_W(ACC_WIDTH)
    ) u_temporal_fusion (
        .clk(clk),
        .rst_n(rst_n),
        // cfg_temporal_mode[1:0] truncation is safe: MODE_CONVLSTM (3'b100) never
        // enters the fusion module because use_temporal_fusion is false for that mode,
        // so the fusion module is bypassed and the truncated bits are don't-care.
        .cfg_fusion_mode(cfg_temporal_mode[1:0]),
        .cfg_history_depth(cfg_history_depth),
        .cfg_enable(use_temporal_fusion),
        .query_valid_in(buffer_valid_out),
        .query_data(buffer_data_out[0]),      // Current frame as query
        .query_ready_in(fusion_ready_in),
        .history_valid_in(buffer_valid_out),
        .history_valid_mask(buffer_valid_mask),
        .history_data(buffer_data_out),
        .attention_weights(temporal_attn_weights),
        .fused_valid_out(fusion_valid_out),
        .fused_data_out(fusion_data_out),
        .fused_ready_in(conv_ready_in),
        .attention_scores(fusion_attention)
    );

    // Bypass for modes that don't use buffer/temporal processing
    always_comb begin
        if (!use_temporal_buffer) begin
            // Buffer disabled (MODE_NONE): pass through directly
            bypass_temporal_valid = valid_in;
            for (int c = 0; c < MAX_C; c++)
                bypass_temporal_data[c] = token_in[c];
        end else if (!use_temporal_fusion) begin
            // Buffer enabled, fusion disabled: use buffer output directly
            bypass_temporal_valid = buffer_valid_out;
            for (int c = 0; c < MAX_C; c++)
                bypass_temporal_data[c] = buffer_data_out[0][c];
        end else begin
            // Buffer and fusion both enabled: use fusion output
            bypass_temporal_valid = fusion_valid_out;
            for (int c = 0; c < MAX_C; c++)
                bypass_temporal_data[c] = fusion_data_out[c];
        end
    end

    // =========================================================================
    // Stage 3: ConvFormer Block
    // =========================================================================
    convformer_block_fixed #(
        .TOTAL_LATENCY(TOTAL_LATENCY)
    ) u_convformer (
        .clk(clk),
        .rst_n(rst_n),
        .cfg_c(cfg_c),
        .cfg_kernel_size(cfg_kernel_size),
        .cfg_act_sel(cfg_act_sel),
        .scale_in(cfg_scale_in),
        .bias_in(cfg_bias_in),
        .cfg_clk_gate(cfg_clk_gate),
        .weight_loaded(1'b1),
        .valid_in(bypass_temporal_valid),
        .valid_in_last(frame_last_in),
        .token_in(bypass_temporal_data),
        .ready_in(conv_ready_in),
        .valid_out(conv_valid_out),
        .valid_out_last(conv_last_out),
        .token_out(conv_data_out),
        // When multi_frame_agg is bypassed (cfg_temporal_mode==MODE_NONE) the
        // agg sets frame_ready_in=1 unconditionally, so its ready cannot
        // signal real downstream back-pressure. Route the top-level out_ready
        // through directly (gated by the held-token latch above) so conv stalls
        // when downstream blocks the chain instead of dropping tokens.
        .out_ready(use_multi_frame_agg ? agg_ready_in
                                       : (out_ready || !valid_out)),
        .pw_wr_en(pw_wr_en),
        .pw_wr_bank(pw_wr_bank),
        .pw_wr_addr(pw_wr_addr),
        .pw_wr_data(pw_wr_data),
        .dw_wr_en(dw_wr_en),
        .dw_wr_ch(dw_wr_ch),
        .dw_wr_k(dw_wr_k),
        .dw_wr_data(dw_wr_data),
        .perf_cycle_count(perf_cycle_count),
        .perf_stall_count(perf_stall_count),
        .perf_token_count(perf_token_count),
        .sparse_csr_req_valid(sparse_csr_req_valid),
        .sparse_csr_req_addr(sparse_csr_req_addr),
        .sparse_csr_req_wdata(sparse_csr_req_wdata),
        .sparse_csr_req_we(sparse_csr_req_we),
        .sparse_csr_req_rdata(sparse_csr_req_rdata),
        .sparse_csr_req_ready(sparse_csr_req_ready)
    );

    // =========================================================================
    // Stage 4: Multi-Frame Aggregation
    // =========================================================================
    multi_frame_agg #(
        .HISTORY_DEPTH(HISTORY_DEPTH),
        .CHANNELS(MAX_C),
        .KERNEL_T(3),  // 3-frame temporal kernel
        .DATA_W(DATA_WIDTH),
        .ACC_W(ACC_WIDTH)
    ) u_multi_frame_agg (
        .clk(clk),
        .rst_n(rst_n),
        .cfg_agg_mode(cfg_temporal_mode),
        .cfg_history_depth(cfg_history_depth),
        .cfg_kernel_t(cfg_kernel_t),
        .cfg_enable(use_multi_frame_agg),
        .frame_valid_in(conv_valid_out),
        .frame_last_in(conv_last_out),
        .frame_data_in(conv_data_out),
        .frame_ready_in(agg_ready_in),
        .temporal_weights(temporal_weights[2:0]),
        .agg_valid_out(agg_valid_out),
        .agg_data_out(agg_data_out),
        .agg_ready_in(out_ready),
        .hidden_state_out(hidden_state),
        .hidden_state_in(hidden_state)  // Recurrent connection
    );

    // =========================================================================
    // Output Selection
    // =========================================================================
    // FIXED: TEMP-LAST-001 — frame_last_out must come from the correct pipeline
    // stage. When multi_frame_agg is active, conv_last_out marks the spatial
    // frame boundary from ConvFormer, which propagates through aggregation at
    // the same cadence. The aggregation module processes each spatial token
    // across temporal frames, preserving the frame-last timing.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            frame_last_out <= 1'b0;
            for (int c = 0; c < MAX_C; c++)
                token_out[c] <= '0;
        end else if (valid_out && !out_ready) begin
            // Hold output stable until the consumer accepts it; otherwise the
            // upstream conv/agg pulse can deassert valid_out mid-handshake,
            // violating TEMPORAL_OUT_STABLE and losing the token. Data is held
            // because conv.out_ready/agg.out_ready below back-pressure the
            // upstream pipeline so it cannot overwrite while we wait.
            valid_out <= 1'b1;
        end else begin
            if (use_multi_frame_agg) begin
                valid_out <= agg_valid_out;
                // conv_last_out marks frame boundary; aggregation preserves it
                frame_last_out <= conv_last_out;
                for (int c = 0; c < MAX_C; c++)
                    token_out[c] <= agg_data_out[c];
            end else begin
                valid_out <= conv_valid_out;
                frame_last_out <= conv_last_out;
                for (int c = 0; c < MAX_C; c++)
                    token_out[c] <= conv_data_out[c];
            end
        end
    end

    // Back-pressure chain (the modules own their own ready logic; we just route it):
    //   upstream ◄─ ready_in ◄─ conv_ready_in  (= u_convformer.ready_in output)
    //                              u_convformer.out_ready ◄─ agg_ready_in
    //                                                          (= u_multi_frame_agg.frame_ready_in output)
    //                                  u_multi_frame_agg.agg_ready_in ◄─ out_ready (top-level)
    //
    // BUG-EROD-TCB-001 (FIXED): the previous `assign agg_ready_in = ...` /
    //   `assign conv_ready_in = ...` lines double-drove these nets — they were
    //   ALREADY driven by u_convformer.ready_in (output, line 278) and
    //   u_multi_frame_agg.frame_ready_in (output, line 321).  Verilator's
    //   MULTIDRIVEN warning was flagging a real synthesis hazard (the two drivers
    //   would short / produce X).  Removed the redundant assigns; the modules'
    //   own ready outputs are the single source of truth.
    assign ready_in = conv_ready_in;

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    TEMPORAL_MODE: assert property (
        @(posedge clk) disable iff (!rst_n)
        cfg_temporal_mode inside {MODE_NONE, MODE_FUSION, MODE_AGG, MODE_BOTH, MODE_CONVLSTM}
    ) else $error("temporal_convformer: Invalid temporal mode");

    TEMPORAL_HISTORY: assert property (
        @(posedge clk) disable iff (!rst_n)
        cfg_history_depth > 0 && cfg_history_depth <= HISTORY_DEPTH
    ) else $error("temporal_convformer: Invalid history depth");

    // Output-stability backpressure invariant: once valid_out is asserted it
    // must stay asserted until the consumer accepts it (out_ready).
    //
    // The previous assertion (valid_out |-> ready_in || !out_ready) was
    // INCORRECT: it required the INPUT side to be ready whenever the OUTPUT
    // was valid. That never holds for this block — it is a deeply buffered
    // pipeline (temporal_buffer + convformer + multi_frame_agg) that
    // legitimately emits output while input-stalled, e.g. when draining its
    // tail at end-of-stream. ready_in=0 during drain loses no data.
    TEMPORAL_OUT_STABLE: assert property (
        @(posedge clk) disable iff (!rst_n)
        (valid_out && !out_ready) |=> valid_out
    ) else $error("temporal_convformer: valid_out dropped before out_ready");
    // synthesis translate_on

endmodule