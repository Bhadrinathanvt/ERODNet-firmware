// =============================================================================
// e_rodnet_unet_top.sv
//
// E-RODNet U-Net encoder-decoder with skip connections, tiled processing,
// and pipelined encoder.
//
// Paper architecture (§III): SSF → N×(Down + ConvFormer) → GFF → N×(ConvFormer + Up + Skip) → Det
//
// Topology (per tile of size TILE_H × TILE_W):
//
//     in (TILE_H×TILE_W, MAX_C, post-SSF) ── E0 ──[skip0]── stride-2 ──>
//                                                  E1 ──[skip1]── stride-2 ──>
//                                                                  E2 ── GFF (bottleneck)
//                                                                            |
//     D1 ← up2 + add(skip0) ← D2 ← up2 + add(skip1) ←──────────────────────┘
//     Detection Head ← D1
//
// Performance / area engineering
// ──────────────────────────────
//  • Pipelined encoder: E0, E1, E2, and GFF all run concurrently. As soon
//    as E0 emits a stride-2 token, E1 consumes it. E1's stride-2 output
//    feeds E2 the same way. Skip buffers are written as the streams flow
//    through. Old design serialised these stages via FSM (S_ENC0/1/2),
//    spending Σ(stage_tokens) ≈ 5376 cycles. New design spends
//    max(stage_tokens) + Σ(pipeline_depths) ≈ 4186 cycles for 64×64 input,
//    ~24 % encoder latency reduction. Inside each ConvFormer the systolic
//    array already provides parallel MACs and weight-stationary dataflow.
//
//  • Sequential decoder: D2 / D1 / Detect run after the encoder drains.
//    NN 2× upsample reads the upstream skip + replay buffer in raster
//    order, which can't safely overlap with the encoder writing them.
//
//  • Tiling (time-multiplexing): the frame is split into
//    NUM_TILES_H × NUM_TILES_W tiles, each of size TILE_H × TILE_W. Skip
//    and replay buffers are sized for one tile, so total BRAM scales with
//    TILE_H × TILE_W rather than FEATURE_H × FEATURE_W. The single
//    ConvFormer/GFF/Det/NMS pipeline is reused across tiles. Detections
//    emerge with absolute (frame-relative) coordinates by adding the
//    tile-origin offset to feat_h_idx / feat_w_idx before the head.
//
//  • Stride-2 backpressure: E0/E1 out_ready propagates from the next
//    stage's ready_in only on stride-2-pass cycles. Off-grid tokens are
//    consumed by the local skip-buffer write (always ready), so the
//    encoder never stalls waiting for an off-grid token to be accepted.
//
// Buffer footprint (default TILE_H = TILE_W = 32, MAX_C = 64, DATA_W = 8):
//   skip0_buf: 32×32×64 B =  64 KB
//   skip1_buf: 16×16×64 B =  16 KB
//   btk_buf  :  8× 8×64 B =   4 KB
//   d2_buf   : 16×16×64 B =  16 KB
//   det_buf  : 32×32×64 B =  64 KB
//   Total: ~164 KB — fits Artix-7-100T (135 BRAM36 ≈ 540 KB) with margin.
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;

module e_rodnet_unet_top #(
    // Frame geometry (full input)
    parameter int FEATURE_H     = 64,     // Frame height (power of 2, ≥ 8)
    parameter int FEATURE_W     = 64,     // Frame width  (power of 2, ≥ 8)

    // Tile geometry (unit of pipeline processing). Default = full frame
    // (no tiling). Set to a smaller power-of-two to reduce on-chip BRAM
    // by the ratio (TILE_H*TILE_W) / (FEATURE_H*FEATURE_W) at no
    // throughput cost (one ConvFormer pipeline is reused across tiles).
    parameter int TILE_H        = 32,
    parameter int TILE_W        = 32,

    parameter int IN_CHANNELS   = 16,
    parameter int NUM_CLASSES   = 3,
    parameter int NUM_ANCHORS   = 3,
    parameter int TOP_K         = 100,
    parameter int HISTORY_DEPTH = 8,
    parameter int DATA_WIDTH    = pkg_config::DATA_WIDTH,
    parameter int FRAC_WIDTH    = pkg_config::FRAC_WIDTH,
    parameter int SKIP_CONNECTION_MODE = pkg_config::SKIP_CONNECTION_MODE
    // 0 = sat_add(upsamp, skip) — default, combinational, 1 cy/token
    // 1 = GELU(GN(Linear(Cat[x1, x2]))) — paper Eq.9, ~74 cy/token sequential
)(
    input  logic clk,
    (* max_fanout = 64 *) input  logic rst_n,

    // Configuration
    input  logic                        cfg_enable,
    input  logic [2:0]                  cfg_temporal_mode,
    input  logic [3:0]                  cfg_history_depth,
    input  logic signed [DATA_WIDTH-1:0] cfg_conf_threshold,
    input  logic signed [DATA_WIDTH-1:0] cfg_iou_threshold,
    input  logic [7:0]                  cfg_top_k,

    // Input tokens — one TILE_H × TILE_W tile at a time, raster order.
    // in_frame_last marks the LAST TOKEN OF A TILE. The wrapper internally
    // iterates tiles across the frame and asserts processing_done after
    // the last detection of the last tile.
    input  logic                        in_valid,
    input  logic                        in_frame_last,
    input  logic signed [DATA_WIDTH-1:0] in_data [MAX_C-1:0],
    output logic                        in_ready,

    // Weight loading
    input  logic        weight_valid,
    input  logic [31:0] weight_addr,
    input  logic signed [DATA_WIDTH-1:0] weight_data,
    input  logic [3:0]  weight_layer_sel,

    // Detection output (post-NMS, frame-absolute coordinates)
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

    // ── Sparse CSR Interface (0xB00–0xB3F) ────────────────────────────────────
    input  logic        sparse_csr_req_valid,
    input  logic [11:0] sparse_csr_req_addr,
    input  logic [31:0] sparse_csr_req_wdata,
    input  logic        sparse_csr_req_we,
    output logic [31:0] sparse_csr_req_rdata,
    output logic        sparse_csr_req_ready,

    // ── Debug: detection pipeline observation ─────────────────────────────────
    output logic [31:0] dbg_dh_emit_count,
    output logic [31:0] dbg_nms_out_count,
    output logic        dbg_det_head_valid,
    output logic        dbg_detecting,
    output logic [3:0]  dbg_unet_state,
    output logic        dbg_det_buf_valid,
    output logic        dbg_d1_stage_done,
    output logic [31:0] dbg_outstanding_det,
    output logic        dbg_det_head_feat_valid,
    output logic        dbg_cf_valid_d1,
    output logic        dbg_finalup_valid,
    output logic        dbg_d1_out_cnt
);

    // =========================================================================
    // Geometry derivations
    // =========================================================================
    localparam int H0 = TILE_H;
    localparam int W0 = TILE_W;
    localparam int H1 = TILE_H >> 1;
    localparam int W1 = TILE_W >> 1;
    localparam int H2 = TILE_H >> 2;
    localparam int W2 = TILE_W >> 2;
    localparam int C  = MAX_C;

    localparam int AW0 = $clog2(H0 * W0);
    localparam int AW1 = $clog2(H1 * W1);
    localparam int AW2 = $clog2(H2 * W2);

    localparam int NUM_TILES_H = FEATURE_H / TILE_H;
    localparam int NUM_TILES_W = FEATURE_W / TILE_W;
    localparam int TH_W = (NUM_TILES_H > 1) ? $clog2(NUM_TILES_H) : 1;
    localparam int TW_W = (NUM_TILES_W > 1) ? $clog2(NUM_TILES_W) : 1;

    // ── Sparse CSR bus: all 5 ConvFormer instances share the same bus ──
    // Writes are broadcast to all instances. Reads return data from
    // whichever instance responds (they all see the same address, so
    // any response is valid). OR-reduce ready signals.
    logic [31:0] sparse_csr_req_rdata_enc0, sparse_csr_req_rdata_enc1;
    logic [31:0] sparse_csr_req_rdata_enc2, sparse_csr_req_rdata_dec2, sparse_csr_req_rdata_dec1;
    logic        sparse_csr_req_ready_enc0,  sparse_csr_req_ready_enc1;
    logic        sparse_csr_req_ready_enc2,  sparse_csr_req_ready_dec2,  sparse_csr_req_ready_dec1;

    assign sparse_csr_req_rdata = sparse_csr_req_rdata_enc0 | sparse_csr_req_rdata_enc1
                                 | sparse_csr_req_rdata_enc2 | sparse_csr_req_rdata_dec2
                                 | sparse_csr_req_rdata_dec1;
    assign sparse_csr_req_ready = sparse_csr_req_ready_enc0 & sparse_csr_req_ready_enc1
                                & sparse_csr_req_ready_enc2 & sparse_csr_req_ready_dec2
                                & sparse_csr_req_ready_dec1;

    // -------------------------------------------------------------------------
    //  Per-layer weight-address decode for ConvFormer blocks.
    //
    //  Within each layer's weight stream, bytes are laid out as:
    //      [0                       .. CB_DW_SIZE-1]   depthwise-conv kernels
    //      [CB_DW_SIZE              .. +CB_PW_SIZE-1]  expand pointwise (bank=1)
    //      [CB_DW_SIZE+CB_PW_SIZE   .. +CB_PW_SIZE-1]  project pointwise (bank=2)
    //  Earlier revisions hard-tied pw_wr_bank/addr/ch/k to 0, which left the
    //  expand+project weight ROMs uninitialised (only address 0 ever written).
    // -------------------------------------------------------------------------
    localparam int CB_EXPANDED_C = MAX_C * pkg_config::EXPANSION_FACTOR;
    localparam int CB_DW_SIZE    = MAX_C * pkg_config::MAX_KERNEL_SIZE;
    localparam int CB_PW_SIZE    = CB_EXPANDED_C * MAX_C;   // same for expand & project

    // Decoded per-layer fields (combinational; valid only when weight_valid).
    logic        cb_in_dw, cb_in_exp, cb_in_proj;
    logic [13:0] cb_pw_addr;
    logic [1:0]  cb_pw_bank;
    logic [15:0] cb_dw_ch, cb_dw_k;
    always_comb begin
        cb_in_dw   = (weight_addr <  CB_DW_SIZE);
        cb_in_exp  = (weight_addr >= CB_DW_SIZE) &&
                     (weight_addr <  CB_DW_SIZE + CB_PW_SIZE);
        cb_in_proj = (weight_addr >= CB_DW_SIZE + CB_PW_SIZE) &&
                     (weight_addr <  CB_DW_SIZE + 2*CB_PW_SIZE);
        cb_pw_bank = cb_in_exp ? 2'd1 : (cb_in_proj ? 2'd2 : 2'd0);
        cb_pw_addr = cb_in_exp  ? 14'(weight_addr - CB_DW_SIZE)
                  :  cb_in_proj ? 14'(weight_addr - CB_DW_SIZE - CB_PW_SIZE)
                  : 14'd0;
        cb_dw_ch   = 16'(weight_addr / pkg_config::MAX_KERNEL_SIZE);
        cb_dw_k    = 16'(weight_addr % pkg_config::MAX_KERNEL_SIZE);
    end

    // Stage indices for the cf_* handshake bus
    localparam int SE0 = 0;
    localparam int SE1 = 1;
    localparam int SE2 = 2;
    localparam int SD2 = 3;
    localparam int SD1 = 4;
    localparam int N_STAGES = 5;

    // =========================================================================
    // FSM
    // =========================================================================
    typedef enum logic [3:0] {
        S_IDLE,
        S_TILE_ENC,    // Concurrent E0/E1/E2/GFF — drain a single tile
        S_DEC2,        // Decoder D2 (NN-up + skip1) over the tile
        S_DEC1,        // Decoder D1 (NN-up + skip0) over the tile
        S_NEXT_TILE    // Advance tile coords; back to S_TILE_ENC or S_IDLE (frame done)
    } state_t;
    state_t state;

    // =========================================================================
    // Interleaved-frame detection flag
    // Detection of the current frame runs concurrently with encoding of the
    // next frame, exploiting GFF's dual-buffer accumulation.  Single-banked
    // skip buffers prevent decode/encode overlap, but detection (which only
    // uses det_buf) can safely overlap with next-frame encode.
    // =========================================================================
    logic [$clog2(TOTAL_TILES+1)-1:0] outstanding_det;
    wire  detecting;
    assign detecting = (outstanding_det > 0);

    // =========================================================================
    // Per-stage handshake bus (5-wide: E0, E1, E2, D2, D1)
    // =========================================================================
    logic                        cf_valid_in       [N_STAGES-1:0];
    logic                        cf_frame_last_in  [N_STAGES-1:0];
    logic signed [DATA_WIDTH-1:0] cf_tok_in        [N_STAGES-1:0][MAX_C-1:0];
    logic                        cf_ready_in       [N_STAGES-1:0];
    logic                        cf_valid_out      [N_STAGES-1:0];
    logic                        cf_frame_last_out [N_STAGES-1:0];
    logic signed [DATA_WIDTH-1:0] cf_tok_out       [N_STAGES-1:0][MAX_C-1:0];
    logic                        cf_out_ready      [N_STAGES-1:0];

    // =========================================================================
    // Tile-sized BRAM buffers (dual-banked for encoder/decoder overlap)
    // Bank 0 / 1 alternate per tile: encoder writes bank N%2 while decoder
    // reads bank (N-1)%2, achieving full overlap of next-tile encode with
    // current-tile decode.
    // =========================================================================
    // -------------------------------------------------------------------------
    // U-Net tile buffers — 1D packed-word LUTRAM with ASYNC (combinational) read.
    //
    // The original 3D unpacked arrays `mem[bank][addr][chan]` mapped to flip-
    // flops in Vivado ([Synth 8-5856]: 3D RAM not supported -> registers), and
    // at MAX_C=16/TILE-32 that was 671,744 FF = 5.3x over the FF budget.  Flat-
    // tening each buffer to a 1D unpacked array of packed words (all C channels
    // of one pixel per word) with `(* ram_style="distributed" *)` makes Vivado
    // infer LUTRAM with an ASYNC read — so the decoder's combinational bilinear
    // upsample + skip logic needs NO read-latency retime (the addresses and
    // consume handshake are unchanged from the original 3D-read design).
    //
    // At the arty_a7_fit profile (MAX_C=4, TILE 16) each buffer is only a few
    // Kb (skip0/det = 2*16*16*4*8 = 16 Kb -> a few hundred LUTs of LUTRAM); the
    // 64-bit-half + registered-BRAM-read restructure (needed at MAX_C=16 where
    // buffers are too large for LUTRAM) is deliberately NOT used here, avoiding
    // the bug-prone 1-cycle read retime.  btk/d2 stay 4-replicated so the 4-tap
    // bilinear upsample's 4 simultaneous reads each hit a dedicated LUTRAM.
    // -------------------------------------------------------------------------
    localparam int PW = C*DATA_WIDTH;            // packed pixel word (32b at C=4,DW=8)

    // Channel extractor: returns channel c out of a packed pixel word.
    function automatic logic signed [DATA_WIDTH-1:0] getch(
        input int c, input logic signed [PW-1:0] w);
        getch = w[c*DATA_WIDTH +: DATA_WIDTH];
    endfunction

    // 1D dual-banked LUTRAM.  btk/d2 replicated 4x for the 4-read upsample.
    (* ram_style = "distributed" *) logic signed [PW-1:0] skip0_buf [0:2*H0*W0-1];
    (* ram_style = "distributed" *) logic signed [PW-1:0] skip1_buf [0:2*H1*W1-1];
    (* ram_style = "distributed" *) logic signed [PW-1:0] btk_buf0  [0:2*H2*W2-1];
    (* ram_style = "distributed" *) logic signed [PW-1:0] btk_buf1  [0:2*H2*W2-1];
    (* ram_style = "distributed" *) logic signed [PW-1:0] btk_buf2  [0:2*H2*W2-1];
    (* ram_style = "distributed" *) logic signed [PW-1:0] btk_buf3  [0:2*H2*W2-1];
    (* ram_style = "distributed" *) logic signed [PW-1:0] d2_buf0   [0:2*H1*W1-1];
    (* ram_style = "distributed" *) logic signed [PW-1:0] d2_buf1   [0:2*H1*W1-1];
    (* ram_style = "distributed" *) logic signed [PW-1:0] d2_buf2   [0:2*H1*W1-1];
    (* ram_style = "distributed" *) logic signed [PW-1:0] d2_buf3   [0:2*H1*W1-1];
    (* ram_style = "distributed" *) logic signed [PW-1:0] det_buf   [0:2*H0*W0-1];

    // Combinational packed-word writes (all C channels of one pixel per word).
    logic signed [PW-1:0] skip0_pw, skip1_pw, btk_pw, d2_pw, det_pw;
    always_comb begin
        for (int c = 0; c < C; c++) begin
            skip0_pw[c*DATA_WIDTH +: DATA_WIDTH] = cf_tok_out[SE0][c];
            skip1_pw[c*DATA_WIDTH +: DATA_WIDTH] = cf_tok_out[SE1][c];
            btk_pw  [c*DATA_WIDTH +: DATA_WIDTH] = gff_out[c];
            d2_pw   [c*DATA_WIDTH +: DATA_WIDTH] = cf_tok_out[SD2][c];
            det_pw  [c*DATA_WIDTH +: DATA_WIDTH] = finalup_data_out[c];
        end
    end

    logic [AW0-1:0] skip0_wr_addr, skip0_rd_addr;
    logic [AW1-1:0] skip1_wr_addr, skip1_rd_addr;
    logic [AW2-1:0] btk_wr_addr,   btk_rd_addr;
    logic [AW1-1:0] d2_buf_wr_addr, d2_buf_rd_addr;
    logic           d2_buf_valid;
    logic [AW0-1:0] det_buf_wr_addr, det_buf_rd_addr;
    logic           det_buf_valid;

    // Dual-bank tile tracking
    localparam int TOTAL_TILES = NUM_TILES_H * NUM_TILES_W;
    localparam int TILE_CNT_W  = $clog2(TOTAL_TILES + 1);
    logic [TILE_CNT_W-1:0] enc_tile_cnt;   // tiles fully encoded
    logic [TILE_CNT_W-1:0] dec_tile_cnt;   // tiles fully decoded (through detect)
    logic                  enc_bank;       // bank select for current encoding tile
    logic                  dec_bank;       // bank select for NEXT decoding tile write
    logic                  det_rd_bank;    // bank select for current detection read (snapshot)
    wire                   all_tiles_enc_done = (enc_tile_cnt == TOTAL_TILES);
    wire                   all_tiles_dec_done = (dec_tile_cnt == TOTAL_TILES);

    // Stride-2 position trackers (advance on stage output handshake)
    logic [$clog2(H0)-1:0] e0_h, e1_h;
    logic [$clog2(W0)-1:0] e0_w, e1_w;

    // Per-tile input token counters for E1/E2 (survive stride-2 frame_last loss)
    logic [AW1-1:0] e1_in_cnt;
    logic [AW2-1:0] e2_in_cnt;

    // Decoder NN-2× sub-pixel phase
    logic [1:0] up2_phase;

    // D1 spatial coords (also reused as detection-head feature index)
    logic [$clog2(H0)-1:0] d1_h_out;
    logic [$clog2(W0)-1:0] d1_w_out;

    // Tile coordinates (raster order)
    logic [TH_W-1:0] tile_h_idx;
    logic [TW_W-1:0] tile_w_idx;

    // Per-tile token counters (encoder-output, decoder, etc.)
    logic [AW2:0] e2_out_cnt;     // E2 (= bottleneck) output count → encoder done
    logic [AW1:0] d2_out_cnt;
    logic [AW0:0] d1_out_cnt;

    wire e2_stage_done = (e2_out_cnt == (AW2+1)'(H2*W2));
    wire d2_stage_done = (d2_out_cnt == (AW1+1)'(H1*W1));
    wire d1_stage_done = (d1_out_cnt == (AW0+1)'(H0*W0));

    // =========================================================================
    // Skip-connection mode 1: handshake + weight storage
    // =========================================================================
    generate
        if (SKIP_CONNECTION_MODE == 1) begin : gen_skip_storage
            // Weight BRAM for skip_connect (layer_sel 9 = D2 skip1, 10 = D1 skip0)
            (* ram_style = "block" *)
            logic signed [DATA_WIDTH-1:0] skip_w_lin  [0:1][0:MAX_C-1][0:2*MAX_C-1];
            (* ram_style = "block" *)
            logic signed [DATA_WIDTH-1:0] skip_gn_gamma[0:1][0:MAX_C-1];
            (* ram_style = "block" *)
            logic signed [DATA_WIDTH-1:0] skip_gn_beta [0:1][0:MAX_C-1];

            // D2 (index 1) and D1 (index 0) skip_connect handshake
            logic sc_valid_in  [0:1];
            logic sc_ready_in  [0:1];
            logic sc_valid_out [0:1];
            logic sc_out_ready [0:1];
            logic signed [DATA_WIDTH-1:0] sc_y_out [0:1][0:MAX_C-1];
            logic signed [DATA_WIDTH-1:0] sc_x1_in [0:1][0:MAX_C-1];
            logic signed [DATA_WIDTH-1:0] sc_x2_in [0:1][0:MAX_C-1];

            // Weight loading (simple linear address counter per layer)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    // BRAM contents undefined after reset — weights loaded by SW
                end else begin
                    if (weight_valid && weight_layer_sel == 4'd9) begin
                        // D2 skip weights — addr maps to {out_ch, in_ch}
                        skip_w_lin[1][weight_addr[$clog2(MAX_C)+:$clog2(MAX_C)]][weight_addr[0+:$clog2(2*MAX_C)]] <= weight_data;
                    end
                    if (weight_valid && weight_layer_sel == 4'd10) begin
                        // D1 skip weights
                        skip_w_lin[0][weight_addr[$clog2(MAX_C)+:$clog2(MAX_C)]][weight_addr[0+:$clog2(2*MAX_C)]] <= weight_data;
                    end
                    if (weight_valid && weight_layer_sel == 4'd11) begin
                        // Gamma/beta for both skips (lower bit selects D1/D2, next bit gamma/beta)
                        if (weight_addr[0]) skip_gn_gamma[weight_addr[1]][weight_addr[2+:$clog2(MAX_C)]] <= weight_data;
                        else                skip_gn_beta [weight_addr[1]][weight_addr[2+:$clog2(MAX_C)]] <= weight_data;
                    end
                end
            end
        end
    endgenerate

    // Feed / output decoupling signals (used in FSM regardless of mode)
    logic d2_feed_valid, d2_feed_ready;
    logic d1_feed_valid, d1_feed_ready;

    // =========================================================================
    // ConvFormer instances (5) — wired through cf_* arrays
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] e0_tw [7:0][MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] e1_tw [7:0][MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] e2_tw [7:0][MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] d2_tw [7:0][MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] d1_tw [7:0][MAX_C-1:0];
    always_comb begin
        for (int t = 0; t < 8; t++) for (int c = 0; c < MAX_C; c++) begin
            e0_tw[t][c] = '0; e1_tw[t][c] = '0; e2_tw[t][c] = '0;
            d2_tw[t][c] = '0; d1_tw[t][c] = '0;
        end
    end


    // =========================================================================
    // SSF Spatial Window Buffer + SSF Module (inserted before E0)
    // =========================================================================
    // Forms 3×3 spatial windows from raster-order tile stream using 3 BRAM line
    // buffers in rotating-pointer fashion. Temporal dimension is zero-padded.
    // Window latency = TILE_W + 1 cycles; frame_last delayed to match.
    // -------------------------------------------------------------------------

    // 3 BRAM line buffers (rotating pointers)
    (* ram_style = "block" *)
    logic signed [DATA_WIDTH-1:0] ssf_lb [0:2][0:TILE_W-1][0:MAX_C-1];
    logic [1:0] ssf_lb_ptr [0:2];        // 0=write, 1=read row-1, 2=read row-2
    logic [$clog2(TILE_W)-1:0] ssf_w;
    logic [$clog2(TILE_H)-1:0] ssf_h;
    logic ssf_window_valid;

    // Horizontal shift registers (per channel)
    logic signed [DATA_WIDTH-1:0] ssf_win_top [MAX_C-1:0][0:2];
    logic signed [DATA_WIDTH-1:0] ssf_win_mid [MAX_C-1:0][0:2];
    logic signed [DATA_WIDTH-1:0] ssf_win_bot [MAX_C-1:0][0:2];

    // Flattened BRAM read data
    logic signed [DATA_WIDTH-1:0] ssf_rd_mid [MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] ssf_rd_top [MAX_C-1:0];

    // BRAM read + write combinational logic
    always_comb begin
        for (int c = 0; c < MAX_C; c++) begin
            case (ssf_lb_ptr[1])
                2'd0: ssf_rd_mid[c] = ssf_lb[0][ssf_w][c];
                2'd1: ssf_rd_mid[c] = ssf_lb[1][ssf_w][c];
                2'd2: ssf_rd_mid[c] = ssf_lb[2][ssf_w][c];
                default: ssf_rd_mid[c] = '0;
            endcase
            case (ssf_lb_ptr[2])
                2'd0: ssf_rd_top[c] = ssf_lb[0][ssf_w][c];
                2'd1: ssf_rd_top[c] = ssf_lb[1][ssf_w][c];
                2'd2: ssf_rd_top[c] = ssf_lb[2][ssf_w][c];
                default: ssf_rd_top[c] = '0;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ssf_w <= '0;
            ssf_h <= '0;
            ssf_window_valid <= 1'b0;
            ssf_lb_ptr[0] <= 2'd0;
            ssf_lb_ptr[1] <= 2'd1;
            ssf_lb_ptr[2] <= 2'd2;
            for (int c = 0; c < MAX_C; c++) begin
                ssf_win_top[c][0] <= '0; ssf_win_top[c][1] <= '0; ssf_win_top[c][2] <= '0;
                ssf_win_mid[c][0] <= '0; ssf_win_mid[c][1] <= '0; ssf_win_mid[c][2] <= '0;
                ssf_win_bot[c][0] <= '0; ssf_win_bot[c][1] <= '0; ssf_win_bot[c][2] <= '0;
            end
        end else if (state == S_NEXT_TILE || state == S_IDLE) begin
            ssf_w <= '0;
            ssf_h <= '0;
            ssf_window_valid <= 1'b0;
            ssf_lb_ptr[0] <= 2'd0;
            ssf_lb_ptr[1] <= 2'd1;
            ssf_lb_ptr[2] <= 2'd2;
        end else if (in_valid && in_ready) begin
            // Update horizontal shift registers
            for (int c = 0; c < MAX_C; c++) begin
                ssf_win_top[c][2] <= ssf_win_top[c][1];
                ssf_win_top[c][1] <= ssf_win_top[c][0];
                ssf_win_top[c][0] <= ssf_rd_top[c];

                ssf_win_mid[c][2] <= ssf_win_mid[c][1];
                ssf_win_mid[c][1] <= ssf_win_mid[c][0];
                ssf_win_mid[c][0] <= ssf_rd_mid[c];

                ssf_win_bot[c][2] <= ssf_win_bot[c][1];
                ssf_win_bot[c][1] <= ssf_win_bot[c][0];
                ssf_win_bot[c][0] <= in_data[c];
            end

            // Write current pixel to active line buffer (ptr cycles 0->1->2->0; 3 is unreachable)
            for (int c = 0; c < MAX_C; c++) begin
                case (ssf_lb_ptr[0])
                    2'd0:    ssf_lb[0][ssf_w][c] <= in_data[c];
                    2'd1:    ssf_lb[1][ssf_w][c] <= in_data[c];
                    2'd2:    ssf_lb[2][ssf_w][c] <= in_data[c];
                    default: ssf_lb[0][ssf_w][c] <= in_data[c];   // ptr==3 never occurs
                endcase
            end

            // Advance column / row
            if (ssf_w == $clog2(TILE_W)'(TILE_W-1)) begin
                ssf_w <= '0;
                ssf_h <= ssf_h + 1;
                // Rotate line-buffer pointers at end of row
                ssf_lb_ptr[0] <= ssf_lb_ptr[2];
                ssf_lb_ptr[1] <= ssf_lb_ptr[0];
                ssf_lb_ptr[2] <= ssf_lb_ptr[1];
            end else begin
                ssf_w <= ssf_w + 1;
            end

            // Window is valid after 2 rows + 2 pixels of warmup
            ssf_window_valid <= (ssf_h >= 2'd2) ||
                                (ssf_h == 2'd1 && ssf_w >= 2'd1);
        end
    end

    // Extract 3x3 spatial window (zero-padded at edges)
    // Temporal dimension (elements 9-26) zero-padded for single-frame input
    logic signed [DATA_WIDTH-1:0] ssf_x_1x1 [MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] ssf_x_3x3 [MAX_C-1:0][26:0];

    always_comb begin
        for (int c = 0; c < MAX_C; c++) begin
            // 1x1 branch: center pixel = middle row, center column
            ssf_x_1x1[c] = ssf_win_mid[c][1];

            // Spatial 3x3 window (elements 0-8)
            // Top row (row h-2)
            ssf_x_3x3[c][0] = ssf_win_top[c][2];  // top-left
            ssf_x_3x3[c][1] = ssf_win_top[c][1];  // top-mid
            ssf_x_3x3[c][2] = ssf_win_top[c][0];  // top-right
            // Mid row (row h-1)
            ssf_x_3x3[c][3] = ssf_win_mid[c][2];  // mid-left
            ssf_x_3x3[c][4] = ssf_win_mid[c][1];  // center
            ssf_x_3x3[c][5] = ssf_win_mid[c][0];  // mid-right
            // Bot row (row h)
            ssf_x_3x3[c][6] = ssf_win_bot[c][2];  // bot-left
            ssf_x_3x3[c][7] = ssf_win_bot[c][1];  // bot-mid
            ssf_x_3x3[c][8] = ssf_win_bot[c][0];  // bot-right

            // Temporal slices (3x3 each) zero-padded
            for (int k = 9; k < 27; k++) ssf_x_3x3[c][k] = '0;
        end
    end

    // Frame-last delay shift register (TILE_W + 2 cycles to match window latency)
    localparam int SSF_LAST_DLY = TILE_W + 2;
    logic [SSF_LAST_DLY-1:0] ssf_last_sr;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) ssf_last_sr <= '0;
        else if (state == S_NEXT_TILE || state == S_IDLE) ssf_last_sr <= '0;
        else if (in_valid && in_ready) ssf_last_sr <= {ssf_last_sr[SSF_LAST_DLY-2:0], in_frame_last};
    end
    logic ssf_last_out;
    assign ssf_last_out = ssf_last_sr[SSF_LAST_DLY-1];

    // SSF weight / BN param defaults (identity / zero until runtime loading)
    // SSF weights are loaded via weight DMA (layer_sel 13).
    // Addressing (weight_addr[7:0] = flat element index):
    //   [0, MAX_C-1]               → w_1x1[idx]
    //   [MAX_C, MAX_C+MAX_C*27-1]  → w_3x3[c][k]  where c=(idx-MAX_C)/27, k=(idx-MAX_C)%27
    //   [MAX_C+MAX_C*27, ...]      → w_linear[c][c2] where c=(idx-b2)/MAX_C, c2=(idx-b2)%MAX_C
    logic signed [DATA_WIDTH-1:0] ssf_w_1x1 [MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] ssf_w_3x3 [MAX_C-1:0][26:0];
    logic signed [DATA_WIDTH-1:0] ssf_w_linear [MAX_C-1:0][MAX_C-1:0];
    logic signed [ACC_WIDTH-1:0] ssf_bn_gamma [MAX_C-1:0];
    logic signed [ACC_WIDTH-1:0] ssf_bn_beta  [MAX_C-1:0];
    logic signed [ACC_WIDTH-1:0] ssf_bn_mean  [MAX_C-1:0];
    logic signed [ACC_WIDTH-1:0] ssf_bn_inv_std [MAX_C-1:0];

    // BN params default to identity (1.0 gamma, 0.0 beta/mean, 1.0 inv_std)
    always_comb begin
        for (int c = 0; c < MAX_C; c++) begin
            ssf_bn_gamma[c] = ACC_WIDTH'(1 << FRAC_WIDTH);
            ssf_bn_beta[c]  = '0;
            ssf_bn_mean[c]  = '0;
            ssf_bn_inv_std[c] = ACC_WIDTH'(1 << FRAC_WIDTH);
        end
    end

    // SSF weight loading via weight DMA (layer_sel 13)
    localparam int SSF_W1_SIZE = MAX_C;
    localparam int SSF_W3_SIZE = MAX_C * 27;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int c = 0; c < MAX_C; c++) begin
                ssf_w_1x1[c] <= '0;
                for (int k = 0; k < 27; k++) ssf_w_3x3[c][k] <= '0;
                for (int c2 = 0; c2 < MAX_C; c2++) ssf_w_linear[c][c2] <= '0;
            end
        end else if (weight_valid && weight_layer_sel == 4'd13) begin
            if (weight_addr[7:0] < SSF_W1_SIZE) begin
                ssf_w_1x1[weight_addr[7:0]] <= weight_data;
            end else if (weight_addr[7:0] < SSF_W1_SIZE + SSF_W3_SIZE) begin
                automatic int offset = weight_addr[7:0] - SSF_W1_SIZE;
                ssf_w_3x3[offset / 27][offset % 27] <= weight_data;
            end else begin
                automatic int offset = weight_addr[7:0] - SSF_W1_SIZE - SSF_W3_SIZE;
                ssf_w_linear[offset / MAX_C][offset % MAX_C] <= weight_data;
            end
        end
    end

    // SSF module instance
    logic                        ssf_out_valid;
    logic                        ssf_out_ready;
    logic                        ssf_out_last;
    logic signed [ACC_WIDTH-1:0] ssf_y_out [MAX_C-1:0];

    ssf_module #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH),
        .CHANNELS(MAX_C)
    ) u_ssf (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(ssf_window_valid),
        .out_valid(ssf_out_valid),
        .ready_in(cf_ready_in[SE0]),      // downstream E0 ready
        .ready_out(ssf_out_ready),        // upstream ready (tells window buffer)
        .last_in(ssf_last_out),
        .last_out(ssf_out_last),
        .x_1x1(ssf_x_1x1),
        .x_3x3(ssf_x_3x3),
        .w_1x1(ssf_w_1x1),
        .w_3x3(ssf_w_3x3),
        .w_linear(ssf_w_linear),
        .bn1_gamma(ssf_bn_gamma),
        .bn1_beta(ssf_bn_beta),
        .bn1_mean(ssf_bn_mean),
        .bn1_inv_std(ssf_bn_inv_std),
        .bn2_gamma(ssf_bn_gamma),
        .bn2_beta(ssf_bn_beta),
        .bn2_mean(ssf_bn_mean),
        .bn2_inv_std(ssf_bn_inv_std),
        .bn3_gamma(ssf_bn_gamma),
        .bn3_beta(ssf_bn_beta),
        .bn3_mean(ssf_bn_mean),
        .bn3_inv_std(ssf_bn_inv_std),
        .y_out(ssf_y_out)
    );

    // Re-quantize SSF output (ACC_WIDTH → DATA_WIDTH) with saturation
    logic signed [DATA_WIDTH-1:0] ssf_y_q [MAX_C-1:0];
    always_comb begin
        for (int c = 0; c < MAX_C; c++) begin
            automatic logic signed [ACC_WIDTH-1:0] val = ssf_y_out[c] >>> FRAC_WIDTH;
            if (val >  ACC_WIDTH'({1'b0, {(DATA_WIDTH-1){1'b1}}}))
                ssf_y_q[c] = {1'b0, {(DATA_WIDTH-1){1'b1}}};
            else if (val < -ACC_WIDTH'(1 << (DATA_WIDTH-1)))
                ssf_y_q[c] = {1'b1, {(DATA_WIDTH-1){1'b0}}};
            else
                ssf_y_q[c] = DATA_WIDTH'(val);
        end
    end

    // -------------------------------------------------------------------------
    temporal_convformer_block #(.HISTORY_DEPTH(HISTORY_DEPTH), .TOTAL_LATENCY(pkg_config::TOTAL_LATENCY)) u_enc0 (
        .clk(clk), .rst_n(rst_n),
        .cfg_temporal_mode(cfg_temporal_mode), .cfg_history_depth(cfg_history_depth),
        .cfg_kernel_t(4'd3), .cfg_c(16'(C)), .cfg_kernel_size(16'd9),
        .cfg_act_sel(3'd4),
        .cfg_scale_in(32'd1 << FRAC_WIDTH), .cfg_bias_in('0),
        .cfg_clk_gate(1'b0), .cfg_enable_temporal(cfg_enable),
        .valid_in(cf_valid_in[SE0]), .frame_last_in(cf_frame_last_in[SE0]),
        .token_in(cf_tok_in[SE0]),    .ready_in(cf_ready_in[SE0]),
        .valid_out(cf_valid_out[SE0]),.frame_last_out(cf_frame_last_out[SE0]),
        .token_out(cf_tok_out[SE0]),  .out_ready(cf_out_ready[SE0]),
        .pw_wr_en(weight_valid && weight_layer_sel == 4'd0 && (cb_in_exp || cb_in_proj)),
        .pw_wr_bank(cb_pw_bank), .pw_wr_addr(cb_pw_addr), .pw_wr_data(weight_data),
        .dw_wr_en(weight_valid && weight_layer_sel == 4'd0 && cb_in_dw),
        .dw_wr_ch(cb_dw_ch), .dw_wr_k(cb_dw_k), .dw_wr_data(weight_data),
        .temporal_weights(e0_tw),
        .perf_cycle_count(), .perf_stall_count(), .perf_token_count(),
        .sparse_csr_req_valid(sparse_csr_req_valid),
        .sparse_csr_req_addr(sparse_csr_req_addr),
        .sparse_csr_req_wdata(sparse_csr_req_wdata),
        .sparse_csr_req_we(sparse_csr_req_we),
        .sparse_csr_req_rdata(sparse_csr_req_rdata_enc0),
        .sparse_csr_req_ready(sparse_csr_req_ready_enc0)
    );

    temporal_convformer_block #(.HISTORY_DEPTH(HISTORY_DEPTH), .TOTAL_LATENCY(pkg_config::TOTAL_LATENCY)) u_enc1 (
        .clk(clk), .rst_n(rst_n),
        .cfg_temporal_mode(cfg_temporal_mode), .cfg_history_depth(cfg_history_depth),
        .cfg_kernel_t(4'd3), .cfg_c(16'(C)), .cfg_kernel_size(16'd9),
        .cfg_act_sel(3'd4),
        .cfg_scale_in(32'd1 << FRAC_WIDTH), .cfg_bias_in('0),
        .cfg_clk_gate(1'b0), .cfg_enable_temporal(cfg_enable),
        .valid_in(cf_valid_in[SE1]), .frame_last_in(cf_frame_last_in[SE1]),
        .token_in(cf_tok_in[SE1]),    .ready_in(cf_ready_in[SE1]),
        .valid_out(cf_valid_out[SE1]),.frame_last_out(cf_frame_last_out[SE1]),
        .token_out(cf_tok_out[SE1]),  .out_ready(cf_out_ready[SE1]),
        .pw_wr_en(weight_valid && weight_layer_sel == 4'd1 && (cb_in_exp || cb_in_proj)),
        .pw_wr_bank(cb_pw_bank), .pw_wr_addr(cb_pw_addr), .pw_wr_data(weight_data),
        .dw_wr_en(weight_valid && weight_layer_sel == 4'd1 && cb_in_dw),
        .dw_wr_ch(cb_dw_ch), .dw_wr_k(cb_dw_k), .dw_wr_data(weight_data),
        .temporal_weights(e1_tw),
        .perf_cycle_count(), .perf_stall_count(), .perf_token_count(),
        .sparse_csr_req_valid(sparse_csr_req_valid),
        .sparse_csr_req_addr(sparse_csr_req_addr),
        .sparse_csr_req_wdata(sparse_csr_req_wdata),
        .sparse_csr_req_we(sparse_csr_req_we),
        .sparse_csr_req_rdata(sparse_csr_req_rdata_enc1),
        .sparse_csr_req_ready(sparse_csr_req_ready_enc1)
    );

    temporal_convformer_block #(.HISTORY_DEPTH(HISTORY_DEPTH), .TOTAL_LATENCY(pkg_config::TOTAL_LATENCY)) u_enc2 (
        .clk(clk), .rst_n(rst_n),
        .cfg_temporal_mode(cfg_temporal_mode), .cfg_history_depth(cfg_history_depth),
        .cfg_kernel_t(4'd3), .cfg_c(16'(C)), .cfg_kernel_size(16'd9),
        .cfg_act_sel(3'd4),
        .cfg_scale_in(32'd1 << FRAC_WIDTH), .cfg_bias_in('0),
        .cfg_clk_gate(1'b0), .cfg_enable_temporal(cfg_enable),
        .valid_in(cf_valid_in[SE2]), .frame_last_in(cf_frame_last_in[SE2]),
        .token_in(cf_tok_in[SE2]),    .ready_in(cf_ready_in[SE2]),
        .valid_out(cf_valid_out[SE2]),.frame_last_out(cf_frame_last_out[SE2]),
        .token_out(cf_tok_out[SE2]),  .out_ready(cf_out_ready[SE2]),
        .pw_wr_en(weight_valid && weight_layer_sel == 4'd2 && (cb_in_exp || cb_in_proj)),
        .pw_wr_bank(cb_pw_bank), .pw_wr_addr(cb_pw_addr), .pw_wr_data(weight_data),
        .dw_wr_en(weight_valid && weight_layer_sel == 4'd2 && cb_in_dw),
        .dw_wr_ch(cb_dw_ch), .dw_wr_k(cb_dw_k), .dw_wr_data(weight_data),
        .temporal_weights(e2_tw),
        .perf_cycle_count(), .perf_stall_count(), .perf_token_count(),
        .sparse_csr_req_valid(sparse_csr_req_valid),
        .sparse_csr_req_addr(sparse_csr_req_addr),
        .sparse_csr_req_wdata(sparse_csr_req_wdata),
        .sparse_csr_req_we(sparse_csr_req_we),
        .sparse_csr_req_rdata(sparse_csr_req_rdata_enc2),
        .sparse_csr_req_ready(sparse_csr_req_ready_enc2)
    );

    temporal_convformer_block #(.HISTORY_DEPTH(HISTORY_DEPTH), .TOTAL_LATENCY(pkg_config::TOTAL_LATENCY)) u_dec2 (
        .clk(clk), .rst_n(rst_n),
        .cfg_temporal_mode(cfg_temporal_mode), .cfg_history_depth(cfg_history_depth),
        .cfg_kernel_t(4'd3), .cfg_c(16'(C)), .cfg_kernel_size(16'd9),
        .cfg_act_sel(3'd4),
        .cfg_scale_in(32'd1 << FRAC_WIDTH), .cfg_bias_in('0),
        .cfg_clk_gate(1'b0), .cfg_enable_temporal(cfg_enable),
        .valid_in(cf_valid_in[SD2]), .frame_last_in(cf_frame_last_in[SD2]),
        .token_in(cf_tok_in[SD2]),    .ready_in(cf_ready_in[SD2]),
        .valid_out(cf_valid_out[SD2]),.frame_last_out(cf_frame_last_out[SD2]),
        .token_out(cf_tok_out[SD2]),  .out_ready(cf_out_ready[SD2]),
        .pw_wr_en(weight_valid && weight_layer_sel == 4'd5 && (cb_in_exp || cb_in_proj)),
        .pw_wr_bank(cb_pw_bank), .pw_wr_addr(cb_pw_addr), .pw_wr_data(weight_data),
        .dw_wr_en(weight_valid && weight_layer_sel == 4'd5 && cb_in_dw),
        .dw_wr_ch(cb_dw_ch), .dw_wr_k(cb_dw_k), .dw_wr_data(weight_data),
        .temporal_weights(d2_tw),
        .perf_cycle_count(), .perf_stall_count(), .perf_token_count(),
        .sparse_csr_req_valid(sparse_csr_req_valid),
        .sparse_csr_req_addr(sparse_csr_req_addr),
        .sparse_csr_req_wdata(sparse_csr_req_wdata),
        .sparse_csr_req_we(sparse_csr_req_we),
        .sparse_csr_req_rdata(sparse_csr_req_rdata_dec2),
        .sparse_csr_req_ready(sparse_csr_req_ready_dec2)
    );

    temporal_convformer_block #(.HISTORY_DEPTH(HISTORY_DEPTH), .TOTAL_LATENCY(pkg_config::TOTAL_LATENCY)) u_dec1 (
        .clk(clk), .rst_n(rst_n),
        .cfg_temporal_mode(cfg_temporal_mode), .cfg_history_depth(cfg_history_depth),
        .cfg_kernel_t(4'd3), .cfg_c(16'(C)), .cfg_kernel_size(16'd9),
        .cfg_act_sel(3'd4),
        .cfg_scale_in(32'd1 << FRAC_WIDTH), .cfg_bias_in('0),
        .cfg_clk_gate(1'b0), .cfg_enable_temporal(cfg_enable),
        .valid_in(cf_valid_in[SD1]), .frame_last_in(cf_frame_last_in[SD1]),
        .token_in(cf_tok_in[SD1]),    .ready_in(cf_ready_in[SD1]),
        .valid_out(cf_valid_out[SD1]),.frame_last_out(cf_frame_last_out[SD1]),
        .token_out(cf_tok_out[SD1]),  .out_ready(cf_out_ready[SD1]),
        .pw_wr_en(weight_valid && weight_layer_sel == 4'd6 && (cb_in_exp || cb_in_proj)),
        .pw_wr_bank(cb_pw_bank), .pw_wr_addr(cb_pw_addr), .pw_wr_data(weight_data),
        .dw_wr_en(weight_valid && weight_layer_sel == 4'd6 && cb_in_dw),
        .dw_wr_ch(cb_dw_ch), .dw_wr_k(cb_dw_k), .dw_wr_data(weight_data),
        .temporal_weights(d1_tw),
        .perf_cycle_count(), .perf_stall_count(), .perf_token_count(),
        .sparse_csr_req_valid(sparse_csr_req_valid),
        .sparse_csr_req_addr(sparse_csr_req_addr),
        .sparse_csr_req_wdata(sparse_csr_req_wdata),
        .sparse_csr_req_we(sparse_csr_req_we),
        .sparse_csr_req_rdata(sparse_csr_req_rdata_dec1),
        .sparse_csr_req_ready(sparse_csr_req_ready_dec1)
    );

    // =========================================================================
    // Temporal Attention at Bottleneck (E2 → [temporal_buffer → attention] → GFF)
    //
    // Synthesis-time enable: EN_TEMPORAL_ATTENTION (from pkg_config).
    // When disabled (0), E2 output bypasses temporal buffer and attention and
    // connects directly to GFF input. Saves all temporal buffer BRAM +
    // attention logic (~15-25 K LUT depending on HEAD_DIM/MAC_UNROLL).
    // =========================================================================
    logic                        bottleneck_attn_valid;
    logic signed [DATA_WIDTH-1:0] bottleneck_attn_data [0:MAX_C-1];

    generate
        if (EN_TEMPORAL_ATTENTION) begin : gen_temporal_attention
            // Runtime bypass: when cfg_temporal_mode==0 (MODE_NONE), route
            // E2 output directly to GFF input, skipping temporal buffer and
            // attention. The temporal buffer requires HISTORY_DEPTH frames
            // before outputting — single-tile tests would deadlock.
            logic temporal_bypass;
            assign temporal_bypass = (cfg_temporal_mode == 3'b000);

            logic                        bottleneck_buf_valid_out;
            logic [HISTORY_DEPTH-1:0]   bottleneck_buf_mask;
            logic signed [DATA_WIDTH-1:0] bottleneck_buf_data [HISTORY_DEPTH-1:0][MAX_C-1:0];
            logic                        bottleneck_buf_ready;
            logic                        bottleneck_attn_ready;
            logic                        bottleneck_attn_valid_int;
            logic signed [DATA_WIDTH-1:0] bottleneck_attn_data_int [0:MAX_C-1];

            // Map buffer [HISTORY_DEPTH-1:0] to attention [0:HISTORY_DEPTH-1] format
            logic signed [DATA_WIDTH-1:0] attn_query_data [0:MAX_C-1];
            logic signed [DATA_WIDTH-1:0] attn_history_data [0:HISTORY_DEPTH-1][0:MAX_C-1];

            always_comb begin
                for (int c = 0; c < MAX_C; c++)
                    attn_query_data[c] = bottleneck_buf_data[0][c];
                for (int t = 0; t < HISTORY_DEPTH; t++)
                    for (int c = 0; c < MAX_C; c++)
                        attn_history_data[t][c] = bottleneck_buf_data[t][c];
            end

            // Temporal buffer: store E2 outputs across frames
            temporal_buffer #(
                .HISTORY_DEPTH(HISTORY_DEPTH),
                .CHANNELS(MAX_C),
                .DATA_W(DATA_WIDTH)
            ) u_bottleneck_buffer (
                .clk(clk),
                .rst_n(rst_n),
                .cfg_history_depth(cfg_history_depth),
                .cfg_enable(cfg_enable),
                .frame_valid_in(cf_valid_out[SE2]),
                .frame_last(cf_frame_last_out[SE2]),
                .frame_data_in(cf_tok_out[SE2]),
                .frame_ready_in(bottleneck_buf_ready),
                .history_valid_out(bottleneck_buf_valid_out),
                .history_valid_mask(bottleneck_buf_mask),
                .history_data(bottleneck_buf_data),
                .history_ready_in(bottleneck_attn_ready),
                .frame_start(cf_frame_last_out[SE2]),
                .buffer_full(),
                .frame_count()
            );

            // Temporal attention: multi-head self-attention on bottleneck features
            temporal_attention_enhanced_v2 #(
                .HISTORY_DEPTH(HISTORY_DEPTH),
                .CHANNELS(MAX_C),
                .NUM_HEADS(pkg_config::NUM_HEADS),
                .HEAD_DIM(pkg_config::HEAD_DIM),
                .DATA_W(DATA_WIDTH),
                .FRAC_W(pkg_module_config::TEMPORAL_ATTENTION_ENHANCED_FRAC_WIDTH),
                .ACC_W(ACC_WIDTH),
                .WEIGHT_LOAD_MODE(pkg_config::WEIGHT_LOAD_MODE),
                .MAC_UNROLL(pkg_config::MAC_UNROLL),
                .OUT_MAC_UNROLL(pkg_config::OUT_MAC_UNROLL),
                .ATTN_HEADS_PER_CYCLE(pkg_config::ATTN_HEADS_PER_CYCLE),
                .HD_UNROLL(pkg_config::HD_UNROLL),
                .CH_OUT_UNROLL(pkg_config::CH_OUT_UNROLL),
                .SCORE_HIST_UNROLL(pkg_config::SCORE_HIST_UNROLL),
                .ENABLE_RESIDUAL(pkg_config::ENABLE_RESIDUAL),
                .EXP_UNROLL(pkg_config::EXP_UNROLL),
                .ENABLE_TEMPERATURE(pkg_config::ENABLE_TEMPERATURE),
                .ENABLE_MASK(pkg_config::ENABLE_MASK),
                .QK_WEIGHT_SHARING(pkg_config::QK_WEIGHT_SHARING)
            ) u_bottleneck_attention (
                .clk(clk),
                .rst_n(rst_n),
                .cfg_attention_mode(3'b000),
                .cfg_history_depth(cfg_history_depth),
                .cfg_local_window(4'd3),
                .cfg_enable(cfg_enable),
                .cfg_temperature(32'(1 << FRAC_WIDTH)),
                .query_valid_in(bottleneck_buf_valid_out),
                .query_data(attn_query_data),
                // BUG-EROD-UNT-001 (FIXED): was `.query_ready_in(bottleneck_buf_ready)`
                //   — that drove `bottleneck_buf_ready` from two outputs (the
                //   temporal_buffer's frame_ready_in AND the attention's
                //   query_ready_in) → MULTIDRIVEN.  And `bottleneck_attn_ready`
                //   (the attention's ready output, read by the buffer's
                //   history_ready_in below) had no driver at all (floating).
                //   Connecting the attention's query_ready_in to
                //   bottleneck_attn_ready fixes both bugs with one edit.
                .query_ready_in(bottleneck_attn_ready),
                .history_valid_in(bottleneck_buf_valid_out),
                .history_valid_mask(bottleneck_buf_mask),
                .history_data(attn_history_data),
                .weight_valid(weight_valid && weight_layer_sel == 4'd4),
                .weight_addr(weight_addr[15:0]),
                .weight_data(weight_data),
                .attended_valid_out(bottleneck_attn_valid_int),
                .attended_data_out(bottleneck_attn_data_int),
                .attended_ready_in(gff_ready_in)
            );

            // E2 out_ready follows temporal buffer backpressure (which cascades
            // from attention → GFF backpressure). In bypass mode, connect
            // directly to GFF readiness.
            assign cf_out_ready[SE2] = temporal_bypass ? gff_ready_in : bottleneck_buf_ready;

            // Bypass mux: route E2 output directly to GFF when temporal is off
            assign bottleneck_attn_valid = temporal_bypass ? cf_valid_out[SE2] : bottleneck_attn_valid_int;
            always_comb begin
                for (int c = 0; c < MAX_C; c++)
                    bottleneck_attn_data[c] = temporal_bypass ? cf_tok_out[SE2][c] : bottleneck_attn_data_int[c];
            end

        end else begin : gen_no_temporal_attention
            // Bypass: E2 output goes directly to GFF input.
            // No temporal buffer or attention instantiated — saves all BRAM + logic.
            assign bottleneck_attn_valid = cf_valid_out[SE2];
            always_comb begin
                for (int c = 0; c < MAX_C; c++)
                    bottleneck_attn_data[c] = cf_tok_out[SE2][c];
            end

            // E2 out_ready connects directly to GFF readiness.
            assign cf_out_ready[SE2] = gff_ready_in;
        end
    endgenerate

    // =========================================================================
    // GFF at bottleneck (after attention)
    //
    // Synthesis-time enable: EN_GFF (from pkg_config).
    // When disabled (0), the GFF module is not instantiated and attention
    // output bypasses directly to the bottleneck buffer. Saves all GFF logic
    // (AvgPool3D + L2-Norm + Fused Projection + Unpool) — ~15-25 K LUT.
    // =========================================================================
    logic                        gff_valid_out;
    logic signed [DATA_WIDTH-1:0] gff_out [MAX_C-1:0];

    // GFF streaming-ready output (tells upstream when GFF can accept)
    logic gff_ready_in;

    generate
        if (EN_GFF) begin : gen_gff
            // Width adapter: upstream features are 8-bit (DATA_WIDTH), but GFF
            // internally processes at GFF_DATA_WIDTH (16-bit) for AvgPool+L2Norm.
            localparam int GFF_W = pkg_module_config::GFF_DATA_WIDTH;

            logic                        gff_valid_in, gff_valid_out_last;
            logic signed [GFF_W-1:0]    gff_in_wide [MAX_C-1:0];
            logic signed [GFF_W-1:0]    gff_out_wide [MAX_C-1:0];

            // Zero-extend 8-bit upstream features to GFF internal width
            always_comb for (int c = 0; c < MAX_C; c++) gff_in_wide[c] = GFF_W'(bottleneck_attn_data[c]);

            assign gff_valid_in = bottleneck_attn_valid;

            // Saturating truncate GFF 16-bit output back to 8-bit for downstream CNN
            always_comb begin
                for (int c = 0; c < MAX_C; c++) begin
                    automatic logic signed [GFF_W-1:0] val = gff_out_wide[c];
                    if (val >  GFF_W'({1'b0, {(DATA_WIDTH-1){1'b1}}}))
                        gff_out[c] = {1'b0, {(DATA_WIDTH-1){1'b1}}};
                    else if (val < -GFF_W'(1 << (DATA_WIDTH-1)))
                        gff_out[c] = {1'b1, {(DATA_WIDTH-1){1'b0}}};
                    else
                        gff_out[c] = DATA_WIDTH'(val);
                end
            end

            logic signed [DATA_WIDTH-1:0] l2norm_scale_reg;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)                                          l2norm_scale_reg <= '0;
                else if (weight_valid && weight_layer_sel == 4'd12)  l2norm_scale_reg <= weight_data;
            end

            // GFF FRAC_WIDTH must be DATA_WIDTH/2 — the GFF hierarchy uses
            // Q(N/2).(N/2) internally (sigmoid/GELU ROM ranges, l2norm rsqrt
            // seed, inline linear extraction all assume FRAC_W = DATA_W/2).
            // Passing the system FRAC_WIDTH=16 here was a bug — see
            // memory/gff_qformat_fix_2026-05-05.md.
            gff #(
                .DATA_WIDTH(GFF_W), .FRAC_WIDTH(GFF_W/2), .STUB_BEHAVIOR(1'b0),
                .FEATURE_H(H2), .FEATURE_W(W2)
            ) u_gff (
                .clk(clk), .rst_n(rst_n),
                .cfg_enable(cfg_enable),
                .valid_in(gff_valid_in),
                .valid_in_last(cf_frame_last_out[SE2]),
                .data_in(gff_in_wide),
                .ready_in(gff_ready_in),
                .valid_out(gff_valid_out),
                .valid_out_last(gff_valid_out_last),
                .data_out(gff_out_wide),
                .out_ready(1'b1),
                .weight_valid(weight_valid && weight_layer_sel == 4'd3),
                .weight_addr(weight_addr),
                .weight_data(weight_data),
                .bias_valid(1'b0), .bias_addr('0), .bias_data('0),
                .l2norm_scale(l2norm_scale_reg),
                .perf_cycle_count()
            );
        end else begin : gen_no_gff
            // Bypass: attention/E2 output goes directly to bottleneck buffer.
            // No GFF instantiated — saves AvgPool, L2-Norm, Fused Projection, Unpool.
            assign gff_valid_out = bottleneck_attn_valid;
            always_comb begin
                for (int c = 0; c < MAX_C; c++)
                    gff_out[c] = bottleneck_attn_data[c];   // already 8-bit
            end
            assign gff_ready_in = 1'b1;     // always ready (no GFF to stall)
        end
    endgenerate

    // =========================================================================
    // Saturating INT8 add (residual skip)
    // =========================================================================
    function automatic logic signed [DATA_WIDTH-1:0] sat_add(
        input logic signed [DATA_WIDTH-1:0] a, b
    );
        automatic logic signed [DATA_WIDTH:0] s = {a[DATA_WIDTH-1], a} + {b[DATA_WIDTH-1], b};
        if (s[DATA_WIDTH] != s[DATA_WIDTH-1])
            sat_add = s[DATA_WIDTH] ? {1'b1, {(DATA_WIDTH-1){1'b0}}}
                                    : {1'b0, {(DATA_WIDTH-1){1'b1}}};
        else
            sat_add = s[DATA_WIDTH-1:0];
    endfunction

    // =========================================================================
    // Stride-2 selection trackers (advance on accepted stage output)
    // =========================================================================
    wire e0_stride2_pass = (e0_h[0] == 1'b0) && (e0_w[0] == 1'b0);
    wire e1_stride2_pass = (e1_h[0] == 1'b0) && (e1_w[0] == 1'b0);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            e0_h <= '0; e0_w <= '0;
            e1_h <= '0; e1_w <= '0;
        end else if (state == S_NEXT_TILE || state == S_IDLE) begin
            e0_h <= '0; e0_w <= '0;
            e1_h <= '0; e1_w <= '0;
        end else begin
            if (cf_valid_out[SE0] && cf_out_ready[SE0]) begin
                if (e0_w == $clog2(W0)'(W0-1)) begin e0_w <= '0; e0_h <= e0_h + 1; end
                else                                  e0_w <= e0_w + 1;
            end
            if (cf_valid_out[SE1] && cf_out_ready[SE1]) begin
                if (e1_w == $clog2(W0)'(W1-1)) begin e1_w <= '0; e1_h <= e1_h + 1; end
                else                                  e1_w <= e1_w + 1;
            end
        end
    end

    // Per-tile input token counters for E1/E2 — reset on new tile encode
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            e1_in_cnt <= '0;
            e2_in_cnt <= '0;
        end else if (enc_new_tile_p) begin
            e1_in_cnt <= '0;
            e2_in_cnt <= '0;
        end else begin
            if (cf_valid_in[SE1] && cf_ready_in[SE1])
                e1_in_cnt <= e1_in_cnt + 1;
            if (cf_valid_in[SE2] && cf_ready_in[SE2])
                e2_in_cnt <= e2_in_cnt + 1;
        end
    end

    // =========================================================================
    // Encoder routing (PIPELINED — all stages run concurrently)
    //
    //   in   → E0 in
    //   E0 out → skip0_buf, and → E1 in (gated by stride-2 pass)
    //   E1 out → skip1_buf, and → E2 in (gated by stride-2 pass)
    //   E2 out → GFF → btk_buf
    //
    // out_ready follows downstream readiness only on stride-2 pass cycles;
    // on stride-2 reject the token is consumed by the local skip-buffer
    // write (always ready), so off-grid tokens never stall the pipeline.
    // =========================================================================
    wire encoding = (state == S_TILE_ENC);

    // SSF sits between external input and E0; E0 receives re-quantized SSF output
    assign cf_valid_in     [SE0] = encoding && ssf_out_valid && cfg_enable;
    assign cf_frame_last_in[SE0] = encoding && ssf_out_last;
    always_comb for (int c = 0; c < MAX_C; c++) cf_tok_in[SE0][c] = ssf_y_q[c];
    assign in_ready              = encoding && ssf_out_ready;
    assign cf_out_ready    [SE0] = e0_stride2_pass ? cf_ready_in[SE1] : 1'b1;

    assign cf_valid_in     [SE1] = cf_valid_out[SE0] && e0_stride2_pass;
    // frame_last anchored on E0's last token; survives stride-2 reject by
    // riding cf_frame_last_out from upstream regardless of position parity
    // frame_last only propagates on valid cycles (gated by stride-2 pass)
    assign cf_frame_last_in[SE1] = cf_valid_in[SE1] && (e1_in_cnt == AW1'(H1*W1 - 1));
    always_comb for (int c = 0; c < MAX_C; c++) cf_tok_in[SE1][c] = cf_tok_out[SE0][c];
    assign cf_out_ready    [SE1] = e1_stride2_pass ? cf_ready_in[SE2] : 1'b1;

    assign cf_valid_in     [SE2] = cf_valid_out[SE1] && e1_stride2_pass;
    assign cf_frame_last_in[SE2] = cf_valid_in[SE2] && (e2_in_cnt == AW2'(H2*W2 - 1));
    always_comb for (int c = 0; c < MAX_C; c++) cf_tok_in[SE2][c] = cf_tok_out[SE1][c];
    // cf_out_ready[SE2] assigned in gen_temporal_attention / gen_no_temporal_attention above.

    // Registered "start new tile encode" pulse — used to reset addr counters
    logic enc_new_tile_p;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) enc_new_tile_p <= 1'b0;
        else        enc_new_tile_p <= e2_stage_done && !all_tiles_enc_done;
    end

    // E0 output → skip0_buf write (banked)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                    skip0_wr_addr <= '0;
        else if (enc_new_tile_p)       skip0_wr_addr <= '0;
        else if (cf_valid_out[SE0]) begin
            skip0_buf[{enc_bank, skip0_wr_addr}] <= skip0_pw;
            skip0_wr_addr <= skip0_wr_addr + 1;
        end
    end

    // E1 output → skip1_buf write (banked)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                    skip1_wr_addr <= '0;
        else if (enc_new_tile_p)       skip1_wr_addr <= '0;
        else if (cf_valid_out[SE1]) begin
            skip1_buf[{enc_bank, skip1_wr_addr}] <= skip1_pw;
            skip1_wr_addr <= skip1_wr_addr + 1;
        end
    end

    // GFF output → btk_buf write (banked); e2_out_cnt drives encoder-done
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            btk_wr_addr <= '0; e2_out_cnt <= '0;
        end else if (enc_new_tile_p) begin
            btk_wr_addr <= '0; e2_out_cnt <= '0;
        end else if (gff_valid_out) begin
            btk_buf0[{enc_bank, btk_wr_addr}] <= btk_pw;
            btk_buf1[{enc_bank, btk_wr_addr}] <= btk_pw;
            btk_buf2[{enc_bank, btk_wr_addr}] <= btk_pw;
            btk_buf3[{enc_bank, btk_wr_addr}] <= btk_pw;
            btk_wr_addr <= btk_wr_addr + 1;
            e2_out_cnt  <= e2_out_cnt  + 1;
        end
    end

    // =========================================================================
    // Bilinear-2x averaging helpers (paper Eq.7 trilinear, in 2D spatial path)
    //
    //   avg2 = (a + b)        >>> 1   (1-bit headroom, arithmetic shift)
    //   avg4 = (a + b + c + d) >>> 2  (2-bit headroom)
    //
    // Inputs are signed INT8; sign-extend through wider intermediate to avoid
    // truncating negative values on shift. Both helpers map to LUT logic
    // (no DSP); attribute below preserves the 240-DSP systolic budget.
    // =========================================================================
    function automatic logic signed [DATA_WIDTH-1:0] avg2(
        input logic signed [DATA_WIDTH-1:0] a, b
    );
        logic signed [DATA_WIDTH:0] s2;
        s2 = $signed({a[DATA_WIDTH-1], a}) + $signed({b[DATA_WIDTH-1], b});
        return s2 >>> 1;
    endfunction

    function automatic logic signed [DATA_WIDTH-1:0] avg4(
        input logic signed [DATA_WIDTH-1:0] a, b, c, d
    );
        logic signed [DATA_WIDTH+1:0] s4;
        s4 = $signed({{2{a[DATA_WIDTH-1]}}, a}) +
             $signed({{2{b[DATA_WIDTH-1]}}, b}) +
             $signed({{2{c[DATA_WIDTH-1]}}, c}) +
             $signed({{2{d[DATA_WIDTH-1]}}, d});
        return s4 >>> 2;
    endfunction

    // =========================================================================
    // D2 input: BILINEAR 2x upsample(btk_buf) + sat_add(skip1)
    //
    // Source grid is H2 x W2 = (TILE_H/4) x (TILE_W/4). The decoder produces
    // 2x denser output at H1 x W1, with up2_phase advancing 0->3 over each
    // source pixel:
    //
    //   phase 0 (TL): dst = src(h, w)
    //   phase 1 (TR): dst = avg2( src(h, w), src(h, w+1) )
    //   phase 2 (BL): dst = avg2( src(h, w), src(h+1, w) )
    //   phase 3 (BR): dst = avg4( src(h, w), src(h, w+1), src(h+1, w), src(h+1, w+1) )
    //
    // Edge handling: when h+1 == H2 or w+1 == W2 the corresponding neighbour
    // is replaced by the centre pixel (replicate-edge, equivalent to PyTorch
    // padding_mode='replicate' / align_corners=True at the boundary).
    // =========================================================================
    wire [$clog2(H2):0] btk_h_idx = btk_rd_addr / W2;
    wire [$clog2(W2):0] btk_w_idx = btk_rd_addr & (W2 - 1);
    wire btk_h_at_edge = (btk_h_idx == H2-1);
    wire btk_w_at_edge = (btk_w_idx == W2-1);

    wire [AW2-1:0] btk_addr_tr = btk_w_at_edge ? btk_rd_addr : btk_rd_addr + 1;
    wire [AW2-1:0] btk_addr_bl = btk_h_at_edge ? btk_rd_addr : btk_rd_addr + W2;
    wire [AW2-1:0] btk_addr_br = (btk_h_at_edge && btk_w_at_edge) ? btk_rd_addr :
                                  btk_h_at_edge ? btk_addr_tr :
                                  btk_w_at_edge ? btk_addr_bl :
                                                  btk_rd_addr + W2 + 1;

    logic signed [DATA_WIDTH-1:0] d2_upsamp [MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] d2_skip   [MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] d2_merged [MAX_C-1:0];

    always_comb begin
        for (int c = 0; c < C; c++) begin
            unique case (up2_phase)
                2'd0: d2_upsamp[c] = getch(c, btk_buf0[{dec_bank, btk_rd_addr}]);
                2'd1: d2_upsamp[c] = avg2(getch(c, btk_buf0[{dec_bank, btk_rd_addr}]),
                                           getch(c, btk_buf1[{dec_bank, btk_addr_tr}]));
                2'd2: d2_upsamp[c] = avg2(getch(c, btk_buf0[{dec_bank, btk_rd_addr}]),
                                           getch(c, btk_buf2[{dec_bank, btk_addr_bl}]));
                2'd3: d2_upsamp[c] = avg4(getch(c, btk_buf0[{dec_bank, btk_rd_addr}]),
                                           getch(c, btk_buf1[{dec_bank, btk_addr_tr}]),
                                           getch(c, btk_buf2[{dec_bank, btk_addr_bl}]),
                                           getch(c, btk_buf3[{dec_bank, btk_addr_br}]));
                default: d2_upsamp[c] = getch(c, btk_buf0[{dec_bank, btk_rd_addr}]);
            endcase
            d2_skip[c] = getch(c, skip1_buf[{dec_bank, skip1_rd_addr}]);
        end
    end

    // D2 feed handshake (advance condition for decoder FSM)
    // Width fix: H2*W2 may equal 2^AW2 (e.g. 4 in 2 bits), so `AW2'(H2*W2)`
    // truncates to 0 and the comparison becomes vacuously false, leaving the
    // DEC2 stage permanently stalled with d2_feed_valid=0. Widen to AW2+1 bits.
    assign d2_feed_valid = (state == S_DEC2) && ({1'b0, btk_rd_addr} < (AW2+1)'(H2*W2));

    generate
        if (SKIP_CONNECTION_MODE == 0) begin : gen_d2_sat_add
            // Mode 0: combinational saturating addition (default, fast)
            always_comb for (int c = 0; c < C; c++) d2_merged[c] = sat_add(d2_upsamp[c], d2_skip[c]);
            assign d2_feed_ready      = cf_ready_in[SD2];
            assign cf_valid_in[SD2]   = d2_feed_valid;
            assign cf_tok_in[SD2]     = d2_merged;
            assign cf_frame_last_in[SD2] = cf_valid_in[SD2] && (btk_rd_addr == AW2'(H2*W2-1))
                                             && (up2_phase == 2'd3);
        end else begin : gen_d2_skip_connect
            // Mode 1: paper-aligned GELU(GN(Linear(Cat[x1, x2])))
            // Latency ~74 cycles/token — decoder stalls while skip_connect processes.
            skip_connect #(
                .DATA_WIDTH(DATA_WIDTH), .FRAC_WIDTH(FRAC_WIDTH),
                .ACC_WIDTH(ACC_WIDTH),  .MAX_C(MAX_C),
                .GROUPS(pkg_config::GROUPS)
            ) u_skip1_sc (
                .clk(clk), .rst_n(rst_n), .cfg_enable(cfg_enable),
                .valid_in (d2_feed_valid),
                .ready_in (d2_feed_ready),
                .x1_in    (d2_upsamp),
                .x2_in    (d2_skip),
                .valid_out(cf_valid_in[SD2]),
                .out_ready(cf_ready_in[SD2]),
                .y_out    (cf_tok_in[SD2]),
                .w_lin    (gen_skip_storage.skip_w_lin[1]),
                .gn_gamma (gen_skip_storage.skip_gn_gamma[1]),
                .gn_beta  (gen_skip_storage.skip_gn_beta[1])
            );
            // frame_last propagated via delayed flag (skip_connect preserves order)
            logic d2_last_reg;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) d2_last_reg <= 1'b0;
                else if (d2_feed_valid && d2_feed_ready && (btk_rd_addr == AW2'(H2*W2-1))
                         && (up2_phase == 2'd3))
                    d2_last_reg <= 1'b1;
                else if (cf_valid_in[SD2] && cf_ready_in[SD2])
                    d2_last_reg <= 1'b0;
            end
            assign cf_frame_last_in[SD2] = cf_valid_in[SD2] && d2_last_reg;
        end
    endgenerate

    // D2 out_ready: in S_DEC2 write d2_buf (always ready); in S_DEC1 route to D1
    assign cf_out_ready    [SD2] = (state == S_DEC1) ? cf_ready_in[SD1] : 1'b1;

    // D2 output → d2_buf (replay buffer for D1)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            d2_buf_wr_addr <= '0; d2_buf_valid <= 1'b0; d2_out_cnt <= '0;
        end else if (state == S_NEXT_TILE) begin
            d2_buf_wr_addr <= '0; d2_buf_valid <= 1'b0; d2_out_cnt <= '0;
        end else if (state == S_DEC2 && cf_valid_out[SD2]) begin
            d2_buf0[{dec_bank, d2_buf_wr_addr}] <= d2_pw;
            d2_buf1[{dec_bank, d2_buf_wr_addr}] <= d2_pw;
            d2_buf2[{dec_bank, d2_buf_wr_addr}] <= d2_pw;
            d2_buf3[{dec_bank, d2_buf_wr_addr}] <= d2_pw;
            d2_out_cnt <= d2_out_cnt + 1;
            if (cf_frame_last_out[SD2]) d2_buf_valid <= 1'b1;
            else                        d2_buf_wr_addr <= d2_buf_wr_addr + 1;
        end
    end

    // =========================================================================
    // D1 input: BILINEAR 2x upsample(d2_buf) + sat_add(skip0)
    //
    // Source grid is H1 x W1 = (TILE_H/2) x (TILE_W/2). Same 4-tap pattern
    // as the D2 stage, with addresses scaled to the H1xW1 buffer.
    // =========================================================================
    wire [$clog2(H1):0] d1_h_idx = d2_buf_rd_addr / W1;
    wire [$clog2(W1):0] d1_w_idx = d2_buf_rd_addr % W1;
    wire d1_h_at_edge = (d1_h_idx == H1-1);
    wire d1_w_at_edge = (d1_w_idx == W1-1);

    wire [AW1-1:0] d1_addr_tr = d1_w_at_edge ? d2_buf_rd_addr : d2_buf_rd_addr + 1;
    wire [AW1-1:0] d1_addr_bl = d1_h_at_edge ? d2_buf_rd_addr : d2_buf_rd_addr + W1;
    wire [AW1-1:0] d1_addr_br = (d1_h_at_edge && d1_w_at_edge) ? d2_buf_rd_addr :
                                 d1_h_at_edge ? d1_addr_tr :
                                 d1_w_at_edge ? d1_addr_bl :
                                                d2_buf_rd_addr + W1 + 1;

    logic signed [DATA_WIDTH-1:0] d1_upsamp [MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] d1_skip   [MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] d1_merged [MAX_C-1:0];

    always_comb begin
        for (int c = 0; c < C; c++) begin
            unique case (up2_phase)
                2'd0: d1_upsamp[c] = getch(c, d2_lo0_q, d2_hi0_q);
                2'd1: d1_upsamp[c] = avg2(getch(c, d2_lo0_q, d2_hi0_q),
                                           getch(c, d2_lo1_q, d2_hi1_q));
                2'd2: d1_upsamp[c] = avg2(getch(c, d2_lo0_q, d2_hi0_q),
                                           getch(c, d2_lo2_q, d2_hi2_q));
                2'd3: d1_upsamp[c] = avg4(getch(c, d2_lo0_q, d2_hi0_q),
                                           getch(c, d2_lo1_q, d2_hi1_q),
                                           getch(c, d2_lo2_q, d2_hi2_q),
                                           getch(c, d2_lo3_q, d2_hi3_q));
                default: d1_upsamp[c] = getch(c, d2_lo0_q, d2_hi0_q);
            endcase
            d1_skip[c] = getch(c, skip0_lo_q, skip0_hi_q);
        end
    end

    // D1 feed handshake (advance condition for decoder FSM)
    assign d1_feed_valid = (state == S_DEC1) && d2_buf_valid;

    generate
        if (SKIP_CONNECTION_MODE == 0) begin : gen_d1_sat_add
            always_comb for (int c = 0; c < C; c++) d1_merged[c] = sat_add(d1_upsamp[c], d1_skip[c]);
            assign d1_feed_ready      = cf_ready_in[SD1];
            assign cf_valid_in[SD1]   = d1_feed_valid;
            assign cf_tok_in[SD1]     = d1_merged;
            assign cf_frame_last_in[SD1] = cf_valid_in[SD1] && (d2_buf_rd_addr == AW1'(H1*W1-1))
                                             && (up2_phase == 2'd3);
        end else begin : gen_d1_skip_connect
            skip_connect #(
                .DATA_WIDTH(DATA_WIDTH), .FRAC_WIDTH(FRAC_WIDTH),
                .ACC_WIDTH(ACC_WIDTH),  .MAX_C(MAX_C),
                .GROUPS(pkg_config::GROUPS)
            ) u_skip0_sc (
                .clk(clk), .rst_n(rst_n), .cfg_enable(cfg_enable),
                .valid_in (d1_feed_valid),
                .ready_in (d1_feed_ready),
                .x1_in    (d1_upsamp),
                .x2_in    (d1_skip),
                .valid_out(cf_valid_in[SD1]),
                .out_ready(cf_ready_in[SD1]),
                .y_out    (cf_tok_in[SD1]),
                .w_lin    (gen_skip_storage.skip_w_lin[0]),
                .gn_gamma (gen_skip_storage.skip_gn_gamma[0]),
                .gn_beta  (gen_skip_storage.skip_gn_beta[0])
            );
            logic d1_last_reg;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) d1_last_reg <= 1'b0;
                else if (d1_feed_valid && d1_feed_ready && (d2_buf_rd_addr == AW1'(H1*W1-1))
                         && (up2_phase == 2'd3))
                    d1_last_reg <= 1'b1;
                else if (cf_valid_in[SD1] && cf_ready_in[SD1])
                    d1_last_reg <= 1'b0;
            end
            assign cf_frame_last_in[SD1] = cf_valid_in[SD1] && d1_last_reg;
        end
    endgenerate

    // =========================================================================
    // Final Upsample (paper Eq. 8): Sigmoid(Linear(GN(trilinear)))
    // Placed between D1 decoder output and detection buffer.
    // D1 already includes spatial upsampling; this adds GN + 1x1 conv + Sigmoid.
    // =========================================================================
    logic                        finalup_valid_in;
    logic                        finalup_ready_in;
    logic                        finalup_valid_out;
    logic                        finalup_out_ready;
    logic signed [DATA_WIDTH-1:0] finalup_data_in [MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] finalup_data_out [MAX_C-1:0];

    always_comb for (int c = 0; c < MAX_C; c++) finalup_data_in[c] = cf_tok_out[SD1][c];
    assign finalup_valid_in = cf_valid_out[SD1];
    assign cf_out_ready[SD1] = finalup_ready_in;

    final_upsample #(
        .CHANNELS(MAX_C),
        .OUT_CH(MAX_C),
        .DATA_W(DATA_WIDTH),
        .FRAC_W(FRAC_WIDTH),
        .GROUPS(pkg_config::GROUPS),
        .PARA_CH(8),
        .WEIGHT_LOAD_MODE(pkg_config::WEIGHT_LOAD_MODE)
    ) u_final_upsample (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(finalup_valid_in),
        .ready_in(finalup_ready_in),
        .data_in(finalup_data_in),
        .valid_out(finalup_valid_out),
        .out_ready(finalup_out_ready),
        .data_out(finalup_data_out),
        .weight_valid(weight_valid && weight_layer_sel == 4'd13),
        .weight_addr(weight_addr[$clog2(MAX_C*MAX_C)-1:0]),
        .weight_data(weight_data),
        .gn_param_valid(weight_valid && weight_layer_sel == 4'd14),
        .gn_param_addr(weight_addr[15:0]),
        .gn_param_data(weight_data)
    );

    // D1 output → final_upsample → det_buf (replay buffer for detection head)
    assign finalup_out_ready = (state == S_DEC1);  // det_buf is always ready during S_DEC1
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            det_buf_wr_addr <= '0; det_buf_valid <= 1'b0; d1_out_cnt <= '0;
        end else if (state == S_NEXT_TILE && !detecting) begin
            // Safe to reset det_buf control when detection of this tile is done
            det_buf_wr_addr <= '0; det_buf_valid <= 1'b0; d1_out_cnt <= '0;
        end else if (state == S_DEC1 && d1_out_cnt == (AW0+1)'(H0*W0)) begin
            // First cycle of S_DEC1 (d1_out_cnt still at previous tile's terminal):
            // reset write address and counter, but PRESERVE det_buf_valid (it was
            // set by cf_frame_last_out on the last token and must stay high for
            // the detection head to read the buffer).
            det_buf_wr_addr <= '0; d1_out_cnt <= '0;
        end else if (state == S_DEC1 && finalup_valid_out) begin
            det_buf[{dec_bank, det_buf_wr_addr}] <= det_pw;
            d1_out_cnt <= d1_out_cnt + 1;
            if (cf_frame_last_out[SD1]) det_buf_valid <= 1'b1;
            else                        det_buf_wr_addr <= det_buf_wr_addr + 1;
        end
    end

    // D1 spatial coords (write side during S_DEC1; reused as read coords in detection)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            d1_h_out <= '0; d1_w_out <= '0; det_buf_rd_addr <= '0;
        end else if (detecting && det_head_last) begin
            // Reset detection addresses when last detection token is accepted
            d1_h_out <= '0; d1_w_out <= '0; det_buf_rd_addr <= '0;
        end else if (detecting) begin
            if (det_buf_valid && det_head_feat_ready) begin
                if (d1_w_out == $clog2(W0)'(W0-1)) begin d1_w_out <= '0; d1_h_out <= d1_h_out + 1; end
                else                                     d1_w_out <= d1_w_out + 1;
                det_buf_rd_addr <= det_buf_rd_addr + 1;
            end
        end else if (state == S_DEC1) begin
            if (cf_valid_out[SD1]) begin
                if (d1_w_out == $clog2(W0)'(W0-1)) begin d1_w_out <= '0; d1_h_out <= d1_h_out + 1; end
                else                                     d1_w_out <= d1_w_out + 1;
            end
        end
    end

    // =========================================================================
    // Main FSM (tile-aware)
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= S_IDLE;
            tile_h_idx         <= '0;
            tile_w_idx         <= '0;
            skip0_rd_addr      <= '0;
            skip1_rd_addr      <= '0;
            btk_rd_addr        <= '0;
            d2_buf_rd_addr     <= '0;
            up2_phase          <= '0;
        end else begin
`ifndef VERILATOR
            if (state == S_TILE_ENC && (perf_cycle_count < 20000 && perf_cycle_count % 500 == 0)) begin
                $display("DEBUG T=%0d:", perf_cycle_count);
                $display("  ssf: v=%b r=%b fifo_cnt=%0d", ssf_out_valid, ssf_out_ready, u_ssf.u_ssf_out_fifo.count);
                $display("  E0: slot=%0d/%0d exp=%b edr=%b proj=%b pdr=%b iv=%b ir=%b ov=%b oor=%b",
                    u_enc0.u_convformer.slot_state[0],
                    u_enc0.u_convformer.slot_state[1],
                    u_enc0.u_convformer.expand_running,
                    u_enc0.u_convformer.expand_drain,
                    u_enc0.u_convformer.project_running,
                    u_enc0.u_convformer.project_drain,
                    cf_valid_in[SE0], cf_ready_in[SE0],
                    cf_valid_out[SE0], cf_out_ready[SE0]);
                $display("  E1: slot=%0d/%0d exp=%b edr=%b proj=%b pdr=%b iv=%b ir=%b ov=%b oor=%b",
                    u_enc1.u_convformer.slot_state[0],
                    u_enc1.u_convformer.slot_state[1],
                    u_enc1.u_convformer.expand_running,
                    u_enc1.u_convformer.expand_drain,
                    u_enc1.u_convformer.project_running,
                    u_enc1.u_convformer.project_drain,
                    cf_valid_in[SE1], cf_ready_in[SE1],
                    cf_valid_out[SE1], cf_out_ready[SE1]);
                $display("  E2: slot=%0d/%0d exp=%b edr=%b proj=%b pdr=%b iv=%b ir=%b ov=%b oor=%b",
                    u_enc2.u_convformer.slot_state[0],
                    u_enc2.u_convformer.slot_state[1],
                    u_enc2.u_convformer.expand_running,
                    u_enc2.u_convformer.expand_drain,
                    u_enc2.u_convformer.project_running,
                    u_enc2.u_convformer.project_drain,
                    cf_valid_in[SE2], cf_ready_in[SE2],
                    cf_valid_out[SE2], cf_out_ready[SE2]);
                $display("  gff: rdy=%b vin=%b vout=%b btk_cnt=%0d",
                    gff_ready_in, cf_valid_out[SE2], gff_valid_out, e2_out_cnt);
                $display("  in_v=%b in_r=%b | e0_h=%0d e0_w=%0d e1_h=%0d e1_w=%0d | st2_0=%b st2_1=%b",
                    in_valid, in_ready,
                    e0_h, e0_w, e1_h, e1_w,
                    e0_stride2_pass, e1_stride2_pass);
            end
`endif

            case (state)
                S_IDLE: begin
                    if (in_valid && cfg_enable) begin
                        state          <= S_TILE_ENC;
                        tile_h_idx     <= '0;
                        tile_w_idx     <= '0;
                        skip0_rd_addr  <= '0;
                        skip1_rd_addr  <= '0;
                        btk_rd_addr    <= '0;
                        d2_buf_rd_addr <= '0;
                        up2_phase      <= '0;
                    end
                end

                // Encoder pipeline drains the current tile. We're done when
                // the bottleneck (E2 → GFF → btk_buf) has filled.
                S_TILE_ENC: begin
                    if (e2_stage_done) begin
                        state       <= S_DEC2;
                        btk_rd_addr <= '0;
                        up2_phase   <= '0;
                    end
                end

                S_DEC2: begin
                    if (d2_feed_valid && d2_feed_ready) begin
                        if (up2_phase == 2'd3) begin
                            up2_phase     <= '0;
                            btk_rd_addr   <= btk_rd_addr   + 1;
                            skip1_rd_addr <= skip1_rd_addr + 1;
                        end else begin
                            up2_phase <= up2_phase + 1;
                        end
                    end
                    if (d2_stage_done) begin
                        state          <= S_DEC1;
                        d2_buf_rd_addr <= '0;
                        up2_phase      <= '0;
                    end
                end

                S_DEC1: begin
                    if (d1_feed_valid && d1_feed_ready) begin
                        if (up2_phase == 2'd3) begin
                            up2_phase      <= '0;
                            d2_buf_rd_addr <= d2_buf_rd_addr + 1;
                            skip0_rd_addr  <= skip0_rd_addr  + 1;
                        end else begin
                            up2_phase <= up2_phase + 1;
                        end
                    end
                    if (d1_stage_done) begin
                        state <= S_NEXT_TILE;
                    end
                end

                S_NEXT_TILE: begin
                    skip0_rd_addr  <= '0;
                    skip1_rd_addr  <= '0;
                    btk_rd_addr    <= '0;
                    d2_buf_rd_addr <= '0;
                    up2_phase      <= '0;
                    // Last tile completed — frame done, return to idle
                    if (tile_h_idx == TH_W'(NUM_TILES_H-1) &&
                        tile_w_idx == TW_W'(NUM_TILES_W-1)) begin
                        state            <= S_IDLE;
                    end else if (tile_w_idx == TW_W'(NUM_TILES_W-1)) begin
                        tile_w_idx <= '0;
                        tile_h_idx <= tile_h_idx + 1;
                        state      <= S_TILE_ENC;
                    end else begin
                        tile_w_idx <= tile_w_idx + 1;
                        state      <= S_TILE_ENC;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // === Debug: detection-head / NMS observation (visible to Verilator) ======
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dbg_dh_emit_count  <= '0;
            dbg_nms_out_count   <= '0;
        end else begin
            if (det_head_valid && nms_det_ready) dbg_dh_emit_count <= dbg_dh_emit_count + 1;
            if (nms_out_valid && nms_skid_ready) dbg_nms_out_count  <= dbg_nms_out_count + 1;
        end
    end
    assign dbg_det_head_valid = det_head_valid;
    assign dbg_detecting      = detecting;
    assign dbg_unet_state     = state;
    assign dbg_det_buf_valid  = det_buf_valid;
    assign dbg_d1_stage_done  = d1_stage_done;
    assign dbg_outstanding_det = outstanding_det;
    assign dbg_det_head_feat_valid = detecting && det_buf_valid && det_head_feat_ready;
    assign dbg_cf_valid_d1    = cf_valid_out[SD1];
    assign dbg_finalup_valid  = finalup_valid_out;
    assign dbg_d1_out_cnt     = {25'd0, d1_out_cnt};

    // === DEBUG: U-Net top FSM state trace ====================================
    // synthesis translate_off
    logic [31:0] unet_dbg_cycle;
    logic [3:0]  unet_dbg_prev_state;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            unet_dbg_cycle      <= 0;
            unet_dbg_prev_state <= '0;
        end else begin
            unet_dbg_cycle <= unet_dbg_cycle + 1'b1;
            if (state != unet_dbg_prev_state) begin
                $display("UNETDBG c=%0d state=%0d -> %0d  e2_done=%b d2_done=%b d1_done=%b  d2_fv=%b d2_fr=%b d1_fv=%b d1_fr=%b",
                    unet_dbg_cycle, unet_dbg_prev_state, state,
                    e2_stage_done, d2_stage_done, d1_stage_done,
                    d2_feed_valid, d2_feed_ready, d1_feed_valid, d1_feed_ready);
            end
            // Periodic state dump while running, every 5000 cycles
            if (unet_dbg_cycle % 5000 == 0 && state != S_IDLE) begin
                $display("UNETDBG c=%0d state=%0d  e2_done=%b d2_done=%b d1_done=%b  d2_fv=%b d2_fr=%b d1_fv=%b d1_fr=%b btk_addr=%0d d2_addr=%0d",
                    unet_dbg_cycle, state,
                    e2_stage_done, d2_stage_done, d1_stage_done,
                    d2_feed_valid, d2_feed_ready, d1_feed_valid, d1_feed_ready,
                    btk_rd_addr, d2_buf_rd_addr);
            end
            unet_dbg_prev_state <= state;
        end
    end
    // synthesis translate_on

    // -------------------------------------------------------------------------
    // Detection tracking + bank/tile accounting (unified)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            outstanding_det <= '0;
            enc_tile_cnt    <= '0;
            dec_tile_cnt    <= '0;
            enc_bank        <= 1'b0;
            dec_bank        <= 1'b0;
            det_rd_bank     <= 1'b0;
        end else begin
            // Decoder tile complete: capture write bank for detection read,
            // toggle write bank for next tile
            if (state == S_DEC1 && d1_stage_done) begin
                outstanding_det <= outstanding_det + 1;
                det_rd_bank     <= dec_bank;
                dec_bank        <= ~dec_bank;
            end
            // Detection head finishes a tile
            if (det_head_last) begin
                if (outstanding_det > 0)
                    outstanding_det <= outstanding_det - 1;
                if (outstanding_det == 1)
                    dec_tile_cnt <= dec_tile_cnt + 1;
            end
            // Encoder tile complete
            if (state == S_TILE_ENC && e2_stage_done) begin
                enc_tile_cnt <= enc_tile_cnt + 1;
                enc_bank     <= ~enc_bank;
                $display("ENC_TILE_DONE at cycle %0d", perf_cycle_count);
            end
            // DEBUG: trace E2→GFF token flow
            if (cf_valid_out[SE2] && cf_out_ready[SE2])
                $display("E2_OUT at cycle %0d, e2_out_cnt=%0d", perf_cycle_count, e2_out_cnt);
            if (gff_valid_out)
                $display("GFF_OUT at cycle %0d, e2_out_cnt→%0d", perf_cycle_count, e2_out_cnt + 1);
        end
    end

    // =========================================================================
    // Detection head: fed from det_buf during detection.
    // Coordinates are tile-local; we add the tile origin to produce absolute
    // (frame-relative) feat_h_idx / feat_w_idx for the detection head.
    // =========================================================================
    localparam int DET_W = pkg_module_config::DETECTION_HEAD_DATA_WIDTH;

    logic                        det_head_valid;
    logic                        det_head_feat_ready;
    logic [15:0]                 det_head_h_idx, det_head_w_idx;
    logic [1:0]                  det_head_anchor;
    logic signed [DET_W-1:0]   det_head_conf;
    logic signed [DET_W-1:0]   det_head_dx, det_head_dy, det_head_dw, det_head_dh;
    logic [NUM_CLASSES-1:0][DET_W-1:0] det_head_cls;
    logic                        det_head_last;
    logic [15:0]                 det_head_cnt;

    // Backpressure from downstream (AXI DMA det_fifo) propagates through
    // detection_head to the decoder stage via feat_ready.
    logic                        nms_det_ready;
    assign nms_det_ready = det_ready_in;

    // synthesis translate_off
    // Skid buffer handles backpressure; valid_out may be high while ready_in is
    // low — the skid buffer holds the data stable until the consumer is ready.
    NMS_SKID_STABLE: assert property (
        @(posedge clk) disable iff (!rst_n)
        (det_valid_out && !det_ready_in) |=> det_valid_out
    ) else $error("e_rodnet_unet_top: skid buffer dropped valid detection while consumer stalled");
    // synthesis translate_on

    logic signed [DET_W-1:0] det_feat_data_wide [MAX_C-1:0];
    always_comb for (int c = 0; c < MAX_C; c++) det_feat_data_wide[c] = DET_W'(getch(c, det_lo_q, det_hi_q));

    // Tile-origin offset → absolute coords for the detection head
    wire [15:0] tile_h_origin = 16'(tile_h_idx) * 16'(TILE_H);
    wire [15:0] tile_w_origin = 16'(tile_w_idx) * 16'(TILE_W);

    detection_head #(
        .FEATURE_H(FEATURE_H), .FEATURE_W(FEATURE_W),
        .IN_CHANNELS(MAX_C),
        .NUM_CLASSES(NUM_CLASSES), .NUM_ANCHORS(NUM_ANCHORS),
        .DATA_WIDTH(DET_W), .FRAC_WIDTH(FRAC_WIDTH)
    ) u_det_head (
        .clk(clk), .rst_n(rst_n),
        .cfg_feature_h(16'(FEATURE_H)), .cfg_feature_w(16'(FEATURE_W)),
        .cfg_num_classes(8'(NUM_CLASSES)), .cfg_num_anchors(8'(NUM_ANCHORS)),
        .cfg_enable(cfg_enable),
        .feat_valid(detecting && det_buf_valid && det_head_feat_ready),
        .feat_ready(det_head_feat_ready),
        .feat_h_idx(tile_h_origin + 16'(d1_h_out)),
        .feat_w_idx(tile_w_origin + 16'(d1_w_out)),
        .feat_data(det_feat_data_wide),
        .feat_last(det_buf_rd_addr == AW0'(H0*W0-1)),
        .cls_weight_valid (weight_valid && weight_layer_sel == 4'd7),
        .cls_weight_addr  (weight_addr[15:0]),
        .cls_weight_data  (weight_data),
        .bbox_weight_valid(weight_valid && weight_layer_sel == 4'd8),
        .bbox_weight_addr (weight_addr[15:0]),
        .bbox_weight_data (weight_data),
        .det_valid(det_head_valid), .det_ready(nms_det_ready),
        .det_h_idx(det_head_h_idx), .det_w_idx(det_head_w_idx),
        .det_anchor_idx(det_head_anchor),
        .det_confidence(det_head_conf),
        .det_dx(det_head_dx), .det_dy(det_head_dy),
        .det_dw(det_head_dw), .det_dh(det_head_dh),
        .det_class_scores(det_head_cls),
        .det_count(det_head_cnt),
        .det_last(det_head_last)
    );

    // -------------------------------------------------------------------------
    // NMS output skid buffer — absorbs backpressure from AXI DMA det_fifo.
    // nms_topk has no out_ready; the skid buffer captures its output and
    // presents it to downstream with proper valid/ready handshake.
    // -------------------------------------------------------------------------
    // NMS outputs are DET_W-wide (16-bit), but UNet output ports are DATA_WIDTH-wide (8-bit).
    // Truncate with saturation before packing to match the unpack target widths.
    localparam int DET_PACKED_W = 16 + 16 + DATA_WIDTH + 16 + 16 + NUM_CLASSES*DATA_WIDTH + 2;

    logic                        nms_out_valid;
    logic [15:0]                 nms_out_h_idx,  nms_out_w_idx;
    logic signed [DET_W-1:0]     nms_out_confidence;
    logic [15:0]                 nms_out_width,  nms_out_height;
    logic [NUM_CLASSES-1:0][DET_W-1:0] nms_out_class_scores;
    logic [1:0]                  nms_out_class_id;
    logic                        nms_out_last;
    logic [DET_PACKED_W-1:0]     nms_packed, skid_packed;

    // Saturating truncation: DET_W → DATA_WIDTH
    logic signed [DATA_WIDTH-1:0] nms_out_confidence_q;
    logic [NUM_CLASSES-1:0][DATA_WIDTH-1:0] nms_out_class_scores_q;
    logic [DATA_WIDTH-1:0] nms_out_width_q, nms_out_height_q;

    always_comb begin
        automatic logic signed [DET_W-1:0] v;
        v = nms_out_confidence;
        if      (v[DET_W-1] && |(~v[DET_W-1:DATA_WIDTH-1])) nms_out_confidence_q = {1'b1, {(DATA_WIDTH-1){1'b0}}};
        else if (!v[DET_W-1] && |v[DET_W-1:DATA_WIDTH-1])  nms_out_confidence_q = {1'b0, {(DATA_WIDTH-1){1'b1}}};
        else                                                 nms_out_confidence_q = v[DATA_WIDTH-1:0];
    end

    // Use upper byte of 16-bit width/height (most significant bits of bounding box size)
    assign nms_out_width_q  = nms_out_width[DATA_WIDTH-1:0];
    assign nms_out_height_q = nms_out_height[DATA_WIDTH-1:0];

    always_comb begin
        for (int c = 0; c < NUM_CLASSES; c++) begin
            automatic logic signed [DET_W-1:0] v = nms_out_class_scores[c];
            if      (v[DET_W-1] && |(~v[DET_W-1:DATA_WIDTH-1])) nms_out_class_scores_q[c] = {1'b1, {(DATA_WIDTH-1){1'b0}}};
            else if (!v[DET_W-1] && |v[DET_W-1:DATA_WIDTH-1])  nms_out_class_scores_q[c] = {1'b0, {(DATA_WIDTH-1){1'b1}}};
            else                                                 nms_out_class_scores_q[c] = v[DATA_WIDTH-1:0];
        end
    end

    assign nms_packed = {
        nms_out_class_id,
        nms_out_class_scores_q,
        nms_out_height_q,
        nms_out_width_q,
        nms_out_confidence_q,
        nms_out_w_idx,
        nms_out_h_idx
    };

    logic nms_skid_ready;  // skid_buffer ready_in (monitored by assertion)

    skid_buffer #(
        .DATA_WIDTH(DET_PACKED_W),
        .INCLUDE_LAST(1)
    ) u_nms_skid (
        .clk(clk), .rst_n(rst_n),
        .data_in (nms_packed),
        .valid_in(nms_out_valid),
        .ready_in(nms_skid_ready),
        .last_in (nms_out_last),
        .data_out (skid_packed),
        .valid_out(det_valid_out),
        .out_ready(det_ready_in),
        .last_out (det_last)
    );

    assign {
        det_class_id,
        det_class_scores,
        det_height,
        det_width,
        det_confidence,
        det_w_idx,
        det_h_idx
    } = skid_packed;

    generate
        if (EN_NMS) begin : gen_nms
            nms_topk #(
                .MAX_DETECTIONS(pkg_config::MAX_DETECTIONS), .NUM_CLASSES(NUM_CLASSES),
                .TOP_K(TOP_K), .DATA_WIDTH(DET_W), .FRAC_WIDTH(FRAC_WIDTH)
            ) u_nms (
                .clk(clk), .rst_n(rst_n),
                .cfg_conf_threshold(cfg_conf_threshold),
                .cfg_iou_threshold (cfg_iou_threshold),
                .cfg_top_k(cfg_top_k),
                .cfg_enable(cfg_enable),
                .det_valid(det_head_valid),
                .det_h_idx(det_head_h_idx), .det_w_idx(det_head_w_idx),
                .det_anchor_idx(det_head_anchor),
                .det_confidence(det_head_conf),
                .det_dx(det_head_dx), .det_dy(det_head_dy),
                .det_dw(det_head_dw), .det_dh(det_head_dh),
                .det_class_scores(det_head_cls),
                .det_last(det_head_last),
                .out_valid(nms_out_valid),
                .out_h_idx(nms_out_h_idx), .out_w_idx(nms_out_w_idx),
                .out_confidence(nms_out_confidence),
                .out_width(nms_out_width), .out_height(nms_out_height),
                .out_class_scores(nms_out_class_scores),
                .out_class_id(nms_out_class_id),
                .out_last(nms_out_last),
                .out_ready(nms_skid_ready),   // FIXED: was unconnected (floating) — backpressure
                                              // from the output skid buffer to the NMS engine
                                              // prevents detection loss when det_ready_in stalls
                .out_count(det_count),
                .suppressed_count()
            );
        end else begin : gen_no_nms
            // Bypass: detection head → output directly.
            // Class ID = argmax of class scores; width/height from dw/dh directly.
            always_comb begin
                nms_out_class_id = 2'd0;
                for (int c = 1; c < NUM_CLASSES; c++) begin
                    if (det_head_cls[c] > det_head_cls[nms_out_class_id])
                        nms_out_class_id = 2'(c);
                end
            end

            assign nms_out_valid    = det_head_valid;
            assign nms_out_h_idx    = det_head_h_idx;
            assign nms_out_w_idx    = det_head_w_idx;
            assign nms_out_confidence = det_head_conf;
            assign nms_out_width    = DET_W'($signed(det_head_dw));  // simplified: use dw/dh directly
            assign nms_out_height   = DET_W'($signed(det_head_dh));
            assign nms_out_class_scores = det_head_cls;
            assign nms_out_last     = det_head_last;
            assign det_count        = det_head_cnt;
        end
    endgenerate

    // synthesis translate_off
    NMS_SKID_OVERFLOW: assert property (
        @(posedge clk) disable iff (!rst_n)
        !(nms_out_valid && !nms_skid_ready)
    ) else $error("e_rodnet_unet_top: NMS skid buffer overflow — downstream DMA cannot keep up");
    // synthesis translate_on

    // =========================================================================
    // Performance counters / status
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_cycle_count <= '0;
            perf_frame_count <= '0;
            processing_busy  <= 1'b0;
            processing_done  <= 1'b0;
        end else begin
            perf_cycle_count <= perf_cycle_count + 1;
            processing_busy <= (state != S_IDLE);
            processing_done <= (state == S_NEXT_TILE) &&
                               (tile_h_idx == TH_W'(NUM_TILES_H-1)) &&
                               (tile_w_idx == TW_W'(NUM_TILES_W-1));
            if (state == S_NEXT_TILE &&
                tile_h_idx == TH_W'(NUM_TILES_H-1) &&
                tile_w_idx == TW_W'(NUM_TILES_W-1))
                perf_frame_count <= perf_frame_count + 1;
        end
    end

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    UNET_GEOMETRY: assert property (
        @(posedge clk) disable iff (!rst_n)
        (H0 >= 8) && (W0 >= 8) && ((H0 & (H0-1)) == 0) && ((W0 & (W0-1)) == 0)
    ) else $error("e_rodnet_unet_top: TILE_H/TILE_W must be power-of-2 >= 8");

    UNET_TILE_DIVIDES: assert property (
        @(posedge clk) disable iff (!rst_n)
        (FEATURE_H % TILE_H == 0) && (FEATURE_W % TILE_W == 0)
    ) else $error("e_rodnet_unet_top: FEATURE_H/W must be a multiple of TILE_H/W");
    // synthesis translate_on

endmodule
