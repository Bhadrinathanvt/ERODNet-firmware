// =============================================================================
// gff_stream_core.sv
//
// STREAMING GFF Synthesizer — token-by-token architecture.
//
// Replaces the batch-parallel gff_synth with a streaming pipeline that processes
// one cell per cycle. The outer gff.sv wrapper provides dual-frame double-buffer:
// while gff_stream_core drains frame K from buffer A, buffer B fills with frame K+1.
//
// PIPELINE STAGES (one cell per cycle propagates):
//   Token flow: Input → [avgpool accum] → [in_l2norm buf+seq] →
//               [proj path: fused_proj→l2norm→unpool] →
//               [concat + linear + gelu] → Output
//
//   Stage 0 (cycles 0..N-1):  BRAM accumulate all N tokens into pooled[4C]
//   Stage 1 (cycle N):        pooled_valid fires
//   Stage 2 (N+1..N+8C+27):  fused_proj pipeline (8C+27 cy)
//   Stage 3 (N+8C+28):        proj_l2norm valid (3+2*ITER cy)
//   Stage 4 (N+8C+28..):      unpooling: proj_l2norm broadcast for N cycles
//   Stage 5 (N+8C+28..):      in_l2norm: seq BRAM→l2norm, N cells streamed out
//   Stage 6 (N+8C+28..):      delay-align (N-entry BRAM delay buffer)
//   Stage 7 (N+8C+28..):      concat (1 cy), result → concat_buf
//   Stage 8 (N+8C+28..):      linear 2C→C (one cell per MAC cycle), output → gelu_buf
//   Stage 9 (ongoing):        gelu (4 cy), final output cell
//
// KEY CONSTANTS:
//   D = 8*C + 28: cycles between unpooled[0] and in_l2norm[0] arriving at concat
//   Linear IN_CH = 2C (64 for C=32). Each cell output takes 2C+3 MAC cycles.
//   Linear throughput: 1 cell per (2C+3) cycles ≈ 67 cy/cell @ C=32
//
// DUAL-BUFFER OVERLAP (gff.sv level):
//   Fill K:   16384 cycles  @ 1 token/cycle
//   Compute:  ~300 cycles   @ 1 cell/cycle (pipeline fills)
//   Drain K:  ~16k cycles   @ 1 token/cycle
//   Fill K+1: overlaps with drain K
//   Net: ~16384 cycles/frame ≈ 60 fps → with overlap: ~32768 cycles/2frames = 120 fps
// =============================================================================

`timescale 1ns / 1ps

import pkg_config::*;
import pkg_module_config::*;

/* verilator lint_off VARHIDDEN */
/* verilator lint_off IMPLICITSTATIC */
module gff_stream_core #(
    parameter int C               = pkg_config::MAX_C,
    parameter int N               = 1024,
    parameter int DATA_W          = pkg_module_config::GFF_STREAM_CORE_DATA_WIDTH,
    parameter int FRAC_W          = pkg_module_config::GFF_STREAM_CORE_DATA_WIDTH / 2,
    parameter int ACC_W           = pkg_config::ACC_WIDTH,
    parameter int L2_ITER         = pkg_config::L2_ITER,
    parameter int AVG_GROUP_SIZE  = pkg_config::AVG_GROUP_SIZE,
    parameter int TIME_MUX       = pkg_config::TIME_MUX,
    parameter string ROM_DIR      = "rtl/mem",
    parameter string WEIGHT_FILE  = "rtl/mem/gff_cat_weight.mem",
    parameter string BIAS_FILE    = "rtl/mem/gff_cat_bias.mem"
) (
    input  logic clk,
    input  logic rst_n,

    // Input: one cell per cycle (token streaming)
    input  logic signed [DATA_W-1:0] feat_in [0:4*C-1],
    input  logic                      feat_valid,
    input  logic                      feat_last,
    output logic                      feat_ready,

    // Output: one cell per cycle
    output logic signed [DATA_W-1:0] feat_out [0:C-1],
    output logic                      feat_out_valid,
    output logic                      feat_out_last,

    // LayerNorm learned parameters
    input  logic signed [DATA_W-1:0] proj_ln_gamma [0:3*C-1],
    input  logic signed [DATA_W-1:0] proj_ln_beta  [0:3*C-1],

    // Weight loading ports
    input  logic                      weight_valid,
    input  logic [31:0]               weight_addr,
    input  logic signed [DATA_W-1:0]  weight_data
);

    // =========================================================================
    // Derived Constants
    // =========================================================================
    localparam int CELL_ADDR_W = $clog2(N);
    localparam int TREE_DEPTH  = $clog2((N + AVG_GROUP_SIZE - 1) / AVG_GROUP_SIZE);
    localparam int T           = TREE_DEPTH + 2;          // avgpool latency
    localparam int D           = 8*C + 28;                // concat alignment delay
    localparam int LINEAR_LAT  = 2*C + 3;                // linear_layer cycles/cell
    localparam int LIN_CNT_W   = $clog2(2*C);
    localparam int GELU_LAT    = 4;

    // =========================================================================
    // Valid Pipeline — removed VP_STAGES shift register (was never read;
    // flow control uses running, frame_done, gelu_output_cnt, etc.)
    // =========================================================================

    // =========================================================================
    // Input State Machine
    // =========================================================================
    // States: IDLE → FILL (during first frame) → RUN (processing) → DONE
    typedef enum logic [1:0] {
        S_IDLE,
        S_FILL,    // Collecting input tokens
        S_COMPUTE  // Processing frame (avgpool + all pipeline stages + drain)
    } core_state_t;
    core_state_t core_state;

    logic [CELL_ADDR_W-1:0] token_cnt;   // Current token index within frame
    logic                   running;     // Currently processing a frame
    logic                   frame_done;   // Single-cycle pulse: frame fully streamed out

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            core_state <= S_IDLE;
            token_cnt  <= '0;
            running   <= 1'b0;
            frame_done <= 1'b0;
        end else begin
            frame_done <= 1'b0;

            case (core_state)
                S_IDLE: begin
                    if (feat_valid) begin
                        core_state <= S_FILL;
                        token_cnt  <= '0;
                        running    <= 1'b1;
                    end
                end

                S_FILL: begin
                    if (feat_valid) begin
                        if (feat_last) begin
                            core_state <= S_COMPUTE;
                            token_cnt  <= '0;
                        end else begin
                            token_cnt <= token_cnt + 1'b1;
                        end
                    end
                end

                S_COMPUTE: begin
                    // Transition when all N cells produced by linear AND
                    // last GELU output fires on the same cycle.
                    if (linear_cells_produced == N &&
                        gelu_v && gelu_output_cnt == N - 1) begin
                        core_state <= S_IDLE;
                        token_cnt  <= '0;
                        running    <= 1'b0;
                        frame_done <= 1'b1;
                    end
                end
            endcase
        end
    end

    assign feat_ready = !running;

    // Combinatorial frame-active signal: true on the first feat_valid (S_IDLE→S_FILL
    // transition) AND all subsequent S_FILL cycles.  Catches the first cell that the
    // registered 'running' signal misses by 1 cycle.
    wire frame_accumulating = (core_state == S_FILL) ||
                              (core_state == S_IDLE && feat_valid);

    // =========================================================================
    // Stage 0: BRAM AvgPool Accumulator
    // =========================================================================
    // Accumulate all N tokens (each 4C-wide) into avgpool_accum[4C] via BRAM.
    // Each feat_valid token (during S_FILL or S_COMPUTE) adds to the accumulator.
    // After N tokens, avgpool_done pulses and we divide by N to get pooled result.

    (* ram_style = "block" *)
    logic signed [ACC_W-1:0] avgpool_accum [0:4*C-1];
    logic [CELL_ADDR_W-1:0]   avgpool_wr_cnt;   // How many tokens accumulated
    logic                     avgpool_summing;
    logic                     avgpool_done;      // Single-cycle pulse
    logic                     avgpool_done_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            avgpool_summing <= 1'b0;
            avgpool_wr_cnt   <= '0;
            avgpool_done     <= 1'b0;
            avgpool_done_r   <= 1'b0;
        end else begin
            avgpool_done   <= 1'b0;
            avgpool_done_r <= avgpool_done;

            if (!avgpool_summing && feat_valid && frame_accumulating) begin
                // First token: mark summing active, wr_ptr stays 0
                avgpool_summing <= 1'b1;
                avgpool_wr_cnt   <= '0;
            end else if (avgpool_summing && feat_valid) begin
                // Subsequent tokens: accumulate. Done on feat_last.
                if (feat_last) begin
                    avgpool_summing <= 1'b0;
                    avgpool_done     <= 1'b1;
                    avgpool_wr_cnt   <= '0;
                end else begin
                    avgpool_wr_cnt <= avgpool_wr_cnt + 1'b1;
                end
            end
        end
    end

    // BRAM write: accumulate each token
    always_ff @(posedge clk) begin
        if ((avgpool_summing || frame_accumulating) && feat_valid) begin
            for (int ch = 0; ch < 4*C; ch++)
                avgpool_accum[ch] <= avgpool_accum[ch] + ACC_W'(feat_in[ch]);
        end
    end

    // Compute final pooled result on avgpool_done
    logic signed [DATA_W-1:0] pooled [0:4*C-1];
    logic                     pooled_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pooled_valid <= 1'b0;
        end else begin
            pooled_valid <= avgpool_done;
            if (avgpool_done) begin
                for (int ch = 0; ch < 4*C; ch++) begin
                    // Power-of-2 N: right shift with rounding
                    pooled[ch] <= DATA_W'((avgpool_accum[ch] + ACC_W'(1 <<< (CELL_ADDR_W-1))) >>> CELL_ADDR_W);
                end
            end
        end
    end

    // =========================================================================
    // Stage 1: Fused Project Pipeline (8C+27 cy latency)
    // =========================================================================
    logic signed [DATA_W-1:0] pooled_reg [0:4*C-1];
    logic                     pooled_valid_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pooled_valid_r <= 1'b0;
            for (int ch = 0; ch < 4*C; ch++)
                pooled_reg[ch] <= '0;
        end else begin
            pooled_valid_r <= pooled_valid;
            if (pooled_valid)
                pooled_reg <= pooled;
        end
    end

    logic signed [DATA_W-1:0] proj_out    [0:C-1];
    logic                     proj_out_v;

    gff_fused_proj #(
        .C       (C),
        .DATA_W  (DATA_W),
        .FRAC_W  (FRAC_W),
        .ACC_W   (ACC_W),
        .USE_DSP (0),
        .GELU_DSP(0),
        .ROM_DIR (ROM_DIR)
    ) u_fused_proj (
        .clk           (clk),
        .rst_n         (rst_n),
        .pooled        (pooled_reg),
        .pooled_valid  (pooled_valid_r),
        .proj_out      (proj_out),
        .proj_out_v    (proj_out_v),
        .ln_gamma      (proj_ln_gamma),
        .ln_beta       (proj_ln_beta),
        .weight_valid  (weight_valid),
        .weight_addr   (weight_addr),
        .weight_data   (weight_data)
    );

    // =========================================================================
    // Stage 2: Proj L2norm (3 + 2*L2_ITER cy)
    // =========================================================================
    logic signed [C*DATA_W-1:0] proj_out_flat, proj_l2norm_flat;
    logic                     proj_l2norm_v;

    always_comb begin
        for (int c = 0; c < C; c++)
            proj_out_flat[c*DATA_W +: DATA_W] = proj_out[c];
    end

    l2norm_nr #(
        .CH         (C),
        .DATA_W     (DATA_W),
        .FRAC_W     (FRAC_W),
        .ITERATIONS (L2_ITER),
        .ACC_W      (ACC_W),
        .USE_DSP    (0)
    ) u_proj_l2norm (
        .clk          (clk),
        .rst_n        (rst_n),
        .data_in_flat (proj_out_flat),
        .valid_in     (proj_out_v),
        .ready_out    (),           // unused
        .data_out_flat(proj_l2norm_flat),
        .valid_out    (proj_l2norm_v),
        .out_ready    (1'b1)        // always ready to consume
    );

    logic signed [DATA_W-1:0] proj_l2norm [0:C-1];
    always_comb begin
        for (int c = 0; c < C; c++)
            proj_l2norm[c] = proj_l2norm_flat[c*DATA_W +: DATA_W];
    end

    // =========================================================================
    // Stage 3: Unpooling — broadcast global C-vector to all N cells
    // =========================================================================
    // proj_l2norm_v is a 1-cycle pulse. Hold proj_l2norm and broadcast for N cycles.
    // unpooled_v = unpooling (asserted for N cycles during broadcast)

    logic signed [DATA_W-1:0] proj_l2norm_reg [0:C-1];
    logic [$clog2(N+1)-1:0]  unpool_cnt;
    logic                     unpooling;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            unpooling <= 1'b0;
            unpool_cnt <= '0;
            for (int c = 0; c < C; c++)
                proj_l2norm_reg[c] <= '0;
        end else begin
            if (proj_l2norm_v) begin
                proj_l2norm_reg <= proj_l2norm;
                unpooling       <= 1'b1;
                unpool_cnt      <= '0;
            end else if (unpooling) begin
                if (unpool_cnt == N - 1) begin
                    unpooling <= 1'b0;
                    unpool_cnt <= '0;
                end else begin
                    unpool_cnt <= unpool_cnt + 1'b1;
                end
            end
        end
    end

    logic unpooled_v;
    assign unpooled_v = unpooling;

    // =========================================================================
    // Stage 4: in_l2norm — sequential BRAM buffer + one l2norm engine
    // =========================================================================
    // Store all N input tokens in BRAM (write during S_FILL).
    // After avgpool_done, sequentially read from BRAM through l2norm (one per cycle).
    //
    // BRAM write: feat_valid during S_FILL → in_l2norm_buf[wr_ptr]
    // BRAM read:  in_l2norm_processing → in_l2norm_buf[rd_ptr] → l2norm_nr
    // Output:    in_l2norm_v fires once per l2norm result

    (* ram_style = "block" *)
    logic signed [DATA_W-1:0] in_l2norm_buf [0:N-1][0:4*C-1];
    logic [CELL_ADDR_W-1:0]   in_buf_wr_ptr;

    // Write all incoming tokens into BRAM
    always_ff @(posedge clk) begin
        if (feat_valid && frame_accumulating) begin
            for (int ch = 0; ch < 4*C; ch++)
                in_l2norm_buf[in_buf_wr_ptr][ch] <= feat_in[ch];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_buf_wr_ptr <= '0;
        end else begin
            if (core_state == S_IDLE)
                in_buf_wr_ptr <= '0;
            else if (feat_valid && frame_accumulating)
                in_buf_wr_ptr <= in_buf_wr_ptr + 1'b1;
        end
    end

    // Sequential processing: start after avgpool_done, read one cell per cycle
    logic [CELL_ADDR_W-1:0]    in_l2norm_rd_ptr;
    logic                       in_l2norm_processing;
    logic                       in_l2norm_v;
    logic signed [DATA_W-1:0]   in_l2norm_result [0:C-1];

    // Flat ports for l2norm_nr (processes first C of 4C input)
    logic signed [4*C*DATA_W-1:0] in_cell_flat, in_l2norm_flat_out;
    logic                             in_l2norm_nr_v;

    always_comb begin
        for (int ch = 0; ch < 4*C; ch++)
            in_cell_flat[ch*DATA_W +: DATA_W] = in_l2norm_buf[in_l2norm_rd_ptr][ch];
    end

    l2norm_nr #(
        .CH         (4*C),
        .DATA_W     (DATA_W),
        .FRAC_W     (FRAC_W),
        .ITERATIONS (L2_ITER),
        .ACC_W      (ACC_W),
        .USE_DSP    (0)
    ) u_in_l2norm (
        .clk          (clk),
        .rst_n        (rst_n),
        .data_in_flat (in_cell_flat),
        .valid_in     (in_l2norm_processing),
        .ready_out    (),           // unused
        .data_out_flat(in_l2norm_flat_out),
        .valid_out    (in_l2norm_nr_v),
        .out_ready    (1'b1)        // always ready to consume
    );

    always_comb begin
        for (int c = 0; c < C; c++)
            in_l2norm_result[c] = in_l2norm_flat_out[c*DATA_W +: DATA_W];
    end

    // Control: start processing after avgpool_done, read one per cycle
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_l2norm_processing <= 1'b0;
            in_l2norm_rd_ptr     <= '0;
            in_l2norm_v          <= 1'b0;
        end else begin
            in_l2norm_v <= 1'b0;

            if (avgpool_done && !in_l2norm_processing) begin
                in_l2norm_processing <= 1'b1;
                in_l2norm_rd_ptr     <= '0;
            end else if (in_l2norm_processing) begin
                in_l2norm_v <= in_l2norm_nr_v;  // Forward l2norm output
                if (in_l2norm_nr_v) begin
                    if (in_l2norm_rd_ptr == N - 1) begin
                        // Last cell done
                        in_l2norm_processing <= 1'b0;
                        in_l2norm_rd_ptr     <= '0;
                    end else begin
                        in_l2norm_rd_ptr <= in_l2norm_rd_ptr + 1'b1;
                    end
                end
            end
        end
    end

    // =========================================================================
    // Stage 5: Delay Alignment Buffer (circular BRAM, D-entry delay)
    // =========================================================================
    // unpooled[k] appears at cycle T+8C+28+k after gff_valid_in
    // in_l2norm[k] appears at cycle T+k after gff_valid_in
    // Delay needed: D = 8C+28 cycles. Use circular BRAM with D-offset read.
    //
    // Write: in_l2norm results (one per cycle, starting when in_l2norm_processing)
    // Read:  N cells starting D cycles after unpooling starts
    // Read addr = (wr_ptr + D) % N at time of read
    // This effectively re-timestamps in_l2norm by D cycles.

    (* ram_style = "block" *)
    logic signed [DATA_W-1:0] delay_buf [0:N-1][0:C-1];
    logic [CELL_ADDR_W-1:0]   delay_wr_ptr;
    logic [CELL_ADDR_W-1:0]   delay_rd_ptr;
    logic                     delay_reading;
    logic                     delay_reading_r;

    // Delay offset within circular buffer (wrap to N)
    localparam int DELAY_OFFSET = D % N;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            delay_wr_ptr    <= '0;
            delay_rd_ptr    <= '0;
            delay_reading   <= 1'b0;
            delay_reading_r <= 1'b0;
        end else begin
            delay_reading_r <= delay_reading;

            // Write: store each in_l2norm result
            if (in_l2norm_v) begin
                for (int c = 0; c < C; c++)
                    delay_buf[delay_wr_ptr][c] <= in_l2norm_result[c];
                delay_wr_ptr <= delay_wr_ptr + 1'b1;
            end

            // Read: start from address 0 (all N cells already written by this point)
            if (unpooling && !delay_reading) begin
                delay_reading <= 1'b1;
                delay_rd_ptr   <= '0;
            end else if (delay_reading) begin
                delay_rd_ptr <= delay_rd_ptr + 1'b1;
            end

            if (frame_done) begin
                delay_wr_ptr    <= '0;
                delay_rd_ptr    <= '0;
                delay_reading   <= 1'b0;
                delay_reading_r <= 1'b0;
            end
        end
    end

    logic                     delayed_in_l2norm_v;
    logic signed [DATA_W-1:0] delayed_in_l2norm [0:C-1];

    assign delayed_in_l2norm_v = delay_reading;
    always_comb begin
        for (int c = 0; c < C; c++)
            delayed_in_l2norm[c] = delay_buf[delay_rd_ptr][c];
    end

    // =========================================================================
    // Stage 6: Concat — unpooled (proj_l2norm_reg) + delayed_in_l2norm
    // =========================================================================
    // Both inputs are synchronized to the same N-cell stream by construction:
    // unpooled_v fires for N cycles (starting when proj_l2norm_v fires)
    // delayed_in_l2norm_v fires for N cycles (starting D cycles later)
    // concat_v = unpooled_v && delayed_in_l2norm_v (they overlap perfectly)

    logic                     concat_v;
    logic signed [DATA_W-1:0] concat_out [0:2*C-1];

    always_comb begin
        for (int c = 0; c < C; c++) begin
            concat_out[c]      = proj_l2norm_reg[c];  // unpooled branch
            concat_out[C + c] = delayed_in_l2norm[c]; // in_l2norm branch
        end
    end

    assign concat_v = unpooled_v && delayed_in_l2norm_v;

    // =========================================================================
    // Stage 7: Concat Buffer + Time-Multiplexed Linear 2C→C
    // =========================================================================
    //
    // ARCHITECTURE:
    //   Concat produces N cells at 1/cycle. The linear 2C→C bottleneck consumed
    //   1 cell per 259 cycles — leaving concat_buf backed up for ~2.15M cycles.
    //
    //   TIME_MUX=4: 4 parallel linear engines, each handles C/4=32 output channels.
    //   All 4 engines process the SAME cell simultaneously (channel parallelism).
    //   Each engine computes its 32 channels in 256 cycles (1 MAC/cycle/ic).
    //   4 engines × 32 ch = 128 ch per cell, done in 256 cycles → 1 cell/256 cy.
    //   Throughput = same as original, but ROM bandwidth reduced 64×.
    //
    //   Concat_buf fills while engines compute. When full (N entries), stall fill.
    //   After all N concat cells consumed, linear processes remaining cells from
    //   partial buffer. Total linear time: N × 256 cy = ~4.2M cycles at C=128.
    //
    // ROM ORGANIZATION (TIME_MUX=4):
    //   weight_rom_tm[0..3]: bank e = channels [e*32 .. e*32+31], 256 words each
    //   Each bank: 32 × 256 = 8192 × 16 b = 128 Kb. 4 banks = 512 Kb total.
    //   Read: 1 weight per bank per cycle → 4 words/cycle total ROM bandwidth.
    //   vs original: 256 words/cycle (one per input channel).
    //
    // FSM: L_IDLE → L_RUN (load input + bias) → L_MAC (256 cy) → L_DONE → L_IDLE
    //       L_IDLE → next cell (if concat_buf has entries)
    // =========================================================================

    localparam int PAR_N       = N / TIME_MUX;          // Cells per TM group
    localparam int GROUP_W    = $clog2(TIME_MUX);       // Slot index width
    localparam int CH_PERSlot = C / TIME_MUX;           // Output channels per engine

    (* ram_style = "block" *)
    logic signed [DATA_W-1:0] concat_buf [0:N-1][0:2*C-1];
    logic [CELL_ADDR_W-1:0]   concat_wr_ptr;
    logic [CELL_ADDR_W-1:0]   concat_rd_ptr;
    logic [$clog2(N+1)-1:0]  concat_filled_cnt;

    // ── Concat BRAM write (streaming from concat stage) ────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            concat_wr_ptr    <= '0;
            concat_rd_ptr    <= '0;
            concat_filled_cnt <= '0;
        end else begin
            // Write concat cell (producer) — guard against overflow
            if (concat_v && concat_filled_cnt < N[$clog2(N+1)-1:0]) begin
                for (int ch = 0; ch < 2*C; ch++)
                    concat_buf[concat_wr_ptr][ch] <= concat_out[ch];
                concat_wr_ptr <= concat_wr_ptr + 1'b1;
            end
            // Advance read pointer after processing one cell (FSM loads next)
            if (linear_cell_done)
                concat_rd_ptr <= concat_rd_ptr + 1'b1;

            // concat_filled_cnt: +1 per concat_v, -1 per cell loaded in L_IDLE
            if (concat_v && !(linear_cell_done))
                concat_filled_cnt <= concat_filled_cnt + 1'b1;
            else if (linear_cell_done && !(concat_v))
                concat_filled_cnt <= concat_filled_cnt - 1'b1;
        end
    end

    // ── Weight / Bias ROMs (TIME_MUX independent banks) ─────────────────────
    // Bank e: channels [e*CH_PERSlot .. e*CH_PERSlot+CH_PERSlot-1]
    localparam int WORDS_PER_BANK = CH_PERSlot * (2*C);
    localparam int WEIGHT_SIZE_FULL = C * (2*C);
    (* rom_style = "block" *) logic signed [DATA_W-1:0] weight_rom_tm [0:TIME_MUX-1][0:WORDS_PER_BANK-1];
    (* rom_style = "block" *) logic signed [DATA_W-1:0] bias_rom_tm   [0:TIME_MUX-1][0:CH_PERSlot-1];

    // Load ROMs from files on init
    initial begin
        for (int e = 0; e < TIME_MUX; e++) begin
            for (int i = 0; i < WORDS_PER_BANK; i++) weight_rom_tm[e][i] = '0;
            for (int i = 0; i < CH_PERSlot; i++)     bias_rom_tm[e][i]   = '0;
        end
`ifndef VERILATOR
        // Load full weight matrix into temp array, then partition per bank.
        // Bank e needs weights for output channels [e*CH_PERSlot .. (e+1)*CH_PERSlot-1],
        // stored as [oc][ic] = weight[(e*CH_PERSlot + oc) * (2*C) + ic].
        // Within bank e, local offset = oc * (2*C) + ic.
        if (WEIGHT_FILE != "") begin
            logic signed [DATA_W-1:0] weight_full [0:WEIGHT_SIZE_FULL-1];
            $readmemh({ROM_DIR, "/gff_cat_weight.mem"}, weight_full);
            for (int e = 0; e < TIME_MUX; e++) begin
                for (int oc = 0; oc < CH_PERSlot; oc++) begin
                    for (int ic = 0; ic < 2*C; ic++) begin
                        weight_rom_tm[e][oc * (2*C) + ic] = weight_full[(e * CH_PERSlot + oc) * (2*C) + ic];
                    end
                end
            end
        end
        if (BIAS_FILE != "") begin
            logic signed [DATA_W-1:0] bias_full [0:C-1];
            $readmemh({ROM_DIR, "/gff_cat_bias.mem"}, bias_full);
            for (int e = 0; e < TIME_MUX; e++) begin
                for (int oc = 0; oc < CH_PERSlot; oc++) begin
                    bias_rom_tm[e][oc] = bias_full[e * CH_PERSlot + oc];
                end
            end
        end
`else
        // weights remain zero-initialized (set by the for-loops above)
`endif
    end

    // ── Linear FSM ────────────────────────────────────────────────────────────
    // States: IDLE (wait for concat cells) → RUN (load+bias+MAC) → DONE → IDLE
    typedef enum logic [1:0] {
        L_IDLE,
        L_RUN      // Loading input + bias; MAC cycles + output on last cycle
    } linear_fsm_t;
    linear_fsm_t linear_fsm;
    logic [$clog2(2*C)-1:0] mac_cnt;      // MAC counter per cell (0..2C-1)
    logic linear_cell_done;                // Pulse when one cell fully processed
    logic [$clog2(N+1)-1:0] lin_group_cnt; // How many concat cells consumed

    // ── Per-cell input register (loaded once per cell from concat_buf) ─────────
    logic signed [DATA_W-1:0] cell_in [0:2*C-1];

    // ── Partial accumulator: TIME_MUX engines × CH_PERSlot channels ───────────
    // accum[e][oc] = accumulator for engine e, output channel oc (within slot)
    (* ram_style = "block" *)
    logic signed [ACC_W-1:0] accum_tm [0:TIME_MUX-1][0:CH_PERSlot-1];

    // ── Linear output: assembled from accum_tm → one cell's C channels ─────────
    logic signed [DATA_W-1:0] linear_out [0:C-1];

    // ── GELU input buffer ──────────────────────────────────────────────────────
    (* ram_style = "block" *)
    logic signed [DATA_W-1:0] gelu_buf [0:N-1][0:C-1];
    logic [CELL_ADDR_W-1:0]   gelu_wr_ptr;
    logic [$clog2(N+1)-1:0]  gelu_filled_cnt;
    logic [$clog2(N+1)-1:0] gelu_rd_cnt;

    // ── Linear FSM ─────────────────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            linear_fsm    <= L_IDLE;
            mac_cnt       <= '0;
            lin_group_cnt <= '0;
            linear_cell_done <= 1'b0;
        end else begin
            linear_cell_done <= 1'b0;

            case (linear_fsm)
                L_IDLE: begin
                    mac_cnt <= '0;
                    if (concat_filled_cnt > 0) begin
                        // Load one cell from concat BRAM
                        for (int ic = 0; ic < 2*C; ic++)
                            cell_in[ic] <= concat_buf[concat_rd_ptr][ic];
                        // Initialize accumulators with bias for each TIME_MUX engine
                        for (int e = 0; e < TIME_MUX; e++) begin
                            for (int oc = 0; oc < CH_PERSlot; oc++) begin
                                accum_tm[e][oc] <= ACC_W'(bias_rom_tm[e][oc]) <<< FRAC_W;
                            end
                        end
                        linear_fsm <= L_RUN;
                    end
                end

                L_RUN: begin
                    mac_cnt <= mac_cnt + 1'b1;
                    // MAC: for each TIME_MUX engine, accumulate weight[ch][ic] * in[ic]
                    for (int e = 0; e < TIME_MUX; e++) begin
                        for (int oc = 0; oc < CH_PERSlot; oc++) begin
                            int w_offset = oc * (2*C) + mac_cnt;
                            accum_tm[e][oc] <= accum_tm[e][oc]
                                + ACC_W'(weight_rom_tm[e][w_offset])
                                * ACC_W'(cell_in[mac_cnt]);
                        end
                    end

                    if (mac_cnt == $clog2(2*C)'(2*C - 1)) begin
                        mac_cnt    <= '0;
                        linear_fsm <= L_IDLE;
                        linear_cell_done <= 1'b1;
                    end
                end

                default: linear_fsm <= L_IDLE;
            endcase

            if (frame_done) begin
                linear_fsm    <= L_IDLE;
                mac_cnt       <= '0;
                lin_group_cnt <= '0;
            end
        end
    end

    // ── Linear output: copy from accum_tm to linear_out (C channels) ───────────
    // linear_out is written once per completed cell (L_DONE cycle)
    // then drained one per cycle to gelu_buf
    logic linear_out_valid;
    logic [$clog2(N+1)-1:0] linear_out_count;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            linear_out_valid <= 1'b0;
            linear_out_count <= '0;
        end else begin
            if (linear_cell_done) begin
                // Assemble C channels from TIME_MUX accumulator banks
                for (int e = 0; e < TIME_MUX; e++) begin
                    for (int oc = 0; oc < CH_PERSlot; oc++) begin
                        int ch = e * CH_PERSlot + oc;
                        // Q-format truncation with saturation (Q8.8 accumulator)
                        if (accum_tm[e][oc] > (1 <<< (FRAC_W + DATA_W - 1)) - 1)
                            linear_out[ch] <= {DATA_W{1'b1}} - 1;  // +MAX
                        else if (accum_tm[e][oc] < -(1 <<< (FRAC_W + DATA_W - 1)))
                            linear_out[ch] <= {DATA_W{1'b1}};       // -MAX-1 (min signed)
                        else
                            linear_out[ch] <= accum_tm[e][oc][FRAC_W+DATA_W-1:FRAC_W];
                    end
                end
                linear_out_valid <= 1'b1;
                linear_out_count <= linear_out_count + 1'b1;
            end else begin
                linear_out_valid <= 1'b0;
            end

            if (frame_done) begin
                linear_out_valid <= 1'b0;
                linear_out_count <= '0;
            end
        end
    end

    // ── Linear cells produced counter (for frame completion detection) ─────────
    logic [$clog2(N+1)-1:0] linear_cells_produced;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            linear_cells_produced <= '0;
        end else begin
            if (frame_done || core_state == S_IDLE)
                linear_cells_produced <= '0;
            else if (linear_cell_done)
                linear_cells_produced <= linear_cells_produced + 1'b1;
        end
    end

    // ── gelu_buf: receives 1 linear cell per cycle when linear_out_valid ───────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gelu_wr_ptr    <= '0;
            gelu_filled_cnt <= '0;
            gelu_rd_cnt    <= '0;
        end else begin
            // Write from linear_out (one cell per linear_out_valid pulse)
            // Guard against overflow — drop cell if buffer full
            if (linear_out_valid && gelu_filled_cnt < N[$clog2(N+1)-1:0]) begin
                for (int c = 0; c < C; c++)
                    gelu_buf[gelu_wr_ptr][c] <= linear_out[c];
                gelu_wr_ptr <= gelu_wr_ptr + 1'b1;
            end

            // Read through GELU (one cell per gelu_v)
            if (frame_done) begin
                gelu_wr_ptr    <= '0;
                gelu_rd_cnt    <= '0;
                gelu_filled_cnt <= '0;
            end else if (gelu_v && gelu_filled_cnt > 0) begin
                if (gelu_rd_cnt == N - 1)
                    gelu_rd_cnt <= '0;
                else
                    gelu_rd_cnt <= gelu_rd_cnt + 1'b1;
            end
        end
    end

    // gelu_filled_cnt
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gelu_filled_cnt <= '0;
        end else begin
            if (linear_out_valid && !(gelu_v))
                gelu_filled_cnt <= gelu_filled_cnt + 1'b1;
            else if (!(linear_out_valid) && gelu_v && gelu_filled_cnt > 0)
                gelu_filled_cnt <= gelu_filled_cnt - 1'b1;
        end
    end

    // ── Shared GELU ROM (1 copy instead of per-instance copies) ──────────────
    localparam int GELU_INDEX_BITS = pkg_config::INDEX_BITS;
    localparam int GELU_NUM_PORTS  = C;
    localparam int GELU_FRAC_W     = DATA_W / 2;

    logic [GELU_INDEX_BITS-1:0]    gelu_shared_rd_addr [GELU_NUM_PORTS-1:0];
    logic signed [DATA_W-1:0]       gelu_shared_y_base  [GELU_NUM_PORTS-1:0];
    logic signed [GELU_FRAC_W-1:0] gelu_shared_slope   [GELU_NUM_PORTS-1:0];

    gelu_rom_shared #(
        .DATA_W     (DATA_W),
        .FRAC_W     (GELU_FRAC_W),
        .INDEX_BITS (GELU_INDEX_BITS),
        .NUM_PORTS  (GELU_NUM_PORTS),
        .ROM_DIR    (ROM_DIR)
    ) u_shared_gelu_rom (
        .clk        (clk),
        .rst_n      (rst_n),
        .rd_addr    (gelu_shared_rd_addr),
        .rd_en      (1'b1),
        .y_base_out (gelu_shared_y_base),
        .slope_out  (gelu_shared_slope)
    );

    // GELU: one cell per cycle
    logic                     gelu_v;
    logic signed [DATA_W-1:0] gelu_out [0:C-1];

    gelu_rom #(
        .DATA_W     (DATA_W),
        .FRAC_W     (GELU_FRAC_W),
        .INDEX_BITS (GELU_INDEX_BITS),
        .CHANNELS   (C),
        .USE_DSP    (0),
        .SHARE_ROM  (1),
        .ROM_DIR    (ROM_DIR)
    ) u_gelu (
        .clk          (clk),
        .rst_n        (rst_n),
        .x_in         (gelu_buf[gelu_rd_cnt]),
        .valid_in     (gelu_filled_cnt > 0),
        .y_out        (gelu_out),
        .valid_out    (gelu_v),
        .gelu_en      (1'b1),
        .index_o      (gelu_shared_rd_addr),
        .rom_y_base_i (gelu_shared_y_base),
        .rom_slope_i  (gelu_shared_slope)
    );

    // ── GELU output count (for frame completion and feat_out_last) ─────────────
    logic [$clog2(N+1)-1:0] gelu_output_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gelu_output_cnt <= '0;
        end else begin
            if (frame_done || core_state == S_IDLE)
                gelu_output_cnt <= '0;
            else if (gelu_v)
                gelu_output_cnt <= gelu_output_cnt + 1'b1;
        end
    end

    // =========================================================================
    // Output Assignment
    // =========================================================================
    assign feat_out       = gelu_out;
    assign feat_out_valid = gelu_v;

    // =========================================================================
    // Stream Done Signal (combinatorial)
    // =========================================================================
    // stream_done is combinatorial: asserted on the last GELU output cell.
    // gelu_output_cnt tracks how many cells GELU has produced (0..N-1).
    // When gelu_output_cnt == N-1 and gelu_v fires, this is the last cell.
    logic stream_done;
    assign stream_done = gelu_v && (gelu_output_cnt == N - 1);
    assign feat_out_last = stream_done;

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off



    // synthesis translate_on

endmodule
