// =============================================================================
//  convformer_block.sv
//
//  Hardware implementation of a ConvFormer block (Token Mixer + Channel Mixer).
//  Includes spatial depthwise convolution, LayerNorm, 1x1 Convs (Systolic),
//  and Residual Skip Connections.
//
//  Improvements over v1:
//    - Full valid/ready backpressure: ready_in / out_ready
//    - Frame boundary signals: valid_in_last / valid_out_last
//    - Runtime weight loading for ALL three banks (DW, expand, project)
//      via unified pw_wr_* + dw_wr_* ports — no .mem files required
//    - Dynamic channel clock gating (cfg_clk_gate) for unused activation lanes
//    - Performance counters: cycle, stall, token counts
//    - Single-issue token pipeline with in-flight tracking
//
//  Weight loading protocol:
//    1. Assert pw_wr_en with pw_wr_bank=1, pw_wr_addr, pw_wr_data to fill
//       expand weight bank (EXPANDED_C × MAX_C = 16384 entries).
//    2. Assert pw_wr_en with pw_wr_bank=2 for project weight bank.
//    3. Use dw_wr_en/dw_wr_ch/dw_wr_k/dw_wr_data for DW conv kernel weights.
//    4. All three banks only need to be loaded ONCE before inference begins.
//    5. token_accepted is 0 (ready_in=1) while weights are being written — safe.
//
//  STUB_BEHAVIOR=1: initialises all weight banks to identity (simulation only).
// =============================================================================

import pkg_config::*;
import pkg_sparse_config::*;
import sat_math_pkg::*;

(* keep_hierarchy = "no" *)
module convformer_block #(
    parameter int    TOTAL_LATENCY    = pkg_config::TOTAL_LATENCY,    // bypass FIFO depth (>= pipeline latency)
    parameter bit    STUB_BEHAVIOR     = 1'b0, // 1 = identity weights for simulation
    parameter int    WEIGHT_LOAD_MODE  = 0,   // 0=RUNTIME (write ports), 1=MEM_FILE ($readmemh)
    parameter string WEIGHT_EXP_FILE   = "",  // .mem file for expand weights
    parameter string WEIGHT_PROJ_FILE  = "",  // .mem file for project weights
    parameter string WEIGHT_DW_FILE    = "",   // .mem file for depthwise kernel weights
    parameter bit    EN_SPARSE_SYSTOLIC = pkg_config::EN_SPARSE_SYSTOLIC
)(
    input  logic clk,
    input  logic rst_n,

    // ── Runtime Configuration ─────────────────────────────────────────────────
    input  logic [15:0]      cfg_c,            // Active channels (<= MAX_C)
    input  logic [15:0]      cfg_kernel_size,  // Active kernel size (<= MAX_KERNEL_SIZE)
    input  logic [2:0]       cfg_act_sel,      // Activation select
    input  logic signed [31:0] scale_in,       // StarReLU scale γ
    input  logic signed [31:0] bias_in,        // StarReLU bias β
    input  logic             cfg_clk_gate,     // 1 = gate unused channel activation lanes

    // ── Input Stream ──────────────────────────────────────────────────────────
    input  logic             valid_in,
    input  logic             valid_in_last,    // marks last token of a frame
    input  logic signed [DATA_WIDTH-1:0] token_in [MAX_C-1:0],
    output logic             ready_in,         // IP ready to accept (backpressure)

    // ── Output Stream ─────────────────────────────────────────────────────────
    output logic             valid_out,
    output logic             valid_out_last,   // marks last output token of frame
    output logic signed [DATA_WIDTH-1:0] token_out [MAX_C-1:0],
    input  logic             out_ready,        // consumer ready (backpressure)

    // ── Weight Loading (all three banks) ─────────────────────────────────────
    // Load once before streaming. Safe while ready_in=1 (token_accepted=0).
    //
    // pw_wr_* — pointwise (expand / project) weight banks:
    //   pw_wr_bank: 1 = expand  (addr range 0..WEIGHT_EXP_SIZE-1)
    //               2 = project (addr range 0..WEIGHT_PROJ_SIZE-1)
    //   pw_wr_addr: 14-bit flat index (row-major: out_ch * in_ch + in_idx)
    //   pw_wr_data: 8-bit signed weight value
    input  logic             pw_wr_en,
    input  logic [1:0]       pw_wr_bank,       // 1=expand, 2=project (0/3 ignored)
    input  logic [13:0]      pw_wr_addr,       // up to 16384 entries
    input  logic signed [DATA_WIDTH-1:0] pw_wr_data,

    // dw_wr_* — depthwise conv kernel weights (one tap per channel):
    input  logic             dw_wr_en,
    input  logic [15:0]      dw_wr_ch,         // channel index (0..MAX_C-1)
    input  logic [15:0]      dw_wr_k,          // kernel tap (0..MAX_KERNEL_SIZE-1)
    input  logic signed [DATA_WIDTH-1:0] dw_wr_data,

    // ── Performance Counters ──────────────────────────────────────────────────
    output logic [31:0]      perf_cycle_count,  // clock cycles while token in-flight
    output logic [31:0]      perf_stall_count,  // cycles upstream wanted in but IP busy
    output logic [31:0]      perf_token_count,  // output tokens delivered to consumer

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
    //  Sparse CSR (single instance, broadcasting to both expand & project SAs)
    // =========================================================================
    logic        sparse_en_cfg;
    logic [1:0]  weight_mode_cfg;
    logic [1:0]  activ_mode_cfg;
    logic [15:0] sparse_threshold_cfg;

    // Aggregated performance counters from both expand and project SAs
    logic [31:0] expand_mac_skip_total,   expand_mac_active_total;
    logic [31:0] project_mac_skip_total, project_mac_active_total;
    logic [31:0] agg_mac_skip_cnt;
    logic [31:0] agg_mac_active_cnt;

    assign agg_mac_skip_cnt   = expand_mac_skip_total   + project_mac_skip_total;
    assign agg_mac_active_cnt = expand_mac_active_total  + project_mac_active_total;

    sparse_csr #(
        .DATA_WIDTH(DATA_WIDTH)
    ) u_sparse_csr (
        .clk               (clk),
        .rst_n             (rst_n),
        .csr_req_valid     (sparse_csr_req_valid),
        .csr_req_addr      (sparse_csr_req_addr),
        .csr_req_wdata     (sparse_csr_req_wdata),
        .csr_req_we        (sparse_csr_req_we),
        .csr_req_rdata     (sparse_csr_req_rdata),
        .csr_req_ready     (sparse_csr_req_ready),
        .sparse_en         (sparse_en_cfg),
        .weight_mode       (weight_mode_cfg),
        .activ_mode        (activ_mode_cfg),
        .sparse_threshold  (sparse_threshold_cfg),
        .isect_busy        (1'b0),                    // placeholder for Feature 2
        .encoder_busy      (1'b0),                    // placeholder for Feature 2
        .sparse_active     (sparse_en_cfg),
        .dense_fallback    (1'b0),
        .elastic_fifo_level(8'd0),
        .nz_weight_cnt     (32'd0),
        .nz_activ_cnt      (32'd0),
        .mac_skip_cnt      (agg_mac_skip_cnt),
        .mac_active_cnt    (agg_mac_active_cnt),
        .cycle_cnt         (32'd0),
        .fifo_stall_cnt    (32'd0)
    );

    // =========================================================================
    // Pointwise Weight BRAMs (expand + project 1×1 convolutions)
    //
    // These are runtime-writable dual-port BRAMs:
    //   - Write port: pw_wr_en / pw_wr_bank / pw_wr_addr / pw_wr_data
    //   - Read port:  combinational, indexed by exp_cnt / proj_cnt
    //
    // Synthesis attribute (* ram_style = "block" *) maps to block RAM on FPGA/ASIC.
    // On reset all entries are 0. Load before starting inference.
    //
    // STUB_BEHAVIOR: initial block fills both banks with identity weights
    // (simulation only — synthesis tools ignore initial blocks for RAMs).
    // =========================================================================
    localparam int WEIGHT_EXP_SIZE  = EXPANDED_C * MAX_C;   // 16384 entries
    localparam int WEIGHT_PROJ_SIZE = MAX_C * EXPANDED_C;   // 16384 entries

    (* ram_style = "block" *) logic signed [DATA_WIDTH-1:0] weight_exp_rom  [0:WEIGHT_EXP_SIZE-1];
    (* ram_style = "block" *) logic signed [DATA_WIDTH-1:0] weight_proj_rom [0:WEIGHT_PROJ_SIZE-1];

    // ── Weight initialization / write logic (parameterized by WEIGHT_LOAD_MODE) ──
    localparam int IDENTITY_SHIFT = DATA_WIDTH / 2;

    generate
    if (WEIGHT_LOAD_MODE == 1) begin : gen_mem_weights
        // MEM mode: weights initialized from .mem files at elaboration time
        initial begin
            if (STUB_BEHAVIOR) begin
                for (int o = 0; o < EXPANDED_C; o++)
                    for (int i = 0; i < MAX_C; i++)
                        weight_exp_rom[o * MAX_C + i] = DATA_WIDTH'(1 << IDENTITY_SHIFT);
                for (int o = 0; o < MAX_C; o++)
                    for (int i = 0; i < EXPANDED_C; i++)
                        weight_proj_rom[o * EXPANDED_C + i] = DATA_WIDTH'(1 << IDENTITY_SHIFT);
            end else begin
                if (WEIGHT_EXP_FILE != "")
                    $readmemh(WEIGHT_EXP_FILE, weight_exp_rom);
                if (WEIGHT_PROJ_FILE != "")
                    $readmemh(WEIGHT_PROJ_FILE, weight_proj_rom);
            end
        end
    end else begin : gen_runtime_weights
        // RUNTIME mode: weights loaded via pw_wr_* write ports at runtime
        initial begin
            if (STUB_BEHAVIOR) begin
                for (int o = 0; o < EXPANDED_C; o++)
                    for (int i = 0; i < MAX_C; i++)
                        weight_exp_rom[o * MAX_C + i] = DATA_WIDTH'(1 << IDENTITY_SHIFT);
                for (int o = 0; o < MAX_C; o++)
                    for (int i = 0; i < EXPANDED_C; i++)
                        weight_proj_rom[o * EXPANDED_C + i] = DATA_WIDTH'(1 << IDENTITY_SHIFT);
            end else begin
                for (int i = 0; i < WEIGHT_EXP_SIZE;  i++) weight_exp_rom[i]  = '0;
                for (int i = 0; i < WEIGHT_PROJ_SIZE; i++) weight_proj_rom[i] = '0;
            end
        end

        always_ff @(posedge clk) begin
            if (pw_wr_en) begin
                case (pw_wr_bank)
                    2'd1: weight_exp_rom [pw_wr_addr] <= pw_wr_data;
                    2'd2: weight_proj_rom[pw_wr_addr] <= pw_wr_data;
                    default: ;
                endcase
            end
        end
    end
    endgenerate

    // =========================================================================
    // Back-pressure: multi-issue token tracking (BUG-EROD-CF-002 fix)
    // =========================================================================
    // Previous single-issue design (ready_in = ~token_accepted) deadlocked when
    // cfg_kernel_size > 1: DW conv needs multiple tokens before producing output,
    // but only one token was allowed in flight. Fix: count tokens in flight and
    // deassert ready_in when the pipeline FIFO is full (TOTAL_LATENCY tokens).
    logic [$clog2(TOTAL_LATENCY+1)-1:0] tokens_in_flight;
    logic valid_in_gated;

    assign valid_in_gated = valid_in & ready_in;
    assign ready_in       = (tokens_in_flight < TOTAL_LATENCY);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tokens_in_flight <= '0;
        end else begin
            case ({valid_in_gated, valid_out & out_ready})
                2'b10: tokens_in_flight <= tokens_in_flight + 1;
                2'b01: tokens_in_flight <= tokens_in_flight - 1;
                // 2'b11: simultaneous in/out — net zero change
                // 2'b00: no change
                default: ;
            endcase
        end
    end

    // =========================================================================
    // 1. Bypass FIFO (Residual Connection)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] residual_token     [MAX_C-1:0];
    logic [MAX_C*DATA_WIDTH-1:0]  token_in_flat;
    logic [MAX_C*DATA_WIDTH-1:0]  residual_token_flat;

    always_comb begin
        for (int i = 0; i < MAX_C; i++) begin
            token_in_flat[i*DATA_WIDTH +: DATA_WIDTH] = token_in[i];
            residual_token[i] = residual_token_flat[i*DATA_WIDTH +: DATA_WIDTH];
        end
    end

    // Residual FIFO rd_en: single-MAC reads 1 entry per pw2_valid_pulse;
    // dual-MAC reads 2 entries (first on pw2_valid_pulse, second on res_rd_en_extra).
    logic residual_fifo_rd_en;
    generate
    if (INT8_PACKED) begin : gen_res_fifo_rd
        assign residual_fifo_rd_en = pw2_valid_pulse | gen_residual_dual.res_rd_en_extra;
    end else begin : gen_res_fifo_rd_single
        assign residual_fifo_rd_en = pw2_valid_pulse;
    end
    endgenerate

    fifo #(
        .WIDTH(MAX_C * DATA_WIDTH),
        .DEPTH(TOTAL_LATENCY)
    ) u_residual_fifo (
        .clk   (clk),
        .rst_n (rst_n),
        .wr_en (valid_in_gated),
        .din   (token_in_flat),
        .rd_en (residual_fifo_rd_en),
        .dout  (residual_token_flat),
        .full  (),
        .empty ()
    );

    // Last-flag bypass FIFO: propagates valid_in_last through the same depth
    // as the residual FIFO so valid_out_last aligns with the correct output token.
    logic last_fifo_out;

    fifo #(
        .WIDTH(1),
        .DEPTH(TOTAL_LATENCY)
    ) u_last_fifo (
        .clk   (clk),
        .rst_n (rst_n),
        .wr_en (valid_in_gated),
        .din   (valid_in_last),
        .rd_en (residual_fifo_rd_en),
        .dout  (last_fifo_out),
        .full  (),
        .empty ()
    );

    // FIFO occupancy monitor (synthesis-time depth check)
    // FIXED: was using pw2_valid (multi-cycle) for decrement — must use
    // residual_fifo_rd_en (= pw2_valid_pulse) to match actual FIFO reads.
    logic [$clog2(TOTAL_LATENCY+1)-1:0] fifo_count;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_count <= '0;
        end else begin
            case ({valid_in_gated, residual_fifo_rd_en})
                2'b10: fifo_count <= fifo_count + 1;
                2'b01: fifo_count <= fifo_count - 1;
                default: ;
            endcase
        end
    end

    // =========================================================================
    // 2. Depthwise Convolution (Token Mixer)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] dw_out [MAX_C-1:0];
    logic dw_valid;

    logic dw_ready;
    logic dw_out_ready;

    depthwise_conv_engine #(
        .MAX_C           (MAX_C),
        .DATA_WIDTH      (DATA_WIDTH),
        .WEIGHT_LOAD_MODE(WEIGHT_LOAD_MODE),
        .WEIGHT_FILE     (WEIGHT_DW_FILE)
    ) u_dw_conv (
        .clk             (clk),
        .rst_n           (rst_n),
        .cfg_kernel_size (cfg_kernel_size),
        .cfg_c           (cfg_c),
        // Weight loading now wired to module ports
        .dw_wr_en        (dw_wr_en),
        .dw_wr_ch        (dw_wr_ch),
        .dw_wr_k         (dw_wr_k),
        .dw_wr_data      (dw_wr_data),
        .valid_in        (valid_in_gated),
        .token_in        (token_in),
        .ready_in        (dw_ready),
        .valid_out       (dw_valid),
        .out_ready       (dw_out_ready),
        .token_out       (dw_out)
    );

    // =========================================================================
    // 2b. DW→LN Inter-stage FIFO (4-deep, absorbs LN stall bubbles)
    // =========================================================================
    // Without this FIFO, DW stalls whenever LN stalls (ln_ready=0). When the
    // expand SA is busy, LN freezes, which back-pressures DW directly. A 4-deep
    // FIFO decouples DW from LN: DW can keep producing tokens while LN is frozen,
    // absorbing up to 4 tokens of mismatch before stalling.
    //
    // The FIFO carries DATA_WIDTH-bit data (before sign-extension to 32-bit
    // Q16.16) to save 75% area vs. placing it after sign-extension
    // (512 bits vs 2048 bits per entry).
    localparam int DW_LN_FIFO_DEPTH = pkg_config::DW_LN_FIFO_DEPTH;

    logic [MAX_C*DATA_WIDTH-1:0] dw_out_flat;
    always_comb begin
        for (int i = 0; i < MAX_C; i++)
            dw_out_flat[i*DATA_WIDTH +: DATA_WIDTH] = dw_out[i];
    end

    logic dw_fifo_full;
    logic dw_fifo_empty;
    logic dw_fifo_rd_en;
    logic [MAX_C*DATA_WIDTH-1:0] dw_fifo_dout;
    logic dw_fifo_valid;

    // DW stalls only when FIFO is full — no longer directly coupled to LN
    assign dw_out_ready = ~dw_fifo_full;

    // Write: push when DW has valid output and FIFO not full
    logic dw_fifo_wr_en;
    assign dw_fifo_wr_en = dw_valid & ~dw_fifo_full;

    // Read: pop when FIFO not empty and LN can accept (ln_ready)
    assign dw_fifo_rd_en = ~dw_fifo_empty & ln_ready;

    // LN valid comes from FIFO (not directly from DW)
    assign dw_fifo_valid = ~dw_fifo_empty;

    fifo #(
        .WIDTH(MAX_C * DATA_WIDTH),
        .DEPTH(DW_LN_FIFO_DEPTH),
        .SHOW_AHEAD(1)    // FWFT: zero-latency read for lower DW→LN delay
    ) u_dw_ln_fifo (
        .clk   (clk),
        .rst_n (rst_n),
        .wr_en (dw_fifo_wr_en),
        .din   (dw_out_flat),
        .rd_en (dw_fifo_rd_en),
        .dout  (dw_fifo_dout),
        .full  (dw_fifo_full),
        .empty (dw_fifo_empty)
    );

    // Unflatten FIFO output back to 2D array for sign-extension
    logic signed [DATA_WIDTH-1:0] dw_fifo_out [MAX_C-1:0];
    always_comb begin
        for (int i = 0; i < MAX_C; i++)
            dw_fifo_out[i] = dw_fifo_dout[i*DATA_WIDTH +: DATA_WIDTH];
    end

    // =========================================================================
    // 3. Layer Normalization
    // =========================================================================
    // BUG-EROD-CF-003/004 fix: LayerNorm requires Q16.16 (32-bit) arithmetic
    // regardless of the systolic array's DATA_WIDTH. Use separate LN width and
    // sign-extend / truncate at the boundaries.
    localparam int LN_DATA_WIDTH  = 32;
    localparam int LN_FRAC_WIDTH = 16;

    logic signed [LN_DATA_WIDTH-1:0] ln_out_wide [MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0]    ln_out      [MAX_C-1:0];
    logic                            ln_valid;

    // Sign-extend FIFO output (DATA_WIDTH-bit) to LN_DATA_WIDTH for LN input
    logic signed [LN_DATA_WIDTH-1:0] dw_out_ext [MAX_C-1:0];
    always_comb begin
        for (int i = 0; i < MAX_C; i++)
            dw_out_ext[i] = LN_DATA_WIDTH'(signed'(dw_fifo_out[i])) << (LN_FRAC_WIDTH - FRAC_WIDTH);
    end

    // Truncate LN output from LN_DATA_WIDTH back to DATA_WIDTH
    // (take upper DATA_WIDTH bits after Q16.16 scaling)
    always_comb begin
        for (int i = 0; i < MAX_C; i++)
            ln_out[i] = DATA_WIDTH'(ln_out_wide[i] >>> LN_FRAC_WIDTH);
    end

    // Default gamma=1, beta=0 (identity affine transform) in Q16.16
    logic signed [LN_DATA_WIDTH-1:0] ln_gamma [MAX_C-1:0];
    logic signed [LN_DATA_WIDTH-1:0] ln_beta  [MAX_C-1:0];
    initial begin
        for (int i = 0; i < MAX_C; i++) begin
            ln_gamma[i] = LN_DATA_WIDTH'(1 << LN_FRAC_WIDTH);  // Q16.16 = 1.0
            ln_beta[i]  = '0;
        end
    end

    logic ln_ready;
    logic ln_out_ready;

    // LN output feeds expansion systolic array.
    // Stall LN when the expand SA is busy (exp_running=1). The LN module
    // freezes its entire pipeline on stall (valid_out and y_out held stable).
    // When exp_running drops, the held LN output is captured by exp_buf.
    assign ln_out_ready = ~exp_running;

    layerNorm_fixed #(
        .MAX_C      (MAX_C),
        .DATA_WIDTH (LN_DATA_WIDTH),
        .FRAC_WIDTH (LN_FRAC_WIDTH)
    ) u_layernorm (
        .clk        (clk),
        .rst_n      (rst_n),
        .cfg_c      (cfg_c),
        .valid_in   (dw_fifo_valid),
        .ready_in   (ln_ready),
        .x_in       (dw_out_ext),
        .valid_out  (ln_valid),
        .out_ready  (ln_out_ready),
        .y_out      (ln_out_wide),
        .gamma      (ln_gamma),
        .beta       (ln_beta)
    );

    // =========================================================================
    // 4. Channel Mixer Expand (1×1 Conv via Systolic Array)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] pw1_out       [EXPANDED_C-1:0];
    logic                         pw1_valid;
    logic signed [ACC_WIDTH-1:0]  pw1_rd_result [EXPANDED_C-1:0][MAX_C-1:0];
    // Dual-MAC hi/lo split read results (INT8_PACKED=1)
    logic signed [ACC_WIDTH-1:0]  pw1_rd_result_lo [EXPANDED_C-1:0][MAX_C-1:0];
    logic signed [ACC_WIDTH-1:0]  pw1_rd_result_hi [EXPANDED_C-1:0][MAX_C-1:0];

    logic ln_valid_d;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) ln_valid_d <= 1'b0;
        else        ln_valid_d <= ln_valid;
    end

    logic exp_running;
    logic [15:0] exp_cnt;
    logic exp_en;
    logic exp_cnt_done;
    logic signed [DATA_WIDTH-1:0] a_in_exp [0:EXPANDED_C-1];
    logic signed [ACC_WIDTH-1:0]  psum_zero_expand [EXPANDED_C-1:0][MAX_C-1:0];
    logic signed [ACC_WIDTH-1:0]  psum_zero_project [MAX_C-1:0][EXPANDED_C-1:0];

    assign exp_cnt_done = (cfg_c > 0) && (exp_cnt == cfg_c - 16'd1);

    // Gate for expand SA start: single-MAC starts on any LN output,
    // dual-MAC waits for both token buffers to be filled.
    logic exp_start_gate;
    generate
    if (INT8_PACKED) begin : gen_exp_start_dual
        assign exp_start_gate = exp_buf_pair_ready && !exp_running;
    end else begin : gen_exp_start_single
        assign exp_start_gate = ln_valid && !exp_running;
    end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            exp_running <= 1'b0;
            exp_cnt     <= '0;
        end else begin
            if (exp_start_gate) begin
                exp_running <= 1'b1;
                exp_cnt     <= '0;
            end else if (exp_running) begin
                if (exp_cnt_done) begin
                    exp_running <= 1'b0;
                    exp_cnt     <= '0;
                end else begin
                    exp_cnt <= exp_cnt + 1'b1;
                end
            end
        end
    end

    assign exp_en = exp_running;

    logic signed [DATA_WIDTH-1:0] exp_buf [MAX_C-1:0];
    // Dual-MAC double buffers for batch processing
    logic signed [DATA_WIDTH-1:0] exp_buf_lo [MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] exp_buf_hi [MAX_C-1:0];
    logic [1:0] exp_buf_count;
    logic       exp_buf_pair_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < MAX_C; i++) begin
                exp_buf[i]    <= '0;
                exp_buf_lo[i] <= '0;
                exp_buf_hi[i] <= '0;
            end
            exp_buf_count <= '0;
            exp_buf_pair_ready <= 1'b0;
        end else if (ln_valid && !exp_running) begin
            // Single-MAC: capture to exp_buf
            exp_buf <= ln_out;
            // Dual-MAC: alternate between lo and hi buffers
            if (INT8_PACKED) begin
                case (exp_buf_count)
                    2'd0: begin
                        exp_buf_lo <= ln_out;
                        exp_buf_count <= 2'd1;
                    end
                    2'd1: begin
                        exp_buf_hi <= ln_out;
                        exp_buf_count <= 2'd2;
                        exp_buf_pair_ready <= 1'b1;
                    end
                    default: ;  // pair already ready, SA not started yet
                endcase
            end
        end else if (exp_running) begin
            // SA started: reset pair tracking
            exp_buf_count <= '0;
            exp_buf_pair_ready <= 1'b0;
        end
    end

    always_comb begin
        for (int i = 0; i < EXPANDED_C; i++)
            a_in_exp[i] = weight_exp_rom[i * MAX_C + exp_cnt];
    end

    // Broadcast activation vector to systolic array columns.
    // For matrix-vector product Y = W · x, all N columns of B must carry
    // the same activation element x[k] at each cycle k. Without broadcast,
    // b_in[j]=x[j] is constant and the PE computes x[j]·ΣW[i][k] instead
    // of ΣW[i][k]·x[k] — a data flow bug.
    logic signed [DATA_WIDTH-1:0] b_in_expand [MAX_C-1:0];
    always_comb begin
        for (int j = 0; j < MAX_C; j++)
            b_in_expand[j] = exp_buf[exp_cnt];
    end

    generate
    if (INT8_PACKED) begin : gen_expand_dual_mac
        // Dual-MAC batch: b_lo_in carries token A activations,
        // b_hi_in carries token B activations (both broadcast).

        logic signed [DATA_WIDTH-1:0] b_in_expand_lo [MAX_C-1:0];
        logic signed [DATA_WIDTH-1:0] b_in_expand_hi [MAX_C-1:0];
        always_comb begin
            for (int j = 0; j < MAX_C; j++) begin
                b_in_expand_lo[j] = exp_buf_lo[exp_cnt];
                b_in_expand_hi[j] = exp_buf_hi[exp_cnt];
            end
        end

        logic exp_pair_valid_d;
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) exp_pair_valid_d <= 1'b0;
            else        exp_pair_valid_d <= exp_buf_pair_ready;
        end

        systolic_ping_pong_fifo_top_dual #(
            .M(EXPANDED_C), .N(MAX_C), .DATA_WIDTH(DATA_WIDTH),
            .ACC_WIDTH(ACC_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .DEPTH(16)
        ) u_pw1_expand_dual (
            .clk         (clk),
            .rst_n       (rst_n),
            .en          (exp_en),
            .clear_acc   (exp_buf_pair_ready & ~exp_pair_valid_d),
            .load_psum   (1'b0),
            .a_in        (a_in_exp),
            .b_lo_in    (b_in_expand_lo),
            .b_hi_in    (b_in_expand_hi),
            .psum_in_lo (psum_zero_expand),
            .psum_in_hi (psum_zero_expand),
            .c_start     (exp_buf_pair_ready),
            .c_done      (~exp_buf_pair_ready & exp_pair_valid_d),
            .rd_idx      ('0),
            .c_data_valid(pw1_valid),
            .rd_result_lo(pw1_rd_result_lo),
            .rd_result_hi(pw1_rd_result_hi),
            .bank_ready  (),
            .owner_error (),
            .result_rdy  ()
        );

        always_comb begin
            for (int i = 0; i < EXPANDED_C; i++)
                for (int j = 0; j < MAX_C; j++)
                    pw1_rd_result[i][j] = pw1_rd_result_lo[i][j];
        end
    end else begin : gen_expand_single_mac
        if (EN_SPARSE_SYSTOLIC) begin : gen_expand_sparse
            systolic_ping_pong_fifo_top_sparse #(
                .EN_SPARSE_SYSTOLIC(1'b1),
                .M(EXPANDED_C), .N(MAX_C), .DATA_WIDTH(DATA_WIDTH),
                .ACC_WIDTH(ACC_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .DEPTH(16)
            ) u_pw1_expand_sparse (
                .clk         (clk),
                .rst_n       (rst_n),
                .en          (exp_en),
                .clear_acc   (ln_valid & ~ln_valid_d),
                .load_psum   (1'b0),
                .a_in        (a_in_exp),
                .b_in        (b_in_expand),
                .psum_in     (psum_zero_expand),
                .c_start     (ln_valid),
                .c_done      (~ln_valid & ln_valid_d),
                .rd_idx      ('0),
                .c_data_valid(pw1_valid),
                .rd_result   (pw1_rd_result),
                .bank_ready  (),
                .owner_error (),
                .result_rdy  (),
                .sparse_en_cfg   (sparse_en_cfg),
                .weight_mode_cfg (weight_mode_cfg),
                .activ_mode_cfg  (activ_mode_cfg),
                .mac_skip_total   (expand_mac_skip_total),
                .mac_active_total (expand_mac_active_total)
            );
        end else begin : gen_expand_normal
            systolic_ping_pong_fifo_top #(
                .M(EXPANDED_C), .N(MAX_C), .DATA_WIDTH(DATA_WIDTH),
                .ACC_WIDTH(ACC_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .DEPTH(16)
            ) u_pw1_expand (
                .clk         (clk),
                .rst_n       (rst_n),
                .en          (exp_en),
                .clear_acc   (ln_valid & ~ln_valid_d),
                .load_psum   (1'b0),
                .a_in        (a_in_exp),
                .b_in        (b_in_expand),
                .psum_in     (psum_zero_expand),
                .c_start     (ln_valid),
                .c_done      (~ln_valid & ln_valid_d),
                .rd_idx      ('0),
                .c_data_valid(pw1_valid),
                .rd_result   (pw1_rd_result),
                .bank_ready  (),
                .owner_error (),
                .result_rdy  ()
            );
        end
    end
    endgenerate

    always_comb begin
        for (int i = 0; i < EXPANDED_C; i++)
            pw1_out[i] = pw1_rd_result[i][0][DATA_WIDTH-1:0];
    end

    // =========================================================================
    // 4b. Dual-MAC Activation Sequencing
    //
    // In dual-MAC batch mode, the expand SA produces results for two tokens
    // simultaneously. The activation lanes are single-token, so we sequence:
    //   1. Feed token A (expand_lo) result to activation → proj_buf_lo
    //   2. Feed token B (expand_hi) result to activation → proj_buf_hi
    // Total activation latency: ~8 cycles for 2 tokens (vs ~4 for 1).
    // =========================================================================
    logic                         act_in_valid_r;
    logic signed [DATA_WIDTH-1:0] act_in_data_r  [EXPANDED_C-1:0];
    logic                         act_seq_hi;
    logic                         act_fire_valid;
    logic                         pw1_out_lo_hi_valid;

    generate
    if (INT8_PACKED) begin : gen_act_seq
        // Extract lo/hi expand results for sequencing
        logic signed [DATA_WIDTH-1:0] pw1_out_lo [EXPANDED_C-1:0];
        logic signed [DATA_WIDTH-1:0] pw1_out_hi [EXPANDED_C-1:0];
        always_comb begin
            for (int i = 0; i < EXPANDED_C; i++) begin
                pw1_out_lo[i] = pw1_rd_result_lo[i][0][DATA_WIDTH-1:0];
                pw1_out_hi[i] = pw1_rd_result_hi[i][0][DATA_WIDTH-1:0];
            end
        end

        // FSM states
        localparam int ACT_SEQ_WIDTH = 2;
        logic [ACT_SEQ_WIDTH-1:0] act_seq_state;
        localparam [ACT_SEQ_WIDTH-1:0] ACT_SEQ_IDLE = 2'd0,
                                        ACT_SEQ_LO  = 2'd1,
                                        ACT_SEQ_HI  = 2'd2;

        // Register expand results when pw1_valid fires
        logic signed [DATA_WIDTH-1:0] pw1_out_lo_reg [EXPANDED_C-1:0];
        logic signed [DATA_WIDTH-1:0] pw1_out_hi_reg [EXPANDED_C-1:0];
        logic pw1_result_held;

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                for (int i = 0; i < EXPANDED_C; i++) begin
                    pw1_out_lo_reg[i] <= '0;
                    pw1_out_hi_reg[i] <= '0;
                end
                pw1_result_held <= 1'b0;
            end else if (pw1_valid && !pw1_result_held) begin
                pw1_out_lo_reg <= pw1_out_lo;
                pw1_out_hi_reg <= pw1_out_hi;
                pw1_result_held <= 1'b1;
            end else if (act_seq_state == ACT_SEQ_HI && act_valid) begin
                pw1_result_held <= 1'b0;
            end
        end

        // Activation sequencing FSM
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                act_seq_state <= ACT_SEQ_IDLE;
            end else begin
                case (act_seq_state)
                    ACT_SEQ_IDLE: begin
                        if (pw1_result_held)
                            act_seq_state <= ACT_SEQ_LO;
                    end
                    ACT_SEQ_LO: begin
                        if (act_valid)
                            act_seq_state <= ACT_SEQ_HI;
                    end
                    ACT_SEQ_HI: begin
                        if (act_valid)
                            act_seq_state <= ACT_SEQ_IDLE;
                    end
                    default: act_seq_state <= ACT_SEQ_IDLE;
                endcase
            end
        end

        assign act_seq_hi = (act_seq_state == ACT_SEQ_HI);
        assign act_fire_valid = (act_seq_state == ACT_SEQ_LO) || (act_seq_state == ACT_SEQ_HI);

        // Mux between lo and hi expand results for activation input
        always_comb begin
            for (int i = 0; i < EXPANDED_C; i++) begin
                act_in_data_r[i] = act_seq_hi ? pw1_out_hi_reg[i]
                                              : pw1_out_lo_reg[i];
            end
        end
        assign act_in_valid_r = act_fire_valid;

        // Capture activation output to double buffers
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                for (int i = 0; i < EXPANDED_C; i++) begin
                    proj_buf_lo[i] <= '0;
                    proj_buf_hi[i] <= '0;
                end
                proj_buf_pair_ready <= 1'b0;
            end else begin
                if (act_seq_state == ACT_SEQ_LO && act_valid) begin
                    for (int i = 0; i < EXPANDED_C; i++)
                        proj_buf_lo[i] <= act_out[i];
                end
                if (act_seq_state == ACT_SEQ_HI && act_valid) begin
                    for (int i = 0; i < EXPANDED_C; i++)
                        proj_buf_hi[i] <= act_out[i];
                    proj_buf_pair_ready <= 1'b1;
                end
                if (proj_running)
                    proj_buf_pair_ready <= 1'b0;
            end
        end

    end else begin : gen_act_single
        // Single-MAC: pass-through, no sequencing needed
        always_comb begin
            for (int i = 0; i < EXPANDED_C; i++)
                act_in_data_r[i] = pw1_out[i];
        end
        assign act_in_valid_r = pw1_valid;
        assign act_seq_hi = 1'b0;
        assign act_fire_valid = pw1_valid;
    end
    endgenerate

    // =========================================================================
    // 5. Activation (runtime-selectable: ReLU/LReLU/ReLU6/Sigmoid/GELU/StarReLU)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] act_out        [EXPANDED_C-1:0];
    logic                         act_valid;
    logic                         act_valid_lanes [EXPANDED_C-1:0];
    logic [EXPANDED_C-1:0]        act_ready_in;

    assign act_valid = act_valid_lanes[0];

    // Channel activity enable: power-gate lanes beyond cfg_c*EXPANSION_FACTOR.
    // With cfg_clk_gate=1 the synthesizer can infer integrated clock gating (ICG).
    logic act_ch_en [EXPANDED_C-1:0];
    always_comb begin
        for (int i = 0; i < EXPANDED_C; i++)
            act_ch_en[i] = ~cfg_clk_gate | (i < (cfg_c * EXPANSION_FACTOR));
    end

    // Shared activation ROMs — one copy per type, multi-read-port
    // ROM values are in ROM_WIDTH=32-bit Q16.16 format for interpolation.
    logic signed [pkg_config::ROM_WIDTH-1:0] shared_sig_y_base [EXPANDED_C-1:0];
    logic signed [pkg_config::ROM_WIDTH-1:0] shared_sig_slope  [EXPANDED_C-1:0];
    logic signed [pkg_config::ROM_WIDTH-1:0] shared_gelu_y_base [EXPANDED_C-1:0];
    logic signed [pkg_config::ROM_WIDTH-1:0] shared_gelu_slope  [EXPANDED_C-1:0];

    shared_activation_roms #(
        .DATA_WIDTH    (DATA_WIDTH),
        .FRAC_WIDTH    (FRAC_WIDTH),
        .INDEX_BITS    (pkg_config::INDEX_BITS),
        .NUM_PORTS     (EXPANDED_C),
        .ROM_DATA_BITS (pkg_config::ROM_DATA_BITS),
        .EN_SIGMOID    (1),
        .EN_GELU       (1)
    ) u_shared_roms (
        .clk           (clk),
        .rst_n         (rst_n),
        .x_in          (act_in_data_r),
        .sigmoid_y_base(shared_sig_y_base),
        .sigmoid_slope (shared_sig_slope),
        .gelu_y_base   (shared_gelu_y_base),
        .gelu_slope    (shared_gelu_slope)
    );

    generate
        for (genvar i = 0; i < EXPANDED_C; i++) begin : gen_act
            unified_activation_lane #(
                .DATA_WIDTH(DATA_WIDTH),
                .FRAC_WIDTH(FRAC_WIDTH)
            ) u_act (
                .clk          (clk),
                .rst_n        (rst_n),
                .in_valid     (act_in_valid_r),
                .out_valid    (act_valid_lanes[i]),
                .act_enable   (act_ch_en[i]),
                .act_sel      (cfg_act_sel),
                .leak_shift   ('0),
                .saturation_in('0),
                .scale_in     (scale_in),
                .bias_in      (bias_in),
                .x_in         (act_in_data_r[i]),
                .y_out        (act_out[i]),
                .ready_in     (act_ready_in[i]),
                .out_ready    (~proj_running),
                .sigmoid_y_base_in(shared_sig_y_base[i]),
                .sigmoid_slope_in (shared_sig_slope[i]),
                .gelu_y_base_in   (shared_gelu_y_base[i]),
                .gelu_slope_in    (shared_gelu_slope[i])
            );
        end
    endgenerate

    // =========================================================================
    // 6. Channel Mixer Project (1×1 Conv via Systolic Array)
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] pw2_out       [MAX_C-1:0];
    logic                         pw2_valid;
    logic                         pw2_valid_pulse;
    logic                         pw2_valid_d;

    // Generate single-cycle pulse from pw2_valid for FIFO rd_en
    // (pw2_valid stays high for multiple cycles; we only want to pop once per result)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) pw2_valid_d <= 1'b0;
        else        pw2_valid_d <= pw2_valid;
    end
    assign pw2_valid_pulse = pw2_valid && !pw2_valid_d;
    logic signed [ACC_WIDTH-1:0]  pw2_rd_result [MAX_C-1:0][EXPANDED_C-1:0];
    // Dual-MAC hi/lo split read results (INT8_PACKED=1)
    logic signed [ACC_WIDTH-1:0]  pw2_rd_result_lo [MAX_C-1:0][EXPANDED_C-1:0];
    logic signed [ACC_WIDTH-1:0]  pw2_rd_result_hi [MAX_C-1:0][EXPANDED_C-1:0];

    logic act_valid_d;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) act_valid_d <= 1'b0;
        else        act_valid_d <= act_valid;
    end

    logic proj_running;
    logic [15:0] proj_cnt;
    logic proj_en;
    logic proj_cnt_done;
    logic signed [DATA_WIDTH-1:0] a_in_proj [0:MAX_C-1];

    assign proj_cnt_done = (proj_cnt == (cfg_c * EXPANSION_FACTOR) - 1);

    // Gate for project SA start: single-MAC starts on any activation output,
    // dual-MAC waits for both token buffers to be filled.
    logic proj_start_gate;
    generate
    if (INT8_PACKED) begin : gen_proj_start_dual
        assign proj_start_gate = proj_buf_pair_ready && !proj_running;
    end else begin : gen_proj_start_single
        assign proj_start_gate = act_valid && !proj_running;
    end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            proj_running <= 1'b0;
            proj_cnt     <= '0;
        end else begin
            if (proj_start_gate) begin
                proj_running <= 1'b1;
                proj_cnt     <= '0;
            end else if (proj_running) begin
                if (proj_cnt_done) begin
                    proj_running <= 1'b0;
                    proj_cnt     <= '0;
                end else begin
                    proj_cnt <= proj_cnt + 1'b1;
                end
            end
        end
    end

    assign proj_en = proj_running;

    logic signed [DATA_WIDTH-1:0] proj_buf [EXPANDED_C-1:0];
    // Dual-MAC double buffers for batch processing
    logic signed [DATA_WIDTH-1:0] proj_buf_lo [EXPANDED_C-1:0];
    logic signed [DATA_WIDTH-1:0] proj_buf_hi [EXPANDED_C-1:0];
    logic       proj_buf_pair_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < EXPANDED_C; i++) begin
                proj_buf[i]    <= '0;
                proj_buf_lo[i] <= '0;
                proj_buf_hi[i] <= '0;
            end
            proj_buf_pair_ready <= 1'b0;
        end else if (act_valid && !proj_running) begin
            proj_buf <= act_out;
        end
    end

    always_comb begin
        for (int i = 0; i < MAX_C; i++)
            a_in_proj[i] = weight_proj_rom[i * EXPANDED_C + proj_cnt];
    end

    // Broadcast activation output to systolic array columns (same pattern
    // as expand: all N columns must carry act[k] at cycle k).
    logic signed [DATA_WIDTH-1:0] b_in_project [EXPANDED_C-1:0];
    always_comb begin
        for (int j = 0; j < EXPANDED_C; j++)
            b_in_project[j] = proj_buf[proj_cnt];
    end

    generate
    if (INT8_PACKED) begin : gen_project_dual_mac
        // Dual-MAC batch: b_lo_in carries token A activation output,
        // b_hi_in carries token B activation output (both broadcast).

        logic signed [DATA_WIDTH-1:0] b_in_project_lo [EXPANDED_C-1:0];
        logic signed [DATA_WIDTH-1:0] b_in_project_hi [EXPANDED_C-1:0];
        always_comb begin
            for (int j = 0; j < EXPANDED_C; j++) begin
                b_in_project_lo[j] = proj_buf_lo[proj_cnt];
                b_in_project_hi[j] = proj_buf_hi[proj_cnt];
            end
        end

        logic proj_pair_valid_d;
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) proj_pair_valid_d <= 1'b0;
            else        proj_pair_valid_d <= proj_buf_pair_ready;
        end

        systolic_ping_pong_fifo_top_dual #(
            .M(MAX_C), .N(EXPANDED_C), .DATA_WIDTH(DATA_WIDTH),
            .ACC_WIDTH(ACC_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .DEPTH(16)
        ) u_pw2_project_dual (
            .clk         (clk),
            .rst_n       (rst_n),
            .en          (proj_en),
            .clear_acc   (proj_buf_pair_ready & ~proj_pair_valid_d),
            .load_psum   (1'b0),
            .a_in        (a_in_proj),
            .b_lo_in    (b_in_project_lo),
            .b_hi_in    (b_in_project_hi),
            .psum_in_lo (psum_zero_project),
            .psum_in_hi (psum_zero_project),
            .c_start     (proj_buf_pair_ready),
            .c_done      (~proj_buf_pair_ready & proj_pair_valid_d),
            .rd_idx      ('0),
            .c_data_valid(pw2_valid),
            .rd_result_lo(pw2_rd_result_lo),
            .rd_result_hi(pw2_rd_result_hi),
            .bank_ready  (),
            .owner_error (),
            .result_rdy  ()
        );

        always_comb begin
            for (int i = 0; i < MAX_C; i++)
                for (int j = 0; j < EXPANDED_C; j++)
                    pw2_rd_result[i][j] = pw2_rd_result_lo[i][j];
        end
    end else begin : gen_project_single_mac
        if (EN_SPARSE_SYSTOLIC) begin : gen_project_sparse
            systolic_ping_pong_fifo_top_sparse #(
                .EN_SPARSE_SYSTOLIC(1'b1),
                .M(MAX_C), .N(EXPANDED_C), .DATA_WIDTH(DATA_WIDTH),
                .ACC_WIDTH(ACC_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .DEPTH(16)
            ) u_pw2_project_sparse (
                .clk         (clk),
                .rst_n       (rst_n),
                .en          (proj_en),
                .clear_acc   (act_valid & ~act_valid_d),
                .load_psum   (1'b0),
                .a_in        (a_in_proj),
                .b_in        (b_in_project),
                .psum_in     (psum_zero_project),
                .c_start     (act_valid),
                .c_done      (~act_valid & act_valid_d),
                .rd_idx      ('0),
                .c_data_valid(pw2_valid),
                .rd_result   (pw2_rd_result),
                .bank_ready  (),
                .owner_error (),
                .result_rdy  (),
                .sparse_en_cfg   (sparse_en_cfg),
                .weight_mode_cfg (weight_mode_cfg),
                .activ_mode_cfg  (activ_mode_cfg),
                .mac_skip_total   (project_mac_skip_total),
                .mac_active_total (project_mac_active_total)
            );
        end else begin : gen_project_normal
            systolic_ping_pong_fifo_top #(
                .M(MAX_C), .N(EXPANDED_C), .DATA_WIDTH(DATA_WIDTH),
                .ACC_WIDTH(ACC_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .DEPTH(16)
            ) u_pw2_project (
                .clk         (clk),
                .rst_n       (rst_n),
                .en          (proj_en),
                .clear_acc   (act_valid & ~act_valid_d),
                .load_psum   (1'b0),
                .a_in        (a_in_proj),
                .b_in        (b_in_project),
                .psum_in     (psum_zero_project),
                .c_start     (act_valid),
                .c_done      (~act_valid & act_valid_d),
                .rd_idx      ('0),
                .c_data_valid(pw2_valid),
                .rd_result   (pw2_rd_result),
                .bank_ready  (),
                .owner_error (),
                .result_rdy  ()
            );
        end
    end
    endgenerate

    always_comb begin
        for (int i = 0; i < MAX_C; i++)
            pw2_out[i] = pw2_rd_result[i][0][DATA_WIDTH-1:0];
    end

    // =========================================================================
    // 6b. Dual-MAC Output Sequencing
    //
    // In dual-MAC batch mode, the project SA produces results for two tokens
    // simultaneously. We output them sequentially:
    //   1. Token A (project_lo) + residual → valid_out
    //   2. Token B (project_hi) + residual → valid_out
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] pw2_out_lo [MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] pw2_out_hi [MAX_C-1:0];
    logic                         pw2_out_pair_valid;
    // Output sequencing FSM (dual-MAC mode)
    logic [1:0] out_seq_state;
    localparam [1:0] OUT_SEQ_IDLE = 2'd0, OUT_SEQ_LO = 2'd1, OUT_SEQ_HI = 2'd2;
    logic signed [DATA_WIDTH-1:0] pw2_out_lo_reg [MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] pw2_out_hi_reg [MAX_C-1:0];

    generate
    if (INT8_PACKED) begin : gen_out_seq
        // Extract lo/hi project results
        always_comb begin
            for (int i = 0; i < MAX_C; i++) begin
                pw2_out_lo[i] = pw2_rd_result_lo[i][0][DATA_WIDTH-1:0];
                pw2_out_hi[i] = pw2_rd_result_hi[i][0][DATA_WIDTH-1:0];
            end
        end

        // Register project results when pw2_valid fires
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                for (int i = 0; i < MAX_C; i++) begin
                    pw2_out_lo_reg[i] <= '0;
                    pw2_out_hi_reg[i] <= '0;
                end
            end else if (pw2_valid) begin
                pw2_out_lo_reg <= pw2_out_lo;
                pw2_out_hi_reg <= pw2_out_hi;
            end
        end

        // Output sequencing state machine
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                out_seq_state <= OUT_SEQ_IDLE;
            end else begin
                case (out_seq_state)
                    OUT_SEQ_IDLE: begin
                        if (pw2_valid)
                            out_seq_state <= OUT_SEQ_LO;
                    end
                    OUT_SEQ_LO: begin
                        if (out_ready)
                            out_seq_state <= OUT_SEQ_HI;
                    end
                    OUT_SEQ_HI: begin
                        if (out_ready)
                            out_seq_state <= OUT_SEQ_IDLE;
                    end
                    default: out_seq_state <= OUT_SEQ_IDLE;
                endcase
            end
        end

        assign pw2_out_pair_valid = (out_seq_state != OUT_SEQ_IDLE);

    end else begin : gen_out_single
        always_comb begin
            for (int i = 0; i < MAX_C; i++) begin
                pw2_out_lo[i] = '0;
                pw2_out_hi[i] = '0;
            end
        end
        assign pw2_out_pair_valid = 1'b0;
    end
    endgenerate

    // =========================================================================
    // 7. Residual Addition + Output Hold Register
    // =========================================================================
    // Single-MAC: pw2_valid fires → capture residual+projection, hold until
    // out_ready. The FIFO dout updates 1 cycle after rd_en (pw2_valid_pulse).
    //
    // Dual-MAC: Output sequencing FSM emits two tokens sequentially.
    // The residual FIFO is read twice (1-cycle latency each) to get both
    // tokens' bypass values. Both are registered before the output FSM starts.

    logic        out_valid_r;
    logic        out_last_r;
    logic signed [DATA_WIDTH-1:0] out_token_r [MAX_C-1:0];

    // Dual-MAC: registered residual values for both tokens
    logic signed [DATA_WIDTH-1:0] residual_token_lo [MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] residual_token_hi [MAX_C-1:0];
    logic        last_fifo_out_lo, last_fifo_out_hi;

    generate
    if (INT8_PACKED) begin : gen_residual_dual
        // Residual read FSM: pop 2 entries from FIFO after pw2_valid
        localparam int RES_RD_WIDTH = 2;
        logic [RES_RD_WIDTH-1:0] res_rd_state;
        localparam [RES_RD_WIDTH-1:0] RES_RD_IDLE = 2'd0,
                                       RES_RD_LO  = 2'd1,
                                       RES_RD_HI  = 2'd2;

        logic res_rd_en_extra;

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                res_rd_state <= RES_RD_IDLE;
            end else begin
                case (res_rd_state)
                    RES_RD_IDLE: begin
                        if (pw2_valid_pulse)
                            res_rd_state <= RES_RD_LO;
                    end
                    RES_RD_LO: begin
                        // First FIFO entry is now on dout; issue second read
                        res_rd_state <= RES_RD_HI;
                    end
                    RES_RD_HI: begin
                        // Both entries read; wait for output FSM to finish
                        if (out_seq_state == 2'd0)  // OUT_SEQ_IDLE
                            res_rd_state <= RES_RD_IDLE;
                    end
                    default: res_rd_state <= RES_RD_IDLE;
                endcase
            end
        end

        // Extra FIFO read for second token (first read is pw2_valid_pulse)
        assign res_rd_en_extra = (res_rd_state == RES_RD_LO);

        // Capture residual values when FIFO dout is valid
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                for (int i = 0; i < MAX_C; i++) begin
                    residual_token_lo[i] <= '0;
                    residual_token_hi[i] <= '0;
                end
                last_fifo_out_lo <= 1'b0;
                last_fifo_out_hi <= 1'b0;
            end else begin
                // First entry available 1 cycle after pw2_valid_pulse
                if (res_rd_state == RES_RD_LO) begin
                    for (int i = 0; i < MAX_C; i++)
                        residual_token_lo[i] <= residual_token[i];
                    last_fifo_out_lo <= last_fifo_out;
                end
                // Second entry available 1 cycle after extra read
                if (res_rd_state == RES_RD_HI) begin
                    for (int i = 0; i < MAX_C; i++)
                        residual_token_hi[i] <= residual_token[i];
                    last_fifo_out_hi <= last_fifo_out;
                end
            end
        end

        // Output logic: use sequencing FSM
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                out_valid_r <= 1'b0;
                out_last_r  <= 1'b0;
                for (int i = 0; i < MAX_C; i++) out_token_r[i] <= '0;
            end else begin
                case (out_seq_state)
                    2'd1: begin  // OUT_SEQ_LO: token A
                        out_valid_r <= 1'b1;
                        out_last_r  <= last_fifo_out_lo;
                        for (int i = 0; i < MAX_C; i++)
                            out_token_r[i] <= (i < cfg_c)
                                ? DATA_WIDTH'(sat_add_signed(residual_token_lo[i], pw2_out_lo_reg[i], DATA_WIDTH))
                                : '0;
                    end
                    2'd2: begin  // OUT_SEQ_HI: token B
                        out_valid_r <= 1'b1;
                        out_last_r  <= last_fifo_out_hi;
                        for (int i = 0; i < MAX_C; i++)
                            out_token_r[i] <= (i < cfg_c)
                                ? DATA_WIDTH'(sat_add_signed(residual_token_hi[i], pw2_out_hi_reg[i], DATA_WIDTH))
                                : '0;
                    end
                    default: begin  // OUT_SEQ_IDLE
                        if (out_ready)
                            out_valid_r <= 1'b0;
                    end
                endcase
            end
        end

    end else begin : gen_residual_single
        // Single-MAC: original output logic
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                out_valid_r <= 1'b0;
                out_last_r  <= 1'b0;
                for (int i = 0; i < MAX_C; i++) out_token_r[i] <= '0;
            end else begin
                if (pw2_valid && (!out_valid_r || out_ready)) begin
                    out_valid_r <= 1'b1;
                    out_last_r  <= last_fifo_out;
                    for (int i = 0; i < MAX_C; i++)
                        out_token_r[i] <= (i < cfg_c) ? DATA_WIDTH'(sat_add_signed(residual_token[i], pw2_out[i], DATA_WIDTH)) : '0;
                end else if (out_valid_r && out_ready && !pw2_valid) begin
                    out_valid_r <= 1'b0;
                    out_last_r  <= 1'b0;
                end
            end
        end
    end
    endgenerate

    assign valid_out      = out_valid_r;
    assign valid_out_last = out_valid_r & out_last_r;
    assign token_out      = out_token_r;

    // =========================================================================
    // Performance Counters
    // =========================================================================
    // perf_cycle_count: clock cycles while a token is in the pipeline
    // perf_stall_count: cycles where upstream presented valid but IP was busy
    // perf_token_count: output tokens successfully delivered to consumer

`ifdef DEBUG
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) perf_cycle_count <= '0;
        else if (tokens_in_flight > 0) perf_cycle_count <= perf_cycle_count + 1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) perf_stall_count <= '0;
        else if (valid_in && !ready_in) perf_stall_count <= perf_stall_count + 1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) perf_token_count <= '0;
        else if (valid_out && out_ready) perf_token_count <= perf_token_count + 1;
    end
`else
    assign perf_cycle_count = '0;
    assign perf_stall_count = '0;
    assign perf_token_count = '0;
`endif

    // =========================================================================
    // Inline SVA Assertions
    // =========================================================================

    // FIFO must never overflow (token_accepted prevents re-entry)
    CONVFORMER_FIFO_NO_OVERFLOW: assert property (
        @(posedge clk) disable iff (!rst_n)
        fifo_count < TOTAL_LATENCY
    ) else $error("convformer_block: residual FIFO overflow — TOTAL_LATENCY too small");

    // Output must remain stable while valid_out && !out_ready
    CONVFORMER_OUT_STABLE: assert property (
        @(posedge clk) disable iff (!rst_n)
        (valid_out && !out_ready) |=> valid_out
    ) else $error("convformer_block: valid_out dropped before out_ready");

    // Pipeline occupancy must never exceed FIFO depth
    CONVFORMER_PIPELINE_BOUNDS: assert property (
        @(posedge clk) disable iff (!rst_n)
        tokens_in_flight <= TOTAL_LATENCY
    ) else $error("convformer_block: tokens_in_flight exceeds TOTAL_LATENCY");

    // LN must not push data while expand SA is busy (backpressure invariant)
    CONVFORMER_LN_BP_EXPAND: assert property (
        @(posedge clk) disable iff (!rst_n)
        exp_running |-> !ln_out_ready
    ) else $error("convformer_block: ln_out_ready high while expand SA busy — data loss risk");

    // Activation must not push data while project SA is busy (backpressure invariant)
    CONVFORMER_ACT_BP_PROJECT: assert property (
        @(posedge clk) disable iff (!rst_n)
        proj_running && act_valid |-> !act_ready_in[0]
    ) else $error("convformer_block: activation not stalled while project SA busy — data loss risk");

    // Broadcast invariant: all columns of expand b_in carry the same element
    CONVFORMER_BROADCAST_EXPAND: assert property (
        @(posedge clk) disable iff (!rst_n)
        exp_running && cfg_c > 0 |-> b_in_expand[0] == b_in_expand[MAX_C-1]
    ) else $error("convformer_block: expand b_in not broadcast — data flow bug");

    // Broadcast invariant: all columns of project b_in carry the same element
    CONVFORMER_BROADCAST_PROJECT: assert property (
        @(posedge clk) disable iff (!rst_n)
        proj_running && cfg_c > 0 |-> b_in_project[0] == b_in_project[EXPANDED_C-1]
    ) else $error("convformer_block: project b_in not broadcast — data flow bug");

    // DW→LN FIFO must never overflow (DW stalls when FIFO full)
    CONVFORMER_DW_LN_FIFO_NO_OVERFLOW: assert property (
        @(posedge clk) disable iff (!rst_n)
        dw_fifo_full |-> !dw_fifo_wr_en
    ) else $error("convformer_block: DW→LN FIFO overflow — backpressure broken");

    // DW→LN FIFO read only when not empty
    CONVFORMER_DW_LN_FIFO_NO_UNDERFLOW: assert property (
        @(posedge clk) disable iff (!rst_n)
        dw_fifo_empty |-> !dw_fifo_rd_en
    ) else $error("convformer_block: DW→LN FIFO underflow");

endmodule
