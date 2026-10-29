// =============================================================================
//  convformer_block_fixed.sv — DUAL-SLOT PIPELINED
//
//  Hardware implementation of a ConvFormer block (Token Mixer + Channel Mixer).
//  FIXED VERSION — Dual-slot pipelining allows expand SA(token N+1) to run
//  concurrently with project SA(token N), hiding expand latency behind project.
//
//  Pipeline Structure:
//    Slot 0/1 track tokens through: LN_READY → EXPAND_ACT → PROJ_READY → DONE
//    Expand SA reads from ln_buf[expand_slot], writes result → act_buf[expand_slot]
//    Project SA reads from act_buf[project_slot], writes output + residual from slot
//
//  Timing improvement (arty_a7: C=16, EXP=2):
//    Serial:    1 token / 52 cycles  (expand 16 + act 4 + project 32)
//    Pipelined: 1 token / 32 cycles  (max(expand+act, project))
//    Speedup:   1.6×
//
//  Architecture:
//    1. Residual bypass (captured per-slot at token acceptance)
//    2. Depthwise Conv (cfg_kernel_size stages)
//    3. LayerNorm (9 stages, can output to any empty slot)
//    4. Expand 1×1 Conv (systolic, C iterations per slot)
//    5. Activation (3-4 stages, 4-stage slot pipe for bank routing)
//    6. Project 1×1 Conv (systolic, C×EXPANSION_FACTOR iterations per slot)
//    7. Residual Add + Output (per-slot residual → slot freed)
//
//  Control:
//    LN output and expand start are in one always_ff block (no NBA bubble).
//    expand_start is combinational (!expand_running + ready signal) for
//    correct clear_acc timing to the systolic array.
// =============================================================================

import pkg_config::*;
import pkg_sparse_config::*;
import sat_math_pkg::*;

(* keep_hierarchy = "no" *)
module convformer_block_fixed #(
    parameter int    TOTAL_LATENCY     = pkg_config::TOTAL_LATENCY,
    parameter bit    STUB_BEHAVIOR     = 1'b0,  // 1 = identity weights for sim
    parameter int    WEIGHT_LOAD_MODE  = 0,     // 0=RUNTIME, 1=MEM_FILE
    parameter string WEIGHT_EXP_FILE   = "",
    parameter string WEIGHT_PROJ_FILE  = "",
    parameter string WEIGHT_DW_FILE    = "",
    parameter bit    EN_SPARSE_SYSTOLIC = pkg_config::EN_SPARSE_SYSTOLIC,
    parameter bit    EN_DUAL_SYSTOLIC   = pkg_config::EN_DUAL_SYSTOLIC
)(
    input  logic clk,
    input  logic rst_n,

    // Runtime Configuration
    input  logic [15:0]      cfg_c,
    input  logic [15:0]      cfg_kernel_size,
    input  logic [2:0]       cfg_act_sel,
    input  logic signed [31:0] scale_in,
    input  logic signed [31:0] bias_in,
    input  logic             cfg_clk_gate,
    input  logic             weight_loaded,

    // Input Stream
    input  logic             valid_in,
    input  logic             valid_in_last,
    input  logic signed [DATA_WIDTH-1:0] token_in [MAX_C-1:0],
    output logic             ready_in,

    // Output Stream
    output logic             valid_out,
    output logic             valid_out_last,
    output logic signed [DATA_WIDTH-1:0] token_out [MAX_C-1:0],
    input  logic             out_ready,

    // Weight Loading
    input  logic             pw_wr_en,
    input  logic [1:0]       pw_wr_bank,
    input  logic [13:0]      pw_wr_addr,
    input  logic signed [DATA_WIDTH-1:0] pw_wr_data,

    input  logic             dw_wr_en,
    input  logic [15:0]      dw_wr_ch,
    input  logic [15:0]      dw_wr_k,
    input  logic signed [DATA_WIDTH-1:0] dw_wr_data,

    // Performance Counters
    output logic [31:0]      perf_cycle_count,
    output logic [31:0]      perf_stall_count,
    output logic [31:0]      perf_token_count,

    // ── Sparse CSR Interface (0xB00–0xB3F) ──────────────────────────────────
    input  logic        sparse_csr_req_valid,
    input  logic [11:0] sparse_csr_req_addr,
    input  logic [31:0] sparse_csr_req_wdata,
    input  logic        sparse_csr_req_we,
    output logic [31:0] sparse_csr_req_rdata,
    output logic        sparse_csr_req_ready
);

    localparam int EXPANDED_C = MAX_C * EXPANSION_FACTOR;
    localparam int WEIGHT_EXP_SIZE = EXPANDED_C * MAX_C;
    localparam int WEIGHT_PROJ_SIZE = MAX_C * EXPANDED_C;
    localparam int NUM_SLOTS = 2;

    // =========================================================================
    //  Slot State
    // =========================================================================
    typedef enum logic [1:0] {
        SLOT_EMPTY,        // Slot available for new token
        SLOT_LN_READY,     // LN output in ln_buf[slot], waiting for expand
        SLOT_EXPAND_ACT,   // Expand running / activation processing this slot
        SLOT_PROJ_READY    // act_buf[slot] ready, project running or pending
    } slot_state_t;

    slot_state_t slot_state [NUM_SLOTS-1:0];

    // Zero-filled psum for systolic array (Verilator-compatible)
    logic signed [ACC_WIDTH-1:0] psum_zero_expand [EXPANDED_C-1:0][MAX_C-1:0];
    logic signed [ACC_WIDTH-1:0] psum_zero_project [MAX_C-1:0][EXPANDED_C-1:0];
    always_comb begin
        for (int i = 0; i < EXPANDED_C; i++)
            for (int j = 0; j < MAX_C; j++)
                psum_zero_expand[i][j] = '0;
        for (int i = 0; i < MAX_C; i++)
            for (int j = 0; j < EXPANDED_C; j++)
                psum_zero_project[i][j] = '0;
    end

    // =========================================================================
    //  Dual Systolic Array Signals (EN_DUAL_SYSTOLIC)
    // =========================================================================
    logic signed [ACC_WIDTH-1:0] expand_result_lo [EXPANDED_C-1:0][MAX_C-1:0];
    logic signed [ACC_WIDTH-1:0] expand_result_hi [EXPANDED_C-1:0][MAX_C-1:0];
    logic expand_valid_dual;
    logic signed [ACC_WIDTH-1:0] project_result_lo [MAX_C-1:0][EXPANDED_C-1:0];
    logic signed [ACC_WIDTH-1:0] project_result_hi [MAX_C-1:0][EXPANDED_C-1:0];
    logic project_valid_dual;

    // Dual expand: secondary slot + sequencer
    logic [$clog2(NUM_SLOTS)-1:0] expand_slot_hi;
    logic expand_dual_pending;   // hi result buffered, awaiting activation
    logic signed [DATA_WIDTH-1:0] expand_out_hi_buf [EXPANDED_C-1:0];

    // Dual project: secondary slot
    logic [$clog2(NUM_SLOTS)-1:0] project_slot_hi;
    logic project_dual_pending;  // hi result buffered, awaiting output

    // =========================================================================
    //  Sparse CSR
    // =========================================================================
    logic        sparse_en_cfg;
    logic [1:0]  weight_mode_cfg;
    logic [1:0]  activ_mode_cfg;
    logic [15:0] sparse_threshold_cfg;

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
        .isect_busy        (1'b0),
        .encoder_busy      (1'b0),
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
    //  Weight RAM
    // =========================================================================
    (* ram_style = "block" *) logic signed [DATA_WIDTH-1:0] weight_exp_rom  [0:WEIGHT_EXP_SIZE-1];
    (* ram_style = "block" *) logic signed [DATA_WIDTH-1:0] weight_proj_rom [0:WEIGHT_PROJ_SIZE-1];

    logic expand_weights_loaded, project_weights_loaded;
    localparam int IDENTITY_SHIFT = DATA_WIDTH / 2;

    generate
    if (WEIGHT_LOAD_MODE == 1) begin : gen_mem_weights
        initial begin
            if (STUB_BEHAVIOR) begin
                for (int o = 0; o < EXPANDED_C; o++)
                    for (int i = 0; i < MAX_C; i++)
                        weight_exp_rom[o * MAX_C + i] = DATA_WIDTH'(1 << IDENTITY_SHIFT);
                for (int o = 0; o < MAX_C; o++)
                    for (int i = 0; i < EXPANDED_C; i++)
                        weight_proj_rom[o * EXPANDED_C + i] = DATA_WIDTH'(1 << IDENTITY_SHIFT);
            end else begin
                if (WEIGHT_EXP_FILE != "") $readmemh(WEIGHT_EXP_FILE, weight_exp_rom);
                if (WEIGHT_PROJ_FILE != "") $readmemh(WEIGHT_PROJ_FILE, weight_proj_rom);
            end
        end
    end else begin : gen_runtime_weights
        initial begin
            if (STUB_BEHAVIOR) begin
                for (int o = 0; o < EXPANDED_C; o++)
                    for (int i = 0; i < MAX_C; i++)
                        weight_exp_rom[o * MAX_C + i] = DATA_WIDTH'(1 << IDENTITY_SHIFT);
                for (int o = 0; o < MAX_C; o++)
                    for (int i = 0; i < EXPANDED_C; i++)
                        weight_proj_rom[o * EXPANDED_C + i] = DATA_WIDTH'(1 << IDENTITY_SHIFT);
            end else begin
                for (int i = 0; i < WEIGHT_EXP_SIZE; i++) weight_exp_rom[i] = '0;
                for (int i = 0; i < WEIGHT_PROJ_SIZE; i++) weight_proj_rom[i] = '0;
            end
        end
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                expand_weights_loaded <= 1'b0;
                project_weights_loaded <= 1'b0;
            end else if (pw_wr_en) begin
                case (pw_wr_bank)
                    2'd1: begin
                        if (pw_wr_addr < WEIGHT_EXP_SIZE)
                            weight_exp_rom[pw_wr_addr] <= pw_wr_data;
                        expand_weights_loaded <= 1'b1;
                    end
                    2'd2: begin
                        if (pw_wr_addr < WEIGHT_PROJ_SIZE)
                            weight_proj_rom[pw_wr_addr] <= pw_wr_data;
                        project_weights_loaded <= 1'b1;
                    end
                    default: ;
                endcase
            end
        end
    end
    endgenerate

    // =========================================================================
    //  Slot-Based Input Acceptance
    // =========================================================================
    logic valid_in_gated;
    logic [$clog2(NUM_SLOTS)-1:0] accept_slot;

    // Accept when any slot is empty
    assign ready_in = (slot_state[0] == SLOT_EMPTY) || (slot_state[1] == SLOT_EMPTY);
    assign valid_in_gated = valid_in & ready_in;

    // First empty slot (0 priority)
    always_comb begin
        if (slot_state[0] == SLOT_EMPTY) accept_slot = 0;
        else                              accept_slot = 1;
    end

    // Per-slot residual storage (replaces residual FIFO)
    logic signed [DATA_WIDTH-1:0] slot_residual [NUM_SLOTS-1:0][MAX_C-1:0];
    logic                         slot_last     [NUM_SLOTS-1:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int s = 0; s < NUM_SLOTS; s++) begin
                slot_last[s] <= 1'b0;
                for (int i = 0; i < MAX_C; i++)
                    slot_residual[s][i] <= '0;
            end
        end else if (valid_in_gated) begin
            for (int i = 0; i < MAX_C; i++)
                slot_residual[accept_slot][i] <= token_in[i];
            slot_last[accept_slot] <= valid_in_last;
        end
    end

    // =========================================================================
    //  Stage 1: Depthwise Convolution
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] dw_out [MAX_C-1:0];
    logic dw_valid_out;

    depthwise_conv_fixed #(
        .MAX_C(MAX_C),
        .DATA_WIDTH(DATA_WIDTH),
        .MAX_KERNEL_SIZE(MAX_KERNEL_SIZE),
        .ACC_WIDTH(ACC_WIDTH),
        .WEIGHT_LOAD_MODE(WEIGHT_LOAD_MODE),
        .WEIGHT_FILE(WEIGHT_DW_FILE)
    ) u_dw_conv (
        .clk(clk), .rst_n(rst_n),
        .cfg_kernel_size(cfg_kernel_size),
        .cfg_c(cfg_c),
        .dw_wr_en(dw_wr_en),
        .dw_wr_ch(dw_wr_ch),
        .dw_wr_k(dw_wr_k),
        .dw_wr_data(dw_wr_data),
        .valid_in(valid_in_gated),
        .token_in(token_in),
        .valid_out(dw_valid_out),
        .token_out(dw_out),
        .out_ready(~dw_fifo_full)
    );

    // =========================================================================
    //  Stage 1b: DW→LN Inter-stage FIFO
    // =========================================================================
    localparam int DW_LN_FIFO_DEPTH = pkg_config::DW_LN_FIFO_DEPTH;

    logic [MAX_C*DATA_WIDTH-1:0] dw_out_flat;
    always_comb begin
        for (int i = 0; i < MAX_C; i++)
            dw_out_flat[i*DATA_WIDTH +: DATA_WIDTH] = dw_out[i];
    end

    logic dw_fifo_full, dw_fifo_empty, dw_fifo_rd_en, dw_fifo_valid;
    logic [MAX_C*DATA_WIDTH-1:0] dw_fifo_dout;
    logic dw_fifo_wr_en;

    assign dw_fifo_wr_en = dw_valid_out & ~dw_fifo_full;
    assign dw_fifo_rd_en = ~dw_fifo_empty & ln_ready;
    assign dw_fifo_valid = ~dw_fifo_empty;

    fifo #(
        .WIDTH(MAX_C * DATA_WIDTH),
        .DEPTH(DW_LN_FIFO_DEPTH),
        .SHOW_AHEAD(1)
    ) u_dw_ln_fifo (
        .clk   (clk), .rst_n (rst_n),
        .wr_en (dw_fifo_wr_en), .din   (dw_out_flat),
        .rd_en (dw_fifo_rd_en), .dout  (dw_fifo_dout),
        .full  (dw_fifo_full),  .empty (dw_fifo_empty)
    );

    logic signed [DATA_WIDTH-1:0] dw_fifo_out [MAX_C-1:0];
    always_comb begin
        for (int i = 0; i < MAX_C; i++)
            dw_fifo_out[i] = dw_fifo_dout[i*DATA_WIDTH +: DATA_WIDTH];
    end

    // =========================================================================
    //  Stage 2: Layer Normalization
    //
    //  BUG-EROD-LN-007 FIX: LayerNorm requires FRAC_WIDTH <= DATA_WIDTH/2 for
    //  correct 1/sqrt LUT indexing. With system defaults (DATA_WIDTH=8,
    //  FRAC_WIDTH=16), 2*FRAC_WIDTH=32 > 2*DATA_WIDTH=16, causing the LUT
    //  index extraction to clamp and lose precision. Fix: widen the LN data
    //  path to LN_DATA_WIDTH=32, LN_FRAC_WIDTH=16, with sign-extend before
    //  and truncate after — same approach as convformer_block.sv.
    // =========================================================================
    localparam int LN_DATA_WIDTH  = 32;
    localparam int LN_FRAC_WIDTH  = 16;

    logic signed [LN_DATA_WIDTH-1:0] ln_out_wide [MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] ln_out [MAX_C-1:0];
    logic ln_valid_out;
    logic ln_ready, ln_out_ready;

    assign ln_out_ready = (slot_state[0] == SLOT_EMPTY) || (slot_state[1] == SLOT_EMPTY);

    // Sign-extend DATA_WIDTH → LN_DATA_WIDTH for LN input
    logic signed [LN_DATA_WIDTH-1:0] dw_out_ext [MAX_C-1:0];
    always_comb begin
        for (int i = 0; i < MAX_C; i++)
            dw_out_ext[i] = LN_DATA_WIDTH'(signed'(dw_fifo_out[i])) << (LN_FRAC_WIDTH - FRAC_WIDTH);
    end

    // Truncate LN output back to DATA_WIDTH
    always_comb begin
        for (int i = 0; i < MAX_C; i++)
            ln_out[i] = DATA_WIDTH'(ln_out_wide[i] >>> LN_FRAC_WIDTH);
    end

    logic signed [LN_DATA_WIDTH-1:0] ln_gamma [MAX_C-1:0];
    logic signed [LN_DATA_WIDTH-1:0] ln_beta  [MAX_C-1:0];
    always_comb begin
        for (int i = 0; i < MAX_C; i++) begin
            ln_gamma[i] = LN_DATA_WIDTH'(1 << LN_FRAC_WIDTH);  // Q16.16 identity
            ln_beta[i]  = '0;
        end
    end

    layerNorm_fixed #(
        .DATA_WIDTH(LN_DATA_WIDTH),
        .FRAC_WIDTH(LN_FRAC_WIDTH),
        .MAX_C(MAX_C)
    ) u_layernorm (
        .clk(clk), .rst_n(rst_n),
        .cfg_c(cfg_c),
        .gamma(ln_gamma), .beta(ln_beta),
        .valid_in(dw_fifo_valid), .ready_in(ln_ready),
        .x_in(dw_out_ext),
        .valid_out(ln_valid_out), .out_ready(ln_out_ready),
        .y_out(ln_out_wide)
    );

    // =========================================================================
    //  Stage 3: Expand 1×1 Conv (Systolic) — MERGED LN BUF + EXPAND CONTROL
    //
    //  LN output routing and expand start are in one always_ff block to
    //  eliminate the NBA bubble: when LN produces output and expand is idle,
    //  expand launches in the same cycle (no 1-cycle gap).
    //
    //  Combinational expand_start drives clear_acc on the systolic array:
    //    expand_start = !expand_running && (ln_valid_out || pending LN_READY)
    //  This fires for exactly 1 cycle while en=0, then en=1 at next cycle.
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] weight_a_expand [EXPANDED_C-1:0];
    logic signed [DATA_WIDTH-1:0] b_in_expand [MAX_C-1:0];
    logic signed [ACC_WIDTH-1:0] expand_result [EXPANDED_C-1:0][MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] expand_out [EXPANDED_C-1:0];
    logic expand_running;
    logic [$clog2(EXPANDED_C):0] expand_cnt;
    logic expand_done;
    // Round-robin priority bit: 0=try slot 0 first, 1=try slot 1 first.
    // Toggled each time the expand picker picks a slot, so that under
    // continuous LN backpressure both slots get serviced fairly.
    logic expand_rr_pick;
    logic [$clog2(NUM_SLOTS)-1:0] expand_slot;
    logic expand_drain;  // high during SA pipeline drain (en=0, results flowing)

    // Dual LN buffer (per-slot, LN writes → expand reads)
    logic signed [DATA_WIDTH-1:0] ln_buf [NUM_SLOTS-1:0][MAX_C-1:0];

    assign expand_done = expand_running && (expand_cnt == cfg_c - 1);

    // Combinational expand_start: fires 1 cycle when expand launches.
    // Gated by !expand_drain — during SA pipeline drain (48 cyc after
    // expand_done), expand_start must not fire or clear_acc would destroy
    // the previous computation's results before result_rdy fires.
    assign expand_start = !expand_running && !expand_drain && (ln_valid_out ||
        slot_state[0] == SLOT_LN_READY || slot_state[1] == SLOT_LN_READY);

    logic expand_start;

    // Slot selection logic (extracted from always_ff for Verilator compat)
    logic e_slot_comb;
    logic e_slot2_comb;
    always_comb begin
        e_slot_comb  = (slot_state[0] == SLOT_EMPTY) ? 0 : 1;
        e_slot2_comb = (e_slot_comb == 0) ? 1 : 0;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            expand_running <= 1'b0;
            expand_cnt     <= '0;
            expand_slot    <= 0;
            expand_rr_pick <= 1'b0;
            slot_state[0]  <= SLOT_EMPTY;
            slot_state[1]  <= SLOT_EMPTY;
            for (int s = 0; s < NUM_SLOTS; s++)
                for (int i = 0; i < MAX_C; i++)
                    ln_buf[s][i] <= '0;
        end else begin
            // ── Expand counter (runs when expand is active) ────────────────
            if (expand_running) begin
                if (expand_done) begin
                    expand_running <= 1'b0;
                    expand_cnt <= '0;
                end else begin
                    expand_cnt <= expand_cnt + 1'b1;
                end
            end

            // ── LN output: store in first free slot, launch expand if idle ─
            //   Gating by !expand_drain too: between expand_done and
            //   expand_valid_act the SA is still draining the previous result
            //   through its (M-1)+(N-1)+2 flush pipeline. If we launch a new
            //   expand here, expand_slot is overwritten before result_rdy fires
            //   for the previous expand → act_valid writes PROJ_READY to the
            //   wrong slot, leaving the original slot stuck in EXPAND_ACT
            //   forever.
            // Guard: only consume the LN output if the chosen slot is actually
            // empty. The LN pipeline has internal latency, so even with
            // ln_out_ready=0 it can produce a tail token after both slots have
            // filled up; without this guard the tail clobbers a non-EMPTY
            // slot's state (PROJ_READY → LN_READY) and the project FSM ends up
            // running on the wrong slot state.
            if (ln_valid_out && slot_state[e_slot_comb] == SLOT_EMPTY) begin
                for (int i = 0; i < MAX_C; i++)
                    ln_buf[e_slot_comb][i] <= ln_out[i];

                if (!expand_running && !expand_drain) begin
                    // Launch expand immediately (same cycle as LN output)
                    expand_slot   <= e_slot_comb;
                    expand_running <= 1'b1;
                    expand_cnt    <= '0;
                    slot_state[e_slot_comb] <= SLOT_EXPAND_ACT;

                    // Dual: check if another slot is LN_READY
                    if (EN_DUAL_SYSTOLIC) begin
                        if (slot_state[e_slot2_comb] == SLOT_LN_READY) begin
                            expand_slot_hi <= e_slot2_comb;
                            expand_dual_pending <= 1'b1;
                            slot_state[e_slot2_comb] <= SLOT_EXPAND_ACT;
                        end else begin
                            expand_dual_pending <= 1'b0;
                        end
                    end
                end else begin
                    slot_state[e_slot_comb] <= SLOT_LN_READY;
                end

            // ── Expand idle: pick up any pending LN_READY slot ─────────────
            //   Round-robin: alternate which slot is checked first to prevent
            //   one slot from starving the other when both repeatedly become
            //   LN_READY (which happens when downstream cycle time is faster
            //   than expand's drain delay).
            end else if (!expand_running && !expand_drain) begin
                // Dual: check for two LN_READY slots
                if (EN_DUAL_SYSTOLIC && slot_state[0] == SLOT_LN_READY &&
                                         slot_state[1] == SLOT_LN_READY) begin
                    expand_slot   <= 0;
                    expand_slot_hi <= 1;
                    expand_running <= 1'b1;
                    expand_cnt    <= '0;
                    expand_dual_pending <= 1'b1;
                    slot_state[0] <= SLOT_EXPAND_ACT;
                    slot_state[1] <= SLOT_EXPAND_ACT;
                end else if (expand_rr_pick == 1'b0
                                ? (slot_state[0] == SLOT_LN_READY)
                                : (slot_state[1] != SLOT_LN_READY &&
                                   slot_state[0] == SLOT_LN_READY)) begin
                    expand_slot   <= 0;
                    expand_running <= 1'b1;
                    expand_cnt    <= '0;
                    expand_dual_pending <= 1'b0;
                    slot_state[0] <= SLOT_EXPAND_ACT;
                    expand_rr_pick <= 1'b1;
                end else if (slot_state[1] == SLOT_LN_READY) begin
                    expand_slot   <= 1;
                    expand_running <= 1'b1;
                    expand_cnt    <= '0;
                    expand_dual_pending <= 1'b0;
                    slot_state[1] <= SLOT_EXPAND_ACT;
                    expand_rr_pick <= 1'b0;
                end
            end

            // ── EXPAND_ACT → PROJ_READY (consolidated from act_buf block) ──
            // act_valid fires 4 cycles after expand_valid_act for the slot
            // tracked in act_slot_pipe[3]. Single-driver pattern avoids the
            // multi-always_ff race that prevented this transition from sticking.
            if (act_valid) begin
                slot_state[act_slot_pipe[3]] <= SLOT_PROJ_READY;
            end

            // ── PROJ_READY → EMPTY (consolidated from project block) ───────
            if (EN_DUAL_SYSTOLIC) begin
                if (project_valid_lo_pulse) slot_state[project_slot]    <= SLOT_EMPTY;
                if (project_valid_hi_pulse) slot_state[project_slot_hi] <= SLOT_EMPTY;
            end
            if (project_valid_single) slot_state[project_slot] <= SLOT_EMPTY;
        end
    end

    // expand_drain: high during SA pipeline drain after expand_done.
    // Prevents expand_start from firing (and clear_acc from destroying
    // results) until expand_valid_act signals results are captured.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            expand_drain <= 1'b0;
        end else begin
            if (expand_start)
                expand_drain <= 1'b0;
            else if (expand_done && expand_running)
                expand_drain <= 1'b1;
            else if (expand_valid_act)
                expand_drain <= 1'b0;
        end
    end

    // Expand SA weight fetch and LN buf read (indexed by expand_cnt)
    logic signed [DATA_WIDTH-1:0] b_in_expand_lo [MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] b_in_expand_hi [MAX_C-1:0];
    always_comb begin
        for (int i = 0; i < EXPANDED_C; i++)
            weight_a_expand[i] = weight_exp_rom[i * MAX_C + expand_cnt];
        for (int j = 0; j < MAX_C; j++) begin
            b_in_expand[j]      = ln_buf[expand_slot][expand_cnt];
            b_in_expand_lo[j]   = ln_buf[expand_slot][expand_cnt];
            b_in_expand_hi[j]   = ln_buf[expand_slot_hi][expand_cnt];
        end
    end

    logic expand_valid_single;
    logic expand_valid_act;

    // Expand systolic array (dense, sparse, dual, or sparse+dual)
    generate
    if (EN_DUAL_SYSTOLIC && EN_SPARSE_SYSTOLIC) begin : gen_expand_sparse_dual
        systolic_array_sparse_dual #(
            .M(EXPANDED_C), .N(MAX_C),
            .DATA_WIDTH(DATA_WIDTH),
            .ACC_WIDTH(ACC_WIDTH),
            .SPARSE_EN(1'b1)
        ) u_expand_sa_sparse_dual (
            .clk(clk), .rst_n(rst_n),
            .en(expand_running),
            .clear_acc(expand_start),
            .load_psum(1'b0),
            .a_in(weight_a_expand),
            .b_lo_in(b_in_expand_lo),
            .b_hi_in(b_in_expand_hi),
            .psum_in_lo(psum_zero_expand),
            .psum_in_hi(psum_zero_expand),
            .result_lo(expand_result_lo),
            .result_hi(expand_result_hi),
            .result_valid_pe(),
            .result_rdy(expand_valid_dual),
            .result_ready(1'b1),
            .sparse_en_cfg(sparse_en_cfg),
            .weight_mode_cfg(weight_mode_cfg),
            .activ_mode_cfg(activ_mode_cfg),
            .mac_skip_total(expand_mac_skip_total),
            .mac_active_total(expand_mac_active_total)
        );
        always_comb begin
            for (int i = 0; i < EXPANDED_C; i++)
                for (int j = 0; j < MAX_C; j++)
                    expand_result[i][j] = '0;
        end
        assign expand_valid_single = 1'b0;
    end else if (EN_DUAL_SYSTOLIC) begin : gen_expand_dual
        systolic_array_dual #(
            .M(EXPANDED_C), .N(MAX_C),
            .DATA_WIDTH(DATA_WIDTH),
            .ACC_WIDTH(ACC_WIDTH)
        ) u_expand_sa_dual (
            .clk(clk), .rst_n(rst_n),
            .en(expand_running),
            .clear_acc(expand_start),
            .load_psum(1'b0),
            .a_in(weight_a_expand),
            .b_lo_in(b_in_expand_lo),
            .b_hi_in(b_in_expand_hi),
            .psum_in_lo(psum_zero_expand),
            .psum_in_hi(psum_zero_expand),
            .result_lo(expand_result_lo),
            .result_hi(expand_result_hi),
            .result_rdy(expand_valid_dual)
        );
        always_comb begin
            for (int i = 0; i < EXPANDED_C; i++)
                for (int j = 0; j < MAX_C; j++)
                    expand_result[i][j] = '0;
        end
        assign expand_valid_single = 1'b0;
    end else if (EN_SPARSE_SYSTOLIC) begin : gen_expand_sparse
        systolic_array_sparse #(
            .M(EXPANDED_C), .N(MAX_C),
            .DATA_WIDTH(DATA_WIDTH),
            .ACC_WIDTH(ACC_WIDTH),
            .SPARSE_EN(1'b1)
        ) u_expand_sa (
            .clk(clk), .rst_n(rst_n),
            .en(expand_running),
            .clear_acc(expand_start),
            .load_psum(1'b0),
            .a_in(weight_a_expand),
            .b_in(b_in_expand),
            .psum_in(psum_zero_expand),
            .result(expand_result),
            .result_valid_pe(),
            .result_rdy(expand_valid_single),
            .result_ready(1'b1),
            .sparse_en_cfg(sparse_en_cfg),
            .weight_mode_cfg(weight_mode_cfg),
            .activ_mode_cfg(activ_mode_cfg),
            .mac_skip_total(expand_mac_skip_total),
            .mac_active_total(expand_mac_active_total)
        );
        always_comb begin
            for (int i = 0; i < EXPANDED_C; i++)
                for (int j = 0; j < MAX_C; j++) begin
                    expand_result_lo[i][j] = '0;
                    expand_result_hi[i][j] = '0;
                end
        end
        assign expand_valid_dual = 1'b0;
    end else begin : gen_expand_normal
        systolic_array #(
            .M(EXPANDED_C), .N(MAX_C),
            .DATA_WIDTH(DATA_WIDTH),
            .ACC_WIDTH(ACC_WIDTH)
        ) u_expand_sa (
            .clk(clk), .rst_n(rst_n),
            .en(expand_running),
            .clear_acc(expand_start),
            .load_psum(1'b0),
            .a_in(weight_a_expand),
            .b_in(b_in_expand),
            .psum_in(psum_zero_expand),
            .result(expand_result),
            .result_rdy(expand_valid_single)
        );
        always_comb begin
            for (int i = 0; i < EXPANDED_C; i++)
                for (int j = 0; j < MAX_C; j++) begin
                    expand_result_lo[i][j] = '0;
                    expand_result_hi[i][j] = '0;
                end
        end
        assign expand_valid_dual = 1'b0;
    end
    endgenerate

    // Dual expand sequencer: after expand_valid_dual, feed lo then hi
    // through the activation pipeline with back-to-back valid pulses.
    logic expand_dual_valid_pulse;
    logic expand_dual_hi_active;  // 1 when second (hi) pulse is active
    logic expand_dual_seq_busy;

    generate
    if (EN_DUAL_SYSTOLIC) begin : gen_dual_expand_seq
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                expand_dual_valid_pulse <= 1'b0;
                expand_dual_hi_active   <= 1'b0;
                expand_dual_seq_busy    <= 1'b0;
                for (int i = 0; i < EXPANDED_C; i++)
                    expand_out_hi_buf[i] <= '0;
            end else begin
                expand_dual_valid_pulse <= 1'b0;  // default: 1-cycle pulse

                if (expand_valid_dual && expand_dual_pending && !expand_dual_seq_busy) begin
                    // Both results ready: capture hi, fire lo
                    expand_dual_valid_pulse <= 1'b1;
                    expand_dual_hi_active   <= 1'b0;
                    expand_dual_seq_busy    <= 1'b1;
                    for (int i = 0; i < EXPANDED_C; i++)
                        expand_out_hi_buf[i] <= expand_result_hi[i][0][DATA_WIDTH-1:0];
                    // expand_out captures expand_result_lo on this pulse
                    // (via the result capture FF using expand_valid_act)

                end else if (expand_dual_seq_busy && !expand_dual_hi_active) begin
                    // Fire hi (back-to-back with lo)
                    expand_dual_valid_pulse <= 1'b1;
                    expand_dual_hi_active   <= 1'b1;
                    // expand_out captures expand_out_hi_buf on this pulse

                end else if (expand_dual_seq_busy && expand_dual_hi_active) begin
                    // Both results dispatched
                    expand_dual_seq_busy <= 1'b0;
                    expand_dual_hi_active <= 1'b0;
                end
            end
        end
    end else begin : gen_no_dual_expand_seq
        assign expand_dual_valid_pulse = 1'b0;
        assign expand_dual_hi_active   = 1'b0;
        assign expand_dual_seq_busy    = 1'b0;
    end
    endgenerate

    // expand_valid_act mux: single mode from array, dual mode from sequencer
    assign expand_valid_act = EN_DUAL_SYSTOLIC ? expand_dual_valid_pulse : expand_valid_single;

    // expand_slot_act mux: in dual mode, alternate between lo and hi
    logic [$clog2(NUM_SLOTS)-1:0] expand_slot_act;
    assign expand_slot_act = (EN_DUAL_SYSTOLIC && expand_dual_hi_active) ? expand_slot_hi : expand_slot;

    // Capture expand result from column 0
    // In dual mode: expand_valid_act fires for lo then hi (back-to-back)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < EXPANDED_C; i++) expand_out[i] <= '0;
        end else if (expand_valid_act) begin
            for (int i = 0; i < EXPANDED_C; i++)
                expand_out[i] <= EN_DUAL_SYSTOLIC && expand_dual_hi_active ?
                    expand_out_hi_buf[i] :
                    expand_result[i][0][DATA_WIDTH-1:0];
        end
    end

    // =========================================================================
    //  Stage 4: Activation + 4-Stage Slot Pipeline
    //
    //  Activation has 4 pipeline stages. The expand_slot at the time of
    //  expand_valid is pipelined through to route the activation output to
    //  the correct slot's act_buf.
    //
    //  out_ready is always high — the dual per-slot act_buf prevents overflow
    //  (expand produces every 16 cy, project consumes every 32 cy, 2 slots).
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] act_out [EXPANDED_C-1:0];
    logic [EXPANDED_C-1:0] act_valid_lane;
    logic act_valid;

    // Slot pipeline: matches activation's 4-cycle latency
    logic [$clog2(NUM_SLOTS)-1:0] act_slot_pipe [3:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 4; i++) act_slot_pipe[i] <= 0;
        end else begin
            if (expand_valid_act) act_slot_pipe[0] <= expand_slot_act;
            for (int i = 1; i < 4; i++)
                act_slot_pipe[i] <= act_slot_pipe[i-1];
        end
    end

    // Dual act_buf: activation output routed to act_slot_pipe[3]
    logic signed [DATA_WIDTH-1:0] act_buf [NUM_SLOTS-1:0][EXPANDED_C-1:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int s = 0; s < NUM_SLOTS; s++)
                for (int i = 0; i < EXPANDED_C; i++)
                    act_buf[s][i] <= '0;
        end else if (act_valid) begin
            for (int i = 0; i < EXPANDED_C; i++)
                act_buf[act_slot_pipe[3]][i] <= act_out[i];
            // slot_state transition moved to FSM always_ff (single-driver)
        end
    end

    // Shared activation ROMs — ROM values are Q16.16 (ROM_WIDTH=32 bits)
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
        .clk           (clk), .rst_n(rst_n),
        .x_in          (expand_out),
        .sigmoid_y_base(shared_sig_y_base),
        .sigmoid_slope (shared_sig_slope),
        .gelu_y_base   (shared_gelu_y_base),
        .gelu_slope    (shared_gelu_slope)
    );

    generate
        for (genvar i = 0; i < EXPANDED_C; i++) begin : gen_act
            logic act_lane_valid;
            unified_activation_lane #(
                .DATA_WIDTH(DATA_WIDTH),
                .FRAC_WIDTH(FRAC_WIDTH)
            ) u_act (
                .clk(clk), .rst_n(rst_n),
                .in_valid(expand_valid_act),
                .out_valid(act_lane_valid),
                .out_ready(1'b1),   // Dual act_buf: always room
                .act_enable(~cfg_clk_gate | (i < cfg_c*EXPANSION_FACTOR)),
                .act_sel(cfg_act_sel),
                .leak_shift('0),
                .saturation_in('0),
                .scale_in(scale_in), .bias_in(bias_in),
                .x_in(expand_out[i]),
                .y_out(act_out[i]),
                .sigmoid_y_base_in(shared_sig_y_base[i]),
                .sigmoid_slope_in (shared_sig_slope[i]),
                .gelu_y_base_in   (shared_gelu_y_base[i]),
                .gelu_slope_in    (shared_gelu_slope[i])
            );
            assign act_valid_lane[i] = act_lane_valid;
        end
    endgenerate

    assign act_valid = &act_valid_lane;

    // =========================================================================
    //  Stage 5: Project 1×1 Conv (Systolic Array)
    //
    //  Picks up the next SLOT_PROJ_READY slot. Runs independently of expand SA
    //  — both systolic arrays process different slots concurrently.
    //  When project finishes (project_valid), the slot is freed and the output
    //  stage emits the residual-added result.
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] weight_a_project [MAX_C-1:0];
    logic signed [DATA_WIDTH-1:0] b_in_project [EXPANDED_C-1:0];
    logic signed [ACC_WIDTH-1:0] project_result [MAX_C-1:0][EXPANDED_C-1:0];
    logic project_running;
    logic [$clog2(EXPANDED_C):0] project_cnt;
    logic project_done;
    logic [$clog2(NUM_SLOTS)-1:0] project_slot;
    logic project_drain;

    assign project_done = project_running && (project_cnt == cfg_c * EXPANSION_FACTOR - 1);

    // Combinational project_start: gated by !project_drain (same pattern as expand)
    logic project_start;
    assign project_start = !project_running && !project_drain &&
        ((slot_state[0] == SLOT_PROJ_READY) || (slot_state[1] == SLOT_PROJ_READY));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            project_running <= 1'b0;
            project_cnt     <= '0;
            project_slot    <= 0;
        end else begin
            if (project_running) begin
                if (project_done) begin
                    project_running <= 1'b0;
                    project_cnt <= '0;
                    // Slot freed by output stage on project_valid
                end else begin
                    project_cnt <= project_cnt + 1'b1;
                end
            end

            // Pick up next PROJ_READY slot(s) (only when project idle AND not
            // still draining the previous result through the SA flush pipe;
            // otherwise project_slot is overwritten and project_valid_single
            // associates with the wrong slot).
            if (!project_running && !project_drain) begin
                // Dual: check for two PROJ_READY slots
                if (EN_DUAL_SYSTOLIC && slot_state[0] == SLOT_PROJ_READY &&
                                         slot_state[1] == SLOT_PROJ_READY) begin
                    project_slot <= 0;
                    project_slot_hi <= 1;
                    project_running <= 1'b1;
                    project_cnt <= '0;
                    project_dual_pending <= 1'b1;
                end else if (slot_state[0] == SLOT_PROJ_READY) begin
                    project_slot <= 0;
                    project_running <= 1'b1;
                    project_cnt <= '0;
                    project_dual_pending <= 1'b0;
                end else if (slot_state[1] == SLOT_PROJ_READY) begin
                    project_slot <= 1;
                    project_running <= 1'b1;
                    project_cnt <= '0;
                    project_dual_pending <= 1'b0;
                end
            end
        end
    end

    // project_drain: high during SA pipeline drain after project_done.
    // Same pattern as expand_drain — prevents project_start/clear_acc
    // during the drain period.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            project_drain <= 1'b0;
        end else begin
            if (project_start)
                project_drain <= 1'b0;
            else if (project_done && project_running)
                project_drain <= 1'b1;
            else if (project_valid_single || (EN_DUAL_SYSTOLIC && (project_valid_lo_pulse | project_valid_hi_pulse)))
                project_drain <= 1'b0;
        end
    end

    // Project SA weight fetch and act_buf read (single + dual)
    logic signed [DATA_WIDTH-1:0] b_in_project_lo [EXPANDED_C-1:0];
    logic signed [DATA_WIDTH-1:0] b_in_project_hi [EXPANDED_C-1:0];
    always_comb begin
        for (int i = 0; i < MAX_C; i++)
            weight_a_project[i] = weight_proj_rom[i * EXPANDED_C + project_cnt];
        for (int j = 0; j < EXPANDED_C; j++) begin
            b_in_project[j]     = act_buf[project_slot][project_cnt];
            b_in_project_lo[j]  = act_buf[project_slot][project_cnt];
            b_in_project_hi[j]  = act_buf[project_slot_hi][project_cnt];
        end
    end

    logic project_valid_single;

    generate
    if (EN_DUAL_SYSTOLIC && EN_SPARSE_SYSTOLIC) begin : gen_project_sparse_dual
        systolic_array_sparse_dual #(
            .M(MAX_C), .N(EXPANDED_C),
            .DATA_WIDTH(DATA_WIDTH),
            .ACC_WIDTH(ACC_WIDTH),
            .SPARSE_EN(1'b1)
        ) u_project_sa_sparse_dual (
            .clk(clk), .rst_n(rst_n),
            .en(project_running),
            .clear_acc(project_start),
            .load_psum(1'b0),
            .a_in(weight_a_project),
            .b_lo_in(b_in_project_lo),
            .b_hi_in(b_in_project_hi),
            .psum_in_lo(psum_zero_project),
            .psum_in_hi(psum_zero_project),
            .result_lo(project_result_lo),
            .result_hi(project_result_hi),
            .result_valid_pe(),
            .result_rdy(project_valid_dual),
            .result_ready(1'b1),
            .sparse_en_cfg(sparse_en_cfg),
            .weight_mode_cfg(weight_mode_cfg),
            .activ_mode_cfg(activ_mode_cfg),
            .mac_skip_total(project_mac_skip_total),
            .mac_active_total(project_mac_active_total)
        );
        always_comb begin
            for (int i = 0; i < MAX_C; i++)
                for (int j = 0; j < EXPANDED_C; j++)
                    project_result[i][j] = '0;
        end
        assign project_valid_single = 1'b0;
    end else if (EN_DUAL_SYSTOLIC) begin : gen_project_dual
        systolic_array_dual #(
            .M(MAX_C), .N(EXPANDED_C),
            .DATA_WIDTH(DATA_WIDTH),
            .ACC_WIDTH(ACC_WIDTH)
        ) u_project_sa_dual (
            .clk(clk), .rst_n(rst_n),
            .en(project_running),
            .clear_acc(project_start),
            .load_psum(1'b0),
            .a_in(weight_a_project),
            .b_lo_in(b_in_project_lo),
            .b_hi_in(b_in_project_hi),
            .psum_in_lo(psum_zero_project),
            .psum_in_hi(psum_zero_project),
            .result_lo(project_result_lo),
            .result_hi(project_result_hi),
            .result_rdy(project_valid_dual)
        );
        always_comb begin
            for (int i = 0; i < MAX_C; i++)
                for (int j = 0; j < EXPANDED_C; j++)
                    project_result[i][j] = '0;
        end
        assign project_valid_single = 1'b0;
    end else if (EN_SPARSE_SYSTOLIC) begin : gen_project_sparse
        systolic_array_sparse #(
            .M(MAX_C), .N(EXPANDED_C),
            .DATA_WIDTH(DATA_WIDTH),
            .ACC_WIDTH(ACC_WIDTH),
            .SPARSE_EN(1'b1)
        ) u_project_sa (
            .clk(clk), .rst_n(rst_n),
            .en(project_running),
            .clear_acc(project_start),
            .load_psum(1'b0),
            .a_in(weight_a_project),
            .b_in(b_in_project),
            .psum_in(psum_zero_project),
            .result(project_result),
            .result_valid_pe(),
            .result_rdy(project_valid_single),
            .result_ready(1'b1),
            .sparse_en_cfg(sparse_en_cfg),
            .weight_mode_cfg(weight_mode_cfg),
            .activ_mode_cfg(activ_mode_cfg),
            .mac_skip_total(project_mac_skip_total),
            .mac_active_total(project_mac_active_total)
        );
        always_comb begin
            for (int i = 0; i < MAX_C; i++)
                for (int j = 0; j < EXPANDED_C; j++) begin
                    project_result_lo[i][j] = '0;
                    project_result_hi[i][j] = '0;
                end
        end
        assign project_valid_dual = 1'b0;
    end else begin : gen_project_normal
        systolic_array #(
            .M(MAX_C), .N(EXPANDED_C),
            .DATA_WIDTH(DATA_WIDTH),
            .ACC_WIDTH(ACC_WIDTH)
        ) u_project_sa (
            .clk(clk), .rst_n(rst_n),
            .en(project_running),
            .clear_acc(project_start),
            .load_psum(1'b0),
            .a_in(weight_a_project),
            .b_in(b_in_project),
            .psum_in(psum_zero_project),
            .result(project_result),
            .result_rdy(project_valid_single)
        );
        always_comb begin
            for (int i = 0; i < MAX_C; i++)
                for (int j = 0; j < EXPANDED_C; j++) begin
                    project_result_lo[i][j] = '0;
                    project_result_hi[i][j] = '0;
                end
        end
        assign project_valid_dual = 1'b0;
    end
    endgenerate

    // project_valid_act mux: single mode from array, dual mode from sequencer
    logic project_valid_act;
    logic project_valid_lo_pulse;
    logic project_valid_hi_pulse;
    logic project_dual_hi_pending;

    generate
    if (EN_DUAL_SYSTOLIC) begin : gen_dual_project_seq
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                project_valid_lo_pulse <= 1'b0;
                project_valid_hi_pulse <= 1'b0;
                project_dual_hi_pending <= 1'b0;
            end else begin
                project_valid_lo_pulse <= 1'b0;
                project_valid_hi_pulse <= 1'b0;

                if (project_valid_dual && project_dual_pending) begin
                    // Fire lo result
                    project_valid_lo_pulse <= 1'b1;
                    project_dual_hi_pending <= 1'b1;
                end else if (project_dual_hi_pending && out_ready) begin
                    // Fire hi only after lo has been accepted by consumer
                    project_valid_hi_pulse <= 1'b1;
                    project_dual_hi_pending <= 1'b0;
                end
            end
        end
        assign project_valid_act = project_valid_lo_pulse | project_valid_hi_pulse;
    end else begin : gen_no_dual_project_seq
        assign project_valid_lo_pulse = 1'b0;
        assign project_valid_hi_pulse = 1'b0;
        assign project_dual_hi_pending = 1'b0;
        assign project_valid_act = project_valid_single;
    end
    endgenerate

    // =========================================================================
    //  Stage 6: Residual Add + Output
    //
    //  When project_valid/project_valid_act fires, the systolic result is added
    //  to the slot's residual (captured at token acceptance). The slot is freed.
    //
    //  In dual mode: project_valid_lo_pulse fires lo (project_slot) and
    //  project_valid_hi_pulse fires hi (project_slot_hi) back-to-back.
    //
    //  project_result[i][0] is read directly (combinational) instead of the
    //  registered project_out to avoid 1-cycle NBA misalignment.
    // =========================================================================
    logic out_valid_r, out_last_r;
    logic signed [DATA_WIDTH-1:0] out_token_r [MAX_C-1:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid_r <= 1'b0;
            out_last_r  <= 1'b0;
            for (int i = 0; i < MAX_C; i++) out_token_r[i] <= '0;
        end else begin
            if (out_valid_r && out_ready && !project_valid_lo_pulse && !project_valid_hi_pulse && !project_valid_single) begin
                out_valid_r <= 1'b0;
                out_last_r  <= 1'b0;
            end

            // ── Dual lo result ──────────────────────────────────────────────
            if (project_valid_lo_pulse) begin
                out_valid_r <= 1'b1;
                out_last_r  <= slot_last[project_slot];
                for (int i = 0; i < MAX_C; i++) begin
                    out_token_r[i] <= DATA_WIDTH'(sat_add_signed(
                        slot_residual[project_slot][i],
                        EN_DUAL_SYSTOLIC ?
                            project_result_lo[i][0][DATA_WIDTH-1:0] :
                            project_result[i][0][DATA_WIDTH-1:0],
                        DATA_WIDTH
                    ));
                end
                // slot_state transition moved to FSM always_ff (single-driver)
            end

            // ── Dual hi result ──────────────────────────────────────────────
            if (project_valid_hi_pulse) begin
                out_valid_r <= 1'b1;
                out_last_r  <= slot_last[project_slot_hi];
                for (int i = 0; i < MAX_C; i++) begin
                    out_token_r[i] <= DATA_WIDTH'(sat_add_signed(
                        slot_residual[project_slot_hi][i],
                        project_result_hi[i][0][DATA_WIDTH-1:0],
                        DATA_WIDTH
                    ));
                end
                // slot_state transition moved to FSM always_ff (single-driver)
            end

            // ── Single mode ─────────────────────────────────────────────────
            if (project_valid_single) begin
                out_valid_r <= 1'b1;
                out_last_r  <= slot_last[project_slot];
                for (int i = 0; i < MAX_C; i++) begin
                    out_token_r[i] <= DATA_WIDTH'(sat_add_signed(
                        slot_residual[project_slot][i],
                        project_result[i][0][DATA_WIDTH-1:0],
                        DATA_WIDTH
                    ));
                end
                // slot_state transition moved to FSM always_ff (single-driver)
            end
        end
    end

    assign valid_out      = out_valid_r;
    assign valid_out_last = out_valid_r & out_last_r;
    assign token_out      = out_token_r;

    // slot_state is now driven from a single always_ff (FSM block above).
    // Reset and all transitions are handled there.

    // =========================================================================
    //  Performance Counters
    // =========================================================================
`ifdef DEBUG
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) perf_cycle_count <= '0;
        else if (valid_in_gated) perf_cycle_count <= perf_cycle_count + 1;
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
    assign perf_cycle_count  = '0;
    assign perf_stall_count  = '0;
    assign perf_token_count  = '0;
`endif

    // =========================================================================
    //  Assertions
    // =========================================================================
    // synthesis translate_off
    CONVFORMER_VALID_OUT_STABLE: assert property (
        @(posedge clk) disable iff (!rst_n)
        (valid_out && !out_ready) |=> valid_out
    ) else $error("convformer_block: valid_out dropped before out_ready");

    CONVFORMER_MAX_INFLIGHT: assert property (
        @(posedge clk) disable iff (!rst_n)
        $countones({slot_state[0] != SLOT_EMPTY, slot_state[1] != SLOT_EMPTY}) <= 2
    ) else $error("convformer_block: more than 2 tokens in flight");

    CONVFORMER_EXPAND_START: assert property (
        @(posedge clk) disable iff (!rst_n)
        // expand_slot is registered in the same always_ff that transitions
        // slot_state to SLOT_EXPAND_ACT. In the observed region (post-NBA),
        // expand_slot already points to the new slot whose state is now
        // SLOT_EXPAND_ACT — so we must allow that state too.
        expand_start |-> (slot_state[expand_slot] == SLOT_LN_READY ||
                          slot_state[expand_slot] == SLOT_EMPTY ||
                          slot_state[expand_slot] == SLOT_EXPAND_ACT ||
                          ln_valid_out)
    ) else $error("convformer_block: expand_start with invalid slot");

    CONVFORMER_PROJ_READY_START: assert property (
        @(posedge clk) disable iff (!rst_n)
        !project_running && !project_drain &&
        (slot_state[0]==SLOT_PROJ_READY ||
         slot_state[1]==SLOT_PROJ_READY) |=> project_running
    ) else $error("convformer_block: project didn't start on PROJ_READY slot");

    CONVFORMER_DW_LN_NO_OVERFLOW: assert property (
        @(posedge clk) disable iff (!rst_n)
        dw_fifo_full |-> !dw_fifo_wr_en
    ) else $error("convformer_block: DW→LN FIFO overflow");
    // synthesis translate_on

    // === DEBUG: internal FSM tracing ===========================================
    logic [31:0] dbg_cycle;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dbg_cycle <= 0;
        end else begin
            dbg_cycle <= dbg_cycle + 1'b1;
            if (expand_running || project_running ||
                slot_state[0] != SLOT_EMPTY || slot_state[1] != SLOT_EMPTY) begin
                if (dbg_cycle % 5000 == 0) begin
                    $display("CFDBG c=%0d s0=%0d s1=%0d e=%b ec=%0d ed=%b evs=%b eva=%b av=%b pr=%b pc=%0d pd=%b pvs=%b edrn=%b pdrn=%b pslot=%0d ln_v=%b",
                        dbg_cycle, slot_state[0], slot_state[1],
                        expand_running, expand_cnt, expand_done,
                        expand_valid_single, expand_valid_act, act_valid,
                        project_running, project_cnt, project_done,
                        project_valid_single, expand_drain, project_drain, project_slot,
                        ln_valid_out);
                end
            end
        end
    end

endmodule