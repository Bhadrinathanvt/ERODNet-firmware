// =============================================================================
//  systolic_ping_pong_fifo_top_sparse.sv
//
//  Drop-in replacement for systolic_ping_pong_fifo_top that uses the
//  sparsity-aware systolic array instead of the original.
//
//  Configuration is driven from the parent convformer_block-level sparse_csr
//  instance via direct input ports (sparse_en_cfg, weight_mode_cfg,
//  activ_mode_cfg). Performance counter outputs (mac_skip_total,
//  mac_active_total) are aggregated at the convformer_block level.
//
//  Dense-mode backward compatibility:
//    When sparse_en_cfg=0, the array operates identically to the original
//    systolic_array, producing bit-identical results with the same timing.
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;
import pkg_sparse_config::*;

module systolic_ping_pong_fifo_top_sparse #(
    parameter int M          = 4,
    parameter int N          = 4,
    parameter int DATA_WIDTH = pkg_module_config::SYSTOLIC_ARRAY_DATA_WIDTH,
    parameter int ACC_WIDTH  = pkg_config::ACC_WIDTH,
    parameter int ADDR_WIDTH = 4,
    parameter int DEPTH      = 16,
    parameter bit EN_SPARSE_SYSTOLIC = pkg_config::EN_SPARSE_SYSTOLIC,
    parameter bit LOAD_PSUM_EN       = 1'b0   // Enable partial-sum load path in systolic array
)(
    input  logic clk,
    input  logic rst_n,

    // ── Systolic array control ──────────────────────────────────────────────
    input  logic en,
    input  logic clear_acc,
    input  logic load_psum,
    input  logic signed [DATA_WIDTH-1:0] a_in [M-1:0],
    input  logic signed [DATA_WIDTH-1:0] b_in [N-1:0],
    input  logic signed [ACC_WIDTH-1:0]  psum_in [M-1:0][N-1:0],

    // ── Consumer interface ───────────────────────────────────────────────────
    input  logic c_start,
    input  logic c_done,
    input  logic [ADDR_WIDTH-1:0] rd_idx,

    output logic c_data_valid,
    output logic signed [ACC_WIDTH-1:0] rd_result [M-1:0][N-1:0],

    output logic [1:0] bank_ready,
    output logic owner_error,
    output logic result_rdy,

    // ── Sparse configuration (from convformer_block-level sparse_csr) ────────
    input  logic        sparse_en_cfg,      // Runtime enable/disable
    input  logic [1:0]  weight_mode_cfg,    // Weight encoding mode
    input  logic [1:0]  activ_mode_cfg,     // Activation encoding mode

    // ── Sparse performance counters ────────────────────────────────────────────
    output logic [31:0] mac_skip_total,
    output logic [31:0] mac_active_total
);

    // =========================================================================
    //  Sparse configuration — driven from convformer_block-level sparse_csr
    // =========================================================================
    // sparse_en_cfg, weight_mode_cfg, activ_mode_cfg are now input ports,
    // driven by the single sparse_csr instance at the convformer_block level.
    // When EN_SPARSE_SYSTOLIC=0, the parent ties them to zero.

    logic [31:0] sa_mac_skip_total;
    logic [31:0] sa_mac_active_total;

    generate
    if (!EN_SPARSE_SYSTOLIC) begin : gen_no_sparse_counters
        // When sparse is disabled at synthesis, counters are zero
        assign sa_mac_skip_total   = 32'd0;
        assign sa_mac_active_total = 32'd0;
    end
    endgenerate

    // =========================================================================
    //  SYSTOLIC ARRAY (SPARSE)
    // =========================================================================
    logic result_rdy_internal;
    logic signed [ACC_WIDTH-1:0] result [M-1:0][N-1:0];

    // Backpressure: when FIFO is full, stall the systolic array
    logic sa_en;
    logic fifo_full;

    assign sa_en = en & ~fifo_full;

    systolic_array_sparse #(
        .M(M), .N(N),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .SPARSE_EN(EN_SPARSE_SYSTOLIC),
        .LOAD_PSUM_EN(LOAD_PSUM_EN)
    ) u_sa_sparse (
        .clk              (clk),
        .rst_n            (rst_n),
        .en               (sa_en),
        .clear_acc        (clear_acc),
        .load_psum        (load_psum),
        .a_in             (a_in),
        .b_in             (b_in),
        .psum_in          (psum_in),
        .result           (result),
        .result_valid_pe  (),              // Not used in this wrapper
        .result_rdy       (result_rdy_internal),
        .sparse_en_cfg    (sparse_en_cfg),
        .weight_mode_cfg  (weight_mode_cfg),
        .activ_mode_cfg   (activ_mode_cfg),
        .mac_skip_total   (sa_mac_skip_total),
        .mac_active_total (sa_mac_active_total)
    );

    assign result_rdy = result_rdy_internal;
    assign mac_skip_total = sa_mac_skip_total;
    assign mac_active_total = sa_mac_active_total;

    // =========================================================================
    //  PACK: flatten M×N result into one FIFO word
    // =========================================================================
    localparam int ROW_WIDTH   = N * ACC_WIDTH;
    localparam int TOTAL_WIDTH = M * ROW_WIDTH;

    logic [ROW_WIDTH-1:0] row_packed [M-1:0];

    always_comb
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                row_packed[i][j*ACC_WIDTH +: ACC_WIDTH] = result[i][j];

    // =========================================================================
    //  FIFO
    // =========================================================================
    logic [TOTAL_WIDTH-1:0] fifo_din, fifo_dout;
    logic fifo_wr, fifo_rd, fifo_empty;

    always_comb
        for (int i = 0; i < M; i++)
            fifo_din[i*ROW_WIDTH +: ROW_WIDTH] = row_packed[i];

    assign fifo_wr = result_rdy_internal & ~fifo_full;

    fifo #(.WIDTH(TOTAL_WIDTH), .DEPTH(16)) u_fifo (
        .clk(clk), .rst_n(rst_n),
        .wr_en(fifo_wr), .din(fifo_din), .full(fifo_full), .almost_full(),
        .rd_en(fifo_rd), .dout(fifo_dout), .empty(fifo_empty), .almost_empty()
    );

    // =========================================================================
    //  ROUND-ROBIN write selector
    // =========================================================================
    logic rr_sel;
    logic [1:0] p_ready;

    assign fifo_rd = ~fifo_empty & (rr_sel ? p_ready[1] : p_ready[0]);

    logic fifo_rd_reg;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fifo_rd_reg <= 1'b0;
        else        fifo_rd_reg <= fifo_rd;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rr_sel <= 1'b0;
        else if (fifo_rd_reg) rr_sel <= ~rr_sel;
    end

    // =========================================================================
    //  WRITE-FIRE pipeline
    // =========================================================================
    logic [1:0] p_start, p_done;
    logic [1:0] write_fire, write_fire_d;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) p_start <= 2'b0;
        else        p_start <= {fifo_rd_reg & rr_sel, fifo_rd_reg & ~rr_sel};
    end
    assign write_fire[0] = p_start[0];
    assign write_fire[1] = p_start[1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) write_fire_d <= 2'b0;
        else        write_fire_d <= write_fire;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p_done[0] <= 1'b0;
            p_done[1] <= 1'b0;
        end else begin
            p_done[0] <= write_fire_d[0];
            p_done[1] <= write_fire_d[1];
        end
    end

    // =========================================================================
    //  WRITE-ADDRESS COUNTER
    // =========================================================================
    logic [ADDR_WIDTH-1:0] wr_idx [1:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_idx[0] <= '0;
            wr_idx[1] <= '0;
        end else begin
            if      (write_fire[0])   wr_idx[0] <= '0;
            else if (write_fire_d[0]) wr_idx[0] <= wr_idx[0] + 1'b1;
            if      (write_fire[1])   wr_idx[1] <= '0;
            else if (write_fire_d[1]) wr_idx[1] <= wr_idx[1] + 1'b1;
        end
    end

    // =========================================================================
    //  CONTROLLERS
    // =========================================================================
    logic [1:0] ctrl_c_data_valid;
    logic [1:0] sel_wr, sel_rd;
    logic [1:0] c_done_ctrl;

    assign c_done_ctrl[0] = c_done & ctrl_c_data_valid[0];
    assign c_done_ctrl[1] = c_done & ctrl_c_data_valid[1];

    generate
    for (genvar k = 0; k < 2; k++) begin : gen_ctrl
        ping_pong_buffer_control u_ctrl (
            .clk          (clk),
            .rst_n        (rst_n),
            .p_start      (p_start[k]),
            .p_ready      (p_ready[k]),
            .p_done       (p_done[k]),
            .c_start      (c_start),
            .c_data_valid (ctrl_c_data_valid[k]),
            .c_done       (c_done_ctrl[k]),
            .sel_wr       (sel_wr[k]),
            .sel_rd       (sel_rd[k]),
            .p_we_gate    ()                   // Unused in this wrapper
        );
    end
    endgenerate

    // =========================================================================
    //  FIFO UNPACK + SKID REGISTER
    // =========================================================================
    logic [TOTAL_WIDTH-1:0] fifo_dout_hold;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            fifo_dout_hold <= '0;
        else if (fifo_rd_reg)
            fifo_dout_hold <= fifo_dout;
    end

    logic [ROW_WIDTH-1:0] fifo_row [M-1:0];

    always_comb
        for (int i = 0; i < M; i++)
            fifo_row[i] = fifo_dout_hold[i*ROW_WIDTH +: ROW_WIDTH];

    // =========================================================================
    //  BRAM ARRAY
    // =========================================================================
    logic [ROW_WIDTH-1:0] rd_row   [1:0][M-1:0];
    logic [1:0]           bank_rdy [1:0][M-1:0];
    logic                 own_err  [1:0][M-1:0];

    // p_we_gate is used to gate writes; compute per-controller
    logic [1:0] p_we_gate;
    assign p_we_gate[0] = write_fire_d[0];
    assign p_we_gate[1] = write_fire_d[1];

    generate
    for (genvar k = 0; k < 2; k++) begin : gen_ctrl_bram
        for (genvar i = 0; i < M; i++) begin : gen_row_bram
            ping_pong_bram #(
                .ADDR_W(ADDR_WIDTH),
                .DATA_W(ROW_WIDTH)
            ) u_bram (
                .clk               (clk),
                .rst_n             (rst_n),
                .p_addr            (wr_idx[k]),
                .p_data_in         (fifo_row[i]),
                .p_we              (p_we_gate[k]),
                .p_done            (p_done[k]),
                .c_addr            (rd_idx),
                .c_data_out        (rd_row[k][i]),
                .c_done            (c_done_ctrl[k]),
                .ping_pong_select_wr(sel_wr[k]),
                .ping_pong_select_rd(sel_rd[k]),
                .bank_ready        (bank_rdy[k][i]),
                .owner_error       (own_err[k][i])
            );
        end
    end
    endgenerate

    // =========================================================================
    //  OUTPUT MUX
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    rd_result[i][j] <= '0;
        end else begin
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    rd_result[i][j] <= ctrl_c_data_valid[0]
                        ? rd_row[0][i][j*ACC_WIDTH +: ACC_WIDTH]
                        : rd_row[1][i][j*ACC_WIDTH +: ACC_WIDTH];
        end
    end

    assign c_data_valid = ctrl_c_data_valid[0] | ctrl_c_data_valid[1];
    assign bank_ready = bank_rdy[0][0] | bank_rdy[1][0];

    always_comb begin
        owner_error = 1'b0;
        for (int k = 0; k < 2; k++)
            for (int i = 0; i < M; i++)
                owner_error |= own_err[k][i];
    end

endmodule