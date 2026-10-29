// =============================================================================
//  systolic_ping_pong_fifo_top_dual.sv
//
//  Dual-MAC variant of systolic_ping_pong_fifo_top.
//  Wraps systolic_array_dual with FIFO + ping-pong BRAM double-buffer.
//
//  Two weight streams (b_lo_in, b_hi_in) produce two independent result
//  matrices (rd_result_lo, rd_result_hi). The FIFO and BRAM array carry
//  both results packed together, and they are unpacked at the output.
//
//  Same round-robin controller topology as the single-MAC version.
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;

module systolic_ping_pong_fifo_top_dual #(
    parameter int M          = 4,
    parameter int N          = 4,
    parameter int DATA_WIDTH = pkg_module_config::SYSTOLIC_ARRAY_DATA_WIDTH,
    parameter int ACC_WIDTH  = pkg_config::ACC_WIDTH,
    parameter int ADDR_WIDTH = 4,
    parameter int DEPTH      = 16,
    parameter bit  LOAD_PSUM_EN = 1'b0   // Enable partial-sum load path in systolic array
)(
    input  logic clk,
    input  logic rst_n,

    input  logic en,
    input  logic clear_acc,
    input  logic load_psum,
    input  logic signed [DATA_WIDTH-1:0] a_in [M-1:0],
    input  logic signed [DATA_WIDTH-1:0] b_lo_in [N-1:0],
    input  logic signed [DATA_WIDTH-1:0] b_hi_in [N-1:0],
    input  logic signed [ACC_WIDTH-1:0]  psum_in_lo [M-1:0][N-1:0],
    input  logic signed [ACC_WIDTH-1:0]  psum_in_hi [M-1:0][N-1:0],

    input  logic c_start,
    input  logic c_done,
    input  logic [ADDR_WIDTH-1:0] rd_idx,

    output logic c_data_valid,
    output logic signed [ACC_WIDTH-1:0] rd_result_lo [M-1:0][N-1:0],
    output logic signed [ACC_WIDTH-1:0] rd_result_hi [M-1:0][N-1:0],

    output logic [1:0] bank_ready,
    output logic owner_error,
    output logic result_rdy
);

    // =========================================================================
    //  SYSTOLIC ARRAY (dual-Mac variant)
    // =========================================================================
    logic result_rdy_internal;
    logic signed [ACC_WIDTH-1:0] result_lo [M-1:0][N-1:0];
    logic signed [ACC_WIDTH-1:0] result_hi [M-1:0][N-1:0];

    systolic_array_dual #(
        .M(M), .N(N),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .LOAD_PSUM_EN(LOAD_PSUM_EN)
    ) u_sa_dual (
        .clk(clk), .rst_n(rst_n),
        .en(en), .clear_acc(clear_acc), .load_psum(load_psum),
        .a_in(a_in),
        .b_lo_in(b_lo_in),
        .b_hi_in(b_hi_in),
        .psum_in_lo(psum_in_lo),
        .psum_in_hi(psum_in_hi),
        .result_lo(result_lo),
        .result_hi(result_hi),
        .result_rdy(result_rdy_internal)
    );

    assign result_rdy = result_rdy_internal;

    // =========================================================================
    //  PACK: flatten M×N results into one FIFO word
    //
    //  Each FIFO word carries both lo and hi results packed together.
    //  ROW_WIDTH = N × ACC_WIDTH × 2 (lo + hi per element)
    // =========================================================================
    localparam int ROW_WIDTH   = N * ACC_WIDTH * 2;  // lo + hi per element
    localparam int TOTAL_WIDTH = M * ROW_WIDTH;

    logic [ROW_WIDTH-1:0] row_packed [M-1:0];

    always_comb
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) begin
                row_packed[i][j*ACC_WIDTH*2 +: ACC_WIDTH]       = result_lo[i][j];
                row_packed[i][j*ACC_WIDTH*2 + ACC_WIDTH +: ACC_WIDTH] = result_hi[i][j];
            end

    // =========================================================================
    //  FIFO (double width to carry both lo and hi results)
    // =========================================================================
    logic [TOTAL_WIDTH-1:0] fifo_din, fifo_dout;
    logic fifo_wr, fifo_rd, fifo_full, fifo_empty;

    always_comb
        for (int i = 0; i < M; i++)
            fifo_din[i*ROW_WIDTH +: ROW_WIDTH] = row_packed[i];

    assign fifo_wr = result_rdy_internal & ~fifo_full;

    fifo #(.WIDTH(TOTAL_WIDTH), .DEPTH(16)) u_fifo (
        .clk(clk), .rst_n(rst_n),
        .wr_en(fifo_wr), .din(fifo_din), .full(fifo_full),
        .rd_en(fifo_rd), .dout(fifo_dout), .empty(fifo_empty)
    );

    // =========================================================================
    //  ROUND-ROBIN write selector (same topology as single-MAC version)
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
    //  WRITE-FIRE pipeline (same as single-MAC version)
    // =========================================================================
    logic [1:0] p_start, p_done;
    logic [1:0] write_fire, write_fire_d;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) p_start <= 2'b0;
        else p_start <= {fifo_rd_reg & rr_sel, fifo_rd_reg & ~rr_sel};
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
    //  WRITE-ADDRESS COUNTER (per controller)
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
    //  CONTROLLERS (same ping_pong_buffer_control as single-MAC version)
    // =========================================================================
    logic [1:0] ctrl_c_data_valid;
    logic [1:0] sel_wr, sel_rd;
    logic [1:0] c_done_ctrl;
    logic [1:0] p_we_gate;

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
            .p_we_gate    (p_we_gate[k])
        );
    end
    endgenerate

    // =========================================================================
    //  FIFO UNPACK + skid register (same as single-MAC)
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
    //  BRAM ARRAY (each bank carries lo+hi packed per element)
    // =========================================================================
    logic [ROW_WIDTH-1:0] rd_row   [1:0][M-1:0];
    logic [1:0]           bank_rdy [1:0][M-1:0];
    logic                 own_err  [1:0][M-1:0];

    generate
    for (genvar k = 0; k < 2; k++) begin : gen_ctrl_bram
        for (genvar i = 0; i < M; i++) begin : gen_row_bram
            ping_pong_bram #(
                .ADDR_W(ADDR_WIDTH),
                .DATA_W(ROW_WIDTH)
            ) u_bram (
                .clk               (clk),
                .rst_n             (rst_n),
                .p_addr             (wr_idx[k]),
                .p_data_in          (fifo_row[i]),
                .p_we               (write_fire_d[k]),
                .p_done             (p_done[k]),
                .c_addr             (rd_idx),
                .c_data_out         (rd_row[k][i]),
                .c_done             (c_done_ctrl[k]),
                .ping_pong_select_wr(sel_wr[k]),
                .ping_pong_select_rd(sel_rd[k]),
                .bank_ready         (bank_rdy[k][i]),
                .owner_error        (own_err[k][i])
            );
        end
    end
    endgenerate

    // =========================================================================
    //  OUTPUT: unpack lo and hi results from packed BRAM data
    // =========================================================================
    // Combinational row data selection (extracted from always_ff for Verilator compat)
    logic [ROW_WIDTH-1:0] row_data_comb [0:M-1];
    always_comb begin
        for (int i = 0; i < M; i++) begin
            row_data_comb[i] = ctrl_c_data_valid[0] ? rd_row[0][i] : rd_row[1][i];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) begin
                    rd_result_lo[i][j] <= '0;
                    rd_result_hi[i][j] <= '0;
                end
        end else begin
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) begin
                    rd_result_lo[i][j] <= row_data_comb[i][j*ACC_WIDTH*2 +: ACC_WIDTH];
                    rd_result_hi[i][j] <= row_data_comb[i][j*ACC_WIDTH*2 + ACC_WIDTH +: ACC_WIDTH];
                end
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