// =============================================================================
//  systolic_ping_pong_fifo_top.sv
//
//  Systolic array → FIFO → dual ping-pong BRAM double-buffer.
//
//  Two independent ping-pong buffer controllers serve as a result pipeline:
//    ─ Alternating matrix-multiply results are routed to alternate controllers
//      via a round-robin selector on each FIFO pop.
//    ─ While the consumer reads result N from controller 0's BRAM banks,
//      controller 1 can be simultaneously filled with result N+1.
//    ─ Each controller manages a FULL copy of M BRAM rows (ROW_WIDTH bits
//      each) so it holds one complete M×N result matrix.
//
//  This gives true producer/consumer pipeline overlap across back-to-back
//  matrix multiplies, instead of the controllers firing in lockstep.
//
//  Key signals:
//    fifo_rd   — pops the FIFO; only fires when the RR-selected controller
//                is ready (p_ready).
//    rr_sel    — round-robin bit; toggles on every fifo_rd.
//    p_start[k]— fires only for the selected controller.
//    c_done_ctrl[k] — c_done routed only to the controller currently serving
//                     (ctrl_c_data_valid[k] high), so the other controller's
//                     bank-fresh flags are not corrupted.
//    rd_result — muxed from the controller whose c_data_valid is asserted
//                (ctrl 0 takes priority if both somehow assert together).
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;

module systolic_ping_pong_fifo_top #(
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
    input  logic signed [DATA_WIDTH-1:0] b_in [N-1:0],
    input  logic signed [ACC_WIDTH-1:0]  psum_in [M-1:0][N-1:0],

    input  logic c_start,
    input  logic c_done,
    input  logic [ADDR_WIDTH-1:0] rd_idx,

    output logic c_data_valid,
    output logic signed [ACC_WIDTH-1:0] rd_result [M-1:0][N-1:0],

    output logic [1:0] bank_ready,
    output logic owner_error,
    output logic result_rdy
);

    // =========================================================================
    //  SYSTOLIC ARRAY
    // =========================================================================
    logic result_rdy_internal;
    logic signed [ACC_WIDTH-1:0] result [M-1:0][N-1:0];

    systolic_array #(
        .M(M), .N(N),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .LOAD_PSUM_EN(LOAD_PSUM_EN)
    ) u_sa (
        .clk(clk), .rst_n(rst_n),
        .en(sa_en), .clear_acc(clear_acc), .load_psum(load_psum),
        .a_in(a_in), .b_in(b_in), .psum_in(psum_in),
        .result(result),
        .result_rdy(result_rdy_internal)
    );

    assign result_rdy = result_rdy_internal;

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
    logic fifo_wr, fifo_rd, fifo_full, fifo_empty;

    always_comb
        for (int i = 0; i < M; i++)
            fifo_din[i*ROW_WIDTH +: ROW_WIDTH] = row_packed[i];

    // FIXED: FIFO-DROP-001 — When FIFO is full, stall the systolic array
    // by gating the enable signal. This prevents silent data loss.
    // result_rdy_internal pulses when the systolic array has a valid result.
    // If the FIFO can't accept it, we must backpressure the array.
    assign fifo_wr = result_rdy_internal & ~fifo_full;

    // Backpressure: when FIFO is full, deassert en to stall the systolic array
    // so it doesn't produce results that would be dropped.
    logic sa_en;
    assign sa_en = en & ~fifo_full;

    fifo #(.WIDTH(TOTAL_WIDTH), .DEPTH(16)) u_fifo (
        .clk(clk), .rst_n(rst_n),
        .wr_en(fifo_wr), .din(fifo_din), .full(fifo_full),
        .rd_en(fifo_rd), .dout(fifo_dout), .empty(fifo_empty)
    );

    // =========================================================================
    //  ROUND-ROBIN write selector
    //
    //  rr_sel chooses which controller receives each FIFO pop.
    //  The FIFO is only popped when the *target* controller is ready,
    //  so controller 1 being busy never stalls controller 0's turn, and
    //  vice-versa.  This allows the two controllers to run independently:
    //    result 0 → ctrl 0,  result 1 → ctrl 1,  result 2 → ctrl 0, …
    // =========================================================================
    logic rr_sel;  // 0 = next pop goes to ctrl 0, 1 = goes to ctrl 1

    logic [1:0] p_ready;

    // Pop only when the target controller is ready
    assign fifo_rd = ~fifo_empty & (rr_sel ? p_ready[1] : p_ready[0]);

    // Registered version to avoid combinational glitches
    logic fifo_rd_reg;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fifo_rd_reg <= 1'b0;
        else fifo_rd_reg <= fifo_rd;
    end

    // Advance selector on every pop
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rr_sel <= 1'b0;
        else if (fifo_rd_reg) rr_sel <= ~rr_sel;
    end

    // =========================================================================
    //  WRITE-FIRE pipeline
    //
    //  write_fire[k]   — registered version of fifo_rd firing for controller k
    //                    (= p_start[k]; FIFO dout is now valid)
    //  write_fire_d[k] — one cycle later, when fifo_dout is stable → actual
    //                    BRAM write enable and p_done generation
    // =========================================================================
    logic [1:0] p_start, p_done;
    logic [1:0] write_fire, write_fire_d;

    // Registered start pulses to avoid combinational glitches
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
    //
    //  Reset to 0 at the start of a new fill (write_fire), then increment
    //  once per BRAM write (write_fire_d) so every bank's result starts at
    //  address 0.  Supports storing multiple rows per bank fill if DEPTH > 1.
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
    //
    //  Two fully independent FSMs.  c_done is routed only to the controller
    //  that currently has c_data_valid asserted so the other controller's
    //  bank-fresh flags are not disturbed.
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
            .c_start      (c_start),           // broadcast; FSM checks own fresh flags
            .c_data_valid (ctrl_c_data_valid[k]),
            .c_done       (c_done_ctrl[k]),    // only fires when this ctrl is serving
            .sel_wr       (sel_wr[k]),
            .sel_rd       (sel_rd[k]),
            .p_we_gate    (p_we_gate[k])
        );
    end
    endgenerate

    // =========================================================================
    //  FIFO UNPACK
    //
    //  In standard (non-show-ahead) mode, fifo_dout goes to zero once the FIFO
    //  becomes empty after a pop.  The BRAM write pipeline consumes the data
    //  several cycles after the FIFO pop (via write_fire_d), so we must capture
    //  and hold the FIFO output in a skid register.
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
    //
    //  Each controller owns M independent ping-pong BRAMs — one per result row.
    //  Both sets write from the same fifo_row[] unpack, but only the selected
    //  controller's write_fire_d is asserted at any time, so there is no
    //  conflict.  This gives 2 × M total BRAM instances, enabling the consumer
    //  to read from controller K's banks while controller (1-K) is being filled.
    // =========================================================================
    logic [ROW_WIDTH-1:0] rd_row   [1:0][M-1:0];  // [ctrl][row]
    logic [1:0]           bank_rdy [1:0][M-1:0];  // [ctrl][row]
    logic                 own_err  [1:0][M-1:0];  // [ctrl][row]

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
                .p_we              (write_fire_d[k]),
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
    //  OUTPUT
    //
    //  rd_result is muxed from whichever controller currently has c_data_valid.
    //  Controller 0 takes priority when both assert simultaneously.
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

    // bank_ready: report the union across all BRAMs of both controllers
    assign bank_ready = bank_rdy[0][0] | bank_rdy[1][0];

    always_comb begin
        owner_error = 1'b0;
        for (int k = 0; k < 2; k++)
            for (int i = 0; i < M; i++)
                owner_error |= own_err[k][i];
    end

endmodule
