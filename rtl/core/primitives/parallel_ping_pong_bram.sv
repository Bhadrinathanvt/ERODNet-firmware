// =============================================================================
// parallel_ping_pong_bram.sv
//
// Parallel Ping-Pong Double-Buffered BRAM Array.
// Buffers M x N matrix results from systolic arrays.
//
// Features:
//   - M x N independent BRAMs for parallel access
//   - Double-buffering: Bank A and Bank B
//   - Parameterized dimensions and depth
//   - Optimal BRAM inference with (* ram_style = "block" *)
// =============================================================================

module parallel_ping_pong_bram #(
    parameter int M          = 4,
    parameter int N          = 4,
    parameter int ACC_W      = 32,
    parameter int DEPTH      = 16,
    parameter bit PIPELINE_RD = 0
)(
    input  logic clk,
    input  logic rst_n,

    // Ping-pong bank select
    input  logic sel_wr,   // 0: write bankA, 1: write bankB
    input  logic sel_rd,   // 0: read bankA,  1: read bankB

    // Write port — from systolic array
    input  logic                                result_rdy,
    input  logic signed [ACC_W-1:0]             result [M-1:0][N-1:0],
    input  logic [$clog2(DEPTH)-1:0]            wr_idx,

    // Read port — from consumer
    input  logic [$clog2(DEPTH)-1:0]            rd_idx,
    output logic signed [ACC_W-1:0]             rd_data [M-1:0][N-1:0]
);

    // =========================================================================
    // Memory Storage
    // =========================================================================
    // M x N independent BRAMs
    (* ram_style = "block" *)
    logic signed [ACC_W-1:0] bank_a [M-1:0][N-1:0][0:DEPTH-1];
    (* ram_style = "block" *)
    logic signed [ACC_W-1:0] bank_b [M-1:0][N-1:0][0:DEPTH-1];

    // =========================================================================
    // Write Logic
    // =========================================================================
    always_ff @(posedge clk) begin
        if (result_rdy) begin
            for (int i = 0; i < M; i++) begin
                for (int j = 0; j < N; j++) begin
                    if (!sel_wr)
                        bank_a[i][j][wr_idx] <= result[i][j];
                    else
                        bank_b[i][j][wr_idx] <= result[i][j];
                end
            end
        end
    end

    // =========================================================================
    // Read Logic
    // =========================================================================
    logic signed [ACC_W-1:0] dout_reg [M-1:0][N-1:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < M; i++) begin
                for (int j = 0; j < N; j++) begin
                    dout_reg[i][j] <= '0;
                end
            end
        end else begin
            for (int i = 0; i < M; i++) begin
                for (int j = 0; j < N; j++) begin
                    dout_reg[i][j] <= sel_rd ? bank_b[i][j][rd_idx] : bank_a[i][j][rd_idx];
                end
            end
        end
    end

    // Optional pipeline register
    generate
        if (PIPELINE_RD) begin : gen_pipeline
            logic signed [ACC_W-1:0] dout_pipe [M-1:0][N-1:0];
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int i = 0; i < M; i++) begin
                        for (int j = 0; j < N; j++) begin
                            dout_pipe[i][j] <= '0;
                        end
                    end
                end else begin
                    dout_pipe <= dout_reg;
                end
            end
            assign rd_data = dout_pipe;
        end else begin : gen_no_pipeline
            assign rd_data = dout_reg;
        end
    endgenerate

endmodule
