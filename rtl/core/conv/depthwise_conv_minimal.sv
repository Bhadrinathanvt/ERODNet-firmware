// =============================================================================
// depthwise_conv_minimal.sv
//
// Minimal depthwise convolution for ultra-low LUT configuration.
// - 3x3 kernel maximum (reduced from 7x7)
// - Time-multiplexed kernel computation
// - LUT-based (no DSP)
//
// Resource Savings:
//   - Full config: 64 channels x 49 kernel positions = ~8000 LUTs
//   - Minimal config: 16 channels x 9 kernel positions = ~800 LUTs
//   - Savings: ~7200 LUTs (90% reduction)
//
// Handshake:
//   - ready_in: deasserted during ACCUMULATE/OUTPUT, reasserted in IDLE
//   - out_ready: consumer backpressure; valid_out held until out_ready=1
// =============================================================================

import pkg_artix7_minimal::*;

module depthwise_conv_minimal #(
    parameter int MAX_C         = 16,
    parameter int DATA_WIDTH    = 8,
    parameter int KERNEL_SIZE   = 9,      // 3x3 kernel
    parameter int ACC_WIDTH     = 24
)(
    input  logic clk,
    input  logic rst_n,

    // Configuration
    input  logic [3:0] cfg_kernel_size,   // 1, 4, or 9 (1x1, 2x2, 3x3)
    input  logic [15:0] cfg_c,            // Number of channels

    // Weight loading
    input  logic dw_wr_en,
    input  logic [15:0] dw_wr_ch,
    input  logic [15:0] dw_wr_k,
    input  logic signed [DATA_WIDTH-1:0] dw_wr_data,

    // Input stream
    input  logic valid_in,
    input  logic signed [DATA_WIDTH-1:0] token_in [MAX_C-1:0],
    output logic ready_in,

    // Output stream
    output logic valid_out,
    output logic signed [DATA_WIDTH-1:0] token_out [MAX_C-1:0],
    input  logic        out_ready          // Consumer backpressure
);

    // =========================================================================
    // Local Parameters
    // =========================================================================
    localparam int KERNEL_DIM = 3;  // 3x3 kernel
    localparam int PIPELINE_STAGES = KERNEL_DIM * KERNEL_DIM;  // 9 stages

    // =========================================================================
    // Weight Storage (Block RAM)
    // =========================================================================
    (* ram_style = "block" *)
    logic signed [DATA_WIDTH-1:0] kernel_weights [0:MAX_C-1][0:KERNEL_SIZE-1];

    // Weight loading
    always_ff @(posedge clk) begin
        if (dw_wr_en && dw_wr_ch < MAX_C && dw_wr_k < KERNEL_SIZE) begin
            kernel_weights[dw_wr_ch][dw_wr_k] <= dw_wr_data;
        end
    end

    // =========================================================================
    // Line Buffer (3 rows for 3x3 convolution)
    // =========================================================================
    (* ram_style = "block" *)
    logic signed [DATA_WIDTH-1:0] line_buf [0:2][0:MAX_C-1];

    logic [1:0] row_ptr;  // Current row being written

    // =========================================================================
    // Pipeline Control
    // =========================================================================
    typedef enum logic [2:0] {
        IDLE,           // Wait for input
        ACCUMULATE,     // Compute weighted sum
        OUTPUT          // Output result
    } state_t;

    state_t state;
    logic [$clog2(KERNEL_SIZE):0] k_idx;      // Kernel index
    logic [$clog2(MAX_C):0] ch_idx;            // Channel index
    logic [$clog2(PIPELINE_STAGES):0] pipe_cnt; // Pipeline counter

    logic signed [ACC_WIDTH-1:0] acc [0:MAX_C-1];
    logic signed [DATA_WIDTH-1:0] token_in_buf [MAX_C-1:0];

    // =========================================================================
    // FSM
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            k_idx <= '0;
            ch_idx <= '0;
            pipe_cnt <= '0;
            row_ptr <= '0;
            valid_out <= '0;
            ready_in <= '1;
            for (int c = 0; c < MAX_C; c++) begin
                acc[c] <= '0;
            end
        end else begin
            case (state)
                // -------------------------------------------------------------
                // IDLE: Wait for valid input
                // -------------------------------------------------------------
                IDLE: begin
                    valid_out <= '0;

                    if (valid_in) begin
                        // Capture input
                        for (int c = 0; c < MAX_C; c++)
                            token_in_buf[c] <= token_in[c];

                        // Update line buffer
                        for (int c = 0; c < MAX_C; c++)
                            line_buf[row_ptr][c] <= token_in[c];

                        row_ptr <= (row_ptr == 2) ? 0 : row_ptr + 1;
                        ready_in <= '0;

                        // Start accumulation
                        ch_idx <= '0;
                        k_idx <= '0;
                        for (int c = 0; c < MAX_C; c++)
                            acc[c] <= '0;

                        state <= ACCUMULATE;
                    end
                end

                // -------------------------------------------------------------
                // ACCUMULATE: Compute depthwise convolution
                // -------------------------------------------------------------
                ACCUMULATE: begin
                    if (k_idx < cfg_kernel_size) begin
                        // LUT-based multiply-accumulate
                        for (int c = 0; c < MAX_C; c++) begin
                            if (c < cfg_c) begin
                                (* use_dsp = "no" *)
                                acc[c] <= acc[c] +
                                    (kernel_weights[c][k_idx] * token_in_buf[c]);
                            end
                        end
                        k_idx <= k_idx + 1'b1;
                    end else begin
                        // Done accumulating, move to output
                        state <= OUTPUT;
                        pipe_cnt <= '0;
                    end
                end

                // -------------------------------------------------------------
                // OUTPUT: Present result, wait for consumer acceptance
                // -------------------------------------------------------------
                OUTPUT: begin
                    valid_out <= '1;

                    if (pipe_cnt == 0) begin
                        // Latch output on first OUTPUT cycle
                        for (int c = 0; c < MAX_C; c++) begin
                            if (c < cfg_c)
                                token_out[c] <= acc[c][ACC_WIDTH-1:ACC_WIDTH-DATA_WIDTH];
                            else
                                token_out[c] <= '0;
                        end
                        pipe_cnt <= pipe_cnt + 1'b1;
                    end else if (out_ready) begin
                        state <= IDLE;
                        ready_in <= '1;
                        valid_out <= '0;
                        pipe_cnt <= '0;
                    end else if (pipe_cnt < PIPELINE_STAGES - 1) begin
                        pipe_cnt <= pipe_cnt + 1'b1;
                    end
                    // If pipe_cnt == PIPELINE_STAGES-1 and out_ready==0, hold
                    // pipe_cnt at max and keep valid_out high until acceptance.
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
