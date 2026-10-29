// =============================================================================
// depthwise_conv_fixed.sv
//
// Fixed pipelined depthwise convolution engine.
//
// Pipeline Stages:
//   Stage 0: Input register (token_in capture)
//   Stage 1: Shift register update
//   Stage 2-3: MAC tree (pipelined accumulation)
//   Stage 4: Output rounding and saturation
//
// Latency: 5 cycles
// Throughput: 1 token/cycle
// =============================================================================

import pkg_config::*;

module depthwise_conv_fixed #(
    parameter int    MAX_C              = 64,
    parameter int    DATA_WIDTH         = 8,
    parameter int    MAX_KERNEL_SIZE    = 49,
    parameter int    ACC_WIDTH          = 32,
    parameter int    WEIGHT_LOAD_MODE   = 0,   // 0=RUNTIME, 1=MEM_FILE
    parameter string WEIGHT_FILE        = ""   // .mem file for depthwise kernel weights
)(
    input  logic clk,
    input  logic rst_n,

    // Runtime config
    input  logic [15:0] cfg_kernel_size,
    input  logic [15:0] cfg_c,

    // Weight load interface
    input  logic        dw_wr_en,
    input  logic [15:0] dw_wr_ch,
    input  logic [15:0] dw_wr_k,
    input  logic signed [DATA_WIDTH-1:0] dw_wr_data,

    // Token stream
    input  logic        valid_in,
    input  logic signed [DATA_WIDTH-1:0] token_in [MAX_C-1:0],
    output logic        ready_out,           // FIXED: backpressure — can accept input
    output logic        valid_out,
    output logic signed [DATA_WIDTH-1:0] token_out [MAX_C-1:0],
    input  logic        out_ready            // FIXED: backpressure — consumer ready
);

    localparam int PIPELINE_DEPTH = 5;
    localparam int ROUND_SHIFT = DATA_WIDTH / 2;
    localparam int HALF_KERNEL = (MAX_KERNEL_SIZE + 1) / 2;

    // =========================================================================
    // Kernel weight RAM (parameterized by WEIGHT_LOAD_MODE)
    // =========================================================================
    (* ram_style = "block" *)
    logic signed [DATA_WIDTH-1:0] kernel [MAX_C-1:0][0:MAX_KERNEL_SIZE-1];

    generate
    if (WEIGHT_LOAD_MODE == 1) begin : gen_mem_weights
        initial begin
            if (WEIGHT_FILE != "")
                $readmemh(WEIGHT_FILE, kernel);
        end
    end else begin : gen_runtime_weights
        always_ff @(posedge clk) begin
            if (dw_wr_en && (dw_wr_ch < MAX_C) && (dw_wr_k < MAX_KERNEL_SIZE)) begin
                kernel[dw_wr_ch][dw_wr_k] <= dw_wr_data;
            end
        end
    end
    endgenerate

    // =========================================================================
    // Stage 0: Input Register
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] token_in_d0 [MAX_C-1:0];
    logic valid_d0;
    logic [15:0] valid_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d0 <= 1'b0;
            valid_cnt <= '0;
            for (int c = 0; c < MAX_C; c++) token_in_d0[c] <= '0;
        end else if (!stall) begin
            valid_d0 <= valid_in;
            if (valid_in && valid_cnt < 16'hFFFF) begin
                valid_cnt <= valid_cnt + 1;
            end
            for (int c = 0; c < MAX_C; c++) begin
                token_in_d0[c] <= token_in[c];
            end
        end
    end

    // =========================================================================
    // Stage 1: Shift Register
    // =========================================================================
    // Shift register for each channel
    logic signed [DATA_WIDTH-1:0] sr [MAX_C-1:0][0:MAX_KERNEL_SIZE-1];
    logic valid_d1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d1 <= 1'b0;
            for (int c = 0; c < MAX_C; c++) begin
                for (int k = 0; k < MAX_KERNEL_SIZE; k++) begin
                    sr[c][k] <= '0;
                end
            end
        end else if (!stall) begin
            valid_d1 <= valid_d0;
            if (valid_d0) begin
                for (int c = 0; c < MAX_C; c++) begin
                    // Shift: new sample at position 0, older samples shift up
                    sr[c][0] <= token_in_d0[c];
                    for (int k = 1; k < MAX_KERNEL_SIZE; k++) begin
                        sr[c][k] <= sr[c][k-1];
                    end
                end
            end
        end
    end

    // =========================================================================
    // Stage 2: MAC Tree — first half (extracted to always_comb)
    logic valid_d2;
    logic signed [ACC_WIDTH-1:0] partial_mac [MAX_C-1:0];
    logic signed [ACC_WIDTH-1:0] partial_mac_comb [MAX_C-1:0];
    always_comb begin
        for (int c = 0; c < MAX_C; c++) begin
            if (c < cfg_c) begin
                automatic logic signed [ACC_WIDTH-1:0] acc = '0;   // per-iteration reset (was implicitly static)
                for (int k = 0; k < HALF_KERNEL; k++) begin
                    if (k < cfg_kernel_size) begin
                        acc = acc + (ACC_WIDTH'(signed'(sr[c][k])) *
                                    ACC_WIDTH'(signed'(kernel[c][k])));
                    end
                end
                partial_mac_comb[c] = acc;
            end else begin
                partial_mac_comb[c] = '0;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d2 <= 1'b0;
            for (int c = 0; c < MAX_C; c++) partial_mac[c] <= '0;
        end else if (!stall) begin
            valid_d2 <= valid_d1;
            if (valid_d1) begin
                for (int c = 0; c < MAX_C; c++) partial_mac[c] <= partial_mac_comb[c];
            end
        end
    end

    // =========================================================================
    // Stage 3: MAC Tree — second half (extracted to always_comb)
    // =========================================================================
    logic valid_d3;
    logic signed [ACC_WIDTH-1:0] mac_result [MAX_C-1:0];
    logic signed [ACC_WIDTH-1:0] mac_result_comb [MAX_C-1:0];
    always_comb begin
        for (int c = 0; c < MAX_C; c++) begin
            if (c < cfg_c) begin
                automatic logic signed [ACC_WIDTH-1:0] acc = partial_mac[c];   // per-iteration reset (was implicitly static)
                for (int k = HALF_KERNEL; k < MAX_KERNEL_SIZE; k++) begin
                    if (k < cfg_kernel_size) begin
                        acc = acc + (ACC_WIDTH'(signed'(sr[c][k])) *
                                    ACC_WIDTH'(signed'(kernel[c][k])));
                    end
                end
                mac_result_comb[c] = acc;
            end else begin
                mac_result_comb[c] = '0;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d3 <= 1'b0;
            for (int c = 0; c < MAX_C; c++) mac_result[c] <= '0;
        end else if (!stall) begin
            valid_d3 <= valid_d2;
            if (valid_d2) begin
                for (int c = 0; c < MAX_C; c++) mac_result[c] <= mac_result_comb[c];
            end
        end
    end

    // =========================================================================
    // Stage 4: Output (Rounding + Saturation)
    // =========================================================================
    localparam logic signed [DATA_WIDTH-1:0] SAT_MAX = (1 << (DATA_WIDTH-1)) - 1;
    localparam logic signed [DATA_WIDTH-1:0] SAT_MIN = -(1 << (DATA_WIDTH-1));

    function automatic logic signed [DATA_WIDTH-1:0] saturate(
        input logic signed [ACC_WIDTH-1:0] val
    );
        logic signed [DATA_WIDTH-1:0] result;
        logic signed [DATA_WIDTH-1:0] shifted;
        shifted = DATA_WIDTH'(val >>> ROUND_SHIFT);

        if (val > ACC_WIDTH'({1'b0, {(ACC_WIDTH-1){1'b1}}})) begin
            result = SAT_MAX;
        end else if (val < ACC_WIDTH'({1'b1, {(ACC_WIDTH-1){1'b0}}})) begin
            result = SAT_MIN;
        end else begin
            result = shifted;
        end
        return result;
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            for (int c = 0; c < MAX_C; c++) token_out[c] <= '0;
        end else begin
            // Output valid when buffer has enough samples AND pipeline is full
            valid_out <= valid_d3 && (valid_cnt >= cfg_kernel_size - 1 + PIPELINE_DEPTH - 1);
            for (int c = 0; c < MAX_C; c++) begin
                token_out[c] <= saturate(mac_result[c]);
            end
        end
    end

    // =========================================================================
    // Backpressure: stall pipeline when consumer is not ready
    // FIXED: added ready_out/out_ready handshake to prevent data loss.
    // =========================================================================
    logic stall;
    assign stall    = valid_out && !out_ready;
    assign ready_out = !stall;   // Accept new input when not stalled

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    DW_KERNEL_BOUNDS: assert property (
        @(posedge clk) disable iff (!rst_n)
        dw_wr_en |-> (dw_wr_ch < MAX_C && dw_wr_k < MAX_KERNEL_SIZE)
    ) else $error("depthwise_conv_fixed: Weight write out of bounds");

    DW_CHANNEL_BOUNDS: assert property (
        @(posedge clk) disable iff (!rst_n)
        cfg_c <= MAX_C
    ) else $error("depthwise_conv_fixed: cfg_c exceeds MAX_C");
    // synthesis translate_on

endmodule