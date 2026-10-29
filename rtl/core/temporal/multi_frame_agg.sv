// =============================================================================
// multi_frame_agg.sv
//
// Multi-frame aggregation module for temporal radar feature processing.
// Combines features from multiple time frames using various strategies:
//   - 3D Convolution: Spatio-temporal convolution across (H, W, T)
//   - Temporal Pooling: Average/Max pooling across time dimension
//   - Recurrent: LSTM/GRU-style temporal state update
//   - ConvLSTM: Convolutional LSTM for spatio-temporal learning
//
// This module is designed for RODNet-style radar object detection where
// temporal information (Doppler velocity) is critical for distinguishing
// static background from moving objects.
//
// Pipeline Stages:
//   Stage 1: Input registration and frame buffer write
//   Stage 2: Temporal buffer read (T frames)
//   Stage 3: Aggregation computation
//   Stage 4: Output with residual connection
//
// Latency: 6-12 cycles (mode dependent)
// Throughput: 1 frame/cycle after fill
// =============================================================================

import pkg_config::*;

module multi_frame_agg #(
    parameter int HISTORY_DEPTH = 8,     // Number of temporal frames (T)
    parameter int CHANNELS      = MAX_C, // Feature channels
    parameter int KERNEL_T      = 3,     // Temporal kernel size
    parameter int DATA_W        = DATA_WIDTH,
    parameter int ACC_W        = ACC_WIDTH
)(
    input  logic clk,
    input  logic rst_n,

    // Configuration
    input  logic [2:0]  cfg_agg_mode,       // Aggregation mode
    input  logic [3:0]  cfg_history_depth,   // T frames to aggregate
    input  logic [3:0]  cfg_kernel_t,        // Temporal kernel size
    input  logic         cfg_enable,

    // Input stream (current frame)
    input  logic                        frame_valid_in,
    input  logic                        frame_last_in,
    input  logic signed [DATA_W-1:0]   frame_data_in [CHANNELS-1:0],
    output logic                        frame_ready_in,

    // Temporal weights (for 3D Conv or Recurrent modes)
    input  logic signed [DATA_W-1:0]   temporal_weights [KERNEL_T-1:0][CHANNELS-1:0],

    // Output stream
    output logic                        agg_valid_out,
    output logic signed [DATA_W-1:0]   agg_data_out [CHANNELS-1:0],
    input  logic                        agg_ready_in,

    // Temporal state (for recurrent modes)
    output logic signed [DATA_W-1:0]   hidden_state_out [CHANNELS-1:0],
    input  logic signed [DATA_W-1:0]   hidden_state_in [CHANNELS-1:0]
);

    // =========================================================================
    // Aggregation Modes
    // =========================================================================
    localparam bit [2:0] MODE_3D_CONV    = 3'b000;  // 3D convolution across time
    localparam bit [2:0] MODE_T_AVG_POOL = 3'b001;  // Temporal average pooling
    localparam bit [2:0] MODE_T_MAX_POOL = 3'b010;  // Temporal max pooling
    localparam bit [2:0] MODE_CONV_LSTM  = 3'b011;  // Convolutional LSTM
    localparam bit [2:0] MODE_RECURRENT   = 3'b100;  // Simple recurrent update

    // =========================================================================
    // Temporal Frame Buffer
    // =========================================================================
    // Store T frames, each with C channels
    (* ram_style = "distributed" *)
    logic signed [DATA_W-1:0] frame_buffer [HISTORY_DEPTH-1:0][CHANNELS-1:0];

    // Buffer pointers
    logic [$clog2(HISTORY_DEPTH)-1:0] wr_ptr;
    logic [HISTORY_DEPTH-1:0] frame_valid;

    // Frame counter for tracking
    logic [$clog2(HISTORY_DEPTH):0] frame_count;

    // =========================================================================
    // Pipeline Stage 1: Input Registration
    // =========================================================================
    logic signed [DATA_W-1:0] input_reg [CHANNELS-1:0];
    logic valid_d1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d1 <= 1'b0;
            wr_ptr <= '0;
            frame_valid <= '0;
            frame_count <= '0;
            for (int c = 0; c < CHANNELS; c++)
                input_reg[c] <= '0;
        end else if (cfg_enable && frame_valid_in && frame_ready_in) begin
            valid_d1 <= 1'b1;

            // Register input
            for (int c = 0; c < CHANNELS; c++)
                input_reg[c] <= frame_data_in[c];

            // Write to circular buffer
            for (int c = 0; c < CHANNELS; c++)
                frame_buffer[wr_ptr][c] <= frame_data_in[c];

            // Mark frame as valid
            frame_valid[wr_ptr] <= 1'b1;

            // Update pointers
            if (frame_last_in) begin
                wr_ptr <= wr_ptr + 1'b1;
                if (frame_count < HISTORY_DEPTH)
                    frame_count <= frame_count + 1'b1;
            end
        end else begin
            valid_d1 <= 1'b0;
        end
    end

    // =========================================================================
    // Pipeline Stage 2: Temporal Buffer Read
    // =========================================================================
    // Read KERNEL_T frames centered around current frame
    logic signed [DATA_W-1:0] temporal_window [KERNEL_T-1:0][CHANNELS-1:0];
    logic [KERNEL_T-1:0] window_valid;
    logic valid_d2;

    // Compute which frames to read based on kernel position
    logic [$clog2(HISTORY_DEPTH)-1:0] center_ptr;
    assign center_ptr = wr_ptr > 0 ? wr_ptr - 1 : HISTORY_DEPTH - 1;

    // Combinational frame index computation (extracted from always_ff for Verilator compat)
    logic [$clog2(HISTORY_DEPTH)-1:0] frame_idx_comb [0:KERNEL_T-1];
    always_comb begin
        for (int t = 0; t < KERNEL_T; t++) begin
            if (t < cfg_kernel_t)
                frame_idx_comb[t] = (center_ptr + HISTORY_DEPTH + t - (cfg_kernel_t >> 1))
                                    & (HISTORY_DEPTH - 1);
            else
                frame_idx_comb[t] = '0;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d2 <= 1'b0;
            window_valid <= '0;
            for (int t = 0; t < KERNEL_T; t++)
                for (int c = 0; c < CHANNELS; c++)
                    temporal_window[t][c] <= '0;
        end else begin
            valid_d2 <= valid_d1;

            for (int t = 0; t < KERNEL_T; t++) begin
                if (t < cfg_kernel_t) begin
                    window_valid[t] <= frame_valid[frame_idx_comb[t]];
                    for (int c = 0; c < CHANNELS; c++)
                        temporal_window[t][c] <= frame_buffer[frame_idx_comb[t]][c];
                end else begin
                    window_valid[t] <= 1'b0;
                    for (int c = 0; c < CHANNELS; c++)
                        temporal_window[t][c] <= '0;
                end
            end
        end
    end

    // =========================================================================
    // Pipeline Stage 3-4: Aggregation Computation
    // =========================================================================
    logic signed [ACC_W-1:0] agg_result [CHANNELS-1:0];
    logic signed [DATA_W-1:0] max_result [CHANNELS-1:0];
    logic signed [DATA_W-1:0] hidden_update [CHANNELS-1:0];
    logic valid_d3, valid_d4;

    // 3D Convolution: combinational weighted sum (extracted from always_ff for Verilator compat)
    logic signed [ACC_W-1:0] conv_result_comb [CHANNELS-1:0];
    always_comb begin
        for (int c = 0; c < CHANNELS; c++) begin
            automatic logic signed [ACC_W-1:0] acc = '0;
            for (int t = 0; t < KERNEL_T; t++) begin
                if (window_valid[t])
                    acc = acc + (ACC_W'(temporal_weights[t][c]) * ACC_W'(temporal_window[t][c]));
            end
            conv_result_comb[c] = acc;
        end
    end

    // T_AVG_POOL: combinational shift-based average (extracted from always_ff)
    logic signed [ACC_W-1:0] avg_result_comb [CHANNELS-1:0];
    always_comb begin
        for (int c = 0; c < CHANNELS; c++) begin
            automatic logic signed [ACC_W-1:0] sum = '0;
            automatic logic [3:0] count = '0;
            for (int t = 0; t < KERNEL_T; t++) begin
                if (window_valid[t]) begin
                    sum = sum + ACC_W'(temporal_window[t][c]);
                    count = count + 1;
                end
            end
            if (count == 4'd0) avg_result_comb[c] = '0;
            else if (count == 4'd1) avg_result_comb[c] = sum;
            else if (count == 4'd2) avg_result_comb[c] = sum >>> 1;
            else if (count == 4'd3) avg_result_comb[c] = (sum >>> 2) + (sum >>> 4) + (sum >>> 8);
            else if (count == 4'd4) avg_result_comb[c] = sum >>> 2;
            else if (count == 4'd5) avg_result_comb[c] = (sum >>> 3) + (sum >>> 5) + (sum >>> 7);
            else if (count == 4'd6) avg_result_comb[c] = (sum >>> 3) + (sum >>> 5);
            else if (count == 4'd7) avg_result_comb[c] = (sum >>> 3) + (sum >>> 6) + (sum >>> 9);
            else avg_result_comb[c] = sum >>> 3;
        end
    end

    // T_MAX_POOL: combinational max over valid frames (extracted from always_ff)
    logic signed [DATA_W-1:0] max_result_comb [CHANNELS-1:0];

    // Residual: combinational saturated add (extracted from always_ff for Verilator compat)
    logic signed [DATA_W-1:0] residual_comb [CHANNELS-1:0];
    always_comb begin
        for (int c = 0; c < CHANNELS; c++) begin
            automatic logic signed [ACC_W-1:0] r = agg_result[c] + ACC_W'(input_reg[c]);
            if (r > (1 <<< (DATA_W - 1)) - 1)
                residual_comb[c] = (1 <<< (DATA_W - 1)) - 1;
            else if (r < -(1 <<< (DATA_W - 1)))
                residual_comb[c] = -(1 <<< (DATA_W - 1));
            else
                residual_comb[c] = DATA_W'(r);
        end
    end
    always_comb begin
        for (int c = 0; c < CHANNELS; c++) begin
            automatic logic signed [DATA_W-1:0] mv = {DATA_W{1'b1}};
            automatic logic fv = 1'b0;
            for (int t = 0; t < KERNEL_T; t++) begin
                if (window_valid[t]) begin
                    if (!fv || temporal_window[t][c] > mv) mv = temporal_window[t][c];
                    fv = 1'b1;
                end
            end
            max_result_comb[c] = fv ? mv : '0;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d3 <= 1'b0;
            for (int c = 0; c < CHANNELS; c++) begin
                agg_result[c] <= '0;
                max_result[c] <= '0;
            end
        end else begin
            valid_d3 <= valid_d2;

            case (cfg_agg_mode)
                MODE_3D_CONV: begin
                    for (int c = 0; c < CHANNELS; c++)
                        agg_result[c] <= conv_result_comb[c];
                end

                MODE_T_AVG_POOL: begin
                    for (int c = 0; c < CHANNELS; c++)
                        agg_result[c] <= avg_result_comb[c];
                end

                MODE_T_MAX_POOL: begin
                    for (int c = 0; c < CHANNELS; c++)
                        max_result[c] <= max_result_comb[c];
                end

                MODE_CONV_LSTM, MODE_RECURRENT: begin
                    for (int c = 0; c < CHANNELS; c++) begin
                        hidden_update[c] <= DATA_W'((ACC_W'(hidden_state_in[c]) +
                                                     ACC_W'(input_reg[c])) >>> 1);
                        agg_result[c] <= ACC_W'(hidden_update[c]);
                    end
                end

                default: begin
                    for (int c = 0; c < CHANNELS; c++)
                        agg_result[c] <= ACC_W'(input_reg[c]);
                end
            endcase
        end
    end

    // =========================================================================
    // Pipeline Stage 4: Output with Residual
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_d4 <= 1'b0;
            agg_valid_out <= 1'b0;
            for (int c = 0; c < CHANNELS; c++) begin
                agg_data_out[c] <= '0;
                hidden_state_out[c] <= '0;
            end
        end else begin
            valid_d4 <= valid_d3;

            if (valid_d3) begin
                agg_valid_out <= 1'b1;

                case (cfg_agg_mode)
                    MODE_T_MAX_POOL: begin
                        for (int c = 0; c < CHANNELS; c++)
                            agg_data_out[c] <= max_result[c];
                    end

                    MODE_CONV_LSTM, MODE_RECURRENT: begin
                        // Output hidden state
                        for (int c = 0; c < CHANNELS; c++) begin
                            hidden_state_out[c] <= hidden_update[c];
                            agg_data_out[c] <= hidden_update[c];
                        end
                    end

                    default: begin
                        for (int c = 0; c < CHANNELS; c++)
                            agg_data_out[c] <= residual_comb[c];
                    end
                endcase
            end else if (agg_valid_out && agg_ready_in) begin
                agg_valid_out <= 1'b0;
            end
        end
    end

    // Backpressure
    assign frame_ready_in = !agg_valid_out || agg_ready_in;

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    MFA_MODE: assert property (
        @(posedge clk) disable iff (!rst_n)
        cfg_agg_mode inside {MODE_3D_CONV, MODE_T_AVG_POOL, MODE_T_MAX_POOL,
                            MODE_CONV_LSTM, MODE_RECURRENT}
    ) else $error("multi_frame_agg: Invalid aggregation mode");

    MFA_KERNEL: assert property (
        @(posedge clk) disable iff (!rst_n)
        cfg_kernel_t > 0 && cfg_kernel_t <= KERNEL_T
    ) else $error("multi_frame_agg: Invalid temporal kernel size");

    MFA_HISTORY: assert property (
        @(posedge clk) disable iff (!rst_n)
        cfg_history_depth > 0 && cfg_history_depth <= HISTORY_DEPTH
    ) else $error("multi_frame_agg: Invalid history depth");
    // synthesis translate_on

endmodule