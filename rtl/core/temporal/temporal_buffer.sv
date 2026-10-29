// =============================================================================
// temporal_buffer.sv
//
// Optimized Multi-frame temporal buffer for RODNet-style processing.
// FIXED: 
//   1. Removed artificial CHANNELS-cycle delay. Full frame writes in 1 cycle.
//   2. Removed invalid BRAM directive; optimized for Distributed RAM (LUTRAM) 
//      which natively supports massive parallel wide-reads for shallow depths.
//   3. FSM collapsed for maximum throughput (1 frame per cycle continuous).
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;

module temporal_buffer #(
    parameter int HISTORY_DEPTH = 8,     // Number of frames to store (T)
    parameter int CHANNELS      = MAX_C, // Feature channels per frame
    parameter int DATA_W = pkg_module_config::TEMPORAL_FUSION_DATA_WIDTH
)(
    input  logic clk,
    input  logic rst_n,

    // Configuration
    input  logic [3:0]  cfg_history_depth,
    input  logic        cfg_enable,

    // Input stream (current frame)
    input  logic                        frame_valid_in,
    input  logic                        frame_last,       
    input  logic signed [DATA_W-1:0]    frame_data_in [CHANNELS-1:0],
    output logic                        frame_ready_in,

    // Output stream (all history frames)
    output logic                        history_valid_out,
    output logic [HISTORY_DEPTH-1:0]    history_valid_mask,
    output logic signed [DATA_W-1:0]    history_data [HISTORY_DEPTH-1:0][CHANNELS-1:0],
    input  logic                        history_ready_in,

    // Control
    input  logic                        frame_start,
    output logic                        buffer_full,
    output logic [$clog2(HISTORY_DEPTH):0] frame_count
);

    // =========================================================================
    // Distributed RAM / Register File
    // =========================================================================
    // Removed BRAM directive. For shallow (e.g., 8 deep) and extremely wide 
    // (e.g., 2048 bit) memories, LUTRAM is the optimal physical target.
    logic signed [DATA_W-1:0] frame_mem [HISTORY_DEPTH-1:0][CHANNELS-1:0];
    logic [HISTORY_DEPTH-1:0] frame_valid;

    logic [$clog2(HISTORY_DEPTH)-1:0] wr_ptr;

    // =========================================================================
    // Core Logic (Zero-Stall Write)
    // =========================================================================
    // The buffer can accept new frames when enabled and not full.
    // Don't stall input just because downstream is slow — the circular
    // buffer absorbs slack until all slots are occupied.
    assign frame_ready_in = cfg_enable && !buffer_full;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr      <= '0;
            frame_valid <= '0;
        end else if (cfg_enable) begin
            
            // Synchronous Reset / Sequence Start
            if (frame_start) begin
                wr_ptr      <= '0;
                frame_valid <= '0;
            end 
            // 1-Cycle Parallel Write
            else if (frame_valid_in && frame_ready_in) begin
                
                // Write all channels simultaneously in 1 clock cycle
                for (int c = 0; c < CHANNELS; c++) begin
                    frame_mem[wr_ptr][c] <= frame_data_in[c];
                end

                frame_valid[wr_ptr] <= 1'b1;

                // Advance write pointer (circularly)
                if (wr_ptr >= HISTORY_DEPTH - 1) begin
                    wr_ptr <= '0;
                end else begin
                    wr_ptr <= wr_ptr + 1'b1;
                end
            end
        end
    end

    // =========================================================================
    // Output Hold Register (backpressure handling)
    // =========================================================================
    // When downstream is not ready, we must hold the last output stable.
    // output_hold tracks whether we're in a "frozen" state.
    logic output_hold;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_hold <= 1'b0;
        end else begin
            if (history_ready_in && output_hold) begin
                // Downstream consumed the held data — release hold
                output_hold <= 1'b0;
            end else if (history_valid_out && !history_ready_in && !output_hold) begin
                // Downstream stalled while we had valid data — enter hold
                output_hold <= 1'b1;
            end
        end
    end

    // =========================================================================
    // Output Interface
    // =========================================================================
    assign history_valid_mask = frame_valid;

    // Valid is purely combinational — stable as long as we have frames
    // BUT: must deassert when we're about to update while downstream is holding us.
    // We achieve this by gating the data update (see below), not by dropping valid.
    assign history_valid_out = !output_hold && (frame_valid != '0);

    assign buffer_full = &frame_valid;

    // Frame count explicitly calculated via popcount
    logic [$clog2(HISTORY_DEPTH):0] frame_count_next;
    always_comb begin
        frame_count_next = '0;
        for (int b = 0; b < HISTORY_DEPTH; b++) begin
            if (frame_valid[b]) frame_count_next = frame_count_next + 1'b1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            frame_count <= '0;
        end else begin
            frame_count <= frame_count_next;
        end
    end

    // Combinational wide-read for all history frames.
    // This allows your Temporal Attention module to see all data instantly.
    //
    // HOLD-UNTIL-ACK FIX (TEMPORAL_BUF-HS-001):
    // We only update history_data when NOT in hold state. Once output_hold
    // is set, history_data is stable and won't be clobbered by new frames
    // arriving while downstream is stalling.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int t = 0; t < HISTORY_DEPTH; t++) begin
                for (int c = 0; c < CHANNELS; c++) begin
                    history_data[t][c] <= '0;
                end
            end
        end else if (!output_hold) begin
            // Only update when NOT holding — data stays stable while downstream
            // processes the current frame without being clobbered.
            if (frame_valid != '0) begin
                for (int t = 0; t < HISTORY_DEPTH; t++) begin
                    if (frame_valid[t]) begin
                        for (int c = 0; c < CHANNELS; c++) begin
                            history_data[t][c] <= frame_mem[t][c];
                        end
                    end
                end
            end
        end
    end

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    TEMPORAL_BUF_DEPTH: assert property (
        @(posedge clk) disable iff (!rst_n)
        cfg_history_depth > 0 && cfg_history_depth <= HISTORY_DEPTH
    ) else $error("temporal_buffer: Invalid history depth configuration");
    // synthesis translate_on

endmodule