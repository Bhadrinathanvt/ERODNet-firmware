// =============================================================================
//  sparse_nz_encoder.sv
//
//  Converts a dense input stream into a non-zero (NZ) packet stream using
//  either Compressed Sparse Row (CSR) or Run-Length Encoding (RLE).
//
//  Modes:
//    SPARSE_MODE_CSR:   Emit {idx, val} for each non-zero element.
//                       idx is the element's position in the dense stream.
//    SPARSE_MODE_RLE:   Emit {run_length, val} pairs. A zero-run is encoded
//                       as {run_length, 0}. A non-zero element has run_length=0.
//    SPARSE_MODE_DENSE: Pass all elements through with sequential indices,
//                       producing bit-identical output to the original array.
//
//  The encoder streams one element per cycle. When a zero element is detected
//  (CSR mode), it is simply skipped — no packet is emitted. This reduces the
//  number of MAC operations downstream.
//
//  Pipeline: 2 stages (input register → NZ packet output register).
//  Handshake: valid/ready on both input and output sides.
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;
import pkg_sparse_config::*;

module sparse_nz_encoder #(
    parameter int DATA_WIDTH = pkg_module_config::SYSTOLIC_ARRAY_DATA_WIDTH,
    parameter int MAX_ELEMENTS = pkg_config::MAX_C * pkg_config::EXPANSION_FACTOR,
    parameter bit  USE_DSP     = 0
)(
    input  logic clk,
    input  logic rst_n,

    // ── Control ─────────────────────────────────────────────────────────────
    input  logic        sparse_en,      // Enable sparse encoding
    input  logic [1:0]  mode_sel,       // sparse_mode_e: CSR, RLE, or Dense

    // ── Input handshake (dense stream) ──────────────────────────────────────
    input  logic                             in_valid,
    output logic                             in_ready,
    input  logic signed [DATA_WIDTH-1:0]    in_data,

    // ── Output handshake (NZ packets) ───────────────────────────────────────
    output logic                             out_valid,
    input  logic                             out_ready,
    output logic                             pkt_last,
    output logic        [NZ_IDX_WIDTH-1:0]  pkt_idx,
    output logic signed [DATA_WIDTH-1:0]    pkt_val
);

    // =========================================================================
    //  Backpressure / Stall
    // =========================================================================
    logic stall;
    assign stall    = out_valid && !out_ready;
    assign in_ready = !stall && !rle_hold;

    // =========================================================================
    //  Internal state
    // =========================================================================
    logic [NZ_IDX_WIDTH-1:0] element_cnt;  // Current element index in dense stream
    logic signed [DATA_WIDTH-1:0] rle_zero_run;  // RLE: count of consecutive zeros
    logic rle_hold;                         // RLE: holding a zero-run packet

    // =========================================================================
    //  Element counter — tracks position in the dense stream
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            element_cnt <= '0;
        end else if (!stall) begin
            if (in_valid && in_ready) begin
                if (element_cnt == MAX_ELEMENTS - 1)
                    element_cnt <= '0;     // Wrap for next tile
                else
                    element_cnt <= element_cnt + 1'b1;
            end
        end
    end

    // =========================================================================
    //  RLE zero-run counter
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rle_zero_run <= '0;
            rle_hold     <= 1'b0;
        end else if (!stall) begin
            if (mode_sel == SPARSE_MODE_RLE && sparse_en) begin
                if (in_valid && in_ready) begin
                    if (in_data == '0) begin
                        // Accumulate zero run
                        rle_zero_run <= rle_zero_run + 1'b1;
                        rle_hold     <= 1'b0;  // Don't emit yet — may extend
                    end else begin
                        // Non-zero element: if we had a pending zero run, hold it
                        rle_hold <= (rle_zero_run != '0);
                        // Don't clear rle_zero_run yet — it will be emitted first
                    end
                end
                if (rle_hold && out_valid && out_ready) begin
                    // Zero-run packet was consumed — clear
                    rle_zero_run <= '0;
                    rle_hold     <= 1'b0;
                end
            end else begin
                rle_zero_run <= '0;
                rle_hold     <= 1'b0;
            end
        end
    end

    // =========================================================================
    //  NZ Packet Output
    // =========================================================================
    logic is_nz;
    logic emit_pkt;
    logic last_element;

    // An element is non-zero if its value != 0 or if we're in dense mode
    assign is_nz = (in_data != '0) || !sparse_en;

    // Determine if this is the last element in the stream
    assign last_element = (element_cnt == MAX_ELEMENTS - 1);

    // Emit packet when:
    //   CSR/Dense: element is non-zero (or any element in dense mode)
    //   RLE: element is non-zero OR we have a pending zero-run to flush
    always_comb begin
        case (mode_sel)
            SPARSE_MODE_CSR:   emit_pkt = sparse_en ? is_nz : 1'b1;
            SPARSE_MODE_RLE:   emit_pkt = sparse_en ? (is_nz || rle_hold) : 1'b1;
            SPARSE_MODE_DENSE: emit_pkt = 1'b1;    // Always emit in dense mode
            default:            emit_pkt = 1'b1;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            pkt_last  <= 1'b0;
            pkt_idx   <= '0;
            pkt_val   <= '0;
        end else if (!stall) begin
            if (rle_hold && (mode_sel == SPARSE_MODE_RLE) && sparse_en) begin
                // Emit pending RLE zero-run packet
                out_valid <= 1'b1;
                pkt_idx   <= NZ_IDX_WIDTH'(rle_zero_run);
                pkt_val   <= '0;   // Zero value signals a run
                pkt_last  <= last_element && !is_nz;
            end else if (in_valid && in_ready) begin
                case (mode_sel)
                    SPARSE_MODE_CSR: begin
                        if (sparse_en) begin
                            out_valid <= is_nz;
                            pkt_idx   <= element_cnt;
                            pkt_val   <= in_data;
                            pkt_last  <= last_element;
                        end else begin
                            // Dense passthrough
                            out_valid <= 1'b1;
                            pkt_idx   <= element_cnt;
                            pkt_val   <= in_data;
                            pkt_last  <= last_element;
                        end
                    end

                    SPARSE_MODE_RLE: begin
                        if (sparse_en && is_nz) begin
                            out_valid <= 1'b1;
                            pkt_idx   <= '0;  // NZ element: run_length=0
                            pkt_val   <= in_data;
                            pkt_last  <= last_element;
                        end else begin
                            out_valid <= 1'b1;
                            pkt_idx   <= element_cnt;
                            pkt_val   <= in_data;
                            pkt_last  <= last_element;
                        end
                    end

                    SPARSE_MODE_DENSE: begin
                        out_valid <= 1'b1;
                        pkt_idx   <= element_cnt;
                        pkt_val   <= in_data;
                        pkt_last  <= last_element;
                    end

                    default: begin
                        out_valid <= 1'b1;
                        pkt_idx   <= element_cnt;
                        pkt_val   <= in_data;
                        pkt_last  <= last_element;
                    end
                endcase
            end else begin
                out_valid <= 1'b0;
            end
        end
        // else: stall — hold all outputs stable
    end

endmodule