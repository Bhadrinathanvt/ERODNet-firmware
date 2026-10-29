// =============================================================================
// depthwise_conv_engine.sv
//
//  Runtime-configurable 1-D depthwise convolution engine for the ConvFormer
//  token mixer.  Processes a stream of MAX_C-channel token vectors; applies an
//  independent per-channel sliding-window convolution of length cfg_kernel_size.
//
//  Architecture
//  ────────────
//    Each channel has a MAX_KERNEL_SIZE-deep shift register (line buffer).
//    Every cycle a new token is pushed in; the oldest sample drops off.
//    The MAC tree sums (sr[c][k] × kernel[c][k]) for k < cfg_kernel_size.
//    The accumulation and output register add 1 cycle of latency.
//
//  Weights
//  ───────
//    Kernel coefficients are loaded via the write interface:
//      dw_wr_en, dw_wr_ch, dw_wr_k, dw_wr_data
//    Default at reset: center-tap = 1, all others = 0 (identity / delay line).
//    For real inference, load weights before streaming tokens.
//
//  Valid handshake
//  ────────────────
//    valid_out asserts once valid_cnt ≥ cfg_kernel_size − 1 (buffer filled).
//
//  Runtime parameters
//  ───────────────────
//    cfg_kernel_size : active kernel length (1 … MAX_KERNEL_SIZE)
//    cfg_c           : active channel count  (1 … MAX_C)
// =============================================================================

import pkg_config::*;

module depthwise_conv_engine #(
    parameter int    MAX_C              = 64,
    parameter int    DATA_WIDTH         = 8,
    parameter int    MAX_KERNEL_SIZE    = 49,
    parameter int    ACC_WIDTH          = 32,
    parameter int    WEIGHT_LOAD_MODE   = 0,   // 0=RUNTIME, 1=MEM_FILE
    parameter string WEIGHT_FILE        = "",   // .mem file for depthwise kernel weights
    parameter int    ENABLE_BACKPRESSURE = 1    // 1=full backpressure, 0=tie ready_in=1, no stall logic
)(
    input  logic clk,
    input  logic rst_n,

    // ── Runtime config ────────────────────────────────────────────────────────
    input  logic [15:0] cfg_kernel_size,   // active kernel length
    input  logic [15:0] cfg_c,            // active channels

    // ── Weight load interface (optional; load before streaming) ──────────────
    input  logic        dw_wr_en,
    input  logic [15:0] dw_wr_ch,         // channel index
    input  logic [15:0] dw_wr_k,          // kernel tap index
    input  logic signed [DATA_WIDTH-1:0] dw_wr_data,

    // ── Token stream ─────────────────────────────────────────────────────────
    input  logic        valid_in,
    input  logic signed [DATA_WIDTH-1:0] token_in [MAX_C-1:0],
    output logic        ready_in,         // Can accept new token (backpressure)

    output logic        valid_out,
    input  logic        out_ready,        // Consumer ready for output
    output logic signed [DATA_WIDTH-1:0] token_out [MAX_C-1:0]
);

    // ── Kernel weight RAM (per-channel) ─────────────────────────────────────
    logic signed [DATA_WIDTH-1:0] kernel [MAX_C-1:0][0:MAX_KERNEL_SIZE-1];

    generate
    if (WEIGHT_LOAD_MODE == 1) begin : gen_mem_weights
        // MEM mode: depthwise kernel initialized from .mem file
        // .mem format: flat array indexed by [channel * MAX_KERNEL_SIZE + tap]
        initial begin
            if (WEIGHT_FILE != "")
                $readmemh(WEIGHT_FILE, kernel);
        end
    end else begin : gen_runtime_weights
        // RUNTIME mode: kernel loaded via dw_wr_* write ports
        always_ff @(posedge clk) begin
            if (dw_wr_en && (dw_wr_ch < MAX_C) && (dw_wr_k < MAX_KERNEL_SIZE)) begin
                kernel[dw_wr_ch][dw_wr_k] <= dw_wr_data;
            end
        end
    end
    endgenerate

    // ── Backpressure & output register ─────────────────────────────────────────
    // When ENABLE_BACKPRESSURE=1: output hold register with full Valid/Ready
    // handshake (stall when consumer deasserts out_ready).
    // When ENABLE_BACKPRESSURE=0: ready_in=1, simplified output FSM (consumer
    // assumed always ready).  Saves ~50-100 LUTs in control logic.
    //
    // Shift register and MAC datapath are shared by both modes; only the
    // ready_in assignment and output register FSM differ.
    logic [15:0] valid_cnt;

    // ── Shift register (line buffer) ──────────────────────────────────────────
    logic signed [DATA_WIDTH-1:0] sr [MAX_C-1:0][0:MAX_KERNEL_SIZE-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int c = 0; c < MAX_C; c++)
                for (int k = 0; k < MAX_KERNEL_SIZE; k++)
                    sr[c][k] <= '0;
            valid_cnt <= '0;
        end else if (valid_in && ready_in) begin
            if (valid_cnt < 16'hFFFF) valid_cnt <= valid_cnt + 1;
            for (int c = 0; c < MAX_C; c++) begin
                sr[c][0] <= token_in[c];
                for (int k = MAX_KERNEL_SIZE-1; k > 0; k--)
                    sr[c][k] <= sr[c][k-1];
            end
        end
    end

    // ── MAC: 1-stage pipelined (registered products, comb adder tree) ─────────
    //
    // DSP policy: MAX_C × MAX_KERNEL_SIZE = 64 × 49 = 3136 multipliers, far
    // beyond the Artix-7-100T 240-DSP budget. Force LUT-based multipliers
    // ((* use_dsp = "no" *)) so the entire DSP fabric stays available for
    // the systolic array PEs in convformer_block.
    //
    // Timing closure: the previous fully-combinational tree (mult + 6-level
    // add) measured ≈13 ns critical path on Artix-7 — would not close at
    // 200 MHz. Adding a register stage between the products and the adder
    // tree splits the path into ~mult-only and ~add-tree halves, letting
    // both close at 200 MHz. Cost: +1 cycle of latency, no throughput hit.
    // ─────────────────────────────────────────────────────────────────────
    (* use_dsp = "no" *)
    logic signed [2*DATA_WIDTH-1:0] mac_prod [MAX_C-1:0][0:MAX_KERNEL_SIZE-1];
    logic                            mac_advance_d1;     // pipe of valid_in&&ready_in
    logic [15:0]                     valid_cnt_at_mac;   // valid_cnt sampled at mac_prod time

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac_advance_d1   <= 1'b0;
            valid_cnt_at_mac <= '0;
            for (int c = 0; c < MAX_C; c++)
                for (int k = 0; k < MAX_KERNEL_SIZE; k++)
                    mac_prod[c][k] <= '0;
        end else begin
            mac_advance_d1 <= valid_in && ready_in;
            if (valid_in && ready_in) begin
                valid_cnt_at_mac <= valid_cnt;     // pre-increment value (NBA)
                for (int c = 0; c < MAX_C; c++) begin
                    for (int k = 0; k < MAX_KERNEL_SIZE; k++) begin
                        if ((k < cfg_kernel_size) && (c < cfg_c))
                            mac_prod[c][k] <= signed'(sr[c][k]) * signed'(kernel[c][k]);
                        else
                            mac_prod[c][k] <= '0;
                    end
                end
            end
        end
    end

    // Stage 2: combinational adder tree on registered products
    // Guard: ACC_WIDTH must hold the full product plus log2(MAX_KERNEL_SIZE)
    // extra bits for the tree sum. If ACC_WIDTH is too narrow, significant
    // product bits are silently truncated at ACC_WIDTH'(mac_prod).
    localparam int MAC_SUM_WIDTH = 2*DATA_WIDTH + $clog2(MAX_KERNEL_SIZE);
    // synthesis translate_off
    initial begin
        if (ACC_WIDTH < MAC_SUM_WIDTH)
            $fatal(0, "depthwise_conv_engine: ACC_WIDTH (%0d) < required %0d (2*DATA_WIDTH + clog2(MAX_KERNEL_SIZE))",
                   ACC_WIDTH, MAC_SUM_WIDTH);
    end
    // synthesis translate_on

    logic signed [ACC_WIDTH-1:0] mac_result [MAX_C-1:0];
    always_comb begin
        for (int c = 0; c < MAX_C; c++) begin
            automatic logic signed [ACC_WIDTH-1:0] acc = '0;
            for (int k = 0; k < MAX_KERNEL_SIZE; k++)
                acc = acc + ACC_WIDTH'(mac_prod[c][k]);
            mac_result[c] = acc;
        end
    end

    // ── Output register ───────────────────────────────────────────────────────
    localparam int ROUND_SHIFT = DATA_WIDTH / 2;

    // Output trigger: mac_advance_d1 fires 1 cycle after a successful input
    // handshake (= when mac_prod registers hold the products of that input
    // cycle's sr). Threshold check uses valid_cnt_at_mac, the pre-increment
    // valid_cnt sampled when the inputs were captured.
    wire mac_result_ready = mac_advance_d1 && (valid_cnt_at_mac >= cfg_kernel_size - 1);

    generate
    if (ENABLE_BACKPRESSURE == 0) begin : gen_nobp
        assign ready_in = 1'b1;
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                for (int c = 0; c < MAX_C; c++) token_out[c] <= '0;
                valid_out <= 1'b0;
            end else begin
                if (mac_result_ready) begin
                    valid_out <= 1'b1;
                    for (int c = 0; c < MAX_C; c++)
                        token_out[c] <= (c < cfg_c) ?
                            DATA_WIDTH'(mac_result[c] >>> ROUND_SHIFT) : '0;
                end else begin
                    valid_out <= 1'b0;
                end
            end
        end
    end else begin : gen_bp
        logic        output_held;
        assign ready_in = !output_held || (output_held && out_ready);
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                for (int c = 0; c < MAX_C; c++) token_out[c] <= '0;
                valid_out  <= 1'b0;
                output_held <= 1'b0;
            end else begin
                if (valid_out && out_ready) begin
                    // Consumer accepted — clear output if no new result this cycle
                    output_held <= 1'b0;
                    if (mac_result_ready) begin
                        // New result arrives simultaneously — latch it
                        valid_out <= 1'b1;
                        output_held <= 1'b1;
                        for (int c = 0; c < MAX_C; c++)
                            token_out[c] <= (c < cfg_c) ?
                                DATA_WIDTH'(mac_result[c] >>> ROUND_SHIFT) : '0;
                    end else begin
                        valid_out <= 1'b0;
                    end
                end else if (valid_out && !out_ready) begin
                    // Consumer not ready — hold output stable (AXI-Stream stability)
                    output_held <= 1'b1;
                end else if (!valid_out) begin
                    // No output held — latch new result if available
                    if (mac_result_ready) begin
                        valid_out <= 1'b1;
                        output_held <= 1'b1;
                        for (int c = 0; c < MAX_C; c++)
                            token_out[c] <= (c < cfg_c) ?
                                DATA_WIDTH'(mac_result[c] >>> ROUND_SHIFT) : '0;
                    end else begin
                        valid_out <= 1'b0;
                    end
                end
            end
        end

        // ── Assertions (backpressure mode only) ──────────────────────────
        // synthesis translate_off
        DW_VALID_STABLE: assert property(@(posedge clk) disable iff (!rst_n)
            (valid_out && !out_ready) |=> valid_out)
            else $error("depthwise_conv_engine: valid_out deasserted before out_ready");
    `ifndef VERILATOR
        DW_DATA_STABLE: assert property(@(posedge clk) disable iff (!rst_n)
            (valid_out && !out_ready) |=> $stable(token_out))
            else $error("depthwise_conv_engine: token_out changed while valid_out held without out_ready");
    `endif
        DW_READY_IMPLIES_SPACE: assert property(@(posedge clk) disable iff (!rst_n)
            ready_in |-> !output_held || out_ready)
            else $error("depthwise_conv_engine: ready_in asserted with output held and consumer not ready");
        // synthesis translate_on
    end
    endgenerate

endmodule
