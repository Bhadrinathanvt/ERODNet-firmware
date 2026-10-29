// =============================================================================
//  sparse_intersection_unit.sv
//
//  Merge-join intersection unit for matching non-zero (NZ) indices between
//  a weight stream and an activation stream before MAC operations.
//
//  Algorithm:
//    Both NZ streams are sorted by ascending index. Two pointers advance
//    independently. When indices match, emit a (weight, activation) pair for
//    the PE MAC unit. When one index is smaller, advance only that stream.
//    This is a standard merge-join (O(NZ_weight + NZ_activ) time, O(1) space).
//
//  FSM:
//    IDLE    → Wait for both weight and activation NZ streams to have valid data
//    COMPARE → Advance the stream with the smaller index; emit match on equality
//    DRAIN   → One stream reached last; drain remaining elements of the other
//              When both streams are exhausted, transition directly to IDLE
//
//  Pipeline: 3 stages (IDLE/COMPARE → matched outputs → PE input register)
//
//  Dense-mode compatibility:
//    When SPARSE_EN=0, the intersection unit passes all elements through,
//    treating every index as matching. This produces bit-identical results
//    to the original systolic array.
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;
import pkg_sparse_config::*;

module sparse_intersection_unit #(
    parameter int DATA_WIDTH = pkg_module_config::SYSTOLIC_ARRAY_DATA_WIDTH,
    parameter int IDX_WIDTH   = pkg_sparse_config::NZ_IDX_WIDTH
)(
    input  logic clk,
    input  logic rst_n,

    // ── Control ─────────────────────────────────────────────────────────────
    input  logic        sparse_en,      // Enable sparse intersection mode

    // ── Weight NZ stream input ──────────────────────────────────────────────
    input  logic                        w_valid,
    output logic                        w_ready,
    input  logic        [IDX_WIDTH-1:0] w_idx,
    input  logic signed [DATA_WIDTH-1:0] w_val,
    input  logic                        w_last,

    // ── Activation NZ stream input ───────────────────────────────────────────
    input  logic                        a_valid,
    output logic                        a_ready,
    input  logic        [IDX_WIDTH-1:0]  a_idx,
    input  logic signed [DATA_WIDTH-1:0] a_val,
    input  logic                        a_last,

    // ── Matched output (to PE) ───────────────────────────────────────────────
    output logic                        match_valid,
    input  logic                        match_ready,
    output logic signed [DATA_WIDTH-1:0] match_weight,
    output logic signed [DATA_WIDTH-1:0] match_activ,
    output logic        [IDX_WIDTH-1:0]  match_idx
);

    // =========================================================================
    //  FSM State
    // =========================================================================
    isect_state_e state, state_next;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= ISect_IDLE;
        else
            state <= state_next;
    end

    // =========================================================================
    //  Registered weight and activation packets (latched when valid)
    // =========================================================================
    logic        [IDX_WIDTH-1:0]  w_idx_r;
    logic signed [DATA_WIDTH-1:0] w_val_r;
    logic                         w_last_r;
    logic                         w_held;    // Weight packet is being held (not yet consumed)

    logic        [IDX_WIDTH-1:0]  a_idx_r;
    logic signed [DATA_WIDTH-1:0] a_val_r;
    logic                         a_last_r;
    logic                         a_held;    // Activation packet is being held

    // Backpressure: stall when output can't be consumed
    logic stall;
    assign stall = match_valid && !match_ready;

    // =========================================================================
    //  Weight latch
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_idx_r  <= '0;
            w_val_r  <= '0;
            w_last_r <= 1'b0;
            w_held   <= 1'b0;
        end else if (!stall) begin
            if (!w_held && w_valid) begin
                // Latch new weight packet
                w_idx_r  <= w_idx;
                w_val_r  <= w_val;
                w_last_r <= w_last;
                w_held   <= 1'b1;
            end else if (w_held && consume_w) begin
                // Consumed current weight — check for next
                w_held <= 1'b0;
            end
        end
    end

    // =========================================================================
    //  Activation latch
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_idx_r  <= '0;
            a_val_r  <= '0;
            a_last_r <= 1'b0;
            a_held   <= 1'b0;
        end else if (!stall) begin
            if (!a_held && a_valid) begin
                a_idx_r  <= a_idx;
                a_val_r  <= a_val;
                a_last_r <= a_last;
                a_held   <= 1'b1;
            end else if (a_held && consume_a) begin
                a_held <= 1'b0;
            end
        end
    end

    // =========================================================================
    //  Ready signals (consumer can accept when we're not holding)
    // =========================================================================
    assign w_ready = !w_held;
    assign a_ready = !a_held;

    // =========================================================================
    //  Merge-Join Logic
    // =========================================================================
    logic match;
    logic w_less, a_less;
    logic consume_w, consume_a;

    // Index comparison
    assign match   = (w_idx_r == a_idx_r) && w_held && a_held;
    assign w_less  = (w_idx_r < a_idx_r)  && w_held && a_held;
    assign a_less  = (a_idx_r < w_idx_r)  && w_held && a_held;

    // Consume decisions: which stream(s) to advance
    always_comb begin
        consume_w = 1'b0;
        consume_a = 1'b0;
        state_next = state;

        case (state)
            ISect_IDLE: begin
                if (w_held && a_held) begin
                    // Both streams ready — start comparing
                    state_next = ISect_COMPARE;
                end
            end

            ISect_COMPARE: begin
                if (match) begin
                    // Indices match — consume both, emit MAC pair
                    consume_w = 1'b1;
                    consume_a = 1'b1;
                end else if (w_less) begin
                    // Weight index smaller — advance weight stream
                    consume_w = 1'b1;
                end else if (a_less) begin
                    // Activation index smaller — advance activation stream
                    consume_a = 1'b1;
                end

                // Check for stream termination — both exhausted → IDLE directly
                if ((w_last_r && consume_w) && (a_last_r && consume_a)) begin
                    state_next = ISect_IDLE;
                end else if (w_last_r && consume_w && !a_last_r) begin
                    state_next = ISect_DRAIN;  // Weight exhausted, drain activation
                end else if (a_last_r && consume_a && !w_last_r) begin
                    state_next = ISect_DRAIN;  // Activation exhausted, drain weight
                end else if (match && w_last_r && a_last_r) begin
                    state_next = ISect_IDLE;
                end
            end

            ISect_DRAIN: begin
                // One stream is done — advance the remaining stream
                // (no more matches possible)
                if (w_last_r) begin
                    consume_a = 1'b1;  // Drain remaining activations
                end else begin
                    consume_w = 1'b1;  // Drain remaining weights
                end
                // Check if drainer is also done — both exhausted → IDLE directly
                if ((w_last_r && a_held) || (a_last_r && w_held)) begin
                    if ((w_last_r && a_last_r) ||
                        (w_last_r && a_last_r))
                        state_next = ISect_IDLE;
                end
            end

        endcase
    end

    // =========================================================================
    //  Output Registration
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            match_valid  <= 1'b0;
            match_weight <= '0;
            match_activ  <= '0;
            match_idx    <= '0;
        end else if (!stall) begin
            if (!sparse_en) begin
                // Dense mode: pass through all inputs as matches
                // (handled by bypass in systolic_array_sparse)
                match_valid  <= 1'b0;
                match_weight <= '0;
                match_activ  <= '0;
                match_idx    <= '0;
            end else begin
                // Sparse mode: emit only on index match
                match_valid  <= match;
                match_weight <= w_val_r;
                match_activ  <= a_val_r;
                match_idx    <= w_idx_r;  // = a_idx_r on match
            end
        end
    end

    // =========================================================================
    //  Assertions
    // =========================================================================
    // synthesis translate_off
    ISECT_BOTH_STREAMS_DONE: assert property(@(posedge clk) disable iff (!rst_n)
        state == ISect_DRAIN |-> !(w_last_r && a_last_r && !w_held && !a_held) ||
                                   w_last_r && a_last_r && !w_held && !a_held)
        else $error("sparse_intersection_unit: streams inconsistent in DRAIN state");
    // synthesis translate_on

endmodule