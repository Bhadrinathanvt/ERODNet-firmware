// =============================================================================
// skid_buffer.sv
//
// Parameterized 1-deep AXI-Stream skid buffer for inter-stage backpressure.
//
// Provides a single-entry buffer that captures producer data when the consumer
// deasserts ready on the same cycle the producer asserts valid. This prevents
// data loss at pipeline boundaries where backpressure can arrive unexpectedly.
//
// Protocol: AXI-Stream valid/ready handshake.
//   - A transfer occurs when valid_out && out_ready on the same clock edge.
//   - valid_out must remain asserted until out_ready is seen (AXI-Stream stability).
//   - ready_in deasserts when the skid buffer is occupied and consumer is not ready.
//
// Architecture:
//   ┌──────────┐     ┌──────────┐     ┌──────────┐
//   │ data_in  │────▶│ skid_reg │────▶│ out_reg  │────▶ data_out
//   │ valid_in │────▶│ skid_v   │     │ out_v    │────▶ valid_out
//   │          │     └──────────┘     └──────────┘     │
//   │          │                              ▲        │
//   │          │◀─── ready_in ───────────────┘  out_ready
//   └──────────┘
//
//   When consumer is ready (out_ready=1): data flows through directly.
//   When consumer stalls (out_ready=0) and valid_in=1: data is captured in skid_reg.
//   ready_in = !skid_valid || (skid_valid && out_ready)
// =============================================================================

module skid_buffer #(
    parameter int DATA_WIDTH = pkg_config::DATA_WIDTH,
    parameter bit INCLUDE_LAST = 1   // Set to 1 if last/frame-boundary signal is needed
)(
    input  logic                          clk,
    input  logic                          rst_n,

    // Slave (input) side — producer interface
    input  logic [DATA_WIDTH-1:0]         data_in,
    input  logic                          valid_in,
    output logic                          ready_in,
    input  logic                          last_in,    // Frame boundary (optional)

    // Master (output) side — consumer interface
    output logic [DATA_WIDTH-1:0]         data_out,
    output logic                          valid_out,
    input  logic                          out_ready,
    output logic                          last_out    // Frame boundary (optional)
);

    // =========================================================================
    // Skid buffer registers
    // =========================================================================
    logic [DATA_WIDTH-1:0]    skid_data;
    logic                    skid_valid;
    logic                    skid_last;

    // Output hold register — holds valid_out asserted until out_ready is seen
    logic [DATA_WIDTH-1:0]   out_data_r;
    logic                    out_valid_r;
    logic                    out_last_r;

    // =========================================================================
    // Transfer indicators
    // =========================================================================
    logic input_transfer;   // Producer → skid buffer transfer
    logic output_transfer;  // Skid/out_reg → consumer transfer

    assign input_transfer  = valid_in && ready_in;
    assign output_transfer = valid_out && out_ready;

    // =========================================================================
    // Storage update — one skid slot + the registered output slot.
    //
    // Each cycle, if the output slot is free (empty or draining) it is refilled
    // from the skid slot first, otherwise directly from the input. If the
    // output is stalled, an incoming beat is captured into the skid slot.
    //
    // A single always_ff (instead of two) avoids the previous bug where a
    // direct pass-through cycle latched the same input into BOTH the skid slot
    // and the output register, duplicating one beat (1 input -> 2 outputs).
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            skid_valid  <= 1'b0;
            skid_data   <= '0;
            skid_last   <= 1'b0;
            out_valid_r <= 1'b0;
            out_data_r  <= '0;
            out_last_r  <= 1'b0;
        end else begin
            if (!out_valid_r || out_ready) begin
                // Output slot free this cycle: refill from skid, else input.
                if (skid_valid) begin
                    out_valid_r <= 1'b1;
                    out_data_r  <= skid_data;
                    out_last_r  <= skid_last;
                    skid_valid  <= 1'b0;
                end else if (input_transfer) begin
                    out_valid_r <= 1'b1;
                    out_data_r  <= data_in;
                    out_last_r  <= last_in;
                end else begin
                    out_valid_r <= 1'b0;
                end
            end else begin
                // Output stalled (valid_out && !out_ready): hold it stable,
                // and capture an incoming beat into the skid slot.
                if (input_transfer) begin
                    skid_valid <= 1'b1;
                    skid_data  <= data_in;
                    skid_last  <= last_in;
                end
            end
        end
    end

    // =========================================================================
    // Output assignments
    // =========================================================================
    assign data_out  = out_data_r;
    assign valid_out = out_valid_r;
    assign last_out  = out_last_r;

    // =========================================================================
    // Ready logic — accept input whenever the skid slot is free. This fully
    // registers the ready path (the purpose of the skid buffer).
    // =========================================================================
    assign ready_in = !skid_valid;

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    SKID_VALID_STABLE: assert property(@(posedge clk) disable iff (!rst_n)
        (valid_out && !out_ready) |=> valid_out)
        else $error("skid_buffer: valid_out deasserted before out_ready");

    SKID_DATA_STABLE: assert property(@(posedge clk) disable iff (!rst_n)
        (valid_out && !out_ready) |=> $stable(data_out))
        else $error("skid_buffer: data_out changed while valid_out asserted without out_ready");

    SKID_LAST_STABLE: assert property(@(posedge clk) disable iff (!rst_n)
        (valid_out && !out_ready) |=> $stable(last_out))
        else $error("skid_buffer: last_out changed while valid_out asserted without out_ready");

    SKID_NO_OVERFLOW: assert property(@(posedge clk) disable iff (!rst_n)
        !(input_transfer && skid_valid))
        else $error("skid_buffer: overflow — input accepted with skid slot full");

    SKID_READY_IMPLIES_SPACE: assert property(@(posedge clk) disable iff (!rst_n)
        ready_in |-> !skid_valid)
        else $error("skid_buffer: ready_in asserted with skid slot full");

    cover property(@(posedge clk) disable iff (!rst_n) valid_in && ready_in);
    cover property(@(posedge clk) disable iff (!rst_n) valid_out && !out_ready);
    cover property(@(posedge clk) disable iff (!rst_n) input_transfer && output_transfer);
    // synthesis translate_on

endmodule
