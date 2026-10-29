//-----------------------------------------------------------------------------
// Module: token_assembler.sv
// Description: Assembles 512-bit tokens from 16 × 32-bit AXI4 beats.
//              Used in source path: DDR read → token assembly → IP stream.
// Architectural Novelty:
//   - Shift-register based assembly (low latency)
//   - Early valid indication when final beat arrives
//   - Backpressure handling with beat stall
// Pipeline: 1 cycle (combinational assembly path)
//-----------------------------------------------------------------------------
module token_assembler #(
    parameter int unsigned AXI_DATA_WIDTH = 32,
    parameter int unsigned TOKEN_WIDTH    = 512,
    parameter int unsigned BEATS_PER_TOKEN = 16
)(
    input  logic                       clk,
    input  logic                       rst_n,

    // AXI4 Read Data Interface (input)
    input  logic [AXI_DATA_WIDTH-1:0]  rdata,
    input  logic [1:0]                 rresp,
    input  logic                       rlast,
    input  logic                       rvalid,
    output logic                       rready,

    // Token Output Interface
    output logic                       token_valid,
    output logic [TOKEN_WIDTH-1:0]      token_data,
    output logic                       token_last,
    input  logic                       token_ready,

    // Control
    input  logic                       assemble_start,
    output logic                       assemble_done,
    output logic                       error            // SLVERR detected
);

    import dma_e_rodnet_pkg::*;

    //==========================================================
    // Internal State
    //==========================================================
    assemble_state_e state;
    assemble_state_e next_state;

    // Beat counter
    logic [$clog2(BEATS_PER_TOKEN):0] beat_count;

    // Shift register for assembly (16 × 32-bit = 512-bit)
    logic [TOKEN_WIDTH-1:0] token_shift;
    logic [TOKEN_WIDTH-1:0] token_shift_next;

    //==========================================================
    // Beat Count Management
    //==========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            beat_count <= '0;
        end else begin
            if (assemble_start) begin
                beat_count <= '0;
            end else if (rvalid && rready) begin
                beat_count <= beat_count + 1'b1;
            end
        end
    end

    //==========================================================
    // Shift Register Assembly
    //==========================================================
    // Shift in new data, LSB first (beat 0 = channels 0-3)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            token_shift <= '0;
        end else if (rvalid && rready) begin
            // Shift left by AXI_DATA_WIDTH, insert new beat at LSB
            token_shift <= {token_shift[TOKEN_WIDTH-AXI_DATA_WIDTH-1:0], rdata};
        end
    end

    //==========================================================
    // State Machine - Sequential
    //==========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ASSEMBLE_IDLE;
        end else begin
            state <= next_state;
        end
    end

    //==========================================================
    // State Machine - Combinational
    //==========================================================
    always_comb begin
        next_state = state;

        case (state)
            ASSEMBLE_IDLE: begin
                if (assemble_start) begin
                    next_state = ASSEMBLE_FILL;
                end
            end

            ASSEMBLE_FILL: begin
                if (rvalid && rready) begin
                    if (beat_count == BEATS_PER_TOKEN - 1) begin
                        // Last beat received
                        if (token_ready) begin
                            next_state = ASSEMBLE_IDLE;
                        end else begin
                            next_state = ASSEMBLE_READY;
                        end
                    end
                end
            end

            ASSEMBLE_READY: begin
                if (token_ready) begin
                    next_state = ASSEMBLE_IDLE;
                end
            end

            default: begin
                next_state = ASSEMBLE_IDLE;
            end
        endcase
    end

    //==========================================================
    // Output Logic
    //==========================================================
    assign rready = (state == ASSEMBLE_FILL) && (beat_count < BEATS_PER_TOKEN);

    assign token_valid = (state == ASSEMBLE_READY) ||
                         (state == ASSEMBLE_FILL && rvalid && rready && beat_count == BEATS_PER_TOKEN - 1);

    assign token_data = token_shift;
    assign token_last = 1'b0; // Set by higher-level controller

    assign assemble_done = (state == ASSEMBLE_FILL) && rvalid && rready && (beat_count == BEATS_PER_TOKEN - 1);

    //==========================================================
    // Error Detection
    //==========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            error <= 1'b0;
        end else if (assemble_start) begin
            error <= 1'b0;
        end else if (rvalid && rresp == AXI_RESP_SLVERR) begin
            error <= 1'b1;
        end
    end

    //==========================================================
    // Assertions
    //==========================================================
    `ifdef FORMAL
        // Verify beat count wraps correctly
        assert property (@(posedge clk) disable iff (!rst_n)
            assemble_done |-> beat_count == BEATS_PER_TOKEN - 1);

        // Verify token_valid only asserted when data is ready
        assert property (@(posedge clk) disable iff (!rst_n)
            token_valid && !token_ready |-> state == ASSEMBLE_READY);
    `endif

endmodule : token_assembler

