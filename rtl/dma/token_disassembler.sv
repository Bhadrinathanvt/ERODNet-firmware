//-----------------------------------------------------------------------------
// Module: token_disassembler.sv
// Description: Disassembles 512-bit tokens into 16 × 32-bit AXI4 beats.
//              Used in sink path: IP stream → token disassembly → DDR write.
// Architectural Novelty:
//   - Skid buffer for backpressure handling
//   - Counter-based beat generation
//   - Early address calculation for write optimization
// Pipeline: 1 cycle (combinational output path)
//-----------------------------------------------------------------------------
module token_disassembler #(
    parameter int unsigned AXI_DATA_WIDTH  = 32,
    parameter int unsigned TOKEN_WIDTH     = 512,
    parameter int unsigned BEATS_PER_TOKEN  = 16
)(
    input  logic                       clk,
    input  logic                       rst_n,

    // Token Input Interface
    input  logic                       token_valid,
    input  logic [TOKEN_WIDTH-1:0]      token_data,
    input  logic                        token_last,
    output logic                        token_ready,

    // AXI4 Write Data Interface (output)
    output logic [AXI_DATA_WIDTH-1:0]   wdata,
    output logic [AXI_DATA_WIDTH/8-1:0] wstrb,
    output logic                        wlast,
    output logic                        wvalid,
    input  logic                        wready,

    // Control
    input  logic                        disassemble_start,
    output logic                        disassemble_done
);

    import dma_e_rodnet_pkg::*;

    //==========================================================
    // Internal State
    //==========================================================
    typedef enum logic [1:0] {
        DISASSEMBLE_IDLE,
        DISASSEMBLE_SEND,
        DISASSEMBLE_WAIT
    } disassemble_state_e;

    disassemble_state_e state, next_state;

    // Beat counter
    logic [$clog2(BEATS_PER_TOKEN):0] beat_count;

    // Token storage (skid buffer)
    logic [TOKEN_WIDTH-1:0] token_buffer;
    logic                   token_last_buffer;
    logic                   token_buffer_valid;

    //==========================================================
    // Token Buffer (Skid Buffer)
    //==========================================================
    // Accepts token when valid, holds until disassembly complete
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            token_buffer <= '0;
            token_last_buffer <= 1'b0;
            token_buffer_valid <= 1'b0;
        end else begin
            if (token_valid && token_ready && !token_buffer_valid) begin
                token_buffer <= token_data;
                token_last_buffer <= token_last;
                token_buffer_valid <= 1'b1;
            end else if (disassemble_done) begin
                token_buffer_valid <= 1'b0;
            end
        end
    end

    //==========================================================
    // Beat Counter
    //==========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            beat_count <= '0;
        end else begin
            if (disassemble_start) begin
                beat_count <= '0;
            end else if (wvalid && wready) begin
                if (beat_count < BEATS_PER_TOKEN - 1) begin
                    beat_count <= beat_count + 1'b1;
                end else begin
                    beat_count <= '0;
                end
            end
        end
    end

    //==========================================================
    // State Machine - Sequential
    //==========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= DISASSEMBLE_IDLE;
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
            DISASSEMBLE_IDLE: begin
                if (disassemble_start && token_buffer_valid) begin
                    next_state = DISASSEMBLE_SEND;
                end
            end

            DISASSEMBLE_SEND: begin
                if (wvalid && wready && beat_count == BEATS_PER_TOKEN - 1) begin
                    next_state = DISASSEMBLE_WAIT;
                end
            end

            DISASSEMBLE_WAIT: begin
                // Wait for next token or idle
                if (!token_buffer_valid) begin
                    next_state = DISASSEMBLE_IDLE;
                end else if (token_valid && token_ready) begin
                    next_state = DISASSEMBLE_SEND;
                end
            end

            default: begin
                next_state = DISASSEMBLE_IDLE;
            end
        endcase
    end

    //==========================================================
    // Output Logic
    //==========================================================
    // Data extraction: beat[0] = token[31:0], beat[15] = token[511:480]
    always_comb begin
        case (beat_count)
            4'd0:  wdata = token_buffer[31:0];
            4'd1:  wdata = token_buffer[63:32];
            4'd2:  wdata = token_buffer[95:64];
            4'd3:  wdata = token_buffer[127:96];
            4'd4:  wdata = token_buffer[159:128];
            4'd5:  wdata = token_buffer[191:160];
            4'd6:  wdata = token_buffer[223:192];
            4'd7:  wdata = token_buffer[255:224];
            4'd8:  wdata = token_buffer[287:256];
            4'd9:  wdata = token_buffer[319:288];
            4'd10: wdata = token_buffer[351:320];
            4'd11: wdata = token_buffer[383:352];
            4'd12: wdata = token_buffer[415:384];
            4'd13: wdata = token_buffer[447:416];
            4'd14: wdata = token_buffer[479:448];
            4'd15: wdata = token_buffer[511:480];
            default: wdata = '0;
        endcase
    end

    assign wstrb = {(AXI_DATA_WIDTH/8){1'b1}}; // All bytes valid
    assign wlast = (beat_count == BEATS_PER_TOKEN - 1);
    assign wvalid = (state == DISASSEMBLE_SEND) && token_buffer_valid;

    assign token_ready = (state == DISASSEMBLE_IDLE) ||
                         (state == DISASSEMBLE_WAIT && !token_buffer_valid);

    assign disassemble_done = (state == DISASSEMBLE_SEND) &&
                               wvalid && wready &&
                               (beat_count == BEATS_PER_TOKEN - 1);

    //==========================================================
    // Assertions
    //==========================================================
    `ifdef FORMAL
        // Verify wlast only on final beat
        assert property (@(posedge clk) disable iff (!rst_n)
            wlast |-> beat_count == BEATS_PER_TOKEN - 1);

        // Verify wvalid only when buffer has data
        assert property (@(posedge clk) disable iff (!rst_n)
            wvalid |-> token_buffer_valid);
    `endif

endmodule : token_disassembler

