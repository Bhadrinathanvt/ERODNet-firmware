//-----------------------------------------------------------------------------
// Module: dma_streaming_intf.sv
// Description: Streaming interface for DMA ↔ e_rodnet token transfer.
//              512-bit wide valid/ready handshake with last flag.
//-----------------------------------------------------------------------------
`ifndef DMA_STREAMING_INTF_SV
`define DMA_STREAMING_INTF_SV

interface dma_streaming_intf #(
    parameter int unsigned DATA_WIDTH = 512
)(
    input logic clk,
    input logic rst_n
);

    //==========================================================
    // Token Signals
    //==========================================================
    logic                   valid;    // Data is valid
    logic [DATA_WIDTH-1:0]  data;     // Full token: {ch63,...,ch0}
    logic                   last;     // Last token of frame
    logic                   ready;    // Receiver can accept

    //==========================================================
    // Master Modport (DMA driving source path)
    //==========================================================
    modport master (
        output valid,
        output data,
        output last,
        input  ready
    );

    //==========================================================
    // Slave Modport (IP receiving source path)
    //==========================================================
    modport slave (
        input  valid,
        input  data,
        input  last,
        output ready
    );

    //==========================================================
    // Monitor Modport (for testbench observation)
    //==========================================================
    modport monitor (
        input valid,
        input data,
        input last,
        input ready
    );

    //==========================================================
    // Clocking Block (for synchronous assertions)
    //==========================================================
    clocking cb @(posedge clk);
        default input #1step output #0;
        input valid, data, last, ready;
    endclocking

    //==========================================================
    // Assertions (Protocol Checks)
    //==========================================================
    // Valid must be stable when ready is low
    // Ready can change independently

    // Task for simple transfer
    task automatic send_token(
        input logic [DATA_WIDTH-1:0] token_data,
        input logic                  token_last
    );
        begin
            @(posedge clk);
            data  = token_data;
            last  = token_last;
            valid = 1'b1;
            while (!ready) begin
                @(posedge clk);
            end
            @(posedge clk);
            valid = 1'b0;
        end
    endtask

    // Task for receiving token
    task automatic receive_token(
        output logic [DATA_WIDTH-1:0] token_data,
        output logic                  token_last
    );
        begin
            ready = 1'b1;
            while (!valid) begin
                @(posedge clk);
            end
            token_data = data;
            token_last = last;
            @(posedge clk);
            ready = 1'b0;
        end
    endtask

endinterface : dma_streaming_intf

`endif // DMA_STREAMING_INTF_SV

