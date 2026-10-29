//-----------------------------------------------------------------------------
// Module: axi4_intf.sv
// Description: AXI4 Full interface for Yamuna SoC (32-bit data, 32-bit address).
//              Used for DMA master port to DDR memory.
//-----------------------------------------------------------------------------
`ifndef AXI4_INTF_SV
`define AXI4_INTF_SV

interface axi4_intf #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 32,
    parameter int unsigned ID_WIDTH   = 4,
    parameter int unsigned LEN_WIDTH = 8
)(
    input logic clk,
    input logic rst_n
);

    //==========================================================
    // Write Address Channel
    //==========================================================
    logic [ID_WIDTH-1:0]    awid;
    logic [ADDR_WIDTH-1:0]  awaddr;
    logic [LEN_WIDTH-1:0]   awlen;      // Burst length - 1
    logic [2:0]             awsize;      // Beat size (010 = 4 bytes)
    logic [1:0]             awburst;     // Burst type (01 = INCR)
    logic                   awlock;
    logic [3:0]             awcache;
    logic [2:0]             awprot;
    logic [3:0]             awqos;
    logic                   awvalid;
    logic                   awready;

    //==========================================================
    // Write Data Channel
    //==========================================================
    logic [DATA_WIDTH-1:0]  wdata;
    logic [DATA_WIDTH/8-1:0] wstrb;
    logic                   wlast;
    logic                   wvalid;
    logic                   wready;

    //==========================================================
    // Write Response Channel
    //==========================================================
    logic [ID_WIDTH-1:0]    bid;
    logic [1:0]             bresp;
    logic                   bvalid;
    logic                   bready;

    //==========================================================
    // Read Address Channel
    //==========================================================
    logic [ID_WIDTH-1:0]    arid;
    logic [ADDR_WIDTH-1:0]  araddr;
    logic [LEN_WIDTH-1:0]   arlen;
    logic [2:0]             arsize;
    logic [1:0]             arburst;
    logic                   arlock;
    logic [3:0]             arcache;
    logic [2:0]             arprot;
    logic [3:0]             arqos;
    logic                   arvalid;
    logic                   arready;

    //==========================================================
    // Read Data Channel
    //==========================================================
    logic [ID_WIDTH-1:0]    rid;
    logic [DATA_WIDTH-1:0]  rdata;
    logic [1:0]             rresp;
    logic                   rlast;
    logic                   rvalid;
    logic                   wready;

    //==========================================================
    // Master Modport (DMA initiator)
    //==========================================================
    modport master (
        // Write Address
        output awid, awaddr, awlen, awsize, awburst, awlock,
               awcache, awprot, awqos, awvalid,
        input  awready,

        // Write Data
        output wdata, wstrb, wlast, wvalid,
        input  wready,

        // Write Response
        input  bid, bresp, bvalid,
        output bready,

        // Read Address
        output arid, araddr, arlen, arsize, arburst, arlock,
               arcache, arprot, arqos, arvalid,
        input  arready,

        // Read Data
        input  rid, rdata, rresp, rlast, rvalid,
        output rready
    );

    //==========================================================
    // Slave Modport (DDR controller / interconnect)
    //==========================================================
    modport slave (
        // Write Address
        input  awid, awaddr, awlen, awsize, awburst, awlock,
               awcache, awprot, awqos, awvalid,
        output awready,

        // Write Data
        input  wdata, wstrb, wlast, wvalid,
        output wready,

        // Write Response
        output bid, bresp, bvalid,
        input  bready,

        // Read Address
        input  arid, araddr, arlen, arsize, arburst, arlock,
               arcache, arprot, arqos, arvalid,
        output arready,

        // Read Data
        output rid, rdata, rresp, rlast, rvalid,
        input  rready
    );

    //==========================================================
    // Clocking Block
    //==========================================================
    clocking cb @(posedge clk);
        default input #1step output #0;
        // Write Address
        output awid, awaddr, awlen, awsize, awburst, awvalid;
        input  awready;
        // Write Data
        output wdata, wstrb, wlast, wvalid;
        input  wready;
        // Write Response
        input  bresp, bvalid;
        output bready;
        // Read Address
        output arid, araddr, arlen, arsize, arburst, arvalid;
        input  arready;
        // Read Data
        input  rdata, rresp, rlast, rvalid;
        output rready;
    endclocking

endinterface : axi4_intf

`endif // AXI4_INTF_SV

