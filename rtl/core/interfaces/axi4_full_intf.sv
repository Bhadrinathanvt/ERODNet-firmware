// =============================================================================
// axi4_full_intf.sv
//
// AXI4-Full Interface for eRODNet Accelerator.
// Replaces AXI-Stream with AXI4-Full for Shakti compatibility.
//
// Features:
//   - Burst support for high-throughput data transfer
//   - Multiple outstanding transactions
//   - Backpressure via READY signals
//   - 512-bit data width for DDR efficiency
//   - Compatible with Shakti uncore DMA
//
// Protocol:
//   1. Master initiates write: AW channel (addr) → W channel (data) → B channel (response)
//   2. Master initiates read: AR channel (addr) → R channel (data + response)
//
// Integration with Shakti DMA:
//   - DMA reads from DDR → AXI4-Full write to accelerator input buffer
//   - Accelerator processes → AXI4-Full read by DMA → write to DDR
// =============================================================================

`ifndef AXI4_FULL_INTF_SV
`define AXI4_FULL_INTF_SV

package axi4_types_pkg;

    //==========================================================
    // AXI4 Burst Types
    //==========================================================
    typedef enum logic [1:0] {
        BURST_FIXED   = 2'b00,
        BURST_INCR    = 2'b01,
        BURST_WRAP    = 2'b10,
        BURST_RESERVED = 2'b11
    } axi4_burst_e;

    //==========================================================
    // AXI4 Response Codes
    //==========================================================
    typedef enum logic [1:0] {
        RESP_OKAY   = 2'b00,
        RESP_EXOKAY = 2'b01,
        RESP_SLVERR = 2'b10,
        RESP_DECERR = 2'b11
    } axi4_resp_e;

    //==========================================================
    // AXI4 Cache Attributes
    //==========================================================
    typedef enum logic [3:0] {
        CACHE_BUFFERABLE        = 4'b0001,
        CACHE_MODIFIABLE        = 4'b0010,
        CACHE_READ_ALLOCATE     = 4'b0100,
        CACHE_WRITE_ALLOCATE    = 4'b1000,
        CACHE_NORMAL_NON_CACHEABLE = 4'b0011,
        CACHE_DEVICE_NON_BUFFERABLE = 4'b0000
    } axi4_cache_e;

endpackage : axi4_types_pkg

// =============================================================================
// AXI4-Full Interface Definition
// =============================================================================

interface axi4_full_intf #(
    parameter int ADDR_WIDTH = 32,
    parameter int DATA_WIDTH = 512,
    parameter int ID_WIDTH   = 8,
    parameter int USER_WIDTH = 1,
    parameter int REGION_WIDTH = 4,
    parameter int QOS_WIDTH  = 4
)(
    input  logic aclk,
    input  logic aresetn
);

    import axi4_types_pkg::*;

    //==========================================================
    // AW Channel (Write Address)
    //==========================================================
    logic [ID_WIDTH-1:0]     awid;
    logic [ADDR_WIDTH-1:0]   awaddr;
    logic [7:0]              awlen;      // Burst length (1-256)
    logic [2:0]              awsize;     // Transfer size (log2 bytes)
    logic [1:0]              awburst;    // Burst type
    logic                    awlock;     // Exclusive access
    logic [3:0]              awcache;    // Cache attributes
    logic [2:0]              awprot;     // Protection type
    logic [REGION_WIDTH-1:0] awregion;   // Region identifier
    logic [QOS_WIDTH-1:0]    awqos;      // QoS identifier
    logic [USER_WIDTH-1:0]   awuser;     // User sideband
    logic                    awvalid;
    logic                    awready;

    //==========================================================
    // W Channel (Write Data)
    //==========================================================
    logic [DATA_WIDTH-1:0]   wdata;
    logic [(DATA_WIDTH/8)-1:0] wstrb;    // Write strobe
    logic                    wlast;      // Last beat of burst
    logic [USER_WIDTH-1:0]   wuser;      // User sideband
    logic                    wvalid;
    logic                    wready;

    //==========================================================
    // B Channel (Write Response)
    //==========================================================
    logic [ID_WIDTH-1:0]     bid;
    logic [1:0]              bresp;      // Response code
    logic [USER_WIDTH-1:0]   buser;      // User sideband
    logic                    bvalid;
    logic                    bready;

    //==========================================================
    // AR Channel (Read Address)
    //==========================================================
    logic [ID_WIDTH-1:0]     arid;
    logic [ADDR_WIDTH-1:0]   araddr;
    logic [7:0]              arlen;      // Burst length (1-256)
    logic [2:0]              arsize;     // Transfer size (log2 bytes)
    logic [1:0]              arburst;    // Burst type
    logic                    arlock;     // Exclusive access
    logic [3:0]              arcache;    // Cache attributes
    logic [2:0]              arprot;     // Protection type
    logic [REGION_WIDTH-1:0] arregion;   // Region identifier
    logic [QOS_WIDTH-1:0]    arqos;      // QoS identifier
    logic [USER_WIDTH-1:0]   aruser;     // User sideband
    logic                    arvalid;
    logic                    arready;

    //==========================================================
    // R Channel (Read Data)
    //==========================================================
    logic [ID_WIDTH-1:0]     rid;
    logic [DATA_WIDTH-1:0]   rdata;
    logic [1:0]              rresp;      // Response code
    logic                    rlast;      // Last beat of burst
    logic [USER_WIDTH-1:0]   ruser;      // User sideband
    logic                    rvalid;
    logic                    rready;

    //==========================================================
    // Modport: Master (DMA or CPU)
    //==========================================================
    modport master (
        // AW Channel
        output awid, awaddr, awlen, awsize, awburst, awlock, awcache, awprot,
        output awregion, awqos, awuser, awvalid,
        input  awready,

        // W Channel
        output wdata, wstrb, wlast, wuser, wvalid,
        input  wready,

        // B Channel
        input  bid, bresp, buser, bvalid,
        output bready,

        // AR Channel
        output arid, araddr, arlen, arsize, arburst, arlock, arcache, arprot,
        output arregion, arqos, aruser, arvalid,
        input  arready,

        // R Channel
        input  rid, rdata, rresp, rlast, ruser, rvalid,
        output rready
    );

    //==========================================================
    // Modport: Slave (Accelerator)
    //==========================================================
    modport slave (
        // AW Channel
        input  awid, awaddr, awlen, awsize, awburst, awlock, awcache, awprot,
        input  awregion, awqos, awuser, awvalid,
        output awready,

        // W Channel
        input  wdata, wstrb, wlast, wuser, wvalid,
        output wready,

        // B Channel
        output bid, bresp, buser, bvalid,
        input  bready,

        // AR Channel
        input  arid, araddr, arlen, arsize, arburst, arlock, arcache, arprot,
        input  arregion, arqos, aruser, arvalid,
        output arready,

        // R Channel
        output rid, rdata, rresp, rlast, ruser, rvalid,
        input  rready
    );

    //==========================================================
    // Modport: Monitor (Debug/Verification)
    //==========================================================
    modport monitor (
        input awid, awaddr, awlen, awsize, awburst, awlock, awcache, awprot,
              awregion, awqos, awuser, awvalid, awready,
        input wdata, wstrb, wlast, wuser, wvalid, wready,
        input bid, bresp, buser, bvalid, bready,
        input arid, araddr, arlen, arsize, arburst, arlock, arcache, arprot,
              arregion, arqos, aruser, arvalid, arready,
        input rid, rdata, rresp, rlast, ruser, rvalid, rready
    );

    //==========================================================
    // SVA Assertions
    //==========================================================

    // AWVALID must stay asserted until AWREADY
    property aw_stable_until_ready;
        @(posedge aclk) disable iff (!aresetn)
        awvalid && !awready |=> awvalid;
    endproperty

    // WVALID must stay asserted until WREADY
    property w_stable_until_ready;
        @(posedge aclk) disable iff (!aresetn)
        wvalid && !wready |=> wvalid;
    endproperty

    // WLAST must only be asserted with WVALID
    property wlast_requires_wvalid;
        @(posedge aclk) disable iff (!aresetn)
        wlast |-> wvalid;
    endproperty

    // BVALID must stay asserted until BREADY
    property b_stable_until_ready;
        @(posedge aclk) disable iff (!aresetn)
        bvalid && !bready |=> bvalid;
    endproperty

    // ARVALID must stay asserted until ARREADY
    property ar_stable_until_ready;
        @(posedge aclk) disable iff (!aresetn)
        arvalid && !arready |=> arvalid;
    endproperty

    // RVALID must stay asserted until RREADY
    property r_stable_until_ready;
        @(posedge aclk) disable iff (!aresetn)
        rvalid && !rready |=> rvalid;
    endproperty

    // RLAST must only be asserted with RVALID
    property rlast_requires_rvalid;
        @(posedge aclk) disable iff (!aresetn)
        rlast |-> rvalid;
    endproperty

    // Generate assertions
    AW_STABLE: assert property (aw_stable_until_ready)
        else $error("axi4_full: AWVALID dropped before AWREADY");

    W_STABLE: assert property (w_stable_until_ready)
        else $error("axi4_full: WVALID dropped before WREADY");

    WLAST_VALID: assert property (wlast_requires_wvalid)
        else $error("axi4_full: WLAST without WVALID");

    B_STABLE: assert property (b_stable_until_ready)
        else $error("axi4_full: BVALID dropped before BREADY");

    AR_STABLE: assert property (ar_stable_until_ready)
        else $error("axi4_full: ARVALID dropped before ARREADY");

    R_STABLE: assert property (r_stable_until_ready)
        else $error("axi4_full: RVALID dropped before RREADY");

    RLAST_VALID: assert property (rlast_requires_rvalid)
        else $error("axi4_full: RLAST without RVALID");

endinterface : axi4_full_intf

`endif // AXI4_FULL_INTF_SV