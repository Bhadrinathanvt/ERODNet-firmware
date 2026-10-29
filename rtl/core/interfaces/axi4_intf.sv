// =============================================================================
// axi4_intf.sv
//
// AXI4 Full interface for e_rodnet — Yamuna SoC (32-bit data bus).
// ARM IHI0022E compliant: 5 channels + burst, WLAST, ID fields.
//
// Parameters:
//   ADDR_WIDTH  — byte address width (Yamuna: 32)
//   DATA_WIDTH  — bus data width (Yamuna: 32)
//   ID_WIDTH    — transaction ID width (recommend 4)
//
// Channels:
//   AW  Write Address  : AWVALID/AWREADY/AWADDR/AWLEN/AWSIZE/AWBURST/AWID/AWCACHE/AWPROT/AWLOCK/AWQOS
//   W   Write Data     : WVALID/WREADY/WDATA/WSTRB/WLAST
//   B   Write Response : BVALID/BREADY/BRESP/BID
//   AR  Read Address   : ARVALID/ARREADY/ARADDR/ARLEN/ARSIZE/ARBURST/ARID/ARCACHE/ARPROT/ARLOCK/ARQOS
//   R   Read Data      : RVALID/RREADY/RDATA/RRESP/RLAST/RID
// =============================================================================

`ifndef AXI4_INTF_SV
`define AXI4_INTF_SV

interface axi4_intf #(
    parameter int ADDR_WIDTH = 32,
    parameter int DATA_WIDTH = 32,
    parameter int ID_WIDTH   = 4
) (
    input logic aclk,
    input logic aresetn
);

    localparam int STRB_WIDTH = DATA_WIDTH / 8;

    // ── AW: Write Address Channel ─────────────────────────────────────────────
    logic                    awvalid;
    logic                    awready;
    logic [ADDR_WIDTH-1:0]   awaddr;
    logic [7:0]              awlen;    // burst length - 1 (0–255)
    logic [2:0]              awsize;   // 2^awsize bytes per beat
    logic [1:0]              awburst;  // FIXED=00 INCR=01 WRAP=10
    logic [ID_WIDTH-1:0]     awid;
    logic [3:0]              awcache;  // memory type attributes
    logic [2:0]              awprot;
    logic                    awlock;   // exclusive access
    logic [3:0]              awqos;

    // ── W: Write Data Channel ─────────────────────────────────────────────────
    logic                    wvalid;
    logic                    wready;
    logic [DATA_WIDTH-1:0]   wdata;
    logic [STRB_WIDTH-1:0]   wstrb;
    logic                    wlast;   // last beat of burst

    // ── B: Write Response Channel ─────────────────────────────────────────────
    logic                    bvalid;
    logic                    bready;
    logic [1:0]              bresp;   // OKAY=00 EXOKAY=01 SLVERR=10 DECERR=11
    logic [ID_WIDTH-1:0]     bid;

    // ── AR: Read Address Channel ──────────────────────────────────────────────
    logic                    arvalid;
    logic                    arready;
    logic [ADDR_WIDTH-1:0]   araddr;
    logic [7:0]              arlen;
    logic [2:0]              arsize;
    logic [1:0]              arburst;
    logic [ID_WIDTH-1:0]     arid;
    logic [3:0]              arcache;
    logic [2:0]              arprot;
    logic                    arlock;
    logic [3:0]              arqos;

    // ── R: Read Data Channel ──────────────────────────────────────────────────
    logic                    rvalid;
    logic                    rready;
    logic [DATA_WIDTH-1:0]   rdata;
    logic [1:0]              rresp;
    logic                    rlast;   // last beat of read burst
    logic [ID_WIDTH-1:0]     rid;

    // =========================================================================
    // Modports
    // =========================================================================

    // Master (DMA / initiator side)
    modport master (
        output awvalid, awaddr, awlen, awsize, awburst, awid, awcache, awprot, awlock, awqos,
        input  awready,
        output wvalid, wdata, wstrb, wlast,
        input  wready,
        input  bvalid, bresp, bid,
        output bready,
        output arvalid, araddr, arlen, arsize, arburst, arid, arcache, arprot, arlock, arqos,
        input  arready,
        input  rvalid, rdata, rresp, rlast, rid,
        output rready
    );

    // Slave (memory / target side)
    modport slave (
        input  awvalid, awaddr, awlen, awsize, awburst, awid, awcache, awprot, awlock, awqos,
        output awready,
        input  wvalid, wdata, wstrb, wlast,
        output wready,
        output bvalid, bresp, bid,
        input  bready,
        input  arvalid, araddr, arlen, arsize, arburst, arid, arcache, arprot, arlock, arqos,
        output arready,
        output rvalid, rdata, rresp, rlast, rid,
        input  rready
    );

    // Monitor (read-only)
    modport monitor (
        input awvalid, awready, awaddr, awlen, awsize, awburst, awid, awcache, awprot, awlock, awqos,
        input wvalid,  wready,  wdata,  wstrb,  wlast,
        input bvalid,  bready,  bresp,  bid,
        input arvalid, arready, araddr, arlen, arsize, arburst, arid, arcache, arprot, arlock, arqos,
        input rvalid,  rready,  rdata,  rresp,  rlast,  rid
    );

    // =========================================================================
    // SVA Protocol Assertions
    // =========================================================================

    // AW: AWVALID must not deassert before AWREADY
    property aw_valid_stable;
        @(posedge aclk) disable iff (!aresetn)
        (awvalid && !awready) |=> awvalid;
    endproperty

    // W: WVALID must not deassert before WREADY
    property w_valid_stable;
        @(posedge aclk) disable iff (!aresetn)
        (wvalid && !wready) |=> wvalid;
    endproperty

    // AR: ARVALID must not deassert before ARREADY
    property ar_valid_stable;
        @(posedge aclk) disable iff (!aresetn)
        (arvalid && !arready) |=> arvalid;
    endproperty

    // AW address/control must remain stable while AWVALID && !AWREADY
    property aw_ctrl_stable;
        @(posedge aclk) disable iff (!aresetn)
        (awvalid && !awready) |=> (awaddr == $past(awaddr)) &&
                                   (awlen  == $past(awlen))  &&
                                   (awsize == $past(awsize)) &&
                                   (awburst == $past(awburst));
    endproperty

    // AR address/control must remain stable while ARVALID && !ARREADY
    property ar_ctrl_stable;
        @(posedge aclk) disable iff (!aresetn)
        (arvalid && !arready) |=> (araddr == $past(araddr)) &&
                                   (arlen  == $past(arlen))  &&
                                   (arsize == $past(arsize)) &&
                                   (arburst == $past(arburst));
    endproperty

    // W data must remain stable while WVALID && !WREADY
    property w_data_stable;
        @(posedge aclk) disable iff (!aresetn)
        (wvalid && !wready) |=> (wdata == $past(wdata)) &&
                                  (wstrb == $past(wstrb)) &&
                                  (wlast == $past(wlast));
    endproperty

    // B response must remain stable while BVALID && !BREADY
    property b_resp_stable;
        @(posedge aclk) disable iff (!aresetn)
        (bvalid && !bready) |=> bvalid && (bresp == $past(bresp));
    endproperty

    // R response must remain stable while RVALID && !RREADY
    property r_resp_stable;
        @(posedge aclk) disable iff (!aresetn)
        (rvalid && !rready) |=> rvalid && (rdata == $past(rdata)) &&
                                           (rresp == $past(rresp)) &&
                                           (rlast == $past(rlast));
    endproperty

    // AWBURST must be FIXED, INCR, or WRAP (RESERVED=2'b11 illegal)
    property aw_burst_legal;
        @(posedge aclk) disable iff (!aresetn)
        awvalid |-> (awburst != 2'b11);
    endproperty

    // AWSIZE: 2^awsize must not exceed DATA_WIDTH/8
    property aw_size_legal;
        @(posedge aclk) disable iff (!aresetn)
        awvalid |-> (awsize <= $clog2(DATA_WIDTH/8));
    endproperty

    // WLAST must be asserted on the last beat of each burst
    // (tracked externally by counting beats; assertion is informational)
    property bresp_legal;
        @(posedge aclk) disable iff (!aresetn)
        bvalid |-> (bresp == 2'b00 || bresp == 2'b01 || bresp == 2'b10 || bresp == 2'b11);
    endproperty

    AXI4_AW_VALID_STABLE : assert property (aw_valid_stable)
        else $error("AXI4 violation: AWVALID dropped before AWREADY");
    AXI4_W_VALID_STABLE  : assert property (w_valid_stable)
        else $error("AXI4 violation: WVALID dropped before WREADY");
    AXI4_AR_VALID_STABLE : assert property (ar_valid_stable)
        else $error("AXI4 violation: ARVALID dropped before ARREADY");
    AXI4_AW_CTRL_STABLE  : assert property (aw_ctrl_stable)
        else $error("AXI4 violation: AW control changed while AWVALID && !AWREADY");
    AXI4_AR_CTRL_STABLE  : assert property (ar_ctrl_stable)
        else $error("AXI4 violation: AR control changed while ARVALID && !ARREADY");
    AXI4_W_DATA_STABLE   : assert property (w_data_stable)
        else $error("AXI4 violation: WDATA/WSTRB/WLAST changed while WVALID && !WREADY");
    AXI4_B_RESP_STABLE   : assert property (b_resp_stable)
        else $error("AXI4 violation: BRESP changed while BVALID && !BREADY");
    AXI4_R_RESP_STABLE   : assert property (r_resp_stable)
        else $error("AXI4 violation: RDATA/RRESP changed while RVALID && !RREADY");
    AXI4_AW_BURST_LEGAL  : assert property (aw_burst_legal)
        else $error("AXI4 violation: AWBURST=2'b11 is reserved");
    AXI4_AW_SIZE_LEGAL   : assert property (aw_size_legal)
        else $error("AXI4 violation: AWSIZE exceeds bus data width");

    // =========================================================================
    // Functional Coverage
    // =========================================================================
    covergroup axi4_cg @(posedge aclk);
        cp_aw_hs   : coverpoint {awvalid, awready} { bins hs = {2'b11}; bins idle = {2'b00}; }
        cp_w_hs    : coverpoint {wvalid, wready}   { bins hs = {2'b11}; bins idle = {2'b00}; }
        cp_b_hs    : coverpoint {bvalid, bready}   { bins hs = {2'b11}; }
        cp_ar_hs   : coverpoint {arvalid, arready} { bins hs = {2'b11}; bins idle = {2'b00}; }
        cp_r_hs    : coverpoint {rvalid, rready}   { bins hs = {2'b11}; }
        cp_awburst : coverpoint awburst { bins incr = {2'b01}; bins fixed = {2'b00}; bins wrap = {2'b10}; }
        cp_awlen   : coverpoint awlen {
            bins single  = {8'h00};
            bins token16 = {8'h0F};
            bins burst256= {8'hFF};
            bins other   = default;
        }
        cp_wlast   : coverpoint wlast iff (wvalid & wready) { bins last = {1'b1}; bins not_last = {1'b0}; }
        cp_bresp   : coverpoint bresp { bins okay = {2'b00}; bins slverr = {2'b10}; }
        cp_rresp   : coverpoint rresp { bins okay = {2'b00}; bins slverr = {2'b10}; }
        cx_rw      : cross cp_aw_hs, cp_ar_hs;
    endgroup

    axi4_cg axi4_cg_inst = new();

endinterface

`endif // AXI4_INTF_SV
