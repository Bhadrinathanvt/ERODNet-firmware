// =============================================================================
// csr_intf.sv
//
// AXI4-Lite Control/Status Register (CSR) interface with shadow registers.
// Compliant with ARM IHI0022E (AXI4-Lite protocol specification).
//
// Features:
//   - Full AXI4-Lite (5-channel: AW, W, B, AR, R)
//   - 32-bit data bus, parameterizable address width
//   - Shadow registers for glitch-free config updates (apply on cfg_update)
//   - Optional dual-clock domain: datapath clk vs. config_clk (2FF CDC)
//   - Write-strobe (WSTRB) byte-lane masking supported
//
// AXI4-Lite Channel Summary:
//   AW  Write Address  : AWVALID / AWREADY / AWADDR / AWPROT
//   W   Write Data     : WVALID  / WREADY  / WDATA  / WSTRB
//   B   Write Response : BVALID  / BREADY  / BRESP
//   AR  Read Address   : ARVALID / ARREADY / ARADDR / ARPROT
//   R   Read Data      : RVALID  / RREADY  / RDATA  / RRESP
//
// BRESP / RRESP encoding:
//   2'b00 = OKAY   (transaction successful)
//   2'b10 = SLVERR (slave error — unmapped address, locked reg, etc.)
// =============================================================================

`ifndef CSR_INTF_SV
`define CSR_INTF_SV

interface csr_intf #(
    parameter int  ADDR_WIDTH = 12,    // 4 KB CSR space (byte-addressed)
    parameter int  DATA_WIDTH = 32,    // AXI4-Lite mandates 32 or 64 b; use 32
    parameter bit  DUAL_CLOCK = 0      // 0 = single clk; 1 = separate config_clk
) (
    input logic clk,
    input logic rst_n,
    input logic config_clk             // tied to clk when DUAL_CLOCK==0
);

    // ── AXI4-Lite: Write Address Channel ─────────────────────────────────────
    logic                      awvalid;
    logic                      awready;
    logic [ADDR_WIDTH-1:0]     awaddr;
    logic [2:0]                awprot;   // [2]=instruction,[1]=secure,[0]=priv

    // ── AXI4-Lite: Write Data Channel ────────────────────────────────────────
    logic                      wvalid;
    logic                      wready;
    logic [DATA_WIDTH-1:0]     wdata;
    logic [DATA_WIDTH/8-1:0]   wstrb;   // byte enables

    // ── AXI4-Lite: Write Response Channel ────────────────────────────────────
    logic                      bvalid;
    logic                      bready;
    logic [1:0]                bresp;   // OKAY=2'b00 / SLVERR=2'b10

    // ── AXI4-Lite: Read Address Channel ──────────────────────────────────────
    logic                      arvalid;
    logic                      arready;
    logic [ADDR_WIDTH-1:0]     araddr;
    logic [2:0]                arprot;

    // ── AXI4-Lite: Read Data Channel ─────────────────────────────────────────
    logic                      rvalid;
    logic                      rready;
    logic [DATA_WIDTH-1:0]     rdata;
    logic [1:0]                rresp;   // OKAY=2'b00 / SLVERR=2'b10

    // ── Shadow-register control ───────────────────────────────────────────────
    // Master asserts cfg_update when it is safe to commit shadow → live regs.
    // Slave holds cfg_busy high while the commit is in flight.
    logic                      cfg_update;
    logic                      cfg_busy;

    // =========================================================================
    // Modports
    // =========================================================================

    // Master (CPU / AXI interconnect side)
    modport master (
        output awvalid, awaddr, awprot,
        input  awready,
        output wvalid,  wdata,  wstrb,
        input  wready,
        input  bvalid,  bresp,
        output bready,
        output arvalid, araddr, arprot,
        input  arready,
        input  rvalid,  rdata,  rresp,
        output rready,
        output cfg_update,
        input  cfg_busy
    );

    // Slave (CSR register file side)
    modport slave (
        input  awvalid, awaddr, awprot,
        output awready,
        input  wvalid,  wdata,  wstrb,
        output wready,
        output bvalid,  bresp,
        input  bready,
        input  arvalid, araddr, arprot,
        output arready,
        output rvalid,  rdata,  rresp,
        input  rready,
        input  cfg_update,
        output cfg_busy
    );

    // =========================================================================
    // Optional 2-FF CDC synchronizers (DUAL_CLOCK mode)
    // =========================================================================
    generate
        if (DUAL_CLOCK) begin : gen_cdc
            // cfg_update: config_clk domain → clk domain
            logic [1:0] cfg_upd_sync_ff;
            logic       cfg_update_sync;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    cfg_upd_sync_ff  <= 2'b00;
                    cfg_update_sync  <= 1'b0;
                end else begin
                    cfg_upd_sync_ff  <= {cfg_upd_sync_ff[0], cfg_update};
                    cfg_update_sync  <= cfg_upd_sync_ff[1];
                end
            end

            // cfg_busy: clk domain → config_clk domain
            logic [1:0] cfg_busy_sync_ff;
            logic       cfg_busy_sync;

            always_ff @(posedge config_clk or negedge rst_n) begin
                if (!rst_n) begin
                    cfg_busy_sync_ff <= 2'b00;
                    cfg_busy_sync    <= 1'b0;
                end else begin
                    cfg_busy_sync_ff <= {cfg_busy_sync_ff[0], cfg_busy};
                    cfg_busy_sync    <= cfg_busy_sync_ff[1];
                end
            end
        end
    endgenerate

    // =========================================================================
    // AXI4-Lite Protocol Assertions (SVA)
    // =========================================================================

    // AW: once AWVALID is asserted it must not deassert until AWREADY
    property aw_valid_stable;
        @(posedge clk) disable iff (!rst_n)
        (awvalid && !awready) |=> awvalid;
    endproperty

    // W: once WVALID is asserted it must not deassert until WREADY
    property w_valid_stable;
        @(posedge clk) disable iff (!rst_n)
        (wvalid && !wready) |=> wvalid;
    endproperty

    // AR: once ARVALID is asserted it must not deassert until ARREADY
    property ar_valid_stable;
        @(posedge clk) disable iff (!rst_n)
        (arvalid && !arready) |=> arvalid;
    endproperty

    // AW address must remain stable while AWVALID && !AWREADY
    property aw_addr_stable;
        @(posedge clk) disable iff (!rst_n)
        (awvalid && !awready) |=> (awaddr == $past(awaddr));
    endproperty

    // AR address must remain stable while ARVALID && !ARREADY
    property ar_addr_stable;
        @(posedge clk) disable iff (!rst_n)
        (arvalid && !arready) |=> (araddr == $past(araddr));
    endproperty

    // W data/strobe must remain stable while WVALID && !WREADY
    property w_data_stable;
        @(posedge clk) disable iff (!rst_n)
        (wvalid && !wready) |=> (wdata == $past(wdata)) && (wstrb == $past(wstrb));
    endproperty

    // B response must remain stable while BVALID && !BREADY
    property b_valid_stable;
        @(posedge clk) disable iff (!rst_n)
        (bvalid && !bready) |=> bvalid && (bresp == $past(bresp));
    endproperty

    // R response must remain stable while RVALID && !RREADY
    property r_valid_stable;
        @(posedge clk) disable iff (!rst_n)
        (rvalid && !rready) |=> rvalid && (rdata == $past(rdata)) && (rresp == $past(rresp));
    endproperty

    // BRESP must be OKAY or SLVERR only (no EXOKAY/DECERR for AXI4-Lite writes)
    property bresp_legal;
        @(posedge clk) disable iff (!rst_n)
        bvalid |-> (bresp == 2'b00 || bresp == 2'b10);
    endproperty

    // RRESP must be OKAY or SLVERR only
    property rresp_legal;
        @(posedge clk) disable iff (!rst_n)
        rvalid |-> (rresp == 2'b00 || rresp == 2'b10);
    endproperty

    // Bind assertions
    AXI4L_AW_VALID_STABLE  : assert property (aw_valid_stable)
        else $error("AXI4L violation: AWVALID dropped before AWREADY");
    AXI4L_W_VALID_STABLE   : assert property (w_valid_stable)
        else $error("AXI4L violation: WVALID dropped before WREADY");
    AXI4L_AR_VALID_STABLE  : assert property (ar_valid_stable)
        else $error("AXI4L violation: ARVALID dropped before ARREADY");
    AXI4L_AW_ADDR_STABLE   : assert property (aw_addr_stable)
        else $error("AXI4L violation: AWADDR changed while AWVALID && !AWREADY");
    AXI4L_AR_ADDR_STABLE   : assert property (ar_addr_stable)
        else $error("AXI4L violation: ARADDR changed while ARVALID && !ARREADY");
    AXI4L_W_DATA_STABLE    : assert property (w_data_stable)
        else $error("AXI4L violation: WDATA/WSTRB changed while WVALID && !WREADY");
    AXI4L_B_VALID_STABLE   : assert property (b_valid_stable)
        else $error("AXI4L violation: BVALID/BRESP changed while !BREADY");
    AXI4L_R_VALID_STABLE   : assert property (r_valid_stable)
        else $error("AXI4L violation: RVALID/RDATA/RRESP changed while !RREADY");
    AXI4L_BRESP_LEGAL      : assert property (bresp_legal)
        else $error("AXI4L violation: illegal BRESP encoding");
    AXI4L_RRESP_LEGAL      : assert property (rresp_legal)
        else $error("AXI4L violation: illegal RRESP encoding");

    // =========================================================================
    // Functional Coverage
    // =========================================================================
    covergroup axi4l_cg @(posedge clk);
        cp_aw_hs : coverpoint {awvalid, awready} {
            bins handshake = {2'b11};
            bins idle      = {2'b00};
        }
        cp_w_hs  : coverpoint {wvalid, wready} {
            bins handshake = {2'b11};
            bins idle      = {2'b00};
        }
        cp_b_hs  : coverpoint {bvalid, bready} {
            bins handshake = {2'b11};
        }
        cp_ar_hs : coverpoint {arvalid, arready} {
            bins handshake = {2'b11};
            bins idle      = {2'b00};
        }
        cp_r_hs  : coverpoint {rvalid, rready} {
            bins handshake = {2'b11};
        }
        cp_bresp : coverpoint bresp {
            bins okay   = {2'b00};
            bins slverr = {2'b10};
        }
        cp_rresp : coverpoint rresp {
            bins okay   = {2'b00};
            bins slverr = {2'b10};
        }
        cp_wstrb : coverpoint wstrb {
            bins byte0 = {4'b0001};
            bins byte1 = {4'b0010};
            bins byte2 = {4'b0100};
            bins byte3 = {4'b1000};
            bins word  = {4'b1111};
            bins other = default;
        }
        cp_awaddr_range : coverpoint awaddr {
            bins low  = {[0  : (1<<(ADDR_WIDTH-2))-1]};
            bins mid  = {[(1<<(ADDR_WIDTH-2)) : (3<<(ADDR_WIDTH-2))-1]};
            bins high = {[(3<<(ADDR_WIDTH-2)) : (1<<ADDR_WIDTH)-1]};
        }
        cx_rw : cross cp_aw_hs, cp_ar_hs;
    endgroup

    axi4l_cg axi4l_cg_inst = new();

endinterface

`endif // CSR_INTF_SV
