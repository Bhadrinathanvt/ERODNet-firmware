//-----------------------------------------------------------------------------
// Module: e_rodnet_dma_wrapper.sv
// Description: Top-level DMA wrapper for e_rodnet integration with Shakti SoC.
//              Bridges AXI4 master interface to e_rodnet CNN accelerator.
//
// Architecture:
//   - AXI4-Lite CSR for CPU configuration
//   - AXI4 Master for DDR memory access
//   - Token-based streaming to e_rodnet IP
//   - PL330-style peripheral flow control
//   - 4KB boundary handling
//
// Integration with e_rodnet:
//   - dma_src_*: DDR read → token assembly → e_rodnet input
//   - dma_dst_*: e_rodnet output → token disassembly → DDR write
//
// Target: Shakti Yamuna SoC (RISECREEK, RIMO, Moushik)
//-----------------------------------------------------------------------------
module e_rodnet_dma_wrapper #(
    parameter int unsigned AXI_DATA_WIDTH   = 32,
    parameter int unsigned AXI_ADDR_WIDTH   = 32,
    parameter int unsigned TOKEN_WIDTH      = 512,
    parameter int unsigned BEATS_PER_TOKEN  = 16,
    parameter int unsigned MAX_TOKENS       = 65535,
    parameter int unsigned FIFO_DEPTH      = 4,
    parameter int unsigned NUM_PERIPHS     = 8
)(
    input  logic                       clk,
    input  logic                       rst_n,

    //==========================================================
    // AXI4-Lite Slave (CPU Configuration)
    //==========================================================
    input  logic [11:0]                s_axi_awaddr,
    input  logic [2:0]                 s_axi_awprot,
    input  logic                       s_axi_awvalid,
    output logic                       s_axi_awready,
    input  logic [31:0]                s_axi_wdata,
    input  logic [3:0]                 s_axi_wstrb,
    input  logic                       s_axi_wvalid,
    output logic                       s_axi_wready,
    output logic [1:0]                 s_axi_bresp,
    output logic                       s_axi_bvalid,
    input  logic                       s_axi_bready,
    input  logic [11:0]                s_axi_araddr,
    input  logic [2:0]                 s_axi_arprot,
    input  logic                       s_axi_arvalid,
    output logic                       s_axi_arready,
    output logic [31:0]                s_axi_rdata,
    output logic [1:0]                 s_axi_rresp,
    output logic                       s_axi_rvalid,
    input  logic                       s_axi_rready,

    //==========================================================
    // AXI4 Master (DDR Memory Access)
    //==========================================================
    output logic [3:0]                 m_axi_awid,
    output logic [AXI_ADDR_WIDTH-1:0]  m_axi_awaddr,
    output logic [7:0]                 m_axi_awlen,
    output logic [2:0]                 m_axi_awsize,
    output logic [1:0]                 m_axi_awburst,
    output logic                       m_axi_awlock,
    output logic [3:0]                 m_axi_awcache,
    output logic [2:0]                 m_axi_awprot,
    output logic [3:0]                 m_axi_awqos,
    output logic                       m_axi_awvalid,
    input  logic                       m_axi_awready,

    output logic [AXI_DATA_WIDTH-1:0]  m_axi_wdata,
    output logic [AXI_DATA_WIDTH/8-1:0]m_axi_wstrb,
    output logic                       m_axi_wlast,
    output logic                       m_axi_wvalid,
    input  logic                       m_axi_wready,

    input  logic [3:0]                 m_axi_bid,
    input  logic [1:0]                 m_axi_bresp,
    input  logic                       m_axi_bvalid,
    output logic                       m_axi_bready,

    output logic [3:0]                 m_axi_arid,
    output logic [AXI_ADDR_WIDTH-1:0]  m_axi_araddr,
    output logic [7:0]                 m_axi_arlen,
    output logic [2:0]                 m_axi_arsize,
    output logic [1:0]                 m_axi_arburst,
    output logic                       m_axi_arlock,
    output logic [3:0]                 m_axi_arcache,
    output logic [2:0]                 m_axi_arprot,
    output logic [3:0]                 m_axi_arqos,
    output logic                       m_axi_arvalid,
    input  logic                       m_axi_arready,

    input  logic [3:0]                 m_axi_rid,
    input  logic [AXI_DATA_WIDTH-1:0]  m_axi_rdata,
    input  logic [1:0]                 m_axi_rresp,
    input  logic                       m_axi_rlast,
    input  logic                       m_axi_rvalid,
    output logic                       m_axi_rready,

    //==========================================================
    // Streaming Interface to e_rodnet IP
    //==========================================================
    output logic                       dma_src_valid,
    output logic [TOKEN_WIDTH-1:0]      dma_src_data,
    output logic                       dma_src_last,
    input  logic                       dma_src_ready,

    input  logic                       dma_dst_valid,
    input  logic [TOKEN_WIDTH-1:0]      dma_dst_data,
    input  logic                       dma_dst_last,
    output logic                       dma_dst_ready,

    //==========================================================
    // Peripheral Request Interface
    //==========================================================
    input  logic [NUM_PERIPHS-1:0]    periph_req_valid,
    input  logic [NUM_PERIPHS-1:0][1:0] periph_req_type,
    output logic [NUM_PERIPHS-1:0]    periph_ack_valid,
    output logic [NUM_PERIPHS-1:0][1:0] periph_ack_type,

    //==========================================================
    // Interrupts
    //==========================================================
    output logic                       irq_done,
    output logic                       irq_error
);

    import dma_e_rodnet_pkg::*;

    //==========================================================
    // Internal Signals
    //==========================================================
    // DMA engine status
    logic                       engine_done;
    logic                       engine_error;
    logic                       engine_busy;
    logic                       engine_src_err;
    logic                       engine_dst_err;
    logic                       engine_boundary_err;
    logic                       engine_timeout_err;
    logic [15:0]               engine_tokens_done;

    // Peripheral status (from peripheral request manager)
    logic [NUM_PERIPHS-1:0]    periph_active_status;
    logic [NUM_PERIPHS-1:0]    periph_error_status;

    //==========================================================
    // DMA Top-Level Instance
    //==========================================================
    dma_e_rodnet_top #(
        .AXI_DATA_WIDTH(AXI_DATA_WIDTH),
        .AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),
        .TOKEN_WIDTH(TOKEN_WIDTH),
        .BEATS_PER_TOKEN(BEATS_PER_TOKEN),
        .MAX_TOKENS(MAX_TOKENS),
        .FIFO_DEPTH(FIFO_DEPTH),
        .NUM_PERIPHS(NUM_PERIPHS)
    ) u_dma_top (
        .clk(clk),
        .rst_n(rst_n),

        // AXI4-Lite Slave
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awprot(s_axi_awprot),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arprot(s_axi_arprot),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),

        // AXI4 Master
        .m_axi_awid(m_axi_awid),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awlock(m_axi_awlock),
        .m_axi_awcache(m_axi_awcache),
        .m_axi_awprot(m_axi_awprot),
        .m_axi_awqos(m_axi_awqos),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arlock(m_axi_arlock),
        .m_axi_arcache(m_axi_arcache),
        .m_axi_arprot(m_axi_arprot),
        .m_axi_arqos(m_axi_arqos),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),

        // Streaming Interface
        .dma_src_valid(dma_src_valid),
        .dma_src_data(dma_src_data),
        .dma_src_last(dma_src_last),
        .dma_src_ready(dma_src_ready),
        .dma_dst_valid(dma_dst_valid),
        .dma_dst_data(dma_dst_data),
        .dma_dst_last(dma_dst_last),
        .dma_dst_ready(dma_dst_ready),

        // Peripheral Request Interface
        .periph_req_valid(periph_req_valid),
        .periph_req_type(periph_req_type),
        .periph_ack_valid(periph_ack_valid),
        .periph_ack_type(periph_ack_type),

        // Interrupt
        .dma_irq(irq_done)  // Combined interrupt
    );

    //==========================================================
    // Interrupt Generation
    //==========================================================
    // Error interrupt (separate from done)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            irq_error <= 1'b0;
        end else begin
            irq_error <= engine_error || engine_src_err || engine_dst_err ||
                         engine_boundary_err || engine_timeout_err ||
                         (|periph_error_status);
        end
    end

    //==========================================================
    // Assertions
    //==========================================================
    `ifdef FORMAL
        // Verify AXI4 protocol compliance
        // Note: Using ##[0:15] instead of ##[0:$] for Verilator compatibility
        assert property (@(posedge clk) disable iff (!rst_n)
            m_axi_awvalid |-> ##[0:15] m_axi_awready)
            else $error("AXI AWVALID held without AWREADY");

        assert property (@(posedge clk) disable iff (!rst_n)
            m_axi_arvalid |-> ##[0:15] m_axi_arready)
            else $error("AXI ARVALID held without ARREADY");

        // Verify streaming protocol
        assert property (@(posedge clk) disable iff (!rst_n)
            dma_src_valid && !dma_src_ready |-> $stable(dma_src_data))
            else $error("Source data changed during backpressure");

        assert property (@(posedge clk) disable iff (!rst_n)
            dma_dst_valid && !dma_dst_ready |-> $stable(dma_dst_data))
            else $error("Destination data changed during backpressure");
    `endif

endmodule : e_rodnet_dma_wrapper
