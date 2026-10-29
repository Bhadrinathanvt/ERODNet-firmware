//-----------------------------------------------------------------------------
// Module: dma_e_rodnet_top.sv
// Description: Top-level DMA controller for e_rodnet with peripheral flow control.
//              Integrates CSR, source engine, sink engine, and peripheral request.
//              Dual-path architecture: DDR→IP and IP→DDR.
// Target: Shakti Yamuna SoC — AXI4 Full, 32-bit data, 32-bit addr
// Token width: 512 bits = 16 AXI4 beats
//
// Features:
//   - AXI4-Lite CSR interface for CPU configuration
//   - AXI4 Master interface for DDR memory access
//   - Token-based streaming (512-bit tokens)
//   - PL330-style peripheral request/acknowledge flow control
//   - 4KB boundary detection and handling
//   - Multi-channel support (up to 8 peripherals)
//   - Error detection and interrupt coalescing
//-----------------------------------------------------------------------------
module dma_e_rodnet_top #(
    parameter int unsigned AXI_DATA_WIDTH  = 32,
    parameter int unsigned AXI_ADDR_WIDTH  = 32,
    parameter int unsigned AXI_ID_WIDTH    = 4,   // FIX: Added parameterized ID width
    parameter int unsigned TOKEN_WIDTH     = 512,
    parameter int unsigned BEATS_PER_TOKEN = 16,
    parameter int unsigned MAX_TOKENS      = 65535,
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
    // Write Address Channel
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

    // Write Data Channel
    output logic [AXI_DATA_WIDTH-1:0]  m_axi_wdata,
    output logic [AXI_DATA_WIDTH/8-1:0]m_axi_wstrb,
    output logic                       m_axi_wlast,
    output logic                       m_axi_wvalid,
    input  logic                       m_axi_wready,

    // Write Response Channel
    input  logic [3:0]                 m_axi_bid,
    input  logic [1:0]                 m_axi_bresp,
    input  logic                       m_axi_bvalid,
    output logic                       m_axi_bready,

    // Read Address Channel
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

    // Read Data Channel
    input  logic [3:0]                 m_axi_rid,
    input  logic [AXI_DATA_WIDTH-1:0]  m_axi_rdata,
    input  logic [1:0]                 m_axi_rresp,
    input  logic                       m_axi_rlast,
    input  logic                       m_axi_rvalid,
    output logic                       m_axi_rready,

    //==========================================================
    // Streaming Interface to e_rodnet IP
    //==========================================================
    // Source path: DDR → IP (input feature map)
    output logic                       dma_src_valid,
    output logic [TOKEN_WIDTH-1:0]      dma_src_data,
    output logic                       dma_src_last,
    input  logic                       dma_src_ready,

    // Sink path: IP → DDR (output feature map)
    input  logic                       dma_dst_valid,
    input  logic [TOKEN_WIDTH-1:0]      dma_dst_data,
    input  logic                       dma_dst_last,
    output logic                       dma_dst_ready,

    //==========================================================
    // Peripheral Request Interface (PL330-style)
    //==========================================================
    input  logic [NUM_PERIPHS-1:0]    periph_req_valid,
    input  logic [NUM_PERIPHS-1:0][1:0] periph_req_type,
    output logic [NUM_PERIPHS-1:0]    periph_ack_valid,
    output logic [NUM_PERIPHS-1:0][1:0] periph_ack_type,

    //==========================================================
    // Interrupt
    //==========================================================
    output logic                       dma_irq
);

    import dma_e_rodnet_pkg::*;

    //==========================================================
    // Internal Signals
    //==========================================================
    // CSR to Engine
    logic [AXI_ADDR_WIDTH-1:0]         engine_src_addr;
    logic [AXI_ADDR_WIDTH-1:0]         engine_dst_addr;
    logic [15:0]                       engine_token_count;
    logic [7:0]                        engine_burst_len;
    logic                              engine_start;
    logic                              engine_abort;
    logic                              engine_loop_mode;
    logic                              engine_src_inc;
    logic                              engine_irq_en_done;
    logic                              engine_irq_en_error;

    // Engine to CSR
    logic                              engine_status_done;
    logic                              engine_status_error;
    logic                              engine_status_busy;
    logic                              engine_status_src_err;
    logic                              engine_status_dst_err;
    logic                              engine_status_boundary_err;
    logic                              engine_status_timeout_err;
    logic [15:0]                       engine_tokens_done;

    //==========================================================
    // CSR Instance
    //==========================================================
    dma_e_rodnet_csr #(
        .ADDR_WIDTH(12),
        .DATA_WIDTH(32)
    ) u_csr (
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

        // DMA Control Outputs
        .dma_src_addr(engine_src_addr),
        .dma_dst_addr(engine_dst_addr),
        .dma_token_count(engine_token_count),
        .dma_burst_len(engine_burst_len),
        .dma_start(engine_start),
        .dma_abort(engine_abort),
        .dma_loop_mode(engine_loop_mode),
        .dma_src_inc(engine_src_inc),
        .dma_irq_en_done(engine_irq_en_done),
        .dma_irq_en_error(engine_irq_en_error),

        // DMA Status Inputs
        .dma_status_done(engine_status_done),
        .dma_status_error(engine_status_error),
        .dma_status_busy(engine_status_busy),
        .dma_status_src_err(engine_status_src_err),
        .dma_status_dst_err(engine_status_dst_err),
        .dma_tokens_done(engine_tokens_done),

        // Interrupt
        .irq(dma_irq)
    );

    //==========================================================
    // DMA Engine Instance
    //==========================================================
    dma_e_rodnet_engine #(
        .AXI_DATA_WIDTH(AXI_DATA_WIDTH),
        .AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),
        .AXI_ID_WIDTH(AXI_ID_WIDTH),  // FIX: Added parameterized ID width
        .TOKEN_WIDTH(TOKEN_WIDTH),
        .BEATS_PER_TOKEN(BEATS_PER_TOKEN),
        .MAX_TOKENS(MAX_TOKENS),
        .FIFO_DEPTH(FIFO_DEPTH),
        .NUM_PERIPHS(NUM_PERIPHS)
    ) u_engine (
        .clk(clk),
        .rst_n(rst_n),

        // AXI4 Master
        .m_awid(m_axi_awid),
        .m_awaddr(m_axi_awaddr),
        .m_awlen(m_axi_awlen),
        .m_awsize(m_axi_awsize),
        .m_awburst(m_axi_awburst),
        .m_awlock(m_axi_awlock),
        .m_awcache(m_axi_awcache),
        .m_awprot(m_axi_awprot),
        .m_awqos(m_axi_awqos),
        .m_awvalid(m_axi_awvalid),
        .m_awready(m_axi_awready),
        .m_wdata(m_axi_wdata),
        .m_wstrb(m_axi_wstrb),
        .m_wlast(m_axi_wlast),
        .m_wvalid(m_axi_wvalid),
        .m_wready(m_axi_wready),
        .m_bid(m_axi_bid),
        .m_bresp(m_axi_bresp),
        .m_bvalid(m_axi_bvalid),
        .m_bready(m_axi_bready),
        .m_arid(m_axi_arid),
        .m_araddr(m_axi_araddr),
        .m_arlen(m_axi_arlen),
        .m_arsize(m_axi_arsize),
        .m_arburst(m_axi_arburst),
        .m_arlock(m_axi_arlock),
        .m_arcache(m_axi_arcache),
        .m_arprot(m_axi_arprot),
        .m_arqos(m_axi_arqos),
        .m_arvalid(m_axi_arvalid),
        .m_arready(m_axi_arready),
        .m_rid(m_axi_rid),
        .m_rdata(m_axi_rdata),
        .m_rresp(m_axi_rresp),
        .m_rlast(m_axi_rlast),
        .m_rvalid(m_axi_rvalid),
        .m_rready(m_axi_rready),

        // Streaming Interface
        .src_valid(dma_src_valid),
        .src_data(dma_src_data),
        .src_last(dma_src_last),
        .src_ready(dma_src_ready),
        .dst_valid(dma_dst_valid),
        .dst_data(dma_dst_data),
        .dst_last(dma_dst_last),
        .dst_ready(dma_dst_ready),

        // Peripheral Request Interface
        .periph_req_valid(periph_req_valid),
        .periph_req_type(periph_req_type),
        .periph_ack_valid(periph_ack_valid),
        .periph_ack_type(periph_ack_type),

        // Control/Status
        .src_addr(engine_src_addr),
        .dst_addr(engine_dst_addr),
        .token_count(engine_token_count),
        .burst_len(engine_burst_len),
        .start(engine_start),
        .abort(engine_abort),
        .loop_mode(engine_loop_mode),
        .src_inc(engine_src_inc),
        .done(engine_status_done),
        .busy(engine_status_busy),
        .error(engine_status_error),
        .src_err(engine_status_src_err),
        .dst_err(engine_status_dst_err),
        .boundary_err(engine_status_boundary_err),
        .timeout_err(engine_status_timeout_err),
        .tokens_done(engine_tokens_done),
        .irq() // Handled by CSR
    );

    //==========================================================
    // Assertions
    //==========================================================
    `ifdef FORMAL
        // Verify token alignment
        assert property (@(posedge clk) disable iff (!rst_n)
            engine_start |-> engine_src_addr[5:0] == '0 && engine_dst_addr[5:0] == '0)
            else $error("DMA start with misaligned addresses");

        // Verify valid/ready protocol on streaming interfaces
        assert property (@(posedge clk) disable iff (!rst_n)
            dma_src_valid && !dma_src_ready |-> $stable(dma_src_data))
            else $error("Source data changed during backpressure");

        assert property (@(posedge clk) disable iff (!rst_n)
            dma_dst_valid && !dma_dst_ready |-> $stable(dma_dst_data))
            else $error("Destination data changed during backpressure");

        // Verify peripheral request/acknowledge protocol
        for (genvar i = 0; i < NUM_PERIPHS; i++) begin : gen_periph_assert
            assert property (@(posedge clk) disable iff (!rst_n)
                periph_req_valid[i] && !periph_ack_valid[i] |-> $stable(periph_req_type[i]))
                else $error("Peripheral %0d request type changed before acknowledge", i);
        end
    `endif

endmodule : dma_e_rodnet_top

