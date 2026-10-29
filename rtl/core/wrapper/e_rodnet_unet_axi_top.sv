// =============================================================================
// e_rodnet_unet_axi_top.sv
//
// AXI-wrapped E-RODNet UNet with split-clock domains.
//   axi_clk  — AXI4-Lite CSR, AXI4 master DMA (bus-facing)
//   core_clk — Accelerator core (e_rodnet_unet_top, radar, compute)
//
// Two radar frontend paths (selected at synthesis via EN_RADAR_PROCESSING):
//   EN_RADAR_PROCESSING=1 — on-FPGA radar_processing_top (ADC → FFT → CFAR)
//   EN_RADAR_PROCESSING=0 — external AWR1843AOP via UART RX (pre-computed RA maps)
//
// CDC at every domain crossing:
//   - CSR config: cdc_sync_vec (static during frame)
//   - Frame buffer: true_dual_port_bram (SAME_CLK=0)
//   - Weight path: cdc_sync_data (toggle handshake)
//   - Sparse CSR: cdc_sync_data request + response
//   - Detection output: cdc_sync_data
//   - Status/perf: cdc_sync_single + cdc_sync_counter
//
// Bus interfaces (per user requirement: full AXI4 / AXI4-Lite, no AXI-Stream):
//
//   1. AXI4-Lite slave  (s_axil_*)   --- 32-bit CSR, 16 registers (0x00..0x3C)
//   2. AXI4   master    (m_axi_in_*) --- frame DMA read  (RA tiles <- memory)
//   3. AXI4   master    (m_axi_wt_*) --- weight DMA read (driven by
//                                        erodnet_weight_controller submodule)
//   4. AXI4   master    (m_axi_out_*) -- detection DMA write (results -> memory)
//
// Data layout (frame DMA):
//   Each token  = MAX_C x DATA_WIDTH bits = 64 x 8 = 512 bits
//   = AXI_DW/64 = 8 AXI beats per token (when AXI_DW = 64 bits)
//   Frame size  = FEATURE_H x FEATURE_W x MAX_C x DATA_WIDTH/8 bytes
//               = 64 x 64 x 64 x 1 = 262144 B = 256 KB (default)
//
// Detection layout (output DMA):
//   Each detection packed into a single 128-bit record:
//     {pad[31:0], class_id[1:0], conf[7:0], pad[5:0], h[15:0], w[15:0],
//      width[15:0], height[15:0]}
//   Two 64-bit AXI beats per detection.
//
// CSR Map (offsets, 32-bit aligned):
//   0x00 CTRL          [0]=enable [1]=soft_reset [2]=start [3]=abort
//   0x04 STATUS        [0]=busy   [1]=done       [2]=error [15:8]=det_count
//   0x08 FRAME_SRC     base address for input frame
//   0x0C FRAME_BYTES   total frame size in bytes
//   0x10 OUT_DST       base address for detection output
//   0x14 OUT_MAX_BYTES output buffer max size
//   0x18 CFG_THR       [7:0]=conf [15:8]=iou [23:16]=top_k [26:24]=temporal_mode
//   0x1C CFG_HDP       [3:0]=history_depth
//   0x20 WT_CSR        weight CSR-mode register (auto-inc + write strobe)
//   0x24 WT_DATA       weight value
//   0x28 WT_DMA_SRC    weight DMA source address
//   0x2C WT_DMA_CNT    weight DMA count + start strobe
//   0x30 PERF_CYCLES   cycle counter (RO)
//   0x34 PERF_FRAMES   frame counter (RO)
//   0x38 IRQ_STATUS    [0]=frame_done [1]=err  (W1C)
//   0x3C IRQ_ENABLE    [0]=frame_done_en [1]=err_en
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;

module e_rodnet_unet_axi_top #(
    parameter int FEATURE_H      = 64,
    parameter int FEATURE_W      = 64,
    parameter int TILE_H         = 32,
    parameter int TILE_W         = 32,
    parameter int IN_CHANNELS    = 16,
    parameter int NUM_CLASSES    = 3,
    parameter int NUM_ANCHORS    = 3,
    parameter int TOP_K          = 100,
    parameter int HISTORY_DEPTH  = 8,
    parameter int DATA_WIDTH     = pkg_config::DATA_WIDTH,
    parameter int FRAC_WIDTH     = pkg_config::FRAC_WIDTH,
    parameter int AXI_DW         = 64,
    parameter int AXI_AW         = 32,
    parameter int AXI_ID_W       = 4,
    parameter int AXI_BURST_LEN  = pkg_config::AXI_BURST_LEN,
    parameter int DET_FIFO_DEPTH = pkg_config::DET_FIFO_DEPTH
)(
    // Clocks / reset --- split domains
    input  logic                       axi_clk,
    input  logic                       core_clk,
    (* max_fanout = 64 *) input  logic                       rst_n,

    // Radar ADC input (direct from mmWave frontend)
    input  logic signed [31:0]         adc_re [3:0],
    input  logic signed [31:0]         adc_im [3:0],
    input  logic                       adc_valid,
    input  logic                       adc_last,
    input  logic [11:0]                adc_sample_idx,
    output logic                       adc_ready,

    // FFT Accelerator interface (for Range FFT inside radar_processing_top)
    input  logic                       fft_done,
    input  logic                       fft_busy,
    output logic                       fft_start,
    output logic [11:0]                fft_size,
    output logic [2:0]                 fft_radix_out,
    output logic                       fft_inverse,

    // UART RX for pre-computed RA maps (AWR1843AOP, 921600 baud)
    input  logic                       uart_rx,

    // AXI4-Lite slave (CSR)
    input  logic [AXI_AW-1:0]          s_axil_awaddr,
    input  logic                       s_axil_awvalid,
    output logic                       s_axil_awready,
    input  logic [31:0]                s_axil_wdata,
    input  logic [3:0]                 s_axil_wstrb,
    input  logic                       s_axil_wvalid,
    output logic                       s_axil_wready,
    output logic [1:0]                 s_axil_bresp,
    output logic                       s_axil_bvalid,
    input  logic                       s_axil_bready,
    input  logic [AXI_AW-1:0]          s_axil_araddr,
    input  logic                       s_axil_arvalid,
    output logic                       s_axil_arready,
    output logic [31:0]                s_axil_rdata,
    output logic [1:0]                 s_axil_rresp,
    output logic                       s_axil_rvalid,
    input  logic                       s_axil_rready,

    // AXI4 master --- frame DMA read
    output logic [AXI_ID_W-1:0]        m_axi_in_arid,
    output logic [AXI_AW-1:0]          m_axi_in_araddr,
    output logic [7:0]                 m_axi_in_arlen,
    output logic [2:0]                 m_axi_in_arsize,
    output logic [1:0]                 m_axi_in_arburst,
    output logic                       m_axi_in_arvalid,
    input  logic                       m_axi_in_arready,
    input  logic [AXI_ID_W-1:0]        m_axi_in_rid,
    input  logic [AXI_DW-1:0]          m_axi_in_rdata,
    input  logic [1:0]                 m_axi_in_rresp,
    input  logic                       m_axi_in_rlast,
    input  logic                       m_axi_in_rvalid,
    output logic                       m_axi_in_rready,

    // AXI4 master --- weight DMA read (proxied to erodnet_weight_controller)
    output logic [AXI_AW-1:0]          m_axi_wt_araddr,
    output logic [7:0]                 m_axi_wt_arlen,
    output logic [2:0]                 m_axi_wt_arsize,
    output logic [1:0]                 m_axi_wt_arburst,
    output logic                       m_axi_wt_arvalid,
    input  logic                       m_axi_wt_arready,
    input  logic [AXI_DW-1:0]          m_axi_wt_rdata,
    input  logic [1:0]                 m_axi_wt_rresp,
    input  logic                       m_axi_wt_rvalid,
    input  logic                       m_axi_wt_rlast,
    output logic                       m_axi_wt_rready,

    // AXI4 master --- detection output DMA write
    output logic [AXI_ID_W-1:0]        m_axi_out_awid,
    output logic [AXI_AW-1:0]          m_axi_out_awaddr,
    output logic [7:0]                 m_axi_out_awlen,
    output logic [2:0]                 m_axi_out_awsize,
    output logic [1:0]                 m_axi_out_awburst,
    output logic                       m_axi_out_awvalid,
    input  logic                       m_axi_out_awready,
    output logic [AXI_DW-1:0]          m_axi_out_wdata,
    output logic [(AXI_DW/8)-1:0]      m_axi_out_wstrb,
    output logic                       m_axi_out_wlast,
    output logic                       m_axi_out_wvalid,
    input  logic                       m_axi_out_wready,
    input  logic [AXI_ID_W-1:0]        m_axi_out_bid,
    input  logic [1:0]                 m_axi_out_bresp,
    input  logic                       m_axi_out_bvalid,
    output logic                       m_axi_out_bready,

    // Interrupt
    output logic                       irq
);

    // =========================================================================
    // Reset Synchronizers
    // =========================================================================
    logic axi_rst_n, core_rst_n;

    cdc_reset_sync #(.STAGES(3)) u_axi_rst_sync (
        .src_rst_n(rst_n),
        .dst_clk(axi_clk),
        .dst_rst_n(axi_rst_n)
    );

    cdc_reset_sync #(.STAGES(3)) u_core_rst_sync (
        .src_rst_n(rst_n),
        .dst_clk(core_clk),
        .dst_rst_n(core_rst_n)
    );

    // =========================================================================
    // Constants
    // =========================================================================
    localparam int BEATS_PER_TOKEN  = (MAX_C * DATA_WIDTH) / AXI_DW;          // 8
    localparam int FRAME_TOKENS     = FEATURE_H * FEATURE_W;                  // 4096
    localparam int FRAME_BEATS      = FRAME_TOKENS * BEATS_PER_TOKEN;         // 32768
    localparam int IN_BURST_BEATS   = AXI_BURST_LEN;
    localparam int OUT_BEATS_PER_DET = 2;                                     // 16B/det

    // =========================================================================
    // AXI Domain: CSR Registers
    // =========================================================================
    logic        reg_enable, reg_soft_reset, reg_start, reg_abort;
    logic [31:0] reg_frame_src, reg_frame_bytes;
    logic [31:0] reg_out_dst,   reg_out_max_bytes;
    logic signed [DATA_WIDTH-1:0] reg_conf_thr, reg_iou_thr;
    logic [7:0]  reg_top_k;
    logic [2:0]  reg_temporal_mode;
    logic [3:0]  reg_history_depth;
    logic [31:0] reg_wt_csr;
    logic        reg_wt_write_strobe;
    logic [31:0] reg_wt_data;
    logic [31:0] reg_wt_dma_src;
    logic [31:0] reg_wt_dma_cnt;
    logic        reg_wt_dma_start_strobe;
    logic [31:0] reg_perf_cycles_axi, reg_perf_frames_axi;
    logic [1:0]  reg_irq_status;       // sticky
    logic [1:0]  reg_irq_enable;

    // Radar config CSR registers
    logic        reg_radar_mode;
    logic [11:0] reg_range_fft_size;
    logic [2:0]  reg_range_fft_radix;
    logic [1:0]  reg_window_type;
    logic        reg_window_enable;
    logic        reg_beamforming_enable;
    logic        reg_angle_fft_enable;
    logic        reg_magnitude_enable;
    logic        reg_log_output;
    logic        reg_cfar_en;
    logic [1:0]  reg_cfar_mode;
    logic [9:0]  reg_cfar_threshold_db;
    logic [3:0]  reg_cfar_guard_cells;
    logic [4:0]  reg_cfar_train_cells;
    logic [7:0]  reg_cfar_scale_factor;
    logic [3:0]  reg_cfar_os_rank_k;

    // =========================================================================
    // AXI Domain: AXI4-Lite Slave
    // =========================================================================
    logic axi_aw_hs, axi_w_hs, axi_ar_hs;

    assign axi_aw_hs = s_axil_awvalid && s_axil_awready;
    assign axi_w_hs  = s_axil_wvalid  && s_axil_wready;
    assign axi_ar_hs = s_axil_arvalid && s_axil_arready;

    // Byte-wise write mask from AXI4-Lite WSTRB
    logic [31:0] wmask;
    always_comb begin
        wmask = '0;
        for (int b = 0; b < 4; b++)
            if (s_axil_wstrb[b]) wmask[b*8 +: 8] = 8'hFF;
    end

    // =========================================================================
    // AXI Domain: Sparse CSR Bridge
    // =========================================================================
    logic        sparse_csr_req_valid;
    logic [11:0] sparse_csr_req_addr;
    logic [31:0] sparse_csr_req_wdata;
    logic        sparse_csr_req_we;
    logic        is_sparse_csr_write, is_sparse_csr_read;

    // Sparse CSR occupies 0xB00-0xB3F (64 bytes, local offsets 0x000-0x03F)
    assign is_sparse_csr_write = axi_w_hs && aw_pending && (aw_addr_q[11:4] == 8'hB0) && (aw_addr_q[7:6] == 2'b00);
    assign is_sparse_csr_read  = axi_ar_hs && (s_axil_araddr[11:4] == 8'hB0) && (s_axil_araddr[7:6] == 2'b00);

    // Sparse CSR request CDC instance (axi -> core) declared below after the request is driven

    // Sparse CSR response CDC (core -> axi)
    logic [31:0] sparse_rdata_cdc;
    logic        sparse_rdata_cdc_valid;
    logic        sparse_ar_pending;

    // =========================================================================
    // AXI Domain: AW/Write Channel
    // =========================================================================
    logic [AXI_AW-1:0] aw_addr_q;
    logic              aw_pending;

    always_ff @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n) begin
            aw_addr_q  <= '0;
            aw_pending <= 1'b0;
            s_axil_awready <= 1'b1;
            s_axil_wready  <= 1'b0;
            s_axil_bvalid  <= 1'b0;
            s_axil_bresp   <= 2'b00;
        end else begin
            if (axi_aw_hs) begin
                aw_addr_q      <= s_axil_awaddr;
                aw_pending     <= 1'b1;
                s_axil_awready <= 1'b0;
                s_axil_wready  <= 1'b1;
            end
            if (axi_w_hs) begin
                aw_pending     <= 1'b0;
                s_axil_wready  <= 1'b0;
                s_axil_bvalid  <= 1'b1;
                s_axil_bresp   <= 2'b00;
            end
            if (s_axil_bvalid && s_axil_bready) begin
                s_axil_bvalid  <= 1'b0;
                s_axil_awready <= 1'b1;
            end
        end
    end

    // CSR write decode (axi_clk domain)
    always_ff @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n) begin
            reg_enable        <= 1'b0;
            reg_soft_reset    <= 1'b0;
            reg_start         <= 1'b0;
            reg_abort         <= 1'b0;
            reg_frame_src     <= '0;
            reg_frame_bytes   <= '0;
            reg_out_dst       <= '0;
            reg_out_max_bytes <= '0;
            reg_conf_thr      <= '0;
            reg_iou_thr       <= '0;
            reg_top_k         <= 8'(TOP_K);
            reg_temporal_mode <= '0;
            reg_history_depth <= 4'(HISTORY_DEPTH);
            reg_wt_csr        <= '0;
            reg_wt_write_strobe <= 1'b0;
            reg_wt_data       <= '0;
            reg_wt_dma_src    <= '0;
            reg_wt_dma_cnt    <= '0;
            reg_wt_dma_start_strobe <= 1'b0;
            reg_irq_enable    <= '0;
            reg_radar_mode          <= 1'b0;
            reg_range_fft_size      <= 12'd256;
            reg_range_fft_radix     <= 3'd2;
            reg_window_type         <= 2'd0;
            reg_window_enable       <= 1'b1;
            reg_beamforming_enable  <= 1'b1;
            reg_angle_fft_enable    <= 1'b1;
            reg_magnitude_enable    <= 1'b1;
            reg_log_output          <= 1'b0;
            reg_cfar_en             <= 1'b0;
            reg_cfar_mode           <= 2'd0;
            reg_cfar_threshold_db   <= 10'd10;
            reg_cfar_guard_cells    <= 4'd4;
            reg_cfar_train_cells    <= 5'd8;
            reg_cfar_scale_factor   <= 8'd128;
            reg_cfar_os_rank_k      <= 4'd4;
        end else begin
            reg_start                <= 1'b0;
            reg_wt_write_strobe      <= 1'b0;
            reg_wt_dma_start_strobe  <= 1'b0;

            if (axi_w_hs && aw_pending) begin
                if (!is_sparse_csr_write) begin
                case (aw_addr_q[7:0])
                    8'h00: if (wmask[0]) begin
                        reg_enable     <= s_axil_wdata[0];
                        reg_soft_reset <= s_axil_wdata[1];
                        reg_start      <= s_axil_wdata[2];
                        reg_abort      <= s_axil_wdata[3];
                    end
                    8'h08: reg_frame_src     <= (reg_frame_src     & ~wmask) | (s_axil_wdata & wmask);
                    8'h0C: reg_frame_bytes   <= (reg_frame_bytes   & ~wmask) | (s_axil_wdata & wmask);
                    8'h10: reg_out_dst       <= (reg_out_dst       & ~wmask) | (s_axil_wdata & wmask);
                    8'h14: reg_out_max_bytes <= (reg_out_max_bytes & ~wmask) | (s_axil_wdata & wmask);
                    8'h18: begin
                        if (wmask[0]) reg_conf_thr      <= s_axil_wdata[DATA_WIDTH-1:0];
                        if (wmask[1]) reg_iou_thr       <= s_axil_wdata[DATA_WIDTH-1+8 -: DATA_WIDTH];
                        if (wmask[2]) reg_top_k         <= s_axil_wdata[23:16];
                        if (wmask[2]) reg_temporal_mode <= s_axil_wdata[26:24];
                    end
                    8'h1C: if (wmask[0]) reg_history_depth <= s_axil_wdata[3:0];
                    8'h20: begin
                        reg_wt_csr          <= (reg_wt_csr & ~wmask) | (s_axil_wdata & wmask);
                        if (|wmask) reg_wt_write_strobe <= 1'b1;
                    end
                    8'h24: reg_wt_data       <= (reg_wt_data       & ~wmask) | (s_axil_wdata & wmask);
                    8'h28: reg_wt_dma_src    <= (reg_wt_dma_src    & ~wmask) | (s_axil_wdata & wmask);
                    8'h2C: begin
                        reg_wt_dma_cnt          <= (reg_wt_dma_cnt & ~wmask) | (s_axil_wdata & wmask);
                        if (|wmask) reg_wt_dma_start_strobe <= 1'b1;
                    end
                    8'h38: if (wmask[0]) reg_irq_status <= reg_irq_status & ~s_axil_wdata[1:0];
                    8'h3C: if (wmask[0]) reg_irq_enable <= s_axil_wdata[1:0];
                    8'h40: if (wmask[0]) reg_radar_mode         <= s_axil_wdata[0];
                    8'h44: begin
                        if (wmask[0]) reg_range_fft_size  <= s_axil_wdata[11:0];
                        if (wmask[0]) reg_range_fft_radix <= s_axil_wdata[14:12];
                        if (wmask[0]) reg_window_type     <= s_axil_wdata[17:16];
                        if (wmask[0]) reg_window_enable   <= s_axil_wdata[18];
                        if (wmask[0]) reg_beamforming_enable <= s_axil_wdata[19];
                        if (wmask[0]) reg_angle_fft_enable  <= s_axil_wdata[20];
                        if (wmask[0]) reg_magnitude_enable  <= s_axil_wdata[21];
                        if (wmask[0]) reg_log_output        <= s_axil_wdata[22];
                        if (wmask[0]) reg_cfar_en           <= s_axil_wdata[23];
                    end
                    8'h48: begin
                        if (wmask[0]) reg_cfar_threshold_db <= s_axil_wdata[9:0];
                        if (wmask[0]) reg_cfar_guard_cells  <= s_axil_wdata[13:10];
                        if (wmask[0]) reg_cfar_train_cells <= s_axil_wdata[18:14];
                        if (wmask[0]) reg_cfar_mode        <= s_axil_wdata[20:19];
                        if (wmask[0]) reg_cfar_scale_factor<= s_axil_wdata[28:21];
                        if (wmask[0]) reg_cfar_os_rank_k  <= s_axil_wdata[31:28];
                    end
                    default: ;
                endcase
                end
            end
        end
    end

    // =========================================================================
    // AXI Domain: CSR Read Decode
    // =========================================================================
    logic [31:0] rdata_q;
    logic        rvalid_q;
    logic        top_busy_axi, top_done_axi, top_err;
    logic [15:0] det_count_axi;
    logic        top_done_axi_prev;
    logic        top_done_rising;

    always_ff @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n) begin
            s_axil_arready <= 1'b1;
            rvalid_q       <= 1'b0;
            rdata_q        <= '0;
            s_axil_rresp   <= 2'b00;
            sparse_ar_pending <= 1'b0;
        end else begin
            if (axi_ar_hs) begin
                s_axil_arready <= 1'b0;
                if (is_sparse_csr_read) begin
                    sparse_ar_pending <= 1'b1;
                end else begin
                    rvalid_q <= 1'b1;
                    case (s_axil_araddr[7:0])
                        8'h00: rdata_q <= {28'h0, reg_abort, reg_start, reg_soft_reset, reg_enable};
                        8'h04: rdata_q <= {det_count_axi, 5'h0, top_err, top_done_axi, top_busy_axi};
                        8'h08: rdata_q <= reg_frame_src;
                        8'h0C: rdata_q <= reg_frame_bytes;
                        8'h10: rdata_q <= reg_out_dst;
                        8'h14: rdata_q <= reg_out_max_bytes;
                        8'h18: rdata_q <= {5'h0, reg_temporal_mode, reg_top_k,
                                           {{(8-DATA_WIDTH){1'b0}}, reg_iou_thr},
                                           {{(8-DATA_WIDTH){1'b0}}, reg_conf_thr}};
                        8'h1C: rdata_q <= {28'h0, reg_history_depth};
                        8'h20: rdata_q <= reg_wt_csr;
                        8'h24: rdata_q <= reg_wt_data;
                        8'h28: rdata_q <= reg_wt_dma_src;
                        8'h2C: rdata_q <= reg_wt_dma_cnt;
                        8'h30: rdata_q <= reg_perf_cycles_axi;
                        8'h34: rdata_q <= reg_perf_frames_axi;
                        8'h38: rdata_q <= {30'h0, reg_irq_status};
                        8'h3C: rdata_q <= {30'h0, reg_irq_enable};
                        8'h40: rdata_q <= {31'h0, reg_radar_mode};
                        8'h44: rdata_q <= {8'h0, reg_cfar_en, reg_log_output, reg_magnitude_enable,
                                              reg_angle_fft_enable, reg_beamforming_enable,
                                              reg_window_enable, reg_window_type,
                                              reg_range_fft_radix, reg_range_fft_size};
                        8'h48: rdata_q <= {reg_cfar_os_rank_k, reg_cfar_scale_factor, reg_cfar_mode,
                                              reg_cfar_train_cells, reg_cfar_guard_cells,
                                              reg_cfar_threshold_db};
                        default: rdata_q <= '0;
                    endcase
                end
            end
            // Sparse CSR read completion (CDC response arrived)
            if (sparse_rdata_cdc_valid && sparse_ar_pending) begin
                rdata_q <= sparse_rdata_cdc;
                rvalid_q <= 1'b1;
                sparse_ar_pending <= 1'b0;
                s_axil_rresp <= 2'b00;
            end
            if (rvalid_q && s_axil_rready) begin
                rvalid_q       <= 1'b0;
                s_axil_arready <= 1'b1;
            end
        end
    end

    assign s_axil_rdata  = rdata_q;
    assign s_axil_rvalid = rvalid_q;

    // =========================================================================
    // AXI Domain: Frame DMA Reader
    // =========================================================================
    typedef enum logic [2:0] {
        F_IDLE, F_REQ, F_DATA, F_ADVANCE, F_DONE, F_ERR
    } fdma_t;
    fdma_t fstate;

    logic [31:0] fdma_addr_cur;
    logic [31:0] fdma_bytes_left;
    logic [$clog2(IN_BURST_BEATS+1)-1:0] fdma_beat_in_burst;
    logic [$clog2(BEATS_PER_TOKEN+1)-1:0] fdma_beat_in_tok;

    // Frame buffer BRAM write data / control (axi_clk domain)
    logic [MAX_C*DATA_WIDTH-1:0] frame_buf_din;
    logic [$clog2(FRAME_TOKENS)-1:0] frame_wr_tok;
    logic frame_buf_we;
    logic frame_buf_full, frame_buf_full_sync;
    logic compute_done_axi;

    // Token-assembly: pack 8 AXI beats into one 512-bit token
    logic signed [DATA_WIDTH-1:0] tok_assembly [0:MAX_C-1];

    always_ff @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n) begin
            fstate              <= F_IDLE;
            fdma_addr_cur       <= '0;
            fdma_bytes_left     <= '0;
            fdma_beat_in_burst  <= '0;
            fdma_beat_in_tok    <= '0;
            frame_wr_tok        <= '0;
            frame_buf_full      <= 1'b0;
            frame_buf_we        <= 1'b0;
            for (int c = 0; c < MAX_C; c++) tok_assembly[c] <= '0;
        end else begin
            frame_buf_we <= 1'b0;  // self-clearing

            case (fstate)
                F_IDLE: begin
                    frame_buf_full <= 1'b0;
                    if (reg_start && reg_enable && reg_frame_bytes != '0) begin
                        fdma_addr_cur   <= reg_frame_src;
                        fdma_bytes_left <= reg_frame_bytes;
                        frame_wr_tok    <= '0;
                        fstate          <= F_REQ;
                    end
                end
                F_REQ: if (m_axi_in_arready) begin
                    fdma_beat_in_burst <= '0;
                    fstate             <= F_DATA;
                end
                F_DATA: if (m_axi_in_rvalid) begin
                    if (m_axi_in_rresp != 2'b00) fstate <= F_ERR;
                    else begin
                        for (int b = 0; b < AXI_DW/DATA_WIDTH; b++) begin
                            tok_assembly[fdma_beat_in_tok * (AXI_DW/DATA_WIDTH) + b]
                                <= m_axi_in_rdata[b*DATA_WIDTH +: DATA_WIDTH];
                        end
                        if (fdma_beat_in_tok == BEATS_PER_TOKEN-1) begin
                            // Write assembled token to BRAM
                            frame_buf_we <= 1'b1;
                            for (int c = 0; c < MAX_C; c++)
                                frame_buf_din[c*DATA_WIDTH +: DATA_WIDTH] <=
                                    (c < (BEATS_PER_TOKEN-1)*(AXI_DW/DATA_WIDTH)) ?
                                        tok_assembly[c] :
                                        ((c < MAX_C) ? m_axi_in_rdata[
                                            (c - (BEATS_PER_TOKEN-1)*(AXI_DW/DATA_WIDTH))*DATA_WIDTH
                                            +: DATA_WIDTH] : '0);
                            frame_wr_tok     <= frame_wr_tok + 1;
                            fdma_beat_in_tok <= '0;
                        end else begin
                            fdma_beat_in_tok <= fdma_beat_in_tok + 1;
                        end

                        if (m_axi_in_rlast) begin
                            fstate <= F_ADVANCE;
                        end
                    end
                end
                F_ADVANCE: begin
                    fdma_addr_cur   <= fdma_addr_cur + IN_BURST_BEATS * (AXI_DW/8);
                    fdma_bytes_left <= fdma_bytes_left - IN_BURST_BEATS * (AXI_DW/8);
                    if (fdma_bytes_left <= IN_BURST_BEATS * (AXI_DW/8)) begin
                        frame_buf_full <= 1'b1;
                        fstate         <= F_DONE;
                    end else begin
                        fstate <= F_REQ;
                    end
                end
                F_DONE: begin
                    if (compute_done_axi) fstate <= F_IDLE;  // wait for core consumption
                end
                F_ERR: begin
                    if (reg_abort || reg_soft_reset) fstate <= F_IDLE;
                end
                default: fstate <= F_IDLE;
            endcase
        end
    end

    assign m_axi_in_arid    = '0;
    assign m_axi_in_araddr  = fdma_addr_cur;
    assign m_axi_in_arlen   = 8'(IN_BURST_BEATS - 1);
    assign m_axi_in_arsize  = 3'd3;     // 8 bytes per beat
    assign m_axi_in_arburst = 2'b01;    // INCR
    assign m_axi_in_arvalid = (fstate == F_REQ);
    assign m_axi_in_rready  = (fstate == F_DATA);

    // =========================================================================
    // CDC Group A: CSR Config (axi -> core)
    // =========================================================================
    // Config signals are quasi-static (change only between frames).
    // cdc_sync_vec is safe: temporary bit-skew in multi-bit bundles is harmless.
    logic cfg_enable_core;
    logic [2:0] cfg_temporal_mode_core;
    logic [3:0] cfg_history_depth_core;
    logic signed [DATA_WIDTH-1:0] cfg_conf_thr_core, cfg_iou_thr_core;
    logic [7:0] cfg_top_k_core;

    cdc_sync_single u_cfg_enable (
        .clk_src(axi_clk), .rst_n_src(axi_rst_n),
        .clk_dst(core_clk), .rst_n_dst(core_rst_n),
        .data_in(reg_enable), .data_out(cfg_enable_core)
    );

    cdc_sync_vec #(.WIDTH(3)) u_cfg_temporal_mode (
        .clk_src(axi_clk), .rst_n_src(axi_rst_n),
        .clk_dst(core_clk), .rst_n_dst(core_rst_n),
        .data_in(reg_temporal_mode), .data_out(cfg_temporal_mode_core)
    );

    cdc_sync_vec #(.WIDTH(4)) u_cfg_history_depth (
        .clk_src(axi_clk), .rst_n_src(axi_rst_n),
        .clk_dst(core_clk), .rst_n_dst(core_rst_n),
        .data_in(reg_history_depth), .data_out(cfg_history_depth_core)
    );

    cdc_sync_vec #(.WIDTH(DATA_WIDTH)) u_cfg_conf_thr (
        .clk_src(axi_clk), .rst_n_src(axi_rst_n),
        .clk_dst(core_clk), .rst_n_dst(core_rst_n),
        .data_in(reg_conf_thr), .data_out(cfg_conf_thr_core)
    );

    cdc_sync_vec #(.WIDTH(DATA_WIDTH)) u_cfg_iou_thr (
        .clk_src(axi_clk), .rst_n_src(axi_rst_n),
        .clk_dst(core_clk), .rst_n_dst(core_rst_n),
        .data_in(reg_iou_thr), .data_out(cfg_iou_thr_core)
    );

    cdc_sync_vec #(.WIDTH(8)) u_cfg_top_k (
        .clk_src(axi_clk), .rst_n_src(axi_rst_n),
        .clk_dst(core_clk), .rst_n_dst(core_rst_n),
        .data_in(reg_top_k), .data_out(cfg_top_k_core)
    );

    // Soft reset pulse synchronizer (axi -> core)
    logic soft_reset_pulse_core;
    cdc_sync_pulse u_soft_reset (
        .clk_src(axi_clk), .rst_n_src(axi_rst_n),
        .clk_dst(core_clk), .rst_n_dst(core_rst_n),
        .pulse_in(reg_soft_reset),
        .pulse_out(soft_reset_pulse_core),
        .busy()
    );

    // =========================================================================
    // CDC Group B: Frame Buffer (true_dual_port_bram, SAME_CLK=0)
    // =========================================================================
    // Port A: axi_clk (DMA write)  / Port B: core_clk (compute read)
    localparam int FRAME_BUF_DATA_W = MAX_C * DATA_WIDTH;  // 512
    localparam int FRAME_BUF_ADDR_W = $clog2(FRAME_TOKENS); // 12

    logic [FRAME_BUF_DATA_W-1:0] frame_buf_dout;

    true_dual_port_bram #(
        .DATA_W(FRAME_BUF_DATA_W),
        .ADDR_W(FRAME_BUF_ADDR_W),
        .DEPTH(FRAME_TOKENS),
        .MAX_DEPTH(FRAME_TOKENS),
        .BYTE_W(FRAME_BUF_DATA_W),
        .PIPELINE_OUT(0),
        .SAME_CLK(0)
    ) u_frame_buf_bram (
        .clk_a(axi_clk),
        .rst_n_a(axi_rst_n),
        .en_a(1'b1),
        .we_a(frame_buf_we),
        .addr_a(frame_wr_tok),
        .din_a(frame_buf_din),
        .be_a({(FRAME_BUF_DATA_W/FRAME_BUF_DATA_W){1'b1}}),
        .dout_a(),
        .clk_b(core_clk),
        .rst_n_b(core_rst_n),
        .en_b(1'b1),
        .we_b(1'b0),
        .addr_b(frame_rd_tok),
        .din_b('0),
        .be_b('0),
        .dout_b(frame_buf_dout),
        .collision_a(),
        .collision_b()
    );

    // frame_buf_full CDC (axi -> core)
    logic frame_buf_full_core;

    cdc_sync_single u_frame_buf_full (
        .clk_src(axi_clk), .rst_n_src(axi_rst_n),
        .clk_dst(core_clk), .rst_n_dst(core_rst_n),
        .data_in(frame_buf_full), .data_out(frame_buf_full_core)
    );

    // compute_done CDC (core -> axi)
    logic compute_done_core;

    cdc_sync_single u_compute_done (
        .clk_src(core_clk), .rst_n_src(core_rst_n),
        .clk_dst(axi_clk), .rst_n_dst(axi_rst_n),
        .data_in(compute_done_core), .data_out(compute_done_axi)
    );

    // =========================================================================
    // Core Domain: Compute Drive FSM
    // =========================================================================
    typedef enum logic [1:0] { C_IDLE, C_STREAM, C_DRAIN } cstate_t;
    cstate_t cstate;
    logic [$clog2(FRAME_TOKENS)-1:0] frame_rd_tok;
    logic                             unet_in_valid, unet_in_ready, unet_in_last;
    logic signed [DATA_WIDTH-1:0]     unet_in_data [0:MAX_C-1];
    logic                             top_done_core;

    always_ff @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            cstate       <= C_IDLE;
            frame_rd_tok <= '0;
            compute_done_core <= 1'b0;
        end else begin
            case (cstate)
                C_IDLE: begin
                    compute_done_core <= 1'b0;
                    if (frame_buf_full_core) begin
                        cstate       <= C_STREAM;
                        frame_rd_tok <= '0;
                    end
                end
                C_STREAM: if (unet_in_valid && unet_in_ready) begin
                    if (frame_rd_tok == FRAME_TOKENS-1) cstate <= C_DRAIN;
                    else                                frame_rd_tok <= frame_rd_tok + 1;
                end
                C_DRAIN: if (top_done_core) begin
                    compute_done_core <= 1'b1;
                    cstate       <= C_IDLE;
                end
                default: cstate <= C_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Core Domain: Radar Processing + RA-to-CNN Bridge
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] radar_cnn_data [MAX_C-1:0];
    logic                         radar_cnn_valid;
    logic                         radar_cnn_last;
    logic                         radar_cnn_ready;

    logic signed [31:0] radar_ra_mag [3:0];
    logic [9:0]         radar_ra_range_idx;
    logic signed [DATA_WIDTH-1:0] radar_ra_mag_q [3:0];
    logic [1:0]         radar_ra_angle_idx;
    logic               radar_ra_valid;
    logic               radar_ra_last;
    logic               radar_ra_ready;

    if (EN_RADAR_PROCESSING) begin : gen_radar_processing
        // Truncate 32-bit radar output to DATA_WIDTH for the CNN bridge
        always_comb begin
            for (int a = 0; a < 4; a++)
                radar_ra_mag_q[a] = radar_ra_mag[a][DATA_WIDTH-1:0];
        end

        radar_processing_top #(
            .NUM_ANTENNAS   (4),
            .NUM_BEAMS      (5),
            .MAX_RANGE_BINS (1024),
            .MAX_ANGLE_BINS (4),
            .DATA_WIDTH     (32),
            .FRAC_WIDTH     (16),
            .MAX_GUARD      (8),
            .MAX_TRAIN      (32)
        ) u_radar (
            .clk                (core_clk),
            .rst_n              (core_rst_n && !soft_reset_pulse_core),

            .range_fft_size     (reg_range_fft_size),
            .range_fft_radix    (reg_range_fft_radix),
            .window_type        (reg_window_type),
            .window_enable      (reg_window_enable),
            .beamforming_enable (reg_beamforming_enable),
            .angle_fft_enable   (reg_angle_fft_enable),
            .magnitude_enable   (reg_magnitude_enable),
            .log_output         (reg_log_output),

            .cfar_threshold_db  (reg_cfar_threshold_db),
            .cfar_guard_cells   (reg_cfar_guard_cells),
            .cfar_train_cells   (reg_cfar_train_cells),
            .cfar_en            (reg_cfar_en),
            .cfar_mode          (reg_cfar_mode),
            .cfar_scale_factor  (reg_cfar_scale_factor),
            .cfar_os_rank_k     (reg_cfar_os_rank_k),

            .adc_re             (adc_re),
            .adc_im             (adc_im),
            .adc_valid          (adc_valid),
            .adc_last           (adc_last),
            .adc_sample_idx     (adc_sample_idx),
            .adc_ready          (adc_ready),

            .fft_start          (fft_start),
            .fft_size           (fft_size),
            .fft_radix_out      (fft_radix_out),
            .fft_inverse        (fft_inverse),
            .fft_done           (fft_done),
            .fft_busy           (fft_busy),

            .detection_value    (),
            .detection_range    (),
            .detection_angle    (),
            .detection_valid    (),
            .detection_last     (),
            .detection_ready    (1'b1),

            .ra_mag             (radar_ra_mag),
            .ra_range_idx       (radar_ra_range_idx),
            .ra_angle_idx       (radar_ra_angle_idx),
            .ra_valid           (radar_ra_valid),
            .ra_last            (radar_ra_last),
            .ra_ready           (radar_ra_ready),

            .det_out_valid      (),
            .det_out_data       (),
            .det_out_ready      (1'b1)
        );
    end else begin : gen_radar_bypass
        // Radar processing excluded — use UART receiver for pre-computed
        // RA maps from AWR1843AOP (921600 baud, 16-bit pixels).
        // UART output feeds the ra_map_to_cnn_bridge directly.

        // 32-bit radar outputs unused in UART mode
        assign radar_ra_mag[0] = '0;
        assign radar_ra_mag[1] = '0;
        assign radar_ra_mag[2] = '0;
        assign radar_ra_mag[3] = '0;
        assign radar_ra_angle_idx = '0;

        // ADC / FFT ports unused
        assign adc_ready    = 1'b0;
        assign fft_start    = 1'b0;
        assign fft_size     = '0;
        assign fft_radix_out = '0;
        assign fft_inverse  = 1'b0;

        ra_map_uart_receiver #(
            .CLK_FREQ_HZ    (200_000_000),
            .DATA_WIDTH     (DATA_WIDTH),
            .MAX_RANGE_BINS (1024),
            .MAX_ANGLE_BINS (4)
        ) u_uart_ra (
            .clk            (core_clk),
            .rst_n          (core_rst_n && !soft_reset_pulse_core),

            .uart_rx        (uart_rx),

            // Fixed config: 256 range bins, 4 angle bins, 921600 baud, 16-bit
            .cfg_update     (1'b1),
            .cfg_range_bins (12'd256),
            .cfg_angle_bins (4'd4),
            .cfg_baud_div   (16'd14),
            .cfg_pixel_16bit(1'b1),

            .frame_active   (),
            .frame_count    (),
            .uart_frame_err (),

            .ra_mag         (radar_ra_mag_q),
            .ra_range_idx   (radar_ra_range_idx),
            .ra_valid       (radar_ra_valid),
            .ra_last        (radar_ra_last),
            .ra_ready       (radar_ra_ready)
        );
    end

    // RA Map -> CNN bridge
    ra_map_to_cnn_bridge #(
        .RADAR_RANGE_BINS (1024),
        .RADAR_ANGLE_BINS (4),
        .FEATURE_H        (FEATURE_H),
        .FEATURE_W        (FEATURE_W),
        .MAX_C            (MAX_C),
        .DATA_WIDTH       (DATA_WIDTH),
        .FRAC_WIDTH       (FRAC_WIDTH)
    ) u_ra_bridge (
        .clk            (core_clk),
        .rst_n          (core_rst_n && !soft_reset_pulse_core),

        .ra_mag         (radar_ra_mag_q),
        .ra_range_idx   (radar_ra_range_idx),
        .ra_valid       (radar_ra_valid),
        .ra_last        (radar_ra_last),
        .ra_ready       (radar_ra_ready),

        .cnn_data       (radar_cnn_data),
        .cnn_valid      (radar_cnn_valid),
        .cnn_frame_last (radar_cnn_last),
        .cnn_ready      (radar_cnn_ready)
    );

    // =========================================================================
    // Core Domain: Input Mux (AXI DMA vs Radar)
    // =========================================================================
    always_comb begin
        if (reg_radar_mode) begin
            unet_in_valid = radar_cnn_valid;
            unet_in_last  = radar_cnn_last;
            for (int c = 0; c < MAX_C; c++)
                unet_in_data[c] = radar_cnn_data[c];
            radar_cnn_ready = unet_in_ready;
        end else begin
            unet_in_valid = (cstate == C_STREAM);
            unet_in_last  = (cstate == C_STREAM) && (frame_rd_tok == FRAME_TOKENS-1);
            for (int c = 0; c < MAX_C; c++)
                unet_in_data[c] = frame_buf_dout[c*DATA_WIDTH +: DATA_WIDTH];
            radar_cnn_ready = 1'b0;
        end
    end

    // =========================================================================
    // AXI Domain: Weight Controller
    // =========================================================================
    logic                              wt_valid;
    logic [31:0]                       wt_addr;
    logic signed [DATA_WIDTH-1:0]      wt_data;
    logic [3:0]                        wt_layer;

    erodnet_weight_controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(32),
        .LAYER_WIDTH(4),
        .AXI_DATA_W(AXI_DW),
        .AXI_ADDR_W(AXI_AW),
        .BURST_LEN(AXI_BURST_LEN)
    ) u_weight_ctrl (
        .clk(axi_clk), .rst_n(axi_rst_n),
        .csr_weight_addr  (reg_wt_csr[26:0]),
        .csr_weight_data  (reg_wt_data[DATA_WIDTH-1:0]),
        .csr_weight_layer (reg_wt_csr[30:27]),
        .csr_weight_write (reg_wt_write_strobe),
        .csr_auto_inc_en  (reg_wt_csr[31]),
        .csr_weight_count (reg_wt_dma_cnt),
        .csr_weight_done_clear(reg_soft_reset),
        .csr_dma_src_addr (reg_wt_dma_src),
        .csr_dma_weight_count(reg_wt_dma_cnt),
        .csr_dma_layer    (reg_wt_csr[30:27]),
        .csr_dma_dst_addr (reg_wt_csr[26:0]),
        .csr_dma_start    (reg_wt_dma_start_strobe),
        .m_axi_wt_araddr  (m_axi_wt_araddr),
        .m_axi_wt_arlen   (m_axi_wt_arlen),
        .m_axi_wt_arsize  (m_axi_wt_arsize),
        .m_axi_wt_arburst (m_axi_wt_arburst),
        .m_axi_wt_arvalid (m_axi_wt_arvalid),
        .m_axi_wt_arready (m_axi_wt_arready),
        .m_axi_wt_rdata   (m_axi_wt_rdata),
        .m_axi_wt_rresp   (m_axi_wt_rresp),
        .m_axi_wt_rvalid  (m_axi_wt_rvalid),
        .m_axi_wt_rlast   (m_axi_wt_rlast),
        .m_axi_wt_rready  (m_axi_wt_rready),
        .weight_valid     (wt_valid),
        .weight_addr      (wt_addr),
        .weight_data      (wt_data),
        .weight_layer_sel (wt_layer)
    );

    // =========================================================================
    // CDC Group C: Weight Path (axi -> core)
    // =========================================================================
    localparam int WT_CDC_W = 32 + DATA_WIDTH + 4;  // addr + data + layer_sel

    logic [WT_CDC_W-1:0] wt_packed;
    assign wt_packed = {wt_addr, wt_data, wt_layer};

    logic [WT_CDC_W-1:0] wt_cdc_data;
    logic                wt_cdc_valid, wt_cdc_ready;

    cdc_sync_data #(.DATA_WIDTH(WT_CDC_W)) u_wt_cdc (
        .src_clk(axi_clk), .src_rst_n(axi_rst_n),
        .dst_clk(core_clk), .dst_rst_n(core_rst_n),
        .data_in(wt_packed),
        .valid_in(wt_valid),
        .ready_out(wt_cdc_ready),
        .data_out(wt_cdc_data),
        .valid_out(wt_cdc_valid),
        .ready_in(1'b1)
    );

    // Unpack weight CDC output
    logic        wt_core_valid;
    logic [31:0] wt_core_addr;
    logic signed [DATA_WIDTH-1:0] wt_core_data;
    logic [3:0]  wt_core_layer;

    assign wt_core_valid = wt_cdc_valid;
    assign wt_core_addr  = wt_cdc_data[WT_CDC_W-1 -: 32];
    assign wt_core_data  = wt_cdc_data[DATA_WIDTH+4-1 -: DATA_WIDTH];
    assign wt_core_layer = wt_cdc_data[3:0];

    // =========================================================================
    // CDC Group D: Sparse CSR Request (axi -> core)
    // =========================================================================
    localparam int SPARSE_REQ_W = 12 + 32 + 1;  // addr + wdata + we

    logic [SPARSE_REQ_W-1:0] sparse_req_packed;
    assign sparse_req_packed = {sparse_csr_req_addr, sparse_csr_req_wdata, sparse_csr_req_we};

    logic [SPARSE_REQ_W-1:0] sparse_req_cdc_data;
    logic                     sparse_req_cdc_valid, sparse_req_cdc_ready;

    cdc_sync_data #(.DATA_WIDTH(SPARSE_REQ_W)) u_sparse_req_cdc (
        .src_clk(axi_clk), .src_rst_n(axi_rst_n),
        .dst_clk(core_clk), .dst_rst_n(core_rst_n),
        .data_in(sparse_req_packed),
        .valid_in(sparse_csr_req_valid),
        .ready_out(sparse_req_cdc_ready),
        .data_out(sparse_req_cdc_data),
        .valid_out(sparse_req_cdc_valid),
        .ready_in(sparse_csr_req_ready_core)
    );

    // Sparse CSR bridge (axi_clk) drives request on rising edge of is_sparse_csr_write/read
    always_ff @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n) begin
            sparse_csr_req_valid <= 1'b0;
            sparse_csr_req_addr  <= '0;
            sparse_csr_req_wdata <= '0;
            sparse_csr_req_we    <= 1'b0;
        end else begin
            sparse_csr_req_valid <= 1'b0;  // single-cycle pulse
            sparse_csr_req_we    <= 1'b0;
            if (is_sparse_csr_write) begin
                sparse_csr_req_valid <= 1'b1;
                sparse_csr_req_addr <= aw_addr_q[11:0] - 12'hB00;
                sparse_csr_req_wdata <= s_axil_wdata;
                sparse_csr_req_we   <= 1'b1;
            end else if (is_sparse_csr_read) begin
                sparse_csr_req_valid <= 1'b1;
                sparse_csr_req_addr <= s_axil_araddr[11:0] - 12'hB00;
                sparse_csr_req_wdata <= '0;
                sparse_csr_req_we   <= 1'b0;
            end
        end
    end

    // Unpack sparse CSR request on core side
    logic        sparse_csr_req_valid_core;
    logic [11:0] sparse_csr_req_addr_core;
    logic [31:0] sparse_csr_req_wdata_core;
    logic        sparse_csr_req_we_core;
    logic        sparse_csr_req_ready_core;
    logic [31:0] sparse_csr_req_rdata_core;

    assign sparse_csr_req_valid_core = sparse_req_cdc_valid;
    assign sparse_csr_req_addr_core  = sparse_req_cdc_data[SPARSE_REQ_W-1 -: 12];
    assign sparse_csr_req_wdata_core = sparse_req_cdc_data[33:2];
    assign sparse_csr_req_we_core    = sparse_req_cdc_data[0];

    // Sparse CSR response CDC (core -> axi)
    // Generate a valid pulse when response is ready
    logic sparse_rsp_valid_core;
    logic [31:0] sparse_rsp_data_core;

    always_ff @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            sparse_rsp_valid_core <= 1'b0;
            sparse_rsp_data_core  <= '0;
        end else begin
            sparse_rsp_valid_core <= 1'b0;  // self-clearing
            if (sparse_csr_req_ready_core && sparse_csr_req_valid_core) begin
                sparse_rsp_data_core  <= sparse_csr_req_rdata_core;
                sparse_rsp_valid_core <= 1'b1;
            end
        end
    end

    cdc_sync_data #(.DATA_WIDTH(32)) u_sparse_rsp_cdc (
        .src_clk(core_clk), .src_rst_n(core_rst_n),
        .dst_clk(axi_clk), .dst_rst_n(axi_rst_n),
        .data_in(sparse_rsp_data_core),
        .valid_in(sparse_rsp_valid_core),
        .ready_out(),
        .data_out(sparse_rdata_cdc),
        .valid_out(sparse_rdata_cdc_valid),
        .ready_in(1'b1)
    );

    // =========================================================================
    // Core Domain: UNet Core Instance
    // =========================================================================
    logic                              det_valid_top;
    logic [15:0]                       det_h_top, det_w_top;
    logic signed [DATA_WIDTH-1:0]      det_conf_top;
    logic [15:0]                       det_width_top, det_height_top;
    logic [NUM_CLASSES-1:0][DATA_WIDTH-1:0] det_class_scores_top;
    logic [1:0]                        det_class_id_top;
    logic                              det_last_top;
    logic [15:0]                       det_count_top;
    logic [31:0]                       perf_cycle_count_core, perf_frame_count_core;
    logic                              top_busy_core;

    e_rodnet_unet_top #(
        .FEATURE_H(FEATURE_H), .FEATURE_W(FEATURE_W),
        .TILE_H(TILE_H), .TILE_W(TILE_W),
        .IN_CHANNELS(IN_CHANNELS),
        .NUM_CLASSES(NUM_CLASSES), .NUM_ANCHORS(NUM_ANCHORS),
        .TOP_K(TOP_K), .HISTORY_DEPTH(HISTORY_DEPTH),
        .DATA_WIDTH(DATA_WIDTH), .FRAC_WIDTH(FRAC_WIDTH)
    ) u_unet (
        .clk(core_clk), .rst_n(core_rst_n && !soft_reset_pulse_core),
        .cfg_enable        (cfg_enable_core),
        .cfg_temporal_mode (cfg_temporal_mode_core),
        .cfg_history_depth (cfg_history_depth_core),
        .cfg_conf_threshold(cfg_conf_thr_core),
        .cfg_iou_threshold (cfg_iou_thr_core),
        .cfg_top_k         (cfg_top_k_core),
        .in_valid          (unet_in_valid),
        .in_frame_last     (unet_in_last),
        .in_data           (unet_in_data),
        .in_ready          (unet_in_ready),
        .weight_valid      (wt_core_valid),
        .weight_addr       (wt_core_addr),
        .weight_data       (wt_core_data),
        .weight_layer_sel  (wt_core_layer),
        .det_valid_out     (det_valid_top),
        .det_h_idx         (det_h_top),
        .det_w_idx         (det_w_top),
        .det_confidence    (det_conf_top),
        .det_width         (det_width_top),
        .det_height        (det_height_top),
        .det_class_scores  (det_class_scores_top),
        .det_class_id      (det_class_id_top),
        .det_last          (det_last_top),
        .det_count         (det_count_top),
        .det_ready_in      (det_core_ready),
        .perf_cycle_count  (perf_cycle_count_core),
        .perf_frame_count  (perf_frame_count_core),
        .processing_busy   (top_busy_core),
        .processing_done   (top_done_core),
        .sparse_csr_req_valid(sparse_csr_req_valid_core),
        .sparse_csr_req_addr (sparse_csr_req_addr_core),
        .sparse_csr_req_wdata(sparse_csr_req_wdata_core),
        .sparse_csr_req_we   (sparse_csr_req_we_core),
        .sparse_csr_req_rdata(sparse_csr_req_rdata_core),
        .sparse_csr_req_ready(sparse_csr_req_ready_core)
    );

    // =========================================================================
    // CDC Group E: Detection Output (core -> axi)
    // =========================================================================
    // DET_CDC_W = 16(h) + 16(w) + DATA_WIDTH(conf) + 16(width) + 16(height)
    //             + NUM_CLASSES*DATA_WIDTH(scores) + 2(class_id) + 1(last)
    localparam int DET_HDR_W = 64 + DATA_WIDTH;  // h(16)+w(16)+conf(DW)+width(16)+height(16)
    localparam int DET_CDC_W = DET_HDR_W + (NUM_CLASSES * DATA_WIDTH) + 2 + 1;

    logic [DET_CDC_W-1:0] det_packed_cdc;
    logic                 det_core_ready;

    // Pack all detection fields into a flat word
    always_comb begin
        det_packed_cdc = '0;
        det_packed_cdc[15:0]                                                  = det_h_top;
        det_packed_cdc[31:16]                                                 = det_w_top;
        det_packed_cdc[32 +: DATA_WIDTH]                                      = det_conf_top;
        det_packed_cdc[32 + DATA_WIDTH +: 16]                                 = det_width_top;
        det_packed_cdc[48 + DATA_WIDTH +: 16]                                 = det_height_top;
        for (int c = 0; c < NUM_CLASSES; c++)
            det_packed_cdc[DET_HDR_W + c*DATA_WIDTH +: DATA_WIDTH]           = det_class_scores_top[c];
        det_packed_cdc[DET_HDR_W + NUM_CLASSES*DATA_WIDTH +: 2]              = det_class_id_top;
        det_packed_cdc[DET_HDR_W + NUM_CLASSES*DATA_WIDTH + 2]               = det_last_top;
    end

    logic [DET_CDC_W-1:0] det_cdc_data;
    logic                  det_cdc_valid, det_cdc_ready;

    cdc_sync_data #(.DATA_WIDTH(DET_CDC_W)) u_det_cdc (
        .src_clk(core_clk), .src_rst_n(core_rst_n),
        .dst_clk(axi_clk), .dst_rst_n(axi_rst_n),
        .data_in(det_packed_cdc),
        .valid_in(det_valid_top && det_core_ready),
        .ready_out(det_core_ready),
        .data_out(det_cdc_data),
        .valid_out(det_cdc_valid),
        .ready_in(!det_fifo_full)
    );

    // =========================================================================
    // CDC Group F: Status & Performance Counters (core -> axi)
    // =========================================================================
    cdc_sync_single u_top_busy (
        .clk_src(core_clk), .rst_n_src(core_rst_n),
        .clk_dst(axi_clk), .rst_n_dst(axi_rst_n),
        .data_in(top_busy_core), .data_out(top_busy_axi)
    );

    cdc_sync_single u_top_done (
        .clk_src(core_clk), .rst_n_src(core_rst_n),
        .clk_dst(axi_clk), .rst_n_dst(axi_rst_n),
        .data_in(top_done_core), .data_out(top_done_axi)
    );

    // Edge detect for top_done_axi (IRQ trigger)
    always_ff @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n) top_done_axi_prev <= 1'b0;
        else            top_done_axi_prev <= top_done_axi;
    end
    assign top_done_rising = top_done_axi && !top_done_axi_prev;

    cdc_sync_counter #(.WIDTH(16)) u_det_count (
        .clk_src(core_clk), .rst_n_src(core_rst_n),
        .clk_dst(axi_clk), .rst_n_dst(axi_rst_n),
        .count_in(det_count_top), .count_out(det_count_axi)
    );

    cdc_sync_counter #(.WIDTH(32)) u_perf_cycles (
        .clk_src(core_clk), .rst_n_src(core_rst_n),
        .clk_dst(axi_clk), .rst_n_dst(axi_rst_n),
        .count_in(perf_cycle_count_core), .count_out(reg_perf_cycles_axi)
    );

    cdc_sync_counter #(.WIDTH(32)) u_perf_frames (
        .clk_src(core_clk), .rst_n_src(core_rst_n),
        .clk_dst(axi_clk), .rst_n_dst(axi_rst_n),
        .count_in(perf_frame_count_core), .count_out(reg_perf_frames_axi)
    );

    // =========================================================================
    // AXI Domain: Detection Output DMA Writer
    // =========================================================================
    // Detection record after CDC unpack
    logic [15:0]                       det_h_axi, det_w_axi;
    logic signed [DATA_WIDTH-1:0]      det_conf_axi;
    logic [15:0]                       det_width_axi, det_height_axi;
    logic [NUM_CLASSES-1:0][DATA_WIDTH-1:0] det_class_scores_axi;
    logic [1:0]                        det_class_id_axi;
    logic                              det_last_axi;

    // Unpack detection record from CDC output
    always_comb begin
        det_h_axi      = det_cdc_data[15:0];
        det_w_axi      = det_cdc_data[31:16];
        det_conf_axi   = det_cdc_data[32 +: DATA_WIDTH];
        det_width_axi  = det_cdc_data[32 + DATA_WIDTH +: 16];
        det_height_axi = det_cdc_data[48 + DATA_WIDTH +: 16];
        for (int c = 0; c < NUM_CLASSES; c++)
            det_class_scores_axi[c] = det_cdc_data[DET_HDR_W + c*DATA_WIDTH +: DATA_WIDTH];
        det_class_id_axi = det_cdc_data[DET_HDR_W + NUM_CLASSES*DATA_WIDTH +: 2];
        det_last_axi     = det_cdc_data[DET_HDR_W + NUM_CLASSES*DATA_WIDTH + 2];
    end

    // Detection FIFO (synchronous, axi_clk domain) to buffer CDC output
    localparam int DET_FIFO_AW    = $clog2(DET_FIFO_DEPTH);

    typedef struct {
        logic [15:0]                 h_idx;
        logic [15:0]                 w_idx;
        logic signed [DATA_WIDTH-1:0] confidence;
        logic [15:0]                 width;
        logic [15:0]                 height;
        logic [NUM_CLASSES-1:0][DATA_WIDTH-1:0] class_scores;
        logic [1:0]                  class_id;
        logic                        last;
    } det_record_t;

    det_record_t det_fifo [0:DET_FIFO_DEPTH-1];
    logic [DET_FIFO_AW:0] det_fifo_wr_ptr, det_fifo_rd_ptr;
    wire det_fifo_full  = (det_fifo_wr_ptr[DET_FIFO_AW] != det_fifo_rd_ptr[DET_FIFO_AW]) &&
                          (det_fifo_wr_ptr[DET_FIFO_AW-1:0] == det_fifo_rd_ptr[DET_FIFO_AW-1:0]);
    wire det_fifo_empty = (det_fifo_wr_ptr == det_fifo_rd_ptr);
    det_record_t det_fifo_rd_data;

    always_ff @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n) begin
            det_fifo_wr_ptr <= '0;
        end else if (det_cdc_valid && !det_fifo_full) begin
            det_fifo[det_fifo_wr_ptr[DET_FIFO_AW-1:0]] <= det_record_t'{
                h_idx:        det_h_axi,
                w_idx:        det_w_axi,
                confidence:   det_conf_axi,
                width:        det_width_axi,
                height:       det_height_axi,
                class_scores: det_class_scores_axi,
                class_id:     det_class_id_axi,
                last:         det_last_axi
            };
            det_fifo_wr_ptr <= det_fifo_wr_ptr + 1;
        end
    end

    always_ff @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n) begin
            det_fifo_rd_ptr <= '0;
        end else if (!det_fifo_empty && ostate == O_IDLE) begin
            det_fifo_rd_ptr <= det_fifo_rd_ptr + 1;
        end
    end

    assign det_fifo_rd_data = det_fifo[det_fifo_rd_ptr[DET_FIFO_AW-1:0]];

    // synthesis translate_off
    DET_FIFO_OVERFLOW: assert property (
        @(posedge axi_clk) disable iff (!axi_rst_n)
        !(det_cdc_valid && det_fifo_full)
    ) else $error("e_rodnet_unet_axi_top: detection FIFO overflow");
    // synthesis translate_on

    // Output DMA writer FSM (axi_clk domain)
    typedef enum logic [2:0] {
        O_IDLE, O_AW, O_W0, O_W1, O_BRESP, O_DONE, O_ERR
    } odma_t;
    odma_t ostate;
    logic [31:0] odma_addr_cur;
    det_record_t det_txn;
    logic det_txn_valid;

    logic [127:0] det_packed;
    always_comb begin
        det_packed = '0;
        det_packed[15:0]  = det_txn.w_idx;
        det_packed[31:16] = det_txn.h_idx;
        det_packed[47:32] = det_txn.width;
        det_packed[63:48] = det_txn.height;
        det_packed[71:64] = det_txn.confidence;
        det_packed[73:72] = det_txn.class_id;
    end

    always_ff @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n) begin
            ostate        <= O_IDLE;
            odma_addr_cur <= '0;
            det_txn_valid <= 1'b0;
        end else begin
            case (ostate)
                O_IDLE: if (!det_fifo_empty) begin
                    if (reg_out_dst != '0)
                        odma_addr_cur <= odma_addr_cur == '0 ? reg_out_dst : odma_addr_cur;
                    det_txn       <= det_fifo_rd_data;
                    det_txn_valid <= 1'b1;
                    ostate        <= O_AW;
                end else begin
                    det_txn_valid <= 1'b0;
                end
                O_AW: if (m_axi_out_awready) ostate <= O_W0;
                O_W0: if (m_axi_out_wready)  ostate <= O_W1;
                O_W1: if (m_axi_out_wready)  ostate <= O_BRESP;
                O_BRESP: if (m_axi_out_bvalid) begin
                    odma_addr_cur <= odma_addr_cur + 16;
                    ostate <= det_txn.last ? O_DONE : O_IDLE;
                    if (m_axi_out_bresp != 2'b00) ostate <= O_ERR;
                end
                O_DONE: ostate <= O_IDLE;
                O_ERR:  if (reg_abort || reg_soft_reset) ostate <= O_IDLE;
                default: ostate <= O_IDLE;
            endcase
        end
    end

    assign m_axi_out_awid    = '0;
    assign m_axi_out_awaddr  = odma_addr_cur;
    assign m_axi_out_awlen   = 8'd1;        // 2 beats per detection
    assign m_axi_out_awsize  = 3'd3;        // 8B per beat
    assign m_axi_out_awburst = 2'b01;
    assign m_axi_out_awvalid = (ostate == O_AW);
    assign m_axi_out_wdata   = (ostate == O_W0) ? det_packed[63:0]   :
                               (ostate == O_W1) ? det_packed[127:64] : '0;
    assign m_axi_out_wstrb   = '1;
    assign m_axi_out_wlast   = (ostate == O_W1);
    assign m_axi_out_wvalid  = (ostate == O_W0) || (ostate == O_W1);
    assign m_axi_out_bready  = (ostate == O_BRESP);

    // =========================================================================
    // AXI Domain: IRQ
    // =========================================================================
    assign top_err = (fstate == F_ERR);

    always_ff @(posedge axi_clk or negedge axi_rst_n) begin
        if (!axi_rst_n) reg_irq_status <= '0;
        else begin
            if (top_done_rising)                 reg_irq_status[0] <= 1'b1;
            if (top_err || ostate == O_ERR)      reg_irq_status[1] <= 1'b1;
        end
    end
    assign irq = |(reg_irq_status & reg_irq_enable);

endmodule
