// =============================================================================
// e_rodnet_csr.sv
//
// AXI4-Lite slave register file for the e_rodnet CNN IP accelerator.
// Implements the full CSR address map defined in csr_regs.sv.
//
// Features:
//   - AXI4-Lite slave with 1-cycle write latency, 1-cycle read latency
//   - Shadow registers for convformer config: write anytime, apply on cfg_update
//   - Byte-lane WSTRB masking on all RW registers
//   - SLVERR on unmapped address access
//   - DW kernel weight loading passthrough (CSR_WEIGHT_CTRL → dw_wr_* ports)
//   - All performance counters readable via AXI4-Lite
//   - DMA CSR bank (0x300-0x31C) feeds external custom DMA
//   - IRQ: level-high, cleared by W1C on CSR_DMA_STATUS
//
// Shadow register protocol:
//   CPU writes CSR_CFG_* → stored in shadow registers
//   CPU writes CSR_CTRL[15]=1 (cfg_update) → shadows copied to live regs
//   CSR_STATUS[14] (cfg_locked) is asserted for 1 cycle during copy
// =============================================================================

import pkg_config::*;
import csr_regs::*;

module e_rodnet_csr (
    input  logic clk,
    input  logic rst_n,

    // ── AXI4-Lite slave (from CPU / AXI interconnect) ─────────────────────────
    csr_intf.slave s_csr,

    // ── ConvFormer block live configuration outputs ───────────────────────────
    output logic [15:0]        cfg_c,
    output logic [15:0]        cfg_kernel_size,
    output logic [2:0]         cfg_act_sel,
    output logic signed [31:0] cfg_scale_in,
    output logic signed [31:0] cfg_bias_in,
    output logic               cfg_act_enable,
    output logic [3:0]         cfg_leak_shift,
    output logic [31:0]        cfg_saturation,
    output logic               cfg_clk_gate,

    // ── Pooling / Quantization outputs ────────────────────────────────────────
    output logic               cfg_pool_en,
    output logic               cfg_pool_mode,
    output logic [31:0]        cfg_quant_mul,
    output logic [7:0]         cfg_quant_shift,
    output logic               cfg_quant_en,

    // ── Global control outputs ────────────────────────────────────────────────
    output logic               ctrl_soft_reset,  // self-clearing pulse
    output logic               ctrl_start,       // self-clearing pulse
    output logic               ctrl_bypass,
    output logic [2:0]         ctrl_precision,
    output logic               ctrl_power_gate,
    output logic               ctrl_cpu_mode,    // 1 = CPU direct I/O; 0 = DMA path

    // ── CPU direct token input path (cpu_mode=1) ──────────────────────────────
    // CSR assembles 16 staging words into a 512-bit token and pulses cpu_src_valid.
    output logic                        cpu_src_valid,   // 1-cycle inject pulse
    output logic [MAX_C*DATA_WIDTH-1:0] cpu_src_data,    // assembled 512-bit token
    output logic                        cpu_src_last,    // marks last token of frame
    input  logic                        cpu_src_ready,   // input FIFO has space

    // ── CPU direct token output path (cpu_mode=1) ─────────────────────────────
    // Top-level presents one output token here; CSR latches it for CPU to read.
    input  logic                        cpu_dst_valid,   // output FIFO has a token
    input  logic [MAX_C*DATA_WIDTH-1:0] cpu_dst_data,    // output token
    input  logic                        cpu_dst_last,
    output logic                        cpu_dst_ack,     // pop the output token

    // ── Weight loading passthrough (all three banks via CSR_WEIGHT_CTRL) ─────
    // DW conv bank (bank=00):
    output logic               dw_wr_en,         // 1-cycle pulse
    output logic [15:0]        dw_wr_ch,
    output logic [15:0]        dw_wr_k,
    output logic [7:0]         dw_wr_data,
    // Pointwise expand/project banks (bank=01/10):
    output logic               pw_wr_en,         // 1-cycle pulse
    output logic [1:0]         pw_wr_bank,       // 1=expand, 2=project
    output logic [13:0]        pw_wr_addr,       // flat address (CSR_WEIGHT_ADDR[13:0])
    output logic [7:0]         pw_wr_data,

    // ── DMA configuration outputs (read by custom DMA) ────────────────────────
    output logic [31:0]        dma_src_addr,
    output logic [31:0]        dma_dst_addr,
    output logic [31:0]        dma_token_cnt,
    output logic               dma_start,        // self-clearing pulse
    output logic               dma_abort,
    output logic               dma_loop_mode,
    output logic               dma_src_inc,
    output logic [31:0]        dma_burst_len,

    // ── Status inputs (from convformer_block + DMA) ───────────────────────────
    input  logic               ip_busy,
    input  logic               ip_error,
    input  logic [3:0]         ip_error_code,
    input  logic [7:0]         ip_fifo_level,
    input  logic [31:0]        perf_cycle_count,
    input  logic [31:0]        perf_stall_count,
    input  logic [31:0]        perf_token_count,
    input  logic               dma_done_in,
    input  logic               dma_busy_in,
    input  logic               dma_src_err_in,
    input  logic               dma_dst_err_in,
    input  logic [31:0]        dma_tokens_done_in,

    // =========================================================================
    // RA Processing Configuration Outputs
    // =========================================================================
    output logic [9:0]         ra_fft_size,
    output logic [2:0]         ra_fft_radix,
    output logic [1:0]         ra_window_type,
    output logic [2:0]         ra_beam_mode,
    output logic               ra_window_en,
    output logic               ra_beamform_en,
    output logic               ra_angle_fft_en,
    output logic               ra_mag_en,
    output logic               ra_cfar_en,
    output logic [1:0]         ra_cfar_mode,
    output logic [9:0]         ra_cfar_threshold,
    output logic [3:0]         ra_cfar_guard,
    output logic [3:0]         ra_cfar_train,

    // RA Processing Status Inputs
    input  logic               ra_busy,
    input  logic               ra_fft_busy,
    input  logic               ra_cfar_active,
    input  logic [15:0]        ra_frame_count,
    input  logic [15:0]        ra_detection_count,

    // =========================================================================
    // Temporal Processing Configuration Outputs
    // =========================================================================
    output logic               temporal_en,
    output logic [2:0]         temporal_mode,
    output logic [3:0]         temporal_history_depth,
    output logic [3:0]         temporal_kernel_t,

    // Temporal Status Inputs
    input  logic               temporal_busy,
    input  logic [3:0]         temporal_frame_count,

    // =========================================================================
    // CFAR Configuration Outputs
    // =========================================================================
    output logic [7:0]         cfar_scale_factor,
    output logic [1:0]         cfar_train_mode,
    output logic [3:0]         cfar_os_rank,

    // =========================================================================
    // Weight Prefetch Configuration Outputs
    // =========================================================================
    output logic               weight_prefetch_en,
    output logic               weight_bank_select,
    output logic [15:0]        weight_prefetch_addr,
    output logic [15:0]        weight_prefetch_len,

    // ── Interrupt output ──────────────────────────────────────────────────────
    output logic               irq
);

    localparam int ADDR_W = 12;
    localparam int DATA_W = 32;

    // =========================================================================
    // AXI4-Lite Write Path
    // =========================================================================
    // Both AW and W channels are buffered independently. A write executes when
    // both are pending. awready/wready are blocked while a B response is live.

    logic aw_pend;
    logic [ADDR_W-1:0] aw_addr_q;
    logic w_pend;
    logic [DATA_W-1:0]   w_data_q;
    logic [DATA_W/8-1:0] w_strb_q;
    logic do_write;

    // AW channel
    assign s_csr.awready = ~aw_pend & ~s_csr.bvalid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_pend   <= 1'b0;
            aw_addr_q <= '0;
        end else begin
            if (s_csr.awvalid & s_csr.awready) begin
                aw_pend   <= 1'b1;
                aw_addr_q <= s_csr.awaddr;
            end else if (do_write) begin
                aw_pend <= 1'b0;
            end
        end
    end

    // W channel
    assign s_csr.wready = ~w_pend & ~s_csr.bvalid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_pend   <= 1'b0;
            w_data_q <= '0;
            w_strb_q <= '0;
        end else begin
            if (s_csr.wvalid & s_csr.wready) begin
                w_pend   <= 1'b1;
                w_data_q <= s_csr.wdata;
                w_strb_q <= s_csr.wstrb;
            end else if (do_write) begin
                w_pend <= 1'b0;
            end
        end
    end

    assign do_write = aw_pend & w_pend;

    // Write response
    logic write_ok;   // driven by register decode
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_csr.bvalid <= 1'b0;
            s_csr.bresp  <= 2'b00;
        end else begin
            if (do_write) begin
                s_csr.bvalid <= 1'b1;
                s_csr.bresp  <= write_ok ? 2'b00 : 2'b10;  // OKAY / SLVERR
            end else if (s_csr.bvalid & s_csr.bready) begin
                s_csr.bvalid <= 1'b0;
            end
        end
    end

    // =========================================================================
    // AXI4-Lite Read Path
    // =========================================================================
    // Single-cycle response: arready=1 when not serving a read; rdata is driven
    // combinatorially from araddr and latched on the read trigger.

    logic [DATA_W-1:0] rd_data_mux;   // combinational decode on araddr

    assign s_csr.arready = ~s_csr.rvalid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_csr.rvalid <= 1'b0;
            s_csr.rdata  <= '0;
            s_csr.rresp  <= 2'b00;
        end else if (s_csr.arvalid & s_csr.arready) begin
            s_csr.rvalid <= 1'b1;
            s_csr.rdata  <= rd_data_mux;
            s_csr.rresp  <= read_ok ? 2'b00 : 2'b10;
        end else if (s_csr.rvalid & s_csr.rready) begin
            s_csr.rvalid <= 1'b0;
        end
    end

    // =========================================================================
    // Shadow Registers (written by CPU, applied on cfg_update)
    // =========================================================================
    logic [15:0]        cfg_c_sh,           cfg_c_live;
    logic [15:0]        cfg_kernel_size_sh, cfg_kernel_size_live;
    logic [2:0]         cfg_act_sel_sh,     cfg_act_sel_live;
    logic signed [31:0] cfg_scale_in_sh,    cfg_scale_in_live;
    logic signed [31:0] cfg_bias_in_sh,     cfg_bias_in_live;
    logic               cfg_act_enable_sh,  cfg_act_enable_live;
    logic [3:0]         cfg_leak_shift_sh,  cfg_leak_shift_live;
    logic [31:0]        cfg_saturation_sh,  cfg_saturation_live;

    logic cfg_update_pulse;   // one-cycle pulse to commit shadows

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cfg_c_live           <= 16'd64;
            cfg_kernel_size_live <= 16'd9;
            cfg_act_sel_live     <= 3'd0;
            cfg_scale_in_live    <= 32'd0;
            cfg_bias_in_live     <= 32'd0;
            cfg_act_enable_live  <= 1'b1;
            cfg_leak_shift_live  <= 4'd0;
            cfg_saturation_live  <= 32'd0;
        end else if (cfg_update_pulse) begin
            cfg_c_live           <= cfg_c_sh;
            cfg_kernel_size_live <= cfg_kernel_size_sh;
            cfg_act_sel_live     <= cfg_act_sel_sh;
            cfg_scale_in_live    <= cfg_scale_in_sh;
            cfg_bias_in_live     <= cfg_bias_in_sh;
            cfg_act_enable_live  <= cfg_act_enable_sh;
            cfg_leak_shift_live  <= cfg_leak_shift_sh;
            cfg_saturation_live  <= cfg_saturation_sh;
        end
    end

    // Shadow initial values (match live resets)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cfg_c_sh           <= 16'd64;
            cfg_kernel_size_sh <= 16'd9;
            cfg_act_sel_sh     <= 3'd0;
            cfg_scale_in_sh    <= 32'd0;
            cfg_bias_in_sh     <= 32'd0;
            cfg_act_enable_sh  <= 1'b1;
            cfg_leak_shift_sh  <= 4'd0;
            cfg_saturation_sh  <= 32'd0;
        end else if (do_write) begin
            // Byte-lane masking helper task (inlined via for loop)
            case (aw_addr_q[ADDR_W-1:0])
                CSR_CFG_C: begin  // 16-bit register: only bytes 0 and 1 are used
                    if (w_strb_q[0]) cfg_c_sh[7:0]  <= w_data_q[7:0];
                    if (w_strb_q[1]) cfg_c_sh[15:8] <= w_data_q[15:8];
                end
                CSR_CFG_KERNEL_SIZE: begin
                    if (w_strb_q[0]) cfg_kernel_size_sh[7:0]  <= w_data_q[7:0];
                    if (w_strb_q[1]) cfg_kernel_size_sh[15:8] <= w_data_q[15:8];
                end
                CSR_CFG_ACT_SEL:   if (w_strb_q[0]) cfg_act_sel_sh     <= w_data_q[2:0];
                CSR_CFG_SCALE_IN:  for (int b=0;b<4;b++) if(w_strb_q[b]) cfg_scale_in_sh[b*8 +: 8]  <= w_data_q[b*8 +: 8];
                CSR_CFG_BIAS_IN:   for (int b=0;b<4;b++) if(w_strb_q[b]) cfg_bias_in_sh[b*8 +: 8]   <= w_data_q[b*8 +: 8];
                CSR_ACT_ENABLE:    if (w_strb_q[0]) cfg_act_enable_sh  <= w_data_q[0];
                CSR_ACT_LEAK_SHIFT:if (w_strb_q[0]) cfg_leak_shift_sh  <= w_data_q[3:0];
                CSR_ACT_SATURATION:for (int b=0;b<4;b++) if(w_strb_q[b]) cfg_saturation_sh[b*8 +: 8] <= w_data_q[b*8 +: 8];
                default: ;
            endcase
        end
    end

    // Drive live values to outputs
    assign cfg_c          = cfg_c_live;
    assign cfg_kernel_size= cfg_kernel_size_live;
    assign cfg_act_sel    = cfg_act_sel_live;
    assign cfg_scale_in   = cfg_scale_in_live;
    assign cfg_bias_in    = cfg_bias_in_live;
    assign cfg_act_enable = cfg_act_enable_live;
    assign cfg_leak_shift = cfg_leak_shift_live;
    assign cfg_saturation = cfg_saturation_live;

    // =========================================================================
    // Control Register (CSR_CTRL)
    // =========================================================================
    logic        ctrl_cfg_update_r;   // shadow update trigger
    logic        ctrl_soft_reset_r;
    logic        ctrl_start_r;
    logic        ctrl_bypass_r;
    logic [2:0]  ctrl_precision_r;
    logic        ctrl_power_gate_r;
    logic        ctrl_clock_gate_r;
    logic        ctrl_cpu_mode_r;     // 0=DMA path, 1=CPU direct I/O

    assign cfg_update_pulse = ctrl_cfg_update_r;
    assign ctrl_soft_reset  = ctrl_soft_reset_r;
    assign ctrl_start       = ctrl_start_r;
    assign ctrl_bypass      = ctrl_bypass_r;
    assign ctrl_precision   = ctrl_precision_r;
    assign ctrl_power_gate  = ctrl_power_gate_r;
    assign cfg_clk_gate     = ctrl_clock_gate_r;
    assign ctrl_cpu_mode    = ctrl_cpu_mode_r;

    // cfg_locked: stretched pulse so CPU can poll it after shadow commit
    logic cfg_locked;
    logic [1:0] cfg_locked_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cfg_locked_cnt <= '0;
        end else if (cfg_update_pulse) begin
            cfg_locked_cnt <= 2'd3;  // stretch for 3+ cycles
        end else if (|cfg_locked_cnt) begin
            cfg_locked_cnt <= cfg_locked_cnt - 1;
        end
    end

    assign cfg_locked = |cfg_locked_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_cfg_update_r <= 1'b0;
            ctrl_soft_reset_r <= 1'b0;
            ctrl_start_r      <= 1'b0;
            ctrl_bypass_r     <= 1'b0;
            ctrl_precision_r  <= 3'd0;
            ctrl_power_gate_r <= 1'b0;
            ctrl_clock_gate_r <= 1'b0;
            ctrl_cpu_mode_r   <= 1'b0;
        end else begin
            // Self-clearing bits: pulse for 1 cycle
            ctrl_cfg_update_r <= 1'b0;
            ctrl_soft_reset_r <= 1'b0;
            ctrl_start_r      <= 1'b0;

            if (do_write && aw_addr_q[ADDR_W-1:0] == CSR_CTRL) begin
                if (w_strb_q[1]) begin
                    ctrl_cfg_update_r <= w_data_q[15];
                    ctrl_soft_reset_r <= w_data_q[14];
                    ctrl_start_r      <= w_data_q[12];
                    ctrl_bypass_r     <= w_data_q[11];
                end
                if (w_strb_q[0]) begin
                    ctrl_precision_r  <= w_data_q[10:8];
                    ctrl_power_gate_r <= w_data_q[7];
                    ctrl_clock_gate_r <= w_data_q[6];
                    ctrl_cpu_mode_r   <= w_data_q[5];   // [5] = cpu_mode
                end
            end
        end
    end

    // =========================================================================
    // Weight Loading (CSR_WEIGHT_ADDR / CSR_WEIGHT_DATA / CSR_WEIGHT_CTRL)
    //
    // Protocol (all three banks):
    //   1. Write flat address to CSR_WEIGHT_ADDR
    //   2. Write 8-bit weight to CSR_WEIGHT_DATA[7:0]
    //   3. Write CSR_WEIGHT_CTRL with wr_en[7]=1, bank[6:5]:
    //        00 = DW conv  (uses channel[23:16] + kernel_tap[15:8] for address)
    //        01 = expand   (uses CSR_WEIGHT_ADDR[13:0])
    //        10 = project  (uses CSR_WEIGHT_ADDR[13:0])
    //   The wr_en pulse is self-clearing (lasts exactly 1 clock cycle).
    //   Repeat steps 1-3 for each weight entry.
    // =========================================================================
    logic [31:0] weight_addr_r;
    logic [31:0] weight_data_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_addr_r <= '0;
            weight_data_r <= '0;
            dw_wr_en      <= 1'b0;
            dw_wr_ch      <= '0;
            dw_wr_k       <= '0;
            dw_wr_data    <= '0;
            pw_wr_en      <= 1'b0;
            pw_wr_bank    <= '0;
            pw_wr_addr    <= '0;
            pw_wr_data    <= '0;
        end else begin
            dw_wr_en <= 1'b0;   // self-clearing every cycle
            pw_wr_en <= 1'b0;

            if (do_write) begin
                case (aw_addr_q[ADDR_W-1:0])
                    CSR_WEIGHT_ADDR: begin
                        for (int b = 0; b < 4; b++)
                            if (w_strb_q[b]) weight_addr_r[b*8 +: 8] <= w_data_q[b*8 +: 8];
                    end
                    CSR_WEIGHT_DATA: begin
                        for (int b = 0; b < 4; b++)
                            if (w_strb_q[b]) weight_data_r[b*8 +: 8] <= w_data_q[b*8 +: 8];
                    end
                    CSR_WEIGHT_CTRL: begin
                        if (w_data_q[7]) begin  // wr_en
                            case (w_data_q[6:5])   // bank select
                                2'b00: begin  // DW conv bank
                                    dw_wr_en   <= 1'b1;
                                    dw_wr_ch   <= {8'b0, w_data_q[23:16]};
                                    dw_wr_k    <= {8'b0, w_data_q[15:8]};
                                    dw_wr_data <= weight_data_r[7:0];
                                end
                                2'b01,          // expand bank
                                2'b10: begin    // project bank
                                    pw_wr_en   <= 1'b1;
                                    pw_wr_bank <= w_data_q[6:5];
                                    pw_wr_addr <= weight_addr_r[13:0];
                                    pw_wr_data <= weight_data_r[7:0];
                                end
                                default: ;
                            endcase
                        end
                    end
                    default: ;
                endcase
            end
        end
    end

    // =========================================================================
    // CPU Direct Token I/O
    // =========================================================================
    //
    // INPUT PATH (CPU → IP):
    //   1. Write TOKEN_IN_0..15 staging words (addresses 0x400..0x43C)
    //   2. Write TOKEN_IN_CTRL[0]=1 (inject, self-clearing) and [1]=last
    //   → CSR asserts cpu_src_valid for 1 cycle; top-level pushes into input FIFO
    //   → TOKEN_IN_CTRL[2] (ready) reads 1 when input FIFO has space
    //
    // OUTPUT PATH (IP → CPU):
    //   1. Poll TOKEN_OUT_STATUS[0] until valid=1
    //   2. Read TOKEN_OUT_0..15 (addresses 0x480..0x4BC) — 16 × 32-bit words
    //   3. Write TOKEN_OUT_ACK[0]=1 to advance (pop latch, allow next token)
    //   → cpu_dst_ack pulses to top-level; token_out_valid clears
    // =========================================================================
    localparam int TOKEN_WORDS = MAX_C * DATA_WIDTH / 32;  // = 16

    // Input staging registers (16 × 32-bit)
    logic [31:0] token_in_staging [0:TOKEN_WORDS-1];

    // Output latch registers (16 × 32-bit)
    logic [31:0]  token_out_latch [0:TOKEN_WORDS-1];
    logic         token_out_valid_r;
    logic         token_out_last_r;

    // ── Token inject (input staging → cpu_src) ────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < TOKEN_WORDS; i++) token_in_staging[i] <= '0;
            cpu_src_valid <= 1'b0;
            cpu_src_data  <= '0;
            cpu_src_last  <= 1'b0;
        end else begin
            cpu_src_valid <= 1'b0;  // self-clearing

            if (do_write) begin
                // Range decode: 0x400–0x43C → word index [3:2]
                if (aw_addr_q[11:6] == 6'h10 && aw_addr_q[5:2] < TOKEN_WORDS[3:0]) begin
                    for (int b = 0; b < 4; b++)
                        if (w_strb_q[b])
                            token_in_staging[aw_addr_q[5:2]][b*8 +: 8] <= w_data_q[b*8 +: 8];
                end
                if (aw_addr_q[ADDR_W-1:0] == CSR_TOKEN_IN_CTRL && w_strb_q[0] && w_data_q[0]) begin
                    // inject bit: pack staging into 512-bit token and present to top
                    cpu_src_valid <= 1'b1;
                    cpu_src_last  <= w_data_q[1];
                    for (int i = 0; i < TOKEN_WORDS; i++)
                        cpu_src_data[i*32 +: 32] <= token_in_staging[i];
                end
            end
        end
    end

    // ── Output latch (cpu_dst → token_out registers) ─────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < TOKEN_WORDS; i++) token_out_latch[i] <= '0;
            token_out_valid_r <= 1'b0;
            token_out_last_r  <= 1'b0;
            cpu_dst_ack       <= 1'b0;
        end else begin
            cpu_dst_ack <= 1'b0;  // self-clearing

            // Latch incoming output token (when CPU mode and latch is free)
            if (cpu_dst_valid && !token_out_valid_r) begin
                token_out_valid_r <= 1'b1;
                token_out_last_r  <= cpu_dst_last;
                for (int i = 0; i < TOKEN_WORDS; i++)
                    token_out_latch[i] <= cpu_dst_data[i*32 +: 32];
            end

            // ACK: CPU writes TOKEN_OUT_ACK[0]=1 → clear latch + pulse ack
            if (do_write && aw_addr_q[ADDR_W-1:0] == CSR_TOKEN_OUT_ACK
                        && w_strb_q[0] && w_data_q[0]) begin
                token_out_valid_r <= 1'b0;
                cpu_dst_ack       <= 1'b1;
            end
        end
    end

    // =========================================================================
    // Pooling / Quantization Registers
    // =========================================================================
    logic        pool_en_r;
    logic        pool_mode_r;
    logic [31:0] quant_mul_r;
    logic [7:0]  quant_shift_r;
    logic        quant_en_r;

    assign cfg_pool_en    = pool_en_r;
    assign cfg_pool_mode  = pool_mode_r;
    assign cfg_quant_mul  = quant_mul_r;
    assign cfg_quant_shift= quant_shift_r;
    assign cfg_quant_en   = quant_en_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pool_en_r    <= 1'b0;
            pool_mode_r  <= 1'b0;
            quant_mul_r  <= 32'h0001_0000;  // Q1.16 = 1.0
            quant_shift_r<= 8'd16;
            quant_en_r   <= 1'b0;
        end else if (do_write) begin
            case (aw_addr_q[ADDR_W-1:0])
                CSR_POOL_EN:   if (w_strb_q[0]) pool_en_r   <= w_data_q[0];
                CSR_POOL_MODE: if (w_strb_q[0]) pool_mode_r <= w_data_q[0];
                CSR_QUANT_MUL: for (int b=0;b<4;b++) if(w_strb_q[b]) quant_mul_r[b*8+:8] <= w_data_q[b*8+:8];
                CSR_QUANT_SHIFT:if(w_strb_q[0]) quant_shift_r <= w_data_q[7:0];
                CSR_QUANT_EN:  if (w_strb_q[0]) quant_en_r  <= w_data_q[0];
                default: ;
            endcase
        end
    end

    // =========================================================================
    // RA Processing Configuration (Shadow Registers)
    // =========================================================================
    // Shadow protocol: CPU writes CSR_RA_* shadow regs anytime.
    // On cfg_update pulse, shadows commit to live regs atomically.
    logic [11:0] ra_fft_size_sh,        ra_fft_size_live;
    logic [2:0]  ra_fft_radix_sh,       ra_fft_radix_live;
    logic [1:0]  ra_window_type_sh,     ra_window_type_live;
    logic [2:0]  ra_beam_mode_sh,       ra_beam_mode_live;
    logic        ra_window_en_sh,        ra_window_en_live;
    logic        ra_beamform_en_sh,      ra_beamform_en_live;
    logic        ra_angle_fft_en_sh,     ra_angle_fft_en_live;
    logic        ra_mag_en_sh,           ra_mag_en_live;
    logic        ra_cfar_en_sh,          ra_cfar_en_live;
    logic [1:0]  ra_cfar_mode_sh,        ra_cfar_mode_live;
    logic [9:0]  ra_cfar_threshold_sh,  ra_cfar_threshold_live;
    logic [3:0]  ra_cfar_guard_sh,      ra_cfar_guard_live;
    logic [3:0]  ra_cfar_train_sh,      ra_cfar_train_live;
    logic        ra_log_output_sh,       ra_log_output_live;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ra_fft_size_live       <= 12'd256;
            ra_fft_radix_live     <= 3'd2;
            ra_window_type_live   <= 2'd0;
            ra_beam_mode_live     <= 3'd0;
            ra_window_en_live     <= 1'b0;
            ra_beamform_en_live   <= 1'b0;
            ra_angle_fft_en_live  <= 1'b0;
            ra_mag_en_live        <= 1'b0;
            ra_cfar_en_live       <= 1'b0;
            ra_cfar_mode_live     <= 2'd0;
            ra_cfar_threshold_live<= 10'd256;
            ra_cfar_guard_live    <= 4'd4;
            ra_cfar_train_live    <= 4'd8;
            ra_log_output_live    <= 1'b0;
        end else if (cfg_update_pulse) begin
            ra_fft_size_live       <= ra_fft_size_sh;
            ra_fft_radix_live     <= ra_fft_radix_sh;
            ra_window_type_live   <= ra_window_type_sh;
            ra_beam_mode_live     <= ra_beam_mode_sh;
            ra_window_en_live     <= ra_window_en_sh;
            ra_beamform_en_live   <= ra_beamform_en_sh;
            ra_angle_fft_en_live  <= ra_angle_fft_en_sh;
            ra_mag_en_live        <= ra_mag_en_sh;
            ra_cfar_en_live       <= ra_cfar_en_sh;
            ra_cfar_mode_live     <= ra_cfar_mode_sh;
            ra_cfar_threshold_live<= ra_cfar_threshold_sh;
            ra_cfar_guard_live    <= ra_cfar_guard_sh;
            ra_cfar_train_live    <= ra_cfar_train_sh;
            ra_log_output_live    <= ra_log_output_sh;
        end
    end

    // Shadow initial values and write decode
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ra_fft_size_sh       <= 12'd256;
            ra_fft_radix_sh     <= 3'd2;
            ra_window_type_sh   <= 2'd0;
            ra_beam_mode_sh     <= 3'd0;
            ra_window_en_sh     <= 1'b0;
            ra_beamform_en_sh   <= 1'b0;
            ra_angle_fft_en_sh  <= 1'b0;
            ra_mag_en_sh        <= 1'b0;
            ra_cfar_en_sh       <= 1'b0;
            ra_cfar_mode_sh     <= 2'd0;
            ra_cfar_threshold_sh<= 10'd256;
            ra_cfar_guard_sh    <= 4'd4;
            ra_cfar_train_sh    <= 4'd8;
            ra_log_output_sh    <= 1'b0;
        end else if (do_write) begin
            case (aw_addr_q[ADDR_W-1:0])
                CSR_RA_CONFIG: if (w_strb_q[0]) begin
                    ra_log_output_sh    <= w_data_q[9];
                    ra_mag_en_sh        <= w_data_q[8];
                    ra_angle_fft_en_sh  <= w_data_q[7];
                    ra_beamform_en_sh   <= w_data_q[6];
                    ra_window_en_sh     <= w_data_q[5];
                    ra_cfar_en_sh       <= w_data_q[4];
                    ra_cfar_mode_sh     <= w_data_q[3:2];
                end
                CSR_RA_FFT_SIZE: if (w_strb_q[1] || w_strb_q[0])
                    ra_fft_size_sh <= w_data_q[11:0];
                CSR_RA_FFT_RADIX: if (w_strb_q[0])
                    ra_fft_radix_sh <= w_data_q[2:0];
                CSR_RA_WINDOW_TYPE: if (w_strb_q[0])
                    ra_window_type_sh <= w_data_q[1:0];
                CSR_RA_BEAM_MODE: if (w_strb_q[0])
                    ra_beam_mode_sh <= w_data_q[2:0];
                CSR_RA_CFAR_THRESH: if (w_strb_q[1] || w_strb_q[0])
                    ra_cfar_threshold_sh <= w_data_q[9:0];
                CSR_RA_CFAR_GUARD: if (w_strb_q[0])
                    ra_cfar_guard_sh <= w_data_q[3:0];
                CSR_RA_CFAR_TRAIN: if (w_strb_q[0])
                    ra_cfar_train_sh <= w_data_q[3:0];
                default: ;
            endcase
        end
    end

    // Drive outputs from live registers
    assign ra_fft_size       = ra_fft_size_live;
    assign ra_fft_radix      = ra_fft_radix_live;
    assign ra_window_type    = ra_window_type_live;
    assign ra_beam_mode      = ra_beam_mode_live;
    assign ra_window_en      = ra_window_en_live;
    assign ra_beamform_en    = ra_beamform_en_live;
    assign ra_angle_fft_en   = ra_angle_fft_en_live;
    assign ra_mag_en         = ra_mag_en_live;
    assign ra_cfar_en        = ra_cfar_en_live;
    assign ra_cfar_mode      = ra_cfar_mode_live;
    assign ra_cfar_threshold = ra_cfar_threshold_live;
    assign ra_cfar_guard     = ra_cfar_guard_live;
    assign ra_cfar_train     = ra_cfar_train_live;

    // =========================================================================
    // CFAR Additional Configuration (Direct-Write Registers)
    // =========================================================================
    logic [7:0]  cfar_scale_factor_r;
    logic [1:0]  cfar_train_mode_r;
    logic [3:0]  cfar_os_rank_r;

    assign cfar_scale_factor = cfar_scale_factor_r;
    assign cfar_train_mode   = cfar_train_mode_r;
    assign cfar_os_rank       = cfar_os_rank_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cfar_scale_factor_r <= 8'd128;  // Q0.8 = 0.5
            cfar_train_mode_r  <= 2'd0;
            cfar_os_rank_r     <= 4'd4;
        end else if (do_write) begin
            case (aw_addr_q[ADDR_W-1:0])
                CSR_CFAR_CTRL: if (w_strb_q[0]) begin
                    cfar_train_mode_r <= w_data_q[3:2];
                    cfar_os_rank_r    <= w_data_q[7:4];  // reuse [7:4] for os_rank
                end
                CSR_CFAR_SCALE: if (w_strb_q[0])
                    cfar_scale_factor_r <= w_data_q[7:0];
                default: ;
            endcase
        end
    end

    // =========================================================================
    // Weight Prefetch Configuration (Direct-Write Registers)
    // =========================================================================
    logic        weight_prefetch_en_r;
    logic        weight_bank_select_r;
    logic [15:0] weight_prefetch_addr_r;
    logic [15:0] weight_prefetch_len_r;

    assign weight_prefetch_en   = weight_prefetch_en_r;
    assign weight_bank_select   = weight_bank_select_r;
    assign weight_prefetch_addr = weight_prefetch_addr_r;
    assign weight_prefetch_len  = weight_prefetch_len_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_prefetch_en_r   <= 1'b0;
            weight_bank_select_r   <= 1'b0;
            weight_prefetch_addr_r <= '0;
            weight_prefetch_len_r  <= '0;
        end else if (do_write) begin
            case (aw_addr_q[ADDR_W-1:0])
                CSR_WEIGHT_PREFETCH_CTRL: if (w_strb_q[0]) begin
                    weight_prefetch_en_r <= w_data_q[1];
                    weight_bank_select_r <= w_data_q[0];
                end
                CSR_WEIGHT_PREFETCH_ADDR: for (int b=0;b<4;b++) if(w_strb_q[b]) weight_prefetch_addr_r[b*8+:8] <= w_data_q[b*8+:8];
                CSR_WEIGHT_PREFETCH_LEN:  for (int b=0;b<4;b++) if(w_strb_q[b]) weight_prefetch_len_r[b*8+:8]  <= w_data_q[b*8+:8];
                default: ;
            endcase
        end
    end

    // =========================================================================
    // Temporal Processing Configuration (Shadow Registers)
    // =========================================================================
    logic        temporal_en_sh,        temporal_en_live;
    logic [2:0]  temporal_mode_sh,     temporal_mode_live;
    logic [3:0]  temporal_history_depth_sh, temporal_history_depth_live;
    logic [3:0]  temporal_kernel_t_sh, temporal_kernel_t_live;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            temporal_en_live            <= 1'b0;
            temporal_mode_live          <= 3'd0;
            temporal_history_depth_live <= 4'd4;
            temporal_kernel_t_live      <= 4'd0;
        end else if (cfg_update_pulse) begin
            temporal_en_live            <= temporal_en_sh;
            temporal_mode_live          <= temporal_mode_sh;
            temporal_history_depth_live <= temporal_history_depth_sh;
            temporal_kernel_t_live      <= temporal_kernel_t_sh;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            temporal_en_sh            <= 1'b0;
            temporal_mode_sh          <= 3'd0;
            temporal_history_depth_sh <= 4'd4;
            temporal_kernel_t_sh      <= 4'd0;
        end else if (do_write) begin
            case (aw_addr_q[ADDR_W-1:0])
                CSR_TEMPORAL_CTRL: if (w_strb_q[0]) begin
                    temporal_en_sh            <= w_data_q[0];
                    temporal_mode_sh          <= w_data_q[1:0];
                    temporal_history_depth_sh <= w_data_q[4:2];
                end
                default: ;
            endcase
        end
    end

    assign temporal_en            = temporal_en_live;
    assign temporal_mode           = temporal_mode_live;
    assign temporal_history_depth  = temporal_history_depth_live;
    assign temporal_kernel_t       = temporal_kernel_t_live;

    // =========================================================================
    // DMA Configuration Registers
    // =========================================================================
    logic [31:0] dma_src_addr_r;
    logic [31:0] dma_dst_addr_r;
    logic [31:0] dma_token_cnt_r;
    logic        dma_start_r;
    logic        dma_abort_r;
    logic        dma_loop_mode_r;
    logic        dma_src_inc_r;
    logic [31:0] dma_burst_len_r;
    logic [1:0]  dma_irq_en_r;

    assign dma_src_addr  = dma_src_addr_r;
    assign dma_dst_addr  = dma_dst_addr_r;
    assign dma_token_cnt = dma_token_cnt_r;
    assign dma_start     = dma_start_r;
    assign dma_abort     = dma_abort_r;
    assign dma_loop_mode = dma_loop_mode_r;
    assign dma_src_inc   = dma_src_inc_r;
    assign dma_burst_len = dma_burst_len_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_src_addr_r  <= '0;
            dma_dst_addr_r  <= '0;
            dma_token_cnt_r <= '0;
            dma_start_r     <= 1'b0;
            dma_abort_r     <= 1'b0;
            dma_loop_mode_r <= 1'b0;
            dma_src_inc_r   <= 1'b0;
            dma_burst_len_r <= 32'd16;   // default: 1 token per AXI4 burst
            dma_irq_en_r    <= 2'b00;
        end else begin
            // Self-clearing
            dma_start_r <= 1'b0;
            dma_abort_r <= 1'b0;

            if (do_write) begin
                case (aw_addr_q[ADDR_W-1:0])
                    CSR_DMA_SRC_ADDR: for(int b=0;b<4;b++) if(w_strb_q[b]) dma_src_addr_r[b*8+:8] <= w_data_q[b*8+:8];
                    CSR_DMA_DST_ADDR: for(int b=0;b<4;b++) if(w_strb_q[b]) dma_dst_addr_r[b*8+:8] <= w_data_q[b*8+:8];
                    CSR_DMA_TOKEN_CNT:for(int b=0;b<4;b++) if(w_strb_q[b]) dma_token_cnt_r[b*8+:8]<= w_data_q[b*8+:8];
                    CSR_DMA_CTRL: begin
                        if (w_strb_q[0]) begin
                            dma_start_r     <= w_data_q[0];
                            dma_abort_r     <= w_data_q[1];
                            dma_loop_mode_r <= w_data_q[2];
                            dma_src_inc_r   <= w_data_q[3];
                        end
                    end
                    CSR_DMA_IRQ_EN:  if(w_strb_q[0]) dma_irq_en_r <= w_data_q[1:0];
                    CSR_DMA_BURST_LEN:for(int b=0;b<4;b++) if(w_strb_q[b]) dma_burst_len_r[b*8+:8]<= w_data_q[b*8+:8];
                    default: ;
                endcase
            end
        end
    end

    // =========================================================================
    // DMA Status Register (W1C)
    // =========================================================================
    logic dma_done_r, dma_error_r, dma_src_err_r, dma_dst_err_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_done_r    <= 1'b0;
            dma_error_r   <= 1'b0;
            dma_src_err_r <= 1'b0;
            dma_dst_err_r <= 1'b0;
        end else begin
            // Set from DMA
            if (dma_done_in)    dma_done_r    <= 1'b1;
            if (dma_src_err_in) dma_src_err_r <= 1'b1;
            if (dma_dst_err_in) dma_dst_err_r <= 1'b1;
            if (dma_src_err_in || dma_dst_err_in) dma_error_r <= 1'b1;

            // W1C clear from CPU: each bit only cleared when its byte's WSTRB is asserted
            // All current W1C bits (done, error, src_err, dst_err) are in byte 0 (bits [4:0])
            if (do_write && aw_addr_q[ADDR_W-1:0] == CSR_DMA_STATUS) begin
                if (w_strb_q[0]) begin  // byte 0: all W1C status bits
                    if (w_data_q[0]) dma_done_r    <= 1'b0;
                    if (w_data_q[1]) dma_error_r   <= 1'b0;
                    if (w_data_q[3]) dma_src_err_r <= 1'b0;
                    if (w_data_q[4]) dma_dst_err_r <= 1'b0;
                end
            end
        end
    end

    // =========================================================================
    // IRQ Generation
    // =========================================================================
    assign irq = (dma_done_r  & dma_irq_en_r[0]) |
                 (dma_error_r & dma_irq_en_r[1]);

    // Also surface cfg_update via interface
    assign s_csr.cfg_busy = cfg_locked;

    // =========================================================================
    // Write Decode: write_ok (SLVERR on unmapped address)
    // =========================================================================
    always_comb begin
        write_ok = 1'b0;
        case (aw_addr_q[ADDR_W-1:0])
            CSR_CTRL,
            CSR_CFG_C, CSR_CFG_KERNEL_SIZE, CSR_CFG_ACT_SEL,
            CSR_CFG_SCALE_IN, CSR_CFG_BIAS_IN,
            CSR_ACT_ENABLE, CSR_ACT_LEAK_SHIFT, CSR_ACT_SATURATION,
            CSR_POOL_EN, CSR_POOL_MODE, CSR_QUANT_MUL, CSR_QUANT_SHIFT, CSR_QUANT_EN,
            CSR_WEIGHT_ADDR, CSR_WEIGHT_DATA, CSR_WEIGHT_CTRL,
            CSR_RA_CONFIG, CSR_RA_FFT_SIZE, CSR_RA_FFT_RADIX,
            CSR_RA_WINDOW_TYPE, CSR_RA_BEAM_MODE, CSR_RA_CFAR_THRESH,
            CSR_RA_CFAR_GUARD, CSR_RA_CFAR_TRAIN,
            CSR_CFAR_CTRL, CSR_CFAR_SCALE,
            CSR_WEIGHT_PREFETCH_CTRL, CSR_WEIGHT_PREFETCH_ADDR, CSR_WEIGHT_PREFETCH_LEN,
            CSR_TEMPORAL_CTRL,
            CSR_DMA_SRC_ADDR, CSR_DMA_DST_ADDR, CSR_DMA_TOKEN_CNT,
            CSR_DMA_CTRL, CSR_DMA_STATUS, CSR_DMA_IRQ_EN, CSR_DMA_BURST_LEN:
                write_ok = 1'b1;
            default: begin
                // Range-based decode for token staging (0x400-0x43C) and ack/ctrl
                if ((aw_addr_q[11:6] == 6'h10 && aw_addr_q[5:2] < TOKEN_WORDS[3:0]) ||
                    aw_addr_q[11:0] == CSR_TOKEN_IN_CTRL ||
                    aw_addr_q[11:0] == CSR_TOKEN_OUT_ACK)
                    write_ok = 1'b1;
                else
                    write_ok = 1'b0;
            end
        endcase
    end

    // =========================================================================
    // Read Decode: rd_data_mux
    // =========================================================================
    logic read_ok;

    always_comb begin
        rd_data_mux = 32'h0;
        read_ok     = 1'b1;

        case (s_csr.araddr[ADDR_W-1:0])
            CSR_CTRL: rd_data_mux = {
                16'b0,
                1'b0,                  // cfg_update (self-clearing, reads 0)
                ctrl_soft_reset_r,
                1'b0,                  // weight_load (not implemented here)
                ctrl_start_r,
                ctrl_bypass_r,
                ctrl_precision_r,
                ctrl_power_gate_r,
                ctrl_clock_gate_r,
                6'b0
            };
            CSR_STATUS: rd_data_mux = {
                8'b0,
                ip_fifo_level,
                1'b0,                  // weight_loaded
                cfg_locked,
                ip_busy,
                ip_error,
                ip_error_code,
                !ip_busy,              // ready
                7'b0
            };
            CSR_VERSION:        rd_data_mux = HW_VERSION;
            CSR_CFG_C:          rd_data_mux = {16'b0, cfg_c_sh};
            CSR_CFG_KERNEL_SIZE:rd_data_mux = {16'b0, cfg_kernel_size_sh};
            CSR_CFG_ACT_SEL:    rd_data_mux = {29'b0, cfg_act_sel_sh};
            CSR_CFG_SCALE_IN:   rd_data_mux = cfg_scale_in_sh;
            CSR_CFG_BIAS_IN:    rd_data_mux = cfg_bias_in_sh;
            CSR_ACT_ENABLE:     rd_data_mux = {31'b0, cfg_act_enable_sh};
            CSR_ACT_LEAK_SHIFT: rd_data_mux = {28'b0, cfg_leak_shift_sh};
            CSR_ACT_SATURATION: rd_data_mux = cfg_saturation_sh;
            CSR_POOL_EN:        rd_data_mux = {31'b0, pool_en_r};
            CSR_POOL_MODE:      rd_data_mux = {31'b0, pool_mode_r};
            CSR_QUANT_MUL:      rd_data_mux = quant_mul_r;
            CSR_QUANT_SHIFT:    rd_data_mux = {24'b0, quant_shift_r};
            CSR_QUANT_EN:       rd_data_mux = {31'b0, quant_en_r};
            CSR_WEIGHT_DATA:    rd_data_mux = weight_data_r;
            CSR_CYCLE_COUNT:    rd_data_mux = perf_cycle_count;
            CSR_INSTR_COUNT:    rd_data_mux = perf_token_count;   // tokens = instructions
            CSR_STALL_COUNT:    rd_data_mux = perf_stall_count;
            CSR_ERROR_COUNT:    rd_data_mux = {28'b0, ip_error_code};
            CSR_DMA_SRC_ADDR:   rd_data_mux = dma_src_addr_r;
            CSR_DMA_DST_ADDR:   rd_data_mux = dma_dst_addr_r;
            CSR_DMA_TOKEN_CNT:  rd_data_mux = dma_token_cnt_r;
            CSR_DMA_CTRL:       rd_data_mux = {28'b0, dma_src_inc_r, dma_loop_mode_r, dma_abort_r, dma_start_r};
            CSR_DMA_STATUS:     rd_data_mux = {
                27'b0,
                dma_dst_err_r, dma_src_err_r,
                dma_busy_in,
                dma_error_r,
                dma_done_r
            };
            CSR_DMA_IRQ_EN:     rd_data_mux = {30'b0, dma_irq_en_r};
            CSR_DMA_BURST_LEN:  rd_data_mux = dma_burst_len_r;
            CSR_DMA_TOKENS_DONE:rd_data_mux = dma_tokens_done_in;
            // ── RA Processing Registers (shadow read-back) ────────────────────
            CSR_RA_CONFIG: rd_data_mux = {22'b0, ra_log_output_sh, ra_mag_en_sh,
                                           ra_angle_fft_en_sh, ra_beamform_en_sh,
                                           ra_window_en_sh, ra_cfar_en_sh, ra_cfar_mode_sh};
            CSR_RA_FFT_SIZE:      rd_data_mux = {20'b0, ra_fft_size_sh};
            CSR_RA_FFT_RADIX:    rd_data_mux = {29'b0, ra_fft_radix_sh};
            CSR_RA_WINDOW_TYPE:  rd_data_mux = {30'b0, ra_window_type_sh};
            CSR_RA_BEAM_MODE:    rd_data_mux = {29'b0, ra_beam_mode_sh};
            CSR_RA_CFAR_THRESH:  rd_data_mux = {22'b0, ra_cfar_threshold_sh};
            CSR_RA_CFAR_GUARD:   rd_data_mux = {28'b0, ra_cfar_guard_sh};
            CSR_RA_CFAR_TRAIN:   rd_data_mux = {28'b0, ra_cfar_train_sh};
            CSR_RA_STATUS:       rd_data_mux = {8'b0, 8'b0,  // [31:16] fifo_level (unused)
                                                1'b0,          // [15] cfar_active (from input)
                                                ra_fft_busy,   // [14]
                                                1'b0,          // [13] cfg_locked (unused for RA)
                                                ra_busy,       // [12]
                                                1'b0,          // [11] error (unused)
                                                4'b0,          // [10:7] error_code (unused)
                                                1'b0,          // [6] done (unused)
                                                6'b0};
            CSR_RA_FRAME_COUNT:  rd_data_mux = {16'b0, ra_frame_count};
            CSR_RA_DETECTION_COUNT: rd_data_mux = {16'b0, ra_detection_count};
            // ── CFAR Registers ─────────────────────────────────────────────────
            CSR_CFAR_CTRL:       rd_data_mux = {28'b0, cfar_train_mode_r, 2'b0, cfar_os_rank_r};
            CSR_CFAR_SCALE:      rd_data_mux = {24'b0, cfar_scale_factor_r};
            CSR_CFAR_STATUS:     rd_data_mux = 32'b0;  // placeholder
            // ── Temporal Processing Registers ──────────────────────────────────
            CSR_TEMPORAL_CTRL:   rd_data_mux = {24'b0, temporal_en_sh,
                                                temporal_mode_sh,
                                                temporal_history_depth_sh, 2'b0};
            CSR_TEMPORAL_STATUS: rd_data_mux = {28'b0, temporal_busy, 3'b0,
                                                temporal_frame_count};
            // ── Weight Prefetch Registers ─────────────────────────────────────
            CSR_WEIGHT_PREFETCH_CTRL:  rd_data_mux = {30'b0, weight_prefetch_en_r, weight_bank_select_r};
            CSR_WEIGHT_PREFETCH_ADDR:  rd_data_mux = {16'b0, weight_prefetch_addr_r};
            CSR_WEIGHT_PREFETCH_LEN:   rd_data_mux = {16'b0, weight_prefetch_len_r};
            // ── CPU direct token I/O ────────────────────────────────────────
            CSR_TOKEN_IN_CTRL: rd_data_mux = {
                29'b0,
                cpu_src_ready,          // [2] ready: FIFO has space
                1'b0,                   // [1] last (WO only)
                1'b0                    // [0] inject (WO, always reads 0)
            };
            CSR_TOKEN_OUT_STATUS: rd_data_mux = {31'b0, token_out_valid_r};
            CSR_TOKEN_OUT_ACK:    rd_data_mux = {31'b0, token_out_last_r};  // reads last flag
            default: begin
                // Range decode for token I/O word arrays
                if (s_csr.araddr[11:6] == 6'h10 && s_csr.araddr[5:2] < TOKEN_WORDS[3:0]) begin
                    // 0x400–0x43C: token input staging (read-back)
                    rd_data_mux = token_in_staging[s_csr.araddr[5:2]];
                    read_ok     = 1'b1;
                end else if (s_csr.araddr[11:6] == 6'h12 && s_csr.araddr[5:2] < TOKEN_WORDS[3:0]) begin
                    // 0x480–0x4BC: token output latch
                    rd_data_mux = token_out_latch[s_csr.araddr[5:2]];
                    read_ok     = 1'b1;
                end else begin
                    rd_data_mux = 32'hDEAD_C0DE;
                    read_ok     = 1'b0;
                end
            end
        endcase
    end

    // =========================================================================
    // SVA Assertions
    // =========================================================================
    CSR_NO_WRITE_WHILE_BUSY: assert property (
        @(posedge clk) disable iff (!rst_n)
        (do_write && aw_addr_q[ADDR_W-1:0] == CSR_CTRL && w_data_q[15]) |->
        !cfg_locked
    ) else $warning("e_rodnet_csr: cfg_update written while cfg_locked — ignored");

endmodule
