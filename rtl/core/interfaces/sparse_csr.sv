// =============================================================================
//  sparse_csr.sv
//
//  CSR register block for sparsity-aware systolic array configuration.
//  Uses the shadow/live register pattern consistent with the project convention:
//  - Write to _sh (shadow) registers at any time
//  - Commit to _live registers via cfg_update_pulse when idle
//  - _live registers drive the actual hardware
//
//  Register Map (0xB00 - 0xB3F):
//    0xB00  CSR_SPARSE_CTRL           RW   Sparse control (enable, modes, update)
//    0xB04  CSR_SPARSE_WEIGHT_MODE    RW   Weight encoding mode (CSR/RLE/Dense)
//    0xB08  CSR_SPARSE_ACTIV_MODE     RW   Activation encoding mode
//    0xB0C  CSR_SPARSE_THRESHOLD      RW   Sparsity threshold for dense fallback
//    0xB10  CSR_SPARSE_STATUS          RO   Status flags
//    0xB14  CSR_SPARSE_NZ_WEIGHT_CNT   RO   Weight NZ count (last tile)
//    0xB18  CSR_SPARSE_NZ_ACTIV_CNT   RO   Activation NZ count (last tile)
//    0xB1C  CSR_SPARSE_MAC_SKIP_CNT    RO   MAC operations skipped
//    0xB30  CSR_SPARSE_MAC_ACTIVE_CNT  RO   MAC operations active (not skipped)
//    0xB20  CSR_SPARSE_CYCLE_CNT       RO   Cycle counter
//    0xB24  CSR_SPARSE_FIFO_STALL_CNT  RO   Elastic FIFO stall cycles
//    0xB28  CSR_SPARSE_WEIGHT_BASE     RW   Weight NZ data base address
//    0xB2C  CSR_SPARSE_ACTIV_BASE      RW   Activation NZ data base address
// =============================================================================

import pkg_config::*;
import pkg_sparse_config::*;

module sparse_csr #(
    parameter int DATA_WIDTH = pkg_config::DATA_WIDTH
)(
    input  logic clk,
    input  logic rst_n,

    // ── AXI4-Lite Slave Interface ───────────────────────────────────────────
    input  logic        csr_req_valid,
    input  logic [11:0] csr_req_addr,
    input  logic [31:0] csr_req_wdata,
    input  logic        csr_req_we,      // 1=write, 0=read
    output logic [31:0] csr_req_rdata,
    output logic        csr_req_ready,

    // ── Committed (live) register outputs ──────────────────────────────────
    output logic        sparse_en,       // Sparse mode enable
    output logic [1:0]  weight_mode,     // Weight encoding mode
    output logic [1:0]  activ_mode,      // Activation encoding mode
    output logic [15:0] sparse_threshold,// Sparsity threshold

    // ── Status inputs (from hardware) ──────────────────────────────────────
    input  logic        isect_busy,
    input  logic        encoder_busy,
    input  logic        sparse_active,
    input  logic        dense_fallback,
    input  logic [7:0]  elastic_fifo_level,
    input  logic [31:0] nz_weight_cnt,
    input  logic [31:0] nz_activ_cnt,
    input  logic [31:0] mac_skip_cnt,
    input  logic [31:0] mac_active_cnt,
    input  logic [31:0] cycle_cnt,
    input  logic [31:0] fifo_stall_cnt
);

    // =========================================================================
    //  Shadow + Live Registers
    // =========================================================================
    csr_sparse_ctrl_t ctrl_sh, ctrl_live;
    logic [1:0]  weight_mode_sh, weight_mode_live;
    logic [1:0]  activ_mode_sh,  activ_mode_live;
    logic [15:0] threshold_sh,   threshold_live;

    // cfg_update_pulse: self-clearing pulse when ctrl_sh.cfg_update is set
    logic cfg_update_pulse;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cfg_update_pulse <= 1'b0;
        end else begin
            cfg_update_pulse <= ctrl_sh.cfg_update;
            if (ctrl_sh.cfg_update) begin
                ctrl_sh.cfg_update <= 1'b0;  // Self-clear
            end
        end
    end

    // Commit shadow → live on cfg_update_pulse
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_live        <= '0;
            weight_mode_live <= SPARSE_MODE_CSR;
            activ_mode_live  <= SPARSE_MODE_CSR;
            threshold_live   <= pkg_sparse_config::SPARSE_THRESHOLD_DEFAULT[15:0];
        end else if (cfg_update_pulse) begin
            ctrl_live        <= ctrl_sh;
            weight_mode_live <= weight_mode_sh;
            activ_mode_live  <= activ_mode_sh;
            threshold_live   <= threshold_sh;
        end
    end

    // Drive outputs from live registers
    assign sparse_en       = ctrl_live.sparse_en;
    assign weight_mode     = weight_mode_live;
    assign activ_mode      = activ_mode_live;
    assign sparse_threshold = threshold_live;

    // =========================================================================
    //  AXI4-Lite CSR Read/Write
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            csr_req_ready <= 1'b0;
            csr_req_rdata <= '0;
        end else begin
            csr_req_ready <= 1'b1;  // Always ready

            if (csr_req_valid && csr_req_we) begin
                // Write
                case (csr_req_addr)
                    CSR_SPARSE_CTRL:       ctrl_sh        <= csr_req_wdata;
                    CSR_SPARSE_WEIGHT_MODE: weight_mode_sh <= csr_req_wdata[1:0];
                    CSR_SPARSE_ACTIV_MODE:  activ_mode_sh  <= csr_req_wdata[1:0];
                    CSR_SPARSE_THRESHOLD:   threshold_sh   <= csr_req_wdata[15:0];
                    default: ;  // Read-only or unmapped
                endcase
            end else if (csr_req_valid && !csr_req_we) begin
                // Read
                case (csr_req_addr)
                    CSR_SPARSE_CTRL:        csr_req_rdata <= ctrl_live;
                    CSR_SPARSE_WEIGHT_MODE: csr_req_rdata <= {30'd0, weight_mode_live};
                    CSR_SPARSE_ACTIV_MODE:  csr_req_rdata <= {30'd0, activ_mode_live};
                    CSR_SPARSE_THRESHOLD:   csr_req_rdata <= {16'd0, threshold_live};
                    CSR_SPARSE_STATUS: begin
                        csr_sparse_status_t status;
                        status = '0;
                        status.elastic_fifo_level <= elastic_fifo_level;
                        status.isect_busy    <= isect_busy;
                        status.encoder_busy  <= encoder_busy;
                        status.sparse_active <= sparse_active;
                        status.dense_fallback<= dense_fallback;
                        status.done          <= 1'b0;  // Updated externally
                        csr_req_rdata <= status;
                    end
                    CSR_SPARSE_NZ_WEIGHT_CNT: csr_req_rdata <= nz_weight_cnt;
                    CSR_SPARSE_NZ_ACTIV_CNT:  csr_req_rdata <= nz_activ_cnt;
                    CSR_SPARSE_MAC_SKIP_CNT:   csr_req_rdata <= mac_skip_cnt;
                    CSR_SPARSE_MAC_ACTIVE_CNT: csr_req_rdata <= mac_active_cnt;
                    CSR_SPARSE_CYCLE_CNT:       csr_req_rdata <= cycle_cnt;
                    CSR_SPARSE_FIFO_STALL_CNT:  csr_req_rdata <= fifo_stall_cnt;
                    default: csr_req_rdata <= '0;
                endcase
            end
        end
    end

endmodule