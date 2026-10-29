// =============================================================================
// erodnet_weight_controller.sv
//
// Dual-mode weight loading engine (rebranded from weight_loader.sv).
// Handles both Auto-Increment CSR streaming and AXI4 DMA bursts.
// =============================================================================

module erodnet_weight_controller #(
    parameter int DATA_WIDTH  = 8,
    parameter int ADDR_WIDTH  = 32,
    parameter int LAYER_WIDTH = 4,
    parameter int AXI_DATA_W  = 64,
    parameter int AXI_ADDR_W  = 32,
    parameter int BURST_LEN   = 16
)(
    input  logic clk,
    input  logic rst_n,

    // --- CSR control ---
    input  logic [ADDR_WIDTH-1:0]            csr_weight_addr,
    input  logic signed [DATA_WIDTH-1:0]     csr_weight_data,
    input  logic [LAYER_WIDTH-1:0]           csr_weight_layer,
    input  logic                             csr_weight_write,
    input  logic                             csr_auto_inc_en,
    input  logic [ADDR_WIDTH-1:0]            csr_weight_count,
    input  logic                             csr_weight_done_clear,

    // --- DMA mode control ---
    input  logic [AXI_ADDR_W-1:0]            csr_dma_src_addr,
    input  logic [ADDR_WIDTH-1:0]            csr_dma_weight_count,
    input  logic [LAYER_WIDTH-1:0]           csr_dma_layer,
    input  logic [ADDR_WIDTH-1:0]            csr_dma_dst_addr,
    input  logic                             csr_dma_start,

    // --- AXI4 Master (weight DMA read from DDR) ---
    output logic [AXI_ADDR_W-1:0]            m_axi_wt_araddr,
    output logic [7:0]                       m_axi_wt_arlen,
    output logic [2:0]                       m_axi_wt_arsize,
    output logic [1:0]                       m_axi_wt_arburst,
    output logic                             m_axi_wt_arvalid,
    input  logic                             m_axi_wt_arready,
    input  logic [AXI_DATA_W-1:0]            m_axi_wt_rdata,
    input  logic [1:0]                       m_axi_wt_rresp,
    input  logic                             m_axi_wt_rvalid,
    input  logic                             m_axi_wt_rlast,
    output logic                             m_axi_wt_rready,

    // --- Weight output (to erodnet_core_top) ---
    output logic                             weight_valid,
    output logic [ADDR_WIDTH-1:0]            weight_addr,
    output logic signed [DATA_WIDTH-1:0]     weight_data,
    output logic [LAYER_WIDTH-1:0]           weight_layer_sel,

    // --- Status (for CSR readback) ---
    output logic                             weight_load_busy,
    output logic                             weight_load_done,
    output logic                             weight_load_error,
    output logic [ADDR_WIDTH-1:0]            weight_count_remain
);

    // ===========================================================
    // FSM States
    // ===========================================================
    typedef enum logic [3:0] {
        WL_IDLE,
        WL_CSR_WRITE,
        WL_DMA_READ_REQ,
        WL_DMA_READ_DATA,
        WL_DMA_EXTRACT,
        WL_DMA_NEXT,
        WL_DONE,
        WL_ERROR
    } state_t;

    state_t state, state_next;

    // ===========================================================
    // CSR Mode Registers
    // ===========================================================
    logic [ADDR_WIDTH-1:0]   curr_addr;
    logic [ADDR_WIDTH-1:0]   curr_count;
    logic [LAYER_WIDTH-1:0]  curr_layer;

    // ===========================================================
    // DMA Mode Registers
    // ===========================================================
    logic [AXI_ADDR_W-1:0]   dma_src_addr_reg;
    logic [ADDR_WIDTH-1:0]   dma_count_reg;
    logic [LAYER_WIDTH-1:0]  dma_layer_reg;
    logic [ADDR_WIDTH-1:0]   dma_dst_addr_reg;

    // Beat extraction: 64-bit beat → 8 weights at 8 bits each
    localparam int WEIGHTS_PER_BEAT = AXI_DATA_W / DATA_WIDTH;
    logic [$clog2(WEIGHTS_PER_BEAT)-1:0] beat_idx;
    logic [AXI_DATA_W-1:0]               beat_data_reg;

    // Weights consumed in current burst (for AXI burst boundary)
    logic [ADDR_WIDTH-1:0] burst_consumed;

    // ===========================================================
    // Status (sticky until cleared)
    // ===========================================================
    logic done_sticky;
    logic error_sticky;

    // ===========================================================
    // FSM Next State
    // ===========================================================
    always_comb begin
        state_next = state;
        case (state)
            WL_IDLE: begin
                if (csr_dma_start && csr_dma_weight_count > 0)
                    state_next = WL_DMA_READ_REQ;
                else if (csr_weight_write)
                    state_next = WL_CSR_WRITE;
            end

            WL_CSR_WRITE: begin
                // Return to IDLE after 1-cycle write pulse
                state_next = WL_IDLE;
            end

            WL_DMA_READ_REQ: begin
                if (m_axi_wt_arready)
                    state_next = WL_DMA_READ_DATA;
            end

            WL_DMA_READ_DATA: begin
                if (m_axi_wt_rvalid) begin
                    if (m_axi_wt_rresp != 2'b00)
                        state_next = WL_ERROR;
                    else
                        state_next = WL_DMA_EXTRACT;
                end
            end

            WL_DMA_EXTRACT: begin
                // After writing the last weight from this beat
                if (dma_count_reg == 0)
                    state_next = WL_DONE;
                else if (beat_idx == WEIGHTS_PER_BEAT - 1)
                    state_next = WL_DMA_NEXT;
                else
                    state_next = WL_DMA_EXTRACT;
            end

            WL_DMA_NEXT: begin
                state_next = (dma_count_reg == 0) ? WL_DONE : WL_DMA_READ_DATA;
            end

            WL_DONE: begin
                state_next = WL_IDLE;
            end

            WL_ERROR: begin
                state_next = WL_IDLE;
            end

            default: state_next = WL_IDLE;
        endcase
    end

    // ===========================================================
    // FSM Sequential Logic
    // ===========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= WL_IDLE;
        end else begin
            state <= state_next;
        end
    end

    // ===========================================================
    // CSR Mode: Address / Count / Layer + burst tracking
    // ===========================================================
    logic csr_burst_active;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            csr_burst_active <= 1'b0;
        end else if (state == WL_IDLE && csr_weight_write && csr_auto_inc_en && csr_weight_count > 0) begin
            csr_burst_active <= 1'b1;
        end else if (state == WL_CSR_WRITE && csr_auto_inc_en && curr_count == 0) begin
            csr_burst_active <= 1'b0;
        end else if (!csr_auto_inc_en) begin
            csr_burst_active <= 1'b0;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            curr_addr   <= '0;
            curr_count  <= '0;
            curr_layer  <= '0;
        end else if (state == WL_IDLE && csr_weight_write && !csr_burst_active) begin
            // Latch CSR values only at start of a new burst or single write
            curr_addr  <= csr_weight_addr;
            curr_count <= csr_weight_count;
            curr_layer <= csr_weight_layer;
        end else if (state == WL_CSR_WRITE && csr_auto_inc_en && curr_count > 0) begin
            // Auto-increment after write
            curr_addr  <= curr_addr + 1'b1;
            curr_count <= curr_count - 1'b1;
        end
    end

    // ===========================================================
    // DMA Mode: Registers
    // ===========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_src_addr_reg  <= '0;
            dma_count_reg     <= '0;
            dma_layer_reg     <= '0;
            dma_dst_addr_reg  <= '0;
            burst_consumed    <= '0;
        end else if (state == WL_IDLE && csr_dma_start) begin
            dma_src_addr_reg  <= csr_dma_src_addr;
            dma_count_reg     <= csr_dma_weight_count;
            dma_layer_reg     <= csr_dma_layer;
            dma_dst_addr_reg  <= csr_dma_dst_addr;
            burst_consumed    <= '0;
        end else if (state == WL_DMA_EXTRACT) begin
            // Each extracted weight decrements count and increments dst addr
            dma_count_reg    <= dma_count_reg - 1'b1;
            dma_dst_addr_reg <= dma_dst_addr_reg + 1'b1;
        end else if (state == WL_DMA_READ_REQ && m_axi_wt_arready) begin
            // Advance source address for next burst
            dma_src_addr_reg <= dma_src_addr_reg + BURST_LEN * (AXI_DATA_W / 8);
            burst_consumed   <= '0;
        end
    end

    // ===========================================================
    // DMA: Beat extraction index and data latch
    // ===========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            beat_idx      <= '0;
            beat_data_reg <= '0;
        end else if (state == WL_DMA_READ_DATA && m_axi_wt_rvalid && m_axi_wt_rresp == 2'b00) begin
            // Latch beat data and reset extraction index
            beat_idx      <= '0;
            beat_data_reg <= m_axi_wt_rdata;
        end else if (state == WL_DMA_EXTRACT) begin
            beat_idx <= beat_idx + 1'b1;
        end
    end

    // ===========================================================
    // Weight Output (combinational)
    // ===========================================================
    always_comb begin
        weight_valid     = 1'b0;
        weight_addr      = '0;
        weight_data      = '0;
        weight_layer_sel = '0;

        case (state)
            WL_CSR_WRITE: begin
                weight_valid     = 1'b1;
                weight_addr      = curr_addr;
                weight_data      = csr_weight_data;
                weight_layer_sel = curr_layer;
            end

            WL_DMA_EXTRACT: begin
                weight_valid     = 1'b1;
                weight_addr      = dma_dst_addr_reg;
                weight_data      = beat_data_reg[beat_idx * DATA_WIDTH +: DATA_WIDTH];
                weight_layer_sel = dma_layer_reg;
            end

            default: ;
        endcase
    end

    // ===========================================================
    // AXI4 Read Channel (DMA mode only)
    // ===========================================================

    always_comb begin
        m_axi_wt_arvalid = (state == WL_DMA_READ_REQ);
        m_axi_wt_araddr  = dma_src_addr_reg;
        m_axi_wt_arlen   = 8'(BURST_LEN - 1);
        m_axi_wt_arsize  = $clog2(AXI_DATA_W / 8);  // bytes per beat
        m_axi_wt_arburst = 2'b01;   // INCR
    end

    assign m_axi_wt_rready = (state == WL_DMA_READ_DATA) ||
                              (state == WL_DMA_NEXT);

    // ===========================================================
    // Status Outputs
    // ===========================================================
    assign weight_load_busy = (state != WL_IDLE);

    // Done pulse: 1-cycle pulse when transitioning to IDLE from DONE
    assign weight_load_done = (state == WL_DONE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done_sticky  <= 1'b0;
            error_sticky <= 1'b0;
        end else if (csr_weight_done_clear) begin
            done_sticky  <= 1'b0;
            error_sticky <= 1'b0;
        end else if (state == WL_DONE) begin
            done_sticky <= 1'b1;
        end else if (state == WL_ERROR) begin
            error_sticky <= 1'b1;
        end
    end

    assign weight_load_error = error_sticky;

    // Remaining count for status readback (matches counter width)
    always_comb begin
        if (state == WL_IDLE)
            weight_count_remain = curr_count;
        else
            weight_count_remain = dma_count_reg;
    end
endmodule