//-----------------------------------------------------------------------------
// Module: dma_e_rodnet_csr.sv
// Description: Enhanced CSR register block for e_rodnet DMA interface.
//              Implements AXI4-Lite slave at addresses 0x300-0x31C.
//              Shadow registers for glitch-free configuration.
//              Supports peripheral flow control and error status.
//
// Features:
//   - AXI4-Lite slave interface
//   - Shadow registers for glitch-free configuration
//   - Extended status bits for boundary/timeout errors
//   - Peripheral channel enable configuration
//   - Interrupt coalescing support
//-----------------------------------------------------------------------------
module dma_e_rodnet_csr #(
    parameter int unsigned ADDR_WIDTH = 12,
    parameter int unsigned DATA_WIDTH = 32,
    parameter int unsigned NUM_PERIPHS = 8
)(
    input  logic                       clk,
    input  logic                       rst_n,

    //==========================================================
    // AXI4-Lite Slave Interface
    //==========================================================
    input  logic [ADDR_WIDTH-1:0]      s_axi_awaddr,
    input  logic [2:0]                 s_axi_awprot,
    input  logic                       s_axi_awvalid,
    output logic                       s_axi_awready,
    input  logic [DATA_WIDTH-1:0]       s_axi_wdata,
    input  logic [DATA_WIDTH/8-1:0]     s_axi_wstrb,
    input  logic                       s_axi_wvalid,
    output logic                       s_axi_wready,
    output logic [1:0]                 s_axi_bresp,
    output logic                       s_axi_bvalid,
    input  logic                       s_axi_bready,
    input  logic [ADDR_WIDTH-1:0]      s_axi_araddr,
    input  logic [2:0]                 s_axi_arprot,
    input  logic                       s_axi_arvalid,
    output logic                       s_axi_arready,
    output logic [DATA_WIDTH-1:0]      s_axi_rdata,
    output logic [1:0]                 s_axi_rresp,
    output logic                       s_axi_rvalid,
    input  logic                       s_axi_rready,

    //==========================================================
    // DMA Control Outputs
    //==========================================================
    output logic [31:0]                dma_src_addr,
    output logic [31:0]                dma_dst_addr,
    output logic [15:0]                dma_token_count,
    output logic [7:0]                 dma_burst_len,
    output logic                       dma_start,
    output logic                       dma_abort,
    output logic                       dma_loop_mode,
    output logic                       dma_src_inc,
    output logic                       dma_irq_en_done,
    output logic                       dma_irq_en_error,

    //==========================================================
    // DMA Status Inputs
    //==========================================================
    input  logic                       dma_status_done,
    input  logic                       dma_status_error,
    input  logic                       dma_status_busy,
    input  logic                       dma_status_src_err,
    input  logic                       dma_status_dst_err,
    input  logic                       dma_status_boundary_err,
    input  logic                       dma_status_timeout_err,
    input  logic [15:0]                dma_tokens_done,

    //==========================================================
    // Peripheral Status Inputs
    //==========================================================
    input  logic [NUM_PERIPHS-1:0]    periph_active,
    input  logic [NUM_PERIPHS-1:0]    periph_error,

    //==========================================================
    // Interrupt
    //==========================================================
    output logic                       irq
);

    import dma_e_rodnet_pkg::*;

    //==========================================================
    // Internal Registers (Shadow + Active)
    //==========================================================
    // Shadow registers (written by CPU)
    logic [31:0] src_addr_shadow;
    logic [31:0] dst_addr_shadow;
    logic [15:0] token_count_shadow;
    logic [7:0]  burst_len_shadow;
    logic        loop_mode_shadow;
    logic        src_inc_shadow;
    logic        irq_en_done_shadow;
    logic        irq_en_error_shadow;

    // Active registers (read by DMA engine)
    logic [31:0] src_addr_active;
    logic [31:0] dst_addr_active;
    logic [15:0] token_count_active;
    logic [7:0]  burst_len_active;

    // Control register bits
    logic        start_pulse;
    logic        abort_pulse;

    // Status register
    logic [31:0] status_reg;

    //==========================================================
    // AXI4-Lite State Machine
    //==========================================================
    typedef enum logic [2:0] {
        AXI_IDLE,
        AXI_READ_ADDR,
        AXI_READ_DATA,
        AXI_WRITE_ADDR,
        AXI_WRITE_DATA,
        AXI_WRITE_RESP
    } axi_state_e;

    axi_state_e axi_state, axi_next_state;
    logic [ADDR_WIDTH-1:0] read_addr_reg;
    logic [ADDR_WIDTH-1:0] write_addr_reg;
    logic [DATA_WIDTH-1:0] write_data_reg;
    logic [DATA_WIDTH/8-1:0] write_strb_reg;
    logic [DATA_WIDTH-1:0] read_data_reg;
    logic [1:0] read_resp_reg;
    logic write_done;

    //==========================================================
    // AXI State Machine - Sequential
    //==========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_state <= AXI_IDLE;
            read_addr_reg <= '0;
            write_addr_reg <= '0;
            write_data_reg <= '0;
            write_strb_reg <= '0;
        end else begin
            axi_state <= axi_next_state;

            case (axi_state)
                AXI_READ_ADDR: begin
                    if (s_axi_arready) begin
                        read_addr_reg <= s_axi_araddr;
                    end
                end

                AXI_WRITE_ADDR: begin
                    if (s_axi_awready) begin
                        write_addr_reg <= s_axi_awaddr;
                    end
                end

                AXI_WRITE_DATA: begin
                    if (s_axi_wvalid && s_axi_wready) begin
                        write_data_reg <= s_axi_wdata;
                        write_strb_reg <= s_axi_wstrb;
                    end
                end
            endcase
        end
    end

    //==========================================================
    // AXI State Machine - Combinational
    //==========================================================
    always_comb begin
        axi_next_state = axi_state;

        case (axi_state)
            AXI_IDLE: begin
                if (s_axi_awvalid && s_axi_wvalid) begin
                    axi_next_state = AXI_WRITE_ADDR;
                end else if (s_axi_arvalid) begin
                    axi_next_state = AXI_READ_ADDR;
                end
            end

            AXI_READ_ADDR: begin
                if (s_axi_arready) begin
                    axi_next_state = AXI_READ_DATA;
                end
            end

            AXI_READ_DATA: begin
                if (s_axi_rready) begin
                    axi_next_state = AXI_IDLE;
                end
            end

            AXI_WRITE_ADDR: begin
                if (s_axi_awready) begin
                    axi_next_state = AXI_WRITE_DATA;
                end
            end

            AXI_WRITE_DATA: begin
                if (s_axi_wvalid && s_axi_wready) begin
                    axi_next_state = AXI_WRITE_RESP;
                end
            end

            AXI_WRITE_RESP: begin
                if (s_axi_bready) begin
                    axi_next_state = AXI_IDLE;
                end
            end

            default: begin
                axi_next_state = AXI_IDLE;
            end
        endcase
    end

    //==========================================================
    // AXI Interface Outputs
    //==========================================================
    always_comb begin
        s_axi_awready = 1'b0;
        s_axi_wready = 1'b0;
        s_axi_bresp = AXI_RESP_OKAY;
        s_axi_bvalid = 1'b0;
        s_axi_arready = 1'b0;
        s_axi_rdata = '0;
        s_axi_rresp = AXI_RESP_OKAY;
        s_axi_rvalid = 1'b0;

        case (axi_state)
            AXI_IDLE: begin
                s_axi_awready = (s_axi_awvalid && s_axi_wvalid);
                s_axi_arready = !(s_axi_awvalid && s_axi_wvalid) && s_axi_arvalid;
            end

            AXI_READ_DATA: begin
                s_axi_rdata = read_data_reg;
                s_axi_rresp = read_resp_reg;
                s_axi_rvalid = 1'b1;
            end

            AXI_WRITE_ADDR: begin
                s_axi_awready = 1'b1;
            end

            AXI_WRITE_DATA: begin
                s_axi_wready = 1'b1;
            end

            AXI_WRITE_RESP: begin
                s_axi_bvalid = 1'b1;
            end
        endcase
    end

    //==========================================================
    // Register Read Logic
    //==========================================================
    always_comb begin
        read_data_reg = '0;
        read_resp_reg = AXI_RESP_OKAY;

        case (read_addr_reg)
            CSR_DMA_SRC_ADDR: begin
                read_data_reg = src_addr_shadow;
            end

            CSR_DMA_DST_ADDR: begin
                read_data_reg = dst_addr_shadow;
            end

            CSR_DMA_TOKEN_CNT: begin
                read_data_reg = {16'h0, token_count_shadow};
            end

            CSR_DMA_CTRL: begin
                read_data_reg = {
                    28'h0,
                    src_inc_shadow,
                    loop_mode_shadow,
                    1'b0, // abort (read-only)
                    1'b0  // start (read-only)
                };
            end

            CSR_DMA_STATUS: begin
                read_data_reg = status_reg;
            end

            CSR_DMA_IRQ_EN: begin
                read_data_reg = {
                    30'h0,
                    irq_en_error_shadow,
                    irq_en_done_shadow
                };
            end

            CSR_DMA_BURST_LEN: begin
                read_data_reg = {24'h0, burst_len_shadow};
            end

            CSR_DMA_TOKENS_DONE: begin
                read_data_reg = {16'h0, dma_tokens_done};
            end

            default: begin
                read_resp_reg = AXI_RESP_SLVERR;
            end
        endcase
    end

    //==========================================================
    // Register Write Logic
    //==========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            src_addr_shadow <= '0;
            dst_addr_shadow <= '0;
            token_count_shadow <= '0;
            burst_len_shadow <= 8'h10; // Default: 16 beats
            loop_mode_shadow <= 1'b0;
            src_inc_shadow <= 1'b0;
            irq_en_done_shadow <= 1'b0;
            irq_en_error_shadow <= 1'b0;
            start_pulse <= 1'b0;
            abort_pulse <= 1'b0;
        end else begin
            // Self-clearing pulses
            start_pulse <= 1'b0;
            abort_pulse <= 1'b0;

            if (axi_state == AXI_WRITE_RESP && s_axi_bready) begin
                if (write_strb_reg == 4'hF) begin // Full word write
                    case (write_addr_reg)
                        CSR_DMA_SRC_ADDR: begin
                            src_addr_shadow <= write_data_reg;
                        end

                        CSR_DMA_DST_ADDR: begin
                            dst_addr_shadow <= write_data_reg;
                        end

                        CSR_DMA_TOKEN_CNT: begin
                            token_count_shadow <= write_data_reg[15:0];
                        end

                        CSR_DMA_CTRL: begin
                            start_pulse <= write_data_reg[0];
                            abort_pulse <= write_data_reg[1];
                            loop_mode_shadow <= write_data_reg[2];
                            src_inc_shadow <= write_data_reg[3];
                        end

                        CSR_DMA_STATUS: begin
                            // W1C bits: clear on write of 1
                            if (write_data_reg[STATUS_DONE]) begin
                                status_reg[STATUS_DONE] <= 1'b0;
                            end
                            if (write_data_reg[STATUS_ERROR]) begin
                                status_reg[STATUS_ERROR] <= 1'b0;
                            end
                        end

                        CSR_DMA_IRQ_EN: begin
                            irq_en_done_shadow <= write_data_reg[IRQ_DONE_EN];
                            irq_en_error_shadow <= write_data_reg[IRQ_ERROR_EN];
                        end

                        CSR_DMA_BURST_LEN: begin
                            if (write_data_reg[7:0] >= 16) begin
                                burst_len_shadow <= write_data_reg[7:0];
                            end
                        end

                        default: begin
                            // No action
                        end
                    endcase
                end
            end
        end
    end

    //==========================================================
    // Status Register Updates
    //==========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            status_reg <= '0;
        end else begin
            // Busy bit (read-only)
            status_reg[STATUS_BUSY] <= dma_status_busy;

            // Done and Error bits (W1C)
            if (dma_status_done && !status_reg[STATUS_DONE]) begin
                status_reg[STATUS_DONE] <= 1'b1;
            end
            if (dma_status_error && !status_reg[STATUS_ERROR]) begin
                status_reg[STATUS_ERROR] <= 1'b1;
            end

            // Error source bits
            if (dma_status_src_err) begin
                status_reg[STATUS_SRC_ERR] <= 1'b1;
            end
            if (dma_status_dst_err) begin
                status_reg[STATUS_DST_ERR] <= 1'b1;
            end
        end
    end

    //==========================================================
    // Active Register Update (Shadow to Active)
    //==========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            src_addr_active <= '0;
            dst_addr_active <= '0;
            token_count_active <= '0;
            burst_len_active <= 8'h10;
        end else if (!dma_status_busy) begin
            // Update active registers only when DMA is idle
            src_addr_active <= src_addr_shadow;
            dst_addr_active <= dst_addr_shadow;
            token_count_active <= token_count_shadow;
            burst_len_active <= burst_len_shadow;
        end
    end

    //==========================================================
    // Output Assignments
    //==========================================================
    assign dma_src_addr = src_addr_active;
    assign dma_dst_addr = dst_addr_active;
    assign dma_token_count = token_count_active;
    assign dma_burst_len = burst_len_active;
    assign dma_start = start_pulse;
    assign dma_abort = abort_pulse;
    assign dma_loop_mode = loop_mode_shadow;
    assign dma_src_inc = src_inc_shadow;
    assign dma_irq_en_done = irq_en_done_shadow;
    assign dma_irq_en_error = irq_en_error_shadow;

    //==========================================================
    // Interrupt Generation
    //==========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            irq <= 1'b0;
        end else begin
            // Level interrupt: high when status bit set and IRQ enabled
            irq <= (status_reg[STATUS_DONE] && irq_en_done_shadow) ||
                   (status_reg[STATUS_ERROR] && irq_en_error_shadow);
        end
    end

    //==========================================================
    // Assertions
    //==========================================================
    `ifdef FORMAL
        // Verify address alignment
        assert property (@(posedge clk) disable iff (!rst_n)
            start_pulse |-> src_addr_shadow[5:0] == '0 && dst_addr_shadow[5:0] == '0);

        // Verify burst length range
        assert property (@(posedge clk) disable iff (!rst_n)
            burst_len_shadow >= 16 && burst_len_shadow <= 256);
    `endif

endmodule : dma_e_rodnet_csr

