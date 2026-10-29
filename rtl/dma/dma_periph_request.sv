//-----------------------------------------------------------------------------
// Module: dma_periph_request.sv
// Description: Peripheral Request Manager for DMA handshaking.
//              Implements PL330-style peripheral request protocol.
//              Manages flow-controlled transfers between peripherals and DMA.
// Architectural Novelty:
//   - Request/Acknowledge handshaking protocol
//   - Single, Burst, and Flush request types
//   - Per-peripheral FIFO buffers for data staging
//   - Backpressure-aware scheduling with timeout
//   - Runtime-configurable burst sizes per peripheral
// Pipeline: 2-stage (Request → Transfer)
//-----------------------------------------------------------------------------
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off CASEINCOMPLETE */
module dma_periph_request #(
    parameter int unsigned NUM_PERIPHS       = 8,
    parameter int unsigned DATA_WIDTH        = 32,
    parameter int unsigned FIFO_DEPTH        = 16,
    parameter int unsigned TIMEOUT_CYCLES    = 1024,
    parameter int unsigned MAX_BURST_BEATS   = 16
)(
    input  logic                       clk,
    input  logic                       rst_n,

    //==========================================================
    // Peripheral Request Interface (PL330-style)
    //==========================================================
    input  logic [NUM_PERIPHS-1:0]    periph_req_valid,
    input  logic [NUM_PERIPHS-1:0][1:0] periph_req_type,  // SINGLE/BURST/FLUSH
    output logic [NUM_PERIPHS-1:0]    periph_ack_valid,
    output logic [NUM_PERIPHS-1:0][1:0] periph_ack_type,  // ACCEPT/WAIT/FLUSH

    //==========================================================
    // Peripheral Data Interface (Streaming)
    //==========================================================
    // Data from peripherals (for DMA write operations)
    input  logic [NUM_PERIPHS-1:0]    periph_wdata_valid,
    input  logic [DATA_WIDTH-1:0]      periph_wdata [NUM_PERIPHS],
    input  logic [NUM_PERIPHS-1:0]    periph_wdata_last,
    output logic [NUM_PERIPHS-1:0]    periph_wdata_ready,

    // Data to peripherals (for DMA read operations)
    output logic [NUM_PERIPHS-1:0]    periph_rdata_valid,
    output logic [DATA_WIDTH-1:0]     periph_rdata [NUM_PERIPHS],
    output logic [NUM_PERIPHS-1:0]    periph_rdata_last,
    input  logic [NUM_PERIPHS-1:0]    periph_rdata_ready,

    //==========================================================
    // DMA Engine Interface
    //==========================================================
    // Channel arbitration
    output logic [NUM_PERIPHS-1:0]    dma_request,
    input  logic [NUM_PERIPHS-1:0]    dma_grant,

    // DMA write data (from peripheral FIFOs to DMA)
    output logic [DATA_WIDTH-1:0]      dma_wdata,
    output logic                       dma_wdata_valid,
    output logic                       dma_wdata_last,
    output logic [3:0]                 dma_wdata_periph_id,
    input  logic                       dma_wdata_ready,

    // DMA read data (from DMA to peripheral FIFOs)
    input  logic [DATA_WIDTH-1:0]      dma_rdata,
    input  logic                       dma_rdata_valid,
    input  logic                       dma_rdata_last,
    input  logic [3:0]                 dma_rdata_periph_id,
    output logic                       dma_rdata_ready,

    //==========================================================
    // CSR Configuration
    //==========================================================
    input  logic [NUM_PERIPHS-1:0]    periph_enable,
    input  logic [NUM_PERIPHS-1:0][3:0] periph_burst_size,  // Beats per burst
    input  logic [NUM_PERIPHS-1:0]    periph_flow_ctrl,      // 1 = hardware flow control
    input  logic [NUM_PERIPHS-1:0]    periph_irq_enable,

    //==========================================================
    // Status and Interrupts
    //==========================================================
    output logic [NUM_PERIPHS-1:0]    periph_active,
    output logic [NUM_PERIPHS-1:0]    periph_error,
    output logic [NUM_PERIPHS-1:0]    periph_timeout,
    output logic [NUM_PERIPHS-1:0]    periph_overflow,
    output logic [NUM_PERIPHS-1:0]    periph_underflow,
    output logic [15:0]                periph_transfer_count [NUM_PERIPHS],
    output logic                       irq
);

    //==========================================================
    // Internal Types (PL330-style)
    //==========================================================
    // Request types - defined locally for portability
    typedef enum logic [1:0] {
        REQ_SINGLE   = 2'b00,  // Single transfer request
        REQ_BURST    = 2'b01,  // Burst transfer request
        REQ_FLUSH    = 2'b10   // Flush request
    } periph_req_type_e;

    // Acknowledge types - defined locally for portability
    typedef enum logic [1:0] {
        ACK_ACCEPT    = 2'b00,  // Request accepted
        ACK_WAIT      = 2'b01,  // Wait for space
        ACK_FLUSH     = 2'b10   // Flush complete
    } periph_ack_type_e;

    //==========================================================
    // Internal Constants
    //==========================================================
    localparam PERIPH_ID_WIDTH = $clog2(NUM_PERIPHS);

    //==========================================================
    // Internal Types
    //==========================================================
    typedef enum logic [2:0] {
        PR_IDLE,
        PR_REQUEST,
        PR_ACCEPTED,
        PR_TRANSFER,
        PR_WAIT_FLUSH,
        PR_COMPLETE,
        PR_ERROR
    } periph_state_e;

    //==========================================================
    // Internal State
    //==========================================================
    periph_state_e periph_state [NUM_PERIPHS];
    periph_state_e periph_next_state [NUM_PERIPHS];

    // Per-peripheral counters
    logic [NUM_PERIPHS-1:0][15:0]    beat_counter;
    logic [NUM_PERIPHS-1:0][15:0]    transfer_counter;
    logic [NUM_PERIPHS-1:0][15:0]    timeout_counter;
    logic [NUM_PERIPHS-1:0][3:0]     burst_beats_remaining;

    // Per-peripheral FIFOs
    logic [NUM_PERIPHS-1:0][$clog2(FIFO_DEPTH):0] fifo_wr_ptr;
    logic [NUM_PERIPHS-1:0][$clog2(FIFO_DEPTH):0] fifo_rd_ptr;
    logic [NUM_PERIPHS-1:0][DATA_WIDTH-1:0]       fifo_mem [0:FIFO_DEPTH-1];
    logic [NUM_PERIPHS-1:0]                       fifo_full;
    logic [NUM_PERIPHS-1:0]                       fifo_empty;
    logic [NUM_PERIPHS-1:0]                       fifo_valid;

    //==========================================================
    // FIFO Management (Per-Peripheral)
    //==========================================================
    generate
        for (genvar p = 0; p < NUM_PERIPHS; p++) begin : gen_periph_fifo
            // FIFO full/empty detection
            assign fifo_full[p] = (fifo_wr_ptr[p][$clog2(FIFO_DEPTH)] !=
                                   fifo_rd_ptr[p][$clog2(FIFO_DEPTH)]) &&
                                  (fifo_wr_ptr[p][$clog2(FIFO_DEPTH)-1:0] ==
                                   fifo_rd_ptr[p][$clog2(FIFO_DEPTH)-1:0]);
            assign fifo_empty[p] = (fifo_wr_ptr[p] == fifo_rd_ptr[p]);

            // FIFO write (from peripheral)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    fifo_wr_ptr[p] <= '0;
                end else if (periph_wdata_valid[p] && periph_wdata_ready[p] && !fifo_full[p]) begin
                    fifo_mem[p][fifo_wr_ptr[p][$clog2(FIFO_DEPTH)-1:0]] <= periph_wdata[p];
                    fifo_wr_ptr[p] <= fifo_wr_ptr[p] + 1'b1;
                end
            end

            // FIFO read (to DMA)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    fifo_rd_ptr[p] <= '0;
                    fifo_valid[p] <= 1'b0;
                end else if (!fifo_empty[p] && dma_wdata_ready && dma_grant[p]) begin
                    fifo_valid[p] <= 1'b1;
                end else begin
                    fifo_valid[p] <= 1'b0;
                end
            end
        end
    endgenerate

    //==========================================================
    // Per-Peripheral State Machine - Sequential
    //==========================================================
    generate
        for (genvar p = 0; p < NUM_PERIPHS; p++) begin : gen_periph_state
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    periph_state[p] <= PR_IDLE;
                    beat_counter[p] <= '0;
                    transfer_counter[p] <= '0;
                    timeout_counter[p] <= '0;
                    burst_beats_remaining[p] <= '0;
                end else begin
                    periph_state[p] <= periph_next_state[p];

                    case (periph_state[p])
                        PR_REQUEST: begin
                            // Wait for acknowledgment
                            if (periph_ack_valid[p] && periph_ack_type[p] == ACK_ACCEPT) begin
                                burst_beats_remaining[p] <= periph_burst_size[p];
                            end
                        end

                        PR_TRANSFER: begin
                            // Count transferred beats
                            if (dma_wdata_valid && dma_wdata_ready && dma_grant[p]) begin
                                beat_counter[p] <= beat_counter[p] + 1'b1;
                                if (burst_beats_remaining[p] > 0) begin
                                    burst_beats_remaining[p] <= burst_beats_remaining[p] - 1'b1;
                                end
                            end
                        end

                        PR_WAIT_FLUSH: begin
                            // Wait for flush completion
                            if (periph_req_valid[p] && periph_req_type[p] == REQ_FLUSH) begin
                                // Flush complete
                            end
                        end

                        PR_COMPLETE: begin
                            // Update transfer count
                            transfer_counter[p] <= transfer_counter[p] + beat_counter[p];
                            beat_counter[p] <= '0;
                        end
                    endcase

                    // Timeout counter
                    if (periph_state[p] == PR_REQUEST || periph_state[p] == PR_TRANSFER) begin
                        if (timeout_counter[p] < TIMEOUT_CYCLES[15:0]) begin
                            timeout_counter[p] <= timeout_counter[p] + 1'b1;
                        end
                    end else begin
                        timeout_counter[p] <= '0;
                    end
                end
            end
        end
    endgenerate

    //==========================================================
    // Per-Peripheral State Machine - Combinational
    //==========================================================
    generate
        for (genvar p = 0; p < NUM_PERIPHS; p++) begin : gen_periph_next_state
            always_comb begin
                periph_next_state[p] = periph_state[p];

                case (periph_state[p])
                    PR_IDLE: begin
                        if (periph_enable[p] && periph_req_valid[p] && !fifo_full[p]) begin
                            periph_next_state[p] = PR_REQUEST;
                        end
                    end

                    PR_REQUEST: begin
                        if (periph_ack_valid[p]) begin
                            case (periph_ack_type[p])
                                ACK_ACCEPT: begin
                                    periph_next_state[p] = PR_ACCEPTED;
                                end
                                ACK_WAIT: begin
                                    periph_next_state[p] = PR_REQUEST; // Stay waiting
                                end
                                ACK_FLUSH: begin
                                    periph_next_state[p] = PR_WAIT_FLUSH;
                                end
                            endcase
                        end else if (timeout_counter[p] >= TIMEOUT_CYCLES[15:0]) begin
                            periph_next_state[p] = PR_ERROR;
                        end
                    end

                    PR_ACCEPTED: begin
                        if (|periph_burst_size[p]) begin
                            periph_next_state[p] = PR_TRANSFER;
                        end
                    end

                    PR_TRANSFER: begin
                        // Check transfer complete conditions
                        if (periph_wdata_last[p] && periph_wdata_valid[p]) begin
                            // Last data received
                            periph_next_state[p] = PR_COMPLETE;
                        end else if (burst_beats_remaining[p] == 0 && !fifo_empty[p]) begin
                            // Burst complete, more data pending
                            periph_next_state[p] = PR_REQUEST;
                        end else if (periph_req_type[p] == REQ_FLUSH) begin
                            periph_next_state[p] = PR_WAIT_FLUSH;
                        end else if (timeout_counter[p] >= TIMEOUT_CYCLES[15:0]) begin
                            periph_next_state[p] = PR_ERROR;
                        end
                    end

                    PR_WAIT_FLUSH: begin
                        if (periph_ack_type[p] == ACK_FLUSH) begin
                            periph_next_state[p] = PR_COMPLETE;
                        end
                    end

                    PR_COMPLETE: begin
                        periph_next_state[p] = PR_IDLE;
                    end

                    PR_ERROR: begin
                        periph_next_state[p] = PR_IDLE;
                    end

                    default: begin
                        periph_next_state[p] = PR_IDLE;
                    end
                endcase
            end
        end
    endgenerate

    //==========================================================
    // Acknowledge Generation
    //==========================================================
    generate
        for (genvar p = 0; p < NUM_PERIPHS; p++) begin : gen_periph_ack
            always_comb begin
                periph_ack_valid[p] = 1'b0;
                periph_ack_type[p] = ACK_WAIT;

                case (periph_state[p])
                    PR_IDLE: begin
                        // Auto-accept if enabled and FIFO not full
                        if (periph_enable[p] && periph_req_valid[p]) begin
                            if (!fifo_full[p]) begin
                                periph_ack_valid[p] = 1'b1;
                                periph_ack_type[p] = ACK_ACCEPT;
                            end else begin
                                periph_ack_valid[p] = 1'b1;
                                periph_ack_type[p] = ACK_WAIT;
                            end
                        end
                    end

                    PR_REQUEST: begin
                        periph_ack_valid[p] = 1'b1;
                        if (!fifo_full[p]) begin
                            periph_ack_type[p] = ACK_ACCEPT;
                        end else begin
                            periph_ack_type[p] = ACK_WAIT;
                        end
                    end

                    PR_TRANSFER: begin
                        // Accept burst continuation
                        if (periph_req_valid[p] && !fifo_full[p]) begin
                            periph_ack_valid[p] = 1'b1;
                            periph_ack_type[p] = ACK_ACCEPT;
                        end
                    end

                    PR_WAIT_FLUSH: begin
                        periph_ack_valid[p] = 1'b1;
                        periph_ack_type[p] = ACK_FLUSH;
                    end

                    default: begin
                        periph_ack_valid[p] = 1'b0;
                    end
                endcase
            end
        end
    endgenerate

    //==========================================================
    // DMA Request Generation
    //==========================================================
    generate
        for (genvar p = 0; p < NUM_PERIPHS; p++) begin : gen_dma_req
            // Request DMA when FIFO has data and peripheral is active
            assign dma_request[p] = periph_enable[p] &&
                                    !fifo_empty[p] &&
                                    periph_state[p] == PR_TRANSFER;
        end
    endgenerate

    //==========================================================
    // Peripheral Data Ready Signals
    //==========================================================
    generate
        for (genvar p = 0; p < NUM_PERIPHS; p++) begin : gen_periph_ready
            assign periph_wdata_ready[p] = !fifo_full[p] &&
                                            periph_state[p] == PR_TRANSFER;
        end
    endgenerate

    //==========================================================
    // DMA Write Data Mux
    //==========================================================
    // Select data from granted peripheral
    always_comb begin
        dma_wdata = '0;
        dma_wdata_valid = 1'b0;
        dma_wdata_last = 1'b0;
        dma_wdata_periph_id = '0;

        for (int p = 0; p < NUM_PERIPHS; p++) begin
            if (dma_grant[p] && fifo_valid[p]) begin
                dma_wdata = fifo_mem[p][fifo_rd_ptr[p][$clog2(FIFO_DEPTH)-1:0]];
                dma_wdata_valid = 1'b1;
                dma_wdata_last = periph_wdata_last[p];
                dma_wdata_periph_id = 4'(p[PERIPH_ID_WIDTH-1:0]);  // Zero-extend to 4 bits
            end
        end
    end

    //==========================================================
    // DMA Read Data Demux (To Peripherals)
    //==========================================================
    generate
        for (genvar p = 0; p < NUM_PERIPHS; p++) begin : gen_periph_rdata
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    periph_rdata_valid[p] <= 1'b0;
                    periph_rdata[p] <= '0;
                    periph_rdata_last[p] <= 1'b0;
                end else begin
                    if (dma_rdata_valid && dma_rdata_periph_id == 4'(p[PERIPH_ID_WIDTH-1:0])) begin
                        periph_rdata_valid[p] <= dma_rdata_valid;
                        periph_rdata[p] <= dma_rdata;
                        periph_rdata_last[p] <= dma_rdata_last;
                    end else begin
                        periph_rdata_valid[p] <= 1'b0;
                    end
                end
            end
        end
    endgenerate

    assign dma_rdata_ready = 1'b1; // Always ready to accept DMA read data

    //==========================================================
    // Status and Error Flags
    //==========================================================
    generate
        for (genvar p = 0; p < NUM_PERIPHS; p++) begin : gen_periph_status
            assign periph_active[p] = (periph_state[p] == PR_TRANSFER) ||
                                      (periph_state[p] == PR_REQUEST);
            assign periph_error[p] = (periph_state[p] == PR_ERROR);
            assign periph_timeout[p] = (timeout_counter[p] >= TIMEOUT_CYCLES[15:0]);
            assign periph_overflow[p] = fifo_full[p] && periph_wdata_valid[p];
            assign periph_underflow[p] = fifo_empty[p] && fifo_valid[p];
        end
    endgenerate

    // Transfer count output
    generate
        for (genvar p = 0; p < NUM_PERIPHS; p++) begin : gen_xfer_count
            assign periph_transfer_count[p] = transfer_counter[p];
        end
    endgenerate

    //==========================================================
    // Interrupt Generation
    //==========================================================
    logic [NUM_PERIPHS-1:0] pending_irq;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pending_irq <= '0;
        end else begin
            for (int p = 0; p < NUM_PERIPHS; p++) begin
                if (periph_error[p] || periph_timeout[p]) begin
                    pending_irq[p] <= 1'b1;
                end else if (!periph_irq_enable[p]) begin
                    pending_irq[p] <= 1'b0;
                end
            end
        end
    end

    assign irq = |(pending_irq & periph_irq_enable);

    //==========================================================
    // Assertions
    //==========================================================
    `ifdef FORMAL
        // Verify only one peripheral granted at a time
        assert property (@(posedge clk) disable iff (!rst_n)
            $onehot0(dma_grant))
            else $error("Multiple peripherals granted simultaneously");

        // Verify FIFO doesn't overflow
        for (genvar p = 0; p < NUM_PERIPHS; p++) begin : gen_fifo_assert
            assert property (@(posedge clk) disable iff (!rst_n)
                !periph_overflow[p])
                else $error("Peripheral %0d FIFO overflow", p);
        end

        // Verify request/acknowledge protocol
        for (genvar p = 0; p < NUM_PERIPHS; p++) begin : gen_protocol_assert
            assert property (@(posedge clk) disable iff (!rst_n)
                periph_req_valid[p] && periph_ack_valid[p] |->
                (periph_ack_type[p] == ACK_ACCEPT ||
                 periph_ack_type[p] == ACK_WAIT ||
                 periph_ack_type[p] == ACK_FLUSH))
                else $error("Peripheral %0d invalid acknowledge", p);
        end
    `endif

endmodule : dma_periph_request

