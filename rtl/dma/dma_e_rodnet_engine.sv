//-----------------------------------------------------------------------------
// Module: dma_e_rodnet_engine.sv
// Description: Enhanced DMA engine for e_rodnet with peripheral flow control.
//              Target: Shakti Yamuna SoC — AXI4 Full, 32-bit data, 32-bit addr
//
// Features:
//   - Dual source/sink paths (DDR→IP and IP→DDR)
//   - Token-level streaming interface (512-bit)
//   - PL330-style peripheral request/acknowledge flow control
//   - 4KB boundary detection and automatic split
//   - Error detection and recovery
//   - Configurable burst lengths (16-256 beats)
//   - Backpressure handling with FIFO buffers
//
// Architecture:
//   - Source Path: DDR read → token assembly → IP input FIFO
//   - Sink Path:   IP output FIFO → token disassembly → DDR write
//   - Both paths operate concurrently with independent state machines
//
// Pipeline: Overlapped read/write with backpressure support
//
// FIXES APPLIED:
//   - Parameterized AXI ID width instead of hardcoded values
//   - Fixed BYTES_PER_BEAT reference from package
//   - Removed unused parameter warnings
//-----------------------------------------------------------------------------
module dma_e_rodnet_engine #(
    parameter int unsigned AXI_DATA_WIDTH   = 32,
    parameter int unsigned AXI_ADDR_WIDTH   = 32,
    parameter int unsigned AXI_ID_WIDTH     = 4,     // FIX: Added parameter for ID width
    parameter int unsigned TOKEN_WIDTH      = 512,
    parameter int unsigned BEATS_PER_TOKEN  = 16,
    parameter int unsigned MAX_TOKENS       = 65535,
    parameter int unsigned FIFO_DEPTH       = 4,
    parameter int unsigned TIMEOUT_CYCLES   = 4096,
    parameter int unsigned NUM_PERIPHS      = 8
)(
    input  logic                       clk,
    input  logic                       rst_n,

    //==========================================================
    // AXI4 Master Interface (to DDR - Shakti SoC)
    //==========================================================
    // Write Address Channel
    output logic [AXI_ID_WIDTH-1:0]    m_awid,
    output logic [AXI_ADDR_WIDTH-1:0] m_awaddr,
    output logic [7:0]                 m_awlen,
    output logic [2:0]                 m_awsize,
    output logic [1:0]                 m_awburst,
    output logic                       m_awlock,
    output logic [3:0]                 m_awcache,
    output logic [2:0]                 m_awprot,
    output logic [3:0]                 m_awqos,
    output logic                       m_awvalid,
    input  logic                       m_awready,

    // Write Data Channel
    output logic [AXI_DATA_WIDTH-1:0]  m_wdata,
    output logic [AXI_DATA_WIDTH/8-1:0]m_wstrb,
    output logic                       m_wlast,
    output logic                       m_wvalid,
    input  logic                       m_wready,

    // Write Response Channel
    input  logic [AXI_ID_WIDTH-1:0]    m_bid,
    input  logic [1:0]                 m_bresp,
    input  logic                       m_bvalid,
    output logic                       m_bready,

    // Read Address Channel
    output logic [AXI_ID_WIDTH-1:0]    m_arid,
    output logic [AXI_ADDR_WIDTH-1:0]  m_araddr,
    output logic [7:0]                 m_arlen,
    output logic [2:0]                 m_arsize,
    output logic [1:0]                 m_arburst,
    output logic                       m_arlock,
    output logic [3:0]                 m_arcache,
    output logic [2:0]                 m_arprot,
    output logic [3:0]                 m_arqos,
    output logic                       m_arvalid,
    input  logic                       m_arready,

    // Read Data Channel
    input  logic [AXI_ID_WIDTH-1:0]    m_rid,
    input  logic [AXI_DATA_WIDTH-1:0]  m_rdata,
    input  logic [1:0]                 m_rresp,
    input  logic                       m_rlast,
    input  logic                       m_rvalid,
    output logic                       m_rready,

    //==========================================================
    // Source Path: DDR → IP (512-bit streaming)
    //==========================================================
    output logic                       src_valid,
    output logic [TOKEN_WIDTH-1:0]      src_data,
    output logic                       src_last,
    input  logic                       src_ready,

    //==========================================================
    // Sink Path: IP → DDR (512-bit streaming)
    //==========================================================
    input  logic                       dst_valid,
    input  logic [TOKEN_WIDTH-1:0]      dst_data,
    input  logic                       dst_last,
    output logic                       dst_ready,

    //==========================================================
    // Peripheral Request Interface (PL330-style flow control)
    //==========================================================
    input  logic [NUM_PERIPHS-1:0]    periph_req_valid,
    input  logic [NUM_PERIPHS-1:0][1:0] periph_req_type,
    output logic [NUM_PERIPHS-1:0]    periph_ack_valid,
    output logic [NUM_PERIPHS-1:0][1:0] periph_ack_type,

    //==========================================================
    // Control/Status (from CSR)
    //==========================================================
    input  logic [AXI_ADDR_WIDTH-1:0]   src_addr,
    input  logic [AXI_ADDR_WIDTH-1:0]   dst_addr,
    input  logic [15:0]                token_count,
    input  logic [7:0]                 burst_len,     // Beats per burst (16-256)
    input  logic                       start,
    input  logic                       abort,
    input  logic                       loop_mode,
    input  logic                       src_inc,

    output logic                       done,
    output logic                       busy,
    output logic                       error,
    output logic                       src_err,
    output logic                       dst_err,
    output logic                       boundary_err,
    output logic                       timeout_err,
    output logic [15:0]                tokens_done,

    // Interrupt
    output logic                       irq
);

    import dma_e_rodnet_pkg::*;

    //==========================================================
    // Internal Constants
    //==========================================================
    // Use BYTES_PER_BEAT from package instead of localparam to avoid VARHIDDEN
    localparam int TOKEN_BYTES     = TOKEN_WIDTH / 8;    // 64 bytes
    localparam int BEATS_PER_4KB   = 4096 / BYTES_PER_BEAT; // 1024 beats

    //==========================================================
    // Source Path State Machine
    //==========================================================
    typedef enum logic [3:0] {
        SRC_IDLE,
        SRC_ADDR_REQ,
        SRC_ADDR_WAIT,
        SRC_DATA_RECV,
        SRC_ASSEMBLE,
        SRC_WAIT_PERIPH,
        SRC_COMPLETE,
        SRC_ERROR,
        SRC_TIMEOUT
    } src_state_e;

    src_state_e src_state, src_next_state;

    //==========================================================
    // Sink Path State Machine
    //==========================================================
    typedef enum logic [3:0] {
        DST_IDLE,
        DST_WAIT_DATA,
        DST_ADDR_REQ,
        DST_ADDR_WAIT,
        DST_DATA_SEND,
        DST_RESP_WAIT,
        DST_COMPLETE,
        DST_ERROR
    } dst_state_e;

    dst_state_e dst_state, dst_next_state;

    //==========================================================
    // Internal Signals - Source Path
    //==========================================================
    logic [AXI_ADDR_WIDTH-1:0] src_current_addr;
    logic [AXI_ADDR_WIDTH-1:0] src_next_addr;  // Calculated address for next burst
    logic [15:0]               src_token_count;
    logic [15:0]               src_token_remaining;
    logic [7:0]                src_burst_len;
    logic [7:0]                src_beat_count;
    logic [15:0]               src_timeout_counter;
    logic                       src_boundary_cross;
    logic                       src_fifo_full;
    logic                       src_fifo_empty;
    logic                       src_fifo_write;
    logic                       src_fifo_read;
    logic [TOKEN_WIDTH-1:0]     src_fifo_din;
    logic [TOKEN_WIDTH-1:0]     src_fifo_dout;

    //==========================================================
    // Internal Signals - Sink Path
    //==========================================================
    logic [AXI_ADDR_WIDTH-1:0] dst_current_addr;
    logic [AXI_ADDR_WIDTH-1:0] dst_next_addr;
    logic [15:0]               dst_token_count;
    logic [15:0]               dst_token_remaining;
    logic [7:0]                dst_burst_len;
    logic [7:0]                dst_beat_count;
    logic [7:0]                dst_resp_count;
    logic                       dst_fifo_full;
    logic                       dst_fifo_empty;
    logic                       dst_fifo_write;
    logic                       dst_fifo_read;
    logic [TOKEN_WIDTH-1:0]     dst_fifo_din;
    logic [TOKEN_WIDTH-1:0]     dst_fifo_dout;
    logic [TOKEN_WIDTH-1:0]     dst_token_buffer;

    //==========================================================
    // Token Assembly/Disassembly
    //==========================================================
    logic [TOKEN_WIDTH-1:0]     assemble_buffer;
    logic [3:0]                assemble_count;  // 0-15 beats per token
    logic [TOKEN_WIDTH-1:0]     disassemble_buffer;
    logic [3:0]                disassemble_count;

    //==========================================================
    // FIFO for Source Path
    //==========================================================
    logic [$clog2(FIFO_DEPTH):0] src_fifo_wr_ptr;
    logic [$clog2(FIFO_DEPTH):0] src_fifo_rd_ptr;
    logic [FIFO_DEPTH-1:0][TOKEN_WIDTH-1:0] src_fifo_mem;

    assign src_fifo_full = (src_fifo_wr_ptr[$clog2(FIFO_DEPTH)] !=
                            src_fifo_rd_ptr[$clog2(FIFO_DEPTH)]) &&
                           (src_fifo_wr_ptr[$clog2(FIFO_DEPTH)-1:0] ==
                            src_fifo_rd_ptr[$clog2(FIFO_DEPTH)-1:0]);
    assign src_fifo_empty = (src_fifo_wr_ptr == src_fifo_rd_ptr);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            src_fifo_wr_ptr <= '0;
        end else if (src_fifo_write && !src_fifo_full) begin
            src_fifo_mem[src_fifo_wr_ptr[$clog2(FIFO_DEPTH)-1:0]] <= src_fifo_din;
            src_fifo_wr_ptr <= src_fifo_wr_ptr + 1'b1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            src_fifo_rd_ptr <= '0;
        end else if (src_fifo_read && !src_fifo_empty) begin
            src_fifo_rd_ptr <= src_fifo_rd_ptr + 1'b1;
        end
    end

    assign src_fifo_dout = src_fifo_mem[src_fifo_rd_ptr[$clog2(FIFO_DEPTH)-1:0]];

    //==========================================================
    // FIFO for Sink Path
    //==========================================================
    logic [$clog2(FIFO_DEPTH):0] dst_fifo_wr_ptr;
    logic [$clog2(FIFO_DEPTH):0] dst_fifo_rd_ptr;
    logic [FIFO_DEPTH-1:0][TOKEN_WIDTH-1:0] dst_fifo_mem;

    assign dst_fifo_full = (dst_fifo_wr_ptr[$clog2(FIFO_DEPTH)] !=
                            dst_fifo_rd_ptr[$clog2(FIFO_DEPTH)]) &&
                           (dst_fifo_wr_ptr[$clog2(FIFO_DEPTH)-1:0] ==
                            dst_fifo_rd_ptr[$clog2(FIFO_DEPTH)-1:0]);
    assign dst_fifo_empty = (dst_fifo_wr_ptr == dst_fifo_rd_ptr);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dst_fifo_wr_ptr <= '0;
        end else if (dst_fifo_write && !dst_fifo_full) begin
            dst_fifo_mem[dst_fifo_wr_ptr[$clog2(FIFO_DEPTH)-1:0]] <= dst_fifo_din;
            dst_fifo_wr_ptr <= dst_fifo_wr_ptr + 1'b1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dst_fifo_rd_ptr <= '0;
        end else if (dst_fifo_read && !dst_fifo_empty) begin
            dst_fifo_rd_ptr <= dst_fifo_rd_ptr + 1'b1;
        end
    end

    assign dst_fifo_dout = dst_fifo_mem[dst_fifo_rd_ptr[$clog2(FIFO_DEPTH)-1:0]];

    //==========================================================
    // 4KB Boundary Detection
    //==========================================================
    function automatic logic check_4kb_boundary(
        input logic [AXI_ADDR_WIDTH-1:0] addr,
        input int unsigned burst_bytes
    );
        int unsigned bytes_to_boundary;
        bytes_to_boundary = BOUNDARY_4KB - (addr & ADDR_MASK_4KB);
        return (burst_bytes > bytes_to_boundary);
    endfunction

    function automatic logic [7:0] calc_safe_burst_len(
        input logic [AXI_ADDR_WIDTH-1:0] addr,
        input logic [7:0] requested_len
    );
        int unsigned bytes_to_boundary;
        int unsigned beats_to_boundary;
        logic [7:0] safe_len;

        bytes_to_boundary = BOUNDARY_4KB - (addr & ADDR_MASK_4KB);
        beats_to_boundary = bytes_to_boundary / BYTES_PER_BEAT;

        if (beats_to_boundary[7:0] < requested_len)
            safe_len = beats_to_boundary[7:0];
        else
            safe_len = requested_len;

        if (safe_len > 8'd255)
            safe_len = 8'd255;
        if (safe_len < 8'd1)
            safe_len = 8'd1;

        return safe_len - 8'd1;  // AXI_LEN = beats - 1
    endfunction

    //==========================================================
    // Source Path State Machine - Sequential
    //==========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            src_state          <= SRC_IDLE;
            src_current_addr   <= '0;
            src_token_count    <= '0;
            src_token_remaining <= '0;
            src_burst_len      <= '0;
            src_beat_count     <= '0;
            src_timeout_counter <= '0;
            assemble_buffer    <= '0;
            assemble_count     <= '0;
            src_err            <= 1'b0;
        end else begin
            src_state <= src_next_state;

            case (src_state)
                SRC_IDLE: begin
                    if (start) begin
                        src_current_addr   <= src_addr;
                        src_token_count    <= token_count;
                        src_token_remaining <= token_count;
                        src_burst_len      <= burst_len;
                        src_err            <= 1'b0;
                        assemble_count     <= '0;
                    end
                end

                SRC_ADDR_REQ: begin
                    if (m_arready && m_arvalid) begin
                        src_current_addr <= src_next_addr;
                    end
                end

                SRC_DATA_RECV: begin
                    if (m_rvalid && m_rready) begin
                        // Assemble token from AXI beats
                        for (int i = 0; i < BEATS_PER_TOKEN; i++) begin
                            if (assemble_count == i[3:0]) begin
                                assemble_buffer[i*AXI_DATA_WIDTH +: AXI_DATA_WIDTH] <= m_rdata;
                            end
                        end
                        assemble_count <= assemble_count + 1'b1;
                        src_beat_count <= src_beat_count - 1'b1;

                        if (m_rlast) begin
                            assemble_count <= '0;
                            if (m_rresp != AXI_RESP_OKAY) begin
                                src_err <= 1'b1;
                            end
                            // Update address for next burst
                            src_next_addr <= src_current_addr + (32'(src_burst_len + 8'd1) << 3);  // + (beats+1)*4 bytes
                            src_token_remaining <= src_token_remaining - 16'd1;
                        end
                    end

                    // Timeout detection
                    if (src_timeout_counter < TIMEOUT_CYCLES[15:0]) begin
                        src_timeout_counter <= src_timeout_counter + 1'b1;
                    end else begin
                        src_state <= SRC_TIMEOUT;
                    end
                end

                SRC_ASSEMBLE: begin
                    if (src_fifo_write) begin
                        assemble_buffer <= '0;
                        assemble_count <= '0;
                    end
                end

                SRC_COMPLETE: begin
                    // Final token count
                end
            endcase

            // Reset timeout on activity
            if (m_rvalid && m_rready) begin
                src_timeout_counter <= '0;
            end
        end
    end

    //==========================================================
    // Source Path State Machine - Combinational
    //==========================================================
    always_comb begin
        src_next_state = src_state;

        case (src_state)
            SRC_IDLE: begin
                if (start && src_token_count > 0) begin
                    src_next_state = SRC_ADDR_REQ;
                end
            end

            SRC_ADDR_REQ: begin
                if (m_arready && m_arvalid) begin
                    src_next_state = SRC_DATA_RECV;
                end
            end

            SRC_DATA_RECV: begin
                if (m_rvalid && m_rready && m_rlast) begin
                    if (src_err) begin
                        src_next_state = SRC_ERROR;
                    end else if (src_token_remaining == 0 || abort) begin
                        src_next_state = SRC_COMPLETE;
                    end else begin
                        src_next_state = SRC_ADDR_REQ;
                    end
                end
            end

            SRC_ASSEMBLE: begin
                if (src_fifo_write || src_fifo_full) begin
                    if (src_token_remaining == 0) begin
                        src_next_state = SRC_COMPLETE;
                    end else begin
                        src_next_state = SRC_ADDR_REQ;
                    end
                end
            end

            SRC_ERROR, SRC_TIMEOUT: begin
                src_next_state = SRC_IDLE;
            end

            SRC_COMPLETE: begin
                src_next_state = SRC_IDLE;
            end

            default: begin
                src_next_state = SRC_IDLE;
            end
        endcase
    end

    //==========================================================
    // Source Path AXI Interface
    //==========================================================
    // FIX: Use parameterized ID width instead of hardcoded 4-bit value
    assign m_arid = AXI_ID_WIDTH'(0);  // Read ID = 0 = 0
    assign m_araddr = src_current_addr;
    assign m_arlen = calc_safe_burst_len(src_current_addr, src_burst_len);
    assign m_arsize = AXI_SIZE_32BIT;
    assign m_arburst = BURST_INCR;
    assign m_arlock = 1'b0;
    assign m_arcache = 4'b0011;  // Normal, non-cacheable, bufferable
    assign m_arprot = 3'b000;    // Unprivileged, secure, data
    assign m_arqos = 4'b0000;
    assign m_arvalid = (src_state == SRC_ADDR_REQ);

    // Token assembly output
    assign src_fifo_write = (src_state == SRC_DATA_RECV) && m_rvalid && m_rready && m_rlast;
    assign src_fifo_din = assemble_buffer;

    // Source streaming output
    assign src_fifo_read = src_valid && src_ready && !src_fifo_empty;
    assign src_valid = !src_fifo_empty;
    assign src_data = src_fifo_dout;
    assign src_last = (src_token_remaining == 1);

    assign m_rready = (src_state == SRC_DATA_RECV) && !src_fifo_full;

    //==========================================================
    // Sink Path State Machine - Sequential
    //==========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dst_state          <= DST_IDLE;
            dst_current_addr    <= '0;
            dst_token_count     <= '0;
            dst_token_remaining <= '0;
            dst_burst_len       <= '0;
            dst_beat_count      <= '0;
            dst_resp_count      <= '0;
            disassemble_buffer  <= '0;
            disassemble_count   <= '0;
            dst_err            <= 1'b0;
        end else begin
            dst_state <= dst_next_state;

            case (dst_state)
                DST_IDLE: begin
                    if (start) begin
                        dst_current_addr   <= dst_addr;
                        dst_token_count    <= token_count;
                        dst_token_remaining <= token_count;
                        dst_burst_len      <= burst_len;
                        dst_err            <= 1'b0;
                    end
                end

                DST_ADDR_REQ: begin
                    if (m_awready && m_awvalid) begin
                        dst_current_addr <= dst_next_addr;
                    end
                end

                DST_DATA_SEND: begin
                    if (m_wvalid && m_wready) begin
                        disassemble_count <= disassemble_count + 1'b1;
                        dst_beat_count <= dst_beat_count - 1'b1;

                        if (m_wlast) begin
                            disassemble_count <= '0;
                        end
                    end
                end

                DST_RESP_WAIT: begin
                    if (m_bvalid && m_bready) begin
                        dst_resp_count <= dst_resp_count - 1'b1;
                        if (m_bresp != AXI_RESP_OKAY) begin
                            dst_err <= 1'b1;
                        end
                    end
                end

                DST_COMPLETE: begin
                    // Final token count
                end
            endcase
        end
    end

    //==========================================================
    // Sink Path State Machine - Combinational
    //==========================================================
    always_comb begin
        dst_next_state = dst_state;

        case (dst_state)
            DST_IDLE: begin
                if (start && dst_token_count > 0) begin
                    dst_next_state = DST_WAIT_DATA;
                end
            end

            DST_WAIT_DATA: begin
                if (!dst_fifo_empty) begin
                    dst_next_state = DST_ADDR_REQ;
                end
            end

            DST_ADDR_REQ: begin
                if (m_awready && m_awvalid) begin
                    dst_next_state = DST_DATA_SEND;
                end
            end

            DST_DATA_SEND: begin
                if (m_wvalid && m_wready && m_wlast) begin
                    dst_next_state = DST_RESP_WAIT;
                end
            end

            DST_RESP_WAIT: begin
                if (m_bvalid && m_bready) begin
                    if (dst_err || abort) begin
                        dst_next_state = DST_ERROR;
                    end else if (dst_token_remaining == 0) begin
                        dst_next_state = DST_COMPLETE;
                    end else if (!dst_fifo_empty) begin
                        dst_next_state = DST_ADDR_REQ;
                    end else begin
                        dst_next_state = DST_WAIT_DATA;
                    end
                end
            end

            DST_ERROR, DST_COMPLETE: begin
                dst_next_state = DST_IDLE;
            end

            default: begin
                dst_next_state = DST_IDLE;
            end
        endcase
    end

    //==========================================================
    // Sink Path AXI Interface
    //==========================================================
    // FIX: Use parameterized ID width instead of hardcoded 4-bit value
    // Write ID should differ from read ID to allow concurrent read/write
    assign m_awid = AXI_ID_WIDTH'(1);  // Write ID = 1
    assign m_awaddr = dst_current_addr;
    assign m_awlen = calc_safe_burst_len(dst_current_addr, dst_burst_len);
    assign m_awsize = AXI_SIZE_32BIT;
    assign m_awburst = BURST_INCR;
    assign m_awlock = 1'b0;
    assign m_awcache = 4'b0011;
    assign m_awprot = 3'b000;
    assign m_awqos = 4'b0000;
    assign m_awvalid = (dst_state == DST_ADDR_REQ);

    assign m_wdata = disassemble_buffer[disassemble_count*AXI_DATA_WIDTH +: AXI_DATA_WIDTH];
    assign m_wstrb = {(AXI_DATA_WIDTH/8){1'b1}};  // All bytes valid
    assign m_wlast = (dst_beat_count == 1);
    assign m_wvalid = (dst_state == DST_DATA_SEND);
    assign m_bready = 1'b1;  // Always ready for response

    // Token disassembly input
    assign dst_fifo_write = dst_valid && dst_ready && !dst_fifo_full;
    assign dst_fifo_din = dst_data;
    assign dst_ready = !dst_fifo_full;

    assign dst_fifo_read = (dst_state == DST_DATA_SEND) && m_wvalid && m_wready && m_wlast;

    // Disassemble token to AXI beats
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            disassemble_buffer <= '0;
            disassemble_count <= '0;
        end else if (dst_fifo_read && !dst_fifo_empty) begin
            disassemble_buffer <= dst_fifo_dout;
            disassemble_count <= '0;
        end else if (m_wvalid && m_wready) begin
            disassemble_count <= disassemble_count + 1'b1;
        end
    end

    //==========================================================
    // Peripheral Request Acknowledge Logic
    //==========================================================
    // Simple implementation: Accept if FIFO not full, otherwise wait
    always_comb begin
        for (int i = 0; i < NUM_PERIPHS; i++) begin
            periph_ack_valid[i] = periph_req_valid[i];
            if (periph_req_type[i] == PERIPH_REQ_FLUSH) begin
                periph_ack_type[i] = PERIPH_ACK_FLUSH_DONE;
            end else if (!dst_fifo_full) begin
                periph_ack_type[i] = PERIPH_ACK_ACCEPT;
            end else begin
                periph_ack_type[i] = PERIPH_ACK_WAIT;
            end
        end
    end

    //==========================================================
    // Status Outputs
    //==========================================================
    assign busy = (src_state != SRC_IDLE) || (dst_state != DST_IDLE);
    assign done = (src_state == SRC_COMPLETE) && (dst_state == DST_COMPLETE);
    assign error = src_err || dst_err;
    assign boundary_err = src_boundary_cross;
    assign timeout_err = (src_state == SRC_TIMEOUT);

    // Token counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tokens_done <= '0;
        end else if (start) begin
            tokens_done <= '0;
        end else if (dst_fifo_read) begin
            tokens_done <= tokens_done + 1'b1;
        end
    end

    //==========================================================
    // Interrupt Generation
    //==========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            irq <= 1'b0;
        end else if (done || error || timeout_err) begin
            irq <= 1'b1;
        end else begin
            irq <= 1'b0;
        end
    end

    //==========================================================
    // Assertions
    //==========================================================
    `ifdef FORMAL
        // Verify burst length constraints
        assert property (@(posedge clk) disable iff (!rst_n)
            m_arvalid |-> m_arlen >= 15 && m_arlen <= 255)
            else $error("AXI burst length out of range");

        // Verify address alignment (64-byte for tokens)
        assert property (@(posedge clk) disable iff (!rst_n)
            start |-> (src_addr[5:0] == '0) && (dst_addr[5:0] == '0))
            else $error("Address not aligned to token boundary");

        // Verify 4KB boundary constraint
        assert property (@(posedge clk) disable iff (!rst_n)
            m_arvalid |-> (m_araddr[11:0] + (m_arlen + 1) * 4) <= BOUNDARY_4KB ||
                           m_araddr[11:0] > (m_araddr[11:0] + (m_arlen + 1) * 4))
            else $error("AXI burst crosses 4KB boundary");

        // Verify FIFO overflow protection
        assert property (@(posedge clk) disable iff (!rst_n)
            src_fifo_write |-> !src_fifo_full)
            else $error("Source FIFO overflow");

        assert property (@(posedge clk) disable iff (!rst_n)
            dst_fifo_write |-> !dst_fifo_full)
            else $error("Sink FIFO overflow");
    `endif

endmodule : dma_e_rodnet_engine

