//-----------------------------------------------------------------------------
// Module: dma_e_rodnet_pkg.sv
// Description: Package for e_rodnet DMA interface.
//              Target: Yamuna (Shakti family) — AXI4 Full, 32-bit data, 32-bit addr
//              Token width: MAX_C(64) × DATA_WIDTH(8) = 512 bits = 64 bytes = 16 AXI4 beats
//
// Features:
//   - AXI4 Full master interface for Shakti SoC
//   - Token-based streaming (512-bit tokens)
//   - PL330-style peripheral request/acknowledge flow control
//   - 4KB boundary crossing detection and handling
//   - Multi-channel support with per-channel FIFOs
//   - Error detection and recovery
//   - Interrupt coalescing
//-----------------------------------------------------------------------------
package dma_e_rodnet_pkg;

    //==========================================================
    // AXI4 Parameters (Yamuna SoC)
    //==========================================================
    localparam int unsigned AXI_DATA_WIDTH = 32;
    localparam int unsigned AXI_ADDR_WIDTH = 32;
    localparam int unsigned AXI_ID_WIDTH    = 4;
    localparam int unsigned AXI_LEN_WIDTH  = 8;
    localparam int unsigned AXI_USER_WIDTH  = 4;   // For cache hints
    localparam int unsigned AXI_STRB_WIDTH  = AXI_DATA_WIDTH / 8;  // 4 bytes

    //==========================================================
    // Token Parameters
    //==========================================================
    // One token = 64 channels × 8 bits = 512 bits = 64 bytes
    localparam int unsigned TOKEN_WIDTH    = 512;
    localparam int unsigned TOKEN_SIZE      = 64;  // bytes
    localparam int unsigned BEATS_PER_TOKEN = 16;  // 16 × 32-bit = 512-bit
    localparam int unsigned BYTES_PER_BEAT = AXI_DATA_WIDTH / 8;  // 4 bytes

    //==========================================================
    // Burst Constraints
    //==========================================================
    // Minimum: 16 beats (one token)
    // Maximum: 256 beats (16 tokens)
    localparam int unsigned MIN_BURST_LEN   = 16;
    localparam int unsigned MAX_BURST_LEN   = 256;

    // 4KB boundary constraint (AXI4 requirement)
    localparam int unsigned BOUNDARY_4KB    = 4096;
    localparam int unsigned ADDR_MASK_4KB   = 32'h00000FFF;  // Lower 12 bits (32-bit)

    // Maximum beats without crossing 4KB boundary
    // 4KB / 4 bytes per beat = 1024 beats max, but AXI limits to 256
    localparam int unsigned MAX_BEATS_4KB   = 256;

    //==========================================================
    // DMA CSR Address Map (extends e_rodnet CSR space at 0x300)
    //==========================================================
    localparam logic [11:0] CSR_DMA_SRC_ADDR     = 12'h300;
    localparam logic [11:0] CSR_DMA_DST_ADDR     = 12'h304;
    localparam logic [11:0] CSR_DMA_TOKEN_CNT    = 12'h308;
    localparam logic [11:0] CSR_DMA_CTRL         = 12'h30C;
    localparam logic [11:0] CSR_DMA_STATUS       = 12'h310;
    localparam logic [11:0] CSR_DMA_IRQ_EN       = 12'h314;
    localparam logic [11:0] CSR_DMA_BURST_LEN    = 12'h318;
    localparam logic [11:0] CSR_DMA_TOKENS_DONE   = 12'h31C;

    //==========================================================
    // CSR_DMA_CTRL Bit Definitions
    //==========================================================
    localparam int unsigned CTRL_START       = 0;
    localparam int unsigned CTRL_ABORT       = 1;
    localparam int unsigned CTRL_LOOP_MODE   = 2;
    localparam int unsigned CTRL_SRC_INC     = 3;

    //==========================================================
    // CSR_DMA_STATUS Bit Definitions
    //==========================================================
    localparam int unsigned STATUS_DONE      = 0;
    localparam int unsigned STATUS_ERROR     = 1;
    localparam int unsigned STATUS_BUSY      = 2;
    localparam int unsigned STATUS_SRC_ERR   = 3;
    localparam int unsigned STATUS_DST_ERR   = 4;

    //==========================================================
    // CSR_DMA_IRQ_EN Bit Definitions
    //==========================================================
    localparam int unsigned IRQ_DONE_EN      = 0;
    localparam int unsigned IRQ_ERROR_EN     = 1;

    //==========================================================
    // DMA Engine States
    //==========================================================
    typedef enum logic [3:0] {
        DMA_IDLE            = 4'b0000,
        DMA_SRC_ARBITRATE   = 4'b0001,
        DMA_SRC_READ_ADDR   = 4'b0010,
        DMA_SRC_READ_DATA   = 4'b0011,
        DMA_SRC_ASSEMBLE    = 4'b0100,
        DMA_DST_WAIT        = 4'b0101,
        DMA_DST_WRITE_ADDR  = 4'b0110,
        DMA_DST_WRITE_DATA  = 4'b0111,
        DMA_DST_WRITE_RESP  = 4'b1000,
        DMA_COMPLETE        = 4'b1001,
        DMA_ERROR           = 4'b1111
    } dma_state_e;

    //==========================================================
    // Token Assembly State
    //==========================================================
    typedef enum logic [1:0] {
        ASSEMBLE_IDLE,
        ASSEMBLE_FILL,
        ASSEMBLE_READY
    } assemble_state_e;

    //==========================================================
    // AXI4 Burst Type
    //==========================================================
    typedef enum logic [1:0] {
        BURST_FIXED = 2'b00,
        BURST_INCR  = 2'b01,
        BURST_WRAP  = 2'b10
    } axi_burst_type_e;

    //==========================================================
    // AXI4 Response
    //==========================================================
    typedef enum logic [1:0] {
        AXI_RESP_OKAY   = 2'b00,
        AXI_RESP_EXOKAY = 2'b01,
        AXI_RESP_SLVERR  = 2'b10,
        AXI_RESP_DECERR  = 2'b11
    } axi_resp_e;

    //==========================================================
    // AXI4 Size Encoding (log2 of bytes per beat)
    //==========================================================
    // 32-bit = 4 bytes = size encoding 010
    localparam logic [2:0] AXI_SIZE_32BIT = 3'b010;

    //==========================================================
    // Peripheral Request Types (PL330-style)
    //==========================================================
    typedef enum logic [1:0] {
        PERIPH_REQ_SINGLE = 2'b00,   // Single transfer request
        PERIPH_REQ_BURST  = 2'b01,   // Burst transfer request
        PERIPH_REQ_FLUSH  = 2'b10    // Flush internal buffers
    } periph_req_type_e;

    typedef enum logic [1:0] {
        PERIPH_ACK_ACCEPT    = 2'b00,  // Request accepted
        PERIPH_ACK_WAIT      = 2'b01,  // Wait for space
        PERIPH_ACK_FLUSH_DONE = 2'b10  // Flush complete
    } periph_ack_type_e;

    //==========================================================
    // Transfer Direction
    //==========================================================
    typedef enum logic [1:0] {
        XFER_READ  = 2'b00,   // DDR → IP (memory read)
        XFER_WRITE = 2'b01,   // IP → DDR (memory write)
        XFER_BIDIR = 2'b10    // Bidirectional
    } transfer_dir_e;

    //==========================================================
    // FIFO Depths
    //==========================================================
    localparam int unsigned SRC_FIFO_DEPTH  = 4;   // Input tokens
    localparam int unsigned DST_FIFO_DEPTH  = 4;   // Output tokens
    localparam int unsigned CMD_FIFO_DEPTH  = 8;   // Command queue

    //==========================================================
    // Timeout and Retry Limits
    //==========================================================
    localparam int unsigned DEFAULT_TIMEOUT   = 4096;  // Cycles
    localparam int unsigned MAX_RETRIES       = 3;
    localparam int unsigned WATCHDOG_TIMEOUT  = 16384; // Cycles

    //==========================================================
    // Peripheral Channel Configuration
    //==========================================================
    localparam int unsigned NUM_PERIPH_CHANNELS = 8;

    //==========================================================
    // DMA Transfer Descriptor
    //==========================================================
    typedef struct packed {
        logic [AXI_ADDR_WIDTH-1:0]   src_addr;      // Source address
        logic [AXI_ADDR_WIDTH-1:0]   dst_addr;      // Destination address
        logic [15:0]                 token_count;   // Tokens to transfer
        logic [7:0]                  burst_len;     // Beats per burst
        logic [2:0]                  xfer_dir;     // Read/Write/Bidir
        logic                        last_desc;    // Last descriptor flag
        logic                        irq_en;       // Interrupt enable
        logic [2:0]                  periph_id;    // Peripheral channel ID
    } dma_xfer_desc_t;

    //==========================================================
    // DMA Status Register
    //==========================================================
    typedef struct packed {
        logic                        busy;          // Transfer in progress
        logic                        done;          // Transfer complete
        logic                        error;         // Error occurred
        logic                        src_err;       // Source (read) error
        logic                        dst_err;       // Destination (write) error
        logic                        timeout;       // Timeout detected
        logic                        boundary_err;  // 4KB boundary violation
        logic [15:0]                 tokens_done;   // Tokens transferred
        logic [3:0]                  periph_id;     // Active peripheral ID
    } dma_status_t;

    //==========================================================
    // AXI4 Burst Calculation Functions
    //==========================================================
    // Calculate maximum burst length without crossing 4KB boundary
    function automatic int unsigned calc_burst_len_4kb(
        input logic [AXI_ADDR_WIDTH-1:0] addr,
        input int unsigned max_len
    );
        int unsigned bytes_to_boundary;
        int unsigned beats_to_boundary;
        int unsigned result;

        bytes_to_boundary = BOUNDARY_4KB - (addr & ADDR_MASK_4KB);
        beats_to_boundary = bytes_to_boundary / BYTES_PER_BEAT;

        if (beats_to_boundary < max_len)
            result = beats_to_boundary;
        else
            result = max_len;

        if (result > 256)
            result = 256;  // AXI max burst

        return result - 1;  // Return AXI_LEN (beats - 1)
    endfunction

    // Check if address crosses 4KB boundary for given burst length
    function automatic logic crosses_4kb_boundary(
        input logic [AXI_ADDR_WIDTH-1:0] start_addr,
        input int unsigned burst_len_bytes
    );
        int unsigned bytes_to_boundary;
        bytes_to_boundary = BOUNDARY_4KB - (start_addr & ADDR_MASK_4KB);
        return (burst_len_bytes > bytes_to_boundary);
    endfunction

endpackage : dma_e_rodnet_pkg


