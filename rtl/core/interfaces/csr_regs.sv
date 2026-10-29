// =============================================================================
// csr_regs.sv
//
// Control and Status Register definitions for e_rodnet ConvFormer accelerator.
// Memory-mapped with shadow registers for glitch-free updates.
// =============================================================================

package csr_regs;

    // ── Register Address Map (32-bit aligned) ────────────────────────────────
    localparam int CSR_BASE_ADDR = 32'h4000_0000;

    typedef enum logic [11:0] {
        // Control Registers
        CSR_CTRL              = 12'h000,  // RW - Global control
        CSR_STATUS            = 12'h004,  // RO - Status flags
        CSR_VERSION           = 12'h008,  // RO - Hardware version

        // ConvFormer Configuration
        CSR_CFG_C             = 12'h010,  // RW - Active channels (≤ MAX_C)
        CSR_CFG_KERNEL_SIZE   = 12'h014,  // RW - Kernel size (≤ MAX_KERNEL_SIZE)
        CSR_CFG_ACT_SEL       = 12'h018,  // RW - Activation select
        CSR_CFG_SCALE_IN      = 12'h01C,  // RW - StarReLU scale γ
        CSR_CFG_BIAS_IN       = 12'h020,  // RW - StarReLU bias β

        // Activation Control Shadow Registers
        CSR_ACT_ENABLE        = 12'h030,  // RW - Activation enable shadow
        CSR_ACT_LEAK_SHIFT    = 12'h034,  // RW - Leaky ReLU shift shadow
        CSR_ACT_SATURATION    = 12'h038,  // RW - ReLU6 saturation shadow

        // Pooling/Quantization Configuration
        CSR_POOL_EN           = 12'h040,  // RW - Pooling enable
        CSR_POOL_MODE         = 12'h044,  // RW - 0=Max, 1=Average
        CSR_QUANT_MUL         = 12'h048,  // RW - Quantization multiplier
        CSR_QUANT_SHIFT       = 12'h04C,  // RW - Quantization shift
        CSR_QUANT_EN          = 12'h050,  // RW - Quantization enable

        // Weight Loading Interface
        CSR_WEIGHT_ADDR       = 12'h100,  // RW - Weight memory address
        CSR_WEIGHT_DATA       = 12'h104,  // WO - Weight data write
        CSR_WEIGHT_CTRL       = 12'h108,  // RW - Weight load control

        // Performance Counters
        CSR_CYCLE_COUNT       = 12'h200,  // RO - Cycle counter
        CSR_INSTR_COUNT       = 12'h204,  // RO - Token-processed counter
        CSR_STALL_COUNT       = 12'h208,  // RO - Stall counter
        CSR_ERROR_COUNT       = 12'h20C,  // RO - Error counter

        // ── CPU Direct Token I/O (cpu_mode = 1) ─────────────────────────────
        // Token input staging: write 16 words (0x400–0x43C) then inject
        // Token output latch:  poll TOKEN_OUT_STATUS, read 16 words, then ack
        // Word n address = CSR_TOKEN_IN_BASE + n*4  (n = 0..15)
        CSR_TOKEN_IN_BASE     = 12'h400,  // WO  Staging word 0  (words 0..15 @ +0..+0x3C)
        CSR_TOKEN_IN_CTRL     = 12'h440,  // RW  [0]=inject(SC), [1]=last, [2]=ready(RO)
        CSR_TOKEN_OUT_BASE    = 12'h480,  // RO  Output latch word 0 (words 0..15 @ +0..+0x3C)
        CSR_TOKEN_OUT_STATUS  = 12'h4C0,  // RO  [0]=valid
        CSR_TOKEN_OUT_ACK     = 12'h4C4,  // WO  [0]=ack (SC, advance output latch)

        // DMA Control Registers (programmed by CPU, read by DMA)
        CSR_DMA_SRC_ADDR      = 12'h300,  // RW - Input feature map DDR base
        CSR_DMA_DST_ADDR      = 12'h304,  // RW - Output feature map DDR base
        CSR_DMA_TOKEN_CNT     = 12'h308,  // RW - Tokens per frame (max 65535)
        CSR_DMA_CTRL          = 12'h30C,  // RW - DMA control (start/abort/loop)
        CSR_DMA_STATUS        = 12'h310,  // RO/W1C - DMA status
        CSR_DMA_IRQ_EN        = 12'h314,  // RW - Interrupt enable mask
        CSR_DMA_BURST_LEN     = 12'h318,  // RW - AXI4 beats per burst (default 16)
        CSR_DMA_TOKENS_DONE   = 12'h31C   // RO - Tokens written to DDR so far
    } csr_addr_e;

    // ── CSR_CTRL Register Bits ───────────────────────────────────────────────
    typedef struct packed {
        logic [31:16] reserved;      // [31:16] Reserved
        logic         cfg_update;    // [15]    Update shadow registers
        logic         soft_reset;    // [14]    Soft reset (self-clearing)
        logic         weight_load;   // [13]    Start weight load (self-clearing)
        logic         start;         // [12]    Start processing (self-clearing)
        logic         bypass;        // [11]    Bypass activation (override)
        logic [2:0]   precision;     // [10:8]  000=INT8, 001=INT16, 010=FP16, 011=FP32
        logic         power_gate;    // [7]     Power gate unused channels
        logic         clock_gate;    // [6]     Dynamic clock gating
        logic [5:0]   reserved_low; // [5:0]   Reserved
    } csr_ctrl_t;

    // ── CSR_STATUS Register Bits ─────────────────────────────────────────────
    typedef struct packed {
        logic [31:24] reserved;      // [31:24] Reserved
        logic [7:0]   fifo_level;    // [23:16] Input FIFO occupancy
        logic         weight_loaded;  // [15]    Weight memory initialized
        logic         cfg_locked;     // [14]    Configuration locked (updating)
        logic         busy;           // [13]    Accelerator busy
        logic         error;          // [12]    Error detected
        logic [3:0]   error_code;    // [11:8]  Error type
        logic         ready;          // [7]     Accelerator ready for commands
        logic [6:0]   reserved_low;  // [6:0]   Reserved
    } csr_status_t;

    // ── CSR_VERSION Register Bits ────────────────────────────────────────────
    localparam logic [31:0] HW_VERSION = 32'h0001_0100;  // v1.1.0

    // ── CSR_WEIGHT_CTRL Register Bits ────────────────────────────────────────
    // bank[1:0]: 00=reserved, 01=expand 1x1, 10=project 1x1, 11=reserved
    //            DW conv bank uses channel+kernel_tap fields (bank=00 implicit
    //            when dw_wr_en path is used separately via CSR_WEIGHT_CTRL legacy)
    // For pointwise banks (01/10): CSR_WEIGHT_ADDR provides the flat address.
    // For DW bank: channel[23:16] and kernel_tap[15:8] provide the address.
    typedef struct packed {
        logic [31:24] reserved;      // [31:24] Reserved
        logic [7:0]   channel;       // [23:16] DW: channel index (0..MAX_C-1)
        logic [7:0]   kernel_tap;    // [15:8]  DW: kernel tap  (0..MAX_KERNEL_SIZE-1)
        logic         wr_en;         // [7]     Write enable (self-clearing)
        logic [1:0]   bank;          // [6:5]   00=DW, 01=expand, 10=project
        logic [4:0]   reserved_low;  // [4:0]   Reserved
    } csr_weight_ctrl_t;

    // ── CSR_DMA_CTRL Register Bits ───────────────────────────────────────────
    typedef struct packed {
        logic [31:4] reserved;
        logic        src_inc;     // [3] Increment src address between frames (tiling)
        logic        loop_mode;   // [2] 1 = repeat until abort; 0 = one-shot
        logic        abort;       // [1] Write 1 to abort; self-clearing
        logic        start;       // [0] Write 1 to start; self-clearing
    } csr_dma_ctrl_t;

    // ── CSR_DMA_STATUS Register Bits ─────────────────────────────────────────
    typedef struct packed {
        logic [31:5] reserved;
        logic        dst_err;     // [4] AXI4 write SLVERR from DMA
        logic        src_err;     // [3] AXI4 read SLVERR from DMA
        logic        busy;        // [2] DMA transfer in progress (RO)
        logic        error;       // [1] Any error; W1C
        logic        done;        // [0] Frame complete; W1C
    } csr_dma_status_t;

    // ── CSR_DMA_IRQ_EN Register Bits ─────────────────────────────────────────
    typedef struct packed {
        logic [31:2] reserved;
        logic        error_en;    // [1] Interrupt on any error
        logic        done_en;     // [0] Interrupt on frame done
    } csr_dma_irq_en_t;

    // ── Error Codes ──────────────────────────────────────────────────────────
    typedef enum logic [3:0] {
        ERR_NONE          = 4'h0,
        ERR_CFG_BOUNDS    = 4'h1,  // cfg_c > MAX_C or cfg_kernel_size > MAX_KERNEL_SIZE
        ERR_FIFO_OVERFLOW = 4'h2,  // Input FIFO overflow
        ERR_FIFO_UNDERFLOW= 4'h3,  // Output FIFO underflow
        ERR_WEIGHT_ADDR   = 4'h4,  // Weight address out of bounds
        ERR_PROTOCOL      = 4'h5,  // Handshake protocol violation
        ERR_OVERFLOW      = 4'h6,  // Arithmetic overflow
        ERR_UNDERFLOW     = 4'h7,  // Arithmetic underflow
        ERR_TIMEOUT       = 4'h8   // Operation timeout
    } error_code_e;

    // =========================================================================
    // SSF BatchNorm Register Map (0x500 - 0x5FF)
    // =========================================================================
    // Each BatchNorm layer has 4 parameter arrays: gamma, beta, mean, inv_std
    // Array size = CHANNELS (default 8) × 32-bit = 32 bytes per parameter type
    // Total: 3 BN layers × 4 params × 32 bytes = 384 bytes (0x180)

    localparam int SSF_CHANNELS = 8;  // Default SSF channel count

    localparam int CSR_SSF_BN1_GAMMA_BASE    = 12'h500;  // BN1 gamma [CHANNELS-1:0]
    localparam int CSR_SSF_BN1_BETA_BASE     = 12'h520;  // BN1 beta [CHANNELS-1:0]
    localparam int CSR_SSF_BN1_MEAN_BASE     = 12'h540;  // BN1 running mean [CHANNELS-1:0]
    localparam int CSR_SSF_BN1_INV_STD_BASE  = 12'h560;  // BN1 inverse std [CHANNELS-1:0]

    localparam int CSR_SSF_BN2_GAMMA_BASE    = 12'h580;  // BN2 gamma [CHANNELS-1:0]
    localparam int CSR_SSF_BN2_BETA_BASE     = 12'h5A0;  // BN2 beta [CHANNELS-1:0]
    localparam int CSR_SSF_BN2_MEAN_BASE     = 12'h5C0;  // BN2 running mean [CHANNELS-1:0]
    localparam int CSR_SSF_BN2_INV_STD_BASE  = 12'h5E0;  // BN2 inverse std [CHANNELS-1:0]

    localparam int CSR_SSF_BN3_GAMMA_BASE    = 12'h600;  // BN3 gamma [CHANNELS-1:0]
    localparam int CSR_SSF_BN3_BETA_BASE     = 12'h620;  // BN3 beta [CHANNELS-1:0]
    localparam int CSR_SSF_BN3_MEAN_BASE     = 12'h640;  // BN3 running mean [CHANNELS-1:0]
    localparam int CSR_SSF_BN3_INV_STD_BASE  = 12'h660;  // BN3 inverse std [CHANNELS-1:0]

    // SSF Linear Weight Register Map (0x680 - 0x6FF)
    localparam int CSR_SSF_LINEAR_W_BASE     = 12'h680;  // Linear layer weights
    localparam int CSR_SSF_LINEAR_BIAS_BASE  = 12'h6C0;  // Linear layer bias

    // ── SSF BatchNorm Register Bits ─────────────────────────────────────────
    typedef struct packed {
        logic [31:0] value;  // 32-bit fixed-point (Q16.16)
    } csr_ssf_bn_param_t;

    // =========================================================================
    // Temporal Fusion Register Map (0x700 - 0x7FF)
    // =========================================================================
    // Temporal attention weights: HISTORY_DEPTH × CHANNELS × 8-bit
    // HISTORY_DEPTH = 8, CHANNELS = 64 → 512 bytes

    localparam int CSR_TEMPORAL_CTRL          = 12'h700;  // RW: Temporal control
    localparam int CSR_TEMPORAL_WEIGHTS_BASE   = 12'h704;  // RW: Attention weights array
    localparam int CSR_TEMPORAL_STATUS         = 12'h780;  // RO: Temporal status
    localparam int CSR_TEMPORAL_FRAME_COUNT    = 12'h784;  // RO: Frames in buffer

    // ── CSR_TEMPORAL_CTRL Register Bits ─────────────────────────────────────
    typedef struct packed {
        logic [31:6] reserved;
        logic        cfar_en;       // [5] CFAR detection enable
        logic [2:0]  history_depth; // [4:2] History depth (1-8 frames)
        logic [2:0]  fusion_mode;   // [1:0] + [2] reserved: Fusion mode
        logic        temporal_en;   // [0] Temporal processing enable
    } csr_temporal_ctrl_t;

    // =========================================================================
    // RA Processing Register Map (0x800 - 0x8FF)
    // =========================================================================

    localparam int CSR_RA_CONFIG          = 12'h800;  // RW: RA processor configuration
    localparam int CSR_RA_FFT_SIZE        = 12'h804;  // RW: Range FFT size (64-1024)
    localparam int CSR_RA_FFT_RADIX       = 12'h808;  // RW: FFT radix (2, 4, 8)
    localparam int CSR_RA_WINDOW_TYPE     = 12'h80C;  // RW: Windowing type
    localparam int CSR_RA_BEAM_MODE       = 12'h810;  // RW: Beamforming mode
    localparam int CSR_RA_CFAR_THRESH     = 12'h814;  // RW: CFAR threshold (dB)
    localparam int CSR_RA_CFAR_GUARD      = 12'h818;  // RW: CFAR guard cells
    localparam int CSR_RA_CFAR_TRAIN      = 12'h81C;  // RW: CFAR training cells
    localparam int CSR_RA_STATUS          = 12'h820;  // RO: RA processor status
    localparam int CSR_RA_FRAME_COUNT     = 12'h824;  // RO: Frames processed
    localparam int CSR_RA_DETECTION_COUNT = 12'h828;  // RO: CFAR detections

    // ── CSR_RA_CONFIG Register Bits ─────────────────────────────────────────
    typedef struct packed {
        logic [31:10] reserved;
        logic         log_output;    // [9] Output log10(magnitude)
        logic         magnitude_en; // [8] Magnitude computation enable
        logic         angle_fft_en; // [7] Angle FFT enable
        logic         beam_en;      // [6] Beamforming enable
        logic         window_en;    // [5] Windowing enable
        logic         cfar_en;      // [4] CFAR detection enable
        logic [3:0]   cfar_mode;    // [3:0] CFAR mode: 0=CA, 1=GO, 2=SO, 3=OS
    } csr_ra_config_t;

    // ── CSR_RA_STATUS Register Bits ─────────────────────────────────────────
    typedef struct packed {
        logic [31:8] reserved;
        logic [7:0]  fifo_level;    // [23:16] Input FIFO level
        logic        cfar_active;   // [15] CFAR detection in progress
        logic        fft_busy;      // [14] FFT accelerator busy
        logic        cfg_locked;    // [13] Configuration locked
        logic        busy;          // [12] RA processor busy
        logic        error;         // [11] Error detected
        logic [3:0]  error_code;    // [10:7] Error type
        logic        done;          // [6] Frame complete
        logic [5:0]  reserved_low; // [5:0] Reserved
    } csr_ra_status_t;

    // =========================================================================
    // CFAR Detection Register Map (0x900 - 0x9FF)
    // =========================================================================

    localparam int CSR_CFAR_CTRL           = 12'h900;  // RW: CFAR control
    localparam int CSR_CFAR_THRESHOLD      = 12'h904;  // RW: Detection threshold (fixed-point)
    localparam int CSR_CFAR_SCALE          = 12'h908;  // RW: Scale factor for threshold
    localparam int CSR_CFAR_GUARD_LEFT     = 12'h90C;  // RW: Guard cells (left)
    localparam int CSR_CFAR_GUARD_RIGHT    = 12'h910;  // RW: Guard cells (right)
    localparam int CSR_CFAR_TRAIN_LEFT     = 12'h914;  // RW: Training cells (left)
    localparam int CSR_CFAR_TRAIN_RIGHT    = 12'h918;  // RW: Training cells (right)
    localparam int CSR_CFAR_STATUS         = 12'h91C;  // RO: CFAR status

    // ── CSR_CFAR_CTRL Register Bits ─────────────────────────────────────────
    typedef struct packed {
        logic [31:4] reserved;
        logic [1:0]  train_mode;    // [3:2] 0=both, 1=left, 2=right, 3=min
        logic [1:0]  method;        // [1:0] 0=CA-CFAR, 1=GO-CFAR, 2=SO-CFAR, 3=OS-CFAR
    } csr_cfar_ctrl_t;

    // =========================================================================
    // Weight Prefetch Register Map (0xA00 - 0xA1F)
    // =========================================================================

    localparam int CSR_WEIGHT_PREFETCH_CTRL   = 12'hA00;  // RW: Prefetch control
    localparam int CSR_WEIGHT_PREFETCH_STATUS = 12'hA04;  // RO: Prefetch status
    localparam int CSR_WEIGHT_PREFETCH_ADDR   = 12'hA08;  // RW: Prefetch start address
    localparam int CSR_WEIGHT_PREFETCH_LEN    = 12'hA0C;  // RW: Prefetch length

    // ── CSR_WEIGHT_PREFETCH_CTRL Register Bits ─────────────────────────────
    typedef struct packed {
        logic [31:2] reserved;
        logic        prefetch_en;   // [1] Enable prefetch
        logic        bank_select;   // [0] 0=Bank A, 1=Bank B
    } csr_weight_prefetch_ctrl_t;

    // ── Precision Modes ──────────────────────────────────────────────────────
    typedef enum logic [2:0] {
        PRECISION_INT8  = 3'b000,
        PRECISION_INT16 = 3'b001,
        PRECISION_FP16  = 3'b010,
        PRECISION_FP32  = 3'b011,
        PRECISION_BFP8  = 3'b100,  // Block floating-point 8-bit
        PRECISION_BFP16 = 3'b101   // Block floating-point 16-bit
    } precision_e;

endpackage