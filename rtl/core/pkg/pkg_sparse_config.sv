// =============================================================================
//  pkg_sparse_config.sv
//
//  Configuration package for the sparsity-aware zero-skipping systolic array.
//  Defines NZ packet formats, CSR register map, and localparams.
// =============================================================================

package pkg_sparse_config;

    import pkg_config::*;

    // =========================================================================
    //  NZ Packet Format
    //
    //  Each NZ (non-zero) packet carries one element of a sparse stream.
    //  Fields are packed into a single bus for efficient interconnect.
    //  Dense mode: all elements are emitted with sequential indices,
    //  producing bit-identical results to the original systolic array.
    // =========================================================================

    // Index width: must address all elements in one dot-product (K dimension).
    // For MAX_C=64 channels with EXPANSION_FACTOR=4, K = MAX_C = 64.
    localparam int NZ_IDX_WIDTH = $clog2(MAX_C * EXPANSION_FACTOR);

    // Total NZ packet width
    localparam int NZ_PKT_WIDTH = 1 + 1 + NZ_IDX_WIDTH + DATA_WIDTH;
    //   [NZ_PKT_WIDTH-1]           = last
    //   [NZ_PKT_WIDTH-2]           = valid
    //   [NZ_IDX_WIDTH+DATA_WIDTH-1 : DATA_WIDTH] = idx
    //   [DATA_WIDTH-1            : 0]            = val

    // Field offsets within NZ packet
    localparam int NZ_VAL_HI   = DATA_WIDTH - 1;
    localparam int NZ_VAL_LO   = 0;
    localparam int NZ_IDX_HI   = NZ_IDX_WIDTH + DATA_WIDTH - 1;
    localparam int NZ_IDX_LO   = DATA_WIDTH;
    localparam int NZ_VALID_BIT = NZ_IDX_HI + 1;
    localparam int NZ_LAST_BIT  = NZ_VALID_BIT + 1;

    // =========================================================================
    //  Encoding Modes
    // =========================================================================
    typedef enum logic [1:0] {
        SPARSE_MODE_CSR  = 2'b00,   // Compressed Sparse Row (index + value)
        SPARSE_MODE_RLE  = 2'b01,   // Run-Length Encoding (run + value)
        SPARSE_MODE_DENSE = 2'b10   // Dense passthrough (no compression)
    } sparse_mode_e;

    // =========================================================================
    //  Intersection Unit FSM States
    // =========================================================================
    typedef enum logic [1:0] {
        ISect_IDLE    = 2'b00,   // Waiting for both NZ streams
        ISect_COMPARE = 2'b01,   // Merge-join: advance smaller index
        ISect_DRAIN   = 2'b10    // One stream exhausted, drain other
                                 // (both streams exhausted → ISect_IDLE directly)
    } isect_state_e;

    // =========================================================================
    //  Elastic FIFO Configuration
    // =========================================================================
    localparam int ELASTIC_FIFO_DEPTH = pkg_config::ELASTIC_FIFO_DEPTH;
    localparam int ELASTIC_FIFO_PTR_W  = $clog2(ELASTIC_FIFO_DEPTH) + 1;  // +1 for full flag

    // =========================================================================
    //  Sparse Array Parameters
    // =========================================================================
    // Maximum run length for RLE encoding (max consecutive zeros)
    localparam int MAX_RLE_RUN = 2 * MAX_C;

    // Sparsity threshold: if NZ count > threshold, fall back to dense mode.
    // Default: 75% of K elements must be NZ before switching to dense.
    localparam int SPARSE_THRESHOLD_DEFAULT = (3 * MAX_C * EXPANSION_FACTOR) / 4;

    // =========================================================================
    //  Sparse Systolic Array Synthesis Flag (re-export from pkg_config)
    // =========================================================================
    localparam bit EN_SPARSE_SYSTOLIC = pkg_config::EN_SPARSE_SYSTOLIC;

    // =========================================================================
    //  CSR Register Map (0xB00 - 0xB3F)
    //  Shadow/live register pattern consistent with csr_regs.sv.
    //  Write to _sh registers, commit via cfg_update_pulse.
    //  Base address from pkg_config::SPARSE_CSR_BASE.
    // =========================================================================
    localparam int CSR_SPARSE_BASE          = pkg_config::SPARSE_CSR_BASE;
    localparam int CSR_SPARSE_CTRL           = CSR_SPARSE_BASE + 12'h000;  // RW: Sparse control
    localparam int CSR_SPARSE_WEIGHT_MODE    = CSR_SPARSE_BASE + 12'h004;  // RW: Weight encoding mode
    localparam int CSR_SPARSE_ACTIV_MODE     = CSR_SPARSE_BASE + 12'h008;  // RW: Activation encoding mode
    localparam int CSR_SPARSE_THRESHOLD      = CSR_SPARSE_BASE + 12'h00C;  // RW: Sparsity threshold
    localparam int CSR_SPARSE_STATUS          = CSR_SPARSE_BASE + 12'h010;  // RO: Sparse status
    localparam int CSR_SPARSE_NZ_WEIGHT_CNT   = CSR_SPARSE_BASE + 12'h014;  // RO: Weight NZ count
    localparam int CSR_SPARSE_NZ_ACTIV_CNT   = CSR_SPARSE_BASE + 12'h018;  // RO: Activation NZ count
    localparam int CSR_SPARSE_MAC_SKIP_CNT    = CSR_SPARSE_BASE + 12'h01C;  // RO: MAC operations skipped
    localparam int CSR_SPARSE_MAC_ACTIVE_CNT  = CSR_SPARSE_BASE + 12'h030;  // RO: MAC operations active (not skipped)
    localparam int CSR_SPARSE_CYCLE_CNT       = CSR_SPARSE_BASE + 12'h020;  // RO: Cycle counter
    localparam int CSR_SPARSE_FIFO_STALL_CNT  = CSR_SPARSE_BASE + 12'h024;  // RO: Elastic FIFO stall cycles
    localparam int CSR_SPARSE_WEIGHT_BASE     = CSR_SPARSE_BASE + 12'h028;  // RW: Weight NZ data base address
    localparam int CSR_SPARSE_ACTIV_BASE      = CSR_SPARSE_BASE + 12'h02C;  // RW: Activation NZ data base address

    // ── CSR_SPARSE_CTRL Register Bits ──────────────────────────────────────────
    typedef struct packed {
        logic [31:6] reserved;
        logic         sparse_en;      // [5] Enable sparsity-aware mode
        logic [1:0]   weight_mode;    // [4:3] Weight encoding: 00=CSR, 01=RLE, 10=Dense
        logic [1:0]  activ_mode;     // [2:1] Activation encoding: 00=CSR, 01=RLE, 10=Dense
        logic         cfg_update;     // [0] Shadow register commit (self-clearing)
    } csr_sparse_ctrl_t;

    // ── CSR_SPARSE_STATUS Register Bits ─────────────────────────────────────────
    typedef struct packed {
        logic [31:16] reserved;
        logic [7:0]   elastic_fifo_level; // [23:16] Elastic FIFO fill level
        logic         isect_busy;          // [15]    Intersection unit active
        logic         encoder_busy;       // [14]    NZ encoder active
        logic         sparse_active;      // [13]    Currently in sparse mode
        logic         dense_fallback;     // [12]    Fell back to dense mode
        logic         cfg_locked;          // [11]    Configuration locked
        logic         error;               // [10]    Error detected
        logic [3:0]  error_code;          // [9:6]   Error type
        logic         done;                // [5]     Computation complete
        logic [4:0]  reserved_low;        // [4:0]   Reserved
    } csr_sparse_status_t;

endpackage