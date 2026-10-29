// ==========================================
// AUTO-GENERATED FILE. DO NOT EDIT MANUALLY.
// Generated from config.yaml
// ==========================================

package pkg_config;

    // ── Matrix Dimensions ──
    localparam int M = 2;
    localparam int N = 2;
    localparam bit INT8_PACKED = 1'b0;
    localparam bit EN_DUAL_SYSTOLIC = 1'b0;

    // ── Bit Widths ──
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH = 24;
    localparam int FRAC_WIDTH = 4;
    localparam int ADDR_WIDTH = 4;

    // ── Conditional Synthesis ──
    localparam bit EN_SIGMOID = 1'b0;
    localparam bit EN_GELU = 1'b0;
    localparam bit EN_STAR_RELU = 1'b0;
    localparam bit EN_SPARSE_SYSTOLIC = 1'b0;
    localparam bit EN_SSF = 1'b1;
    localparam bit EN_RADAR_PROCESSING = 1'b0;
    localparam bit EN_TEMPORAL_ATTENTION = 1'b1;
    localparam bit EN_CONVFORMER = 1'b1;
    localparam bit EN_GFF = 1'b1;
    localparam bit EN_DETECTION_HEAD = 1'b1;
    localparam bit EN_NMS = 1'b1;

    // ── Sparse Systolic ──
    localparam int ELASTIC_FIFO_DEPTH = 4;
    localparam int SPARSE_CSR_BASE = 2816;

    // ── Weight Loading ──
    localparam int WEIGHT_LOAD_MODE = 0;

    // ── Convformer ──
    localparam int MAX_C = 4;
    localparam int MAX_KERNEL_SIZE = 9;
    localparam int EXPANSION_FACTOR = 1;
    localparam int SKIP_CONNECTION_MODE = 0;

    // ── Activation Rom ──
    localparam int INDEX_BITS = 8;
    localparam int ROM_DATA_BITS = 32;
    localparam int NUM_PORTS = 16;
    localparam int INDEX_START_BIT = 12;
    localparam int ROM_WIDTH = 32;
    localparam int GELU_INPUT_RANGE = 8;

    // ── Prehead Widths ──
    localparam int SIGMOID_INPUT_INT_W = 4;
    localparam int SIGMOID_INPUT_FRAC_W = 16;
    localparam int SIGMOID_INPUT_WIDTH = 20;

    // ── Detection ──
    localparam int MAX_DETECTIONS = 32;
    localparam int NUM_CLASSES = 3;
    localparam int TOP_K = 16;
    localparam int PARA_CH = 2;
    localparam int NUM_ANCHORS = 3;
    localparam int PARALLEL_IOU = 1;
    localparam int NMS_DATA_WIDTH = 8;
    localparam int NMS_FRAC_WIDTH = 4;
    localparam int IN_CHANNELS = 8;
    localparam int DEFAULT_ANCHOR_H = 16;
    localparam int DEFAULT_ANCHOR_W = 16;
    localparam int BBOX_COORD_W = 16;

    // ── Prehead ──
    localparam int PREHEAD_PARA_CH = 8;

    // ── Temporal ──
    localparam int HISTORY_DEPTH = 2;
    localparam int KERNEL_T = 1;
    localparam int TM_CH = 4;

    // ── Norm ──
    localparam int GROUPS = 4;

    // ── Temporal Attention ──
    localparam int NUM_HEADS = 1;
    localparam int HEAD_DIM = 4;
    localparam int MAC_UNROLL = 1;
    localparam int OUT_MAC_UNROLL = 1;
    localparam int ATTN_HEADS_PER_CYCLE = 1;
    localparam int PARALLEL_DIV = 1;
    localparam int PARALLEL_HIST = 1;
    localparam int HD_UNROLL = 2;
    localparam int CH_OUT_UNROLL = 16;
    localparam int SCORE_HIST_UNROLL = 4;
    localparam int ENABLE_RESIDUAL = 1;
    localparam int EXP_UNROLL = 8;
    localparam int ENABLE_TEMPERATURE = 0;
    localparam int ENABLE_MASK = 1;
    localparam int QK_WEIGHT_SHARING = 1;

    // ── Pipeline ──
    localparam int TOTAL_LATENCY = 12;
    localparam int DW_LN_FIFO_DEPTH = 4;

    // ── Synthesis Flags ──
    localparam int DEBUG = 0;

    // ── Axi Config ──
    localparam int AXI_DATA_WIDTH = 64;
    localparam int AXI_ADDR_WIDTH = 32;
    localparam int AXI_ID_WIDTH = 4;
    localparam int AXI_BURST_LEN = 16;
    localparam int AXI_STRB_WIDTH = 8;
    localparam int DMA_DESCRIPTOR_DEPTH = 8;
    localparam int DMA_MAX_BURSTS = 4;
    localparam int CSR_BASE_ADDR = 268435456;
    localparam int CSR_ADDR_WIDTH = 12;

    // ── Frame Geometry ──
    localparam int FEATURE_H = 64;
    localparam int FEATURE_W = 64;
    localparam int TILE_H = 16;
    localparam int TILE_W = 16;
    localparam int PAD_MODE = 0;
    localparam int STRIDE = 1;

    // ── Memory ──
    localparam int DEPTH = 16;
    localparam int BRAM_THRESHOLD = 2;
    localparam int LUTRAM_THRESHOLD = 1;
    localparam int PING_PONG_DEPTH = 0;
    localparam int SKID_BUFFER_DEPTH = 4;
    localparam int WEIGHT_FIFO_DEPTH = 8;
    localparam int RESIDUAL_FIFO_DEPTH = 12;
    localparam int DET_FIFO_DEPTH = 16;
    localparam int SSF_FIFO_DEPTH = 64;

    // ── Design Targets ──
    localparam int CLOCK_FREQ_MHZ = 200;
    localparam int OPTIMIZATION_GOAL = 0;
    localparam int MAX_FANOUT = 64;
    localparam int PIPELINE_EXTRA = 0;

    // ── Module Impl ──
    localparam int SYSTOLIC_ARRAY_IMPL = 0;
    localparam int ACTIVATION_IMPL = 0;
    localparam int NORM_IMPL = 0;
    localparam int POOLING_IMPL = 0;
    localparam int NMS_IMPL = 0;
    localparam int ATTENTION_IMPL = 0;

    // ── Gff ──
    localparam int TIME_MUX = 16;
    localparam int L2_ITER = 2;
    localparam int AVG_GROUP_SIZE = 8;
    localparam int STUB_BEHAVIOR = 0;
    localparam int GFF_N = 1024;
    localparam int LINEAR_IN_CH = 16;
    localparam int LINEAR_OUT_CH = 12;

    // ── Impl Flags ──
    localparam bit SYSTOLIC_USE_DSP = 1'b1;
    localparam bit NORM_USE_DSP = 1'b0;
    localparam bit ACT_USE_DSP = 1'b0;
    localparam bit GFF_USE_DSP = 1'b0;
    localparam bit MAGNITUDE_USE_CORDIC = 1'b0;
    localparam int BATCH_NORM_3D_PIPE = 4;

    // ── Norm Math ──
    localparam int LUT_SIZE = 256;
    localparam int LUT_INDEX_BITS = 8;
    localparam int GROUP_NORM_PIPELINE_STAGES = 8;
    localparam int LAYER_NORM_PIPELINE_STAGES = 10;
    localparam int BATCH_NORM_3D_PIPELINE_STAGES = 4;

endpackage