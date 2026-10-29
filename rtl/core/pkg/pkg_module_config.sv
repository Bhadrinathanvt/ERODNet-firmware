// ==========================================
// AUTO-GENERATED FILE. DO NOT EDIT MANUALLY.
// Generated from config.yaml — module_widths section
//
// Usage in a module:
//   import pkg_module_config::*;
//   module my_module #(
//       parameter int DATA_WIDTH = MY_MODULE_DATA_WIDTH,
//       parameter int ACC_WIDTH  = pkg_config::ACC_WIDTH,
//       parameter int FRAC_WIDTH = pkg_config::FRAC_WIDTH
//   ) (...);
//
// This allows both:
//   (a) Global change via config.yaml → regenerate
//   (b) Local override at instantiation #(.DATA_WIDTH(12))
// ==========================================

package pkg_module_config;

    // ── Per-Module Data Width Defaults ──
    localparam int RADAR_PROCESSING_TOP_DATA_WIDTH = 32;
    localparam int RA_MAP_PROCESSOR_DATA_WIDTH = 32;
    localparam int CFAR_DETECTOR_DATA_WIDTH = 32;
    localparam int WINDOWING_UNIT_DATA_WIDTH = 32;
    localparam int DIGITAL_BEAMFORMING_DATA_WIDTH = 32;
    localparam int ANGLE_FFT_4PT_DATA_WIDTH = 32;
    localparam int MAGNITUDE_UNIT_DATA_WIDTH = 32;
    localparam int SSF_MODULE_DATA_WIDTH = 8;
    localparam int SYSTOLIC_ARRAY_DATA_WIDTH = 8;
    localparam int DEPTHWISE_CONV2D_DATA_WIDTH = 8;
    localparam int DEPTHWISE_CONV_ENGINE_DATA_WIDTH = 8;
    localparam int UNIFIED_ACTIVATION_DATA_WIDTH = 8;
    localparam int LAYER_NORM_FIXED_DATA_WIDTH = 16;
    localparam int BATCH_NORM_3D_DATA_WIDTH = 16;
    localparam int GROUP_NORM_DATA_WIDTH = 16;
    localparam int GFF_DATA_WIDTH = 8;
    localparam int GFF_STREAM_CORE_DATA_WIDTH = 8;
    localparam int LINEAR_LAYER_DATA_WIDTH = 8;
    localparam int TEMPORAL_ATTENTION_ENHANCED_DATA_WIDTH = 8;
    localparam int TEMPORAL_ATTENTION_ENHANCED_FRAC_WIDTH = 8;
    localparam int TEMPORAL_CONVLSTM_DATA_WIDTH = 8;
    localparam int TEMPORAL_FUSION_DATA_WIDTH = 8;
    localparam int DETECTION_HEAD_DATA_WIDTH = 8;
    localparam int NMS_TOPK_DATA_WIDTH = 8;

endpackage