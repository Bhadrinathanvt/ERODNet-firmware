// =============================================================================
// pkg_nms.sv
//
// Package for Non-Maximum Suppression (NMS) types and constants.
// =============================================================================

package pkg_nms;
    import pkg_config::*;

    // NMS field width is config-driven (pkg_config::NMS_DATA_WIDTH, from the
    // detection section) so a resource-fit profile can shrink the heap_mem/nms_buf
    // register arrays. h_idx/w_idx stay 16-bit (pixel coordinates — BBOX_COORD_W).
    localparam int NMS_DATA_W = pkg_config::NMS_DATA_WIDTH;

    typedef struct packed {
        logic [15:0]                  h_idx;
        logic [15:0]                  w_idx;
        logic signed [NMS_DATA_W-1:0] confidence;
        logic signed [NMS_DATA_W-1:0] width;
        logic signed [NMS_DATA_W-1:0] height;
        logic [pkg_config::NUM_CLASSES-1:0][NMS_DATA_W-1:0] class_scores;
        logic [1:0]                   class_id;
        logic                         valid;
    } detection_t;

endpackage
