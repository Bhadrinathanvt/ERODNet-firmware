// =============================================================================
// iou_unit.sv
//
// Division-free Intersection over Union (IoU) calculation unit.
// Implements the "Comparison Trick" to avoid Newton-Raphson reciprocals:
//   I / U > T  <=>  I * (1 + T) > T * (A1 + A2)
// In fixed point:
//   I * (2^FRAC + T_fixed) > T_fixed * (A1 + A2)
// =============================================================================

import pkg_nms::*;
import pkg_module_config::*;

typedef pkg_nms::detection_t detection_t;

module iou_unit #(
    parameter int DATA_WIDTH = pkg_module_config::NMS_TOPK_DATA_WIDTH,
    parameter int FRAC_WIDTH = DATA_WIDTH/2
)(
    input  detection_t det1,
    input  detection_t det2,
    input  logic signed [47:0] area_thresh1, // area1 * active_iou_threshold
    input  logic signed [47:0] area_thresh2, // area2 * active_iou_threshold
    input  logic signed [DATA_WIDTH:0] one_plus_T, // (1 << FRAC_WIDTH) + active_iou_threshold
    output logic suppressed
);

    logic signed [31:0] x1_min, x1_max, y1_min, y1_max;
    logic signed [31:0] x2_min, x2_max, y2_min, y2_max;
    logic signed [31:0] inter_w, inter_h;
    logic signed [31:0] inter_area;
    logic signed [63:0] lhs;
    logic signed [63:0] rhs;

    always_comb begin
        // Bounding box coordinates (assuming center-based indices)
        // Explicit 32-bit casting to avoid WIDTHEXPAND warnings
        x1_min = 32'($signed(det1.w_idx)) - (32'($signed(det1.width)) >>> 1);
        x1_max = 32'($signed(det1.w_idx)) + (32'($signed(det1.width)) >>> 1);
        y1_min = 32'($signed(det1.h_idx)) - (32'($signed(det1.height)) >>> 1);
        y1_max = 32'($signed(det1.h_idx)) + (32'($signed(det1.height)) >>> 1);

        x2_min = 32'($signed(det2.w_idx)) - (32'($signed(det2.width)) >>> 1);
        x2_max = 32'($signed(det2.w_idx)) + (32'($signed(det2.width)) >>> 1);
        y2_min = 32'($signed(det2.h_idx)) - (32'($signed(det2.height)) >>> 1);
        y2_max = 32'($signed(det2.h_idx)) + (32'($signed(det2.height)) >>> 1);

        // Intersection dimensions
        inter_w = (x1_max < x2_max ? x1_max : x2_max) -
                  (x1_min > x2_min ? x1_min : x2_min);
        inter_h = (y1_max < y2_max ? y1_max : y2_max) -
                  (y1_min > y2_min ? y1_min : y2_min);

        if (inter_w < 0) inter_w = 0;
        if (inter_h < 0) inter_h = 0;

        inter_area = inter_w * inter_h;

        // Comparison Trick: I * (1 + T) > T * (A1 + A2)
        lhs = $signed(inter_area) * $signed(one_plus_T);
        rhs = 64'($signed(area_thresh1)) + 64'($signed(area_thresh2));

        // Suppress if IoU > T and detections are valid and not the same index
        // (Same index check usually handled by loop indices in top module)
        suppressed = (det1.valid && det2.valid) && (lhs > rhs);
    end

endmodule
