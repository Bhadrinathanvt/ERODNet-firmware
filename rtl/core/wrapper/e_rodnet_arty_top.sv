// =============================================================================
//  e_rodnet_arty_top.sv
//
//  Reduced E-RODNet top-level for Artix-7 100T.
//
//  Pipeline (single-path, no U-Net tiling):
//    Input stream → ConvFormer block → Detection head → NMS → Output
//
//  Resource strategy:
//    - One ConvFormer block instead of five (saves ~4× logic + BRAM).
//    - No encoder/decoder skip buffers (saves 164 KB BRAM).
//    - No GFF / temporal attention (saves l2norm_nr + temporal_buffer).
//    - No sparse systolic variant (saves systolic_array_sparse + NZ encoder).
//    - All dimensions reduced via pkg_config (M=2,N=2, MAX_C=8, etc.).
//
//  Expected utilization (Yosys estimate):
//    LUTs   ~35-50 K   (budget 101 K)
//    FFs    ~40-60 K   (budget 127 K)
//    DSPs   ~40-80     (budget 240)
//    BRAMs  ~20-40     (budget 135)
// =============================================================================

import pkg_config::*;

module e_rodnet_arty_top (
    input  logic clk,
    input  logic rst_n,

    // ── AXI4-Lite CSR Slave (control) ────────────────────────────────────────
    input  logic        csr_awvalid,
    output logic        csr_awready,
    input  logic [11:0] csr_awaddr,
    input  logic        csr_wvalid,
    output logic        csr_wready,
    input  logic [31:0] csr_wdata,
    input  logic [3:0]  csr_wstrb,
    output logic        csr_bvalid,
    input  logic        csr_bready,
    output logic [1:0]  csr_bresp,
    input  logic        csr_arvalid,
    output logic        csr_arready,
    input  logic [11:0] csr_araddr,
    output logic        csr_rvalid,
    input  logic        csr_rready,
    output logic [31:0] csr_rdata,
    output logic [1:0]  csr_rresp,

    // ── Streaming input (one pixel at a time, raster order) ──────────────────
    input  logic                        in_valid,
    input  logic                        in_last,      // last pixel of frame
    input  logic signed [DATA_WIDTH-1:0] in_data [MAX_C-1:0],
    output logic                        in_ready,

    // ── Weight loading (broadcast to all submodules) ─────────────────────────
    input  logic        weight_valid,
    input  logic [31:0] weight_addr,
    input  logic signed [DATA_WIDTH-1:0] weight_data,
    input  logic [3:0]  weight_layer_sel,

    // ── Detection output (post-NMS) ──────────────────────────────────────────
    output logic                        det_valid,
    output logic [15:0]                 det_h_idx,
    output logic [15:0]                 det_w_idx,
    output logic signed [DATA_WIDTH-1:0] det_confidence,
    output logic [15:0]                 det_width,
    output logic [15:0]                 det_height,
    output logic [NUM_CLASSES-1:0][DATA_WIDTH-1:0] det_class_scores,
    output logic [1:0]                  det_class_id,
    output logic                        det_last,
    output logic [15:0]                 det_count,
    output logic [15:0]                 det_suppressed_count,
    input  logic                        det_ready,

    // ── Status ───────────────────────────────────────────────────────────────
    output logic [31:0] perf_cycle_count,
    output logic [31:0] perf_frame_count,
    output logic        processing_busy,
    output logic        processing_done
);

    // =========================================================================
    //  CSR registers (shadow + live)
    // =========================================================================
    logic        cfg_enable;
    logic [2:0]  cfg_act_sel;
    logic        cfg_clk_gate;
    logic [15:0] cfg_c;
    logic [15:0] cfg_kernel_size;
    logic signed [DATA_WIDTH-1:0] cfg_conf_threshold;
    logic signed [DATA_WIDTH-1:0] cfg_iou_threshold;
    logic [7:0]  cfg_top_k;
    logic        cfg_update_sh, cfg_update_live;
    logic [31:0] scale_in, bias_in;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cfg_enable         <= 1'b0;
            cfg_act_sel        <= 3'd5;               // StarReLU default
            cfg_clk_gate       <= 1'b0;
            cfg_c              <= MAX_C[15:0];
            cfg_kernel_size    <= 16'd9;
            cfg_conf_threshold <= 8'h20;
            cfg_iou_threshold  <= 8'h10;
            cfg_top_k          <= TOP_K[7:0];
            scale_in           <= 32'd16384;          // 0.25 Q16.16
            bias_in            <= 32'd0;
            cfg_update_sh      <= 1'b0;
        end else begin
            if (csr_awvalid && csr_wvalid && csr_awready) begin
                case (csr_awaddr)
                    12'h000: {cfg_update_sh, cfg_clk_gate, cfg_act_sel, cfg_enable}
                             <= {1'b1, csr_wdata[4:0]};
                    12'h004: cfg_c           <= csr_wdata[15:0];
                    12'h008: cfg_kernel_size <= csr_wdata[15:0];
                    12'h00C: cfg_conf_threshold <= csr_wdata[7:0];
                    12'h010: cfg_iou_threshold  <= csr_wdata[7:0];
                    12'h014: cfg_top_k       <= csr_wdata[7:0];
                    12'h018: scale_in        <= csr_wdata;
                    12'h01C: bias_in         <= csr_wdata;
                    default: ;
                endcase
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cfg_update_live <= 1'b0;
        end else begin
            cfg_update_live <= cfg_update_sh;
            if (cfg_update_sh) cfg_update_sh <= 1'b0;
        end
    end

    // AXI4-Lite write response
    assign csr_awready = 1'b1;
    assign csr_wready  = 1'b1;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) csr_bvalid <= 1'b0;
        else        csr_bvalid <= (csr_awvalid && csr_wvalid);
    end
    assign csr_bresp = 2'b00;

    // AXI4-Lite read
    assign csr_arready = 1'b1;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            csr_rvalid <= 1'b0;
            csr_rdata  <= '0;
        end else begin
            csr_rvalid <= csr_arvalid;
            if (csr_arvalid) begin
                case (csr_araddr)
                    12'h000: csr_rdata <= {27'd0, cfg_clk_gate, cfg_act_sel, cfg_enable};
                    12'h004: csr_rdata <= {16'd0, cfg_c};
                    12'h008: csr_rdata <= {16'd0, cfg_kernel_size};
                    12'h00C: csr_rdata <= {24'd0, cfg_conf_threshold};
                    12'h010: csr_rdata <= {24'd0, cfg_iou_threshold};
                    12'h014: csr_rdata <= {24'd0, cfg_top_k};
                    12'h018: csr_rdata <= scale_in;
                    12'h01C: csr_rdata <= bias_in;
                    12'h020: csr_rdata <= perf_cycle_count;
                    12'h024: csr_rdata <= perf_frame_count;
                    12'h028: csr_rdata <= {16'd0, det_count};
                    default: csr_rdata <= '0;
                endcase
            end
        end
    end
    assign csr_rresp = 2'b00;

    // =========================================================================
    //  Frame-level control
    // =========================================================================
    logic [31:0] cycle_cnt;
    logic [31:0] frame_cnt;
    logic        frame_active;
    logic        frame_last_seen;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_cnt       <= '0;
            frame_cnt       <= '0;
            frame_active    <= 1'b0;
            frame_last_seen <= 1'b0;
        end else begin
            if (cfg_enable && !frame_active && in_valid) begin
                frame_active <= 1'b1;
                cycle_cnt    <= '0;
            end else if (frame_active) begin
                cycle_cnt <= cycle_cnt + 1'b1;
            end

            if (frame_active && in_valid && in_last) begin
                frame_last_seen <= 1'b1;
            end

            if (frame_last_seen && det_last) begin
                frame_active    <= 1'b0;
                frame_last_seen <= 1'b0;
                frame_cnt       <= frame_cnt + 1'b1;
            end
        end
    end

    assign perf_cycle_count = cycle_cnt;
    assign perf_frame_count = frame_cnt;
    assign processing_busy  = frame_active;
    assign processing_done  = !frame_active && (frame_cnt > '0);

    // =========================================================================
    //  ConvFormer block (single instance)
    // =========================================================================
    logic                        cf_valid_in;
    logic                        cf_valid_in_last;
    logic signed [DATA_WIDTH-1:0] cf_token_in [MAX_C-1:0];
    logic                        cf_ready_in;
    logic                        cf_valid_out;
    logic                        cf_valid_out_last;
    logic signed [DATA_WIDTH-1:0] cf_token_out [MAX_C-1:0];
    logic                        cf_out_ready;

    assign cf_valid_in      = in_valid && cfg_enable;
    assign cf_valid_in_last = in_last;
    assign cf_token_in      = in_data;
    assign in_ready         = cf_ready_in;

    convformer_block #(
        .TOTAL_LATENCY     (TOTAL_LATENCY),
        .STUB_BEHAVIOR     (1'b0),
        .WEIGHT_LOAD_MODE  (WEIGHT_LOAD_MODE),
        .EN_SPARSE_SYSTOLIC(1'b0)
    ) u_convformer (
        .clk                (clk),
        .rst_n              (rst_n),
        .cfg_c              (cfg_c),
        .cfg_kernel_size    (cfg_kernel_size),
        .cfg_act_sel        (cfg_act_sel),
        .scale_in           (scale_in),
        .bias_in            (bias_in),
        .cfg_clk_gate       (cfg_clk_gate),
        .valid_in           (cf_valid_in),
        .valid_in_last      (cf_valid_in_last),
        .token_in           (cf_token_in),
        .ready_in           (cf_ready_in),
        .valid_out          (cf_valid_out),
        .valid_out_last     (cf_valid_out_last),
        .token_out          (cf_token_out),
        .out_ready          (cf_out_ready),
        .pw_wr_en           (weight_valid && weight_layer_sel == 4'd1),
        .pw_wr_bank         (weight_addr[17:16]),
        .pw_wr_addr         (weight_addr[13:0]),
        .pw_wr_data         (weight_data),
        .dw_wr_en           (weight_valid && weight_layer_sel == 4'd2),
        .dw_wr_ch           (weight_addr[15:0]),
        .dw_wr_k            (weight_addr[31:16]),
        .dw_wr_data         (weight_data),
        .perf_cycle_count   (),
        .perf_stall_count   (),
        .perf_token_count   (),
        .sparse_csr_req_valid (1'b0),
        .sparse_csr_req_addr  ('0),
        .sparse_csr_req_wdata ('0),
        .sparse_csr_req_we  (1'b0),
        .sparse_csr_req_rdata (),
        .sparse_csr_req_ready ()
    );

    // =========================================================================
    //  Detection head
    // =========================================================================
    // Spatial position counters (raster order) driven by convformer output handshake.
    // These feed feat_h_idx/feat_w_idx so the detection head computes correct
    // absolute bounding box coordinates.
    logic [$clog2(FEATURE_H)-1:0] feat_h;
    logic [$clog2(FEATURE_W)-1:0] feat_w;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            feat_h <= '0;
            feat_w <= '0;
        end else if (cf_valid_out && cf_out_ready) begin
            if (feat_w == $clog2(FEATURE_W)'(FEATURE_W-1)) begin
                feat_w <= '0;
                feat_h <= feat_h + 1;
            end else begin
                feat_w <= feat_w + 1;
            end
        end
    end

    logic                        dh_valid;
    logic [15:0]                 dh_h_idx;
    logic [15:0]                 dh_w_idx;
    logic signed [DATA_WIDTH-1:0] dh_confidence;
    logic signed [DATA_WIDTH-1:0] dh_dx, dh_dy, dh_dw, dh_dh;
    logic [NUM_CLASSES-1:0][DATA_WIDTH-1:0] dh_class_scores;
    logic [$clog2(2)-1:0]        dh_anchor_idx;
    logic [15:0]                 dh_count;
    logic                        dh_last;

    detection_head #(
        .FEATURE_H     (FEATURE_H),
        .FEATURE_W     (FEATURE_W),
        .IN_CHANNELS   (MAX_C),
        .NUM_CLASSES   (NUM_CLASSES),
        .NUM_ANCHORS   (2),
        .DATA_WIDTH    (DATA_WIDTH),
        .FRAC_WIDTH    (FRAC_WIDTH),
        .WEIGHT_LOAD_MODE(WEIGHT_LOAD_MODE),
        .PARA_CH       (PARA_CH)
    ) u_detection_head (
        .clk               (clk),
        .rst_n             (rst_n),
        .cfg_feature_h     (FEATURE_H[15:0]),
        .cfg_feature_w     (FEATURE_W[15:0]),
        .cfg_num_classes   (NUM_CLASSES[7:0]),
        .cfg_num_anchors   (8'd2),
        .cfg_enable        (cfg_enable),
        .feat_valid        (cf_valid_out),
        .feat_ready        (cf_out_ready),
        .feat_h_idx        ({ {(16-$clog2(FEATURE_H)){1'b0}}, feat_h }),
        .feat_w_idx        ({ {(16-$clog2(FEATURE_W)){1'b0}}, feat_w }),
        .feat_data         (cf_token_out),
        .feat_last         (cf_valid_out_last),
        .cls_weight_valid  (weight_valid && weight_layer_sel == 4'd3),
        .cls_weight_addr   (weight_addr[15:0]),
        .cls_weight_data   (weight_data),
        .bbox_weight_valid (weight_valid && weight_layer_sel == 4'd4),
        .bbox_weight_addr  (weight_addr[15:0]),
        .bbox_weight_data  (weight_data),
        .det_valid         (dh_valid),
        .det_ready         (1'b1),           // NMS has no backpressure; always accept
        .det_h_idx         (dh_h_idx),
        .det_w_idx         (dh_w_idx),
        .det_anchor_idx    (dh_anchor_idx),
        .det_confidence    (dh_confidence),
        .det_dx            (dh_dx),
        .det_dy            (dh_dy),
        .det_dw            (dh_dw),
        .det_dh            (dh_dh),
        .det_class_scores  (dh_class_scores),
        .det_count         (dh_count),
        .det_last          (dh_last)
    );

    // =========================================================================
    //  NMS + Top-K
    // =========================================================================
    nms_topk #(
        .MAX_DETECTIONS (MAX_DETECTIONS),
        .TOP_K          (TOP_K),
        .NUM_CLASSES    (NUM_CLASSES),
        .DATA_WIDTH     (DATA_WIDTH),
        .FRAC_WIDTH     (FRAC_WIDTH)
    ) u_nms (
        .clk                (clk),
        .rst_n              (rst_n),
        .cfg_enable         (cfg_enable),
        .cfg_conf_threshold (cfg_conf_threshold),
        .cfg_iou_threshold  (cfg_iou_threshold),
        .cfg_top_k          (cfg_top_k),
        .det_valid          (dh_valid),
        .det_h_idx          (dh_h_idx),
        .det_w_idx          (dh_w_idx),
        .det_anchor_idx     ({1'b0, dh_anchor_idx}),
        .det_confidence     (dh_confidence),
        .det_dx             (dh_dx),
        .det_dy             (dh_dy),
        .det_dw             (dh_dw),
        .det_dh             (dh_dh),
        .det_class_scores   (dh_class_scores),
        .det_last           (dh_last),
        .out_valid          (det_valid),
        .out_h_idx          (det_h_idx),
        .out_w_idx          (det_w_idx),
        .out_confidence     (det_confidence),
        .out_width          (det_width),
        .out_height         (det_height),
        .out_class_scores   (det_class_scores),
        .out_class_id       (det_class_id),
        .out_last           (det_last),
        .out_ready          (out_ready),   // FIXED: was unconnected (floating) —
                                           // propagate the detection-consumer's ready
                                           // so the NMS engine can be back-pressured
        .out_count          (det_count),
        .suppressed_count   (det_suppressed_count)
    );

endmodule
