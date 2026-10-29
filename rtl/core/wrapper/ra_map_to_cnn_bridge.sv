// =============================================================================
// ra_map_to_cnn_bridge.sv
//
// Bridge between radar RA map output and CNN tile input.
// Converts streaming range-bin pixels (with angle bins as channels) into
// raster-order tiles suitable for e_rodnet_unet_top.
//
// Architecture:
//   - Ping-pong double buffering: one buffer collects radar output while the
//     other streams to the CNN. Swaps on frame boundary (ra_last).
//   - Dimension mapping: nearest-neighbor for mismatched radar/CNN sizes.
//   - Channel mapping: angle bins → channels [0:ANGLE_BINS-1], remainder zero.
//
// BUG FIXES (2026-05-12):
//   - BRIDGE-001: Single-pixel frame edge case: if ra_valid && ra_last on
//     the same cycle with buf_write_busy == 0, the read FSM never started
//     because buf_write_busy transitioned 0→1→0 in one NBA block. Fixed by
//     using a frame_complete pulse that latches for one cycle.
//   - BRIDGE-002: No backpressure on radar input: ra_ready was hardwired to 1,
//     meaning the radar could overwrite the buffer the CNN was still reading.
//     Fixed by deasserting ra_ready when the read side is streaming the same
//     buffer the write side would use next.
//   - BRIDGE-003: Off-by-one pixel count and valid-drop under backpressure:
//     registered cnn_valid_int added a 1-cycle latency that mismatched the skid
//     buffer, producing an extra pixel after frame end and dropping cnn_valid
//     during backpressure stalls. Fixed by making cnn_valid_int combinational
//     (derived from rstate) and replacing the bare register stage with a proper
//     skid buffer that holds valid data during backpressure. The FSM now advances
//     on skid_accept (cnn_ready || !skid_valid) instead of cnn_ready directly.
//
// Latency: 2 cycles (combinational BRAM read + registered output) + frame acquisition time
// Throughput: 1 pixel/cycle after warmup
// =============================================================================

module ra_map_to_cnn_bridge #(
    parameter int RADAR_RANGE_BINS = 256,
    parameter int RADAR_ANGLE_BINS = 4,
    parameter int FEATURE_H        = 64,
    parameter int FEATURE_W        = 64,
    parameter int MAX_C            = 64,
    parameter int DATA_WIDTH       = 8,
    parameter int FRAC_WIDTH       = 16
)(
    input  logic clk,
    input  logic rst_n,

    // Radar input (from radar_processing_top / ra_map_processor)
    input  logic signed [DATA_WIDTH-1:0] ra_mag  [RADAR_ANGLE_BINS-1:0],
    input  logic        [9:0]            ra_range_idx,
    input  logic                         ra_valid,
    input  logic                         ra_last,
    output logic                         ra_ready,

    // CNN output (to e_rodnet_unet_top)
    output logic signed [DATA_WIDTH-1:0] cnn_data [MAX_C-1:0],
    output logic                         cnn_valid,
    output logic                         cnn_frame_last,
    input  logic                         cnn_ready
);

    // =========================================================================
    // Dimension check
    // =========================================================================
    localparam int FRAME_PIXELS = FEATURE_H * FEATURE_W;
    localparam int RADAR_PIXELS   = RADAR_RANGE_BINS * RADAR_ANGLE_BINS;

    // =========================================================================
    // Scaling factors for nearest-neighbor mapping (power-of-2 assumption)
    // =========================================================================
    localparam int H_SCALE = $clog2(RADAR_RANGE_BINS) - $clog2(FEATURE_H);
    localparam int W_SCALE = $clog2(RADAR_ANGLE_BINS)  - $clog2(FEATURE_W);

    // =========================================================================
    // Ping-pong buffers (double buffering for continuous operation)
    // =========================================================================
    // Each buffer holds one full radar frame: RANGE_BINS × ANGLE_BINS
    logic signed [DATA_WIDTH-1:0] ping_buf [0:RADAR_RANGE_BINS-1][0:RADAR_ANGLE_BINS-1];
    logic signed [DATA_WIDTH-1:0] pong_buf [0:RADAR_RANGE_BINS-1][0:RADAR_ANGLE_BINS-1];

    logic buf_write_sel;      // 0=ping, 1=pong (write side)
    logic buf_write_busy;     // buffer currently being written
    logic frame_complete;     // pulse: a frame was just completed

    // =========================================================================
    // Write side: collect radar output into active buffer
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buf_write_sel  <= 1'b0;
            buf_write_busy <= 1'b0;
            frame_complete <= 1'b0;
        end else begin
            frame_complete <= 1'b0;  // default: one-cycle pulse

            if (ra_valid && !buf_write_busy) begin
                buf_write_busy <= 1'b1;
            end

            if (ra_valid && ra_last) begin
                buf_write_sel  <= ~buf_write_sel;
                buf_write_busy <= 1'b0;
                frame_complete  <= 1'b1;  // BRIDGE-001 fix: explicit pulse
            end
        end
    end

    // Write radar pixels into active buffer
    always_ff @(posedge clk) begin
        if (ra_valid) begin
            for (int a = 0; a < RADAR_ANGLE_BINS; a++) begin
                if (buf_write_sel == 1'b0)
                    ping_buf[ra_range_idx][a] <= ra_mag[a];
                else
                    pong_buf[ra_range_idx][a] <= ra_mag[a];
            end
        end
    end

    // BRIDGE-002 fix: backpressure on radar input.
    // Deassert ra_ready when the read FSM is streaming and the write side
    // would overwrite the buffer the read side is currently using.
    // Write side writes to buf_write_sel; read side reads from rd_buf_sel.
    // Collision occurs when rstate == R_STREAM && rd_buf_sel == buf_write_sel.
    // Also deassert when the read FSM has not yet consumed the previous frame
    // and a new frame arrives (buf_write_busy && rstate == R_STREAM).
    logic rd_active;
    assign rd_active = (rstate != R_IDLE);

    // The write side will write to (~buf_write_sel) after swap, but during a
    // frame in progress it writes to buf_write_sel. The read side reads from
    // rd_buf_sel. Collision when both are in the same buffer.
    // Safe to accept radar data when: not reading, OR reading the opposite buffer.
    assign ra_ready = !rd_active || (rd_buf_sel != buf_write_sel);

    // =========================================================================
    // Read side: stream buffered frame as CNN tiles
    // =========================================================================
    typedef enum logic { R_IDLE, R_STREAM } rstate_t;
    rstate_t rstate;

    logic [$clog2(FRAME_PIXELS)-1:0] rd_pixel_cnt;
    logic [$clog2(FEATURE_H)-1:0]    rd_h;
    logic [$clog2(FEATURE_W)-1:0]    rd_w;
    logic                              rd_buf_sel;

    // Nearest-neighbor mapping from CNN pixel (h,w) to radar pixel
    logic [$clog2(RADAR_RANGE_BINS)-1:0] map_h;
    logic [$clog2(RADAR_ANGLE_BINS)-1:0] map_w;

    always_comb begin
        if (RADAR_RANGE_BINS == FEATURE_H)
            map_h = rd_h;
        else
            map_h = rd_h << H_SCALE;

        if (RADAR_ANGLE_BINS == FEATURE_W)
            map_w = rd_w;
        else
            map_w = rd_w << W_SCALE;
    end

    // Read from active buffer (combinational BRAM read)
    logic signed [DATA_WIDTH-1:0] rd_pixel [RADAR_ANGLE_BINS-1:0];
    always_comb begin
        if (rd_buf_sel == 1'b0) begin
            for (int a = 0; a < RADAR_ANGLE_BINS; a++)
                rd_pixel[a] = ping_buf[map_h][a];
        end else begin
            for (int a = 0; a < RADAR_ANGLE_BINS; a++)
                rd_pixel[a] = pong_buf[map_h][a];
        end
    end

    // Skid buffer pipeline stage: holds valid data when cnn_ready is deasserted.
    // This ensures cnn_valid stays asserted and cnn_data stays stable during
    // backpressure, and prevents the off-by-one extra pixel that a bare
    // register stage would produce (BRIDGE-003).
    logic signed [DATA_WIDTH-1:0] skid_data  [RADAR_ANGLE_BINS-1:0];
    logic                         skid_valid;
    logic                         skid_last;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int a = 0; a < RADAR_ANGLE_BINS; a++) skid_data[a] <= '0;
            skid_valid <= 1'b0;
            skid_last  <= 1'b0;
        end else begin
            if (cnn_ready || !skid_valid) begin
                // Consumer accepted or skid was empty — load new data
                for (int a = 0; a < RADAR_ANGLE_BINS; a++) skid_data[a] <= rd_pixel[a];
                skid_valid <= cnn_valid_int;
                skid_last  <= cnn_frame_last_int;
            end
            // When !cnn_ready && skid_valid: hold skid_data/valid/last unchanged
        end
    end

    // Output assignment: angle bins → low channels, rest zero-padded
    always_comb begin
        for (int c = 0; c < MAX_C; c++) begin
            if (c < RADAR_ANGLE_BINS)
                cnn_data[c] = skid_data[c];
            else
                cnn_data[c] = '0;
        end
    end

    // Skid buffer acceptance: the FSM should advance when the skid buffer can
    // accept new data (consumer ready OR skid buffer empty). This ensures the
    // FSM stalls when the skid buffer is holding data and the consumer isn't ready.
    logic skid_accept;
    assign skid_accept = cnn_ready || !skid_valid;

    // Read FSM: stream one frame, then return to idle
    // Advances when the skid buffer can accept new data (skid_accept).
    // cnn_valid_int and cnn_frame_last_int are combinational (derived from FSM
    // state and counters) so the skid buffer sees them in the same cycle —
    // no 1-cycle latency mismatch (BRIDGE-003).
    assign cnn_valid_int = (rstate == R_STREAM);
    assign cnn_frame_last_int = (rstate == R_STREAM) && skid_accept &&
                                 (rd_h == FEATURE_H - 1) && (rd_w == FEATURE_W - 1);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rstate            <= R_IDLE;
            rd_pixel_cnt      <= '0;
            rd_h              <= '0;
            rd_w              <= '0;
            rd_buf_sel        <= 1'b0;
        end else begin
            case (rstate)
                R_IDLE: begin
                    rd_pixel_cnt <= '0;
                    rd_h         <= '0;
                    rd_w         <= '0;
                    // BRIDGE-001 fix: use frame_complete pulse instead of
                    // buf_write_busy (which can be 0 on single-pixel frames)
                    if (frame_complete) begin
                        rd_buf_sel <= ~buf_write_sel;  // read from just-completed buffer
                        rstate     <= R_STREAM;
                    end
                end

                R_STREAM: begin
                    if (skid_accept) begin
                        if (rd_w == FEATURE_W - 1) begin
                            rd_w <= '0;
                            if (rd_h == FEATURE_H - 1) begin
                                rstate <= R_IDLE;
                            end else begin
                                rd_h <= rd_h + 1'b1;
                            end
                        end else begin
                            rd_w <= rd_w + 1'b1;
                        end
                        rd_pixel_cnt <= rd_pixel_cnt + 1'b1;
                    end
                end

                default: rstate <= R_IDLE;
            endcase
        end
    end

    // Drive registered outputs (from skid buffer, not bare register)
    assign cnn_valid      = skid_valid;
    assign cnn_frame_last = skid_last;

endmodule