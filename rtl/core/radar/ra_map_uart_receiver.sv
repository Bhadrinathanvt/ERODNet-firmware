// =============================================================================
// ra_map_uart_receiver.sv
//
// UART receiver for AWR1843AOP pre-computed RA map data.
// Receives raw RA map pixels over UART and drives the ra_map_to_cnn_bridge
// interface, bypassing the on-FPGA radar processing pipeline.
//
// AWR1843AOP UART: 921600 baud, 8N1 (8 data bits, no parity, 1 stop bit)
// Max throughput: 92,160 bytes/sec → ~45 fps for 256×4×2-byte frames
//
// Architecture:
//   - 16x oversampling UART RX (mid-bit sampling for noise immunity)
//   - Byte → pixel assembler (1 or 2 bytes per pixel, configurable)
//   - Angle bin accumulator (ANGLE_BINS pixels per range row)
//   - Inter-frame gap detector (timeout for frame boundary detection)
//   - CSR shadow-register interface for runtime configurability
//
// Data format (raw RA map, row-major):
//   For range_idx 0..RANGE_BINS-1:
//     For angle_idx 0..ANGLE_BINS-1:
//       Pixel[range_idx][angle_idx] (MSB first if 16-bit)
//
// Frame sync: inter-frame gap ≥ 100 bit-times at configured baud rate
// =============================================================================

/* verilator lint_off UNUSEDPARAM */
module ra_map_uart_receiver #(
    parameter int CLK_FREQ_HZ     = 200_000_000,
    parameter int DATA_WIDTH      = 16,
    parameter int MAX_RANGE_BINS  = 1024,
    parameter int MAX_ANGLE_BINS  = 8
)(
    input  logic clk,
    input  logic rst_n,

    // ── UART RX pin ───────────────────────────────────────────────────────
    input  logic uart_rx,

    // ── CSR Configuration (shadow-register compatible) ────────────────────
    input  logic        cfg_update,          // Commit configuration
    input  logic [11:0] cfg_range_bins,      // Range bins per frame (64–1024)
    input  logic [3:0]  cfg_angle_bins,      // Angle bins (1–MAX_ANGLE_BINS)
    input  logic [15:0] cfg_baud_div,        // CLK_FREQ / (BAUD × 16)
    input  logic        cfg_pixel_16bit,     // 0: 8-bit pixel, 1: 16-bit pixel

    // ── Status ────────────────────────────────────────────────────────────
    output logic        frame_active,        // High while receiving a frame
    output logic [15:0] frame_count,         // Total frames received
    output logic        uart_frame_err,      // Stop bit error

    // ── Output to ra_map_to_cnn_bridge ────────────────────────────────────
    output logic signed [DATA_WIDTH-1:0] ra_mag  [MAX_ANGLE_BINS-1:0],
    output logic [9:0]  ra_range_idx,
    output logic        ra_valid,
    output logic        ra_last,
    input  logic        ra_ready
);

    // =========================================================================
    // Shadow registers
    // =========================================================================
    logic [11:0] range_bins;
    logic [3:0]  angle_bins;
    logic [15:0] baud_div;
    logic        pixel_16bit;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            range_bins  <= 12'd256;
            angle_bins  <= 4'd4;
            baud_div    <= 16'd14;   // 200MHz / (921600 × 16) ≈ 13.56 → 14
            pixel_16bit <= 1'b1;
        end else if (cfg_update) begin
            range_bins  <= (cfg_range_bins >= 64)  ? cfg_range_bins  : 12'd64;
            angle_bins  <= (cfg_angle_bins >= 1)   ? cfg_angle_bins  : 4'd1;
            baud_div    <= (cfg_baud_div >= 2)     ? cfg_baud_div    : 16'd2;
            pixel_16bit <= cfg_pixel_16bit;
        end
    end

    // =========================================================================
    // Baud rate generator: produces 16× baud enable pulse
    // =========================================================================
    logic [15:0] baud_cnt;
    logic       baud_16x_tick;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            baud_cnt      <= 16'd0;
            baud_16x_tick <= 1'b0;
        end else begin
            baud_16x_tick <= 1'b0;
            if (baud_cnt >= baud_div - 1) begin
                baud_cnt      <= 16'd0;
                baud_16x_tick <= 1'b1;
            end else begin
                baud_cnt <= baud_cnt + 16'd1;
            end
        end
    end

    // =========================================================================
    // UART RX FSM — 16x oversampling, mid-bit sampling
    // =========================================================================
    typedef enum logic [1:0] {
        RX_IDLE      = 2'd0,
        RX_START_BIT = 2'd1,
        RX_DATA_BITS = 2'd2,
        RX_STOP_BIT  = 2'd3
    } rx_state_t;

    rx_state_t rx_state;
    logic [3:0] oversample_cnt;   // 0–15 counts within a bit
    logic [2:0] bit_cnt;          // 0–7 data bits received
    logic [7:0] rx_shift_reg;
    logic       rx_byte_valid;
    logic [7:0] rx_byte;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_state       <= RX_IDLE;
            oversample_cnt <= 4'd0;
            bit_cnt        <= 3'd0;
            rx_shift_reg   <= 8'd0;
            rx_byte_valid  <= 1'b0;
            rx_byte        <= 8'd0;
        end else if (baud_16x_tick) begin
            rx_byte_valid <= 1'b0;  // pulse

            case (rx_state)
                RX_IDLE: begin
                    oversample_cnt <= 4'd0;
                    bit_cnt        <= 3'd0;
                    if (!uart_rx) begin  // start bit detected (falling edge)
                        rx_state <= RX_START_BIT;
                    end
                end

                RX_START_BIT: begin
                    // Sample mid-bit (count 8 of 16 oversamples)
                    if (oversample_cnt == 4'd7) begin
                        if (uart_rx) begin
                            // False start — return to idle
                            rx_state <= RX_IDLE;
                        end else begin
                            oversample_cnt <= 4'd0;
                            rx_state       <= RX_DATA_BITS;
                        end
                    end else begin
                        oversample_cnt <= oversample_cnt + 4'd1;
                    end
                end

                RX_DATA_BITS: begin
                    // Sample mid-bit (count 8 of 16)
                    if (oversample_cnt == 4'd7) begin
                        rx_shift_reg[bit_cnt] <= uart_rx;
                        oversample_cnt <= 4'd0;
                        if (bit_cnt == 3'd7) begin
                            rx_state <= RX_STOP_BIT;
                        end else begin
                            bit_cnt <= bit_cnt + 3'd1;
                        end
                    end else begin
                        oversample_cnt <= oversample_cnt + 4'd1;
                    end
                end

                RX_STOP_BIT: begin
                    if (oversample_cnt == 4'd7) begin
                        rx_state      <= RX_IDLE;
                        rx_byte       <= rx_shift_reg;
                        rx_byte_valid <= !uart_rx;  // 0 on frame error (stop=0)
                        // Note: frame error latched externally
                    end else begin
                        oversample_cnt <= oversample_cnt + 4'd1;
                    end
                end

                default: rx_state <= RX_IDLE;
            endcase
        end
    end

    // Stop bit / frame error
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            uart_frame_err <= 1'b0;
        end else if (baud_16x_tick && rx_state == RX_STOP_BIT &&
                     oversample_cnt == 4'd7) begin
            uart_frame_err <= uart_rx;  // stop bit should be 1
        end
    end

    // =========================================================================
    // Inter-frame gap detector — ~100 bit-times of idle = frame boundary
    // =========================================================================
    localparam int GAP_BITS = 100;  // 100 bit-times = 1.09 ms at 921600 baud
    localparam int GAP_TICKS = GAP_BITS * 16;  // in 16x oversample ticks

    logic [$clog2(GAP_TICKS+1)-1:0] gap_cnt;
    logic       gap_detected;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gap_cnt       <= '0;
            gap_detected  <= 1'b0;
        end else begin
            gap_detected <= 1'b0;
            if (rx_state != RX_IDLE || !uart_rx) begin
                gap_cnt <= '0;
            end else if (baud_16x_tick && gap_cnt < GAP_TICKS) begin
                gap_cnt <= gap_cnt + 1'b1;
                if (gap_cnt == GAP_TICKS - 1) begin
                    gap_detected <= 1'b1;
                end
            end
        end
    end

    // =========================================================================
    // Byte → pixel assembler
    // =========================================================================
    logic [15:0] pixel_buf;
    logic        pixel_byte_sel;   // 0: low byte, 1: high byte (for 16-bit)
    logic        pixel_valid;      // complete pixel ready

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pixel_buf      <= 16'd0;
            pixel_byte_sel <= 1'b0;
            pixel_valid    <= 1'b0;
        end else begin
            pixel_valid <= 1'b0;

            if (rx_byte_valid) begin
                if (pixel_16bit) begin
                    if (pixel_byte_sel == 1'b0) begin
                        pixel_buf[15:8] <= rx_byte;   // MSB first (big-endian)
                        pixel_byte_sel  <= 1'b1;
                    end else begin
                        pixel_buf[7:0]  <= rx_byte;   // LSB second
                        pixel_byte_sel  <= 1'b0;
                        pixel_valid     <= 1'b1;
                    end
                end else begin
                    pixel_buf   <= {8'd0, rx_byte};
                    pixel_valid <= 1'b1;
                end
            end
        end
    end

    // =========================================================================
    // Frame controller — counts range bins and angle bins, drives output
    // =========================================================================
    typedef enum logic [1:0] {
        FSM_WAIT_FRAME = 2'd0,
        FSM_ACTIVE     = 2'd1,
        FSM_HOLD       = 2'd2   // backpressure stall
    } frame_state_t;

    frame_state_t frm_state;

    logic [$clog2(MAX_RANGE_BINS)-1:0] range_cnt;
    logic [$clog2(MAX_ANGLE_BINS)-1:0] angle_cnt;
    logic signed [DATA_WIDTH-1:0] angle_buf [MAX_ANGLE_BINS-1:0];
    logic        angle_buf_valid;
    logic        is_last_row;            // current row is the final one
    logic        held_last;               // was the held output the last row?

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            frm_state       <= FSM_WAIT_FRAME;
            range_cnt       <= '0;
            angle_cnt       <= '0;
            angle_buf_valid <= 1'b0;
            frame_active    <= 1'b0;
            frame_count     <= 16'd0;
            ra_valid        <= 1'b0;
            ra_last         <= 1'b0;
            ra_range_idx    <= 10'd0;
            for (int i = 0; i < MAX_ANGLE_BINS; i++) begin
                angle_buf[i] <= '0;
                ra_mag[i]    <= '0;
            end
        end else begin
            // Default: deassert valid/last unless held or newly asserted
            if (ra_valid && ra_ready) begin
                ra_valid <= 1'b0;
                ra_last  <= 1'b0;
            end

            // Gap detection resets frame state
            if (gap_detected && frm_state != FSM_HOLD) begin
                range_cnt <= '0;
                angle_cnt <= '0;
                frm_state <= FSM_ACTIVE;
                frame_active <= 1'b1;
            end

            // Accumulate pixels into angle buffer
            if (pixel_valid) begin
                if (angle_cnt < angle_bins) begin
                    angle_buf[angle_cnt] <= $signed(pixel_buf);
                    angle_cnt <= angle_cnt + 1'b1;
                end

                // When all angle bins for this range are collected
                if (angle_cnt == angle_bins - 1) begin
                    angle_cnt <= '0;
                    range_cnt <= range_cnt + 1'b1;
                    is_last_row <= (range_cnt == range_bins - 1);
                    angle_buf_valid <= 1'b1;
                end
            end

            // Output FSM with backpressure handling
            case (frm_state)
                FSM_WAIT_FRAME: begin
                    frame_active <= 1'b0;
                    if (gap_detected) begin
                        frm_state    <= FSM_ACTIVE;
                        range_cnt    <= '0;
                        angle_cnt    <= '0;
                        frame_active <= 1'b1;
                    end
                end

                FSM_ACTIVE: begin
                    if (angle_buf_valid && !ra_valid) begin
                        angle_buf_valid <= 1'b0;
                        held_last       <= is_last_row;
                        ra_range_idx    <= range_cnt;
                        for (int i = 0; i < MAX_ANGLE_BINS; i++) begin
                            if (i < angle_bins)
                                ra_mag[i] <= angle_buf[i];
                            else
                                ra_mag[i] <= '0;
                        end
                        ra_valid <= 1'b1;
                        ra_last  <= is_last_row;
                        if (is_last_row) begin
                            frame_active <= 1'b0;
                            frame_count  <= frame_count + 16'd1;
                            frm_state    <= FSM_WAIT_FRAME;
                        end
                    end

                    if (ra_valid && !ra_ready) begin
                        frm_state <= FSM_HOLD;
                    end
                end

                FSM_HOLD: begin
                    if (ra_ready) begin
                        ra_valid <= 1'b0;
                        ra_last  <= 1'b0;
                        if (held_last) begin
                            frame_active <= 1'b0;
                            frame_count  <= frame_count + 16'd1;
                            frm_state    <= FSM_WAIT_FRAME;
                        end else begin
                            frm_state <= FSM_ACTIVE;
                        end
                    end
                end

                default: frm_state <= FSM_WAIT_FRAME;
            endcase
        end
    end

    // =========================================================================
    // SVA: protocol assertions
    // =========================================================================
`ifdef FORMAL
    always @(posedge clk) begin
        if (rst_n) begin
            // Angle bins must not exceed parameter
            assert(angle_bins <= MAX_ANGLE_BINS);
            // Range bins must be ≥ minimum
            assert(range_bins >= 64);
            // Baud divider must be ≥ 2
            assert(baud_div >= 2);
        end
    end
`endif

endmodule
