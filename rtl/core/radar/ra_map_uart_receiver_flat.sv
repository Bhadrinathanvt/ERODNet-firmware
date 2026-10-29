// =============================================================================
// ra_map_uart_receiver_flat.sv
//
// Flat-port wrapper for ra_map_uart_receiver — Verilator compatibility.
// Replaces unpacked array ports with 1D packed equivalents.
// =============================================================================

module ra_map_uart_receiver_flat #(
    parameter int CLK_FREQ_HZ    = 200_000_000,
    parameter int DATA_WIDTH     = 16,
    parameter int MAX_RANGE_BINS = 1024,
    parameter int MAX_ANGLE_BINS = 8
)(
    input  logic clk,
    input  logic rst_n,

    // UART
    input  logic uart_rx,

    // CSR
    input  logic        cfg_update,
    input  logic [11:0] cfg_range_bins,
    input  logic [3:0]  cfg_angle_bins,
    input  logic [15:0] cfg_baud_div,
    input  logic        cfg_pixel_16bit,

    // Status
    output logic        frame_active,
    output logic [15:0] frame_count,
    output logic        uart_frame_err,

    // Output to ra_map_to_cnn_bridge (flat)
    output logic [DATA_WIDTH*MAX_ANGLE_BINS-1:0] ra_mag_flat,
    output logic [9:0]  ra_range_idx,
    output logic        ra_valid,
    output logic        ra_last,
    input  logic        ra_ready
);

    // Unpacked intermediate signals
    logic signed [DATA_WIDTH-1:0] ra_mag_unpacked [MAX_ANGLE_BINS-1:0];

    // Pack outputs
    always_comb begin
        for (int i = 0; i < MAX_ANGLE_BINS; i++) begin
            ra_mag_flat[i*DATA_WIDTH +: DATA_WIDTH] = ra_mag_unpacked[i];
        end
    end

    ra_map_uart_receiver #(
        .CLK_FREQ_HZ    (CLK_FREQ_HZ),
        .DATA_WIDTH     (DATA_WIDTH),
        .MAX_RANGE_BINS (MAX_RANGE_BINS),
        .MAX_ANGLE_BINS (MAX_ANGLE_BINS)
    ) u_receiver (
        .clk             (clk),
        .rst_n           (rst_n),
        .uart_rx         (uart_rx),
        .cfg_update      (cfg_update),
        .cfg_range_bins  (cfg_range_bins),
        .cfg_angle_bins  (cfg_angle_bins),
        .cfg_baud_div    (cfg_baud_div),
        .cfg_pixel_16bit (cfg_pixel_16bit),
        .frame_active    (frame_active),
        .frame_count     (frame_count),
        .uart_frame_err  (uart_frame_err),
        .ra_mag          (ra_mag_unpacked),
        .ra_range_idx    (ra_range_idx),
        .ra_valid        (ra_valid),
        .ra_last         (ra_last),
        .ra_ready        (ra_ready)
    );

endmodule
