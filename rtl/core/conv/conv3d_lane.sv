// =============================================================================
// conv3d_lane.sv
//
// 3D convolution lane for a single output channel.
// Computes the dot product of a flattened spatial kernel window with weights.
//
// Parameters:
//   DATA_WIDTH — input data precision
//   ACC_WIDTH  — accumulator width
//   KERNEL     — cubic kernel size (1 or 3)
//                KERNEL=1 → 1x1x1 = 1 element
//                KERNEL=3 → 3x3x3 = 27 elements
//
// Ports:
//   x_win    — flattened input window [KERNEL^3-1:0]
//   w_kernel — flattened weights      [KERNEL^3-1:0]
//   y_out    — dot-product result (signed, ACC_WIDTH)
//   ready_in — upstream can accept new data (backpressure)
//
// Latency: 1 cycle (registered output)
// Throughput: 1 result/cycle when consumer is always ready
// =============================================================================

module conv3d_lane #(
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32,
    parameter int KERNEL     = 1   // 1 or 3
)(
    input  logic clk,
    input  logic rst_n,

    input  logic                       in_valid,
    output logic                       ready_in,
    input  logic                       out_ready,

    output logic                       out_valid,

    input  logic signed [DATA_WIDTH-1:0] x_win    [KERNEL*KERNEL*KERNEL-1:0],
    input  logic signed [DATA_WIDTH-1:0] w_kernel [KERNEL*KERNEL*KERNEL-1:0],

    output logic signed [ACC_WIDTH-1:0] y_out
);

    localparam int NUM_ELEMS = KERNEL * KERNEL * KERNEL;

    // Combinational MAC
    logic signed [ACC_WIDTH-1:0] acc;

    always_comb begin
        acc = '0;
        for (int i = 0; i < NUM_ELEMS; i++) begin
            acc = acc + ACC_WIDTH'(x_win[i]) * ACC_WIDTH'(w_kernel[i]);
        end
    end

    // Skid buffer: hold output when consumer is not ready
    logic                       skid_valid;
    logic signed [ACC_WIDTH-1:0] skid_data;

    assign ready_in = !skid_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid  <= 1'b0;
            y_out      <= '0;
            skid_valid <= 1'b0;
            skid_data  <= '0;
        end else begin
            if (out_ready) begin
                // Consumer accepted — clear skid, pass through
                skid_valid <= 1'b0;
                out_valid  <= in_valid || skid_valid;
                if (skid_valid) begin
                    y_out <= skid_data;
                end else begin
                    y_out <= acc;
                end
            end else if (in_valid && !skid_valid) begin
                // Consumer not ready, new input — park in skid buffer
                skid_valid <= 1'b1;
                skid_data  <= acc;
                out_valid  <= 1'b1;
                y_out      <= acc;
            end
            // else: consumer not ready, no new input — hold current output
        end
    end

endmodule
