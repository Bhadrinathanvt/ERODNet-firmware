// =============================================================================
// unpooling.sv
//
// Nearest-neighbor upsampling (broadcast) from a single global vector to N spatial
// positions. Takes a C-dimensional vector and replicates it across all N positions,
// producing an N x C output.
//
// Pipeline: 1 stage (broadcast)
// =============================================================================

module unpooling #(
    parameter int C      = 64,
    parameter int N      = 1024,
    parameter int DWIDTH = 16
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic signed [DWIDTH-1:0]    data_in [0:C-1],
    input  logic                        valid_in,
    output logic signed [DWIDTH-1:0]    data_out [0:N-1][0:C-1],
    output logic                        valid_out
);

    // Register input and broadcast on valid
    logic signed [DWIDTH-1:0] data_reg [0:C-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            for (int ch = 0; ch < C; ch++)
                data_reg[ch] <= '0;
        end else begin
            valid_out <= 1'b0;
            if (valid_in) begin
                for (int ch = 0; ch < C; ch++)
                    data_reg[ch] <= data_in[ch];
                valid_out <= 1'b1;
            end
        end
    end

    // Broadcast: every spatial position gets the same C-dimensional vector
    always_comb begin
        for (int n = 0; n < N; n++)
            for (int ch = 0; ch < C; ch++)
                data_out[n][ch] = data_reg[ch];
    end

endmodule