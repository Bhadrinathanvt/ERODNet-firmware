// =============================================================================
// cat_layer.sv
//
// Channel Concatenation Layer.
// Takes two input streams (branch_a: N x C_A, branch_b: N x C_B) and produces
// one output stream (N x (C_A + C_B)) by concatenating channels dimension-wise.
// Both inputs must arrive on the same cycle (valid_a && valid_b).
//
// Pipeline: 1 stage (register + combine)
// =============================================================================

module cat_layer #(
    parameter int N      = 1024,
    parameter int C      = 64,
    parameter int DWIDTH = 16
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic signed [DWIDTH-1:0]    branch_a [0:N-1][0:C-1],
    input  logic                        valid_a,
    input  logic signed [DWIDTH-1:0]    branch_b [0:N-1][0:C-1],
    input  logic                        valid_b,
    output logic signed [DWIDTH-1:0]    cat_out [0:N-1][0:2*C-1],
    output logic                        valid_out
);

    // Register both branches and combine
    logic signed [DWIDTH-1:0] reg_a [0:N-1][0:C-1];
    logic signed [DWIDTH-1:0] reg_b [0:N-1][0:C-1];
    logic                     both_valid;

    assign both_valid = valid_a && valid_b;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            for (int n = 0; n < N; n++)
                for (int ch = 0; ch < C; ch++) begin
                    reg_a[n][ch] <= '0;
                    reg_b[n][ch] <= '0;
                end
        end else begin
            valid_out <= 1'b0;
            if (both_valid) begin
                for (int n = 0; n < N; n++)
                    for (int ch = 0; ch < C; ch++) begin
                        reg_a[n][ch] <= branch_a[n][ch];
                        reg_b[n][ch] <= branch_b[n][ch];
                    end
                valid_out <= 1'b1;
            end
        end
    end

    // Concatenate: lower channels = branch_a, upper channels = branch_b
    always_comb begin
        for (int n = 0; n < N; n++) begin
            for (int ch = 0; ch < C; ch++)
                cat_out[n][ch] = reg_a[n][ch];
            for (int ch = 0; ch < C; ch++)
                cat_out[n][C + ch] = reg_b[n][ch];
        end
    end

endmodule