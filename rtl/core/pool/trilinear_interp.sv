module trilinear_interp #(
    parameter DATA_WIDTH = 16,
    parameter FRAC_WIDTH = 8
)(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,

    // 8 neighbors
    input  logic signed [DATA_WIDTH-1:0] c000, c001, c010, c011,
    input  logic signed [DATA_WIDTH-1:0] c100, c101, c110, c111,

    // fractional distances
    input  logic [FRAC_WIDTH-1:0] xd,
    input  logic [FRAC_WIDTH-1:0] yd,
    input  logic [FRAC_WIDTH-1:0] zd,

    output logic valid_out,
    output logic signed [DATA_WIDTH-1:0] out
);

    // =========================================================================
    // Stage 0: Precompute (1 - x), (1 - y), (1 - z)
    // FIXED: Widen one_minus_* to FRAC_WIDTH+1 bits to prevent overflow when
    // frac=0 (256 wraps to 0 in 8 bits). Cast to signed for multiplication
    // with signed corner values — unsigned * signed treats signed as unsigned
    // per SV rules, producing wrong results for negative corners.
    // =========================================================================
    logic signed [FRAC_WIDTH:0] one_minus_xd, one_minus_yd, one_minus_zd;
    logic signed [FRAC_WIDTH:0] xd_s, yd_s, zd_s;

    always_comb begin
        one_minus_xd = signed'((1 << FRAC_WIDTH) - xd);
        one_minus_yd = signed'((1 << FRAC_WIDTH) - yd);
        one_minus_zd = signed'((1 << FRAC_WIDTH) - zd);
        xd_s = signed'({1'b0, xd});
        yd_s = signed'({1'b0, yd});
        zd_s = signed'({1'b0, zd});
    end

    // =========================================================================
    // Stage 1: X interpolation (pipelined with fraction values)
    // FIXED: All multiplications use signed operands. Pipeline fraction
    // values so Stage 2/3 use delayed values matching their data inputs.
    // =========================================================================
    logic signed [DATA_WIDTH+FRAC_WIDTH:0] c00, c01, c10, c11;
    logic signed [FRAC_WIDTH:0] one_minus_yd_s1, yd_s1;
    logic signed [FRAC_WIDTH:0] one_minus_zd_s1, zd_s1;
    logic valid_s1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c00 <= 0; c01 <= 0; c10 <= 0; c11 <= 0;
            one_minus_yd_s1 <= 0; yd_s1 <= 0;
            one_minus_zd_s1 <= 0; zd_s1 <= 0;
            valid_s1 <= 0;
        end else begin
            valid_s1 <= valid_in;
            // Pipeline fraction values for later stages
            one_minus_yd_s1 <= one_minus_yd;
            yd_s1 <= yd_s;
            one_minus_zd_s1 <= one_minus_zd;
            zd_s1 <= zd_s;

            c00 <= c000 * one_minus_xd + c100 * xd_s;
            c01 <= c001 * one_minus_xd + c101 * xd_s;
            c10 <= c010 * one_minus_xd + c110 * xd_s;
            c11 <= c011 * one_minus_xd + c111 * xd_s;
        end
    end

    // =========================================================================
    // Stage 2: Y interpolation
    // FIXED: Uses pipelined fraction values from Stage 1
    // =========================================================================
    logic signed [DATA_WIDTH+2*FRAC_WIDTH:0] c0, c1;
    logic signed [FRAC_WIDTH:0] one_minus_zd_s2, zd_s2;
    logic valid_s2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c0 <= 0; c1 <= 0;
            one_minus_zd_s2 <= 0; zd_s2 <= 0;
            valid_s2 <= 0;
        end else begin
            valid_s2 <= valid_s1;
            // Pipeline z-fraction for Stage 3
            one_minus_zd_s2 <= one_minus_zd_s1;
            zd_s2 <= zd_s1;

            c0 <= c00 * one_minus_yd_s1 + c10 * yd_s1;
            c1 <= c01 * one_minus_yd_s1 + c11 * yd_s1;
        end
    end

    // =========================================================================
    // Stage 3: Z interpolation
    // FIXED: Uses pipelined fraction values from Stage 2
    // =========================================================================
    logic signed [DATA_WIDTH+3*FRAC_WIDTH:0] c_final;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c_final <= 0;
            valid_out <= 0;
        end else begin
            valid_out <= valid_s2;

            c_final <= c0 * one_minus_zd_s2 + c1 * zd_s2;
        end
    end

    // =========================================================================
    // Output scaling (fixed-point normalize)
    // =========================================================================
    assign out = c_final >>> (3 * FRAC_WIDTH);

endmodule