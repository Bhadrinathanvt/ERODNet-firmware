// =============================================================================
//  sat_math.sv — Saturating arithmetic helpers
//
//  Provides signed saturating add/subtract functions.  These clip to the
//  symmetric INT range [-(2^(W-1)) , 2^(W-1)-1] instead of wrapping.
//
//  Usage:
//      import sat_math_pkg::*;
//      out <= DATA_WIDTH'(sat_add_signed(a, b, DATA_WIDTH));
// =============================================================================

package sat_math_pkg;

    // Signed saturating addition — clips on overflow / underflow.
    // Width is passed explicitly because SV functions in packages cannot be
    // parameterised; the caller casts the result back to the desired width.
    function automatic logic signed [31:0] sat_add_signed(
        input logic signed [31:0] a,
        input logic signed [31:0] b,
        input int width
    );
        logic signed [32:0] sum;
        logic signed [31:0] max_pos;
        logic signed [31:0] min_neg;

        sum     = a + b;
        max_pos = (32'sb1 << (width - 1)) - 1;
        min_neg = -(32'sb1 << (width - 1));

        // Overflow: a and b have same sign, but sum sign differs.
        if ((a[width-1] == b[width-1]) && (a[width-1] != sum[width-1])) begin
            return a[width-1] ? min_neg : max_pos;
        end else begin
            return sum[31:0];
        end
    endfunction

    // Signed saturating subtraction — a - b
    function automatic logic signed [31:0] sat_sub_signed(
        input logic signed [31:0] a,
        input logic signed [31:0] b,
        input int width
    );
        logic signed [32:0] diff;
        logic signed [31:0] max_pos;
        logic signed [31:0] min_neg;

        diff    = a - b;
        max_pos = (32'sb1 << (width - 1)) - 1;
        min_neg = -(32'sb1 << (width - 1));

        // Overflow: a and -b have same sign (i.e. a and b differ), but diff sign differs from a.
        if ((a[width-1] != b[width-1]) && (a[width-1] != diff[width-1])) begin
            return a[width-1] ? min_neg : max_pos;
        end else begin
            return diff[31:0];
        end
    endfunction

endpackage
