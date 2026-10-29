// =============================================================================
// avgpool3d_unroll.sv
//
// 3D Average Pooling with Hybrid Tree-Fold Reduction.
// Reduces N x 4C feature map to a single 1 x 4C global vector by averaging
// across all N spatial positions.
//
// Architecture:
//   - Group-combinational add: N points split into groups of TREE_GROUP_SIZE.
//     Within each group, all additions are combinational (1 cycle to register).
//   - Sequential tree fold: NUM_GROUPS partial sums reduced through
//     log2(NUM_GROUPS) levels, one level per cycle, each halving the count.
//   - Final division: 1 cycle to divide by N (right shift for power-of-2).
//
// Latency: log2(NUM_GROUPS) + 2 cycles
//   For N=1024, TREE_GROUP_SIZE=16: log2(64) + 2 = 8 cycles
//   For N=256,  TREE_GROUP_SIZE=16: log2(16) + 2 = 6 cycles
//
// BUG FIX (GFF-AVGPOOL-001): Previous implementation had a critical bug
// where valid_in was a single-cycle pulse but the accumulation loop
// required N cycles of valid_in held high. The early-termination condition
// (!valid_in && count > 0) fired after only 1 accumulation, producing an
// incorrect global vector. This version uses a tree-fold that correctly
// accumulates all N points in log2(N/GROUP_SIZE)+2 cycles.
// =============================================================================

module avgpool3d_unroll #(
    parameter int C      = 64,
    parameter int N      = 1024,
    parameter int DWIDTH = 16,
    parameter int TREE_GROUP_SIZE = 16  // Points per combinational group (power of 2)
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic signed [DWIDTH-1:0]    feat_in [0:N-1][0:4*C-1],
    input  logic                        valid_in,
    output logic signed [DWIDTH-1:0]    feat_out [0:4*C-1],
    output logic                        valid_out
);

    // -------------------------------------------------------------------------
    // Derived parameters
    // -------------------------------------------------------------------------
    localparam int NUM_GROUPS    = (N + TREE_GROUP_SIZE - 1) / TREE_GROUP_SIZE;
    localparam int TREE_DEPTH    = $clog2(NUM_GROUPS);
    localparam int ACC_WIDTH     = DWIDTH + $clog2(N);

    // -------------------------------------------------------------------------
    // Stage 0: Combinational group reduction
    //   Each group of up to TREE_GROUP_SIZE points is reduced to a single partial
    //   sum via combinational adder tree. All NUM_GROUPS partial sums are
    //   registered in one cycle.
    // -------------------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] group_sums [0:NUM_GROUPS-1][0:4*C-1];
    logic                         stage0_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage0_valid <= 1'b0;
            for (int g = 0; g < NUM_GROUPS; g++)
                for (int ch = 0; ch < 4*C; ch++)
                    group_sums[g][ch] <= '0;
        end else begin
            stage0_valid <= valid_in;
            if (valid_in) begin
                for (int g = 0; g < NUM_GROUPS; g++) begin
                    for (int ch = 0; ch < 4*C; ch++) begin
                        // Combinational reduction within group
                        logic signed [ACC_WIDTH-1:0] group_acc;
                        group_acc = '0;
                        for (int p = 0; p < TREE_GROUP_SIZE; p++) begin
                            if (g * TREE_GROUP_SIZE + p < N)
                                group_acc = group_acc + ACC_WIDTH'(feat_in[g * TREE_GROUP_SIZE + p][ch]);
                        end
                        group_sums[g][ch] <= group_acc;
                    end
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Stages 1..TREE_DEPTH: Sequential tree fold
    //   Each cycle halves the number of active partial sums by pairwise adding.
    //   Results are written back to the lower half of work_sums.
    //   After TREE_DEPTH levels, work_sums[0] holds the total sum.
    // -------------------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] work_sums [0:NUM_GROUPS-1][0:4*C-1];
    logic [$clog2(NUM_GROUPS+1)-1:0] active_count;
    logic                            reducing;

    // verilator lint_off CMPCONST
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int g = 0; g < NUM_GROUPS; g++)
                for (int ch = 0; ch < 4*C; ch++)
                    work_sums[g][ch] <= '0;
            active_count <= '0;
            reducing     <= 1'b0;
            valid_out    <= 1'b0;
            for (int ch = 0; ch < 4*C; ch++)
                feat_out[ch] <= '0;
        end else begin
            valid_out <= 1'b0;

            if (stage0_valid && !reducing) begin
                // Load group sums into working array
                for (int g = 0; g < NUM_GROUPS; g++)
                    for (int ch = 0; ch < 4*C; ch++)
                        work_sums[g][ch] <= group_sums[g][ch];
                active_count <= $clog2(NUM_GROUPS+1)'(NUM_GROUPS);
                reducing     <= 1'b1;
            end else if (reducing && active_count > 1) begin
                // Fold: pairwise add, write results to lower half
                /* verilator lint_off WIDTHTRUNC */
                for (int g = 0; g < (NUM_GROUPS+1)/2; g++) begin
                    for (int ch = 0; ch < 4*C; ch++) begin
                        if (2*g + 1 < active_count)
                            work_sums[g][ch] <= work_sums[2*g][ch] + work_sums[2*g+1][ch];
                        else if (2*g < active_count)
                            work_sums[g][ch] <= work_sums[2*g][ch];
                    end
                end
                active_count <= (active_count + 1) / 2;  // Ceiling division
                /* verilator lint_on WIDTHTRUNC */
            end else if (reducing && active_count <= 1) begin
                // Final sum is in work_sums[0], divide by N and output
                for (int ch = 0; ch < 4*C; ch++) begin
                    if (N == (1 << $clog2(N)))
                        // Power-of-2 N: right shift with rounding
                        feat_out[ch] <= DWIDTH'((work_sums[0][ch] + ACC_WIDTH'(1 <<< ($clog2(N)-1))) >>> $clog2(N));
                    else
                        // Non-power-of-2: approximate division
                        feat_out[ch] <= DWIDTH'((work_sums[0][ch] + ACC_WIDTH'(N/2)) / ACC_WIDTH'(N));
                end
                valid_out  <= 1'b1;
                reducing   <= 1'b0;
                active_count <= '0;
            end
        end
    end
    // verilator lint_on CMPCONST

    // Assertions
    // synthesis translate_off
    initial begin
        if (N == 0)
            $error("avgpool3d_unroll: N must be > 0 (N=%0d)", N);
        if (TREE_GROUP_SIZE == 0)
            $error("avgpool3d_unroll: TREE_GROUP_SIZE must be > 0 (TREE_GROUP_SIZE=%0d)", TREE_GROUP_SIZE);
        if (N % TREE_GROUP_SIZE != 0)
            $warning("avgpool3d_unroll: N is not divisible by TREE_GROUP_SIZE (N=%0d, TREE_GROUP_SIZE=%0d) — partial last group handled automatically", N, TREE_GROUP_SIZE);
        if ((TREE_GROUP_SIZE & (TREE_GROUP_SIZE - 1)) != 0)
            $warning("avgpool3d_unroll: TREE_GROUP_SIZE should be power of 2 for optimal synthesis");
    end
    // synthesis translate_on

endmodule
