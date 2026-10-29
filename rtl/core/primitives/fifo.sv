// =============================================================================
// fifo.sv
//
// Runtime-Configurable FIFO with proper BRAM inference.
// Supports both show-ahead (FWFT) and standard modes.
//
// Features:
//   - Runtime-configurable depth via parameter
//   - Proper BRAM inference with (* ram_style = "block" *)
//   - Show-ahead (FWFT) mode for lower read latency
//   - Overflow/underflow protection
//   - Full/empty status with almost thresholds
//   - Parameterized data width
//   - Pipeline output option for better timing
//
// Usage:
//   - Standard mode: data appears after rd_en, read latency = 1 cycle
//   - Show-ahead mode: data appears immediately, read latency = 0 cycles
//
// BRAM Mapping:
//   - Uses block RAM for DEPTH >= 16
//   - Uses distributed RAM for DEPTH < 16
// =============================================================================

/* verilator lint_off DECLFILENAME */

module fifo #(
    parameter int WIDTH          = 32,
    parameter int DEPTH          = 8,
    parameter bit SHOW_AHEAD     = 0,    // First-Word Fall-Through mode
    parameter bit PIPELINE_DOUT  = 0     // Additional output register
)(
    input  logic clk,
    input  logic rst_n,

    // Write side
    input  logic             wr_en,
    input  logic [WIDTH-1:0] din,
    output logic             full,
    output logic             almost_full,

    // Read side
    input  logic             rd_en,
    output logic [WIDTH-1:0] dout,
    output logic             empty,
    output logic             almost_empty
);

    // =========================================================================
    // Derived Parameters
    // =========================================================================
    localparam int ADDR_W = $clog2(DEPTH);
    localparam int PTR_W  = ADDR_W + 1;  // Extra bit for full detection

    // =========================================================================
    // Memory Storage
    // =========================================================================
    // Synthesis tools auto-infer block RAM for DEPTH >= 16, LUT RAM otherwise.
    logic [WIDTH-1:0] mem [DEPTH-1:0];

    // =========================================================================
    // Pointer Logic
    // =========================================================================
    logic [PTR_W-1:0] wr_ptr_bin;
    logic [PTR_W-1:0] rd_ptr_bin;

    // =========================================================================
    // Write Logic
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr_bin  <= '0;
        end else if (wr_en && !full) begin
            mem[wr_ptr_bin[ADDR_W-1:0]] <= din;
            wr_ptr_bin <= wr_ptr_bin + 1'b1;
        end
    end

    // =========================================================================
    // Read Logic
    // =========================================================================
    logic [WIDTH-1:0] dout_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr_bin  <= '0;
            dout_reg    <= '0;
        end else if (rd_en && !empty) begin
            dout_reg <= mem[rd_ptr_bin[ADDR_W-1:0]];
            rd_ptr_bin <= rd_ptr_bin + 1'b1;
        end
    end

    // Show-ahead mode: output data immediately on empty check
    logic [WIDTH-1:0] dout_shownext;
    generate
        if (SHOW_AHEAD) begin : gen_show_ahead
            // First-Word Fall-Through: data available immediately
            logic [WIDTH-1:0] peek_data;
            logic peek_valid;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    peek_data  <= '0;
                    peek_valid <= 1'b0;
                end else begin
                    if (empty && !(wr_en && !full)) begin
                        // FIFO empty and no simultaneous write — invalidate output
                        peek_valid <= 1'b0;
                    end else if (rd_en && peek_valid) begin
                        // Consumer read current item — load next.
                        // Only keep peek_valid if there will be data after the read.
                        if (count_bin > 1) begin
                            peek_data  <= mem[(rd_ptr_bin[ADDR_W-1:0] + 1'b1)];
                            peek_valid <= 1'b1;
                        end else if (wr_en && !full) begin
                            // Simultaneously writing while reading last item
                            peek_data  <= din;
                            peek_valid <= 1'b1;
                        end else begin
                            peek_valid <= 1'b0;
                        end
                    end else if (!peek_valid && !empty) begin
                        // FIFO has data, nothing peeked — load head item
                        peek_data  <= mem[rd_ptr_bin[ADDR_W-1:0]];
                        peek_valid <= 1'b1;
                    end
                end
            end

            assign dout_shownext = peek_valid ? peek_data : '0;
        end else begin : gen_standard
            // Standard mode: output registered data
            assign dout_shownext = dout_reg;
        end
    endgenerate

    // Optional pipeline register for timing
    generate
        if (PIPELINE_DOUT) begin : gen_pipeline
            logic [WIDTH-1:0] dout_pipeline;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    dout_pipeline <= '0;
                else
                    dout_pipeline <= dout_shownext;
            end
            assign dout = dout_pipeline;
        end else begin : gen_no_pipeline
            assign dout = dout_shownext;
        end
    endgenerate

    // =========================================================================
    // Status Logic
    // =========================================================================
    assign empty = (wr_ptr_bin == rd_ptr_bin);

    assign full  = (wr_ptr_bin[PTR_W-1]     != rd_ptr_bin[PTR_W-1]) &&
                   (wr_ptr_bin[ADDR_W-1:0]  == rd_ptr_bin[ADDR_W-1:0]);

    // Almost thresholds
    logic [PTR_W-1:0] count_bin;
    assign count_bin = wr_ptr_bin - rd_ptr_bin;

    assign almost_full  = (32'(count_bin) >= (DEPTH - 2));
    assign almost_empty = (count_bin <= PTR_W'(2));

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    /* verilator lint_off SYNCASYNCNET */
    FIFO_DEPTH_VALID: assert property (
        @(posedge clk) disable iff (!rst_n)
        DEPTH >= 2 && DEPTH <= 65536
    ) else $error("fifo: DEPTH must be between 2 and 65536");
    /* verilator lint_on SYNCASYNCNET */
    // synthesis translate_on

endmodule

// =============================================================================
// fifo_param - Parameterized FIFO with CSR depth configuration
// =============================================================================

module fifo_param #(
    parameter int WIDTH       = 32,
    parameter int MAX_DEPTH   = 64,
    parameter int ADDR_W      = $clog2(MAX_DEPTH)
)(
    input  logic clk,
    input  logic rst_n,

    // CSR Configuration
    input  logic [ADDR_W:0] cfg_depth,     // Runtime depth (2 to MAX_DEPTH)
    input  logic            cfg_valid,    // Configuration valid

    // Write side
    input  logic             wr_en,
    input  logic [WIDTH-1:0] din,
    output logic             full,
    output logic             almost_full,

    // Read side
    input  logic             rd_en,
    output logic [WIDTH-1:0] dout,
    output logic             empty,
    output logic             almost_empty
);

    // =========================================================================
    // Memory Storage
    // =========================================================================
    (* ram_style = "block" *)
    logic [WIDTH-1:0] mem [MAX_DEPTH-1:0];

    // =========================================================================
    // Pointers and State
    // =========================================================================
    logic [ADDR_W:0] wr_ptr;
    logic [ADDR_W:0] rd_ptr;
    logic [ADDR_W:0] count;

    // Runtime depth tracking
    logic [ADDR_W:0] effective_depth;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            effective_depth <= MAX_DEPTH[ADDR_W:0];
        end else if (cfg_valid) begin
            if (cfg_depth >= 2 && cfg_depth <= MAX_DEPTH[ADDR_W:0])
                effective_depth <= cfg_depth;
            else
                effective_depth <= MAX_DEPTH[ADDR_W:0];
        end
    end

    // =========================================================================
    // Control Logic
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
            count  <= '0;
        end else begin
            // Pointer management with non-power-of-two wrap
            if (wr_en && !full) begin
                mem[wr_ptr[ADDR_W-1:0]] <= din;
                if (wr_ptr == effective_depth - 1'b1)
                    wr_ptr <= '0;
                else
                    wr_ptr <= wr_ptr + 1'b1;
            end

            if (rd_en && !empty) begin
                if (rd_ptr == effective_depth - 1'b1)
                    rd_ptr <= '0;
                else
                    rd_ptr <= rd_ptr + 1'b1;
            end

            // Count management
            case ({wr_en && !full, rd_en && !empty})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end

    // Output data
    logic [WIDTH-1:0] dout_reg;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            dout_reg <= '0;
        else if (rd_en && !empty)
            dout_reg <= mem[rd_ptr[ADDR_W-1:0]];
    end

    assign dout = dout_reg;

    // =========================================================================
    // Status Logic
    // =========================================================================
    assign full         = (count == effective_depth);
    assign empty        = (count == '0);
    assign almost_full  = (count >= effective_depth - (ADDR_W+1)'(2));
    assign almost_empty = (count <= (ADDR_W+1)'(2));

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    /* verilator lint_off SYNCASYNCNET */
    FIFO_PARAM_DEPTH: assert property (
        @(posedge clk) disable iff (!rst_n)
        effective_depth >= 2
    ) else $error("fifo_param: Invalid depth configuration");
    /* verilator lint_on SYNCASYNCNET */
    // synthesis translate_on

endmodule
