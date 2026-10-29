// =============================================================================
// true_dual_port_bram.sv
//
// True Dual-Port Block RAM with runtime-configurable depth.
// Properly infers Xilinx/Intel BRAM with both read/write ports.
//
// Features:
//   - True dual-port: simultaneous read/write on both ports
//   - Runtime-configurable depth (up to MAX_DEPTH)
//   - Synchronous reads with optional output registration
//   - Byte-enable support for partial writes
//   - Pipeline output option for better timing
//   - Initialize from memory file (for ROM mode)
//
// Port A: Read/Write
// Port B: Read/Write (independent clock domain optional)
//
// BRAM Inference:
//   - Xilinx: (* ram_style = "block" *)
//   - Intel: (* ramstyle = "M9K" *) or "M10K"
//   - Proper read-during-write behavior
// =============================================================================

/* verilator lint_off DECLFILENAME */

module true_dual_port_bram #(
    parameter int DATA_W       = 32,
    parameter int ADDR_W       = 10,
    parameter int DEPTH        = 1024,
    parameter int MAX_DEPTH    = 4096,      // Maximum depth (BRAM size)
    parameter int BYTE_W       = 8,         // Byte width for byte-enable
    parameter bit  PIPELINE_OUT = 1,        // Add output register for timing
    parameter bit  INIT_FILE_EN = 0,        // Initialize from file
    parameter string INIT_FILE  = "",       // Memory initialization file
    parameter bit  SAME_CLK     = 0         // 1 when both ports share the same clock
)(
    // Port A (primary read/write)
    input  logic                        clk_a,
    input  logic                        rst_n_a,
    input  logic                        en_a,
    input  logic                        we_a,
    input  logic [ADDR_W-1:0]           addr_a,
    input  logic [DATA_W-1:0]           din_a,
    input  logic [(DATA_W/BYTE_W)-1:0]  be_a,  // Byte enable
    output logic [DATA_W-1:0]           dout_a,

    // Port B (secondary read/write)
    input  logic                        clk_b,
    input  logic                        rst_n_b,
    input  logic                        en_b,
    input  logic                        we_b,
    input  logic [ADDR_W-1:0]           addr_b,
    input  logic [DATA_W-1:0]           din_b,
    input  logic [(DATA_W/BYTE_W)-1:0]  be_b,
    output logic [DATA_W-1:0]           dout_b,

    // Status
    output logic                        collision_a,
    output logic                        collision_b
);

    // =========================================================================
    // Derived Parameters
    // =========================================================================
    localparam int NUM_BYTES = DATA_W / BYTE_W;

    // =========================================================================
    // BRAM Storage
    // =========================================================================
    (* ram_style = "block" *)
    logic [DATA_W-1:0] mem [MAX_DEPTH-1:0];

    // =========================================================================
    // Initialization (for ROM mode)
    // =========================================================================
    // synthesis translate_off
    initial begin
        if (INIT_FILE_EN && INIT_FILE != "") begin
            $readmemh(INIT_FILE, mem, 0, DEPTH-1);
        end else begin
            // Initialize to zero for simulation
            for (int i = 0; i < MAX_DEPTH; i++) begin
                mem[i] = '0;
            end
        end
    end
    // synthesis translate_on

    // =========================================================================
    // Port A: Write Logic with Byte-Enable
    // =========================================================================
    logic [DATA_W-1:0] dout_a_reg;
    logic [DATA_W-1:0] dout_a_pipeline;

    always_ff @(posedge clk_a) begin
        if (en_a) begin
            if (we_a) begin
                // Write with byte-enable
                for (int b = 0; b < NUM_BYTES; b++) begin
                    if (be_a[b]) begin
                        mem[addr_a[$clog2(MAX_DEPTH)-1:0]][b*BYTE_W +: BYTE_W] <= din_a[b*BYTE_W +: BYTE_W];
                    end
                end
            end
            // Synchronous read (inferred BRAM)
            dout_a_reg <= mem[addr_a[$clog2(MAX_DEPTH)-1:0]];
        end
    end

    // Optional pipeline register for timing
    generate
        if (PIPELINE_OUT) begin : gen_pipeline_a
            always_ff @(posedge clk_a or negedge rst_n_a) begin
                if (!rst_n_a)
                    dout_a_pipeline <= '0;
                else
                    dout_a_pipeline <= dout_a_reg;
            end
            assign dout_a = dout_a_pipeline;
        end else begin : gen_no_pipeline_a
            assign dout_a = dout_a_reg;
        end
    endgenerate

    // =========================================================================
    // Port B: Write Logic with Byte-Enable
    // =========================================================================
    logic [DATA_W-1:0] dout_b_reg;
    logic [DATA_W-1:0] dout_b_pipeline;

    always_ff @(posedge clk_b) begin
        if (en_b) begin
            if (we_b) begin
                // Write with byte-enable
                for (int b = 0; b < NUM_BYTES; b++) begin
                    if (be_b[b]) begin
                        mem[addr_b[$clog2(MAX_DEPTH)-1:0]][b*BYTE_W +: BYTE_W] <= din_b[b*BYTE_W +: BYTE_W];
                    end
                end
            end
            // Synchronous read
            dout_b_reg <= mem[addr_b[$clog2(MAX_DEPTH)-1:0]];
        end
    end

    // Optional pipeline register for timing
    generate
        if (PIPELINE_OUT) begin : gen_pipeline_b
            always_ff @(posedge clk_b or negedge rst_n_b) begin
                if (!rst_n_b)
                    dout_b_pipeline <= '0;
                else
                    dout_b_pipeline <= dout_b_reg;
            end
            assign dout_b = dout_b_pipeline;
        end else begin : gen_no_pipeline_b
            assign dout_b = dout_b_reg;
        end
    endgenerate

    // =========================================================================
    // Collision Detection (same-clock domain)
    // =========================================================================
    // Note: Only meaningful when both ports share the same clock domain
    assign collision_a = SAME_CLK && en_a && we_a && en_b && we_b && (addr_a == addr_b);
    assign collision_b = collision_a;

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    /* verilator lint_off SYNCASYNCNET */
    BRAM_DEPTH_VALID: assert property (
        @(posedge clk_a) DEPTH <= MAX_DEPTH
    );

    BRAM_ADDR_VALID_A: assert property (
        @(posedge clk_a) disable iff (!rst_n_a)
        en_a |-> 32'(addr_a) < DEPTH
    ) else $error("true_dual_port_bram: Port A address out of bounds");

    BRAM_ADDR_VALID_B: assert property (
        @(posedge clk_b) disable iff (!rst_n_b)
        en_b |-> 32'(addr_b) < DEPTH
    ) else $error("true_dual_port_bram: Port B address out of bounds");
    /* verilator lint_on SYNCASYNCNET */
    // synthesis translate_on

endmodule


// =============================================================================
// true_dual_port_bram_same_clk - Simplified same-clock variant
// =============================================================================
// Single clock domain, simpler interface

module true_dual_port_bram_same_clk #(
    parameter int DATA_W       = 32,
    parameter int ADDR_W       = 10,
    parameter int DEPTH        = 1024,
    parameter int BYTE_W       = 8,
    parameter bit  PIPELINE_OUT = 1
)(
    input  logic clk,
    input  logic rst_n,

    // Port A
    input  logic                          en_a,
    input  logic                          we_a,
    input  logic [ADDR_W-1:0]             addr_a,
    input  logic [DATA_W-1:0]             din_a,
    input  logic [(DATA_W/BYTE_W)-1:0]    be_a,
    output logic [DATA_W-1:0]             dout_a,

    // Port B
    input  logic                          en_b,
    input  logic                          we_b,
    input  logic [ADDR_W-1:0]             addr_b,
    input  logic [DATA_W-1:0]             din_b,
    input  logic [(DATA_W/BYTE_W)-1:0]    be_b,
    output logic [DATA_W-1:0]             dout_b
);

    // =========================================================================
    // BRAM Storage
    // =========================================================================
    (* ram_style = "block" *)
    logic [DATA_W-1:0] mem [DEPTH-1:0];

    localparam int NUM_BYTES = DATA_W / BYTE_W;

    // =========================================================================
    // Port A
    // =========================================================================
    logic [DATA_W-1:0] dout_a_reg;

    always_ff @(posedge clk) begin
        if (en_a) begin
            if (we_a) begin
                for (int b = 0; b < NUM_BYTES; b++) begin
                    if (be_a[b])
                        mem[addr_a[$clog2(DEPTH)-1:0]][b*BYTE_W +: BYTE_W] <= din_a[b*BYTE_W +: BYTE_W];
                end
            end
            dout_a_reg <= mem[addr_a[$clog2(DEPTH)-1:0]];
        end
    end

    generate
        if (PIPELINE_OUT) begin : gen_pipe_a
            logic [DATA_W-1:0] dout_a_pipe;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    dout_a_pipe <= '0;
                else
                    dout_a_pipe <= dout_a_reg;
            end
            assign dout_a = dout_a_pipe;
        end else begin : gen_no_pipe_a
            assign dout_a = dout_a_reg;
        end
    endgenerate

    // =========================================================================
    // Port B
    // =========================================================================
    logic [DATA_W-1:0] dout_b_reg;

    always_ff @(posedge clk) begin
        if (en_b) begin
            if (we_b) begin
                for (int b = 0; b < NUM_BYTES; b++) begin
                    if (be_b[b])
                        mem[addr_b[$clog2(DEPTH)-1:0]][b*BYTE_W +: BYTE_W] <= din_b[b*BYTE_W +: BYTE_W];
                end
            end
            dout_b_reg <= mem[addr_b[$clog2(DEPTH)-1:0]];
        end
    end

    generate
        if (PIPELINE_OUT) begin : gen_pipe_b
            logic [DATA_W-1:0] dout_b_pipe;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    dout_b_pipe <= '0;
                else
                    dout_b_pipe <= dout_b_reg;
            end
            assign dout_b = dout_b_pipe;
        end else begin : gen_no_pipe_b
            assign dout_b = dout_b_reg;
        end
    endgenerate

endmodule
