// =============================================================================
// linear_layer.sv
//
// Fully-connected linear layer: y = W * x + b
//
// Parameters:
//   DATA_WIDTH   : Total bit width (integer + fractional) — from pkg_config
//   FRAC_WIDTH   : Number of fractional bits — from pkg_config
//   IN_CH        : Number of input channels
//   OUT_CH       : Number of output channels
//   STUB_BEHAVIOR: 0 = use ROM files, 1 = stub behavior (sum inputs)
//   WEIGHT_FILE  : Path to weight ROM file (hex values, row-major flattened)
//   BIAS_FILE    : Path to bias ROM file (hex values)
//
// Inputs:
//   clk, rst_n : Clock and active-low reset
//   in_valid   : Assert for one cycle when input vector is valid
//   data_in    : Input vector [0:IN_CH-1] (signed fixed-point)
//
// Outputs:
//   out_valid  : Assert for one cycle when output vector is valid
//   data_out   : Output vector [0:OUT_CH-1] (signed fixed-point)
//
// Pipeline latency: IN_CH + 3 cycles
//   Stage 0: Register input vector
//   Stage 1..IN_CH: Sequential dot product (one input channel per cycle)
//   Stage IN_CH+1: Add bias, round/saturate
//   Stage IN_CH+2: Register output
//
// Handshake:
//   No backpressure (ready) - assumes consumer can accept data every IN_CH+3 cycles.
//   in_valid must be held until computation starts (one cycle sufficient).
// =============================================================================

module linear_layer #(
    parameter int DATA_WIDTH = pkg_module_config::LINEAR_LAYER_DATA_WIDTH,
    parameter int FRAC_WIDTH = pkg_config::FRAC_WIDTH,
    parameter int IN_CH      = 256,
    parameter int OUT_CH     = 192,
    parameter bit STUB_BEHAVIOR = 1'b0,
    parameter int WEIGHT_LOAD_MODE = pkg_config::WEIGHT_LOAD_MODE,
    parameter string WEIGHT_FILE = "",
    parameter string BIAS_FILE   = ""
)(
    input  logic clk,
    input  logic rst_n,

    // Handshake
    input  logic                       in_valid,
    output logic                       out_valid,

    // Data
    input  logic signed [DATA_WIDTH-1:0] data_in [0:IN_CH-1],
    output logic signed [DATA_WIDTH-1:0] data_out[0:OUT_CH-1],

    // Runtime weight loading (active when WEIGHT_LOAD_MODE=0)
    input  logic                       weight_valid,
    input  logic [31:0]                weight_addr,
    input  logic signed [DATA_WIDTH-1:0] weight_data
);

    // -------------------------------------------------------------------------
    // Local parameters
    // -------------------------------------------------------------------------
    localparam int ACC_WIDTH = DATA_WIDTH * 2 + $clog2(IN_CH);   // accumulator width
    localparam int CNT_WIDTH = $clog2(IN_CH);
    localparam int WEIGHT_SIZE = OUT_CH * IN_CH;

    // Rounding constant: 0.5 ULP at FRAC_WIDTH position before truncation
    localparam logic signed [ACC_WIDTH-1:0] ROUND_CONST =
        {{(ACC_WIDTH-FRAC_WIDTH){1'b0}}, 1'b1, {(FRAC_WIDTH-1){1'b0}}};

    // Signed saturation thresholds (fix LIN-SAT-003)
    // Before truncation by FRAC_WIDTH, the max representable output is:
    //   ((1 << (DATA_WIDTH-1)) - 1) << FRAC_WIDTH
    localparam logic signed [ACC_WIDTH-1:0] POS_SAT =
        (ACC_WIDTH'((1 <<< (DATA_WIDTH - 1)) - 1)) <<< FRAC_WIDTH;
    localparam logic signed [ACC_WIDTH-1:0] NEG_SAT =
        -(ACC_WIDTH'(1 <<< (DATA_WIDTH - 1))) <<< FRAC_WIDTH;

    // -------------------------------------------------------------------------
    // Weight and Bias ROMs (parameterized by WEIGHT_LOAD_MODE)
    // -------------------------------------------------------------------------
    (* ram_style = "block" *) logic signed [DATA_WIDTH-1:0] weight_rom [0:WEIGHT_SIZE-1];
    (* ram_style = "block" *) logic signed [DATA_WIDTH-1:0] bias_rom   [0:OUT_CH-1];

    generate
    if (WEIGHT_LOAD_MODE == 1) begin : gen_mem_weights
        // MEM mode: weights initialized from .mem files
        initial begin
            if (STUB_BEHAVIOR) begin
                for (int o = 0; o < OUT_CH; o++) begin
                    for (int i = 0; i < IN_CH; i++)
                        weight_rom[o * IN_CH + i] = {DATA_WIDTH{1'b1}};
                    bias_rom[o] = '0;
                end
            end else begin
                if (WEIGHT_FILE != "") $readmemh(WEIGHT_FILE, weight_rom);
                else for (int i = 0; i < WEIGHT_SIZE; i++) weight_rom[i] = '0;
                if (BIAS_FILE != "") $readmemh(BIAS_FILE, bias_rom);
                else for (int i = 0; i < OUT_CH; i++) bias_rom[i] = '0;
            end
        end
    end else begin : gen_runtime_weights
        // RUNTIME mode: weights loaded via write ports
        initial begin
            if (STUB_BEHAVIOR) begin
                for (int o = 0; o < OUT_CH; o++) begin
                    for (int i = 0; i < IN_CH; i++)
                        weight_rom[o * IN_CH + i] = {DATA_WIDTH{1'b1}};
                    bias_rom[o] = '0;
                end
            end else begin
                for (int i = 0; i < WEIGHT_SIZE; i++) weight_rom[i] = '0;
                for (int i = 0; i < OUT_CH; i++) bias_rom[i] = '0;
            end
        end

        // Runtime weight write port
        always_ff @(posedge clk) begin
            if (weight_valid) begin
                if (weight_addr < WEIGHT_SIZE)
                    weight_rom[weight_addr] <= weight_data;
                else if (weight_addr[31] == 1'b0 && (weight_addr - WEIGHT_SIZE) < OUT_CH)
                    bias_rom[weight_addr - WEIGHT_SIZE] <= weight_data;
            end
        end
    end
    endgenerate

    // -------------------------------------------------------------------------
    // Pipeline control
    // -------------------------------------------------------------------------
    logic [CNT_WIDTH-1:0] cnt;
    logic cnt_en, cnt_done;
    logic running;

    assign cnt_done = (cnt == IN_CH-1);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running <= 1'b0;
            cnt <= '0;
        end else begin
            if (in_valid && !running) begin
                running <= 1'b1;
                cnt <= '0;
            end else if (running) begin
                if (cnt_done) begin
                    running <= 1'b0;
                    cnt <= '0;
                end else begin
                    cnt <= cnt + 1'b1;
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Input registers
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] data_in_reg[0:IN_CH-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < IN_CH; i++) data_in_reg[i] <= '0;
        end else if (in_valid) begin
            data_in_reg <= data_in;
        end
    end

    // -------------------------------------------------------------------------
    // Accumulators
    // -------------------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] acc[0:OUT_CH-1];

    // Combinational MAC computation (extracted from always_ff for Verilator compat)
    logic signed [ACC_WIDTH-1:0] acc_update[0:OUT_CH-1];
    always_comb begin
        for (int o = 0; o < OUT_CH; o++) begin
            automatic logic signed [2*DATA_WIDTH-1:0] prod = weight_rom[o * IN_CH + cnt] * data_in_reg[cnt];
            acc_update[o] = acc[o] + ACC_WIDTH'(prod);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int o = 0; o < OUT_CH; o++) acc[o] <= '0;
        end else begin
            if (in_valid && !running) begin
                // start accumulation
                for (int o = 0; o < OUT_CH; o++) begin
                    if (STUB_BEHAVIOR) begin
                        acc[o] <= '0; // stub: no bias
                    end else begin
                        acc[o] <= ACC_WIDTH'(bias_rom[o]) <<< FRAC_WIDTH; // bias scaled to accumulator format
                    end
                end
            end else if (running) begin
                for (int o = 0; o < OUT_CH; o++) begin
                    acc[o] <= acc_update[o];
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Output stage: round and saturate to DATA_WIDTH
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] out_reg[0:OUT_CH-1];
    logic out_valid_reg;

    // Combinational round/saturate (extracted from always_ff for Verilator compat)
    logic signed [DATA_WIDTH-1:0] out_comb[0:OUT_CH-1];
    always_comb begin
        for (int o = 0; o < OUT_CH; o++) begin
            automatic logic signed [ACC_WIDTH-1:0] round = acc[o] + ROUND_CONST;
            automatic logic signed [DATA_WIDTH-1:0] shift = round[ACC_WIDTH-1:FRAC_WIDTH];
            if (round > POS_SAT) begin
                out_comb[o] = (DATA_WIDTH-1)'(1) - 1;
            end else if (round < NEG_SAT) begin
                out_comb[o] = {(DATA_WIDTH){1'b1}};
            end else begin
                out_comb[o] = shift;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid_reg <= 1'b0;
            for (int o = 0; o < OUT_CH; o++) out_reg[o] <= '0;
        end else begin
            out_valid_reg <= running && cnt_done;
            if (running && cnt_done) begin
                for (int o = 0; o < OUT_CH; o++) begin
                    out_reg[o] <= out_comb[o];
                end
            end
        end
    end

    assign out_valid = out_valid_reg;
    assign data_out = out_reg;

endmodule