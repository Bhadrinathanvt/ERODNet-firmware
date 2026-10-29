// =============================================================================
//  sigmoid_linear_prehead.sv
//
//  Pre-detection-head feature refinement: Sigmoid (LUT-based) → Linear (1×1).
//
//  Replaces the crude approx_sigmoid in final_upsample/detection_head with a
//  proper ROM-interpolated sigmoid, followed by a learnable pointwise linear
//  projection.  No GroupNorm, no upsampling (FinalUp) — just Sigmoid + Linear.
//
//  Width conversion: input at DATA_W is sign-extended to SIG_W (Q4.16 = 20 bits)
//  for the sigmoid ROM lookup (which extracts bits [19:12] for the LUT index).
//  SIG_W is sized to the minimum that preserves the [INDEX_START_BIT+INDEX_BITS-1
//  : INDEX_START_BIT] = [19:12] index window — see Phase 5I LUT-reduction notes.
//  Sigmoid saturates outside |x|>~6, well within the Q4.16 range [-8, +8).
//  The sigmoid output is truncated back to DATA_W for the linear stage.
//
//  Pipeline:
//    Stage 1: Input capture + sign-extend to SIG_W (1 cycle)
//    Stage 2: Sigmoid — C parallel lanes with shared ROMs at SIG_W (4-cycle latency)
//    Stage 3: Truncate sigmoid output back to DATA_W
//    Stage 4: Linear — sequential MAC, PARA_CH lanes (CHANNELS/PARA_CH + 3 cycles)
//    Stage 5: Output hold register with valid/ready handshake
//
//  Total latency per token: 4 + CHANNELS/PARA_CH + 6 cycles
//  Throughput: 1 token per CHANNELS/PARA_CH + 10 cycles (no overlap)
//
//  Weight loading:
//    - Linear weights: weight_valid/weight_addr/weight_data
//      Address space: 0..CHANNELS*OUT_CH-1  (row-major: out_idx*CHANNELS + in_idx)
//    - Linear biases:  address CHANNELS*OUT_CH..CHANNELS*OUT_CH+OUT_CH-1
// =============================================================================

import pkg_config::*;

module sigmoid_linear_prehead #(
    parameter int CHANNELS    = MAX_C,
    parameter int OUT_CH      = MAX_C,
    parameter int DATA_W      = DATA_WIDTH,
    parameter int FRAC_W      = FRAC_WIDTH,
    parameter int INDEX_BITS  = pkg_config::INDEX_BITS,
    parameter int ROM_DATA_BITS = pkg_config::ROM_DATA_BITS,
    parameter int PARA_CH     = pkg_config::PREHEAD_PARA_CH,
    parameter int WEIGHT_LOAD_MODE = pkg_config::WEIGHT_LOAD_MODE
)(
    input  logic clk,
    input  logic rst_n,

    // ── Input from upstream (GFF / decoder output) ────────────────────────────
    input  logic                        valid_in,
    output logic                        ready_in,
    input  logic signed [DATA_W-1:0]    data_in  [CHANNELS-1:0],
    input  logic [15:0]                 h_idx_in,
    input  logic [15:0]                 w_idx_in,
    input  logic                        last_in,

    // ── Output to downstream (detection_head) ───────────────────────────────────
    output logic                        valid_out,
    input  logic                        out_ready,
    output logic signed [DATA_W-1:0]    data_out [OUT_CH-1:0],
    output logic [15:0]                 h_idx_out,
    output logic [15:0]                 w_idx_out,
    output logic                        last_out,

    // ── Weight loading (Linear: CHANNELS × OUT_CH + OUT_CH biases) ────────────
    input  logic                        weight_valid,
    input  logic [31:0]                 weight_addr,
    input  logic signed [DATA_W-1:0]    weight_data
);

    // SIG_FRAC_W / SIG_INT_W / SIG_W come from pkg_config (config.yaml ::
    // prehead_widths). Default Q4.16 = 20 bits — the minimum width that
    // preserves the [INDEX_START_BIT+INDEX_BITS-1:INDEX_START_BIT] = [19:12]
    // ROM index window. (Phase 5I LUT reduction: 32→20.)
    localparam int SIG_FRAC_W  = pkg_config::SIGMOID_INPUT_FRAC_W;
    localparam int SIG_INT_W   = pkg_config::SIGMOID_INPUT_INT_W;
    localparam int SIG_W       = pkg_config::SIGMOID_INPUT_WIDTH;
    localparam int ACC_W       = 2*DATA_W + $clog2(CHANNELS);
    localparam int MAC_CYCLES  = (CHANNELS + PARA_CH - 1) / PARA_CH;
    localparam int WEIGHT_SIZE = CHANNELS * OUT_CH;

    // =========================================================================
    // FSM
    // =========================================================================
    typedef enum logic [2:0] {
        PH_IDLE,        // Wait for valid_in
        PH_SIGMOID,     // Sigmoid lanes processing (4-cycle pipeline)
        PH_LINEAR,      // Sequential MAC for linear projection
        PH_ROUND,       // Round/saturate accumulator → register output
        PH_OUTPUT       // Hold output until consumer accepts
    } state_t;
    state_t state;

    // =========================================================================
    // Stage 1: Input capture + sign-extend to SIG_W
    // =========================================================================
    logic signed [DATA_W-1:0] feat_reg [CHANNELS-1:0];
    logic signed [SIG_W-1:0]  feat_ext [CHANNELS-1:0];
    logic [15:0] h_idx_reg, w_idx_reg;
    logic        last_reg;

    assign ready_in = (state == PH_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int c = 0; c < CHANNELS; c++) feat_reg[c] <= '0;
            h_idx_reg <= '0;
            w_idx_reg <= '0;
            last_reg  <= 1'b0;
        end else if (valid_in && ready_in) begin
            for (int c = 0; c < CHANNELS; c++)
                feat_reg[c] <= data_in[c];
            h_idx_reg <= h_idx_in;
            w_idx_reg <= w_idx_in;
            last_reg  <= last_in;
        end
    end

    // Sign-extend DATA_W → SIG_W for sigmoid ROM lookup
    always_comb begin
        for (int c = 0; c < CHANNELS; c++)
            feat_ext[c] = SIG_W'(signed'(feat_reg[c])) << (SIG_FRAC_W - FRAC_W);
    end

    // =========================================================================
    // Stage 2: Sigmoid — C parallel lanes with shared ROMs at SIG_W
    // =========================================================================
    logic signed [SIG_W-1:0]  sig_out       [CHANNELS-1:0];
    logic signed [DATA_W-1:0] sig_out_trunc  [CHANNELS-1:0];
    logic        sig_valid;
    logic        sig_out_ready;
    logic [4:0]  sig_cnt;
    logic        sig_active;

    // Shared sigmoid ROMs emit Q16.16 values at pkg_config::ROM_WIDTH.
    // SIG_W is the *input* width to the ROM index extractor; the output
    // y_base/slope are always ROM_WIDTH wide regardless of input.
    localparam int ROM_W = pkg_config::ROM_WIDTH;
    logic signed [ROM_W-1:0] shared_sig_y_base [CHANNELS-1:0];
    logic signed [ROM_W-1:0] shared_sig_slope  [CHANNELS-1:0];

    shared_activation_roms #(
        .DATA_WIDTH    (SIG_W),
        .FRAC_WIDTH    (SIG_FRAC_W),
        .INDEX_BITS    (INDEX_BITS),
        .NUM_PORTS     (CHANNELS),
        // ROM_DATA_BITS defaults to ROM_WIDTH (32) — required for the .mem
        // files. Earlier we passed pkg_config::ROM_DATA_BITS (6) which
        // truncated all ROM values to zero.
        .EN_SIGMOID    (1),
        .EN_GELU       (0)
    ) u_sig_roms (
        .clk           (clk),
        .rst_n         (rst_n),
        .x_in          (feat_ext),
        .sigmoid_y_base(shared_sig_y_base),
        .sigmoid_slope (shared_sig_slope),
        .gelu_y_base   (),
        .gelu_slope    ()
    );

    // C parallel sigmoid lanes
    logic [CHANNELS-1:0] sig_lane_valid;

    generate
        for (genvar i = 0; i < CHANNELS; i++) begin : gen_sig_lane
            unified_activation_lane #(
                .DATA_WIDTH   (SIG_W),
                .FRAC_WIDTH   (SIG_FRAC_W),
                .EN_SIGMOID   (1),
                .EN_GELU      (0),
                .EN_STAR_RELU (0),
                .USE_DSP      (0)
            ) u_sig_lane (
                .clk             (clk),
                .rst_n           (rst_n),
                // Pulse for one cycle on entry to PH_SIGMOID. (Earlier code
                // gated this with sig_active, but sig_active was registered to
                // 1 only on the *second* PH_SIGMOID cycle — after sig_cnt had
                // already incremented past 0 — so the lane never saw a valid
                // pulse. Fixed: sample on first PH_SIGMOID cycle.)
                .in_valid        (state == PH_SIGMOID && sig_cnt == 0),
                .ready_in        (),
                .out_valid       (sig_lane_valid[i]),
                .out_ready       (sig_out_ready),
                .act_enable      (1'b1),
                .act_sel         (3'd3),           // Sigmoid
                .leak_shift      ('0),
                .saturation_in   ('0),
                .scale_in        ('0),
                .bias_in         ('0),
                .x_in            (feat_ext[i]),
                .y_out           (sig_out[i]),
                .sigmoid_y_base_in(shared_sig_y_base[i]),
                .sigmoid_slope_in (shared_sig_slope[i]),
                .gelu_y_base_in   ('0),
                .gelu_slope_in    ('0)
            );
        end
    endgenerate

    assign sig_valid = &sig_lane_valid;

    // Truncate sigmoid output from SIG_W back to DATA_W
    always_comb begin
        for (int c = 0; c < CHANNELS; c++)
            sig_out_trunc[c] = DATA_W'(sig_out[c] >>> SIG_FRAC_W);
    end

    // Sigmoid pipeline counter: pulse in_valid for 1 cycle, then wait for output
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sig_cnt    <= 5'd0;
            sig_active <= 1'b0;
        end else begin
            case (state)
                PH_IDLE: begin
                    sig_active <= 1'b0;
                    sig_cnt    <= 5'd0;
                end
                PH_SIGMOID: begin
                    sig_active <= 1'b1;
                    if (sig_cnt < 5'd8)
                        sig_cnt <= sig_cnt + 1'b1;
                end
                default: begin
                    sig_active <= 1'b0;
                end
            endcase
        end
    end

    // Capture truncated sigmoid output once all lanes are valid
    logic signed [DATA_W-1:0] sig_buf [CHANNELS-1:0];
    logic                     sig_captured;

    assign sig_out_ready = ~sig_captured;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sig_captured <= 1'b0;
            for (int c = 0; c < CHANNELS; c++) sig_buf[c] <= '0;
        end else if (sig_valid && !sig_captured) begin
            for (int c = 0; c < CHANNELS; c++)
                sig_buf[c] <= sig_out_trunc[c];
            sig_captured <= 1'b1;
        end else if (state == PH_LINEAR) begin
            sig_captured <= 1'b0;
        end
    end

    // =========================================================================
    // Stage 3: Linear — Sequential MAC with PARA_CH parallelism
    // =========================================================================
    (* ram_style = "distributed" *)
    logic signed [DATA_W-1:0] pw_weights [0:CHANNELS-1][0:OUT_CH-1];
    logic signed [DATA_W-1:0] bias_rom   [0:OUT_CH-1];

    generate
    if (WEIGHT_LOAD_MODE == 1) begin : gen_mem_linear_weights
        initial begin
            for (int i = 0; i < CHANNELS; i++)
                for (int o = 0; o < OUT_CH; o++)
                    pw_weights[i][o] = '0;
            for (int o = 0; o < OUT_CH; o++)
                bias_rom[o] = '0;
        end
    end else begin : gen_runtime_linear_weights
        initial begin
            for (int i = 0; i < CHANNELS; i++)
                for (int o = 0; o < OUT_CH; o++)
                    pw_weights[i][o] = '0;
            for (int o = 0; o < OUT_CH; o++)
                bias_rom[o] = '0;
        end

        always_ff @(posedge clk) begin
            if (weight_valid) begin
                if (weight_addr < WEIGHT_SIZE) begin
                    for (int c = 0; c < CHANNELS; c++) begin
                        if (weight_addr[$clog2(CHANNELS)-1:0] ==
                            c[$clog2(CHANNELS)-1:0])
                            pw_weights[c][weight_addr >>> $clog2(CHANNELS)] <= weight_data;
                    end
                end else if (weight_addr < WEIGHT_SIZE + OUT_CH) begin
                    bias_rom[weight_addr - WEIGHT_SIZE] <= weight_data;
                end
            end
        end
    end
    endgenerate

    logic [$clog2(MAC_CYCLES):0] mac_cycle;
    logic signed [ACC_W-1:0] accum [OUT_CH-1:0];
    logic signed [ACC_W-1:0] mac_result [OUT_CH-1:0];

    // =========================================================================
    // FSM + MAC + output
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= PH_IDLE;
            valid_out <= 1'b0;
            mac_cycle <= '0;
            h_idx_out <= '0;
            w_idx_out <= '0;
            last_out  <= 1'b0;
            for (int o = 0; o < OUT_CH; o++) begin
                accum[o]    <= '0;
                mac_result[o] <= '0;
                data_out[o] <= '0;
            end
        end else begin
            valid_out <= 1'b0;

            case (state)
                // ── Wait for input token ──────────────────────────────────
                PH_IDLE: begin
                    if (valid_in) begin
                        state <= PH_SIGMOID;
                    end
                end

                // ── Wait for sigmoid pipeline to complete ──────────────────
                PH_SIGMOID: begin
                    if (sig_captured) begin
                        // Start linear MAC
                        mac_cycle <= '0;
                        for (int o = 0; o < OUT_CH; o++)
                            accum[o] <= ACC_W'(bias_rom[o]) <<< FRAC_W;
                        state <= PH_LINEAR;
                    end
                end

                // ── Sequential MAC: PARA_CH channels per cycle ────────────
                PH_LINEAR: begin
                    for (int o = 0; o < OUT_CH; o++) begin
                        automatic logic signed [ACC_W-1:0] acc = accum[o];
                        for (int p = 0; p < PARA_CH; p++) begin
                            automatic int idx = int'(mac_cycle) * PARA_CH + p;
                            if (idx < CHANNELS) begin
                                acc = acc + ACC_W'(sig_buf[idx] * pw_weights[idx][o]);
                            end
                        end
                        accum[o] <= acc;
                    end

                    if (mac_cycle + 1 >= MAC_CYCLES) begin
                        for (int o = 0; o < OUT_CH; o++)
                            mac_result[o] <= (accum[o] + (1 <<< (FRAC_W - 1))) >>> FRAC_W;
                        state <= PH_ROUND;
                    end else begin
                        mac_cycle <= mac_cycle + 1'b1;
                    end
                end

                // ── Round / saturate / register output ─────────────────────
                PH_ROUND: begin
                    for (int o = 0; o < OUT_CH; o++) begin
                        automatic logic signed [ACC_W-1:0] x = mac_result[o];
                        if (x > (1 <<< (DATA_W - 1)) - 1)
                            data_out[o] <= (DATA_W-1)'(1) - 1;
                        else if (x < -(1 <<< (DATA_W - 1)))
                            data_out[o] <= {1'b1, {(DATA_W-1){1'b0}}};
                        else
                            data_out[o] <= DATA_W'(x);
                    end
                    h_idx_out <= h_idx_reg;
                    w_idx_out <= w_idx_reg;
                    last_out  <= last_reg;
                    valid_out <= 1'b1;
                    state     <= PH_OUTPUT;
                end

                // ── Hold output until consumer accepts ─────────────────────
                PH_OUTPUT: begin
                    valid_out <= 1'b1;
                    if (out_ready) begin
                        valid_out <= 1'b0;
                        state     <= PH_IDLE;
                    end
                end

                default: state <= PH_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    PREHEAD_VALID_STABLE: assert property (
        @(posedge clk) disable iff (!rst_n)
        (valid_out && !out_ready) |=> valid_out
    ) else $error("sigmoid_linear_prehead: valid_out dropped before out_ready");

    PREHEAD_NO_OVERRUN: assert property (
        @(posedge clk) disable iff (!rst_n)
        (state != PH_IDLE) |-> !ready_in
    ) else $error("sigmoid_linear_prehead: accepted input while busy");
    // synthesis translate_on

endmodule