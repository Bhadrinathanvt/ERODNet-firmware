// =============================================================================
// final_upsample.sv
//
// Final upsampling layer per E-RODNet paper Eq. 8:
//   FinalUp = Sigmoid(Linear(GN(trilinear)))
//
// Maps decoder output to per-pixel sigmoid scores.
// In our pipeline D1 already includes spatial upsampling (bilinear 2x),
// so this module performs the GN + Linear(1x1) + Sigmoid on full-resolution
// features before the detection head.
//
// Pipeline: Input → GroupNorm → Pointwise Linear (1x1 conv) → Sigmoid → Output
// Throughput: 1 token per ~IN_CHANNELS/PARA_CH + GN latency + 2 cycles
// =============================================================================

import pkg_config::*;

module final_upsample #(
    parameter int CHANNELS    = MAX_C,
    parameter int OUT_CH      = MAX_C,          // Can be NUM_CLASSES*NUM_ANCHORS for paper fidelity
    parameter int DATA_W      = DATA_WIDTH,
    parameter int FRAC_W      = FRAC_WIDTH,
    parameter int GROUPS      = pkg_config::GROUPS,
    parameter int PARA_CH     = pkg_config::PARA_CH,   // Parallel MAC lanes (must divide CHANNELS)
    parameter int WEIGHT_LOAD_MODE = pkg_config::WEIGHT_LOAD_MODE
)(
    input  logic clk,
    input  logic rst_n,

    // Streaming handshake
    input  logic                        valid_in,
    output logic                        ready_in,
    input  logic signed [DATA_W-1:0]  data_in [CHANNELS-1:0],

    output logic                        valid_out,
    input  logic                        out_ready,
    output logic signed [DATA_W-1:0]  data_out [OUT_CH-1:0],

    // Weight loading (pointwise conv weights: CHANNELS x OUT_CH)
    input  logic                        weight_valid,
    input  logic [$clog2(CHANNELS*OUT_CH)-1:0] weight_addr,
    input  logic signed [DATA_W-1:0]    weight_data,

    // GroupNorm gamma/beta loading (optional; tied to identity if not loaded)
    input  logic                        gn_param_valid,
    input  logic [15:0]                 gn_param_addr,
    input  logic signed [DATA_W-1:0]    gn_param_data
);

    // =========================================================================
    // GroupNorm stage
    // =========================================================================
    logic                        gn_valid_out;
    logic                        gn_out_ready;
    logic signed [DATA_W-1:0]    gn_out [CHANNELS-1:0];
    logic signed [DATA_W-1:0]    gn_gamma [CHANNELS-1:0];
    logic signed [DATA_W-1:0]    gn_beta  [CHANNELS-1:0];

    // Default to identity gamma=1.0, beta=0.0 in Q format
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int c = 0; c < CHANNELS; c++) begin
                gn_gamma[c] <= (1 << FRAC_W);
                gn_beta[c]  <= '0;
            end
        end else if (gn_param_valid) begin
            if (gn_param_addr < CHANNELS)
                gn_gamma[gn_param_addr] <= gn_param_data;
            else if (gn_param_addr < 2*CHANNELS)
                gn_beta[gn_param_addr - CHANNELS] <= gn_param_data;
        end
    end

    group_norm #(
        .DATA_WIDTH(DATA_W),
        .FRAC_WIDTH(FRAC_W),
        .MAX_C(CHANNELS),
        .GROUPS(GROUPS),
        .USE_DSP(0)
    ) u_gn (
        .clk(clk),
        .rst_n(rst_n),
        .cfg_c(16'(CHANNELS)),
        .valid_in(valid_in),
        .ready_in(ready_in),
        .valid_out(gn_valid_out),
        .out_ready(gn_out_ready),
        .gamma(gn_gamma),
        .beta(gn_beta),
        .x_in(data_in),
        .y_out(gn_out)
    );

    // =========================================================================
    // Weight storage for pointwise Linear (CHANNELS x OUT_CH)
    // =========================================================================
    (* ram_style = "distributed" *)
    logic signed [DATA_W-1:0] pw_weights [0:CHANNELS-1][0:OUT_CH-1];

    generate
        if (WEIGHT_LOAD_MODE == 1) begin : gen_mem_weights
            // Static weights loaded at elaboration
        end else begin : gen_runtime_weights
            always_ff @(posedge clk) begin
                if (weight_valid) begin
                    for (int c = 0; c < CHANNELS; c++) begin
                        if (weight_addr[$clog2(CHANNELS)-1:0] == c[$clog2(CHANNELS)-1:0])
                            pw_weights[c][weight_addr >>> $clog2(CHANNELS)] <= weight_data;
                    end
                end
            end
        end
    endgenerate

    // =========================================================================
    // Pointwise Linear FSM (multi-cycle MAC)
    // =========================================================================
    localparam int ACC_W = 2*DATA_W + $clog2(CHANNELS);
    localparam int MAC_CYCLES = (CHANNELS + PARA_CH - 1) / PARA_CH;

    typedef enum logic [1:0] {
        S_IDLE,
        S_MAC,
        S_SIGMOID,
        S_OUTPUT
    } state_t;
    state_t state;

    logic [$clog2(MAC_CYCLES):0] mac_cycle;
    logic signed [ACC_W-1:0] accum [OUT_CH-1:0];
    logic signed [ACC_W-1:0] mac_result [OUT_CH-1:0];
    logic signed [DATA_W-1:0] feat_reg [CHANNELS-1:0];
    logic        feat_valid_reg;

    // Latch GN output when accepted
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            feat_valid_reg <= 1'b0;
            for (int c = 0; c < CHANNELS; c++)
                feat_reg[c] <= '0;
        end else if (gn_valid_out && gn_out_ready) begin
            feat_valid_reg <= gn_valid_out;
            for (int c = 0; c < CHANNELS; c++)
                feat_reg[c] <= gn_out[c];
        end else if (state == S_IDLE) begin
            feat_valid_reg <= 1'b0;
        end
    end

    // MAC accumulation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            mac_cycle <= '0;
            valid_out <= 1'b0;
            for (int o = 0; o < OUT_CH; o++) begin
                accum[o] <= '0;
                mac_result[o] <= '0;
            end
        end else begin
            valid_out <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (feat_valid_reg) begin
                        mac_cycle <= '0;
                        for (int o = 0; o < OUT_CH; o++)
                            accum[o] <= '0;
                        state <= S_MAC;
                    end
                end

                S_MAC: begin
                    for (int o = 0; o < OUT_CH; o++) begin
                        automatic logic signed [ACC_W-1:0] acc = accum[o];
                        for (int p = 0; p < PARA_CH; p++) begin
                            automatic int idx = int'(mac_cycle) * PARA_CH + p;
                            if (idx < CHANNELS) begin
                                /* verilator lint_off WIDTH */
                                acc = acc + feat_reg[idx] * pw_weights[idx][o];
                                /* verilator lint_on WIDTH */
                            end
                        end
                        accum[o] <= acc;
                    end

                    if (mac_cycle + 1 >= MAC_CYCLES) begin
                        for (int o = 0; o < OUT_CH; o++)
                            mac_result[o] <= (accum[o] + (1 <<< (FRAC_W - 1))) >>> FRAC_W;
                        state <= S_SIGMOID;
                    end else begin
                        mac_cycle <= mac_cycle + 1;
                    end
                end

                S_SIGMOID: begin
                    state <= S_OUTPUT;
                end

                S_OUTPUT: begin
                    valid_out <= 1'b1;
                    for (int o = 0; o < OUT_CH; o++) begin
                        automatic logic signed [ACC_W-1:0] x = mac_result[o];
                        // Sigmoid approximation: clamp to [0, 1] in Q format
                        if (x > (1 <<< FRAC_W))
                            data_out[o] <= DATA_W'((1 <<< FRAC_W) - 1);
                        else if (x < -(1 <<< FRAC_W))
                            data_out[o] <= '0;
                        else
                            data_out[o] <= DATA_W'(((1 <<< (FRAC_W - 1)) + (x >>> 2)));
                    end
                    if (out_ready) begin
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // Backpressure
    assign gn_out_ready = (state == S_IDLE);

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    FINALUP_VALID: assert property (
        @(posedge clk) disable iff (!rst_n)
        valid_out && !out_ready |=> valid_out
    ) else $error("final_upsample: valid_out deasserted before out_ready");
    // synthesis translate_on

endmodule