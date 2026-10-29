// =============================================================================
// skip_connect.sv
//
// Paper-aligned UNet skip connection (E-RODNet, IEEE Sensors Journal 2024,
// equation 9):
//
//     skipConnect(x1, x2) = GELU(GN(Linear(Cat[x1, x2])))
//
// Where:
//   • x1 is the upstream decoder feature   (MAX_C channels)
//   • x2 is the encoder skip               (MAX_C channels)
//   • Cat[x1, x2] is the 2*MAX_C concatenation
//   • Linear projects 2*MAX_C → MAX_C
//   • GN normalises across CHAN_PER_GROUP = MAX_C/GROUPS
//   • GELU is applied per-channel (LUT-based to preserve DSP budget)
//
// Pipeline (sequential, single-token-in / single-token-out):
//   • S_IDLE      : wait for valid_in
//   • S_LINEAR    : 2*MAX_C cycles (2 cycles per output channel).
//                   Phase 0: compute group partial sums (MAC_GROUP_SIZE terms/group).
//                   Phase 1: sum groups → lin_out[oc].
//                   Uses LUT multiplies (use_dsp = "no").
//                   Reduces combinational depth from 2*MAX_C to MAC_GROUP_SIZE MACs.
//   • S_GN        : feed accumulated linear-output token into group_norm.sv
//                   (8-stage pipeline). gn_valid_in_q is asserted for one
//                   cycle (gn_launched flag), then the state waits for
//                   gn_valid_out_w before advancing.
//   • S_GELU      : 1-cycle per-channel LUT lookup via shared gelu_rom_shared.
//   • S_DRAIN     : raise valid_out, hold y_out until out_ready.
//
// Latency: 2*MAX_C + 8 + 1 + 1 ≈ 138 cycles per token at MAX_C = 64.
//
// Synthesis directives:
//   • (* use_dsp = "no" *) on the linear-projection multiply (LUT path,
//     keeps the 240-DSP Artix-7 budget reserved for the systolic array).
//   • Accumulator width ACC_WIDTH = 32, sufficient for 2*64 INT8 inputs.
//
// Drop-in usage in e_rodnet_unet_top.sv: replace the in-line sat_add()
// calls in the D2 / D1 NN-up + skip merge stages with this module to
// realise paper Eq.~9 exactly. See docs/paper_updates_proposal.tex §2.3.
// =============================================================================

import pkg_config::*;

module skip_connect #(
    parameter int DATA_WIDTH = 8,
    parameter int FRAC_WIDTH = 16,
    parameter int ACC_WIDTH  = 32,
    parameter int MAX_C      = 64,
    parameter int GROUPS     = 8,    // MAX_C must be divisible by GROUPS
    parameter int MAC_GROUP_SIZE = 8  // Adder tree group size (combinational depth)
)(
    input  logic clk,
    input  logic rst_n,
    input  logic cfg_enable,

    // Input handshake
    input  logic                              valid_in,
    output logic                              ready_in,
    input  logic signed [DATA_WIDTH-1:0]      x1_in [0:MAX_C-1],   // decoder upstream
    input  logic signed [DATA_WIDTH-1:0]      x2_in [0:MAX_C-1],   // encoder skip

    // Output handshake
    output logic                              valid_out,
    input  logic                              out_ready,
    output logic signed [DATA_WIDTH-1:0]      y_out  [0:MAX_C-1],

    // Linear weights: w_lin[OUT_CH][IN_CH], IN_CH = 2*MAX_C
    input  logic signed [DATA_WIDTH-1:0]      w_lin [0:MAX_C-1][0:2*MAX_C-1],

    // GroupNorm affine params
    input  logic signed [DATA_WIDTH-1:0]      gn_gamma [0:MAX_C-1],
    input  logic signed [DATA_WIDTH-1:0]      gn_beta  [0:MAX_C-1]
);

    // =========================================================================
    // Constants
    // =========================================================================
    localparam int IN_C  = 2*MAX_C;
    localparam int OUT_C = MAX_C;
    localparam int NUM_GROUPS = (IN_C + MAC_GROUP_SIZE - 1) / MAC_GROUP_SIZE;

    // =========================================================================
    // FSM
    // =========================================================================
    typedef enum logic [2:0] {
        S_IDLE,
        S_LINEAR,
        S_GN,
        S_GELU,
        S_DRAIN
    } state_t;

    state_t state, state_next;

    // Flag: GN has been launched (gn_valid_in_q pulsed for one cycle)
    logic gn_launched;

    // Output-channel index walked through during S_LINEAR
    logic [$clog2(OUT_C):0] lin_oc;

    // =========================================================================
    // Concatenated input register
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] cat_reg [0:IN_C-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < IN_C; i++) cat_reg[i] <= '0;
        end else if (state == S_IDLE && valid_in && ready_in) begin
            for (int c = 0; c < MAX_C; c++) begin
                cat_reg[c]         <= x1_in[c];
                cat_reg[MAX_C + c] <= x2_in[c];
            end
        end
    end

    // =========================================================================
    // Linear stage — 2-cycle pipelined per-output-channel MAC (Phase 0/1)
    //
    // For output channel `lin_oc`, compute
    //   acc = sum_{i=0..IN_C-1} w_lin[lin_oc][i] * cat_reg[i]
    // then store in lin_out[lin_oc] after fixed-point scaling.
    //
    // Phase 0 (mac_phase=0): Compute group partial sums (MAC_GROUP_SIZE terms/group)
    //   combinationally and register them. This reduces combinational depth from
    //   IN_C=128 to MAC_GROUP_SIZE=8 multiply-adds.
    // Phase 1 (mac_phase=1): Sum the registered group sums → lin_acc, write
    //   saturated result to lin_out[lin_oc], increment lin_oc.
    // =========================================================================
    logic signed [DATA_WIDTH-1:0]                  lin_out [0:OUT_C-1];
    logic                                          mac_phase;
    logic signed [ACC_WIDTH-1:0]                   group_sums [0:NUM_GROUPS-1];
    logic signed [ACC_WIDTH-1:0]                   group_sums_r [0:NUM_GROUPS-1];
    logic signed [ACC_WIDTH-1:0]                   lin_acc;

    // Phase counter: toggles every cycle during S_LINEAR
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)              mac_phase <= 1'b0;
        else if (state == S_LINEAR) mac_phase <= mac_phase + 1'b1;
        else                     mac_phase <= 1'b0;
    end

    // Output-channel index: increments on phase 1
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                                    lin_oc <= '0;
        else if (state == S_IDLE && valid_in && ready_in) lin_oc <= '0;
        else if (state == S_LINEAR && mac_phase == 1'b1) lin_oc <= lin_oc + 1'b1;
    end

    // Phase 0 — Combinational per-group partial sums (MAC_GROUP_SIZE terms each)
    always_comb begin
        for (int g = 0; g < NUM_GROUPS; g++) begin
            automatic logic signed [ACC_WIDTH-1:0] gs = '0;
            for (int i = 0; i < MAC_GROUP_SIZE; i++) begin
                automatic int idx = g * MAC_GROUP_SIZE + i;
                if (idx < IN_C) begin
                    (* use_dsp = "no" *)
                    gs = gs + ACC_WIDTH'(w_lin[lin_oc[$clog2(OUT_C)-1:0]][idx]) *
                            ACC_WIDTH'(cat_reg[idx]);
                end
            end
            group_sums[g] = gs;
        end
    end

    // Register group sums at end of phase 0 (ready for phase 1)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int g = 0; g < NUM_GROUPS; g++) group_sums_r[g] <= '0;
        end else if (state == S_LINEAR && mac_phase == 1'b0) begin
            for (int g = 0; g < NUM_GROUPS; g++)
                group_sums_r[g] <= group_sums[g];
        end
    end

    // Phase 1 — Combinational sum of registered group partials
    always_comb begin
        automatic logic signed [ACC_WIDTH-1:0] s = '0;
        for (int g = 0; g < NUM_GROUPS; g++) begin
            s = s + group_sums_r[g];
        end
        lin_acc = s;
    end

    // Write saturated lin_out[lin_oc] on phase 1
    // Combinational saturate (extracted from always_ff for Verilator compat)
    logic signed [DATA_WIDTH-1:0] lin_out_comb;
    always_comb begin
        automatic logic signed [ACC_WIDTH-1:0] q = lin_acc >>> FRAC_WIDTH;
        if      (q[ACC_WIDTH-1] && |(~q[ACC_WIDTH-1:DATA_WIDTH-1]))
            lin_out_comb = {1'b1, {(DATA_WIDTH-1){1'b0}}};
        else if (!q[ACC_WIDTH-1] && |q[ACC_WIDTH-1:DATA_WIDTH-1])
            lin_out_comb = {1'b0, {(DATA_WIDTH-1){1'b1}}};
        else
            lin_out_comb = q[DATA_WIDTH-1:0];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int c = 0; c < OUT_C; c++) lin_out[c] <= '0;
        end else if (state == S_LINEAR && mac_phase == 1'b1) begin
            lin_out[lin_oc[$clog2(OUT_C)-1:0]] <= lin_out_comb;
        end
    end

    // =========================================================================
    // GroupNorm submodule
    // =========================================================================
    logic                              gn_valid_in_q;
    logic                              gn_ready_in_w;
    logic                              gn_valid_out_w;
    logic signed [DATA_WIDTH-1:0]      gn_y_out [0:MAX_C-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                          gn_valid_in_q <= 1'b0;
        else if (state == S_GN && !gn_launched) gn_valid_in_q <= 1'b1;
        else                                 gn_valid_in_q <= 1'b0;
    end

    // gn_launched: set on first cycle of S_GN, cleared on exit
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                          gn_launched <= 1'b0;
        else if (state == S_GN && !gn_launched) gn_launched <= 1'b1;
        else if (state != S_GN)              gn_launched <= 1'b0;
    end

    // group_norm.ready_in is an OUTPUT (tells upstream it is ready).
    // skip_connect always presents valid data when gn_valid_in_q is high,
    // so the ready signal is intentionally unused here.
    wire _unused_gn_ready_in = gn_ready_in_w;

    group_norm #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH),
        .MAX_C(MAX_C),
        .GROUPS(GROUPS),
        .USE_DSP(0)
    ) u_gn (
        .clk(clk), .rst_n(rst_n),
        .cfg_c(16'(MAX_C)),
        .valid_in (gn_valid_in_q),
        .ready_in (gn_ready_in_w),
        .valid_out(gn_valid_out_w),
        .out_ready(1'b1),
        .gamma(gn_gamma),
        .beta (gn_beta),
        .x_in (lin_out),
        .y_out(gn_y_out)
    );

    // =========================================================================
    // GELU stage — per-channel LUT via shared gelu_rom_shared
    // =========================================================================
    localparam int GELU_INDEX_BITS = pkg_config::INDEX_BITS;

    logic                                  gelu_rd_en;
    logic [GELU_INDEX_BITS-1:0]            gelu_rd_addr [0:MAX_C-1];
    logic signed [ACC_WIDTH-1:0]           gelu_y_base  [0:MAX_C-1];
    logic signed [FRAC_WIDTH-1:0]          gelu_slope   [0:MAX_C-1];

    always_comb begin
        for (int c = 0; c < MAX_C; c++) begin
            // Map signed INT8 to [0, 256) index (offset binary)
            gelu_rd_addr[c] = {~gn_y_out[c][DATA_WIDTH-1], gn_y_out[c][DATA_WIDTH-2:0]};
        end
    end
    assign gelu_rd_en = (state == S_GELU) || (state == S_DRAIN);

    gelu_rom_shared #(
        .DATA_W   (ACC_WIDTH),
        .FRAC_W   (FRAC_WIDTH),
        .INDEX_BITS(GELU_INDEX_BITS),
        .NUM_PORTS(MAX_C)
    ) u_gelu_rom (
        .clk(clk), .rst_n(rst_n),
        .rd_addr(gelu_rd_addr),
        .rd_en  (gelu_rd_en),
        .y_base_out(gelu_y_base),
        .slope_out (gelu_slope)
    );

    // Linear-interpolated GELU output: y = y_base + (slope * x_low_bits)
    // Combinational GELU interpolation (extracted from always_ff for Verilator compat)
    logic signed [DATA_WIDTH-1:0] y_out_comb [0:MAX_C-1];
    always_comb begin
        for (int c = 0; c < MAX_C; c++) begin
            (* use_dsp = "no" *)
            automatic logic signed [ACC_WIDTH-1:0] interp = gelu_y_base[c] +
                    ((ACC_WIDTH'(gelu_slope[c]) * $signed({ {(ACC_WIDTH-4){1'b0}}, gn_y_out[c][3:0] })) >>> 4);
            if      (interp[ACC_WIDTH-1] && |(~interp[ACC_WIDTH-1:DATA_WIDTH-1]))
                y_out_comb[c] = {1'b1, {(DATA_WIDTH-1){1'b0}}};
            else if (!interp[ACC_WIDTH-1] && |interp[ACC_WIDTH-1:DATA_WIDTH-1])
                y_out_comb[c] = {1'b0, {(DATA_WIDTH-1){1'b1}}};
            else
                y_out_comb[c] = interp[DATA_WIDTH-1:0];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int c = 0; c < MAX_C; c++) y_out[c] <= '0;
        end else if (state == S_GELU) begin
            for (int c = 0; c < MAX_C; c++)
                y_out[c] <= y_out_comb[c];
        end
    end

    // =========================================================================
    // FSM transitions
    // =========================================================================
    always_comb begin
        state_next = state;
        case (state)
            S_IDLE:    if (valid_in && cfg_enable)              state_next = S_LINEAR;
            S_LINEAR:  if (lin_oc == OUT_C-1 && mac_phase == 1'b1) state_next = S_GN;
            S_GN:      if (gn_valid_out_w)                      state_next = S_GELU;
            S_GELU:                                             state_next = S_DRAIN;
            S_DRAIN:   if (out_ready)                           state_next = S_IDLE;
            default:                                            state_next = S_IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)             state <= S_IDLE;
        else if (cfg_enable)    state <= state_next;
        else                    state <= S_IDLE;
    end

    assign ready_in  = (state == S_IDLE);
    assign valid_out = (state == S_DRAIN);

endmodule
