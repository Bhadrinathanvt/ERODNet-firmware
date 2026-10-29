// =============================================================================
//  ssf_module.sv
//
//  SSF-Module: Spatial-Spectral Fusion block from the architecture diagram.
//  Updated to use a single shared GELU ROM across all branches and channels.
// =============================================================================

import pkg_config::*;
import pkg_module_config::*;

module ssf_module #(
    parameter int DATA_WIDTH = pkg_module_config::SSF_MODULE_DATA_WIDTH,
    parameter int ACC_WIDTH  = pkg_config::ACC_WIDTH,
    parameter int FRAC_WIDTH = pkg_config::FRAC_WIDTH,
    parameter int CHANNELS   = 8        // must be even
)(
    input  logic clk,
    input  logic rst_n,
    input  logic in_valid,
    output logic out_valid,
    input  logic ready_in,          // downstream ready
    output logic ready_out,         // upstream ready (skid buffer capacity)
    input  logic last_in,           // frame boundary
    output logic last_out,          // delayed frame boundary

    // ── Input windows (flattened spatial kernel, all channels) ───────────────
    input  logic signed [DATA_WIDTH-1:0] x_1x1 [CHANNELS-1:0],
    input  logic signed [DATA_WIDTH-1:0] x_3x3 [CHANNELS-1:0][26:0],

    // ── Weights ──────────────────────────────────────────────────────────────
    input  logic signed [DATA_WIDTH-1:0] w_1x1    [CHANNELS-1:0],
    input  logic signed [DATA_WIDTH-1:0] w_3x3    [CHANNELS-1:0][26:0],
    input  logic signed [DATA_WIDTH-1:0] w_linear [CHANNELS-1:0][CHANNELS-1:0],

    // ── BatchNorm parameters (3 BN layers) ───────────────────────────────────
    input  logic signed [ACC_WIDTH-1:0] bn1_gamma   [CHANNELS-1:0],
    input  logic signed [ACC_WIDTH-1:0] bn1_beta    [CHANNELS-1:0],
    input  logic signed [ACC_WIDTH-1:0] bn1_mean    [CHANNELS-1:0],
    input  logic signed [ACC_WIDTH-1:0] bn1_inv_std [CHANNELS-1:0],

    input  logic signed [ACC_WIDTH-1:0] bn2_gamma   [CHANNELS-1:0],
    input  logic signed [ACC_WIDTH-1:0] bn2_beta    [CHANNELS-1:0],
    input  logic signed [ACC_WIDTH-1:0] bn2_mean    [CHANNELS-1:0],
    input  logic signed [ACC_WIDTH-1:0] bn2_inv_std [CHANNELS-1:0],

    input  logic signed [ACC_WIDTH-1:0] bn3_gamma   [CHANNELS-1:0],
    input  logic signed [ACC_WIDTH-1:0] bn3_beta    [CHANNELS-1:0],
    input  logic signed [ACC_WIDTH-1:0] bn3_mean    [CHANNELS-1:0],
    input  logic signed [ACC_WIDTH-1:0] bn3_inv_std [CHANNELS-1:0],

    // ── Output ───────────────────────────────────────────────────────────────
    output logic signed [ACC_WIDTH-1:0] y_out [CHANNELS-1:0]
);

    // ─────────────────────────────────────────────────────────────────────────
    //  SHARED GELU ROM (Serves Left, Right, and Output branches)
    // ─────────────────────────────────────────────────────────────────────────
    localparam int INDEX_BITS = pkg_config::INDEX_BITS;
    localparam int NUM_GELU_PORTS = 3 * CHANNELS; // Left, Right, and Out branches

    logic [INDEX_BITS-1:0]    shared_rd_addr [NUM_GELU_PORTS-1:0];
    logic signed [ACC_WIDTH-1:0] shared_y_base  [NUM_GELU_PORTS-1:0];
    logic signed [FRAC_WIDTH-1:0] shared_slope   [NUM_GELU_PORTS-1:0];

    gelu_rom_shared #(
        .DATA_W(ACC_WIDTH),
        .FRAC_W(FRAC_WIDTH),
        .INDEX_BITS(INDEX_BITS),
        .NUM_PORTS(NUM_GELU_PORTS)
    ) u_shared_gelu_rom (
        .clk(clk),
        .rst_n(rst_n),
        .rd_addr(shared_rd_addr),
        .rd_en(1'b1),
        .y_base_out(shared_y_base),
        .slope_out(shared_slope)
    );

    // Interface signals mapping GELU branches to the Shared ROM
    logic [INDEX_BITS-1:0]    left_index_o  [CHANNELS-1:0], right_index_o  [CHANNELS-1:0], out_index_o  [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] left_y_base_i [CHANNELS-1:0], right_y_base_i [CHANNELS-1:0], out_y_base_i [CHANNELS-1:0];
    logic signed [FRAC_WIDTH-1:0] left_slope_i  [CHANNELS-1:0], right_slope_i  [CHANNELS-1:0], out_slope_i  [CHANNELS-1:0];

    always_comb begin
        for (int c = 0; c < CHANNELS; c++) begin
            // Left Branch Mappings (Ports 0 to C-1)
            shared_rd_addr[c] = left_index_o[c];
            left_y_base_i[c]  = shared_y_base[c];
            left_slope_i[c]   = shared_slope[c];

            // Right Branch Mappings (Ports C to 2C-1)
            shared_rd_addr[CHANNELS + c] = right_index_o[c];
            right_y_base_i[c]            = shared_y_base[CHANNELS + c];
            right_slope_i[c]             = shared_slope[CHANNELS + c];

            // Output Branch Mappings (Ports 2C to 3C-1)
            shared_rd_addr[2*CHANNELS + c] = out_index_o[c];
            out_y_base_i[c]                = shared_y_base[2*CHANNELS + c];
            out_slope_i[c]                 = shared_slope[2*CHANNELS + c];
        end
    end

    // ─────────────────────────────────────────────────────────────────────────
    //  LEFT BRANCH   Conv3D[1×1×1] → BN1_left → GELU
    // ─────────────────────────────────────────────────────────────────────────
    logic signed [ACC_WIDTH-1:0] left_conv  [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] left_conv_q [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] left_bn    [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] left_gelu  [CHANNELS-1:0];
    logic                        left_conv_v, left_bn_v, left_gelu_v;
    logic left_conv_v_arr [CHANNELS-1:0];

    always_comb begin
        left_conv_v = 1'b1;
        for (int i = 0; i < CHANNELS; i++) left_conv_v &= left_conv_v_arr[i];
    end

    generate
        for (genvar c = 0; c < CHANNELS; c++) begin : gen_left_conv
            logic signed [DATA_WIDTH-1:0] win1 [0:0]; assign win1[0] = x_1x1[c];
            logic signed [DATA_WIDTH-1:0] wk1  [0:0]; assign wk1[0]  = w_1x1[c];

            // out_ready tied high: SSF consumes conv3d_lane free-running (samples
            // out_valid only); the lane's skid buffer never back-pressures here.
            conv3d_lane #(.DATA_WIDTH(DATA_WIDTH),.ACC_WIDTH(ACC_WIDTH),.KERNEL(1))
            u_conv1 (.clk(clk),.rst_n(rst_n),.in_valid(in_valid),.ready_in(),.out_ready(1'b1),.out_valid(left_conv_v_arr[c]),.x_win(win1),.w_kernel(wk1),.y_out(left_conv[c]));
            
            assign left_conv_q[c] = left_conv[c] * (1 << FRAC_WIDTH);
        end
    endgenerate

    logic signed [CHANNELS*ACC_WIDTH-1:0] bn1_gamma_flat, bn1_beta_flat, bn1_mean_flat, bn1_inv_std_flat, left_conv_q_flat, left_bn_flat;
    always_comb begin
        for (int c = 0; c < CHANNELS; c++) begin
            bn1_gamma_flat[c*ACC_WIDTH +: ACC_WIDTH]   = bn1_gamma[c];
            bn1_beta_flat[c*ACC_WIDTH +: ACC_WIDTH]    = bn1_beta[c];
            bn1_mean_flat[c*ACC_WIDTH +: ACC_WIDTH]    = bn1_mean[c];
            bn1_inv_std_flat[c*ACC_WIDTH +: ACC_WIDTH] = bn1_inv_std[c];
            left_conv_q_flat[c*ACC_WIDTH +: ACC_WIDTH] = left_conv_q[c];
            left_bn[c] = left_bn_flat[c*ACC_WIDTH +: ACC_WIDTH];
        end
    end

    batch_norm_3d #(.DATA_WIDTH(ACC_WIDTH),.FRAC_WIDTH(FRAC_WIDTH),.CHANNELS(CHANNELS))
    u_bn1_left (
        .clk(clk),.rst_n(rst_n),.in_valid(left_conv_v),.out_valid(left_bn_v),
        .gamma_flat(bn1_gamma_flat),.beta_flat(bn1_beta_flat),.run_mean_flat(bn1_mean_flat),.inv_std_flat(bn1_inv_std_flat),
        .x_in_flat(left_conv_q_flat),.y_out_flat(left_bn_flat)
    );

    // Vectorized GELU instance (replaces the generate loop)
    gelu_rom #(
        .DATA_W(ACC_WIDTH), .FRAC_W(FRAC_WIDTH), .INDEX_BITS(INDEX_BITS),
        .CHANNELS(CHANNELS), .SHARE_ROM(1)
    ) u_gelu_l (
        .clk(clk), .rst_n(rst_n), .gelu_en(1'b1),
        .x_in(left_bn), .valid_in(left_bn_v),
        .y_out(left_gelu), .valid_out(left_gelu_v),
        .index_o(left_index_o), .rom_y_base_i(left_y_base_i), .rom_slope_i(left_slope_i)
    );

    // ─────────────────────────────────────────────────────────────────────────
    //  RIGHT BRANCH  Conv3D[3×3×3] → BN2_right → GELU → MaxPool3D (identity)
    // ─────────────────────────────────────────────────────────────────────────
    logic signed [ACC_WIDTH-1:0] right_conv [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] right_conv_q [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] right_bn   [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] right_gelu [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] right_pool [CHANNELS-1:0]; 
    logic right_conv_v, right_bn_v, right_gelu_v;
    logic right_conv_v_arr [CHANNELS-1:0];

    always_comb begin
        right_conv_v = 1'b1;
        for (int i = 0; i < CHANNELS; i++) right_conv_v &= right_conv_v_arr[i];
    end

    generate
        for (genvar c = 0; c < CHANNELS; c++) begin : gen_right_conv
            conv3d_lane #(.DATA_WIDTH(DATA_WIDTH),.ACC_WIDTH(ACC_WIDTH),.KERNEL(3))
            u_conv3 (.clk(clk),.rst_n(rst_n),.in_valid(in_valid),.ready_in(),.out_ready(1'b1),.out_valid(right_conv_v_arr[c]),.x_win(x_3x3[c]),.w_kernel(w_3x3[c]),.y_out(right_conv[c]));
            
            assign right_conv_q[c] = right_conv[c] * (1 << FRAC_WIDTH);
        end
    endgenerate

    logic signed [CHANNELS*ACC_WIDTH-1:0] bn2_gamma_flat, bn2_beta_flat, bn2_mean_flat, bn2_inv_std_flat, right_conv_q_flat, right_bn_flat;
    always_comb begin
        for (int c = 0; c < CHANNELS; c++) begin
            bn2_gamma_flat[c*ACC_WIDTH +: ACC_WIDTH]   = bn2_gamma[c];
            bn2_beta_flat[c*ACC_WIDTH +: ACC_WIDTH]    = bn2_beta[c];
            bn2_mean_flat[c*ACC_WIDTH +: ACC_WIDTH]    = bn2_mean[c];
            bn2_inv_std_flat[c*ACC_WIDTH +: ACC_WIDTH] = bn2_inv_std[c];
            right_conv_q_flat[c*ACC_WIDTH +: ACC_WIDTH] = right_conv_q[c];
            right_bn[c] = right_bn_flat[c*ACC_WIDTH +: ACC_WIDTH];
        end
    end

    batch_norm_3d #(.DATA_WIDTH(ACC_WIDTH),.FRAC_WIDTH(FRAC_WIDTH),.CHANNELS(CHANNELS))
    u_bn2_right (
        .clk(clk),.rst_n(rst_n),.in_valid(right_conv_v),.out_valid(right_bn_v),
        .gamma_flat(bn2_gamma_flat),.beta_flat(bn2_beta_flat),.run_mean_flat(bn2_mean_flat),.inv_std_flat(bn2_inv_std_flat),
        .x_in_flat(right_conv_q_flat),.y_out_flat(right_bn_flat)
    );

    // Vectorized GELU instance (replaces the generate loop)
    gelu_rom #(
        .DATA_W(ACC_WIDTH), .FRAC_W(FRAC_WIDTH), .INDEX_BITS(INDEX_BITS),
        .CHANNELS(CHANNELS), .SHARE_ROM(1)
    ) u_gelu_r (
        .clk(clk), .rst_n(rst_n), .gelu_en(1'b1),
        .x_in(right_bn), .valid_in(right_bn_v),
        .y_out(right_gelu), .valid_out(right_gelu_v),
        .index_o(right_index_o), .rom_y_base_i(right_y_base_i), .rom_slope_i(right_slope_i)
    );

    assign right_pool = right_gelu;

    // ─────────────────────────────────────────────────────────────────────────
    //  CONCAT  [left_gelu[C/2-1:0] | right_pool[C/2-1:0]]  → C channels
    // ─────────────────────────────────────────────────────────────────────────
    localparam int HALF = CHANNELS / 2;
    logic signed [ACC_WIDTH-1:0] concat_in [CHANNELS-1:0];
    generate
        for (genvar c = 0; c < HALF; c++) begin : gen_concat
            assign concat_in[c]        = left_gelu[c];
            assign concat_in[c + HALF] = right_pool[c];
        end
    endgenerate

    // ─────────────────────────────────────────────────────────────────────────
    //  LINEAR (pointwise Conv1×1×1 across C) → BN3 → GELU
    // ─────────────────────────────────────────────────────────────────────────
    logic signed [ACC_WIDTH-1:0] lin_out  [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn3_out  [CHANNELS-1:0];
    logic                        lin_v, bn3_v;

    // Linear accumulation (automatic extracted to always_comb)
    logic signed [ACC_WIDTH-1:0] lin_out_comb [CHANNELS-1:0];
    always_comb begin
        for (int co = 0; co < CHANNELS; co++) begin
            logic signed [ACC_WIDTH-1:0] acc;
            acc = '0;
            for (int ci = 0; ci < CHANNELS; ci++)
                acc = acc + concat_in[ci] * ACC_WIDTH'(w_linear[co][ci]);
            lin_out_comb[co] = acc >>> FRAC_WIDTH;
        end
    end

    generate
        for (genvar co = 0; co < CHANNELS; co++) begin : gen_linear_out
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) lin_out[co] <= '0;
                else        lin_out[co] <= lin_out_comb[co];
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) lin_v <= '0;
        else        lin_v <= left_gelu_v && right_gelu_v;

    logic signed [CHANNELS*ACC_WIDTH-1:0] bn3_gamma_flat, bn3_beta_flat, bn3_mean_flat, bn3_inv_std_flat, lin_out_flat, bn3_out_flat;
    always_comb begin
        for (int c = 0; c < CHANNELS; c++) begin
            bn3_gamma_flat[c*ACC_WIDTH +: ACC_WIDTH]   = bn3_gamma[c];
            bn3_beta_flat[c*ACC_WIDTH +: ACC_WIDTH]    = bn3_beta[c];
            bn3_mean_flat[c*ACC_WIDTH +: ACC_WIDTH]    = bn3_mean[c];
            bn3_inv_std_flat[c*ACC_WIDTH +: ACC_WIDTH] = bn3_inv_std[c];
            lin_out_flat[c*ACC_WIDTH +: ACC_WIDTH]     = lin_out[c];
            bn3_out[c] = bn3_out_flat[c*ACC_WIDTH +: ACC_WIDTH];
        end
    end

    batch_norm_3d #(.DATA_WIDTH(ACC_WIDTH),.FRAC_WIDTH(FRAC_WIDTH),.CHANNELS(CHANNELS))
    u_bn3 (
        .clk(clk),.rst_n(rst_n),.in_valid(lin_v),.out_valid(bn3_v),
        .gamma_flat(bn3_gamma_flat),.beta_flat(bn3_beta_flat),.run_mean_flat(bn3_mean_flat),.inv_std_flat(bn3_inv_std_flat),
        .x_in_flat(lin_out_flat),.y_out_flat(bn3_out_flat)
    );

    // Vectorized GELU instance (replaces the generate loop)
    // Internal pipeline output before skid buffer
    logic signed [ACC_WIDTH-1:0] y_out_internal [CHANNELS-1:0];
    logic out_valid_internal;
    logic last_out_internal;

    gelu_rom #(
        .DATA_W(ACC_WIDTH), .FRAC_W(FRAC_WIDTH), .INDEX_BITS(INDEX_BITS),
        .CHANNELS(CHANNELS), .SHARE_ROM(1)
    ) u_gelu_out (
        .clk(clk), .rst_n(rst_n), .gelu_en(1'b1),
        .x_in(bn3_out), .valid_in(bn3_v),
        .y_out(y_out_internal), .valid_out(out_valid_internal),
        .index_o(out_index_o), .rom_y_base_i(out_y_base_i), .rom_slope_i(out_slope_i)
    );

    // Pipeline latency = 19 cycles (verified from submodule latencies)
    localparam int SSF_LATENCY = 19;
    logic [SSF_LATENCY-1:0] last_sr;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) last_sr <= '0;
        else        last_sr <= {last_sr[SSF_LATENCY-2:0], last_in};
    end
    assign last_out_internal = last_sr[SSF_LATENCY-1];

    // Output FIFO for backpressure handling.
    // Replaced 1-deep skid_buffer with elastic_fifo (depth 64) to absorb the
    // rate mismatch between SSF (1 token/cycle after pipeline fill) and E0
    // convformer (~1 token/16 cycles, expand SA rate).  Depth 64 holds all
    // 64 tokens of one 8x8 tile.
    //
    // Input gating: ready_out goes low when fewer than SSF_LATENCY (19) free
    // slots remain, preventing pipeline mid-flight overflow since the SSF
    // pipeline has fixed latency and cannot be stalled mid-stage.
    localparam int SSF_PAYLOAD_W = CHANNELS * ACC_WIDTH;
    localparam int SSF_FIFO_DW  = SSF_PAYLOAD_W + 1;  // +1 for last bit (MSB)
    localparam int SSF_FIFO_DEPTH = pkg_config::SSF_FIFO_DEPTH;
    logic [SSF_FIFO_DW-1:0] ssf_fifo_din, ssf_fifo_dout;
    logic                    ssf_fifo_full, ssf_fifo_empty;
    logic [7:0]              ssf_fifo_count;
    logic                    ssf_fifo_almost_full;
    logic                    ready_out_internal;

    assign ssf_fifo_almost_full = ssf_fifo_count >= (SSF_FIFO_DEPTH - SSF_LATENCY - 1);

    always_comb begin
        // Pack: {last, data[CHANNELS*ACC_WIDTH-1:0]}
        ssf_fifo_din = {last_out_internal, y_out_internal[0]};
        for (int c = 1; c < CHANNELS; c++)
            ssf_fifo_din[c*ACC_WIDTH +: ACC_WIDTH] = y_out_internal[c];
        // Unpack
        last_out = ssf_fifo_dout[SSF_FIFO_DW-1];
        for (int c = 0; c < CHANNELS; c++)
            y_out[c] = ssf_fifo_dout[c*ACC_WIDTH +: ACC_WIDTH];
    end

    elastic_fifo #(
        .DATA_WIDTH(SSF_FIFO_DW),
        .DEPTH(SSF_FIFO_DEPTH)
    ) u_ssf_out_fifo (
        .clk(clk),
        .rst_n(rst_n),
        .wr_valid(out_valid_internal),
        .wr_ready(ready_out_internal),
        .din(ssf_fifo_din),
        .rd_valid(out_valid),
        .rd_ready(ready_in),
        .dout(ssf_fifo_dout),
        .count(ssf_fifo_count)
    );

    // Gate upstream ready: stall input when fewer than pipeline-latency
    // free slots remain, to prevent mid-flight overflow.
    assign ready_out = ready_out_internal && !ssf_fifo_almost_full;

endmodule