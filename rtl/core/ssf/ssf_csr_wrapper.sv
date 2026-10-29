// =============================================================================
// ssf_csr_wrapper.sv
//
// CSR Wrapper for SSF (Spatial-Spectral Fusion) Module.
// Provides AXI4-Lite CSR interface for runtime configuration of:
//   - BatchNorm parameters (gamma, beta, mean, inv_std for each layer)
//   - Linear layer weights
//   - Enable/configuration signals
//
// CSR Address Map:
//   0x500-0x53F: BN1 parameters (gamma, beta, mean, inv_std)
//   0x540-0x57F: BN2 parameters
//   0x580-0x5BF: BN3 parameters
//   0x5C0-0x5FF: Linear layer weights
//
// All parameters use shadow registers for glitch-free updates.
// Parameters are committed to live registers on cfg_update pulse.
// =============================================================================

import pkg_config::*;
import csr_regs::*;

module ssf_csr_wrapper #(
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32,
    parameter int FRAC_WIDTH = 16,
    parameter int CHANNELS   = 8,
    parameter int CHANNELS_W = $clog2(CHANNELS)
)(
    input  logic clk,
    input  logic rst_n,

    // AXI4-Lite CSR Interface
    csr_intf.slave s_csr,

    // SSF Data Interface (AXI-Stream style)
    input  logic                        in_valid,
    input  logic signed [DATA_WIDTH-1:0] x_1x1 [CHANNELS-1:0],
    input  logic signed [DATA_WIDTH-1:0] x_3x3 [CHANNELS-1:0][26:0],
    input  logic signed [DATA_WIDTH-1:0] w_1x1 [CHANNELS-1:0],
    input  logic signed [DATA_WIDTH-1:0] w_3x3 [CHANNELS-1:0][26:0],
    input  logic signed [DATA_WIDTH-1:0] w_linear [CHANNELS-1:0][CHANNELS-1:0],
    output logic                        out_valid,
    output logic signed [ACC_WIDTH-1:0] y_out [CHANNELS-1:0],

    // Configuration update
    input  logic cfg_update,

    // Enable
    input  logic ssf_en
);

    // =========================================================================
    // Shadow Registers for BatchNorm Parameters
    // =========================================================================
    // Each BatchNorm layer has 4 parameters: gamma, beta, mean, inv_std
    // Each parameter is an array of CHANNELS elements (32-bit each)

    // BN1 parameters
    logic signed [ACC_WIDTH-1:0] bn1_gamma_shadow   [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn1_beta_shadow    [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn1_mean_shadow    [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn1_inv_std_shadow [CHANNELS-1:0];

    // BN2 parameters
    logic signed [ACC_WIDTH-1:0] bn2_gamma_shadow   [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn2_beta_shadow    [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn2_mean_shadow    [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn2_inv_std_shadow [CHANNELS-1:0];

    // BN3 parameters
    logic signed [ACC_WIDTH-1:0] bn3_gamma_shadow   [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn3_beta_shadow    [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn3_mean_shadow    [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn3_inv_std_shadow [CHANNELS-1:0];

    // Live registers (committed from shadow on cfg_update)
    logic signed [ACC_WIDTH-1:0] bn1_gamma_live   [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn1_beta_live    [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn1_mean_live    [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn1_inv_std_live [CHANNELS-1:0];

    logic signed [ACC_WIDTH-1:0] bn2_gamma_live   [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn2_beta_live    [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn2_mean_live    [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn2_inv_std_live [CHANNELS-1:0];

    logic signed [ACC_WIDTH-1:0] bn3_gamma_live   [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn3_beta_live    [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn3_mean_live    [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] bn3_inv_std_live [CHANNELS-1:0];

    // =========================================================================
    // CSR Address Decode
    // =========================================================================
    localparam int ADDR_W = 12;
    localparam int DATA_W = 32;

    logic aw_pend, w_pend;
    logic [ADDR_W-1:0] aw_addr_q;
    logic [DATA_W-1:0] w_data_q;
    logic [DATA_W/8-1:0] w_strb_q;
    logic do_write;

    assign s_csr.awready = ~aw_pend & ~s_csr.bvalid;
    assign s_csr.wready  = ~w_pend & ~s_csr.bvalid;
    assign do_write       = aw_pend & w_pend;

    // AW channel
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_pend   <= 1'b0;
            aw_addr_q <= '0;
        end else begin
            if (s_csr.awvalid & s_csr.awready) begin
                aw_pend   <= 1'b1;
                aw_addr_q <= s_csr.awaddr;
            end else if (do_write) begin
                aw_pend <= 1'b0;
            end
        end
    end

    // W channel
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_pend   <= 1'b0;
            w_data_q <= '0;
            w_strb_q <= '0;
        end else begin
            if (s_csr.wvalid & s_csr.wready) begin
                w_pend   <= 1'b1;
                w_data_q <= s_csr.wdata;
                w_strb_q <= s_csr.wstrb;
            end else if (do_write) begin
                w_pend <= 1'b0;
            end
        end
    end

    // Write response
    logic write_ok;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_csr.bvalid <= 1'b0;
            s_csr.bresp  <= 2'b00;
        end else begin
            if (do_write) begin
                s_csr.bvalid <= 1'b1;
                s_csr.bresp  <= write_ok ? 2'b00 : 2'b10;
            end else if (s_csr.bvalid & s_csr.bready) begin
                s_csr.bvalid <= 1'b0;
            end
        end
    end

    // =========================================================================
    // Shadow Register Write Logic
    // =========================================================================
    // Decode address and write to appropriate shadow register array

    logic [CHANNELS_W-1:0] channel_idx;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Initialize BN1 shadow registers
            for (int c = 0; c < CHANNELS; c++) begin
                bn1_gamma_shadow[c]   <= ACC_WIDTH'(1 << FRAC_WIDTH);  // gamma = 1
                bn1_beta_shadow[c]    <= '0;  // beta = 0
                bn1_mean_shadow[c]     <= '0;  // mean = 0
                bn1_inv_std_shadow[c]  <= ACC_WIDTH'(1 << FRAC_WIDTH);  // inv_std = 1
            end
            // BN2 shadow registers
            for (int c = 0; c < CHANNELS; c++) begin
                bn2_gamma_shadow[c]   <= ACC_WIDTH'(1 << FRAC_WIDTH);
                bn2_beta_shadow[c]    <= '0;
                bn2_mean_shadow[c]     <= '0;
                bn2_inv_std_shadow[c]  <= ACC_WIDTH'(1 << FRAC_WIDTH);
            end
            // BN3 shadow registers
            for (int c = 0; c < CHANNELS; c++) begin
                bn3_gamma_shadow[c]   <= ACC_WIDTH'(1 << FRAC_WIDTH);
                bn3_beta_shadow[c]    <= '0;
                bn3_mean_shadow[c]     <= '0;
                bn3_inv_std_shadow[c]  <= ACC_WIDTH'(1 << FRAC_WIDTH);
            end
        end else if (do_write) begin
            // Extract channel index from address (bits [4:2])
            // Note: channel_idx is combinational (always_comb at line 344)

            case (aw_addr_q[11:5])
                // BN1 parameters (0x500-0x55F) — mask channel bits [4:2]
                CSR_SSF_BN1_GAMMA_BASE[11:5]: begin
                    if (channel_idx < CHANNELS)
                        for (int b = 0; b < 4; b++)
                            if (w_strb_q[b])
                                bn1_gamma_shadow[channel_idx][b*8 +: 8] <= w_data_q[b*8 +: 8];
                end
                CSR_SSF_BN1_BETA_BASE[11:5]: begin
                    if (channel_idx < CHANNELS)
                        for (int b = 0; b < 4; b++)
                            if (w_strb_q[b])
                                bn1_beta_shadow[channel_idx][b*8 +: 8] <= w_data_q[b*8 +: 8];
                end
                CSR_SSF_BN1_MEAN_BASE[11:5]: begin
                    if (channel_idx < CHANNELS)
                        for (int b = 0; b < 4; b++)
                            if (w_strb_q[b])
                                bn1_mean_shadow[channel_idx][b*8 +: 8] <= w_data_q[b*8 +: 8];
                end
                CSR_SSF_BN1_INV_STD_BASE[11:5]: begin
                    if (channel_idx < CHANNELS)
                        for (int b = 0; b < 4; b++)
                            if (w_strb_q[b])
                                bn1_inv_std_shadow[channel_idx][b*8 +: 8] <= w_data_q[b*8 +: 8];
                end

                // BN2 parameters (0x580-0x5DF)
                CSR_SSF_BN2_GAMMA_BASE[11:5]: begin
                    if (channel_idx < CHANNELS)
                        for (int b = 0; b < 4; b++)
                            if (w_strb_q[b])
                                bn2_gamma_shadow[channel_idx][b*8 +: 8] <= w_data_q[b*8 +: 8];
                end
                CSR_SSF_BN2_BETA_BASE[11:5]: begin
                    if (channel_idx < CHANNELS)
                        for (int b = 0; b < 4; b++)
                            if (w_strb_q[b])
                                bn2_beta_shadow[channel_idx][b*8 +: 8] <= w_data_q[b*8 +: 8];
                end
                CSR_SSF_BN2_MEAN_BASE[11:5]: begin
                    if (channel_idx < CHANNELS)
                        for (int b = 0; b < 4; b++)
                            if (w_strb_q[b])
                                bn2_mean_shadow[channel_idx][b*8 +: 8] <= w_data_q[b*8 +: 8];
                end
                CSR_SSF_BN2_INV_STD_BASE[11:5]: begin
                    if (channel_idx < CHANNELS)
                        for (int b = 0; b < 4; b++)
                            if (w_strb_q[b])
                                bn2_inv_std_shadow[channel_idx][b*8 +: 8] <= w_data_q[b*8 +: 8];
                end

                // BN3 parameters (0x600-0x65F)
                CSR_SSF_BN3_GAMMA_BASE[11:5]: begin
                    if (channel_idx < CHANNELS)
                        for (int b = 0; b < 4; b++)
                            if (w_strb_q[b])
                                bn3_gamma_shadow[channel_idx][b*8 +: 8] <= w_data_q[b*8 +: 8];
                end
                CSR_SSF_BN3_BETA_BASE[11:5]: begin
                    if (channel_idx < CHANNELS)
                        for (int b = 0; b < 4; b++)
                            if (w_strb_q[b])
                                bn3_beta_shadow[channel_idx][b*8 +: 8] <= w_data_q[b*8 +: 8];
                end
                CSR_SSF_BN3_MEAN_BASE[11:5]: begin
                    if (channel_idx < CHANNELS)
                        for (int b = 0; b < 4; b++)
                            if (w_strb_q[b])
                                bn3_mean_shadow[channel_idx][b*8 +: 8] <= w_data_q[b*8 +: 8];
                end
                CSR_SSF_BN3_INV_STD_BASE[11:5]: begin
                    if (channel_idx < CHANNELS)
                        for (int b = 0; b < 4; b++)
                            if (w_strb_q[b])
                                bn3_inv_std_shadow[channel_idx][b*8 +: 8] <= w_data_q[b*8 +: 8];
                end

                default: ;
            endcase
        end else if (cfg_update) begin
            // Commit shadow registers to live registers
            for (int c = 0; c < CHANNELS; c++) begin
                bn1_gamma_live[c]   <= bn1_gamma_shadow[c];
                bn1_beta_live[c]    <= bn1_beta_shadow[c];
                bn1_mean_live[c]     <= bn1_mean_shadow[c];
                bn1_inv_std_live[c]  <= bn1_inv_std_shadow[c];

                bn2_gamma_live[c]   <= bn2_gamma_shadow[c];
                bn2_beta_live[c]    <= bn2_beta_shadow[c];
                bn2_mean_live[c]     <= bn2_mean_shadow[c];
                bn2_inv_std_live[c]  <= bn2_inv_std_shadow[c];

                bn3_gamma_live[c]   <= bn3_gamma_shadow[c];
                bn3_beta_live[c]    <= bn3_beta_shadow[c];
                bn3_mean_live[c]     <= bn3_mean_shadow[c];
                bn3_inv_std_live[c]  <= bn3_inv_std_shadow[c];
            end
        end
    end

    // =========================================================================
    // Write OK Decode
    // =========================================================================
    always_comb begin
        write_ok = 1'b0;
        case (aw_addr_q[11:0])
            CSR_SSF_BN1_GAMMA_BASE[11:0],
            CSR_SSF_BN1_BETA_BASE[11:0],
            CSR_SSF_BN1_MEAN_BASE[11:0],
            CSR_SSF_BN1_INV_STD_BASE[11:0],
            CSR_SSF_BN2_GAMMA_BASE[11:0],
            CSR_SSF_BN2_BETA_BASE[11:0],
            CSR_SSF_BN2_MEAN_BASE[11:0],
            CSR_SSF_BN2_INV_STD_BASE[11:0],
            CSR_SSF_BN3_GAMMA_BASE[11:0],
            CSR_SSF_BN3_BETA_BASE[11:0],
            CSR_SSF_BN3_MEAN_BASE[11:0],
            CSR_SSF_BN3_INV_STD_BASE[11:5]: begin
                write_ok = (channel_idx < CHANNELS);
            end
            default: write_ok = 1'b0;
        endcase
    end

    // =========================================================================
    // Read Path
    // =========================================================================
    logic [DATA_W-1:0] rd_data_mux;
    logic read_ok;

    assign s_csr.arready = ~s_csr.rvalid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_csr.rvalid <= 1'b0;
            s_csr.rdata  <= '0;
            s_csr.rresp  <= 2'b00;
        end else if (s_csr.arvalid & s_csr.arready) begin
            s_csr.rvalid <= 1'b1;
            s_csr.rdata  <= rd_data_mux;
            s_csr.rresp  <= read_ok ? 2'b00 : 2'b10;
        end else if (s_csr.rvalid & s_csr.rready) begin
            s_csr.rvalid <= 1'b0;
        end
    end

    always_comb begin
        rd_data_mux = 32'h0;
        read_ok     = 1'b1;

        // Extract channel index for read
        channel_idx = s_csr.araddr[4:2];

        case (s_csr.araddr[11:0])
            CSR_SSF_BN1_GAMMA_BASE[11:0]: begin
                if (channel_idx < CHANNELS)
                    rd_data_mux = bn1_gamma_shadow[channel_idx];
                else
                    read_ok = 1'b0;
            end
            CSR_SSF_BN1_BETA_BASE[11:5]: begin
                if (channel_idx < CHANNELS)
                    rd_data_mux = bn1_beta_shadow[channel_idx];
                else
                    read_ok = 1'b0;
            end
            CSR_SSF_BN1_MEAN_BASE[11:5]: begin
                if (channel_idx < CHANNELS)
                    rd_data_mux = bn1_mean_shadow[channel_idx];
                else
                    read_ok = 1'b0;
            end
            CSR_SSF_BN1_INV_STD_BASE[11:5]: begin
                if (channel_idx < CHANNELS)
                    rd_data_mux = bn1_inv_std_shadow[channel_idx];
                else
                    read_ok = 1'b0;
            end
            CSR_SSF_BN2_GAMMA_BASE[11:5]: begin
                if (channel_idx < CHANNELS)
                    rd_data_mux = bn2_gamma_shadow[channel_idx];
                else
                    read_ok = 1'b0;
            end
            CSR_SSF_BN2_BETA_BASE[11:5]: begin
                if (channel_idx < CHANNELS)
                    rd_data_mux = bn2_beta_shadow[channel_idx];
                else
                    read_ok = 1'b0;
            end
            CSR_SSF_BN2_MEAN_BASE[11:5]: begin
                if (channel_idx < CHANNELS)
                    rd_data_mux = bn2_mean_shadow[channel_idx];
                else
                    read_ok = 1'b0;
            end
            CSR_SSF_BN2_INV_STD_BASE[11:5]: begin
                if (channel_idx < CHANNELS)
                    rd_data_mux = bn2_inv_std_shadow[channel_idx];
                else
                    read_ok = 1'b0;
            end
            CSR_SSF_BN3_GAMMA_BASE[11:5]: begin
                if (channel_idx < CHANNELS)
                    rd_data_mux = bn3_gamma_shadow[channel_idx];
                else
                    read_ok = 1'b0;
            end
            CSR_SSF_BN3_BETA_BASE[11:5]: begin
                if (channel_idx < CHANNELS)
                    rd_data_mux = bn3_beta_shadow[channel_idx];
                else
                    read_ok = 1'b0;
            end
            CSR_SSF_BN3_MEAN_BASE[11:5]: begin
                if (channel_idx < CHANNELS)
                    rd_data_mux = bn3_mean_shadow[channel_idx];
                else
                    read_ok = 1'b0;
            end
            CSR_SSF_BN3_INV_STD_BASE[11:5]: begin
                if (channel_idx < CHANNELS)
                    rd_data_mux = bn3_inv_std_shadow[channel_idx];
                else
                    read_ok = 1'b0;
            end
            default: begin
                rd_data_mux = 32'hDEAD_C0DE;
                read_ok = 1'b0;
            end
        endcase
    end

    // =========================================================================
    // Sign-Extension Adapter (SSF-WIDTH-MISMATCH-001)
    // =========================================================================
    // Input ports are DATA_WIDTH (8-bit), ssf_module expects ACC_WIDTH (32-bit).
    // Sign-extend narrow inputs to wide internal signals.
    logic signed [ACC_WIDTH-1:0] x_1x1_wide   [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] x_3x3_wide   [CHANNELS-1:0][26:0];
    logic signed [ACC_WIDTH-1:0] w_1x1_wide   [CHANNELS-1:0];
    logic signed [ACC_WIDTH-1:0] w_3x3_wide   [CHANNELS-1:0][26:0];
    logic signed [ACC_WIDTH-1:0] w_linear_wide [CHANNELS-1:0][CHANNELS-1:0];

    always_comb begin
        for (int c = 0; c < CHANNELS; c++) begin
            x_1x1_wide[c] = ACC_WIDTH'(x_1x1[c]);
            w_1x1_wide[c] = ACC_WIDTH'(w_1x1[c]);
            for (int k = 0; k < 27; k++) begin
                x_3x3_wide[c][k] = ACC_WIDTH'(x_3x3[c][k]);
                w_3x3_wide[c][k] = ACC_WIDTH'(w_3x3[c][k]);
            end
            for (int c2 = 0; c2 < CHANNELS; c2++)
                w_linear_wide[c][c2] = ACC_WIDTH'(w_linear[c][c2]);
        end
    end

    // =========================================================================
    // SSF Module Instantiation
    // =========================================================================
    ssf_module #(
        .DATA_WIDTH (ACC_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH),
        .FRAC_WIDTH (FRAC_WIDTH),
        .CHANNELS   (CHANNELS)
    ) u_ssf (
        .clk        (clk),
        .rst_n      (rst_n),
        .in_valid   (in_valid),
        .out_valid  (out_valid),
        .ready_in   (1'b1),
        .ready_out  (),
        .last_in    (1'b0),
        .last_out   (),

        .x_1x1      (x_1x1_wide),
        .x_3x3      (x_3x3_wide),
        .w_1x1      (w_1x1_wide),
        .w_3x3      (w_3x3_wide),
        .w_linear   (w_linear_wide),

        .bn1_gamma   (bn1_gamma_live),
        .bn1_beta    (bn1_beta_live),
        .bn1_mean    (bn1_mean_live),
        .bn1_inv_std (bn1_inv_std_live),

        .bn2_gamma   (bn2_gamma_live),
        .bn2_beta    (bn2_beta_live),
        .bn2_mean    (bn2_mean_live),
        .bn2_inv_std (bn2_inv_std_live),

        .bn3_gamma   (bn3_gamma_live),
        .bn3_beta    (bn3_beta_live),
        .bn3_mean    (bn3_mean_live),
        .bn3_inv_std (bn3_inv_std_live),

        .y_out      (y_out)
    );

    // cfg_busy output
    assign s_csr.cfg_busy = cfg_update;

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    SSF_CSR_CHANNEL_RANGE: assert property (
        @(posedge clk) disable iff (!rst_n)
        CHANNELS > 0 && CHANNELS <= 256
    ) else $error("ssf_csr_wrapper: Invalid channel count");

    SSF_CSR_WRITE_OK: assert property (
        @(posedge clk) disable iff (!rst_n)
        do_write |-> write_ok || aw_addr_q[11:8] == 4'h5 || aw_addr_q[11:8] == 4'h6
    ) else $error("ssf_csr_wrapper: Invalid write address");
    // synthesis translate_on

endmodule