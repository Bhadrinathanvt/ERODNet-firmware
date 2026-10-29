// =============================================================================
// shared_activation_roms.sv
//
// Shared activation function ROMs for the unified_activation array.
//
// Replaces per-lane private ROMs in sigmoid.sv / GELU.sv with a single
// instance per activation type.  One read port per lane — the module
// computes the LUT index from each lane's x_in and returns the
// corresponding y_base / slope pair.
//
// Index extraction:
//   The ROM is generated for a 32-bit Q16.16 fixed-point input (ROM_WIDTH=32,
//   INDEX_START_BIT=12).  The x_in port carries narrow DATA_WIDTH-bit values
//   (e.g., INT8).  Before extracting the index, each lane is sign-extended
//   to ROM_WIDTH and shifted left by FRAC_WIDTH to convert to Q16.16 format.
//
// Output format:
//   y_base and slope outputs are in ROM_WIDTH=32-bit Q16.16 format, matching
//   the ROM storage width.  The activation submodules (sigmoid, GELU) perform
//   interpolation in this width and truncate to DATA_WIDTH at the output.
//
// ROM_DATA_BITS:
//   Must equal ROM_WIDTH (32) for correct $readmemh loading.  Previous
//   ROM_DATA_BITS=6 truncated all ROM values to zero — this was the root
//   cause of BUG-EROD-ACT-001 (all activation outputs zero).
// =============================================================================

import pkg_config::*;

module shared_activation_roms #(
    parameter int DATA_WIDTH    = 32,
    parameter int FRAC_WIDTH    = 16,
    parameter int INDEX_BITS    = pkg_config::INDEX_BITS,
    parameter int NUM_PORTS     = 16,           // M×N lanes
    parameter int ROM_DATA_BITS = 32,           // Must be 32 for Q16.16 ROM values
    parameter bit EN_SIGMOID    = 1,
    parameter bit EN_GELU       = 1
)(
    input  logic clk,
    input  logic rst_n,

    // One x_in per lane → index computed internally
    input  logic signed [DATA_WIDTH-1:0] x_in [0:NUM_PORTS-1],

    // Sigmoid outputs (one per lane) — ROM_WIDTH = Q16.16 format
    output logic signed [ROM_DATA_BITS-1:0] sigmoid_y_base [0:NUM_PORTS-1],
    output logic signed [ROM_DATA_BITS-1:0] sigmoid_slope  [0:NUM_PORTS-1],

    // GELU outputs (one per lane) — ROM_WIDTH = Q16.16 format
    output logic signed [ROM_DATA_BITS-1:0] gelu_y_base [0:NUM_PORTS-1],
    output logic signed [ROM_DATA_BITS-1:0] gelu_slope  [0:NUM_PORTS-1]
);

    // ROM internal width: Q16.16 format for piecewise-linear interpolation.
    // DATA_WIDTH may be narrower (e.g., 8-bit INT8).  We sign-extend and
    // shift each lane to ROM_WIDTH before extracting the LUT index.
    // ROM_WIDTH/INDEX_START_BIT come from pkg_config (config.yaml::activation_rom).
    // Tied to .mem file geometry — regenerate via scripts/generate_roms.py if changed.
    localparam int ROM_WIDTH       = pkg_config::ROM_WIDTH;
    localparam int INDEX_START_BIT = pkg_config::INDEX_START_BIT;
    localparam int ROM_ENTRIES     = 1 << INDEX_BITS;

    // Scale each lane from DATA_WIDTH to ROM_WIDTH (Q16.16).
    // Sign-extend then shift left by FRAC_WIDTH to place fractional bits.
    logic signed [ROM_WIDTH-1:0] x_in_scaled [0:NUM_PORTS-1];

    always_comb begin
        for (int p = 0; p < NUM_PORTS; p++) begin
            x_in_scaled[p] = { {(ROM_WIDTH-DATA_WIDTH){x_in[p][DATA_WIDTH-1]}}, x_in[p] } <<< FRAC_WIDTH;
        end
    end

    // =========================================================================
    // Sigmoid ROMs (shared, read-only)
    // =========================================================================
    generate
        if (EN_SIGMOID) begin : gen_sigmoid_rom
            (* rom_style = "block" *)
            logic signed [ROM_DATA_BITS-1:0] rom_sig_y_base [0:ROM_ENTRIES-1];
            (* rom_style = "block" *)
            logic signed [ROM_DATA_BITS-1:0] rom_sig_slope  [0:ROM_ENTRIES-1];

            initial begin
                $readmemh("rtl/sigmoid_y_base.mem",  rom_sig_y_base);
                $readmemh("rtl/sigmoid_m_slope.mem", rom_sig_slope);
            end

            always_comb begin
                for (int p = 0; p < NUM_PORTS; p++) begin
                    logic [INDEX_BITS-1:0] idx;
                    idx = x_in_scaled[p][INDEX_START_BIT+INDEX_BITS-1 : INDEX_START_BIT];
                    sigmoid_y_base[p] = rom_sig_y_base[idx];
                    sigmoid_slope [p] = rom_sig_slope [idx];
                end
            end
        end else begin : gen_no_sigmoid_rom
            always_comb begin
                for (int p = 0; p < NUM_PORTS; p++) begin
                    sigmoid_y_base[p] = '0;
                    sigmoid_slope [p] = '0;
                end
            end
        end
    endgenerate

    // =========================================================================
    // GELU ROMs (shared, read-only)
    // =========================================================================
    generate
        if (EN_GELU) begin : gen_gelu_rom
            (* rom_style = "block" *)
            logic signed [ROM_DATA_BITS-1:0] rom_gelu_y_base [0:ROM_ENTRIES-1];
            (* rom_style = "block" *)
            logic signed [ROM_DATA_BITS-1:0] rom_gelu_slope  [0:ROM_ENTRIES-1];

            initial begin
                $readmemh("rtl/gelu_y_base.mem",  rom_gelu_y_base);
                $readmemh("rtl/gelu_m_slope.mem", rom_gelu_slope);
            end

            always_comb begin
                for (int p = 0; p < NUM_PORTS; p++) begin
                    logic [INDEX_BITS-1:0] idx;
                    idx = x_in_scaled[p][INDEX_START_BIT+INDEX_BITS-1 : INDEX_START_BIT];
                    gelu_y_base[p] = rom_gelu_y_base[idx];
                    gelu_slope [p] = rom_gelu_slope [idx];
                end
            end
        end else begin : gen_no_gelu_rom
            always_comb begin
                for (int p = 0; p < NUM_PORTS; p++) begin
                    gelu_y_base[p] = '0;
                    gelu_slope [p] = '0;
                end
            end
        end
    endgenerate

endmodule