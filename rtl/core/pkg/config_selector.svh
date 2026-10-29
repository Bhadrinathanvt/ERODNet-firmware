// =============================================================================
// config_selector.svh
//
// Configuration selector include file.
// Include this file at the top of your design to select a configuration.
//
// Usage in Makefile:
//   # For FULL configuration (default)
//   DEFINES += +define+CONFIG_FULL
//
//   # For BALANCED configuration
//   DEFINES += +define+CONFIG_BALANCED
//
//   # For MINIMAL configuration (15% LUT target)
//   DEFINES += +define+CONFIG_MINIMAL
//
// Usage in Verilator:
//   verilator --define CONFIG_MINIMAL ...
//   verilator --define CONFIG_BALANCED ...
//   verilator --define CONFIG_FULL ...
//
// Usage in Vivado:
//   set_property generic {CONFIG_MINIMAL} [current_fileset]
//   Or add to synthesis options: -define CONFIG_MINIMAL
// =============================================================================

// Default to FULL if no configuration specified
`ifndef CONFIG_FULL
    `ifndef CONFIG_BALANCED
        `ifndef CONFIG_MINIMAL
            `ifndef CONFIG_ARTY_A7
                `define CONFIG_FULL
            `endif
        `endif
    `endif
`endif

// Display selected configuration
`ifdef CONFIG_FULL
    `define CONFIG_PROFILE "FULL"
    `define CONFIG_PROFILE_NUM 3
`elsif CONFIG_BALANCED
    `define CONFIG_PROFILE "BALANCED"
    `define CONFIG_PROFILE_NUM 2
`elsif CONFIG_MINIMAL
    `define CONFIG_PROFILE "MINIMAL"
    `define CONFIG_PROFILE_NUM 1
`elsif CONFIG_ARTY_A7
    `define CONFIG_PROFILE "ARTY_A7"
    `define CONFIG_PROFILE_NUM 4
`else
    `define CONFIG_PROFILE "FULL"
    `define CONFIG_PROFILE_NUM 3
`endif

// =============================================================================
// Resource Estimates (for synthesis-time checking)
// =============================================================================

`ifdef CONFIG_FULL
    `define EST_DSP_COUNT    96
    `define EST_LUT_COUNT    48300
    `define EST_BRAM_COUNT   72
    `define EST_LUT_PERCENT  76
`elsif CONFIG_BALANCED
    `define EST_DSP_COUNT    48
    `define EST_LUT_COUNT    25000
    `define EST_BRAM_COUNT   40
    `define EST_LUT_PERCENT  40
`elsif CONFIG_MINIMAL
    `define EST_DSP_COUNT    16
    `define EST_LUT_COUNT    7200
    `define EST_BRAM_COUNT   18
    `define EST_LUT_PERCENT  11
`elsif CONFIG_ARTY_A7
    `define EST_DSP_COUNT    216
    `define EST_LUT_COUNT    20288
    `define EST_BRAM_COUNT   115
    `define EST_LUT_PERCENT  20
`endif

// =============================================================================
// Timing Constraints (for synthesis)
// =============================================================================

// Clock periods (200MHz target = 5ns)
`define CLK_PERIOD_NS    5
`define CLK_PERIOD_PS    5000

// Critical path margins
`ifdef CONFIG_FULL
    `define TIMING_MARGIN_NS   1.5   // Aggressive, minimal margin
`elsif CONFIG_BALANCED
    `define TIMING_MARGIN_NS   2.0   // Moderate margin
`elsif CONFIG_MINIMAL
    `define TIMING_MARGIN_NS   2.5   // Generous margin for safety
`elsif CONFIG_ARTY_A7
    `define TIMING_MARGIN_NS   2.5   // Generous margin — smaller design closes timing easily
`endif

// =============================================================================
// Synthesis Directives
// =============================================================================

// DSP inference control
`ifdef CONFIG_FULL
    `define SYSTOLIC_USE_DSP   1   // Systolic array uses DSP
    `define NORM_USE_DSP       0   // LUT-based
    `define ACT_USE_DSP        0   // LUT-based
    `define GFF_USE_DSP        0   // LUT-based
`elsif CONFIG_BALANCED
    `define SYSTOLIC_USE_DSP   1   // Systolic uses DSP
    `define NORM_USE_DSP       0   // LUT-based
    `define ACT_USE_DSP        0   // LUT-based
    `define GFF_USE_DSP        0   // LUT-based
`elsif CONFIG_MINIMAL
    `define SYSTOLIC_USE_DSP   1   // Systolic uses DSP
    `define NORM_USE_DSP       0   // LUT-based
    `define ACT_USE_DSP        0   // LUT-based
    `define GFF_USE_DSP        0   // LUT-based
`elsif CONFIG_ARTY_A7
    `define SYSTOLIC_USE_DSP   1   // Systolic uses DSP
    `define NORM_USE_DSP       1   // DSP-based (push math onto DSP48Es, save LUTs)
    `define ACT_USE_DSP        1   // DSP-based (activation interpolation)
    `define GFF_USE_DSP        1   // DSP-based (GFF linear + l2norm)
`endif

// RAM style
`ifdef CONFIG_FULL
    `define WEIGHT_RAM_STYLE   "block"
    `define FIFO_RAM_STYLE     "block"
`elsif CONFIG_BALANCED
    `define WEIGHT_RAM_STYLE   "block"
    `define FIFO_RAM_STYLE     "block"
`elsif CONFIG_MINIMAL
    `define WEIGHT_RAM_STYLE   "distributed"
    `define FIFO_RAM_STYLE     "distributed"
`elsif CONFIG_ARTY_A7
    `define WEIGHT_RAM_STYLE   "block"
    `define FIFO_RAM_STYLE     "block"
`endif

// =============================================================================
// Print Configuration (synthesis info)
// =============================================================================

// synthesis translate_off
module config_selector_info;
    initial begin
        $display("========================================================");
        $display("eRODNet Configuration: %s", `CONFIG_PROFILE);
        $display("========================================================");
        $display("Estimated Resources:");
        $display("  DSP48:  %0d", `EST_DSP_COUNT);
        $display("  LUT:    %0d (%0d%% of 63,400)", `EST_LUT_COUNT, `EST_LUT_PERCENT);
        $display("  BRAM36: %0d", `EST_BRAM_COUNT);
        $display("");
        $display("Timing:  %0d ns period, %0d ns margin", `CLK_PERIOD_NS, `TIMING_MARGIN_NS);
        $display("========================================================");
    end
endmodule
// synthesis translate_on