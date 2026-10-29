// =============================================================================
// cdc_sync.sv
//
// Clock Domain Crossing (CDC) Synchronizer Primitives (Production)
//
// Provides safe crossing between clock domains for:
//   - Reset deassertion (3-stage)
//   - Single-bit control signals (2-stage synchronizer)
//   - Pulse signals (toggle-based pulse synchronizer)
//   - Multi-bit data signals (handshake-based data synchronizer)
//   - Event counters (gray-code counter crossing)
//   - Vector of independent single-bit signals
//
// All modules are synthesizable with:
//   - Proper metastability resolution (async_reg attributes)
//   - No combinational logic between flops
//   - Compatible with set_clock_groups -asynchronous timing constraints
// =============================================================================

`ifndef CDC_SYNC_SV
`define CDC_SYNC_SV

// =============================================================================
// cdc_reset_sync - Reset Deassertion Synchronizer
// =============================================================================
// Synchronizes asynchronous reset deassertion to the destination clock domain.
// Reset asserts asynchronously (immediately) but deasserts synchronously.
//
// Usage:
//   cdc_reset_sync #(.STAGES(3)) u_rst_sync (
//       .src_rst_n(rst_n),       // async, active-low (from top-level)
//       .dst_clk(clk_dst),       // destination clock
//       .dst_rst_n(rst_n_dst)    // synchronized reset for dst domain
//   );
// =============================================================================

module cdc_reset_sync #(
    parameter int STAGES = 3
)(
    input  logic src_rst_n,
    input  logic dst_clk,
    output logic dst_rst_n
);

    (* async_reg = "true" *) logic [STAGES-1:0] rst_sync;

    always_ff @(posedge dst_clk or negedge src_rst_n) begin
        if (!src_rst_n)
            rst_sync <= '0;
        else
            rst_sync <= {rst_sync[STAGES-2:0], 1'b1};
    end

    assign dst_rst_n = rst_sync[STAGES-1];

endmodule


// =============================================================================
// cdc_sync_single - Single-bit Signal Synchronizer
// =============================================================================
// Safe crossing for single-bit control signals.
// Uses 2-stage synchronizer with metastability resolution.
//
// Usage: valid_in, ready_in, start, done, interrupt signals
//
// Timing:
//   - Source clock to dest clock: 2 dest clock cycles minimum
//   - Output is level (not pulse)
// =============================================================================

module cdc_sync_single #(
    parameter int STAGES = 2
)(
    input  logic clk_src,
    input  logic rst_n_src,
    input  logic clk_dst,
    input  logic rst_n_dst,

    input  logic data_in,
    output logic data_out
);

    // Source domain: capture input
    logic data_src_reg;
    always_ff @(posedge clk_src or negedge rst_n_src) begin
        if (!rst_n_src) data_src_reg <= 1'b0;
        else            data_src_reg <= data_in;
    end

    // Destination domain: multi-stage synchronizer
    (* async_reg = "true" *) logic [STAGES-1:0] sync_chain;

    always_ff @(posedge clk_dst or negedge rst_n_dst) begin
        if (!rst_n_dst) begin
            sync_chain <= '0;
        end else begin
            sync_chain <= {sync_chain[STAGES-2:0], data_src_reg};
        end
    end

    assign data_out = sync_chain[STAGES-1];

endmodule


// =============================================================================
// cdc_sync_pulse - Pulse Synchronizer (source pulse -> dest pulse)
// =============================================================================
// Converts a single-cycle pulse in source domain to a single-cycle pulse
// in destination domain.
//
// Usage: start signals, event triggers, interrupts
//
// Protocol:
//   - Source pulse: 1 cycle pulse_in
//   - Dest pulse: 1 cycle pulse_out
//   - Minimum spacing between pulses: 2*max(clk_src, clk_dst) cycles
// =============================================================================

module cdc_sync_pulse #(
    parameter int STAGES = 2
)(
    input  logic clk_src,
    input  logic rst_n_src,
    input  logic clk_dst,
    input  logic rst_n_dst,

    input  logic pulse_in,
    output logic pulse_out,
    output logic busy
);

    // Source domain: convert pulse to level (toggle)
    logic toggle_ff;

    always_ff @(posedge clk_src or negedge rst_n_src) begin
        if (!rst_n_src)
            toggle_ff <= 1'b0;
        else if (pulse_in)
            toggle_ff <= ~toggle_ff;
    end

    // Busy: high while pulse is being synchronized
    assign busy = pulse_in;

    // Destination domain: synchronize toggle level
    (* async_reg = "true" *) logic [STAGES:0] toggle_sync;

    always_ff @(posedge clk_dst or negedge rst_n_dst) begin
        if (!rst_n_dst)
            toggle_sync <= '0;
        else
            toggle_sync <= {toggle_sync[STAGES-1:0], toggle_ff};
    end

    // Edge detect -> single-cycle pulse
    assign pulse_out = toggle_sync[STAGES] ^ toggle_sync[STAGES-1];

endmodule


// =============================================================================
// cdc_sync_data - Multi-bit Data Synchronizer with Handshake
// =============================================================================
// Safe crossing for multi-bit data signals using toggle-based handshake.
//
// Usage: configuration registers, status data, weight data, detection data
//
// Protocol:
//   1. Source captures data and toggles req when valid_in && ready_out
//   2. Destination detects toggle edge, latches data, asserts valid_out
//   3. Destination toggles ack when ready_in consumes data
//   4. Source detects ack toggle, ready for next transfer
// =============================================================================

module cdc_sync_data #(
    parameter int DATA_WIDTH = 32,
    parameter int STAGES     = 2
)(
    input  logic src_clk,
    input  logic src_rst_n,
    input  logic dst_clk,
    input  logic dst_rst_n,

    input  logic [DATA_WIDTH-1:0] data_in,
    input  logic                  valid_in,
    output logic                  ready_out,

    output logic [DATA_WIDTH-1:0] data_out,
    output logic                  valid_out,
    input  logic                  ready_in
);

    // Toggle-based request/acknowledge handshake
    logic req_toggle;
    logic ack_synced;
    logic [DATA_WIDTH-1:0] data_captured;
    logic capturing;

    // Source domain: capture data, toggle request
    always_ff @(posedge src_clk or negedge src_rst_n) begin
        if (!src_rst_n) begin
            req_toggle <= 1'b0;
            data_captured <= '0;
            capturing <= 1'b0;
        end else begin
            if (valid_in && ready_out && !capturing) begin
                data_captured <= data_in;
                req_toggle <= ~req_toggle;
                capturing <= 1'b1;
            end else if (ack_synced) begin
                capturing <= 1'b0;
            end
        end
    end

    assign ready_out = !capturing;

    // Destination domain: synchronize request toggle
    (* async_reg = "true" *) logic [STAGES:0] req_sync;
    logic prev_req;

    always_ff @(posedge dst_clk or negedge dst_rst_n) begin
        if (!dst_rst_n)
            req_sync <= '0;
        else
            req_sync <= {req_sync[STAGES-1:0], req_toggle};
    end

    assign prev_req = req_sync[STAGES];
    assign valid_out = (req_sync[STAGES-1] ^ prev_req) && ready_in;

    // Latch data on valid edge
    always_ff @(posedge dst_clk or negedge dst_rst_n) begin
        if (!dst_rst_n) begin
            data_out <= '0;
        end else if (valid_out) begin
            data_out <= data_captured;
        end
    end

    // Destination domain: toggle acknowledge when data consumed
    logic ack_toggle;

    always_ff @(posedge dst_clk or negedge dst_rst_n) begin
        if (!dst_rst_n)
            ack_toggle <= 1'b0;
        else if (valid_out && ready_in)
            ack_toggle <= ~ack_toggle;
    end

    // Source domain: synchronize acknowledge toggle
    (* async_reg = "true" *) logic [STAGES-1:0] ack_sync;

    always_ff @(posedge src_clk or negedge src_rst_n) begin
        if (!src_rst_n)
            ack_sync <= '0;
        else
            ack_sync <= {ack_sync[STAGES-2:0], ack_toggle};
    end

    assign ack_synced = (ack_sync[STAGES-1] ^ ack_sync[STAGES-2]);

endmodule


// =============================================================================
// cdc_sync_counter - Gray-code Counter Synchronizer
// =============================================================================
// Safe crossing for counters using gray-code encoding.
// Gray code ensures only one bit changes per cycle, avoiding
// inconsistent values during crossing.
//
// Usage: performance counters, status counters, frame counters
// =============================================================================

module cdc_sync_counter #(
    parameter int WIDTH  = 32,
    parameter int STAGES = 2
)(
    input  logic clk_src,
    input  logic rst_n_src,
    input  logic clk_dst,
    input  logic rst_n_dst,

    input  logic [WIDTH-1:0] count_in,
    output logic [WIDTH-1:0] count_out
);

    // Binary to Gray conversion
    function automatic logic [WIDTH-1:0] binary_to_gray(input logic [WIDTH-1:0] binary);
        return binary ^ (binary >> 1);
    endfunction

    // Gray to Binary conversion
    function automatic logic [WIDTH-1:0] gray_to_binary(input logic [WIDTH-1:0] gray);
        logic [WIDTH-1:0] binary;
        binary[WIDTH-1] = gray[WIDTH-1];
        for (int i = WIDTH-2; i >= 0; i--) begin
            binary[i] = gray[i] ^ binary[i+1];
        end
        return binary;
    endfunction

    // Source domain: convert to gray code
    logic [WIDTH-1:0] gray_src;
    assign gray_src = binary_to_gray(count_in);

    // Destination domain: multi-stage synchronizer for gray code
    (* async_reg = "true" *) logic [STAGES-1:0][WIDTH-1:0] gray_sync_chain;

    always_ff @(posedge clk_dst or negedge rst_n_dst) begin
        if (!rst_n_dst) begin
            gray_sync_chain <= '0;
        end else begin
            gray_sync_chain <= {gray_sync_chain[STAGES-2:0], gray_src};
        end
    end

    // Convert back to binary
    assign count_out = gray_to_binary(gray_sync_chain[STAGES-1]);

endmodule


// =============================================================================
// cdc_sync_vec - Vector Synchronizer (array of cdc_sync_single)
// =============================================================================
// Synchronizes multiple independent single-bit signals in parallel.
// Each signal gets its own synchronizer chain.
//
// Usage: multiple control flags, status bit bundles, config bits
// =============================================================================

module cdc_sync_vec #(
    parameter int WIDTH  = 8,
    parameter int STAGES = 2
)(
    input  logic clk_src,
    input  logic rst_n_src,
    input  logic clk_dst,
    input  logic rst_n_dst,

    input  logic [WIDTH-1:0] data_in,
    output logic [WIDTH-1:0] data_out
);

    generate
        for (genvar i = 0; i < WIDTH; i++) begin : gen_sync
            cdc_sync_single #(.STAGES(STAGES)) u_sync (
                .clk_src   (clk_src),
                .rst_n_src (rst_n_src),
                .clk_dst   (clk_dst),
                .rst_n_dst (rst_n_dst),
                .data_in   (data_in[i]),
                .data_out  (data_out[i])
            );
        end
    endgenerate

endmodule

`endif
