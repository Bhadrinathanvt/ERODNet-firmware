// =============================================================================
// shadow_reg.sv
//
// Shadow register with glitch-free update on idle boundary.
// Two-phase update: write to shadow -> commit on safe boundary.
//
// Features:
// - Prevents glitches during configuration updates
// - Commit only when module is idle (idle_i=1)
// - Optional commit pulse (auto_commit=0)
// =============================================================================

module shadow_reg #(
    parameter int WIDTH = 32,
    parameter bit AUTO_COMMIT = 1,      // 1=auto commit when idle, 0=manual commit
    parameter bit SYNC_UPDATE = 0       // 1=synchronize update to clk domain
) (
    input  logic clk,
    input  logic rst_n,

    // Configuration interface
    input  logic              cfg_wr_i,     // Write to shadow register
    input  logic [WIDTH-1:0] cfg_data_i,   // New configuration value
    input  logic              cfg_commit_i, // Commit shadow to active (if AUTO_COMMIT=0)

    // Module status
    input  logic              idle_i,       // Module is idle (safe to update)

    // Output registers
    output logic [WIDTH-1:0] active_o,     // Active configuration
    output logic              updated_o     // Pulse when active register updated
);

    // ── Shadow (pending) and Active (current) registers ───────────────────────
    logic [WIDTH-1:0] shadow_q, active_q;
    logic              commit_pending_q;

    // ── Synchronizer for cross-domain updates ────────────────────────────────
    generate
        if (SYNC_UPDATE) begin : gen_sync
            logic [WIDTH-1:0] cfg_data_sync;
            logic              cfg_wr_sync;
            logic [2:0]        sync_ff;

            // 3-stage synchronizer for write pulse
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    sync_ff <= '0;
                    cfg_wr_sync <= '0;
                end else begin
                    sync_ff <= {sync_ff[1:0], cfg_wr_i};
                    cfg_wr_sync <= sync_ff[2];
                end
            end

            // Data synchronizer (assumes stable during sync)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    cfg_data_sync <= '0;
                end else if (cfg_wr_i) begin
                    cfg_data_sync <= cfg_data_i;
                end
            end

            // Shadow register update (synchronized domain)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    shadow_q <= '0;
                    commit_pending_q <= '0;
                end else if (cfg_wr_sync) begin
                    shadow_q <= cfg_data_sync;
                    commit_pending_q <= 1'b1;
                end else if (commit_pending_q && idle_i) begin
                    commit_pending_q <= 1'b0;
                end
            end
        end else begin : gen_nosync
            // Single clock domain
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    shadow_q <= '0;
                    commit_pending_q <= '0;
                end else if (cfg_wr_i) begin
                    shadow_q <= cfg_data_i;
                    commit_pending_q <= 1'b1;
                end else if (commit_pending_q && idle_i) begin
                    commit_pending_q <= 1'b0;
                end
            end
        end
    endgenerate

    // ── Commit logic ─────────────────────────────────────────────────────────
    logic commit_en;

    generate
        if (AUTO_COMMIT) begin : gen_auto_commit
            // Auto-commit when module is idle
            assign commit_en = commit_pending_q && idle_i;
        end else begin : gen_manual_commit
            // Manual commit pulse
            logic commit_pulse_sync;
            logic [2:0] commit_sync_ff;

            // Synchronize commit pulse (3-stage)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    commit_sync_ff <= '0;
                    commit_pulse_sync <= '0;
                end else begin
                    commit_sync_ff <= {commit_sync_ff[1:0], cfg_commit_i};
                    commit_pulse_sync <= commit_sync_ff[2];
                end
            end

            assign commit_en = commit_pulse_sync && commit_pending_q && idle_i;
        end
    endgenerate

    // ── Active register update ──────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_q <= '0;
            updated_o <= '0;
        end else if (commit_en) begin
            active_q <= shadow_q;
            updated_o <= 1'b1;
        end else begin
            updated_o <= 1'b0;
        end
    end

    assign active_o = active_q;

    // ── SystemVerilog Assertions ─────────────────────────────────────────────
    // Shadow register should only update when module is idle
    property shadow_update_only_when_idle;
        @(posedge clk) disable iff (!rst_n)
        (commit_en) |-> idle_i;
    endproperty

    SHADOW_UPDATE_ONLY_WHEN_IDLE: assert property (shadow_update_only_when_idle)
        else $error("Shadow register committed while module busy");

`ifndef VERILATOR
    // Configuration write should not be lost
    // (Unsupported in Verilator: ## range cycle delay)
    property cfg_write_eventually_commits;
        @(posedge clk) disable iff (!rst_n)
        cfg_wr_i |-> ##[1:1000] (active_o == cfg_data_i);
    endproperty

    CFG_WRITE_EVENTUALLY_COMMITS: assert property (cfg_write_eventually_commits)
        else $error("Configuration write not committed within timeout");
`endif

    // No glitches on active_o during commit
    // Note: commit_en is a combinational signal derived from commit_pending_q
    // which clears on the same edge as active_q updates, so $changed(active_o)
    // can fire when commit_en has already de-asserted. Use $past(commit_en) for
    // correct timing.
`ifndef VERILATOR
    property no_glitch_on_active;
        @(posedge clk) disable iff (!rst_n)
        $changed(active_o) |-> $past(commit_en);
    endproperty

    NO_GLITCH_ON_ACTIVE: assert property (no_glitch_on_active)
        else $error("Active register changed without commit enable");
`endif

endmodule