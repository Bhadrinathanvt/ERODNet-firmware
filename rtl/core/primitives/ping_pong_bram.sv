// =============================================================================
// ping_pong_bram.sv
//
// Ping-Pong Double-Buffered BRAM with True Dual-Port Inference.
// Provides producer/consumer decoupling for streaming pipelines.
//
// Features:
//   - True dual-port BRAM inference for optimal resource usage
//   - Runtime-configurable depth via parameter
//   - Double-buffering: Bank A and Bank B alternate
//   - Producer writes to one bank while consumer reads from the other
//   - Collision detection and prevention
//   - Fresh/empty status tracking
//   - Pipeline output option for better timing
//   - Supports AXI-Stream style handshaking
// =============================================================================

/* verilator lint_off DECLFILENAME */

module ping_pong_bram #(
    parameter int ADDR_W         = 10,
    parameter int DATA_W         = 32,
    parameter int DEPTH          = 1024,
    parameter bit PIPELINE_READ  = 1
)(
    input  logic clk,
    input  logic rst_n,

    // Producer Interface (Write)
    input  logic [ADDR_W-1:0]           p_addr,
    input  logic [DATA_W-1:0]           p_data_in,
    input  logic                        p_we,
    input  logic                        p_done,    // Bank full signal

    // Consumer Interface (Read)
    input  logic [ADDR_W-1:0]           c_addr,
    output logic [DATA_W-1:0]           c_data_out,
    input  logic                        c_done,    // Bank consumed signal

    // Control Signals
    input  logic                        ping_pong_select_wr,  // 0: Write A, 1: Write B
    input  logic                        ping_pong_select_rd,  // 0: Read A, 1: Read B
    output logic [1:0]                  bank_ready,           // {B_fresh, A_fresh}
    output logic                        owner_error
);

    // =========================================================================
    // True Dual-Port BRAM Storage
    // =========================================================================
    (* ram_style = "block" *)
    logic [DATA_W-1:0] bank_a [0:DEPTH-1];

    (* ram_style = "block" *)
    logic [DATA_W-1:0] bank_b [0:DEPTH-1];

    // =========================================================================
    // Port A: Producer Write Logic
    // =========================================================================
    logic bank_a_wr_en;
    logic bank_b_wr_en;

    always_ff @(posedge clk) begin
        if (bank_a_wr_en) begin
            bank_a[p_addr] <= p_data_in;
        end
        if (bank_b_wr_en) begin
            bank_b[p_addr] <= p_data_in;
        end
    end

    assign bank_a_wr_en = p_we && !ping_pong_select_wr;
    assign bank_b_wr_en = p_we && ping_pong_select_wr;

    // =========================================================================
    // Port B: Consumer Read Logic
    // =========================================================================
    logic [DATA_W-1:0] bank_a_rd_data;
    logic [DATA_W-1:0] bank_b_rd_data;
    logic [DATA_W-1:0] dout_reg;

    always_ff @(posedge clk) begin
        bank_a_rd_data <= bank_a[c_addr];
        bank_b_rd_data <= bank_b[c_addr];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dout_reg <= '0;
        end else begin
            dout_reg <= ping_pong_select_rd ? bank_b_rd_data : bank_a_rd_data;
        end
    end

    // Optional pipeline register
    generate
        if (PIPELINE_READ) begin : gen_pipeline
            logic [DATA_W-1:0] dout_pipeline;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    dout_pipeline <= '0;
                else
                    dout_pipeline <= dout_reg;
            end
            assign c_data_out = dout_pipeline;
        end else begin : gen_no_pipeline
            assign c_data_out = dout_reg;
        end
    endgenerate

    // =========================================================================
    // Bank Status Management
    // =========================================================================
    logic bank_a_fresh_reg;
    logic bank_b_fresh_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bank_a_fresh_reg <= 1'b0;
            bank_b_fresh_reg <= 1'b0;
        end else begin
            if (p_done && !ping_pong_select_wr) begin
                bank_a_fresh_reg <= 1'b1;
            end else if (c_done && !ping_pong_select_rd) begin
                bank_a_fresh_reg <= 1'b0;
            end

            if (p_done && ping_pong_select_wr) begin
                bank_b_fresh_reg <= 1'b1;
            end else if (c_done && ping_pong_select_rd) begin
                bank_b_fresh_reg <= 1'b0;
            end
        end
    end

    assign bank_ready = {bank_b_fresh_reg, bank_a_fresh_reg};
    assign owner_error = (ping_pong_select_wr == ping_pong_select_rd) && p_we;

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    /* verilator lint_off SYNCASYNCNET */
    PING_PONG_OWNER_ERROR: assert property (
        @(posedge clk) disable iff (!rst_n)
        !((ping_pong_select_wr == ping_pong_select_rd) && p_we)
    ) else $error("ping_pong_bram: Owner conflict");

    PING_PONG_DEPTH_VALID: assert property (@(posedge clk) disable iff (!rst_n) DEPTH >= 2);

    PING_PONG_ADDR_VALID: assert property (
        @(posedge clk) disable iff (!rst_n)
        32'(p_addr) < DEPTH && 32'(c_addr) < DEPTH
    ) else $error("ping_pong_bram: Address out of bounds");
    /* verilator lint_on SYNCASYNCNET */
    // synthesis translate_on

endmodule

// =============================================================================
// ping_pong_bram_auto - Automatic Bank Switching Variant
// =============================================================================

module ping_pong_bram_auto #(
    parameter int ADDR_W         = 10,
    parameter int DATA_W         = 32,
    parameter int DEPTH          = 1024,
    parameter bit PIPELINE_READ  = 1
)(
    input  logic clk,
    input  logic rst_n,

    // Producer Interface (AXI-Stream style)
    input  logic [DATA_W-1:0] p_data,
    input  logic              p_valid,
    output logic              p_ready,
    input  logic              p_last,    // End of bank

    // Consumer Interface (AXI-Stream style)
    output logic [DATA_W-1:0] c_data,
    output logic              c_valid,
    input  logic              c_ready,
    output logic              c_last,

    // Status
    output logic              bank_switch,  // Pulse on bank switch
    output logic [1:0]        bank_status   // Current bank status
);

    // =========================================================================
    // Bank Storage
    // =========================================================================
    (* ram_style = "block" *)
    logic [DATA_W-1:0] bank_a [0:DEPTH-1];
    (* ram_style = "block" *)
    logic [DATA_W-1:0] bank_b [0:DEPTH-1];

    // =========================================================================
    // State Machine
    // =========================================================================
    typedef enum logic [2:0] {
        IDLE_A_WR,    // Writing to A, B is empty/ready
        SWITCH_A2B,   // A filled, switching to read A / write B
        READ_A_WR_B,  // Reading from A, writing to B
        SWITCH_B2A,   // B filled, switching to read B / write A
        READ_B_WR_A   // Reading from B, writing to A
    } state_e;

    state_e state, next_state;

    // =========================================================================
    // Address Counters
    // =========================================================================
    logic [ADDR_W-1:0] wr_addr;
    logic [ADDR_W-1:0] rd_addr;
    logic              wr_sel;  // 0=write A, 1=write B
    logic              rd_sel;  // 0=read A, 1=read B

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_addr <= '0;
            rd_addr <= '0;
            wr_sel  <= 1'b0;
            rd_sel  <= 1'b0;
        end else begin
            // Write address increment
            if (p_valid && p_ready) begin
                if (wr_addr == ADDR_W'(DEPTH - 1) || p_last) begin
                    wr_addr <= '0;
                end else begin
                    wr_addr <= wr_addr + 1'b1;
                end
            end

            // Read address increment
            if (c_valid && c_ready) begin
                if (rd_addr == ADDR_W'(DEPTH - 1) || c_last) begin
                    rd_addr <= '0;
                end else begin
                    rd_addr <= rd_addr + 1'b1;
                end
            end

            // Bank select update on state change
            case (state)
                IDLE_A_WR: begin
                    wr_sel <= 1'b0;
                    rd_sel <= 1'b0;
                end
                READ_A_WR_B: begin
                    wr_sel <= 1'b1;
                    rd_sel <= 1'b0;
                end
                READ_B_WR_A: begin
                    wr_sel <= 1'b0;
                    rd_sel <= 1'b1;
                end
                default: ;
            endcase
        end
    end

    // =========================================================================
    // Write Logic
    // =========================================================================
    always_ff @(posedge clk) begin
        if (p_valid && p_ready) begin
            if (!wr_sel)
                bank_a[wr_addr] <= p_data;
            else
                bank_b[wr_addr] <= p_data;
        end
    end

    // =========================================================================
    // Read Logic
    // =========================================================================
    logic [DATA_W-1:0] rd_data_a, rd_data_b;
    logic [DATA_W-1:0] c_data_reg;

    always_ff @(posedge clk) begin
        rd_data_a <= bank_a[rd_addr];
        rd_data_b <= bank_b[rd_addr];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            c_data_reg <= '0;
        else
            c_data_reg <= rd_sel ? rd_data_b : rd_data_a;
    end

    // Optional pipeline
    generate
        if (PIPELINE_READ) begin : gen_pipe
            logic [DATA_W-1:0] c_data_pipe;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    c_data_pipe <= '0;
                else
                    c_data_pipe <= c_data_reg;
            end
            assign c_data = c_data_pipe;
        end else begin : gen_no_pipe
            assign c_data = c_data_reg;
        end
    endgenerate

    // =========================================================================
    // State Machine
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE_A_WR;
        else
            state <= next_state;
    end

    always_comb begin
        next_state = state;
        bank_switch = 1'b0;

        case (state)
            IDLE_A_WR: begin
                if (p_last && p_valid && p_ready) begin
                    next_state = SWITCH_A2B;
                    bank_switch = 1'b1;
                end
            end

            SWITCH_A2B: begin
                next_state = READ_A_WR_B;
            end

            READ_A_WR_B: begin
                if (c_last && c_valid && c_ready) begin
                    if (p_last && p_valid && p_ready)
                        next_state = READ_B_WR_A;
                    else
                        next_state = READ_A_WR_B;
                    bank_switch = 1'b1;
                end
            end

            READ_B_WR_A: begin
                if (c_last && c_valid && c_ready) begin
                    if (p_last && p_valid && p_ready)
                        next_state = READ_A_WR_B;
                    else
                        next_state = READ_B_WR_A;
                    bank_switch = 1'b1;
                end
            end

            default: next_state = IDLE_A_WR;
        endcase
    end

    assign p_ready = (state inside {IDLE_A_WR, READ_A_WR_B, READ_B_WR_A});
    assign c_valid = (state inside {READ_A_WR_B, READ_B_WR_A});
    assign c_last = (rd_addr == ADDR_W'(DEPTH - 1));
    assign bank_status = {rd_sel, wr_sel};

    // =========================================================================
    // Assertions
    // =========================================================================
    // synthesis translate_off
    /* verilator lint_off SYNCASYNCNET */
    PING_PONG_AUTO_ADDR: assert property (
        @(posedge clk) disable iff (!rst_n)
        32'(wr_addr) < DEPTH && 32'(rd_addr) < DEPTH
    ) else $error("ping_pong_bram_auto: Address overflow");

    PING_PONG_AUTO_VALID: assert property (
        @(posedge clk) disable iff (!rst_n)
        (p_valid && p_ready) |-> !c_valid || c_ready || p_last
    ) else $error("ping_pong_bram_auto: Potential deadlock");
    /* verilator lint_on SYNCASYNCNET */
    // synthesis translate_on

endmodule
