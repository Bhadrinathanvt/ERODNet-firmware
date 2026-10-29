// =============================================================================
// ping_pong_buffer_control.sv
//
// FSM controller for Double-Buffered Ping-Pong BRAMs.
// Manages producer/consumer handshaking and bank selection.
// =============================================================================

module ping_pong_buffer_control (
    input  logic clk,
    input  logic rst_n,

    // Producer Interface
    input  logic p_start,
    output logic p_ready,
    input  logic p_done,

    // Consumer Interface
    input  logic c_start,
    output logic c_data_valid,
    input  logic c_done,

    // Buffer Controls
    output logic sel_wr,
    output logic sel_rd,
    output logic p_we_gate,

    // DEBUG
    output logic [2:0] debug_state
);

    typedef enum logic [2:0] {
        IDLE        = 3'b000,
        FILL_A      = 3'b001,
        FILL_B      = 3'b010,
        PROCESS_A   = 3'b011,
        PROCESS_B   = 3'b100,
        SWAP_WAIT   = 3'b101
    } state_e;

    state_e state, next_state;
    logic bank_a_fresh, bank_b_fresh;
    logic c_data_valid_comb;

    // =========================================================================
    // Bank Freshness Tracking
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bank_a_fresh <= 1'b0;
            bank_b_fresh <= 1'b0;
        end else begin
            // Consumer marks bank as consumed
            if (c_done) begin
                if (!sel_rd) bank_a_fresh <= 1'b0;
                else         bank_b_fresh <= 1'b0;
            end

            // Producer marks bank as fresh (p_done has priority)
            if (p_done) begin
                if (!sel_wr) bank_a_fresh <= 1'b1;
                else         bank_b_fresh <= 1'b1;
            end
        end
    end

    // =========================================================================
    // FSM State Transitions
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state  <= IDLE;
            debug_state <= IDLE;
            sel_wr <= 1'b0;
            sel_rd <= 1'b1; // Default read from B while writing to A
        end else begin
            state <= next_state;
            debug_state <= next_state;

            case (next_state)
                FILL_A:     sel_wr <= 1'b0;
                FILL_B:     sel_wr <= 1'b1;
                PROCESS_A:  begin
                    sel_rd <= 1'b0;
                    sel_wr <= 1'b1; // Next write to B
                end
                PROCESS_B:  begin
                    sel_rd <= 1'b1;
                    sel_wr <= 1'b0; // Next write to A
                end
                default: ;
            endcase
        end
    end

    // =========================================================================
    // FSM Next State and Control Logic
    // =========================================================================
    always_comb begin
        next_state = state;
        p_ready    = 1'b0;
        p_we_gate  = 1'b0;
        c_data_valid_comb = 1'b0;

        case (state)
            IDLE: begin
                p_ready = 1'b1;
                if (p_start) begin
                    next_state = (!sel_wr) ? FILL_A : FILL_B;
                end else if (c_start) begin
                    if (bank_a_fresh)      next_state = PROCESS_A;
                    else if (bank_b_fresh) next_state = PROCESS_B;
                end
            end

            FILL_A: begin
                p_we_gate = 1'b1;
                c_data_valid_comb = bank_b_fresh;
                if (p_done) begin
                    next_state = bank_b_fresh ? PROCESS_B : SWAP_WAIT;
                end
            end

            FILL_B: begin
                p_we_gate = 1'b1;
                c_data_valid_comb = bank_a_fresh;
                if (p_done) begin
                    next_state = bank_a_fresh ? PROCESS_A : SWAP_WAIT;
                end
            end

            PROCESS_A: begin
                c_data_valid_comb = 1'b1;
                p_ready = !bank_b_fresh;
                if (c_done) begin
                    next_state = bank_b_fresh ? PROCESS_B : IDLE;
                end
            end

            PROCESS_B: begin
                c_data_valid_comb = 1'b1;
                p_ready = !bank_a_fresh;
                if (c_done) begin
                    next_state = bank_a_fresh ? PROCESS_A : IDLE;
                end
            end

            SWAP_WAIT: begin
                p_ready = 1'b1;
                if (bank_a_fresh)      next_state = PROCESS_A;
                else if (bank_b_fresh) next_state = PROCESS_B;
                else if (p_start)      next_state = (!sel_wr) ? FILL_B : FILL_A;
            end

            default: next_state = IDLE;
        endcase
    end

    // =========================================================================
    // Output Registration
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c_data_valid <= 1'b0;
        end else begin
            c_data_valid <= c_data_valid_comb;
        end
    end

endmodule
