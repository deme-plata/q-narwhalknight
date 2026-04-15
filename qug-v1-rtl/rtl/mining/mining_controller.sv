// =============================================================================
// mining_controller.sv — Autonomous Nonce-Scanning Mining FSM
// QUG-V1 Mining SoC — Phase 5: Mining Controller
// =============================================================================
// Project  : QUG-V1 RISC-V Mining SoC
// Target   : Xilinx Kintex-7 XC7K325T (FPGA prototype)
// Author   : Quillon Foundation / Dragon Ball Miner
// License  : MIT
// =============================================================================
// Autonomous mining controller that scans nonces without CPU intervention.
// Writes the challenge+nonce message block into the Xcrypto scratchpad,
// issues blake3.chain commands with the configured VDF depth, and checks
// the LZC result against the difficulty target.
//
// Message block layout (matches gpu.rs):
//   Words 0-7:   challenge hash (256 bits, big-endian word order)
//   Words 8-9:   nonce (64-bit, little-endian: word 8 = lo, word 9 = hi)
//   Words 10-15: zero padding
//
// FSM:
//   S_IDLE -> S_WRITE_SCRATCH -> S_ISSUE_CHAIN -> S_WAIT_RESULT ->
//   S_CHECK -> S_SOLUTION (if found) or S_NEXT_NONCE (loop back)
//
// The controller continuously scans nonces from nonce_start upward until
// either a valid solution is found or the stop signal is asserted.
// =============================================================================

module mining_controller
    import qug_pkg::*;
(
    input  logic        clk,
    input  logic        rst_n,

    // =========================================================================
    // Control interface (from CPU / MMIO difficulty_regs)
    // =========================================================================
    input  logic        start,           // Pulse: begin mining
    input  logic        stop,            // Pulse: halt mining
    input  logic [255:0] challenge,      // 256-bit challenge hash
    input  logic [63:0]  nonce_start,    // Starting nonce value
    input  logic [7:0]   difficulty,     // Required leading zero bits
    input  logic [13:0]  vdf_depth,      // VDF chain depth (100 legacy, 5000+ genus-2)

    // =========================================================================
    // Status output
    // =========================================================================
    output logic        mining_active,   // Currently scanning nonces
    output logic        solution_found,  // Valid solution discovered
    output logic [63:0]  solution_nonce, // Winning nonce value
    output logic [7:0]   solution_lzc,   // Leading zeros of winning hash
    output logic [31:0]  nonces_tried,   // Counter for hashrate measurement

    // =========================================================================
    // Xcrypto scratchpad bulk-write interface
    // =========================================================================
    output logic [31:0]  scratch_data [0:15],  // 16-word message block
    output logic         scratch_wr_en,        // Bulk write enable

    // =========================================================================
    // Xcrypto command interface (trigger blake3.chain)
    // =========================================================================
    output logic         xc_cmd_valid,   // Issue blake3.chain command
    output logic [6:0]   xc_cmd_funct7,  // F7_CHAIN = 2
    output logic [31:0]  xc_cmd_rs2,     // Chain depth in rs2
    input  logic         xc_cmd_ready,   // Xcrypto unit ready
    input  logic         xc_resp_valid,  // Chain complete
    input  logic [31:0]  xc_resp_data    // {solution_found, 23'd0, lzc_count}
);

    // =========================================================================
    // Xcrypto funct7 encoding (from xcrypto_pkg)
    // =========================================================================
    localparam logic [6:0] F7_CHAIN = 7'd2;

    // =========================================================================
    // FSM state encoding
    // =========================================================================
    typedef enum logic [2:0] {
        S_IDLE          = 3'd0,  // Waiting for start
        S_WRITE_SCRATCH = 3'd1,  // Write challenge+nonce to scratchpad
        S_ISSUE_CHAIN   = 3'd2,  // Issue blake3.chain with vdf_depth
        S_WAIT_RESULT   = 3'd3,  // Wait for xcrypto response
        S_CHECK         = 3'd4,  // Check if solution_found bit is set
        S_SOLUTION      = 3'd5,  // Latch winning nonce, assert solution_found
        S_NEXT_NONCE    = 3'd6   // Increment nonce, loop back
    } state_t;

    state_t fsm_state, fsm_next;

    // =========================================================================
    // Internal registers
    // =========================================================================
    logic [63:0]  current_nonce;       // Running nonce counter
    logic [255:0] latched_challenge;   // Latched challenge hash
    logic [7:0]   latched_difficulty;  // Latched difficulty target
    logic [13:0]  latched_vdf_depth;   // Latched VDF depth
    logic [31:0]  nonce_counter;       // Nonces tried (for hashrate)
    logic         active_flag;         // Mining is active
    logic         found_flag;          // Solution has been found
    logic [63:0]  found_nonce;         // Winning nonce
    logic [7:0]   found_lzc;           // LZC of winning hash

    // Response latch
    logic         resp_sol_found;      // Bit 31 of xc_resp_data
    logic [7:0]   resp_lzc;            // Bits [7:0] of xc_resp_data

    // =========================================================================
    // FSM: next-state logic
    // =========================================================================
    always_comb begin
        fsm_next = fsm_state;

        case (fsm_state)
            S_IDLE: begin
                if (start && !stop) begin
                    fsm_next = S_WRITE_SCRATCH;
                end
            end

            S_WRITE_SCRATCH: begin
                // Single-cycle scratchpad write, advance immediately
                fsm_next = S_ISSUE_CHAIN;
            end

            S_ISSUE_CHAIN: begin
                // Wait for xcrypto handshake
                if (xc_cmd_ready) begin
                    fsm_next = S_WAIT_RESULT;
                end
            end

            S_WAIT_RESULT: begin
                if (xc_resp_valid) begin
                    fsm_next = S_CHECK;
                end
            end

            S_CHECK: begin
                if (resp_sol_found) begin
                    fsm_next = S_SOLUTION;
                end else begin
                    fsm_next = S_NEXT_NONCE;
                end
            end

            S_SOLUTION: begin
                // Hold here until stop or new start clears it
                if (stop) begin
                    fsm_next = S_IDLE;
                end
            end

            S_NEXT_NONCE: begin
                // Increment nonce and loop back
                fsm_next = S_WRITE_SCRATCH;
            end

            default: fsm_next = S_IDLE;
        endcase

        // Global stop override: return to idle from any active state
        if (stop && fsm_state != S_IDLE && fsm_state != S_SOLUTION) begin
            fsm_next = S_IDLE;
        end
    end

    // =========================================================================
    // FSM: state register and datapath
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm_state         <= S_IDLE;
            current_nonce     <= 64'd0;
            latched_challenge <= 256'd0;
            latched_difficulty <= 8'd0;
            latched_vdf_depth <= 14'd0;
            nonce_counter     <= 32'd0;
            active_flag       <= 1'b0;
            found_flag        <= 1'b0;
            found_nonce       <= 64'd0;
            found_lzc         <= 8'd0;
            resp_sol_found    <= 1'b0;
            resp_lzc          <= 8'd0;
        end else begin
            fsm_state <= fsm_next;

            case (fsm_state)
                S_IDLE: begin
                    active_flag <= 1'b0;

                    // Latch parameters on start pulse
                    if (start && !stop) begin
                        latched_challenge  <= challenge;
                        current_nonce      <= nonce_start;
                        latched_difficulty <= difficulty;
                        latched_vdf_depth  <= vdf_depth;
                        nonce_counter      <= 32'd0;
                        found_flag         <= 1'b0;
                        found_nonce        <= 64'd0;
                        found_lzc          <= 8'd0;
                        active_flag        <= 1'b1;
                    end
                end

                S_WRITE_SCRATCH: begin
                    active_flag <= 1'b1;
                end

                S_ISSUE_CHAIN: begin
                    // Nothing to update; handshake handled combinationally
                end

                S_WAIT_RESULT: begin
                    // Latch response when valid
                    if (xc_resp_valid) begin
                        resp_sol_found <= xc_resp_data[31];
                        resp_lzc       <= xc_resp_data[7:0];
                    end
                end

                S_CHECK: begin
                    // Increment nonce counter (one hash attempt completed)
                    nonce_counter <= nonce_counter + 32'd1;
                end

                S_SOLUTION: begin
                    // Latch winning nonce and LZC
                    found_flag  <= 1'b1;
                    found_nonce <= current_nonce;
                    found_lzc   <= resp_lzc;
                    active_flag <= 1'b0;

                    // Clear on stop
                    if (stop) begin
                        active_flag <= 1'b0;
                    end
                end

                S_NEXT_NONCE: begin
                    // Increment nonce for next attempt
                    current_nonce <= current_nonce + 64'd1;
                end

                default: ;
            endcase

            // Global stop: clear active flag
            if (stop && fsm_state != S_IDLE) begin
                active_flag <= 1'b0;
            end
        end
    end

    // =========================================================================
    // Scratchpad write: build 16-word message block
    // =========================================================================
    // Layout (matching gpu.rs):
    //   Words 0-7:   challenge[255:0] big-endian word order
    //                 word 0 = challenge[255:224] (MSW)
    //                 word 7 = challenge[31:0]    (LSW)
    //   Words 8-9:   nonce little-endian
    //                 word 8 = nonce[31:0]  (lo)
    //                 word 9 = nonce[63:32] (hi)
    //   Words 10-15: zero padding
    always_comb begin
        // Default: all zeros
        for (int i = 0; i < 16; i++) begin
            scratch_data[i] = 32'd0;
        end

        // Challenge hash: big-endian word order (word 0 = MSW)
        scratch_data[0] = latched_challenge[255:224];
        scratch_data[1] = latched_challenge[223:192];
        scratch_data[2] = latched_challenge[191:160];
        scratch_data[3] = latched_challenge[159:128];
        scratch_data[4] = latched_challenge[127:96];
        scratch_data[5] = latched_challenge[95:64];
        scratch_data[6] = latched_challenge[63:32];
        scratch_data[7] = latched_challenge[31:0];

        // Nonce: little-endian (lo word first)
        scratch_data[8]  = current_nonce[31:0];
        scratch_data[9]  = current_nonce[63:32];

        // Words 10-15 already zeroed by default

        // Assert write enable only during S_WRITE_SCRATCH
        scratch_wr_en = (fsm_state == S_WRITE_SCRATCH);
    end

    // =========================================================================
    // Xcrypto command interface: issue blake3.chain
    // =========================================================================
    always_comb begin
        xc_cmd_valid  = (fsm_state == S_ISSUE_CHAIN);
        xc_cmd_funct7 = F7_CHAIN;
        xc_cmd_rs2    = {18'd0, latched_vdf_depth};  // VDF depth in rs2[13:0]
    end

    // =========================================================================
    // Output assignments
    // =========================================================================
    assign mining_active  = active_flag;
    assign solution_found = found_flag;
    assign solution_nonce = found_nonce;
    assign solution_lzc   = found_lzc;
    assign nonces_tried   = nonce_counter;

endmodule
