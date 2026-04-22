// =============================================================================
// mining_controller.sv — Autonomous Nonce-Scanning Mining FSM v2
// QUG-V1 Mining SoC — Phase 5: Mining Controller (v4.1 Enhanced)
// =============================================================================
// Project  : QUG-V1 RISC-V Mining SoC
// Target   : Xilinx Kintex-7 XC7K325T (FPGA prototype)
// Author   : Quillon Foundation / Dragon Ball Miner
// License  : MIT
// =============================================================================
//
// Implements all control signal semantics from technical-review-rtl-v4.1.md
// Section 4 (signal types) and Section 6 (SPI register map):
//
//   sticky solution_found     — set when hash meets target; cleared by
//                               clear_solution pulse or new start
//   best-hash double-buffer   — shadow registers update freely on each new
//                               best; snapshot registers promoted atomically
//                               on snapshot_best pulse (tear-free SPI reads)
//   hardware hashrate         — CLK_FREQ_HZ / 1s window sampling counter
//   nonce-range exhaustion    — stops and signals when current_nonce >= nonce_end
//   JOB_ID echo               — latched with solution and best-hash snapshots
//   partial scratchpad write  — after first nonce, only words 8-9 (nonce_lo,
//                               nonce_hi) change; scratch_partial_wr signals
//                               the scratchpad to skip words 0-7 and 10-15
//
// FSM:
//   S_IDLE → S_WRITE_SCRATCH → S_ISSUE_CHAIN → S_WAIT_RESULT →
//   S_CHECK → S_SOLUTION (if found) or S_NEXT_NONCE (loop)
//   S_NEXT_NONCE checks nonce_end and goes to S_IDLE on exhaustion.
//
// Best-hash double-buffer SPI protocol (per v4.1 Section 4):
//   1. Host writes CTRL.SNAPSHOT_BEST
//   2. Shadow registers → snapshot registers (atomic copy)
//   3. Host reads BEST_NONCE (2 SPI words)
//   4. Host reads BEST_HASH (8 SPI words)
//   All reads return stable snapshot — no tearing across SPI beats.
// =============================================================================

module mining_controller
    import qug_pkg::*;
(
    input  logic        clk,
    input  logic        rst_n,

    // =========================================================================
    // Work assignment (written by host before asserting start)
    // =========================================================================
    input  logic        start,             // Pulse: begin new job
    input  logic        stop,              // Pulse: halt immediately
    input  logic [255:0] challenge,        // 256-bit challenge hash
    input  logic [63:0]  nonce_start,      // Starting nonce
    input  logic [63:0]  nonce_end,        // Ending nonce (0 = unlimited)
    input  logic [7:0]   difficulty,       // Required leading zero bits
    input  logic [13:0]  vdf_depth,        // VDF chain depth (100 legacy, 5000+ genus-2)
    input  logic [31:0]  job_id,           // Host-assigned job ID (echoed in results)

    // =========================================================================
    // Control commands (write-1-to-act, maps to SPI CTRL register bits)
    // =========================================================================
    input  logic        clear_solution,    // CTRL[2]: clear sticky solution_found
    input  logic        clear_best,        // CTRL[3]: clear best_updated sticky flag
    input  logic        snapshot_best,     // CTRL[4]: atomic shadow→snapshot promotion

    // =========================================================================
    // Mining status (maps to SPI STATUS register)
    // =========================================================================
    output logic        mining_active,     // Level: mining in progress
    output logic        nonce_exhausted,   // Level: nonce range exhausted, no solution

    // =========================================================================
    // Solution snapshot (sticky; stable until new start or clear_solution)
    // solution_found remains high so the host can poll STATUS at low frequency.
    // =========================================================================
    output logic        solution_found,
    output logic [63:0]  solution_nonce,
    output logic [31:0]  solution_hash [0:7],
    output logic [7:0]   solution_lzc,
    output logic [31:0]  solution_job_id,

    // =========================================================================
    // Best-hash double-buffer (shadow → snapshot on snapshot_best pulse)
    // Shadow is updated on every new best; snapshot is the read-stable copy.
    // =========================================================================
    output logic        best_updated,      // Sticky: new best since last clear_best
    output logic [63:0]  best_nonce_snap,
    output logic [31:0]  best_hash_snap [0:7],
    output logic [7:0]   best_lzc_snap,
    output logic [31:0]  best_job_snap,

    // =========================================================================
    // Telemetry counters (free-running; read at any time without side effects)
    // =========================================================================
    output logic [63:0]  hash_count,       // Total completed VDF chains since reset
    output logic [31:0]  nonces_tried,     // Completed nonces in current job
    output logic [31:0]  hashrate,         // Hashes/second — hardware 1s window

    // =========================================================================
    // Xcrypto scratchpad interface
    // scratch_wr_en  = full 16-word write (first nonce, or after start)
    // scratch_partial_wr = only words 8-9 (nonce_lo, nonce_hi) have changed
    // =========================================================================
    output logic [31:0]  scratch_data [0:15],
    output logic         scratch_wr_en,
    output logic         scratch_partial_wr,   // Hint: only nonce words changed

    // =========================================================================
    // Xcrypto command interface (trigger blake3.chain)
    // =========================================================================
    output logic         xc_cmd_valid,
    output logic [6:0]   xc_cmd_funct7,
    output logic [31:0]  xc_cmd_rs2,
    input  logic         xc_cmd_ready,
    input  logic         xc_resp_valid,
    input  logic [31:0]  xc_resp_data,     // {solution_found[31], 23'd0, lzc[7:0]}

    // =========================================================================
    // Hash result from xcrypto (hash_words_out port from xcrypto_unit.sv)
    // Valid and stable whenever xc_resp_valid is asserted.
    // =========================================================================
    input  logic [31:0]  hash_words_in [0:7]
);

    // =========================================================================
    // Xcrypto funct7 encoding
    // =========================================================================
    localparam logic [6:0] F7_CHAIN = 7'd2;

    // =========================================================================
    // FSM state encoding
    // =========================================================================
    typedef enum logic [2:0] {
        S_IDLE          = 3'd0,
        S_WRITE_SCRATCH = 3'd1,
        S_ISSUE_CHAIN   = 3'd2,
        S_WAIT_RESULT   = 3'd3,
        S_CHECK         = 3'd4,
        S_SOLUTION      = 3'd5,
        S_NEXT_NONCE    = 3'd6
    } state_t;

    state_t fsm_state, fsm_next;

    // =========================================================================
    // Work registers (latched on start)
    // =========================================================================
    logic [63:0]  current_nonce;
    logic [255:0] latched_challenge;
    logic [7:0]   latched_difficulty;
    logic [13:0]  latched_vdf_depth;
    logic [31:0]  latched_job_id;
    logic [63:0]  latched_nonce_end;

    // =========================================================================
    // Response latch
    // =========================================================================
    logic         resp_sol_found;
    logic [7:0]   resp_lzc;

    // =========================================================================
    // First-nonce flag: enables partial scratchpad writes on subsequent nonces
    // =========================================================================
    logic         first_nonce_done;

    // =========================================================================
    // Solution sticky registers
    // =========================================================================
    logic         solution_found_r;
    logic [63:0]  solution_nonce_r;
    logic [31:0]  solution_hash_r [0:7];
    logic [7:0]   solution_lzc_r;
    logic [31:0]  solution_job_id_r;

    // =========================================================================
    // Best-hash shadow registers (updated on every new best)
    // =========================================================================
    logic         best_updated_r;
    logic [63:0]  best_nonce_shadow;
    logic [31:0]  best_hash_shadow [0:7];
    logic [7:0]   best_lzc_shadow;
    logic [31:0]  best_job_shadow;

    // Best-hash snapshot registers (stable for host SPI reads)
    logic [63:0]  best_nonce_snap_r;
    logic [31:0]  best_hash_snap_r [0:7];
    logic [7:0]   best_lzc_snap_r;
    logic [31:0]  best_job_snap_r;

    // =========================================================================
    // Telemetry registers
    // =========================================================================
    logic [63:0]  hash_count_r;
    logic [31:0]  nonces_tried_r;

    // Hashrate: count hashes in a 1-second hardware window, snapshot at rollover
    localparam int CYCLES_PER_SECOND = CLK_FREQ_HZ;  // 100_000_000
    logic [26:0]  cycle_cnt;      // 2^27 = 134M > 100M clock cycles
    logic [31:0]  hash_window_r;  // Hashes counted in current second
    logic [31:0]  hashrate_r;     // Latched at second rollover

    // Status flags
    logic         active_flag;
    logic         exhausted_flag;

    // =========================================================================
    // FSM: next-state logic
    // =========================================================================
    always_comb begin
        fsm_next = fsm_state;

        case (fsm_state)
            S_IDLE: begin
                if (start && !stop)
                    fsm_next = S_WRITE_SCRATCH;
            end

            S_WRITE_SCRATCH: begin
                fsm_next = S_ISSUE_CHAIN;
            end

            S_ISSUE_CHAIN: begin
                if (xc_cmd_ready)
                    fsm_next = S_WAIT_RESULT;
            end

            S_WAIT_RESULT: begin
                if (xc_resp_valid)
                    fsm_next = S_CHECK;
            end

            S_CHECK: begin
                if (resp_sol_found)
                    fsm_next = S_SOLUTION;
                else
                    fsm_next = S_NEXT_NONCE;
            end

            S_SOLUTION: begin
                if (stop)
                    fsm_next = S_IDLE;
            end

            S_NEXT_NONCE: begin
                // Check nonce exhaustion before looping
                if (latched_nonce_end != 64'd0 &&
                    current_nonce + 64'd1 >= latched_nonce_end) begin
                    fsm_next = S_IDLE;  // Exhaust → idle (nonce_exhausted will be set)
                end else begin
                    fsm_next = S_WRITE_SCRATCH;
                end
            end

            default: fsm_next = S_IDLE;
        endcase

        // Global stop: return to idle from any active state
        if (stop && fsm_state != S_IDLE && fsm_state != S_SOLUTION)
            fsm_next = S_IDLE;
    end

    // =========================================================================
    // FSM: state register and datapath
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm_state          <= S_IDLE;
            current_nonce      <= 64'd0;
            latched_challenge  <= 256'd0;
            latched_difficulty <= 8'd0;
            latched_vdf_depth  <= 14'd0;
            latched_job_id     <= 32'd0;
            latched_nonce_end  <= 64'd0;
            nonces_tried_r     <= 32'd0;
            active_flag        <= 1'b0;
            exhausted_flag     <= 1'b0;
            first_nonce_done   <= 1'b0;
            resp_sol_found     <= 1'b0;
            resp_lzc           <= 8'd0;
            // Solution sticky
            solution_found_r   <= 1'b0;
            solution_nonce_r   <= 64'd0;
            for (int i = 0; i < 8; i++) solution_hash_r[i] <= 32'd0;
            solution_lzc_r     <= 8'd0;
            solution_job_id_r  <= 32'd0;
            // Best-hash shadow
            best_updated_r     <= 1'b0;
            best_nonce_shadow  <= 64'd0;
            for (int i = 0; i < 8; i++) best_hash_shadow[i] <= 32'd0;
            best_lzc_shadow    <= 8'd0;
            best_job_shadow    <= 32'd0;
            // Best-hash snapshot
            best_nonce_snap_r  <= 64'd0;
            for (int i = 0; i < 8; i++) best_hash_snap_r[i] <= 32'd0;
            best_lzc_snap_r    <= 8'd0;
            best_job_snap_r    <= 32'd0;
            // Telemetry
            hash_count_r       <= 64'd0;
            cycle_cnt          <= 27'd0;
            hash_window_r      <= 32'd0;
            hashrate_r         <= 32'd0;
        end else begin
            fsm_state <= fsm_next;

            // ------------------------------------------------------------------
            // Hashrate measurement: 1-second rolling window
            // ------------------------------------------------------------------
            if (cycle_cnt >= CYCLES_PER_SECOND - 1) begin
                cycle_cnt    <= 27'd0;
                hashrate_r   <= hash_window_r;  // Snapshot this second's count
                hash_window_r <= 32'd0;
            end else begin
                cycle_cnt <= cycle_cnt + 27'd1;
            end

            // ------------------------------------------------------------------
            // Control commands (can fire any cycle)
            // ------------------------------------------------------------------
            if (clear_solution)
                solution_found_r <= 1'b0;

            if (clear_best)
                best_updated_r <= 1'b0;

            if (snapshot_best) begin
                // Atomic shadow → snapshot promotion (all fields in one cycle)
                best_nonce_snap_r <= best_nonce_shadow;
                best_lzc_snap_r   <= best_lzc_shadow;
                best_job_snap_r   <= best_job_shadow;
                for (int i = 0; i < 8; i++)
                    best_hash_snap_r[i] <= best_hash_shadow[i];
            end

            // ------------------------------------------------------------------
            // FSM datapath
            // ------------------------------------------------------------------
            case (fsm_state)
                S_IDLE: begin
                    active_flag    <= 1'b0;
                    exhausted_flag <= 1'b0;

                    if (start && !stop) begin
                        latched_challenge  <= challenge;
                        current_nonce      <= nonce_start;
                        latched_nonce_end  <= nonce_end;
                        latched_difficulty <= difficulty;
                        latched_vdf_depth  <= vdf_depth;
                        latched_job_id     <= job_id;
                        nonces_tried_r     <= 32'd0;
                        active_flag        <= 1'b1;
                        first_nonce_done   <= 1'b0;
                        // Clear stickies on new job
                        solution_found_r   <= 1'b0;
                        best_updated_r     <= 1'b0;
                        best_lzc_shadow    <= 8'd0;  // Reset best-hash baseline
                    end
                end

                S_WRITE_SCRATCH: begin
                    active_flag      <= 1'b1;
                    first_nonce_done <= 1'b1;  // After first write, enable partial mode
                end

                S_ISSUE_CHAIN: ;  // Nothing: combinational handshake

                S_WAIT_RESULT: begin
                    // Latch response when xcrypto signals completion
                    if (xc_resp_valid) begin
                        resp_sol_found <= xc_resp_data[31];
                        resp_lzc       <= xc_resp_data[7:0];
                    end
                end

                S_CHECK: begin
                    // Count this hash attempt
                    nonces_tried_r <= nonces_tried_r + 32'd1;
                    hash_count_r   <= hash_count_r   + 64'd1;
                    hash_window_r  <= hash_window_r  + 32'd1;

                    // ----------------------------------------------------------
                    // Best-hash tracking: update shadow if this hash is harder
                    // (more leading zeros) than the current best.
                    // Shadow updated here; snapshot promoted on snapshot_best.
                    // ----------------------------------------------------------
                    if (resp_lzc > best_lzc_shadow) begin
                        best_lzc_shadow   <= resp_lzc;
                        best_nonce_shadow <= current_nonce;
                        best_job_shadow   <= latched_job_id;
                        best_updated_r    <= 1'b1;
                        for (int i = 0; i < 8; i++)
                            best_hash_shadow[i] <= hash_words_in[i];
                    end
                end

                S_SOLUTION: begin
                    // Latch solution snapshot (sticky — survives multiple clock cycles)
                    if (!solution_found_r) begin
                        solution_found_r  <= 1'b1;
                        solution_nonce_r  <= current_nonce;
                        solution_lzc_r    <= resp_lzc;
                        solution_job_id_r <= latched_job_id;
                        for (int i = 0; i < 8; i++)
                            solution_hash_r[i] <= hash_words_in[i];
                    end
                    active_flag <= 1'b0;

                    if (stop)
                        active_flag <= 1'b0;
                end

                S_NEXT_NONCE: begin
                    current_nonce <= current_nonce + 64'd1;

                    // Mark exhausted when the incremented nonce reaches nonce_end
                    if (latched_nonce_end != 64'd0 &&
                        current_nonce + 64'd1 >= latched_nonce_end) begin
                        exhausted_flag <= 1'b1;
                        active_flag    <= 1'b0;
                    end
                end

                default: ;
            endcase

            // Global stop: clear active flag
            if (stop && fsm_state != S_IDLE)
                active_flag <= 1'b0;
        end
    end

    // =========================================================================
    // Scratchpad write: build 16-word message block
    // Layout (matching gpu.rs and technical-review-rtl-v4.1.md Section 0.3):
    //   Words 0-7:   challenge hash (LE word order: word 0 = challenge bytes 0-3)
    //   Words 8-9:   nonce LE (word 8 = nonce[31:0], word 9 = nonce[63:32])
    //   Words 10-15: zero padding
    //
    // scratch_partial_wr: asserted on all iterations after the first.
    // A smart scratchpad can skip writing words 0-7 and 10-15 (they are constant
    // within a job), saving 14 of the 16 write ports per nonce iteration.
    // The current bulk scratchpad ignores this hint and writes all 16 words in
    // one cycle regardless, so there is no latency impact on FPGA prototype.
    // =========================================================================
    always_comb begin
        for (int i = 0; i < 16; i++) scratch_data[i] = 32'd0;

        // Challenge: LE word order (word 0 = bytes 0-3 of challenge hash)
        scratch_data[0]  = latched_challenge[31:0];
        scratch_data[1]  = latched_challenge[63:32];
        scratch_data[2]  = latched_challenge[95:64];
        scratch_data[3]  = latched_challenge[127:96];
        scratch_data[4]  = latched_challenge[159:128];
        scratch_data[5]  = latched_challenge[191:160];
        scratch_data[6]  = latched_challenge[223:192];
        scratch_data[7]  = latched_challenge[255:224];

        // Nonce: LE 64-bit (lo word first)
        scratch_data[8]  = current_nonce[31:0];
        scratch_data[9]  = current_nonce[63:32];

        // Words 10-15 remain zero (set by default above)

        scratch_wr_en      = (fsm_state == S_WRITE_SCRATCH);
        scratch_partial_wr = (fsm_state == S_WRITE_SCRATCH) && first_nonce_done;
    end

    // =========================================================================
    // Xcrypto command interface
    // =========================================================================
    always_comb begin
        xc_cmd_valid  = (fsm_state == S_ISSUE_CHAIN);
        xc_cmd_funct7 = F7_CHAIN;
        xc_cmd_rs2    = {18'd0, latched_vdf_depth};
    end

    // =========================================================================
    // Output assignments
    // =========================================================================
    assign mining_active  = active_flag;
    assign nonce_exhausted = exhausted_flag;

    assign solution_found  = solution_found_r;
    assign solution_nonce  = solution_nonce_r;
    assign solution_lzc    = solution_lzc_r;
    assign solution_job_id = solution_job_id_r;
    generate
        for (genvar i = 0; i < 8; i++) begin : gen_sol_hash
            assign solution_hash[i] = solution_hash_r[i];
        end
    endgenerate

    assign best_updated   = best_updated_r;
    assign best_nonce_snap = best_nonce_snap_r;
    assign best_lzc_snap   = best_lzc_snap_r;
    assign best_job_snap   = best_job_snap_r;
    generate
        for (genvar i = 0; i < 8; i++) begin : gen_best_hash
            assign best_hash_snap[i] = best_hash_snap_r[i];
        end
    endgenerate

    assign hash_count   = hash_count_r;
    assign nonces_tried = nonces_tried_r;
    assign hashrate     = hashrate_r;

endmodule
