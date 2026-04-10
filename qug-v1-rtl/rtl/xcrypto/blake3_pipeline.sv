// =============================================================================
// blake3_pipeline.sv — 14-Stage Pipelined BLAKE3 Compression Function
// QUG-V1 Mining SoC — Xcrypto BLAKE3 Hardware Pipeline
// =============================================================================
//
// Fully pipelined BLAKE3 compression: one new compression can be started every
// clock cycle, with results emerging 14 cycles later (2 stages per round x 7
// rounds).
//
// After the pipeline is full, throughput is 1 hash / clock cycle.
// Total latency: 14 cycles (was 7 in the single-stage version).
//
// Each blake3_round is now a 2-stage pipelined module:
//   Stage A: column quarter-rounds (registered)
//   Stage B: diagonal quarter-rounds (registered)
// 7 rounds x 2 stages = 14 pipeline stages.
//
// Input interface:
//   - chaining_value: 256-bit (8 x 32-bit) input chaining value (or IV)
//   - block_words:    512-bit (16 x 32-bit) message block
//   - counter:        64-bit block counter
//   - block_len:      32-bit number of input bytes in this block
//   - flags:          32-bit domain separation flags
//   - in_valid:       input handshake
//   - in_ready:       backpressure from pipeline (always ready)
//
// Output interface:
//   - hash_out:       256-bit output hash (upper 8 words of final XOR)
//   - out_valid:      output handshake
//
// The BLAKE3 compression function:
//   1. Initialize 16-word state from chaining_value, IV, counter, block_len, flags
//   2. Apply 7 rounds with message permutation (14 pipeline stages)
//   3. XOR upper/lower halves of final state
//   4. Output lower 8 words as hash
// =============================================================================

module blake3_pipeline #(
    parameter int NUM_ROUNDS = 7   // BLAKE3 uses exactly 7 rounds
) (
    input  logic        clk,
    input  logic        rst_n,

    // Input: compression function arguments
    input  logic [31:0] chaining_value [0:7],   // h[0..7] — 256-bit CV
    input  logic [31:0] block_words   [0:15],   // m[0..15] — 512-bit block
    input  logic [63:0] counter,                 // t — 64-bit counter
    input  logic [31:0] block_len,               // b — bytes in block
    input  logic [31:0] flags,                   // d — domain separation
    input  logic        in_valid,
    output logic        in_ready,

    // Output: 256-bit hash
    output logic [31:0] hash_out [0:7],
    output logic        out_valid
);

    // Total pipeline stages = 2 * NUM_ROUNDS = 14
    localparam int NUM_STAGES = 2 * NUM_ROUNDS;

    // =========================================================================
    // BLAKE3 IV constants
    // =========================================================================
    localparam logic [31:0] IV [0:7] = '{
        32'h6A09E667, 32'hBB67AE85, 32'h3C6EF372, 32'hA54FF53A,
        32'h510E527F, 32'h9B05688C, 32'h1F83D9AB, 32'h5BE0CD19
    };

    // Pipeline is always ready (no backpressure within pipeline)
    assign in_ready = 1'b1;

    // =========================================================================
    // Initial state construction
    // =========================================================================
    logic [31:0] init_state [0:15];

    always_comb begin
        init_state[ 0] = chaining_value[0];
        init_state[ 1] = chaining_value[1];
        init_state[ 2] = chaining_value[2];
        init_state[ 3] = chaining_value[3];
        init_state[ 4] = chaining_value[4];
        init_state[ 5] = chaining_value[5];
        init_state[ 6] = chaining_value[6];
        init_state[ 7] = chaining_value[7];
        init_state[ 8] = IV[0];
        init_state[ 9] = IV[1];
        init_state[10] = IV[2];
        init_state[11] = IV[3];
        init_state[12] = counter[31:0];
        init_state[13] = counter[63:32];
        init_state[14] = block_len;
        init_state[15] = flags;
    end

    // =========================================================================
    // Input latch register (stage 0 entry)
    // =========================================================================
    // Latch inputs into first pipeline register so round 0 reads from flops.
    logic [31:0] state_s0 [0:15];
    logic [31:0] msg_s0   [0:15];
    logic [31:0] cv_s0    [0:7];
    logic        valid_s0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s0 <= 1'b0;
        end else begin
            valid_s0 <= in_valid;
            if (in_valid) begin
                for (int j = 0; j < 16; j++) begin
                    state_s0[j] <= init_state[j];
                    msg_s0[j]   <= block_words[j];
                end
                for (int j = 0; j < 8; j++) begin
                    cv_s0[j] <= chaining_value[j];
                end
            end
        end
    end

    // =========================================================================
    // Round instantiation — 7 pipelined rounds, each 2 stages
    // =========================================================================
    // Wiring between rounds:
    //   round[0] input = state_s0/msg_s0  (from input latch)
    //   round[r] input = round[r-1] output (for r > 0)

    // State/msg/valid wires between rounds (NUM_ROUNDS + 1 boundaries)
    logic [31:0] inter_state [0:NUM_ROUNDS][0:15];
    logic [31:0] inter_msg   [0:NUM_ROUNDS][0:15];
    logic        inter_valid [0:NUM_ROUNDS];

    // Chaining value pipeline — travels alongside the rounds
    // Needs NUM_STAGES + 1 entries (input latch + 14 round stages)
    // We propagate CV through each round's 2-stage delay.
    logic [31:0] cv_pipe [0:NUM_STAGES][0:7];
    logic        valid_pipe [0:NUM_STAGES];

    // Connect input latch to round 0 input
    always_comb begin
        for (int j = 0; j < 16; j++) begin
            inter_state[0][j] = state_s0[j];
            inter_msg[0][j]   = msg_s0[j];
        end
        inter_valid[0] = valid_s0;
    end

    // Round output valid signals (from each round's out_valid)
    logic round_out_valid [0:NUM_ROUNDS-1];

    genvar r;
    generate
        for (r = 0; r < NUM_ROUNDS; r++) begin : gen_rounds
            blake3_round u_round (
                .clk       (clk),
                .rst_n     (rst_n),
                .state_in  (inter_state[r]),
                .msg       (inter_msg[r]),
                .round_idx (3'(r)),
                .in_valid  (inter_valid[r]),
                .state_out (inter_state[r+1]),
                .out_valid (round_out_valid[r])
            );

            // Message words pass through unchanged — register them to match
            // the 2-cycle latency of each round.
            // Stage 1 register (after column step)
            logic [31:0] msg_s1 [0:15];
            logic [31:0] msg_s2 [0:15];

            always_ff @(posedge clk) begin
                for (int j = 0; j < 16; j++) begin
                    msg_s1[j] <= inter_msg[r][j];
                    msg_s2[j] <= msg_s1[j];
                end
            end

            // Connect delayed message to next round input
            always_comb begin
                for (int j = 0; j < 16; j++) begin
                    inter_msg[r+1][j] = msg_s2[j];
                end
            end

            // Valid propagation: round module handles this internally,
            // connect round out_valid to next round in_valid
            assign inter_valid[r+1] = round_out_valid[r];
        end
    endgenerate

    // =========================================================================
    // Chaining value pipeline — delay CV by NUM_STAGES cycles (14)
    // =========================================================================
    // Input latch already added 1 cycle. Each round adds 2 cycles.
    // Total delay from in_valid to final output = 1 (input latch) + 14 (rounds) = 15 cycles.
    // CV must be delayed the same 14 cycles as the rounds (from s0 to final).

    logic [31:0] cv_delay [0:NUM_STAGES-1][0:7];

    always_ff @(posedge clk) begin
        // First delay stage: from input latch
        for (int j = 0; j < 8; j++) begin
            cv_delay[0][j] <= cv_s0[j];
        end
        // Subsequent delay stages
        for (int i = 1; i < NUM_STAGES; i++) begin
            for (int j = 0; j < 8; j++) begin
                cv_delay[i][j] <= cv_delay[i-1][j];
            end
        end
    end

    // =========================================================================
    // Output: XOR final state halves
    // =========================================================================
    // BLAKE3 finalization:
    //   output[i] = state[i] ^ state[i+8]   for i in 0..7

    // The final round's out_valid is the pipeline output valid
    assign out_valid = round_out_valid[NUM_ROUNDS-1];

    always_comb begin
        for (int i = 0; i < 8; i++) begin
            hash_out[i] = inter_state[NUM_ROUNDS][i] ^ inter_state[NUM_ROUNDS][i + 8];
        end
    end

endmodule
