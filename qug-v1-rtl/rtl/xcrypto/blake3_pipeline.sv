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
//
// NOTE: iverilog 11 does not support genvar-indexed 2D array slices as module
// port connections (assertion elaborate.cc:1474). Rounds are fully unrolled
// with explicit 1D wire arrays for every inter-round signal.
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

    // =========================================================================
    // BLAKE3 IV constants
    // =========================================================================
`ifdef SYNTHESIS
    localparam logic [31:0] IV [0:7] = '{
        32'h6A09E667, 32'hBB67AE85, 32'h3C6EF372, 32'hA54FF53A,
        32'h510E527F, 32'h9B05688C, 32'h1F83D9AB, 32'h5BE0CD19
    };
`else
    // iverilog 11 does not support array localparams in module scope
    reg [31:0] IV [0:7];
    initial begin
        IV[0] = 32'h6A09E667; IV[1] = 32'hBB67AE85;
        IV[2] = 32'h3C6EF372; IV[3] = 32'hA54FF53A;
        IV[4] = 32'h510E527F; IV[5] = 32'h9B05688C;
        IV[6] = 32'h1F83D9AB; IV[7] = 32'h5BE0CD19;
    end
`endif

    // Pipeline is always ready (no backpressure within pipeline)
    assign in_ready = 1'b1;

    // =========================================================================
    // Initial state construction
    // =========================================================================
    // Initial state as individual assigns — iverilog 11 always_comb may not
    // re-trigger correctly on unpacked array port element changes.
    // Continuous assigns are unambiguous in their sensitivity.
    logic [31:0] init_state [0:15];
    assign init_state[ 0] = chaining_value[0];
    assign init_state[ 1] = chaining_value[1];
    assign init_state[ 2] = chaining_value[2];
    assign init_state[ 3] = chaining_value[3];
    assign init_state[ 4] = chaining_value[4];
    assign init_state[ 5] = chaining_value[5];
    assign init_state[ 6] = chaining_value[6];
    assign init_state[ 7] = chaining_value[7];
    // BLAKE3 IV[0..3] for state positions 8-11
    assign init_state[ 8] = 32'h6A09E667;
    assign init_state[ 9] = 32'hBB67AE85;
    assign init_state[10] = 32'h3C6EF372;
    assign init_state[11] = 32'hA54FF53A;
    assign init_state[12] = counter[31:0];
    assign init_state[13] = counter[63:32];
    assign init_state[14] = block_len;
    assign init_state[15] = flags;

    // =========================================================================
    // Scalar bridges for block_words and init_state[12..15]
    // =========================================================================
    // iverilog 11 bug: always_ff reading directly from an unpacked array input
    // port does not see updates from the connected testbench signal — the NBA
    // reads the stale (X) port copy, not the driven value. Routing each element
    // through an individual continuous-assign scalar bridge ensures the always_ff
    // reads the correct post-update value.
    //
    // Also: init_state[12..15] are driven from packed signals (counter parts,
    // block_len, flags_in) via continuous assigns to unpacked array elements.
    // In iverilog 11, always_ff reading those unpacked array elements can get
    // stale X values. Fix: bypass init_state[12..15] and read packed scalars
    // directly via named scalar bridges.
    logic [31:0] bw0,  bw1,  bw2,  bw3;
    logic [31:0] bw4,  bw5,  bw6,  bw7;
    logic [31:0] bw8,  bw9,  bw10, bw11;
    logic [31:0] bw12, bw13, bw14, bw15;

    assign bw0  = block_words[ 0]; assign bw1  = block_words[ 1];
    assign bw2  = block_words[ 2]; assign bw3  = block_words[ 3];
    assign bw4  = block_words[ 4]; assign bw5  = block_words[ 5];
    assign bw6  = block_words[ 6]; assign bw7  = block_words[ 7];
    assign bw8  = block_words[ 8]; assign bw9  = block_words[ 9];
    assign bw10 = block_words[10]; assign bw11 = block_words[11];
    assign bw12 = block_words[12]; assign bw13 = block_words[13];
    assign bw14 = block_words[14]; assign bw15 = block_words[15];

    // Scalar bridges for counter, block_len, flags — bypass init_state[12..15]
    logic [31:0] cnt_lo, cnt_hi, blen, flg;
    assign cnt_lo = counter[31: 0];
    assign cnt_hi = counter[63:32];
    assign blen   = block_len;
    assign flg    = flags;

    // =========================================================================
    // Input latch register (stage 0 entry)
    // =========================================================================
    logic [31:0] state_s0 [0:15];
    logic [31:0] msg_s0   [0:15];
    logic        valid_s0;

    // iverilog 11: for-loop NBA assignments to unpacked arrays in always_ff
    // do not execute — unrolled to individual statements.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s0      <= 1'b0;
            state_s0[ 0]  <= '0; state_s0[ 1]  <= '0; state_s0[ 2]  <= '0; state_s0[ 3]  <= '0;
            state_s0[ 4]  <= '0; state_s0[ 5]  <= '0; state_s0[ 6]  <= '0; state_s0[ 7]  <= '0;
            state_s0[ 8]  <= '0; state_s0[ 9]  <= '0; state_s0[10]  <= '0; state_s0[11]  <= '0;
            state_s0[12]  <= '0; state_s0[13]  <= '0; state_s0[14]  <= '0; state_s0[15]  <= '0;
            msg_s0[ 0]    <= '0; msg_s0[ 1]    <= '0; msg_s0[ 2]    <= '0; msg_s0[ 3]    <= '0;
            msg_s0[ 4]    <= '0; msg_s0[ 5]    <= '0; msg_s0[ 6]    <= '0; msg_s0[ 7]    <= '0;
            msg_s0[ 8]    <= '0; msg_s0[ 9]    <= '0; msg_s0[10]    <= '0; msg_s0[11]    <= '0;
            msg_s0[12]    <= '0; msg_s0[13]    <= '0; msg_s0[14]    <= '0; msg_s0[15]    <= '0;
        end else begin
            valid_s0 <= in_valid;
            if (in_valid) begin
                state_s0[ 0] <= init_state[ 0]; state_s0[ 1] <= init_state[ 1];
                state_s0[ 2] <= init_state[ 2]; state_s0[ 3] <= init_state[ 3];
                state_s0[ 4] <= init_state[ 4]; state_s0[ 5] <= init_state[ 5];
                state_s0[ 6] <= init_state[ 6]; state_s0[ 7] <= init_state[ 7];
                state_s0[ 8] <= init_state[ 8]; state_s0[ 9] <= init_state[ 9];
                state_s0[10] <= init_state[10]; state_s0[11] <= init_state[11];
                state_s0[12] <= counter[31: 0];  state_s0[13] <= counter[63:32];
                state_s0[14] <= block_len;       state_s0[15] <= flags;
                msg_s0[ 0]  <= bw0;  msg_s0[ 1]  <= bw1;
                msg_s0[ 2]  <= bw2;  msg_s0[ 3]  <= bw3;
                msg_s0[ 4]  <= bw4;  msg_s0[ 5]  <= bw5;
                msg_s0[ 6]  <= bw6;  msg_s0[ 7]  <= bw7;
                msg_s0[ 8]  <= bw8;  msg_s0[ 9]  <= bw9;
                msg_s0[10]  <= bw10; msg_s0[11]  <= bw11;
                msg_s0[12]  <= bw12; msg_s0[13]  <= bw13;
                msg_s0[14]  <= bw14; msg_s0[15]  <= bw15;
            end
        end
    end

    // =========================================================================
    // Inter-round state wires — one per round output (explicit 1D arrays)
    // =========================================================================
    logic [31:0] state_r0 [0:15];   // output of round 0
    logic [31:0] state_r1 [0:15];   // output of round 1
    logic [31:0] state_r2 [0:15];   // output of round 2
    logic [31:0] state_r3 [0:15];   // output of round 3
    logic [31:0] state_r4 [0:15];   // output of round 4
    logic [31:0] state_r5 [0:15];   // output of round 5
    logic [31:0] state_r6 [0:15];   // output of round 6 (final)

    logic        valid_r0, valid_r1, valid_r2, valid_r3,
                 valid_r4, valid_r5, valid_r6;

    // =========================================================================
    // Message delay pipeline — explicit 1D arrays per even delay stage
    //
    // Round r needs msg_s0 delayed by 2*r clock cycles:
    //   round 0 → msg_s0         (0 delay, direct)
    //   round 1 → msg_d2         (2 cycles)
    //   round 2 → msg_d4         (4 cycles)
    //   round 3 → msg_d6         (6 cycles)
    //   round 4 → msg_d8         (8 cycles)
    //   round 5 → msg_d10        (10 cycles)
    //   round 6 → msg_d12        (12 cycles)
    //
    // Intermediate odd stages needed to chain registers:
    //   msg_d1  = msg_s0  >> 1
    //   msg_d3  = msg_d2  >> 1   etc.
    // =========================================================================
    logic [31:0] msg_d1  [0:15];
    logic [31:0] msg_d2  [0:15];
    logic [31:0] msg_d3  [0:15];
    logic [31:0] msg_d4  [0:15];
    logic [31:0] msg_d5  [0:15];
    logic [31:0] msg_d6  [0:15];
    logic [31:0] msg_d7  [0:15];
    logic [31:0] msg_d8  [0:15];
    logic [31:0] msg_d9  [0:15];
    logic [31:0] msg_d10 [0:15];
    logic [31:0] msg_d11 [0:15];
    logic [31:0] msg_d12 [0:15];

    always_ff @(posedge clk) begin
        msg_d1[ 0] <= msg_s0[ 0];  msg_d1[ 1] <= msg_s0[ 1];  msg_d1[ 2] <= msg_s0[ 2];  msg_d1[ 3] <= msg_s0[ 3];
        msg_d1[ 4] <= msg_s0[ 4];  msg_d1[ 5] <= msg_s0[ 5];  msg_d1[ 6] <= msg_s0[ 6];  msg_d1[ 7] <= msg_s0[ 7];
        msg_d1[ 8] <= msg_s0[ 8];  msg_d1[ 9] <= msg_s0[ 9];  msg_d1[10] <= msg_s0[10];  msg_d1[11] <= msg_s0[11];
        msg_d1[12] <= msg_s0[12];  msg_d1[13] <= msg_s0[13];  msg_d1[14] <= msg_s0[14];  msg_d1[15] <= msg_s0[15];
        msg_d2[ 0] <= msg_d1[ 0];  msg_d2[ 1] <= msg_d1[ 1];  msg_d2[ 2] <= msg_d1[ 2];  msg_d2[ 3] <= msg_d1[ 3];
        msg_d2[ 4] <= msg_d1[ 4];  msg_d2[ 5] <= msg_d1[ 5];  msg_d2[ 6] <= msg_d1[ 6];  msg_d2[ 7] <= msg_d1[ 7];
        msg_d2[ 8] <= msg_d1[ 8];  msg_d2[ 9] <= msg_d1[ 9];  msg_d2[10] <= msg_d1[10];  msg_d2[11] <= msg_d1[11];
        msg_d2[12] <= msg_d1[12];  msg_d2[13] <= msg_d1[13];  msg_d2[14] <= msg_d1[14];  msg_d2[15] <= msg_d1[15];
        msg_d3[ 0] <= msg_d2[ 0];  msg_d3[ 1] <= msg_d2[ 1];  msg_d3[ 2] <= msg_d2[ 2];  msg_d3[ 3] <= msg_d2[ 3];
        msg_d3[ 4] <= msg_d2[ 4];  msg_d3[ 5] <= msg_d2[ 5];  msg_d3[ 6] <= msg_d2[ 6];  msg_d3[ 7] <= msg_d2[ 7];
        msg_d3[ 8] <= msg_d2[ 8];  msg_d3[ 9] <= msg_d2[ 9];  msg_d3[10] <= msg_d2[10];  msg_d3[11] <= msg_d2[11];
        msg_d3[12] <= msg_d2[12];  msg_d3[13] <= msg_d2[13];  msg_d3[14] <= msg_d2[14];  msg_d3[15] <= msg_d2[15];
        msg_d4[ 0] <= msg_d3[ 0];  msg_d4[ 1] <= msg_d3[ 1];  msg_d4[ 2] <= msg_d3[ 2];  msg_d4[ 3] <= msg_d3[ 3];
        msg_d4[ 4] <= msg_d3[ 4];  msg_d4[ 5] <= msg_d3[ 5];  msg_d4[ 6] <= msg_d3[ 6];  msg_d4[ 7] <= msg_d3[ 7];
        msg_d4[ 8] <= msg_d3[ 8];  msg_d4[ 9] <= msg_d3[ 9];  msg_d4[10] <= msg_d3[10];  msg_d4[11] <= msg_d3[11];
        msg_d4[12] <= msg_d3[12];  msg_d4[13] <= msg_d3[13];  msg_d4[14] <= msg_d3[14];  msg_d4[15] <= msg_d3[15];
        msg_d5[ 0] <= msg_d4[ 0];  msg_d5[ 1] <= msg_d4[ 1];  msg_d5[ 2] <= msg_d4[ 2];  msg_d5[ 3] <= msg_d4[ 3];
        msg_d5[ 4] <= msg_d4[ 4];  msg_d5[ 5] <= msg_d4[ 5];  msg_d5[ 6] <= msg_d4[ 6];  msg_d5[ 7] <= msg_d4[ 7];
        msg_d5[ 8] <= msg_d4[ 8];  msg_d5[ 9] <= msg_d4[ 9];  msg_d5[10] <= msg_d4[10];  msg_d5[11] <= msg_d4[11];
        msg_d5[12] <= msg_d4[12];  msg_d5[13] <= msg_d4[13];  msg_d5[14] <= msg_d4[14];  msg_d5[15] <= msg_d4[15];
        msg_d6[ 0] <= msg_d5[ 0];  msg_d6[ 1] <= msg_d5[ 1];  msg_d6[ 2] <= msg_d5[ 2];  msg_d6[ 3] <= msg_d5[ 3];
        msg_d6[ 4] <= msg_d5[ 4];  msg_d6[ 5] <= msg_d5[ 5];  msg_d6[ 6] <= msg_d5[ 6];  msg_d6[ 7] <= msg_d5[ 7];
        msg_d6[ 8] <= msg_d5[ 8];  msg_d6[ 9] <= msg_d5[ 9];  msg_d6[10] <= msg_d5[10];  msg_d6[11] <= msg_d5[11];
        msg_d6[12] <= msg_d5[12];  msg_d6[13] <= msg_d5[13];  msg_d6[14] <= msg_d5[14];  msg_d6[15] <= msg_d5[15];
        msg_d7[ 0] <= msg_d6[ 0];  msg_d7[ 1] <= msg_d6[ 1];  msg_d7[ 2] <= msg_d6[ 2];  msg_d7[ 3] <= msg_d6[ 3];
        msg_d7[ 4] <= msg_d6[ 4];  msg_d7[ 5] <= msg_d6[ 5];  msg_d7[ 6] <= msg_d6[ 6];  msg_d7[ 7] <= msg_d6[ 7];
        msg_d7[ 8] <= msg_d6[ 8];  msg_d7[ 9] <= msg_d6[ 9];  msg_d7[10] <= msg_d6[10];  msg_d7[11] <= msg_d6[11];
        msg_d7[12] <= msg_d6[12];  msg_d7[13] <= msg_d6[13];  msg_d7[14] <= msg_d6[14];  msg_d7[15] <= msg_d6[15];
        msg_d8[ 0] <= msg_d7[ 0];  msg_d8[ 1] <= msg_d7[ 1];  msg_d8[ 2] <= msg_d7[ 2];  msg_d8[ 3] <= msg_d7[ 3];
        msg_d8[ 4] <= msg_d7[ 4];  msg_d8[ 5] <= msg_d7[ 5];  msg_d8[ 6] <= msg_d7[ 6];  msg_d8[ 7] <= msg_d7[ 7];
        msg_d8[ 8] <= msg_d7[ 8];  msg_d8[ 9] <= msg_d7[ 9];  msg_d8[10] <= msg_d7[10];  msg_d8[11] <= msg_d7[11];
        msg_d8[12] <= msg_d7[12];  msg_d8[13] <= msg_d7[13];  msg_d8[14] <= msg_d7[14];  msg_d8[15] <= msg_d7[15];
        msg_d9[ 0] <= msg_d8[ 0];  msg_d9[ 1] <= msg_d8[ 1];  msg_d9[ 2] <= msg_d8[ 2];  msg_d9[ 3] <= msg_d8[ 3];
        msg_d9[ 4] <= msg_d8[ 4];  msg_d9[ 5] <= msg_d8[ 5];  msg_d9[ 6] <= msg_d8[ 6];  msg_d9[ 7] <= msg_d8[ 7];
        msg_d9[ 8] <= msg_d8[ 8];  msg_d9[ 9] <= msg_d8[ 9];  msg_d9[10] <= msg_d8[10];  msg_d9[11] <= msg_d8[11];
        msg_d9[12] <= msg_d8[12];  msg_d9[13] <= msg_d8[13];  msg_d9[14] <= msg_d8[14];  msg_d9[15] <= msg_d8[15];
        msg_d10[ 0] <= msg_d9[ 0];  msg_d10[ 1] <= msg_d9[ 1];  msg_d10[ 2] <= msg_d9[ 2];  msg_d10[ 3] <= msg_d9[ 3];
        msg_d10[ 4] <= msg_d9[ 4];  msg_d10[ 5] <= msg_d9[ 5];  msg_d10[ 6] <= msg_d9[ 6];  msg_d10[ 7] <= msg_d9[ 7];
        msg_d10[ 8] <= msg_d9[ 8];  msg_d10[ 9] <= msg_d9[ 9];  msg_d10[10] <= msg_d9[10];  msg_d10[11] <= msg_d9[11];
        msg_d10[12] <= msg_d9[12];  msg_d10[13] <= msg_d9[13];  msg_d10[14] <= msg_d9[14];  msg_d10[15] <= msg_d9[15];
        msg_d11[ 0] <= msg_d10[ 0];  msg_d11[ 1] <= msg_d10[ 1];  msg_d11[ 2] <= msg_d10[ 2];  msg_d11[ 3] <= msg_d10[ 3];
        msg_d11[ 4] <= msg_d10[ 4];  msg_d11[ 5] <= msg_d10[ 5];  msg_d11[ 6] <= msg_d10[ 6];  msg_d11[ 7] <= msg_d10[ 7];
        msg_d11[ 8] <= msg_d10[ 8];  msg_d11[ 9] <= msg_d10[ 9];  msg_d11[10] <= msg_d10[10];  msg_d11[11] <= msg_d10[11];
        msg_d11[12] <= msg_d10[12];  msg_d11[13] <= msg_d10[13];  msg_d11[14] <= msg_d10[14];  msg_d11[15] <= msg_d10[15];
        msg_d12[ 0] <= msg_d11[ 0];  msg_d12[ 1] <= msg_d11[ 1];  msg_d12[ 2] <= msg_d11[ 2];  msg_d12[ 3] <= msg_d11[ 3];
        msg_d12[ 4] <= msg_d11[ 4];  msg_d12[ 5] <= msg_d11[ 5];  msg_d12[ 6] <= msg_d11[ 6];  msg_d12[ 7] <= msg_d11[ 7];
        msg_d12[ 8] <= msg_d11[ 8];  msg_d12[ 9] <= msg_d11[ 9];  msg_d12[10] <= msg_d11[10];  msg_d12[11] <= msg_d11[11];
        msg_d12[12] <= msg_d11[12];  msg_d12[13] <= msg_d11[13];  msg_d12[14] <= msg_d11[14];  msg_d12[15] <= msg_d11[15];
    end

    // =========================================================================
    // Round instantiations — fully unrolled, all ports are simple 1D arrays
    // =========================================================================

    blake3_round u_r0 (
        .clk       (clk),
        .rst_n     (rst_n),
        .state_in  (state_s0),
        .msg       (msg_s0),
        .round_idx (3'd0),
        .in_valid  (valid_s0),
        .state_out (),          // unconnected — hierarchical bypass below
        .out_valid (valid_r0)
    );

    blake3_round u_r1 (
        .clk       (clk),
        .rst_n     (rst_n),
        .state_in  (state_r0),
        .msg       (msg_d2),
        .round_idx (3'd1),
        .in_valid  (valid_r0),
        .state_out (),          // unconnected — hierarchical bypass below
        .out_valid (valid_r1)
    );

    blake3_round u_r2 (
        .clk       (clk),
        .rst_n     (rst_n),
        .state_in  (state_r1),
        .msg       (msg_d4),
        .round_idx (3'd2),
        .in_valid  (valid_r1),
        .state_out (),          // unconnected — hierarchical bypass below
        .out_valid (valid_r2)
    );

    blake3_round u_r3 (
        .clk       (clk),
        .rst_n     (rst_n),
        .state_in  (state_r2),
        .msg       (msg_d6),
        .round_idx (3'd3),
        .in_valid  (valid_r2),
        .state_out (),          // unconnected — hierarchical bypass below
        .out_valid (valid_r3)
    );

    blake3_round u_r4 (
        .clk       (clk),
        .rst_n     (rst_n),
        .state_in  (state_r3),
        .msg       (msg_d8),
        .round_idx (3'd4),
        .in_valid  (valid_r3),
        .state_out (),          // unconnected — hierarchical bypass below
        .out_valid (valid_r4)
    );

    blake3_round u_r5 (
        .clk       (clk),
        .rst_n     (rst_n),
        .state_in  (state_r4),
        .msg       (msg_d10),
        .round_idx (3'd5),
        .in_valid  (valid_r4),
        .state_out (),          // unconnected — hierarchical bypass below
        .out_valid (valid_r5)
    );

    blake3_round u_r6 (
        .clk       (clk),
        .rst_n     (rst_n),
        .state_in  (state_r5),
        .msg       (msg_d12),
        .round_idx (3'd6),
        .in_valid  (valid_r5),
        .state_out (),          // unconnected — hash_out uses u_r6.s2_state directly
        .out_valid (valid_r6)
    );

    // =========================================================================
    // Hierarchical inter-round state bypass (iverilog 11 unpacked array
    // output-port update-propagation bug: connected wires don't see updates).
    // state_out ports are left unconnected above; s2_state is read directly.
    // =========================================================================
    assign state_r0[ 0] = u_r0.s2_state[ 0]; assign state_r0[ 1] = u_r0.s2_state[ 1];
    assign state_r0[ 2] = u_r0.s2_state[ 2]; assign state_r0[ 3] = u_r0.s2_state[ 3];
    assign state_r0[ 4] = u_r0.s2_state[ 4]; assign state_r0[ 5] = u_r0.s2_state[ 5];
    assign state_r0[ 6] = u_r0.s2_state[ 6]; assign state_r0[ 7] = u_r0.s2_state[ 7];
    assign state_r0[ 8] = u_r0.s2_state[ 8]; assign state_r0[ 9] = u_r0.s2_state[ 9];
    assign state_r0[10] = u_r0.s2_state[10]; assign state_r0[11] = u_r0.s2_state[11];
    assign state_r0[12] = u_r0.s2_state[12]; assign state_r0[13] = u_r0.s2_state[13];
    assign state_r0[14] = u_r0.s2_state[14]; assign state_r0[15] = u_r0.s2_state[15];

    assign state_r1[ 0] = u_r1.s2_state[ 0]; assign state_r1[ 1] = u_r1.s2_state[ 1];
    assign state_r1[ 2] = u_r1.s2_state[ 2]; assign state_r1[ 3] = u_r1.s2_state[ 3];
    assign state_r1[ 4] = u_r1.s2_state[ 4]; assign state_r1[ 5] = u_r1.s2_state[ 5];
    assign state_r1[ 6] = u_r1.s2_state[ 6]; assign state_r1[ 7] = u_r1.s2_state[ 7];
    assign state_r1[ 8] = u_r1.s2_state[ 8]; assign state_r1[ 9] = u_r1.s2_state[ 9];
    assign state_r1[10] = u_r1.s2_state[10]; assign state_r1[11] = u_r1.s2_state[11];
    assign state_r1[12] = u_r1.s2_state[12]; assign state_r1[13] = u_r1.s2_state[13];
    assign state_r1[14] = u_r1.s2_state[14]; assign state_r1[15] = u_r1.s2_state[15];

    assign state_r2[ 0] = u_r2.s2_state[ 0]; assign state_r2[ 1] = u_r2.s2_state[ 1];
    assign state_r2[ 2] = u_r2.s2_state[ 2]; assign state_r2[ 3] = u_r2.s2_state[ 3];
    assign state_r2[ 4] = u_r2.s2_state[ 4]; assign state_r2[ 5] = u_r2.s2_state[ 5];
    assign state_r2[ 6] = u_r2.s2_state[ 6]; assign state_r2[ 7] = u_r2.s2_state[ 7];
    assign state_r2[ 8] = u_r2.s2_state[ 8]; assign state_r2[ 9] = u_r2.s2_state[ 9];
    assign state_r2[10] = u_r2.s2_state[10]; assign state_r2[11] = u_r2.s2_state[11];
    assign state_r2[12] = u_r2.s2_state[12]; assign state_r2[13] = u_r2.s2_state[13];
    assign state_r2[14] = u_r2.s2_state[14]; assign state_r2[15] = u_r2.s2_state[15];

    assign state_r3[ 0] = u_r3.s2_state[ 0]; assign state_r3[ 1] = u_r3.s2_state[ 1];
    assign state_r3[ 2] = u_r3.s2_state[ 2]; assign state_r3[ 3] = u_r3.s2_state[ 3];
    assign state_r3[ 4] = u_r3.s2_state[ 4]; assign state_r3[ 5] = u_r3.s2_state[ 5];
    assign state_r3[ 6] = u_r3.s2_state[ 6]; assign state_r3[ 7] = u_r3.s2_state[ 7];
    assign state_r3[ 8] = u_r3.s2_state[ 8]; assign state_r3[ 9] = u_r3.s2_state[ 9];
    assign state_r3[10] = u_r3.s2_state[10]; assign state_r3[11] = u_r3.s2_state[11];
    assign state_r3[12] = u_r3.s2_state[12]; assign state_r3[13] = u_r3.s2_state[13];
    assign state_r3[14] = u_r3.s2_state[14]; assign state_r3[15] = u_r3.s2_state[15];

    assign state_r4[ 0] = u_r4.s2_state[ 0]; assign state_r4[ 1] = u_r4.s2_state[ 1];
    assign state_r4[ 2] = u_r4.s2_state[ 2]; assign state_r4[ 3] = u_r4.s2_state[ 3];
    assign state_r4[ 4] = u_r4.s2_state[ 4]; assign state_r4[ 5] = u_r4.s2_state[ 5];
    assign state_r4[ 6] = u_r4.s2_state[ 6]; assign state_r4[ 7] = u_r4.s2_state[ 7];
    assign state_r4[ 8] = u_r4.s2_state[ 8]; assign state_r4[ 9] = u_r4.s2_state[ 9];
    assign state_r4[10] = u_r4.s2_state[10]; assign state_r4[11] = u_r4.s2_state[11];
    assign state_r4[12] = u_r4.s2_state[12]; assign state_r4[13] = u_r4.s2_state[13];
    assign state_r4[14] = u_r4.s2_state[14]; assign state_r4[15] = u_r4.s2_state[15];

    assign state_r5[ 0] = u_r5.s2_state[ 0]; assign state_r5[ 1] = u_r5.s2_state[ 1];
    assign state_r5[ 2] = u_r5.s2_state[ 2]; assign state_r5[ 3] = u_r5.s2_state[ 3];
    assign state_r5[ 4] = u_r5.s2_state[ 4]; assign state_r5[ 5] = u_r5.s2_state[ 5];
    assign state_r5[ 6] = u_r5.s2_state[ 6]; assign state_r5[ 7] = u_r5.s2_state[ 7];
    assign state_r5[ 8] = u_r5.s2_state[ 8]; assign state_r5[ 9] = u_r5.s2_state[ 9];
    assign state_r5[10] = u_r5.s2_state[10]; assign state_r5[11] = u_r5.s2_state[11];
    assign state_r5[12] = u_r5.s2_state[12]; assign state_r5[13] = u_r5.s2_state[13];
    assign state_r5[14] = u_r5.s2_state[14]; assign state_r5[15] = u_r5.s2_state[15];

    assign state_r6[ 0] = u_r6.s2_state[ 0]; assign state_r6[ 1] = u_r6.s2_state[ 1];
    assign state_r6[ 2] = u_r6.s2_state[ 2]; assign state_r6[ 3] = u_r6.s2_state[ 3];
    assign state_r6[ 4] = u_r6.s2_state[ 4]; assign state_r6[ 5] = u_r6.s2_state[ 5];
    assign state_r6[ 6] = u_r6.s2_state[ 6]; assign state_r6[ 7] = u_r6.s2_state[ 7];
    assign state_r6[ 8] = u_r6.s2_state[ 8]; assign state_r6[ 9] = u_r6.s2_state[ 9];
    assign state_r6[10] = u_r6.s2_state[10]; assign state_r6[11] = u_r6.s2_state[11];
    assign state_r6[12] = u_r6.s2_state[12]; assign state_r6[13] = u_r6.s2_state[13];
    assign state_r6[14] = u_r6.s2_state[14]; assign state_r6[15] = u_r6.s2_state[15];

    // =========================================================================
    // Output: XOR final state halves
    // =========================================================================
    // BLAKE3 finalization: output[i] = state[i] ^ state[i+8]  for i in 0..7
    //
    // iverilog 11: continuous assigns using two elements of the same NBA-driven
    // hierarchical array may not re-evaluate. Fix: route each element through a
    // scalar bridge first (single-element hierarchical → scalar is confirmed to
    // work), then XOR pairs of scalars.

    assign out_valid = valid_r6;

    logic [31:0] r6h_0,  r6h_1,  r6h_2,  r6h_3;
    logic [31:0] r6h_4,  r6h_5,  r6h_6,  r6h_7;
    logic [31:0] r6h_8,  r6h_9,  r6h_10, r6h_11;
    logic [31:0] r6h_12, r6h_13, r6h_14, r6h_15;

    assign r6h_0  = u_r6.s2_state[ 0]; assign r6h_8  = u_r6.s2_state[ 8];
    assign r6h_1  = u_r6.s2_state[ 1]; assign r6h_9  = u_r6.s2_state[ 9];
    assign r6h_2  = u_r6.s2_state[ 2]; assign r6h_10 = u_r6.s2_state[10];
    assign r6h_3  = u_r6.s2_state[ 3]; assign r6h_11 = u_r6.s2_state[11];
    assign r6h_4  = u_r6.s2_state[ 4]; assign r6h_12 = u_r6.s2_state[12];
    assign r6h_5  = u_r6.s2_state[ 5]; assign r6h_13 = u_r6.s2_state[13];
    assign r6h_6  = u_r6.s2_state[ 6]; assign r6h_14 = u_r6.s2_state[14];
    assign r6h_7  = u_r6.s2_state[ 7]; assign r6h_15 = u_r6.s2_state[15];

    assign hash_out[0] = r6h_0 ^ r6h_8;
    assign hash_out[1] = r6h_1 ^ r6h_9;
    assign hash_out[2] = r6h_2 ^ r6h_10;
    assign hash_out[3] = r6h_3 ^ r6h_11;
    assign hash_out[4] = r6h_4 ^ r6h_12;
    assign hash_out[5] = r6h_5 ^ r6h_13;
    assign hash_out[6] = r6h_6 ^ r6h_14;
    assign hash_out[7] = r6h_7 ^ r6h_15;

endmodule
