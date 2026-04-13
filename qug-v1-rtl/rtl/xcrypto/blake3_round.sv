// =============================================================================
// blake3_round.sv — 2-Stage Pipelined BLAKE3 Round (Column then Diagonal)
// QUG-V1 Mining SoC — Xcrypto BLAKE3 Hardware Pipeline
// =============================================================================
//
// Implements one full BLAKE3 round split into 2 pipeline stages:
//   Stage 1 (clk edge): Latch column quarter-round results
//   Stage 2 (clk edge): Latch diagonal quarter-round results
//
// This halves the combinational depth vs the single-stage version (~6 levels
// of 32-bit add per stage instead of ~12), meeting 100 MHz on Kintex-7.
//
// BLAKE3 state layout (16 x 32-bit words):
//   [ v0  v1  v2  v3 ]   <- a row
//   [ v4  v5  v6  v7 ]   <- b row
//   [ v8  v9  v10 v11]   <- c row
//   [ v12 v13 v14 v15]   <- d row
//
// Column round operates on columns:  (0,4,8,12) (1,5,9,13) (2,6,10,14) (3,7,11,15)
// Diagonal round operates on diags:  (0,5,10,15) (1,6,11,12) (2,7,8,13) (3,4,9,14)
//
// Each quarter-round (G function):
//   a = a + b + mx;  d = (d ^ a) >>> 16;
//   c = c + d;       b = (b ^ c) >>> 12;
//   a = a + b + my;  d = (d ^ a) >>> 8;
//   c = c + d;       b = (b ^ c) >>> 7;
//
// Latency: 2 clock cycles per round.
// =============================================================================

module blake3_round (
    input  logic        clk,
    input  logic        rst_n,

    // Input state: 16 x 32-bit words
    input  logic [31:0] state_in  [0:15],
    // Message words for this round: 16 x 32-bit
    input  logic [31:0] msg       [0:15],
    // Round index (0-6) — selects message schedule permutation
    input  logic [2:0]  round_idx,
    // Input valid — indicates state_in/msg/round_idx are valid
    input  logic        in_valid,

    // Output state: 16 x 32-bit words (after both column + diagonal)
    output logic [31:0] state_out [0:15],
    // Output valid — high for 1 cycle when state_out is ready
    output logic        out_valid
);

    // =========================================================================
    // BLAKE3 message schedule permutations
    // =========================================================================
    // Pre-computed permutation tables for all 7 rounds.
    // sigma[round][i] gives the message word index for position i.

    logic [3:0] sigma [0:6][0:15];

    always_comb begin
        // Round 0: identity
        sigma[0][ 0] = 4'd0;  sigma[0][ 1] = 4'd1;  sigma[0][ 2] = 4'd2;  sigma[0][ 3] = 4'd3;
        sigma[0][ 4] = 4'd4;  sigma[0][ 5] = 4'd5;  sigma[0][ 6] = 4'd6;  sigma[0][ 7] = 4'd7;
        sigma[0][ 8] = 4'd8;  sigma[0][ 9] = 4'd9;  sigma[0][10] = 4'd10; sigma[0][11] = 4'd11;
        sigma[0][12] = 4'd12; sigma[0][13] = 4'd13; sigma[0][14] = 4'd14; sigma[0][15] = 4'd15;

        // Round 1
        sigma[1][ 0] = 4'd2;  sigma[1][ 1] = 4'd6;  sigma[1][ 2] = 4'd3;  sigma[1][ 3] = 4'd10;
        sigma[1][ 4] = 4'd7;  sigma[1][ 5] = 4'd0;  sigma[1][ 6] = 4'd4;  sigma[1][ 7] = 4'd13;
        sigma[1][ 8] = 4'd1;  sigma[1][ 9] = 4'd11; sigma[1][10] = 4'd12; sigma[1][11] = 4'd5;
        sigma[1][12] = 4'd9;  sigma[1][13] = 4'd14; sigma[1][14] = 4'd15; sigma[1][15] = 4'd8;

        // Round 2
        sigma[2][ 0] = 4'd3;  sigma[2][ 1] = 4'd4;  sigma[2][ 2] = 4'd10; sigma[2][ 3] = 4'd12;
        sigma[2][ 4] = 4'd13; sigma[2][ 5] = 4'd2;  sigma[2][ 6] = 4'd7;  sigma[2][ 7] = 4'd14;
        sigma[2][ 8] = 4'd6;  sigma[2][ 9] = 4'd5;  sigma[2][10] = 4'd9;  sigma[2][11] = 4'd0;
        sigma[2][12] = 4'd11; sigma[2][13] = 4'd15; sigma[2][14] = 4'd8;  sigma[2][15] = 4'd1;

        // Round 3
        sigma[3][ 0] = 4'd10; sigma[3][ 1] = 4'd7;  sigma[3][ 2] = 4'd12; sigma[3][ 3] = 4'd9;
        sigma[3][ 4] = 4'd14; sigma[3][ 5] = 4'd3;  sigma[3][ 6] = 4'd13; sigma[3][ 7] = 4'd15;
        sigma[3][ 8] = 4'd4;  sigma[3][ 9] = 4'd0;  sigma[3][10] = 4'd11; sigma[3][11] = 4'd2;
        sigma[3][12] = 4'd5;  sigma[3][13] = 4'd8;  sigma[3][14] = 4'd1;  sigma[3][15] = 4'd6;

        // Round 4
        sigma[4][ 0] = 4'd12; sigma[4][ 1] = 4'd13; sigma[4][ 2] = 4'd9;  sigma[4][ 3] = 4'd11;
        sigma[4][ 4] = 4'd15; sigma[4][ 5] = 4'd10; sigma[4][ 6] = 4'd14; sigma[4][ 7] = 4'd8;
        sigma[4][ 8] = 4'd7;  sigma[4][ 9] = 4'd2;  sigma[4][10] = 4'd5;  sigma[4][11] = 4'd3;
        sigma[4][12] = 4'd0;  sigma[4][13] = 4'd1;  sigma[4][14] = 4'd6;  sigma[4][15] = 4'd4;

        // Round 5
        sigma[5][ 0] = 4'd9;  sigma[5][ 1] = 4'd14; sigma[5][ 2] = 4'd11; sigma[5][ 3] = 4'd5;
        sigma[5][ 4] = 4'd8;  sigma[5][ 5] = 4'd12; sigma[5][ 6] = 4'd15; sigma[5][ 7] = 4'd1;
        sigma[5][ 8] = 4'd13; sigma[5][ 9] = 4'd3;  sigma[5][10] = 4'd0;  sigma[5][11] = 4'd10;
        sigma[5][12] = 4'd2;  sigma[5][13] = 4'd6;  sigma[5][14] = 4'd4;  sigma[5][15] = 4'd7;

        // Round 6
        sigma[6][ 0] = 4'd11; sigma[6][ 1] = 4'd15; sigma[6][ 2] = 4'd5;  sigma[6][ 3] = 4'd0;
        sigma[6][ 4] = 4'd1;  sigma[6][ 5] = 4'd9;  sigma[6][ 6] = 4'd8;  sigma[6][ 7] = 4'd6;
        sigma[6][ 8] = 4'd14; sigma[6][ 9] = 4'd10; sigma[6][10] = 4'd2;  sigma[6][11] = 4'd12;
        sigma[6][12] = 4'd3;  sigma[6][13] = 4'd4;  sigma[6][14] = 4'd7;  sigma[6][15] = 4'd13;
    end

    // =========================================================================
    // Scheduled message words for this round
    // =========================================================================
    logic [31:0] m [0:15];

    always_comb begin
        for (int i = 0; i < 16; i++) begin
            m[i] = msg[sigma[round_idx][i]];
        end
    end

    // =========================================================================
    // Quarter-round G function (pure combinational)
    // =========================================================================
    function automatic logic [127:0] quarter_round(
        input logic [31:0] a, b, c, d, mx, my
    );
        logic [31:0] a1, b1, c1, d1;
        logic [31:0] a2, b2, c2, d2;

        // Step 1
        a1 = a + b + mx;
        d1 = {(d ^ a1)[15:0], (d ^ a1)[31:16]};  // ror32 by 16
        c1 = c + d1;
        b1 = {(b ^ c1)[11:0], (b ^ c1)[31:12]};  // ror32 by 12

        // Step 2
        a2 = a1 + b1 + my;
        d2 = {(d1 ^ a2)[7:0], (d1 ^ a2)[31:8]};  // ror32 by 8
        c2 = c1 + d2;
        b2 = {(b1 ^ c2)[6:0], (b1 ^ c2)[31:7]};  // ror32 by 7

        quarter_round = {a2, b2, c2, d2};
    endfunction

    // =========================================================================
    // Stage 1: Column quarter-rounds (combinational)
    // =========================================================================
    // Column 0: G(v0, v4, v8,  v12, m[0],  m[1])
    // Column 1: G(v1, v5, v9,  v13, m[2],  m[3])
    // Column 2: G(v2, v6, v10, v14, m[4],  m[5])
    // Column 3: G(v3, v7, v11, v15, m[6],  m[7])

    logic [31:0] col_state [0:15];
    logic [127:0] col0_result, col1_result, col2_result, col3_result;

    always_comb begin
        col0_result = quarter_round(state_in[ 0], state_in[ 4], state_in[ 8], state_in[12], m[ 0], m[ 1]);
        col1_result = quarter_round(state_in[ 1], state_in[ 5], state_in[ 9], state_in[13], m[ 2], m[ 3]);
        col2_result = quarter_round(state_in[ 2], state_in[ 6], state_in[10], state_in[14], m[ 4], m[ 5]);
        col3_result = quarter_round(state_in[ 3], state_in[ 7], state_in[11], state_in[15], m[ 6], m[ 7]);

        // Unpack column results: {a, b, c, d}
        col_state[ 0] = col0_result[127:96]; col_state[ 4] = col0_result[95:64];
        col_state[ 8] = col0_result[ 63:32]; col_state[12] = col0_result[31: 0];

        col_state[ 1] = col1_result[127:96]; col_state[ 5] = col1_result[95:64];
        col_state[ 9] = col1_result[ 63:32]; col_state[13] = col1_result[31: 0];

        col_state[ 2] = col2_result[127:96]; col_state[ 6] = col2_result[95:64];
        col_state[10] = col2_result[ 63:32]; col_state[14] = col2_result[31: 0];

        col_state[ 3] = col3_result[127:96]; col_state[ 7] = col3_result[95:64];
        col_state[11] = col3_result[ 63:32]; col_state[15] = col3_result[31: 0];
    end

    // =========================================================================
    // Stage 1 pipeline register: latch column results + message words
    // =========================================================================
    logic [31:0] s1_state [0:15];
    logic [31:0] s1_msg   [0:15];
    logic        s1_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            for (int j = 0; j < 16; j++) begin
                s1_state[j] <= '0;
                s1_msg[j]   <= '0;
            end
        end else begin
            s1_valid <= in_valid;
            if (in_valid) begin
                for (int j = 0; j < 16; j++) begin
                    s1_state[j] <= col_state[j];
                    s1_msg[j]   <= m[j];
                end
            end
        end
    end

    // =========================================================================
    // Stage 2: Diagonal quarter-rounds (combinational, fed from s1 regs)
    // =========================================================================
    // Diag 0: G(v0, v5, v10, v15, m[8],  m[9])
    // Diag 1: G(v1, v6, v11, v12, m[10], m[11])
    // Diag 2: G(v2, v7, v8,  v13, m[12], m[13])
    // Diag 3: G(v3, v4, v9,  v14, m[14], m[15])

    logic [31:0] diag_state [0:15];
    logic [127:0] diag0_result, diag1_result, diag2_result, diag3_result;

    always_comb begin
        diag0_result = quarter_round(s1_state[ 0], s1_state[ 5], s1_state[10], s1_state[15], s1_msg[ 8], s1_msg[ 9]);
        diag1_result = quarter_round(s1_state[ 1], s1_state[ 6], s1_state[11], s1_state[12], s1_msg[10], s1_msg[11]);
        diag2_result = quarter_round(s1_state[ 2], s1_state[ 7], s1_state[ 8], s1_state[13], s1_msg[12], s1_msg[13]);
        diag3_result = quarter_round(s1_state[ 3], s1_state[ 4], s1_state[ 9], s1_state[14], s1_msg[14], s1_msg[15]);

        // Unpack diagonal results back into linear state
        diag_state[ 0] = diag0_result[127:96];
        diag_state[ 5] = diag0_result[ 95:64];
        diag_state[10] = diag0_result[ 63:32];
        diag_state[15] = diag0_result[ 31: 0];

        diag_state[ 1] = diag1_result[127:96];
        diag_state[ 6] = diag1_result[ 95:64];
        diag_state[11] = diag1_result[ 63:32];
        diag_state[12] = diag1_result[ 31: 0];

        diag_state[ 2] = diag2_result[127:96];
        diag_state[ 7] = diag2_result[ 95:64];
        diag_state[ 8] = diag2_result[ 63:32];
        diag_state[13] = diag2_result[ 31: 0];

        diag_state[ 3] = diag3_result[127:96];
        diag_state[ 4] = diag3_result[ 95:64];
        diag_state[ 9] = diag3_result[ 63:32];
        diag_state[14] = diag3_result[ 31: 0];
    end

    // =========================================================================
    // Stage 2 pipeline register: latch diagonal results (final output)
    // =========================================================================
    logic [31:0] s2_state [0:15];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            for (int j = 0; j < 16; j++) begin
                s2_state[j] <= '0;
            end
        end else begin
            out_valid <= s1_valid;
            if (s1_valid) begin
                for (int j = 0; j < 16; j++) begin
                    s2_state[j] <= diag_state[j];
                end
            end
        end
    end

    // Output assignment
    always_comb begin
        for (int i = 0; i < 16; i++) begin
            state_out[i] = s2_state[i];
        end
    end

endmodule
