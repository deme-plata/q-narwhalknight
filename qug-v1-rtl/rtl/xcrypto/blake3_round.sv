// =============================================================================
// blake3_round.sv — 2-Stage Pipelined BLAKE3 Round (Column then Diagonal)
// QUG-V1 Mining SoC — Xcrypto BLAKE3 Hardware Pipeline
// =============================================================================
//
// Implements one full BLAKE3 round split into 2 pipeline stages:
//   Stage 1 (clk edge): Latch column quarter-round results
//   Stage 2 (clk edge): Latch diagonal quarter-round results
//
// BLAKE3 state layout (16 x 32-bit words):
//   [ v0  v1  v2  v3 ]   <- a row
//   [ v4  v5  v6  v7 ]   <- b row
//   [ v8  v9  v10 v11]   <- c row
//   [ v12 v13 v14 v15]   <- d row
//
// Column round:   (0,4,8,12) (1,5,9,13) (2,6,10,14) (3,7,11,15)
// Diagonal round: (0,5,10,15) (1,6,11,12) (2,7,8,13) (3,4,9,14)
//
// Each quarter-round (G function):
//   a = a + b + mx;  d = (d ^ a) >>> 16;
//   c = c + d;       b = (b ^ c) >>> 12;
//   a = a + b + my;  d = (d ^ a) >>> 8;
//   c = c + d;       b = (b ^ c) >>> 7;
//
// Latency: 2 clock cycles per round.
//
// iverilog 11 compatibility notes:
//   - Dynamic array indexing in always_* not supported → case statement for sigma
//   - Function calls + part-selects in always_* broken → use assign statements
// =============================================================================

module blake3_round (
    input  logic        clk,
    input  logic        rst_n,

    input  logic [31:0] state_in  [0:15],
    input  logic [31:0] msg       [0:15],
    input  logic [2:0]  round_idx,
    input  logic        in_valid,

    output logic [31:0] state_out [0:15],
    output logic        out_valid
);

    // =========================================================================
    // Message schedule — case on round_idx (avoids dynamic 2D array indexing)
    // =========================================================================
    // BLAKE3 permutation: {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8} applied
    // cumulatively each round. Pre-computed for rounds 0-6.
    // =========================================================================
    logic [31:0] m [0:15];

    always_comb begin
        case (round_idx)
            3'd0: begin  // identity
                m[ 0] = msg[ 0]; m[ 1] = msg[ 1]; m[ 2] = msg[ 2]; m[ 3] = msg[ 3];
                m[ 4] = msg[ 4]; m[ 5] = msg[ 5]; m[ 6] = msg[ 6]; m[ 7] = msg[ 7];
                m[ 8] = msg[ 8]; m[ 9] = msg[ 9]; m[10] = msg[10]; m[11] = msg[11];
                m[12] = msg[12]; m[13] = msg[13]; m[14] = msg[14]; m[15] = msg[15];
            end
            3'd1: begin  // perm^1: {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8}
                m[ 0] = msg[ 2]; m[ 1] = msg[ 6]; m[ 2] = msg[ 3]; m[ 3] = msg[10];
                m[ 4] = msg[ 7]; m[ 5] = msg[ 0]; m[ 6] = msg[ 4]; m[ 7] = msg[13];
                m[ 8] = msg[ 1]; m[ 9] = msg[11]; m[10] = msg[12]; m[11] = msg[ 5];
                m[12] = msg[ 9]; m[13] = msg[14]; m[14] = msg[15]; m[15] = msg[ 8];
            end
            3'd2: begin  // perm^2
                m[ 0] = msg[ 3]; m[ 1] = msg[ 4]; m[ 2] = msg[10]; m[ 3] = msg[12];
                m[ 4] = msg[13]; m[ 5] = msg[ 2]; m[ 6] = msg[ 7]; m[ 7] = msg[14];
                m[ 8] = msg[ 6]; m[ 9] = msg[ 5]; m[10] = msg[ 9]; m[11] = msg[ 0];
                m[12] = msg[11]; m[13] = msg[15]; m[14] = msg[ 8]; m[15] = msg[ 1];
            end
            3'd3: begin  // perm^3
                m[ 0] = msg[10]; m[ 1] = msg[ 7]; m[ 2] = msg[12]; m[ 3] = msg[ 9];
                m[ 4] = msg[14]; m[ 5] = msg[ 3]; m[ 6] = msg[13]; m[ 7] = msg[15];
                m[ 8] = msg[ 4]; m[ 9] = msg[ 0]; m[10] = msg[11]; m[11] = msg[ 2];
                m[12] = msg[ 5]; m[13] = msg[ 8]; m[14] = msg[ 1]; m[15] = msg[ 6];
            end
            3'd4: begin  // perm^4
                m[ 0] = msg[12]; m[ 1] = msg[13]; m[ 2] = msg[ 9]; m[ 3] = msg[11];
                m[ 4] = msg[15]; m[ 5] = msg[10]; m[ 6] = msg[14]; m[ 7] = msg[ 8];
                m[ 8] = msg[ 7]; m[ 9] = msg[ 2]; m[10] = msg[ 5]; m[11] = msg[ 3];
                m[12] = msg[ 0]; m[13] = msg[ 1]; m[14] = msg[ 6]; m[15] = msg[ 4];
            end
            3'd5: begin  // perm^5
                m[ 0] = msg[ 9]; m[ 1] = msg[14]; m[ 2] = msg[11]; m[ 3] = msg[ 5];
                m[ 4] = msg[ 8]; m[ 5] = msg[12]; m[ 6] = msg[15]; m[ 7] = msg[ 1];
                m[ 8] = msg[13]; m[ 9] = msg[ 3]; m[10] = msg[ 0]; m[11] = msg[10];
                m[12] = msg[ 2]; m[13] = msg[ 6]; m[14] = msg[ 4]; m[15] = msg[ 7];
            end
            default: begin  // round 6: perm^6
                m[ 0] = msg[11]; m[ 1] = msg[15]; m[ 2] = msg[ 5]; m[ 3] = msg[ 0];
                m[ 4] = msg[ 1]; m[ 5] = msg[ 9]; m[ 6] = msg[ 8]; m[ 7] = msg[ 6];
                m[ 8] = msg[14]; m[ 9] = msg[10]; m[10] = msg[ 2]; m[11] = msg[12];
                m[12] = msg[ 3]; m[13] = msg[ 4]; m[14] = msg[ 7]; m[15] = msg[13];
            end
        endcase
    end

    // =========================================================================
    // Quarter-round G function (pure combinational, returns {a,b,c,d} packed)
    // =========================================================================
    function automatic logic [127:0] quarter_round(
        input logic [31:0] a, b, c, d, mx, my
    );
        logic [31:0] a1, b1, c1, d1, a2, b2, c2, d2, tmp;
        a1  = a + b + mx;
        tmp = d ^ a1;  d1 = {tmp[15:0], tmp[31:16]};
        c1  = c + d1;
        tmp = b ^ c1;  b1 = {tmp[11:0], tmp[31:12]};
        a2  = a1 + b1 + my;
        tmp = d1 ^ a2; d2 = {tmp[7:0],  tmp[31:8]};
        c2  = c1 + d2;
        tmp = b1 ^ c2; b2 = {tmp[6:0],  tmp[31:7]};
        quarter_round = {a2, b2, c2, d2};
    endfunction

    // =========================================================================
    // Stage 1: Column quarter-rounds
    // =========================================================================
    // Use continuous assigns (not always_comb) to avoid iverilog 11 issues
    // with function calls and constant part-selects inside always_* blocks.
    //
    // iverilog 11 bug: continuous assigns that read multiple elements of the
    // same NBA-driven (always_ff) unpacked array do not re-evaluate when those
    // elements are written by NBA — only the first element triggers sensitivity.
    // state_in is an input port connected to NBA-driven state_s0 in the pipeline.
    // Fix: route each element through an individual scalar wire first.
    // =========================================================================
    logic [31:0] si0,  si1,  si2,  si3;
    logic [31:0] si4,  si5,  si6,  si7;
    logic [31:0] si8,  si9,  si10, si11;
    logic [31:0] si12, si13, si14, si15;

    assign si0  = state_in[ 0]; assign si1  = state_in[ 1];
    assign si2  = state_in[ 2]; assign si3  = state_in[ 3];
    assign si4  = state_in[ 4]; assign si5  = state_in[ 5];
    assign si6  = state_in[ 6]; assign si7  = state_in[ 7];
    assign si8  = state_in[ 8]; assign si9  = state_in[ 9];
    assign si10 = state_in[10]; assign si11 = state_in[11];
    assign si12 = state_in[12]; assign si13 = state_in[13];
    assign si14 = state_in[14]; assign si15 = state_in[15];

    logic [127:0] col0_r, col1_r, col2_r, col3_r;

    assign col0_r = quarter_round(si0,  si4,  si8,  si12, m[ 0], m[ 1]);
    assign col1_r = quarter_round(si1,  si5,  si9,  si13, m[ 2], m[ 3]);
    assign col2_r = quarter_round(si2,  si6,  si10, si14, m[ 4], m[ 5]);
    assign col3_r = quarter_round(si3,  si7,  si11, si15, m[ 6], m[ 7]);

    // Unpack column results into col_state: {a,b,c,d} = [127:96],[95:64],[63:32],[31:0]
    logic [31:0] col_state [0:15];
    assign col_state[ 0] = col0_r[127:96]; assign col_state[ 4] = col0_r[ 95:64];
    assign col_state[ 8] = col0_r[ 63:32]; assign col_state[12] = col0_r[ 31: 0];
    assign col_state[ 1] = col1_r[127:96]; assign col_state[ 5] = col1_r[ 95:64];
    assign col_state[ 9] = col1_r[ 63:32]; assign col_state[13] = col1_r[ 31: 0];
    assign col_state[ 2] = col2_r[127:96]; assign col_state[ 6] = col2_r[ 95:64];
    assign col_state[10] = col2_r[ 63:32]; assign col_state[14] = col2_r[ 31: 0];
    assign col_state[ 3] = col3_r[127:96]; assign col_state[ 7] = col3_r[ 95:64];
    assign col_state[11] = col3_r[ 63:32]; assign col_state[15] = col3_r[ 31: 0];

    // =========================================================================
    // Stage 1 pipeline register
    // =========================================================================
    logic [31:0] s1_state [0:15];
    logic [31:0] s1_msg   [0:15];
    logic        s1_valid;

    // iverilog 11: for-loop NBA assignments to unpacked arrays in always_ff
    // do not execute — unrolled to individual statements.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid      <= 1'b0;
            s1_state[ 0]  <= '0; s1_state[ 1]  <= '0; s1_state[ 2]  <= '0; s1_state[ 3]  <= '0;
            s1_state[ 4]  <= '0; s1_state[ 5]  <= '0; s1_state[ 6]  <= '0; s1_state[ 7]  <= '0;
            s1_state[ 8]  <= '0; s1_state[ 9]  <= '0; s1_state[10]  <= '0; s1_state[11]  <= '0;
            s1_state[12]  <= '0; s1_state[13]  <= '0; s1_state[14]  <= '0; s1_state[15]  <= '0;
            s1_msg[ 0]    <= '0; s1_msg[ 1]    <= '0; s1_msg[ 2]    <= '0; s1_msg[ 3]    <= '0;
            s1_msg[ 4]    <= '0; s1_msg[ 5]    <= '0; s1_msg[ 6]    <= '0; s1_msg[ 7]    <= '0;
            s1_msg[ 8]    <= '0; s1_msg[ 9]    <= '0; s1_msg[10]    <= '0; s1_msg[11]    <= '0;
            s1_msg[12]    <= '0; s1_msg[13]    <= '0; s1_msg[14]    <= '0; s1_msg[15]    <= '0;
        end else begin
            s1_valid <= in_valid;
            if (in_valid) begin
                s1_state[ 0] <= col_state[ 0]; s1_state[ 1] <= col_state[ 1];
                s1_state[ 2] <= col_state[ 2]; s1_state[ 3] <= col_state[ 3];
                s1_state[ 4] <= col_state[ 4]; s1_state[ 5] <= col_state[ 5];
                s1_state[ 6] <= col_state[ 6]; s1_state[ 7] <= col_state[ 7];
                s1_state[ 8] <= col_state[ 8]; s1_state[ 9] <= col_state[ 9];
                s1_state[10] <= col_state[10]; s1_state[11] <= col_state[11];
                s1_state[12] <= col_state[12]; s1_state[13] <= col_state[13];
                s1_state[14] <= col_state[14]; s1_state[15] <= col_state[15];
                s1_msg[ 0]  <= m[ 0]; s1_msg[ 1]  <= m[ 1];
                s1_msg[ 2]  <= m[ 2]; s1_msg[ 3]  <= m[ 3];
                s1_msg[ 4]  <= m[ 4]; s1_msg[ 5]  <= m[ 5];
                s1_msg[ 6]  <= m[ 6]; s1_msg[ 7]  <= m[ 7];
                s1_msg[ 8]  <= m[ 8]; s1_msg[ 9]  <= m[ 9];
                s1_msg[10]  <= m[10]; s1_msg[11]  <= m[11];
                s1_msg[12]  <= m[12]; s1_msg[13]  <= m[13];
                s1_msg[14]  <= m[14]; s1_msg[15]  <= m[15];
            end
        end
    end

    // =========================================================================
    // Stage 2: Diagonal quarter-rounds
    // =========================================================================
    // iverilog 11 bug: continuous assigns that read multiple elements of the
    // same NBA-driven (always_ff) unpacked array do not re-evaluate when those
    // elements are written by NBA — the sensitivity list is not updated for
    // post-NBA array element changes. Fix: route each element through an
    // individual scalar wire first; single-element array-to-scalar assigns
    // (e.g. "assign s = arr[i]") are confirmed to track NBA updates correctly.
    logic [31:0] s1s_0,  s1s_1,  s1s_2,  s1s_3;
    logic [31:0] s1s_4,  s1s_5,  s1s_6,  s1s_7;
    logic [31:0] s1s_8,  s1s_9,  s1s_10, s1s_11;
    logic [31:0] s1s_12, s1s_13, s1s_14, s1s_15;
    logic [31:0] s1m_8,  s1m_9,  s1m_10, s1m_11;
    logic [31:0] s1m_12, s1m_13, s1m_14, s1m_15;

    assign s1s_0  = s1_state[ 0]; assign s1s_1  = s1_state[ 1];
    assign s1s_2  = s1_state[ 2]; assign s1s_3  = s1_state[ 3];
    assign s1s_4  = s1_state[ 4]; assign s1s_5  = s1_state[ 5];
    assign s1s_6  = s1_state[ 6]; assign s1s_7  = s1_state[ 7];
    assign s1s_8  = s1_state[ 8]; assign s1s_9  = s1_state[ 9];
    assign s1s_10 = s1_state[10]; assign s1s_11 = s1_state[11];
    assign s1s_12 = s1_state[12]; assign s1s_13 = s1_state[13];
    assign s1s_14 = s1_state[14]; assign s1s_15 = s1_state[15];

    assign s1m_8  = s1_msg[ 8]; assign s1m_9  = s1_msg[ 9];
    assign s1m_10 = s1_msg[10]; assign s1m_11 = s1_msg[11];
    assign s1m_12 = s1_msg[12]; assign s1m_13 = s1_msg[13];
    assign s1m_14 = s1_msg[14]; assign s1m_15 = s1_msg[15];

    logic [127:0] diag0_r, diag1_r, diag2_r, diag3_r;

    assign diag0_r = quarter_round(s1s_0,  s1s_5,  s1s_10, s1s_15, s1m_8,  s1m_9);
    assign diag1_r = quarter_round(s1s_1,  s1s_6,  s1s_11, s1s_12, s1m_10, s1m_11);
    assign diag2_r = quarter_round(s1s_2,  s1s_7,  s1s_8,  s1s_13, s1m_12, s1m_13);
    assign diag3_r = quarter_round(s1s_3,  s1s_4,  s1s_9,  s1s_14, s1m_14, s1m_15);

    logic [31:0] diag_state [0:15];
    assign diag_state[ 0] = diag0_r[127:96]; assign diag_state[ 5] = diag0_r[ 95:64];
    assign diag_state[10] = diag0_r[ 63:32]; assign diag_state[15] = diag0_r[ 31: 0];
    assign diag_state[ 1] = diag1_r[127:96]; assign diag_state[ 6] = diag1_r[ 95:64];
    assign diag_state[11] = diag1_r[ 63:32]; assign diag_state[12] = diag1_r[ 31: 0];
    assign diag_state[ 2] = diag2_r[127:96]; assign diag_state[ 7] = diag2_r[ 95:64];
    assign diag_state[ 8] = diag2_r[ 63:32]; assign diag_state[13] = diag2_r[ 31: 0];
    assign diag_state[ 3] = diag3_r[127:96]; assign diag_state[ 4] = diag3_r[ 95:64];
    assign diag_state[ 9] = diag3_r[ 63:32]; assign diag_state[14] = diag3_r[ 31: 0];

    // =========================================================================
    // Stage 2 pipeline register
    // =========================================================================
    logic [31:0] s2_state [0:15];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid     <= 1'b0;
            s2_state[ 0]  <= '0; s2_state[ 1]  <= '0; s2_state[ 2]  <= '0; s2_state[ 3]  <= '0;
            s2_state[ 4]  <= '0; s2_state[ 5]  <= '0; s2_state[ 6]  <= '0; s2_state[ 7]  <= '0;
            s2_state[ 8]  <= '0; s2_state[ 9]  <= '0; s2_state[10]  <= '0; s2_state[11]  <= '0;
            s2_state[12]  <= '0; s2_state[13]  <= '0; s2_state[14]  <= '0; s2_state[15]  <= '0;
        end else begin
            out_valid <= s1_valid;
            if (s1_valid) begin
                s2_state[ 0] <= diag_state[ 0]; s2_state[ 1] <= diag_state[ 1];
                s2_state[ 2] <= diag_state[ 2]; s2_state[ 3] <= diag_state[ 3];
                s2_state[ 4] <= diag_state[ 4]; s2_state[ 5] <= diag_state[ 5];
                s2_state[ 6] <= diag_state[ 6]; s2_state[ 7] <= diag_state[ 7];
                s2_state[ 8] <= diag_state[ 8]; s2_state[ 9] <= diag_state[ 9];
                s2_state[10] <= diag_state[10]; s2_state[11] <= diag_state[11];
                s2_state[12] <= diag_state[12]; s2_state[13] <= diag_state[13];
                s2_state[14] <= diag_state[14]; s2_state[15] <= diag_state[15];
            end
        end
    end

    // Output — individual assigns: iverilog 11 always_comb for-loop does not
    // track array element sensitivity correctly.
    assign state_out[ 0] = s2_state[ 0]; assign state_out[ 1] = s2_state[ 1];
    assign state_out[ 2] = s2_state[ 2]; assign state_out[ 3] = s2_state[ 3];
    assign state_out[ 4] = s2_state[ 4]; assign state_out[ 5] = s2_state[ 5];
    assign state_out[ 6] = s2_state[ 6]; assign state_out[ 7] = s2_state[ 7];
    assign state_out[ 8] = s2_state[ 8]; assign state_out[ 9] = s2_state[ 9];
    assign state_out[10] = s2_state[10]; assign state_out[11] = s2_state[11];
    assign state_out[12] = s2_state[12]; assign state_out[13] = s2_state[13];
    assign state_out[14] = s2_state[14]; assign state_out[15] = s2_state[15];

endmodule
