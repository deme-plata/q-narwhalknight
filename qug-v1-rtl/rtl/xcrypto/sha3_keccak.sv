// =============================================================================
// sha3_keccak.sv — Iterative Keccak-f[1600] Core for SHA-3-256
// QUG-V1 Mining SoC — Xcrypto SHA-3 Hardware Accelerator
// =============================================================================
//
// Implements the full Keccak-f[1600] permutation used in SHA-3-256.
// 1600-bit state (25 x 64-bit lanes, 5x5 grid), 24 rounds at 1 round/cycle.
//
// SHA-3-256 parameters:
//   Rate     = 1088 bits (136 bytes, 17 lanes)
//   Capacity = 512 bits
//   Output   = 256 bits (4 lanes)
//
// FSM: IDLE -> ABSORB -> PERMUTE (24 cycles) -> SQUEEZE -> DONE
//
// Each round executes the five Keccak step mappings in combinational logic:
//   theta -> rho -> pi -> chi -> iota
//
// Target: ~3000 LUT + 1600 FF on Kintex-7 XC7K325T @ 100 MHz
//
// Reference: FIPS 202 (SHA-3 Standard), Keccak Reference Implementation
// =============================================================================

module sha3_keccak (
    input  logic          clk,
    input  logic          rst_n,

    // Control interface
    input  logic          start,          // Pulse to begin hashing
    input  logic          init,           // Initialize (zero) state before absorb

    // Data interface
    input  logic [1087:0] msg_block,      // Rate block (1088 bits for SHA-3-256)
    output logic [255:0]  hash_out,       // 256-bit hash output
    output logic          done,           // Pulse when hash ready
    output logic          busy            // High during permutation
);

    // =========================================================================
    // Keccak Round Constants (RC[0..23])
    // =========================================================================
    // These are derived from a degree-8 LFSR as specified in FIPS 202.

    localparam logic [63:0] RC [0:23] = '{
        64'h0000000000000001, 64'h0000000000008082,
        64'h800000000000808A, 64'h8000000080008000,
        64'h000000000000808B, 64'h0000000080000001,
        64'h8000000080008081, 64'h8000000000008009,
        64'h000000000000008A, 64'h0000000000000088,
        64'h0000000080008009, 64'h000000008000000A,
        64'h000000008000808B, 64'h800000000000008B,
        64'h8000000000008089, 64'h8000000000008003,
        64'h8000000000008002, 64'h8000000000000080,
        64'h000000000000800A, 64'h800000008000000A,
        64'h8000000080008081, 64'h8000000000008080,
        64'h0000000080000001, 64'h8000000080008008
    };

    // =========================================================================
    // Rho Rotation Offsets [x + 5*y]
    // =========================================================================
    // Number of bits to left-rotate each lane during the rho step.

    localparam int unsigned RHO_OFFSET [0:24] = '{
         0,  1, 62, 28, 27,    // y=0: x=0..4
        36, 44,  6, 55, 20,    // y=1: x=0..4
         3, 10, 43, 25, 39,    // y=2: x=0..4
        41, 45, 15, 21,  8,    // y=3: x=0..4
        18,  2, 61, 56, 14     // y=4: x=0..4
    };

    // =========================================================================
    // FSM States
    // =========================================================================

    typedef enum logic [2:0] {
        S_IDLE    = 3'd0,
        S_ABSORB  = 3'd1,
        S_PERMUTE = 3'd2,
        S_SQUEEZE = 3'd3,
        S_DONE    = 3'd4
    } state_e;

    state_e              fsm_q, fsm_d;
    logic [4:0]          round_cnt_q, round_cnt_d;   // 0..23

    // =========================================================================
    // Keccak State: 25 lanes x 64 bits = 1600 bits
    // =========================================================================
    // Indexed as state[x + 5*y] where x=0..4, y=0..4

    logic [63:0] state_q [0:24];
    logic [63:0] state_d [0:24];

    // =========================================================================
    // Helper: 64-bit left rotate
    // =========================================================================

    function automatic logic [63:0] rotl64(
        input logic [63:0] x,
        input int unsigned n
    );
        if (n == 0)
            return x;
        else
            return (x << n) | (x >> (64 - n));
    endfunction : rotl64

    // =========================================================================
    // Keccak-f Round Function (combinational)
    // =========================================================================
    // Computes one full round: theta -> rho -> pi -> chi -> iota
    // Input: current state + round index
    // Output: next state after one round

    logic [63:0] round_out [0:24];

    always_comb begin
        // Local variables for intermediate step results
        logic [63:0] c [0:4];         // theta: column parities
        logic [63:0] d [0:4];         // theta: column adjustments
        logic [63:0] a_theta [0:24];  // state after theta
        logic [63:0] a_rho   [0:24];  // state after rho
        logic [63:0] a_pi    [0:24];  // state after pi
        logic [63:0] a_chi   [0:24];  // state after chi

        // ---------------------------------------------------------------------
        // Step 1: THETA
        // ---------------------------------------------------------------------
        // C[x] = state[x,0] ^ state[x,1] ^ state[x,2] ^ state[x,3] ^ state[x,4]
        for (int x = 0; x < 5; x++) begin
            c[x] = state_q[x]      ^ state_q[x + 5]  ^ state_q[x + 10]
                  ^ state_q[x + 15] ^ state_q[x + 20];
        end

        // D[x] = C[(x-1) mod 5] ^ rotl64(C[(x+1) mod 5], 1)
        for (int x = 0; x < 5; x++) begin
            d[x] = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
        end

        // Apply theta: state[x,y] ^= D[x]
        for (int x = 0; x < 5; x++) begin
            for (int y = 0; y < 5; y++) begin
                a_theta[x + 5*y] = state_q[x + 5*y] ^ d[x];
            end
        end

        // ---------------------------------------------------------------------
        // Step 2: RHO — rotate each lane by its fixed offset
        // ---------------------------------------------------------------------
        for (int i = 0; i < 25; i++) begin
            a_rho[i] = rotl64(a_theta[i], RHO_OFFSET[i]);
        end

        // ---------------------------------------------------------------------
        // Step 3: PI — rearrange lanes
        // B[y, 2*x + 3*y mod 5] = A_rho[x, y]
        // Equivalently: a_pi[(2*x + 3*y) % 5 + 5*x] ... but the standard
        // formulation is: a_pi[y + 5*((2*x + 3*y) % 5)] = a_rho[x + 5*y]
        // which rearranges to: new[y, (2x+3y) mod 5] = old[x, y]
        // In flat indexing: new[(2*x + 3*y) % 5 + 5*y] ... NO.
        //
        // FIPS 202 pi: A'[y, 2x+3y mod 5] = A[x, y]
        // flat: A'[(2*x + 3*y) % 5  +  5 * y] = ... WRONG
        // Let's be precise:
        //   If A[x,y] is at index x + 5*y, then
        //   A'[y, (2*x+3*y) mod 5] is at index y + 5*((2*x+3*y) mod 5)
        // ---------------------------------------------------------------------
        for (int x = 0; x < 5; x++) begin
            for (int y = 0; y < 5; y++) begin
                a_pi[y + 5 * ((2*x + 3*y) % 5)] = a_rho[x + 5*y];
            end
        end

        // ---------------------------------------------------------------------
        // Step 4: CHI — non-linear mixing
        // A'[x,y] = A[x,y] ^ (~A[(x+1)%5, y] & A[(x+2)%5, y])
        // ---------------------------------------------------------------------
        for (int x = 0; x < 5; x++) begin
            for (int y = 0; y < 5; y++) begin
                a_chi[x + 5*y] = a_pi[x + 5*y]
                    ^ (~a_pi[((x + 1) % 5) + 5*y] & a_pi[((x + 2) % 5) + 5*y]);
            end
        end

        // ---------------------------------------------------------------------
        // Step 5: IOTA — XOR round constant into lane [0,0]
        // ---------------------------------------------------------------------
        for (int i = 0; i < 25; i++) begin
            round_out[i] = a_chi[i];
        end
        round_out[0] = a_chi[0] ^ RC[round_cnt_q];
    end

    // =========================================================================
    // FSM & State Register Update
    // =========================================================================

    always_comb begin
        // Default: hold state
        fsm_d       = fsm_q;
        round_cnt_d = round_cnt_q;
        for (int i = 0; i < 25; i++) begin
            state_d[i] = state_q[i];
        end

        case (fsm_q)
            // -----------------------------------------------------------------
            S_IDLE: begin
                if (start) begin
                    if (init) begin
                        // Zero all lanes before absorb
                        for (int i = 0; i < 25; i++) begin
                            state_d[i] = 64'd0;
                        end
                    end
                    fsm_d = S_ABSORB;
                end
            end

            // -----------------------------------------------------------------
            S_ABSORB: begin
                // XOR the rate block (1088 bits = 17 lanes) into state lanes 0..16
                for (int i = 0; i < 17; i++) begin
                    state_d[i] = state_q[i] ^ msg_block[i*64 +: 64];
                end
                round_cnt_d = 5'd0;
                fsm_d       = S_PERMUTE;
            end

            // -----------------------------------------------------------------
            S_PERMUTE: begin
                // Apply one round of Keccak-f per clock cycle
                for (int i = 0; i < 25; i++) begin
                    state_d[i] = round_out[i];
                end

                if (round_cnt_q == 5'd23) begin
                    // All 24 rounds complete
                    fsm_d = S_SQUEEZE;
                end else begin
                    round_cnt_d = round_cnt_q + 5'd1;
                end
            end

            // -----------------------------------------------------------------
            S_SQUEEZE: begin
                // Output is available; transition to DONE
                fsm_d = S_DONE;
            end

            // -----------------------------------------------------------------
            S_DONE: begin
                // Return to IDLE; done pulse is generated for this cycle
                fsm_d = S_IDLE;
            end

            default: begin
                fsm_d = S_IDLE;
            end
        endcase
    end

    // =========================================================================
    // Sequential Logic
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm_q       <= S_IDLE;
            round_cnt_q <= 5'd0;
            for (int i = 0; i < 25; i++) begin
                state_q[i] <= 64'd0;
            end
        end else begin
            fsm_q       <= fsm_d;
            round_cnt_q <= round_cnt_d;
            for (int i = 0; i < 25; i++) begin
                state_q[i] <= state_d[i];
            end
        end
    end

    // =========================================================================
    // Output Assignments
    // =========================================================================

    // Hash output: lanes 0..3 (256 bits) — available after SQUEEZE
    assign hash_out = {state_q[3], state_q[2], state_q[1], state_q[0]};

    // Done pulse: high for one cycle in S_DONE
    assign done = (fsm_q == S_DONE);

    // Busy: high during ABSORB and PERMUTE phases
    assign busy = (fsm_q == S_ABSORB) || (fsm_q == S_PERMUTE);

endmodule
