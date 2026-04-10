// =============================================================================
// mod_mul_256.sv -- 256-bit Modular Multiplier (F_p, p = 2^255 - 19)
// QUG-V1 Mining SoC -- Xlattice Genus-2 VDF Field Arithmetic
// =============================================================================
// Project  : QUG-V1 RISC-V Mining SoC
// Target   : Xilinx Kintex-7 XC7K325T (FPGA prototype)
// Author   : Quillon Foundation / Dragon Ball Miner
// License  : MIT
// =============================================================================
//
// Architecture: Digit-serial schoolbook multiplication with 8 parallel DSP48E1
//
// Operand decomposition: 256-bit = 8 x 32-bit digits (little-endian)
//   A = a[0] + a[1]*2^32 + a[2]*2^64 + ... + a[7]*2^224
//   B = b[0] + b[1]*2^32 + b[2]*2^64 + ... + b[7]*2^224
//
// Algorithm:
//   Phase 1 (8 cycles): Raw 512-bit product via digit-serial multiply
//     For each digit b[j] of B (j = 0..7):
//       Multiply all 8 digits of A by b[j] using 8 parallel DSP48E1 slices
//       Accumulate into partial product register with carry propagation
//
//   Phase 2 (~4 cycles): Barrett reduction mod p
//     Since p = 2^255 - 19 has special form, reduction is efficient:
//     For a 512-bit product P = P_hi * 2^256 + P_lo:
//       result = P_lo + P_hi * 38 (mod p)   [since 2^256 = 38 mod p]
//     The multiply-by-38 uses shift+add, then conditional subtraction.
//
// Total latency: 14 cycles per modular multiplication
// Throughput: 1 result / 14 cycles (no pipelining -- sequential VDF chain)
//
// Resource estimate: 8 DSP48E1 (or 16 if Vivado splits 32x32) + ~2K LUT + ~1K FF
// =============================================================================

module mod_mul_256 (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,       // pulse to begin
    input  logic [255:0] op_a,
    input  logic [255:0] op_b,
    input  logic [255:0] modulus,     // p = 2^255 - 19
    output logic [255:0] result,
    output logic         done         // pulse when result valid
);

    // =========================================================================
    // Digit decomposition
    // =========================================================================
    localparam int unsigned N  = 8;           // Number of digits
    localparam int unsigned DW = 32;          // Digit width
    localparam int unsigned PN = 2 * N;       // Product digit count (16)

    // =========================================================================
    // FSM
    // =========================================================================
    // Pipeline: LOAD -> MUL0..MUL7 -> ACCUM_LAST -> CARRY1 -> CARRY2 -> REDUCE1 -> REDUCE2 -> COND_SUB -> DONE
    // MUL states overlap: DSP launches in cycle i, result accumulated in cycle i+1.

    typedef enum logic [3:0] {
        S_IDLE,
        S_LOAD,         // Latch operands into digit arrays
        S_MUL,          // 9 cycles: launch DSP (0..7) + accumulate last (8th)
        S_CARRY1,       // Carry propagation pass 1
        S_CARRY2,       // Carry propagation pass 2
        S_REDUCE,       // First fold: t = P_lo + P_hi * 38 (up to ~262 bits)
        S_REDUCE2,      // Second fold: t2 = t_lo + t_hi * 38 (now < 2^256 + small)
        S_COND_SUB,     // Conditional subtract of p (up to 2x)
        S_DONE
    } state_t;

    state_t state, state_next;

    // =========================================================================
    // Operand digit registers
    // =========================================================================
    logic [DW-1:0] a_dig [N];
    logic [DW-1:0] b_dig [N];

    // =========================================================================
    // Multiply phase control
    // =========================================================================
    logic [3:0] mul_step;   // 0..8: step 0 launches first DSP, step 8 accumulates last

    // =========================================================================
    // DSP48E1-inferred 32x32 -> 64 multipliers (8 parallel)
    // =========================================================================
    // Xilinx synthesis attribute on the product registers for DSP inference.
    (* use_dsp = "yes" *) logic [63:0] dsp_out [N];

    // Current B digit fed to all 8 DSPs
    logic [DW-1:0] b_cur;

    // DSP input mux: during S_MUL, pick b_dig[mul_step] (or 0 if past end)
    always_comb begin
        if (state == S_LOAD) begin
            b_cur = op_b[DW-1:0];  // Pre-load first digit
        end else if (state == S_MUL && mul_step < N[3:0]) begin
            b_cur = b_dig[mul_step];
        end else begin
            b_cur = {DW{1'b0}};
        end
    end

    // Registered DSP multiplies: a_dig[i] * b_cur, 1-cycle latency
    generate
        genvar gi;
        for (gi = 0; gi < N; gi++) begin : gen_dsp
            always_ff @(posedge clk) begin
                dsp_out[gi] <= {32'd0, a_dig[gi]} * {32'd0, b_cur};
            end
        end
    endgenerate

    // =========================================================================
    // Product accumulator: 16 x 64-bit digits
    // =========================================================================
    // Each digit can temporarily hold values > 32 bits during accumulation.
    // Final carry propagation normalizes all digits to 32 bits.
    logic [63:0] prod [PN];

    // Track which B-digit position the current DSP outputs correspond to
    // (1 cycle behind mul_step due to DSP register)
    logic [3:0] accum_pos;

    // =========================================================================
    // Reduction registers
    // =========================================================================
    logic [263:0] reduce_sum;   // First fold result (up to ~262 bits)
    logic [263:0] reduce_sum2;  // Second fold result (< 2^256 + small)

    // =========================================================================
    // FSM next-state
    // =========================================================================
    always_comb begin
        state_next = state;
        case (state)
            S_IDLE:     if (start) state_next = S_LOAD;
            S_LOAD:     state_next = S_MUL;
            S_MUL:      if (mul_step == 4'd8) state_next = S_CARRY1;
            S_CARRY1:   state_next = S_CARRY2;
            S_CARRY2:   state_next = S_REDUCE;
            S_REDUCE:   state_next = S_REDUCE2;
            S_REDUCE2:  state_next = S_COND_SUB;
            S_COND_SUB: state_next = S_DONE;
            S_DONE:     state_next = S_IDLE;
            default:    state_next = S_IDLE;
        endcase
    end

    // =========================================================================
    // Carry propagation: normalize 64-bit digits to 32-bit
    // =========================================================================
    // Combinational carry chain across all 16 product digits.
    // Two passes (S_CARRY1, S_CARRY2) ensure full propagation even
    // when a carry into digit i causes digit i to overflow 32 bits.
    logic [63:0] carry_out [PN];

    always_comb begin
        carry_out[0] = prod[0];
        for (int i = 1; i < PN; i++) begin
            carry_out[i] = prod[i] + {32'd0, carry_out[i-1][63:32]};
        end
    end

    // =========================================================================
    // Extract 256-bit halves from normalized product
    // =========================================================================
    logic [255:0] p_lo, p_hi;

    always_comb begin
        for (int i = 0; i < N; i++) begin
            p_lo[i*32 +: 32] = prod[i][31:0];
            p_hi[i*32 +: 32] = prod[i + N][31:0];
        end
    end

    // =========================================================================
    // Main datapath
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            mul_step  <= 4'd0;
            accum_pos <= 4'd0;
            done      <= 1'b0;
            result    <= 256'd0;
            reduce_sum  <= 264'd0;
            reduce_sum2 <= 264'd0;
            for (int i = 0; i < N; i++) begin
                a_dig[i] <= 32'd0;
                b_dig[i] <= 32'd0;
            end
            for (int i = 0; i < PN; i++) begin
                prod[i] <= 64'd0;
            end
        end else begin
            state <= state_next;
            done  <= 1'b0;

            case (state)
                // ---------------------------------------------------------
                // IDLE: Wait for start
                // ---------------------------------------------------------
                S_IDLE: begin
                    if (start) begin
                        mul_step  <= 4'd0;
                        accum_pos <= 4'd0;
                    end
                end

                // ---------------------------------------------------------
                // LOAD: Decompose operands into digit arrays, clear product
                // ---------------------------------------------------------
                S_LOAD: begin
                    for (int i = 0; i < N; i++) begin
                        a_dig[i] <= op_a[i*32 +: 32];
                        b_dig[i] <= op_b[i*32 +: 32];
                    end
                    for (int i = 0; i < PN; i++) begin
                        prod[i] <= 64'd0;
                    end
                    mul_step  <= 4'd0;
                    accum_pos <= 4'd0;
                end

                // ---------------------------------------------------------
                // MUL: 9 sub-steps (0..8)
                //   Step 0: DSPs compute a[i]*b[0] (output valid next cycle)
                //   Step 1: Accumulate a[i]*b[0] into prod[i+0]; DSPs compute a[i]*b[1]
                //   ...
                //   Step 7: Accumulate a[i]*b[6] into prod[i+6]; DSPs compute a[i]*b[7]
                //   Step 8: Accumulate a[i]*b[7] into prod[i+7]; done
                // ---------------------------------------------------------
                S_MUL: begin
                    // Accumulate DSP results from previous step (valid when mul_step >= 1)
                    if (mul_step >= 4'd1) begin
                        for (int i = 0; i < N; i++) begin
                            prod[i + accum_pos] <= prod[i + accum_pos] + dsp_out[i];
                        end
                    end

                    // Advance
                    accum_pos <= mul_step;  // DSP outputs at step S correspond to b[S-1]
                    mul_step  <= mul_step + 4'd1;
                end

                // ---------------------------------------------------------
                // CARRY1: Ripple carry normalization pass 1
                // ---------------------------------------------------------
                S_CARRY1: begin
                    for (int i = 0; i < PN; i++) begin
                        prod[i] <= {32'd0, carry_out[i][31:0]};
                    end
                end

                // ---------------------------------------------------------
                // CARRY2: Second carry pass (resolves cascaded carries)
                // ---------------------------------------------------------
                S_CARRY2: begin
                    for (int i = 0; i < PN; i++) begin
                        prod[i] <= {32'd0, carry_out[i][31:0]};
                    end
                end

                // ---------------------------------------------------------
                // REDUCE (first fold): t = P_lo + P_hi * 38
                //
                // For p = 2^255 - 19:
                //   2^256 mod p = 2^256 - (2^255 - 19)*2 = 2^256 - 2^256 + 38 = 38
                //   So P_hi * 2^256 + P_lo  =  P_lo + P_hi * 38 (mod p)
                //
                // P_hi is up to 256 bits, so P_hi * 38 can be up to ~262 bits.
                // reduce_sum (t) can be up to ~262 bits -- too large for
                // two conditional subtractions of p (~255 bits) to suffice.
                //
                // 38 = 32 + 4 + 2 (shift-and-add, no DSP needed)
                // ---------------------------------------------------------
                S_REDUCE: begin
                    reduce_sum <= {8'd0, p_lo}
                               + ({8'd0, p_hi} << 5)   // * 32
                               + ({8'd0, p_hi} << 2)   // * 4
                               + ({8'd0, p_hi} << 1);  // * 2
                end

                // ---------------------------------------------------------
                // REDUCE2 (second fold): t2 = t_lo[255:0] + t_hi[263:256] * 38
                //
                // After the first fold, reduce_sum is up to ~262 bits.
                // Split into t_lo (low 256 bits) and t_hi (high bits 263:256,
                // at most ~6 bits wide, value <= 39).
                // t2 = t_lo + t_hi * 38.  Since t_hi <= 39, t_hi * 38 <= 1482
                // so t2 < 2^256 + 1482. This is within range for two
                // conditional subtractions of p to bring into [0, p).
                // ---------------------------------------------------------
                S_REDUCE2: begin
                    reduce_sum2 <= {8'd0, reduce_sum[255:0]}
                                 + ({256'd0, reduce_sum[263:256]} << 5)   // * 32
                                 + ({256'd0, reduce_sum[263:256]} << 2)   // * 4
                                 + ({256'd0, reduce_sum[263:256]} << 1);  // * 2
                end

                // ---------------------------------------------------------
                // COND_SUB: Conditional subtract of p
                //
                // After the second fold, reduce_sum2 < 2^256 + ~1482.
                // At most 2 subtractions of p suffice to bring into [0, p).
                // ---------------------------------------------------------
                S_COND_SUB: begin
                    if (reduce_sum2 >= ({9'd0, modulus} + {9'd0, modulus})) begin
                        // >= 2p: subtract 2p
                        result <= reduce_sum2[255:0] - modulus - modulus;
                    end else if (reduce_sum2 >= {8'd0, modulus}) begin
                        // >= p: subtract p
                        result <= reduce_sum2[255:0] - modulus;
                    end else begin
                        result <= reduce_sum2[255:0];
                    end
                end

                // ---------------------------------------------------------
                // DONE: Signal completion
                // ---------------------------------------------------------
                S_DONE: begin
                    done <= 1'b1;
                end

                default: ;
            endcase
        end
    end

endmodule
