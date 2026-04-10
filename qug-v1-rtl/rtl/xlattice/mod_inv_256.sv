// =============================================================================
// mod_inv_256.sv -- 256-bit Modular Inversion via Fermat's Little Theorem
// QUG-V1 Mining SoC -- Xlattice Genus-2 VDF Field Arithmetic
// =============================================================================
// Project  : QUG-V1 RISC-V Mining SoC
// Target   : Xilinx Kintex-7 XC7K325T (FPGA prototype)
// Author   : Quillon Foundation / Dragon Ball Miner
// License  : MIT
// =============================================================================
//
// Computes:  result = op_a^(-1) mod p   (where p = 2^255 - 19)
//
// Method: Fermat's little theorem
//   a^(-1) = a^(p-2) mod p
//
//   p - 2 = 2^255 - 21
//         = 0x7FFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFEB
//
// Algorithm: Right-to-left binary exponentiation (square-and-multiply)
//   For each bit of (p-2), from bit 0 to bit 254:
//     1. If bit is set: result = result * base (mod p)
//     2. base = base * base (mod p)                     [squaring]
//
// Bit pattern of p-2:
//   Bit 0 = 1, bit 1 = 1, bit 2 = 0, bit 3 = 1, bits 4..254 = 1
//   (p-2 = ...1111111_11101011 in binary)
//   Total set bits: 252 out of 255 (bits 2 and 4 are 0)
//   Wait -- let me be precise:
//   p-2 = 2^255 - 21
//   21 = 10101 in binary
//   So p-2 = 111...111_11101011 (255 bits, with bits 2 and 4 cleared)
//
// Cost per modular multiply: 12 cycles (mod_mul_256)
// Total operations: 254 squarings + ~252 multiplications = ~506 mod_muls
// Total cycles: ~506 * 12 = ~6,072 cycles
//
// At 100 MHz: ~60.7 us per inversion
//
// Resource estimate: 1 x mod_mul_256 instance + ~256 FF control logic
// =============================================================================

module mod_inv_256 (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,       // pulse to begin
    input  logic [255:0] op_a,
    input  logic [255:0] modulus,     // p = 2^255 - 19
    output logic [255:0] result,
    output logic         done         // pulse when result valid
);

    // =========================================================================
    // Exponent: p - 2 = 2^255 - 21
    // =========================================================================
    //   p   = 0x7FFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFED
    //   p-2 = 0x7FFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFEB
    //
    // Binary of p-2: bits 254..5 are all 1, then low 5 bits = 01011
    //   21 = 10101_b, so p-2 = 2^255 - 21 has bits {2, 4} clear among the low 5.
    //   Total set bits: 253 out of 255. Hamming weight is high.
    //
    // Cost: 254 squarings + 253 multiplications = 507 modular multiplications.
    //       At 12 cycles each: ~6,084 cycles (~61 us at 100 MHz).

    localparam logic [255:0] P_MINUS_2 =
        256'h7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEB;

    // =========================================================================
    // FSM
    // =========================================================================
    typedef enum logic [2:0] {
        S_IDLE,
        S_CHECK_BIT,    // Check current exponent bit
        S_MUL_START,    // Start result * base multiplication
        S_MUL_WAIT,     // Wait for multiplication to complete
        S_SQR_START,    // Start base * base squaring
        S_SQR_WAIT,     // Wait for squaring to complete
        S_DONE
    } state_t;

    state_t state, state_next;

    // =========================================================================
    // Internal registers
    // =========================================================================
    logic [255:0] base_reg;      // Current base (a^(2^i))
    logic [255:0] accum_reg;     // Accumulated result
    logic [8:0]   bit_idx;       // Current bit index (0..254)
    logic         need_multiply; // Current exponent bit is set

    // =========================================================================
    // mod_mul_256 interface
    // =========================================================================
    logic         mul_start;
    logic [255:0] mul_op_a;
    logic [255:0] mul_op_b;
    logic [255:0] mul_result;
    logic         mul_done;

    mod_mul_256 u_mul (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (mul_start),
        .op_a    (mul_op_a),
        .op_b    (mul_op_b),
        .modulus (modulus),
        .result  (mul_result),
        .done    (mul_done)
    );

    // =========================================================================
    // State machine
    // =========================================================================
    always_comb begin
        state_next = state;
        case (state)
            S_IDLE:      if (start) state_next = S_CHECK_BIT;
            S_CHECK_BIT: begin
                if (bit_idx > 9'd254) begin
                    state_next = S_DONE;
                end else if (need_multiply) begin
                    state_next = S_MUL_START;
                end else begin
                    state_next = S_SQR_START;
                end
            end
            S_MUL_START: state_next = S_MUL_WAIT;
            S_MUL_WAIT:  if (mul_done) state_next = S_SQR_START;
            S_SQR_START: state_next = S_SQR_WAIT;
            S_SQR_WAIT:  if (mul_done) state_next = S_CHECK_BIT;
            S_DONE:      state_next = S_IDLE;
            default:     state_next = S_IDLE;
        endcase
    end

    // =========================================================================
    // Exponent bit lookup
    // =========================================================================
    always_comb begin
        if (bit_idx <= 9'd255) begin
            need_multiply = P_MINUS_2[bit_idx[7:0]];
        end else begin
            need_multiply = 1'b0;
        end
    end

    // =========================================================================
    // Multiplier input muxing
    // =========================================================================
    always_comb begin
        mul_start = 1'b0;
        mul_op_a  = 256'd0;
        mul_op_b  = 256'd0;

        case (state)
            S_MUL_START: begin
                // result = result * base
                mul_start = 1'b1;
                mul_op_a  = accum_reg;
                mul_op_b  = base_reg;
            end
            S_SQR_START: begin
                // base = base * base (squaring)
                mul_start = 1'b1;
                mul_op_a  = base_reg;
                mul_op_b  = base_reg;
            end
            default: ;
        endcase
    end

    // =========================================================================
    // Datapath
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            base_reg  <= 256'd0;
            accum_reg <= 256'd0;
            bit_idx   <= 9'd0;
            result    <= 256'd0;
            done      <= 1'b0;
        end else begin
            state <= state_next;
            done  <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        base_reg  <= op_a;       // base = a
                        accum_reg <= 256'd1;     // result = 1
                        bit_idx   <= 9'd0;       // start from LSB
                    end
                end

                S_CHECK_BIT: begin
                    // Bit index check and branch handled by FSM
                    // If bit is 0, skip multiply and go straight to square
                end

                S_MUL_WAIT: begin
                    if (mul_done) begin
                        accum_reg <= mul_result;  // result = result * base
                    end
                end

                S_SQR_WAIT: begin
                    if (mul_done) begin
                        base_reg <= mul_result;   // base = base^2
                        bit_idx  <= bit_idx + 9'd1;
                    end
                end

                S_DONE: begin
                    result <= accum_reg;
                    done   <= 1'b1;
                end

                default: ;
            endcase
        end
    end

endmodule
