// =============================================================================
// mod_sub_256.sv -- 256-bit Modular Subtractor (F_p, p = 2^255 - 19)
// QUG-V1 Mining SoC -- Xlattice Genus-2 VDF Field Arithmetic
// =============================================================================
// Project  : QUG-V1 RISC-V Mining SoC
// Target   : Xilinx Kintex-7 XC7K325T (FPGA prototype)
// Author   : Quillon Foundation / Dragon Ball Miner
// License  : MIT
// =============================================================================
//
// Computes:  result = (op_a - op_b) mod modulus
//
// Single-cycle combinational subtract with conditional addition of modulus.
// Output is registered for timing closure at 100 MHz.
//
// If op_a >= op_b:  result = op_a - op_b
// If op_a <  op_b:  result = op_a - op_b + modulus  (wrap into [0, p))
//
// Critical path: 256-bit subtract + 257-bit add + mux + register
// At 100 MHz on Kintex-7, this comfortably meets timing (carry chains are fast).
//
// Resource estimate: ~512 LUT6 + 256 FF (pure fabric, no DSP needed)
// =============================================================================

module mod_sub_256 (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,       // pulse to begin (result valid next cycle)
    input  logic [255:0] op_a,
    input  logic [255:0] op_b,
    input  logic [255:0] modulus,     // p = 2^255 - 19
    output logic [255:0] result,
    output logic         done         // pulse when result valid
);

    // =========================================================================
    // Combinational subtraction with modular correction
    // =========================================================================
    logic [256:0] diff;           // 257-bit to capture borrow
    logic [256:0] diff_plus_p;    // diff + modulus
    logic [255:0] result_comb;

    always_comb begin
        diff       = {1'b0, op_a} - {1'b0, op_b};
        diff_plus_p = diff + {1'b0, modulus};

        // If op_a >= op_b, diff is non-negative: use diff directly
        // diff[256] is the borrow bit: 1 means op_a < op_b (underflow)
        if (diff[256] == 1'b0) begin
            result_comb = diff[255:0];
        end else begin
            result_comb = diff_plus_p[255:0];
        end
    end

    // =========================================================================
    // Registered output
    // =========================================================================
    logic done_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 256'd0;
            done_r <= 1'b0;
        end else begin
            done_r <= start;
            if (start) begin
                result <= result_comb;
            end
        end
    end

    assign done = done_r;

endmodule
