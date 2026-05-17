// =============================================================================
// QUG-V1 Mining SoC - Arithmetic Logic Unit
// =============================================================================
// Pure combinational ALU for RV32IM.
// Operations: ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT, SLTU,
//             MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU
//
// M-extension multiply produces a 64-bit result split across two cycles
// (EX1 computes, EX2 captures high word if needed).
//
// Target: Kintex-7 @ 100 MHz (DSP48E1 infers for multiply)
// =============================================================================

module qug_alu
  import qug_core_pkg::*;
#(
  parameter int XLEN = 32
)(
  input  logic [XLEN-1:0] operand_a,
  input  logic [XLEN-1:0] operand_b,
  input  alu_op_e          op,

  output logic [XLEN-1:0] result,
  output logic             zero,       // result == 0
  output logic             carry,      // unsigned overflow
  output logic             overflow    // signed overflow
);

  // -------------------------------------------------------------------------
  // Internal wires
  // -------------------------------------------------------------------------
  logic [XLEN:0]   add_result;   // 33-bit for carry detection
  logic [XLEN:0]   sub_result;
  logic [2*XLEN-1:0] mul_result_ss; // signed x signed
  logic [2*XLEN-1:0] mul_result_uu; // unsigned x unsigned
  logic signed [XLEN-1:0] a_signed;
  logic signed [XLEN-1:0] b_signed;
  logic [XLEN-1:0] div_result;
  logic [XLEN-1:0] rem_result;

  assign a_signed = signed'(operand_a);
  assign b_signed = signed'(operand_b);

  // -------------------------------------------------------------------------
  // Adder / subtractor (shared logic)
  // -------------------------------------------------------------------------
  assign add_result = {1'b0, operand_a} + {1'b0, operand_b};
  assign sub_result = {1'b0, operand_a} - {1'b0, operand_b};

  // -------------------------------------------------------------------------
  // Multiplier (infers DSP48E1 on Kintex-7)
  // -------------------------------------------------------------------------
  assign mul_result_ss = {{XLEN{operand_a[XLEN-1]}}, operand_a} *
                         {{XLEN{operand_b[XLEN-1]}}, operand_b};
  assign mul_result_uu = {32'b0, operand_a} * {32'b0, operand_b};

  // -------------------------------------------------------------------------
  // Divider (combinational -- acceptable at 100 MHz for 32-bit on Kintex-7
  // with multi-cycle execution via EX1+EX2 stages)
  // -------------------------------------------------------------------------
  always_comb begin
    if (operand_b == '0) begin
      div_result = '1;               // div by zero -> all-ones per RISC-V spec
      rem_result = operand_a;        // rem by zero -> dividend
    end else begin
      div_result = $unsigned(a_signed / b_signed);
      rem_result = $unsigned(a_signed % b_signed);
    end
  end

  // -------------------------------------------------------------------------
  // Result mux
  // -------------------------------------------------------------------------
  always_comb begin
    result   = '0;
    carry    = 1'b0;
    overflow = 1'b0;

    unique case (op)
      ALU_ADD: begin
        result   = add_result[XLEN-1:0];
        carry    = add_result[XLEN];
        overflow = (operand_a[XLEN-1] == operand_b[XLEN-1]) &&
                   (result[XLEN-1]    != operand_a[XLEN-1]);
      end

      ALU_SUB: begin
        result   = sub_result[XLEN-1:0];
        carry    = sub_result[XLEN];
        overflow = (operand_a[XLEN-1] != operand_b[XLEN-1]) &&
                   (result[XLEN-1]    != operand_a[XLEN-1]);
      end

      ALU_AND:  result = operand_a & operand_b;
      ALU_OR:   result = operand_a | operand_b;
      ALU_XOR:  result = operand_a ^ operand_b;

      ALU_SLL:  result = operand_a << operand_b[4:0];
      ALU_SRL:  result = operand_a >> operand_b[4:0];
      ALU_SRA:  result = $unsigned(a_signed >>> operand_b[4:0]);

      ALU_SLT:  result = {{(XLEN-1){1'b0}}, (a_signed < b_signed)};
      ALU_SLTU: result = {{(XLEN-1){1'b0}}, (operand_a < operand_b)};

      ALU_MUL:  result = mul_result_ss[XLEN-1:0];       // lower 32 bits
      ALU_MULH: result = mul_result_ss[2*XLEN-1:XLEN];  // upper 32 bits (signed)

      ALU_DIV:  result = div_result;
      ALU_REM:  result = rem_result;

      ALU_PASS: result = operand_a;

      default:  result = '0;
    endcase
  end

  // -------------------------------------------------------------------------
  // Zero flag
  // -------------------------------------------------------------------------
  assign zero = (result == '0);

endmodule : qug_alu
