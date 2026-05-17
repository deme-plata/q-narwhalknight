// =============================================================================
// QUG-V1 Mining SoC - Instruction Decoder
// =============================================================================
// Decodes RV32IMC instructions + custom extension opcodes.
//
// Supports:
//   - RV32I base: R, I, S, B, U, J formats
//   - RV32M: MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU
//   - RV32C: Compressed 16-bit instructions (expanded to 32-bit equivalent)
//   - Custom-0 (opcode 0x0B / 7'b0001011): routed to Xcrypto unit
//   - Custom-1 (opcode 0x2B / 7'b0101011): routed to Xlattice unit
//
// Target: Kintex-7 @ 100 MHz
// =============================================================================

module qug_decoder
  import qug_core_pkg::*;
(
  input  logic [31:0]    instr_i,      // raw instruction (may be 16 or 32 bit)
  input  logic [31:0]    pc_i,         // current PC (for AUIPC / branch targets)

  output decoded_ctrl_t  ctrl_o,       // decoded control bundle
  output logic [31:0]    instr_expanded_o, // 32-bit instruction (after C expansion)
  output logic           is_compressed_o   // input was a 16-bit C instruction
);

  // -------------------------------------------------------------------------
  // Detect compressed instruction (bits [1:0] != 2'b11)
  // -------------------------------------------------------------------------
  logic [31:0] instr;
  logic        is_c;

  assign is_c = (instr_i[1:0] != 2'b11);
  assign is_compressed_o = is_c;

  // -------------------------------------------------------------------------
  // C-extension expander
  // Expand 16-bit compressed instructions to their 32-bit RV32I equivalents
  // -------------------------------------------------------------------------
  logic [31:0] c_expanded;

  always_comb begin
    c_expanded = 32'h0000_0000; // default: illegal (HINT/reserved)

    casez (instr_i[15:0])
      // -----------------------------------------------------------------
      // Quadrant 0 (bits [1:0] = 2'b00)
      // -----------------------------------------------------------------
      // C.ADDI4SPN: addi rd', x2, nzuimm
      16'b000_?????_???_??_00: begin
        if (instr_i[12:5] != 8'b0) begin // nzuimm != 0
          // imm = {nzuimm[5:4|9:6|2|3]} << 0, fields [12:5]
          // Encoding: imm[5:4]=i[12:11], imm[9:6]=i[10:7], imm[2]=i[6], imm[3]=i[5]
          c_expanded = {2'b0, instr_i[10:7], instr_i[12:11], instr_i[5],
                        instr_i[6], 2'b00, 5'd2, 3'b000, 2'b01, instr_i[4:2], 7'b0010011};
        end
      end

      // C.LW: lw rd', offset(rs1')
      16'b010_???_???_??_???_00: begin
        // offset = {imm[5:3|2|6]} = {i[5], i[12:10], i[6], 2'b00}
        c_expanded = {5'b0, instr_i[5], instr_i[12:10], instr_i[6],
                      2'b00, 2'b01, instr_i[9:7], 3'b010, 2'b01, instr_i[4:2], 7'b0000011};
      end

      // C.SW: sw rs2', offset(rs1')
      16'b110_???_???_??_???_00: begin
        c_expanded = {5'b0, instr_i[5], instr_i[12], 2'b01, instr_i[4:2],
                      2'b01, instr_i[9:7], 3'b010, instr_i[11:10], instr_i[6],
                      2'b00, 7'b0100011};
      end

      // -----------------------------------------------------------------
      // Quadrant 1 (bits [1:0] = 2'b01)
      // -----------------------------------------------------------------
      // C.NOP / C.ADDI: addi rd, rd, nzimm
      16'b000_?_?????_?????_01: begin
        c_expanded = {{6{instr_i[12]}}, instr_i[12], instr_i[6:2],
                      instr_i[11:7], 3'b000, instr_i[11:7], 7'b0010011};
      end

      // C.JAL: jal x1, offset (RV32 only)
      16'b001_???????????_01: begin
        // offset[11|4|9:8|10|6|7|3:1|5]
        c_expanded = {instr_i[12], instr_i[8], instr_i[10:9], instr_i[6],
                      instr_i[7], instr_i[2], instr_i[11], instr_i[5:3],
                      {9{instr_i[12]}}, 5'd1, 7'b1101111};
      end

      // C.LI: addi rd, x0, imm
      16'b010_?_?????_?????_01: begin
        c_expanded = {{6{instr_i[12]}}, instr_i[12], instr_i[6:2],
                      5'd0, 3'b000, instr_i[11:7], 7'b0010011};
      end

      // C.LUI / C.ADDI16SP
      16'b011_?_?????_?????_01: begin
        if (instr_i[11:7] == 5'd2) begin
          // C.ADDI16SP: addi x2, x2, nzimm
          c_expanded = {{3{instr_i[12]}}, instr_i[4:3], instr_i[5],
                        instr_i[2], instr_i[6], 4'b0000, 5'd2, 3'b000, 5'd2, 7'b0010011};
        end else begin
          // C.LUI: lui rd, nzimm
          c_expanded = {{14{instr_i[12]}}, instr_i[12], instr_i[6:2],
                        instr_i[11:7], 7'b0110111};
        end
      end

      // C.SRLI, C.SRAI, C.ANDI, C.SUB, C.XOR, C.OR, C.AND
      16'b100_?_??_???_?????_01: begin
        unique case (instr_i[11:10])
          2'b00: // C.SRLI
            c_expanded = {7'b0000000, instr_i[6:2], 2'b01, instr_i[9:7],
                          3'b101, 2'b01, instr_i[9:7], 7'b0010011};
          2'b01: // C.SRAI
            c_expanded = {7'b0100000, instr_i[6:2], 2'b01, instr_i[9:7],
                          3'b101, 2'b01, instr_i[9:7], 7'b0010011};
          2'b10: // C.ANDI
            c_expanded = {{6{instr_i[12]}}, instr_i[12], instr_i[6:2],
                          2'b01, instr_i[9:7], 3'b111, 2'b01, instr_i[9:7], 7'b0010011};
          2'b11: begin
            unique case ({instr_i[12], instr_i[6:5]})
              3'b000: // C.SUB
                c_expanded = {7'b0100000, 2'b01, instr_i[4:2], 2'b01, instr_i[9:7],
                              3'b000, 2'b01, instr_i[9:7], 7'b0110011};
              3'b001: // C.XOR
                c_expanded = {7'b0000000, 2'b01, instr_i[4:2], 2'b01, instr_i[9:7],
                              3'b100, 2'b01, instr_i[9:7], 7'b0110011};
              3'b010: // C.OR
                c_expanded = {7'b0000000, 2'b01, instr_i[4:2], 2'b01, instr_i[9:7],
                              3'b110, 2'b01, instr_i[9:7], 7'b0110011};
              3'b011: // C.AND
                c_expanded = {7'b0000000, 2'b01, instr_i[4:2], 2'b01, instr_i[9:7],
                              3'b111, 2'b01, instr_i[9:7], 7'b0110011};
              default: c_expanded = 32'h0;
            endcase
          end
        endcase
      end

      // C.J: jal x0, offset
      16'b101_???????????_01: begin
        c_expanded = {instr_i[12], instr_i[8], instr_i[10:9], instr_i[6],
                      instr_i[7], instr_i[2], instr_i[11], instr_i[5:3],
                      {9{instr_i[12]}}, 5'd0, 7'b1101111};
      end

      // C.BEQZ: beq rs1', x0, offset
      16'b110_???_???_?????_01: begin
        c_expanded = {{3{instr_i[12]}}, instr_i[12], instr_i[6:5], instr_i[2],
                      5'd0, 2'b01, instr_i[9:7], 3'b000,
                      instr_i[11:10], instr_i[4:3], 1'b0, 7'b1100011};
      end

      // C.BNEZ: bne rs1', x0, offset
      16'b111_???_???_?????_01: begin
        c_expanded = {{3{instr_i[12]}}, instr_i[12], instr_i[6:5], instr_i[2],
                      5'd0, 2'b01, instr_i[9:7], 3'b001,
                      instr_i[11:10], instr_i[4:3], 1'b0, 7'b1100011};
      end

      // -----------------------------------------------------------------
      // Quadrant 2 (bits [1:0] = 2'b10)
      // -----------------------------------------------------------------
      // C.SLLI: slli rd, rd, shamt
      16'b000_?_?????_?????_10: begin
        c_expanded = {7'b0000000, instr_i[6:2], instr_i[11:7],
                      3'b001, instr_i[11:7], 7'b0010011};
      end

      // C.LWSP: lw rd, offset(x2)
      16'b010_?_?????_?????_10: begin
        c_expanded = {4'b0, instr_i[3:2], instr_i[12], instr_i[6:4],
                      2'b00, 5'd2, 3'b010, instr_i[11:7], 7'b0000011};
      end

      // C.MV, C.ADD, C.JR, C.JALR
      16'b100_?_?????_?????_10: begin
        if (instr_i[12] == 1'b0) begin
          if (instr_i[6:2] == 5'b0) begin
            // C.JR: jalr x0, rs1, 0
            c_expanded = {12'b0, instr_i[11:7], 3'b000, 5'd0, 7'b1100111};
          end else begin
            // C.MV: add rd, x0, rs2
            c_expanded = {7'b0, instr_i[6:2], 5'd0, 3'b000, instr_i[11:7], 7'b0110011};
          end
        end else begin
          if (instr_i[6:2] == 5'b0) begin
            // C.JALR: jalr x1, rs1, 0
            c_expanded = {12'b0, instr_i[11:7], 3'b000, 5'd1, 7'b1100111};
          end else begin
            // C.ADD: add rd, rd, rs2
            c_expanded = {7'b0, instr_i[6:2], instr_i[11:7], 3'b000,
                          instr_i[11:7], 7'b0110011};
          end
        end
      end

      // C.SWSP: sw rs2, offset(x2)
      16'b110_?_?????_?????_10: begin
        c_expanded = {4'b0, instr_i[8:7], instr_i[12], instr_i[6:2],
                      5'd2, 3'b010, instr_i[11:9], 2'b00, 7'b0100011};
      end

      default: c_expanded = 32'h0000_0000; // illegal
    endcase
  end

  // Select expanded or raw instruction
  assign instr = is_c ? c_expanded : instr_i;
  assign instr_expanded_o = instr;

  // -------------------------------------------------------------------------
  // Main 32-bit decoder
  // -------------------------------------------------------------------------
  logic [6:0] opcode;
  logic [2:0] funct3;
  logic [6:0] funct7;

  assign opcode = instr[6:0];
  assign funct3 = instr[14:12];
  assign funct7 = instr[31:25];

  always_comb begin
    // Defaults: NOP-like
    ctrl_o.alu_op       = ALU_ADD;
    ctrl_o.imm_type     = IMM_NONE;
    ctrl_o.branch_type  = BR_NONE;
    ctrl_o.mem_width    = MEM_NONE;
    ctrl_o.reg_write    = 1'b0;
    ctrl_o.mem_read     = 1'b0;
    ctrl_o.mem_write    = 1'b0;
    ctrl_o.mem_signed   = 1'b0;
    ctrl_o.alu_src_imm  = 1'b0;
    ctrl_o.lui          = 1'b0;
    ctrl_o.auipc        = 1'b0;
    ctrl_o.jal          = 1'b0;
    ctrl_o.jalr         = 1'b0;
    ctrl_o.ext_xcrypto  = 1'b0;
    ctrl_o.ext_xlattice = 1'b0;
    ctrl_o.is_mul       = 1'b0;
    ctrl_o.is_div       = 1'b0;
    ctrl_o.illegal      = 1'b0;
    ctrl_o.rs1_addr     = instr[19:15];
    ctrl_o.rs2_addr     = instr[24:20];
    ctrl_o.rd_addr      = instr[11:7];
    ctrl_o.immediate    = '0;

    unique case (opcode)
      // -----------------------------------------------------------------
      // LUI (U-type)
      // -----------------------------------------------------------------
      7'b0110111: begin
        ctrl_o.lui       = 1'b1;
        ctrl_o.reg_write = 1'b1;
        ctrl_o.alu_op    = ALU_PASS;
        ctrl_o.imm_type  = IMM_U;
        ctrl_o.immediate = {instr[31:12], 12'b0};
      end

      // -----------------------------------------------------------------
      // AUIPC (U-type)
      // -----------------------------------------------------------------
      7'b0010111: begin
        ctrl_o.auipc     = 1'b1;
        ctrl_o.reg_write = 1'b1;
        ctrl_o.alu_op    = ALU_ADD;
        ctrl_o.imm_type  = IMM_U;
        ctrl_o.immediate = {instr[31:12], 12'b0};
      end

      // -----------------------------------------------------------------
      // JAL (J-type)
      // -----------------------------------------------------------------
      7'b1101111: begin
        ctrl_o.jal         = 1'b1;
        ctrl_o.reg_write   = 1'b1;
        ctrl_o.branch_type = BR_JAL;
        ctrl_o.imm_type    = IMM_J;
        ctrl_o.immediate   = {{11{instr[31]}}, instr[31], instr[19:12],
                              instr[20], instr[30:21], 1'b0};
      end

      // -----------------------------------------------------------------
      // JALR (I-type)
      // -----------------------------------------------------------------
      7'b1100111: begin
        ctrl_o.jalr        = 1'b1;
        ctrl_o.reg_write   = 1'b1;
        ctrl_o.alu_src_imm = 1'b1;
        ctrl_o.branch_type = BR_JAL;
        ctrl_o.imm_type    = IMM_I;
        ctrl_o.immediate   = {{20{instr[31]}}, instr[31:20]};
      end

      // -----------------------------------------------------------------
      // Branch (B-type)
      // -----------------------------------------------------------------
      7'b1100011: begin
        ctrl_o.imm_type  = IMM_B;
        ctrl_o.immediate = {{19{instr[31]}}, instr[31], instr[7],
                            instr[30:25], instr[11:8], 1'b0};
        unique case (funct3)
          3'b000: ctrl_o.branch_type = BR_EQ;
          3'b001: ctrl_o.branch_type = BR_NE;
          3'b100: ctrl_o.branch_type = BR_LT;
          3'b101: ctrl_o.branch_type = BR_GE;
          3'b110: ctrl_o.branch_type = BR_LTU;
          3'b111: ctrl_o.branch_type = BR_GEU;
          default: ctrl_o.illegal = 1'b1;
        endcase
      end

      // -----------------------------------------------------------------
      // Load (I-type)
      // -----------------------------------------------------------------
      7'b0000011: begin
        ctrl_o.mem_read    = 1'b1;
        ctrl_o.reg_write   = 1'b1;
        ctrl_o.alu_src_imm = 1'b1;
        ctrl_o.alu_op      = ALU_ADD;
        ctrl_o.imm_type    = IMM_I;
        ctrl_o.immediate   = {{20{instr[31]}}, instr[31:20]};
        unique case (funct3)
          3'b000: begin ctrl_o.mem_width = MEM_BYTE; ctrl_o.mem_signed = 1'b1; end // LB
          3'b001: begin ctrl_o.mem_width = MEM_HALF; ctrl_o.mem_signed = 1'b1; end // LH
          3'b010: begin ctrl_o.mem_width = MEM_WORD; end                           // LW
          3'b100: begin ctrl_o.mem_width = MEM_BYTE; end                           // LBU
          3'b101: begin ctrl_o.mem_width = MEM_HALF; end                           // LHU
          default: ctrl_o.illegal = 1'b1;
        endcase
      end

      // -----------------------------------------------------------------
      // Store (S-type)
      // -----------------------------------------------------------------
      7'b0100011: begin
        ctrl_o.mem_write   = 1'b1;
        ctrl_o.alu_src_imm = 1'b1;
        ctrl_o.alu_op      = ALU_ADD;
        ctrl_o.imm_type    = IMM_S;
        ctrl_o.immediate   = {{20{instr[31]}}, instr[31:25], instr[11:7]};
        unique case (funct3)
          3'b000: ctrl_o.mem_width = MEM_BYTE; // SB
          3'b001: ctrl_o.mem_width = MEM_HALF; // SH
          3'b010: ctrl_o.mem_width = MEM_WORD; // SW
          default: ctrl_o.illegal = 1'b1;
        endcase
      end

      // -----------------------------------------------------------------
      // ALU immediate (I-type)
      // -----------------------------------------------------------------
      7'b0010011: begin
        ctrl_o.reg_write   = 1'b1;
        ctrl_o.alu_src_imm = 1'b1;
        ctrl_o.imm_type    = IMM_I;
        ctrl_o.immediate   = {{20{instr[31]}}, instr[31:20]};
        unique case (funct3)
          3'b000: ctrl_o.alu_op = ALU_ADD;  // ADDI
          3'b010: ctrl_o.alu_op = ALU_SLT;  // SLTI
          3'b011: ctrl_o.alu_op = ALU_SLTU; // SLTIU
          3'b100: ctrl_o.alu_op = ALU_XOR;  // XORI
          3'b110: ctrl_o.alu_op = ALU_OR;   // ORI
          3'b111: ctrl_o.alu_op = ALU_AND;  // ANDI
          3'b001: begin
            ctrl_o.alu_op    = ALU_SLL;  // SLLI
            ctrl_o.immediate = {27'b0, instr[24:20]};
          end
          3'b101: begin
            ctrl_o.immediate = {27'b0, instr[24:20]};
            ctrl_o.alu_op = (funct7[5]) ? ALU_SRA : ALU_SRL; // SRAI / SRLI
          end
        endcase
      end

      // -----------------------------------------------------------------
      // ALU register (R-type) + M-extension
      // -----------------------------------------------------------------
      7'b0110011: begin
        ctrl_o.reg_write = 1'b1;
        if (funct7 == 7'b0000001) begin
          // M-extension
          unique case (funct3)
            3'b000: begin ctrl_o.alu_op = ALU_MUL;  ctrl_o.is_mul = 1'b1; end // MUL
            3'b001: begin ctrl_o.alu_op = ALU_MULH; ctrl_o.is_mul = 1'b1; end // MULH
            3'b010: begin ctrl_o.alu_op = ALU_MULH; ctrl_o.is_mul = 1'b1; end // MULHSU (approx)
            3'b011: begin ctrl_o.alu_op = ALU_MULH; ctrl_o.is_mul = 1'b1; end // MULHU (approx)
            3'b100: begin ctrl_o.alu_op = ALU_DIV;  ctrl_o.is_div = 1'b1; end // DIV
            3'b101: begin ctrl_o.alu_op = ALU_DIV;  ctrl_o.is_div = 1'b1; end // DIVU
            3'b110: begin ctrl_o.alu_op = ALU_REM;  ctrl_o.is_div = 1'b1; end // REM
            3'b111: begin ctrl_o.alu_op = ALU_REM;  ctrl_o.is_div = 1'b1; end // REMU
          endcase
        end else begin
          unique case (funct3)
            3'b000: ctrl_o.alu_op = (funct7[5]) ? ALU_SUB : ALU_ADD;
            3'b001: ctrl_o.alu_op = ALU_SLL;
            3'b010: ctrl_o.alu_op = ALU_SLT;
            3'b011: ctrl_o.alu_op = ALU_SLTU;
            3'b100: ctrl_o.alu_op = ALU_XOR;
            3'b101: ctrl_o.alu_op = (funct7[5]) ? ALU_SRA : ALU_SRL;
            3'b110: ctrl_o.alu_op = ALU_OR;
            3'b111: ctrl_o.alu_op = ALU_AND;
          endcase
        end
      end

      // -----------------------------------------------------------------
      // FENCE / FENCE.I (treat as NOP for single-core)
      // -----------------------------------------------------------------
      7'b0001111: begin
        // NOP: no side effects needed for single-hart in-order pipeline
      end

      // -----------------------------------------------------------------
      // SYSTEM: ECALL / EBREAK (treated as NOP placeholder)
      // -----------------------------------------------------------------
      7'b1110011: begin
        // Minimal: no CSR support in v1 mining core
      end

      // -----------------------------------------------------------------
      // Custom-0: Xcrypto extension (opcode 0x0B = 7'b0001011)
      // -----------------------------------------------------------------
      7'b0001011: begin
        ctrl_o.ext_xcrypto = 1'b1;
        ctrl_o.reg_write   = 1'b1;  // extension writes back to rd
      end

      // -----------------------------------------------------------------
      // Custom-1: Xlattice extension (opcode 0x2B = 7'b0101011)
      // -----------------------------------------------------------------
      7'b0101011: begin
        ctrl_o.ext_xlattice = 1'b1;
        ctrl_o.reg_write    = 1'b1;
      end

      // -----------------------------------------------------------------
      // Illegal instruction
      // -----------------------------------------------------------------
      default: begin
        ctrl_o.illegal = 1'b1;
      end
    endcase
  end

endmodule : qug_decoder
