// =============================================================================
// QUG-V1 Mining SoC - Package Definitions
// =============================================================================
// Common types, enums, and structs for the RV32IMC core pipeline.
// Target: Kintex-7 @ 100 MHz
// =============================================================================

`ifndef QUG_PKG_SV
`define QUG_PKG_SV

package qug_core_pkg;

  // -------------------------------------------------------------------------
  // ISA parameters
  // -------------------------------------------------------------------------
  parameter int XLEN       = 32;
  parameter int REG_ADDR_W = 5;
  parameter int NUM_REGS   = 32;

  // -------------------------------------------------------------------------
  // ALU operation encoding
  // -------------------------------------------------------------------------
  typedef enum logic [3:0] {
    ALU_ADD  = 4'b0000,
    ALU_SUB  = 4'b0001,
    ALU_AND  = 4'b0010,
    ALU_OR   = 4'b0011,
    ALU_XOR  = 4'b0100,
    ALU_SLL  = 4'b0101,
    ALU_SRL  = 4'b0110,
    ALU_SRA  = 4'b0111,
    ALU_SLT  = 4'b1000,
    ALU_SLTU = 4'b1001,
    ALU_MUL  = 4'b1010,
    ALU_MULH = 4'b1011,
    ALU_DIV  = 4'b1100,
    ALU_REM  = 4'b1101,
    ALU_PASS = 4'b1110  // pass operand_a through
  } alu_op_e;

  // -------------------------------------------------------------------------
  // Immediate type encoding
  // -------------------------------------------------------------------------
  typedef enum logic [2:0] {
    IMM_I = 3'b000,
    IMM_S = 3'b001,
    IMM_B = 3'b010,
    IMM_U = 3'b011,
    IMM_J = 3'b100,
    IMM_C = 3'b101,  // compressed
    IMM_NONE = 3'b111
  } imm_type_e;

  // -------------------------------------------------------------------------
  // Branch condition encoding
  // -------------------------------------------------------------------------
  typedef enum logic [2:0] {
    BR_NONE = 3'b000,
    BR_EQ   = 3'b001,
    BR_NE   = 3'b010,
    BR_LT   = 3'b011,
    BR_GE   = 3'b100,
    BR_LTU  = 3'b101,
    BR_GEU  = 3'b110,
    BR_JAL  = 3'b111
  } branch_type_e;

  // -------------------------------------------------------------------------
  // Memory operation width
  // -------------------------------------------------------------------------
  typedef enum logic [1:0] {
    MEM_WORD = 2'b00,
    MEM_HALF = 2'b01,
    MEM_BYTE = 2'b10,
    MEM_NONE = 2'b11
  } mem_width_e;

  // -------------------------------------------------------------------------
  // Decoded control signals from decoder
  // -------------------------------------------------------------------------
  typedef struct packed {
    alu_op_e      alu_op;
    imm_type_e    imm_type;
    branch_type_e branch_type;
    mem_width_e   mem_width;
    logic         reg_write;     // write-back to rd
    logic         mem_read;      // load from memory
    logic         mem_write;     // store to memory
    logic         mem_signed;    // sign-extend load
    logic         alu_src_imm;   // operand B = immediate (vs rs2)
    logic         lui;           // LUI instruction
    logic         auipc;         // AUIPC instruction
    logic         jal;           // JAL
    logic         jalr;          // JALR
    logic         ext_xcrypto;   // custom-0 opcode 0x0B
    logic         ext_xlattice;  // custom-1 opcode 0x2B
    logic         is_mul;        // M-extension multiply
    logic         is_div;        // M-extension divide
    logic         illegal;       // illegal instruction
    logic [4:0]   rs1_addr;
    logic [4:0]   rs2_addr;
    logic [4:0]   rd_addr;
    logic [31:0]  immediate;
  } decoded_ctrl_t;

  // -------------------------------------------------------------------------
  // Pipeline register: IF -> ID
  // -------------------------------------------------------------------------
  typedef struct packed {
    logic [31:0] pc;
    logic [31:0] instr;
    logic        valid;
  } if_id_reg_t;

  // -------------------------------------------------------------------------
  // Pipeline register: ID -> EX1
  // -------------------------------------------------------------------------
  typedef struct packed {
    logic [31:0]  pc;
    logic [31:0]  rs1_data;
    logic [31:0]  rs2_data;
    decoded_ctrl_t ctrl;
    logic          valid;
  } id_ex1_reg_t;

  // -------------------------------------------------------------------------
  // Pipeline register: EX1 -> EX2
  // -------------------------------------------------------------------------
  typedef struct packed {
    logic [31:0]  pc;
    logic [31:0]  alu_result;
    logic [31:0]  rs2_data;     // for store forwarding
    decoded_ctrl_t ctrl;
    logic          valid;
  } ex1_ex2_reg_t;

  // -------------------------------------------------------------------------
  // Pipeline register: EX2 -> MEM
  // -------------------------------------------------------------------------
  typedef struct packed {
    logic [31:0]  pc;
    logic [31:0]  alu_result;
    logic [31:0]  rs2_data;
    decoded_ctrl_t ctrl;
    logic          valid;
  } ex2_mem_reg_t;

  // -------------------------------------------------------------------------
  // Pipeline register: MEM -> WB1
  // -------------------------------------------------------------------------
  typedef struct packed {
    logic [31:0]  pc;
    logic [31:0]  result;       // alu or mem load
    decoded_ctrl_t ctrl;
    logic          valid;
  } mem_wb1_reg_t;

  // -------------------------------------------------------------------------
  // Pipeline register: WB1 -> WB2
  // -------------------------------------------------------------------------
  typedef struct packed {
    logic [31:0]  result;
    logic [4:0]   rd_addr;
    logic         reg_write;
    logic         valid;
  } wb1_wb2_reg_t;

  // -------------------------------------------------------------------------
  // Extension interface signals (Xcrypto / Xlattice)
  // -------------------------------------------------------------------------
  typedef struct packed {
    logic [31:0] instruction;
    logic [31:0] rs1_data;
    logic [31:0] rs2_data;
    logic        valid;
  } ext_request_t;

  typedef struct packed {
    logic [31:0] result;
    logic        ready;
  } ext_response_t;

endpackage : qug_core_pkg

`endif
