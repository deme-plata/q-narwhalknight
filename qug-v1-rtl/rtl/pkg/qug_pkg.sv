// =============================================================================
// QUG-V1 Mining SoC — Global Package
// =============================================================================
// Project  : QUG-V1 RISC-V Mining SoC
// Target   : Xilinx Kintex-7 XC7K325T (FPGA prototype)
// Author   : Quillon Foundation / Dragon Ball Miner
// License  : MIT
// =============================================================================
// Defines global parameters, types, opcode encodings, and AXI4-Lite interface
// structures used across all RTL modules.
// =============================================================================

package qug_pkg;

  // ===========================================================================
  // Global Parameters
  // ===========================================================================

  localparam int NUM_CORES         = 16;       // 4x4 mesh of tiles
  localparam int MESH_ROWS         = 4;
  localparam int MESH_COLS         = 4;
  localparam int XLEN              = 32;       // RV32IMC base ISA
  localparam int ILEN              = 32;       // Instruction width
  localparam int BLAKE3_ROUNDS     = 7;        // BLAKE3 compression rounds
  localparam int NTT_DEPTH         = 256;      // NTT butterfly depth
  localparam int RF_ADDR_W         = 5;        // Register file address width (x0..x31)
  localparam int PHYS_ADDR_W       = 32;       // Physical address width
  localparam int L1I_SIZE_KB       = 8;        // L1 instruction cache per tile
  localparam int L1D_SIZE_KB       = 8;        // L1 data cache per tile
  localparam int L1_LINE_BYTES     = 32;       // Cache line size
  localparam int L2_SIZE_KB        = 256;      // Shared L2 cache
  localparam int AXI_DATA_W        = 32;       // AXI data bus width
  localparam int AXI_ADDR_W        = 32;       // AXI address bus width
  localparam int AXI_STRB_W        = AXI_DATA_W / 8;
  localparam int PIPELINE_STAGES   = 7;        // IF-ID-IS-EX-M1-M2-WB

  // Target clock frequency (Hz)
  localparam int CLK_FREQ_HZ       = 100_000_000; // 100 MHz FPGA prototype

  // ===========================================================================
  // Common Types
  // ===========================================================================

  typedef logic [XLEN-1:0]         word_t;      // General-purpose word
  typedef logic [XLEN-1:0]         addr_t;      // Address
  typedef logic [XLEN-1:0]         data_t;      // Data word
  typedef logic [RF_ADDR_W-1:0]    rf_addr_t;   // Register file index (5-bit)
  typedef logic [ILEN-1:0]         instr_t;     // Encoded instruction

  // ===========================================================================
  // RISC-V Opcode Map (RV32IMC base)
  // ===========================================================================
  // Standard encoding: instr[6:0] = opcode
  //   funct7[31:25]  rs2[24:20]  rs1[19:15]  funct3[14:12]  rd[11:7]  opcode[6:0]

  localparam logic [6:0] OPC_LUI       = 7'b011_0111;  // U-type
  localparam logic [6:0] OPC_AUIPC     = 7'b001_0111;  // U-type
  localparam logic [6:0] OPC_JAL       = 7'b110_1111;  // J-type
  localparam logic [6:0] OPC_JALR      = 7'b110_0111;  // I-type
  localparam logic [6:0] OPC_BRANCH    = 7'b110_0011;  // B-type
  localparam logic [6:0] OPC_LOAD      = 7'b000_0011;  // I-type
  localparam logic [6:0] OPC_STORE     = 7'b010_0011;  // S-type
  localparam logic [6:0] OPC_OP_IMM    = 7'b001_0011;  // I-type (ADDI, SLTI, ...)
  localparam logic [6:0] OPC_OP        = 7'b011_0011;  // R-type (ADD, MUL, ...)
  localparam logic [6:0] OPC_MISC_MEM  = 7'b000_1111;  // FENCE
  localparam logic [6:0] OPC_SYSTEM    = 7'b111_0011;  // CSR, ECALL, EBREAK

  // Custom extension opcodes (RISC-V reserved for custom use)
  localparam logic [6:0] OPC_CUSTOM_0  = 7'b000_1011;  // 0x0B — Xcrypto (BLAKE3)
  localparam logic [6:0] OPC_CUSTOM_1  = 7'b010_1011;  // 0x2B — Xlattice (NTT)

  // ===========================================================================
  // Xcrypto Extension — funct3 encodings (within custom-0)
  // ===========================================================================
  // R-type format: funct7 | rs2 | rs1 | funct3 | rd | OPC_CUSTOM_0

  localparam logic [2:0] XCRYPTO_F3_BLAKE3   = 3'b000;  // BLAKE3 operations
  localparam logic [2:0] XCRYPTO_F3_SHA256    = 3'b001;  // Reserved: SHA-256
  localparam logic [2:0] XCRYPTO_F3_AES       = 3'b010;  // Reserved: AES-256
  localparam logic [2:0] XCRYPTO_F3_KECCAK    = 3'b011;  // Reserved: Keccak

  // ===========================================================================
  // Xlattice Extension — funct3 encodings (within custom-1)
  // ===========================================================================
  // R-type format: funct7 | rs2 | rs1 | funct3 | rd | OPC_CUSTOM_1

  localparam logic [2:0] XLATTICE_F3_NTT      = 3'b000;  // NTT operations
  localparam logic [2:0] XLATTICE_F3_POLY     = 3'b001;  // Polynomial arithmetic
  localparam logic [2:0] XLATTICE_F3_REDUCE   = 3'b010;  // Modular reduction
  localparam logic [2:0] XLATTICE_F3_SAMPLE   = 3'b011;  // Sampling / rejection

  // ===========================================================================
  // Pipeline Stage Enumeration
  // ===========================================================================

  typedef enum logic [2:0] {
    STAGE_IF  = 3'd0,   // Instruction Fetch
    STAGE_ID  = 3'd1,   // Instruction Decode
    STAGE_IS  = 3'd2,   // Issue / Operand Read
    STAGE_EX  = 3'd3,   // Execute / ALU / Xcrypto / Xlattice
    STAGE_M1  = 3'd4,   // Memory Access 1
    STAGE_M2  = 3'd5,   // Memory Access 2 (for multi-cycle loads)
    STAGE_WB  = 3'd6    // Write-Back
  } pipe_stage_e;

  // ===========================================================================
  // ALU Operation Enumeration
  // ===========================================================================

  typedef enum logic [3:0] {
    ALU_ADD   = 4'd0,
    ALU_SUB   = 4'd1,
    ALU_SLL   = 4'd2,
    ALU_SLT   = 4'd3,
    ALU_SLTU  = 4'd4,
    ALU_XOR   = 4'd5,
    ALU_SRL   = 4'd6,
    ALU_SRA   = 4'd7,
    ALU_OR    = 4'd8,
    ALU_AND   = 4'd9,
    ALU_MUL   = 4'd10,
    ALU_MULH  = 4'd11,
    ALU_DIV   = 4'd12,
    ALU_REM   = 4'd13,
    ALU_PASS  = 4'd14,  // Pass-through (LUI, AUIPC)
    ALU_NOP   = 4'd15
  } alu_op_e;

  // ===========================================================================
  // AXI4-Lite Write Address Channel
  // ===========================================================================

  typedef struct packed {
    logic [AXI_ADDR_W-1:0]  awaddr;
    logic [2:0]              awprot;
    logic                    awvalid;
  } axi4l_aw_t;

  // ===========================================================================
  // AXI4-Lite Write Data Channel
  // ===========================================================================

  typedef struct packed {
    logic [AXI_DATA_W-1:0]  wdata;
    logic [AXI_STRB_W-1:0]  wstrb;
    logic                    wvalid;
  } axi4l_w_t;

  // ===========================================================================
  // AXI4-Lite Write Response Channel
  // ===========================================================================

  typedef struct packed {
    logic [1:0]              bresp;
    logic                    bvalid;
  } axi4l_b_t;

  // ===========================================================================
  // AXI4-Lite Read Address Channel
  // ===========================================================================

  typedef struct packed {
    logic [AXI_ADDR_W-1:0]  araddr;
    logic [2:0]              arprot;
    logic                    arvalid;
  } axi4l_ar_t;

  // ===========================================================================
  // AXI4-Lite Read Data Channel
  // ===========================================================================

  typedef struct packed {
    logic [AXI_DATA_W-1:0]  rdata;
    logic [1:0]              rresp;
    logic                    rvalid;
  } axi4l_r_t;

  // ===========================================================================
  // AXI4-Lite Master Interface (bundled)
  // ===========================================================================

  typedef struct packed {
    axi4l_aw_t  aw;
    axi4l_w_t   w;
    logic        bready;
    axi4l_ar_t  ar;
    logic        rready;
  } axi4l_master_t;

  // ===========================================================================
  // AXI4-Lite Slave Interface (bundled)
  // ===========================================================================

  typedef struct packed {
    logic        awready;
    logic        wready;
    axi4l_b_t   b;
    logic        arready;
    axi4l_r_t   r;
  } axi4l_slave_t;

  // ===========================================================================
  // AXI Response Codes
  // ===========================================================================

  localparam logic [1:0] AXI_RESP_OKAY   = 2'b00;
  localparam logic [1:0] AXI_RESP_EXOKAY = 2'b01;
  localparam logic [1:0] AXI_RESP_SLVERR = 2'b10;
  localparam logic [1:0] AXI_RESP_DECERR = 2'b11;

  // ===========================================================================
  // Tile ID Type (for mesh NoC addressing)
  // ===========================================================================

  typedef struct packed {
    logic [$clog2(MESH_ROWS)-1:0]  row;
    logic [$clog2(MESH_COLS)-1:0]  col;
  } tile_id_t;

  // ===========================================================================
  // Exception Codes
  // ===========================================================================

  typedef enum logic [3:0] {
    EXC_NONE              = 4'd0,
    EXC_INSTR_MISALIGN    = 4'd1,
    EXC_INSTR_ACCESS      = 4'd2,
    EXC_ILLEGAL_INSTR     = 4'd3,
    EXC_BREAKPOINT        = 4'd4,
    EXC_LOAD_MISALIGN     = 4'd5,
    EXC_LOAD_ACCESS       = 4'd6,
    EXC_STORE_MISALIGN    = 4'd7,
    EXC_STORE_ACCESS      = 4'd8,
    EXC_ECALL             = 4'd9
  } exc_code_e;

  // ===========================================================================
  // Mining Algorithm Parameters (v10.3.0+ parity)
  // ===========================================================================

  // VDF chain depth — widened to support Genus-2 VDF (5K-10K iterations)
  localparam int VDF_CHAIN_DEPTH_W   = 14;       // Supports up to 16,383 iterations
  localparam int VDF_CHAIN_DEFAULT    = 100;      // Legacy BLAKE3 chain (Quillon v4.1)
  localparam int VDF_CHAIN_GENUS2     = 5000;     // Genus-2 VDF minimum iterations
  localparam int VDF_CHAIN_GENUS2_MAX = 10000;    // Genus-2 VDF maximum iterations

  // LWMA difficulty adjustment — hardware stores result, software computes
  localparam int DIFFICULTY_REG_W     = 8;        // Leading-zero-bit target (max 255)
  localparam int LWMA_WINDOW_SIZE     = 60;       // Block window for LWMA (firmware ref)

endpackage : qug_pkg
