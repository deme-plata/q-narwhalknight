// =============================================================================
// QUG-V1 Mining SoC — Xcrypto (BLAKE3) Package
// =============================================================================
// Project  : QUG-V1 RISC-V Mining SoC
// Target   : Xilinx Kintex-7 XC7K325T (FPGA prototype)
// Author   : Quillon Foundation / Dragon Ball Miner
// License  : MIT
// =============================================================================
// Defines BLAKE3 initialization vectors, message schedule permutations,
// Xcrypto instruction encodings, and state register file types.
//
// Reference: BLAKE3 specification (https://github.com/BLAKE3-team/BLAKE3-specs)
// The BLAKE3 compression function operates on a 4x4 matrix of 32-bit words.
// QUG-V1 implements 7-round BLAKE3 compression in hardware via the Xcrypto
// custom instruction extension (RISC-V custom-0 opcode 0x0B).
// =============================================================================

package xcrypto_pkg;

  import qug_pkg::*;

  // ===========================================================================
  // BLAKE3 Initialization Vectors (same as SHA-256 H0..H7)
  // ===========================================================================
  // These are the first 32 bits of the fractional parts of the square roots
  // of the first 8 prime numbers (2, 3, 5, 7, 11, 13, 17, 19).

  localparam int BLAKE3_NUM_IV = 8;
  // BLAKE3_IV defined as individual scalars for tool compatibility.
  // Modules that need the array form define it locally (same values).
  localparam logic [31:0] BLAKE3_IV_0 = 32'h6A09E667; // sqrt(2)
  localparam logic [31:0] BLAKE3_IV_1 = 32'hBB67AE85; // sqrt(3)
  localparam logic [31:0] BLAKE3_IV_2 = 32'h3C6EF372; // sqrt(5)
  localparam logic [31:0] BLAKE3_IV_3 = 32'hA54FF53A; // sqrt(7)
  localparam logic [31:0] BLAKE3_IV_4 = 32'h510E527F; // sqrt(11)
  localparam logic [31:0] BLAKE3_IV_5 = 32'h9B05688C; // sqrt(13)
  localparam logic [31:0] BLAKE3_IV_6 = 32'h1F83D9AB; // sqrt(17)
  localparam logic [31:0] BLAKE3_IV_7 = 32'h5BE0CD19; // sqrt(19)

  // ===========================================================================
  // BLAKE3 State Dimensions
  // ===========================================================================

  localparam int BLAKE3_STATE_WORDS  = 16;  // 4x4 matrix of 32-bit words
  localparam int BLAKE3_STATE_BYTES  = BLAKE3_STATE_WORDS * 4;  // 64 bytes
  localparam int BLAKE3_MSG_WORDS    = 16;  // 16 message words per block
  localparam int BLAKE3_BLOCK_BYTES  = 64;  // 64-byte input block
  localparam int BLAKE3_KEY_WORDS    = 8;
  localparam int BLAKE3_OUT_WORDS    = 8;   // 256-bit output (words 0..7)
  localparam int BLAKE3_ROUNDS       = 7;

  // ===========================================================================
  // BLAKE3 Domain Separation Flags
  // ===========================================================================

  localparam logic [7:0] BLAKE3_FLAG_CHUNK_START  = 8'h01;
  localparam logic [7:0] BLAKE3_FLAG_CHUNK_END    = 8'h02;
  localparam logic [7:0] BLAKE3_FLAG_PARENT       = 8'h04;
  localparam logic [7:0] BLAKE3_FLAG_ROOT         = 8'h08;
  localparam logic [7:0] BLAKE3_FLAG_KEYED_HASH   = 8'h10;
  localparam logic [7:0] BLAKE3_FLAG_DERIVE_KEY_C = 8'h20; // derive_key_context
  localparam logic [7:0] BLAKE3_FLAG_DERIVE_KEY_M = 8'h40; // derive_key_material

  // ===========================================================================
  // Quarter-Round Rotation Constants
  // ===========================================================================
  // BLAKE3 quarter-round: G(a, b, c, d) uses four rotations.
  //   a = a + b + mx;  d = (d ^ a) >>> R1;
  //   c = c + d;       b = (b ^ c) >>> R2;
  //   a = a + b + my;  d = (d ^ a) >>> R3;
  //   c = c + d;       b = (b ^ c) >>> R4;

  localparam int BLAKE3_ROT_1 = 16;
  localparam int BLAKE3_ROT_2 = 12;
  localparam int BLAKE3_ROT_3 =  8;
  localparam int BLAKE3_ROT_4 =  7;

  // ===========================================================================
  // BLAKE3 Message Schedule Permutation
  // ===========================================================================
  // Each round permutes the 16 message words according to a fixed schedule.
  // BLAKE3 uses a single permutation applied repeatedly (unlike BLAKE2's 10
  // distinct sigma permutations).
  //
  // Permutation: {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8}
  //
  // MSG_SCHEDULE[round][i] gives the message word index to use at position i
  // in the given round. Round 0 uses identity (0,1,2,...,15), subsequent
  // rounds apply the permutation cumulatively.

  // MSG_PERM and MSG_SCHEDULE defined locally in blake3_round.sv for tool compatibility.

  // ===========================================================================
  // Xcrypto Instruction Encodings (funct7 field)
  // ===========================================================================
  // All Xcrypto instructions use OPC_CUSTOM_0 (0x0B) with funct3 = 3'b000.
  // The funct7 field selects the specific operation.
  //
  // Encoding: {funct7[6:0], rs2[4:0], rs1[4:0], 3'b000, rd[4:0], 7'b000_1011}
  //
  //   blake3.init     rd, rs1       — Initialize state from chaining value addr
  //   blake3.round    rd, rs1, rs2  — Execute one BLAKE3 round
  //   blake3.chain    rd            — Read chaining value output
  //   blake3.finalize rd            — Finalize and output hash to rd (multi-cycle)

  localparam logic [6:0] XCRYPTO_INIT      = 7'b000_0000;  // funct7 = 0
  localparam logic [6:0] XCRYPTO_ROUND     = 7'b000_0001;  // funct7 = 1
  localparam logic [6:0] XCRYPTO_CHAIN     = 7'b000_0010;  // funct7 = 2
  localparam logic [6:0] XCRYPTO_FINALIZE  = 7'b000_0011;  // funct7 = 3
  localparam logic [6:0] XCRYPTO_LOAD_MSG  = 7'b000_0100;  // funct7 = 4, load msg block
  localparam logic [6:0] XCRYPTO_STATUS    = 7'b000_0101;  // funct7 = 5, read engine status
  localparam logic [6:0] XCRYPTO_SET_VDF_DEPTH = 7'b000_0110;  // funct7 = 6, set VDF depth
  localparam logic [6:0] XCRYPTO_SET_DIFFICULTY = 7'b000_0111;  // funct7 = 7, set difficulty target

  // ===========================================================================
  // Xcrypto Engine Status Bits
  // ===========================================================================

  typedef struct packed {
    logic [23:0] reserved;
    logic [2:0]  current_round;   // Current round index (0..6)
    logic        finalized;       // Hash output is valid
    logic        msg_loaded;      // Message block has been loaded
    logic        state_valid;     // State register file is initialized
    logic        error;           // Error flag (e.g., invalid sequence)
    logic        busy;            // Engine is processing a round
  } xcrypto_status_t;

  // ===========================================================================
  // State Register File Type
  // ===========================================================================
  // The BLAKE3 compression function operates on a 16-word (512-bit) state
  // arranged as a 4x4 matrix:
  //
  //   | v0  v1  v2  v3  |     h[0..3]           — chaining value
  //   | v4  v5  v6  v7  |     h[4..7]           — chaining value
  //   | v8  v9  v10 v11 |     IV[0..3]          — constants
  //   | v12 v13 v14 v15 |     counter_lo, counter_hi, block_len, flags

  typedef logic [31:0] blake3_state_t [0:15];
  typedef logic [31:0] blake3_msg_t   [0:15];

  // Packed 256-bit hash output
  typedef logic [255:0] blake3_hash_t;

  // ===========================================================================
  // Xcrypto Decoded Instruction
  // ===========================================================================

  typedef struct packed {
    logic [6:0]  funct7;
    rf_addr_t    rs2;
    rf_addr_t    rs1;
    logic [2:0]  funct3;
    rf_addr_t    rd;
    logic        valid;          // Instruction is a valid Xcrypto op
  } xcrypto_instr_t;

  // ===========================================================================
  // Helper Function: Right-rotate a 32-bit word
  // ===========================================================================

  function automatic logic [31:0] rotr32(
    input logic [31:0] x,
    input int n
  );
    return (x >> n) | (x << (32 - n));
  endfunction : rotr32

  // ===========================================================================
  // SHA-3 / Keccak-f[1600] Constants (v10.3.0 — Hybrid Mining Support)
  // ===========================================================================

  localparam int KECCAK_ROUNDS     = 24;
  localparam int KECCAK_LANES      = 25;   // 5x5 grid
  localparam int KECCAK_LANE_W     = 64;   // 64-bit lanes
  localparam int SHA3_256_RATE     = 1088;  // Rate in bits (136 bytes)
  localparam int SHA3_256_CAPACITY = 512;   // Capacity in bits
  localparam int SHA3_256_OUTPUT   = 256;   // Output bits
  localparam int SHA3_RATE_LANES   = 17;    // 1088 / 64 = 17 lanes

  // KECCAK_RC and KECCAK_RHO defined locally in sha3_keccak.sv for tool compatibility.

  // Xcrypto funct7 encodings for SHA-3 (under funct3 = XCRYPTO_F3_KECCAK = 3'b011)
  localparam logic [6:0] KECCAK_INIT    = 7'b000_0000;  // funct7 = 0: Zero state
  localparam logic [6:0] KECCAK_ABSORB  = 7'b000_0001;  // funct7 = 1: XOR rate block
  localparam logic [6:0] KECCAK_SQUEEZE = 7'b000_0010;  // funct7 = 2: Permute + extract
  localparam logic [6:0] KECCAK_CHAIN   = 7'b000_0011;  // funct7 = 3: SHA-3 VDF chain

endpackage : xcrypto_pkg
