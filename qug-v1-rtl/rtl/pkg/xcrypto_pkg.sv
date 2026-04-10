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

  localparam int unsigned BLAKE3_NUM_IV = 8;

  localparam logic [31:0] BLAKE3_IV [BLAKE3_NUM_IV] = '{
    32'h6A09E667,   // IV[0] = sqrt(2)
    32'hBB67AE85,   // IV[1] = sqrt(3)
    32'h3C6EF372,   // IV[2] = sqrt(5)
    32'hA54FF53A,   // IV[3] = sqrt(7)
    32'h510E527F,   // IV[4] = sqrt(11)
    32'h9B05688C,   // IV[5] = sqrt(13)
    32'h1F83D9AB,   // IV[6] = sqrt(17)
    32'h5BE0CD19    // IV[7] = sqrt(19)
  };

  // ===========================================================================
  // BLAKE3 State Dimensions
  // ===========================================================================

  localparam int unsigned BLAKE3_STATE_WORDS  = 16;  // 4x4 matrix of 32-bit words
  localparam int unsigned BLAKE3_STATE_BYTES  = BLAKE3_STATE_WORDS * 4;  // 64 bytes
  localparam int unsigned BLAKE3_MSG_WORDS    = 16;  // 16 message words per block
  localparam int unsigned BLAKE3_BLOCK_BYTES  = 64;  // 64-byte input block
  localparam int unsigned BLAKE3_KEY_WORDS    = 8;
  localparam int unsigned BLAKE3_OUT_WORDS    = 8;   // 256-bit output (words 0..7)
  localparam int unsigned BLAKE3_ROUNDS       = 7;

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

  localparam int unsigned BLAKE3_ROT_1 = 16;
  localparam int unsigned BLAKE3_ROT_2 = 12;
  localparam int unsigned BLAKE3_ROT_3 =  8;
  localparam int unsigned BLAKE3_ROT_4 =  7;

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

  localparam int unsigned MSG_PERM [16] = '{
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8
  };

  // Pre-computed cumulative permutations for all 7 rounds.
  // Round 0: identity ordering
  // Round N: apply MSG_PERM to round N-1 ordering
  localparam int unsigned MSG_SCHEDULE [BLAKE3_ROUNDS][16] = '{
    // Round 0 (identity)
    '{ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    // Round 1 (apply permutation once)
    '{ 2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8},
    // Round 2
    '{ 3,  4, 10, 12,  13, 2,  7, 14,  6,  5, 9,   0, 11, 15,  8,  1},
    // Round 3
    '{10,  7, 12,  9,  14, 3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6},
    // Round 4
    '{12, 13,  9, 11,  15, 10, 14,  8,  7,  2,  5,  3,  0,  1,  6,  4},
    // Round 5
    '{ 9, 14, 11,  5,   8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7},
    // Round 6
    '{11, 15,  5,  0,   1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13}
  };

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

  typedef logic [31:0] blake3_state_t [BLAKE3_STATE_WORDS];
  typedef logic [31:0] blake3_msg_t   [BLAKE3_MSG_WORDS];

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
    input int unsigned n
  );
    return (x >> n) | (x << (32 - n));
  endfunction : rotr32

endpackage : xcrypto_pkg
