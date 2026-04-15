// =============================================================================
// difficulty_regs.sv — MMIO Register Bank for Mining Control
// QUG-V1 Mining SoC — Phase 5: Difficulty Registers
// =============================================================================
// Project  : QUG-V1 RISC-V Mining SoC
// Target   : Xilinx Kintex-7 XC7K325T (FPGA prototype)
// Author   : Quillon Foundation / Dragon Ball Miner
// License  : MIT
// =============================================================================
// AXI4-Lite accessible register bank at base address 0x2000_0000.
// Provides CPU-writable mining parameters (challenge, difficulty, VDF depth,
// nonce start) and read-only status registers (solution nonce, LZC, hashrate
// counter). Start/stop are write-pulse registers.
//
// Register Map (byte offsets from base 0x2000_0000):
//   0x00  CTRL           — W: {30'd0, stop, start}  R: {31'd0, mining_active}
//   0x04  DIFFICULTY      — W/R: {24'd0, difficulty_target[7:0]}
//   0x08  VDF_DEPTH       — W/R: {18'd0, vdf_depth[13:0]}
//   0x0C  STATUS          — R: {solution_found, 23'd0, solution_lzc[7:0]}
//   0x10  CHALLENGE[0]    — W/R: challenge[255:224]  (MSW)
//   0x14  CHALLENGE[1]    — W/R: challenge[223:192]
//   0x18  CHALLENGE[2]    — W/R: challenge[191:160]
//   0x1C  CHALLENGE[3]    — W/R: challenge[159:128]
//   0x20  CHALLENGE[4]    — W/R: challenge[127:96]
//   0x24  CHALLENGE[5]    — W/R: challenge[95:64]
//   0x28  CHALLENGE[6]    — W/R: challenge[63:32]
//   0x2C  CHALLENGE[7]    — W/R: challenge[31:0]    (LSW)
//   0x30  NONCE_LO        — W/R: nonce_start[31:0]
//   0x34  NONCE_HI        — W/R: nonce_start[63:32]
//   0x38  SOL_NONCE_LO    — R: solution_nonce[31:0]
//   0x3C  SOL_NONCE_HI    — R: solution_nonce[63:32]
//   0x40  NONCES_TRIED    — R: nonces_tried[31:0]
// =============================================================================

module difficulty_regs
    import qug_pkg::*;
(
    input  logic        clk,
    input  logic        rst_n,

    // =========================================================================
    // AXI4-Lite slave interface (simplified: addr + data + we)
    // =========================================================================
    input  logic [7:0]  reg_addr,    // Register address (byte offset)
    input  logic [31:0] reg_wdata,   // Write data
    input  logic        reg_we,      // Write enable
    output logic [31:0] reg_rdata,   // Read data

    // =========================================================================
    // Mining controller interface — outputs
    // =========================================================================
    output logic [7:0]  difficulty_target,    // LWMA result (leading zero bits)
    output logic [255:0] challenge_hash,      // Current challenge
    output logic [63:0]  nonce_start,         // Starting nonce
    output logic [13:0]  vdf_depth,           // VDF chain depth
    output logic         mining_start,        // Pulse: begin mining
    output logic         mining_stop,         // Pulse: halt mining

    // =========================================================================
    // Status inputs (from mining_controller)
    // =========================================================================
    input  logic         mining_active,
    input  logic         solution_found,
    input  logic [63:0]  solution_nonce,
    input  logic [7:0]   solution_lzc,
    input  logic [31:0]  nonces_tried
);

    // =========================================================================
    // Register address offsets
    // =========================================================================
    localparam logic [7:0] ADDR_CTRL          = 8'h00;
    localparam logic [7:0] ADDR_DIFFICULTY     = 8'h04;
    localparam logic [7:0] ADDR_VDF_DEPTH      = 8'h08;
    localparam logic [7:0] ADDR_STATUS         = 8'h0C;
    localparam logic [7:0] ADDR_CHALLENGE_0    = 8'h10;
    localparam logic [7:0] ADDR_CHALLENGE_1    = 8'h14;
    localparam logic [7:0] ADDR_CHALLENGE_2    = 8'h18;
    localparam logic [7:0] ADDR_CHALLENGE_3    = 8'h1C;
    localparam logic [7:0] ADDR_CHALLENGE_4    = 8'h20;
    localparam logic [7:0] ADDR_CHALLENGE_5    = 8'h24;
    localparam logic [7:0] ADDR_CHALLENGE_6    = 8'h28;
    localparam logic [7:0] ADDR_CHALLENGE_7    = 8'h2C;
    localparam logic [7:0] ADDR_NONCE_LO       = 8'h30;
    localparam logic [7:0] ADDR_NONCE_HI       = 8'h34;
    localparam logic [7:0] ADDR_SOL_NONCE_LO   = 8'h38;
    localparam logic [7:0] ADDR_SOL_NONCE_HI   = 8'h3C;
    localparam logic [7:0] ADDR_NONCES_TRIED   = 8'h40;

    // =========================================================================
    // Writable register storage
    // =========================================================================
    logic [7:0]   reg_difficulty;
    logic [13:0]  reg_vdf_depth;
    logic [31:0]  reg_challenge [0:7];
    logic [31:0]  reg_nonce_lo;
    logic [31:0]  reg_nonce_hi;

    // Start/stop pulse registers (active for exactly one cycle)
    logic         start_pulse;
    logic         stop_pulse;

    // =========================================================================
    // Write logic
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_difficulty <= 8'd0;
            reg_vdf_depth  <= VDF_CHAIN_DEFAULT[13:0];
            for (int i = 0; i < 8; i++) begin
                reg_challenge[i] <= 32'd0;
            end
            reg_nonce_lo   <= 32'd0;
            reg_nonce_hi   <= 32'd0;
            start_pulse    <= 1'b0;
            stop_pulse     <= 1'b0;
        end else begin
            // Default: clear pulses after one cycle
            start_pulse <= 1'b0;
            stop_pulse  <= 1'b0;

            if (reg_we) begin
                case (reg_addr)
                    ADDR_CTRL: begin
                        // Bit 0 = start pulse, Bit 1 = stop pulse
                        start_pulse <= reg_wdata[0];
                        stop_pulse  <= reg_wdata[1];
                    end

                    ADDR_DIFFICULTY: begin
                        reg_difficulty <= reg_wdata[7:0];
                    end

                    ADDR_VDF_DEPTH: begin
                        reg_vdf_depth <= reg_wdata[13:0];
                    end

                    // Challenge hash: 8 words, big-endian word order
                    ADDR_CHALLENGE_0: reg_challenge[0] <= reg_wdata;
                    ADDR_CHALLENGE_1: reg_challenge[1] <= reg_wdata;
                    ADDR_CHALLENGE_2: reg_challenge[2] <= reg_wdata;
                    ADDR_CHALLENGE_3: reg_challenge[3] <= reg_wdata;
                    ADDR_CHALLENGE_4: reg_challenge[4] <= reg_wdata;
                    ADDR_CHALLENGE_5: reg_challenge[5] <= reg_wdata;
                    ADDR_CHALLENGE_6: reg_challenge[6] <= reg_wdata;
                    ADDR_CHALLENGE_7: reg_challenge[7] <= reg_wdata;

                    ADDR_NONCE_LO: reg_nonce_lo <= reg_wdata;
                    ADDR_NONCE_HI: reg_nonce_hi <= reg_wdata;

                    // SOL_NONCE_LO, SOL_NONCE_HI, NONCES_TRIED are read-only
                    default: ; // Ignore writes to read-only or unmapped registers
                endcase
            end
        end
    end

    // =========================================================================
    // Read logic (combinational)
    // =========================================================================
    always_comb begin
        reg_rdata = 32'd0;

        case (reg_addr)
            ADDR_CTRL:         reg_rdata = {31'd0, mining_active};
            ADDR_DIFFICULTY:   reg_rdata = {24'd0, reg_difficulty};
            ADDR_VDF_DEPTH:    reg_rdata = {18'd0, reg_vdf_depth};
            ADDR_STATUS:       reg_rdata = {solution_found, 23'd0, solution_lzc};
            ADDR_CHALLENGE_0:  reg_rdata = reg_challenge[0];
            ADDR_CHALLENGE_1:  reg_rdata = reg_challenge[1];
            ADDR_CHALLENGE_2:  reg_rdata = reg_challenge[2];
            ADDR_CHALLENGE_3:  reg_rdata = reg_challenge[3];
            ADDR_CHALLENGE_4:  reg_rdata = reg_challenge[4];
            ADDR_CHALLENGE_5:  reg_rdata = reg_challenge[5];
            ADDR_CHALLENGE_6:  reg_rdata = reg_challenge[6];
            ADDR_CHALLENGE_7:  reg_rdata = reg_challenge[7];
            ADDR_NONCE_LO:     reg_rdata = reg_nonce_lo;
            ADDR_NONCE_HI:     reg_rdata = reg_nonce_hi;
            ADDR_SOL_NONCE_LO: reg_rdata = solution_nonce[31:0];
            ADDR_SOL_NONCE_HI: reg_rdata = solution_nonce[63:32];
            ADDR_NONCES_TRIED: reg_rdata = nonces_tried;
            default:           reg_rdata = 32'd0;
        endcase
    end

    // =========================================================================
    // Output assignments
    // =========================================================================

    // Assemble 256-bit challenge from 8 words (big-endian word order)
    assign challenge_hash = {
        reg_challenge[0], reg_challenge[1], reg_challenge[2], reg_challenge[3],
        reg_challenge[4], reg_challenge[5], reg_challenge[6], reg_challenge[7]
    };

    // Assemble 64-bit nonce from lo/hi words
    assign nonce_start = {reg_nonce_hi, reg_nonce_lo};

    assign difficulty_target = reg_difficulty;
    assign vdf_depth         = reg_vdf_depth;
    assign mining_start      = start_pulse;
    assign mining_stop       = stop_pulse;

endmodule
