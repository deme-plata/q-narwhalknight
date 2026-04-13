// =============================================================================
// xcrypto_scratchpad.sv -- 512-bit Tightly-Coupled Scratchpad for Xcrypto
// QUG-V1 Mining SoC -- BLAKE3 Message Block Storage
// =============================================================================
// Project  : QUG-V1 RISC-V Mining SoC
// Target   : Xilinx Kintex-7 (FPGA) / TSMC 12nm (ASIC)
// Author   : Quillon Foundation / Dragon Ball Miner
// License  : MIT
// =============================================================================
//
// Stores the 64-byte (16 x 32-bit) BLAKE3 message block for Xcrypto.
//
// Mining message layout (40 bytes of data in 16-word block):
//   word[ 0..7] = challenge_hash[0..31]  (32 bytes, 8 LE u32 words)
//   word[ 8..9] = nonce                  (8 bytes, LE u64 split into 2 words)
//   word[10..15] = 0x00000000            (24 bytes zero padding)
//
// Two access modes:
//   1. Narrow write port (32-bit): CPU/mining controller writes individual
//      words to construct the message block.
//   2. Wide read port (512-bit): Xcrypto reads all 16 words in a single
//      cycle when blake3.round or blake3.chain is issued.
//
// Read model: SYNCHRONOUS — both rd_data and rd_valid are registered.
// Data and valid change on the same clock edge. This prevents the
// consumer from seeing stale data with a premature valid.
//
// This replaces the zero-fill stub in qug_tile.sv that caused all BLAKE3
// chain hashes to compute on zero input.
//
// Resource estimate: 1024 FF + ~50 LUT (512 storage + 512 read reg + decode)
// =============================================================================

module xcrypto_scratchpad (
    input  logic        clk,
    input  logic        rst_n,

    // =========================================================================
    // Narrow write port (from CPU data bus or mining controller)
    // =========================================================================
    input  logic [3:0]  wr_idx,        // Word index 0..15
    input  logic [31:0] wr_data,       // 32-bit write data
    input  logic        wr_en,         // Write enable

    // =========================================================================
    // Bulk write port (from mining controller -- write all 16 words at once)
    // =========================================================================
    input  logic [31:0] bulk_wr_data [0:15],  // 512-bit bulk write
    input  logic        bulk_wr_en,            // Bulk write enable

    // =========================================================================
    // Wide read port (to xcrypto_unit -- 512-bit single-cycle read)
    // =========================================================================
    input  logic        rd_en,         // Read enable (from xcrypto FSM)
    output logic [31:0] rd_data [0:15],// 16 x 32-bit message words
    output logic        rd_valid       // Data valid (1-cycle latency)
);

    // =========================================================================
    // Storage: 16 x 32-bit registers
    // =========================================================================
    logic [31:0] mem [0:15];

    // =========================================================================
    // Synchronous read registers (data + valid change on same clock edge)
    // =========================================================================
    logic [31:0] rd_data_r [0:15];
    logic        rd_valid_r;

    // =========================================================================
    // Write logic: bulk write takes priority over narrow write
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 16; i++) begin
                mem[i] <= 32'd0;
            end
        end else begin
            // Bulk write (from mining controller) has priority
            if (bulk_wr_en) begin
                for (int i = 0; i < 16; i++) begin
                    mem[i] <= bulk_wr_data[i];
                end
            end else if (wr_en) begin
                mem[wr_idx] <= wr_data;
            end
        end
    end

    // =========================================================================
    // Synchronous read: data and valid registered together
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 16; i++) begin
                rd_data_r[i] <= 32'd0;
            end
            rd_valid_r <= 1'b0;
        end else begin
            rd_valid_r <= rd_en;
            if (rd_en) begin
                for (int i = 0; i < 16; i++) begin
                    rd_data_r[i] <= mem[i];
                end
            end
        end
    end

    // Output assignment
    always_comb begin
        for (int i = 0; i < 16; i++) begin
            rd_data[i] = rd_data_r[i];
        end
    end

    assign rd_valid = rd_valid_r;

endmodule
