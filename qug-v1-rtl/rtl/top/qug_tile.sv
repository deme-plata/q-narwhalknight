// =============================================================================
// qug_tile.sv -- Single Mining Tile
// QUG-V1 Mining SoC -- Tile Wrapper
// =============================================================================
// Project  : QUG-V1 RISC-V Mining SoC
// Target   : Xilinx Kintex-7 XC7K325T (FPGA prototype)
// Author   : Quillon Foundation / Dragon Ball Miner
// License  : MIT
// =============================================================================
//
// A tile wraps one RV32IMC core with its Xcrypto (BLAKE3) and Xlattice
// (256-bit field arithmetic) extension units.  The tile exposes flat
// instruction-memory and data-memory interfaces so the SoC top-level can
// attach shared BRAM or a cache hierarchy.
//
// For the FPGA prototype (NUM_TILES=1) there is no cache -- memory
// interfaces connect directly to the mem_subsystem arbiter.
//
//              +-----------------------------+
//              |         QUG Tile            |
//              |  +--------+  +-----------+  |
//  imem_* <----|  | Core   |--| Xcrypto   |  |
//  dmem_* <----|  |RV32IMC |--| BLAKE3    |  |
//              |  +--------+  +-----------+  |
//              |       |      +-----------+  |
//              |       +------| Xlattice  |  |
//              |              | 256-bit   |  |
//              |              +-----------+  |
//              +-----------------------------+
//
// =============================================================================

module qug_tile
    import qug_pkg::*;
#(
    parameter int unsigned TILE_ID = 0   // Multi-tile identification
) (
    input  logic        clk,
    input  logic        rst_n,

    // =========================================================================
    // Instruction memory interface (read-only, word-aligned)
    // =========================================================================
    output logic [31:0] imem_addr,
    input  logic [31:0] imem_rdata,
    output logic        imem_req,
    input  logic        imem_gnt,

    // =========================================================================
    // Data memory interface (read/write, byte-lane enables)
    // =========================================================================
    output logic [31:0] dmem_addr,
    output logic [31:0] dmem_wdata,
    input  logic [31:0] dmem_rdata,
    output logic [3:0]  dmem_we,
    output logic        dmem_req,
    input  logic        dmem_gnt
);

    // =========================================================================
    // Internal wires: core <-> xcrypto
    // =========================================================================
    logic [31:0] core_xcrypto_instr;
    logic [31:0] core_xcrypto_rs1;
    logic [31:0] core_xcrypto_rs2;
    logic        core_xcrypto_valid;
    logic [31:0] core_xcrypto_result;
    logic        core_xcrypto_ready;

    // =========================================================================
    // Internal wires: core <-> xlattice
    // =========================================================================
    logic [31:0] core_xlattice_instr;
    logic [31:0] core_xlattice_rs1;
    logic [31:0] core_xlattice_rs2;
    logic        core_xlattice_valid;
    logic [31:0] core_xlattice_result;
    logic        core_xlattice_ready;

    // =========================================================================
    // Xcrypto decoded fields (extract from instruction word)
    // =========================================================================
    logic [6:0]  xc_funct7;
    logic [2:0]  xc_funct3;
    logic [4:0]  xc_rd_addr;

    assign xc_funct7  = core_xcrypto_instr[31:25];
    assign xc_funct3  = core_xcrypto_instr[14:12];
    assign xc_rd_addr = core_xcrypto_instr[11:7];

    // =========================================================================
    // Xlattice decoded fields
    // =========================================================================
    logic [6:0]  xl_funct7;
    logic [2:0]  xl_funct3;
    logic [4:0]  xl_rd_addr;

    assign xl_funct7  = core_xlattice_instr[31:25];
    assign xl_funct3  = core_xlattice_instr[14:12];
    assign xl_rd_addr = core_xlattice_instr[11:7];

    // =========================================================================
    // Xcrypto unit response signals
    // =========================================================================
    logic        xc_resp_valid;
    logic [4:0]  xc_resp_rd_addr;
    logic [31:0] xc_resp_data;
    logic        xc_resp_wr_en;
    logic        xc_req_ready;

    // Xcrypto message-block memory interface (tightly coupled to data memory)
    logic [31:0] xc_mem_addr;
    logic        xc_mem_rd_en;
    logic [31:0] xc_mem_block [0:15];
    logic        xc_mem_valid;

    // =========================================================================
    // Xcrypto scratchpad: 512-bit tightly-coupled message block storage
    // =========================================================================
    // Replaces the zero-fill stub. The mining controller or CPU writes the
    // message block (challenge[0:7] + nonce[8:9] + zeros[10:15]) via the
    // narrow write port. Xcrypto reads all 16 words in one cycle.
    //
    // Scratchpad address decode: data memory writes to address range
    // 0x0002_0000 - 0x0002_003F are routed to the scratchpad.
    // Word index = dmem_addr[5:2] (16 words x 4 bytes = 64 bytes).

    localparam logic [31:0] SCRATCH_BASE = 32'h0002_0000;
    localparam logic [31:0] SCRATCH_END  = 32'h0002_003F;

    logic scratch_wr_en;
    assign scratch_wr_en = dmem_req && (|dmem_we) &&
                           (dmem_addr >= SCRATCH_BASE) &&
                           (dmem_addr <= SCRATCH_END);

    // Bulk write interface (active when mining controller is wired up)
    logic [31:0] scratch_bulk_data [0:15];
    logic        scratch_bulk_en;

    // Default: no bulk write (mining controller will drive these when added)
    always_comb begin
        for (int i = 0; i < 16; i++) scratch_bulk_data[i] = 32'd0;
        scratch_bulk_en = 1'b0;
    end

    (* dont_touch = "true" *)
    xcrypto_scratchpad u_xc_scratch (
        .clk          (clk),
        .rst_n        (rst_n),
        .wr_idx       (dmem_addr[5:2]),
        .wr_data      (dmem_wdata),
        .wr_en        (scratch_wr_en),
        .bulk_wr_data (scratch_bulk_data),
        .bulk_wr_en   (scratch_bulk_en),
        .rd_en        (xc_mem_rd_en),
        .rd_data      (xc_mem_block),
        .rd_valid     (xc_mem_valid)
    );

    // =========================================================================
    // Xlattice unit response signals
    // =========================================================================
    logic        xl_resp_valid;
    logic [4:0]  xl_resp_rd_addr;
    logic [31:0] xl_resp_data;
    logic        xl_resp_wr_en;
    logic        xl_req_ready;

    // Xlattice 256-bit SRAM interface -- stubbed for FPGA prototype.
    // In production, connects to a dedicated 256-bit-wide scratchpad.
    logic [31:0]  xl_mem_rd_addr_a;
    logic         xl_mem_rd_en_a;
    logic [255:0] xl_mem_rd_data_a;
    logic         xl_mem_rd_valid_a;
    logic [31:0]  xl_mem_rd_addr_b;
    logic         xl_mem_rd_en_b;
    logic [255:0] xl_mem_rd_data_b;
    logic         xl_mem_rd_valid_b;
    logic [31:0]  xl_mem_wr_addr;
    logic         xl_mem_wr_en;
    logic [255:0] xl_mem_wr_data;

    // Stub: single-cycle valid, data = 0 (firmware uses register-based ops)
    logic xl_mem_rd_valid_a_r, xl_mem_rd_valid_b_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            xl_mem_rd_valid_a_r <= 1'b0;
            xl_mem_rd_valid_b_r <= 1'b0;
        end else begin
            xl_mem_rd_valid_a_r <= xl_mem_rd_en_a;
            xl_mem_rd_valid_b_r <= xl_mem_rd_en_b;
        end
    end

    assign xl_mem_rd_valid_a = xl_mem_rd_valid_a_r;
    assign xl_mem_rd_valid_b = xl_mem_rd_valid_b_r;
    assign xl_mem_rd_data_a  = 256'd0;
    assign xl_mem_rd_data_b  = 256'd0;

    // =========================================================================
    // Map xcrypto response back to core result interface
    // =========================================================================
    // The core expects a simple result/ready handshake.  When the xcrypto unit
    // signals resp_valid, we drive the core's xcrypto_result and xcrypto_ready.
    assign core_xcrypto_result = xc_resp_data;
    assign core_xcrypto_ready  = xc_req_ready | xc_resp_valid;

    // Same for xlattice
    assign core_xlattice_result = xl_resp_data;
    assign core_xlattice_ready  = xl_req_ready | xl_resp_valid;

    // =========================================================================
    // RISC-V Core
    // =========================================================================
    (* dont_touch = "true" *)
    qug_core u_core (
        .clk             (clk),
        .rst_n           (rst_n),

        // Instruction memory
        .imem_addr       (imem_addr),
        .imem_rdata      (imem_rdata),
        .imem_req        (imem_req),
        .imem_gnt        (imem_gnt),

        // Data memory
        .dmem_addr       (dmem_addr),
        .dmem_wdata      (dmem_wdata),
        .dmem_rdata      (dmem_rdata),
        .dmem_we         (dmem_we),
        .dmem_req        (dmem_req),
        .dmem_gnt        (dmem_gnt),

        // Xcrypto extension port
        .xcrypto_instr   (core_xcrypto_instr),
        .xcrypto_rs1     (core_xcrypto_rs1),
        .xcrypto_rs2     (core_xcrypto_rs2),
        .xcrypto_valid   (core_xcrypto_valid),
        .xcrypto_result  (core_xcrypto_result),
        .xcrypto_ready   (core_xcrypto_ready),

        // Xlattice extension port
        .xlattice_instr  (core_xlattice_instr),
        .xlattice_rs1    (core_xlattice_rs1),
        .xlattice_rs2    (core_xlattice_rs2),
        .xlattice_valid  (core_xlattice_valid),
        .xlattice_result (core_xlattice_result),
        .xlattice_ready  (core_xlattice_ready)
    );

    // =========================================================================
    // Xcrypto Extension Unit (BLAKE3 hardware pipeline)
    // =========================================================================
    (* dont_touch = "true" *)
    xcrypto_unit u_xcrypto (
        .clk           (clk),
        .rst_n         (rst_n),

        .req_valid     (core_xcrypto_valid),
        .req_ready     (xc_req_ready),
        .req_funct7    (xc_funct7),
        .req_funct3    (xc_funct3),
        .req_rs1       (core_xcrypto_rs1),
        .req_rs2       (core_xcrypto_rs2),
        .req_rd_addr   (xc_rd_addr),

        .resp_valid    (xc_resp_valid),
        .resp_rd_addr  (xc_resp_rd_addr),
        .resp_data     (xc_resp_data),
        .resp_wr_en    (xc_resp_wr_en),

        .mem_addr      (xc_mem_addr),
        .mem_rd_en     (xc_mem_rd_en),
        .mem_block     (xc_mem_block),
        .mem_valid     (xc_mem_valid)
    );

    // =========================================================================
    // Xlattice Extension Unit (256-bit field arithmetic)
    // =========================================================================
    (* dont_touch = "true" *)
    xlattice_unit u_xlattice (
        .clk             (clk),
        .rst_n           (rst_n),

        .req_valid       (core_xlattice_valid),
        .req_ready       (xl_req_ready),
        .req_funct7      (xl_funct7),
        .req_funct3      (xl_funct3),
        .req_rs1         (core_xlattice_rs1),
        .req_rs2         (core_xlattice_rs2),
        .req_rd_addr     (xl_rd_addr),

        .resp_valid      (xl_resp_valid),
        .resp_rd_addr    (xl_resp_rd_addr),
        .resp_data       (xl_resp_data),
        .resp_wr_en      (xl_resp_wr_en),

        .mem_rd_addr_a   (xl_mem_rd_addr_a),
        .mem_rd_en_a     (xl_mem_rd_en_a),
        .mem_rd_data_a   (xl_mem_rd_data_a),
        .mem_rd_valid_a  (xl_mem_rd_valid_a),

        .mem_rd_addr_b   (xl_mem_rd_addr_b),
        .mem_rd_en_b     (xl_mem_rd_en_b),
        .mem_rd_data_b   (xl_mem_rd_data_b),
        .mem_rd_valid_b  (xl_mem_rd_valid_b),

        .mem_wr_addr     (xl_mem_wr_addr),
        .mem_wr_en       (xl_mem_wr_en),
        .mem_wr_data     (xl_mem_wr_data)
    );

endmodule : qug_tile
