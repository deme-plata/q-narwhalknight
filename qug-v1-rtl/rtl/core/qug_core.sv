// =============================================================================
// QUG-V1 Mining SoC - Top-Level Core
// =============================================================================
// RV32IMC in-order 7-stage pipeline with Xcrypto/Xlattice extension interfaces.
//
// Instantiates: register file, decoder, ALU, pipeline controller
// Exposes:
//   - Instruction memory interface (Harvard architecture)
//   - Data memory interface with byte-lane enables
//   - Xcrypto extension port (custom-0, opcode 0x0B)
//   - Xlattice extension port (custom-1, opcode 0x2B)
//
// Target: Kintex-7 @ 100 MHz
// =============================================================================

module qug_core
  import qug_core_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,

  // -------------------------------------------------------------------
  // Instruction memory interface (read-only, word-aligned)
  // -------------------------------------------------------------------
  output logic [31:0] imem_addr,
  input  logic [31:0] imem_rdata,
  output logic        imem_req,
  input  logic        imem_gnt,

  // -------------------------------------------------------------------
  // Data memory interface (read/write, byte-lane enables)
  // -------------------------------------------------------------------
  output logic [31:0] dmem_addr,
  output logic [31:0] dmem_wdata,
  input  logic [31:0] dmem_rdata,
  output logic [3:0]  dmem_we,
  output logic        dmem_req,
  input  logic        dmem_gnt,

  // -------------------------------------------------------------------
  // Xcrypto extension interface (custom-0, opcode 0x0B)
  // -------------------------------------------------------------------
  output logic [31:0] xcrypto_instr,
  output logic [31:0] xcrypto_rs1,
  output logic [31:0] xcrypto_rs2,
  output logic        xcrypto_valid,
  input  logic [31:0] xcrypto_result,
  input  logic        xcrypto_ready,

  // -------------------------------------------------------------------
  // Xlattice extension interface (custom-1, opcode 0x2B)
  // -------------------------------------------------------------------
  output logic [31:0] xlattice_instr,
  output logic [31:0] xlattice_rs1,
  output logic [31:0] xlattice_rs2,
  output logic        xlattice_valid,
  input  logic [31:0] xlattice_result,
  input  logic        xlattice_ready
);

  // -------------------------------------------------------------------
  // Internal extension bus wiring
  // -------------------------------------------------------------------
  ext_request_t  xcrypto_req_w;
  ext_response_t xcrypto_resp_w;
  ext_request_t  xlattice_req_w;
  ext_response_t xlattice_resp_w;

  // Map internal packed structs to flat top-level ports
  assign xcrypto_instr = xcrypto_req_w.instruction;
  assign xcrypto_rs1   = xcrypto_req_w.rs1_data;
  assign xcrypto_rs2   = xcrypto_req_w.rs2_data;
  assign xcrypto_valid = xcrypto_req_w.valid;
  assign xcrypto_resp_w.result = xcrypto_result;
  assign xcrypto_resp_w.ready  = xcrypto_ready;

  assign xlattice_instr = xlattice_req_w.instruction;
  assign xlattice_rs1   = xlattice_req_w.rs1_data;
  assign xlattice_rs2   = xlattice_req_w.rs2_data;
  assign xlattice_valid = xlattice_req_w.valid;
  assign xlattice_resp_w.result = xlattice_result;
  assign xlattice_resp_w.ready  = xlattice_ready;

  // -------------------------------------------------------------------
  // Pipeline (contains regfile, decoder, ALU internally)
  // -------------------------------------------------------------------
  qug_pipeline u_pipeline (
    .clk           (clk),
    .rst_n         (rst_n),

    // Instruction memory
    .imem_addr     (imem_addr),
    .imem_rdata    (imem_rdata),
    .imem_req      (imem_req),
    .imem_gnt      (imem_gnt),

    // Data memory
    .dmem_addr     (dmem_addr),
    .dmem_wdata    (dmem_wdata),
    .dmem_rdata    (dmem_rdata),
    .dmem_we       (dmem_we),
    .dmem_req      (dmem_req),
    .dmem_gnt      (dmem_gnt),

    // Extensions
    .xcrypto_req   (xcrypto_req_w),
    .xcrypto_resp  (xcrypto_resp_w),
    .xlattice_req  (xlattice_req_w),
    .xlattice_resp (xlattice_resp_w)
  );

endmodule : qug_core
