// =============================================================================
// QUG-V1 Mining SoC - Register File
// =============================================================================
// 32-entry x XLEN-bit register file.
//   - 2 asynchronous read ports (rs1, rs2) for LUTRAM inference
//   - 1 synchronous write port (rd)
//   - x0 hardwired to zero
//
// Target: Kintex-7 FPGA LUTRAM (distributed RAM)
// =============================================================================

module qug_regfile
  import qug_core_pkg::*;
#(
  parameter int XLEN       = 32,
  parameter int REG_ADDR_W = 5
)(
  input  logic                  clk,
  input  logic                  rst_n,

  // Read port 1 (rs1) - asynchronous
  input  logic [REG_ADDR_W-1:0] rs1_addr,
  output logic [XLEN-1:0]       rs1_data,

  // Read port 2 (rs2) - asynchronous
  input  logic [REG_ADDR_W-1:0] rs2_addr,
  output logic [XLEN-1:0]       rs2_data,

  // Write port (rd) - synchronous
  input  logic [REG_ADDR_W-1:0] rd_addr,
  input  logic [XLEN-1:0]       rd_data,
  input  logic                   rd_we
);

  // -------------------------------------------------------------------------
  // Register storage
  // Using (* ram_style = "distributed" *) for Xilinx LUTRAM inference
  // -------------------------------------------------------------------------
  (* ram_style = "distributed" *)
  logic [XLEN-1:0] regs [1:2**REG_ADDR_W-1];  // x1..x31 (x0 is implicit zero)

  // -------------------------------------------------------------------------
  // Asynchronous reads with x0 hardwired to zero
  // Write-through: if reading the same register being written this cycle,
  // return the new value (avoids 1-cycle stale read).
  // -------------------------------------------------------------------------
  always_comb begin
    if (rs1_addr == '0) begin
      rs1_data = '0;
    end else if (rd_we && (rs1_addr == rd_addr)) begin
      rs1_data = rd_data;  // write-through bypass
    end else begin
      rs1_data = regs[rs1_addr];
    end
  end

  always_comb begin
    if (rs2_addr == '0) begin
      rs2_data = '0;
    end else if (rd_we && (rs2_addr == rd_addr)) begin
      rs2_data = rd_data;  // write-through bypass
    end else begin
      rs2_data = regs[rs2_addr];
    end
  end

  // -------------------------------------------------------------------------
  // Synchronous write (x0 writes are silently discarded)
  // -------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 1; i < 2**REG_ADDR_W; i++) begin
        regs[i] <= '0;
      end
    end else if (rd_we && (rd_addr != '0)) begin
      regs[rd_addr] <= rd_data;
    end
  end

  // -------------------------------------------------------------------------
  // Assertions (simulation only)
  // -------------------------------------------------------------------------
  // synthesis translate_off
  always_comb begin
    assert (rs1_addr == '0 || rs1_data !== 'x)
      else $warning("qug_regfile: rs1 read X from x%0d", rs1_addr);
  end
  // synthesis translate_on

endmodule : qug_regfile
