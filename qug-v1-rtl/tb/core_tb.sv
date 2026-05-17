// =============================================================================
// QUG-V1 Mining SoC - Core Testbench
// =============================================================================
// Functional verification of the RV32IMC pipeline:
//   - ADD, SUB, AND, OR
//   - Branch (BEQ, BNE)
//   - Load/Store (LW, SW)
//   - Custom-0 (Xcrypto) dispatch verification
//
// Uses simple BRAM memory models for instruction and data memory.
// Self-checking with pass/fail status.
// =============================================================================

`timescale 1ns / 1ps

module core_tb;

  import qug_pkg::*;

  // -------------------------------------------------------------------------
  // Clock and reset
  // -------------------------------------------------------------------------
  logic clk;
  logic rst_n;

  initial clk = 1'b0;
  always #5 clk = ~clk;  // 100 MHz

  // -------------------------------------------------------------------------
  // Memory models
  // -------------------------------------------------------------------------
  localparam int IMEM_DEPTH = 1024;  // 4 KB instruction memory
  localparam int DMEM_DEPTH = 1024;  // 4 KB data memory

  logic [31:0] imem [0:IMEM_DEPTH-1];
  logic [31:0] dmem [0:DMEM_DEPTH-1];

  // -------------------------------------------------------------------------
  // DUT signals
  // -------------------------------------------------------------------------
  logic [31:0] imem_addr,  dmem_addr;
  logic [31:0] imem_rdata, dmem_rdata, dmem_wdata;
  logic        imem_req,   dmem_req;
  logic        imem_gnt,   dmem_gnt;
  logic [3:0]  dmem_we;

  // Xcrypto extension
  logic [31:0] xcrypto_instr, xcrypto_rs1, xcrypto_rs2, xcrypto_result;
  logic        xcrypto_valid, xcrypto_ready;

  // Xlattice extension
  logic [31:0] xlattice_instr, xlattice_rs1, xlattice_rs2, xlattice_result;
  logic        xlattice_valid, xlattice_ready;

  // -------------------------------------------------------------------------
  // DUT instantiation
  // -------------------------------------------------------------------------
  qug_core u_dut (
    .clk             (clk),
    .rst_n           (rst_n),

    .imem_addr       (imem_addr),
    .imem_rdata      (imem_rdata),
    .imem_req        (imem_req),
    .imem_gnt        (imem_gnt),

    .dmem_addr       (dmem_addr),
    .dmem_wdata      (dmem_wdata),
    .dmem_rdata      (dmem_rdata),
    .dmem_we         (dmem_we),
    .dmem_req        (dmem_req),
    .dmem_gnt        (dmem_gnt),

    .xcrypto_instr   (xcrypto_instr),
    .xcrypto_rs1     (xcrypto_rs1),
    .xcrypto_rs2     (xcrypto_rs2),
    .xcrypto_valid   (xcrypto_valid),
    .xcrypto_result  (xcrypto_result),
    .xcrypto_ready   (xcrypto_ready),

    .xlattice_instr  (xlattice_instr),
    .xlattice_rs1    (xlattice_rs1),
    .xlattice_rs2    (xlattice_rs2),
    .xlattice_valid  (xlattice_valid),
    .xlattice_result (xlattice_result),
    .xlattice_ready  (xlattice_ready)
  );

  // -------------------------------------------------------------------------
  // Instruction memory model (single-cycle, word-aligned)
  // -------------------------------------------------------------------------
  assign imem_gnt   = imem_req;
  assign imem_rdata = imem[imem_addr[31:2] % IMEM_DEPTH];

  // -------------------------------------------------------------------------
  // Data memory model (single-cycle, byte-lane write)
  // -------------------------------------------------------------------------
  assign dmem_gnt   = dmem_req;
  assign dmem_rdata = dmem[dmem_addr[31:2] % DMEM_DEPTH];

  always_ff @(posedge clk) begin
    if (dmem_req) begin
      if (dmem_we[0]) dmem[dmem_addr[31:2] % DMEM_DEPTH][ 7: 0] <= dmem_wdata[ 7: 0];
      if (dmem_we[1]) dmem[dmem_addr[31:2] % DMEM_DEPTH][15: 8] <= dmem_wdata[15: 8];
      if (dmem_we[2]) dmem[dmem_addr[31:2] % DMEM_DEPTH][23:16] <= dmem_wdata[23:16];
      if (dmem_we[3]) dmem[dmem_addr[31:2] % DMEM_DEPTH][31:24] <= dmem_wdata[31:24];
    end
  end

  // -------------------------------------------------------------------------
  // Stub extension units (Xcrypto returns rs1 XOR rs2, Xlattice returns 0)
  // -------------------------------------------------------------------------
  assign xcrypto_result = xcrypto_rs1 ^ xcrypto_rs2;
  assign xcrypto_ready  = xcrypto_valid;  // single-cycle response

  assign xlattice_result = 32'hDEAD_BEEF;
  assign xlattice_ready  = xlattice_valid;

  // -------------------------------------------------------------------------
  // RV32I instruction encoders (for readability)
  // -------------------------------------------------------------------------
  function automatic logic [31:0] rv_addi(input logic [4:0] rd, rs1,
                                          input logic [11:0] imm);
    return {imm, rs1, 3'b000, rd, 7'b0010011};
  endfunction

  function automatic logic [31:0] rv_add(input logic [4:0] rd, rs1, rs2);
    return {7'b0000000, rs2, rs1, 3'b000, rd, 7'b0110011};
  endfunction

  function automatic logic [31:0] rv_sub(input logic [4:0] rd, rs1, rs2);
    return {7'b0100000, rs2, rs1, 3'b000, rd, 7'b0110011};
  endfunction

  function automatic logic [31:0] rv_and(input logic [4:0] rd, rs1, rs2);
    return {7'b0000000, rs2, rs1, 3'b111, rd, 7'b0110011};
  endfunction

  function automatic logic [31:0] rv_or(input logic [4:0] rd, rs1, rs2);
    return {7'b0000000, rs2, rs1, 3'b110, rd, 7'b0110011};
  endfunction

  function automatic logic [31:0] rv_sw(input logic [4:0] rs2, rs1,
                                        input logic [11:0] offset);
    return {offset[11:5], rs2, rs1, 3'b010, offset[4:0], 7'b0100011};
  endfunction

  function automatic logic [31:0] rv_lw(input logic [4:0] rd, rs1,
                                        input logic [11:0] offset);
    return {offset, rs1, 3'b010, rd, 7'b0000011};
  endfunction

  // BEQ: B-type encoding
  function automatic logic [31:0] rv_beq(input logic [4:0] rs1, rs2,
                                         input logic [12:0] offset);
    return {offset[12], offset[10:5], rs2, rs1, 3'b000,
            offset[4:1], offset[11], 7'b1100011};
  endfunction

  // BNE: B-type encoding
  function automatic logic [31:0] rv_bne(input logic [4:0] rs1, rs2,
                                         input logic [12:0] offset);
    return {offset[12], offset[10:5], rs2, rs1, 3'b001,
            offset[4:1], offset[11], 7'b1100011};
  endfunction

  // Custom-0 R-type (Xcrypto): opcode = 0x0B
  function automatic logic [31:0] rv_custom0(input logic [4:0] rd, rs1, rs2,
                                             input logic [6:0] funct7);
    return {funct7, rs2, rs1, 3'b000, rd, 7'b0001011};
  endfunction

  // NOP: addi x0, x0, 0
  function automatic logic [31:0] rv_nop();
    return rv_addi(5'd0, 5'd0, 12'd0);
  endfunction

  // -------------------------------------------------------------------------
  // Test program
  // -------------------------------------------------------------------------
  //
  // Test 1: ADDI / ADD / SUB
  //   x1 = 10
  //   x2 = 20
  //   x3 = x1 + x2  (expect 30)
  //   x4 = x2 - x1  (expect 10)
  //
  // Test 2: AND / OR
  //   x5 = x1 & x2  (10 & 20 = 0)
  //   x6 = x1 | x2  (10 | 20 = 30)
  //
  // Test 3: SW / LW
  //   SW x3, 0(x0)   -> dmem[0] = 30
  //   LW x7, 0(x0)   -> x7 = 30
  //
  // Test 4: BEQ (not taken), BNE (taken)
  //   BEQ x1, x2, +8   (10 != 20 -> not taken)
  //   ADDI x8, x0, 0x55  (should execute)
  //   BNE x1, x2, +8   (10 != 20 -> taken, skip next)
  //   ADDI x9, x0, 0xBB  (should be skipped)
  //   ADDI x10, x0, 0xCC  (branch target, should execute)
  //
  // Test 5: Custom-0 (Xcrypto)
  //   x11 = custom0(x1, x2)  -> expect x1 XOR x2 = 10 ^ 20 = 30
  //
  // End: infinite loop (JAL x0, 0)

  initial begin
    int idx;

    // Clear memories
    for (int i = 0; i < IMEM_DEPTH; i++) imem[i] = rv_nop();
    for (int i = 0; i < DMEM_DEPTH; i++) dmem[i] = 32'd0;

    idx = 0;

    // --- Test 1: Arithmetic ---
    imem[idx++] = rv_addi(5'd1, 5'd0, 12'd10);    // x1 = 10
    imem[idx++] = rv_addi(5'd2, 5'd0, 12'd20);    // x2 = 20
    imem[idx++] = rv_nop();                         // bubble for data path settling
    imem[idx++] = rv_add(5'd3, 5'd1, 5'd2);        // x3 = x1 + x2 = 30
    imem[idx++] = rv_sub(5'd4, 5'd2, 5'd1);        // x4 = x2 - x1 = 10

    // --- Test 2: Logic ---
    imem[idx++] = rv_and(5'd5, 5'd1, 5'd2);        // x5 = 10 & 20 = 0
    imem[idx++] = rv_or(5'd6, 5'd1, 5'd2);         // x6 = 10 | 20 = 30

    // --- Test 3: Store / Load ---
    imem[idx++] = rv_sw(5'd3, 5'd0, 12'd0);        // dmem[0] = x3 (30)
    imem[idx++] = rv_nop();                         // wait for store
    imem[idx++] = rv_nop();
    imem[idx++] = rv_lw(5'd7, 5'd0, 12'd0);        // x7 = dmem[0] (30)

    // --- Test 4: Branches ---
    // BEQ x1, x2, +8 (not taken: 10 != 20)
    imem[idx++] = rv_beq(5'd1, 5'd2, 13'd8);       // idx=11
    imem[idx++] = rv_addi(5'd8, 5'd0, 12'h55);     // x8 = 0x55 (should execute)
    // BNE x1, x2, +8 (taken: 10 != 20, skip next instr)
    imem[idx++] = rv_bne(5'd1, 5'd2, 13'd8);       // idx=13
    imem[idx++] = rv_addi(5'd9, 5'd0, 12'hBB);     // x9 = 0xBB (SKIPPED)
    imem[idx++] = rv_addi(5'd10, 5'd0, 12'hCC);    // x10 = 0xCC (branch target)

    // --- Test 5: Custom-0 (Xcrypto) ---
    imem[idx++] = rv_custom0(5'd11, 5'd1, 5'd2, 7'b0000000); // x11 = xcrypto(x1, x2)

    // Fill remaining with NOPs then infinite loop
    for (int i = idx; i < idx + 20; i++) imem[i] = rv_nop();
    // JAL x0, 0 (infinite loop at end)
    imem[idx + 20] = {12'b0, 5'd0, 3'b000, 5'd0, 7'b1101111}; // jal x0, 0 -> self-loop
  end

  // -------------------------------------------------------------------------
  // Test monitor
  // -------------------------------------------------------------------------
  int cycle_count;
  int pass_count;
  int fail_count;

  task automatic check_reg(
    input string name,
    input int    reg_idx,
    input logic [31:0] expected
  );
    logic [31:0] actual;
    // Access register file through hierarchy
    if (reg_idx == 0)
      actual = 32'd0;
    else
      actual = u_dut.u_pipeline.u_regfile.regs[reg_idx];

    if (actual === expected) begin
      $display("[PASS] %s: x%0d = 0x%08h (expected 0x%08h)", name, reg_idx, actual, expected);
      pass_count++;
    end else begin
      $display("[FAIL] %s: x%0d = 0x%08h (expected 0x%08h)", name, reg_idx, actual, expected);
      fail_count++;
    end
  endtask

  task automatic check_mem(
    input string name,
    input int    addr,
    input logic [31:0] expected
  );
    logic [31:0] actual;
    actual = dmem[addr >> 2];

    if (actual === expected) begin
      $display("[PASS] %s: dmem[0x%04h] = 0x%08h", name, addr, actual);
      pass_count++;
    end else begin
      $display("[FAIL] %s: dmem[0x%04h] = 0x%08h (expected 0x%08h)", name, addr, actual, expected);
      fail_count++;
    end
  endtask

  // -------------------------------------------------------------------------
  // Xcrypto dispatch monitor
  // -------------------------------------------------------------------------
  logic xcrypto_fired;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      xcrypto_fired <= 1'b0;
    else if (xcrypto_valid)
      xcrypto_fired <= 1'b1;
  end

  // -------------------------------------------------------------------------
  // Main test sequence
  // -------------------------------------------------------------------------
  initial begin
    $display("=============================================================");
    $display("  QUG-V1 RV32IMC Core Testbench");
    $display("  7-stage pipeline: IF-ID-EX1-EX2-MEM-WB1-WB2");
    $display("=============================================================");

    pass_count = 0;
    fail_count = 0;
    cycle_count = 0;

    // Apply reset
    rst_n = 1'b0;
    repeat (5) @(posedge clk);
    rst_n = 1'b1;

    $display("\n--- Reset released, pipeline running ---\n");

    // Let the pipeline run enough cycles for all instructions to retire.
    // 7-stage pipeline + ~20 instructions + bubbles/branches ~ 60 cycles is safe.
    repeat (80) begin
      @(posedge clk);
      cycle_count++;
    end

    $display("\n--- Checking results after %0d cycles ---\n", cycle_count);

    // Test 1: Arithmetic
    check_reg("ADDI x1=10",     1, 32'd10);
    check_reg("ADDI x2=20",     2, 32'd20);
    check_reg("ADD  x3=30",     3, 32'd30);
    check_reg("SUB  x4=10",     4, 32'd10);

    // Test 2: Logic
    check_reg("AND  x5=0",      5, 32'd0);    // 10 & 20 = 0 (0xA & 0x14 = 0)
    check_reg("OR   x6=30",     6, 32'd30);   // 10 | 20 = 30 (0xA | 0x14 = 0x1E)

    // Test 3: Load/Store
    check_mem("SW dmem[0]=30",  0, 32'd30);
    check_reg("LW x7=30",       7, 32'd30);

    // Test 4: Branches
    check_reg("BEQ not-taken x8=0x55",  8, 32'h55);
    check_reg("BNE taken, x9 skipped",  9, 32'd0);    // should NOT have been written
    check_reg("BNE target x10=0xCC",   10, 32'hCC);

    // Test 5: Xcrypto dispatch
    if (xcrypto_fired) begin
      $display("[PASS] Xcrypto extension: custom-0 instruction dispatched");
      pass_count++;
    end else begin
      $display("[FAIL] Xcrypto extension: custom-0 instruction was NOT dispatched");
      fail_count++;
    end
    // The Xcrypto stub returns rs1 ^ rs2 = 10 ^ 20 = 30
    check_reg("Xcrypto x11 = x1^x2 = 30", 11, 32'd30);

    // Summary
    $display("\n=============================================================");
    $display("  TEST SUMMARY: %0d passed, %0d failed", pass_count, fail_count);
    $display("=============================================================");

    if (fail_count == 0)
      $display("  >>> ALL TESTS PASSED <<<");
    else
      $display("  >>> FAILURES DETECTED <<<");

    $display("");
    $finish;
  end

  // -------------------------------------------------------------------------
  // Waveform dump (VCD)
  // -------------------------------------------------------------------------
  initial begin
    $dumpfile("qug_core_tb.vcd");
    $dumpvars(0, core_tb);
  end

  // -------------------------------------------------------------------------
  // Timeout watchdog
  // -------------------------------------------------------------------------
  initial begin
    #50000;
    $display("[ERROR] Simulation timeout at %0t ns", $time);
    $finish;
  end

endmodule : core_tb
