// =============================================================================
// QUG-V1 Mining SoC - 7-Stage Pipeline Controller
// =============================================================================
// 7-stage in-order pipeline: IF -> ID -> EX1 -> EX2 -> MEM -> WB1 -> WB2
//
// Features:
//   - Full data forwarding from EX1, EX2, MEM, WB1, WB2 back to ID
//   - Load-use hazard detection with 1-cycle stall
//   - Branch misprediction flush (predict not-taken)
//   - Extension stall (hold pipeline while Xcrypto/Xlattice busy)
//   - PC management: sequential, branch, jump
//
// Target: Kintex-7 @ 100 MHz
// =============================================================================

module qug_pipeline
  import qug_core_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,

  // Instruction memory interface
  output logic [31:0] imem_addr,
  input  logic [31:0] imem_rdata,
  output logic        imem_req,
  input  logic        imem_gnt,

  // Data memory interface
  output logic [31:0] dmem_addr,
  output logic [31:0] dmem_wdata,
  input  logic [31:0] dmem_rdata,
  output logic [3:0]  dmem_we,       // byte-lane write enables
  output logic        dmem_req,
  input  logic        dmem_gnt,

  // Extension interface: Xcrypto
  output ext_request_t  xcrypto_req,
  input  ext_response_t xcrypto_resp,

  // Extension interface: Xlattice
  output ext_request_t  xlattice_req,
  input  ext_response_t xlattice_resp
);

  // =========================================================================
  // Pipeline registers
  // =========================================================================
  if_id_reg_t   if_id_r,   if_id_next;
  id_ex1_reg_t  id_ex1_r,  id_ex1_next;
  ex1_ex2_reg_t ex1_ex2_r, ex1_ex2_next;
  ex2_mem_reg_t ex2_mem_r, ex2_mem_next;
  mem_wb1_reg_t mem_wb1_r, mem_wb1_next;
  wb1_wb2_reg_t wb1_wb2_r, wb1_wb2_next;

  // =========================================================================
  // PC register
  // =========================================================================
  logic [31:0] pc_r, pc_next;

  // =========================================================================
  // Hazard / stall / flush signals
  // =========================================================================
  logic stall_if, stall_id;
  logic flush_if, flush_id, flush_ex1;
  logic load_use_hazard;
  logic ext_stall;
  logic branch_taken;
  logic [31:0] branch_target;

  // =========================================================================
  // Register file interface
  // =========================================================================
  logic [4:0]  rf_rs1_addr, rf_rs2_addr;
  logic [31:0] rf_rs1_data, rf_rs2_data;
  logic [4:0]  rf_rd_addr;
  logic [31:0] rf_rd_data;
  logic        rf_rd_we;

  // =========================================================================
  // Decoder interface
  // =========================================================================
  decoded_ctrl_t dec_ctrl;
  logic [31:0]   dec_instr_expanded;
  logic          dec_is_compressed;

  // =========================================================================
  // ALU interface
  // =========================================================================
  logic [31:0] alu_operand_a, alu_operand_b;
  logic [31:0] alu_result;
  // ALU flags — currently unused, reserved for branch optimization
  // verilator lint_off UNUSED
  logic        alu_zero, alu_carry, alu_overflow;
  // verilator lint_on UNUSED

  // =========================================================================
  // Forwarded operand values (resolved in ID stage)
  // =========================================================================
  logic [31:0] fwd_rs1_data, fwd_rs2_data;

  // =========================================================================
  // Instantiate sub-modules
  // =========================================================================

  qug_regfile #(.XLEN(XLEN)) u_regfile (
    .clk      (clk),
    .rst_n    (rst_n),
    .rs1_addr (rf_rs1_addr),
    .rs1_data (rf_rs1_data),
    .rs2_addr (rf_rs2_addr),
    .rs2_data (rf_rs2_data),
    .rd_addr  (rf_rd_addr),
    .rd_data  (rf_rd_data),
    .rd_we    (rf_rd_we)
  );

  qug_decoder u_decoder (
    .instr_i          (if_id_r.instr),
    .pc_i             (if_id_r.pc),
    .ctrl_o           (dec_ctrl),
    .instr_expanded_o (dec_instr_expanded),
    .is_compressed_o  (dec_is_compressed)
  );

  qug_alu #(.XLEN(XLEN)) u_alu (
    .operand_a (alu_operand_a),
    .operand_b (alu_operand_b),
    .op        (id_ex1_r.ctrl.alu_op),
    .result    (alu_result),
    .zero      (alu_zero),
    .carry     (alu_carry),
    .overflow  (alu_overflow)
  );

  // =========================================================================
  // Stage 1: Instruction Fetch (IF)
  // =========================================================================
  assign imem_addr = pc_r;
  assign imem_req  = !stall_if;

  always_comb begin
    if (branch_taken)
      pc_next = branch_target;
    else if (stall_if)
      pc_next = pc_r;
    else if (dec_is_compressed && if_id_r.valid)
      pc_next = pc_r + 32'd2;
    else
      pc_next = pc_r + 32'd4;
  end

  always_comb begin
    if_id_next.pc    = pc_r;
    if_id_next.instr = imem_rdata;
    if_id_next.valid = imem_gnt && !flush_if;
  end

  // =========================================================================
  // Stage 2: Instruction Decode (ID) + Register Read + Forwarding
  // =========================================================================
  assign rf_rs1_addr = dec_ctrl.rs1_addr;
  assign rf_rs2_addr = dec_ctrl.rs2_addr;

  // ----- Data forwarding network -----
  // Priority: EX1 > EX2 > MEM > WB1 > WB2 (youngest first)
  always_comb begin
    // RS1 forwarding
    fwd_rs1_data = rf_rs1_data;
    if (dec_ctrl.rs1_addr == 5'd0) begin
      fwd_rs1_data = 32'd0;
    end else if (id_ex1_r.valid && id_ex1_r.ctrl.reg_write &&
                 id_ex1_r.ctrl.rd_addr == dec_ctrl.rs1_addr) begin
      // Cannot forward from EX1 if it's a load (load-use hazard handled separately)
      if (!id_ex1_r.ctrl.mem_read)
        fwd_rs1_data = alu_result; // forward from EX1 combinationally
    end else if (ex1_ex2_r.valid && ex1_ex2_r.ctrl.reg_write &&
                 ex1_ex2_r.ctrl.rd_addr == dec_ctrl.rs1_addr) begin
      fwd_rs1_data = ex1_ex2_r.alu_result;
    end else if (ex2_mem_r.valid && ex2_mem_r.ctrl.reg_write &&
                 ex2_mem_r.ctrl.rd_addr == dec_ctrl.rs1_addr) begin
      fwd_rs1_data = ex2_mem_r.alu_result;
    end else if (mem_wb1_r.valid && mem_wb1_r.ctrl.reg_write &&
                 mem_wb1_r.ctrl.rd_addr == dec_ctrl.rs1_addr) begin
      fwd_rs1_data = mem_wb1_r.result;
    end else if (wb1_wb2_r.valid && wb1_wb2_r.reg_write &&
                 wb1_wb2_r.rd_addr == dec_ctrl.rs1_addr) begin
      fwd_rs1_data = wb1_wb2_r.result;
    end
  end

  always_comb begin
    // RS2 forwarding (identical structure)
    fwd_rs2_data = rf_rs2_data;
    if (dec_ctrl.rs2_addr == 5'd0) begin
      fwd_rs2_data = 32'd0;
    end else if (id_ex1_r.valid && id_ex1_r.ctrl.reg_write &&
                 id_ex1_r.ctrl.rd_addr == dec_ctrl.rs2_addr) begin
      if (!id_ex1_r.ctrl.mem_read)
        fwd_rs2_data = alu_result;
    end else if (ex1_ex2_r.valid && ex1_ex2_r.ctrl.reg_write &&
                 ex1_ex2_r.ctrl.rd_addr == dec_ctrl.rs2_addr) begin
      fwd_rs2_data = ex1_ex2_r.alu_result;
    end else if (ex2_mem_r.valid && ex2_mem_r.ctrl.reg_write &&
                 ex2_mem_r.ctrl.rd_addr == dec_ctrl.rs2_addr) begin
      fwd_rs2_data = ex2_mem_r.alu_result;
    end else if (mem_wb1_r.valid && mem_wb1_r.ctrl.reg_write &&
                 mem_wb1_r.ctrl.rd_addr == dec_ctrl.rs2_addr) begin
      fwd_rs2_data = mem_wb1_r.result;
    end else if (wb1_wb2_r.valid && wb1_wb2_r.reg_write &&
                 wb1_wb2_r.rd_addr == dec_ctrl.rs2_addr) begin
      fwd_rs2_data = wb1_wb2_r.result;
    end
  end

  // ----- Load-use hazard detection -----
  // If EX1 has a load and ID needs that register, stall 1 cycle
  assign load_use_hazard = id_ex1_r.valid && id_ex1_r.ctrl.mem_read &&
                           ((id_ex1_r.ctrl.rd_addr == dec_ctrl.rs1_addr && dec_ctrl.rs1_addr != 5'd0) ||
                            (id_ex1_r.ctrl.rd_addr == dec_ctrl.rs2_addr && dec_ctrl.rs2_addr != 5'd0));

  // ----- Extension stall -----
  // Stall while extension unit is busy (valid request, no ready)
  assign ext_stall = (id_ex1_r.valid && id_ex1_r.ctrl.ext_xcrypto  && !xcrypto_resp.ready) ||
                     (id_ex1_r.valid && id_ex1_r.ctrl.ext_xlattice && !xlattice_resp.ready);

  // ----- Stall/flush control -----
  assign stall_id = load_use_hazard || ext_stall;
  assign stall_if = stall_id;  // back-pressure to IF

  assign flush_ex1 = branch_taken;
  assign flush_id  = branch_taken;
  assign flush_if  = branch_taken;

  // ----- ID -> EX1 register -----
  always_comb begin
    id_ex1_next.pc       = if_id_r.pc;
    id_ex1_next.rs1_data = fwd_rs1_data;
    id_ex1_next.rs2_data = fwd_rs2_data;
    id_ex1_next.ctrl     = dec_ctrl;
    id_ex1_next.valid    = if_id_r.valid && !flush_id && !stall_id;
  end

  // =========================================================================
  // Stage 3: Execute 1 (EX1) - ALU operation + branch resolution
  // =========================================================================

  // ALU operand A mux
  always_comb begin
    if (id_ex1_r.ctrl.auipc)
      alu_operand_a = id_ex1_r.pc;
    else if (id_ex1_r.ctrl.lui)
      alu_operand_a = id_ex1_r.ctrl.immediate;
    else
      alu_operand_a = id_ex1_r.rs1_data;
  end

  // ALU operand B mux
  always_comb begin
    if (id_ex1_r.ctrl.alu_src_imm || id_ex1_r.ctrl.auipc)
      alu_operand_b = id_ex1_r.ctrl.immediate;
    else
      alu_operand_b = id_ex1_r.rs2_data;
  end

  // ----- Branch resolution (predict not-taken) -----
  always_comb begin
    branch_taken  = 1'b0;
    branch_target = 32'd0;

    if (id_ex1_r.valid) begin
      unique case (id_ex1_r.ctrl.branch_type)
        BR_EQ:  branch_taken = (id_ex1_r.rs1_data == id_ex1_r.rs2_data);
        BR_NE:  branch_taken = (id_ex1_r.rs1_data != id_ex1_r.rs2_data);
        BR_LT:  branch_taken = ($signed(id_ex1_r.rs1_data) <  $signed(id_ex1_r.rs2_data));
        BR_GE:  branch_taken = ($signed(id_ex1_r.rs1_data) >= $signed(id_ex1_r.rs2_data));
        BR_LTU: branch_taken = (id_ex1_r.rs1_data <  id_ex1_r.rs2_data);
        BR_GEU: branch_taken = (id_ex1_r.rs1_data >= id_ex1_r.rs2_data);
        BR_JAL: branch_taken = 1'b1;
        BR_NONE: branch_taken = 1'b0;
      endcase

      // Branch/jump target
      if (id_ex1_r.ctrl.jalr)
        branch_target = (id_ex1_r.rs1_data + id_ex1_r.ctrl.immediate) & ~32'd1;
      else
        branch_target = id_ex1_r.pc + id_ex1_r.ctrl.immediate;
    end
  end

  // Extension request dispatch
  always_comb begin
    xcrypto_req.instruction = id_ex1_r.ctrl.immediate; // full instruction in immediate field
    xcrypto_req.rs1_data    = id_ex1_r.rs1_data;
    xcrypto_req.rs2_data    = id_ex1_r.rs2_data;
    xcrypto_req.valid       = id_ex1_r.valid && id_ex1_r.ctrl.ext_xcrypto;

    xlattice_req.instruction = id_ex1_r.ctrl.immediate;
    xlattice_req.rs1_data    = id_ex1_r.rs1_data;
    xlattice_req.rs2_data    = id_ex1_r.rs2_data;
    xlattice_req.valid       = id_ex1_r.valid && id_ex1_r.ctrl.ext_xlattice;
  end

  // EX1 -> EX2 register
  always_comb begin
    ex1_ex2_next.pc         = id_ex1_r.pc;
    ex1_ex2_next.rs2_data   = id_ex1_r.rs2_data;
    ex1_ex2_next.ctrl       = id_ex1_r.ctrl;
    ex1_ex2_next.valid      = id_ex1_r.valid && !flush_ex1;

    // Result mux: ALU, extension, or PC+4 for JAL/JALR
    if (id_ex1_r.ctrl.jal || id_ex1_r.ctrl.jalr)
      ex1_ex2_next.alu_result = id_ex1_r.pc + 32'd4;
    else if (id_ex1_r.ctrl.ext_xcrypto && xcrypto_resp.ready)
      ex1_ex2_next.alu_result = xcrypto_resp.result;
    else if (id_ex1_r.ctrl.ext_xlattice && xlattice_resp.ready)
      ex1_ex2_next.alu_result = xlattice_resp.result;
    else
      ex1_ex2_next.alu_result = alu_result;
  end

  // =========================================================================
  // Stage 4: Execute 2 (EX2) - Multi-cycle completion / passthrough
  // =========================================================================
  // For single-cycle ops this is a simple passthrough.
  // M-extension divide could use this stage for iterative completion.
  always_comb begin
    ex2_mem_next.pc         = ex1_ex2_r.pc;
    ex2_mem_next.alu_result = ex1_ex2_r.alu_result;
    ex2_mem_next.rs2_data   = ex1_ex2_r.rs2_data;
    ex2_mem_next.ctrl       = ex1_ex2_r.ctrl;
    ex2_mem_next.valid      = ex1_ex2_r.valid;
  end

  // =========================================================================
  // Stage 5: Memory Access (MEM)
  // =========================================================================
  assign dmem_addr = ex2_mem_r.alu_result;
  assign dmem_req  = ex2_mem_r.valid && (ex2_mem_r.ctrl.mem_read || ex2_mem_r.ctrl.mem_write);

  // Write data with byte-lane steering
  always_comb begin
    dmem_wdata = 32'd0;
    dmem_we    = 4'b0000;

    if (ex2_mem_r.valid && ex2_mem_r.ctrl.mem_write) begin
      unique case (ex2_mem_r.ctrl.mem_width)
        MEM_BYTE: begin
          dmem_wdata = {4{ex2_mem_r.rs2_data[7:0]}};
          unique case (ex2_mem_r.alu_result[1:0])
            2'b00: dmem_we = 4'b0001;
            2'b01: dmem_we = 4'b0010;
            2'b10: dmem_we = 4'b0100;
            2'b11: dmem_we = 4'b1000;
          endcase
        end
        MEM_HALF: begin
          dmem_wdata = {2{ex2_mem_r.rs2_data[15:0]}};
          dmem_we    = ex2_mem_r.alu_result[1] ? 4'b1100 : 4'b0011;
        end
        MEM_WORD: begin
          dmem_wdata = ex2_mem_r.rs2_data;
          dmem_we    = 4'b1111;
        end
        default: begin
          dmem_wdata = 32'd0;
          dmem_we    = 4'b0000;
        end
      endcase
    end
  end

  // Load data alignment and sign extension
  logic [31:0] mem_load_data;

  always_comb begin
    mem_load_data = 32'd0;

    if (ex2_mem_r.ctrl.mem_read) begin
      unique case (ex2_mem_r.ctrl.mem_width)
        MEM_BYTE: begin
          logic [7:0] byte_val;
          unique case (ex2_mem_r.alu_result[1:0])
            2'b00: byte_val = dmem_rdata[7:0];
            2'b01: byte_val = dmem_rdata[15:8];
            2'b10: byte_val = dmem_rdata[23:16];
            2'b11: byte_val = dmem_rdata[31:24];
          endcase
          mem_load_data = ex2_mem_r.ctrl.mem_signed ?
                          {{24{byte_val[7]}}, byte_val} :
                          {24'b0, byte_val};
        end
        MEM_HALF: begin
          logic [15:0] half_val;
          half_val = ex2_mem_r.alu_result[1] ? dmem_rdata[31:16] : dmem_rdata[15:0];
          mem_load_data = ex2_mem_r.ctrl.mem_signed ?
                          {{16{half_val[15]}}, half_val} :
                          {16'b0, half_val};
        end
        MEM_WORD: begin
          mem_load_data = dmem_rdata;
        end
        default: mem_load_data = 32'd0;
      endcase
    end
  end

  // MEM -> WB1 register
  always_comb begin
    mem_wb1_next.pc   = ex2_mem_r.pc;
    mem_wb1_next.ctrl = ex2_mem_r.ctrl;
    mem_wb1_next.valid = ex2_mem_r.valid;

    // Select between ALU result and memory load
    if (ex2_mem_r.ctrl.mem_read)
      mem_wb1_next.result = mem_load_data;
    else
      mem_wb1_next.result = ex2_mem_r.alu_result;
  end

  // =========================================================================
  // Stage 6: Write-Back 1 (WB1)
  // =========================================================================
  // WB1 -> WB2: propagate for late forwarding / register write
  always_comb begin
    wb1_wb2_next.result    = mem_wb1_r.result;
    wb1_wb2_next.rd_addr   = mem_wb1_r.ctrl.rd_addr;
    wb1_wb2_next.reg_write = mem_wb1_r.ctrl.reg_write;
    wb1_wb2_next.valid     = mem_wb1_r.valid;
  end

  // =========================================================================
  // Stage 7: Write-Back 2 (WB2) - Commit to register file
  // =========================================================================
  assign rf_rd_addr = wb1_wb2_r.rd_addr;
  assign rf_rd_data = wb1_wb2_r.result;
  assign rf_rd_we   = wb1_wb2_r.valid && wb1_wb2_r.reg_write;

  // =========================================================================
  // Sequential pipeline register updates
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pc_r      <= 32'h0000_0000;  // reset vector
      if_id_r   <= '0;
      id_ex1_r  <= '0;
      ex1_ex2_r <= '0;
      ex2_mem_r <= '0;
      mem_wb1_r <= '0;
      wb1_wb2_r <= '0;
    end else begin
      // PC update
      pc_r <= pc_next;

      // IF/ID
      if (!stall_if)
        if_id_r <= if_id_next;
      else if (flush_if)
        if_id_r.valid <= 1'b0;

      // ID/EX1
      if (!stall_id)
        id_ex1_r <= id_ex1_next;
      else begin
        // Insert bubble on stall
        id_ex1_r.valid <= 1'b0;
      end

      // EX1/EX2 (no stall, but can flush)
      if (flush_ex1)
        ex1_ex2_r.valid <= 1'b0;
      else if (!ext_stall)
        ex1_ex2_r <= ex1_ex2_next;

      // EX2/MEM
      ex2_mem_r <= ex2_mem_next;

      // MEM/WB1
      mem_wb1_r <= mem_wb1_next;

      // WB1/WB2
      wb1_wb2_r <= wb1_wb2_next;
    end
  end

endmodule : qug_pipeline
