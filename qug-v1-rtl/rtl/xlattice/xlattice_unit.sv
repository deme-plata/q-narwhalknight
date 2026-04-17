// =============================================================================
// xlattice_unit.sv -- Top-Level Xlattice Extension Unit for RISC-V Core
// QUG-V1 Mining SoC -- Genus-2 VDF Field Arithmetic Coprocessor
// =============================================================================
// Project  : QUG-V1 RISC-V Mining SoC
// Target   : Xilinx Kintex-7 XC7K325T (FPGA prototype)
// Author   : Quillon Foundation / Dragon Ball Miner
// License  : MIT
// =============================================================================
//
// Interfaces with the RISC-V core pipeline via the custom-1 opcode (0x2B).
// Decodes funct7 field to select Xlattice operations.
//
// Xlattice ISA instructions (R-type encoding, opcode = 7'b0101011):
//   funct7 = 0: ntt.fwd      -- Forward NTT (Phase 1B stub, returns 0)
//   funct7 = 1: ntt.inv      -- Inverse NTT (Phase 1B stub, returns 0)
//   funct7 = 2: poly.add     -- 256-bit modular addition (1 cycle)
//   funct7 = 3: poly.mul     -- 256-bit modular multiplication (12 cycles)
//   funct7 = 4: poly.reduce  -- Barrett reduction (Phase 1B stub, returns 0)
//
// R-type encoding: [funct7 | rs2 | rs1 | funct3 | rd | opcode]
//   rs1: source register 1 (operand A pointer or register index)
//   rs2: source register 2 (operand B pointer or register index)
//   rd:  destination register (result pointer or status)
//   funct3: sub-function (0 = default)
//
// Operand convention for poly.add / poly.mul:
//   rs1 = base address of 256-bit operand A in tightly-coupled SRAM
//   rs2 = base address of 256-bit operand B in tightly-coupled SRAM
//   rd  = receives status word (0 = success)
//   The 256-bit result is written back to the address in rs1 (in-place).
//
// Memory interface:
//   Two 256-bit read ports (operand A and B) and one 256-bit write port.
//   Assumes tightly-coupled SRAM delivers 256 bits per cycle.
//
// Pipeline handshake:
//   The core asserts req_valid when it has an Xlattice instruction.
//   This unit asserts req_ready when it can accept.
//   resp_valid signals completion; resp_data carries the status for rd.
//   While a multi-cycle operation (poly.mul) is in progress, req_ready
//   is deasserted, stalling the core pipeline.
//
// Modulus: p = 2^255 - 19 (Curve25519 prime, hardwired)
// =============================================================================

module xlattice_unit
    import qug_pkg::*;
(
    input  logic        clk,
    input  logic        rst_n,

    // =========================================================================
    // RISC-V core interface (valid/ready handshake)
    // =========================================================================
    input  logic        req_valid,      // Core has an Xlattice instruction
    output logic        req_ready,      // Unit can accept
    input  logic [6:0]  req_funct7,     // Operation select
    input  logic [2:0]  req_funct3,     // Sub-function (reserved)
    input  logic [31:0] req_rs1,        // Source register 1 value
    input  logic [31:0] req_rs2,        // Source register 2 value
    input  logic [4:0]  req_rd_addr,    // Destination register address

    output logic        resp_valid,     // Result ready
    output logic [4:0]  resp_rd_addr,   // Destination register address
    output logic [31:0] resp_data,      // Result data for rd (status word)
    output logic        resp_wr_en,     // Write-back enable

    // =========================================================================
    // Tightly-coupled SRAM interface (256-bit wide)
    // =========================================================================
    // Read port A (operand A)
    output logic [31:0] mem_rd_addr_a,  // Read address for operand A
    output logic        mem_rd_en_a,    // Read enable A
    input  logic [255:0] mem_rd_data_a, // 256-bit operand A
    input  logic        mem_rd_valid_a, // Data valid A

    // Read port B (operand B)
    output logic [31:0] mem_rd_addr_b,  // Read address for operand B
    output logic        mem_rd_en_b,    // Read enable B
    input  logic [255:0] mem_rd_data_b, // 256-bit operand B
    input  logic        mem_rd_valid_b, // Data valid B

    // Write port (result, written to address of operand A)
    output logic [31:0] mem_wr_addr,    // Write address (= rs1)
    output logic        mem_wr_en,      // Write enable
    output logic [255:0] mem_wr_data    // 256-bit result
);

    // =========================================================================
    // Funct7 operation encoding
    // =========================================================================
    localparam logic [6:0] F7_NTT_FWD    = 7'd0;   // Forward NTT (stub)
    localparam logic [6:0] F7_NTT_INV    = 7'd1;   // Inverse NTT (stub)
    localparam logic [6:0] F7_POLY_ADD   = 7'd2;   // Modular addition
    localparam logic [6:0] F7_POLY_MUL   = 7'd3;   // Modular multiplication
    localparam logic [6:0] F7_POLY_RED   = 7'd4;   // Barrett reduction (stub)
    localparam logic [6:0] F7_MOD_SUB    = 7'd5;   // Modular subtraction
    localparam logic [6:0] F7_MOD_SQR    = 7'd6;   // Modular squaring (a*a mod p)
    localparam logic [6:0] F7_SET_MOD    = 7'd7;   // Set custom modulus from SRAM

    // =========================================================================
    // Curve25519 prime: p = 2^255 - 19
    // =========================================================================
    localparam logic [255:0] CURVE25519_P =
        256'h7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFED;

    // =========================================================================
    // FSM states
    // =========================================================================
    typedef enum logic [3:0] {
        S_IDLE,             // Waiting for instruction
        S_FETCH_OPS,        // Fetching operands from SRAM
        S_WAIT_OPS,         // Waiting for SRAM read valid
        S_EXEC_ADD,         // Execute modular addition
        S_WAIT_ADD,         // Wait for add result
        S_EXEC_SUB,         // Execute modular subtraction
        S_WAIT_SUB,         // Wait for sub result
        S_EXEC_MUL,         // Start modular multiplication
        S_WAIT_MUL,         // Wait for multiplication (12 cycles)
        S_WRITEBACK,        // Write result to SRAM
        S_RESPOND,          // Send response to core
        S_STUB_RESPOND      // Immediate response for stub instructions
    } state_t;

    state_t fsm_state, fsm_next;

    // =========================================================================
    // Latched instruction fields
    // =========================================================================
    logic [6:0]  lat_funct7;
    logic [31:0] lat_rs1;
    logic [31:0] lat_rs2;
    logic [4:0]  lat_rd_addr;

    // =========================================================================
    // Operand registers (latched from SRAM reads)
    // =========================================================================
    logic [255:0] operand_a;
    logic [255:0] operand_b;

    // =========================================================================
    // Configurable modulus register (default: Curve25519)
    // =========================================================================
    logic [255:0] active_modulus;

    // =========================================================================
    // mod_add_256 interface
    // =========================================================================
    logic         add_start;
    logic [255:0] add_result;
    logic         add_done;

    (* dont_touch = "true" *)
    mod_add_256 u_mod_add (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (add_start),
        .op_a    (operand_a),
        .op_b    (operand_b),
        .modulus (active_modulus),
        .result  (add_result),
        .done    (add_done)
    );

    // =========================================================================
    // mod_mul_256 interface
    // =========================================================================
    logic         mul_start;
    logic [255:0] mul_result;
    logic         mul_done;

    (* dont_touch = "true" *)
    mod_mul_256 u_mod_mul (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (mul_start),
        .op_a    (operand_a),
        .op_b    (operand_b),
        .modulus (active_modulus),
        .result  (mul_result),
        .done    (mul_done)
    );

    // =========================================================================
    // mod_sub_256 interface (Genus-2 VDF support)
    // =========================================================================
    logic         sub_start;
    logic [255:0] sub_result;
    logic         sub_done;

    (* dont_touch = "true" *)
    mod_sub_256 u_mod_sub (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (sub_start),
        .op_a    (operand_a),
        .op_b    (operand_b),
        .modulus (active_modulus),
        .result  (sub_result),
        .done    (sub_done)
    );

    // =========================================================================
    // Result register
    // =========================================================================
    logic [255:0] result_reg;

    // =========================================================================
    // FSM: next state logic
    // =========================================================================
    always_comb begin
        fsm_next = fsm_state;

        case (fsm_state)
            S_IDLE: begin
                if (req_valid) begin
                    case (req_funct7)
                        F7_NTT_FWD,
                        F7_NTT_INV,
                        F7_POLY_RED:  fsm_next = S_STUB_RESPOND;  // Stubs
                        F7_POLY_ADD,
                        F7_POLY_MUL,
                        F7_MOD_SUB,
                        F7_MOD_SQR:   fsm_next = S_FETCH_OPS;     // Real ops
                        F7_SET_MOD:   fsm_next = S_FETCH_OPS;     // Load modulus from SRAM addr
                        default:      fsm_next = S_STUB_RESPOND;  // Unknown
                    endcase
                end
            end

            S_FETCH_OPS: begin
                fsm_next = S_WAIT_OPS;
            end

            S_WAIT_OPS: begin
                if (mem_rd_valid_a && mem_rd_valid_b) begin
                    case (lat_funct7)
                        F7_POLY_ADD: fsm_next = S_EXEC_ADD;
                        F7_MOD_SUB:  fsm_next = S_EXEC_SUB;
                        F7_POLY_MUL,
                        F7_MOD_SQR:  fsm_next = S_EXEC_MUL;  // SQR uses mul with op_b=op_a
                        F7_SET_MOD:  fsm_next = S_RESPOND;    // Just latch, no writeback
                        default:     fsm_next = S_RESPOND;
                    endcase
                end
            end

            S_EXEC_ADD: begin
                fsm_next = S_WAIT_ADD;
            end

            S_WAIT_ADD: begin
                if (add_done) fsm_next = S_WRITEBACK;
            end

            S_EXEC_SUB: begin
                fsm_next = S_WAIT_SUB;
            end

            S_WAIT_SUB: begin
                if (sub_done) fsm_next = S_WRITEBACK;
            end

            S_EXEC_MUL: begin
                fsm_next = S_WAIT_MUL;
            end

            S_WAIT_MUL: begin
                if (mul_done) fsm_next = S_WRITEBACK;
            end

            S_WRITEBACK: begin
                fsm_next = S_RESPOND;
            end

            S_RESPOND: begin
                fsm_next = S_IDLE;
            end

            S_STUB_RESPOND: begin
                fsm_next = S_IDLE;
            end

            default: fsm_next = S_IDLE;
        endcase
    end

    // =========================================================================
    // FSM: state register and datapath
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm_state      <= S_IDLE;
            lat_funct7     <= 7'd0;
            lat_rs1        <= 32'd0;
            lat_rs2        <= 32'd0;
            lat_rd_addr    <= 5'd0;
            operand_a      <= 256'd0;
            operand_b      <= 256'd0;
            result_reg     <= 256'd0;
            active_modulus <= 256'h7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFED; // Curve25519 default
        end else begin
            fsm_state <= fsm_next;

            // Latch instruction on acceptance
            if (fsm_state == S_IDLE && req_valid) begin
                lat_funct7  <= req_funct7;
                lat_rs1     <= req_rs1;
                lat_rs2     <= req_rs2;
                lat_rd_addr <= req_rd_addr;
            end

            // Latch operands when SRAM reads complete
            if (fsm_state == S_WAIT_OPS && mem_rd_valid_a && mem_rd_valid_b) begin
                operand_a <= mem_rd_data_a;
                operand_b <= mem_rd_data_b;
            end

            // For squaring (mod_sqr), set operand_b = operand_a
            if (fsm_state == S_WAIT_OPS && mem_rd_valid_a && mem_rd_valid_b && lat_funct7 == F7_MOD_SQR) begin
                operand_b <= mem_rd_data_a;
            end

            // Latch custom modulus from operand A
            if (fsm_state == S_WAIT_OPS && mem_rd_valid_a && mem_rd_valid_b && lat_funct7 == F7_SET_MOD) begin
                active_modulus <= mem_rd_data_a;
            end

            // Latch results
            if (fsm_state == S_WAIT_ADD && add_done) begin
                result_reg <= add_result;
            end
            if (fsm_state == S_WAIT_SUB && sub_done) begin
                result_reg <= sub_result;
            end
            if (fsm_state == S_WAIT_MUL && mul_done) begin
                result_reg <= mul_result;
            end
        end
    end

    // =========================================================================
    // Memory interface
    // =========================================================================
    always_comb begin
        mem_rd_addr_a = lat_rs1;
        mem_rd_addr_b = lat_rs2;
        mem_rd_en_a   = (fsm_state == S_FETCH_OPS || fsm_state == S_WAIT_OPS);
        mem_rd_en_b   = (fsm_state == S_FETCH_OPS || fsm_state == S_WAIT_OPS);

        mem_wr_addr   = lat_rs1;  // Write result back to operand A address
        mem_wr_en     = (fsm_state == S_WRITEBACK);
        mem_wr_data   = result_reg;
    end

    // =========================================================================
    // Arithmetic unit start signals
    // =========================================================================
    always_comb begin
        add_start = (fsm_state == S_EXEC_ADD);
        sub_start = (fsm_state == S_EXEC_SUB);
        mul_start = (fsm_state == S_EXEC_MUL);
    end

    // =========================================================================
    // Core response interface
    // =========================================================================
    always_comb begin
        req_ready    = (fsm_state == S_IDLE);
        resp_valid   = 1'b0;
        resp_data    = 32'd0;
        resp_rd_addr = lat_rd_addr;
        resp_wr_en   = 1'b0;

        case (fsm_state)
            S_RESPOND: begin
                resp_valid   = 1'b1;
                resp_data    = 32'd0;       // Status: 0 = success
                resp_rd_addr = lat_rd_addr;
                resp_wr_en   = 1'b1;
            end

            S_STUB_RESPOND: begin
                // Phase 1B stubs: return 0 immediately
                resp_valid   = 1'b1;
                resp_data    = 32'd0;       // Stub: result = 0
                resp_rd_addr = (fsm_state == S_IDLE) ? req_rd_addr : lat_rd_addr;
                resp_wr_en   = 1'b1;
            end

            default: ;
        endcase
    end

endmodule
