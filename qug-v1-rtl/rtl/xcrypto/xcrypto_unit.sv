// =============================================================================
// xcrypto_unit.sv — Top-Level Xcrypto Extension Unit for RISC-V Core
// QUG-V1 Mining SoC — Xcrypto BLAKE3 Hardware Pipeline
// =============================================================================
//
// Interfaces with the RISC-V core pipeline via the custom-0 opcode (0x0B).
// Decodes funct7 field to select BLAKE3 operations.
//
// Xcrypto ISA instructions (R-type encoding, opcode = 7'b0001011):
//   funct7 = 0: blake3.init     — Load IV into state, reset pipeline
//   funct7 = 1: blake3.round    — Start pipelined compression (7 rounds)
//   funct7 = 2: blake3.chain    — Feed hash output back as chaining value
//   funct7 = 3: blake3.finalize — Read final hash word from state into rd
//
// R-type encoding: [funct7 | rs2 | rs1 | funct3 | rd | opcode]
//   rs1: source register 1 (address of message block in memory, or word index)
//   rs2: source register 2 (counter/flags depending on instruction)
//   rd:  destination register (hash word output for finalize)
//   funct3: sub-function (0 = default)
//
// Pipeline handshake:
//   The core asserts req_valid when it has an Xcrypto instruction.
//   This unit asserts req_ready when it can accept (init/chain are 1-cycle,
//   round takes 7 cycles but is pipelined, finalize is 1-cycle).
//   resp_valid signals completion; resp_data carries the result for rd.
//
// VDF chain operation (blake3.chain):
//   Takes the 256-bit hash output from the pipeline and feeds it back into
//   the state registers as the new chaining value. This enables the 100-hash
//   sequential chain required for QUG mining proof-of-work without round-
//   tripping through the RISC-V register file.
//
// Memory interface:
//   blake3.round needs 16 message words. The unit provides a memory read
//   interface to fetch the 64-byte block from the address in rs1.
//   For simplicity, we assume a tightly-coupled SRAM that delivers 512 bits
//   in a single cycle (message_block input port).
// =============================================================================

module xcrypto_unit
    import xcrypto_pkg::*;
(
    input  logic        clk,
    input  logic        rst_n,

    // =========================================================================
    // RISC-V core interface
    // =========================================================================
    input  logic        req_valid,      // Core has an Xcrypto instruction
    output logic        req_ready,      // Unit can accept
    input  logic [6:0]  req_funct7,     // Operation select
    input  logic [2:0]  req_funct3,     // Sub-function (unused, reserved)
    input  logic [31:0] req_rs1,        // Source register 1 value
    input  logic [31:0] req_rs2,        // Source register 2 value
    input  logic [4:0]  req_rd_addr,    // Destination register address

    output logic        resp_valid,     // Result ready
    output logic [4:0]  resp_rd_addr,   // Destination register address
    output logic [31:0] resp_data,      // Result data for rd
    output logic        resp_wr_en,     // Write-back enable
    output logic [31:0] hash_words_out [0:7], // Latest hash (for best-hash tracking in mining_controller)

    // =========================================================================
    // Message block interface (tightly-coupled SRAM / cache)
    // =========================================================================
    // For blake3.round: 16 x 32-bit message words loaded from memory
    // Address comes from rs1; assume single-cycle 512-bit read
    output logic [31:0] mem_addr,       // Message block base address
    output logic        mem_rd_en,      // Memory read enable
    input  logic [511:0] mem_block,        // 512-bit message block data (packed)
    input  logic        mem_valid       // Memory data valid
);

    // =========================================================================
    // Funct7 operation encoding
    // =========================================================================
    localparam logic [6:0] F7_INIT     = 7'd0;
    localparam logic [6:0] F7_ROUND    = 7'd1;
    localparam logic [6:0] F7_CHAIN    = 7'd2;
    localparam logic [6:0] F7_FINALIZE = 7'd3;

    // =========================================================================
    // FSM states
    // =========================================================================
    typedef enum logic [2:0] {
        S_IDLE            = 3'd0,  // Waiting for instruction
        S_FETCH_MSG       = 3'd1,  // Waiting for message block from memory
        S_COMPRESS        = 3'd2,  // Pipeline compression running (first iteration)
        S_WAIT_PIPELINE   = 3'd3,  // Waiting for pipeline result
        S_CHAIN_WRITEBACK = 3'd4,  // Legacy writeback (non-chain round)
        S_FINALIZE        = 3'd5,  // Reading hash word
        S_RELAUNCH        = 3'd6   // Approach B: 1-cycle chain re-launch (saves 1 cycle/iteration vs WRITEBACK+COMPRESS)
    } state_t;

    state_t fsm_state, fsm_next;

    // =========================================================================
    // Internal signals
    // =========================================================================

    // State register file interface
    logic [2:0]  state_op;
    logic [3:0]  state_rd_idx;
    logic [31:0] state_wr_scalar;
    logic [3:0]  state_wr_idx;
    logic [31:0] state_bulk_in [0:15];
    logic        state_bulk_wr_en;
    logic [31:0] state_cv_in [0:7];
    logic [63:0] state_counter;
    logic [31:0] state_block_len;
    logic [31:0] state_flags;
    logic [31:0] state_rd_data;
    logic [31:0] state_out [0:15];

    // Pipeline interface
    logic [31:0] pipe_cv [0:7];
    logic [31:0] pipe_block [0:15];
    logic [63:0] pipe_counter;
    logic [31:0] pipe_block_len;
    logic [31:0] pipe_flags;
    logic        pipe_in_valid;
    logic        pipe_in_ready;
    logic [31:0] pipe_hash_out [0:7];
    logic        pipe_out_valid;

    // Latched instruction fields
    logic [6:0]  lat_funct7;
    logic [31:0] lat_rs1;
    // Pre-computed slice — iverilog 11 does not support constant part-selects inside always_comb
    logic [3:0]  lat_rs1_lo4;
    assign lat_rs1_lo4 = lat_rs1[3:0];
    logic [31:0] lat_rs2;
    logic [4:0]  lat_rd_addr;

    // Chain counter — widened to support Genus-2 VDF (up to 16,383 iterations)
    logic [qug_pkg::VDF_CHAIN_DEPTH_W-1:0] chain_count;
    logic [qug_pkg::VDF_CHAIN_DEPTH_W-1:0] chain_target;

    // Pipeline completion flag (for single compression)
    logic        compress_started;

    // FSM watchdog — widened for deep VDF chains (proportional to chain_target)
    logic [19:0] fsm_timeout_cnt;
    logic        fsm_timeout_error;

    // LWMA difficulty target register (written by firmware via SET_DIFFICULTY)
    logic [qug_pkg::DIFFICULTY_REG_W-1:0] difficulty_target;

    // Leading-zero counter output
    logic [7:0]  lzc_count;
    logic        solution_found;

    // =========================================================================
    // Submodule instantiation: BLAKE3 state register file
    // =========================================================================
    blake3_state u_state (
        .clk         (clk),
        .rst_n       (rst_n),
        .op          (state_op),
        .rd_idx      (state_rd_idx),
        .wr_scalar   (state_wr_scalar),
        .wr_idx      (state_wr_idx),
        .bulk_in     (state_bulk_in),
        .bulk_wr_en  (state_bulk_wr_en),
        .cv_in       (state_cv_in),
        .counter     (state_counter),
        .block_len   (state_block_len),
        .flags_in    (state_flags),
        .rd_data     (state_rd_data),
        .state_out   (state_out)
    );

    // =========================================================================
    // Submodule instantiation: BLAKE3 7-stage pipeline
    // =========================================================================
    blake3_pipeline #(
        .NUM_ROUNDS(7)
    ) u_pipeline (
        .clk            (clk),
        .rst_n          (rst_n),
        .chaining_value (pipe_cv),
        .block_words    (pipe_block),
        .counter        (pipe_counter),
        .block_len      (pipe_block_len),
        .flags          (pipe_flags),
        .in_valid       (pipe_in_valid),
        .in_ready       (pipe_in_ready),
        .hash_out       (pipe_hash_out),  // unconnected — pipe_hash_live used below
        .out_valid      (pipe_out_valid)
    );

    // =========================================================================
    // iverilog 11: unpacked-array output-port connections never propagate.
    // pipe_hash_out stays X; bypass by reading u_pipeline.u_r6.s2_state directly
    // through scalar bridges (single-element hierarchical assigns confirmed to
    // track NBA updates; XOR pairs of scalars to get BLAKE3 finalized output).
    // =========================================================================
    logic [31:0] pr6s0,  pr6s1,  pr6s2,  pr6s3;
    logic [31:0] pr6s4,  pr6s5,  pr6s6,  pr6s7;
    logic [31:0] pr6s8,  pr6s9,  pr6s10, pr6s11;
    logic [31:0] pr6s12, pr6s13, pr6s14, pr6s15;

    assign pr6s0  = u_pipeline.u_r6.s2_state[ 0]; assign pr6s8  = u_pipeline.u_r6.s2_state[ 8];
    assign pr6s1  = u_pipeline.u_r6.s2_state[ 1]; assign pr6s9  = u_pipeline.u_r6.s2_state[ 9];
    assign pr6s2  = u_pipeline.u_r6.s2_state[ 2]; assign pr6s10 = u_pipeline.u_r6.s2_state[10];
    assign pr6s3  = u_pipeline.u_r6.s2_state[ 3]; assign pr6s11 = u_pipeline.u_r6.s2_state[11];
    assign pr6s4  = u_pipeline.u_r6.s2_state[ 4]; assign pr6s12 = u_pipeline.u_r6.s2_state[12];
    assign pr6s5  = u_pipeline.u_r6.s2_state[ 5]; assign pr6s13 = u_pipeline.u_r6.s2_state[13];
    assign pr6s6  = u_pipeline.u_r6.s2_state[ 6]; assign pr6s14 = u_pipeline.u_r6.s2_state[14];
    assign pr6s7  = u_pipeline.u_r6.s2_state[ 7]; assign pr6s15 = u_pipeline.u_r6.s2_state[15];

    // iverilog 11: continuous assign to unpacked-array elements does not propagate.
    // Use individual scalars instead.
    logic [31:0] pipe_hash_live0, pipe_hash_live1, pipe_hash_live2, pipe_hash_live3;
    logic [31:0] pipe_hash_live4, pipe_hash_live5, pipe_hash_live6, pipe_hash_live7;
    assign pipe_hash_live0 = pr6s0 ^ pr6s8;
    assign pipe_hash_live1 = pr6s1 ^ pr6s9;
    assign pipe_hash_live2 = pr6s2 ^ pr6s10;
    assign pipe_hash_live3 = pr6s3 ^ pr6s11;
    assign pipe_hash_live4 = pr6s4 ^ pr6s12;
    assign pipe_hash_live5 = pr6s5 ^ pr6s13;
    assign pipe_hash_live6 = pr6s6 ^ pr6s14;
    assign pipe_hash_live7 = pr6s7 ^ pr6s15;

    // =========================================================================
    // FSM: next state logic
    // =========================================================================
    always_comb begin
        fsm_next = fsm_state;

        // Watchdog: proportional to chain_target (chain_target * 16 cycles per iteration)
        if (fsm_timeout_cnt == {chain_target, 6'd0} &&
            (fsm_state == S_FETCH_MSG || fsm_state == S_WAIT_PIPELINE)) begin
            fsm_next = S_IDLE;
        end else begin
            case (fsm_state)
                S_IDLE: begin
                    if (req_valid) begin
                        case (req_funct7)
                            F7_INIT:     fsm_next = S_IDLE;       // Single-cycle
                            F7_ROUND:    fsm_next = S_FETCH_MSG;  // Need message block
                            F7_CHAIN:    fsm_next = S_FETCH_MSG;  // Need message block for chain
                            F7_FINALIZE: fsm_next = S_FINALIZE;   // Single-cycle read
                            default:     fsm_next = S_IDLE;
                        endcase
                    end
                end

                S_FETCH_MSG: begin
                    if (mem_valid) begin
                        fsm_next = S_COMPRESS;
                    end
                end

                S_COMPRESS: begin
                    // Compression launched into pipeline
                    fsm_next = S_WAIT_PIPELINE;
                end

                S_WAIT_PIPELINE: begin
                    if (pipe_out_valid) begin
                        if (lat_funct7 == F7_CHAIN && chain_count < chain_target) begin
                            // Approach B: 1-cycle S_RELAUNCH replaces S_CHAIN_WRITEBACK+S_COMPRESS
                            // Saves 1 cycle per chain iteration (6.25% throughput gain at depth=100)
                            fsm_next = S_RELAUNCH;
                        end else begin
                            fsm_next = S_IDLE;
                        end
                    end
                end

                S_CHAIN_WRITEBACK: begin
                    // Legacy path for non-chain blake3.round writeback
                    fsm_next = S_COMPRESS;
                end

                S_RELAUNCH: begin
                    // 1-cycle re-launch: pipeline in_valid asserted this cycle,
                    // then immediately back to S_WAIT_PIPELINE for 14 more cycles
                    fsm_next = S_WAIT_PIPELINE;
                end

                S_FINALIZE: begin
                    fsm_next = S_IDLE;
                end

                default: fsm_next = S_IDLE;
            endcase
        end
    end

    // =========================================================================
    // FSM: state register
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm_state      <= S_IDLE;
            lat_funct7     <= 7'd0;
            lat_rs1        <= 32'd0;
            lat_rs2        <= 32'd0;
            lat_rd_addr    <= 5'd0;
            chain_count    <= '0;
            chain_target   <= '0;
            compress_started <= 1'b0;
            fsm_timeout_cnt  <= 20'd0;
            fsm_timeout_error <= 1'b0;
            difficulty_target <= '0;
            lzc_count        <= 8'd0;
            solution_found   <= 1'b0;
        end else begin
            fsm_state <= fsm_next;

            // Watchdog timeout counter
            if (fsm_state == S_FETCH_MSG || fsm_state == S_WAIT_PIPELINE) begin
                if (fsm_timeout_cnt < {chain_target, 6'd0})
                    fsm_timeout_cnt <= fsm_timeout_cnt + 20'd1;
            end else begin
                fsm_timeout_cnt <= 20'd0;
            end

            // Assert error on timeout, clear when FSM returns to idle
            if (fsm_timeout_cnt == {chain_target, 6'd0})
                fsm_timeout_error <= 1'b1;
            else if (fsm_state == S_IDLE)
                fsm_timeout_error <= 1'b0;

            // Latch instruction on acceptance
            if (fsm_state == S_IDLE && req_valid) begin
                lat_funct7  <= req_funct7;
                lat_rs1     <= req_rs1;
                lat_rs2     <= req_rs2;
                lat_rd_addr <= req_rd_addr;

                if (req_funct7 == F7_CHAIN) begin
                    // rs2[VDF_CHAIN_DEPTH_W-1:0] = chain length (100 legacy, up to 16383)
                    chain_target <= req_rs2[qug_pkg::VDF_CHAIN_DEPTH_W-1:0];
                    chain_count  <= '0;
                end

                // SET_DIFFICULTY: write LWMA difficulty target from rs1[7:0]
                if (req_funct7 == 7'd7) begin
                    difficulty_target <= req_rs1[qug_pkg::DIFFICULTY_REG_W-1:0];
                end
            end

            // Track compression start
            if (fsm_state == S_COMPRESS) begin
                compress_started <= 1'b1;
            end
            if (fsm_state == S_IDLE) begin
                compress_started <= 1'b0;
            end

            // Increment chain counter on S_RELAUNCH (Approach B hot path) or legacy S_CHAIN_WRITEBACK
            if (fsm_state == S_RELAUNCH || fsm_state == S_CHAIN_WRITEBACK) begin
                chain_count <= chain_count + 1'b1;
            end
        end
    end

    // =========================================================================
    // Memory interface
    // =========================================================================
    always_comb begin
        mem_addr  = lat_rs1;  // Message block base address from rs1
        mem_rd_en = (fsm_state == S_FETCH_MSG);
    end

    // =========================================================================
    // Latched message block register (hold message for chain iterations)
    // iverilog 11: continuous assign reading from unpacked array element does
    // not re-trigger — use 16 scalar FFs so always_comb sensitivity works.
    // =========================================================================
    logic [31:0] mb0,  mb1,  mb2,  mb3;
    logic [31:0] mb4,  mb5,  mb6,  mb7;
    logic [31:0] mb8,  mb9,  mb10, mb11;
    logic [31:0] mb12, mb13, mb14, mb15;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mb0  <= 32'd0; mb1  <= 32'd0; mb2  <= 32'd0; mb3  <= 32'd0;
            mb4  <= 32'd0; mb5  <= 32'd0; mb6  <= 32'd0; mb7  <= 32'd0;
            mb8  <= 32'd0; mb9  <= 32'd0; mb10 <= 32'd0; mb11 <= 32'd0;
            mb12 <= 32'd0; mb13 <= 32'd0; mb14 <= 32'd0; mb15 <= 32'd0;
        end else if (fsm_state == S_FETCH_MSG && mem_valid) begin
            // mem_block is 512-bit packed: word i = bits [32*i+31 : 32*i]
            mb0  <= mem_block[ 31:  0]; mb1  <= mem_block[ 63: 32];
            mb2  <= mem_block[ 95: 64]; mb3  <= mem_block[127: 96];
            mb4  <= mem_block[159:128]; mb5  <= mem_block[191:160];
            mb6  <= mem_block[223:192]; mb7  <= mem_block[255:224];
            mb8  <= mem_block[287:256]; mb9  <= mem_block[319:288];
            mb10 <= mem_block[351:320]; mb11 <= mem_block[383:352];
            mb12 <= mem_block[415:384]; mb13 <= mem_block[447:416];
            mb14 <= mem_block[479:448]; mb15 <= mem_block[511:480];
        end
    end

    // =========================================================================
    // Latched hash output (for chain feedback)
    // =========================================================================
    logic [31:0] hash_latched [0:7];
    // Scalar bridges — always_comb sensitivity fix (iverilog 11)
    logic [31:0] hl0, hl1, hl2, hl3, hl4, hl5, hl6, hl7;
    assign hl0 = hash_latched[0]; assign hl1 = hash_latched[1];
    assign hl2 = hash_latched[2]; assign hl3 = hash_latched[3];
    assign hl4 = hash_latched[4]; assign hl5 = hash_latched[5];
    assign hl6 = hash_latched[6]; assign hl7 = hash_latched[7];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hash_latched[0] <= 32'd0; hash_latched[1] <= 32'd0;
            hash_latched[2] <= 32'd0; hash_latched[3] <= 32'd0;
            hash_latched[4] <= 32'd0; hash_latched[5] <= 32'd0;
            hash_latched[6] <= 32'd0; hash_latched[7] <= 32'd0;
        end else if (pipe_out_valid) begin
            hash_latched[0] <= pipe_hash_live0; hash_latched[1] <= pipe_hash_live1;
            hash_latched[2] <= pipe_hash_live2; hash_latched[3] <= pipe_hash_live3;
            hash_latched[4] <= pipe_hash_live4; hash_latched[5] <= pipe_hash_live5;
            hash_latched[6] <= pipe_hash_live6; hash_latched[7] <= pipe_hash_live7;
        end
    end

    // =========================================================================
    // State register file control
    // =========================================================================
    always_comb begin
        // Defaults
        state_op        = 3'd0;  // NOP
        state_rd_idx    = 4'd0;
        state_wr_scalar = 32'd0;
        state_wr_idx    = 4'd0;
        state_bulk_wr_en = 1'b0;
        state_counter   = 64'd0;
        state_block_len = 32'd0;
        state_flags     = 32'd0;
        state_bulk_in[ 0] = 32'd0; state_bulk_in[ 1] = 32'd0;
        state_bulk_in[ 2] = 32'd0; state_bulk_in[ 3] = 32'd0;
        state_bulk_in[ 4] = 32'd0; state_bulk_in[ 5] = 32'd0;
        state_bulk_in[ 6] = 32'd0; state_bulk_in[ 7] = 32'd0;
        state_bulk_in[ 8] = 32'd0; state_bulk_in[ 9] = 32'd0;
        state_bulk_in[10] = 32'd0; state_bulk_in[11] = 32'd0;
        state_bulk_in[12] = 32'd0; state_bulk_in[13] = 32'd0;
        state_bulk_in[14] = 32'd0; state_bulk_in[15] = 32'd0;
        state_cv_in[0] = 32'd0; state_cv_in[1] = 32'd0;
        state_cv_in[2] = 32'd0; state_cv_in[3] = 32'd0;
        state_cv_in[4] = 32'd0; state_cv_in[5] = 32'd0;
        state_cv_in[6] = 32'd0; state_cv_in[7] = 32'd0;

        case (fsm_state)
            S_IDLE: begin
                if (req_valid && req_funct7 == F7_INIT) begin
                    state_op = 3'd1;  // OP_INIT
                end
            end

            S_CHAIN_WRITEBACK: begin
                // Load hash output as new chaining value
                state_op = 3'd2;  // OP_LOAD_CV
                state_cv_in[0] = hl0; state_cv_in[1] = hl1;
                state_cv_in[2] = hl2; state_cv_in[3] = hl3;
                state_cv_in[4] = hl4; state_cv_in[5] = hl5;
                state_cv_in[6] = hl6; state_cv_in[7] = hl7;
            end

            S_FINALIZE: begin
                // Read state register for finalize — rs1[3:0] selects word
                state_op     = 3'd4;  // OP_READ
                state_rd_idx = lat_rs1_lo4;
            end

            S_WAIT_PIPELINE: begin
                // When pipeline produces output and we are done, update state
                if (pipe_out_valid && !(lat_funct7 == F7_CHAIN && chain_count < chain_target)) begin
                    state_bulk_wr_en = 1'b1;
                    state_bulk_in[0] = pipe_hash_live0; state_bulk_in[8]  = 32'd0;
                    state_bulk_in[1] = pipe_hash_live1; state_bulk_in[9]  = 32'd0;
                    state_bulk_in[2] = pipe_hash_live2; state_bulk_in[10] = 32'd0;
                    state_bulk_in[3] = pipe_hash_live3; state_bulk_in[11] = 32'd0;
                    state_bulk_in[4] = pipe_hash_live4; state_bulk_in[12] = 32'd0;
                    state_bulk_in[5] = pipe_hash_live5; state_bulk_in[13] = 32'd0;
                    state_bulk_in[6] = pipe_hash_live6; state_bulk_in[14] = 32'd0;
                    state_bulk_in[7] = pipe_hash_live7; state_bulk_in[15] = 32'd0;
                end
            end

            default: ;
        endcase
    end

    // =========================================================================
    // BLAKE3 IV — local copy for tool compatibility (matches xcrypto_pkg scalars)
    // =========================================================================
`ifdef SYNTHESIS
    localparam logic [31:0] BLAKE3_IV [0:7] = '{
        32'h6A09E667, 32'hBB67AE85, 32'h3C6EF372, 32'hA54FF53A,
        32'h510E527F, 32'h9B05688C, 32'h1F83D9AB, 32'h5BE0CD19
    };
`else
    // iverilog 11 does not support array localparams in module scope
    reg [31:0] BLAKE3_IV [0:7];
    initial begin
        BLAKE3_IV[0] = 32'h6A09E667; BLAKE3_IV[1] = 32'hBB67AE85;
        BLAKE3_IV[2] = 32'h3C6EF372; BLAKE3_IV[3] = 32'hA54FF53A;
        BLAKE3_IV[4] = 32'h510E527F; BLAKE3_IV[5] = 32'h9B05688C;
        BLAKE3_IV[6] = 32'h1F83D9AB; BLAKE3_IV[7] = 32'h5BE0CD19;
    end
`endif

    // =========================================================================
    // BLAKE3 flag constants for mining (single-chunk, single-block)
    // =========================================================================
    localparam logic [31:0] MINING_FLAGS = {24'd0,
        BLAKE3_FLAG_CHUNK_START | BLAKE3_FLAG_CHUNK_END | BLAKE3_FLAG_ROOT};

    // =========================================================================
    // Pipeline input control
    // =========================================================================
    // CRITICAL (v4.1 fix): Quillon mining protocol requires:
    //   H₀ = BLAKE3(input[40])  — CV = IV, message = scratchpad, block_len = 40
    //   Hᵢ = BLAKE3(Hᵢ₋₁)     — CV = IV, message = prev hash || zeros, block_len = 32
    //
    // Every hash in the 100-round chain uses the BLAKE3 IV as chaining value.
    // The previous hash output goes into the MESSAGE BLOCK (words 0-7), NOT
    // the chaining value. This matches gpu.rs:199-208 (blake3_hash_32).
    //
    // All hashes use flags = CHUNK_START | CHUNK_END | ROOT (single chunk).
    // Counter is always 0.
    always_comb begin
        pipe_in_valid  = 1'b0;
        pipe_counter   = 64'd0;             // Counter = 0 for all mining hashes
        pipe_block_len = 32'd64;            // Default: full 64-byte block
        pipe_flags     = MINING_FLAGS;      // Default: single-chunk mining flags

        // Default: IV as chaining value, scratchpad as message
        // BLAKE3_IV inlined — iverilog 11 does not re-trigger always_comb on reg+initial writes
        pipe_cv[0] = 32'h6A09E667; pipe_cv[1] = 32'hBB67AE85;
        pipe_cv[2] = 32'h3C6EF372; pipe_cv[3] = 32'hA54FF53A;
        pipe_cv[4] = 32'h510E527F; pipe_cv[5] = 32'h9B05688C;
        pipe_cv[6] = 32'h1F83D9AB; pipe_cv[7] = 32'h5BE0CD19;
        // Scalar FFs — iverilog 11 always_comb sensitivity fix
        pipe_block[ 0] = mb0;  pipe_block[ 1] = mb1;
        pipe_block[ 2] = mb2;  pipe_block[ 3] = mb3;
        pipe_block[ 4] = mb4;  pipe_block[ 5] = mb5;
        pipe_block[ 6] = mb6;  pipe_block[ 7] = mb7;
        pipe_block[ 8] = mb8;  pipe_block[ 9] = mb9;
        pipe_block[10] = mb10; pipe_block[11] = mb11;
        pipe_block[12] = mb12; pipe_block[13] = mb13;
        pipe_block[14] = mb14; pipe_block[15] = mb15;

        if (fsm_state == S_COMPRESS) begin
            pipe_in_valid = 1'b1;

            if (lat_funct7 == F7_CHAIN && chain_count == 7'd0) begin
                // ── First iteration (H₀): hash the 40-byte mining input ──
                // CV = BLAKE3 IV (already set by default above)
                // Message = scratchpad contents (challenge[0:7] + nonce[8:9] + zeros[10:15])
                // block_len = 40 (40 bytes of actual data in the 64-byte block)
                pipe_block_len = 32'd40;
                // pipe_block = msg_block_lat (already set by default)
                // pipe_flags = MINING_FLAGS (already set by default)

            end else if (lat_funct7 == F7_CHAIN && chain_count > 7'd0) begin
                // ── Chain iterations (H₁..H₉₉): hash the 32-byte prev output ──
                // CV = BLAKE3 IV (always! NOT the previous hash)
                // Message = previous hash in words 0-7, zeros in words 8-15
                // block_len = 32 (32 bytes of hash data)
                pipe_block[0] = hl0; pipe_block[1] = hl1;
                pipe_block[2] = hl2; pipe_block[3] = hl3;
                pipe_block[4] = hl4; pipe_block[5] = hl5;
                pipe_block[6] = hl6; pipe_block[7] = hl7;
                pipe_block[ 8] = 32'd0; pipe_block[ 9] = 32'd0;
                pipe_block[10] = 32'd0; pipe_block[11] = 32'd0;
                pipe_block[12] = 32'd0; pipe_block[13] = 32'd0;
                pipe_block[14] = 32'd0; pipe_block[15] = 32'd0;
                pipe_block_len = 32'd32;
                // pipe_cv = BLAKE3_IV (already set by default)
                // pipe_flags = MINING_FLAGS (already set by default)

            end else begin
                // ── Single blake3.round (non-chain): use state registers ──
                pipe_cv[0] = state_out[0]; pipe_cv[1] = state_out[1];
                pipe_cv[2] = state_out[2]; pipe_cv[3] = state_out[3];
                pipe_cv[4] = state_out[4]; pipe_cv[5] = state_out[5];
                pipe_cv[6] = state_out[6]; pipe_cv[7] = state_out[7];
                pipe_block_len = 32'd64;
                pipe_flags     = 32'd0;
            end
        end

        // ── S_RELAUNCH (Approach B): chain hot-path re-launch ──
        // hash_latched was updated on the previous clock edge (when pipe_out_valid fired
        // in S_WAIT_PIPELINE), so it now holds Hᵢ and is safe to use as message block.
        if (fsm_state == S_RELAUNCH) begin
            pipe_in_valid = 1'b1;
            // CV = BLAKE3 IV (already set by default above)
            pipe_block[0] = hl0; pipe_block[1] = hl1;
            pipe_block[2] = hl2; pipe_block[3] = hl3;
            pipe_block[4] = hl4; pipe_block[5] = hl5;
            pipe_block[6] = hl6; pipe_block[7] = hl7;
            pipe_block[ 8] = 32'd0; pipe_block[ 9] = 32'd0;
            pipe_block[10] = 32'd0; pipe_block[11] = 32'd0;
            pipe_block[12] = 32'd0; pipe_block[13] = 32'd0;
            pipe_block[14] = 32'd0; pipe_block[15] = 32'd0;
            pipe_block_len = 32'd32;
            pipe_flags     = MINING_FLAGS;
        end
    end

    // =========================================================================
    // Leading-Zero Counter (LZC) for difficulty checking
    // =========================================================================
    // Timing fix: use pipe_hash_out directly when pipe_out_valid to avoid the
    // 1-cycle lag that would otherwise read hash_latched BEFORE it is updated.
    // hash_words_for_lzc[0] is the most significant word (big-endian convention).

    // Select hash source: live pipeline output during valid cycle, else latch.
    // Individual continuous assigns (not always_comb) to avoid iverilog 11
    // dynamic-index sensitivity issues with unpacked arrays.
    logic [31:0] hash_words_for_lzc [0:7];
    assign hash_words_for_lzc[0] = pipe_out_valid ? pipe_hash_live0 : hash_latched[0];
    assign hash_words_for_lzc[1] = pipe_out_valid ? pipe_hash_live1 : hash_latched[1];
    assign hash_words_for_lzc[2] = pipe_out_valid ? pipe_hash_live2 : hash_latched[2];
    assign hash_words_for_lzc[3] = pipe_out_valid ? pipe_hash_live3 : hash_latched[3];
    assign hash_words_for_lzc[4] = pipe_out_valid ? pipe_hash_live4 : hash_latched[4];
    assign hash_words_for_lzc[5] = pipe_out_valid ? pipe_hash_live5 : hash_latched[5];
    assign hash_words_for_lzc[6] = pipe_out_valid ? pipe_hash_live6 : hash_latched[6];
    assign hash_words_for_lzc[7] = pipe_out_valid ? pipe_hash_live7 : hash_latched[7];

    always_comb begin
        lzc_count = 8'd0;
        solution_found = 1'b0;

        // Count leading zero bits across 8 words (big-endian: word 0 = MSW)
        begin : lzc_block
            logic done_lzc;
            done_lzc = 1'b0;
            for (int w = 0; w < 8; w++) begin
                if (!done_lzc) begin
                    if (hash_words_for_lzc[w] == 32'd0) begin
                        lzc_count = lzc_count + 8'd32;
                    end else begin
                        // CLZ32 on first non-zero word
                        for (int b = 31; b >= 0; b--) begin
                            if (!done_lzc) begin
                                if (hash_words_for_lzc[w][b] == 1'b0) begin
                                    lzc_count = lzc_count + 8'd1;
                                end else begin
                                    done_lzc = 1'b1;
                                end
                            end
                        end
                    end
                end
            end
        end

        if (difficulty_target > 0) begin
            solution_found = (lzc_count >= difficulty_target);
        end
    end

    // hash_words_out: expose current hash to mining_controller for best-hash tracking
    // Same timing-corrected source as LZC: live pipeline output during valid cycle
    always_comb begin
        hash_words_out[0] = hash_words_for_lzc[0]; hash_words_out[1] = hash_words_for_lzc[1];
        hash_words_out[2] = hash_words_for_lzc[2]; hash_words_out[3] = hash_words_for_lzc[3];
        hash_words_out[4] = hash_words_for_lzc[4]; hash_words_out[5] = hash_words_for_lzc[5];
        hash_words_out[6] = hash_words_for_lzc[6]; hash_words_out[7] = hash_words_for_lzc[7];
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
            S_IDLE: begin
                // blake3.init completes in one cycle
                if (req_valid && req_funct7 == F7_INIT) begin
                    resp_valid   = 1'b1;
                    resp_data    = 32'd0;  // No meaningful return value
                    resp_rd_addr = req_rd_addr;
                    resp_wr_en   = 1'b0;   // No writeback for init
                end
            end

            S_FINALIZE: begin
                // Return selected hash word to rd
                resp_valid   = 1'b1;
                resp_data    = state_rd_data;
                resp_rd_addr = lat_rd_addr;
                resp_wr_en   = 1'b1;
            end

            S_WAIT_PIPELINE: begin
                // Pipeline completed — signal done for round/chain
                if (pipe_out_valid && !(lat_funct7 == F7_CHAIN && chain_count < chain_target)) begin
                    resp_valid   = 1'b1;
                    // Return solution_found in bit 31, lzc_count in bits [7:0]
                    resp_data    = {solution_found, 23'd0, lzc_count};
                    resp_rd_addr = lat_rd_addr;
                    resp_wr_en   = 1'b1;
                end
            end

            default: ;
        endcase
    end

endmodule
