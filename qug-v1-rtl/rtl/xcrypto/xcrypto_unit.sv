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

    // =========================================================================
    // Message block interface (tightly-coupled SRAM / cache)
    // =========================================================================
    // For blake3.round: 16 x 32-bit message words loaded from memory
    // Address comes from rs1; assume single-cycle 512-bit read
    output logic [31:0] mem_addr,       // Message block base address
    output logic        mem_rd_en,      // Memory read enable
    input  logic [31:0] mem_block [0:15], // 512-bit message block data
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
        S_IDLE,             // Waiting for instruction
        S_FETCH_MSG,        // Waiting for message block from memory
        S_COMPRESS,         // Pipeline compression running
        S_WAIT_PIPELINE,    // Waiting for pipeline result
        S_CHAIN_WRITEBACK,  // Writing chain result back to state
        S_FINALIZE          // Reading hash word
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
    logic [31:0] lat_rs2;
    logic [4:0]  lat_rd_addr;

    // Chain counter — tracks VDF chain iteration
    logic [6:0]  chain_count;
    logic [6:0]  chain_target;

    // Pipeline completion flag (for single compression)
    logic        compress_started;

    // FSM watchdog timeout counter
    logic [9:0]  fsm_timeout_cnt;
    logic        fsm_timeout_error;

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
        .hash_out       (pipe_hash_out),
        .out_valid      (pipe_out_valid)
    );

    // =========================================================================
    // FSM: next state logic
    // =========================================================================
    always_comb begin
        fsm_next = fsm_state;

        // Watchdog: force return to S_IDLE on timeout
        if (fsm_timeout_cnt == 10'd1000 &&
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
                            // More chain iterations needed
                            fsm_next = S_CHAIN_WRITEBACK;
                        end else begin
                            fsm_next = S_IDLE;
                        end
                    end
                end

                S_CHAIN_WRITEBACK: begin
                    // Write hash back to state as CV, then re-compress
                    fsm_next = S_COMPRESS;
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
            chain_count    <= 7'd0;
            chain_target   <= 7'd0;
            compress_started <= 1'b0;
            fsm_timeout_cnt  <= 10'd0;
            fsm_timeout_error <= 1'b0;
        end else begin
            fsm_state <= fsm_next;

            // Watchdog timeout counter
            if (fsm_state == S_FETCH_MSG || fsm_state == S_WAIT_PIPELINE) begin
                if (fsm_timeout_cnt < 10'd1000)
                    fsm_timeout_cnt <= fsm_timeout_cnt + 10'd1;
            end else begin
                fsm_timeout_cnt <= 10'd0;
            end

            // Assert error on timeout, clear when FSM returns to idle
            if (fsm_timeout_cnt == 10'd1000)
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
                    // rs2[6:0] = chain length (default 100 for mining)
                    chain_target <= req_rs2[6:0];
                    chain_count  <= 7'd0;
                end
            end

            // Track compression start
            if (fsm_state == S_COMPRESS) begin
                compress_started <= 1'b1;
            end
            if (fsm_state == S_IDLE) begin
                compress_started <= 1'b0;
            end

            // Increment chain counter on writeback
            if (fsm_state == S_CHAIN_WRITEBACK) begin
                chain_count <= chain_count + 7'd1;
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
    // =========================================================================
    logic [31:0] msg_block_lat [0:15];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 16; i++) begin
                msg_block_lat[i] <= 32'd0;
            end
        end else if (fsm_state == S_FETCH_MSG && mem_valid) begin
            for (int i = 0; i < 16; i++) begin
                msg_block_lat[i] <= mem_block[i];
            end
        end
    end

    // =========================================================================
    // Latched hash output (for chain feedback)
    // =========================================================================
    logic [31:0] hash_latched [0:7];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 8; i++) begin
                hash_latched[i] <= 32'd0;
            end
        end else if (pipe_out_valid) begin
            for (int i = 0; i < 8; i++) begin
                hash_latched[i] <= pipe_hash_out[i];
            end
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
        for (int i = 0; i < 16; i++) state_bulk_in[i] = 32'd0;
        for (int i = 0; i < 8; i++)  state_cv_in[i] = 32'd0;

        case (fsm_state)
            S_IDLE: begin
                if (req_valid && req_funct7 == F7_INIT) begin
                    state_op = 3'd1;  // OP_INIT
                end
            end

            S_CHAIN_WRITEBACK: begin
                // Load hash output as new chaining value
                state_op = 3'd2;  // OP_LOAD_CV
                for (int i = 0; i < 8; i++) begin
                    state_cv_in[i] = hash_latched[i];
                end
            end

            S_FINALIZE: begin
                // Read state register for finalize — rs1[3:0] selects word
                state_op     = 3'd4;  // OP_READ
                state_rd_idx = lat_rs1[3:0];
            end

            S_WAIT_PIPELINE: begin
                // When pipeline produces output and we are done, update state
                if (pipe_out_valid && !(lat_funct7 == F7_CHAIN && chain_count < chain_target)) begin
                    state_bulk_wr_en = 1'b1;
                    for (int i = 0; i < 8; i++) begin
                        state_bulk_in[i]     = pipe_hash_out[i];
                        state_bulk_in[i + 8] = 32'd0;
                    end
                end
            end

            default: ;
        endcase
    end

    // =========================================================================
    // Pipeline input control
    // =========================================================================
    always_comb begin
        pipe_in_valid  = 1'b0;
        pipe_counter   = {lat_rs2, 32'd0};  // Upper 32 bits from rs2
        pipe_block_len = 32'd64;             // Default: full 64-byte block
        pipe_flags     = 32'd0;

        for (int i = 0; i < 8; i++)  pipe_cv[i]    = state_out[i];
        for (int i = 0; i < 16; i++) pipe_block[i]  = msg_block_lat[i];

        if (fsm_state == S_COMPRESS) begin
            pipe_in_valid = 1'b1;

            if (lat_funct7 == F7_CHAIN && chain_count > 7'd0) begin
                // Chain iteration: use latched hash as chaining value
                for (int i = 0; i < 8; i++) begin
                    pipe_cv[i] = hash_latched[i];
                end
                // For chain iterations after the first, the "message" is
                // the same block (the original input being hashed repeatedly)
                pipe_flags = 32'h0;  // No special flags for inner chain hashes
            end
        end
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
                    resp_data    = pipe_hash_out[0];  // First hash word as status
                    resp_rd_addr = lat_rd_addr;
                    resp_wr_en   = 1'b1;
                end
            end

            default: ;
        endcase
    end

endmodule
