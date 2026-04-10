// =============================================================================
// blake3_state.sv — Dedicated 64-Byte State Register File for Xcrypto
// QUG-V1 Mining SoC — Xcrypto BLAKE3 Hardware Pipeline
// =============================================================================
//
// 16 x 32-bit registers (s0-s15) forming the BLAKE3 working state.
// Separate from the RISC-V integer register file (x0-x31).
//
// Operations:
//   INIT     — Load BLAKE3 IV into s0-s7, zero s8-s15
//   LOAD_CV  — Load chaining value into s0-s7 from external bus
//   UPDATE   — Bulk-write all 16 registers from round output
//   READ     — Read individual register by index (for finalize)
//   SETUP    — Configure s8-s15 with IV, counter, block_len, flags
//
// This register file is the architectural state visible to the Xcrypto ISA.
// The pipeline reads/writes through this interface.
// =============================================================================

module blake3_state (
    input  logic        clk,
    input  logic        rst_n,

    // Control interface
    input  logic [2:0]  op,           // Operation select
    input  logic [3:0]  rd_idx,       // Register index for scalar read
    input  logic [31:0] wr_scalar,    // Scalar write data (for individual reg writes)
    input  logic [3:0]  wr_idx,       // Scalar write index

    // Bulk write interface (from pipeline output)
    input  logic [31:0] bulk_in [0:15],
    input  logic        bulk_wr_en,

    // Chaining value load interface
    input  logic [31:0] cv_in [0:7],

    // Compression setup interface
    input  logic [63:0] counter,
    input  logic [31:0] block_len,
    input  logic [31:0] flags_in,

    // Read interface
    output logic [31:0] rd_data,           // Scalar read output
    output logic [31:0] state_out [0:15]   // Full state read (to pipeline)
);

    // =========================================================================
    // Operation encoding
    // =========================================================================
    localparam logic [2:0] OP_NOP     = 3'd0;
    localparam logic [2:0] OP_INIT    = 3'd1;  // Load IV, reset state
    localparam logic [2:0] OP_LOAD_CV = 3'd2;  // Load chaining value into s0-s7
    localparam logic [2:0] OP_UPDATE  = 3'd3;  // Bulk update from round output
    localparam logic [2:0] OP_READ    = 3'd4;  // Scalar read (no state change)
    localparam logic [2:0] OP_SETUP   = 3'd5;  // Setup s8-s15 for compression
    localparam logic [2:0] OP_WR_REG  = 3'd6;  // Write single register

    // =========================================================================
    // BLAKE3 IV constants
    // =========================================================================
    localparam logic [31:0] IV [0:7] = '{
        32'h6A09E667, 32'hBB67AE85, 32'h3C6EF372, 32'hA54FF53A,
        32'h510E527F, 32'h9B05688C, 32'h1F83D9AB, 32'h5BE0CD19
    };

    // =========================================================================
    // State register file: 16 x 32-bit
    // =========================================================================
    logic [31:0] s [0:15];

    // Continuous output — full state always available
    always_comb begin
        for (int i = 0; i < 16; i++) begin
            state_out[i] = s[i];
        end
    end

    // Scalar read output
    always_comb begin
        rd_data = s[rd_idx];
    end

    // =========================================================================
    // Register file update logic
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset: load IV into upper half, zero lower half
            for (int i = 0; i < 8; i++) begin
                s[i] <= IV[i];
            end
            for (int i = 8; i < 16; i++) begin
                s[i] <= 32'd0;
            end
        end else begin
            // Bulk write takes priority (from pipeline completion)
            if (bulk_wr_en) begin
                for (int i = 0; i < 16; i++) begin
                    s[i] <= bulk_in[i];
                end
            end else begin
                case (op)
                    OP_INIT: begin
                        // Load IV into s0-s7, zero s8-s15
                        for (int i = 0; i < 8; i++) begin
                            s[i] <= IV[i];
                        end
                        for (int i = 8; i < 16; i++) begin
                            s[i] <= 32'd0;
                        end
                    end

                    OP_LOAD_CV: begin
                        // Load external chaining value into s0-s7
                        for (int i = 0; i < 8; i++) begin
                            s[i] <= cv_in[i];
                        end
                    end

                    OP_SETUP: begin
                        // Set up s8-s15 for compression function
                        // s8-s11  = IV[0..3]
                        // s12     = counter[31:0]
                        // s13     = counter[63:32]
                        // s14     = block_len
                        // s15     = flags
                        s[ 8] <= IV[0];
                        s[ 9] <= IV[1];
                        s[10] <= IV[2];
                        s[11] <= IV[3];
                        s[12] <= counter[31:0];
                        s[13] <= counter[63:32];
                        s[14] <= block_len;
                        s[15] <= flags_in;
                    end

                    OP_WR_REG: begin
                        // Write a single register
                        s[wr_idx] <= wr_scalar;
                    end

                    // OP_NOP, OP_READ: no state change
                    default: ;
                endcase
            end
        end
    end

endmodule
