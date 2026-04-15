// =============================================================================
// sha3_state.sv — Keccak State Register File for SHA-3-256
// QUG-V1 Mining SoC — Xcrypto SHA-3 Hardware Accelerator
// =============================================================================
//
// 25 x 64-bit register file holding the full Keccak-f[1600] state.
// Provides INIT, ABSORB, and READ operations for the SHA-3 data path.
//
// Operations:
//   NOP    (2'd0) — No state change, outputs remain valid
//   INIT   (2'd1) — Zero all 25 lanes (prepare for new hash)
//   ABSORB (2'd2) — XOR rate block (1088 bits) into lanes 0..16
//   READ   (2'd3) — No state change (read port always active)
//
// The hash output port continuously reflects lanes 0-3 (256 bits).
// The full lane array is available for the permutation core to read.
//
// Target: 1600 FF + minimal LUT on Kintex-7 XC7K325T
// =============================================================================

module sha3_state (
    input  logic          clk,
    input  logic          rst_n,

    // Control interface
    input  logic [1:0]    op,           // 0=NOP, 1=INIT, 2=ABSORB, 3=READ

    // Data interface
    input  logic [1087:0] rate_in,      // Rate block for absorption (1088 bits)

    // Bulk write interface (from permutation core)
    input  logic [63:0]   wr_lanes [0:24],
    input  logic          wr_en,         // Write-enable from permutation

    // Read interface
    output logic [63:0]   lanes [0:24],  // Full state (25 lanes, always readable)
    output logic [255:0]  hash_out       // Extracted hash: {lane3, lane2, lane1, lane0}
);

    // =========================================================================
    // Operation encoding
    // =========================================================================

    localparam logic [1:0] OP_NOP    = 2'd0;
    localparam logic [1:0] OP_INIT   = 2'd1;
    localparam logic [1:0] OP_ABSORB = 2'd2;
    localparam logic [1:0] OP_READ   = 2'd3;

    // =========================================================================
    // State register file: 25 x 64-bit lanes
    // =========================================================================

    logic [63:0] s [0:24];

    // =========================================================================
    // Continuous read outputs
    // =========================================================================

    // Full state — always available for the permutation core
    always_comb begin
        for (int i = 0; i < 25; i++) begin
            lanes[i] = s[i];
        end
    end

    // Hash output — first 4 lanes (256 bits), little-endian lane order
    assign hash_out = {s[3], s[2], s[1], s[0]};

    // =========================================================================
    // State register update logic
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset: zero all lanes
            for (int i = 0; i < 25; i++) begin
                s[i] <= 64'd0;
            end
        end else begin
            // Bulk write from permutation core takes priority
            if (wr_en) begin
                for (int i = 0; i < 25; i++) begin
                    s[i] <= wr_lanes[i];
                end
            end else begin
                case (op)
                    OP_INIT: begin
                        // Zero all 25 lanes — prepare for new hash
                        for (int i = 0; i < 25; i++) begin
                            s[i] <= 64'd0;
                        end
                    end

                    OP_ABSORB: begin
                        // XOR rate block into lanes 0..16
                        // Rate = 1088 bits = 17 lanes x 64 bits
                        for (int i = 0; i < 17; i++) begin
                            s[i] <= s[i] ^ rate_in[i*64 +: 64];
                        end
                        // Lanes 17..24 (capacity region) unchanged
                    end

                    // OP_NOP, OP_READ: no state change
                    default: ;
                endcase
            end
        end
    end

endmodule
