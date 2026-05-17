// =============================================================================
// bram_dp.sv — Parameterized True Dual-Port Synchronous BRAM
// QUG-V1 Mining SoC — Memory Subsystem
// =============================================================================
//
// Infers Xilinx BRAM36K on Kintex-7 via (* ram_style = "block" *) attribute.
//
// Parameters:
//   DEPTH — Number of words (e.g., 16384 for 64KB with WIDTH=32)
//   WIDTH — Bits per word (default 32)
//
// Port A: read/write (CPU data access)
// Port B: read-only  (debug / JTAG scan, optional — active only when en_b=1)
//
// Both ports are synchronous with 1-cycle read latency.
// Simultaneous write to Port A and read to Port B at the same address:
//   Port B returns the OLD value (read-first on both ports).
//
// =============================================================================

module bram_dp #(
    parameter int DEPTH = 16384,   // Number of words
    parameter int WIDTH = 32       // Bits per word
) (
    input  logic                    clk,
    input  logic                    rst_n,

    // Port A — read/write (CPU)
    input  logic                    en_a,
    input  logic                    we_a,
    input  logic [$clog2(DEPTH)-1:0] addr_a,
    input  logic [WIDTH-1:0]        wdata_a,
    output logic [WIDTH-1:0]        rdata_a,

    // Port B — read-only (debug / JTAG)
    input  logic                    en_b,
    input  logic [$clog2(DEPTH)-1:0] addr_b,
    output logic [WIDTH-1:0]        rdata_b
);

    // Xilinx synthesis attribute — force BRAM inference
    (* ram_style = "block" *)
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    // Port A: read/write
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rdata_a <= '0;
        end else if (en_a) begin
            if (we_a) begin
                mem[addr_a] <= wdata_a;
            end
            rdata_a <= mem[addr_a];
        end
    end

    // Port B: read-only
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rdata_b <= '0;
        end else if (en_b) begin
            rdata_b <= mem[addr_b];
        end
    end

endmodule
