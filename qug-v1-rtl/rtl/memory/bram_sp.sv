// =============================================================================
// bram_sp.sv — Parameterized Single-Port Synchronous BRAM
// QUG-V1 Mining SoC — Memory Subsystem
// =============================================================================
//
// Infers Xilinx BRAM36K on Kintex-7 via (* ram_style = "block" *) attribute.
//
// Parameters:
//   DEPTH — Number of words (e.g., 16384 for 64KB with WIDTH=32)
//   WIDTH — Bits per word (default 32)
//
// Interface:
//   Synchronous read with 1-cycle latency.
//   Synchronous write (write-first: read returns new data on simultaneous R/W).
//
// =============================================================================

module bram_sp #(
    parameter int DEPTH = 16384,   // Number of words
    parameter int WIDTH = 32       // Bits per word
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    en,      // Enable (read or write)
    input  logic                    we,      // Write enable
    input  logic [$clog2(DEPTH)-1:0] addr,   // Word address
    input  logic [WIDTH-1:0]        wdata,   // Write data
    output logic [WIDTH-1:0]        rdata    // Read data (1-cycle latency)
);

    // Xilinx synthesis attribute — force BRAM inference
    (* ram_style = "block" *)
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rdata <= '0;
        end else if (en) begin
            if (we) begin
                mem[addr] <= wdata;
            end
            rdata <= mem[addr];  // Read-first behavior
        end
    end

endmodule
