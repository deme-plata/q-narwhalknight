// =============================================================================
// mem_subsystem.sv — Memory Subsystem Top-Level
// QUG-V1 Mining SoC — Memory Subsystem
// =============================================================================
//
// Memory map:
//   0x0000_0000 - 0x0000_FFFF : Instruction memory (64KB, read-only from CPU)
//   0x0001_0000 - 0x0001_FFFF : Data memory       (64KB, read-write)
//   0x1000_0000 - 0x1000_000F : UART I/O          (memory-mapped stub)
//
// Interface: valid/ready handshake with addr/wdata/rdata/we.
// Read latency: 1 cycle for BRAM regions.
// UART region: 0-cycle combinational read for status, 1-cycle for data.
//
// Address decoding:
//   [31:28] == 4'h0 && [16] == 0 : Instruction memory (addr[15:2] word index)
//   [31:28] == 4'h0 && [16] == 1 : Data memory        (addr[15:2] word index)
//   [31:28] == 4'h1              : UART I/O
//   Otherwise                     : Bus error (rdata = 0, ready = 1)
//
// =============================================================================

module mem_subsystem (
    input  logic        clk,
    input  logic        rst_n,

    // CPU bus interface
    input  logic        req_valid,   // Request valid
    output logic        req_ready,   // Request accepted this cycle
    input  logic [31:0] req_addr,    // Byte address
    input  logic [31:0] req_wdata,   // Write data
    input  logic        req_we,      // Write enable (0 = read, 1 = write)

    output logic [31:0] resp_rdata,  // Read data (valid 1 cycle after req accepted)
    output logic        resp_valid,  // Response valid (1 cycle after accepted request)

    // UART external pins (directly exposed for I/O stub)
    output logic [7:0]  uart_tx_data,
    output logic        uart_tx_valid,
    input  logic        uart_tx_ready,
    input  logic [7:0]  uart_rx_data,
    input  logic        uart_rx_valid,
    output logic        uart_rx_ack
);

    // =========================================================================
    // Address decode constants
    // =========================================================================
    localparam int IMEM_DEPTH = 16384;  // 64KB / 4 bytes = 16K words
    localparam int DMEM_DEPTH = 16384;  // 64KB / 4 bytes = 16K words

    // =========================================================================
    // Address decode
    // =========================================================================
    logic sel_imem;
    logic sel_dmem;
    logic sel_uart;

    logic [13:0] word_addr;  // 14-bit word address (16K words)

    always_comb begin
        sel_imem = (req_addr[31:28] == 4'h0) && !req_addr[16];
        sel_dmem = (req_addr[31:28] == 4'h0) &&  req_addr[16];
        sel_uart = (req_addr[31:28] == 4'h1);
        word_addr = req_addr[15:2];  // Byte addr to word addr (drop bit[1:0])
    end

    // =========================================================================
    // BRAM always enabled when request is valid and targeting that region
    // =========================================================================
    logic imem_en;
    logic dmem_en;
    logic dmem_we;

    always_comb begin
        imem_en = req_valid && sel_imem;
        dmem_en = req_valid && sel_dmem;
        // Instruction memory is read-only from CPU side
        dmem_we = req_valid && sel_dmem && req_we;
    end

    // =========================================================================
    // Instruction memory — single-port BRAM (read-only from CPU)
    // =========================================================================
    logic [31:0] imem_rdata;

    bram_sp #(
        .DEPTH (IMEM_DEPTH),
        .WIDTH (32)
    ) u_imem (
        .clk   (clk),
        .en    (imem_en),
        .we    (1'b0),          // CPU cannot write instruction memory
        .addr  (word_addr),
        .wdata (32'd0),
        .rdata (imem_rdata)
    );

    // =========================================================================
    // Data memory — dual-port BRAM (Port A = CPU, Port B = debug)
    // =========================================================================
    logic [31:0] dmem_rdata;

    bram_dp #(
        .DEPTH (DMEM_DEPTH),
        .WIDTH (32)
    ) u_dmem (
        .clk     (clk),
        // Port A — CPU
        .en_a    (dmem_en),
        .we_a    (dmem_we),
        .addr_a  (word_addr),
        .wdata_a (req_wdata),
        .rdata_a (dmem_rdata),
        // Port B — unused / debug (active-low, tie off)
        .en_b    (1'b0),
        .addr_b  (14'd0),
        .rdata_b ()             // Unconnected
    );

    // =========================================================================
    // UART I/O stub (memory-mapped)
    // =========================================================================
    // 0x1000_0000 (offset 0x0): TX data register (write: send byte)
    // 0x1000_0004 (offset 0x4): TX status        (read: bit[0] = tx_ready)
    // 0x1000_0008 (offset 0x8): RX data register (read: received byte, auto-ack)
    // 0x1000_000C (offset 0xC): RX status        (read: bit[0] = rx_valid)

    logic [31:0] uart_rdata;
    logic        uart_tx_wr;
    logic        uart_rx_rd;

    always_comb begin
        uart_rdata  = 32'd0;
        uart_tx_wr  = 1'b0;
        uart_rx_rd  = 1'b0;

        if (sel_uart) begin
            case (req_addr[3:2])
                2'd0: begin  // TX data — write-only
                    uart_tx_wr = req_valid && req_we;
                end
                2'd1: begin  // TX status — read-only
                    uart_rdata = {31'd0, uart_tx_ready};
                end
                2'd2: begin  // RX data — read-only (auto-ack)
                    uart_rdata = {24'd0, uart_rx_data};
                    uart_rx_rd = req_valid && !req_we;
                end
                2'd3: begin  // RX status — read-only
                    uart_rdata = {31'd0, uart_rx_valid};
                end
                default: ;
            endcase
        end
    end

    // UART TX output — directly drive from write
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            uart_tx_valid <= 1'b0;
            uart_tx_data  <= 8'd0;
        end else begin
            if (uart_tx_wr) begin
                uart_tx_data  <= req_wdata[7:0];
                uart_tx_valid <= 1'b1;
            end else if (uart_tx_ready) begin
                uart_tx_valid <= 1'b0;
            end
        end
    end

    // UART RX ack — pulse on read of RX data register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            uart_rx_ack <= 1'b0;
        end else begin
            uart_rx_ack <= uart_rx_rd;
        end
    end

    // =========================================================================
    // Ready signal — always accept in 1 cycle (no wait states)
    // =========================================================================
    assign req_ready = 1'b1;

    // =========================================================================
    // Response mux — select read data based on which region was accessed
    // =========================================================================
    // Pipeline the select signals 1 cycle to match BRAM read latency.
    logic sel_imem_r, sel_dmem_r, sel_uart_r;
    logic req_valid_r;
    logic [31:0] uart_rdata_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sel_imem_r   <= 1'b0;
            sel_dmem_r   <= 1'b0;
            sel_uart_r   <= 1'b0;
            req_valid_r  <= 1'b0;
            uart_rdata_r <= 32'd0;
        end else begin
            sel_imem_r   <= sel_imem;
            sel_dmem_r   <= sel_dmem;
            sel_uart_r   <= sel_uart;
            req_valid_r  <= req_valid;
            uart_rdata_r <= uart_rdata;  // UART data is combinational, latch it
        end
    end

    always_comb begin
        resp_valid = req_valid_r;
        if (sel_imem_r)
            resp_rdata = imem_rdata;
        else if (sel_dmem_r)
            resp_rdata = dmem_rdata;
        else if (sel_uart_r)
            resp_rdata = uart_rdata_r;
        else
            resp_rdata = 32'd0;  // Bus error / unmapped
    end

endmodule
