// =============================================================================
// qug_soc_top.sv -- Full SoC Top-Level
// QUG-V1 Mining SoC -- Parameterized Multi-Tile SoC
// =============================================================================
// Project  : QUG-V1 RISC-V Mining SoC
// Target   : Xilinx Kintex-7 XC7K325T (FPGA prototype)
// Author   : Quillon Foundation / Dragon Ball Miner
// License  : MIT
// =============================================================================
//
// Parameterized SoC.  For FPGA prototype: NUM_TILES=1.
//
// Contains:
//   - NUM_TILES x qug_tile (RISC-V core + Xcrypto + Xlattice)
//   - mem_subsystem (shared BRAM + UART I/O, arbitrated for multi-tile)
//   - Heartbeat counter (toggles LED every ~0.5s at 100 MHz)
//   - Boot logic: core fetches first instruction from address 0x0000_0000
//
// Memory map (from mem_subsystem):
//   0x0000_0000 - 0x0000_FFFF : Instruction BRAM (64KB)
//   0x0001_0000 - 0x0001_FFFF : Data BRAM        (64KB)
//   0x1000_0000 - 0x1000_000F : UART I/O
//
// =============================================================================

module qug_soc_top
    import qug_pkg::*;
#(
    parameter int unsigned NUM_TILES = 1
) (
    input  logic       clk,
    input  logic       rst_n,

    // UART
    output logic       uart_tx,
    input  logic       uart_rx,

    // Status LEDs
    output logic [7:0] led,

    // Debug heartbeat
    output logic       heartbeat
);

    // =========================================================================
    // Heartbeat counter -- toggle every ~0.5s at 100 MHz (25M cycles)
    // =========================================================================
    localparam int unsigned HEARTBEAT_DIV = CLK_FREQ_HZ / 2;  // 50_000_000

    logic [25:0] hb_cnt;
    logic        hb_toggle;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hb_cnt    <= '0;
            hb_toggle <= 1'b0;
        end else begin
            if (hb_cnt == HEARTBEAT_DIV[25:0] - 1) begin
                hb_cnt    <= '0;
                hb_toggle <= ~hb_toggle;
            end else begin
                hb_cnt <= hb_cnt + 1;
            end
        end
    end

    assign heartbeat = hb_toggle;
    assign led[0]    = hb_toggle;     // LED 0 = heartbeat
    assign led[7:1]  = '0;            // Remaining LEDs reserved

    // =========================================================================
    // UART TX/RX -- simple 8N1 shift register (directly exposed)
    // =========================================================================
    // For FPGA prototype we use a minimal UART to keep LUT count low.
    // The mem_subsystem exposes a byte-wide data/valid/ready interface;
    // the actual serialisation to the pin is done here.

    // --- UART TX ---
    localparam int unsigned UART_BAUD    = 115200;
    localparam int unsigned UART_DIV     = CLK_FREQ_HZ / UART_BAUD;  // ~868 @ 100 MHz
    localparam int unsigned UART_DIV_W   = $clog2(UART_DIV + 1);

    logic [7:0]            tx_data_in;
    logic                  tx_valid_in;
    logic                  tx_ready_out;

    logic [UART_DIV_W-1:0] tx_baud_cnt;
    logic [3:0]            tx_bit_cnt;     // 0=idle, 1=start, 2..9=data, 10=stop
    logic [9:0]            tx_shift;       // {stop, data[7:0], start}
    logic                  tx_active;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_baud_cnt <= '0;
            tx_bit_cnt  <= '0;
            tx_shift    <= 10'h3FF;        // Idle = all 1s
            tx_active   <= 1'b0;
            uart_tx     <= 1'b1;           // Idle high
        end else begin
            if (!tx_active) begin
                uart_tx <= 1'b1;
                if (tx_valid_in) begin
                    // Load shift register: {stop=1, data[7:0], start=0}
                    tx_shift    <= {1'b1, tx_data_in, 1'b0};
                    tx_active   <= 1'b1;
                    tx_bit_cnt  <= 4'd0;
                    tx_baud_cnt <= '0;
                end
            end else begin
                if (tx_baud_cnt == UART_DIV[UART_DIV_W-1:0] - 1) begin
                    tx_baud_cnt <= '0;
                    uart_tx     <= tx_shift[0];
                    tx_shift    <= {1'b1, tx_shift[9:1]};
                    tx_bit_cnt  <= tx_bit_cnt + 1;
                    if (tx_bit_cnt == 4'd9) begin
                        tx_active <= 1'b0;
                    end
                end else begin
                    tx_baud_cnt <= tx_baud_cnt + 1;
                end
            end
        end
    end

    assign tx_ready_out = ~tx_active;

    // --- UART RX ---
    logic [7:0]            rx_data_out;
    logic                  rx_valid_out;
    logic                  rx_ack_in;

    logic [UART_DIV_W-1:0] rx_baud_cnt;
    logic [3:0]            rx_bit_cnt;
    logic [7:0]            rx_shift;
    logic                  rx_active;
    logic                  rx_sync_0, rx_sync_1;  // Metastability synchronizer
    logic                  rx_sample;
    logic                  rx_data_ready;

    // Double-flop synchronizer for async RX input
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_sync_0 <= 1'b1;
            rx_sync_1 <= 1'b1;
        end else begin
            rx_sync_0 <= uart_rx;
            rx_sync_1 <= rx_sync_0;
        end
    end

    assign rx_sample = rx_sync_1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_baud_cnt   <= '0;
            rx_bit_cnt    <= '0;
            rx_shift      <= 8'd0;
            rx_active     <= 1'b0;
            rx_data_out   <= 8'd0;
            rx_data_ready <= 1'b0;
        end else begin
            // Clear data ready on ack
            if (rx_ack_in)
                rx_data_ready <= 1'b0;

            if (!rx_active) begin
                // Detect start bit (falling edge)
                if (!rx_sample) begin
                    rx_active   <= 1'b1;
                    rx_baud_cnt <= UART_DIV[UART_DIV_W-1:0] >> 1;  // Sample mid-bit
                    rx_bit_cnt  <= 4'd0;
                end
            end else begin
                if (rx_baud_cnt == UART_DIV[UART_DIV_W-1:0] - 1) begin
                    rx_baud_cnt <= '0;
                    rx_bit_cnt  <= rx_bit_cnt + 1;

                    if (rx_bit_cnt == 4'd0) begin
                        // Verify start bit
                        if (rx_sample) begin
                            // False start, abort
                            rx_active <= 1'b0;
                        end
                    end else if (rx_bit_cnt >= 4'd1 && rx_bit_cnt <= 4'd8) begin
                        // Data bits (LSB first)
                        rx_shift <= {rx_sample, rx_shift[7:1]};
                    end else if (rx_bit_cnt == 4'd9) begin
                        // Stop bit -- latch received byte
                        if (rx_sample) begin
                            rx_data_out   <= rx_shift;
                            rx_data_ready <= 1'b1;
                        end
                        rx_active <= 1'b0;
                    end
                end else begin
                    rx_baud_cnt <= rx_baud_cnt + 1;
                end
            end
        end
    end

    assign rx_valid_out = rx_data_ready;

    // =========================================================================
    // Tile memory bus signals (per-tile)
    // =========================================================================
    logic [31:0] tile_imem_addr  [NUM_TILES];
    logic [31:0] tile_imem_rdata [NUM_TILES];
    logic        tile_imem_req   [NUM_TILES];
    logic        tile_imem_gnt   [NUM_TILES];

    logic [31:0] tile_dmem_addr  [NUM_TILES];
    logic [31:0] tile_dmem_wdata [NUM_TILES];
    logic [31:0] tile_dmem_rdata [NUM_TILES];
    logic [3:0]  tile_dmem_we    [NUM_TILES];
    logic        tile_dmem_req   [NUM_TILES];
    logic        tile_dmem_gnt   [NUM_TILES];

    // =========================================================================
    // Tile instantiation
    // =========================================================================
    generate
        for (genvar t = 0; t < NUM_TILES; t++) begin : gen_tiles
            qug_tile #(
                .TILE_ID (t)
            ) u_tile (
                .clk        (clk),
                .rst_n      (rst_n),

                .imem_addr  (tile_imem_addr[t]),
                .imem_rdata (tile_imem_rdata[t]),
                .imem_req   (tile_imem_req[t]),
                .imem_gnt   (tile_imem_gnt[t]),

                .dmem_addr  (tile_dmem_addr[t]),
                .dmem_wdata (tile_dmem_wdata[t]),
                .dmem_rdata (tile_dmem_rdata[t]),
                .dmem_we    (tile_dmem_we[t]),
                .dmem_req   (tile_dmem_req[t]),
                .dmem_gnt   (tile_dmem_gnt[t])
            );
        end
    endgenerate

    // =========================================================================
    // Memory arbiter (round-robin for NUM_TILES > 1, pass-through for 1)
    // =========================================================================
    // Merged bus to mem_subsystem
    logic        mem_req_valid;
    logic        mem_req_ready;
    logic [31:0] mem_req_addr;
    logic [31:0] mem_req_wdata;
    logic        mem_req_we;
    logic [31:0] mem_resp_rdata;
    logic        mem_resp_valid;

    generate
        if (NUM_TILES == 1) begin : gen_single_tile
            // -----------------------------------------------------------------
            // Single tile: direct connection, no arbiter needed
            // -----------------------------------------------------------------
            // Instruction fetch path -- map imem_req to mem bus for reads
            // Data path -- map dmem_req for reads and writes
            //
            // Simple priority: dmem has priority over imem (data hazards stall
            // fetch, which is normal for in-order pipelines).
            // -----------------------------------------------------------------

            logic imem_pending;
            logic dmem_pending;

            assign dmem_pending = tile_dmem_req[0];
            assign imem_pending = tile_imem_req[0] && !dmem_pending;

            // Mux request to mem_subsystem
            always_comb begin
                if (dmem_pending) begin
                    mem_req_valid = 1'b1;
                    mem_req_addr  = tile_dmem_addr[0];
                    mem_req_wdata = tile_dmem_wdata[0];
                    mem_req_we    = |tile_dmem_we[0];
                end else if (imem_pending) begin
                    mem_req_valid = 1'b1;
                    mem_req_addr  = tile_imem_addr[0];
                    mem_req_wdata = 32'd0;
                    mem_req_we    = 1'b0;
                end else begin
                    mem_req_valid = 1'b0;
                    mem_req_addr  = 32'd0;
                    mem_req_wdata = 32'd0;
                    mem_req_we    = 1'b0;
                end
            end

            // Grant signals
            assign tile_dmem_gnt[0]   = dmem_pending && mem_req_ready;
            assign tile_imem_gnt[0]   = imem_pending && mem_req_ready;

            // Response routing -- track which port made the last accepted request
            logic last_was_dmem_r;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    last_was_dmem_r <= 1'b0;
                else if (mem_req_valid && mem_req_ready)
                    last_was_dmem_r <= dmem_pending;
            end

            assign tile_dmem_rdata[0] = mem_resp_rdata;
            assign tile_imem_rdata[0] = mem_resp_rdata;

        end else begin : gen_multi_tile
            // -----------------------------------------------------------------
            // Multi-tile: round-robin arbiter
            // -----------------------------------------------------------------
            // For multi-tile ASIC targets.  FPGA prototype uses NUM_TILES=1
            // so this path is not synthesized in the prototype.
            // -----------------------------------------------------------------

            logic [$clog2(NUM_TILES)-1:0] arb_sel;
            logic [$clog2(NUM_TILES)-1:0] arb_sel_r;

            // Simple round-robin priority
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    arb_sel <= '0;
                else if (mem_req_valid && mem_req_ready) begin
                    if (arb_sel == NUM_TILES[$clog2(NUM_TILES)-1:0] - 1)
                        arb_sel <= '0;
                    else
                        arb_sel <= arb_sel + 1;
                end
            end

            // Track granted tile for response routing
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    arb_sel_r <= '0;
                else if (mem_req_valid && mem_req_ready)
                    arb_sel_r <= arb_sel;
            end

            // Mux selected tile's dmem port to mem_subsystem
            // (instruction fetches happen through dmem addr range in multi-tile
            //  -- all tiles share the same BRAM program image)
            always_comb begin
                mem_req_valid = tile_dmem_req[arb_sel];
                mem_req_addr  = tile_dmem_addr[arb_sel];
                mem_req_wdata = tile_dmem_wdata[arb_sel];
                mem_req_we    = |tile_dmem_we[arb_sel];
            end

            // Grant and response routing
            for (genvar t = 0; t < NUM_TILES; t++) begin : gen_arb_route
                assign tile_dmem_gnt[t]   = (arb_sel == t[$clog2(NUM_TILES)-1:0]) &&
                                             mem_req_valid && mem_req_ready;
                assign tile_dmem_rdata[t] = mem_resp_rdata;

                // For multi-tile, imem uses a separate read port of the
                // dual-port BRAM.  For now, grant immediately (BRAM is
                // always ready) and return data next cycle.
                assign tile_imem_gnt[t]   = tile_imem_req[t];
                assign tile_imem_rdata[t] = mem_resp_rdata;  // Simplified
            end
        end
    endgenerate

    // =========================================================================
    // Memory subsystem
    // =========================================================================
    logic [7:0]  uart_subsys_tx_data;
    logic        uart_subsys_tx_valid;
    logic        uart_subsys_tx_ready;
    logic [7:0]  uart_subsys_rx_data;
    logic        uart_subsys_rx_valid;
    logic        uart_subsys_rx_ack;

    mem_subsystem u_mem (
        .clk           (clk),
        .rst_n         (rst_n),

        .req_valid     (mem_req_valid),
        .req_ready     (mem_req_ready),
        .req_addr      (mem_req_addr),
        .req_wdata     (mem_req_wdata),
        .req_we        (mem_req_we),

        .resp_rdata    (mem_resp_rdata),
        .resp_valid    (mem_resp_valid),

        .uart_tx_data  (uart_subsys_tx_data),
        .uart_tx_valid (uart_subsys_tx_valid),
        .uart_tx_ready (uart_subsys_tx_ready),
        .uart_rx_data  (uart_subsys_rx_data),
        .uart_rx_valid (uart_subsys_rx_valid),
        .uart_rx_ack   (uart_subsys_rx_ack)
    );

    // Connect UART subsystem signals to UART TX/RX logic
    assign tx_data_in          = uart_subsys_tx_data;
    assign tx_valid_in         = uart_subsys_tx_valid;
    assign uart_subsys_tx_ready = tx_ready_out;
    assign uart_subsys_rx_data  = rx_data_out;
    assign uart_subsys_rx_valid = rx_valid_out;
    assign rx_ack_in            = uart_subsys_rx_ack;

endmodule : qug_soc_top
