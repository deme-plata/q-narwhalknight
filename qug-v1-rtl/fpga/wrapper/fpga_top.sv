// =============================================================================
// fpga_top.sv -- FPGA Top-Level for Kintex-7
// QUG-V1 Mining SoC -- FPGA Wrapper
// =============================================================================
// Project  : QUG-V1 RISC-V Mining SoC
// Target   : Xilinx Kintex-7 XC7K325T-FFG900-2
// Author   : Quillon Foundation / Dragon Ball Miner
// License  : MIT
// =============================================================================
//
// Wraps qug_soc_top with:
//   - Xilinx MMCM for clock generation (100 MHz from external oscillator)
//   - Reset synchronizer (async assert, sync deassert)
//   - Pin assignments for UART (directly to FPGA pins)
//   - LED output for heartbeat/status
//
// External clock input: differential or single-ended, board-dependent.
// The MMCM generates a clean 100 MHz clock for the SoC domain.
//
// =============================================================================

module fpga_top (
    // External clock (from board oscillator)
    input  logic       clk_in,

    // Active-low reset button
    input  logic       rst_n_in,

    // UART pins
    output logic       uart_tx,
    input  logic       uart_rx,

    // Status LEDs
    output logic [7:0] led
);

    // =========================================================================
    // Clock generation -- Xilinx MMCM / PLL
    // =========================================================================
    // Input:  board oscillator (assumed 100 MHz single-ended)
    // Output: 100 MHz system clock, aligned and jitter-cleaned
    //
    // For boards with a different input frequency, adjust CLKFBOUT_MULT_F
    // and CLKOUT0_DIVIDE_F accordingly:
    //   f_VCO  = f_in * CLKFBOUT_MULT_F / DIVCLK_DIVIDE
    //   f_out  = f_VCO / CLKOUT0_DIVIDE_F
    //
    // Example (100 MHz in, 100 MHz out):
    //   f_VCO = 100 * 10.0 / 1 = 1000 MHz
    //   f_out = 1000 / 10.0    = 100 MHz

    logic clk_mmcm;
    logic clk_fb;
    logic mmcm_locked;

    // Buffered input clock
    logic clk_ibuf;

    (* DONT_TOUCH = "true" *)
    IBUF u_clk_ibuf (
        .I  (clk_in),
        .O  (clk_ibuf)
    );

    (* DONT_TOUCH = "true" *)
    MMCME2_ADV #(
        .BANDWIDTH           ("OPTIMIZED"),
        .CLKFBOUT_MULT_F     (10.0),        // VCO = 100 * 10 = 1000 MHz
        .CLKFBOUT_PHASE       (0.0),
        .CLKIN1_PERIOD        (10.0),        // 100 MHz input = 10 ns period
        .CLKOUT0_DIVIDE_F     (10.0),        // 1000 / 10 = 100 MHz output
        .CLKOUT0_DUTY_CYCLE   (0.5),
        .CLKOUT0_PHASE        (0.0),
        .DIVCLK_DIVIDE        (1),
        .REF_JITTER1          (0.010),
        .STARTUP_WAIT         ("FALSE")
    ) u_mmcm (
        .CLKFBOUT     (clk_fb),
        .CLKFBOUTB    (),
        .CLKOUT0      (clk_mmcm),
        .CLKOUT0B     (),
        .CLKOUT1      (),
        .CLKOUT1B     (),
        .CLKOUT2      (),
        .CLKOUT2B     (),
        .CLKOUT3      (),
        .CLKOUT3B     (),
        .CLKOUT4      (),
        .CLKOUT5      (),
        .CLKOUT6      (),
        .LOCKED       (mmcm_locked),
        // Input clocks
        .CLKFBIN      (clk_fb),
        .CLKIN1       (clk_ibuf),
        .CLKIN2       (1'b0),
        .CLKINSEL     (1'b1),         // Select CLKIN1
        // Control
        .DADDR        (7'd0),
        .DCLK         (1'b0),
        .DEN          (1'b0),
        .DI           (16'd0),
        .DO           (),
        .DRDY         (),
        .DWE          (1'b0),
        .PSCLK        (1'b0),
        .PSEN         (1'b0),
        .PSINCDEC     (1'b0),
        .PSDONE       (),
        .PWRDWN       (1'b0),
        .RST          (~rst_n_in)     // Reset MMCM when button pressed
    );

    // Global clock buffer for the system clock
    logic sys_clk;

    (* DONT_TOUCH = "true" *)
    BUFG u_bufg_clk (
        .I  (clk_mmcm),
        .O  (sys_clk)
    );

    // =========================================================================
    // Reset synchronizer (async assert, sync deassert)
    // =========================================================================
    // Ensures the internal reset deasserts synchronously to sys_clk, even
    // though the external reset button is asynchronous.  The reset is held
    // active until the MMCM locks.

    logic rst_sync_0, rst_sync_1, rst_sync_2;
    logic sys_rst_n;

    always_ff @(posedge sys_clk or negedge rst_n_in) begin
        if (!rst_n_in) begin
            rst_sync_0 <= 1'b0;
            rst_sync_1 <= 1'b0;
            rst_sync_2 <= 1'b0;
        end else begin
            rst_sync_0 <= mmcm_locked;    // Only deassert when PLL is locked
            rst_sync_1 <= rst_sync_0;
            rst_sync_2 <= rst_sync_1;
        end
    end

    assign sys_rst_n = rst_sync_2;

    // =========================================================================
    // SoC instantiation (single tile for FPGA prototype)
    // =========================================================================
    logic       soc_uart_tx;
    logic       soc_uart_rx;
    logic [7:0] soc_led;
    logic       soc_heartbeat;

    (* DONT_TOUCH = "true" *)
    qug_soc_top #(
        .NUM_TILES (1)
    ) u_soc (
        .clk       (sys_clk),
        .rst_n     (sys_rst_n),
        .uart_tx   (soc_uart_tx),
        .uart_rx   (soc_uart_rx),
        .led       (soc_led),
        .heartbeat (soc_heartbeat)
    );

    // =========================================================================
    // I/O assignments
    // =========================================================================
    assign uart_tx    = soc_uart_tx;
    assign soc_uart_rx = uart_rx;

    // LED[0] = heartbeat (alive indicator)
    // LED[1] = MMCM locked
    // LED[7:2] = SoC status
    assign led[0]   = soc_heartbeat;
    assign led[1]   = mmcm_locked;
    assign led[7:2] = soc_led[7:2];

endmodule : fpga_top
