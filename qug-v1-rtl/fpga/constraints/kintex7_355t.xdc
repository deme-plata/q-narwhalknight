## =============================================================================
## kintex7_355t.xdc -- Pin Constraints for Kintex-7 XC7K355T-FFG901-2
## QUG-V1 Mining SoC -- FPGA Constraints (Dragon Ball Miner Board)
## =============================================================================
## Project  : QUG-V1 RISC-V Mining SoC
## Target   : Xilinx Kintex-7 XC7K355T-FFG901-2
## Author   : Quillon Foundation / Dragon Ball Miner
## License  : MIT
## =============================================================================
##
## Pin assignments for Kintex-7 XC7K355T (FFG901 package).
##
## !!! IMPORTANT !!!
## The XC7K355T uses the FFG901 package (NOT FFG900 like XC7K325T).
## ALL pin assignments below are PLACEHOLDERS.  You MUST verify every
## PACKAGE_PIN against your specific board's schematic before synthesis.
##
## To adapt:
##   1. Open your board schematic (PDF / OrCAD / Altium)
##   2. Find the FPGA ball for each signal (clock osc, USB-UART, LEDs, etc.)
##   3. Replace the PACKAGE_PIN value in each set_property line
##
## =============================================================================

## =============================================================================
## Clock Input (100 MHz single-ended oscillator)
## =============================================================================
## >>> Replace E3 with your board's clock oscillator output pin <<<

set_property PACKAGE_PIN E3  [get_ports clk_in]
set_property IOSTANDARD LVCMOS33 [get_ports clk_in]

create_clock -period 10.000 -name sys_clk -waveform {0.000 5.000} [get_ports clk_in]

## =============================================================================
## Reset Button (active-low, directly connected to FPGA pin)
## =============================================================================
## >>> Replace C12 with your board's reset pushbutton pin <<<

set_property PACKAGE_PIN C12 [get_ports rst_n_in]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n_in]

## =============================================================================
## UART
## =============================================================================
## >>> Replace D10/A9 with your board's USB-UART or header TX/RX pins <<<

set_property PACKAGE_PIN D10 [get_ports uart_tx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx]
set_property SLEW FAST [get_ports uart_tx]
set_property DRIVE 12 [get_ports uart_tx]

set_property PACKAGE_PIN A9  [get_ports uart_rx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rx]

## =============================================================================
## Status LEDs
## =============================================================================
## LED[0] = heartbeat, LED[1] = MMCM locked, LED[7:2] = reserved
## >>> Replace all pin letters/numbers with your board's LED pins <<<

set_property PACKAGE_PIN H17 [get_ports {led[0]}]
set_property PACKAGE_PIN K15 [get_ports {led[1]}]
set_property PACKAGE_PIN J13 [get_ports {led[2]}]
set_property PACKAGE_PIN N14 [get_ports {led[3]}]
set_property PACKAGE_PIN R18 [get_ports {led[4]}]
set_property PACKAGE_PIN V17 [get_ports {led[5]}]
set_property PACKAGE_PIN U17 [get_ports {led[6]}]
set_property PACKAGE_PIN U16 [get_ports {led[7]}]

set_property IOSTANDARD LVCMOS33 [get_ports {led[*]}]

## =============================================================================
## Timing Constraints
## =============================================================================

## MMCM output clock (auto-derived from sys_clk through MMCM)
set_false_path -from [get_ports rst_n_in]

## UART is asynchronous -- set false path on RX input
set_false_path -from [get_ports uart_rx]

## =============================================================================
## Configuration and Bitstream Settings
## =============================================================================

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

## SPI flash programming (adjust for your board's flash)
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 33 [current_design]
set_property CONFIG_MODE SPIx4 [current_design]

## Pull unused pins down to prevent floating
set_property BITSTREAM.CONFIG.UNUSEDPIN PULLDOWN [current_design]

## =============================================================================
## Power Analysis
## =============================================================================

set_switching_activity -toggle_rate 12.5 -static_probability 0.5 [get_ports clk_in]

## =============================================================================
## XC7K355T Resource Budget (for reference)
## =============================================================================
##   LUTs:      226,800
##   Flip-Flops: 453,600
##   BRAM 36Kb:  445
##   DSP48E1:    1,440
##
##   vs XC7K325T:
##     LUTs:    +23,000 (203,800 -> 226,800)
##     FFs:     +46,000 (407,600 -> 453,600)
##     DSP:     +600    (840 -> 1,440)  <-- major upgrade for Xlattice
##     BRAM:    same    (445)
##
##   The XC7K355T has 71% more DSP slices than the 325T, which significantly
##   benefits the mod_mul_256 multiplier chains in the Xlattice unit.
##   This headroom allows exploring 2-tile configurations on a single FPGA.
## =============================================================================
