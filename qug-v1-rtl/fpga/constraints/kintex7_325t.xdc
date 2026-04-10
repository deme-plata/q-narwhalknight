## =============================================================================
## kintex7_325t.xdc -- Pin Constraints for Kintex-7 XC7K325T-FFG900-2
## QUG-V1 Mining SoC -- FPGA Constraints
## =============================================================================
## Project  : QUG-V1 RISC-V Mining SoC
## Target   : Xilinx Kintex-7 XC7K325T-FFG900-2
## Author   : Quillon Foundation / Dragon Ball Miner
## License  : MIT
## =============================================================================
##
## Generic pin assignments for Kintex-7 XC7K325T.
## Dragon Ball will adapt these to their specific board layout.
##
## IMPORTANT: Verify all PACKAGE_PIN assignments against your board schematic
## before synthesis.  The pins below are representative examples.
##
## =============================================================================

## =============================================================================
## Clock Input (100 MHz single-ended oscillator)
## =============================================================================
## Adjust PACKAGE_PIN to match your board's oscillator output.

set_property PACKAGE_PIN E3  [get_ports clk_in]
set_property IOSTANDARD LVCMOS33 [get_ports clk_in]

create_clock -period 10.000 -name sys_clk -waveform {0.000 5.000} [get_ports clk_in]

## =============================================================================
## Reset Button (active-low, directly connected to FPGA pin)
## =============================================================================

set_property PACKAGE_PIN C12 [get_ports rst_n_in]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n_in]

## =============================================================================
## UART
## =============================================================================
## Connect to USB-UART bridge or header pins on your board.

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

## MMCM output clock (auto-derived from sys_clk through MMCM, but set
## false path from async reset to avoid spurious timing violations)
set_false_path -from [get_ports rst_n_in]

## UART is asynchronous -- set false path on RX input
set_false_path -from [get_ports uart_rx]

## =============================================================================
## Configuration and Bitstream Settings
## =============================================================================

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

## SPI flash programming (adjust for your board)
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 33 [current_design]
set_property CONFIG_MODE SPIx4 [current_design]

## Internal voltage monitoring (recommended)
set_property BITSTREAM.CONFIG.UNUSEDPIN PULLDOWN [current_design]

## =============================================================================
## Power Analysis
## =============================================================================
## Set realistic switching activity for power estimation

set_switching_activity -toggle_rate 12.5 -static_probability 0.5 [get_ports clk_in]
