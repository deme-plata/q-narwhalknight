# ==============================================================================
# synth_vivado.tcl -- Vivado Synthesis & Implementation Script
# QUG-V1 Mining SoC -- FPGA Build Flow
# ==============================================================================
# Project  : QUG-V1 RISC-V Mining SoC
# Target   : Xilinx Kintex-7 XC7K325T-FFG900-2
# Author   : Quillon Foundation / Dragon Ball Miner
# License  : MIT
# ==============================================================================
#
# Usage:
#   vivado -mode batch -source fpga/scripts/synth_vivado.tcl
#
# Or in Vivado GUI:
#   source fpga/scripts/synth_vivado.tcl
#
# Output:
#   build/qug_v1_fpga/    -- Vivado project directory
#   build/reports/         -- Timing, utilization, power reports
#   build/bitstream/       -- Generated bitstream (.bit) file
#
# ==============================================================================

# ==============================================================================
# Configuration
# ==============================================================================
# Supported FPGA parts:
#   xc7k325tffg900-2   -- Kintex-7 325T (FFG900 package)
#   xc7k355tffg901-2   -- Kintex-7 355T (FFG901 package, Dragon Ball board)
#
# Override from command line:
#   vivado -mode batch -source fpga/scripts/synth_vivado.tcl -tclargs xc7k355tffg901-2

set project_name    "qug_v1_fpga"
if { $argc > 0 } {
    set part [lindex $argv 0]
} else {
    set part "xc7k325tffg900-2"
}
set top_module      "fpga_top"
set build_dir       "build"
set report_dir      "${build_dir}/reports"
set bitstream_dir   "${build_dir}/bitstream"

# Source file root (relative to where Vivado is invoked)
set rtl_root        "."

# ==============================================================================
# Create output directories
# ==============================================================================
file mkdir ${build_dir}
file mkdir ${report_dir}
file mkdir ${bitstream_dir}

# ==============================================================================
# Create project (in-memory for batch mode)
# ==============================================================================
puts "================================================================"
puts " QUG-V1 Mining SoC -- Vivado Synthesis Flow"
puts " Target: ${part}"
puts " Top:    ${top_module}"
puts "================================================================"

create_project ${project_name} ${build_dir}/${project_name} -part ${part} -force

# ==============================================================================
# Add RTL sources
# ==============================================================================
puts "Adding RTL sources..."

# Package files (must be compiled first)
add_files -norecurse [list \
    ${rtl_root}/rtl/pkg/qug_pkg.sv \
    ${rtl_root}/rtl/pkg/xcrypto_pkg.sv \
]

# Core
add_files -norecurse [list \
    ${rtl_root}/rtl/core/qug_alu.sv \
    ${rtl_root}/rtl/core/qug_decoder.sv \
    ${rtl_root}/rtl/core/qug_regfile.sv \
    ${rtl_root}/rtl/core/qug_pipeline.sv \
    ${rtl_root}/rtl/core/qug_core.sv \
]

# Xcrypto (BLAKE3 + SHA-3)
add_files -norecurse [list \
    ${rtl_root}/rtl/xcrypto/blake3_round.sv \
    ${rtl_root}/rtl/xcrypto/blake3_state.sv \
    ${rtl_root}/rtl/xcrypto/blake3_pipeline.sv \
    ${rtl_root}/rtl/xcrypto/sha3_state.sv \
    ${rtl_root}/rtl/xcrypto/sha3_keccak.sv \
    ${rtl_root}/rtl/xcrypto/xcrypto_unit.sv \
]

# Xlattice (256-bit field arithmetic)
add_files -norecurse [list \
    ${rtl_root}/rtl/xlattice/mod_add_256.sv \
    ${rtl_root}/rtl/xlattice/mod_sub_256.sv \
    ${rtl_root}/rtl/xlattice/mod_mul_256.sv \
    ${rtl_root}/rtl/xlattice/mod_inv_256.sv \
    ${rtl_root}/rtl/xlattice/xlattice_unit.sv \
]

# Memory
add_files -norecurse [list \
    ${rtl_root}/rtl/memory/bram_sp.sv \
    ${rtl_root}/rtl/memory/bram_dp.sv \
    ${rtl_root}/rtl/memory/xcrypto_scratchpad.sv \
    ${rtl_root}/rtl/memory/mem_subsystem.sv \
]

# Mining controller
add_files -norecurse [list \
    ${rtl_root}/rtl/mining/difficulty_regs.sv \
    ${rtl_root}/rtl/mining/mining_controller.sv \
]

# SoC top-level
add_files -norecurse [list \
    ${rtl_root}/rtl/top/qug_tile.sv \
    ${rtl_root}/rtl/top/qug_soc_top.sv \
]

# FPGA wrapper
add_files -norecurse [list \
    ${rtl_root}/fpga/wrapper/fpga_top.sv \
]

# Set file compile order -- packages first
set_property file_type SystemVerilog [get_files *.sv]

# ==============================================================================
# Add constraints
# ==============================================================================
puts "Adding constraints..."

# Auto-select constraints file based on target part
if { [string match "*355t*" $part] } {
    set xdc_file "${rtl_root}/fpga/constraints/kintex7_355t.xdc"
} else {
    set xdc_file "${rtl_root}/fpga/constraints/kintex7_325t.xdc"
}
puts "Using constraints: ${xdc_file}"
add_files -fileset constrs_1 -norecurse [list ${xdc_file}]

# ==============================================================================
# Set top module and design properties
# ==============================================================================
set_property top ${top_module} [current_fileset]
set_property target_language Verilog [current_project]

# ==============================================================================
# Synthesis settings
# ==============================================================================
puts "Configuring synthesis..."

set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {-mode out_of_context} -objects [get_runs synth_1]

# Override: use default mode (not OOC) for the top-level
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {} -objects [get_runs synth_1]

# Enable retiming for better Fmax
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]

# Keep hierarchy to prevent constant propagation through crypto modules.
# 'rebuilt' flattens the design and allows Vivado to prove xcrypto_valid/xlattice_valid
# are always 0 (BRAM inits to 0 = NOP instructions), removing Xcrypto/Xlattice entirely
# despite dont_touch on module instances. 'none' preserves hierarchy boundaries.
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY none [get_runs synth_1]

# ==============================================================================
# Run Synthesis
# ==============================================================================
puts "================================================================"
puts " Running Synthesis..."
puts "================================================================"

launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Check synthesis status
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
    puts "ERROR: Synthesis failed!"
    exit 1
}

# ==============================================================================
# Post-Synthesis Reports
# ==============================================================================
puts "Generating post-synthesis reports..."

open_run synth_1 -name synth_1

report_timing_summary -delay_type min_max -report_unconstrained \
    -check_timing_verbose -max_paths 10 -input_pins -routable_nets \
    -file ${report_dir}/post_synth_timing_summary.rpt

report_utilization -hierarchical \
    -file ${report_dir}/post_synth_utilization.rpt

report_design_analysis -timing \
    -file ${report_dir}/post_synth_design_analysis.rpt

# ==============================================================================
# Implementation settings
# ==============================================================================
puts "Configuring implementation..."

set_property strategy Performance_ExtraTimingOpt [get_runs impl_1]

# ==============================================================================
# Run Implementation (Place & Route)
# ==============================================================================
puts "================================================================"
puts " Running Implementation (Place & Route)..."
puts "================================================================"

launch_runs impl_1 -jobs 4
wait_on_run impl_1

# Check implementation status
if {[get_property STATUS [get_runs impl_1]] ne "route_design Complete!"} {
    puts "ERROR: Implementation failed!"
    exit 1
}

# ==============================================================================
# Post-Implementation Reports
# ==============================================================================
puts "Generating post-implementation reports..."

open_run impl_1 -name impl_1

report_timing_summary -delay_type min_max -report_unconstrained \
    -check_timing_verbose -max_paths 20 -input_pins -routable_nets \
    -file ${report_dir}/post_impl_timing_summary.rpt

report_utilization -hierarchical \
    -file ${report_dir}/post_impl_utilization.rpt

report_power -advisory \
    -file ${report_dir}/post_impl_power.rpt

report_clock_utilization \
    -file ${report_dir}/post_impl_clock_utilization.rpt

report_drc -file ${report_dir}/post_impl_drc.rpt

report_methodology -file ${report_dir}/post_impl_methodology.rpt

# ==============================================================================
# Check Timing
# ==============================================================================
set wns [get_property STATS.WNS [get_runs impl_1]]
set tns [get_property STATS.TNS [get_runs impl_1]]
set whs [get_property STATS.WHS [get_runs impl_1]]

puts "================================================================"
puts " Timing Results:"
puts "   WNS (Worst Negative Slack):  ${wns} ns"
puts "   TNS (Total Negative Slack):  ${tns} ns"
puts "   WHS (Worst Hold Slack):      ${whs} ns"
puts "================================================================"

if {$wns < 0} {
    puts "WARNING: Timing not met! WNS = ${wns} ns"
    puts "         Consider reducing clock frequency or optimizing critical paths."
}

# ==============================================================================
# Generate Bitstream
# ==============================================================================
puts "================================================================"
puts " Generating Bitstream..."
puts "================================================================"

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# Copy bitstream to output directory
set bit_file [glob -nocomplain ${build_dir}/${project_name}/${project_name}.runs/impl_1/*.bit]
if {$bit_file ne ""} {
    file copy -force $bit_file ${bitstream_dir}/${project_name}.bit
    puts "Bitstream generated: ${bitstream_dir}/${project_name}.bit"
} else {
    puts "WARNING: Bitstream file not found."
}

# ==============================================================================
# Final Summary
# ==============================================================================
puts ""
puts "================================================================"
puts " QUG-V1 Mining SoC -- Build Complete"
puts "================================================================"
puts " Reports:   ${report_dir}/"
puts " Bitstream: ${bitstream_dir}/${project_name}.bit"
puts ""
puts " Key Reports:"
puts "   - post_synth_utilization.rpt     (resource usage after synthesis)"
puts "   - post_impl_timing_summary.rpt   (timing closure)"
puts "   - post_impl_utilization.rpt      (resource usage after P&R)"
puts "   - post_impl_power.rpt            (power estimation)"
puts "================================================================"

# Close project
close_project

puts "Done."
