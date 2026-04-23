# QUG-V1 RTL Simulation Makefile
# Supports: Verilator (free), Icarus Verilog (free), Vivado Simulator (Xilinx)

RTL_DIR  := ../rtl
TB_DIR   := ../tb
PKG_DIR  := $(RTL_DIR)/pkg
CORE_DIR := $(RTL_DIR)/core
XCRY_DIR := $(RTL_DIR)/xcrypto

# SystemVerilog source files (order matters for packages)
PKG_FILES := $(PKG_DIR)/qug_pkg.sv $(PKG_DIR)/xcrypto_pkg.sv $(CORE_DIR)/qug_core_pkg.sv
XCRY_FILES := $(XCRY_DIR)/blake3_round.sv $(XCRY_DIR)/blake3_state.sv $(XCRY_DIR)/blake3_pipeline.sv $(XCRY_DIR)/xcrypto_unit.sv
CORE_FILES := $(CORE_DIR)/qug_regfile.sv $(CORE_DIR)/qug_alu.sv $(CORE_DIR)/qug_decoder.sv $(CORE_DIR)/qug_pipeline.sv $(CORE_DIR)/qug_core.sv

ALL_RTL := $(PKG_FILES) $(XCRY_FILES) $(CORE_FILES)

# Default target
.PHONY: all clean blake3 core help

all: blake3 core

help:
	@echo "QUG-V1 RTL Simulation"
	@echo "  make blake3    - Run BLAKE3 pipeline testbench (iverilog)"
	@echo "  make core      - Run RISC-V core testbench (iverilog)"
	@echo "  make all       - Run all tests"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make waves     - Open waveform viewer (GTKWave)"

# ═══════════════════════════════════════
# Icarus Verilog targets
# ═══════════════════════════════════════
blake3: $(TB_DIR)/blake3_tb.sv $(XCRY_FILES) $(PKG_FILES)
	@echo "═══ BLAKE3 Pipeline Test ═══"
	iverilog -g2012 -o blake3_tb.vvp \
		$(PKG_DIR)/qug_pkg.sv $(PKG_DIR)/xcrypto_pkg.sv \
		$(XCRY_FILES) $(TB_DIR)/blake3_tb.sv
	vvp blake3_tb.vvp
	@echo ""

core: $(TB_DIR)/core_tb.sv $(CORE_FILES) $(PKG_FILES)
	@echo "═══ RISC-V Core Test ═══"
	iverilog -g2012 -o core_tb.vvp \
		$(PKG_DIR)/qug_pkg.sv $(PKG_DIR)/xcrypto_pkg.sv $(CORE_DIR)/qug_core_pkg.sv \
		$(CORE_FILES) $(TB_DIR)/core_tb.sv
	vvp core_tb.vvp
	@echo ""

waves:
	gtkwave *.vcd &

clean:
	rm -f *.vvp *.vcd *.log
