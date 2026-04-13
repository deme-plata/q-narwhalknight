# QUG-V1 Mining SoC -- SystemVerilog RTL

## Overview

QUG-V1 is a 16-core RISC-V (RV32IMC) mining system-on-chip designed for the
Quillon blockchain. Each core includes two custom ISA extensions:

- **Xcrypto** -- Hardware-accelerated BLAKE3 hashing (custom-0, opcode `0x0B`)
- **Xlattice** -- NTT/polynomial arithmetic for post-quantum lattice operations
  (custom-1, opcode `0x2B`)

The SoC is organized as a 4x4 mesh-of-tiles interconnected by a lightweight
network-on-chip. Each tile contains one RISC-V core, private L1I/L1D caches
(8 KB each), and dedicated Xcrypto + Xlattice execution units.

**Target platform:** Xilinx Kintex-7 XC7K325T FPGA at 100 MHz (prototype).
ASIC tapeout planned for TSMC 7nm in collaboration with Dragon Ball Miner.

## Directory Structure

```
qug-v1-rtl/
  README.md                  This file
  doc/
    architecture.md          SoC architecture, ISA encoding, memory map
    coding-guidelines.md     SystemVerilog coding standards
  rtl/
    pkg/
      qug_pkg.sv             Global parameters, types, AXI4-Lite structs
      xcrypto_pkg.sv         BLAKE3 constants, Xcrypto instruction encodings
    core/                    RISC-V pipeline stages (implemented)
      qug_pipeline.sv        7-stage in-order pipeline with forwarding
    memory/                  Memory subsystem (implemented)
      bram_sp.sv             Single-port synchronous BRAM
      bram_dp.sv             True dual-port synchronous BRAM
    xcrypto/                 BLAKE3 execution unit (implemented)
      blake3_round.sv        2-stage pipelined BLAKE3 round
      xcrypto_unit.sv        Top-level Xcrypto extension unit with FSM
    xlattice/                (planned) NTT/polynomial execution unit
    tile/                    (planned) Tile wrapper (core + L1 + extensions)
    mesh/                    (planned) 4x4 mesh NoC + L2
    soc/                     (planned) Top-level SoC (tiles + peripherals)
  tb/                        (planned) Verilator / cocotb testbenches
  syn/
    vivado/                  (planned) Vivado project and constraints (XDC)
  sw/
    tests/                   (planned) Bare-metal ISA compliance tests
    blake3/                  (planned) BLAKE3 reference / mining firmware
```

## ISA Summary

### Xcrypto Extension (custom-0, opcode 0x0B, funct3=000)

| Mnemonic         | funct7  | Operands      | Description                                |
|------------------|---------|---------------|--------------------------------------------|
| `blake3.init`    | 0x00    | rs1           | Load IV into state, reset pipeline (no rd writeback) |
| `blake3.round`   | 0x01    | rd, rs1, rs2  | Fetch message from memory[rs1], run 7-round compression |
| `blake3.chain`   | 0x02    | rd, rs1, rs2  | VDF chain: rs1=msg addr, rs2[6:0]=chain length |
| `blake3.finalize`| 0x03    | rd, rs1       | Read hash word rs1[3:0] into rd            |
| `blake3.ldmsg`   | 0x04    | rd, rs1       | Load 64-byte message block from memory[rs1]|
| `blake3.status`  | 0x05    | rd            | Read engine status register to rd          |

### Xlattice Extension (custom-1, opcode 0x2B, funct3=000)

| Mnemonic         | funct7  | Operands      | Description                                |
|------------------|---------|---------------|--------------------------------------------|
| `ntt.fwd`        | 0x00    | rd, rs1, rs2  | Forward NTT on 256-element polynomial      |
| `ntt.inv`        | 0x01    | rd, rs1, rs2  | Inverse NTT                                |
| `poly.add`       | 0x02    | rd, rs1, rs2  | Coefficient-wise polynomial addition mod q |
| `poly.mul`       | 0x03    | rd, rs1, rs2  | Coefficient-wise polynomial multiply mod q |
| `poly.reduce`    | 0x04    | rd, rs1       | Barrett reduction mod q on all coefficients|

## Build Instructions

### Prerequisites

- **Verilator** >= 5.006 (simulation)
- **Vivado** >= 2023.2 (FPGA synthesis for Kintex-7)
- **Python** >= 3.10 with cocotb (optional, for advanced testbenches)
- **RISC-V GCC toolchain** (`riscv32-unknown-elf-gcc`) for firmware compilation

### Verilator Simulation

```bash
# Lint check (no simulation, just syntax + elaboration)
verilator --lint-only -sv \
  rtl/pkg/qug_pkg.sv \
  rtl/pkg/xcrypto_pkg.sv

# Full simulation (when testbench is available)
verilator --cc --exe --build -sv \
  -Irtl/pkg \
  rtl/pkg/qug_pkg.sv \
  rtl/pkg/xcrypto_pkg.sv \
  rtl/core/*.sv \
  rtl/xcrypto/*.sv \
  tb/tb_top.cpp \
  -o qug_v1_sim

./obj_dir/qug_v1_sim
```

### Vivado Synthesis (Kintex-7 XC7K325T)

```bash
cd syn/vivado
vivado -mode batch -source build.tcl
# Output: qug_v1_top.bit in syn/vivado/output/
```

### FPGA Resource Estimates (Kintex-7 XC7K325T)

| Resource     | Available | Estimated Use | Utilization |
|--------------|-----------|---------------|-------------|
| LUTs         | 203,800   | ~145,000      | ~71%        |
| FFs          | 407,600   | ~98,000       | ~24%        |
| BRAM (36Kb)  | 445       | ~280          | ~63%        |
| DSP48E1      | 840       | ~320          | ~38%        |

## Performance Targets

| Metric                 | Target              |
|------------------------|---------------------|
| Clock frequency        | 100 MHz (FPGA)      |
| BLAKE3 throughput      | 1 hash / 14 cycles  |
| Mining hashrate (16c)  | ~114 MH/s (FPGA)    |
| NTT-256 latency        | ~512 cycles          |
| Power (FPGA estimate)  | ~12 W                |

## Collaboration

This project is a joint effort between the **Quillon Foundation** and
**Dragon Ball Miner** (ASIC manufacturer). The RTL is developed for FPGA
prototyping on Kintex-7 before ASIC tapeout.

## Contact

- **Quillon Foundation** -- https://quillon.xyz
- **Repository** -- code.quillon.xyz

## License

MIT License. See individual source files for copyright headers.
