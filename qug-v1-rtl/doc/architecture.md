# QUG-V1 Mining SoC -- Architecture Document

## 1. SoC Block Diagram

The QUG-V1 is a 16-core tiled architecture arranged in a 4x4 mesh. Each tile
contains a RISC-V RV32IMC core with private L1 caches and dedicated
cryptographic execution units. A shared L2 cache and AXI4-Lite interconnect
provide access to off-chip DRAM and peripherals.

```
                          AXI4-Lite Interconnect
        ┌────────────────────────┬───────────────────────┐
        │                        │                       │
   ┌────┴────┐            ┌──────┴──────┐         ┌──────┴──────┐
   │  UART   │            │  SPI Flash  │         │  DDR3 Ctrl  │
   │ (Debug) │            │  (FW Boot)  │         │  (512 MB)   │
   └─────────┘            └─────────────┘         └─────────────┘
                                │
                         ┌──────┴──────┐
                         │  Shared L2  │
                         │  256 KB     │
                         │  8-way set  │
                         └──────┬──────┘
                                │
            ┌───────────────────┼───────────────────┐
            │           Mesh Network-on-Chip        │
            │            (4x4 XY routing)           │
            │                                       │
   ┌────────┼────────┬────────┬────────┬────────────┤
   │        │        │        │        │            │
 ┌─┴──┐  ┌─┴──┐  ┌─┴──┐  ┌─┴──┐   (12 more      │
 │T00 │──│T01 │──│T02 │──│T03 │    tiles in       │
 └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘    rows 1-3)     │
   │        │        │        │                     │
 ┌─┴──┐  ┌─┴──┐  ┌─┴──┐  ┌─┴──┐                  │
 │T10 │──│T11 │──│T12 │──│T13 │                   │
 └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘                   │
   │        │        │        │                     │
 ┌─┴──┐  ┌─┴──┐  ┌─┴──┐  ┌─┴──┐                  │
 │T20 │──│T21 │──│T22 │──│T23 │                   │
 └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘                   │
   │        │        │        │                     │
 ┌─┴──┐  ┌─┴──┐  ┌─┴──┐  ┌─┴──┐                  │
 │T30 │──│T31 │──│T32 │──│T33 │                   │
 └────┘  └────┘  └────┘  └────┘                   │
            │                                       │
            └───────────────────────────────────────┘
```

### Tile Internal Structure

Each tile (Trc) contains:

```
  ┌─────────────────────────────────────────┐
  │                  Tile                    │
  │                                         │
  │  ┌──────────────────────────────────┐   │
  │  │        RV32IMC Core              │   │
  │  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐    │   │
  │  │  │ IF │→│ ID │→│ IS │→│ EX │──┐ │   │
  │  │  └────┘ └────┘ └────┘ └────┘  │ │   │
  │  │                        ┌────┐  │ │   │
  │  │                     ┌──│ M1 │←─┘ │   │
  │  │                     │  └────┘    │   │
  │  │                     │  ┌────┐    │   │
  │  │                     └─→│ M2 │    │   │
  │  │                        └──┬─┘    │   │
  │  │                        ┌──┴─┐    │   │
  │  │                        │ WB │    │   │
  │  │  ┌──────┐              └────┘    │   │
  │  │  │ RF   │  32x32-bit registers  │   │
  │  │  │ x0-  │                        │   │
  │  │  │ x31  │                        │   │
  │  │  └──────┘                        │   │
  │  └──────────────────────────────────┘   │
  │                                         │
  │  ┌──────────┐       ┌──────────┐        │
  │  │ Xcrypto  │       │ Xlattice │        │
  │  │ (BLAKE3) │       │ (NTT)    │        │
  │  │          │       │          │        │
  │  │ 16x32b   │       │ 256-pt   │        │
  │  │ state RF │       │ butterfly│        │
  │  └──────────┘       └──────────┘        │
  │                                         │
  │  ┌────────┐          ┌────────┐         │
  │  │  L1I   │          │  L1D   │         │
  │  │  8 KB  │          │  8 KB  │         │
  │  │ 2-way  │          │ 2-way  │         │
  │  └────┬───┘          └────┬───┘         │
  │       └─────────┬────────┘              │
  │            ┌────┴────┐                  │
  │            │ NoC Port│                  │
  │            └─────────┘                  │
  └─────────────────────────────────────────┘
```

## 2. ISA Encoding Tables

### 2.1 Xcrypto Instructions (custom-0, opcode = 0000_1011)

All Xcrypto instructions use R-type encoding:

```
  31       25  24   20  19   15  14  12  11    7  6      0
 ┌──────────┬────────┬────────┬───────┬────────┬──────────┐
 │  funct7  │  rs2   │  rs1   │funct3 │   rd   │  opcode  │
 │  [6:0]   │ [4:0]  │ [4:0]  │ [2:0] │ [4:0]  │  [6:0]   │
 └──────────┴────────┴────────┴───────┴────────┴──────────┘
```

| Instruction      | funct7    | rs2   | rs1     | funct3 | rd      | Operation                          |
|------------------|-----------|-------|---------|--------|---------|------------------------------------|
| `blake3.init`    | 000_0000  | 00000 | cv_addr | 000    | status  | State = IV; load CV from mem[rs1]  |
| `blake3.round`   | 000_0001  | rnd#  | msg_adr | 000    | status  | Execute round rs2 with msg[rs1]    |
| `blake3.chain`   | 000_0010  | idx   | 00000   | 000    | word    | rd = chaining_value[rs2]           |
| `blake3.finalize`| 000_0011  | idx   | 00000   | 000    | word    | rd = hash_output[rs2]; finalize    |
| `blake3.ldmsg`   | 000_0100  | 00000 | blk_adr | 000    | status  | Load 64B message from mem[rs1]     |
| `blake3.status`  | 000_0101  | 00000 | 00000   | 000    | status  | rd = {busy, error, valid, ...}     |

### 2.2 Xlattice Instructions (custom-1, opcode = 010_1011)

| Instruction      | funct7    | rs2     | rs1     | funct3 | rd      | Operation                          |
|------------------|-----------|---------|---------|--------|---------|------------------------------------|
| `ntt.fwd`        | 000_0000  | src2    | src1    | 000    | dst     | Forward NTT: dst = NTT(src1)       |
| `ntt.inv`        | 000_0001  | src2    | src1    | 000    | dst     | Inverse NTT: dst = INTT(src1)      |
| `poly.add`       | 000_0010  | src2    | src1    | 001    | dst     | dst[i] = (src1[i]+src2[i]) mod q   |
| `poly.mul`       | 000_0011  | src2    | src1    | 001    | dst     | dst[i] = (src1[i]*src2[i]) mod q   |
| `poly.reduce`    | 000_0100  | 00000   | src1    | 010    | dst     | Barrett reduce all coefficients    |

For NTT/poly operations, rs1 and rs2 point to base addresses of 256-element
coefficient arrays in the Xlattice scratchpad (1024 bytes each, 32-bit coefficients).

## 3. Memory Map

```
  Address Range             Size      Description
  ─────────────────────────────────────────────────────────
  0x0000_0000 - 0x0000_FFFF   64 KB   Boot ROM (SPI flash shadow)
  0x0001_0000 - 0x0001_FFFF   64 KB   On-chip SRAM (stack, globals)
  0x1000_0000 - 0x1FFF_FFFF  256 MB   DDR3 DRAM (via AXI)
  0x2000_0000 - 0x2000_0FFF    4 KB   UART registers
  0x2000_1000 - 0x2000_1FFF    4 KB   SPI controller registers
  0x2000_2000 - 0x2000_2FFF    4 KB   Timer / watchdog
  0x2000_3000 - 0x2000_3FFF    4 KB   Interrupt controller (PLIC)
  0x2000_4000 - 0x2000_4FFF    4 KB   Mining control registers
  0x4000_0000 - 0x4000_003F   64 B    Xcrypto state (per-core, banked)
  0x4000_0040 - 0x4000_007F   64 B    Xcrypto message buffer (per-core)
  0x4000_0080 - 0x4000_009F   32 B    Xcrypto output (per-core)
  0x5000_0000 - 0x5000_03FF    1 KB   Xlattice scratchpad A (per-core)
  0x5000_0400 - 0x5000_07FF    1 KB   Xlattice scratchpad B (per-core)
  0x5000_0800 - 0x5000_0BFF    1 KB   Xlattice scratchpad C (per-core)
  0xF000_0000 - 0xF000_0FFF    4 KB   Debug module (JTAG TAP)
```

## 4. Pipeline Diagram

The QUG-V1 core uses a 7-stage in-order pipeline. The extra M2 stage
accommodates multi-cycle DRAM reads and Xcrypto/Xlattice operations that
stall for completion.

```
  Cycle:  1     2     3     4     5     6     7
         ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐
  Instr: │ IF ││ ID ││ IS ││ EX ││ M1 ││ M2 ││ WB │
         └────┘└────┘└────┘└────┘└────┘└────┘└────┘

  IF  - Instruction Fetch     : PC generation, I-cache access
  ID  - Instruction Decode    : Opcode decode, immediate extraction
  IS  - Issue / Operand Read  : Register file read, hazard check, forwarding
  EX  - Execute               : ALU / branch / Xcrypto dispatch / Xlattice dispatch
  M1  - Memory Access 1       : D-cache tag lookup, address translation
  M2  - Memory Access 2       : D-cache data return, Xcrypto result capture
  WB  - Write-Back            : Register file write, exception commit
```

### Xcrypto Timing

BLAKE3 operations are multi-cycle. The core pipeline stalls during execution:

| Instruction      | Latency (cycles) | Pipeline behavior             |
|------------------|-------------------|-------------------------------|
| `blake3.init`    | 3                 | Stall EX for 2 extra cycles   |
| `blake3.round`   | 2                 | Stall EX for 1 extra cycle    |
| `blake3.chain`   | 1                 | No stall (register read)      |
| `blake3.finalize`| 2                 | Stall EX for 1 extra cycle    |
| `blake3.ldmsg`   | 4                 | Stall for memory load         |
| Full hash (7 rnd)| 14                | Init(3) + 7*Round(2) - 3 overlap |

### Xlattice Timing

NTT operations are deeply pipelined through the butterfly network:

| Instruction      | Latency (cycles) | Pipeline behavior             |
|------------------|-------------------|-------------------------------|
| `ntt.fwd`        | 512               | Core stalls or context-switches|
| `ntt.inv`        | 512               | Core stalls or context-switches|
| `poly.add`       | 256               | Element-parallel, pipelined   |
| `poly.mul`       | 256               | Element-parallel, pipelined   |
| `poly.reduce`    | 256               | Element-parallel, pipelined   |

## 5. Clock Domains

```
  Domain        Frequency     Driven by          Used by
  ─────────────────────────────────────────────────────────────
  clk_core      100 MHz       MMCM (PLL)         Cores, L1, NoC, L2
  clk_ddr       200 MHz       MMCM (PLL)         DDR3 controller (4:1 mode)
  clk_ref       200 MHz       MMCM (PLL)         DDR3 IDELAY reference
  clk_uart      50 MHz        clk_core / 2       UART peripheral
  clk_spi       25 MHz        clk_core / 4       SPI flash controller
  clk_jtag      20 MHz        External TCK        Debug module (async)
```

All domain crossings use dual-flop synchronizers or async FIFOs. The DDR3
controller uses Xilinx MIG IP with its own internal clock generation from
`clk_ddr`.

### Reset Tree

```
         External                 MMCM
        POR button ──────┐       locked ──────┐
                         ▼                    ▼
                    ┌─────────────────────────────┐
                    │     Reset Synchronizer       │
                    │  (2-FF per domain, async     │
                    │   assert, sync deassert)     │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
         rst_core_n       rst_ddr_n        rst_jtag_n
         (active-low)     (active-low)     (active-low)
```

## 6. FPGA Resource Estimates (Kintex-7 XC7K325T)

### Per-Tile Breakdown

| Component               | LUTs    | FFs     | BRAM 36K | DSP48E1 |
|--------------------------|---------|---------|----------|---------|
| RV32IMC core (7-stage)   | 3,800   | 2,100   | 0        | 4       |
| Register file (32x32)    | 200     | 1,024   | 0        | 0       |
| L1I cache (8 KB, 2-way)  | 300     | 180     | 4        | 0       |
| L1D cache (8 KB, 2-way)  | 400     | 220     | 4        | 0       |
| Xcrypto BLAKE3 engine    | 2,200   | 1,100   | 1        | 0       |
| Xlattice NTT engine      | 1,800   | 900     | 3        | 16      |
| NoC router port          | 400     | 200     | 0        | 0       |
| **Tile total**           | **9,100** | **5,724** | **12** | **20**  |

### SoC Total

| Component                     | LUTs     | FFs      | BRAM 36K | DSP48E1 |
|-------------------------------|----------|----------|----------|---------|
| 16 tiles                      | 145,600  | 91,584   | 192      | 320     |
| Shared L2 cache (256 KB)      | 1,200    | 800      | 64       | 0       |
| Mesh NoC crossbar             | 2,400    | 1,600    | 8        | 0       |
| DDR3 controller (MIG)         | 3,500    | 2,800    | 12       | 0       |
| Peripherals (UART/SPI/Timer)  | 800      | 500      | 2        | 0       |
| Debug module (JTAG)           | 600      | 400      | 1        | 0       |
| **SoC total**                 | **154,100** | **97,684** | **279** | **320** |
| **XC7K325T available**        | 203,800  | 407,600  | 445      | 840     |
| **Utilization**               | **75.6%** | **24.0%** | **62.7%** | **38.1%** |

The design fits comfortably within the Kintex-7 XC7K325T with margin for
timing closure at 100 MHz. BRAM is the tightest resource due to L1/L2 caches
and NTT scratchpads.
