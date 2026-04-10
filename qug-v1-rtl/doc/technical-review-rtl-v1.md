# Technical Review: QUG-V1 RTL Design (v1 — Initial Delivery)

**Date:** 2026-04-10
**Reviewer:** Self-assessment (Quillon engineering)
**Purpose:** Honest evaluation of the current RTL before sharing with Dragon Ball. Identify gaps where external AI review (DeepSeek, ChatGPT) can strengthen the design.

---

## 1. What We Built

19 files, 4,624 lines of SystemVerilog:
- RV32IMC core with 7-stage pipeline, forwarding, hazard detection
- BLAKE3 Xcrypto pipeline (7 rounds parallel, 9 cycles/hash)
- Xcrypto unit with 100-hash VDF chain support
- Testbenches with known-answer vectors
- Documentation + build scripts

---

## 2. Honest Assessment: What's Good

### The BLAKE3 pipeline is solid
- Correct BLAKE3 round function with all 8 quarter-rounds
- Pre-computed message schedule permutations for 7 rounds
- Pipeline registers between stages for throughput
- VDF chain support (output → input loopback without register file round-trip)
- Testbench includes behavioral reference model for comparison

### The core architecture is reasonable
- Standard 7-stage in-order pipeline
- Full data forwarding from EX1/EX2/MEM/WB1/WB2
- Load-use hazard detection with 1-cycle stall
- Branch resolution in EX1 with IF/ID/EX1 flush
- Extension interface with valid/ready handshake
- Compressed (C) instruction expansion

### The documentation is professional
- Architecture doc with ISA encoding tables
- Coding guidelines suitable for multi-team collaboration
- README with build instructions

---

## 3. Honest Assessment: What's Weak or Missing

### 3.1 The RISC-V core is NOT verified against the spec

**Problem:** We wrote a decoder, ALU, and pipeline from scratch. We have NOT run the RISC-V compliance test suite (riscv-tests / riscv-arch-test). The decoder likely has bugs in edge cases:
- CSR instructions (not implemented at all)
- FENCE/ECALL/EBREAK (not implemented)
- Misaligned memory access handling
- Integer overflow behavior for MUL/DIV
- Lots of C-extension corner cases

**Risk:** Dragon Ball's engineers will synthesize this and find it fails on real programs.

**Ask DeepSeek/ChatGPT:**
> "Review our RV32IMC decoder (qug_decoder.sv). We implemented all R/I/S/B/U/J types plus C-extension expansion. What common RISC-V decoder bugs should we check for? Which riscv-tests cases are most likely to fail on a first implementation?"

### 3.2 The pipeline has no formal verification

**Problem:** We wrote forwarding logic and hazard detection by hand. Pipeline correctness is notoriously hard to get right — off-by-one errors in forwarding, missing stall conditions, and incorrect flush logic cause silent data corruption.

**Risk:** A subtle forwarding bug means the core silently computes wrong results. Mining would produce invalid hashes and waste power.

**Ask DeepSeek/ChatGPT:**
> "Here is our 7-stage RISC-V pipeline (qug_pipeline.sv). The forwarding network bypasses from EX1, EX2, MEM, WB1, WB2 back to ID. Can you identify any missing forwarding paths, incorrect stall conditions, or flush logic errors? Specifically: (1) Does the load-use hazard detection cover all cases including loads followed by branches? (2) Is the branch flush correct — should we flush EX1 or only IF/ID? (3) Can a store-to-load forwarding hazard occur?"

### 3.3 The BLAKE3 pipeline may have incorrect message scheduling

**Problem:** BLAKE3's message schedule permutation is critical. We hardcoded 7 permutation tables (SIGMA[0..6]). If ANY entry is wrong, every hash is wrong, and mining produces invalid results.

**Risk:** A single wrong index in the permutation table makes the entire miner useless.

**Ask DeepSeek/ChatGPT:**
> "Verify these BLAKE3 message schedule permutations are correct:
> Round 0: {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
> Round 1: {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8}
> Round 2: {3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1}
> ... (all 7 rounds)
> Compare against the BLAKE3 reference implementation in C and Rust."

### 3.4 No timing analysis for FPGA

**Problem:** We claim 100 MHz on Kintex-7 but have done ZERO synthesis. The 512-bit BLAKE3 datapath with 8 parallel quarter-rounds may not meet timing — each quarter-round has 4 additions, 4 XORs, and 4 rotations in a combinational chain. That's a long critical path.

**Risk:** The design may only run at 50-70 MHz on Kintex-7, halving the claimed hashrate.

**Ask DeepSeek/ChatGPT:**
> "Our BLAKE3 round module (blake3_round.sv) executes all 8 quarter-rounds (4 column + 4 diagonal) in a single combinational stage. Each quarter-round is: a+=b; d^=a; d>>>=16; c+=d; b^=c; b>>>=12; a+=b; d^=a; d>>>=8; c+=d; b^=c; b>>>=7. Is this critical path too long for 100 MHz on Kintex-7 (6ns cycle)? Should we split each round into 2 pipeline stages (column + diagonal) to meet timing?"

### 3.5 Xlattice extension is completely missing

**Problem:** The Xlattice bignum accelerator (256-bit modular arithmetic for Genus-2 VDF) is not implemented. Only stub files exist. This is half the QUG-V1's value proposition.

**Risk:** Dragon Ball can only evaluate the BLAKE3 path, not the Genus-2 VDF path.

**Ask DeepSeek/ChatGPT:**
> "We need to implement a 256-bit modular multiplier in SystemVerilog for FPGA (Kintex-7 with 840 DSP48E1 slices). The multiplier must perform Barrett reduction over a 256-bit prime field (p = 2^255 - 19). What is the most DSP-efficient architecture? Should we use schoolbook multiplication (O(n^2) DSPs), Karatsuba (fewer DSPs, more logic), or iterative digit-serial (fewer DSPs, more cycles)? Target: 1 multiplication per 4-8 clock cycles at 100 MHz."

### 3.6 Memory subsystem is trivial

**Problem:** We have a basic single-port SRAM and no real cache. The full QUG-V1 spec calls for 32KB 4-way L1 + 4MB shared L2 + DDR4 controller. The FPGA prototype doesn't need all of that, but it needs at least an instruction/data memory interface that works.

**Risk:** The core testbench uses a simple memory array. Real FPGA deployment needs proper BRAM inference and AXI4 protocol compliance.

**Ask DeepSeek/ChatGPT:**
> "For a RISC-V core targeting Kintex-7 FPGA, what is the minimal memory subsystem needed for a working prototype? We need: (1) instruction memory (read-only, BRAM), (2) data memory (read-write, BRAM), (3) memory-mapped I/O for UART. Should we use AXI4-Lite or a simpler custom bus for the FPGA prototype? How do we infer dual-port BRAM on Xilinx 7-series?"

### 3.7 No NoC implementation

**Problem:** The 4x4 mesh Network-on-Chip is not implemented. For the FPGA prototype (1-4 cores), we don't need the full NoC, but we need at least a shared bus or crossbar for multi-core communication.

**Risk:** Single-core prototype works, but multi-core scaling is unproven.

**Ask DeepSeek/ChatGPT:**
> "For a 2-4 core RISC-V cluster on FPGA, what is the simplest interconnect that scales? Options: (1) Shared AXI4 bus with arbiter, (2) AXI4 crossbar, (3) Simple ring bus. We need to share L2 SRAM and communicate mining challenges between cores. Which is most area-efficient for Kintex-7?"

### 3.8 The Xcrypto unit's memory interface is assumed

**Problem:** `xcrypto_unit.sv` assumes a 512-bit wide single-cycle SRAM read for message block fetch. This doesn't exist on Kintex-7 — BRAM is 36-bit wide (or 72-bit in SDP mode). We'd need 8 sequential reads or a custom wide BRAM configuration.

**Risk:** The Xcrypto unit won't work as-is on FPGA without a memory adapter.

**Ask DeepSeek/ChatGPT:**
> "Our BLAKE3 Xcrypto unit needs to fetch a 512-bit (16x32-bit) message block from memory. On Kintex-7, BRAM is 36Kb with max 72-bit port width. What's the best approach: (1) 8 sequential 64-bit reads (adds 8 cycles latency), (2) Use 8 parallel BRAM instances (costs 8 BRAM36K), (3) Use distributed RAM (LUTRAM) for the message buffer?"

---

## 4. What Dragon Ball Will Ask (And We Don't Have Answers)

| Question | Our Current Answer | What We Need |
|----------|-------------------|--------------|
| "Does this pass riscv-tests?" | No — untested | Run compliance suite, fix failures |
| "What's the critical path?" | Unknown — no synthesis | Run Vivado synthesis, report timing |
| "How many LUTs/DSPs/BRAMs?" | Estimated only | Actual synthesis utilization report |
| "Can we simulate with VCS?" | Maybe — untested | Verify with commercial simulator |
| "Where's the Xlattice?" | Not implemented | Need 256-bit modular arithmetic RTL |
| "What about interrupts/exceptions?" | Not implemented | At minimum: illegal instruction trap |
| "Debug interface (JTAG)?" | Not implemented | Need for FPGA bring-up |

---

## 5. Recommended Questions for DeepSeek/ChatGPT

### Architecture-level (give them the full architecture.md):
1. "Review our QUG-V1 SoC architecture. Is a 7-stage pipeline overkill for a 100 MHz FPGA target? Would a 5-stage pipeline be simpler and still meet timing?"
2. "We plan 16 cores in a 4x4 mesh. For FPGA prototyping, should we start with 1 core or 4 cores? What's the minimum viable multi-core configuration?"
3. "Our Xcrypto BLAKE3 unit is a fully pipelined 7-stage design. For a single-core FPGA prototype doing sequential VDF chains (100 hashes), is pipelining wasteful? Would a single-round iterative design (7 cycles, 1/7th the area) be more FPGA-appropriate?"

### RTL-level (give them specific .sv files):
4. "Review blake3_round.sv for correctness. Compare quarter-round implementation against BLAKE3 spec. Are the rotation amounts correct (16, 12, 8, 7)?"
5. "Review qug_pipeline.sv forwarding logic. We bypass from 5 stages. Is this correct for a 7-stage pipeline, or are we missing forwarding from some stages?"
6. "Review qug_decoder.sv C-extension expansion. We handle ~25 C-extension instructions. Which ones are we missing? Are our register mappings correct (CIW/CL/CS formats use registers x8-x15 mapped from 3-bit fields)?"

### FPGA-specific:
7. "For Kintex-7 XC7K325T at 100 MHz, what is the maximum combinational depth we can afford per pipeline stage? Our BLAKE3 round has ~12 levels of logic (4 adds + 4 XORs + rotations). Will this meet timing?"
8. "How should we structure the BRAM for instruction and data memory to get maximum bandwidth with minimum BRAM usage? We need 64KB instruction + 64KB data."

### Verification:
9. "What is the minimum set of riscv-tests we should pass before claiming RV32IMC compliance? Can you list the specific tests?"
10. "We have a behavioral BLAKE3 reference in our testbench. How do we verify it's correct? Should we generate test vectors from the official BLAKE3 Rust implementation instead?"

---

## 6. Priority Action Items

| Priority | Task | Effort | Blocks Dragon Ball? |
|----------|------|--------|---------------------|
| **P0** | Run Vivado synthesis — get real timing/area numbers | 1 day | Yes — they need this |
| **P0** | Verify BLAKE3 permutation tables against reference | 2 hours | Yes — wrong tables = wrong hashes |
| **P1** | Run riscv-tests compliance suite | 2-3 days | Yes — core must be correct |
| **P1** | Fix BRAM memory interface for FPGA | 1 day | Yes — testbench-only memory won't synthesize |
| **P2** | Implement Xlattice mod_mul_256 | 1 week | No — Phase 1A is BLAKE3 only |
| **P2** | Add JTAG debug interface | 3 days | No — but needed for FPGA bring-up |
| **P3** | Implement NoC router | 1 week | No — single-core prototype first |
| **P3** | Add interrupt/exception handling | 3 days | No — bare-metal mining doesn't need it |

---

## 7. Bottom Line

**What we have:** A reasonable first-pass RTL that captures the architecture correctly. The BLAKE3 pipeline design is sound. The RISC-V core has the right structure.

**What we don't have:** Verification that any of it actually works. No synthesis results, no compliance testing, no timing analysis.

**Honest risk assessment:** If we send this to Dragon Ball today, their engineers will find bugs within a day of synthesis. The BLAKE3 pipeline is probably close to correct. The RISC-V core probably has decoder bugs and pipeline hazards we haven't caught.

**What we need from DeepSeek/ChatGPT:** Code review of the critical modules (decoder, pipeline, BLAKE3 round). They can spot the kinds of bugs that are invisible in small testbenches but break on real programs. We should give them the full .sv files and ask specific questions (Section 5 above).

**What we need from Dragon Ball:** Synthesis on their Kintex-7 boards to get real numbers. We can fix bugs in RTL quickly — but we can't fix timing without synthesis data from their tools.

---

*This review is intentionally harsh. We would rather find problems now than after Dragon Ball synthesizes and finds a broken design. Honesty builds trust.*
