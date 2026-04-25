# Dragon Ball Miner Reply — 2026-04-10
## Copy-paste ready for Discord DM to chong_dragonballminer

---

Hi Chong! Thank you for the detailed and thoughtful questions. Your experience with FPGA prototyping is exactly what we need in this partnership. Let me address each point:

**Q1: Which algorithm are we validating at the FPGA stage?**

Both — but in two phases within Phase 1:

**Phase 1A (BLAKE3 — first 2-3 months):**
- Validate the Xcrypto BLAKE3 pipeline (100 sequential hashes per nonce)
- This is the simpler datapath — single-issue pipelined hash unit
- Purpose: prove the mining core works, validate hashrate/watt, test the 16-core cluster interconnect
- This is essentially the same class of problem as your ALPH miner — familiar territory for Dragon Ball

**Phase 1B (Genus-2 VDF — months 3-6):**
- Validate the Xlattice bignum accelerator (256-bit modular arithmetic)
- This requires: Barrett reduction, modular multiplication, modular inversion (Fermat's method)
- Purpose: prove the VDF acceleration works, measure doublings/sec/watt
- This is the novel part — no one has done Genus-2 Jacobian hardware acceleration before

We can start Phase 1A immediately while the Genus-2 VDF algorithm is finalized on mainnet (expected Q3 2026).

**Q2: Artix-7 resource concerns — you're probably right**

You raise an excellent point. Our original spec assumed Artix-7 (XC7A200T) which has:
- 215K logic cells, 740 DSP slices, 13Mb BRAM

For the BLAKE3-only core, Artix-7 is likely sufficient — a single BLAKE3 pipeline is compact (~5K LUTs + 8 DSP slices for the compression function).

BUT for the Xlattice bignum accelerator (256-bit parallel multiplier with Barrett reduction), you're correct that Artix-7 may be too small. A single 256×256 modular multiplier using DSP48E1 slices consumes ~64-128 DSPs depending on implementation. The 8-butterfly parallel unit in our spec would need 512+ DSPs — exceeding Artix-7's 740 DSP limit when combined with everything else.

**Your Xilinx 355T (Virtex/Kintex UltraScale?) suggestion makes sense** for the full SoC validation. We could do:

| Target | FPGA | Purpose | Est. Resources |
|--------|------|---------|----------------|
| BLAKE3 core only | Artix-7 200T | Phase 1A — hash pipeline validation | ~30K LUTs, 64 DSPs |
| Full SoC (1 core) | Kintex-7 355T or KU040 | Phase 1B — complete mining core with Xlattice | ~150K LUTs, 400 DSPs |
| Multi-core cluster | Virtex UltraScale+ | Optional — validate 4-core interconnect | ~500K LUTs |

**Q3: Your offer to collaborate on FPGA — YES, absolutely**

We would welcome Dragon Ball's involvement in the FPGA phase. Here's what we propose:

**What Quillon provides:**
- Complete RTL design (Verilog/SystemVerilog) for the QUG-V1 SoC
- Mining algorithm firmware (BLAKE3 + Genus-2 VDF)
- Testbench and verification suite
- Engineering support for ISA/algorithm questions

**What Dragon Ball provides:**
- FPGA device selection expertise (you know better than us which Xilinx parts fit)
- Board design and manufacturing (if Dragon Ball wants to build the FPGA dev board)
- FPGA synthesis optimization (your team has done this many times)
- Manufacturing cost estimates for the 7nm phase (informed by FPGA results)

**What we need from you to start:**
1. Which FPGA device do you recommend for the full SoC? (We're open to Kintex/Virtex/UltraScale)
2. Do you have existing FPGA dev boards we could target, or should we design a new one?
3. Can you share your ALPH FPGA prototype specs as a reference? (resource utilization, clock speed achieved)
4. Timeline: how quickly can your team synthesize and test if we provide RTL?

**Cost sharing proposal:**
- Quillon funds the RTL design and engineering (already budgeted at $67.5K)
- Dragon Ball funds FPGA boards and synthesis tools (if using your existing infrastructure, this may be near-zero)
- Alternatively: Quillon funds everything, Dragon Ball contributes engineering time only
- We're flexible — the goal is to get hardware in hand as fast as possible

**Next step:** If you're interested, I can share the RTL architecture document (block diagram, ISA encoding, memory map, bus protocol) so your team can assess FPGA resource requirements. This is non-confidential — it's the hardware spec, not trade secrets. We can sign a mutual NDA if you prefer.

Looking forward to your thoughts! This is exactly the kind of practical collaboration that moves both companies forward.

Best regards,
Demetri

---

## Attachments to include:
1. `papers/dragon-ball-response-v2.pdf` (already sent — has the full NRE analysis)
2. `papers/qug-v1-pocket-supercomputer.pdf` (already sent — QUG-V1 SoC whitepaper)
3. `papers/genus2-jacobian-vdf-mining-whitepaper.pdf` (Genus-2 VDF algorithm details)
4. **NEW: RTL architecture document** (to be created — block diagram + ISA encoding)
