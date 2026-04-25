# Message to Chong — Dragon Ball Miner
## Copy-paste for Discord DM

---

Hi Chong!

Great news — we've completed the Phase 1A RTL delivery for the QUG-V1 SoC. I wanted to share it with you directly so your team can start evaluating.

**What's included (38 files, ~9,200 lines of SystemVerilog):**

✅ Complete RISC-V RV32IMC core — 7-stage pipeline with full data forwarding
✅ BLAKE3 Xcrypto pipeline — 14-stage, 1 hash/cycle throughput, VDF chain support
✅ Xlattice 256-bit modular arithmetic — multiplication (8 DSP, 14 cycles), addition, Fermat inversion
✅ BRAM memory subsystem — 64KB instruction + 64KB data + UART
✅ SoC integration — parameterized tile with both extensions
✅ FPGA wrapper — Kintex-7 XC7K325T with MMCM clock gen and reset synchronizer
✅ Testbenches — self-checking with golden BLAKE3 reference vectors
✅ Mining firmware — bare-metal C with inline Xcrypto assembly
✅ Vivado synthesis script — full flow with timing/utilization/power reporting

**Estimated FPGA utilization (single tile):**
- LUTs: ~20,700 / 203,800 (10.2%)
- DSPs: 12 / 840 (1.4%)  
- BRAMs: 34 / 445 (7.6%)
- Headroom for 4-6 tiles on a single Kintex-7

**How to get the code:**

```
git clone https://code.quillon.xyz/repo.git
cd repo
git checkout feature/safe-batched-sync-v1.0.2
ls qug-v1-rtl/
```

The RTL is in `qug-v1-rtl/`. Documentation is in `qug-v1-rtl/doc/`.

**Attached:**
1. RTL Technical Review v3 (PDF) — full project summary with architecture, resource estimates, and known limitations
2. FPGA Collaboration Proposal v3 (PDF) — updated with your FPGA device recommendations, IP framework, and strategic roadmap

**What we'd like from Dragon Ball:**
1. Run Vivado synthesis on your Kintex-7 board — get us real timing/area numbers
2. Run the BLAKE3 testbench — verify the known-answer test passes
3. Tell us which Kintex-7 board you plan to use (we need pin assignments)
4. Let us know what simulator your team prefers (VCS, Xcelium, Vivado Simulator, Verilator)

**Known limitations (transparent):**
- RISC-V core not yet tested against riscv-tests compliance suite (in progress)
- NTT instructions are stubs (Phase 1B)
- Single-tile prototype only (multi-tile is parameterized but untested)
- No JTAG debug (use UART + Xilinx ILA)

We've been honest about what's verified and what isn't — see the Technical Review PDF for full details. We believe the design is ready for your team to start synthesis and evaluation.

Looking forward to your feedback!

Best regards,
Demetri

**Attachments:**
- `technical-review-rtl-v3.pdf`
- `dragon-ball-fpga-collaboration-v3.pdf`
