# Dragon Ball — Reply to Chong, 2026-05-19

Addresses: (1) the recurring RTL synthesis failure where the optimizer removes Xcrypto/Xlattice, (2) Chong's question about payment milestone structure, (3) his team-manpower availability concern.

## Root-cause analysis (the RTL "doesn't work" problem)

Chong saw the same symptom twice in a row even after the FLATTEN_HIERARCHY=none + dont_touch fixes landed (commit visible in `qug-v1-rtl/syn/synth_vivado.tcl:170`). The optimizer is still pruning Xcrypto/Xlattice because the **deeper cause was never the synthesis attributes** — it's the instruction BRAM initial contents.

When Vivado synthesizes a design where the instruction memory initializes to all-zeros (RV32 NOP), constant propagation can prove at elaboration time that:
- Every instruction fetch yields 0x00000000
- Therefore decoder outputs `xcrypto_valid = 0` and `xlattice_valid = 0` permanently
- Therefore those units have no observable fan-out
- Therefore the optimizer eliminates them, dont_touch attributes notwithstanding (dont_touch prevents *the module from being dissolved*, not *internal logic from being pruned via constant inputs*).

**The fix is to give Vivado actual mining firmware in the BRAM at synthesis time** — not synthesis attributes. Three options, increasing in fidelity:

1. **Quick fix** (15 min): pre-init the instruction BRAM with the BLAKE3 mining loop firmware (`qug-v1-rtl/sw/firmware.bin` compiled to a `.coe` file). Once Vivado sees real custom-0/custom-1 opcodes in the elaborated BRAM contents, constant propagation can no longer prove the crypto units are unused.

2. **Robust fix** (1 day): wire a UART-triggered test-pattern injector that feeds known custom-opcode sequences from outside the BRAM. This makes the design synthesizable even without firmware preload.

3. **Production fix** (2-3 days): replace the static-BRAM model with a streaming instruction fetch (DMA from external memory). The on-chip BRAM is then only used for low-MHz boot ROM; the mining loop runs from instruction cache fed by external DDR or SPI flash.

For the FPGA validation phase, option 1 is what we should ship. The `.coe` file generation is a single line in `qug-v1-rtl/sw/Makefile`:

```makefile
firmware.coe: firmware.bin
	echo "memory_initialization_radix=16;" > $@
	echo "memory_initialization_vector=" >> $@
	xxd -p $< | sed 's/.\{8\}/&,/g; s/,$/;/' >> $@
```

Then add `add_files -fileset constrs_1 firmware.coe` and reference it in the BRAM instantiation's INIT_VALUES parameter. We will push this fix today and notify you when it's pushable.

## Payment milestones — clarification

Per your 2026-05-12 enumeration, the five FPGA-phase milestones with USDT payment are:

| Milestone | Trigger | Amount (USDT) |
|---|---|---|
| **M1** | Cooperation agreement signed | **$20,000** initial manpower |
| **M2** | BLAKE3 Xcrypto pipeline validated on XC7K355T (testbench + FPGA boot test pass) | **$10,000** |
| **M3** | Genus-2 VDF Xlattice validated (no new hardware purchase) | **$10,000** |
| **M4** | If multi-core verification requires Virtex UltraScale+ board purchase | **$15,000** (hardware) |
| **M5** | ASIC feasibility evaluation report (area, power, perf/W per core) | **$10,000** |

**Total FPGA-phase commitment**: **$50,000 USDT minimum, $65,000 if M4 hardware purchase is required.**

We commit to paying these in USDT from accumulated mining revenue regardless of whether the project proceeds to tape-out. Your team's engineering time is not at risk if the ASIC round fails to close.

Tape-out and manufacturing phases are gated by external investor funding. If that round closes, we proceed with your full contract-manufacturing model (you produce, we buy at cost + markup, we distribute). If not, the FPGA work has independent IP value to your portfolio and we have paid you for it.

## On team manpower availability

We understand. The two failed synthesis attempts represented real engineering time spent diagnosing optimizer behavior, and we appreciate your patience. To reduce friction on your side:

1. We will provide the **`.coe` firmware preload file** before your next synthesis attempt, so the optimizer-pruning issue is gone before your engineers touch it.
2. We will provide a **smoke-test checklist** of the three things you can run in ~30 minutes to verify the synthesis is healthy *before* committing your team's hours to the full validation: (a) check that `xcrypto_inst` shows >1000 LUTs in utilization report, (b) check that `xlattice_inst` shows >500 LUTs, (c) check critical path is in the BLAKE3 compression function (not in the decoder or BRAM read).
3. We will set up a **shared synthesis log review channel** (Discord or shared doc) where your team can paste partial results and our team can flag whether to proceed or stop.

If your team currently has limited capacity, we can also offer an alternative pacing: **monthly milestone reviews** instead of continuous engagement. You commit blocks of engineering hours when convenient, we commit USDT on milestone completion. This way your team's schedule is not driven by ours.

## Concrete asks of Dragon Ball this week

1. Confirm that the M1-M5 milestone structure is acceptable (or propose modifications)
2. Confirm preferred pacing (continuous vs monthly review)
3. Confirm preferred contract language source (English, Chinese, or both)
4. Provide an official corporate email address for the cooperation agreement signing

We are ready to push the firmware-preload fix today and have the cooperation agreement drafted within 72 hours.

Best regards,
Demetri / Quillon Team

---

## Handoff document references

For Chong's team to absorb the project state:

- **`qug-v1-rtl/doc/technical-review-rtl-v4.1.md`** — latest RTL technical review, comprehensive
- **`qug-v1-rtl/doc/architecture.md`** — module-level architecture
- **`qug-v1-rtl/IVERILOG_COMPAT_STATUS.md`** — iverilog simulator compatibility notes
- **`qug-v1-rtl/syn/synth_vivado.tcl`** — synthesis script with FLATTEN_HIERARCHY=none fix
- **`papers/dragon-ball-fpga-collaboration-v3.pdf`** — formal collaboration proposal
- **`papers/asic-12nm-cost-model-v1.pdf`** — 12nm economics + break-even analysis
- **`papers/qugusd-dragonball-whitepaper.pdf`** — QUGUSD allocation mechanism (20M for partnership)
- **`docs/dragon-ball-reply-2026-05-12.md`** — previous reply with pre-mining + miner pricing answers

Synthesis script status verified at `qug-v1-rtl/syn/synth_vivado.tcl:170` — comment confirms FLATTEN_HIERARCHY=none is the active setting. The firmware-preload `.coe` fix is the next-required addition.
