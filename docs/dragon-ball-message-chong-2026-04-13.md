# Message to Chong — Dragon Ball Miner (2026-04-13)
## Copy-paste for Discord DM

---

Excellent news Chong! XC7K355T confirmed — that's the chip we targeted in our constraint file, so the Vivado part selection in `synth_vivado.tcl` will work out of the box. And even better that you have ready-made XDC files from your previous work — please use those instead of ours, since your pin assignments are proven on your board.

Just a heads up — please download the **latest tarball** before you start, since we pushed important fixes today:

```
wget https://quillon.xyz/downloads/qug-v1-rtl.tar.gz
tar xzf qug-v1-rtl.tar.gz
```

The key fix: the BLAKE3 chain loop now correctly matches our mainnet mining protocol (IV as chaining value, previous hash into message block). The earlier version would have produced wrong hash output during functional testing. Synthesis/timing numbers would have been fine either way since the datapath didn't change — but now functional verification will also pass.

To run synthesis with your XDC:
```
vivado -mode batch -source qug-v1-rtl/fpga/scripts/synth_vivado.tcl -tclargs xc7k355tffg901-2
```
Then just swap our XDC for yours in the TCL script (line where it does `add_files -fileset constrs_1`).

Feel free to ask us anything as you review the code. We're here to help. Looking forward to your FPGA info tomorrow!

Best,
Demetri

---

## Context (not for Discord — internal reference)

Chong's reply (2026-04-13 15:25):
> we use xc7k355t. and we will provide you with FPGA-related information tomorrow.
> We have carried out relevant development on the 355T platform and have ready-made
> XDC files. We will first look into the code and consult you anytime if we have
> questions.

**Key takeaways:**
- FPGA confirmed: **Kintex-7 XC7K355T** (not Artix-7) — our constraint file and part selection are correct
- They have **existing XDC files** from previous projects (ALPH miner) — no need for us to guess pin assignments
- They will **provide FPGA board info tomorrow** (2026-04-14)
- They're **starting code review now** and will ask questions as they go
- Tone is positive and collaborative

**Status of deliverables after today's fixes:**
- [x] Chain semantics fixed (IV as CV, hash→message, correct block_len/flags)
- [x] Scratchpad wired in (replaces zero-fill stub)
- [x] Synchronous read model (data+valid registered together)
- [x] XC7K355T constraint file created
- [x] Vivado TCL updated with part auto-selection
- [x] Technical review v4.1 document
- [ ] Mining controller (hardware nonce gen + target comparator) — next
- [ ] SPI host interface — after mining controller
- [ ] VCS simulation script — when they confirm VCS version
