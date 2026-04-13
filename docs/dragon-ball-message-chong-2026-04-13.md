# Message to Chong — Dragon Ball Miner (2026-04-13)
## Copy-paste for Discord DM

---

Hi Chong! Important update from our side.

We completed a deep technical review (v4.1) of the RTL code we sent you, cross-referencing it against our live mining protocol implementation. We found a significant correctness issue that we want to be transparent about before your team starts synthesis.

**The BLAKE3 chain loop in the RTL has a semantic bug.** The current `xcrypto_unit.sv` feeds the hash output back as the BLAKE3 chaining value (CV) for the next iteration. But our actual mining protocol works differently — every hash in the 100-round chain uses the standard BLAKE3 IV as the chaining value, and the previous hash output goes into the message block (words 0-7, zero-padded). These compute different hash functions. If you synthesized and ran the v3 RTL as-is, the FPGA would produce valid BLAKE3 output, but NOT the same output as our mainnet miners. We've identified exactly what needs to change (about 20 lines in `xcrypto_unit.sv`) and we're implementing the fix now.

We also corrected a spec error in our earlier documentation — the mining input is **40 bytes** (32-byte challenge hash + 8-byte nonce in little-endian), NOT 72 bytes. The miner address is submitted alongside the solution but is not part of the hash input. This means the scratchpad only needs to hold 10 words (8 challenge + 2 nonce), not 18.

**What this means for you:** The RTL you downloaded is still useful for Vivado synthesis and resource/timing analysis — the BLAKE3 round logic, pipeline structure, memory subsystem, and RISC-V core are all correct. The fix only changes the chain loop control signals inside `xcrypto_unit.sv`, not the datapath. Your synthesis numbers (LUT count, DSP usage, timing) will be nearly identical after the fix. But do NOT use the current RTL for functional verification of hash output — wait for our updated delivery.

**New in v4.1 (in addition to the chain fix):**
- Xcrypto scratchpad module (replaces the zero-fill stub — the old code hashed zeros instead of real data)
- Mining controller with hardware nonce generator and target comparator
- Correct byte-order handling for difficulty comparison (our hash is stored as LE u32 words, compared byte-by-byte)
- Best-hash tracker with double-buffered registers for safe SPI reads
- Full SPI register map with VERSION, IRQ, JOB_ID, and atomic work-update protocol
- Defined signal semantics (sticky/pulse/level) for all control outputs

**Updated tarball available now:**
```
wget https://quillon.xyz/downloads/qug-v1-rtl.tar.gz
tar xzf qug-v1-rtl.tar.gz
```
This contains the v4.1 technical review document. The corrected RTL code will follow within the next few days as branch `dev/mining-controller`.

**Still waiting on from your side:**
1. Which FPGA — XC7K355T (Kintex-7) or XC7A355T (Artix-7)?
2. Board schematic (pin assignments for clock, reset, UART, LEDs)
3. VCS version (for genvar compatibility)
4. Does your board have an onboard MCU for SPI, or UART only?
5. 12nm confirmed for first ASIC tapeout?
6. How many mining tiles on the ASIC die?
7. Does the FPGA board have a GPIO pin we can use as an interrupt output?

No rush on these — we can proceed with the RTL fixes independently. But we need your board info before we can deliver a pin-correct constraint file.

Best regards,
Demetri
