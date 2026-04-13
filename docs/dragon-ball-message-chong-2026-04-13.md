# Message to Chong — Dragon Ball Miner (2026-04-13)
## Copy-paste for Discord DM

---

Hi Chong! Important update from our side.

We completed a deep technical review (v4.1) of the RTL code, cross-referencing every module against our live mining protocol in Rust. We found and **already fixed** a significant correctness issue before your team starts synthesis.

**What was wrong:** The BLAKE3 chain loop in `xcrypto_unit.sv` fed the hash output back as the BLAKE3 chaining value (CV) for the next iteration. But our mining protocol works differently — every hash in the 100-round chain uses the standard BLAKE3 IV as the chaining value, and the previous hash output goes into the message block (words 0-7, zero-padded). These compute different functions. If you had synthesized the old RTL and tested it against our miner, the hash output would not match.

**What we fixed (already in the updated tarball):**

1. `xcrypto_unit.sv` — Chain loop now uses IV as CV and puts the previous hash into the message block with correct `block_len` (40 for the first hash, 32 for rounds 1-99) and correct flags (`CHUNK_START | CHUNK_END | ROOT`). This matches our GPU kernel exactly (`gpu.rs:199-208`).

2. `xcrypto_scratchpad.sv` — New module that replaces the zero-fill stub. The old code had all 16 message words hardwired to zero, so the BLAKE3 engine was hashing zeroes instead of real data. The scratchpad uses a synchronous read model (data and valid registered on the same clock edge) for clean timing.

3. `qug_tile.sv` — Scratchpad wired in, memory-mapped at `0x0002_0000`. The CPU (or future mining controller) writes the challenge hash and nonce to this address range, then issues `blake3.chain` which reads all 16 words in one cycle.

We also corrected a spec error — the mining input is **40 bytes** (32-byte challenge hash + 8-byte nonce in little-endian), NOT 72 bytes. The miner address is not part of the hash input.

**Updated tarball (with all fixes applied):**
```
wget https://quillon.xyz/downloads/qug-v1-rtl.tar.gz
tar xzf qug-v1-rtl.tar.gz
```

This is ready for Vivado synthesis AND functional verification — the hash output will now match our mainnet miners. The v4.1 technical review document is included in `doc/technical-review-rtl-v4.1.md` with full protocol spec, byte ordering, and the remaining roadmap items (mining controller, SPI interface, best-hash tracker).

**What's new in the tarball since last time:**
- Fixed chain semantics (the critical bug)
- Scratchpad module (real data instead of zeroes)
- Kintex-7 XC7K355T constraint file
- Vivado TCL script with part auto-selection
- Technical review v4 + v4.1 documents

**Still need from your side:**
1. Which FPGA — XC7K355T (Kintex-7) or XC7A355T (Artix-7)?
2. Board schematic (pin assignments for clock, reset, UART, LEDs)
3. VCS version (for genvar compatibility)
4. Does your board have an onboard MCU for SPI, or UART only?
5. 12nm confirmed for first ASIC tapeout?
6. How many mining tiles on the ASIC die?
7. Does the board have a GPIO pin for interrupt output?

No rush — we can proceed independently. But we need pin assignments before we can give you a board-specific constraint file.

Best regards,
Demetri
