# Technical Review v4.1: SystemVerilog RTL Update Roadmap (Corrected)

**Date:** 2026-04-13
**Reviewer:** Quillon Foundation Engineering
**Status:** Corrections to v4 based on independent technical review. All ambiguities resolved.
**Audience:** Dragon Ball Miner engineering team + Quillon RTL developers
**Supersedes:** v4 (2026-04-13)

---

## Changes from v4

v4 had 7 weaknesses identified by independent review:

| # | Issue | Resolution in v4.1 |
|---|-------|-------------------|
| 1 | 72-byte message handling underspecified | Section 0 now provides exact protocol spec from Rust source |
| 2 | Scratchpad timing contract fuzzy | Section 1 specifies synchronous read model |
| 3 | Target comparison endianness ambiguous | Section 2 specifies exact byte extraction from LE u32 words |
| 4 | `<` vs `<=` inconsistency | Section 2 resolves: strict `<` for solutions, `<=` for target |
| 5 | Control signal semantics undefined | Section 4 defines pulse/level/sticky for every signal |
| 6 | Chain-loop optimization overprescribed | Section 5 downgraded to implementation target |
| 7 | SPI register map missing practical fields | Section 6 adds VERSION, IRQ, JOB_ID, atomicity |

Additionally: "do not modify" language in Section 15 softened per review.

---

## Section 0: MINING ALGORITHM SPECIFICATION (Protocol Truth)

**Source of truth:** `crates/q-mining/src/gpu.rs` lines 9-15, 159-228, 230-266.
Verified against: `crates/q-mining/src/optimized_miner.rs` lines 324-334,
`crates/q-mining/src/difficulty.rs` lines 308-345.

### 0.1 Algorithm Definition (Exact)

```
input[40] = challenge_hash[32] || nonce_le[8]

H₀  = BLAKE3(input[40])         // 40 bytes, block_len=40
Hᵢ  = BLAKE3(Hᵢ₋₁)   for i = 1..99   // 32 bytes, block_len=32

result = H₉₉

Valid solution iff:  result < difficulty_target   (strict less-than)
```

**This is NOT 72 bytes.** The v4 document incorrectly stated "72 bytes (challenge + nonce + address)." The actual mining input is **40 bytes**: 32-byte challenge hash + 8-byte nonce in little-endian. The miner address is NOT part of the hash input — it is submitted alongside the solution but does not affect the hash computation.

The challenge hash itself is derived server-side from block height, difficulty target, VDF iterations, and protocol version. Miners receive it as a 32-byte opaque value.

### 0.2 BLAKE3 Compression Details (Per Hash)

Each of the 100 BLAKE3 hashes is a **single-chunk, single-block** compression:

| Parameter | H₀ (40-byte input) | H₁..H₉₉ (32-byte input) |
|-----------|--------------------|-----------------------|
| Chaining value | BLAKE3 IV | BLAKE3 IV |
| Message block | `input[0..39] \|\| zeros[40..63]` | `Hᵢ₋₁[0..31] \|\| zeros[32..63]` |
| Counter | 0 | 0 |
| Block length | 40 | 32 |
| Flags | `CHUNK_START \| CHUNK_END \| ROOT` | `CHUNK_START \| CHUNK_END \| ROOT` |

**Critical observation:** Every hash in the chain uses the **BLAKE3 IV** as the chaining value, NOT the previous hash output. The previous hash output is placed in the **message block** (words 0-7), not the chaining value. This means the `blake3.chain` instruction in `xcrypto_unit.sv` has a semantic mismatch — it feeds the hash output back as the CV, but the protocol requires it to go into the message block.

**Source:** `gpu.rs:199-208` — `blake3_hash_32()` copies `input[8]` into `block[0..7]`, zeroes `block[8..15]`, and compresses with `iv` as the chaining value.

### 0.3 Message Block Layout (16 x 32-bit words, little-endian)

**For H₀ (40-byte input):**

```
word[ 0] = challenge_hash[ 0.. 3] (LE u32)
word[ 1] = challenge_hash[ 4.. 7]
word[ 2] = challenge_hash[ 8..11]
word[ 3] = challenge_hash[12..15]
word[ 4] = challenge_hash[16..19]
word[ 5] = challenge_hash[20..23]
word[ 6] = challenge_hash[24..27]
word[ 7] = challenge_hash[28..31]
word[ 8] = nonce[0..3]   (lower 32 bits of u64 LE)
word[ 9] = nonce[4..7]   (upper 32 bits of u64 LE)
word[10] = 0x00000000    (zero padding)
word[11] = 0x00000000
word[12] = 0x00000000
word[13] = 0x00000000
word[14] = 0x00000000
word[15] = 0x00000000
```

**For H₁..H₉₉ (32-byte input):**

```
word[ 0] = Hᵢ₋₁[0]   (hash word 0, LE u32)
word[ 1] = Hᵢ₋₁[1]
word[ 2] = Hᵢ₋₁[2]
word[ 3] = Hᵢ₋₁[3]
word[ 4] = Hᵢ₋₁[4]
word[ 5] = Hᵢ₋₁[5]
word[ 6] = Hᵢ₋₁[6]
word[ 7] = Hᵢ₋₁[7]
word[ 8] = 0x00000000
word[ 9] = 0x00000000
word[10] = 0x00000000
word[11] = 0x00000000
word[12] = 0x00000000
word[13] = 0x00000000
word[14] = 0x00000000
word[15] = 0x00000000
```

### 0.4 Target Comparison (Byte-Wise, LE Words → Bytes)

The hash output is 8 x u32 words in **little-endian** format. The difficulty target is 32 bytes.

Comparison is **byte-by-byte from byte 0 to byte 31** where bytes are extracted from words in LE order:

```
byte[i*4 + j] = (word[i] >> (j * 8)) & 0xFF    for i=0..7, j=0..3
```

A solution is valid iff: for the **first** byte index `k` where `hash_byte[k] != target_byte[k]`, `hash_byte[k] < target_byte[k]`.

**Equality is also valid** (`meets_target` returns `true` when all bytes are equal).

**Source:** `gpu.rs:217-228`

### 0.5 Difficulty Target Encoding

Difficulty is specified as **leading zero bits**. A target with N leading zero bits has:
- Bytes `0..(N/8 - 1)` = `0x00`
- Byte `N/8` = `0xFF >> (N % 8)`
- Remaining bytes = `0xFF`

Example: 16-bit difficulty → `target[0..1] = 0x00, target[2..31] = 0xFF`

**Source:** `difficulty.rs:308-344`

### 0.6 Difficulty-Weighted Rewards

Per-solution weight = `2^(leading_zero_bits)`, capped at `2^64`.

Leading zero bits counted from `hash[0]` byte-by-byte:
- Each `0x00` byte contributes 8 bits
- First non-zero byte contributes `byte.leading_zeros()` (0-7 bits)
- Stop counting at first non-zero bit

Reward for solution i = `(block_miner_total × weight_i) / total_weight` (integer division with rounding remainder to highest-weight miner).

**Source:** `block_producer.rs:1703-1748`

---

## Section 1: P0 — Xcrypto Message Block Interface (CRITICAL)

### 1.1 Problem (unchanged from v4)

`qug_tile.sv` lines 117-139 hard-wire `xc_mem_block[0:15]` to zero.

### 1.2 Additional Problem (NEW in v4.1)

The `blake3.chain` instruction in `xcrypto_unit.sv` feeds the hash output back as the **chaining value** (`pipe_cv`). But per Section 0.2, the Quillon mining protocol puts the previous hash into the **message block** and always uses the **BLAKE3 IV** as the chaining value.

This is a semantic mismatch between the RTL and the protocol. If the RTL chains via CV, it computes a different function than `BLAKE3(BLAKE3(...))`.

### 1.3 Fix: Correct Chain Semantics

The `blake3.chain` loop in `xcrypto_unit.sv` must be modified:

**Current behavior (WRONG for mining):**
```
Iteration i:
  CV     = hash_output[i-1]     ← previous hash fed as chaining value
  block  = msg_block_lat[0:15]  ← original message (frozen)
  flags  = 0
```

**Required behavior (matches Rust/GPU):**
```
Iteration i:
  CV     = BLAKE3_IV            ← always the IV
  block  = hash_output[i-1][0:7] || zeros[8:15]  ← hash goes into message
  block_len = 32                ← 32 bytes, not 64
  flags  = CHUNK_START | CHUNK_END | ROOT
```

**Changes to `xcrypto_unit.sv`:**

In the pipeline input control block (line 398-420), when in chain mode:

```systemverilog
if (fsm_state == S_COMPRESS) begin
    pipe_in_valid = 1'b1;

    if (lat_funct7 == F7_CHAIN && chain_count > 7'd0) begin
        // Chain iteration: IV as CV, previous hash as message
        for (int i = 0; i < 8; i++) begin
            pipe_cv[i] = IV[i];                    // Always use BLAKE3 IV
            pipe_block[i] = hash_latched[i];       // Hash output → message words 0-7
        end
        for (int i = 8; i < 16; i++) begin
            pipe_block[i] = 32'd0;                 // Zero-pad words 8-15
        end
        pipe_block_len = 32'd32;                   // 32 bytes, not 64
        pipe_flags = CHUNK_START | CHUNK_END | ROOT;
    end else begin
        // First iteration (H₀): use scratchpad message, IV as CV
        for (int i = 0; i < 8; i++) begin
            pipe_cv[i] = IV[i];                    // IV
        end
        // pipe_block = msg_block_lat (already set by default)
        pipe_block_len = 32'd40;                   // 40 bytes for initial hash
        pipe_flags = CHUNK_START | CHUNK_END | ROOT;
    end
end
```

Also add the BLAKE3 IV constant to `xcrypto_unit.sv`:

```systemverilog
localparam logic [31:0] IV [0:7] = '{
    32'h6A09E667, 32'hBB67AE85, 32'h3C6EF372, 32'hA54FF53A,
    32'h510E527F, 32'h9B05688C, 32'h1F83D9AB, 32'h5BE0CD19
};
```

The `blake3_pipeline.sv` already initializes `init_state[8:11]` from IV and sets `init_state[14] = block_len`, `init_state[15] = flags`, so these inputs will propagate correctly.

### 1.4 Scratchpad: Synchronous Read Model (Corrected from v4)

v4 had a mixed async-data / sync-valid model. v4.1 uses **fully synchronous read**:

```systemverilog
// Synchronous read: both data and valid are registered
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (int i = 0; i < 16; i++) rd_data_r[i] <= 32'd0;
        rd_valid_r <= 1'b0;
    end else begin
        rd_valid_r <= rd_en;
        if (rd_en) begin
            for (int i = 0; i < 16; i++) rd_data_r[i] <= mem[i];
        end
    end
end

always_comb begin
    for (int i = 0; i < 16; i++) rd_data[i] = rd_data_r[i];
end
assign rd_valid = rd_valid_r;
```

This guarantees `rd_data` and `rd_valid` change on the same clock edge. `xcrypto_unit.sv` already waits for `mem_valid` in `S_FETCH_MSG` before transitioning to `S_COMPRESS`, so this 1-cycle latency is handled.

---

## Section 2: P0 — Target Comparator (Corrected)

### 2.1 Byte Order Contract

The BLAKE3 pipeline outputs 8 x 32-bit words in **little-endian** format. The target is stored as 32 bytes where `target[0]` is the most significant byte for comparison purposes.

The comparator must extract bytes from LE words:

```systemverilog
// Extract byte[k] from hash word array (LE word order, LE byte-within-word)
// byte_index = i*4 + j, where i = word index, j = byte lane
function automatic logic [7:0] extract_byte(
    input logic [31:0] words [0:7],
    input int unsigned byte_idx
);
    int unsigned word_idx = byte_idx / 4;
    int unsigned lane     = byte_idx % 4;
    return words[word_idx][lane*8 +: 8];
endfunction
```

### 2.2 Comparison Logic (Corrected: strict `<`, with equality for `meets_target`)

```systemverilog
// Solution valid iff hash < target OR hash == target
// (matches Rust: meets_target returns true on equality)
function automatic logic meets_target(
    input logic [31:0] hash_words [0:7],
    input logic [7:0]  target [0:31]
);
    for (int k = 0; k < 32; k++) begin
        logic [7:0] hb = extract_byte(hash_words, k);
        if (hb < target[k]) return 1'b1;  // hash < target → valid
        if (hb > target[k]) return 1'b0;  // hash > target → invalid
    end
    return 1'b1;  // all equal → valid
endfunction
```

**Note:** The `<` vs `<=` ambiguity from v4 is resolved. The server-side check at `main.rs:15472` uses `!(hash < target)` to reject, which means `hash == target` is accepted (it doesn't trigger the rejection). The `meets_target` functions in `difficulty.rs:335`, `hybrid_mining.rs:726`, and `gpu.rs:217` all return `true` on equality. So `<=` is correct for the solution check.

### 2.3 Best-Hash Comparison

For best-hash tracking (difficulty-weighted rewards), the comparison is strict `<`:

```systemverilog
// Is this hash better (harder) than current best?
wire is_new_best = !meets_target(best_hash_words, hash_bytes);
// Equivalently: hash_bytes < best_hash_bytes (strict less-than)
// We reuse meets_target in the inverse: if best doesn't meet hash-as-target,
// then hash is harder.
```

Or more directly, count leading zero bits (which is what the reward formula uses):

```systemverilog
function automatic logic [5:0] count_leading_zero_bits(
    input logic [31:0] hash_words [0:7]
);
    logic [5:0] zeros = 6'd0;
    for (int k = 0; k < 32; k++) begin
        logic [7:0] b = extract_byte(hash_words, k);
        if (b == 8'd0) begin
            zeros = zeros + 6'd8;
        end else begin
            // Count leading zeros in first non-zero byte
            for (int bit = 7; bit >= 0; bit--) begin
                if (b[bit] == 1'b0) zeros = zeros + 6'd1;
                else return zeros;
            end
            return zeros;
        end
    end
    return zeros;  // All zero hash (impossibly hard)
endfunction
```

---

## Section 3: P0 — Mining Controller (Corrected)

### 3.1 Corrected Interface

The v4 interface had `chain_msg[0:15]` and `chain_length`, implying the controller constructs a message and tells Xcrypto to chain. With the corrected chain semantics from Section 1.3, the controller's role simplifies:

1. Write 10 words to the scratchpad (challenge[0:7] + nonce[8:9], words 10-15 = 0)
2. Issue `blake3.chain rd, scratchpad_addr, 100`
3. Read hash result (8 words)
4. Compare against target (byte-extracted from LE words)
5. Track best hash
6. Increment nonce, repeat

The controller does NOT need to manage chain internals — that's handled by `xcrypto_unit.sv` with the corrected chain semantics.

### 3.2 Simplified Controller FSM

```
IDLE → LOAD_SCRATCHPAD → ISSUE_CHAIN → WAIT_CHAIN → CHECK_RESULT → (loop or REPORT)
```

| State | Cycles | Action |
|-------|--------|--------|
| IDLE | 1 | Wait for `start` |
| LOAD_SCRATCHPAD | 10 | Write 10 words (challenge[0:7] + nonce_lo + nonce_hi) via narrow port |
| ISSUE_CHAIN | 1 | Assert Xcrypto request with chain_length=100 |
| WAIT_CHAIN | ~1,401+ | Wait for `blake3.chain` to complete (100 compressions) |
| CHECK_RESULT | 1 | Read hash, compare target, update best |
| NEXT_NONCE | 1 | Increment nonce, go to LOAD_SCRATCHPAD |

**Optimization:** Only words 8-9 (nonce) change per nonce. After the first scratchpad load, subsequent iterations only write 2 words (nonce_lo, nonce_hi), reducing LOAD_SCRATCHPAD to 2 cycles.

---

## Section 4: Control Signal Semantics (NEW — addresses v4 weakness)

Every output signal from `mining_controller` has a defined persistence model:

| Signal | Type | Definition |
|--------|------|-----------|
| `busy` | **Level** | High while mining active. Low when idle. Cleared by FSM returning to IDLE. |
| `solution_found` | **Sticky** | Set when hash meets target. Remains high until cleared by host writing `CTRL.clear_solution` or new `start`. |
| `solution_nonce` | **Snapshot** | Latched when `solution_found` goes high. Stable until next solution or `start`. |
| `solution_hash` | **Snapshot** | Latched atomically with `solution_nonce`. 8 words stable together. |
| `best_updated` | **Sticky** | Set when a new best hash is found. Cleared by host writing `CTRL.clear_best` or new `start`. |
| `best_nonce` | **Double-buffered** | Shadow register updated on each new best. Promoted to read register only when host writes `CTRL.snapshot_best`. Prevents torn reads across SPI bus beats. |
| `best_hash` | **Double-buffered** | Same as `best_nonce`. Shadow + snapshot model. |
| `best_leading_zeros` | **Level** | Leading zero count of current best hash. Updated combinationally from best_hash. |
| `hashes_computed` | **Free-running counter** | 64-bit, increments on each completed chain. Wraps. Read at any time. |
| `hashrate` | **Sampled** | Hardware timer counts hashes per 1-second window. Updated every second. |

**Double-buffer protocol for best-hash reads over SPI:**

```
1. Host writes CTRL.snapshot_best (1 cycle)
2. Shadow registers → snapshot registers (atomic copy)
3. Host reads BEST_NONCE (2 SPI reads for 64 bits)
4. Host reads BEST_HASH (8 SPI reads for 256 bits)
   All reads return stable snapshot values — no tearing
```

---

## Section 5: Chain-Loop Optimization (Downgraded to Implementation Target)

### 5.1 Goal

Eliminate the 2 dead cycles per chain iteration (`S_CHAIN_WRITEBACK` + `S_COMPRESS` launch) to reduce per-nonce latency from ~1,600 to ~1,400 cycles.

### 5.2 v4.1 Position (Changed from v4)

v4 prescribed an exact FSM state deletion. The independent review correctly identified that this may create:

- A long combinational path from `pipe_hash_out` through the CV mux to the pipeline input register
- A potential launch/accept race if `pipe_out_valid` and `pipe_in_valid` interact in the same cycle
- Timing violations at high clock speeds (particularly on ASIC at 500 MHz+)

**v4.1 recommendation:** Treat this as an **implementation target**, not a prescribed patch. Two acceptable approaches:

**Approach A (safe):** Keep the current 2-dead-cycle design for FPGA validation. Optimize in ASIC backend if timing permits.

**Approach B (aggressive):** Collapse to 1 dead cycle by doing CV writeback and compress launch in the same cycle, but through a **bypass register** rather than direct combinational path:

```
S_WAIT_PIPELINE (pipe_out_valid) → S_RELAUNCH (1 cycle: latch hash into message block, set CV=IV, launch pipeline) → S_WAIT_PIPELINE (14 cycles)
```

This is 15 cycles per chain iteration instead of 16 (current) or 14 (v4's overly aggressive target). The 6.25% improvement is less dramatic but timing-safe.

**Dragon Ball should decide** based on their synthesis timing reports. If the FPGA meets 100 MHz with margin, Approach B is safe. If timing is tight, keep Approach A.

---

## Section 6: SPI Host Interface (Updated Register Map)

### 6.1 Register Map (Updated with missing fields from v4 review)

| Offset | Width | R/W | Name | Description |
|--------|-------|-----|------|-------------|
| 0x00 | 32-bit | R | VERSION | `{8'h01, 8'h00, 16'hNNNN}` (major.minor.build) |
| 0x04 | 32-bit | R | CAPS | Capability bits: `[0]=blake3, [1]=vdf, [2]=best_hash, [3]=irq` |
| 0x08 | 32-bit | R/W | CTRL | Control register (see below) |
| 0x0C | 32-bit | R | STATUS | Status register (see below) |
| 0x10 | 32-bit | R | IRQ_STATUS | Sticky interrupt bits (write-1-to-clear) |
| 0x14 | 32-bit | R/W | IRQ_ENABLE | Interrupt enable mask |
| 0x20 | 256-bit | W | CHALLENGE | Challenge hash (8 x 32-bit words, LE) |
| 0x40 | 256-bit | W | TARGET | Difficulty target (32 bytes) |
| 0x60 | 64-bit | W | NONCE_START | Starting nonce (LE u64) |
| 0x68 | 64-bit | W | NONCE_END | Ending nonce (LE u64) |
| 0x70 | 32-bit | W | JOB_ID | Host-assigned job ID (echoed back with results) |
| 0x74 | 32-bit | R/W | WORK_SEQ | Atomic work update sequence counter |
| 0x80 | 32-bit | R | SOL_JOB_ID | Job ID that produced this solution |
| 0x84 | 64-bit | R | SOL_NONCE | Solution nonce (snapshot) |
| 0x8C | 256-bit | R | SOL_HASH | Solution hash (snapshot, 8 words) |
| 0xAC | 32-bit | R | SOL_LEADING_ZEROS | Leading zero bits of solution hash |
| 0xB0 | 32-bit | R | BEST_JOB_ID | Job ID of best hash |
| 0xB4 | 64-bit | R | BEST_NONCE | Best nonce (double-buffered snapshot) |
| 0xBC | 256-bit | R | BEST_HASH | Best hash (double-buffered snapshot) |
| 0xDC | 32-bit | R | BEST_LEADING_ZEROS | Leading zero bits of best hash |
| 0xE0 | 64-bit | R | HASH_COUNT | Total hashes computed (free-running) |
| 0xE8 | 32-bit | R | HASHRATE | Hashes/sec (hardware-measured, 1s window) |
| 0xEC | 32-bit | R | UPTIME | Seconds since last reset |
| 0xF0 | 32-bit | R | TEMPERATURE | Die temperature (if sensor available) |

### 6.2 CTRL Register (0x08)

| Bit | Name | Description |
|-----|------|-------------|
| 0 | START | Write 1 to begin mining (self-clearing) |
| 1 | STOP | Write 1 to stop mining (self-clearing) |
| 2 | CLEAR_SOLUTION | Write 1 to clear `solution_found` sticky flag |
| 3 | CLEAR_BEST | Write 1 to clear `best_updated` sticky flag |
| 4 | SNAPSHOT_BEST | Write 1 to promote shadow best to readable snapshot |
| 5 | SOFT_RESET | Write 1 to reset all state (self-clearing) |

### 6.3 STATUS Register (0x0C)

| Bit | Name | Description |
|-----|------|-------------|
| 0 | BUSY | Mining in progress (level) |
| 1 | SOLUTION_FOUND | Solution found (sticky until cleared) |
| 2 | BEST_UPDATED | New best hash since last clear (sticky) |
| 3 | NONCE_EXHAUSTED | Reached nonce_end without solution |
| 7:4 | TILE_COUNT | Number of active mining tiles |

### 6.4 IRQ_STATUS Register (0x10) — Write-1-to-Clear

| Bit | Name | Description |
|-----|------|-------------|
| 0 | IRQ_SOLUTION | Fired when a valid solution is found |
| 1 | IRQ_BEST | Fired when a new best hash is found |
| 2 | IRQ_EXHAUSTED | Fired when nonce range is exhausted |
| 3 | IRQ_TIMEOUT | Fired if no solution found within configurable timeout |

### 6.5 Atomic Work Update Protocol

To prevent the miner from using a partially-written challenge+target+nonce:

```
1. Host writes CTRL.STOP                     // Pause mining
2. Host writes CHALLENGE (8 words)
3. Host writes TARGET (8 words)
4. Host writes NONCE_START, NONCE_END
5. Host writes JOB_ID
6. Host increments WORK_SEQ                  // Signals all fields are consistent
7. Host writes CTRL.START                    // Resume mining
```

The mining controller checks `WORK_SEQ` changes and only loads new work after `START`. This prevents split-brain where the target is from job N but the challenge is from job N+1.

### 6.6 Endian Definition

All multi-word registers are **little-endian**: word at lowest offset = least significant word. Within each word, bytes are little-endian (matching BLAKE3 internal representation). This means the host reads `CHALLENGE[0x20]` = challenge word 0 = bytes 0-3 of the challenge hash.

---

## Section 7: Multi-Tile Integration (Clarified)

v4 left ambiguous whether each tile gets its own mining controller or shares one.

### 7.1 Architecture Decision

**Each tile gets its own mining controller.** The host partitions the nonce space across tiles:

```
Tile 0: nonce_start = N,           nonce_end = N + range/K
Tile 1: nonce_start = N + range/K, nonce_end = N + 2*range/K
...
```

Each tile has its own SPI register set, offset by tile stride in the address map:

```
Tile 0: 0x0000 - 0x00FF
Tile 1: 0x0100 - 0x01FF
...
```

A shared "global" register at 0xF000 reports aggregate hashrate and best-of-all-tiles hash.

### 7.2 Rationale

- Zero inter-tile coordination overhead
- Each tile is independently testable
- Dragon Ball's host MCU dispatches work to tiles via simple SPI addressing
- Matches their existing ALPH miner architecture (multiple independent hash engines per die)

---

## Section 8: Corrected File Change Summary

| File | Changes Needed | Priority |
|------|---------------|----------|
| `rtl/memory/xcrypto_scratchpad.sv` | **NEW** — synchronous read model | P0 |
| `rtl/mining/mining_controller.sv` | **NEW** — nonce gen, target compare, best-hash, control signals | P0 |
| `rtl/xcrypto/xcrypto_unit.sv` | **FIX** chain semantics: IV as CV, hash→message block, block_len/flags | P0 |
| `rtl/top/qug_tile.sv` | Wire scratchpad, fix chain interface | P0 |
| `rtl/top/qug_soc_top.sv` | Instantiate mining controller | P0 |
| `rtl/pkg/qug_pkg.sv` | Add mining params, BLAKE3 flag constants | P0 |
| `rtl/periph/spi_slave.sv` | **NEW** — SPI slave + register map with IRQ, double-buffer | P2 |
| `sim/Makefile.vcs` | **NEW** — VCS compile script | P2 |
| `fpga/constraints/artix7_355t.xdc` | **NEW** (if Artix-7 confirmed) | P1 |

---

## Section 9: Modules With No Currently Identified Issues

The following modules have no known issues, **pending synthesis results and protocol conformance review.** They may require changes if:

- Vivado synthesis reveals timing violations
- Protocol conformance testing shows behavioral mismatches
- Dragon Ball's FPGA board has constraints we haven't anticipated

| Module | Status |
|--------|--------|
| `blake3_round.sv` | SIGMA tables verified. 2-stage pipeline correct. |
| `blake3_pipeline.sv` | 14-stage pipeline, CV delay chain, finalization XOR. |
| `blake3_state.sv` | Register file operations. May need flag constant additions. |
| `mod_mul_256.sv` | Double-fold Barrett reduction verified. |
| `mod_add_256.sv` | Single-cycle add with conditional subtract. |
| `mod_inv_256.sv` | Fermat's method, correct exponent. |
| `qug_pipeline.sv` | 5-way forwarding, hazard detection. |
| `qug_decoder.sv` | RV32IMC decode (not compliance-tested). |
| `qug_alu.sv` | Standard ALU ops. |
| `qug_regfile.sv` | 32x32 register file. |
| `mem_subsystem.sv` | BRAM + UART mux. |
| `fpga_top.sv` | MMCM + reset sync. |

---

## Section 10: Updated Questions for Dragon Ball

1-6 unchanged from v4. Adding:

7. **Interrupt pin?** Does the FPGA board have a GPIO that can be used as an interrupt output to the host MCU? If yes, which pin? This enables the IRQ_STATUS register to drive a physical interrupt rather than requiring polling.

8. **Multi-tile target count?** How many independent mining tiles on the ASIC? This determines the SPI address map stride and global aggregation logic.

---

*v4.1 is the corrected work order. The most significant change from v4 is the discovery that the BLAKE3 chain loop has incorrect CV/message semantics — the hash output must go into the message block, not the chaining value. This is a functional bug that would produce wrong hashes on mainnet. All other corrections are precision improvements to signal semantics, byte ordering, and interface contracts.*
