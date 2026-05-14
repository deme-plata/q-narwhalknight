# DeepSeek Handoff (Round 3) — Path A: Full Multi-Block BLAKE3 Gadget

**Date:** 2026-05-13
**Project:** Q-NarwhalKnight (Quillon Graph)
**Production status:** **LIVE MAINNET · ~$2 BILLION USD MARKET CAP**
**Supersedes:** the `hash_message` placeholder in `deepseek-handoff-merkle-and-delta-2026-05-13.md`. The blueprint there assumed a primitive that doesn't exist. This document specifies the primitive itself.

---

# WHY THIS DOCUMENT EXISTS

Both prior submissions of Blueprint 1B (Merkle gadget) failed at the same point: `hash_message` was specified as a small helper, but the existing `crates/q-ivc/src/gadgets/blake3.rs` has only a **single-block (64-byte) `verify_hash`**. The Merkle gadget needs to hash 75-byte node messages, and the δ-circuit needs to hash block headers that exceed 1 KB. Neither is single-block. So before any Merkle gadget can land, we need a real multi-block BLAKE3-in-circuit primitive.

That is the deliverable of this handoff: **build the multi-block BLAKE3 gadget on top of the existing `compress` function**. This is one self-contained PR. No Merkle gadget, no δ-circuit, no Nova — just BLAKE3, done correctly.

---

# 🚨 MAINNET PRODUCTION CONSTRAINTS — READ FIRST

You are extending a **live, production blockchain** with a **$2 B market cap**. The constraints below are non-negotiable:

1. **Only touch `crates/q-ivc/src/gadgets/blake3.rs`.** No changes anywhere else in the repo. No `Cargo.toml` changes (the existing dev-deps are sufficient). No new files.
2. **The existing public surface stays compatible.** `Blake3Gadget::compress`, `g_function`, `alloc_as_uint32`, `alloc_bytes_as_words`, `alloc_hash`, `verify_hash` — none of their signatures or behavior may change. Add new functions; do not modify old ones.
3. **No changes to balance_root computation anywhere in the codebase.** This is purely a circuit-side primitive. It does not run during consensus, sync, mining, or block validation. Its only consumer in this PR is the new test suite. The Merkle gadget (separate PR) will be its first non-test consumer.
4. **No `unsafe`, no FFI, no external process calls.**
5. **No `unwrap()` or `panic!()` in non-test code.** All errors return `SynthesisError`.
6. **Match the existing code's API patterns.** Read the file end-to-end before writing. Notice that `compress` returns `Vec<UInt32<F>>` (not `[UInt32; 8]`), that flag values are `u32` literals, that `fpvar_to_uint32` exists for the FpVar bridge. Use these.
7. **Every new function must compile clean and pass tests** in the q-ivc crate. The crate as-is compiles; do not regress that.

If you are about to make a change that doesn't fit these constraints — stop and ask. The cost of a clarifying question is zero. The cost of a misplaced edit on this codebase is huge.

---

# WHAT'S ALREADY IN `blake3.rs` (USE THESE — DO NOT REINVENT)

Reading order: open the file, read top to bottom. Key items you will compose against:

## Constants (already declared, do not redefine)

```rust
const BLAKE3_IV: [u32; 8] = [...];     // 8-word initial chaining value
const BLAKE3_SIGMA: [[usize; 16]; 7];  // 7 round-permutations
const BLAKE3_FLAG_SINGLE: u32 = 0b00001011;  // CHUNK_START|CHUNK_END|ROOT = 11
```

You will ADD new flag constants (see §3 below). Do not change `BLAKE3_FLAG_SINGLE`.

## Functions (existing — use, don't modify)

| Function | Purpose | Returns |
|---|---|---|
| `Blake3Gadget::g_function(a, b, c, d, mx, my)` | One BLAKE3 quarter-round | `(UInt32, UInt32, UInt32, UInt32)` |
| `Blake3Gadget::compress(cs, cv, msg, counter_lo, counter_hi, block_len, flags)` | **The core primitive — call this for every block** | `Vec<UInt32<F>>` (8 words = new CV) |
| `Blake3Gadget::alloc_as_uint32(cs, bytes)` | Native bytes → witnessed `UInt32` words (LE) | `Vec<UInt32<F>>` |
| `Blake3Gadget::alloc_bytes_as_words(cs, bytes)` | Native bytes → witnessed FpVar words (LE) | `Vec<FpVar<F>>` |
| `Blake3Gadget::alloc_hash(cs, &[u8; 32])` | 32-byte hash → 8 FpVar words | `Vec<FpVar<F>>` |
| `Blake3Gadget::verify_hash(cs, preimage_words, expected_hash_words)` | Single-block verify | `Result<()>` |
| `Blake3Gadget::fpvar_to_uint32(v)` | FpVar → constrained UInt32 (bridge) | `Result<UInt32<F>>` (private) |
| `Blake3Gadget::uint32_to_fpvar(w)` | UInt32 → FpVar (bridge) | `Result<FpVar<F>>` (private) |

**`compress` is the only primitive you need to call.** Everything else is glue.

## What `compress` does, exactly

```rust
pub fn compress<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    cv: &[UInt32<F>],      // 8 words: input chaining value
    msg: &[UInt32<F>],     // 16 words: one 64-byte block as LE u32s
    counter_lo: u32,       // chunk counter low 32 bits
    counter_hi: u32,       // chunk counter high 32 bits (always 0 for our use; chunk index < 2^32)
    block_len: u32,        // actual bytes in this block (≤ 64)
    flags: u32,            // see flag bits below
) -> Result<Vec<UInt32<F>>, SynthesisError>;
```

- Returns 8 words = output chaining value
- Constant cost: ~36K constraints per call (7 rounds × 8 G-calls × 640)
- Counter is per-CHUNK, not per-block. All blocks within one chunk share the same counter.

---

# THE BLAKE3 SPEC YOU MUST IMPLEMENT

BLAKE3 hashes any byte string by:

**Step 1 — chunking:** Split input into 1024-byte chunks. Last chunk may be shorter.

**Step 2 — block compression within a chunk:** Each chunk is up to 16 blocks × 64 bytes. Process blocks sequentially:
- Block 0 of chunk: flags include `CHUNK_START` (bit 0 = 1)
- Block N-1 (last block) of chunk: flags include `CHUNK_END` (bit 1 = 2)
- Other blocks: flags = 0 (or just continuation flags if applicable, see ROOT below)
- Counter for every block = chunk index (NOT block index within chunk)
- `block_len` = actual bytes in this block. Last block of last chunk may be < 64; pad message bytes with zeros to fill 64-byte block, but pass the real `block_len` to compress.
- CV for block 0 of chunk = IV (for chunk 0) OR `chaining_value` derived from parent in tree (for ≥ chunk 1, only used in tree mode)
- CV for block N>0 of chunk = output of block N-1's compress
- The output of the **last block of a chunk** is the chunk's chaining value.

**Step 3 — tree construction:**
- If input ≤ 1024 bytes (≤ 1 chunk), the last block of the chunk gets `ROOT` flag added (bit 3 = 8). The output is the final hash.
- If input > 1024 bytes (≥ 2 chunks), do NOT set `ROOT` on chunks. Each chunk's output CV (8 words) becomes a leaf in a binary tree:
  - Pair chunks (CV_0, CV_1) → parent_cv via `compress(IV, [CV_0 || CV_1], counter=0, block_len=64, flags=PARENT)`. PARENT = bit 2 = 4.
  - The 16-word msg input for parent compression is `CV_left[0..8] || CV_right[0..8]` (no padding — both are exactly 8 words = 32 bytes, concatenated to 64 bytes).
  - Continue building the tree pairwise upward.
  - If a level has odd number of nodes, the leftover bubbles up to the next level WITHOUT recompression.
  - The **root parent compression** gets `PARENT | ROOT = 4 | 8 = 12` flag.

**Step 4 — output:** The output of the final compress (whether it's a single-chunk-with-ROOT or root-PARENT) is 8 words = 32 bytes = the BLAKE3 digest.

## Flag bit values (constants you will define)

```rust
const BLAKE3_FLAG_CHUNK_START: u32 = 1 << 0;   // 1
const BLAKE3_FLAG_CHUNK_END:   u32 = 1 << 1;   // 2
const BLAKE3_FLAG_PARENT:      u32 = 1 << 2;   // 4
const BLAKE3_FLAG_ROOT:        u32 = 1 << 3;   // 8
```

(The existing `BLAKE3_FLAG_SINGLE = 11 = CHUNK_START | CHUNK_END | ROOT` is consistent with these.)

## Chunk size constants

```rust
const BLAKE3_BLOCK_LEN: usize = 64;
const BLAKE3_CHUNK_LEN: usize = 1024;
const BLAKE3_BLOCKS_PER_CHUNK: usize = BLAKE3_CHUNK_LEN / BLAKE3_BLOCK_LEN; // 16
```

---

# WHAT TO IMPLEMENT — TIER 1 + TIER 2

## Tier 1 — single-chunk multi-block (REQUIRED for Merkle gadget)

Handles messages of 0 to 1024 bytes. This is what the Merkle gadget needs (max 75 bytes per node hash).

```rust
impl Blake3Gadget {
    /// Hash a single-chunk (≤ 1024 byte) message in-circuit.
    /// Returns 8 UInt32 words (256-bit BLAKE3 digest).
    pub fn hash_single_chunk<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        msg_bytes: &[UInt8<F>],
    ) -> Result<Vec<UInt32<F>>, SynthesisError>;
}
```

### Tier 1 algorithm

1. **Length check:** `msg_bytes.len() ≤ 1024`. Otherwise return `Err(SynthesisError::UnexpectedInput)` (or define a clean error).
2. **Compute block count:** `n_blocks = ceil(msg_bytes.len() / 64).max(1)`. (At least one block even for empty input, per BLAKE3 spec.)
3. **For each block `i` in `0..n_blocks`:**
   - Compute `start = i * 64`, `end = (start + 64).min(msg_bytes.len())`
   - `block_len = end - start` (this is the real-bytes count for this block, 1..=64; spec also allows block_len=0 for empty-message case, in which case n_blocks=1 with block_len=0)
   - Build the 16-word msg array:
     - Take bytes `msg_bytes[start..end]`
     - Pad with `UInt8::constant(0)` to length 64
     - Group into 16 LE u32 words: words[j] = bytes[j*4..j*4+4] interpreted as LE u32
     - Convert each group to `UInt32<F>` via bit decomposition. **CRITICAL:** the words are made of `UInt8<F>` values; you cannot just constant-construct them. Use this pattern:

       ```rust
       fn bytes_to_uint32_le<F: PrimeField>(
           cs: ConstraintSystemRef<F>,
           bytes: &[UInt8<F>],  // exactly 4 bytes
       ) -> Result<UInt32<F>, SynthesisError> {
           assert_eq!(bytes.len(), 4);
           // Get the LE bits from the 4 bytes:
           // bits = byte0_bits_LE || byte1_bits_LE || byte2_bits_LE || byte3_bits_LE
           // UInt32::from_bits_le takes 32 bits LSB-first.
           let mut bits = Vec::with_capacity(32);
           for byte in bytes {
               bits.extend(byte.to_bits_le());
           }
           Ok(UInt32::from_bits_le(&bits))
       }
       ```
       Note: the existing `alloc_as_uint32` allocates fresh witnesses from native bytes — it does NOT work for `UInt8<F>` inputs. You must wire bytes to UInt32 via bit decomposition as shown.
   - Determine flags for this block:
     - `flags = 0u32`
     - If `i == 0`: `flags |= BLAKE3_FLAG_CHUNK_START`
     - If `i == n_blocks - 1`: `flags |= BLAKE3_FLAG_CHUNK_END | BLAKE3_FLAG_ROOT` (last block of last (only) chunk gets ROOT)
   - Determine CV:
     - If `i == 0`: `cv = BLAKE3_IV.iter().map(UInt32::constant).collect::<Vec<_>>()` (starts at IV for chunk 0)
     - Else: `cv` = output of previous compress
   - Call: `cv = Blake3Gadget::compress(cs.clone(), &cv, &msg_words, 0u32 /* counter_lo = chunk_idx = 0 */, 0u32 /* counter_hi */, block_len as u32, flags)?`
4. **Return** the final `cv` (8 UInt32 words = the BLAKE3 hash).

### Edge cases

- **Empty input (len 0):** still compresses one block with `block_len = 0`, all-zero msg, flags = `CHUNK_START | CHUNK_END | ROOT`. The BLAKE3 spec defines this; the output is the well-known empty-string BLAKE3 hash. **Required to handle correctly** — the gadget's correctness depends on it for boundary tests.
- **Exactly 64-byte input (1 block):** n_blocks = 1, block 0 gets all three flags (CHUNK_START | CHUNK_END | ROOT). This must produce IDENTICAL output to the existing `verify_hash` on the same input. A cross-check test against `verify_hash` is required.
- **Exactly 1024-byte input (16 blocks):** n_blocks = 16, blocks 0..14 have flags 0/CHUNK_START/0/... but only block 0 has CHUNK_START and only block 15 has CHUNK_END | ROOT.

## Tier 2 — multi-chunk tree mode (REQUIRED for δ-circuit block header)

Handles messages > 1024 bytes. The δ-circuit's witnessed block header (postcard-serialized BlockHeader) often exceeds 1 KB.

```rust
impl Blake3Gadget {
    /// Hash a message of any length in-circuit using BLAKE3 tree mode.
    /// Returns 8 UInt32 words (256-bit BLAKE3 digest).
    ///
    /// This is the universal entry point — Merkle gadget calls this for 75-byte
    /// node hashes (which dispatches internally to single-chunk mode), and the
    /// δ-circuit calls this for arbitrary-length block headers.
    pub fn hash_message<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        msg_bytes: &[UInt8<F>],
    ) -> Result<Vec<UInt32<F>>, SynthesisError>;
}
```

### Tier 2 algorithm

1. **If `msg_bytes.len() ≤ 1024`:** delegate to `hash_single_chunk` (Tier 1 above), but with a small modification: ROOT flag is only added if this is the only chunk. So:
   - Add a parameter to `hash_single_chunk` (or make an internal variant) that takes `is_root: bool`. When called from Tier 2's chunk loop, pass `is_root = false` (no ROOT on intermediate chunks). The Tier 1 public API always uses `is_root = true`.
2. **Else (multi-chunk):**
   - Split into chunks of 1024 bytes (last chunk may be shorter).
   - For each chunk `c` in `0..n_chunks`:
     - Compute chunk's CV by processing its blocks (exactly like Tier 1 but **without ROOT flag**, and with `counter_lo = c`, `counter_hi = 0`)
     - The last block of each chunk gets `CHUNK_END` but NOT `ROOT`
   - Collect the n_chunks CVs into a `Vec<Vec<UInt32<F>>>` (each inner vec is 8 words).
   - **Build the binary tree** by repeated pairwise parent compressions:
     ```
     while cvs.len() > 1:
         next_level = []
         for pair in cvs.chunks(2):
             if pair.len() == 2:
                 // parent = compress(IV, [pair[0] || pair[1]], 0, 0, 64, PARENT [| ROOT if this becomes the last])
                 parent = compress(cs, &IV, &concat_16_words(&pair[0], &pair[1]),
                                   0, 0, 64,
                                   if next_level.is_empty() && cvs.chunks(2).next_is_last() && top_level
                                       { PARENT | ROOT } else { PARENT })
                 next_level.push(parent)
             else:
                 // odd one out — bubbles up
                 next_level.push(pair[0])
         cvs = next_level
     return cvs[0]  // root (with ROOT flag was applied at the last parent compress)
     ```
   - **CRITICAL:** the ROOT flag must be set on the **final compress call** (whether single-chunk-direct or tree-root-parent), and ONLY on that final call. Get this wrong and the gadget produces a hash that doesn't match native BLAKE3.
   - Helper function: `fn concat_16_words(left: &[UInt32<F>], right: &[UInt32<F>]) -> Vec<UInt32<F>>` — just concatenates 8 + 8 = 16. No allocation, just clones.

### Edge case for Tier 2

- **Exactly 1024-byte input:** dispatches to single-chunk mode. ROOT flag on the last block of the (only) chunk. NO tree mode invoked. Test this boundary explicitly.
- **1025-byte input:** two chunks. First is full (1024 bytes), second is 1 byte. Build a tree of two leaves, root parent compress with `PARENT | ROOT`. Test this boundary explicitly.
- **Multi-level tree (e.g., 4097 bytes = 5 chunks):** tree levels collapse pairwise. With 5 leaves: level 1 = [pair(0,1), pair(2,3), leaf(4)]. Level 2 = [pair(L1.0, L1.1), L1.2]. Level 3 = [pair(L2.0, L2.1)]. Final compress has PARENT | ROOT. Test this with 5 chunks of varying real sizes.

---

# CONSTRAINT COST EXPECTATIONS

Each `compress` call is ~36K constraints. Total cost for the gadget:

| Message size | Compresses | Constraints |
|---|---|---|
| 0-64 bytes (1 block, 1 chunk) | 1 | ~36K |
| 75 bytes (Merkle node, 2 blocks, 1 chunk) | 2 | ~72K |
| 1024 bytes (16 blocks, 1 chunk) | 16 | ~576K |
| 1025-2048 bytes (2 chunks + 1 parent) | 17-32 + 1 = ~33 | ~1.2M |
| 4096 bytes (4 chunks + 3 parents) | 64 + 3 = 67 | ~2.4M |

Plus byte-to-UInt32 bridge cost: 32 constraints per byte × `msg.len()` for the witness allocation.

**Per Merkle path:** 256 node hashes × ~72K = ~18M constraints per path. For 100 transactions × 4 paths each = ~7.2 billion constraints. **This is huge but expected** — it's why the IVC fold step takes 30-90 seconds; Nova's relaxed R1CS handles arbitrary-size single-step circuits.

If the constraint count is significantly different (e.g., >2× expected), there's a bug somewhere. Print the constraint count in the test (see test plan) and compare against this table.

---

# WHAT TESTS YOU MUST INCLUDE

All tests go in the existing `#[cfg(test)] mod tests` block in `blake3.rs`. Field for tests: continue using `ark_bls12_381::Fr` (matches existing convention in the file).

For each test, use the `blake3` crate (already a workspace dep) to compute the native expected hash, and assert byte-for-byte equality against the in-circuit result.

The output of `hash_message` is `Vec<UInt32<F>>` — to compare with `blake3::hash().as_bytes()` (which returns `[u8; 32]`), convert each UInt32 to its 4 little-endian bytes:

```rust
fn uint32_words_to_bytes<F: PrimeField>(words: &[UInt32<F>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        let val = w.value().unwrap();
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}
```

(The `value()` call requires `cs.is_satisfied()` to have been checked, AND the test must use `UInt32::new_witness` / `new_constant` paths that yield retrievable values. This pattern is standard in arkworks tests.)

## Required tests (minimum 10)

| # | Name | Input | What it asserts |
|---|---|---|---|
| 1 | `hash_message_empty` | 0 bytes | Output matches `blake3::hash(b"").as_bytes()` byte-for-byte |
| 2 | `hash_message_one_block_partial` | 11 bytes (`b"smt_leaf_v2"`) | Matches native |
| 3 | `hash_message_one_block_full` | 64 bytes (random pattern, e.g., `[0x55; 64]`) | Matches native; ALSO assert this matches the existing `verify_hash` result on the same input |
| 4 | `hash_message_two_blocks` | 75 bytes (the Merkle node case: `b"smt_node_v2" || [0xAA; 32] || [0xBB; 32]`) | Matches native |
| 5 | `hash_message_one_chunk_full` | 1024 bytes | Matches native; assert n_blocks = 16, n_chunks = 1 internally (via debug-only assertion or by inspecting constraint count) |
| 6 | `hash_message_one_chunk_plus_one` | 1025 bytes | Matches native; assert tree mode kicks in (2 chunks + 1 parent) |
| 7 | `hash_message_two_chunks` | 2048 bytes | Matches native |
| 8 | `hash_message_three_chunks_odd` | 3000 bytes (5 chunks of varying sizes, exercising odd-bubble-up) | Matches native |
| 9 | `hash_message_large_real_world` | a serialized fake BlockHeader, ~1.5-3 KB | Matches native |
| 10 | `hash_message_constraint_count_sanity` | 75 bytes | Print constraint count; assert it's between 50K and 150K (sanity bound for two-block compress + bridge) |

The cross-check via native `blake3::hash` is what verifies correctness. Every test must use it.

Additionally, **DO NOT remove or break the existing tests** (`test_g_function_compiles`, `test_blake3_alloc_helpers`, `test_blake3_gadget_compiles`, plus any others in the file). Run `cargo test --package q-ivc` and confirm everything passes.

---

# IMPLEMENTATION ORDER (FOLLOW THIS)

If you implement in this order, each step is verifiable independently before moving on:

1. **Add flag constants** (`BLAKE3_FLAG_CHUNK_START`, etc.) and size constants. Compile-check. ← 5 minutes.
2. **Write `bytes_to_uint32_le` private helper.** Add a unit test that exercises just this — feed it 4 known bytes and assert the resulting UInt32 value. ← 30 minutes.
3. **Write `hash_single_chunk` (Tier 1)** for inputs of length 1-64 (single block only). Skip the multi-block case for now. Test with empty input + 64-byte input + cross-check with `verify_hash`. ← 2 hours.
4. **Extend `hash_single_chunk` to handle 65-1024 bytes** (multi-block within one chunk). Test with 75 bytes, 128 bytes, 1024 bytes. ← 2 hours.
5. **Write `hash_message` Tier 2 entry point.** If len ≤ 1024, delegate to single-chunk. Else, implement chunked tree-mode. Test with 1025 bytes, 2048, 3000, 4096. ← 4 hours.
6. **Add the constraint-count sanity test.** Print actual numbers, compare against expected table. Adjust if off. ← 30 minutes.
7. **Run `cargo check --package q-ivc` and `cargo test --package q-ivc`.** Both must pass. ← incremental.

**Total estimated work: ~1 working day if everything is straightforward, ~3 working days if BLAKE3 spec subtleties trip you up.**

---

# THE MOST COMMON WAYS THIS WILL GO WRONG

Read these BEFORE you start coding. They are the things that have wasted time historically:

1. **ROOT flag on the wrong call.** Only the FINAL compress (in single-chunk mode: last block of the chunk; in tree mode: the topmost parent) gets ROOT. If you accidentally set ROOT on a non-final block, the output will be wrong AND it will pass any test that doesn't cross-check against native BLAKE3. **Cross-check is mandatory.**

2. **Counter confusion.** Counter is per-CHUNK, NOT per-block. All 16 blocks of chunk 0 share counter=0. All 16 blocks of chunk 1 share counter=1. PARENT compressions in tree mode always have counter=0.

3. **block_len for the last block.** Even if you pad the block with zeros to 64 bytes for the message words, `block_len` must equal the REAL byte count (1 to 64). Empty input has block_len=0. This is critical for BLAKE3's domain separation.

4. **Byte ordering.** The 16 message words are built from the byte buffer as little-endian u32: `words[j] = bytes[j*4..j*4+4]` interpreted as LE. Get the endianness wrong here and every hash will be wrong. **Test 3** (full 64-byte block matches `verify_hash`) catches this.

5. **`UInt8` → `UInt32` allocation.** Do NOT call `UInt32::new_witness(cs, || Ok(value_from_uint8))`. That would allocate a fresh witness with no constraint linking it to the bytes you started with — soundness hole, the gadget would silently accept wrong inputs. Use `UInt32::from_bits_le(&bits)` where `bits` come from `UInt8::to_bits_le()`. This makes the UInt32 a *linear function of the bytes*, not a separate witness.

6. **Odd-bubble-up in tree mode.** When a level has an odd number of nodes, the leftover bubbles up WITHOUT recompression. Common mistake: pairing the leftover with itself and recompressing. Wrong. Just propagate it.

7. **Tier 2 for ≤1024 bytes calls Tier 1 with ROOT.** If your Tier 2 implementation calls a "Tier 1 with no root" helper internally for the multi-chunk case, make sure the single-chunk case (when `len ≤ 1024`) still gets ROOT. One clean way is to make `hash_single_chunk` take an `is_root: bool` parameter internally, with the public API hard-coding `true`.

8. **Allocating zeros for padding.** The padding bytes are CONSTANTS, not witnesses. Use `UInt8::constant(0u8)`, not `UInt8::new_witness(cs, || Ok(0u8))`. Both produce the same R1CS values but the former emits 0 constraints; the latter emits ~8 unnecessary constraints per zero byte.

9. **Forgetting `cs.clone()` everywhere.** `ConstraintSystemRef<F>` is `Clone`. Every internal call needs its own clone, both for borrow-checker reasons and because some arkworks APIs consume the cs.

10. **Native BLAKE3 cross-check assertion is non-negotiable.** If a test passes without comparing against `blake3::hash()`, it doesn't prove the gadget is correct — only that the gadget is internally consistent. Every test must end with `assert_eq!(circuit_bytes, blake3_hash_bytes)`.

---

# ACCEPTANCE CRITERIA

A reviewable PR satisfies all of these:

- [ ] `cargo check --package q-ivc` — clean
- [ ] `cargo test --package q-ivc` — all tests pass (old + 10 new)
- [ ] No files modified outside `crates/q-ivc/src/gadgets/blake3.rs`
- [ ] No `Cargo.toml` changes
- [ ] Existing public functions (`compress`, `verify_hash`, `g_function`, etc.) unchanged
- [ ] No `unsafe`, no `unwrap()` in production code, no `panic!()`
- [ ] Every required test cross-checks against `blake3::hash()` byte-for-byte
- [ ] `hash_message` handles empty input, 1-block, 16-block (1 KB), 17-block (1 KB + 1 byte), 5-chunk
- [ ] Constraint count for 75-byte input is between 50K and 150K (sanity)
- [ ] All 10 listed test cases exist and pass

If any checkbox fails, the PR is rejected. The deliverable is "BLAKE3 multi-block + tree mode in-circuit, cross-checked against native BLAKE3, with no regressions." Anything less is incomplete.

---

# WHAT'S NEXT AFTER THIS

Once this lands and soaks for ~1 week in the test suite + no regressions:

1. The Merkle gadget (Blueprint 1B from `deepseek-handoff-merkle-and-delta-2026-05-13.md`) can be implemented for real — it now has a working `hash_message` to call. The blueprint there is correct in shape; only the assumption about `hash_message` was wrong, and this PR fixes that.
2. The δ-circuit (Blueprint 2) can use `hash_message` for the block-header hash check.
3. Tests in the Merkle gadget and δ-circuit will use this BLAKE3 gadget to verify SmtProof outputs from the production `BalanceSmt` (already shipped at `crates/q-storage/src/balance_smt.rs`).

This is the gating piece. Get it right and the rest of the IVC stack unblocks cleanly.

— Server Beta, 2026-05-13
