# Technical Review: Legacy Block-Pack Parsing for Operation Twelve Leagues Deep

**Date:** 2026-04-07  
**Severity:** Critical (mainnet, $920M cap)  
**Prepared for:** DeepSeek peer review  
**Classification:** Safe code change — read-only block parsing, no chain modification  

## 1. Problem

The only block-producing peer sends `Vec<QBlock>` responses in bincode, but with an OLDER QBlock struct layout. Our current fallback at `block_pack.rs:188` tries:

```rust
bincode::deserialize::<Vec<QBlock>>(buf)  // Fails — current QBlock struct incompatible
```

The old peer's QBlock is missing fields like `phase`, `producer_id`, `producer_public_key`, etc. that were added in later versions.

## 2. Existing Infrastructure

The codebase ALREADY has legacy QBlock definitions and proven conversion code in `crates/q-types/src/legacy.rs`:

| Struct | Era | Key Differences |
|--------|-----|-----------------|
| `LegacyQBlock` | pre-v1.0.60 | `LegacyTransaction` (no tx_type), `LegacyBalanceUpdate` (u64 not u128) |
| `LegacyQBlockV2` | v1.0.60-v1.0.85 | `LegacyQuantumMetadata` (pre-sqisign), `LegacyTransactionV2` (has tx_type, u64 amounts) |
| `LegacyQBlockV3` | pre-v1.0.60 old QM | `LegacyQuantumMetadata` + `LegacyTransaction` |

Each has `impl From<LegacyQBlock*> for QBlock` (lines 111, 254, 283 of legacy.rs).

The function `deserialize_qblock_with_fallback()` at line 306-346 already tries all 4 formats for single-block deserialization. This is used by the storage layer for EVERY block read from RocksDB — proven in production on 13M+ blocks.

## 3. Proposed Fix

In `crates/q-types/src/block_pack.rs`, after the `Vec<QBlock>` attempt (line 188), add:

```rust
// v10.2.8: Try legacy QBlock formats for older peers
use crate::legacy::{LegacyQBlock, LegacyQBlockV2, LegacyQBlockV3};

// Try Vec<LegacyQBlockV2> (most likely format for recent-but-old peers)
if let Ok(legacy_blocks) = bincode::deserialize::<Vec<LegacyQBlockV2>>(buf) {
    if !legacy_blocks.is_empty() {
        let blocks: Vec<QBlock> = legacy_blocks.into_iter().map(|b| b.into()).collect();
        let start_height = blocks.first().unwrap().header.height;
        let end_height = blocks.last().unwrap().header.height;
        return Ok(BlockPackResponse {
            blocks, start_height, end_height, has_more: false, peer_height: 0,
        });
    }
}

// Try Vec<LegacyQBlockV3>
if let Ok(legacy_blocks) = bincode::deserialize::<Vec<LegacyQBlockV3>>(buf) { ... }

// Try Vec<LegacyQBlock> (oldest format)
if let Ok(legacy_blocks) = bincode::deserialize::<Vec<LegacyQBlock>>(buf) { ... }
```

## 4. Safety Analysis

**Why this is safe:**
- Read-only parsing — no blocks are created, modified, or signed
- Uses proven conversion code from `legacy.rs` (production-tested on 13M+ blocks)
- The `From<LegacyQBlock*> for QBlock` conversions fill missing fields with defaults
- If parsing fails for all formats, falls through to existing CBOR/JSON fallbacks
- No consensus, validation, or balance changes
- No chain state modification — just receiving blocks from peers

**What could go wrong:**
- Legacy struct matches wrong data -> corrupt blocks accepted
  - Mitigated: blocks go through full validation after deserialization (signature, hash, parent chain)
- Performance: trying 3 extra deserializations per response
  - Mitigated: bincode deserialization is microseconds; only runs on parse failure

**What this does NOT do:**
- Does NOT produce blocks
- Does NOT fork the chain
- Does NOT modify balances
- Does NOT change consensus rules
- Does NOT require miner cooperation

## 5. Files to Modify

- `crates/q-types/src/block_pack.rs` lines 188-200 — add 3 legacy Vec fallbacks

## 6. Testing Plan

1. Build on Epsilon (Debian 12 Docker, 25 min)
2. Deploy to Epsilon
3. Monitor: `journalctl | grep 'Downloaded.*applied'` — blocks from miner peer
4. Verify: `current_height` advances past 13,475,449
5. If no progress: check which legacy format parsed (add logging for each attempt)
6. If all fail: hex dump shows the EXACT QBlock layout and we write a targeted deserializer

## 7. Recommendation

Implement Path A only (legacy parsing). Do NOT attempt Path B (independent block production / forking) — too risky for $920M mainnet.

---

**Prepared by:** Claude Code (Server Alpha)  
**Review requested from:** DeepSeek  
**Classification:** Safe read-only change
