# Technical Review: New Node Sync Blocked by Missing Early Blocks

**Date:** 2026-04-15
**Project:** Quillon Graph (Q-NarwhalKnight) — $1B market cap, 15.3M blocks, mainnet-genesis
**Severity:** High — no new nodes can join the network
**Constraint:** Zero-risk changes only (mainnet with real money)

---

## 1. Problem Statement

New nodes cannot sync from the network. They start at height 0 and request blocks via P2P turbo sync, but all peers respond with **0 blocks** for heights 0 through ~1.6M (Epsilon) or ~10.7M (Beta).

**Evidence from logs:**
```
multi_get: 0/200 keys found for heights 1001..=1200     ← Epsilon (176GB DB)
multi_get: 0/200 keys found for heights 5001..=5200     ← Epsilon
multi_get: 200/200 keys found for heights 14726001..=14726200  ← Epsilon (blocks exist here)
multi_get: 0/200 keys found for heights 1001..=1200     ← Beta (100GB DB)
multi_get: 134/200 keys found for heights 10699001..=10699200  ← Beta (earliest blocks)
```

**Root cause:** Early blocks (heights 0 through ~1.6M) were pruned or lost. They don't exist on any node. The 176GB on Epsilon contains blocks from ~1.6M to 15.3M.

---

## 2. Current Architecture (How Sync Works)

### 2.1 New Node Bootstrap Sequence

When a fresh node starts:

```
1. P2P Connect       → Finds peers via bootstrap addresses
2. State Sync        → HTTP GET /api/v1/sync/full-state (after 10s)
                        Returns: 279 wallet_balances, 28 contracts, 23 pools, 1913 token_balances
                        ✅ Balances are immediately correct (no block replay needed)
3. Turbo Sync        → Requests blocks starting from height 0
                        Gets 0 blocks for heights 0..1,600,000 → STUCK
4. Challenge API     → Returns "network height unknown" because local height = 0
                        Miners can't mine. Node appears dead.
```

The problem is step 3: turbo sync starts from 0 and never advances because nobody has those blocks.

### 2.2 Balance Safety (CONFIRMED SAFE)

Mining rewards are stored in **separate RocksDB column families** (`CF_BALANCES`, `CF_MANIFEST`) independent of `CF_BLOCKS`. The state sync endpoint serves all 279 wallet balances directly from memory — no block replay required. Early miners' rewards are:

- ✅ Stored in persistent RocksDB `manifest` column family
- ✅ Transferred to new nodes via HTTP state sync (`/api/v1/sync/full-state`)
- ✅ Propagated via P2P gossipsub when new blocks are mined
- ✅ Deterministically derived from block processing (idempotent, LRU-cached)

**Balances do NOT depend on replaying blocks from genesis.** A node can have correct balances at height 15M without ever seeing blocks 0-1.6M.

### 2.3 Block-Pack Handler Flow

```
New node sends:  "Give me blocks 1001-2000"
                      ↓
Peer receives:   get_qblocks_range(1001, 200)
                      ↓
Storage layer:   multi_get(CF_BLOCKS, ["qblock:height:1001", ..., "qblock:height:1200"])
                      ↓
Result:          0/200 found  →  response: empty block pack
                      ↓
New node:        Got 0 blocks. Retry same range... forever.
```

The block-pack handler queries only the current key format (`qblock:height:{N}`), then tries two legacy formats. All return 0 for early heights because the blocks genuinely don't exist in any format.

### 2.4 Turbo Sync Chunking

Turbo sync divides the total range into 1000-block chunks and requests them in parallel:

```rust
// turbo_sync.rs — simplified
let chunks: Vec<(u64, u64)> = (0..network_height)
    .step_by(1000)
    .map(|start| (start, start + 999))
    .collect();

for chunk in chunks {
    // Request blocks chunk.0..chunk.1 from a random peer
    // If peer returns 0 blocks → chunk stays in queue → retried later
}
```

The problem: chunks 0-1000, 1001-2000, ..., 1600001-1601000 all fail permanently. They clog the queue and prevent the sync from ever reaching heights where blocks exist.

---

## 3. What We Need

A mechanism for new nodes to **skip the gap** and start syncing from the first height where blocks actually exist on the network. Requirements:

1. **Zero risk to existing nodes** — no changes to block validation, consensus, or storage
2. **Correct balances** — new nodes must have accurate wallet balances (already solved by state sync)
3. **Correct block history** — from the skip-point forward, all blocks must be verified normally
4. **Automatic** — new node operators shouldn't need to manually configure anything
5. **Transparent** — the node should clearly indicate it has a partial history

---

## 4. Proposed Solutions (For DeepSeek Review)

### Solution A: Turbo Sync Gap Detection + Auto-Skip

**Concept:** When turbo sync requests N consecutive chunk ranges and gets 0 blocks for all of them, exponentially skip forward to find where blocks start.

```
Pseudocode:
1. Request blocks 0-1000       → 0 blocks
2. Request blocks 1001-2000    → 0 blocks
3. Request blocks 2001-3000    → 0 blocks
   (3 consecutive failures)
4. Skip forward: try 10,000    → 0 blocks
5. Skip forward: try 100,000   → 0 blocks
6. Skip forward: try 1,000,000 → 0 blocks
7. Skip forward: try 1,600,000 → 3 blocks found!
8. Binary search: 1,500,000-1,600,000 to find exact start
9. Set sync_start = first_available_height
10. Continue normal turbo sync from there
```

**Implementation location:** `crates/q-storage/src/turbo_sync.rs`
- Add `gap_detection_state` to track consecutive empty responses
- After K consecutive empty chunks (e.g., K=5), start probing ahead
- Use a "probe request" that asks a peer "what's your lowest block height?"
- Or probe with exponential jumps: 10K, 100K, 1M, 5M, 10M, 14M

**Pros:**
- Fully automatic, no configuration needed
- Works regardless of where the gap ends
- Zero impact on existing synced nodes (they never trigger gap detection)

**Cons:**
- Takes 10-30 seconds to probe and find the start
- Requires the node to accept a "partial history" state

### Solution B: Peer-Advertised Earliest Height

**Concept:** Each peer announces its `earliest_available_height` alongside its `current_height` in the P2P height announcements. New nodes use the minimum `earliest_available_height` across all peers as their sync start.

```rust
// In peer height announcement (gossipsub):
struct PeerHeightAnnouncement {
    peer_id: PeerId,
    current_height: u64,
    earliest_height: u64,  // NEW: lowest height this peer can serve
    network_id: String,
    version: String,
}
```

**Implementation:**
- `crates/q-network/src/unified_network_manager.rs`: Add `earliest_height` to announcements
- `crates/q-storage/src/turbo_sync.rs`: Use `min(peer.earliest_height)` as sync start
- Compute `earliest_height` on startup by scanning the first few hundred blocks

**Pros:**
- New nodes know immediately where to start (no probing delay)
- Protocol-level solution, works for all future nodes
- Old nodes (without this field) are ignored (field is optional)

**Cons:**
- Requires a P2P message format change (backward compatible via `#[serde(default)]`)
- Needs scanning on startup to determine own `earliest_height`

### Solution C: Bootstrap Snapshot with Block Skip Certificate

**Concept:** The state sync endpoint already serves balances, contracts, and pools. Extend it with a "Block Skip Certificate" — a signed attestation from a trusted peer that says "the chain is valid from height H, here's the state at H."

```rust
struct FullStateSnapshot {
    // Existing:
    wallet_balances: HashMap<String, String>,
    contracts: Vec<Contract>,
    liquidity_pools: Vec<Pool>,
    block_height: u64,
    
    // NEW:
    earliest_available_block: u64,
    state_root_hash: [u8; 32],  // Merkle root of all balances at this height
    last_block_hash: String,     // Hash of the block at block_height
    attesting_peers: Vec<PeerId>, // Peers that confirmed this state
}
```

New nodes would:
1. Fetch state snapshot (already happens)
2. Start block sync from `earliest_available_block` (skip the gap)
3. Validate the chain from that point forward
4. The `state_root_hash` provides cryptographic assurance that the balances are correct

**Pros:**
- Cryptographic proof of state validity
- New nodes don't need to trust any single peer blindly
- Fastest possible sync start

**Cons:**
- Most complex to implement
- Requires defining a Merkle root over balances (new code)

### Solution D: Hybrid — Immediate State + Lazy Block Fill

**Concept:** Combine what already works:
1. State sync gives correct balances immediately (already works)
2. Set the node's `current_height` to the state sync height (new)
3. Start mining/serving API immediately (currently blocked)
4. Turbo sync fills blocks backward from the tip in the background
5. Mark blocks 0-earliest_available as "historical gap — not available"

```rust
// On startup after state sync:
if local_height == 0 && state_sync_height > 0 {
    // We have state but no blocks. Start operating at state_sync height.
    current_height_atomic.store(state_sync_height, Ordering::SeqCst);
    
    // Start turbo sync from (state_sync_height - 1000) forward to fill recent blocks
    turbo_sync.sync_from(state_sync_height.saturating_sub(1000));
    
    // The challenge API can now serve challenges
    // The node can participate in mining and validation
}
```

**Pros:**
- Uses existing infrastructure (state sync already works)
- New nodes operational within seconds, not hours
- No protocol changes needed
- Block history fills in over time from the tip

**Cons:**
- Node briefly operates without full block history (but has correct state)
- Need to handle "block not found" API responses gracefully for missing ranges

---

## 5. Network State Summary

| Server | Earliest Block | Latest Block | DB Size | Gap |
|--------|---------------|-------------|---------|-----|
| Epsilon (10Gbit) | ~1,645,947 (sparse) | 15,312,000+ | 176GB | 0 → 1.6M lost |
| Beta (production) | ~10,761,000 | 15,297,000+ | 100GB | 0 → 10.7M lost |
| Gamma (backup) | unknown | ~15M | unknown | likely similar |
| New node | 0 | 0 | 0 | cannot sync at all |

Total blocks on network: ~13.7M (from 1.6M to 15.3M on Epsilon)
Missing blocks: ~1.6M (from genesis to first available)
These blocks are permanently lost unless a backup exists.

---

## 6. Relevant Code Locations

| File | Lines | Component |
|------|-------|-----------|
| `crates/q-storage/src/turbo_sync.rs` | 5787-5900 | `sync_to_height()` — decides where to start syncing |
| `crates/q-storage/src/turbo_sync.rs` | 3083-3085 | `get_local_height()` — returns highest contiguous block |
| `crates/q-storage/src/lib.rs` | 2003-2090 | `get_qblocks_range()` — block-pack serving (multi_get) |
| `crates/q-storage/src/lib.rs` | 2195-2276 | `get_qblock_any_format()` — multi-format fallback |
| `crates/q-storage/src/lib.rs` | 3065-3137 | `scan_highest_contiguous_block_internal()` — height recovery |
| `crates/q-api-server/src/state_sync_api.rs` | 37-45 | `FullStateSnapshot` struct |
| `crates/q-api-server/src/state_sync_api.rs` | 146-191 | `spawn_state_sync_task()` — initial + periodic sync |
| `crates/q-api-server/src/state_sync_api.rs` | 1052+ | HTTP endpoint serving state snapshots |
| `crates/q-network/src/unified_network_manager.rs` | 3379-3505 | Block-pack request handler |
| `crates/q-storage/src/balance_consensus.rs` | 1-150 | Balance processing (deterministic, idempotent) |

---

## 7. Questions for Review

1. Which solution (A, B, C, D, or a combination) is safest for a $1B mainnet?
2. For Solution A (gap detection): what's a safe value for K (consecutive empty chunks before skipping)? We need to avoid false positives where a peer is just slow.
3. For Solution D (hybrid): is it safe for a node to serve mining challenges without full block history if it has correct balances from state sync?
4. Should we combine approaches? E.g., Solution B (peer-advertised earliest height) + Solution D (immediate operation after state sync)?
5. How should the node communicate to its operator that it has a partial block history?

---

## 8. Constraints

- **MAINNET** — $1B market cap, 279 wallets, 40+ active miners
- **Zero risk to balances** — mining rewards must be preserved exactly
- **Backward compatible** — old node versions must still work alongside new ones
- **No consensus changes** — block validation rules stay identical
- **No hard fork** — this is an operational improvement, not a protocol change

---

*This review is part of the Quillon Graph infrastructure. The sync gap affects new node operators trying to join the network — currently impossible without a manual database copy.*
