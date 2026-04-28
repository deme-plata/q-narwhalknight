# Q-NarwhalKnight — Balance Divergence Root Cause Technical Review
**Date:** 2026-04-28  
**Status:** For external consultation (DeepSeek / ChatGPT)  
**Auditors:** Claude Code (4 parallel subagents) + Claude Sonnet 4.6 synthesis  
**Codebase:** `/opt/orobit/shared/q-narwhalknight` (branch: `feature/safe-batched-sync-v1.0.2`)  
**Network:** mainnet-genesis, ~16.5M blocks, live production

---

## 1. Executive Summary

The Q-NarwhalKnight blockchain has a critical architectural defect: **wallet balance state is not deterministically derivable from block chain data alone.** Different nodes that have processed the same set of blocks end up with different wallet balances. A new node syncing from genesis will produce different balances than Epsilon (the authoritative genesis node, running since day 1). The convergence migration designed to fix this made things worse (9,082 QUG post-migration vs. 479 QUG on Epsilon for the master wallet — a 19× discrepancy).

This document presents four root causes, identified by auditing 23 distinct balance write paths, the full P2P gossip subsystem, the block application pipeline, and the convergence migration code.

---

## 2. Architecture Overview: How Balances Are Supposed to Work

### What a correct blockchain does

In a standard blockchain (Bitcoin, Ethereum), the balance of every address is 100% derivable from the block chain:

```
balance(addr) = sum(coinbase_txs to addr) + sum(incoming transfers) - sum(outgoing transfers)
```

Any node replaying blocks from genesis in order must arrive at the same balance for every address. Balance state is **fully deterministic from block data**.

### What Q-NarwhalKnight actually does

Q-NarwhalKnight has **five independent systems** that each modify wallet balances:

1. **Block transaction processing** — deterministic, from block coinbase/transfer TXs
2. **P2P balance gossip** — non-deterministic delta messages broadcast over gossipsub, outside of blocks (now disabled by default)
3. **Migration system** — 9 sequential startup migrations that rebuild, scale, or adjust balances
4. **DEX adjustment system** — startup idempotent recalculation of DEX-related QUG debits/credits
5. **Authority sync / bootstrap sync** — one-time import of full balance snapshot from a trusted peer

Systems 2–5 all operate **outside the block chain**. Their effects cannot be reconstructed by a node replaying blocks from genesis. This is why new nodes diverge from Epsilon — they have the same block data, but Epsilon has applied years of off-chain corrections that a fresh node has never seen.

---

## 3. Root Cause #1 — Inflated Historical Coinbase Amounts (The 34× Bug)

### What happened

Before approximately v7.1.2 (early 2026), block producers used a fixed-rate emission formula:

```rust
// OLD (buggy): hardcoded 1.0 blocks/sec assumption
fn calculate_block_reward_time_based(genesis_ts: u64, block_ts: u64) -> u128 {
    let elapsed = block_ts - genesis_ts;
    let annual = annual_emission(era_at_time(elapsed));
    annual / 31_557_600  // seconds/year — assumes 1 block/sec
}
```

The actual mainnet block rate was approximately 1.44 blocks/sec (measured over 1000 rolling 10-second windows). Each block was given the full per-second reward, so each second's worth of emission was paid out to 1.44 blocks. Total annual inflation: 1.44× intended.

Over months, with compounding, the cumulative overshoot was documented in the codebase as approximately **34×**.

### The permanent damage

Every block from the early era has its inflated `block_tx.amount` baked into the block's Merkle tree root and cryptographic hash. **These amounts cannot be changed without invalidating those blocks.** The chain permanently encodes the inflation.

```
block.transactions[0].amount = INFLATED_REWARD   // 34× too high
                                                  // SHA3-256 covers this field
                                                  // Cannot be corrected retroactively
```

### What this means for a fresh node

A fresh node syncing from genesis reads `block_tx.amount` from each stored block and adds it directly to the miner's balance:

```rust
// In safe_batched_convergence_v103, lib.rs:6738
if is_coinbase {
    if block_tx.to != [0u8; 32] && block_tx.amount > 0 {
        let entry = balances.entry(block_tx.to).or_insert(0);
        *entry = entry.saturating_add(block_tx.amount);  // reads INFLATED value from block
    }
}
```

The fresh node faithfully applies the inflated amounts from old blocks. Epsilon, by contrast, ran various migrations (v8.5.1, v8.8.1, v8.8.5) that scaled those balances back down. The corrections exist only in Epsilon's RocksDB — not in the block chain data.

**Consequence:** A fresh node summing coinbase TXs from all 16.5M blocks will produce a `chain_total` dramatically higher than Epsilon's corrected balance state.

---

## 4. Root Cause #2 — P2P Balance Gossip Creates Non-Deterministic State

### The gossip system (now disabled)

There is a dedicated gossipsub topic `/qnk/mainnet-genesis/balance-updates` where nodes broadcast balance delta messages as `P2PBalanceUpdate` structs (defined in `crates/q-types/src/balance_update.rs`):

```rust
pub struct P2PBalanceUpdate {
    pub wallet_address: String,
    pub amount: u128,        // DELTA to add (not absolute balance)
    pub new_balance: u128,   // informational only
    pub block_height: u64,
    pub update_type: BalanceUpdateType,  // MiningReward, FaucetClaim, etc.
    pub solution_hash: [u8; 32],
    pub signature: Vec<u8>,
    // ...
}
```

When received, the node applies the delta:

```rust
// main.rs:9162 — P2P balance gossip receive handler
let rocks_balance = app_state.storage_engine.get_balance(&wallet_hex).await.unwrap_or(0);
let new_balance = rocks_balance.saturating_add(update.amount);  // delta add
```

### Why this was disabled (v8.2.0)

The comment in `main.rs:8952` is explicit:

```rust
// v8.2.0: DETERMINISTIC BALANCE CONSENSUS — gossip balance updates DISABLED by default
// Gossip balance updates caused divergence between nodes (different balances per server).
// To opt-in (NOT recommended): set Q_ENABLE_BALANCE_GOSSIP=1
let balance_gossip_enabled = std::env::var("Q_ENABLE_BALANCE_GOSSIP")
    .map(|v| v == "1" || v.to_lowercase() == "true")
    .unwrap_or(false);  // defaults to FALSE
```

The divergence was caused by the fundamental impossibility of reliable gossip delivery in an asynchronous network:

- **Node A** is online when a mining reward gossip message is broadcast → applies +2 QUG delta
- **Node B** is offline or not yet connected → never receives the message
- **Result:** A and B have different balances for the same wallet, forever

There is no recovery mechanism. The state sync's bootstrap wallet sync (`state_sync_api.rs:785`) is a **one-time operation** that fires only on a node's first startup. After that, the comment reads:

```rust
if already_done {
    debug!("🔒 [STATE SYNC v8.5.4] Skipping {} wallet balances (bootstrap already done)");
}
```

### Historical evidence of production double-counting

Version `v10.3.7` added block-level deduplication for gossip with this comment:

```rust
// v10.3.7: BLOCK-LEVEL DEDUP for gossipsub balance updates
// Prevents double-counting when the same block's effects arrive via BOTH
// gossipsub balance-update (this path) AND block processing.
// Without this, 1 QUG transfer can become 2-3 QUG.
```

This confirms that production double-counting events occurred — the same mining reward was applied twice (once via block processing, once via gossip). This is additional evidence of the non-determinism.

---

## 5. Root Cause #3 — The Convergence Migration Produces Wrong Results (19× Error)

### Overview of the migration

`safe_batched_convergence_v103()` (`lib.rs:6631`) attempts to fix balance divergence by:
1. Purging all wallet balances from RocksDB
2. Replaying all 16.5M blocks to recompute balances from scratch
3. Scaling all balances so the total matches the expected emission from first principles
4. Persisting the corrected state

The migration was intended to produce the same result on every node. In practice, it produced 9,082 QUG for the master wallet — approximately 19× Epsilon's 479 QUG.

### Defect A — Legacy block path uses per-second reward, not per-block

For blocks without embedded transactions (the "legacy" path), the migration computes reward as:

```rust
// lib.rs:6861
let reward = (annual / 31_557_600u128).max(MIN_REWARD);
```

`annual / 31_557_600` is the **per-second** emission rate. This value is applied to **each mining solution in each block**. At a block rate of 2.91 blocks/second:

- Per-second emission (Era 0): 2,625,000 / 31,557,600 ≈ **0.0832 QUG/sec**
- Per-block, correctly: 0.0832 / 2.91 ≈ **0.0286 QUG/block**
- What the migration computes: **0.0832 QUG/block** (2.91× too high)

If there are many legacy blocks and multiple mining solutions per block (DagKnight can have many parallel solutions), the overcounting multiplies further.

### Defect B — Scaling amplifies the error rather than correcting it

Step 4 of the migration scales all balances:

```rust
// lib.rs:6896
if chain_total > 0 && chain_total != expected_total {
    for (_addr, amount) in balances.iter_mut() {
        // amount := amount × (expected_total / chain_total)
        let quot = *amount / chain_total;
        let rem = *amount % chain_total;
        *amount = quot.saturating_mul(expected_total)
                  + (rem as f64 * expected_total as f64 / chain_total as f64) as u128;
    }
}
```

`expected_total` is computed from the emission formula at **wall-clock time** (not block time), giving approximately the correct total supply.

The problem: if `chain_total` from the block scan is much **smaller** than `expected_total` (because legacy blocks have few or zero valid mining solutions, or because most transactions are coinbase-typed with zero-address recipients filtered out), the scaling factor becomes large, amplifying all balances.

**Hypothetical calculation:**
- `expected_total` at 65 days post-genesis: ~467,000 QUG × 10²⁴ (raw units)
- `chain_total` from block scan: ~20,000 QUG × 10²⁴ (if most blocks are legacy with poor reward accounting)
- Scale factor: 467,000 / 20,000 ≈ **23.35×**
- Master wallet pre-scale: ~397 QUG → post-scale: ~397 × 23.35 ≈ **9,271 QUG** (matches observed ~9,082 QUG)

The scaling step was designed to correct for historical inflation, but instead it **amplifies the proportional distribution** of whatever the chain scan found, which may bear no resemblance to actual earned balances.

### Defect C — The migration's expected_total uses wall-clock time, not chain time

```rust
// lib.rs:6651
let now_secs = std::time::SystemTime::now()...as_secs();
let elapsed_since_genesis = now_secs.saturating_sub(genesis_ts);
let expected_total = target_cumulative_at_time(elapsed_since_genesis);
```

If the node's clock is different from Epsilon's clock when each ran the migration, `expected_total` will differ. Even a 1-hour clock difference at 65 days of operation represents 1/(65×24) ≈ 0.06% of emission — small but real. More critically, if the genesis timestamp itself is ambiguous or the migration runs at different points in time, `expected_total` varies across nodes.

---

## 6. Root Cause #4 — 23 Write Paths With No Ordering Guarantee

The full audit found **23 distinct code paths** that write wallet QUG balances, with different callers, different atomicity guarantees, and different persistence semantics:

| Category | Count | Atomicity | Deterministic? |
|----------|-------|-----------|----------------|
| Block TX processing (add_balance_tx via RocksDB transaction) | 1 | Atomic TX | ✅ Yes |
| DEX atomic batch operations | 2 | WriteBatch | ✅ Yes |
| Migration rebuilds (purge + rebuild) | 5 | Sequential | ✅ Yes (if migration is correct) |
| 15-second periodic sync (HashMap ← RocksDB) | 1 | Per-wallet | ✅ Yes |
| P2P gossip balance updates (delta add) | 1 | put_sync per wallet | ❌ No |
| Bootstrap P2P sync (one-time overwrite) | 1 | Conditional | ❌ No |
| Authority sync (full snapshot overwrite) | 1 | db_put per wallet | ❌ No |
| DEX startup adjustment (apply_dex_qug_adjustments) | 1 | Batch | ⚠️ Depends on applied_net flag |
| Reorg handler (disabled v10.3.2) | 1 | put_sync | ❌ Race condition |
| In-memory HashMap insert (block pipeline) | 4 | In-memory only | ⚠️ Sync to RocksDB via 15s task |
| Admin API rebuild | 1 | Chain replay | ⚠️ Same issues as migration |

**Key observation:** Of the 23 paths, only the block TX processing path (Path #1) is both atomic and fully deterministic from chain data. The remaining 22 paths introduce non-determinism, timing dependencies, or off-chain state that cannot be reconstructed by a fresh syncing node.

### The 15-second periodic sync as a source of divergence

`main.rs:21004–21081` runs every 15 seconds and corrects mismatches between the in-memory HashMap and RocksDB:

```rust
// Every 15s: for each wallet in RocksDB where RocksDB value ≠ HashMap value:
//   → update HashMap to match RocksDB
// This fires as 🔴 [BALANCE WRITE] 15s_sync_hashmap_correction(): caller=PERIODIC_15S_SYNC
```

If RocksDB has been written by **any** of the other 22 paths (including incorrect values), this sync propagates that error into the HashMap. The task is designed to correct HashMap drift, but it also propagates RocksDB errors silently.

---

## 7. Why Fresh Nodes Cannot Match Epsilon — Summary

A fresh node syncing from genesis applies operations in this sequence:

1. **Reads coinbase TX amounts from blocks** — includes pre-fix inflated amounts from early blocks (Root Cause #1)
2. **Does NOT apply P2P balance gossip** — disabled by default, so even if Epsilon received historical gossip messages, the fresh node won't (Root Cause #2)
3. **Does NOT apply historical migration corrections** — Epsilon's v8.5.1, v8.8.1, v8.8.5 corrections exist only in Epsilon's RocksDB (Root Cause #3)
4. **May run convergence migration if chain_rebuild_enabled=1** — but this produces wrong results (Root Cause #3)

The fresh node has the same block data as Epsilon. It does not have the same migration history, gossip history, or RocksDB correction history.

**In a well-designed blockchain, balance state derived from blocks alone should be sufficient. In this system, it is not.**

---

## 8. The Balance Hash Divergence Chain of Events

The `balance_state_hash` (Blake3 hash of all wallet balances sorted by address) is computed and compared during periodic state sync:

```rust
// state_sync_api.rs:952 — fires every 5 minutes
if (our_height as i64 - peer_height as i64).unsigned_abs() < 100 {
    if our_hash != peer_hash {
        error!("🚨 [DIVERGENCE CHECK] CRITICAL: Balance hash MISMATCH with peer!");
        error!("   Run convergence migration to fix: delete RocksDB flag and restart");
        // ← NO AUTO-FIX. Manual intervention only.
    }
}
```

The system detects divergence but does not heal it. The suggested fix (running the convergence migration) itself produces incorrect results (as demonstrated by the 9,082 vs 479 QUG discrepancy).

The network-wide picture (all nodes diverge from each other, not just from Epsilon) is explained by:
- Each node applied its migrations at different calendar times → different `expected_total` from wall-clock
- Each node received different subsets of historical P2P gossip
- Each node ran different versions of the migration code with different bugs

---

## 9. Proposed Solutions

### 9.1 Short-term: Balance Checkpoint (Immediate, No Architecture Change)

Embed Epsilon's current balance state as a hardcoded constant in the binary:

```rust
// In q-types/src/lib.rs or q-storage/src/lib.rs
pub const BALANCE_CHECKPOINT_HEIGHT: u64 = 16_540_000;
pub const BALANCE_CHECKPOINT_HASH: &str = "abc123...";  // Blake3 of the snapshot
pub const BALANCE_CHECKPOINT: &[(&str, u128)] = &[
    ("efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723", 479_457_238_470_962_989_623_408_338),
    // ... all 1332 wallets from Epsilon's snapshot at height 16,540,000
    // generated via: curl http://89.149.241.126:8080/api/v1/sync/full-state
];
```

At startup, if the node's RocksDB has no wallet balances AND the chain height is above `BALANCE_CHECKPOINT_HEIGHT`:
1. Apply the checkpoint balances
2. Replay blocks from `BALANCE_CHECKPOINT_HEIGHT + 1` onwards only
3. Never touch pre-checkpoint history

**Properties:**
- Every node running this binary gets the same starting balance state (Epsilon's verified state)
- No dependency on gossip, migrations, or wall-clock timing
- Blocks above the checkpoint are still replayed deterministically
- The checkpoint is verified by `BALANCE_CHECKPOINT_HASH` on startup

**Risks:**
- If Epsilon's balance state at checkpoint height contains any error, that error is baked in
- Must regenerate checkpoint for future binary releases as chain grows

### 9.2 Medium-term: Remove Off-Chain Balance Mutations

All balance changes that are not in blocks must be eliminated or encoded on-chain:

1. **P2P balance gossip (`/balance-updates` topic)** — Already disabled (v8.2.0). Remove the code entirely to prevent accidental re-enabling.

2. **Migration corrections** — The v8.5.1, v8.8.5, v1.0.3 migrations corrected historical inflation by off-chain scaling. These corrections need to be encoded as on-chain transactions at a specific block height (a "correction block") so every node applying the chain applies the same corrections.

3. **DEX QUG adjustments** — `apply_dex_qug_adjustments()` runs at every startup and adjusts balances based on tracked DEX debits/credits. This must become part of block transaction processing.

4. **Authority sync** — Can remain as an emergency tool for new nodes, but must validate the imported snapshot against the checkpoint hash.

### 9.3 Long-term: Make Balance State Fully Deterministic

The block chain must become the single source of truth for all balance state:

1. Every coinbase transaction must use the adaptive emission formula (not fixed-rate). This is partially done for new blocks (v7.1.2+) but old blocks are permanently inflated.

2. Define a **chain fork height** at which all historical balances are replaced by a canonical snapshot (encoded as a special transaction in the fork block). All nodes that apply this block get the same corrected state.

3. Remove the `wallet_balances` in-memory HashMap as a separate data structure from RocksDB. The HashMap is a cache of RocksDB. Currently, the 15-second sync that reconciles them masks errors rather than preventing them.

4. Add strict ordering guarantees: DEX operations must produce block transactions, not off-chain balance adjustments.

---

## 10. Immediate Questions for DeepSeek / ChatGPT Review

The following questions require external consultation because they involve architectural trade-offs beyond the scope of code-level analysis:

**Q1: Balance Checkpoint Correctness**  
If we snapshot Epsilon's balance at height H and embed it as a hardcoded constant, we are trusting that Epsilon's balance state at height H is correct. Given that Epsilon itself ran buggy migrations (v8.5.1 with inflated coinbase replay), is Epsilon's current balance state actually correct? Or is it the least-wrong state we have?

**Q2: Correction Block Design**  
What is the safest way to encode a mass balance correction as an on-chain transaction? Options include:
- A special `BalanceCorrection` transaction type validated by all nodes at a specific height
- A hard fork that replaces the genesis balance distribution
- A checkpoint mechanism where nodes agree on a balance state hash via consensus

**Q3: The Convergence Migration Scaling Step**  
The migration scales all balances by `expected_total / chain_total`. If `chain_total` accurately reflects proportional ownership (even if the absolute values are wrong), the scaling correctly preserves relative shares. Is this approach fundamentally sound, and what breaks the proportional accuracy in this codebase?

**Q4: DEX State and Balance Integrity**  
The `apply_dex_qug_adjustments()` function adjusts balances at startup based on tracked DEX debits/credits. If this runs on a node that already had the migration corrections applied (which included DEX TX processing), would it double-apply DEX adjustments? What is the expected idempotency guarantee of this function?

**Q5: Safe Deployment of Balance Checkpoint**  
If we deploy a binary with a hardcoded balance checkpoint, nodes that are already fully synced will have their old (possibly incorrect) balances AND the checkpoint applied. How do we ensure the transition is safe and doesn't result in double-counting for nodes upgrading from an older binary?

---

## 11. Key Files Reference

| File | Relevance |
|------|-----------|
| `crates/q-storage/src/lib.rs:6631` | `safe_batched_convergence_v103()` — convergence migration |
| `crates/q-storage/src/lib.rs:5909` | `purge_and_rebuild_balances()` — v8.5.1 migration |
| `crates/q-storage/src/lib.rs:4123` | `save_wallet_balance()` — absolute overwrite with logging |
| `crates/q-storage/src/lib.rs:8610` | `add_balance()` — BalanceStorage trait delta add |
| `crates/q-storage/src/balance_consensus.rs:1278` | `add_balance_tx()` — atomic block TX balance write |
| `crates/q-api-server/src/main.rs:3372` | Startup migration sequence (chain_rebuild_enabled gate) |
| `crates/q-api-server/src/main.rs:8952` | P2P balance gossip receive handler (disabled default) |
| `crates/q-api-server/src/main.rs:21004` | 15-second periodic HashMap←RocksDB sync |
| `crates/q-api-server/src/state_sync_api.rs:952` | Balance hash mismatch detection (no auto-heal) |
| `crates/q-api-server/src/state_sync_api.rs:1103` | `do_authoritative_balance_sync()` — Q_BALANCE_AUTHORITY_PEER |
| `crates/q-api-server/src/block_producer.rs:1266` | `create_coinbase_transactions()` — current emission calc |
| `crates/q-types/src/balance_update.rs:50` | `P2PBalanceUpdate` message format |
| `crates/q-storage/src/emission_controller.rs` | `target_cumulative_at_time()`, `annual_emission()` |

---

## 12. Observed Data Points

| Measurement | Value | Source |
|------------|-------|--------|
| Epsilon master wallet balance (UI, ~2026-04-27) | 479 QUG | Live observation |
| Container (post-convergence-migration) master wallet | 9,082 QUG | Log: `old=9082655…/10²⁴` |
| Discrepancy ratio | ~19× | |
| Epsilon wallet count (full-state snapshot) | 1,332 wallets | `curl /api/v1/sync/full-state` |
| Container wallet count (post-migration) | 1,980 wallets | BATCH_OVERWRITE log count |
| Chain height at time of analysis | ~16,537,163 blocks | Epsilon snapshot |
| Balance hash: all nodes match Epsilon? | NO — all diverge | `🚨 DIVERGENCE CHECK` log |
| Q_ENABLE_BALANCE_GOSSIP default | false (disabled v8.2.0) | `main.rs:8952` |
| Number of startup migrations | 9 sequential | `main.rs:3370–3855` |
| Balance write paths identified | 23 distinct paths | This audit |

---

*This document was generated by automated code audit (4 parallel subagents) and Claude Sonnet 4.6 synthesis. All line numbers reference the `feature/safe-batched-sync-v1.0.2` branch as of 2026-04-28.*
