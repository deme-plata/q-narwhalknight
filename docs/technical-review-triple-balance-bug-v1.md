# Technical Review: Triple Balance Bug — 1 Coin Becomes 3
## Three Independent Code Paths Process the Same Transfer
### Date: 2026-04-18 | Status: Identified, not yet reproducible

---

## 1. The Bug

Transferring 1 QUG to a new address results in 3 QUG appearing over time.

## 2. Root Cause: Three Independent Balance Write Paths

A single block containing a transfer can be processed by three separate code paths, each adding the transfer amount independently:

### Path 1: Block Processing (deduped)

```
File: crates/q-api-server/src/main.rs:6083 (turbo sync batch)
      crates/q-api-server/src/main.rs:10451 (gossipsub block received)

Flow: Block received → process_block_mining_rewards_tx() or
      process_block_coinbase_only_tx()
      → subtract sender balance
      → add receiver balance
      → set persistent dedup flag: "processed_balance_block:{hash}"

Dedup: YES (v10.3.2 persistent RocksDB flag + LRU cache)
```

### Path 2: P2P Gossipsub Balance-Update (NO dedup)

```
File: crates/q-api-server/src/main.rs:8900-8926

Flow: Peer broadcasts balance-update gossipsub message
      → security checks (signature, height range, rate limit)
      → reads current balance from RocksDB
      → saturating_add(update.amount)    ← ADDS on top of whatever is already there
      → saves new balance to RocksDB

Dedup: PARTIAL (LRU dedup_key check at line 8806, but NO persistent/block-level dedup)
```

**This is the primary suspect.** When a node produces a block with a transfer:
1. The node processes the block locally (Path 1) → receiver gets +1
2. The node broadcasts a balance-update gossipsub message to all peers
3. Other nodes receive the gossipsub message → `saturating_add(amount)` → receiver gets +1 AGAIN
4. If the block is also synced via turbo sync → Path 1 dedup catches it (no +1)
5. Net result on other nodes: +2 (once from gossipsub, once from block processing if dedup fails)

### Path 3: Authority Sync / State Sync (ABSOLUTE_OVERWRITE)

```
File: crates/q-api-server/src/state_sync_api.rs:164 (authority)
      crates/q-api-server/src/state_sync_api.rs:198 (P2P periodic)

Flow: Every 5 minutes, fetch ALL wallet balances from peers
      → overwrite local values with peer's values
      → "ABSOLUTE_OVERWRITE" in logs

Dedup: N/A — it's an overwrite, not an addition
```

If a peer's balance is already inflated (from Path 2 double-counting), the state sync imports the inflated value. This can turn a 2x into a 3x if the timing is right.

## 3. The Triple-Count Scenario

```
Timeline for a 1 QUG transfer from Alice to Bob:

T+0s:  Block produced on Node A containing transfer (Alice→Bob, 1 QUG)
       Node A: Path 1 processes block → Bob = 0 + 1 = 1 QUG ✅
       
T+0.1s: Node A broadcasts gossipsub balance-update: "Bob +1 QUG at height H"
       
T+0.5s: Node B receives gossipsub balance-update
       Node B: Path 2: Bob = 0 + 1 = 1 QUG (correct so far)
       
T+1.0s: Node B receives the block via gossipsub or turbo sync  
       Node B: Path 1: process_block → subtract Alice, add Bob
       IF dedup catches it: Bob stays at 1 QUG ✅
       IF dedup MISSES (different function, LRU empty): Bob = 1 + 1 = 2 QUG ❌
       
T+300s: State sync runs, imports balances from Node A
       Node A has Bob = 1 QUG (correct)
       Node B OVERWRITES its 2 QUG with Node A's 1 QUG → Bob = 1 QUG ✅
       OR: Node B is chosen as the "higher height" peer → Node A imports 2 QUG ❌
       
T+600s: Next state sync cycle
       If BOTH nodes now have 2 QUG, the inflated value propagates everywhere
       With 3+ nodes exchanging inflated values, it compounds to 3x
```

## 4. Why It's Hard to Reproduce

The bug requires specific timing:
- Gossipsub balance-update must arrive BEFORE the block is processed (otherwise block dedup flag is already set)
- The LRU dedup for gossipsub must miss (LRU eviction or different dedup key format)
- State sync must import from an inflated peer (timing-dependent)

In normal operation with stable nodes, the timing may rarely align. After restarts (empty LRU), the probability increases.

## 5. The Fix

### Fix A: Add block-hash dedup to the gossipsub balance-update path (RECOMMENDED)

```rust
// crates/q-api-server/src/main.rs:8900 — BEFORE the saturating_add

// v10.3.7: Block-level dedup for gossipsub balance updates
// Prevents double-counting when the same block's effects arrive via
// both gossipsub balance-update AND block processing
let dedup_key = format!("processed_balance_block_gossip:{}:{}", 
    update.block_height, &update.wallet_address[..16]);
if let Ok(Some(_)) = app_state_gossip.storage_engine
    .hot_db.get(CF_MANIFEST, dedup_key.as_bytes()).await {
    debug!("⏭️ [P2P BALANCE] Already applied for this block+wallet, skipping");
    continue;
}

// ... existing saturating_add code ...

// After successful apply:
let _ = app_state_gossip.storage_engine
    .hot_db.put(CF_MANIFEST, dedup_key.as_bytes(), b"1").await;
```

### Fix B: Make gossipsub balance-updates use ABSOLUTE values, not deltas

Instead of `saturating_add(amount)`, broadcast the ABSOLUTE balance and only accept if it's higher:

```rust
// BEFORE (additive — causes double-counting):
let new_balance = rocks_balance.saturating_add(update.amount);

// AFTER (absolute — idempotent):
let new_balance = update.absolute_balance;
if new_balance <= rocks_balance {
    continue; // Peer's balance is not higher — skip
}
```

This requires changing the gossipsub message format (protocol change). More invasive but eliminates the entire class of bugs.

### Fix C: Disable gossipsub balance-updates entirely

```bash
# In service file:
Environment="Q_ENABLE_BALANCE_GOSSIP=0"
```

Line 8755 already checks this:
```rust
debug!("Ignoring gossipsub balance update (deterministic mode — set Q_ENABLE_BALANCE_GOSSIP=1 to opt-in)");
```

This is the nuclear option — removes the real-time balance propagation. Nodes would only get balance updates from block processing + state sync (every 5 min). Less real-time but eliminates the double-counting path entirely.

## 6. Recommendation

**Fix A** (block-hash dedup) for immediate deployment — same pattern as the existing dedup, minimal code change, zero risk.

**Fix B** (absolute values) as a medium-term protocol improvement — requires message format change and peer coordination.

**Fix C** (disable gossipsub) as an emergency fallback if the bug becomes exploitable.

## 7. Why It's Not Urgent Right Now

1. The user couldn't reproduce it
2. The persistent dedup (our v10.3.6 fix) catches most double-counting scenarios
3. The authority sync from Beta acts as a correction mechanism
4. The state sync (every 5 min) tends to converge balances across nodes

But it IS a real vulnerability — if someone discovers a way to reliably trigger it, they could inflate their balance by 2-3x on every transfer. This should be fixed before the exchange listing.

---

*Generated 2026-04-18 — Quillon Foundation*
*Status: Bug identified, not yet reproducible, fix designed*
