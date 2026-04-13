# Technical Review v5: Balance Corruption — Complete Audit & Debugging

**Date:** 2026-04-13  
**Severity:** CRITICAL  
**Network:** Q-NarwhalKnight mainnet-genesis ($1B market cap, ~40 days old, ~4.6M blocks)  
**Core Finding:** THREE independent mechanisms corrupt balances in different directions  
**Prepared for:** DeepSeek + ChatGPT peer review

---

## 0. Corrections From Previous Reviews

| Claim in v1-v4 | Reality |
|----------------|---------|
| "Blocks 0-13M are pruned" | Chain is 40 days old. Pruning kept 30 days. Only ~10 days pruned (~25% of history) |
| "14.9M blocks exist" | Height ≠ block count in DAG-Knight. ~4.6M actual blocks at height 14.9M |
| "Beta has the correct balance" | **WRONG** — Beta copies Epsilon's corrupted balance via `Q_BALANCE_AUTHORITY_PEER` |
| "Restart caused the corruption" | User says balance went to zero **immediately after swap**, not after restart |
| "Persistent dedup is the core fix" | It's ONE of many needed fixes — 16 write paths exist, only one uses the dedup |

---

## 1. The Three Bugs Creating Three Different Kinds of Wrong

### Bug A: Epsilon — Balance DEFLATED after restart

**Mechanism:** After restart, `balance_consensus` rebuilds the balance by processing blocks from turbo sync. With ~25% of chain history pruned (first 10 days), and the LRU dedup cache empty, the rebuild produces a partial total. But even 75% of history should produce much more than [10-100] QUG.

**Unanswered question:** Why does the rebuild produce ~17 QUG when 75% of blocks are available? This suggests either:
- The rebuild only processes a SMALL window of blocks (not all available ones)
- Or another writer overwrites the rebuilt value with something lower
- Or turbo sync doesn't deliver all available blocks during the startup window

**Node behavior:** Balance starts near zero on boot, slowly grows from new mining rewards.

### Bug B: Beta — Balance OVERWRITTEN by Epsilon's wrong value

**Mechanism:** Beta's service file contains:
```
Environment="Q_BALANCE_AUTHORITY_PEER=http://89.149.241.126:8080"
```

On every startup (10 seconds after boot), Beta fetches ALL wallet balances from Epsilon and **overwrites its own RocksDB** (`state_sync_api.rs:1119`):

```rust
// Line 1108: "Overwrite ALL wallet balances in RocksDB"
for (address_hex, balance_str) in &snapshot.wallet_balances {
    let key = format!("wallet_balance_{}", address_hex);
    app_state.storage_engine.db_put("manifest", key.as_bytes(), &balance.to_le_bytes()).await;
}
```

This is a **raw `db_put` with NO safety checks** — no dedup, no DEX counter reconciliation, no validation. Whatever Epsilon has, Beta copies blindly.

**Result:** Epsilon corrupted → Beta copies corruption → Beta also corrupted.

**No other node has `Q_BALANCE_AUTHORITY_PEER` set.** This is Beta-specific.

### Bug C: Delta/Gamma — Balance INFLATED by P2P sync

**Mechanism:** The P2P bootstrap sync (`state_sync_api.rs:822-823`) only writes if `peer_amount > current_local`:

```rust
let current = balances.get(&addr_bytes).copied().unwrap_or(0);
if amount > current {
    app_state.storage_engine.save_wallet_balance(&addr_bytes, amount).await;
}
```

This can only INCREASE balances, never decrease. DEX swap deductions happen locally on the swap node (Epsilon/Beta) but are never propagated to peers. So Delta/Gamma accumulate the full mining total with zero DEX deductions.

**Result:** Delta/Gamma show [1K-10K] QUG — the user's TOTAL lifetime mining rewards without ANY DEX swap deductions applied. The user says they "always swap QUG to QUGUSD" so the real balance should be much lower.

---

## 2. Full Audit: 16 Write Paths to wallet_balance_

### Absolute Writers (SET a value — dangerous if value is stale/wrong)

| # | Function | File:Line | Called When | Source of Value | Risk |
|---|----------|-----------|-------------|-----------------|------|
| 1 | `save_wallet_balance()` | `lib.rs:3956` | Various callers | Parameter | **HIGH** — caller determines value |
| 2 | `save_wallet_balances()` | `lib.rs:4106` | Migrations | Parameter HashMap | **HIGH** — batch overwrite |
| 3 | `set_balance()` | `lib.rs:8452` | Reorg, corrections | Parameter | **HIGH** — absolute overwrite |
| 4 | Authority sync `db_put` | `state_sync_api.rs:1119` | Startup (Beta only) | **Epsilon's RocksDB** | **CRITICAL** — blind copy |
| 5 | P2P bootstrap `save_wallet_balance` | `state_sync_api.rs:825` | Startup (one-time) | **Peer's in-memory** | **HIGH** — inflate only |
| 6 | Block producer persist | `main.rs:18566` | Block with transfers | **In-memory HashMap** | **HIGH** — stale if cache lags |
| 7 | `purge_and_rebuild_balances()` | `lib.rs:5673` | Migration (once) | Chain replay | **CRITICAL** — deletes all first |
| 8 | `reconcile_balances_with_dex_swaps()` | `lib.rs:5827` | Migration (once) | Emission + miner shares | **CRITICAL** — redistributes total |
| 9 | `rebuild_balances_from_chain()` | `lib.rs:5550` | Admin endpoint | Chain replay | **HIGH** — partial if pruned |

### Delta Writers (ADD or SUBTRACT — safe if not double-applied)

| # | Function | File:Line | Called When | Dedup? | Risk |
|---|----------|-----------|-------------|--------|------|
| 10 | `add_balance()` | `lib.rs:8356` | Coinbase processing | LRU (volatile!) | **MEDIUM** — double-count after restart |
| 11 | `subtract_balance()` | `lib.rs:8390` | DEX, transfers | None | **LOW** — fails if insufficient |
| 12 | `add_balance_tx()` | `balance_consensus.rs:1196` | Block processing (TX) | LRU (volatile!) | **MEDIUM** — double-count after restart |
| 13 | `subtract_balance_tx()` | `balance_consensus.rs:1255` | Transfer processing (TX) | LRU (volatile!) | **LOW** |
| 14 | `atomic_subtract_and_record_dex_debit()` | `lib.rs:8521` | DEX swap (v10.3.1) | WriteBatch | **LOW** |
| 15 | `atomic_add_and_record_dex_credit()` | `lib.rs:8603` | DEX swap (v10.3.1) | WriteBatch | **LOW** |
| 16 | `apply_dex_qug_adjustments()` | `lib.rs:8876` | Manual/startup | Delta tracking | **MEDIUM** — if base is wrong |

---

## 3. Why No Node Has the Correct Balance

| Node | Balance | Why It's Wrong |
|------|---------|---------------|
| **Epsilon** [10-100] | Rebuilt from partial chain after restart, only recent blocks counted |
| **Beta** [10-100] | Blind copy of Epsilon's wrong balance via `Q_BALANCE_AUTHORITY_PEER` |
| **Gamma** [1K-10K] | Inflated by P2P sync (only increases, never decreases for DEX) |
| **Delta** [1K-10K] | Same as Gamma — never received DEX deduction from swap node |

**The correct balance** = (total mining rewards from all blocks) - (total DEX swaps).

Neither side of this equation is reliably available:
- Total mining: pruned blocks + incomplete replay = unknown exact total
- Total DEX swaps: `dex_qug_debited:{wallet}` counter exists on Epsilon but wasn't propagated

**Best estimate:** User had 386 QUG at swap time (verified from swap log). After swapping 193, correct balance = 193 + mining since swap (~18 QUG in ~5 hours at current rate) ≈ **~211 QUG**.

---

## 4. The User's Timeline (Corrected)

The user reports: *"after swapping the 50% of my total qug balance to qugusd the whole qug amount was gone instantly after without restart."*

**Possible explanation WITH the new audit data:**

1. In-memory showed ~193 QUG (stale — RocksDB had 386)
2. User swapped → `subtract_balance` wrote 193 to RocksDB (386-193=193)
3. In-memory was set to 193 (matching RocksDB)
4. **WITHIN SECONDS:** The 15-second balance sync or P2P state sync ran
5. The sync read RocksDB on a PEER (or the sync's own stale source) and found a LOWER value
6. The sync OVERWROTE 193 with the lower value
7. User refreshed → saw near-zero

**OR:**

1. The swap happened on Epsilon which was already corrupted (balance was ~17 in RocksDB)
2. But the swap log shows `RocksDB: 386 → 193` — so RocksDB had 386 at swap time
3. **AFTER the swap:** `balance_consensus` processed a new block
4. `balance_consensus` did `add_balance()`: read RocksDB (193), add coinbase, write 193.0002
5. That's correct... unless another path simultaneously wrote a lower value

**We need the debugging logs to know for certain.** The v5 debugging additions will show EVERY write to this wallet's balance with before/after values and caller identity.

---

## 5. Debugging Additions (v10.3.1-debug)

Every write path now logs at WARN level (loud, visible in journalctl):

```
🔴 [BALANCE WRITE] save_wallet_balance(): wallet=efca1e8c... old=193036328290000000000000000000 new=17929594201021128441198016 delta=-175106734088978871558802000000 caller=authority_sync height=14930000
```

For writes where `new < old` (potential corruption), the log is at ERROR level:

```
🔴🔴 [BALANCE DROP] save_wallet_balance(): wallet=efca1e8c... DECREASED from 193.04 to 17.93 QUG (-175.11 QUG) caller=authority_sync
```

### What To Watch For

After deploying the debug binary to Delta Docker:

1. **`🔴🔴 [BALANCE DROP]`** — Any balance decrease not caused by a DEX swap or transfer is a bug
2. **`caller=authority_sync`** — If authority sync overwrites, we found the problem
3. **`caller=p2p_bootstrap`** — If P2P sync writes, we found the problem
4. **`caller=save_wallet_balance_block_producer`** — If block producer writes stale values
5. **`caller=balance_consensus_add_balance_tx`** — Should only INCREASE, never decrease
6. **`caller=set_balance_reorg`** — Fork recovery overwriting

---

## 6. Immediate Actions (Safe, No Balance Modification)

### 6.1 Remove Q_BALANCE_AUTHORITY_PEER from Beta

```bash
# On Beta: remove the line that copies Epsilon's wrong balance
sed -i '/Q_BALANCE_AUTHORITY_PEER/d' /etc/systemd/system/q-api-server.service
systemctl daemon-reload
# DO NOT restart Beta yet — removing the env var just prevents future damage
```

**This prevents Beta from copying Epsilon's wrong balance on next restart.** Does not modify any balance, does not require restart.

### 6.2 Deploy Debug Binary to Delta Docker (NOT production)

1. Build with debugging additions
2. Deploy to Delta (test node)
3. Run the miner
4. Watch logs for all `🔴` entries
5. If no corruption after 1 hour: attempt a swap and watch what happens

### 6.3 DO NOT Restart Gamma or Delta

Gamma and Delta have inflated but stable balances. Restarting them would trigger the same rebuild bug as Epsilon. Keep them running until the fix is deployed.

---

## 7. The Fix Plan (After Debugging Confirms Root Cause)

### Phase 1: Stop the bleeding (no balance modification)

1. **Remove `Q_BALANCE_AUTHORITY_PEER`** from all service files — never blind-copy balances
2. **DEX safety gate** — block swaps until node is synced
3. **Persistent processed-block tracking** — prevent balance_consensus from reprocessing
4. **Fix P2P bootstrap sync** — use MAX(local, peer) only for wallets that DON'T have DEX history

### Phase 2: Fix balance propagation

5. **Propagate DEX deductions via P2P** — when a swap happens, broadcast the debit to all nodes
6. **Or: make swaps on-chain transactions** — the long-term fix from DeepSeek's review

### Phase 3: Manual balance recovery (user-triggered)

7. **Admin forensic report** — read-only, shows all data sources and their values
8. **Admin set-balance** — if user approves, with full audit trail
9. **No automatic recovery** — ever

---

## 8. Open Questions for Peer Review

1. **Why does balance_consensus produce [10-100] when 75% of blocks are available?** Even with 25% pruned, the balance should be ~75% of correct. Something prevents full replay.

2. **What is the exact mechanism of the "immediate" post-swap corruption?** The user says it happened instantly after the swap, not after restart. Is a background task overwriting within seconds?

3. **Should the P2P bootstrap sync be disabled entirely?** It can only inflate balances (never deflate). This creates permanent divergence between nodes.

4. **How should DEX deductions propagate across nodes?** Currently they're local-only. Options: (a) gossipsub broadcast of swap events, (b) on-chain swap transactions, (c) periodic DEX counter sync.

5. **Can we reconstruct the correct balance from the DEX debit counter + available blocks?** If the `dex_qug_debited:{wallet}` counter on Epsilon is correct (recording all historical swaps), and we can replay the available 75% of blocks for coinbase rewards, we could estimate the correct balance.

---

## 9. Summary

| What | Finding |
|------|---------|
| **Root cause** | Three independent bugs: deflation (Epsilon rebuild), blind copy (Beta authority sync), inflation (P2P one-way sync) |
| **Affected nodes** | ALL — every node has a different wrong balance |
| **Correct balance** | Unknown exact value. Best estimate: ~211 QUG based on swap log |
| **Immediate fix** | Remove `Q_BALANCE_AUTHORITY_PEER`, deploy debug binary to Delta |
| **Full fix** | Persistent dedup + fix all 16 write paths + proper DEX propagation |
| **User recovery** | Manual, admin-triggered, after debugging confirms the correct value |
| **Debugging** | Every balance write now logged with before/after/caller/height |
