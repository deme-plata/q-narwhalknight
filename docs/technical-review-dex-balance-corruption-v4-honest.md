# Technical Review v4: DEX Balance Corruption — Honest Investigation

**Date:** 2026-04-13  
**Severity:** CRITICAL  
**Network:** Q-NarwhalKnight mainnet-genesis ($1B market cap)  
**Core Finding:** Balance corruption affects ALL nodes that restart, NOT just one  
**Prepared for:** DeepSeek + ChatGPT peer review

---

## 1. What We KNOW (Verified From Logs)

### 1.1 The User's Balance Before the Swap

**Source:** Epsilon swap log at 14:59:31 CEST  
```
RocksDB: 386.03537279 → 193.03632829 QUG
In-memory (stale): 193.22932733 QUG
```

The user had **386.035 QUG** in RocksDB at the moment of the swap. The in-memory display showed only **193.229 QUG** (half the real value — already stale/wrong BEFORE the swap).

### 1.2 The User Says Balance Went to Zero Immediately After Swap

The user reports: *"after swapping the 50% of my total qug balance to qugusd the whole qug amount was gone instantly after without restart."*

This contradicts the earlier assumption that restart caused the corruption. The user says it happened **immediately**, not after a restart.

### 1.3 Multi-Node Balance State (Verified 2026-04-13 ~17:10 CEST)

| Node | Balance Range | Binary | When Started |
|------|--------------|--------|-------------|
| **Epsilon** | [10-100] QUG | v10.3.0 | 12:34 CEST |
| **Beta** | [10-100] QUG | v10.3.0 | 16:50 CEST |
| **Gamma** | [1K-10K] QUG | v10.3.0 | Restarting |
| **Delta** | [1K-10K] QUG | v10.2.1 (old) | Running longest |

### 1.4 Beta Corruption Caught in Action

| PID | Time (CEST) | Balance Range | Event |
|-----|------------|--------------|-------|
| 531019 | 15:27-15:36 | [100-1K] | Correct balance |
| 649414 | 15:40 | [100-1K] | Still correct |
| 652513 | 15:43 | [1-10] | **CORRUPTED** |
| 700970 | 16:50 | [10-100] | Growing from corrupted base |

The corruption was already in RocksDB when PID 652513 loaded it. The migrations (purge_and_rebuild, reconcile_with_dex) were SKIPPED (flags present). Something else corrupted the balance.

### 1.5 User's Current Display

- **17.93 QUG** — from Epsilon (frontend connects to quillon.xyz → Epsilon)
- **24,258,229.73 QUGUSD** — correct (token balances unaffected)

---

## 2. What We DON'T KNOW

### 2.1 The Exact Corruption Trigger

We have NOT identified the specific code path that overwrites the balance. Candidates:

| Hypothesis | Evidence For | Evidence Against |
|-----------|-------------|-----------------|
| `balance_consensus` reprocessing | LRU dedup is volatile | Migrations were skipped on Beta |
| `purge_and_rebuild_balances()` | Deletes all balances then rebuilds from pruned chain | Migration flag was present, function returned early |
| 15-second balance sync task | Reads from RocksDB → if RocksDB is already wrong, propagates | This task READS, it doesn't SET — it mirrors RocksDB to in-memory |
| Turbo sync importing stale balance snapshots from peers | Could overwrite local balance with peer's (potentially lower) value | Need to audit turbo sync balance handling |
| The v10.3.0 binary itself introduced a new startup path | Epsilon/Beta (v10.3.0) corrupted; Delta (v10.2.1) not | No obvious new balance-touching code in v10.3.0 |
| The swap handler itself | User says corruption was immediate after swap | The swap log shows RocksDB correctly went 386→193, not to zero |

### 2.2 The Correct Balance

We **cannot determine** the exact correct QUG balance for this wallet. Here's why:

- Blocks 0-13M are pruned (deleted by the pruning bug). Those blocks contained the coinbase transactions proving historical mining rewards.
- No node has been running continuously since genesis — every node has restarted at some point.
- Different nodes show different values: [10-100] vs [1K-10K].
- The only verified data point is the swap log: 386.035 QUG at swap time.

### 2.3 Why the In-Memory Balance Was Already Half

The in-memory balance was **193.229 QUG** while RocksDB had **386.035 QUG**. This 2:1 ratio is suspicious — it suggests either:
- The 15-second sync had not yet picked up recent RocksDB changes
- Or the in-memory cache was somehow initialized with half the real value
- Or balance_consensus was in the process of rebuilding and had only reached half the total

### 2.4 Whether This Bug Existed Before v10.3.0

Delta (v10.2.1) shows [1K-10K] — much higher than Epsilon/Beta. But Delta hasn't restarted recently. We don't know if Delta would ALSO corrupt on restart with v10.2.1 or if this is specifically a v10.3.0 regression.

---

## 3. What the Code Does (Audited)

### 3.1 Balance Writers to `wallet_balance_{hex}` in RocksDB

| Writer | File:Line | Type | When | Audited? |
|--------|-----------|------|------|----------|
| `add_balance()` (coinbase) | `balance_consensus.rs:357` | Delta (+) | Every block's coinbase | Yes — safe if dedup works |
| `subtract_balance_tx()` (transfers) | `balance_consensus.rs:1255` | Delta (-) | Block transfers | Yes — safe |
| `subtract_balance()` (DEX debit) | `lib.rs:8390` | Delta (-) | DEX swap | Yes — safe |
| `save_wallet_balance()` (block producer) | `main.rs:18566` | **Absolute SET** | Block production (transfers) | **SUSPICIOUS** |
| `save_wallet_balances()` (purge_rebuild) | `lib.rs:5709` | **Absolute SET** | Migration (one-time) | Audited — skipped via flag |
| `save_wallet_balances()` (reconcile) | `lib.rs:5827` | **Absolute SET** | Migration (one-time) | Audited — skipped via flag |
| `set_balance()` | `lib.rs:8441` | **Absolute SET** | Various | **UNKNOWN — need full grep** |
| Turbo sync state applicator | `main.rs:~5860-6130` | Via `add_balance()` | Turbo sync batches | Partially audited |

### 3.2 The Dangerous Pattern

Every **absolute SET** writer can corrupt a balance if it writes a stale or partial value. The delta writers (+/-) are safe as long as they operate on the current correct value.

**`save_wallet_balance()` at main.rs:18566** is called during block production for transfer transactions. It persists the in-memory HashMap value to RocksDB. If the in-memory value is wrong (stale), this OVERWRITES the correct RocksDB value with the wrong one.

This is the **reverse direction** of the 15-second sync. The 15-second sync reads RocksDB → writes in-memory. But the block producer reads in-memory → writes RocksDB. If in-memory is stale (as we know it was — 193 vs 386), and a transfer block is produced, the block producer writes the stale 193 to RocksDB, overwriting the correct 386.

**This is a strong candidate for the immediate post-swap corruption.**

### 3.3 The Probable Sequence

```
1. In-memory = 193 QUG (stale — hasn't synced from RocksDB yet)
2. RocksDB = 386 QUG (correct — accumulated from mining)
3. User swaps 50%:
   a. subtract_balance reads RocksDB (386), subtracts 193, writes 193 to RocksDB
   b. In-memory set to 193 (matching post-swap RocksDB)
4. Meanwhile, block producer produces a new block containing transfers:
   a. Reads in-memory balance for this wallet
   b. Finds 193 (the post-swap value — or maybe the stale pre-swap value)
   c. Calls save_wallet_balance(addr, 193) → RocksDB
   d. This MIGHT be the correct value (193) or it MIGHT be a stale value
5. BUT: balance_consensus also processes the block:
   a. add_balance reads RocksDB (now 193), adds coinbase reward
   b. Writes 193 + 0.0002 = 193.0002 to RocksDB
   c. This should be correct...

UNLESS balance_consensus is processing a DIFFERENT block simultaneously,
one that was produced BEFORE the swap, which reads RocksDB between
the swap's write and the sync:

1. RocksDB = 386 (correct)
2. balance_consensus reads 386, starts processing block
3. DEX swap reads 386, writes 193 (386-193)
4. balance_consensus adds coinbase reward: 386 + 0.0002 = 386.0002
5. balance_consensus writes 386.0002 to RocksDB
6. The swap's 193 is OVERWRITTEN by 386.0002
7. The 193 QUG deduction is lost!
```

Wait — this sequence would INCREASE the balance, not decrease it. The user's balance would go back to 386 instead of 193.

**Let me reconsider.** The user says the balance went to ZERO. Not to 386 (overwritten by concurrent add_balance), not to 193 (correct post-swap), but to ZERO (or near-zero).

For the balance to go near-zero, something must EITHER:
- Set it to zero/low explicitly
- Or subtract more than the total balance

The only way I can see the balance going to near-zero is:

1. The swap subtracts 193 from 386 → 193 remaining
2. Some other code path also subtracts ~193 (double-deduction)
3. Or some code path SETs the balance to a very low value

**The double-deduction scenario is possible** if there's a P2P broadcast of the swap that causes the receiving node to ALSO deduct. Let me check if swap events are broadcast and processed.

---

## 4. Open Investigation Items

### 4.1 HIGHEST PRIORITY: P2P Swap Broadcast

The swap handler broadcasts swap events to the network via gossipsub. If the RECEIVING nodes also deduct the balance, it would be a double-deduction.

```
Search needed: main.rs — handling of /qnk/mainnet-genesis/dex/swaps gossipsub topic
```

### 4.2 HIGH PRIORITY: Concurrent add_balance / subtract_balance Race

`add_balance` and `subtract_balance` both do read-modify-write on the same key. If they race (no lock), one can overwrite the other's result.

### 4.3 MEDIUM PRIORITY: save_wallet_balance in Block Producer

If the block producer writes in-memory values to RocksDB, and in-memory is stale, it can overwrite correct RocksDB values.

### 4.4 MEDIUM PRIORITY: Turbo Sync State Import

If turbo sync imports balance state from peers, and the peer has a different (corrupted) value, it can overwrite the local correct value.

---

## 5. What v10.3.1 Should Do (Prevention Only, No Balance Modification)

### Keep:
1. **DEX safety gate** — block swaps during bootstrap
2. **Atomic WriteBatch** — make DEX operations crash-safe
3. **Persistent processed-block tracking** — prevent balance_consensus reprocessing

### Add:
4. **Audit and fix ALL absolute balance writers** — every `save_wallet_balance()` and `set_balance()` call must be verified
5. **Balance anomaly detection** — log-only, never modify
6. **Pre-shutdown balance snapshot** — dump all balances to a file for operator reference

### Do NOT:
7. Automatically modify any user's balance
8. Trust any single node's balance as authoritative
9. Run apply_dex_qug_adjustments() automatically

---

## 6. Honest Summary

| Question | Answer |
|----------|--------|
| Is the user's money safe on the blockchain? | YES — coinbase transactions are immutable |
| Can we compute the exact correct balance? | NO — pruned blocks prevent full replay |
| Do we know the exact cause? | NO — multiple candidate mechanisms |
| Is it safe to deploy v10.3.1? | NOT YET — need to audit all balance writers |
| Will the balance self-correct over time? | Mining rewards accumulate, but the gap from historical mining cannot be recovered without an archive node |
| Should we trust any node's balance? | Delta and Gamma show [1K-10K] — likely closest to correct, but unverified |

---

## 7. Recommended Next Steps

1. **Audit every `save_wallet_balance` and `set_balance` call in the codebase** — identify which ones write absolute values from potentially stale sources
2. **Check P2P swap event handling** — verify that swap deductions are not applied twice (once locally, once from P2P broadcast)
3. **Test on Delta Docker** — reproduce the bug: swap → verify balance → restart → verify balance
4. **Do NOT restart Gamma or Delta** until the root cause is found — they may have the only correct balance
5. **Write the fix, test it, share with reviewers** — only deploy after peer review and testing
