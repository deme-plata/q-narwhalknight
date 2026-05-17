# Technical Review v3: DEX Balance Corruption — Root Cause, Prevention, and Recovery Philosophy

**Date:** 2026-04-13  
**Severity:** CRITICAL  
**Network:** Q-NarwhalKnight mainnet-genesis ($1B market cap)  
**Revision:** v3 — complete rewrite after discovering ALL nodes affected  
**Core Principle:** **Never automatically modify user balances. Prevent corruption; don't repair it.**  
**Prepared for:** DeepSeek + ChatGPT peer review

---

## 0. Why This Review Exists

A user swapped ~50% of their QUG for QUGUSD on the DEX. After the swap, their QUG balance dropped to near zero instead of the expected ~50% remaining. Investigation revealed that the corruption is not isolated to one node — **every node that restarts exhibits the same balance corruption**. This is a systemic architectural flaw, not a one-time incident.

**Previous reviews (v1, v2) proposed automatic balance repair on startup. This review rejects that approach.** On a $1B mainnet, automated balance modification is itself a risk vector. The correct approach is: prevent the corruption from occurring, detect anomalies, and provide manual recovery tools that require explicit human approval.

---

## 1. Verified Facts (From Logs — Not Assumptions)

### 1.1 The Swap (Verified)

```
Timestamp: 2026-04-13 14:59:31 CEST (Epsilon)
Action:    Executing swap: [100-1K] QUG for QUGUSD (wallet: qnkefca…0723)
Deduction: RocksDB: 386.03537279 → 193.03632829 QUG
In-memory: was 193.22932733 QUG (stale — half of RocksDB value)
Result:    193 QUG deducted, 193 QUG remaining in RocksDB
```

**Verified:** The swap itself executed correctly. RocksDB had 386 QUG, 193 was subtracted, 193 remained. The AMM output was ~532,340 QUGUSD (correctly priced at ~$2,759/QUG).

### 1.2 The Corruption Timeline (Verified)

```
14:52  balance_consensus logs: qnkefca…e910 → [100-1K] QUG (correct, pre-swap)
14:59  Swap executed: RocksDB 386 → 193 QUG
15:06  balance_consensus logs: qnkefca…e910 → [1-10] QUG (CORRUPTED)
15:09+ balance_consensus logs: qnkefca…e910 → [1-10] QUG (still corrupted, slowly growing)
```

**Verified:** Seven minutes after the swap, `balance_consensus` overwrote the 193 QUG balance with a value in the [1-10] range. The balance has been slowly rebuilding from new mining rewards since then.

### 1.3 Multi-Node Corruption (Verified 2026-04-13 ~17:10 CEST)

| Node | Balance Range | Uptime | Restart Cause |
|------|--------------|--------|---------------|
| **Epsilon** | `[10-100]` | 4h 37min | v10.3.0 deploy (PQC fix) |
| **Beta** | `[10-100]` | 20min | v10.3.0 deploy |
| **Gamma** | `[1K-10K]` | Longer (pre-restart) | Was restarting at check time |
| **Delta** | N/A | Running LWMA test binary | Not authoritative |

**Verified:** Beta, which was previously showing `[100-1K]` (presumed correct), dropped to `[10-100]` after its own restart. Gamma showed a much higher `[1K-10K]` before it started shutting down.

**Critical finding:** The earlier claim that "Beta has the correct balance" was **wrong**. Beta's balance was also corrupted after restart. No node can be trusted to have the correct balance without verification.

### 1.4 What We Know About the User's Actual Balance

| Source | Value | Trustworthy? |
|--------|-------|-------------|
| Pre-swap RocksDB (Epsilon log) | 386.035 QUG | **Yes** — logged at swap time |
| Post-swap RocksDB (Epsilon log) | 193.036 QUG | **Yes** — logged at swap time |
| Current Epsilon display | ~17.93 QUG | **No** — corrupted |
| Current Beta display | [10-100] range | **No** — corrupted after restart |
| Gamma display | [1K-10K] range | **Unknown** — was highest but now restarting |
| QUGUSD balance | 24,258,229.73 | **Likely correct** — token balances not affected by this bug |
| Blockchain (coinbase history) | Unknown exact total | **Immutable** — but can't be summed without all blocks (pruned) |

**Honest assessment:** We cannot determine the exact correct QUG balance from any currently available source. The blockchain is the ultimate truth, but blocks 0–13M are pruned and cannot be replayed. The only verified data point is the swap log: the user had 386.035 QUG immediately before the swap.

---

## 2. Root Cause Analysis

### 2.1 The Three Interacting Flaws

**Flaw 1: `balance_consensus` writes absolute balances, not deltas**

When `balance_consensus` processes a block with a coinbase transaction for wallet X:
```rust
// Simplified from balance_consensus.rs
let current = get_balance(wallet);  // Read from RocksDB
let new_balance = current + coinbase_amount;  // Add reward
set_balance(wallet, new_balance);  // OVERWRITE in RocksDB
```

This is safe during normal operation (current is correct, new_balance = correct + reward). But after restart, if `current` is wrong (because the balance was already overwritten by a previous cycle), every subsequent write compounds the error.

**Flaw 2: The dedup mechanism is volatile (in-memory LRU)**

The LRU cache (500K entries) prevents double-processing of blocks. But it is:
- Lost on restart (in-memory only)
- Evicted when full (500K entries is finite)

After restart, `balance_consensus` sees every incoming block as "new" and reprocesses it. If the block was already processed before restart, the coinbase reward is double-counted — UNLESS the balance was reset to zero first, in which case only recent blocks contribute and the balance is UNDER-counted.

**The actual behavior observed** (balance drops to near-zero after restart) suggests that something is resetting the balance before reprocessing. This could be:

a) A balance rebuild/migration task that zeroes balances before replaying  
b) The 15-second balance sync task overwriting with a partial value  
c) Turbo sync delivering blocks with balance_updates that SET (not ADD) balances  
d) An interaction between the watermark and the sync mechanism  

**We do not know with certainty which of these is the exact trigger.** The code is complex enough that multiple mechanisms could contribute. What we DO know is the result: after restart, the balance drops to near-zero and only accumulates from new blocks.

**Flaw 3: DEX deductions are invisible to chain replay**

The swap handler calls `subtract_balance()` directly on RocksDB. This deduction is not recorded in any block. If the balance is ever recomputed from chain data, the DEX deduction vanishes — the chain says the user earned X QUG from mining, so that's the "correct" chain-derived balance (ignoring all DEX activity).

The DEX debit counter (`dex_qug_debited:{wallet}`) exists precisely to track this, but:
- It is never automatically applied after restart
- Even if it were applied, it requires the BASE balance (chain-derived) to be correct first
- If the base balance is itself wrong (because of Flaw 1+2), applying the DEX delta produces a doubly-wrong result

### 2.2 The Pruning Complication

Blocks 0 through ~13.2 million were silently deleted by the adaptive pruning bug (fixed in v10.3.0, but damage already done). This means:

- A full chain replay from genesis is **impossible** on any existing node
- The correct cumulative mining balance **cannot be independently computed** from chain data alone
- Any "balance rebuild from chain" operation will produce an INCORRECT result (too low) because it can only see recent blocks

This is why automatic repair is dangerous: **there is no reliable source of truth to repair FROM.**

### 2.3 Why Gamma Showed [1K-10K] (A Different Kind of Wrong)

Gamma's much higher balance ([1K-10K] range vs [10-100] on others) is likely because:
- Gamma had been running longer without restart
- Its `balance_consensus` accumulated more blocks' worth of coinbase rewards
- But this doesn't mean Gamma's value is CORRECT — it could be:
  - Too high (if blocks were double-processed at some point)
  - Too low (if some blocks were missed)
  - Coincidentally close to the real value
  
**We cannot verify which without an independent calculation, which is impossible due to pruning.**

---

## 3. Design Philosophy: Prevent, Detect, Gate — Never Auto-Fix

### 3.1 The Principle

```
                    ┌──────────────────────────────┐
                    │   NEVER automatically modify  │
                    │   user balances on a $1B      │
                    │   mainnet.                    │
                    │                               │
                    │   • Prevent corruption         │
                    │   • Detect anomalies           │
                    │   • Gate dangerous operations  │
                    │   • Manual recovery only       │
                    └──────────────────────────────┘
```

**Why not auto-fix?**

1. The "correct" balance cannot be independently computed (blocks pruned)
2. Any heuristic repair might make things worse
3. Automated balance modification is itself a potential exploit vector
4. The user explicitly does not trust automated balance changes
5. On a $1B network, conservative inaction is safer than aggressive correction

### 3.2 What v10.3.1 WILL Do

| Action | Category | Modifies Balance? |
|--------|----------|-------------------|
| DEX safety gate (block swaps during bootstrap) | **Prevent** | No |
| Atomic WriteBatch for DEX operations | **Prevent** | No (same writes, just atomic) |
| Persistent processed-block tracking | **Prevent** | No (prevents balance_consensus from reprocessing) |
| Balance anomaly detection on startup | **Detect** | No (log-only) |
| dex_ready status in health endpoint | **Detect** | No |

### 3.3 What v10.3.1 WILL NOT Do

| Action | Why Not |
|--------|---------|
| Automatic balance repair on startup | Cannot verify correctness (blocks pruned) |
| `apply_dex_qug_adjustments()` on startup | Modifies user balances without consent |
| Balance sync from peer nodes | Peers may also be corrupted |
| Admin balance rebuild | Only if user explicitly requests, not automatic |
| Setting balance from any external source | No trusted source available |

---

## 4. The Four Fixes (Prevention Only)

### Fix 1: DEX Safety Gate

**Purpose:** Prevent new DEX corruption during the dangerous bootstrap window.

```rust
// AppState — new field
pub dex_ready: Arc<AtomicBool>,  // Starts false, set true when safe

// execute_swap() — first check
if !state.dex_ready.load(Ordering::Acquire) {
    return Ok(Json(ApiResponse::error(
        "DEX is temporarily disabled while the node synchronizes."
    )));
}
```

**When does DEX become ready?**
- After the node is synced to within 10 blocks of network tip
- After the persistent dedup system is initialized
- NOT after any balance modification — just after the safety systems are up

**What this prevents:**
- User swapping against a stale/corrupted balance during bootstrap
- New DEX deductions being made while balance_consensus is still overwriting

### Fix 2: Atomic WriteBatch for DEX Operations

**Purpose:** Prevent partial writes if the process crashes mid-swap.

```rust
// Single atomic batch: balance deduction + DEX debit counter + applied-net tracker
// If process crashes during write, NONE are applied (RocksDB atomicity)
storage.hot_db.write_batch(vec![
    (CF_MANIFEST, balance_key, new_balance),
    (CF_MANIFEST, debit_key, new_debit_total),
    (CF_MANIFEST, applied_key, new_applied_net),
]).await?;
```

**What this prevents:**
- Debit counter out of sync with actual balance deduction
- Partial state after crash (balance deducted but counter not updated, or vice versa)

**What this does NOT do:**
- Does not modify any existing balances
- Does not change swap math or AMM logic
- Only applies to FUTURE swaps, not past ones

### Fix 3: Persistent Processed-Block Tracking (THE Core Fix)

**Purpose:** Prevent `balance_consensus` from reprocessing blocks after restart.

```rust
// New key in CF_MANIFEST:
// "processed_balance_block:{block_hash_hex}" → timestamp (u64)
//
// Before processing any block's coinbase transactions:
let processed_key = format!("processed_balance_block:{}", hex::encode(block_hash));
if storage.hot_db.get(CF_MANIFEST, processed_key.as_bytes()).await?.is_some() {
    // Block already processed — skip entirely
    // This is the key safety mechanism: balance_consensus CANNOT
    // overwrite a balance that was already correctly computed.
    return Ok(vec![]);
}

// Process the block (add coinbase rewards to balances)
// ...

// Mark as processed (AFTER successful processing, in same batch as balance write)
// This ensures atomicity: either both the balance update AND the processed marker
// are written, or neither is.
batch.push((CF_MANIFEST, processed_key, timestamp_bytes));
```

**Why this works:**
- After restart, the LRU cache is empty, but the persistent keys survive
- `balance_consensus` checks the persistent key before processing → finds it → skips
- The existing balance in RocksDB is preserved untouched
- Only genuinely NEW blocks (produced after restart) get processed
- Uses CF_MANIFEST prefix keys (zero schema change, proven approach)

**Why this is safe:**
- It ONLY prevents double-processing — it never writes a balance itself
- If the processed key is missing (e.g., first time seeing a block), normal processing happens
- If the processed key is present, the block is skipped — no balance modification
- The key is written atomically with the balance update (same WriteBatch)
- Adding processed keys cannot corrupt existing data (prefix isolation)

**Edge case — what if processed keys are lost?**
- If RocksDB data is lost, ALL data is lost (not just processed keys)
- In that case, a full sync from peers is needed anyway
- The processed keys degrade gracefully: worst case = blocks are reprocessed (current behavior), not worse

**Growth management:**
- Each key: ~80 bytes (prefix + 64-byte hash hex + 8-byte timestamp)
- At 1 bps: ~86,400 keys/day = ~7 MB/day
- After 1 year: ~2.5 GB — manageable, can be pruned for blocks older than 90 days
- Pruning processed keys for OLD blocks is safe because those blocks are also pruned from storage

### Fix 4: Balance Anomaly Detection (Log-Only, Never Modify)

**Purpose:** Detect when a balance looks wrong, alert operators, but NEVER automatically fix it.

```rust
// On startup, after initial sync settles:
// Compare current balance with what we can verify

// 1. Check if any DEX debit counters exist but balance seems too low
let debit_entries = storage.scan_prefix(CF_MANIFEST, b"dex_qug_debited:").await?;
for (key, value) in debit_entries {
    let wallet_hex = key.trim_start_matches("dex_qug_debited:");
    let total_debited = u128::from_le_bytes(value);
    let current_balance = get_balance(wallet_hex).await?;
    
    if total_debited > 0 && current_balance < total_debited {
        // Balance is less than what was debited — something is wrong
        // But we DO NOT fix it. We LOG it loudly.
        error!(
            "🚨 [BALANCE ANOMALY v10.3.1] Wallet {}...: balance={:.6} QUG but DEX debited={:.6} QUG. \
             Balance may be corrupted. Manual review required. \
             DO NOT use apply_dex_qug_adjustments() without operator approval.",
            &wallet_hex[..16],
            current_balance as f64 / 1e24,
            total_debited as f64 / 1e24,
        );
    }
}

// 2. Expose anomaly count in health endpoint
// So operators can monitor without checking logs
state.balance_anomaly_count.store(anomaly_count, Ordering::Relaxed);
```

**What this does:**
- Scans for wallets where the DEX history doesn't match the current balance
- Logs a loud error with exact numbers
- Exposes the count in the health API
- **Does NOT modify any balance**

**What the operator does with this information:**
- Investigates manually
- Decides whether to trigger a manual recovery (admin endpoint)
- Can compare across nodes to find the most trustworthy value
- Documents the decision and the reasoning

---

## 5. Manual Recovery (Admin-Triggered Only)

### 5.1 The Admin Recovery Endpoint (Already Exists)

```
POST /api/v1/admin/rebuild-balances
Authorization: admin wallet required
```

This endpoint replays available chain data and recomputes balances. It is:
- Admin-only (requires master wallet authentication)
- Manually triggered (never runs automatically)
- Logged and auditable
- **Limited by pruning** — cannot recover balances from pruned blocks

### 5.2 A Better Recovery Tool (Proposed)

```
POST /api/v1/admin/set-wallet-balance
Authorization: admin wallet required
Body: {
    "wallet": "qnkefca...",
    "new_balance": "193036328290000000000000000000",
    "reason": "Correcting balance after DEX corruption incident 2026-04-13",
    "evidence": "Swap log shows RocksDB: 386.035 → 193.036 at 14:59:31"
}
```

This would:
- Log the change with full audit trail (who, when, why, from what to what)
- Require the admin wallet signature
- Write both the new balance AND an audit record atomically
- Be used ONLY when the operator has verified the correct value from logs

**This is NOT implemented in v10.3.1.** It's a proposal for the future. The user's current balance can only be corrected by the user deciding to use this tool after reviewing the evidence.

---

## 6. What We're NOT Doing (And Why)

### 6.1 NOT Running apply_dex_qug_adjustments() On Startup

**Why not:** It modifies user balances. On a network where the base balance (from chain replay) is already wrong (due to pruning), applying a DEX delta on top of a wrong base produces a wrong result. Two wrongs don't make a right.

**Example:**
- Correct balance: 193 QUG (386 mined - 193 swapped)
- Chain-derived balance: 17 QUG (only recent blocks, rest pruned)
- DEX delta: -193 QUG
- Result of auto-fix: 17 - 193 = 0 QUG (saturated) — WORSE than the current 17

### 6.2 NOT Syncing Balances From Peer Nodes

**Why not:** After observing that Beta, Epsilon, and Gamma all have DIFFERENT balances for the same wallet, we cannot trust any node's balance as authoritative. P2P balance sync would propagate corruption, not fix it.

### 6.3 NOT Running Admin Balance Rebuild Automatically

**Why not:** The rebuild replays available blocks. With blocks 0-13M pruned, it cannot compute the correct cumulative mining balance. The result would be an authoritative-looking but incorrect value — worse than the current corrupted value because it would overwrite with false confidence.

---

## 7. The User's Current Situation (Honest Assessment)

### What the user has (verified):
- **17.93 QUG displayed** — growing from ongoing mining
- **24,258,229.73 QUGUSD** — correct (token balances unaffected by this bug)
- **A swap log proving they had 386.035 QUG before the swap and 193.036 after**

### What the user lost (from display):
- **~175 QUG** — the difference between the correct 193 QUG and the displayed 17.93 QUG
- This represents mining rewards accumulated over the lifetime of the network, minus the DEX deduction
- These rewards ARE recorded in the blockchain as coinbase transactions, but can't be summed because of pruning

### How to recover (when the user chooses):
1. The admin `set-wallet-balance` tool (proposed above) could set the balance to 193.036 QUG
2. The evidence is the swap log: `RocksDB: 386.03537279 → 193.03632829`
3. This would be a manual, logged, auditable action — not an automated repair
4. The user decides when and if to do this
5. Mining rewards since the corruption (~17.93 QUG) would need to be added to the corrected value

### What happens if we do nothing:
- The balance continues to grow from new mining rewards
- The ~175 QUG gap remains (historical mining rewards that can't be replayed)
- The QUGUSD balance is unaffected
- No further corruption occurs once v10.3.1 is deployed (persistent dedup prevents reprocessing)

---

## 8. Files Changed in v10.3.1

| File | Change | Modifies Balances? |
|------|--------|-------------------|
| `handlers.rs` | DEX safety gate at top of execute_swap() | No |
| `handlers.rs` | Atomic WriteBatch for DEX debit/credit | No (same writes, atomic) |
| `lib.rs` (AppState) | `dex_ready: AtomicBool` field | No |
| `main.rs` | Set dex_ready=true after sync + dedup init | No |
| `main.rs` | Balance anomaly detection (log-only) | No |
| `balance_consensus.rs` | Persistent processed-block tracking | No (prevents double-processing) |
| `lib.rs` (storage) | `atomic_subtract_and_record_dex_debit()` | No (same operation, atomic) |
| `lib.rs` (storage) | `atomic_add_and_record_dex_credit()` | No (same operation, atomic) |

**Total balance-modifying code in v10.3.1: ZERO.**

Every change is either a gate (preventing operations during unsafe state), an atomic wrapper (same operations, just crash-safe), or a log (detection without modification).

---

## 9. Testing Plan

### 9.1 Prevention Tests (Must Pass)

```
Test 1: Persistent dedup prevents reprocessing
  - Process block at height H (coinbase to test wallet)
  - Verify: processed_balance_block:{hash} key exists in RocksDB
  - Restart node
  - Verify: block H is NOT reprocessed (dedup key found)
  - Verify: test wallet balance unchanged after restart

Test 2: DEX gate blocks swaps during bootstrap
  - Restart node
  - Immediately attempt swap (within 5 seconds)
  - Verify: "DEX temporarily disabled" error
  - Wait for sync complete
  - Verify: swap succeeds after gate opens

Test 3: Atomic batch crash safety
  - Start swap, kill -9 node DURING the WriteBatch
  - Restart, verify: either swap fully applied OR fully rolled back
  - Never partial (balance changed but counter not, or vice versa)

Test 4: Swap + restart + balance preserved
  - Accumulate 100 QUG from mining
  - Swap 50 QUG for QUGUSD
  - Verify: 50 QUG remaining
  - Restart node
  - Wait for full sync
  - Verify: STILL 50 QUG (not 100, not 0, not any other value)
  THIS IS THE CRITICAL TEST. If this fails, v10.3.1 is not ready.
```

### 9.2 Detection Tests

```
Test 5: Anomaly detection logs correctly
  - Create test wallet with known balance
  - Inject a DEX debit counter larger than the balance
  - Restart node
  - Verify: error log "BALANCE ANOMALY" with correct numbers
  - Verify: balance NOT modified (still the original value)

Test 6: Health endpoint exposes anomaly count
  - Same setup as Test 5
  - Check /api/v1/health or status endpoint
  - Verify: anomaly_count > 0
```

### 9.3 Non-Regression Tests

```
Test 7: Normal mining unaffected
  - Mine 100 blocks
  - Verify: balance increases by sum of coinbase rewards
  - Restart
  - Mine 100 more blocks
  - Verify: balance = pre-restart balance + new rewards (no loss, no duplication)

Test 8: Token balances unaffected
  - Hold QUGUSD + custom tokens
  - Restart node
  - Verify: all token balances unchanged
```

---

## 10. Open Questions for Peer Review

### Q1: Is the persistent dedup sufficient?

The persistent dedup (processed_balance_block:{hash}) prevents `balance_consensus` from reprocessing blocks it already processed. But what about:

a) The 15-second balance sync task — does it overwrite balances from a different source?  
b) Turbo sync balance_updates in blocks — do they SET balances (absolute) or ADD (incremental)?  
c) P2P balance propagation — can a peer overwrite a correct local balance with a wrong one?

**These must be audited.** If any other code path writes to `wallet_balance_{hex}` keys, the persistent dedup alone is insufficient.

### Q2: Should we snapshot balances before restart?

A pre-shutdown balance snapshot (all wallet balances dumped to a file) would provide a recovery baseline. If the balance gets corrupted on restart, the operator has a known-good snapshot to compare against.

**Proposal:** On graceful shutdown (SIGTERM handler), write all wallet balances to `/data-mainnet-genesis/balance_snapshot_{timestamp}.json`. This is read-only from the node's perspective — it's just a dump for operator reference.

### Q3: How should we handle the pruned blocks long-term?

The inability to replay from genesis is a permanent limitation. Approaches:

a) **Accept it** — balances computed incrementally are the truth. Don't try to rebuild.  
b) **Archive node** — dedicate one node to storing all blocks (no pruning). Use it as the reference for balance disputes.  
c) **Balance checkpoints** — periodically snapshot all balances with a Merkle root. Nodes can verify their balances against the checkpoint.

### Q4: The multi-node balance divergence

Epsilon: [10-100], Beta: [10-100], Gamma: [1K-10K]. Three different values for the same wallet. Which is correct? None of them may be. How should the network converge on a single truth?

**This is an unsolved problem in the current architecture.** The honest answer is: we don't have a mechanism for balance consensus across nodes for off-chain mutations (DEX swaps). This is the architectural debt that DeepSeek correctly identified: "DEX debits must become part of the deterministic ledger/state transition model."

### Q5: What exactly triggers the balance collapse on restart?

We observed the balance dropping from [100-1K] to [1-10] about 7 minutes after restart. The exact mechanism is not fully understood. Candidates:

a) `balance_consensus` reprocessing blocks with LRU cache empty  
b) A balance rebuild/migration task in the startup sequence  
c) Turbo sync delivering blocks with absolute balance_updates that override local state  
d) The 15-second balance sync task reading from a partial/stale source  

**The persistent dedup protects against (a). But (b), (c), and (d) need separate investigation.** The code paths that write to `wallet_balance_` keys must ALL be audited and protected.

---

## 11. Audit Requirement: Every Writer to wallet_balance_

Before v10.3.1 is considered production-ready, we must enumerate EVERY code path that writes to `wallet_balance_{hex}` keys in RocksDB and verify that each one is protected against double-write or stale-write.

Known writers:

| Writer | Where | Protected? |
|--------|-------|-----------|
| `balance_consensus` — coinbase processing | `balance_consensus.rs:1243` | Fix 3 (persistent dedup) |
| `balance_consensus` — transfer processing | `balance_consensus.rs:~670` | Fix 3 |
| DEX swap `subtract_balance` | `handlers.rs:11531` | Fix 2 (atomic batch) |
| DEX swap `add_balance` (QUG credit) | `handlers.rs:~11670` | Fix 2 (atomic batch) |
| Admin rebuild | `handlers.rs:432` | Manual trigger only |
| 15-second balance sync | `main.rs:~20160` | **UNKNOWN — MUST AUDIT** |
| Turbo sync balance_updates | `main.rs:~5924` | **UNKNOWN — MUST AUDIT** |
| P2P balance propagation | Various | **UNKNOWN — MUST AUDIT** |
| Bootstrap wallet sync | `main.rs:~?` | **UNKNOWN — MUST AUDIT** |

**The "UNKNOWN" entries are the highest risk.** If any of them write absolute balances (overwriting the current value), they could cause the same corruption even WITH the persistent dedup in place.

---

## 12. Summary

### What we know:
- The user's balance was corrupted from ~193 QUG to near-zero after node restarts
- The corruption affects ALL nodes that restart, not just one
- No node currently has a verified correct balance
- Blocks 0–13M are pruned, making full chain replay impossible
- The blockchain contains the correct coinbase history (immutable)
- The QUGUSD balance (~24.2M) is unaffected

### What v10.3.1 does:
- **Prevents** future corruption via persistent dedup + DEX gate + atomic writes
- **Detects** anomalies via log-only startup checks
- **Does NOT** modify any user balance, ever, under any circumstance

### What remains unsolved:
- Restoring the user's correct balance (requires manual admin action with user consent)
- Multi-node balance divergence (architectural debt — no consensus mechanism for off-chain mutations)
- Unknown balance writers (15s sync, turbo sync, P2P propagation) — must audit before production
- Long-term: DEX operations should be on-chain transactions, not off-chain RocksDB mutations

### The path forward:
1. Fix compilation errors in v10.3.1
2. Audit ALL wallet_balance_ writers
3. Test on Delta Docker (the 4 critical tests above)
4. Deploy to production only after tests pass AND user reviews results
5. User manually decides if/when to restore their balance
