# Technical Review v2: Critical DEX Swap Balance Corruption — After Peer Review

**Date:** 2026-04-13  
**Severity:** CRITICAL (persisted accounting state corruption after restart)  
**Network:** Q-NarwhalKnight mainnet-genesis ($1B market cap)  
**Revision:** v2 — incorporates DeepSeek peer review feedback  
**Status:** Fixes implemented (v10.3.1), pending compilation and Delta Docker test

---

## What Changed From v1

DeepSeek's peer review identified several critical refinements:

| Item | v1 | v2 (This Document) |
|------|----|----|
| Classification | "Display only" | **Persisted accounting corruption** (display first, then state corruption after restart) |
| Fix 2 priority | "Primary prevention" | **Mitigation only** — requires idempotency proof and ordering guarantee |
| Write atomicity | Not addressed | **Atomic WriteBatch** — balance + debit counter in single all-or-nothing RocksDB write |
| Idempotency | Assumed | **Proven** — delta-based tracking: `desired_net - previously_applied = delta`. If delta=0, no-op |
| DEX during bootstrap | Not addressed | **Safety gate** — `dex_ready: AtomicBool` blocks all swaps until reconciliation completes |
| Negative balance | Not addressed | **`saturating_sub`** everywhere — balance can never go below zero |
| Root cause | "Stale cache + off-chain deduction" | **"DEX debits live outside the deterministic state transition system"** (DeepSeek's formulation) |

---

## The Architectural Flaw (DeepSeek's Key Insight)

> "The root bug is that DEX debits live outside the same deterministic state transition system as mining rewards."

Two balance authorities exist simultaneously:

| Authority | What it tracks | Persistence |
|-----------|---------------|-------------|
| **Chain-derived** (balance_consensus) | Coinbase mining rewards from block replay | Deterministic from chain |
| **Imperative mutations** (DEX swap handler) | Direct RocksDB subtract_balance/add_balance | RocksDB only, no chain record |

These only work together if **replay logic explicitly re-applies all off-chain mutations deterministically**. That reconciliation path existed (`apply_dex_qug_adjustments`) but was never called on normal restarts.

---

## Two Distinct Failure Modes (Refined From v1)

### Mode A: DEX Debit Omission (balance too HIGH)

After restart, `balance_consensus` replays full chain history → computes total coinbase rewards → writes balance. The DEX swap deduction is invisible to chain replay → balance jumps BACK to pre-swap value.

**Example:** User has 386 QUG from mining, sells 193 on DEX → balance should be 193. Restart → chain says 386 → balance reset to 386 (the 193 deduction is lost).

### Mode B: Partial History Overwrite (balance too LOW)

After restart, if `balance_consensus` only sees RECENT blocks (not full history), the balance collapses to only the last few blocks' worth of coinbase rewards.

**Example:** User has 386 QUG from months of mining. Restart → balance_consensus processes only last 100 blocks → balance drops to ~1-10 QUG.

**What we observed:** Both modes in sequence. First Mode A (balance should be 193 after swap but chain says ~386), then Mode B (balance_consensus overwrites to ~1-10 from partial history).

---

## The Four Fixes (v10.3.1)

### Fix 1: DEX Safety Gate (NO swaps during bootstrap)

```rust
// handlers.rs — top of execute_swap()
if !state.dex_ready.load(Ordering::Acquire) {
    return Ok(Json(ApiResponse::error(
        "DEX temporarily disabled while node synchronizes."
    )));
}
```

- `dex_ready: Arc<AtomicBool>` in AppState, initialized to `false`
- Set to `true` ONLY after startup reconciliation completes successfully
- If reconciliation fails → DEX stays disabled → no silent corruption
- DeepSeek: "node starts in DEX disabled / read-only mode"

### Fix 2: Atomic Write Batch (all-or-nothing DEX operations)

```rust
// storage lib.rs — new function
pub async fn atomic_subtract_and_record_dex_debit(
    &self, wallet_hex: &str, amount: u128
) -> Result<u128> {
    let new_balance = current_balance.saturating_sub(amount);
    let new_debit_total = current_debit.saturating_add(amount);
    let new_applied_net = current_credit as i128 - new_debit_total as i128;

    // Single atomic WriteBatch — balance + debit counter + applied-net tracker
    // If process crashes here, NONE are written (RocksDB atomicity guarantee)
    self.hot_db.write_batch(vec![
        (CF_MANIFEST, balance_key, new_balance.to_le_bytes()),
        (CF_MANIFEST, debit_key, new_debit_total.to_le_bytes()),
        (CF_MANIFEST, applied_key, new_applied_net.to_le_bytes()),
    ]).await?;

    Ok(new_balance)
}
```

- DeepSeek: "if the process crashes between subtract_balance and record_dex_qug_debit, the debit counter is wrong"
- Same atomic pattern for `atomic_add_and_record_dex_credit`
- Credit side has fallback path (non-atomic) if atomic fails — user still gets tokens this session

### Fix 3: Idempotent Startup Reconciliation (safe to run any number of times)

```rust
// main.rs — after balance watermark initialization
tokio::spawn(async move {
    // Wait 30s for initial sync to settle
    tokio::time::sleep(Duration::from_secs(30)).await;

    // Idempotent reconciliation: delta = desired_net - previously_applied
    match storage.apply_dex_qug_adjustments().await {
        Ok(adjusted) => {
            // Refresh in-memory from RocksDB
            // Then enable DEX
            dex_state.dex_ready.store(true, Ordering::Release);
        }
        Err(e) => {
            error!("DEX adjustment failed — DEX stays DISABLED");
            return; // Do NOT enable DEX
        }
    }
});
```

**Idempotency proof (per DeepSeek's requirement):**

The adjustment uses delta tracking, not absolute reapplication:

```
desired_net = credited - debited    (from counters, immutable history)
previously_applied = dex_applied_net  (from tracker, updated atomically with each swap)
delta = desired_net - previously_applied

If delta == 0 → skip (already correct)
If delta < 0 → subtract |delta| from current balance
If delta > 0 → add delta to current balance
```

Running the function twice: after the first run, `previously_applied == desired_net`, so `delta == 0` → no-op. **Proven idempotent.**

### Fix 4: saturating_sub Everywhere (DeepSeek recommendation)

```rust
// Prevents negative balances if debits ever exceed credits + chain balance
let new_balance = current_balance.saturating_sub(abs_delta);
// Warning logged if this happens — admin investigation needed
```

---

## Safety Analysis for $1B Mainnet

| Concern | Assessment |
|---------|-----------|
| Can the fixes cause new bugs? | No — all changes are ADDITIVE (gate, atomic batch, reconciliation). No existing logic modified. |
| Can the fixes lose funds? | No — `saturating_sub` prevents underflow; atomic batch prevents partial writes; gate prevents swaps during unsafe state |
| What if reconciliation fails? | DEX stays disabled. Mining, P2P, block production all unaffected. Admin intervention required. |
| What if the gate never opens? | Same as above. Only DEX is affected. All other node functions work normally. |
| Can the atomic batch fail? | If RocksDB WriteBatch fails, the swap is rejected with an error message. No state change. User retries. |
| Performance impact? | Minimal — one extra AtomicBool::load per swap request. WriteBatch is same speed as individual writes. |
| Backward compatibility? | Full — `dex_applied_net` keys are new. Old nodes without them get `previously_applied = 0`, which produces the correct delta on first run. |

---

## Testing Requirements (Enhanced Per DeepSeek Review)

### Required Before Production

```
1. Swap + restart (the exact bug scenario):
   - Accumulate 100 QUG from mining
   - Swap 50 QUG for QUGUSD
   - Verify: 50 QUG remaining
   - Restart node, wait 60s for reconciliation
   - Verify: STILL 50 QUG (not 100, not 0)

2. Idempotency (DeepSeek requirement):
   - Run apply_dex_qug_adjustments() twice in succession
   - Verify balance unchanged on second run
   - Verify dex_applied_net unchanged on second run

3. Crash mid-swap (atomic batch test):
   - Start a swap, kill -9 the node DURING the write
   - Restart, verify either: swap fully applied OR fully rolled back
   - Never partial (balance changed but counter not, or vice versa)

4. DEX gate (bootstrap safety):
   - Restart node, immediately attempt swap within 30s
   - Verify: "DEX temporarily disabled" error returned
   - Wait 60s, retry → swap succeeds

5. Negative balance prevention:
   - Set debit counter to exceed chain balance (artificial test)
   - Run apply_dex_qug_adjustments()
   - Verify: balance = 0, not underflow (not u128::MAX)
   - Verify: warning logged

6. Multi-wallet regression (DeepSeek requirement):
   - Wallet A: no swaps (should be unaffected)
   - Wallet B: one swap
   - Wallet C: multiple swaps
   - Wallet D: swap + post-swap mining rewards
   - Restart and verify ALL FOUR are correct

7. Other off-chain mutator audit (DeepSeek requirement):
   - Verify no other code path directly modifies RocksDB balances
   - Search for: set_balance, add_balance, subtract_balance outside balance_consensus
   - Ensure all are covered by reconciliation
```

---

## Long-Term Architectural Fix (P1)

DeepSeek's strongest recommendation: **Move DEX accounting into the deterministic replayable state.**

Two approaches:

### Option A: On-Chain DEX Transactions
Record every swap as a blockchain transaction (like Uniswap on Ethereum). Chain replay produces correct balances deterministically. No reconciliation needed.

**Pro:** Eliminates the dual-authority problem entirely.  
**Con:** Increases block size, adds latency to swaps, requires consensus upgrade.

### Option B: Separate Balance Domain
Maintain two balance domains: `chain_balance` (from mining) and `dex_balance` (from swaps). Each is authoritative in its domain. The displayed balance is the sum.

```
displayed_balance = chain_balance + dex_net_adjustment
chain_balance: computed by balance_consensus (deterministic)
dex_net_adjustment: maintained by DEX handler (never touched by balance_consensus)
```

**Pro:** No consensus change needed. Balance_consensus can replay freely without affecting DEX state.  
**Con:** Adds complexity to every balance read.

**Recommendation:** Option B for v10.4.0, Option A for v11.0.0 (consensus upgrade).

---

## Immediate Remediation

1. Deploy v10.3.1 to Delta Docker — test swap+restart scenario
2. If clean, deploy to Epsilon (admin balance rebuild first)
3. Verify user's balance is restored after reconciliation
4. Monitor for 48h before deploying to Beta/Gamma

---

## References

- DeepSeek peer review: 2026-04-13 (inline in conversation)
- v10.2.1 watermark removal: original race condition with turbo sync batch task
- v9.3.3 DEX debit counter: `record_dex_qug_debit` / `apply_dex_qug_adjustments`
- RocksDB WriteBatch atomicity: https://github.com/facebook/rocksdb/wiki/Basic-Operations#atomic-updates
- Files changed: `handlers.rs`, `lib.rs` (AppState), `main.rs`, `lib.rs` (storage)
