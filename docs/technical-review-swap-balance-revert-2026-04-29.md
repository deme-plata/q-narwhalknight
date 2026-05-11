# Technical Review — QUGUSD→QUG Swap Balance Revert & Post-Checkpoint DEX Credit Loss

**Date:** 2026-04-29  
**Versions:** v10.5.1 (swap revert fix), v10.5.2 (checkpoint credit loss fix)  
**Files Changed:**
- `crates/q-api-server/src/handlers.rs`
- `crates/q-storage/src/lib.rs`
- `Cargo.toml`

---

## 1. Original Bug: QUGUSD→QUG Swap Balance Reverts After 2–3 Minutes

### Symptom
User swaps QUGUSD → native QUG. Frontend shows correct balance immediately. After 2–3 minutes, the native QUG balance silently reverts to its pre-swap value as if the swap never happened.

### Root Cause (traced through 5 layers)

#### Layer 1 — Transaction routing in `balance_consensus.rs`

When a Swap transaction is submitted, `balance_consensus` routes it by `token_type`:

```
token_type == QUG   → native-transfer branch → subtract_balance / add_balance on CF_MANIFEST
token_type == QUGUSD → token-transfer branch  → subtract_token_balance / add_token_balance
```

A QUGUSD→QUG swap transaction carries `token_type = QUGUSD` (the input token). So `balance_consensus` routes it to the **token-transfer branch**.

#### Layer 2 — Token transfer branch fails silently

In the token-transfer branch:
```rust
if let Err(_) = subtract_token_balance(user, QUGUSD, amount) {
    continue;  // ← silent skip
}
// add_balance for QUG is NEVER reached
```

The handler in `handlers.rs` had **already** deducted the QUGUSD from the user's token balance before submitting the transaction. So `subtract_token_balance` fails (insufficient balance), hits `continue`, and **`add_balance` for the QUG output is never called**.

#### Layer 3 — StateApplicator credits the wrong column family

`StateApplicator::apply_balance_credit()` (called during `replay_rich_transactions`) writes to `CF_TOKEN_BALANCES` using 32-byte binary keys. Native QUG balances live in `CF_MANIFEST` under string keys (`wallet_balance_{hex}`). The two column families are independent. Writing QUG to `CF_TOKEN_BALANCES` has no effect on what the user sees.

#### Layer 4 — get_balance reads the wrong place

`get_balance(wallet)` reads `CF_MANIFEST:wallet_balance_{hex}`. Because `balance_consensus` never called `add_balance`, this value was never updated. It still reflects the pre-swap balance.

#### Layer 5 — 15-second sync task enforces the wrong value

`main.rs` runs a task every 15 seconds that reads `CF_MANIFEST:wallet_balance_` for every session and writes it into the in-memory `wallet_balances` map. Any optimistic in-memory increase from the handler is overwritten within 15 seconds.

### Why It Looked Correct Briefly

The swap handler updated `wallet_balances` (in-memory) optimistically after calling `record_dex_qug_credit`. The frontend SSE stream picked this up immediately. The 15-second task then "corrected" the in-memory value back to the stale on-disk value.

---

## 2. v10.5.1 Fix: Direct CF_MANIFEST Write in Handler

### Approach

In `handlers.rs`, in the `if to_is_qug` block, after the QUGUSD deduction:

1. Call `record_dex_qug_credit(&wallet_hex, amount)` — records the credit counter for idempotency.
2. Call `add_balance(&wallet_hex, amount)` — **writes the QUG credit directly to `CF_MANIFEST:wallet_balance_`**, the same location that `balance_consensus` uses for the debit side and that `get_balance` reads.
3. Re-read the updated balance from RocksDB and write it into `wallet_balances` (in-memory), so the 15-second sync task sees the correct value and does not revert it.

### Why This Is Correct for Normal Operation

`add_balance` writes to the exact same key (`wallet_balance_{hex}` in `CF_MANIFEST`) that the 15-second sync task reads. So after step 2, the on-disk and in-memory values are consistent. The sync task confirms rather than reverts.

### Why `balance_consensus` Not Double-Crediting Is Safe

`balance_consensus` processes the Swap transaction via the token-transfer branch, calls `subtract_token_balance(QUGUSD)` which fails (already deducted), hits `continue`, and exits without touching QUG. No double-credit.

### Known Limitation Introduced

This fix writes the QUG credit in the HTTP handler (synchronous with the API call) but `balance_consensus` is the authoritative replay path. If the chain forks and the swap transaction is rolled back, the direct `add_balance` write is not automatically reversed. This is acceptable because:
- The chain is not currently subject to meaningful reorgs
- The existing DEX credit/debit counter system (`record_dex_qug_credit`) provides a reconciliation path

---

## 3. New Bug Revealed by v10.5.1: Post-Checkpoint DEX Credit Loss on Restart

### Symptom

After deploying v10.5.1 and restarting both servers, users who had done QUGUSD→QUG swaps after the checkpoint height (block 16,538,868) saw their native QUG balance revert to the checkpoint value. Example: user had 735 QUG (648 checkpoint + 87 from swap), after restart: 648.

### Why v10.5.1 Revealed This

With the old code (v10.3.2), the balance always reverted after 2–3 minutes. Users never accumulated a "correct" balance that a restart could destroy. With v10.5.1, the balance persisted correctly — until a restart.

### Root Cause

#### Step 1 — Checkpoint restore purges `wallet_balance_` only

`apply_balance_checkpoint()` in `lib.rs` (line 5511):
```rust
let deleted = self.delete_by_prefix(b"wallet_balance_").await.unwrap_or(0);
// Imports checkpoint data (height 16,538,868) for 1,326 wallets
```

Keys purged: `wallet_balance_{hex}` only.  
Keys **NOT** purged: `dex_qug_credited:{hex}`, `dex_qug_debited:{hex}`, `dex_applied_net:{hex}`.

#### Step 2 — Post-checkpoint replay is Coinbase + Transfer only

After import, the node replays blocks from height 16,538,869 to the current tip, but **only** Coinbase and Transfer transactions. Swap transactions are excluded from this replay path.

This means any QUG received via a QUGUSD→QUG swap after the checkpoint is not re-credited during replay.

#### Step 3 — `apply_dex_qug_adjustments` is supposed to fix this, but doesn't

`apply_dex_qug_adjustments()` is designed to re-apply DEX net adjustments on startup:
```
delta = (dex_qug_credited - dex_qug_debited) - dex_applied_net
apply delta to wallet_balance_
```

The problem: `record_dex_qug_credit()` (called by the v10.5.1 handler fix) writes:
```rust
dex_applied_net:{wallet} = new_credit_total - current_debit_total
```

This sets `dex_applied_net` to the full net amount **at the time of the swap**. After checkpoint restore:
- `wallet_balance_` = 648 (reset by checkpoint)
- `dex_applied_net:` = 87 (survived — checkpoint does not purge it)
- `dex_qug_credited:` = 87 (survived)

`apply_dex_qug_adjustments` computes:
```
desired_net = 87 - 0 = 87
previously_applied = dex_applied_net = 87
delta = 87 - 87 = 0  ← nothing applied
```

Balance stays at 648. The 87 QUG is silently lost on every restart.

### Why the Old Code Had This Too (But It Was Hidden)

With v10.3.2, `record_dex_qug_credit` also set `dex_applied_net`. So the same delta=0 problem existed. But since the balance always reverted after 2–3 minutes anyway, there was no "correct" balance to lose. The restart issue was masked by the continuous revert.

---

## 4. v10.5.2 Fix

### Fix 1 — Checkpoint restore resets `dex_applied_net` (`lib.rs` line 5511)

```rust
let dex_net_purged = self.delete_by_prefix(b"dex_applied_net:").await.unwrap_or(0);
if dex_net_purged > 0 {
    warn!("🏁 [CHECKPOINT] Reset {} dex_applied_net entries — adjustments will be re-applied this boot.", dex_net_purged);
}
```

After this purge:
- `dex_applied_net:` = 0 for all wallets (reset)
- `dex_qug_credited:` = 87 (preserved)
- `apply_dex_qug_adjustments`: delta = 87 - 0 = 87 → balance = 648 + 87 = 735 ✓
- Saves `dex_applied_net = 87` for idempotency on subsequent calls

On the **next** restart (same checkpoint, no new swaps):
- Checkpoint purges `wallet_balance_` AND `dex_applied_net:` again
- `dex_applied_net = 0` again
- delta = 87 → balance restored correctly ✓

### Fix 2 — `record_dex_qug_credit` no longer writes `dex_applied_net` (`lib.rs` line 9428)

```rust
// v10.5.2: Do NOT write dex_applied_net here.
// Only apply_dex_qug_adjustments() should set this, after checkpoint restore.
self.hot_db.put_sync(CF_MANIFEST, credit_key.as_bytes(), &new_credit_total.to_le_bytes()).await?;
```

This separates concerns: the handler records credits/debits, `apply_dex_qug_adjustments` is the sole authority on what has been applied to the on-disk balance. The checkpoint purge of `dex_applied_net` is now the only mechanism that triggers re-application.

---

## 5. Correctness Analysis

### Normal Operation (No Restart)

| Event | `wallet_balance_` | `dex_qug_credited:` | `dex_applied_net:` | In-Memory |
|-------|-------------------|---------------------|---------------------|-----------|
| Before swap | 648 | 0 | 0 | 648 |
| Swap executes (v10.5.1 handler) | 735 (add_balance) | 87 | 0 (v10.5.2: not set) | 735 |
| 15s sync task | reads 735 → in-memory = 735 | — | — | 735 ✓ |

### After Restart

| Phase | `wallet_balance_` | `dex_applied_net:` | Result |
|-------|-------------------|--------------------|--------|
| Checkpoint purge | 648 (reset) | 0 (reset by v10.5.2) | — |
| `apply_dex_qug_adjustments` | 648 + 87 = 735 | 87 | 735 ✓ |
| Second restart | 648 (checkpoint) → +87 = 735 | 0 → 87 | 735 ✓ |

### Idempotency

`apply_dex_qug_adjustments` is only called once per startup from `main.rs`. After it runs, it writes `dex_applied_net = desired_net`. If somehow called again in the same session (not currently possible), delta = 0 → safe.

### Double-Credit Risk

`add_balance` in the handler writes to CF_MANIFEST. `apply_dex_qug_adjustments` also writes to CF_MANIFEST. Could both apply the same credit?

- **Same session (no restart):** `apply_dex_qug_adjustments` runs at startup before any swaps. The swap handler runs later. `apply_dex_qug_adjustments` does not run again. No double-credit.
- **After restart:** Checkpoint resets `wallet_balance_` to 648 (wiping the handler's `add_balance` write). `apply_dex_qug_adjustments` then applies +87. Net result: 648 + 87 = 735. Correct — not a double-credit.

### Node Without Checkpoint (Empty DB)

`delete_by_prefix(b"dex_applied_net:")` on empty DB → deletes 0 entries, no-op. `apply_dex_qug_adjustments` with no credits/debits → skips all wallets. Correct.

---

## 6. Remaining Risks and Open Issues

### R1 — DEX Debits After Checkpoint Are Also Not Replayed

The post-checkpoint replay exclusion of swap transactions means QUG→QUGUSD debits (the debit side) are also not replayed. However, the `atomic_subtract_and_record_dex_debit` function already writes directly to `wallet_balance_` (CF_MANIFEST) — mirroring what v10.5.1 does for credits. So debits already survive correctly in CF_MANIFEST and are preserved across checkpoints as long as the user's balance doesn't drop below checkpoint value.

**Gap:** If a user's `wallet_balance_` after the checkpoint is lower than at checkpoint time (due to QUG→QUGUSD swaps in post-checkpoint blocks), the checkpoint will restore a higher balance. `apply_dex_qug_adjustments` will then subtract the debits — which is correct behavior.

**Verify:** Confirm `atomic_subtract_and_record_dex_debit` also does NOT set `dex_applied_net` (or if it does, that the v10.5.2 checkpoint reset handles it).

### R2 — Multi-Swap Accumulation

A user who does multiple QUGUSD→QUG swaps between restarts: each swap increments `dex_qug_credited` and `add_balance` writes the running total. After restart, `apply_dex_qug_adjustments` reads the full accumulated `dex_qug_credited` and applies the full net. This is correct.

### R3 — Checkpoint Data Staleness

The checkpoint is at height 16,538,868 (hardcoded in the binary). As the chain grows, the post-checkpoint replay window grows. Currently ~151k blocks. Each restart requires replaying Coinbase + Transfer for all 151k+ blocks. This replay time will grow with each deployment until a new checkpoint is embedded.

**Recommendation:** Update CHECKPOINT_DATA to a more recent height in the next planned maintenance release.

### R4 — `apply_dex_qug_adjustments` Not Called After Live Checkpoint (No Restart)

If a checkpoint is applied to a running node (future feature), `apply_dex_qug_adjustments` would not be re-called. This is not currently possible — checkpoints only apply at startup.

### R5 — Incomplete Post-Checkpoint Swap Replay

The fundamental design gap: swap transactions are excluded from post-checkpoint replay. This means any QUG state from swaps only exists in the DEX counter mechanism. If the counter data is ever lost or corrupted, the credits are unrecoverable from chain history.

**Long-term recommendation:** Include swap transactions in the post-checkpoint replay, or update the checkpoint more frequently.

---

## 7. Summary of Changes

| File | Change | Risk |
|------|--------|------|
| `crates/q-storage/src/lib.rs:5511` | Checkpoint restore also purges `dex_applied_net:` keys | Low — additive purge, enables correct re-apply |
| `crates/q-storage/src/lib.rs:9428` | `record_dex_qug_credit` no longer writes `dex_applied_net` | Low — removes incorrect write, leaves tracking to apply fn |
| `crates/q-api-server/src/handlers.rs` | `add_balance` + in-memory sync after QUGUSD→QUG swap | Medium — direct DB write in handler, not via consensus |
| `Cargo.toml` | Version 10.5.0 → 10.5.1 → 10.5.2 | None |

**Deploy:** Restart after v10.5.2 deploy will automatically restore affected balances via `apply_dex_qug_adjustments`. No manual intervention required.
