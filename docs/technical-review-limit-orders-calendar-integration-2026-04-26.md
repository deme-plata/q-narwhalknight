# Technical Review: Limit Orders + Calendar P2P Integration
**Date:** 2026-04-26  
**Status:** UPDATED after pre-deploy checks — see Check Results section  
**Mainnet MCap:** ~$1.5B — all changes require canary validation before Beta/Epsilon deploy

---

## 1. Scope

Two distinct features, very different risk profiles:

| Feature | Risk | Financial Impact | Files Changed |
|---------|------|-----------------|---------------|
| Calendar gossipsub publish | 🟢 LOW | None — social/scheduling data only | 1 file, ~5 lines |
| Limit order API wiring | 🔴 BLOCKED | Executes real swaps, modifies balances | Must fix 3 bugs first |

---

## 2. Calendar P2P Publish — ✅ CLEARED

### What's Missing
`create_event()` in `calendar_api.rs:166` saves events locally and emits SSE but does NOT call `publish_calendar_event_p2p()`. The function already exists at `calendar_api.rs:639`. One missing conditional call.

Email publish was already fully wired — no changes needed there.

### Fix
```rust
// In create_event(), after save and SSE emit:
if event.shared {
    calendar_api::publish_calendar_event_p2p(&state, &event).await;
}
```

- `if event.shared` preserves the explicit opt-in privacy model
- `publish_calendar_event_p2p()` is already proven correct (used by `share_event()`)
- Message format: `CalendarEvent` JSON — matches what receive handler deserializes
- Topic: `format!("/qnk/{}/calendar", network_id)` — already subscribed on all nodes

**Verdict: 🟢 SAFE. Implement now. No canary needed.**

---

## 3. Limit Orders — Pre-Deploy Check Results

### Check 1: execute_limit_swap() Atomicity — ❌ FAILED (3 blockers)

#### Blocker A: Function Does Not Exist (Compilation Error)
`limit_order_api.rs:526` calls `crate::dca_api::execute_limit_swap()` — **this function does not exist anywhere in the codebase**. The only related function is `execute_dca_swap` (private, different signature, 6 params vs 5).

Wiring the routes as-is produces a binary that cannot compile. This is the most important blocker — nothing else matters until this is resolved.

**Fix options:**
- A) Add `pub async fn execute_limit_swap(...)` to `dca_api.rs` that wraps `execute_dca_swap` with public visibility and the correct 5-param signature
- B) Rename the call in `limit_order_api.rs` to match `execute_dca_swap`'s actual signature

#### Blocker B: TOCTOU — Status Updated AFTER Swap (Double-Execution Risk)
The polling loop (lines 373–442) works like this:
```
T0: Snapshot all Open orders into Vec (line 373-379)
T1: For each order, fetch price (line 404)
T2: Check trigger condition (line 410-412)
T3: Execute swap (line 430)          ← swap fires here
T4: Acquire write lock, set Filled (line 432-434)   ← status updated HERE
T5: Persist to storage (via save_limit_order)
```

**Race 1 (User cancellation):** User calls `DELETE /dex/limit-orders/wallet/order_X` between T0 and T3. Cancel handler marks order `Cancelled` in storage. Loop still holds stale `Open` snapshot, executes swap anyway. User's explicit cancellation is defeated.

**Race 2 (Crash between T3 and T5):** Swap succeeds, balances change, but node crashes before status persists. On restart, order is still `Open` in storage → executes again. User gets double-charged.

**Fix:** Pre-swap status update pattern:
```
1. Acquire write lock
2. Re-read order from storage (not from snapshot)
3. If status != Open → skip (already cancelled/filled)
4. Set status = "Processing" in storage (fsync) ← NEW: crash-safe "in flight" marker
5. Release write lock
6. Execute swap
7. Set status = "Filled" or "Open" (on failure) in storage
```
This requires adding `LimitOrderStatus::Processing` variant and using `put_sync()` (not `put()`) for step 4.

#### Blocker C: Storage Methods Missing
`LimitOrderStorage.save_order()` calls `storage.save_limit_order()` — this method does not exist in `StorageEngine`. Same for `load_all_limit_orders()` and `delete_limit_order()`.

**Fix (key-prefix approach — no new CF):**
Use `CF_DCA_ORDERS` with key prefix `"limitorder:{order_id}"`:
```rust
// In q-storage/src/lib.rs — add alongside DCA methods:
pub async fn save_limit_order(&self, order_id: &str, bytes: &[u8]) -> Result<()> {
    let key = format!("limitorder:{}", order_id);
    self.hot_db.put(CF_DCA_ORDERS, key.as_bytes(), bytes).await
}
pub async fn delete_limit_order(&self, order_id: &str) -> Result<()> {
    let key = format!("limitorder:{}", order_id);
    self.hot_db.delete(CF_DCA_ORDERS, key.as_bytes()).await
}
pub async fn load_all_limit_orders(&self) -> Result<Vec<(String, Vec<u8>)>> {
    let pairs = self.hot_db.scan_prefix(CF_DCA_ORDERS, b"limitorder:").await?;
    Ok(pairs.into_iter()
        .filter_map(|(k, v)| String::from_utf8(k).ok().map(|s| (s, v)))
        .collect())
}
```
`scan_prefix()` is proven available and used throughout the codebase. No new CF, no DB schema change, no startup risk.

---

### Check 2: Oracle Zero-Guard — ✅ PASSED

`get_token_price_usd()` (lines 476–517):
- Returns `None` (not `Some(0.0)`) in all missing-data cases
- QUG: explicit `if p > 0.0` guard before returning
- Custom tokens: checks `qug_price <= 0.0` and `token_reserve == 0` before any arithmetic
- Never panics, never divides by zero

Polling loop (line 405):
```rust
let Some(price) = current_price else {
    debug!("Cannot get price for {} — skipping order {}", ...);
    continue;
};
```
Explicit `None → continue` guard. If price is missing for any reason, the order is skipped for this cycle. No unintended triggers possible.

**Verdict: ✅ Oracle zero-guard is sound. No changes needed.**

---

### Check 3: CF Registration — ✅ RESOLVED (key-prefix approach)

Using `CF_DCA_ORDERS` + `"limitorder:"` prefix eliminates this check entirely. No new CF, no DB init changes, no startup risk. Check 3 drops off the list.

### Check 4: AppState Init Sites — Deferred until Blockers A/B/C resolved

Once the 3 blockers are fixed, AppState needs `limit_order_storage: Some(Arc::new(LimitOrderStorage::new()))` in both init blocks (~lines 3147 and 4625). Compilation enforces completeness — if either is missed, the build fails.

---

## 4. Additional Finding: DCA Has Same TOCTOU Bug

The DCA execution loop (`dca_api.rs:904-940`) has the same snapshot-before-swap, status-after-swap pattern. This is already in production. Since the question was specifically about limit orders, DCA is out of scope for this PR — but the pattern should be noted for a follow-up hardening pass.

---

## 5. Revised Implementation Plan

### Step 1: Calendar (ship now — zero risk)
- Add `if event.shared { publish_calendar_event_p2p(&state, &event).await; }` in `calendar_api.rs:create_event()`
- Build, deploy to Epsilon, verify logs show calendar propagation
- Commit separately

### Step 2: Limit Orders — Fix 3 Blockers First

**Fix A — Storage layer (q-storage/src/lib.rs):**
- Add `save_limit_order()`, `delete_limit_order()`, `load_all_limit_orders()` using `CF_DCA_ORDERS` + `"limitorder:"` prefix
- No new CF, no DB init changes

**Fix B — Missing function (dca_api.rs):**
- Add `pub async fn execute_limit_swap(state, from_token, to_token, amount, wallet) -> Result<...>`
- Wraps existing `execute_dca_swap` with correct public signature and 5-param interface

**Fix C — TOCTOU in polling loop (limit_order_api.rs):**
- Add `LimitOrderStatus::Processing` variant
- Before swap: re-read from storage, check still Open, set Processing in storage via `put_sync()`
- After swap success: set Filled
- After swap failure: reset to Open (retriable) with failure count increment
- Add `failure_count: u32` field to `LimitOrder` struct; after 5 failures → auto-cancel

### Step 3: Wire Routes
After all fixes compile and `cargo check` passes:
- Add `pub mod limit_order_api;` to `lib.rs`
- Add `limit_order_storage` field to AppState + both init sites
- Register `.nest("/api/v1/dex/limit-orders", ...)` in main.rs
- Spawn `limit_order_check_loop` in main.rs background task block

### Step 4: Canary
- 24h soak on Alpha Docker
- Place 3 test orders: one Above trigger, one Below trigger, one cancel-before-fill
- Verify: triggered orders execute once and only once; cancelled orders do not execute; crash-recovery (kill -9, restart) does not double-execute

---

## 6. Updated Verdict

| Feature | Verdict | Blocker |
|---------|---------|---------|
| Calendar shared-event P2P | ✅ Ship now | None |
| Limit orders — storage | 🔴 Fix first | `save/load/delete_limit_order` missing |
| Limit orders — swap function | 🔴 Fix first | `execute_limit_swap` doesn't exist |
| Limit orders — TOCTOU | 🔴 Fix first | Status updated after swap, no re-read guard |
| Limit orders — routes | ⏳ After above | Compile blocked until fixes land |
| Limit orders — canary | ⏳ After routes | 24h Alpha soak required |
