# Technical Review: QUGUSD Double-Credit Bug & SSE Notification Fix

**Version**: v10.2.9
**Date**: 2026-04-09
**Severity**: Critical (financial impact)
**Status**: Fixed and deployed

---

## 1. Executive Summary

A double-credit bug was discovered in the QUGUSD token transfer path on a blockchain with approximately $920M in total value. Sending 1 QUGUSD to a recipient resulted in the receiver being credited with 2 QUGUSD. The root cause was two independent code paths both modifying `token_balances` for the same transfer: one in `handlers.rs` during local consensus confirmation, and another in `balance_consensus.rs` during block processing. The fix removes the duplicate path in `handlers.rs`, making `balance_consensus.rs` the sole authority for QUGUSD balance changes.

A secondary bug was also identified: `TokenBalanceUpdated` SSE events were silently dropped by a catch-all filter in `streaming.rs`, preventing real-time frontend notifications for QUGUSD transfers.

---

## 2. Double-Credit Root Cause

Two code paths independently modified `token_balances` for the same QUGUSD transfer:

**Path 1 — `handlers.rs` (consensus confirmation loop, ~line 3079):**
When a block containing a QUGUSD transaction reached consensus confirmation, the handler's confirmation loop called `token_balances.insert()` directly on both the sender and receiver keys. This path applied balance changes immediately in the in-memory HashMap.

**Path 2 — `balance_consensus.rs` (block processing, ~line 415-427):**
When the same block was processed by `process_block_mining_rewards_tx()`, it called `subtract_token_balance()` and `add_token_balance()` via the storage engine. These are RocksDB-backed operations in the `token_balances` column family.

Both paths executed for every block containing QUGUSD transfers. Path 1 ran during local consensus confirmation in the main event loop; Path 2 ran during block processing in the balance consensus subsystem. The net effect: every QUGUSD transfer credited the receiver twice.

**Why QUG was unaffected:** Native QUG transfers use a separate `wallet_balances` HashMap (line 3099-3114 in `handlers.rs`). The balance consensus path for QUG also writes to `wallet_balances`, but the two paths are reconciled by the consensus watermark mechanism. QUGUSD, introduced in v10.2.0, used the `token_balances` map without this reconciliation, and the duplicate path was introduced at that time.

---

## 3. Fix Verification (Audit Results)

An exhaustive search of all QUGUSD balance modification paths identified 5 total locations where `token_balances` is written:

| # | Location | Operation | Status After Fix |
|---|----------|-----------|-----------------|
| 1 | `handlers.rs:3079-3092` | Consensus confirmation (was: insert) | **REMOVED** — now logs only, defers to block processing |
| 2 | `balance_consensus.rs:415-427` | `add_token_balance` / `subtract_token_balance` in `process_block_mining_rewards_tx` | **SOLE transfer handler** |
| 3 | `handlers.rs:11411-11414` | DEX swap QUGUSD debit | Separate operation (not a transfer) |
| 4 | `handlers.rs:11536-11539` | DEX swap QUGUSD credit | Separate operation (not a transfer) |
| 5 | `balance_consensus.rs:1116-1128` | Fast-sync token transfer replay | Separate code path (turbo sync only) |

After the fix, exactly one path credits QUGUSD for peer-to-peer transfers: `balance_consensus.rs` via `add_token_balance()` / `subtract_token_balance()`.

**Dedup mechanism:** The balance consensus subsystem maintains a `processed_blocks` LRU cache (100,000 entries, ~5MB) keyed by block hash. Before processing any block's transactions, the system checks this cache with an atomic read-then-write pattern. This prevents re-processing the same block on restart (when the LRU is empty, a persisted watermark height provides the fallback guard). The LRU check is TOCTOU-safe because it runs under a `RwLock` write guard.

**Concurrent double-processing is not possible** because `process_block_mining_rewards_tx` acquires the balance consensus write lock before modifying any balances, and the LRU insert happens within the same critical section.

---

## 4. SSE Notification Bug

**Problem:** `TokenBalanceUpdated` events were defined in the `StreamEvent` enum (streaming.rs:204) and had a corresponding event name mapping (`"token-balance-updated"`, streaming.rs:1286), but the SSE filter function that determines which events reach which connected client did not include `TokenBalanceUpdated` in its match arms. The event fell through to the catch-all:

```rust
// streaming.rs:849
_ => false,
```

This meant every `TokenBalanceUpdated` event was silently discarded. The frontend had listeners ready in `sseManager.ts`, `App.tsx`, and `Dashboard.tsx`, but they never received events.

**Fix:** Added `TokenBalanceUpdated` to the SSE filter with wallet address matching (streaming.rs:838-846), following the same pattern used by `BalanceUpdated`:

```rust
StreamEvent::TokenBalanceUpdated { ref wallet_address, .. } => {
    let normalized_event = if wallet_address.starts_with("qnk") {
        wallet_address[3..].to_string()
    } else {
        wallet_address.clone()
    };
    normalized_event == normalized_filter
}
```

**Additional fix:** A receiver-side optimistic SSE event was added in `handlers.rs`. Previously, only the sender received an immediate balance update event upon transaction submission. The receiver had to wait for block confirmation. Now both sender and receiver get optimistic events at submission time, with confirmed events following after block processing.

---

## 5. Complete QUGUSD Transfer Event Chain (After Fix)

1. User submits QUGUSD transfer via `POST /v1/transactions/send`
2. `handlers.rs` validates the transaction and emits an optimistic `TokenBalanceUpdated` SSE event for the **sender** (balance deducted)
3. `handlers.rs` emits an optimistic `TokenBalanceUpdated` SSE event for the **receiver** (balance credited) -- **NEW in v10.2.9**
4. Transaction is included in the next block by the block producer
5. `balance_consensus.rs` processes the block via `process_block_mining_rewards_tx()`, calling `subtract_token_balance()` and `add_token_balance()` against RocksDB
6. Block production loop emits confirmed `TokenBalanceUpdated` SSE events for both parties
7. `streaming.rs` SSE filter now passes `TokenBalanceUpdated` events to the matching wallet connection -- **FIXED in v10.2.9**
8. Frontend receives the event and updates Dashboard / TransactionScreen in real-time

---

## 6. Files Modified

| File | Change |
|------|--------|
| `crates/q-api-server/src/handlers.rs` | Removed duplicate QUGUSD `token_balances.insert()` in consensus confirmation loop (lines 3079-3092 replaced with log-only stub). Added receiver-side optimistic SSE event on transfer submission. |
| `crates/q-api-server/src/streaming.rs` | Added `TokenBalanceUpdated` match arm to the SSE wallet filter function (lines 838-846), preventing events from falling to the `_ => false` catch-all. |
| `gui/quantum-wallet/src/components/Dashboard.tsx` | Added `isValidBalance` cap for QUGUSD display amounts to guard against stale double-credited values in client cache. |
| `gui/quantum-wallet/src/components/TransactionScreenV2.tsx` | Added `isValidBalance` cap and balance source priority logic (confirmed SSE > optimistic > cached). |

---

## 7. Testing Recommendations

- **Transfer accuracy**: Send 1 QUGUSD between two wallets. Verify the receiver's balance increases by exactly 1 (not 2). Check both the in-memory state and the RocksDB persisted value.
- **SSE real-time update**: Open the receiver's wallet in a separate browser session before sending the transfer. Verify the balance updates in real-time without requiring a page refresh.
- **Server log single-entry**: After a QUGUSD transfer, check `journalctl -u q-api-server --since "1 minute ago" | grep "TOKEN TRANSFER"`. There should be exactly one `[TOKEN TRANSFER] Processed` log line per transfer (from `balance_consensus.rs`), not two.
- **Dedup on restart**: Restart the node and verify that the `processed_blocks` LRU rebuild does not re-process already-applied QUGUSD transfers. Check that balances remain stable across restarts.
- **DEX swap isolation**: Execute a QUGUSD DEX swap and verify the swap path (handlers.rs:11411-11539) still functions correctly — it was not modified and should remain unaffected.
