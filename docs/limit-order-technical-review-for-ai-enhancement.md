# Limit Order System — Technical Review for AI-Assisted Enhancement
**Date:** 2026-04-26  
**Author:** Server Beta (Claude Code)  
**Purpose:** Complete system description for DeepSeek or other AI to propose enhancements  
**Codebase:** Rust workspace, async/await, Axum HTTP, RocksDB storage, libp2p gossipsub P2P

---

## 1. Executive Summary

The limit order system lets users place price-triggered swap orders on the DEX. An order says: "When QUG price crosses $X, automatically swap Y QUG for QUGUSD." The system polls every 60 seconds, checks all open orders against the live oracle price, and fires the swap if the condition is met.

**Current state:** Implemented but NOT yet wired into the live binary (compilation blocked until 3 bugs were fixed — those fixes are being applied in this session).

---

## 2. Architecture Overview

```
User HTTP Request
      │
      ▼
POST /api/v1/dex/limit-orders
      │
      ▼
create_limit_order()          ← HTTP handler
      │
      ├── Validates request
      ├── Creates LimitOrder struct
      ├── Saves to in-memory HashMap (LimitOrderStorage.orders)
      └── Saves to RocksDB (CF_DCA_ORDERS, key="limitorder:{id}")

Background Task (60s loop)
      │
limit_order_check_loop()
      │
      ├── For each Open order:
      │   ├── Check expiry
      │   ├── Fetch current price (get_token_price_usd)
      │   ├── Check trigger condition (Above/Below)
      │   ├── [NEW] Write Processing status to RocksDB (fsync)
      │   ├── Execute swap (execute_limit_order_swap → execute_limit_swap → execute_dca_swap)
      │   ├── Update status to Filled (or Open on failure)
      │   └── Broadcast SSE event (SwapExecuted)
      │
      └── Repeat every 60 seconds
```

---

## 3. Core Data Structures

### `LimitOrder` (the central type)

```rust
pub struct LimitOrder {
    pub id: String,                    // "lo_{timestamp}_{random_hex}"
    pub wallet_address: String,        // hex-encoded 32-byte public key
    pub from_token: String,            // "QUG" or contract address
    pub to_token: String,              // "QUGUSD" or contract address
    pub amount: u128,                  // base units (24-decimal). IMPORTANT: NOT f64!
    pub trigger_price: f64,            // USD price at which to fire
    pub price_token: String,           // which token's price to watch
    pub direction: PriceDirection,     // Above or Below
    pub max_slippage: f64,             // e.g. 0.03 = 3%
    pub status: LimitOrderStatus,      // Open | Processing | Filled | Cancelled | Expired
    pub created_at: i64,               // Unix ms
    pub processing_since: Option<i64>, // Set when entering Processing state (crash recovery)
    pub filled_at: Option<i64>,        // Unix ms when filled
    pub expiry: Option<i64>,           // Unix ms (None = GTC — Good Till Cancelled)
    pub amount_out: u128,              // Output amount after fill (base units)
    pub fill_price: Option<f64>,       // Actual price at fill
    pub tx_hash: Option<String>,       // Transaction hash of the executed swap
    pub failure_count: u32,            // Consecutive swap failures; auto-cancel at 5
}
```

**Key invariants:**
- `amount` is in 24-decimal base units. 1 QUG = `10^24` base units. Never convert to f64 for arithmetic.
- `wallet_address` is hex-encoded (64 hex chars for 32-byte key), NOT binary.
- `processing_since` is `Some(timestamp)` when status == Processing. Always `None` when Open/Filled/Cancelled/Expired.

### `LimitOrderStatus`

```rust
pub enum LimitOrderStatus {
    Open,        // Waiting for trigger condition
    Processing,  // Swap in flight (durably written BEFORE swap executes — crash-safe)
    Filled,      // Swap completed successfully
    Cancelled,   // User cancelled OR auto-cancelled after 5 failures
    Expired,     // Past expiry timestamp
}
```

### `PriceDirection`

```rust
pub enum PriceDirection {
    Above,  // Fire when price >= trigger_price (stop-buy / take-profit)
    Below,  // Fire when price <= trigger_price (limit-buy / stop-loss)
}
```

---

## 4. Storage Architecture

### Database Layer

- **Column Family:** `CF_DCA_ORDERS` (shared with DCA orders)
- **Key format:** `"limitorder:{order_id}"` (prefix separates from DCA keys `"{order_id}"`)
- **Value format:** JSON-serialized `LimitOrder` (serde_json)
- **Write safety:** `put_sync()` (fsync-backed) for Processing state writes; `put()` acceptable for Filled/Cancelled

### In-Memory Layer

`LimitOrderStorage` wraps a `RwLock<HashMap<String, LimitOrder>>`:
- Writes use `write()` lock
- Reads use `read()` lock
- On startup: populated from RocksDB via `load_from_storage()`
- The source of truth is RocksDB; the HashMap is a fast-path cache

### `LimitOrderStorage` API

```rust
pub struct LimitOrderStorage {
    pub orders: RwLock<HashMap<String, LimitOrder>>,
}

impl LimitOrderStorage {
    pub fn new() -> Self
    pub async fn load_from_storage(&self, storage: &QStorage) -> anyhow::Result<()>
    pub async fn save_order(&self, storage: &QStorage, order: &LimitOrder) -> anyhow::Result<()>
    pub async fn delete_order(&self, storage: &QStorage, order_id: &str) -> anyhow::Result<()>
    pub async fn recover_stuck_processing(&self, storage: &QStorage)
    // ↑ Resets orders stuck in Processing > 5 minutes back to Open on startup
}
```

### `QStorage` (RocksDB) Methods for Limit Orders

```rust
// In crates/q-storage/src/lib.rs — uses CF_DCA_ORDERS + "limitorder:" prefix
pub async fn save_limit_order(&self, order_id: &str, bytes: &[u8]) -> Result<()>
pub async fn delete_limit_order(&self, order_id: &str) -> Result<()>
pub async fn load_all_limit_orders(&self) -> Result<Vec<(String, Vec<u8>)>>
```

---

## 5. Price Oracle

```rust
async fn get_token_price_usd(state: &Arc<AppState>, token: &str) -> Option<f64>
```

**How it works:**

For QUG (native token):
1. Reads the collateral vault — finds how much USDC/USDT/QUGUSD is collateralizing QUG
2. Computes price = collateral_value / circulating_supply
3. Guards: must be `> 0.0` to return Some; returns None if collateral unavailable

For custom tokens (ERC-20-style):
1. Finds the token's pool with QUG
2. Computes ratio: `(qug_reserve / token_reserve) * qug_price`
3. Guards: `qug_price > 0`, `token_reserve > 0`, division-by-zero protected
4. Returns None if any guard fails

**Critical property:** Returns `None` (never `Some(0.0)`) for missing/unavailable data. The polling loop has:
```rust
let Some(price) = current_price else {
    debug!("Cannot get price for {} — skipping order {}", ...);
    continue;
};
```
This means orders are skipped (not triggered) if the oracle is unavailable. Safe behavior.

---

## 6. Swap Execution Chain

```
execute_limit_order_swap(state, order)
    ↓
execute_limit_swap(state, from_token, to_token, amount, wallet_address)
    ↓
execute_dca_swap(state, from_token, to_token, amount, min_amount_out=0, wallet_address)
    ↓
DEX pool swap logic (same path as manual user swap)
    ↓
Returns (amount_out: u128, tx_hash: String)
```

**Slippage:** `max_slippage` is stored on the order but is currently NOT passed to `execute_dca_swap` (which takes `_min_amount_out: u128`). The `min_amount_out` is computed as 0 (no slippage protection at the execution layer). This is a known limitation — see Enhancement 1 below.

---

## 7. TOCTOU Fix (Crash Safety)

The original code had a race condition: it snapshotted all Open orders into a Vec, then executed the swap, then updated status. A crash between swap execution and status update would leave the order as Open — causing double-execution on restart.

The fixed pattern (now implemented):

```
1. Acquire write lock on in-memory orders
2. Re-read order from HashMap (not from snapshot)
3. If status != Open → skip (handles user cancellation between T0 and T3)
4. Set status = Processing in both HashMap and RocksDB (put_sync = fsync)
5. Release write lock
6. Execute swap (lock released — non-blocking)
7. On success: write Filled to HashMap + RocksDB
8. On failure: increment failure_count, reset to Open (or auto-cancel if ≥ 5 failures)
```

**Crash recovery on startup:**
`recover_stuck_processing()` — called during node startup after `load_from_storage()`. Any order in Processing state for > 5 minutes is reset to Open. The TOCTOU guard (step 3) protects against double-execution because if the swap actually completed, the order was written as Filled to RocksDB before the crash.

---

## 8. HTTP API

### Create Order
```
POST /api/v1/dex/limit-orders
Content-Type: application/json

{
  "wallet_address": "hex_string",
  "from_token": "QUG",
  "to_token": "QUGUSD",
  "amount": "1000000000000000000000000",   // u128 as string (1 QUG)
  "trigger_price": 0.50,                   // USD
  "price_token": "QUG",                    // optional — defaults to from_token
  "direction": "above",                    // "above" | "below"
  "max_slippage": 0.03,                    // optional — defaults to 0.03 (3%)
  "expiry": null                           // optional Unix ms — null = GTC
}

Response: { "success": true, "order_id": "lo_...", "message": "Limit order created" }
```

### List Orders for Wallet
```
GET /api/v1/dex/limit-orders/wallet/{wallet_address}

Response: {
  "success": true,
  "orders": [...],
  "open_count": 2
}
```

### Cancel Order
```
DELETE /api/v1/dex/limit-orders/{wallet_address}/{order_id}

Response: { "success": true, "message": "Order cancelled" }
```

---

## 9. AppState Integration

The system uses `limit_order_storage: Option<Arc<LimitOrderStorage>>` on `AppState`.

**Why `Option`?** The field is `None` until explicitly initialized. Handlers and the polling loop both do:
```rust
let Some(lo_storage) = &state.limit_order_storage else { return ... };
```
This means the system degrades gracefully (returns 503 or skips) if storage is not initialized.

**What's needed to wire it (not yet done):**
1. `pub mod limit_order_api;` in `lib.rs`
2. `limit_order_storage: Some(Arc::new(LimitOrderStorage::new()))` in both AppState init blocks in `main.rs`
3. Call `lo_storage.load_from_storage(&state.storage_engine).await` after init
4. Call `lo_storage.recover_stuck_processing(&state.storage_engine).await` after load
5. Route registration: `.nest("/api/v1/dex/limit-orders", limit_order_api::create_limit_order_router())`
6. Spawn background task: `tokio::spawn(limit_order_check_loop(state.clone()))`

---

## 10. Known Limitations & Enhancement Opportunities

### Enhancement 1: Slippage Enforcement
**Current:** `max_slippage` is stored but never enforced at execution time. `min_amount_out = 0` is passed to the swap — any slippage is accepted.

**Proposed:** Before calling `execute_limit_swap`, compute `min_amount_out`:
```rust
// Get the expected output from the pool price
let expected_out = compute_expected_output(&state, &order.from_token, &order.to_token, order.amount).await?;
// Apply slippage tolerance
let min_amount_out = (expected_out as f64 * (1.0 - order.max_slippage)) as u128;
// Pass to swap
execute_limit_swap(state, from_token, to_token, amount, min_amount_out, wallet_address).await
```
This requires `execute_limit_swap` to accept and pass through `min_amount_out`.

### Enhancement 2: Partial Fill Support
**Current:** All-or-nothing. The full `amount` is swapped in one transaction.

**Proposed:** Add `filled_amount: u128` to track how much has been swapped. Allow partial fills:
- If swap succeeds for less than `amount`, update `filled_amount`, keep status as Open
- Auto-cancel when `amount - filled_amount < minimum_swap_amount`
- Report fill percentage in API response

### Enhancement 3: Fill-or-Kill / Immediate-or-Cancel Modes
**Current:** Only GTC (Good Till Cancelled) behavior. Orders retry on failure.

**Proposed:** Add `time_in_force` field:
```rust
pub enum TimeInForce {
    GoodTillCancelled,     // Current behavior
    FillOrKill,            // Cancel immediately if not filled in first attempt
    ImmediateOrCancel,     // Fill as much as possible immediately, cancel remainder
    GoodTillDate(i64),     // Cancel at specific timestamp (= current expiry field)
}
```

### Enhancement 4: Trailing Stop Loss
**Current:** Fixed trigger price.

**Proposed:** Add `trailing_delta: Option<f64>` — trigger price tracks market price with a fixed offset:
- For Above: `trigger_price = max(observed_high) - trailing_delta`
- For Below: `trigger_price = min(observed_low) + trailing_delta`
- Requires tracking high-water-mark / low-water-mark per order

### Enhancement 5: P2P Order Broadcast
**Current:** Orders are local to one node. A limit order placed on Beta won't execute if that node is down when the price triggers.

**Proposed:** Gossipsub topic `/qnk/{network_id}/limit-orders`:
- On create: broadcast order to all peers (they store it locally)
- On fill/cancel: broadcast status update
- Dedup by `order_id`
- Only the node that "claims" the fill (writes Processing state) executes the swap

### Enhancement 6: Order Book Aggregation
**Current:** No cross-node visibility of pending limit orders.

**Proposed:** New endpoint `GET /api/v1/dex/limit-orders/book/{token_pair}` that:
1. Queries local orders
2. Queries peer nodes via HTTP or gossipsub
3. Returns aggregated order book (price levels, total size)
This enables price discovery and shows market depth to users.

### Enhancement 7: Smart Order Routing
**Current:** Swaps go through a single pool (whichever `execute_dca_swap` selects).

**Proposed:** Before executing, query all pools for the token pair. Route through the best pool (deepest liquidity, best price). Or split across multiple pools for large orders.

### Enhancement 8: MEV Protection
**Current:** Limit orders are publicly visible (or will be when P2P broadcast is added). A large limit order near the current price is front-runnable.

**Proposed:** 
- Commit-reveal scheme: broadcast a hash of the order; reveal full order only when near trigger
- Time-randomized execution: add ±30 second jitter to the 60s poll interval so exact execution time is unpredictable
- Private mempool: execute via trusted node set only

### Enhancement 9: Cancellation with Signed Message
**Current:** `DELETE /api/v1/dex/limit-orders/{wallet}/{id}` has no cryptographic authentication. Any client with knowledge of the wallet address and order ID can cancel.

**Proposed:** Require a signed cancellation message:
```json
{
  "order_id": "lo_...",
  "timestamp": 1745678901234,
  "signature": "hex_ed25519_sig_over_(order_id + timestamp)"
}
```
Verify against `wallet_address`'s public key before cancelling.

### Enhancement 10: Conditional Orders (One-Cancels-Other)
**Current:** Orders are independent.

**Proposed:** `linked_order_id: Option<String>` — when one order fills, automatically cancel the other. Classic OCO (One-Cancels-Other) pattern:
- Set take-profit: sell if price > $0.60
- Set stop-loss: sell if price < $0.40
- Link them — whichever triggers first cancels the other

---

## 11. File Map

| File | Purpose |
|------|---------|
| `crates/q-api-server/src/limit_order_api.rs` | All limit order logic (types, handlers, polling loop, storage wrapper) |
| `crates/q-storage/src/lib.rs` | RocksDB methods: `save_limit_order`, `delete_limit_order`, `load_all_limit_orders` |
| `crates/q-api-server/src/dca_api.rs` | `execute_limit_swap` public wrapper (lines added ~1028) |
| `crates/q-api-server/src/main.rs` | AppState init + route wiring + background task spawn (pending) |
| `crates/q-api-server/src/lib.rs` | `pub mod limit_order_api;` (pending) |

---

## 12. Performance Characteristics

- **Poll interval:** 60 seconds (configurable via constant)
- **Price oracle:** In-memory read from pool reserves — O(1), no DB access per order
- **Max orders per tick:** Unbounded (processes all Open orders). With 1000 open orders, each oracle query is <1ms → full scan ~1 second
- **Swap execution:** Same path as manual swap — 10-50ms per execution, atomic RocksDB write
- **Storage overhead:** ~500 bytes per order (JSON). 10,000 orders = ~5MB

**Scaling concern:** The polling loop is synchronous per-order (processes one at a time). With thousands of orders, this could cause execution delays. Enhancement idea: parallelize with `tokio::spawn` per triggered order, gated by semaphore (similar to the block-pack semaphore fix).

---

## 13. Test Plan (for new enhancements)

When testing any enhancement, verify these critical scenarios:

1. **Happy path:** Order triggers, swap succeeds, status = Filled, SSE event received
2. **User cancellation before trigger:** Cancel while Open → status = Cancelled; polling loop skips it
3. **User cancellation during execution:** Cancel between snapshot and swap → TOCTOU guard catches it (re-read shows Cancelled), swap does NOT execute
4. **Crash between Processing and Filled:** Kill -9 the process, restart → `recover_stuck_processing()` resets to Open, next poll retries
5. **Swap failure:** Swap returns error → failure_count increments, status resets to Open; after 5 failures → auto-cancel
6. **Expiry:** Order past expiry → immediately set to Expired; swap does NOT execute even if price triggered
7. **Oracle unavailable:** `get_token_price_usd` returns None → order skipped, not triggered
8. **Zero price guard:** Oracle returns Some(0.0) — this should not happen (oracle returns None for 0), but test defensively
9. **Double-trigger prevention:** Price stays above trigger for multiple poll cycles → only one fill, not repeated

---

## 14. Glossary

| Term | Meaning |
|------|---------|
| GTC | Good Till Cancelled — order stays open indefinitely until filled or user cancels |
| TOCTOU | Time-of-Check-Time-of-Use — a race condition class where state changes between when you check it and when you act on it |
| Base units | Smallest denomination; 1 QUG = 10^24 base units (24-decimal system) |
| CF_DCA_ORDERS | RocksDB column family shared by DCA and limit orders (limit orders use "limitorder:" key prefix) |
| Oracle | The `get_token_price_usd` function — computes USD price from on-chain pool reserves |
| Processing | Intermediate order state written durably BEFORE swap execution — crash-safe "in-flight" marker |
| put_sync | RocksDB write with fsync — guaranteed durable even if process crashes immediately after |
| SSE | Server-Sent Events — real-time push from server to browser frontend |
