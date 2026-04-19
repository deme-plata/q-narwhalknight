# Technical Review: Critical DEX Swap Balance Corruption Bug

**Date:** 2026-04-13  
**Severity:** CRITICAL (user funds display lost to zero after DEX swap + node restart)  
**Network:** Q-NarwhalKnight mainnet-genesis ($1B market cap)  
**Status:** Root cause identified, fix required before any further DEX swaps  
**Prepared for:** DeepSeek peer review

---

## 1. What Happened

User swapped 50% of QUG balance for QUGUSD. After the swap:
- Expected: ~193 QUG remaining + ~532,340 QUGUSD
- Actual: ~0 QUG displayed (balance dropped to near zero within minutes)

**Timeline (Epsilon node, all times UTC+2):**

| Time | Event | Balance |
|------|-------|---------|
| 14:52 | Normal mining, balance accumulating | ~386 QUG (RocksDB) |
| 14:59:31 | Swap executed: QUG -> QUGUSD | RocksDB: 386.04 -> 193.04 QUG (correct deduction) |
| 14:59:31 | In-memory cache was stale | mem showed: 193.23 QUG (half of real) |
| 15:06:45 | `balance_consensus` reprocesses blocks | Balance OVERWRITTEN to ~1-10 QUG |
| 15:09+ | Continuing to mine, balance stuck at ~1-10 | Only new coinbase counted |

---

## 2. Root Cause: Three Interacting Bugs

### Bug 1: In-Memory Balance Cache Stale (Display Bug)

**File:** `crates/q-api-server/src/handlers.rs` lines 10269-10287

The `wallet_balances` HashMap (in-memory cache) is synced from RocksDB every 15 seconds. But mining rewards are credited to RocksDB directly by `balance_consensus` for every block. Between sync intervals, the in-memory cache falls behind.

```
RocksDB (truth):  386 QUG (accumulated from all coinbase since genesis)
In-memory (stale): 193 QUG (last sync was 15s ago, missed recent rewards)
Frontend displays: 193 QUG (reads in-memory, not RocksDB)
```

**Impact:** User sees half their real balance. When they request "50% swap", the frontend calculates 50% of the displayed (wrong) value. The server then processes the full requested amount against the real RocksDB balance.

### Bug 2: DEX Swap Deduction Is Off-Chain (Architecture Flaw)

**File:** `crates/q-api-server/src/handlers.rs` lines 11506-11561

DEX swaps deduct QUG via a direct RocksDB write (`subtract_balance`), NOT via an on-chain transaction that gets included in a block. This means:

```
Chain data (blocks):    Only knows about coinbase credits (mining rewards)
RocksDB (direct write): Knows about both coinbase AND DEX deductions
```

When `balance_consensus` recomputes balances from the chain, it sees only coinbase — the DEX deduction is invisible to it. The chain says the user earned X QUG from mining, so the "correct" chain-derived balance is X QUG (ignoring all DEX swaps).

### Bug 3: Balance Consensus Overwrites RocksDB After Restart (THE KILLER)

**File:** `crates/q-storage/src/balance_consensus.rs` lines 241-246 (watermark checks REMOVED in v10.2.1)

After the v10.3.0 restart of Epsilon:
1. The in-memory LRU dedup cache (500K entries) is empty
2. `balance_consensus` starts processing blocks from turbo sync
3. Without the watermark check (removed in v10.2.1), it recomputes balances from scratch
4. The recomputed balance only includes coinbase rewards — **the DEX swap deduction is lost**
5. Worse: the recomputation starts from ZERO for wallets whose blocks are still being synced, so the balance goes from 193 -> ~1-10 QUG

The v9.3.3 DEX debit counter (`dex_qug_debited:{wallet}`) was designed to fix this, but `apply_dex_qug_adjustments()` is **never called on normal restarts** — only during explicit balance rebuild migrations.

---

## 3. Why This Wasn't Caught Before

1. **The watermark removal (v10.2.1) created the vulnerability.** Before removal, the watermark prevented `balance_consensus` from overwriting already-processed blocks. It was removed because of a race condition with turbo sync — but no replacement was added.

2. **DEX swaps are rare on this network.** Most users mine and hold. The bug only manifests when: (a) a DEX swap happens, AND (b) the node restarts within the same session. Without both conditions, the balance looks correct.

3. **The DEX debit counter exists but is dormant.** `record_dex_qug_debit()` correctly records the deduction at swap time. But `apply_dex_qug_adjustments()` is never called automatically — it requires a manual admin trigger.

4. **The stale in-memory cache is usually close enough** (15s lag) that nobody notices. But during rapid mining (3.46 bps), the balance can diverge significantly within one sync interval.

---

## 4. The Three Fixes Required

### Fix 1: Balance API Must Read From RocksDB (Immediate)

**Every balance read visible to the user must come from RocksDB, not the in-memory cache.**

```rust
// BEFORE (stale cache):
let balance = wallet_balances.get(&wallet_addr).copied().unwrap_or(0);

// AFTER (authoritative):
let balance = state.storage_engine.get_balance(&hex::encode(wallet_addr)).await.unwrap_or(0);
```

**Files to change:**
- `handlers.rs` — Every endpoint that returns balance to the user
- SSE events — Balance updates pushed via SSE must read RocksDB
- Frontend — Must not cache balance locally; always use latest API response

### Fix 2: Auto-Apply DEX Adjustments After Every Restart (Critical)

**Call `apply_dex_qug_adjustments()` automatically during startup, AFTER the initial balance sync.**

```rust
// In main.rs, after balance_consensus initialization:
if let Ok(adjusted) = state.storage_engine.apply_dex_qug_adjustments().await {
    if adjusted > 0 {
        info!("🔄 [STARTUP] Applied {} DEX QUG adjustments after restart", adjusted);
    }
}
```

**File:** `crates/q-api-server/src/main.rs` — after the balance watermark is set (around line 3350)

### Fix 3: Persistent Dedup for Balance Consensus (P0-3 — Prevents Root Cause)

**Replace the in-memory LRU dedup with a persistent RocksDB column family.**

```rust
// New CF: "processed_balance_blocks"
// Key: block_hash (32 bytes)
// Value: empty (existence = processed)
// On process_block_*: check CF first, skip if exists, insert after processing
// Survives restarts. No watermark race. No LRU eviction.
```

**File:** `crates/q-storage/src/balance_consensus.rs`

This prevents `balance_consensus` from ever reprocessing a block after restart, eliminating the overwrite entirely.

---

## 5. Interaction Diagram (How The Bug Flows)

```
USER                    FRONTEND                   SERVER (Epsilon)
  |                        |                            |
  |  "Show balance"        |                            |
  |----------------------->|  GET /api/v1/balance       |
  |                        |--------------------------->|
  |                        |  Returns: 193 QUG          | (in-memory, STALE)
  |                        |<---------------------------|
  |  Sees: 193 QUG         |                            | (RocksDB has 386 QUG)
  |                        |                            |
  |  "Swap 50%"            |                            |
  |----------------------->|  POST swap: 96.5 QUG       |
  |                        |  (50% of displayed 193)    |
  |                        |--------------------------->|
  |                        |  ACTUAL: deduct 96.5       | WRONG: should deduct from
  |                        |  from RocksDB 386          | real balance, but amount_in
  |                        |  386 - 96.5 = 289.5        | was calculated from stale
  |                        |                            | display, not real balance
  |                        |                            |
  | ... 7 minutes later... |                            |
  |                        |                            | balance_consensus
  |                        |                            | reprocesses blocks
  |                        |                            | computes: only 1-10 QUG
  |                        |                            | from recent coinbase
  |                        |                            | OVERWRITES 289.5 -> 1-10
  |                        |                            |
  |  "Where's my QUG?!"   |                            |
  |----------------------->|  GET /api/v1/balance       |
  |                        |--------------------------->|
  |                        |  Returns: 1-10 QUG         | (corrupted)
  |                        |<---------------------------|
  |  Sees: ~0 QUG          |                            |
```

---

## 6. Immediate Remediation (Before Fix Is Deployed)

1. **Trigger admin balance rebuild on Epsilon:**
   ```
   POST /api/v1/admin/rebuild-balances (admin auth required)
   ```
   This recomputes ALL balances from full chain history, then needs `apply_dex_qug_adjustments()` to restore DEX deductions.

2. **Disable DEX swaps until fix is deployed:**
   The swap endpoint should return an error message until Fix 2 and Fix 3 are implemented.

3. **Verify other users' balances:**
   Check if any other wallet has a `dex_qug_debited:` counter in RocksDB but a chain-only balance (indicating the same corruption).

---

## 7. Why The DEX Debit Counter (v9.3.3) Doesn't Save Us

The counter works correctly:
```rust
// At swap time (correct):
storage_engine.record_dex_qug_debit(&wallet_hex, amount).await;
// Records: dex_qug_debited:efca1e8c... = 193000000000000000000000000 (193 QUG in base units)
```

But `apply_dex_qug_adjustments()` is only called during explicit migrations:
```rust
/// ONLY call this after a balance rebuild migration has run. On a normal restart
/// (no rebuild), subtract_balance()/add_balance() already maintain correct balances.
pub async fn apply_dex_qug_adjustments(&self) -> Result<u64> { ... }
```

The comment "on a normal restart, subtract_balance() already maintains correct balances" is **wrong**. After restart, `balance_consensus` reprocesses blocks and overwrites the subtract_balance'd value. The assumption that RocksDB values survive restart is incorrect when `balance_consensus` actively overwrites them.

---

## 8. The v10.2.1 Watermark Removal (Why It Was Removed)

**Original code (had watermark):**
```rust
fn process_block_mining_rewards(&self, block: &QBlock) -> Result<Vec<BalanceUpdate>> {
    if block.header.height <= self.balance_processed_watermark.load(...) {
        return Ok(vec![]); // Already processed — skip
    }
    // ... process block ...
}
```

**Why it was removed:** The turbo sync batch task updated the watermark to the network tip every 15 seconds. This caused locally-produced blocks (which had a height BELOW the synced tip) to be skipped — their transfers were never processed.

**What should have replaced it:** A per-block-hash check (e.g., RocksDB CF `processed_balance_blocks`), which is height-independent and doesn't race with sync.

---

## 9. Testing Requirements

```
1. Swap + restart test:
   - Start node, accumulate 100 QUG mining rewards
   - Execute DEX swap: sell 50 QUG for QUGUSD
   - Verify balance: 50 QUG remaining
   - Restart node (systemctl restart)
   - Wait 5 minutes for balance_consensus to run
   - Verify balance: STILL 50 QUG (not 100, not 0)

2. Multiple swap + restart:
   - Execute 3 swaps: sell 10, 20, 30 QUG
   - Restart node
   - Verify DEX debit counter: 60 QUG total
   - Verify balance: chain_balance - 60 QUG

3. Stale cache test:
   - Mine at 3.46 bps for 30 seconds
   - Read balance from API
   - Read balance from RocksDB directly
   - Verify they match (within 1 block reward tolerance)

4. Concurrent swap + mining:
   - Start mining (accumulating rewards)
   - Execute swap mid-mining
   - Verify no double-deduction, no lost deduction
```

---

## 10. Files Involved

| File | Role | Bug |
|------|------|-----|
| `crates/q-api-server/src/handlers.rs:10269` | Balance reload from RocksDB before swap | Only on swap, not on display |
| `crates/q-api-server/src/handlers.rs:11506` | QUG deduction via `subtract_balance` | Correct, but off-chain |
| `crates/q-api-server/src/handlers.rs:11558` | DEX debit counter recording | Correct, but never applied on restart |
| `crates/q-storage/src/balance_consensus.rs:241` | Watermark check (REMOVED) | Enables balance overwrite |
| `crates/q-storage/src/balance_consensus.rs:138` | LRU dedup cache (in-memory) | Lost on restart |
| `crates/q-storage/src/lib.rs:8667` | `apply_dex_qug_adjustments()` | Never called on restart |
| `crates/q-api-server/src/main.rs:3350` | Startup watermark init | Doesn't trigger DEX adjustment |

---

## 11. Classification

| Category | Assessment |
|----------|-----------|
| Funds at risk | Display only — chain has correct coinbase history |
| Data loss | DEX deduction lost from RocksDB after restart |
| Affected users | Any user who (a) did a DEX swap AND (b) node restarted after |
| Recovery | Admin rebuild + DEX adjustment restores correct balance |
| Prevention | Persistent dedup CF + auto-apply DEX adjustments on startup |

---

## 12. Recommendation

**Do NOT deploy any node restarts until Fix 2 (auto-apply DEX adjustments) is implemented.**

Every restart risks corrupting DEX swap balances for any user who has ever swapped. The fix is straightforward (add one function call to startup sequence + persistent dedup CF), but must be tested on Delta Docker before production.

Priority order:
1. **Fix 2 (auto-apply on startup)** — 1 hour, prevents future corruption
2. **Fix 1 (RocksDB reads for display)** — 2 hours, prevents stale display  
3. **Fix 3 (persistent dedup CF)** — 1-2 days, prevents root cause
4. **Admin rebuild + adjustment** — immediate, restores current user's balance
