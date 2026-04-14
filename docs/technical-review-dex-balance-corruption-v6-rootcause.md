# Technical Review v6: DEX Double Deduction — ROOT CAUSE FOUND

**Date:** 2026-04-14  
**Severity:** CRITICAL (user funds lost on every swap)  
**Network:** Q-NarwhalKnight mainnet-genesis ($1B market cap)  
**Root Cause:** CONFIRMED — every swap deducts QUG TWICE  
**Prepared for:** DeepSeek + ChatGPT peer review

---

## 1. Root Cause: Proven by Debug Logs

**Every DEX swap deducts the input amount TWICE from the sender's wallet.**

### Evidence (from Delta Docker test, captured 2026-04-14)

Test: swap 0.7 QUG → QUGUSD for wallet `1cb46f94...`

```
05:22:25  🔴 atomic_subtract_and_record_dex_debit(): 7.42 → 6.72 QUG  (FIRST deduction: -0.7)
05:22:25  💸 [SWAP v10.3.1] Deducted 0.7 QUG (RocksDB: 7.42 → 6.72)
05:22:29  🔴 subtract_balance(): 6.72 → 6.02 QUG  (SECOND deduction: -0.7 AGAIN)
```

**4 seconds apart. Same wallet. Same amount. Two different code paths.**

### Previous evidence (user's production swap)

The user swapped ~193 QUG (50% of 386 QUG balance):
- First deduction: 386 → 193 (handler)
- Second deduction: 193 → 0 (balance_consensus)
- User saw: 0 QUG (all gone)

This matches the user's report: *"after swapping 50% of my QUG, the whole amount was gone instantly."*

---

## 2. The Two Code Paths

### Path 1: Direct RocksDB Deduction (Swap Handler)

**File:** `crates/q-api-server/src/handlers.rs` line ~11531

```rust
// v10.3.1: ATOMIC subtract-balance + record-dex-debit
let rocks_new_balance = state.storage_engine
    .atomic_subtract_and_record_dex_debit(&wallet_hex, request.amount_in)
    .await?;
```

This runs IMMEDIATELY when the swap API is called. Deducts from RocksDB, records DEX counter, updates in-memory cache. The user sees the deduction instantly.

### Path 2: Consensus Transaction Processing (balance_consensus)

**File:** `crates/q-api-server/src/handlers.rs` line ~11060

```rust
// Step 5: Create the swap transaction with proper binary format
let swap_tx = transaction_utils::create_swap_transaction(
    wallet_addr, pool_id_bytes, request.amount_in, ...
);

// Step 6: Submit transaction to mempool for block inclusion
let submission_result = transaction_utils::submit_transaction(swap_tx, ...);
```

This creates a **consensus transaction** that gets included in the next block. When `balance_consensus` processes the block, it sees the Swap transaction and calls `subtract_balance()` on the sender — **deducting the same amount a second time.**

**File:** `crates/q-storage/src/balance_consensus.rs` line ~670

```rust
// Process transfers in the block (including Swap tx_type)
if !block_tx.is_coinbase() {
    // Subtract from sender
    storage.subtract_balance(&hex::encode(&block_tx.from), block_tx.amount).await?;
    // Add to receiver  
    storage.add_balance(&hex::encode(&block_tx.to), block_tx.amount).await?;
}
```

---

## 3. Why This Happens

The swap was designed in two phases that were never properly unified:

1. **Phase 1 (v3.6.8):** Direct RocksDB deduction for instant balance update ("CRITICAL FIX — Credit output token to user's balance IMMEDIATELY")
2. **Phase 2 (v2.4.0):** Consensus-verified swap transactions for cross-node agreement ("Instead of modifying local state directly, we submit a transaction to the mempool")

Both phases were implemented, but neither was removed when the other was added. The result: **both run for every swap.**

---

## 4. The Fix

### Option A: Remove the direct deduction (let consensus handle it)

Remove the `atomic_subtract_and_record_dex_debit` call from the swap handler. Let `balance_consensus` handle the deduction when the Swap transaction is processed from the block.

**Pro:** Clean architecture — one deduction path, consensus-verified  
**Con:** Balance update is delayed until the block is produced (1-3 seconds). User sees a brief window where their balance hasn't decreased yet.

### Option B: Remove the consensus transaction (keep direct deduction)

Remove `create_swap_transaction` / `submit_transaction` from the swap handler. The direct RocksDB deduction is the only deduction.

**Pro:** Instant balance update  
**Con:** Swap is not recorded on-chain. Other nodes don't see it. This is the current DEX architectural debt (off-chain mutations).

### Option C: Keep both, but make balance_consensus skip Swap transactions

Add a check in `balance_consensus` to skip `tx_type=Swap` transactions for balance processing, since they're already handled by the swap handler directly.

```rust
// In process_block_mining_rewards / process_block_transfers:
if block_tx.tx_type == TxType::Swap {
    // Skip — swap handler already deducted via atomic_subtract_and_record_dex_debit
    continue;
}
```

**Pro:** Minimal code change, preserves both the instant update and the on-chain record  
**Con:** The on-chain Swap transaction exists but is NOT processed for balances — could confuse auditors

### Recommended: Option C (safest for production)

Option C is the smallest change with the lowest risk. It preserves existing behavior:
- Swap handler deducts immediately (instant UX)
- Swap transaction is still recorded on-chain (auditability)
- balance_consensus skips Swap tx_type for balance processing (prevents double deduction)

---

## 5. Testing

```
Test 1: Swap with fix applied
  - Balance: 100 QUG
  - Swap 50 QUG → QUGUSD
  - Verify: balance = 50 QUG (not 0)
  - Wait 10 seconds (for block to be produced)
  - Verify: balance still 50 QUG (balance_consensus didn't deduct again)

Test 2: Verify Swap tx still recorded on-chain
  - After swap, check block transactions
  - Verify Swap tx exists in the block
  - Verify balance_consensus logged "skipping Swap tx"

Test 3: Multiple swaps
  - Swap 10, 20, 30 QUG in sequence
  - Verify total deduction = 60 (not 120)
  
Test 4: Reverse swap (QUGUSD → QUG)  
  - Verify no double CREDIT either
```

---

## 6. Impact Assessment

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| QUG deducted per swap | 2× amount_in | 1× amount_in |
| User sees after 50% swap | 0 QUG (all gone) | 50% remaining (correct) |
| On-chain record | Swap tx in block | Swap tx in block (unchanged) |
| Cross-node propagation | Swap tx propagated | Swap tx propagated (unchanged) |

---

## 7. Summary

The root cause was not restart, not pruning, not jemalloc, not authority sync. It was a simple **double deduction**: the swap handler deducts directly from RocksDB, AND submits a consensus transaction that balance_consensus also deducts. Every swap loses 2× the intended amount.

Found by adding `🔴 [BALANCE WRITE]` debug logging to all 16 write paths and watching a live test swap on Delta Docker.
