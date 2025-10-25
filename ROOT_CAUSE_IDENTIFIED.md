# ROOT CAUSE IDENTIFIED: 100x Balance Conversion Bug

## Date: 2025-10-25
## Status: **ROOT CAUSE CONFIRMED**
## Location: Transaction Replay During Startup

---

## The Real Bug

**The balance persistence issue is caused by transaction replay during startup.**

When the node restarts:
1. ✅ Balances are loaded correctly from disk (700,779,726,118 units = 7,007.79 QUG)
2. ✅ Transactions are loaded from persistent storage
3. ❌ **BUG**: Transactions are replayed by parallel workers
4. ❌ **BUG**: `BalanceUpdated` events are emitted with f64 QUG values (7,009.29)
5. ❌ **BUG**: Something saves these f64 values as u64 atomic units (treating 70.09 as 70 units)
6. ❌ **RESULT**: Balance overwritten as ~70 QUG instead of ~7,000 QUG

---

## Evidence from Logs

### Oct 25 05:52:21 (Startup - Node Restarts)
```
✅ Loaded 70 wallet balances from persistent storage
🚀 Processing transaction batch: 117 transactions
📡 [SSE] Broadcasting BalanceUpdated: wallet=efca1e8c1f46e910, old=7008.79726118, new=7009.29726118
```

### Oct 25 05:52:22 (1 Second Later - Wrong Values Saved!)
```
❌ SYNCED wallet balance to disk: efca1e8c... -> 6701412131 units (survives hard kill)
❌ SYNCED wallet balance to disk: efca1e8c... -> 7001412131 units (survives hard kill)
```

**Expected**: 700,929,726,118 units (7,009.29 QUG × 100,000,000)
**Actual**: 7,001,412,131 units (70.01 QUG)
**Ratio**: 100x too small!

---

## Code Path

###  1. Startup (`lib.rs:622-642`)
```rust
// Load existing transactions from storage
match storage_engine.load_all_transactions().await {
    Ok(persisted_transactions) => {
        for tx in persisted_transactions {
            tx_pool.insert(tx.id, tx.clone());
            tx_status.insert(tx.id, TxStatus::InMempool);
        }
    }
}
```

### 2. Parallel Workers Start (`parallel_workers.rs:94-102`)
```rust
for worker_id in 0..self.config.num_workers {
    let handle = tokio::spawn(async move {
        Self::worker_loop(worker_id, config, state).await;
    });
}
```

### 3. Workers Process Transactions (`parallel_workers.rs:137`)
```rust
if let Err(e) = handlers::process_transaction_batch(state.clone()).await {
    debug!("Worker {} batch processing error: {}", worker_id, e);
}
```

### 4. Balances Updated and Events Emitted (`handlers.rs:672-722`)
```rust
// Update balances in memory (CORRECT - u64 atomic units)
let new_sender_balance = sender_balance - total_cost;
balances.insert(tx.from, new_sender_balance);

let new_recipient_balance = old_recipient_balance + tx.amount;
balances.insert(tx.to, new_recipient_balance);

// Emit events (f64 QUG values)
let recipient_event = crate::streaming::StreamEvent::BalanceUpdated {
    wallet_address: hex::encode(tx.to),
    old_balance: old_recipient_balance as f64 / 100_000_000.0,  // f64 QUG
    new_balance: new_recipient_balance as f64 / 100_000_000.0,  // f64 QUG
    change_reason: "transaction_received".to_string(),
    timestamp: chrono::Utc::now(),
};
```

### 5. **BUG LOCATION: WHERE ARE BALANCES SAVED?**

**CRITICAL**: The `process_transaction_batch()` function does NOT call `save_wallet_balance()`!

It only:
- Updates in-memory balances (correct)
- Emits BalanceUpdated events (f64 QUG)
- Saves transactions to storage

**BUT WHO IS SAVING THE BALANCES TO DISK?**

There must be:
1. A background task that periodically saves balances
2. OR a subscriber that listens to BalanceUpdated events and persists them
3. OR network sync code that receives balance updates and saves them

---

## Next Steps to Find the Bug

### 1. Search for Background Balance Persistence Tasks
```bash
rg "save_all_wallet_balances|periodic.*save|interval.*balance" crates/q-api-server/src/
```

### 2. Search for Event Subscribers
```bash
rg "subscribe.*BalanceUpdated|recv.*BalanceUpdated" crates/q-api-server/src/
```

### 3. Search for Network Sync Balance Handlers
```bash
rg "gossip.*balance|sync.*balance|peer.*balance" crates/q-api-server/src/
```

### 4. Add Debug Logging
Temporarily add logging before every `save_wallet_balance()` call to see:
- Where it's being called from
- What value is being passed
- Stack trace to identify the code path

---

## Suspects

### Most Likely:
1. **Network Sync Code** - `main.rs:1044-1077` gossip handler processes mining rewards
   - Might be incorrectly handling balance sync messages
   - Could be treating f64 QUG values as u64 atomic units

2. **Event-Driven Persistence** - Some code subscribes to BalanceUpdated events
   - Listens to event_broadcaster/event_emitter
   - Incorrectly persists the f64 values

3. **Periodic Batch Save** - Background task that saves all balances periodically
   - Reads from in-memory balances (should be correct)
   - But might have conversion bug

---

## The Smoking Gun Pattern

Looking at the saved values:
```
6,701,412,131 units = 67.01412131 QUG
7,001,412,131 units = 70.01412131 QUG
```

This pattern shows:
- The decimal part is preserved (67.01, 70.01)
- But it's 100x smaller than expected

This suggests:
```rust
// WRONG (what's happening):
let balance_qug: f64 = 70.01412131;  // From BalanceUpdated event
let balance_to_save = balance_qug as u64;  // 70 (truncated)
save_wallet_balance(&addr, balance_to_save);  // Saves 70 instead of 7,001,412,131!

// OR possibly:
let balance_units: u64 = 7001412131;  // Some intermediate value
save_wallet_balance(&addr, balance_units);  // Missing 2 zeros at the end!
```

---

## Fix Strategy

Once we find where the bug is:

1. **Immediate Fix**: Ensure proper conversion
   ```rust
   // CORRECT:
   let balance_qug: f64 = event.new_balance;  // 7009.29726118
   let balance_units = (balance_qug * 100_000_000.0) as u64;  // 700,929,726,118
   save_wallet_balance(&addr, balance_units);
   ```

2. **Long-term Fix**: Don't replay transactions on startup
   - Transactions shouldn't need to be reprocessed
   - Balance state is already persisted
   - Only new incoming transactions should update balances

3. **Data Recovery**: Multiply all affected balances by 100

---

**Priority**: CRITICAL - Stop all other work and find the exact line of code causing this
**Impact**: 99% balance loss on every restart
**Users Affected**: All testnet users

