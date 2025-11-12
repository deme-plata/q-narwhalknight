# Swap Balance 10x Bug - Investigation Report

## Symptom
User balance appears as 10x LESS in DEX swap checks, causing swap failures.

## Evidence
```
❌ Swap failed: Insufficient QUG balance. Required: 76544496003, Available: 935656004
```

## Analysis

### Expected Values
- User should have: ~93.56 QUG
- In base units: 9,356,560,040 (93.56 × 100,000,000)

### Actual Values
- Available shows: 935,656,004 base units
- This equals: 9.35656004 QUG
- **This is exactly 10x less!**

### Calculation Proof
```
9,356,560,040 ÷ 10 = 935,656,004 ✓ EXACT MATCH
```

## Code Path Investigation

### 1. Database Storage (`q-storage/src/lib.rs`)
- `save_wallet_balance()` at line 1280: Stores `amount.to_le_bytes()` directly ✅ CORRECT
- `load_wallet_balance()` at line 1297: Reads `u64::from_le_bytes()` directly ✅ CORRECT
- `load_wallet_balances()` at line 1358: Same pattern ✅ CORRECT

### 2. Balance Loading in AppState (`q-api-server/src/lib.rs`)
- Line 798: `wallet_balances = persisted_balances` ✅ Direct assignment, no division

### 3. Swap Handler (`q-api-server/src/handlers.rs`)
- Line 4844: Reloads from RocksDB using `load_wallet_balances()`
- Line 4848: `wallet_balances_write.insert(addr, bal)` ✅ Direct insert
- Line 4860: `wallet_balances.get(&wallet_addr).copied().unwrap_or(0)` ✅ Direct retrieval

## Hypothesis: The Bug is in Balance WRITING, Not Reading

Since all the READ paths look correct, the bug must be during WRITE. Somewhere, when a balance is being saved, it's being divided by 10 before storage.

## Potential Bug Locations

### Theory 1: Mining Reward Processing
When mining rewards are credited, the balance might be divided by 10 before saving.

**Check**: `crates/q-api-server/src/main.rs` - Mining solution processing
**Check**: `crates/q-api-server/src/block_producer.rs` - Block production and rewards

### Theory 2: Transaction Processing
When processing transactions, sender/recipient balances might be divided by 10.

**Check**: `crates/q-api-server/src/handlers.rs` - Transaction submission

### Theory 3: Balance Consensus Engine
The Balance Consensus Engine might be dividing balances during consensus updates.

**Check**: `crates/q-storage/src/balance_consensus.rs` - Balance updates

## Next Steps

1. ✅ **Added diagnostic logging** to `execute_swap` handler to see:
   - Balance value from RocksDB when loaded
   - Balance value in HashMap before comparison

2. **Deploy diagnostic build** and test swap to see actual values in logs

3. **Search for "/" or "div" operations** on balances before `save_wallet_balance()` calls

4. **Check mining reward calculation** - most likely culprit

## Critical Files to Audit

- `crates/q-api-server/src/main.rs` (mining rewards)
- `crates/q-api-server/src/block_producer.rs` (block rewards)
- `crates/q-storage/src/balance_consensus.rs` (consensus updates)

## Diagnostic Build Status

Building with added logging at:
- `handlers.rs:4848` - Log balance when loading from RocksDB
- `handlers.rs:4863` - Log balance before swap check

This will show us the EXACT values being stored and retrieved.
