# Balance Update Bug - Root Cause Found

## Problem Summary

**Symptoms**:
1. Receiver's wallet balance increases ✅
2. Sender's wallet balance does NOT decrease ❌
3. Recent activity doesn't show transactions ❌

## Root Cause: Address Format Mismatch

The system has **TWO different places** where it handles wallet addresses:

### 1. Transaction Balance Check (handlers.rs:908-911)
Checks **3 different address formats**:
```rust
let sender_balance = balances.get(&sender_address).copied()          // From Ed25519 signature
    .or_else(|| balances.get(&derived_address).copied())             // From public key
    .or_else(|| balances.get(&mnemonic_hash_address).copied())       // From mnemonic
    .unwrap_or(0);
```

### 2. Balance Update After Consensus (handlers.rs:552-563)
Only checks **1 address format**:
```rust
// Deduct from sender
let sender_balance = balances.get(&tx.from).copied().unwrap_or(0);  // ❌ Only checks tx.from!
let total_cost = tx.amount + tx.fee;

if sender_balance >= total_cost {
    balances.insert(tx.from, sender_balance - total_cost);  // Updates tx.from
    balances.insert(tx.to, old_recipient_balance + tx.amount);  // Updates tx.to
}
```

## Why This Causes the Bug

1. **Faucet** stores your balance under address format A (e.g., `mnemonic_hash_address`)
2. **Transaction validation** finds the balance because it checks all 3 formats (A, B, C)
3. **Transaction passes validation** ✅
4. **Consensus processes transaction**
5. **Balance update** tries to deduct from `tx.from` (address format B)
6. **Address format B has 0 balance** (your balance is under format A!)
7. **Condition fails**: `if sender_balance >= total_cost` is false (0 < 2.00001)
8. **Sender balance NOT deducted** ❌
9. **But recipient gets credited** because `balances.insert(tx.to, ...)` always works
10. **Result**: Money created from thin air! 🪄💰

## Why Recent Activity Doesn't Show

The transaction IS being created and stored, but the frontend might be:
1. Not authenticated properly to query recent transactions
2. Querying the wrong endpoint
3. Or the transactions aren't being returned because they don't match the wallet address filter

## The Fix

We need to make the balance update code (line 552) use the SAME logic as the balance check (lines 908-911):

```rust
// BEFORE (BROKEN):
let sender_balance = balances.get(&tx.from).copied().unwrap_or(0);

// AFTER (FIXED):
// Check all 3 address formats, same as balance validation
let sender_balance = balances.get(&tx.from).copied()
    .or_else(|| balances.get(&derived_address).copied())
    .or_else(|| balances.get(&mnemonic_hash_address).copied())
    .unwrap_or(0);

// Also need to update the SAME address format that has the balance:
let sender_address_key = if balances.contains_key(&tx.from) {
    tx.from
} else if balances.contains_key(&derived_address) {
    derived_address
} else {
    mnemonic_hash_address
};

// Now update using the correct address key:
balances.insert(sender_address_key, new_sender_balance);
```

## Additional Issues

### Issue 1: Multiple Address Formats
The root problem is that the system allows balances to be stored under multiple address formats. This creates confusion and bugs.

**Solution**: Normalize all addresses to ONE canonical format:
- Option A: Always use `derived_address` (from public key)
- Option B: Always use `mnemonic_hash_address` (from mnemonic)
- Option C: Add address normalization function that ALL endpoints use

### Issue 2: Faucet Address Format
The faucet endpoint (lines 2179-2202) uses different logic:
```rust
let requested_address = if hex_part.len() == 64 {
    hex::decode(hex_part)  // Full address
} else {
    // Hash the string - creates DIFFERENT address than mnemonic derivation!
    let mut hasher = Sha3_256::new();
    hasher.update(wallet_address.as_bytes());
    hasher.finalize().into()
}
```

If the wallet address string is less than 64 chars, it gets **hashed**, which produces a completely different address than the one derived from the mnemonic/public key.

## Recommended Solution Strategy

### Short-term Fix (Immediate)
1. Update `process_transaction_batch` balance update logic to check all 3 address formats
2. Find which format has the balance and update that one
3. Test thoroughly

### Long-term Fix (Proper)
1. **Choose ONE canonical address format** (e.g., Ed25519 public key derived address)
2. **Add address normalization function** that converts ANY address representation to canonical form
3. **Update ALL endpoints** to use normalized addresses:
   - Faucet
   - Balance query
   - Transaction creation
   - Transaction processing
   - Balance updates
4. **Migration script** to convert existing balances to canonical format

## Impact

This bug allows **money creation**:
- Sender keeps original balance (not deducted)
- Recipient gets new balance (credited)
- Total QUG in circulation increases with each transaction
- This is a critical vulnerability that breaks the economic model

## Files to Modify

### Immediate Fix:
1. `crates/q-api-server/src/handlers.rs:547-574` - Balance update after consensus

### Long-term Fix:
1. `crates/q-api-server/src/handlers.rs` - All balance operations
2. `crates/q-types/src/lib.rs` - Add address normalization
3. `crates/q-wallet/src/lib.rs` - Update wallet address derivation
4. Migration script for existing database

## Status

- ✅ Root cause identified
- ✅ Impact understood
- ⏳ Fix being implemented
- ⏳ Testing required

---

**Date**: 2025-10-15
**Severity**: CRITICAL - Money creation bug
**Priority**: IMMEDIATE FIX REQUIRED
