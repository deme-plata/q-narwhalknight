# Balance Detection Bug - Address Mismatch Investigation

## Problem Summary

User experiences balance detection bug where:
- **Balance API shows**: 10 QUG ✅ Correct
- **Transaction validation shows**: 0 QUG ❌ Wrong
- **Error message**: "Insufficient balance. Have: 0 QUG, Need: 2.00001 QUG"

## Root Cause Hypothesis

**Address mismatch between faucet storage and transaction validation**

The issue appears to be that:
1. **Faucet endpoint** stores balance under one address format (lines 2179-2202 in `handlers.rs`)
2. **Transaction endpoint** checks balance under different address formats (lines 903-911 in `handlers.rs`)

### Address Representations

The transaction endpoint checks **3 possible address representations**:

```rust
// handlers.rs:904-911
let sender_address = signed_transaction.from;  // From Ed25519 signature
let sender_balance = balances.get(&sender_address).copied()
    .or_else(|| balances.get(&derived_address).copied())  // From public key
    .or_else(|| balances.get(&mnemonic_hash_address).copied())  // From mnemonic
    .unwrap_or(0);
```

### Faucet Balance Storage

The faucet stores balance under:

```rust
// handlers.rs:2186-2202
let requested_address = if hex_part.len() == 64 {
    // Full 32-byte hex address
    hex::decode(hex_part)
} else {
    // Short address - hash the string
    use q_types::{Sha3_256, Digest};
    let mut hasher = Sha3_256::new();
    hasher.update(wallet_address.as_bytes());
    hasher.finalize().into()
}
```

**Problem**: If the wallet address is short (less than 64 hex chars), it's **hashed**. But the transaction endpoint derives addresses from the mnemonic/signature, which may not match the hashed short address.

## Diagnostic Logging Added

Added comprehensive logging to `handlers.rs` lines 906-925:

```rust
info!("🔍 BALANCE CHECK DIAGNOSTIC:");
info!("   sender_address (from signature): {}", hex::encode(sender_address));
info!("   derived_address (from public key): {}", hex::encode(derived_address));
info!("   mnemonic_hash_address (from mnemonic): {}", hex::encode(mnemonic_hash_address));

info!("   balance_from_sender: {:?} QUG", balance_from_sender);
info!("   balance_from_derived: {:?} QUG", balance_from_derived);
info!("   balance_from_mnemonic: {:?} QUG", balance_from_mnemonic);

info!("   📋 All wallet addresses in HashMap ({} total):", balances.len());
for (addr, bal) in balances.iter().take(10) {
    info!("      {} -> {} QUG", hex::encode(addr), *bal as f64 / 100_000_000.0);
}
```

## Testing Instructions

**User**: Please try sending 2 QNK again and copy the backend logs.

The new diagnostic logs will show:
1. **Three address formats** being checked
2. **Balance values** for each format (or None if not found)
3. **All wallet addresses** currently in the HashMap (first 10)

This will help us identify:
- Which address format the faucet stored the balance under
- Which address format the transaction is checking
- Whether they match or not

## Expected Output

When you try to send 2 QNK, you should see logs like:

```
🔍 BALANCE CHECK DIAGNOSTIC:
   sender_address (from signature): 9d1cf74ac9595e4a39a3e84416f264f243efef472f4825b89f76462d7e59b15c
   derived_address (from public key): [some address]
   mnemonic_hash_address (from mnemonic): [some address]
   balance_from_sender: None QUG
   balance_from_derived: None QUG
   balance_from_mnemonic: Some(10.0) QUG
   📋 All wallet addresses in HashMap (72 total):
      [address 1] -> X QUG
      [address 2] -> Y QUG
      ...
```

## Solution Strategy

Once we identify the mismatch, we'll need to:
1. **Fix the faucet** to store balance under the correct address format
2. **OR** fix the transaction validation to check the same format the faucet uses
3. **OR** add address normalization to ensure all endpoints use the same format

## STARK Question Answer

Regarding your question: **"on starks they claim millions of tps. but how is that pissible if one txn takes 2 seconds?"**

STARKs achieve millions of TPS through **batching and parallelization**:

### 1. Batch Proving (Main Technique)
- **Single tx**: 1 proof / 2 sec = 0.5 TPS ❌
- **Batched (1,000 txs)**: 1,000 txs / 2 sec = 500 TPS ✅
- **Batched (1,000,000 txs)**: 1,000,000 txs / 2 sec = 500,000 TPS ✅✅

Instead of proving each transaction individually, STARKs prove **1000+ transactions in a single proof** that still takes 2 seconds.

### 2. Parallel Provers
- **10 provers** × 500k TPS each = 5 million TPS
- **100 provers** × 500k TPS each = 50 million TPS

Multiple STARK provers run in parallel, each processing a batch.

### 3. Progressive Proving
```
Transaction execution:      Instant (microseconds)
Transaction finality:       2 seconds (proof generation)
Throughput (TPS):          Based on execution speed, not proving speed
```

Transactions execute and get included in blocks immediately. STARK proofs generate **asynchronously in the background** for finality.

**Analogy**: It's like a factory assembly line:
- **Assembly line speed**: 1 million products/hour (TPS = transaction execution)
- **Quality control testing**: 2 seconds per batch of 1000 products (STARK proving)
- **Throughput**: Limited by assembly line, not quality control

### Real-World STARK Systems

- **StarkNet**: ~100-300 TPS (current), targeting millions with future optimizations
- **StarkEx (for DEXs)**: 18,000 TPS proven in production
- **Polygon Miden**: Targeting 1M+ TPS with recursive STARKs

The "millions of TPS" claims are:
1. **Theoretical maximum** with optimal batching and parallelization
2. **Future target** with hardware acceleration (GPUs, ASICs)
3. **Marketing claims** that may be exaggerated

## Current Status

✅ **Backend rebuilt** with diagnostic logging (3.27s compilation)
✅ **Server running** on port 8080
⏳ **Awaiting test transaction** from user to analyze logs

---

**Next Step**: User sends 2 QNK transaction, we analyze the diagnostic output to identify address mismatch.

**Date**: 2025-10-15
**File**: `crates/q-api-server/src/handlers.rs:906-925`
