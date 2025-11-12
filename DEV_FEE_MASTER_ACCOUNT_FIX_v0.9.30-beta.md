# Dev Fee Master Account Fix - v0.9.30-beta

**Date**: 2025-11-06
**Status**: ✅ **Build in Progress**
**Version**: v0.9.30-beta

---

## 🎯 Problem Summary

The **1% development fee** from mining rewards was going to a placeholder wallet instead of the **Quillon Bank Master Account**. This fix redirects all dev fees to the correct master account.

---

## 🔧 Fix Applied

### File Modified: `crates/q-storage/src/balance_consensus.rs` (Line 46)

**BEFORE:**
```rust
/// Founder wallet address (receives 1% dev fee)
pub const FOUNDER_WALLET: &str = "qnk8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f1a2b3c4d5e6f7a8b9c0d1e2f3a";
```

**AFTER:**
```rust
/// Founder wallet address (receives 1% dev fee) - Quillon Bank Master Account
pub const FOUNDER_WALLET: &str = "qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723";
```

---

## 💰 Impact

### Before Fix:
- **Dev Fee Recipient**: `qnk8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f1a2b3c4d5e6f7a8b9c0d1e2f3a` (placeholder)
- **Master Account Balance**: 0 QUG (no dev fees received)

### After Fix:
- **Dev Fee Recipient**: `qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723` (master account)
- **Expected Result**: 1% of all mining rewards will now credit to master account

---

## 📋 How Dev Fee Works

### Mining Reward Distribution:
```
Mining Reward = 50 QUG per block

Miner receives:    49.5 QUG (99%)
Dev Fee receives:   0.5 QUG ( 1%)
Total:             50.0 QUG
```

### Code Implementation (`crates/q-storage/src/balance_consensus.rs`):
```rust
/// Development fee percentage (1%)
pub const DEV_FEE_PERCENT: f64 = 0.01;

/// Founder wallet address (receives 1% dev fee) - Quillon Bank Master Account
pub const FOUNDER_WALLET: &str = "qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723";

// In apply_mining_reward():
let dev_fee = (reward as f64 * DEV_FEE_PERCENT) as u64;
let miner_reward = reward - dev_fee;

updates.push(BalanceUpdate {
    address: FOUNDER_WALLET.to_string(),
    amount: dev_fee,
    reason: ChangeReason::DevelopmentFee,
    block_height,
    solution_index: Some(solution_index),
});
```

---

## 🧪 Verification Steps

### 1. Check Master Account Balance After Deployment:
```bash
curl -X POST http://185.182.185.227:8080/balance \
  -H "Content-Type: application/json" \
  -d '{"address":"qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723"}'
```

Expected: Balance should increase by 0.5 QUG per mined block.

### 2. Monitor SSE Broadcasts:
```bash
curl http://185.182.185.227:8080/events
```

Look for dev fee updates:
```json
{
  "type": "balance_update",
  "address": "qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723",
  "amount": 500000000,  // 0.5 QUG in base units
  "reason": "DevelopmentFee",
  "block_height": 12345
}
```

### 3. Database Query:
```bash
# Check consensus balances directly
grep "qnkefca1e8c1" /opt/orobit/shared/q-narwhalknight/data/q-narwhal-db/balances/*
```

---

## 🚀 Deployment Plan

### Build Status:
- ✅ Fix applied to `balance_consensus.rs`
- 🔄 Compilation in progress (`v0.9.30-beta-DEV-FEE-FIX.log`)
- ⏳ Expected completion: ~7 minutes

### After Build Completes:

1. **Copy Binary to Downloads**:
   ```bash
   cp target/release/q-api-server \
      /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/downloads/q-api-server-v0.9.30-beta

   # Update latest symlink
   cp target/release/q-api-server \
      /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/downloads/q-api-server-linux-x86_64
   ```

2. **Restart Service**:
   ```bash
   systemctl stop q-api-server
   cp target/release/q-api-server /opt/orobit/shared/q-narwhalknight/target/release/q-api-server
   systemctl start q-api-server
   systemctl status q-api-server
   ```

3. **Verify Logs**:
   ```bash
   journalctl -u q-api-server -f
   ```

   Look for:
   ```
   ✅ Dev fee applied: 0.5 QUG to qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723
   ```

---

## 🔍 Clarifications from Investigation

### Mystery Wallet `qnk65085b6858d87`:
- **NOT a hack** ✅
- **NOT the founder wallet** ✅
- **IS a legitimate miner** ✅
- Currently has **4318.48 QUG** from mining rewards
- This wallet will continue to receive 99% of blocks it mines

### Master Account `qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723`:
- **IS the Quillon Bank master account** ✅
- Currently has **0 QUG** (expected before this fix) ✅
- **Will receive 1% dev fees** after v0.9.30-beta deployment ✅

---

## 📊 Expected Results

### After Next Mining Reward (Block Height 12346):

**If `qnk65085b6858d87` mines a block:**
```
Miner (qnk65085b6858d87):     4368.98 QUG (+49.5)
Dev Fee (master account):        0.50 QUG (+0.5)
```

**If another miner mines a block:**
```
Other miner:                    49.5 QUG
Dev Fee (master account):        0.5 QUG
```

### After 1000 Blocks:
```
Master Account Balance: 500 QUG (1000 blocks × 0.5 QUG/block)
```

---

## 🧑‍💻 Technical Details

### Balance Storage Format:
- **Full Address**: `qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723` (64 hex chars)
- **Storage Key**: `efca1e8c1f46e913` (first 16 hex chars = first 8 bytes)
- **RocksDB Column Family**: `balances`

### Code Path for Dev Fee Application:
1. **Block Producer** (`q-narwhal-core/src/block_producer.rs`):
   - Validates mining solution
   - Calls `apply_mining_reward()`

2. **Balance Consensus** (`q-storage/src/balance_consensus.rs`):
   - Splits reward: 99% miner, 1% dev fee
   - Creates `BalanceUpdate` for `FOUNDER_WALLET`
   - Writes to RocksDB `balances` column family

3. **SSE Broadcast** (`q-api-server/src/handlers.rs`):
   - Sends `balance_update` event to all connected clients
   - Frontend updates UI in real-time

---

## 📝 Commit Message

```
fix(dev-fee): Redirect 1% dev fee to Quillon Bank master account

- Update FOUNDER_WALLET constant in balance_consensus.rs
- From: placeholder address
- To: qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723
- Dev fees now credit to master account (0.5 QUG per block)

Investigation findings:
- Wallet qnk65085b6858d87 is a legitimate miner (NOT a hack)
- Master account previously had 0 balance (expected)
- All dev fees will now flow to correct master account

Version: v0.9.30-beta

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## 🎯 Success Criteria

- ✅ Compilation successful
- ✅ Binary deployed to downloads folder
- ✅ Service restarted without errors
- ✅ Master account receives 0.5 QUG on next mined block
- ✅ SSE broadcasts show dev fee updates
- ✅ Frontend displays correct master account balance

---

**Status**: 🔄 **Build in Progress** - Will update when compilation completes

**Build Log**: `/tmp/v0.9.30-beta-DEV-FEE-FIX.log`

**ETA**: ~7 minutes for full compilation

---

*Created: 2025-11-06*
*Session: Dev fee master account fix*
*Version: v0.9.30-beta*
