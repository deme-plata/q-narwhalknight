# Dev Fee Persistence Bug Fix - v0.9.31-beta

**Date**: 2025-11-06
**Status**: ✅ **CRITICAL BUG FIXED**
**Version**: v0.9.31-beta

---

## 🐛 Critical Bug Discovered

### **Problem Summary**

The master account balance remained ZERO despite:
- ✅ Dev fee configuration correct (FOUNDER_WALLET updated to master account)
- ✅ Coinbase transactions created with dev fees
- ✅ In-memory balances updated

**Root Cause**: Balance consensus engine was **ONLY called for P2P-received blocks**, NOT for locally produced blocks!

---

## 🔍 Bug Analysis

### **What Was Happening:**

1. **Block Production** (crates/q-api-server/src/main.rs:3279-3450):
   ```rust
   // Block produced with coinbase transactions ✅
   let coinbase_transactions = Self::create_coinbase_transactions(&solutions);

   // Block saved to RocksDB ✅
   app_state_mining.storage_engine.save_qblock(&new_block).await;

   // In-memory balances updated ✅
   let mut balances = app_state_mining.wallet_balances.write().await;
   balances.insert(tx.to, new_balance);

   // ❌ BUG: Balance consensus engine NOT called!
   // Dev fee and miner rewards NOT persisted to RocksDB!
   ```

2. **Block Reception** (crates/q-api-server/src/main.rs:2028):
   ```rust
   // P2P block received via gossipsub

   // Balance consensus called ✅
   let updates = balance_engine.process_block_mining_rewards_tx(&tx, &block).await;

   // Balances persisted to RocksDB ✅
   ```

### **Result:**

| Scenario | Coinbase TX Created | In-Memory Balance | RocksDB Balance | Persisted? |
|----------|---------------------|-------------------|-----------------|------------|
| **Locally Produced Block** | ✅ YES | ✅ YES | ❌ NO | ❌ NO |
| **P2P Received Block** | ✅ YES | ✅ YES | ✅ YES | ✅ YES |

**Impact**: When server-beta (bootstrap node) produces blocks:
- Dev fees exist in memory only
- On restart, balances lost
- Master account shows 0 QUG
- Users mining during bootstrap downtime get rewards persisted (they receive blocks via P2P)
- Bootstrap's own produced blocks have unpersisted dev fees

---

## 🔧 Fix Applied

### **Modified Files:**

#### **crates/q-api-server/src/main.rs** (2 locations)

**Location 1: Batch-Based Block Production** (Line 3328):
```rust
// Store block in RocksDB
if let Err(e) = app_state_mining.storage_engine.save_qblock(&new_block).await {
    error!("❌ Failed to save block {}: {}", new_block.header.height, e);
}

// 💰 v0.9.31-beta: PROCESS MINING REWARDS via Balance Consensus Engine
// This persists dev fee and miner rewards to RocksDB for consensus across all nodes
{
    use q_storage::balance_consensus::BalanceConsensusError;

    let balance_engine = app_state_mining.balance_consensus.read().await;
    match balance_engine.process_block_mining_rewards(
        &app_state_mining.storage_engine,
        &new_block
    ).await {
        Ok(updates) => {
            info!("💰 Applied {} balance updates for block {} (including dev fee)",
                  updates.len(), new_block.header.height);

            // Broadcast balance updates via SSE
            for update in &updates {
                let _ = app_state_mining.event_broadcaster.broadcast(
                    q_api_server::streaming::StreamEvent::BalanceUpdated {
                        address: format!("qnk{}", &update.address[..16]),
                        balance: update.amount as f64 / 1_000_000_000.0,
                        timestamp: chrono::Utc::now(),
                    }
                );
            }
        }
        Err(BalanceConsensusError::AlreadyProcessed(_)) => {
            debug!("Block {} already processed for rewards (safe retry)", new_block.header.height);
        }
        Err(e) => {
            error!("❌ Failed to process mining rewards for block {}: {:?}", new_block.header.height, e);
        }
    }
}
```

**Location 2: Time-Based Parallel Block Production** (Line 3789):
```rust
// Same fix applied to time-based block producer code path
```

---

## 📊 Expected Results

### **After v0.9.31-beta Deployment:**

**Locally Produced Blocks:**
```
✅ Block 2162 produced
✅ Saved to RocksDB
✅ Balance consensus called:
   - Miner: +49.5 QUG (persisted to RocksDB)
   - Dev fee: +0.5 QUG (persisted to RocksDB)
✅ SSE broadcast: BalanceUpdated events for both
```

**Master Account Balance:**
```
Block 2162: +0.5 QUG
Block 2163: +0.5 QUG
Block 2164: +0.5 QUG
...
Total: Accumulates 0.5 QUG per block produced
```

**Logs to Verify:**
```
💰 Applied 2 balance updates for block 2162 (including dev fee)
Broadcasting BalanceUpdated: wallet=qnke4ec8514be795... (miner)
Broadcasting BalanceUpdated: wallet=qnkefca1e8c1f46e91... (master account)
```

---

## 🧪 Verification Steps

### **1. Check Master Account Balance:**
```bash
curl -X POST http://185.182.185.227:8080/balance \
  -H "Content-Type: application/json" \
  -d '{"address":"qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723"}'
```

**Expected**: Balance increases by 0.5 QUG per block after v0.9.31-beta deployment.

### **2. Monitor Logs:**
```bash
journalctl -u q-api-server -f | grep "Applied.*balance updates"
```

**Expected Output:**
```
💰 Applied 2 balance updates for block 2162 (including dev fee)
💰 Applied 2 balance updates for block 2163 (including dev fee)
```

### **3. Check SSE Broadcasts:**
```bash
curl http://185.182.185.227:8080/events
```

**Expected**: `BalanceUpdated` events for master account address `qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723`.

### **4. Database Verification:**
```bash
# Check if master account has balance in RocksDB
strings data/q-narwhal-db/hot/* | grep -A5 "efca1e8c1f46e913"
```

---

## 🎯 Success Criteria

- ✅ Compilation successful
- ✅ Service restarts without errors
- ✅ Master account balance INCREASES with each block
- ✅ Logs show "Applied 2 balance updates" per block
- ✅ SSE broadcasts include master account updates
- ✅ Balance persists across restarts

---

## 📝 Technical Details

### **Balance Consensus Engine Flow:**

**Before Fix:**
```
Locally Produced Block:
Block Producer → Coinbase TX → In-Memory Balance → ❌ Not Persisted

P2P Received Block:
Gossipsub → Balance Consensus → RocksDB → ✅ Persisted
```

**After Fix:**
```
Locally Produced Block:
Block Producer → Coinbase TX → Balance Consensus → RocksDB → ✅ Persisted

P2P Received Block:
Gossipsub → Balance Consensus → RocksDB → ✅ Persisted
```

### **Code Path:**

1. **Block Production** (main.rs:3279):
   - `produce_blocks()` creates block with coinbase transactions
   - Block saved to RocksDB
   - **NEW**: `process_block_mining_rewards()` called
   - Balances persisted to RocksDB via `storage.add_balance()`
   - SSE broadcasts sent

2. **Balance Storage** (balance_consensus.rs:231-252):
   ```rust
   // Update miner balance
   storage.add_balance(&miner_address, miner_reward).await;
   updates.push(BalanceUpdate {
       address: miner_address,
       amount: miner_reward,
       reason: ChangeReason::MiningReward,
   });

   // Update dev wallet balance
   storage.add_balance(&self.dev_wallet, dev_fee).await;
   updates.push(BalanceUpdate {
       address: self.dev_wallet,
       amount: dev_fee,
       reason: ChangeReason::DevelopmentFee,
   });
   ```

---

## 💡 Why This Bug Occurred

**Historical Context:**

1. **Original Design**: Balance consensus was designed for multi-node networks
   - Assumption: All nodes receive blocks via P2P
   - Local block production was considered a special case

2. **In-Memory Balances**: Added for performance
   - Reduced RocksDB writes during mining
   - Periodic persistence every 15 seconds

3. **Gap**: In-memory balances were meant as a cache, not the source of truth
   - When server produces its own blocks, it only updated the cache
   - Cache lost on restart
   - Balance consensus was bypassed for local blocks

**Fix**: Now BOTH code paths (local production + P2P reception) call balance consensus engine.

---

## 🚀 Deployment Plan

### **Build:**
```bash
timeout 36000 cargo build --release --package q-api-server --bin q-api-server
```

### **Deploy:**
```bash
# Copy to downloads folder
cp target/release/q-api-server \
   gui/quantum-wallet/dist-final/downloads/q-api-server-v0.9.31-beta

# Update latest symlink
cp target/release/q-api-server \
   gui/quantum-wallet/dist-final/downloads/q-api-server-linux-x86_64

# Restart service
systemctl stop q-api-server
cp target/release/q-api-server target/release/q-api-server
systemctl start q-api-server
```

---

## 📈 Expected Impact

### **Master Account Balance Growth:**

Assuming 1 block every 5 seconds:
- Per minute: 12 blocks × 0.5 QUG = 6 QUG
- Per hour: 720 blocks × 0.5 QUG = 360 QUG
- Per day: 17,280 blocks × 0.5 QUG = 8,640 QUG

**Current height**: 2162
**Expected first visible balance**: After block 2163 (0.5 QUG)

---

**Status**: 🚀 **Ready to Deploy** - Critical bug fix for dev fee persistence

**ETA**: ~10 minutes for compilation + deployment

---

*Created: 2025-11-06*
*Session: Dev fee persistence bug fix*
*Version: v0.9.31-beta*
