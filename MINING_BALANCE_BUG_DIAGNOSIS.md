# Mining Balance Not Showing in Dashboard - Root Cause Analysis

**Date**: 2025-11-06
**Status**: 🔴 **CRITICAL BUG IDENTIFIED**

---

## 🔴 **PROBLEM**

Mining rewards are being earned and processed, but the dashboard shows **zero balance** for the mining wallet.

**Symptoms:**
- Miner is actively finding blocks (670+ blocks mined)
- Logs show "Processing balance consensus transactions"
- Dashboard frontend shows 0 QNK balance
- `/api/v1/wallet/{address}/balance` returns 0

---

## 🔍 **ROOT CAUSE**

### **TWO SEPARATE BALANCE SYSTEMS**

The codebase has **two independent balance tracking systems** that don't communicate:

1. **Balance Consensus Engine (RocksDB)**
   - Location: `crates/q-storage/src/balance_consensus.rs`
   - Writes to: `balances` column family in RocksDB
   - Function: `add_balance_tx()` at line 443-472
   - Used by: Mining reward processing

2. **In-Memory HashMap (`wallet_balances`)**
   - Location: `AppState.wallet_balances`
   - Type: `Arc<RwLock<HashMap<[u8; 32], u64>>>`
   - Read by: `/api/v1/wallet/{address}/balance` endpoint
   - Used by: Wallet API

### **The Disconnect:**

```rust
// Mining rewards are written HERE:
// crates/q-storage/src/balance_consensus.rs:469
tx.put("balances", address.as_bytes(), &new_balance.to_be_bytes()).await?;

// But the API reads from HERE:
// crates/q-api-server/src/handlers.rs:2684
let balances = state.wallet_balances.read().await;
balances.get(&address_bytes).copied().unwrap_or(0)
```

**Result**: Mining rewards accumulate in RocksDB but the wallet API reads from the empty in-memory HashMap!

---

## ✅ **THE FIX**

### **Option 1: Make Wallet API Read from Balance Consensus (RECOMMENDED)**

Update the wallet balance endpoint to read from the balance consensus engine instead of the in-memory HashMap.

**Changes Needed:**

1. **Add `balance_engine` to `AppState`**:
```rust
pub struct AppState {
    // ... existing fields ...
    pub balance_engine: Arc<BalanceConsensusEngine>,
}
```

2. **Update `get_wallet_balance()` handler**:
```rust
// OLD CODE (crates/q-api-server/src/handlers.rs:2683-2686)
let balance = {
    let balances = state.wallet_balances.read().await;
    balances.get(&address_bytes).copied().unwrap_or(0)
};

// NEW CODE (read from balance consensus)
let balance = {
    let address_hex = hex::encode(&address_bytes);
    match state.balance_engine.get_balance(&state.storage_engine, &address_hex).await {
        Ok(balance) => balance,
        Err(e) => {
            warn!("Failed to get balance for {}: {}", address_hex, e);
            0
        }
    }
};
```

3. **Update node status endpoint** (crates/q-api-server/src/handlers.rs:40):
```rust
// OLD CODE
let balance = {
    let balances = state.wallet_balances.read().await;
    balances.get(&wallet_address).copied().unwrap_or(0)
};

// NEW CODE
let balance = {
    let address_hex = hex::encode(&wallet_address);
    match state.balance_engine.get_balance(&state.storage_engine, &address_hex).await {
        Ok(balance) => balance,
        Err(_) => 0
    }
};
```

**Why This is Better:**
- ✅ Single source of truth (RocksDB via balance consensus)
- ✅ Mining rewards immediately visible
- ✅ Consistent across all nodes
- ✅ Survives restarts (persisted to disk)
- ✅ No sync lag or race conditions

---

### **Option 2: Sync In-Memory HashMap from Balance Consensus**

Update the periodic balance sync task to read from balance consensus and update the HashMap.

**Changes Needed:**

Update the periodic sync task (crates/q-api-server/src/main.rs:4661-4700):

```rust
// CURRENT: Syncs HashMap → RocksDB (wrong direction!)
match app_state_balance_sync.storage_engine.save_wallet_balances(&balances_snapshot).await {
    // ...
}

// NEW: Sync RocksDB → HashMap (correct direction!)
// Read all balances from balance consensus
let all_balances = balance_engine.get_all_balances(&storage_engine).await?;

// Update in-memory HashMap
let mut balances = app_state.wallet_balances.write().await;
for (address_hex, balance) in all_balances {
    if let Ok(bytes) = hex::decode(&address_hex) {
        if bytes.len() == 32 {
            let mut addr = [0u8; 32];
            addr.copy_from_slice(&bytes);
            balances.insert(addr, balance);
        }
    }
}
```

**Why This is Less Ideal:**
- ⚠️ Introduces sync lag (15-second delay)
- ⚠️ Two sources of truth
- ⚠️ Race conditions possible
- ⚠️ More complex

---

## 📊 **IMPACT**

**Affected Users:**
- All miners on Server Alpha
- All wallet queries via API
- All dashboard balance displays

**Data Loss:**
- ❌ **NO DATA LOSS** - Balances are safely stored in RocksDB
- ✅ Once fixed, all historical mining rewards will be visible

**Severity**: **HIGH**
- Mining rewards ARE being tracked
- They just don't show up in the UI
- Fix is straightforward

---

## 🚀 **IMPLEMENTATION PLAN**

### **Phase 1: Immediate Fix (Option 1)**

1. Add `balance_engine` to `AppState`
2. Update `get_wallet_balance()` to read from balance consensus
3. Update node status endpoint
4. Test with mining wallet
5. Deploy v0.9.27-beta

**ETA**: 30 minutes

### **Phase 2: Cleanup (Optional)**

1. Remove `wallet_balances` HashMap from `AppState`
2. Remove periodic sync task (no longer needed)
3. Update all balance-related code to use balance consensus
4. Clean up old balance storage code

**ETA**: 1 hour

---

## 🧪 **TESTING CHECKLIST**

After fix:
- [ ] Mining wallet shows correct balance in dashboard
- [ ] `/api/v1/wallet/{address}/balance` returns correct amount
- [ ] `/api/v1/node/status` shows correct node balance
- [ ] Balance persists across node restarts
- [ ] Multiple wallets tracked correctly
- [ ] Dev fee wallet shows accumulated fees

---

## 📝 **FILES TO MODIFY**

1. `crates/q-api-server/src/main.rs`
   - Line 983: Already creates `balance_engine`
   - Add to AppState initialization

2. `crates/q-api-server/src/handlers.rs`
   - Line 40: Update node status balance query
   - Line 2683-2686: Update wallet balance query

3. `crates/q-storage/src/balance_consensus.rs`
   - Verify `get_balance()` function exists (line 585)
   - May need implementation if it's just a trait

---

## 💡 **LESSONS LEARNED**

1. **Never maintain two sources of truth for critical data**
2. **Always trace data flow from write to read**
3. **Test balance updates end-to-end (mining → storage → API → UI)**
4. **Use a single authoritative data source (RocksDB)**

---

**Next Steps**: Implement Option 1 (recommended fix)
