# 🔴 CRITICAL: DEX Data Loss on Restart

**Date**: 2025-10-16
**Severity**: 🔴 **CRITICAL - DATA LOSS**
**Status**: IDENTIFIED - REQUIRES IMMEDIATE FIX

---

## 🚨 Problem Description

After restarting the API server, **ALL liquidity pools are lost**. This causes:

1. ❌ All liquidity pools disappear (`/api/v1/dex/pools` returns empty array)
2. ❌ QUGUSD token not found (swaps fail with "Token 'QUGUSD' not found")
3. ❌ Users cannot trade or swap tokens
4. ❌ Liquidity providers lose their pool positions

---

## 🔍 Root Cause Analysis

### Issue 1: Liquidity Pools NOT Persisted

**File**: `crates/q-api-server/src/lib.rs`

Liquidity pools are stored in memory only:
```rust
pub liquidity_pools: Arc<RwLock<HashMap<String, LiquidityPool>>>,
```

**Problem**: No save/load logic for liquidity pools in `AppState::new()` or `AppState::new_with_networks()`

**Evidence**:
- Wallet balances: ✅ Persisted (lines 167-183 in lib.rs)
- Token balances: ✅ Persisted (lines 186-201 in lib.rs)
- Liquidity pools: ❌ NOT persisted (initialized empty on line 266 & 607)

```rust
// lib.rs:266 & 607
liquidity_pools: Arc::new(RwLock::new(HashMap::new())), // ❌ ALWAYS EMPTY
```

### Issue 2: QUGUSD Token Not Registered

**File**: `crates/q-api-server/src/handlers.rs:3670-3686`

The swap endpoint tries to resolve token addresses:
```rust
let to_token_addr = if !to_is_native {
    match resolve_token_address(&state, &request.to_token).await {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(format!("To token not found: {}", e)))),
    }
} else {
    [0u8; 32]
};
```

**Problem**: `resolve_token_address()` looks for deployed contracts, but QUGUSD may be a special stablecoin that's not registered as a deployed contract.

---

## 💥 Impact

### User Impact:
- 🔴 **100% of liquidity pools lost on every restart**
- 🔴 **Unable to perform swaps** (no pools available)
- 🔴 **QUGUSD swaps fail** ("Token not found")
- 🔴 **Liquidity providers cannot withdraw**

### Business Impact:
- 🚨 **DEX is completely non-functional** after restart
- 🚨 **Production deployment impossible**
- 🚨 **User funds technically "locked"** (pools don't exist)

---

## ✅ Solution: Implement Liquidity Pool Persistence

### Step 1: Add Storage Methods to `q-storage`

**File**: `crates/q-storage/src/lib.rs`

Add methods:
```rust
/// Save liquidity pool to storage
pub async fn save_liquidity_pool(&self, pool_id: &str, pool: &LiquidityPool) -> Result<()> {
    let key = format!("liquidity_pool:{}", pool_id);
    let value = serde_json::to_vec(pool)?;
    self.hot_db.put(key.as_bytes(), &value)?;
    Ok(())
}

/// Load all liquidity pools from storage
pub async fn load_liquidity_pools(&self) -> Result<HashMap<String, LiquidityPool>> {
    let mut pools = HashMap::new();
    let prefix = b"liquidity_pool:";

    let iter = self.hot_db.prefix_iterator(prefix);
    for item in iter {
        let (key, value) = item?;
        let key_str = String::from_utf8_lossy(&key);
        let pool_id = key_str.strip_prefix("liquidity_pool:").unwrap().to_string();
        let pool: LiquidityPool = serde_json::from_slice(&value)?;
        pools.insert(pool_id, pool);
    }

    Ok(pools)
}

/// Delete liquidity pool from storage
pub async fn delete_liquidity_pool(&self, pool_id: &str) -> Result<()> {
    let key = format!("liquidity_pool:{}", pool_id);
    self.hot_db.delete(key.as_bytes())?;
    Ok(())
}
```

### Step 2: Load Liquidity Pools on Startup

**File**: `crates/q-api-server/src/lib.rs`

**In `AppState::new()` (around line 264)**:
```rust
// Load existing liquidity pools from persistent storage
let mut liquidity_pools_map = HashMap::new();
match storage_engine.load_liquidity_pools().await {
    Ok(persisted_pools) => {
        liquidity_pools_map = persisted_pools;
        tracing::info!(
            "💧 Loaded {} liquidity pools from persistent storage",
            liquidity_pools_map.len()
        );
    }
    Err(e) => {
        tracing::warn!(
            "Failed to load liquidity pools from storage: {}, starting with empty pools",
            e
        );
    }
}

Ok(Self {
    // ...
    liquidity_pools: Arc::new(RwLock::new(liquidity_pools_map)),
    // ...
})
```

**In `AppState::new_with_networks()` (around line 605)**: Same logic

### Step 3: Persist Liquidity Pools on Creation

**File**: `crates/q-api-server/src/handlers.rs`

**In `create_pool()` handler** (after creating pool):
```rust
// Persist pool to storage
if let Err(e) = state.storage_engine.save_liquidity_pool(&pool_id, &pool).await {
    warn!("Failed to persist liquidity pool to storage: {}", e);
    // Continue anyway - pool is in memory
}

info!("💾 Persisted liquidity pool {} to storage", pool_id);
```

**In swap operations** (after updating pool reserves in handlers.rs:3870-3883):
```rust
// Update pool reserves
{
    let mut pools = state.liquidity_pools.write().await;
    if let Some(pool_mut) = pools.get_mut(&pool_id) {
        if !is_reversed {
            pool_mut.reserve0 += request.amount_in;
            pool_mut.reserve1 -= amount_out;
        } else {
            pool_mut.reserve1 += request.amount_in;
            pool_mut.reserve0 -= amount_out;
        }
        info!("🔄 Updated pool reserves: {} / {}", pool_mut.reserve0, pool_mut.reserve1);

        // ✅ Persist updated pool to storage
        if let Err(e) = state.storage_engine.save_liquidity_pool(&pool_id, &pool_mut).await {
            warn!("Failed to persist updated pool reserves: {}", e);
        }

        (pool_mut.reserve0, pool_mut.reserve1, pool_mut.reserve0 + pool_mut.reserve1)
    } else {
        (0, 0, 0)
    }
}
```

---

## ✅ Solution: Register QUGUSD Token

### Option 1: Auto-Register QUGUSD on Startup

**File**: `crates/q-api-server/src/lib.rs`

In `AppState::new()` and `AppState::new_with_networks()`:
```rust
// Auto-register QUGUSD stablecoin token
let qugusd_address = [0xFFu8; 32]; // Special address for QUGUSD
let qugusd_symbol = "QUGUSD".to_string();

// Register in contract registry
contract_registry.register_token(qugusd_address, qugusd_symbol.clone()).await?;

tracing::info!("💵 Auto-registered QUGUSD stablecoin token");
```

### Option 2: Handle QUGUSD as Special Case

**File**: `crates/q-api-server/src/handlers.rs`

In swap endpoint (before resolving token addresses):
```rust
// Special handling for QUGUSD stablecoin
let to_token_addr = if !to_is_native {
    if to_is_qugusd {
        // QUGUSD has special address
        [0xFFu8; 32]
    } else {
        match resolve_token_address(&state, &request.to_token).await {
            Ok(addr) => addr,
            Err(e) => return Ok(Json(ApiResponse::error(format!("To token not found: {}", e)))),
        }
    }
} else {
    [0u8; 32]
};
```

---

## 🧪 Testing Plan

### Test 1: Liquidity Pool Persistence
```bash
# 1. Start server and create pool
curl -X POST http://localhost:8080/api/v1/dex/pools/create \
  -H "Content-Type: application/json" \
  -d '{"token0":"QUG","token1":"QUGUSD","reserve0":1000,"reserve1":1000,"provider":"..."}'

# 2. Verify pool exists
curl http://localhost:8080/api/v1/dex/pools
# Expected: pool exists

# 3. Restart server
killall q-api-server && Q_DB_PATH=./data-stark-test ./target/release/q-api-server --port 8080

# 4. Check pools again
curl http://localhost:8080/api/v1/dex/pools
# Expected: ✅ Pool still exists (currently ❌ FAILS)
```

### Test 2: QUGUSD Token Resolution
```bash
# Execute swap with QUGUSD
curl -X POST http://localhost:8080/api/v1/dex/swap \
  -H "Content-Type: application/json" \
  -d '{"from_token":"QUG","to_token":"QUGUSD","amount_in":100,"min_amount_out":90,"wallet_address":"..."}'

# Expected: ✅ Swap succeeds (currently ❌ FAILS with "Token not found")
```

---

## 📊 Priority

**Priority**: 🔴 **P0 - BLOCKER**
**Impact**: 🔴 **CRITICAL - DEX COMPLETELY BROKEN**
**Effort**: 🟡 **Medium** (2-3 hours)

### Why P0:
1. DEX is **completely unusable** after any restart
2. **100% data loss** for liquidity pools
3. **Blocks production deployment**
4. **User funds at risk** (technically locked)

---

## 🚀 Implementation Checklist

- [ ] Add liquidity pool storage methods to `q-storage/src/lib.rs`
- [ ] Add `save_liquidity_pool()` method
- [ ] Add `load_liquidity_pools()` method
- [ ] Add `delete_liquidity_pool()` method
- [ ] Load liquidity pools in `AppState::new()`
- [ ] Load liquidity pools in `AppState::new_with_networks()`
- [ ] Persist pools on creation in `create_pool()` handler
- [ ] Persist pool updates after swaps
- [ ] Persist pool updates on add/remove liquidity
- [ ] Register QUGUSD token on startup OR handle as special case
- [ ] Test pool persistence across restart
- [ ] Test QUGUSD swaps
- [ ] Update documentation

---

## 📝 Related Issues

- DEX_SECURITY_FIXES_APPLIED.md - Security fixes (authentication, overflow)
- DEX_BACKEND_ANALYSIS.md - Original security audit
- This bug was NOT identified in the security audit (focused on authentication/overflow)

---

## 🎯 Temporary Workaround

**For immediate testing only** (NOT production solution):

1. Don't restart the server
2. Keep liquidity pools in memory
3. Re-create pools manually after restart

**WARNING**: This is NOT a solution - just a workaround for testing!

---

## 💡 Recommendation

**MUST FIX IMMEDIATELY** before:
- Any production deployment
- Adding more security features
- Frontend integration testing
- User testing

**Order of Operations**:
1. ✅ Security fixes (completed) handlers.rs:3665-3888
2. 🔴 **THIS BUG** - Liquidity pool persistence (MUST DO NOW)
3. 🟡 Rate limiting (can wait)
4. 🟡 Transaction atomicity (can wait)

The security fixes are useless if the DEX doesn't work after restart!
