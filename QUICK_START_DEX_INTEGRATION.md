# Quick Start: DEX Token Registry Integration

## 🚀 Getting Started in 5 Minutes

This guide shows you how to integrate the new token registry system with your existing Q-NarwhalKnight node.

---

## Step 1: Update Dependencies

The implementation uses existing dependencies. Ensure your `Cargo.toml` includes:

```toml
[dependencies]
q-storage = { path = "../q-storage" }
q-dex = { path = "../q-dex" }
q-oracle = { path = "../q-oracle" }
bigdecimal = "0.4"
rocksdb = "0.22"
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"
chrono = "0.4"
```

---

## Step 2: Initialize in Your Main Application

**File: `crates/q-api-server/src/main.rs`**

Add this initialization code:

```rust
use q_storage::token_registry::TokenRegistry;
use q_storage::price_history::PriceHistoryManager;
use q_dex::{QuantumDexManager, OraclePriceBridge, create_default_bridge};
use q_oracle::QuantumOracle;
use rocksdb::DB;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Open RocksDB (or reuse existing DB handle)
    let db_path = "./data/q-narwhal-db";
    let db = Arc::new(DB::open_default(db_path)?);

    // 2. Initialize Token Registry
    let token_registry = Arc::new(TokenRegistry::new(db.clone()));
    token_registry.initialize().await?;
    println!("✅ Token Registry initialized");

    // 3. Initialize Price History Manager
    let price_history = Arc::new(PriceHistoryManager::new(db.clone()));
    price_history.initialize().await?;
    println!("✅ Price History Manager initialized");

    // 4. Create DEX Manager with Registry
    let dex_manager = Arc::new(QuantumDexManager::new(
        token_registry.clone(),
        price_history.clone(),
    )?);
    dex_manager.initialize().await?;
    println!("✅ DEX Manager initialized");

    // 5. Setup Oracle (if you have oracle enabled)
    let node_id = [0u8; 32]; // Your node ID
    let phase = q_types::Phase::Phase1;
    let oracle_config = q_oracle::QuantumOracleConfig::default();
    let oracle = Arc::new(QuantumOracle::new(node_id, phase, oracle_config).await?);
    oracle.initialize().await?;

    // 6. Create Oracle-DEX Price Bridge
    let price_bridge = Arc::new(create_default_bridge(
        dex_manager.clone(),
        oracle.clone(),
    ));
    price_bridge.initialize().await?;
    println!("✅ Oracle-DEX Price Bridge initialized");

    // 7. Store in AppState for handlers
    let app_state = Arc::new(AppState {
        dex_manager,
        token_registry,
        price_history,
        oracle,
        price_bridge,
        // ... other fields
    });

    // 8. Start server
    start_server(app_state).await?;

    Ok(())
}
```

---

## Step 3: Hook Up VM Token Creation

**File: `crates/q-vm/src/contracts/token_contract.rs` (or wherever you create tokens)**

When a token is created in the VM, register it with the DEX:

```rust
pub async fn create_token(
    symbol: String,
    name: String,
    total_supply: u64,
    decimals: u8,
    creator: String,
) -> Result<String> {
    // 1. Create token in VM (your existing code)
    let contract_address = vm.deploy_contract(/* ... */)?;

    // 2. 🆕 NEW: Register with DEX
    let dex_manager = get_dex_manager_from_context(); // Get from your context
    dex_manager.register_token_from_vm(
        contract_address.clone(),
        symbol,
        name,
        decimals,
        BigDecimal::from(total_supply),
        creator,
    ).await?;

    println!("✅ Token {} registered in DEX", symbol);

    Ok(contract_address)
}
```

---

## Step 4: Hook Up Liquidity Pool Creation

**File: `crates/q-dex/src/liquidity.rs` or your pool creation handler**

When a liquidity pool is created, register it:

```rust
pub async fn create_liquidity_pool_handler(
    base_token: String,
    quote_token: String,
    amount_a: BigDecimal,
    amount_b: BigDecimal,
    creator: String,
) -> Result<String> {
    // 1. Create pool (your existing code)
    let pool_address = create_amm_pool(/* ... */)?;

    // 2. 🆕 NEW: Register with DEX registry
    let dex_manager = get_dex_manager_from_context();
    let pool_id = dex_manager.register_liquidity_pool(
        pool_address.clone(),
        base_token,
        quote_token,
        amount_a,
        amount_b,
        creator,
    ).await?;

    println!("✅ Liquidity pool {} registered", pool_id);

    Ok(pool_id)
}
```

---

## Step 5: Hook Up Trade Recording

**File: `crates/q-dex/src/trading.rs` or your trade execution handler**

After each trade, record it in price history:

```rust
pub async fn execute_trade_handler(
    trader_id: String,
    pair_id: String,
    side: TradeSide,
    amount: BigDecimal,
) -> Result<QuantumTradeResult> {
    // 1. Execute trade (your existing code)
    let trade_result = execute_amm_swap(/* ... */)?;

    // 2. 🆕 NEW: Record in price history
    let dex_manager = get_dex_manager_from_context();
    dex_manager.record_trade(&trade_result).await?;

    // 3. 🆕 NEW: Submit to oracle (if enabled)
    let price_bridge = get_price_bridge_from_context();
    price_bridge.on_trade_executed(&trade_result).await?;

    println!("✅ Trade recorded: {} @ {}", pair_id, trade_result.price);

    Ok(trade_result)
}
```

---

## Step 6: Update API Endpoints

**File: `crates/q-api-server/src/handlers.rs`**

Update your token list endpoint to use the registry:

```rust
// BEFORE (hardcoded):
async fn list_tokens() -> Result<Json<Vec<Token>>, StatusCode> {
    let tokens = vec![
        Token { symbol: "ORB", ... },
        Token { symbol: "ORBUSD", ... },
    ];
    Ok(Json(tokens))
}

// AFTER (dynamic from registry):
async fn list_tokens(
    State(app_state): State<Arc<AppState>>,
) -> Result<Json<Vec<QuantumTokenInfo>>, StatusCode> {
    let tokens = app_state.dex_manager
        .get_all_available_tokens()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(tokens))
}
```

Add new endpoints for historical data:

```rust
// Get historical prices
async fn get_historical_prices(
    State(app_state): State<Arc<AppState>>,
    Path(pair_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Vec<QuantumOhlcvData>>, StatusCode> {
    let interval = params.get("interval").map(|s| s.as_str()).unwrap_or("1h");
    let limit = params.get("limit").and_then(|s| s.parse().ok());

    let candles = app_state.dex_manager
        .get_historical_prices(&pair_id, interval, limit)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(candles))
}

// Get 24h stats
async fn get_24h_stats(
    State(app_state): State<Arc<AppState>>,
    Path(pair_id): Path<String>,
) -> Result<Json<DayStatistics>, StatusCode> {
    let stats = app_state.price_history
        .get_24h_stats(&pair_id)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(stats))
}
```

---

## Step 7: Test Your Integration

### Test 1: List Tokens (Should Include Custom Tokens)
```bash
curl http://localhost:8080/api/dex/tokens

# Expected: JSON array with ORB, ORBUSD, and any custom tokens you've created
```

### Test 2: Create a Test Token
```bash
curl -X POST http://localhost:8080/api/vm/token/create \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "TEST",
    "name": "Test Token",
    "total_supply": "1000000",
    "decimals": 18
  }'

# Then verify it appears:
curl http://localhost:8080/api/dex/tokens | grep TEST
```

### Test 3: Create Liquidity Pool
```bash
curl -X POST http://localhost:8080/api/dex/pools/create \
  -H "Content-Type: application/json" \
  -d '{
    "base_token": "TEST",
    "quote_token": "ORB",
    "initial_reserve_base": "10000",
    "initial_reserve_quote": "1000",
    "creator": "your_address"
  }'

# Verify pool exists:
curl http://localhost:8080/api/dex/pools
```

### Test 4: Execute Trade and Check History
```bash
# Execute a swap
curl -X POST http://localhost:8080/api/dex/swap/execute \
  -H "Content-Type: application/json" \
  -d '{
    "from_token": "ORB",
    "to_token": "TEST",
    "amount": "10"
  }'

# Check historical prices (should show the trade)
curl "http://localhost:8080/api/dex/prices/historical/TEST/ORB?interval=1m&limit=10"

# Check 24h stats
curl http://localhost:8080/api/dex/prices/TEST/ORB/24h-stats
```

---

## Troubleshooting

### Issue: Compilation errors about missing types

**Solution:** Ensure you've added the modules to `lib.rs`:

```rust
// In crates/q-storage/src/lib.rs
pub mod token_registry;
pub mod price_history;

// In crates/q-dex/src/lib.rs
pub mod oracle_price_bridge;
```

### Issue: "Token not found" when creating pool

**Solution:** Make sure the token was registered first via `register_token_from_vm()`.

### Issue: Historical prices not showing

**Solution:** Ensure `record_trade()` is being called after each trade. Check logs for "Trade recorded" messages.

### Issue: Database errors

**Solution:** Ensure RocksDB path exists and has write permissions:
```bash
mkdir -p ./data/q-narwhal-db
chmod 755 ./data/q-narwhal-db
```

---

## Next Steps

Once the basic integration is working:

1. ✅ **Add frontend integration** - Update your UI to fetch tokens from `/api/dex/tokens`
2. ✅ **Add charting** - Use historical price endpoints to display charts
3. ✅ **Enable oracle pairs** - Call `price_bridge.enable_oracle_for_pair()` for pairs you want oracle data
4. ✅ **Add token verification** - Implement admin panel to verify tokens
5. ✅ **Add advanced features** - Limit orders, stop-loss, etc.

---

## Reference

- **Full Implementation Guide:** `DEX_TOKEN_REGISTRY_IMPLEMENTATION.md`
- **Token Registry API:** `crates/q-storage/src/token_registry.rs`
- **Price History API:** `crates/q-storage/src/price_history.rs`
- **DEX Manager API:** `crates/q-dex/src/lib.rs`
- **Oracle Bridge API:** `crates/q-dex/src/oracle_price_bridge.rs`

---

**Need help?** Check the implementation guide or review the test cases in each module.

**Happy coding!** 🚀
