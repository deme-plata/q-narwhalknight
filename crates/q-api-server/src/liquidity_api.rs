/// Liquidity Provision API for DEX
///
/// This module handles adding and managing liquidity pools
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{AppState, LiquidityPool};
use q_types::{Transaction, TxStatus};

/// API response wrapper
#[derive(Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: u64,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: current_timestamp(),
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message),
            timestamp: current_timestamp(),
        }
    }
}

/// Add liquidity request
#[derive(Debug, Deserialize)]
pub struct AddLiquidityRequest {
    pub token0: String, // "QUG" for native or token contract address
    pub token1: String, // Token contract address
    pub amount0: u64,
    pub amount1: u64,
    pub provider: String, // Wallet address
}

/// Add liquidity response
#[derive(Debug, Serialize)]
pub struct AddLiquidityResponse {
    pub pool_id: String,
    pub token0: String,
    pub token1: String,
    pub amount0: u64,
    pub amount1: u64,
    pub transaction_id: String,
}

/// Remove liquidity request
#[derive(Debug, Deserialize)]
pub struct RemoveLiquidityRequest {
    pub pool_id: String,
    pub percentage: u64,  // Percentage to remove (0-100)
    pub provider: String, // Wallet address
}

/// Remove liquidity response
#[derive(Debug, Serialize)]
pub struct RemoveLiquidityResponse {
    pub pool_id: String,
    pub amount0_returned: u64,
    pub amount1_returned: u64,
    pub transaction_id: String,
}

/// Create liquidity router
pub fn create_liquidity_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/add", post(add_liquidity))
        .route("/remove", post(remove_liquidity))
        .route("/pools", get(get_all_pools))
        .route("/pools/:pool_id", get(get_pool_info))
}

/// Add liquidity to a pool
pub async fn add_liquidity(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AddLiquidityRequest>,
) -> Result<Json<ApiResponse<AddLiquidityResponse>>, StatusCode> {
    // Parse provider address
    let provider = match parse_address(&request.provider) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    // Validate amounts
    if request.amount0 == 0 || request.amount1 == 0 {
        return Ok(Json(ApiResponse::error(
            "Both amounts must be greater than 0".to_string(),
        )));
    }

    // Check if token0 is native QUG or a token contract
    let is_native_token0 =
        request.token0.to_uppercase() == "QUG" || request.token0.to_lowercase() == "native-qug";

    // Check if token1 is native QUG or a token contract
    let is_native_token1 =
        request.token1.to_uppercase() == "QUG" || request.token1.to_lowercase() == "native-qug";

    // Resolve token1 symbol to contract address if needed (unless it's native QUG)
    let token1_addr = if is_native_token1 {
        // Native QUG doesn't have a contract address, use zero address as placeholder
        [0u8; 32]
    } else if request.token1.starts_with("0x") || request.token1.starts_with("qnk") {
        // Already an address
        match parse_address(&request.token1) {
            Ok(addr) => addr,
            Err(e) => return Ok(Json(ApiResponse::error(e))),
        }
    } else {
        // It's a token symbol, look it up in deployed contracts
        match resolve_token_symbol(&state, &request.token1).await {
            Ok(addr) => addr,
            Err(e) => {
                return Ok(Json(ApiResponse::error(format!(
                    "Token symbol '{}' not found: {}",
                    request.token1, e
                ))))
            }
        }
    };

    // Track which token balances changed for persistence
    let mut token_balance_changes: Vec<([u8; 32], [u8; 32], u64)> = Vec::new(); // (wallet, token, new_balance)

    // Deduct balances
    {
        let mut wallet_balances = state.wallet_balances.write().await;
        let mut token_balances = state.token_balances.write().await;

        // Deduct token0 (native QUG or token)
        if is_native_token0 {
            // Deduct native QUG - initialize if needed
            let balance = wallet_balances.entry(provider).or_insert(0);
            if *balance < request.amount0 {
                return Ok(Json(ApiResponse::error(format!(
                    "Insufficient QUG balance. Required: {}, Available: {}",
                    request.amount0, *balance
                ))));
            }
            *balance -= request.amount0;
            tracing::info!(
                "💸 Deducted {} QUG from {} for liquidity. New balance: {}",
                request.amount0 as f64 / 100_000_000.0,
                hex::encode(provider),
                *balance as f64 / 100_000_000.0
            );
        } else {
            // Deduct token0 (token contract) - resolve symbol if needed
            let token0_addr =
                if request.token0.starts_with("0x") || request.token0.starts_with("qnk") {
                    // Already an address
                    match parse_address(&request.token0) {
                        Ok(addr) => addr,
                        Err(e) => return Ok(Json(ApiResponse::error(e))),
                    }
                } else {
                    // It's a token symbol, look it up in deployed contracts
                    match resolve_token_symbol(&state, &request.token0).await {
                        Ok(addr) => addr,
                        Err(e) => {
                            return Ok(Json(ApiResponse::error(format!(
                                "Token symbol '{}' not found: {}",
                                request.token0, e
                            ))))
                        }
                    }
                };

            let balance_key = (provider, token0_addr);

            // AUTO-RESTORE: Check if this wallet deployed the token and restore balance if missing
            if !token_balances.contains_key(&balance_key) {
                // Try to find and restore the balance from deployed contracts
                let deployed_contracts = state.orobit_ecosystem.deployed_contracts.read().await;
                for contract in deployed_contracts.values() {
                    if contract.deployer == provider && contract.address.0 == token0_addr {
                        if let Some(supply_value) = contract
                            .deployment_params
                            .get("initialSupply")
                            .or_else(|| contract.deployment_params.get("initial_supply"))
                        {
                            let initial_supply = supply_value.as_u64().or_else(|| {
                                supply_value.as_str().and_then(|s| s.parse::<u64>().ok())
                            });

                            if let Some(supply) = initial_supply {
                                token_balances.insert(balance_key, supply);
                                tracing::info!(
                                    "💰 Auto-restored token0 balance for {} (contract {}): {} tokens",
                                    hex::encode(&provider[..8]),
                                    hex::encode(&token0_addr[..8]),
                                    supply
                                );
                                break;
                            }
                        }
                    }
                }
                drop(deployed_contracts);
            }

            if let Some(balance) = token_balances.get_mut(&balance_key) {
                if *balance < request.amount0 {
                    return Ok(Json(ApiResponse::error(format!(
                        "Insufficient token0 balance. Required: {}, Available: {}",
                        request.amount0, *balance
                    ))));
                }
                *balance -= request.amount0;
                token_balance_changes.push((provider, token0_addr, *balance)); // Track for persistence
                tracing::info!(
                    "💸 Deducted {} token0 from {} for liquidity",
                    request.amount0,
                    hex::encode(provider)
                );
            } else {
                return Ok(Json(ApiResponse::error(
                    "Insufficient token0 balance".to_string(),
                )));
            }
        }

        // Deduct token1 (native QUG or token)
        if is_native_token1 {
            // Deduct native QUG - initialize if needed
            let balance = wallet_balances.entry(provider).or_insert(0);
            if *balance < request.amount1 {
                return Ok(Json(ApiResponse::error(format!(
                    "Insufficient QUG balance. Required: {}, Available: {}",
                    request.amount1, *balance
                ))));
            }
            *balance -= request.amount1;
            tracing::info!(
                "💸 Deducted {} QUG from {} for liquidity. New balance: {}",
                request.amount1 as f64 / 100_000_000.0,
                hex::encode(provider),
                *balance as f64 / 100_000_000.0
            );
        } else {
            // Deduct token1 (token contract)
            let balance_key = (provider, token1_addr);

            // AUTO-RESTORE: Check if this wallet deployed the token and restore balance if missing
            if !token_balances.contains_key(&balance_key) {
                // Try to find and restore the balance from deployed contracts
                let deployed_contracts = state.orobit_ecosystem.deployed_contracts.read().await;
                for contract in deployed_contracts.values() {
                    if contract.deployer == provider && contract.address.0 == token1_addr {
                        if let Some(supply_value) = contract
                            .deployment_params
                            .get("initialSupply")
                            .or_else(|| contract.deployment_params.get("initial_supply"))
                        {
                            let initial_supply = supply_value.as_u64().or_else(|| {
                                supply_value.as_str().and_then(|s| s.parse::<u64>().ok())
                            });

                            if let Some(supply) = initial_supply {
                                token_balances.insert(balance_key, supply);
                                tracing::info!(
                                    "💰 Auto-restored token1 balance for {} (contract {}): {} tokens",
                                    hex::encode(&provider[..8]),
                                    hex::encode(&token1_addr[..8]),
                                    supply
                                );
                                break;
                            }
                        }
                    }
                }
                drop(deployed_contracts);
            }

            if let Some(balance) = token_balances.get_mut(&balance_key) {
                if *balance < request.amount1 {
                    return Ok(Json(ApiResponse::error(format!(
                        "Insufficient token1 balance. Required: {}, Available: {}",
                        request.amount1, *balance
                    ))));
                }
                *balance -= request.amount1;
                token_balance_changes.push((provider, token1_addr, *balance)); // Track for persistence
                tracing::info!(
                    "💸 Deducted {} token1 from {} for liquidity",
                    request.amount1,
                    hex::encode(provider)
                );
            } else {
                return Ok(Json(ApiResponse::error(
                    "Insufficient token1 balance".to_string(),
                )));
            }
        }
    }

    // Persist all token balance changes to storage
    for (wallet_addr, token_addr, new_balance) in token_balance_changes {
        if let Err(e) = state
            .storage_engine
            .save_token_balance(&wallet_addr, &token_addr, new_balance)
            .await
        {
            tracing::warn!("Failed to persist token balance after liquidity: {}", e);
        }
    }

    // Check if a pool already exists for this token pair (with same provider)
    let pool_id = {
        let pools = state.liquidity_pools.read().await;

        // Look for existing pool with matching token pair and provider
        pools
            .values()
            .find(|p| {
                (p.token0 == request.token0 && p.token1 == request.token1 && p.provider == provider)
                    || (p.token0 == request.token1
                        && p.token1 == request.token0
                        && p.provider == provider)
            })
            .map(|p| p.pool_id.clone())
    };

    let (final_pool_id, action) = if let Some(existing_pool_id) = pool_id {
        // Pool exists - add to reserves
        let mut pools = state.liquidity_pools.write().await;
        if let Some(pool) = pools.get_mut(&existing_pool_id) {
            // Check token order and add to correct reserves
            if pool.token0 == request.token0 && pool.token1 == request.token1 {
                pool.reserve0 += request.amount0;
                pool.reserve1 += request.amount1;
            } else {
                // Swapped order
                pool.reserve0 += request.amount1;
                pool.reserve1 += request.amount0;
            }
            tracing::info!(
                "💰 Added to existing liquidity pool {} - New reserves: {} / {}",
                existing_pool_id,
                pool.reserve0,
                pool.reserve1
            );

            // ✅ Persist updated liquidity pool to storage
            let pool_clone = pool.clone();
            drop(pools); // Release write lock before async I/O

            if let Ok(pool_data) = serde_json::to_vec(&pool_clone) {
                if let Err(e) = state
                    .storage_engine
                    .save_liquidity_pool(&existing_pool_id, &pool_data)
                    .await
                {
                    tracing::warn!("Failed to persist updated liquidity pool: {}", e);
                } else {
                    tracing::info!("💾 Persisted updated liquidity pool: {}", existing_pool_id);
                }
            }

            (existing_pool_id.clone(), "added")
        } else {
            // Pool was removed between read and write locks - create new one
            let new_pool_id = format!(
                "pool-{}-{}-{}",
                request.token0,
                request.token1,
                chrono::Utc::now().timestamp_millis()
            );
            let pool = LiquidityPool {
                pool_id: new_pool_id.clone(),
                token0: request.token0.clone(),
                token1: request.token1.clone(),
                reserve0: request.amount0,
                reserve1: request.amount1,
                provider,
                created_at: chrono::Utc::now(),
            };
            let pool_clone = pool.clone();
            pools.insert(new_pool_id.clone(), pool);
            tracing::info!(
                "💰 Created liquidity pool {} with reserves: {} / {}",
                new_pool_id,
                request.amount0,
                request.amount1
            );

            // ✅ Persist new liquidity pool to storage
            if let Ok(pool_data) = serde_json::to_vec(&pool_clone) {
                if let Err(e) = state
                    .storage_engine
                    .save_liquidity_pool(&new_pool_id, &pool_data)
                    .await
                {
                    tracing::warn!("Failed to persist new liquidity pool: {}", e);
                } else {
                    tracing::info!("💾 Persisted new liquidity pool: {}", new_pool_id);
                }
            }

            (new_pool_id, "created")
        }
    } else {
        // No existing pool - create new one
        let new_pool_id = format!(
            "pool-{}-{}-{}",
            request.token0,
            request.token1,
            chrono::Utc::now().timestamp_millis()
        );
        let pool = LiquidityPool {
            pool_id: new_pool_id.clone(),
            token0: request.token0.clone(),
            token1: request.token1.clone(),
            reserve0: request.amount0,
            reserve1: request.amount1,
            provider,
            created_at: chrono::Utc::now(),
        };

        let pool_clone = pool.clone();
        let mut pools = state.liquidity_pools.write().await;
        pools.insert(new_pool_id.clone(), pool);
        tracing::info!(
            "💰 Created liquidity pool {} with reserves: {} / {}",
            new_pool_id,
            request.amount0,
            request.amount1
        );

        // ✅ Persist new liquidity pool to storage
        if let Ok(pool_data) = serde_json::to_vec(&pool_clone) {
            if let Err(e) = state
                .storage_engine
                .save_liquidity_pool(&new_pool_id, &pool_data)
                .await
            {
                tracing::warn!("Failed to persist new liquidity pool: {}", e);
            } else {
                tracing::info!("💾 Persisted new liquidity pool: {}", new_pool_id);
            }
        }

        (new_pool_id, "created")
    };

    // Create transaction history entry
    let tx_hash = format!(
        "liquidity-{}-{}",
        hex::encode(provider),
        chrono::Utc::now().timestamp_millis()
    );
    let transaction = Transaction {
        id: [0u8; 32], // Would be properly hashed in production
        from: provider,
        to: [0u8; 32],
        amount: request.amount0,
        fee: 0,
        nonce: 0,
        signature: vec![],
        timestamp: chrono::Utc::now(),
        data: format!(
            "Add liquidity: {} {} + {} {}",
            request.amount0, request.token0, request.amount1, request.token1
        )
        .into_bytes(),
        token_type: q_types::TokenType::QUG,
        fee_token_type: q_types::TokenType::QUGUSD,
    };

    // Store transaction
    let tx_id = transaction.id;
    state.tx_pool.insert(tx_id, transaction);
    state.tx_status.insert(
        tx_id,
        TxStatus::Confirmed {
            block_height: 0,
            round: 0,
        },
    );

    Ok(Json(ApiResponse::success(AddLiquidityResponse {
        pool_id: final_pool_id.clone(),
        token0: request.token0,
        token1: request.token1,
        amount0: request.amount0,
        amount1: request.amount1,
        transaction_id: tx_hash,
    })))
}

/// Get all liquidity pools
pub async fn get_all_pools(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<PoolInfo>>>, StatusCode> {
    let pools = state.liquidity_pools.read().await;

    let pool_infos: Vec<PoolInfo> = pools
        .values()
        .map(|pool| PoolInfo {
            pool_id: pool.pool_id.clone(),
            token0: pool.token0.clone(),
            token1: pool.token1.clone(),
            reserve0: pool.reserve0,
            reserve1: pool.reserve1,
            provider: format!("qnk{}", hex::encode(pool.provider)),
            created_at: pool.created_at.timestamp() as u64,
        })
        .collect();

    Ok(Json(ApiResponse::success(pool_infos)))
}

/// Get specific pool info
pub async fn get_pool_info(
    Path(pool_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<PoolInfo>>, StatusCode> {
    let pools = state.liquidity_pools.read().await;

    match pools.get(&pool_id) {
        Some(pool) => Ok(Json(ApiResponse::success(PoolInfo {
            pool_id: pool.pool_id.clone(),
            token0: pool.token0.clone(),
            token1: pool.token1.clone(),
            reserve0: pool.reserve0,
            reserve1: pool.reserve1,
            provider: format!("qnk{}", hex::encode(pool.provider)),
            created_at: pool.created_at.timestamp() as u64,
        }))),
        None => Ok(Json(ApiResponse::error("Pool not found".to_string()))),
    }
}

/// Remove liquidity from a pool
pub async fn remove_liquidity(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RemoveLiquidityRequest>,
) -> Result<Json<ApiResponse<RemoveLiquidityResponse>>, StatusCode> {
    // Parse provider address
    let provider = match parse_address(&request.provider) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    // Validate percentage
    if request.percentage == 0 || request.percentage > 100 {
        return Ok(Json(ApiResponse::error(
            "Percentage must be between 1 and 100".to_string(),
        )));
    }

    // Get the pool
    let pool = {
        let pools = state.liquidity_pools.read().await;
        match pools.get(&request.pool_id) {
            Some(p) => p.clone(),
            None => return Ok(Json(ApiResponse::error("Pool not found".to_string()))),
        }
    };

    // Verify ownership
    if pool.provider != provider {
        return Ok(Json(ApiResponse::error(
            "You can only remove liquidity from your own pools".to_string(),
        )));
    }

    // Calculate amounts to return
    let amount0_to_return = (pool.reserve0 * request.percentage) / 100;
    let amount1_to_return = (pool.reserve1 * request.percentage) / 100;

    // Check if tokens are native QUG or custom tokens
    let is_native_token0 =
        pool.token0.to_uppercase() == "QUG" || pool.token0.to_lowercase() == "native-qug";
    let is_native_token1 =
        pool.token1.to_uppercase() == "QUG" || pool.token1.to_lowercase() == "native-qug";

    // Resolve token addresses for custom tokens
    let token0_addr = if !is_native_token0 {
        if pool.token0.starts_with("0x") || pool.token0.starts_with("qnk") {
            match parse_address(&pool.token0) {
                Ok(addr) => addr,
                Err(e) => return Ok(Json(ApiResponse::error(e))),
            }
        } else {
            match resolve_token_symbol(&state, &pool.token0).await {
                Ok(addr) => addr,
                Err(e) => return Ok(Json(ApiResponse::error(e))),
            }
        }
    } else {
        [0u8; 32]
    };

    let token1_addr = if !is_native_token1 {
        if pool.token1.starts_with("0x") || pool.token1.starts_with("qnk") {
            match parse_address(&pool.token1) {
                Ok(addr) => addr,
                Err(e) => return Ok(Json(ApiResponse::error(e))),
            }
        } else {
            match resolve_token_symbol(&state, &pool.token1).await {
                Ok(addr) => addr,
                Err(e) => return Ok(Json(ApiResponse::error(e))),
            }
        }
    } else {
        [0u8; 32]
    };

    let mut token_balance_changes: Vec<([u8; 32], [u8; 32], u64)> = Vec::new();

    // Return balances to provider
    {
        let mut wallet_balances = state.wallet_balances.write().await;
        let mut token_balances = state.token_balances.write().await;

        // Return token0
        if is_native_token0 {
            *wallet_balances.entry(provider).or_insert(0) += amount0_to_return;
            tracing::info!(
                "💰 Returned {} QUG to {} from liquidity removal",
                amount0_to_return,
                hex::encode(provider)
            );
        } else {
            let balance_key = (provider, token0_addr);
            *token_balances.entry(balance_key).or_insert(0) += amount0_to_return;
            token_balance_changes.push((
                provider,
                token0_addr,
                *token_balances.get(&balance_key).unwrap(),
            ));
            tracing::info!(
                "💰 Returned {} token0 to {} from liquidity removal",
                amount0_to_return,
                hex::encode(provider)
            );
        }

        // Return token1
        if is_native_token1 {
            *wallet_balances.entry(provider).or_insert(0) += amount1_to_return;
            tracing::info!(
                "💰 Returned {} QUG to {} from liquidity removal",
                amount1_to_return,
                hex::encode(provider)
            );
        } else {
            let balance_key = (provider, token1_addr);
            *token_balances.entry(balance_key).or_insert(0) += amount1_to_return;
            token_balance_changes.push((
                provider,
                token1_addr,
                *token_balances.get(&balance_key).unwrap(),
            ));
            tracing::info!(
                "💰 Returned {} token1 to {} from liquidity removal",
                amount1_to_return,
                hex::encode(provider)
            );
        }
    }

    // Persist token balance changes
    for (wallet_addr, token_addr, new_balance) in token_balance_changes {
        if let Err(e) = state
            .storage_engine
            .save_token_balance(&wallet_addr, &token_addr, new_balance)
            .await
        {
            tracing::warn!(
                "Failed to persist token balance after liquidity removal: {}",
                e
            );
        }
    }

    // Update or remove pool
    {
        let mut pools = state.liquidity_pools.write().await;
        if request.percentage == 100 {
            // Remove pool entirely
            pools.remove(&request.pool_id);
            tracing::info!(
                "🗑️ Removed liquidity pool {} (100% withdrawn)",
                request.pool_id
            );

            // ✅ Delete pool from storage
            drop(pools); // Release write lock before async I/O
            if let Err(e) = state
                .storage_engine
                .delete_liquidity_pool(&request.pool_id)
                .await
            {
                tracing::warn!("Failed to delete liquidity pool from storage: {}", e);
            } else {
                tracing::info!(
                    "💾 Deleted liquidity pool from storage: {}",
                    request.pool_id
                );
            }
        } else {
            // Update pool reserves
            if let Some(pool) = pools.get_mut(&request.pool_id) {
                pool.reserve0 -= amount0_to_return;
                pool.reserve1 -= amount1_to_return;
                tracing::info!(
                    "📉 Reduced liquidity pool {} reserves: {} / {}",
                    request.pool_id,
                    pool.reserve0,
                    pool.reserve1
                );

                // ✅ Persist updated pool to storage
                let pool_clone = pool.clone();
                drop(pools); // Release write lock before async I/O

                if let Ok(pool_data) = serde_json::to_vec(&pool_clone) {
                    if let Err(e) = state
                        .storage_engine
                        .save_liquidity_pool(&request.pool_id, &pool_data)
                        .await
                    {
                        tracing::warn!(
                            "Failed to persist updated liquidity pool after removal: {}",
                            e
                        );
                    } else {
                        tracing::info!("💾 Persisted updated liquidity pool: {}", request.pool_id);
                    }
                }
            }
        }
    }

    // Create transaction history
    let tx_hash = format!(
        "remove-liquidity-{}-{}",
        hex::encode(provider),
        chrono::Utc::now().timestamp_millis()
    );

    Ok(Json(ApiResponse::success(RemoveLiquidityResponse {
        pool_id: request.pool_id,
        amount0_returned: amount0_to_return,
        amount1_returned: amount1_to_return,
        transaction_id: tx_hash,
    })))
}

#[derive(Debug, Serialize)]
pub struct PoolInfo {
    pub pool_id: String,
    pub token0: String,
    pub token1: String,
    pub reserve0: u64,
    pub reserve1: u64,
    pub provider: String,
    pub created_at: u64,
}

// Helper functions
fn parse_address(address_str: &str) -> Result<[u8; 32], String> {
    // Support both 0x (Ethereum-style) and qnk (Q-NarwhalKnight) prefixes
    let hex_str = if address_str.starts_with("0x") {
        if address_str.len() != 42 && address_str.len() != 66 {
            return Err(format!(
                "Invalid 0x address format (expected 42 or 66 chars, got {})",
                address_str.len()
            ));
        }
        &address_str[2..]
    } else if address_str.starts_with("qnk") {
        // Q-NarwhalKnight addresses: qnk + 64 hex chars = 67 total
        if address_str.len() != 43 && address_str.len() != 67 {
            return Err(format!(
                "Invalid qnk address format (expected 43 or 67 chars, got {})",
                address_str.len()
            ));
        }
        &address_str[3..]
    } else {
        return Err(format!(
            "Address must start with 0x or qnk (got: {})",
            address_str
        ));
    };

    match hex::decode(hex_str) {
        Ok(bytes) => {
            if bytes.len() == 32 {
                // Q-NarwhalKnight native format (32 bytes)
                let mut result = [0u8; 32];
                result.copy_from_slice(&bytes);
                Ok(result)
            } else if bytes.len() == 20 {
                // Ethereum-style address (20 bytes), pad to 32 bytes
                let mut padded = [0u8; 32];
                padded[12..].copy_from_slice(&bytes);
                Ok(padded)
            } else {
                Err(format!(
                    "Address must be 20 or 32 bytes, got {}",
                    bytes.len()
                ))
            }
        }
        Err(_) => Err("Invalid hex in address".to_string()),
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Resolve a token symbol to its contract address by searching deployed contracts
async fn resolve_token_symbol(state: &Arc<AppState>, symbol: &str) -> Result<[u8; 32], String> {
    // Search through all deployed contracts to find one with matching symbol
    let ecosystem = &state.orobit_ecosystem;

    // Access the deployed contracts directly
    let deployed_contracts = ecosystem.deployed_contracts.read().await;

    // Search for a contract with matching symbol
    for contract in deployed_contracts.values() {
        if let Some(contract_symbol) = &contract.metadata.symbol {
            if contract_symbol.eq_ignore_ascii_case(symbol) {
                return Ok(contract.address.0);
            }
        }
    }

    Err(format!("No contract found with symbol '{}'", symbol))
}
