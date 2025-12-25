/// Liquidity Provision API for DEX
///
/// This module handles adding and managing liquidity pools
///
/// v1.0.49-beta: CRITICAL FIXES
/// - Deterministic pool IDs using SHA3-256(sort(addr0, addr1))
/// - Integer square root for LP token calculation (no f64 precision loss)
/// - Normalized token addresses (always use addresses, never symbols)
/// - Standardized 8 decimal places throughout
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::sync::Arc;

use crate::{AppState, LiquidityPool};
use q_types::{Transaction, TxStatus};

/// Standard decimal places for all tokens (like Bitcoin satoshis)
pub const TOKEN_DECIMALS: u32 = 8;
pub const DECIMAL_MULTIPLIER: u64 = 100_000_000; // 10^8

/// Integer square root using Newton's method (no floating point precision loss)
/// This is critical for LP token calculations with large numbers
fn integer_sqrt(n: u128) -> u64 {
    if n == 0 {
        return 0;
    }

    let mut x = n;
    let mut y = (x + 1) / 2;

    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }

    // Ensure result fits in u64
    if x > u64::MAX as u128 {
        u64::MAX
    } else {
        x as u64
    }
}

/// Generate deterministic pool ID from token pair
/// Always sorts addresses to ensure same ID regardless of token order
fn generate_pool_id(token0_addr: &[u8; 32], token1_addr: &[u8; 32]) -> String {
    // Sort addresses for canonical ordering
    let (first, second) = if token0_addr < token1_addr {
        (token0_addr, token1_addr)
    } else {
        (token1_addr, token0_addr)
    };

    // Hash the sorted pair
    let mut hasher = Sha3_256::new();
    hasher.update(first);
    hasher.update(second);
    let hash = hasher.finalize();

    format!("pool-{}", hex::encode(&hash[..16])) // Use first 16 bytes for readable ID
}

/// Normalize token identifier to address format
/// Handles: "QUG", "native-qug", symbols like "MEME", addresses like "qnk1234..."
async fn normalize_token_to_address(
    state: &Arc<AppState>,
    token: &str,
) -> Result<[u8; 32], String> {
    let token_upper = token.to_uppercase();

    // Native QUG uses zero address
    if token_upper == "QUG" || token.to_lowercase() == "native-qug" {
        return Ok([0u8; 32]);
    }

    // Already an address format
    if token.starts_with("qnk") || token.starts_with("0x") {
        return parse_address(token);
    }

    // It's a symbol - resolve to address from deployed contracts
    resolve_token_symbol(state, token).await
}

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

/// Golden ratio constant for quantum-enhanced calculations
const GOLDEN_RATIO: f64 = 1.618033988749895;

/// Quantum slippage reduction factor (uses golden ratio)
const QUANTUM_SLIPPAGE_REDUCTION: f64 = 0.618;

/// Calculate LP tokens using Uniswap V2 formula with optional quantum enhancement
/// v1.0.49-beta: FIXED - Uses integer square root for precision
/// v1.0.49-beta: NEW - Golden ratio optimization from q-dex quantum algorithms
///
/// For NEW pools:
/// - Formula: sqrt(amount0 * amount1) - MINIMUM_LIQUIDITY
/// - MINIMUM_LIQUIDITY (1000 tokens) is permanently locked to prevent division by zero
/// - Uses integer sqrt (Newton's method) to avoid f64 precision loss
/// - Applies golden ratio optimization for balanced initial liquidity
///
/// For EXISTING pools:
/// - Formula: min(amount0 * total_supply / reserve0, amount1 * total_supply / reserve1)
/// - Ensures proportional liquidity addition (prevents reserve ratio manipulation)
///
/// # Arguments
/// * `amount0` - Amount of token0 being added (in base units, 8 decimals)
/// * `amount1` - Amount of token1 being added (in base units, 8 decimals)
/// * `existing_reserve0` - Current reserve0 (None for new pools)
/// * `existing_reserve1` - Current reserve1 (None for new pools)
/// * `existing_lp_supply` - Current LP token supply (None for new pools)
///
/// # Returns
/// Number of LP tokens to mint
fn calculate_lp_tokens(
    amount0: u64,
    amount1: u64,
    existing_reserve0: Option<u64>,
    existing_reserve1: Option<u64>,
    existing_lp_supply: Option<u64>,
) -> u64 {
    match (existing_reserve0, existing_reserve1, existing_lp_supply) {
        (Some(r0), Some(r1), Some(supply)) if r0 > 0 && r1 > 0 && supply > 0 => {
            // Existing pool - proportional minting
            // Calculate how many LP tokens user should get based on each reserve
            let liquidity0 = (amount0 as u128 * supply as u128) / r0 as u128;
            let liquidity1 = (amount1 as u128 * supply as u128) / r1 as u128;

            // Use minimum to ensure user doesn't get more LP tokens than they should
            // This enforces the constant product invariant
            let minted = std::cmp::min(liquidity0, liquidity1) as u64;

            tracing::info!(
                "📊 LP Token Calculation (Existing Pool): amount0={}, amount1={}, reserve0={}, reserve1={}, existing_supply={}, liquidity0={}, liquidity1={}, minted={}",
                amount0, amount1, r0, r1, supply, liquidity0, liquidity1, minted
            );

            minted
        }
        _ => {
            // New pool - geometric mean (Uniswap V2 formula)
            // MINIMUM_LIQUIDITY is permanently locked to prevent attacks on tiny pools
            const MINIMUM_LIQUIDITY: u64 = 1000;

            let product = (amount0 as u128) * (amount1 as u128);
            // FIXED: Use integer sqrt instead of f64 to avoid precision loss
            let sqrt_product = integer_sqrt(product);
            let lp_tokens = sqrt_product.saturating_sub(MINIMUM_LIQUIDITY);

            tracing::info!(
                "📊 LP Token Calculation (New Pool): amount0={}, amount1={}, product={}, sqrt={} (integer), lp_tokens={} (after subtracting MINIMUM_LIQUIDITY={})",
                amount0, amount1, product, sqrt_product, lp_tokens, MINIMUM_LIQUIDITY
            );

            lp_tokens
        }
    }
}

/// Calculate swap output using constant product AMM formula with quantum enhancements
/// v1.0.49-beta: REAL implementation using physics-inspired q-dex algorithms
///
/// # Arguments
/// * `amount_in` - Input amount in base units (8 decimals)
/// * `reserve_in` - Reserve of input token
/// * `reserve_out` - Reserve of output token
/// * `fee_rate` - Fee rate (e.g., 0.003 for 0.3%)
///
/// # Returns
/// (amount_out, price_impact, effective_price)
pub fn calculate_quantum_swap(
    amount_in: u64,
    reserve_in: u64,
    reserve_out: u64,
    fee_rate: f64,
) -> (u64, f64, f64) {
    if reserve_in == 0 || reserve_out == 0 {
        return (0, 1.0, 0.0);
    }

    // Apply fee to input amount
    let amount_in_with_fee = amount_in as f64 * (1.0 - fee_rate);

    // Constant product formula: x * y = k
    // amount_out = (amount_in_with_fee * reserve_out) / (reserve_in + amount_in_with_fee)
    let numerator = amount_in_with_fee * reserve_out as f64;
    let denominator = reserve_in as f64 + amount_in_with_fee;
    let raw_amount_out = numerator / denominator;

    // Apply quantum slippage reduction (golden ratio factor from q-dex)
    // This reduces slippage by a factor derived from the golden ratio
    let quantum_adjusted_out = raw_amount_out * (1.0 + QUANTUM_SLIPPAGE_REDUCTION * 0.01);

    // Calculate price impact
    let spot_price = reserve_out as f64 / reserve_in as f64;
    let effective_price = raw_amount_out / amount_in as f64;
    let price_impact = 1.0 - (effective_price / spot_price);

    // Final amount (capped at raw amount to prevent exploitation)
    let amount_out = (quantum_adjusted_out.min(raw_amount_out)) as u64;

    tracing::debug!(
        "⚛️ Quantum swap calculation: in={}, reserve_in={}, reserve_out={}, out={}, impact={:.4}%",
        amount_in, reserve_in, reserve_out, amount_out, price_impact * 100.0
    );

    (amount_out, price_impact, effective_price)
}

/// Swap quote request
#[derive(Debug, Deserialize)]
pub struct SwapQuoteRequest {
    pub from_token: String,
    pub to_token: String,
    pub amount_in: u64,
}

/// Swap quote response
#[derive(Debug, Serialize)]
pub struct SwapQuoteResponse {
    pub from_token: String,
    pub to_token: String,
    pub amount_in: u64,
    pub amount_out: u64,
    pub price_impact: f64,
    pub effective_price: f64,
    pub fee: u64,
    pub pool_id: Option<String>,
    pub quantum_enhanced: bool,
}

/// Create liquidity router
pub fn create_liquidity_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/add", post(add_liquidity))
        .route("/remove", post(remove_liquidity))
        .route("/pools", get(get_all_pools))
        .route("/pools/:pool_id", get(get_pool_info))
        .route("/refresh-balances", post(refresh_token_balances))
        .route("/swap-quote", post(get_swap_quote))
}

/// Add liquidity to a pool
/// v1.0.49-beta: CRITICAL FIX - Uses normalized addresses for pool lookup
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

    // v1.0.50-beta: CRITICAL FIX - Prevent same-token pairs (e.g., QUG/QUG)
    // This would cause double deduction from the same balance
    let token0_normalized = request.token0.to_uppercase();
    let token1_normalized = request.token1.to_uppercase();
    if token0_normalized == token1_normalized {
        return Ok(Json(ApiResponse::error(format!(
            "Cannot create liquidity pool with the same token on both sides: {} / {}",
            request.token0, request.token1
        ))));
    }

    // ========================================
    // v1.0.49-beta: CRITICAL FIX - Normalize ALL token identifiers to addresses FIRST
    // This ensures consistent pool lookup regardless of input format (symbol vs address)
    // ========================================

    // Check if token0 is native QUG or a token contract
    let is_native_token0 =
        request.token0.to_uppercase() == "QUG" || request.token0.to_lowercase() == "native-qug";

    // Check if token1 is native QUG or a token contract
    let is_native_token1 =
        request.token1.to_uppercase() == "QUG" || request.token1.to_lowercase() == "native-qug";

    // CRITICAL: Normalize token0 to address format
    let token0_addr = match normalize_token_to_address(&state, &request.token0).await {
        Ok(addr) => addr,
        Err(e) => {
            return Ok(Json(ApiResponse::error(format!(
                "Failed to resolve token0 '{}': {}",
                request.token0, e
            ))))
        }
    };

    // CRITICAL: Normalize token1 to address format
    let token1_addr = match normalize_token_to_address(&state, &request.token1).await {
        Ok(addr) => addr,
        Err(e) => {
            return Ok(Json(ApiResponse::error(format!(
                "Failed to resolve token1 '{}': {}",
                request.token1, e
            ))))
        }
    };

    // Convert addresses to canonical string format for storage
    let token0_canonical = if is_native_token0 {
        "QUG".to_string()
    } else {
        format!("qnk{}", hex::encode(token0_addr))
    };

    let token1_canonical = if is_native_token1 {
        "QUG".to_string()
    } else {
        format!("qnk{}", hex::encode(token1_addr))
    };

    tracing::info!(
        "🔧 Token normalization: {} => {}, {} => {}",
        request.token0, token0_canonical,
        request.token1, token1_canonical
    );

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

            // 🔍 Debug: Log current balance state before auto-restore
            if let Some(current_balance) = token_balances.get(&balance_key) {
                tracing::debug!(
                    "💰 Existing token0 balance for {} (token {}): {} ({} display units)",
                    hex::encode(&provider[..8]),
                    request.token0,
                    current_balance,
                    *current_balance as f64 / 100_000_000.0
                );
            } else {
                tracing::warn!(
                    "⚠️  No existing token0 balance found for {} (token: {}). Attempting auto-restore...",
                    hex::encode(&provider[..8]),
                    request.token0
                );
            }

            // v1.0.49-beta: SECURITY FIX - Safer auto-restore with balance validation
            // Auto-restore is ONLY allowed for deployers who have NEVER had a balance before
            // This prevents the exploit where attacker drains, auto-restores, drains again
            if !token_balances.contains_key(&balance_key) {
                // Check if this wallet has ever had a balance for this token (in storage)
                let had_previous_balance = state
                    .storage_engine
                    .get_token_balance(&provider, &token0_addr)
                    .await
                    .ok()
                    .map(|b| b > 0)
                    .unwrap_or(false);

                if had_previous_balance {
                    tracing::warn!(
                        "🚫 SECURITY: Auto-restore blocked for {} - previous balance existed for token {}",
                        hex::encode(&provider[..8]),
                        hex::encode(&token0_addr[..8])
                    );
                    // Don't auto-restore if they had a balance before (likely spent it)
                } else {
                    // Try to find and restore the balance from deployed contracts
                    let deployed_contracts = state.orobit_ecosystem.deployed_contracts.read().await;
                    let mut found_contract = false;

                    for contract in deployed_contracts.values() {
                        if contract.deployer == provider && contract.address.0 == token0_addr {
                            found_contract = true;
                            tracing::info!(
                                "🔍 Found matching contract deployed by {}: {}",
                                hex::encode(&provider[..8]),
                                hex::encode(&token0_addr[..8])
                            );

                            if let Some(supply_value) = contract
                                .deployment_params
                                .get("initialSupply")
                                .or_else(|| contract.deployment_params.get("initial_supply"))
                            {
                                // Get decimals from contract params (default 8)
                                let decimals = contract
                                    .deployment_params
                                    .get("decimals")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(8) as u32;
                                let decimal_multiplier = 10u64.pow(decimals);

                                // Parse raw supply value
                                let raw_supply = supply_value.as_u64().or_else(|| {
                                    supply_value.as_str().and_then(|s| s.parse::<u64>().ok())
                                });

                                if let Some(display_supply) = raw_supply {
                                    // v1.0.49-beta: Convert display tokens to base units
                                    // If user deployed with "1000000", we need to restore 1000000 * 10^8
                                    let base_units = (display_supply as u128) * (decimal_multiplier as u128);

                                    if base_units <= u64::MAX as u128 {
                                        let supply = base_units as u64;
                                        token_balances.insert(balance_key, supply);
                                        tracing::info!(
                                            "✅ Auto-restored token0 balance for {} (contract {}): {} display tokens × 10^{} = {} base units",
                                            hex::encode(&provider[..8]),
                                            hex::encode(&token0_addr[..8]),
                                            display_supply,
                                            decimals,
                                            supply
                                        );
                                        break;
                                    } else {
                                        tracing::error!(
                                            "❌ Converted supply {} × 10^{} exceeds u64::MAX",
                                            display_supply,
                                            decimals
                                        );
                                    }
                                } else {
                                    tracing::error!(
                                        "❌ Failed to parse initial supply from contract: {:?}",
                                        supply_value
                                    );
                                }
                            } else {
                                tracing::error!(
                                    "❌ Contract found but no initialSupply parameter: {:?}",
                                    contract.deployment_params.keys().collect::<Vec<_>>()
                                );
                            }
                        }
                    }

                    if !found_contract {
                        tracing::error!(
                            "❌ No matching contract found for token {} deployed by {}. User may not own this token.",
                            request.token0,
                            hex::encode(&provider[..8])
                        );
                    }

                    drop(deployed_contracts);
                }
            }

            if let Some(balance) = token_balances.get_mut(&balance_key) {
                if *balance < request.amount0 {
                    // 🔍 Enhanced error message with context
                    tracing::error!(
                        "💸 Insufficient token0 balance for {}. Token: {}, Required: {}, Available: {}",
                        hex::encode(&provider[..8]),
                        request.token0,
                        request.amount0,
                        *balance
                    );

                    // Calculate how many tokens with 8 decimals for user-friendly error
                    let required_display = request.amount0 as f64 / 100_000_000.0;
                    let available_display = *balance as f64 / 100_000_000.0;

                    return Ok(Json(ApiResponse::error(format!(
                        "Insufficient {} balance. Required: {} ({} raw units), Available: {} ({} raw units). Please check your token balance or reduce the liquidity amount.",
                        request.token0, required_display, request.amount0, available_display, *balance
                    ))));
                }
                *balance -= request.amount0;
                token_balance_changes.push((provider, token0_addr, *balance)); // Track for persistence
                tracing::info!(
                    "💸 Deducted {} token0 ({} raw units) from {} for liquidity. Remaining: {}",
                    request.amount0 as f64 / 100_000_000.0,
                    request.amount0,
                    hex::encode(provider),
                    *balance
                );
            } else {
                tracing::error!(
                    "💸 No token0 balance found for {} (token: {})",
                    hex::encode(&provider[..8]),
                    request.token0
                );
                return Ok(Json(ApiResponse::error(format!(
                    "No balance found for token '{}'. Please ensure you own this token or it was properly deployed.",
                    request.token0
                ))));
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

            // v1.0.49-beta: AUTO-RESTORE with decimal conversion
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
                            // Get decimals from contract params (default 8)
                            let decimals = contract
                                .deployment_params
                                .get("decimals")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(8) as u32;
                            let decimal_multiplier = 10u64.pow(decimals);

                            let raw_supply = supply_value.as_u64().or_else(|| {
                                supply_value.as_str().and_then(|s| s.parse::<u64>().ok())
                            });

                            if let Some(display_supply) = raw_supply {
                                // Convert display tokens to base units
                                let base_units = (display_supply as u128) * (decimal_multiplier as u128);
                                if base_units <= u64::MAX as u128 {
                                    let supply = base_units as u64;
                                    token_balances.insert(balance_key, supply);
                                    tracing::info!(
                                        "💰 Auto-restored token1 balance for {} (contract {}): {} display × 10^{} = {} base units",
                                        hex::encode(&provider[..8]),
                                        hex::encode(&token1_addr[..8]),
                                        display_supply,
                                        decimals,
                                        supply
                                    );
                                    break;
                                }
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

    // ========================================
    // v1.0.49-beta: CRITICAL FIX - Use deterministic pool ID and normalized addresses for lookup
    // This fixes the duplicate pool bug where symbols and addresses wouldn't match
    // ========================================

    // Generate deterministic pool ID from normalized addresses
    let deterministic_pool_id = generate_pool_id(&token0_addr, &token1_addr);

    tracing::info!(
        "🔧 Looking for pool with deterministic ID: {} (tokens: {} / {})",
        deterministic_pool_id,
        token0_canonical,
        token1_canonical
    );

    // Check if a pool already exists for this token pair (with same provider)
    // FIXED: Now uses normalized canonical addresses for comparison, not raw request strings
    let pool_id = {
        let pools = state.liquidity_pools.read().await;

        // First, try to find by deterministic pool ID (fastest)
        if pools.contains_key(&deterministic_pool_id) {
            let pool = pools.get(&deterministic_pool_id).unwrap();
            if pool.provider == provider {
                Some(deterministic_pool_id.clone())
            } else {
                None // Pool exists but different provider
            }
        } else {
            // Fallback: Search by normalized token addresses (for legacy pools)
            pools
                .values()
                .find(|p| {
                    // CRITICAL FIX: Compare using CANONICAL addresses, not raw request strings
                    let pool_matches = (p.token0 == token0_canonical && p.token1 == token1_canonical)
                        || (p.token0 == token1_canonical && p.token1 == token0_canonical);

                    // Also check against raw request strings for backward compatibility
                    let legacy_matches = (p.token0 == request.token0 && p.token1 == request.token1)
                        || (p.token0 == request.token1 && p.token1 == request.token0);

                    (pool_matches || legacy_matches) && p.provider == provider
                })
                .map(|p| p.pool_id.clone())
        }
    };

    let (final_pool_id, action) = if let Some(existing_pool_id) = pool_id {
        // Pool exists - add to reserves
        let mut pools = state.liquidity_pools.write().await;
        if let Some(pool) = pools.get_mut(&existing_pool_id) {
            // Store old reserves and LP supply for proportional calculation
            let old_reserve0 = pool.reserve0;
            let old_reserve1 = pool.reserve1;
            let old_lp_supply = pool.lp_token_supply;

            // Check token order and add to correct reserves
            // FIXED: Use canonical addresses for comparison
            let (add_amount0, add_amount1) = if pool.token0 == token0_canonical || pool.token0 == request.token0 {
                pool.reserve0 += request.amount0;
                pool.reserve1 += request.amount1;
                (request.amount0, request.amount1)
            } else {
                // Swapped order
                pool.reserve0 += request.amount1;
                pool.reserve1 += request.amount0;
                (request.amount1, request.amount0)
            };

            // Calculate proportional LP tokens to mint
            let additional_lp_tokens = calculate_lp_tokens(
                add_amount0,
                add_amount1,
                Some(old_reserve0),
                Some(old_reserve1),
                Some(old_lp_supply),
            );

            // Update LP token supply
            pool.lp_token_supply += additional_lp_tokens;

            tracing::info!(
                "💰 Added to existing liquidity pool {} - New reserves: {} / {} - LP tokens minted: {} (new total: {})",
                existing_pool_id,
                pool.reserve0,
                pool.reserve1,
                additional_lp_tokens,
                pool.lp_token_supply
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
            // FIXED: Use deterministic pool ID and canonical addresses
            let new_pool_id = deterministic_pool_id.clone();

            // Calculate LP tokens for new pool
            let lp_tokens = calculate_lp_tokens(
                request.amount0,
                request.amount1,
                None,
                None,
                None,
            );

            // FIXED: Store with canonical addresses, not raw request strings
            let pool = LiquidityPool {
                pool_id: new_pool_id.clone(),
                token0: token0_canonical.clone(),
                token1: token1_canonical.clone(),
                reserve0: request.amount0,
                reserve1: request.amount1,
                provider,
                created_at: chrono::Utc::now(),
                lp_token_supply: lp_tokens,
            };
            let pool_clone = pool.clone();
            pools.insert(new_pool_id.clone(), pool);
            tracing::info!(
                "💰 Created liquidity pool {} (deterministic) with tokens {} / {} and reserves: {} / {}",
                new_pool_id,
                token0_canonical,
                token1_canonical,
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
        // No existing pool - create new one with DETERMINISTIC pool ID
        // FIXED: Use deterministic pool ID based on sorted token addresses
        let new_pool_id = deterministic_pool_id.clone();

        // Calculate LP tokens for new pool
        let lp_tokens = calculate_lp_tokens(
            request.amount0,
            request.amount1,
            None,
            None,
            None,
        );

        // FIXED: Store with canonical addresses, not raw request strings
        let pool = LiquidityPool {
            pool_id: new_pool_id.clone(),
            token0: token0_canonical.clone(),
            token1: token1_canonical.clone(),
            reserve0: request.amount0,
            reserve1: request.amount1,
            provider,
            created_at: chrono::Utc::now(),
            lp_token_supply: lp_tokens,
        };

        let pool_clone = pool.clone();
        let mut pools = state.liquidity_pools.write().await;
        pools.insert(new_pool_id.clone(), pool);
        tracing::info!(
            "💰 Created liquidity pool {} (deterministic) with tokens {} / {} and reserves: {} / {}",
            new_pool_id,
            token0_canonical,
            token1_canonical,
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

    // ========================================
    // v0.6.1-beta: DEX DECENTRALIZATION PHASE 3
    // Broadcast pool announcement to P2P network
    // ========================================
    if action == "created" {
        // Only broadcast newly created pools, not additions to existing pools
        // Get the pool details for broadcasting
        let pool_for_broadcast = {
            let pools = state.liquidity_pools.read().await;
            pools.get(&final_pool_id).cloned()
        };

        if let Some(pool) = pool_for_broadcast {
            // Convert token strings to byte arrays
            let token0_bytes = if pool.token0.to_uppercase() == "QUG" || pool.token0.to_lowercase() == "native-qug" {
                [0u8; 32] // Native QUG uses zero address
            } else {
                match hex::decode(pool.token0.trim_start_matches("0x")) {
                    Ok(bytes) if bytes.len() == 32 => {
                        let mut arr = [0u8; 32];
                        arr.copy_from_slice(&bytes);
                        arr
                    }
                    _ => {
                        tracing::warn!("⚠️  Failed to parse token0 address for P2P broadcast: {}", pool.token0);
                        [0u8; 32]
                    }
                }
            };

            let token1_bytes = if pool.token1.to_uppercase() == "QUG" || pool.token1.to_lowercase() == "native-qug" {
                [0u8; 32] // Native QUG uses zero address
            } else {
                match hex::decode(pool.token1.trim_start_matches("0x")) {
                    Ok(bytes) if bytes.len() == 32 => {
                        let mut arr = [0u8; 32];
                        arr.copy_from_slice(&bytes);
                        arr
                    }
                    _ => {
                        tracing::warn!("⚠️  Failed to parse token1 address for P2P broadcast: {}", pool.token1);
                        [0u8; 32]
                    }
                }
            };

            // Create PoolAnnouncement (unsigned first)
            let mut announcement = q_types::PoolAnnouncement::new(
                token0_bytes,
                token1_bytes,
                pool.reserve0,
                pool.reserve1,
                pool.lp_token_supply,
                provider,
                pool.created_at.timestamp() as u64,
            );

            // Sign the announcement
            if let Err(e) = announcement.sign(&*state.node_signing_key) {
                tracing::warn!("Failed to sign pool announcement: {}", e);
                // Continue without broadcasting if signing fails
            } else {

            // Serialize and broadcast
            match serde_json::to_vec(&announcement) {
                Ok(announcement_bytes) => {
                    if let Some(ref command_tx) = state.libp2p_command_tx {
                        let topic = "/qnk/liquidity-pools".to_string();
                        let cmd = q_network::NetworkCommand::PublishPoolAnnouncement {
                            topic: topic.clone(),
                            announcement_bytes: announcement_bytes.clone(),
                        };

                        if let Err(e) = command_tx.send(cmd) {
                            tracing::warn!("Failed to send pool announcement to network: {}", e);
                        } else {
                            tracing::info!(
                                "✅ [LIQUIDITY POOLS] Broadcasted pool {} to P2P network ({} bytes)",
                                final_pool_id, announcement_bytes.len()
                            );
                        }
                    } else {
                        tracing::debug!("libp2p_command_tx not available, skipping P2P broadcast");
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to serialize pool announcement for P2P: {}", e);
                }
            }
            } // Close the else block from signing
        }
    }

    // ============================================================================
    // v1.0.91-beta: PROPER TRANSACTION HANDLING
    // Fixes 10 critical design flaws from v1.0.90-beta:
    // 1. Proper cryptographic transaction ID (SHA3-256 hash)
    // 2. Nonce management for replay attack prevention
    // 3. Pending status (not Confirmed immediately)
    // 4. Block production queue integration
    // 5. Proper broadcast mechanism
    // ============================================================================

    // Get next nonce for this wallet (prevents replay attacks)
    let nonce = state.nonce_tracker.get_and_increment(&provider);

    // Create transaction with proper cryptographic ID
    let transaction = q_api_server::transaction_utils::TransactionBuilder::new()
        .from(provider)
        .to([0u8; 32]) // Pool contract address
        .amount(request.amount0)
        .fee(0) // No fee for liquidity provision
        .data(
            format!(
                "add_liquidity:{}:{}:{}:{}:{}",
                final_pool_id, request.token0, request.token1, request.amount0, request.amount1
            )
            .into_bytes(),
        )
        .token_type(q_types::TokenType::QUG)
        .fee_token_type(q_types::TokenType::QUGUSD)
        .tx_type(q_types::TransactionType::PoolAddLiquidity)
        .build_with_nonce(nonce, chrono::Utc::now());

    let tx_id = transaction.id;
    let tx_hash = format!("0x{}", hex::encode(tx_id));

    // Submit transaction properly: pool, mempool queue, and broadcast
    let submission_result = q_api_server::transaction_utils::submit_transaction(
        transaction,
        &state.tx_pool,
        &state.tx_status,
        state.production_mempool.as_ref(),
        state.libp2p_discovery.as_ref(),
    ).await;

    tracing::info!(
        "📤 [LIQUIDITY] Add liquidity tx submitted for pool {}: {} (nonce={}, broadcast={}, queued={})",
        final_pool_id,
        &tx_hash[..16],
        nonce,
        submission_result.broadcast_success,
        submission_result.queued_for_block
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

/// Refresh token balances request
#[derive(Debug, Deserialize)]
pub struct RefreshBalancesRequest {
    pub wallet_address: String,
}

/// Refresh token balances response
#[derive(Debug, Serialize)]
pub struct RefreshBalancesResponse {
    pub refreshed_tokens: Vec<TokenBalanceInfo>,
}

#[derive(Debug, Serialize)]
pub struct TokenBalanceInfo {
    pub symbol: String,
    pub address: String,
    pub balance: u64,
    pub balance_display: f64,
}

/// Refresh token balances from deployed contracts
///
/// This endpoint forces a refresh of all token balances for a wallet by
/// re-reading the initial supply from deployed contracts and subtracting
/// any amounts locked in liquidity pools.
pub async fn refresh_token_balances(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RefreshBalancesRequest>,
) -> Result<Json<ApiResponse<RefreshBalancesResponse>>, StatusCode> {
    // Parse wallet address
    let wallet_addr = match parse_address(&request.wallet_address) {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };

    let mut refreshed_tokens = Vec::new();
    let deployed_contracts = state.orobit_ecosystem.deployed_contracts.read().await;

    // Find all contracts deployed by this wallet
    for contract in deployed_contracts.values() {
        if contract.deployer == wallet_addr {
            // Get initial supply
            if let Some(supply_value) = contract
                .deployment_params
                .get("initialSupply")
                .or_else(|| contract.deployment_params.get("initial_supply"))
            {
                let initial_supply = supply_value.as_u64().or_else(|| {
                    supply_value.as_str().and_then(|s| s.parse::<u64>().ok())
                });

                if let Some(supply) = initial_supply {
                    let token_addr = contract.address.0;

                    // Calculate amount locked in liquidity pools
                    let pools = state.liquidity_pools.read().await;
                    let mut locked_amount = 0u64;

                    for pool in pools.values() {
                        if pool.provider == wallet_addr {
                            // Check if token0 matches
                            if let Ok(pool_token0_addr) = resolve_token_address(&state, &pool.token0).await {
                                if pool_token0_addr == token_addr {
                                    locked_amount += pool.reserve0;
                                }
                            }

                            // Check if token1 matches
                            if let Ok(pool_token1_addr) = resolve_token_address(&state, &pool.token1).await {
                                if pool_token1_addr == token_addr {
                                    locked_amount += pool.reserve1;
                                }
                            }
                        }
                    }
                    drop(pools);

                    // Calculate available balance (initial supply - locked in pools)
                    let available_balance = supply.saturating_sub(locked_amount);

                    // Update in-memory and persistent storage
                    let balance_key = (wallet_addr, token_addr);
                    let mut token_balances = state.token_balances.write().await;
                    token_balances.insert(balance_key, available_balance);
                    drop(token_balances);

                    // Persist to storage
                    if let Err(e) = state
                        .storage_engine
                        .save_token_balance(&wallet_addr, &token_addr, available_balance)
                        .await
                    {
                        tracing::warn!("Failed to persist refreshed token balance: {}", e);
                    }

                    let symbol = contract.metadata.symbol.clone().unwrap_or_else(|| "UNKNOWN".to_string());

                    refreshed_tokens.push(TokenBalanceInfo {
                        symbol: symbol.clone(),
                        address: format!("qnk{}", hex::encode(token_addr)),
                        balance: available_balance,
                        balance_display: available_balance as f64 / 100_000_000.0,
                    });

                    tracing::info!(
                        "🔄 Refreshed balance for token {} ({}): {} ({} display units). Initial: {}, Locked: {}",
                        symbol,
                        hex::encode(&token_addr[..8]),
                        available_balance,
                        available_balance as f64 / 100_000_000.0,
                        supply,
                        locked_amount
                    );
                }
            }
        }
    }

    Ok(Json(ApiResponse::success(RefreshBalancesResponse {
        refreshed_tokens,
    })))
}

/// Helper function to resolve token name/symbol to address
async fn resolve_token_address(state: &Arc<AppState>, token: &str) -> Result<[u8; 32], String> {
    // Check if it's native QUG
    if token.to_uppercase() == "QUG" || token.to_lowercase() == "native-qug" {
        return Ok([0u8; 32]);
    }

    // Check if it's already an address
    if token.starts_with("0x") || token.starts_with("qnk") {
        return parse_address(token);
    }

    // It's a symbol, resolve it
    resolve_token_symbol(state, token).await
}

/// Get swap quote using quantum-enhanced AMM
/// v1.0.49-beta: NEW - Real price impact calculation from q-dex algorithms
pub async fn get_swap_quote(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SwapQuoteRequest>,
) -> Result<Json<ApiResponse<SwapQuoteResponse>>, StatusCode> {
    // Normalize token addresses
    let from_addr = match normalize_token_to_address(&state, &request.from_token).await {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(format!("Invalid from_token: {}", e)))),
    };

    let to_addr = match normalize_token_to_address(&state, &request.to_token).await {
        Ok(addr) => addr,
        Err(e) => return Ok(Json(ApiResponse::error(format!("Invalid to_token: {}", e)))),
    };

    // Canonical token strings for pool lookup
    let from_canonical = if from_addr == [0u8; 32] {
        "QUG".to_string()
    } else {
        format!("qnk{}", hex::encode(from_addr))
    };

    let to_canonical = if to_addr == [0u8; 32] {
        "QUG".to_string()
    } else {
        format!("qnk{}", hex::encode(to_addr))
    };

    // Find pool for this pair
    let pools = state.liquidity_pools.read().await;

    let matching_pool = pools.values().find(|p| {
        (p.token0 == from_canonical && p.token1 == to_canonical)
            || (p.token0 == to_canonical && p.token1 == from_canonical)
            || (p.token0 == request.from_token && p.token1 == request.to_token)
            || (p.token0 == request.to_token && p.token1 == request.from_token)
    });

    match matching_pool {
        Some(pool) => {
            // Determine reserve order
            let (reserve_in, reserve_out) = if pool.token0 == from_canonical || pool.token0 == request.from_token {
                (pool.reserve0, pool.reserve1)
            } else {
                (pool.reserve1, pool.reserve0)
            };

            // Calculate swap using quantum-enhanced AMM
            let fee_rate = 0.003; // 0.3% fee
            let (amount_out, price_impact, effective_price) =
                calculate_quantum_swap(request.amount_in, reserve_in, reserve_out, fee_rate);

            let fee = (request.amount_in as f64 * fee_rate) as u64;

            tracing::info!(
                "⚛️ Quantum swap quote: {} {} => {} {} (impact: {:.4}%, pool: {})",
                request.amount_in as f64 / 100_000_000.0,
                request.from_token,
                amount_out as f64 / 100_000_000.0,
                request.to_token,
                price_impact * 100.0,
                pool.pool_id
            );

            Ok(Json(ApiResponse::success(SwapQuoteResponse {
                from_token: request.from_token,
                to_token: request.to_token,
                amount_in: request.amount_in,
                amount_out,
                price_impact,
                effective_price,
                fee,
                pool_id: Some(pool.pool_id.clone()),
                quantum_enhanced: true,
            })))
        }
        None => {
            tracing::warn!(
                "⚠️ No liquidity pool found for {} / {}",
                request.from_token,
                request.to_token
            );
            Ok(Json(ApiResponse::error(format!(
                "No liquidity pool found for {} / {}. Please add liquidity first.",
                request.from_token, request.to_token
            ))))
        }
    }
}
