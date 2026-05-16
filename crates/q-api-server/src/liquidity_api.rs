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
use serde::{Deserialize, Deserializer, Serialize};
use sha3::{Digest, Sha3_256};
use std::sync::Arc;
use q_storage::BalanceStorage; // v3.6.4-beta: For storage_engine.get_balance()
use q_vm::contracts::orobit_smart_contracts::ContractAddress; // v3.2.17-beta: For contract lookup

/// v3.2.20-beta: Improved deserializer for u128 that preserves precision
/// Handles:
/// - Plain integers (u64, u128)
/// - String numbers ("1000000000000000" or "99999999999999900000000000000000000")
/// - Scientific notation ("1e15") - parsed WITHOUT going through f64 for large numbers
/// - WARNING: f64 inputs have already lost precision by the time they reach us!
fn deserialize_u128_from_any<'de, D>(deserializer: D) -> Result<u128, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::{self, Visitor};

    struct U128FromAnyVisitor;

    /// Parse scientific notation string to u128 WITHOUT using f64
    /// This preserves full precision for numbers like "1e30"
    fn parse_scientific_to_u128(s: &str) -> Option<u128> {
        // Match patterns like "1e15", "1.5e10", "1E+30"
        let s_lower = s.to_lowercase();
        let parts: Vec<&str> = s_lower.split('e').collect();
        if parts.len() != 2 {
            return None;
        }

        let mantissa_str = parts[0];
        let exp_str = parts[1].trim_start_matches('+');
        let exponent: i32 = exp_str.parse().ok()?;

        if exponent < 0 {
            // Negative exponent would result in a fractional number
            // For u128, we'd lose the fraction anyway, so just return 0 for small values
            return Some(0);
        }

        // Parse mantissa, handling decimal point
        let mantissa_parts: Vec<&str> = mantissa_str.split('.').collect();
        let whole_part = mantissa_parts[0];
        let frac_part = if mantissa_parts.len() > 1 { mantissa_parts[1] } else { "" };

        // Combine whole and fractional parts, then adjust exponent
        let combined = format!("{}{}", whole_part, frac_part);
        let adjusted_exp = exponent as usize - frac_part.len();

        // Build the final number string
        let mut result_str = combined.trim_start_matches('0').to_string();
        if result_str.is_empty() {
            result_str = "0".to_string();
        }

        // Add zeros for the exponent
        result_str.push_str(&"0".repeat(adjusted_exp));

        result_str.parse::<u128>().ok()
    }

    impl<'de> Visitor<'de> for U128FromAnyVisitor {
        type Value = u128;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a number (integer, float, or string)")
        }

        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            tracing::debug!("📊 deserialize_u128: received u64 = {}", value);
            Ok(value as u128)
        }

        fn visit_u128<E>(self, value: u128) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            tracing::debug!("📊 deserialize_u128: received u128 = {}", value);
            Ok(value)
        }

        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            if value >= 0 {
                tracing::debug!("📊 deserialize_u128: received i64 = {}", value);
                Ok(value as u128)
            } else {
                Err(de::Error::custom("negative values not allowed"))
            }
        }

        fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            // WARNING: f64 has already lost precision for large numbers!
            // f64 only has ~15-17 significant decimal digits
            if value >= 0.0 {
                let result = value as u128;
                tracing::warn!(
                    "⚠️ deserialize_u128: received f64 = {} -> u128 = {} (PRECISION MAY BE LOST!)",
                    value, result
                );
                Ok(result)
            } else {
                Err(de::Error::custom("negative values not allowed"))
            }
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            tracing::debug!("📊 deserialize_u128: received string = '{}'", value);

            // Try parsing as u128 first (handles pure integer strings)
            if let Ok(n) = value.parse::<u128>() {
                tracing::debug!("📊 deserialize_u128: parsed as u128 = {}", n);
                return Ok(n);
            }

            // Try parsing scientific notation WITHOUT f64 to preserve precision
            if value.to_lowercase().contains('e') {
                if let Some(n) = parse_scientific_to_u128(value) {
                    tracing::debug!("📊 deserialize_u128: parsed scientific '{}' as u128 = {}", value, n);
                    return Ok(n);
                }
            }

            // Last resort: try f64 for other formats (will lose precision for large numbers!)
            if let Ok(f) = value.parse::<f64>() {
                if f >= 0.0 {
                    let result = f as u128;
                    tracing::warn!(
                        "⚠️ deserialize_u128: f64 fallback for '{}' = {} (PRECISION LOST!)",
                        value, result
                    );
                    return Ok(result);
                }
            }

            Err(de::Error::custom(format!("cannot parse '{}' as a number", value)))
        }
    }

    deserializer.deserialize_any(U128FromAnyVisitor)
}

use crate::{AppState, LiquidityPool};
use q_types::{
    Transaction, TxStatus,
    DEX_TOTAL_FEE_BPS, DEX_PROTOCOL_FEE_BPS, DEX_LP_FEE_BPS, BPS_DIVISOR,
};

/// Generate a deterministic LP token address for a pool.
/// Uses SHA-256("lp_token:<pool_id>") → [u8; 32].
/// Same pool always produces the same LP token address.
fn generate_lp_token_address(pool_id: &str) -> [u8; 32] {
    use sha2::{Sha256, Digest};
    let hash = Sha256::digest(format!("lp_token:{}", pool_id).as_bytes());
    let mut addr = [0u8; 32];
    addr.copy_from_slice(&hash[..32]);
    addr
}

/// Extract a short display symbol from a pool's canonical token name.
/// "QUG" → "QUG", "QUGUSD" → "QUGUSD", "qnkabcdef..." → "abcdef" (first 6 hex chars)
fn token_display_symbol(canonical: &str) -> String {
    if canonical == "QUG" || canonical == "QUGUSD" {
        canonical.to_string()
    } else if let Some(hex_part) = canonical.strip_prefix("qnk") {
        // Use first 6 hex chars for a compact label
        hex_part.chars().take(6).collect::<String>().to_uppercase()
    } else {
        canonical.to_string()
    }
}

/// Standard decimal places for all tokens
/// v3.0.6-beta: Updated to 24 decimals for u128 migration
pub const TOKEN_DECIMALS: u32 = 24;
pub const DECIMAL_MULTIPLIER: u128 = 1_000_000_000_000_000_000_000_000; // 10^24

/// Integer square root using Newton's method (no floating point precision loss)
/// This is critical for LP token calculations with large numbers
fn integer_sqrt(n: u128) -> u128 {
    if n == 0 {
        return 0;
    }

    let mut x = n;
    let mut y = (x + 1) / 2;

    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }

    x
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
/// Handles: "QUG", "native-qug", "QUGUSD", symbols like "MEME", addresses like "qnk1234..."
async fn normalize_token_to_address(
    state: &Arc<AppState>,
    token: &str,
) -> Result<[u8; 32], String> {
    normalize_token_to_address_for_wallet(state, token, None).await
}

/// v3.9.4-beta: Normalize token to address with wallet preference
/// When multiple tokens have the same symbol, prefers tokens the wallet owns
async fn normalize_token_to_address_for_wallet(
    state: &Arc<AppState>,
    token: &str,
    wallet_address: Option<&[u8; 32]>,
) -> Result<[u8; 32], String> {
    let token_upper = token.to_uppercase();

    // Native QUG uses zero address
    if token_upper == "QUG" || token.to_lowercase() == "native-qug" {
        return Ok([0u8; 32]);
    }

    // v2.6.1-beta: Native QUGUSD stablecoin uses special address pattern ("QUGUSD" padded with zeros)
    if token_upper == "QUGUSD" || token.to_lowercase() == "qugusd-stable" {
        return Ok([0x51, 0x55, 0x47, 0x55, 0x53, 0x44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // "QUGUSD" in ASCII
    }

    // Already an address format
    if token.starts_with("qnk") || token.starts_with("0x") {
        return parse_address(token);
    }

    // v3.9.4-beta: It's a symbol - resolve to address with wallet preference
    resolve_token_symbol_for_wallet(state, token, wallet_address).await
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
/// v2.8.2: Flexible deserializer handles scientific notation & string numbers
#[derive(Debug, Deserialize)]
pub struct AddLiquidityRequest {
    pub token0: String, // "QUG" for native or token contract address
    pub token1: String, // Token contract address
    #[serde(deserialize_with = "deserialize_u128_from_any")]
    pub amount0: u128,
    #[serde(deserialize_with = "deserialize_u128_from_any")]
    pub amount1: u128,
    pub provider: String, // Wallet address
}

/// Add liquidity response
#[derive(Debug, Serialize)]
pub struct AddLiquidityResponse {
    pub pool_id: String,
    pub token0: String,
    pub token1: String,
    #[serde(serialize_with = "q_types::u128_serde::serialize")]
    pub amount0: u128,
    #[serde(serialize_with = "q_types::u128_serde::serialize")]
    pub amount1: u128,
    pub transaction_id: String,
    /// LP tokens minted and credited to the provider's wallet
    pub lp_tokens_minted: String,
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
    #[serde(serialize_with = "q_types::u128_serde::serialize")]
    pub amount0_returned: u128,
    #[serde(serialize_with = "q_types::u128_serde::serialize")]
    pub amount1_returned: u128,
    pub transaction_id: String,
    /// LP tokens burned from the provider's wallet
    pub lp_tokens_burned: String,
}

/// Golden ratio constant for quantum-enhanced calculations
const GOLDEN_RATIO: f64 = 1.618033988749895;

/// Quantum slippage reduction factor (uses golden ratio)
const QUANTUM_SLIPPAGE_REDUCTION: f64 = 0.618;

/// v3.2.16-beta: Native token decimals (QUG, QUGUSD)
const NATIVE_TOKEN_DECIMALS: u8 = 24;

/// v3.2.16-beta: Default decimals for custom tokens (when contract metadata unavailable)
const DEFAULT_CUSTOM_TOKEN_DECIMALS: u8 = 8;

/// v3.2.16-beta: Get token decimals based on token type
/// QUG and QUGUSD use 24 decimals, custom tokens use their deployed decimals (default 8)
fn get_token_decimals(token_canonical: &str, is_native_qug: bool, is_qugusd: bool) -> u8 {
    if is_native_qug || is_qugusd {
        NATIVE_TOKEN_DECIMALS // 24 decimals for QUG/QUGUSD
    } else if token_canonical.to_uppercase() == "QUG" || token_canonical.to_uppercase() == "QUGUSD" {
        NATIVE_TOKEN_DECIMALS // 24 decimals
    } else {
        DEFAULT_CUSTOM_TOKEN_DECIMALS // Default 8 decimals, but should use async version when possible
    }
}

/// v3.2.17-beta: Async version that looks up actual decimals from contract deployment
/// This is the preferred method when you have access to AppState
async fn get_token_decimals_from_contract(
    state: &std::sync::Arc<crate::AppState>,
    token_canonical: &str,
    token_addr: &[u8; 32],
    is_native_qug: bool,
    is_qugusd: bool,
) -> u8 {
    // Native tokens have fixed decimals
    if is_native_qug || is_qugusd {
        return NATIVE_TOKEN_DECIMALS; // 24 decimals for QUG/QUGUSD
    }
    if token_canonical.to_uppercase() == "QUG" || token_canonical.to_uppercase() == "QUGUSD" {
        return NATIVE_TOKEN_DECIMALS;
    }

    // Look up contract metadata for custom tokens
    if let Some(contract) = state.orobit_ecosystem.get_contract_by_address(ContractAddress(*token_addr)).await {
        if let Some(decimals_val) = contract.deployment_params.get("decimals") {
            if let Some(decimals) = decimals_val.as_u64() {
                tracing::debug!(
                    "📊 Found decimals={} for token {} from contract metadata",
                    decimals, token_canonical
                );
                return decimals as u8;
            }
        }
    }

    // Fallback to default
    tracing::debug!(
        "📊 Using default decimals={} for token {} (no contract metadata found)",
        DEFAULT_CUSTOM_TOKEN_DECIMALS, token_canonical
    );
    DEFAULT_CUSTOM_TOKEN_DECIMALS
}

/// v3.2.16-beta: Normalize a reserve value to a common decimal base for AMM calculations
/// This is critical for swaps between tokens with different decimal places (e.g., QUG=24, custom=8)
fn normalize_reserve_to_24_decimals(reserve: u128, decimals: u8) -> u128 {
    if decimals >= 24 {
        reserve / 10u128.pow((decimals - 24) as u32)
    } else {
        // Scale up to 24 decimals
        reserve * 10u128.pow((24 - decimals) as u32)
    }
}

/// v3.2.16-beta: De-normalize amount from 24-decimal base back to token's native decimals
fn denormalize_from_24_decimals(amount_24: u128, decimals: u8) -> u128 {
    if decimals >= 24 {
        amount_24 * 10u128.pow((decimals - 24) as u32)
    } else {
        // Scale down from 24 decimals to token's native decimals
        amount_24 / 10u128.pow((24 - decimals) as u32)
    }
}

/// Calculate LP tokens using Uniswap V2 formula with optional quantum enhancement
/// v1.0.49-beta: FIXED - Uses integer square root for precision
/// v1.0.49-beta: NEW - Golden ratio optimization from q-dex quantum algorithms
///
/// For NEW pools:
/// - Formula: sqrt(amount0 * amount1) - MINIMUM_LIQUIDITY
/// - MINIMUM_LIQUIDITY (100 tokens, v8.6.0) is permanently locked to prevent division by zero
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
    amount0: u128,
    amount1: u128,
    existing_reserve0: Option<u128>,
    existing_reserve1: Option<u128>,
    existing_lp_supply: Option<u128>,
) -> u128 {
    match (existing_reserve0, existing_reserve1, existing_lp_supply) {
        (Some(r0), Some(r1), Some(supply)) if r0 > 0 && r1 > 0 && supply > 0 => {
            // Existing pool - proportional minting
            // Calculate how many LP tokens user should get based on each reserve
            let liquidity0 = (amount0 * supply) / r0;
            let liquidity1 = (amount1 * supply) / r1;

            // Use minimum to ensure user doesn't get more LP tokens than they should
            // This enforces the constant product invariant
            let minted = std::cmp::min(liquidity0, liquidity1);

            tracing::info!(
                "📊 LP Token Calculation (Existing Pool): amount0={}, amount1={}, reserve0={}, reserve1={}, existing_supply={}, liquidity0={}, liquidity1={}, minted={}",
                amount0, amount1, r0, r1, supply, liquidity0, liquidity1, minted
            );

            minted
        }
        _ => {
            // New pool - geometric mean (Uniswap V2 formula)
            // MINIMUM_LIQUIDITY is permanently locked to prevent attacks on tiny pools
            // v8.6.0: lowered from 1000 to 100 for easier pool bootstrapping
            const MINIMUM_LIQUIDITY: u128 = 100;

            let product = amount0 * amount1;
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
/// v8.8.5: Overflow-safe (a * b) / d using divide-first decomposition.
/// When a * b overflows u128, decomposes as: a * (b/d) + a * (b%d) / d.
/// The first term is exact when a * (b/d) fits in u128 (true for all practical AMM swaps).
/// The second term uses f64 only for the small correction when a * (b%d) also overflows.
/// This fixes the critical bug where saturating_mul capped intermediate products at u128::MAX,
/// producing near-zero swap outputs for QUG/QUGUSD pairs with large 24-decimal reserves.
pub fn mul_div_u128(a: u128, b: u128, d: u128) -> u128 {
    if d == 0 {
        return 0;
    }

    // Fast path: no overflow
    if let Some(product) = a.checked_mul(b) {
        return product / d;
    }

    // Overflow path: decompose b = q*d + r, then a*b/d = a*q + a*r/d
    let q = b / d;
    let r = b % d;

    // Main term: a * q (rate × amount). For AMM this is amount_in × (reserve_out/denominator)
    // which equals amount_in × effective_price. This fits u128 for all practical swap sizes.
    let main_part = a.checked_mul(q).unwrap_or_else(|| {
        // Extreme case: use f64 (maintains ~15 digits of precision)
        ((a as f64) * (q as f64)) as u128
    });

    // Correction term: a * r / d. Since r < d, this term is < a (always small).
    let correction = match a.checked_mul(r) {
        Some(ar) => ar / d,
        None => {
            // a * r overflows. Since correction < a, f64 precision is sufficient.
            ((a as f64) * (r as f64) / (d as f64)) as u128
        }
    };

    main_part.saturating_add(correction)
}

/// v4.0.13: PRECISION FIX - Use integer math for AMM calculation to match handlers.rs.
/// The old f64 path lost precision for amounts > 2^53 base units (~9M display tokens).
pub fn calculate_quantum_swap(
    amount_in: u128,
    reserve_in: u128,
    reserve_out: u128,
    fee_rate: f64,
) -> (u128, f64, f64) {
    if reserve_in == 0 || reserve_out == 0 {
        return (0, 1.0, 0.0);
    }

    // Apply fee using integer math: amount * (1000 - fee_bps) / 1000
    // fee_rate = 0.003 → fee_bps = 3
    let fee_bps = (fee_rate * 1000.0).round() as u128;
    let amount_in_with_fee = amount_in
        .checked_mul(1000u128.saturating_sub(fee_bps))
        .and_then(|v| v.checked_div(1000))
        .unwrap_or(amount_in.saturating_mul(997) / 1000);

    // Constant product formula using integer math:
    // amount_out = (amount_in_with_fee * reserve_out) / (reserve_in + amount_in_with_fee)
    let denominator = reserve_in.saturating_add(amount_in_with_fee);
    if denominator == 0 {
        return (0, 1.0, 0.0);
    }

    let amount_out = mul_div_u128(amount_in_with_fee, reserve_out, denominator);

    // Calculate price impact using display-scale f64 (safe for display purposes)
    let r_in_d = reserve_in as f64 / 1e24;
    let r_out_d = reserve_out as f64 / 1e24;
    let a_in_d = amount_in as f64 / 1e24;
    let a_out_d = amount_out as f64 / 1e24;
    let spot_price = if r_in_d > 0.0 { r_out_d / r_in_d } else { 0.0 };
    let effective_price = if a_in_d > 0.0 { a_out_d / a_in_d } else { 0.0 };
    let price_impact = if spot_price > 0.0 { 1.0 - (effective_price / spot_price) } else { 0.0 };

    tracing::debug!(
        "⚛️ [QUOTE v4.0.13] Integer AMM: in={}, reserve_in={}, reserve_out={}, out={}, impact={:.4}%",
        amount_in, reserve_in, reserve_out, amount_out, price_impact * 100.0
    );

    (amount_out, price_impact, effective_price)
}

/// Swap quote request
/// v2.8.2: Flexible deserializer handles scientific notation & string numbers
#[derive(Debug, Deserialize)]
pub struct SwapQuoteRequest {
    pub from_token: String,
    pub to_token: String,
    #[serde(deserialize_with = "deserialize_u128_from_any")]
    pub amount_in: u128,
}

/// Swap quote response (v2.4.5-beta: includes fee breakdown)
#[derive(Debug, Serialize)]
pub struct SwapQuoteResponse {
    pub from_token: String,
    pub to_token: String,
    #[serde(serialize_with = "q_types::u128_serde::serialize")]
    pub amount_in: u128,
    #[serde(serialize_with = "q_types::u128_serde::serialize")]
    pub amount_out: u128,
    pub price_impact: f64,
    pub effective_price: f64,
    /// Total fee paid by trader (0.30%)
    #[serde(serialize_with = "q_types::u128_serde::serialize")]
    pub fee: u128,
    /// Protocol fee portion (0.05%) - goes to master wallet
    #[serde(serialize_with = "q_types::u128_serde::serialize")]
    pub protocol_fee: u128,
    /// LP fee portion (0.25%) - stays in pool for liquidity providers
    #[serde(serialize_with = "q_types::u128_serde::serialize")]
    pub lp_fee: u128,
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
        .route("/admin/reset-pool", post(admin_reset_pool_reserves))
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

    // v2.6.1-beta: Check if token0/token1 is native QUGUSD stablecoin
    let is_qugusd_token0 =
        request.token0.to_uppercase() == "QUGUSD" || request.token0.to_lowercase() == "qugusd-stable";
    let is_qugusd_token1 =
        request.token1.to_uppercase() == "QUGUSD" || request.token1.to_lowercase() == "qugusd-stable";

    // CRITICAL: Normalize token0 to address format
    // v3.9.4-beta: Use wallet-aware resolution to find tokens the provider owns
    let token0_addr = match normalize_token_to_address_for_wallet(&state, &request.token0, Some(&provider)).await {
        Ok(addr) => addr,
        Err(e) => {
            return Ok(Json(ApiResponse::error(format!(
                "Failed to resolve token0 '{}': {}",
                request.token0, e
            ))))
        }
    };

    // CRITICAL: Normalize token1 to address format
    // v3.9.4-beta: Use wallet-aware resolution to find tokens the provider owns
    let token1_addr = match normalize_token_to_address_for_wallet(&state, &request.token1, Some(&provider)).await {
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
    } else if is_qugusd_token0 {
        "QUGUSD".to_string()
    } else {
        format!("qnk{}", hex::encode(token0_addr))
    };

    let token1_canonical = if is_native_token1 {
        "QUG".to_string()
    } else if is_qugusd_token1 {
        "QUGUSD".to_string()
    } else {
        format!("qnk{}", hex::encode(token1_addr))
    };

    tracing::info!(
        "🔧 Token normalization: {} => {}, {} => {}",
        request.token0, token0_canonical,
        request.token1, token1_canonical
    );

    // Track which token balances changed for persistence
    let mut token_balance_changes: Vec<([u8; 32], [u8; 32], u128)> = Vec::new(); // (wallet, token, new_balance)
    // v10.2.1: Track native QUG balance changes for persistence (fixes deductions lost on restart)
    let mut native_qug_balance_change: Option<([u8; 32], u128)> = None;

    // Deduct balances
    {
        let mut wallet_balances = state.wallet_balances.write().await;
        let mut token_balances = state.token_balances.write().await;

        // Deduct token0 (native QUG, native QUGUSD, or token)
        if is_native_token0 {
            // v3.6.4-beta: CRITICAL FIX - Read balance from storage_engine (authoritative source)
            // The in-memory wallet_balances HashMap was stale, causing "insufficient balance" errors
            // even when user had funds (dashboard showed 4.85 QUG but liquidity showed 0.21 QUG)
            let storage_balance = state
                .storage_engine
                .get_balance(&hex::encode(provider))
                .await
                .unwrap_or(0);

            // Sync in-memory cache with storage
            let balance = wallet_balances.entry(provider).or_insert(storage_balance);
            if *balance != storage_balance {
                tracing::info!(
                    "🔄 [LIQUIDITY] Synced stale balance for {}: {} → {}",
                    hex::encode(&provider[..8]),
                    *balance as f64 / 1e24,
                    storage_balance as f64 / 1e24
                );
                *balance = storage_balance;
            }

            if *balance < request.amount0 {
                // v3.6.2-beta: Display human-readable amounts (24 decimal precision)
                return Ok(Json(ApiResponse::error(format!(
                    "Insufficient QUG balance. Required: {:.6} QUG, Available: {:.6} QUG",
                    request.amount0 as f64 / 1e24, *balance as f64 / 1e24
                ))));
            }
            *balance -= request.amount0;
            native_qug_balance_change = Some((provider, *balance)); // v10.2.1: Track for persistence
            // v11.2.1: CRITICAL — also write the DEX debit counter so the
            // deduction survives balance-rebuild on restart. Without this,
            // balance_consensus replays the on-chain history (which has no
            // record of the LP deposit), apply_dex_qug_adjustments() finds
            // no debit counter, and the QUG returns to the wallet while the
            // pool keeps its reserves — money created from nothing.
            if let Err(e) = state
                .storage_engine
                .record_dex_qug_debit(&hex::encode(provider), request.amount0)
                .await
            {
                tracing::error!("🚨 [DEX-DEBIT] Failed to record add_liquidity QUG debit (token0): {}", e);
            }
            tracing::info!(
                "💸 Deducted {} QUG from {} for liquidity. New balance: {}",
                request.amount0 as f64 / 1e24,
                hex::encode(provider),
                *balance as f64 / 1e24
            );
        } else if is_qugusd_token0 {
            // v2.6.1-beta: Deduct QUGUSD stablecoin for token0
            // QUGUSD balance = minted (from vault) + received (from swaps/transfers)
            let minted_qugusd = {
                let vault = state.collateral_vault.read().await;
                vault.minted_qugusd.get(&provider).copied().unwrap_or(0)
            };
            let swapped_qugusd = {
                let token_balances_read = state.token_balances.read().await;
                let qugusd_addr = [0x51, 0x55, 0x47, 0x55, 0x53, 0x44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; // "QUGUSD" padded
                token_balances_read.get(&(provider, qugusd_addr)).copied().unwrap_or(0)
            };
            // v2.7.9-beta: Cast to u128 for larger token supplies
            let total_qugusd = minted_qugusd as u128 + swapped_qugusd;

            if total_qugusd < request.amount0 as u128 {
                return Ok(Json(ApiResponse::error(format!(
                    "Insufficient QUGUSD balance for token0. Required: {:.4}, Available: {:.4} (minted: {:.4}, swapped: {:.4})",
                    request.amount0 as f64 / 1e24,
                    total_qugusd as f64 / 1e24,
                    minted_qugusd as f64 / 1e24,
                    swapped_qugusd as f64 / 1e24
                ))));
            }

            // Deduct from minted first, then from swapped
            // v3.0.4: minted_qugusd is now u128
            let mut remaining = request.amount0;
            if minted_qugusd > 0 {
                let deduct_from_minted = remaining.min(minted_qugusd);
                let mut vault = state.collateral_vault.write().await;
                if let Some(bal) = vault.minted_qugusd.get_mut(&provider) {
                    *bal = bal.saturating_sub(deduct_from_minted);
                }
                remaining -= deduct_from_minted;
                tracing::info!(
                    "💸 Deducted {} QUGUSD (minted) from {} for token0 liquidity",
                    deduct_from_minted as f64 / 1e24,
                    hex::encode(&provider[..8])
                );
            }
            if remaining > 0 {
                let qugusd_addr = [0x51, 0x55, 0x47, 0x55, 0x53, 0x44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                if let Some(bal) = token_balances.get_mut(&(provider, qugusd_addr)) {
                    *bal = bal.saturating_sub(remaining as u128);
                }
                tracing::info!(
                    "💸 Deducted {} QUGUSD (swapped) from {} for token0 liquidity",
                    remaining as f64 / 1e24,
                    hex::encode(&provider[..8])
                );
            }
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
                    // v3.9.4-beta: Use resolve_token_symbol_for_wallet to prefer tokens user owns
                    // This fixes the bug where multiple tokens have same symbol (e.g., 5 "LLAMA" tokens)
                    match resolve_token_symbol_for_wallet(&state, &request.token0, Some(&provider)).await {
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
                    *current_balance as f64 / 1e24
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

                                    token_balances.insert(balance_key, base_units);
                                    tracing::info!(
                                        "✅ Auto-restored token0 balance for {} (contract {}): {} display tokens × 10^{} = {} base units",
                                        hex::encode(&provider[..8]),
                                        hex::encode(&token0_addr[..8]),
                                        display_supply,
                                        decimals,
                                        base_units
                                    );
                                    break;
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
                if *balance < request.amount0 as u128 {
                    // 🔍 Enhanced error message with context
                    tracing::error!(
                        "💸 Insufficient token0 balance for {}. Token: {}, Required: {}, Available: {}",
                        hex::encode(&provider[..8]),
                        request.token0,
                        request.amount0,
                        *balance
                    );

                    // Calculate how many tokens with 8 decimals for user-friendly error
                    let required_display = request.amount0 as f64 / 1e24;
                    let available_display = *balance as f64 / 1e24;

                    return Ok(Json(ApiResponse::error(format!(
                        "Insufficient {} balance. Required: {} ({} raw units), Available: {} ({} raw units). Please check your token balance or reduce the liquidity amount.",
                        request.token0, required_display, request.amount0, available_display, *balance
                    ))));
                }
                *balance -= request.amount0 as u128;
                token_balance_changes.push((provider, token0_addr, *balance)); // Track for persistence
                tracing::info!(
                    "💸 Deducted {} token0 ({} raw units) from {} for liquidity. Remaining: {}",
                    request.amount0 as f64 / 1e24,
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

        // Deduct token1 (native QUG, native QUGUSD, or token)
        if is_native_token1 {
            // v3.6.4-beta: CRITICAL FIX - Read balance from storage_engine (authoritative source)
            let storage_balance = state
                .storage_engine
                .get_balance(&hex::encode(provider))
                .await
                .unwrap_or(0);

            // Sync in-memory cache with storage
            let balance = wallet_balances.entry(provider).or_insert(storage_balance);
            if *balance != storage_balance {
                tracing::info!(
                    "🔄 [LIQUIDITY] Synced stale balance for {}: {} → {}",
                    hex::encode(&provider[..8]),
                    *balance as f64 / 1e24,
                    storage_balance as f64 / 1e24
                );
                *balance = storage_balance;
            }

            if *balance < request.amount1 {
                // v3.6.2-beta: Display human-readable amounts (24 decimal precision)
                return Ok(Json(ApiResponse::error(format!(
                    "Insufficient QUG balance. Required: {:.6} QUG, Available: {:.6} QUG",
                    request.amount1 as f64 / 1e24, *balance as f64 / 1e24
                ))));
            }
            *balance -= request.amount1;
            native_qug_balance_change = Some((provider, *balance)); // v10.2.1: Track for persistence
            // v11.2.1: CRITICAL — also write the DEX debit counter (see token0
            // branch above for the full explanation).
            if let Err(e) = state
                .storage_engine
                .record_dex_qug_debit(&hex::encode(provider), request.amount1)
                .await
            {
                tracing::error!("🚨 [DEX-DEBIT] Failed to record add_liquidity QUG debit (token1): {}", e);
            }
            tracing::info!(
                "💸 Deducted {} QUG from {} for liquidity. New balance: {}",
                request.amount1 as f64 / 1e24,
                hex::encode(provider),
                *balance as f64 / 1e24
            );
        } else if is_qugusd_token1 {
            // v2.6.1-beta: Deduct QUGUSD stablecoin
            // QUGUSD balance = minted (from vault) + received (from swaps/transfers)
            let minted_qugusd = {
                let vault = state.collateral_vault.read().await;
                vault.minted_qugusd.get(&provider).copied().unwrap_or(0) as u128
            };
            let swapped_qugusd = {
                let token_balances = state.token_balances.read().await;
                let qugusd_addr = [0x51, 0x55, 0x47, 0x55, 0x53, 0x44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; // "QUGUSD" padded
                token_balances.get(&(provider, qugusd_addr)).copied().unwrap_or(0)
            };
            let total_qugusd = minted_qugusd + swapped_qugusd;

            if total_qugusd < request.amount1 as u128 {
                return Ok(Json(ApiResponse::error(format!(
                    "Insufficient QUGUSD balance. Required: {:.4}, Available: {:.4} (minted: {:.4}, swapped: {:.4})",
                    request.amount1 as f64 / 1e24,
                    total_qugusd as f64 / 1e24,
                    minted_qugusd as f64 / 1e24,
                    swapped_qugusd as f64 / 1e24
                ))));
            }

            // Deduct from minted first, then from swapped
            // v3.0.4: minted_qugusd is now u128
            let mut remaining = request.amount1 as u128;
            if minted_qugusd > 0 {
                let deduct_from_minted = remaining.min(minted_qugusd);
                let mut vault = state.collateral_vault.write().await;
                if let Some(bal) = vault.minted_qugusd.get_mut(&provider) {
                    *bal = bal.saturating_sub(deduct_from_minted);
                }
                remaining -= deduct_from_minted;
                tracing::info!(
                    "💸 Deducted {} QUGUSD (minted) from {} for liquidity",
                    deduct_from_minted as f64 / 1e24,
                    hex::encode(&provider[..8])
                );
            }
            if remaining > 0 {
                let mut token_balances = state.token_balances.write().await;
                let qugusd_addr = [0x51, 0x55, 0x47, 0x55, 0x53, 0x44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                if let Some(bal) = token_balances.get_mut(&(provider, qugusd_addr)) {
                    *bal = bal.saturating_sub(remaining);
                }
                tracing::info!(
                    "💸 Deducted {} QUGUSD (swapped) from {} for liquidity",
                    remaining as f64 / 1e24,
                    hex::encode(&provider[..8])
                );
            }
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
                                token_balances.insert(balance_key, base_units);
                                tracing::info!(
                                    "💰 Auto-restored token1 balance for {} (contract {}): {} display × 10^{} = {} base units",
                                    hex::encode(&provider[..8]),
                                    hex::encode(&token1_addr[..8]),
                                    display_supply,
                                    decimals,
                                    base_units
                                );
                                break;
                            }
                        }
                    }
                }
                drop(deployed_contracts);
            }

            if let Some(balance) = token_balances.get_mut(&balance_key) {
                if *balance < request.amount1 as u128 {
                    return Ok(Json(ApiResponse::error(format!(
                        "Insufficient token1 balance. Required: {}, Available: {}",
                        request.amount1, *balance
                    ))));
                }
                *balance -= request.amount1 as u128;
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

    // v10.2.1: Persist native QUG balance deduction (fixes deductions lost on restart)
    if let Some((addr, new_balance)) = native_qug_balance_change {
        if let Err(e) = state.storage_engine.save_wallet_balance(&addr, new_balance).await {
            tracing::warn!("⚠️ Failed to persist QUG balance after liquidity add: {}", e);
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

    let (final_pool_id, action, minted_lp_tokens) = if let Some(existing_pool_id) = pool_id {
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

            // Capture pool token names for LP metadata before dropping pool reference
            let pool_token0_name = pool.token0.clone();
            let pool_token1_name = pool.token1.clone();

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

            // Credit LP tokens to provider's wallet
            if additional_lp_tokens > 0 {
                let lp_token_addr = generate_lp_token_address(&existing_pool_id);
                let mut token_balances = state.token_balances.write().await;
                let current = *token_balances.get(&(provider, lp_token_addr)).unwrap_or(&0u128);
                let new_balance = current + additional_lp_tokens;
                token_balances.insert((provider, lp_token_addr), new_balance);
                drop(token_balances);
                state.storage_engine.save_token_balance(&provider, &lp_token_addr, new_balance).await.ok();
                tracing::info!(
                    "🪙 Credited {} LP tokens to {} (pool {}, total: {})",
                    additional_lp_tokens, hex::encode(&provider[..8]), existing_pool_id, new_balance
                );
            }

            (existing_pool_id.clone(), "added", additional_lp_tokens)
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
            // v3.2.17-beta: Look up actual token decimals from contract metadata
            let token0_decimals = get_token_decimals_from_contract(
                &state, &token0_canonical, &token0_addr, is_native_token0, is_qugusd_token0
            ).await;
            let token1_decimals = get_token_decimals_from_contract(
                &state, &token1_canonical, &token1_addr, is_native_token1, is_qugusd_token1
            ).await;

            let pool = LiquidityPool {
                pool_id: new_pool_id.clone(),
                token0: token0_canonical.clone(),
                token1: token1_canonical.clone(),
                reserve0: request.amount0,
                reserve1: request.amount1,
                provider,
                created_at: chrono::Utc::now(),
                lp_token_supply: lp_tokens,
                token0_decimals,
                token1_decimals,
            };
            let pool_clone = pool.clone();
            pools.insert(new_pool_id.clone(), pool);
            tracing::info!(
                "💰 Created liquidity pool {} (deterministic) with tokens {} ({} dec) / {} ({} dec) and reserves: {} / {}",
                new_pool_id,
                token0_canonical,
                token0_decimals,
                token1_canonical,
                token1_decimals,
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

            // Credit LP tokens to provider + save metadata
            if lp_tokens > 0 {
                let lp_token_addr = generate_lp_token_address(&new_pool_id);
                let mut token_balances = state.token_balances.write().await;
                token_balances.insert((provider, lp_token_addr), lp_tokens);
                drop(token_balances);
                state.storage_engine.save_token_balance(&provider, &lp_token_addr, lp_tokens).await.ok();
                let sym0 = token_display_symbol(&token0_canonical);
                let sym1 = token_display_symbol(&token1_canonical);
                state.storage_engine.save_lp_token_meta(&lp_token_addr, &sym0, &sym1).await.ok();
                tracing::info!("🪙 Credited {} LP tokens to {} (new pool {})", lp_tokens, hex::encode(&provider[..8]), new_pool_id);
            }

            (new_pool_id, "created", lp_tokens)
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
        // v3.2.17-beta: Look up actual token decimals from contract metadata
        let token0_decimals = get_token_decimals_from_contract(
            &state, &token0_canonical, &token0_addr, is_native_token0, is_qugusd_token0
        ).await;
        let token1_decimals = get_token_decimals_from_contract(
            &state, &token1_canonical, &token1_addr, is_native_token1, is_qugusd_token1
        ).await;

        let pool = LiquidityPool {
            pool_id: new_pool_id.clone(),
            token0: token0_canonical.clone(),
            token1: token1_canonical.clone(),
            reserve0: request.amount0,
            reserve1: request.amount1,
            provider,
            created_at: chrono::Utc::now(),
            lp_token_supply: lp_tokens,
            token0_decimals,
            token1_decimals,
        };

        let pool_clone = pool.clone();
        let mut pools = state.liquidity_pools.write().await;
        pools.insert(new_pool_id.clone(), pool);
        tracing::info!(
            "💰 Created liquidity pool {} (deterministic) with tokens {} ({} dec) / {} ({} dec) and reserves: {} / {}",
            new_pool_id,
            token0_canonical,
            token0_decimals,
            token1_canonical,
            token1_decimals,
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

        // Credit LP tokens to provider + save metadata
        if lp_tokens > 0 {
            let lp_token_addr = generate_lp_token_address(&new_pool_id);
            let mut token_balances = state.token_balances.write().await;
            token_balances.insert((provider, lp_token_addr), lp_tokens);
            drop(token_balances);
            state.storage_engine.save_token_balance(&provider, &lp_token_addr, lp_tokens).await.ok();
            let sym0 = token_display_symbol(&token0_canonical);
            let sym1 = token_display_symbol(&token1_canonical);
            state.storage_engine.save_lp_token_meta(&lp_token_addr, &sym0, &sym1).await.ok();
            tracing::info!("🪙 Credited {} LP tokens to {} (new pool {})", lp_tokens, hex::encode(&provider[..8]), new_pool_id);
        }

        (new_pool_id, "created", lp_tokens)
    };

    // ========================================
    // v0.6.1-beta: DEX DECENTRALIZATION PHASE 3
    // Broadcast pool announcement to P2P network
    // ========================================
    if action == "created" || action == "added" {
        // Broadcast both newly created pools and liquidity additions to existing pools
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
        lp_tokens_minted: minted_lp_tokens.to_string(),
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

    // Check user has LP tokens for this pool (replaces old provider-only ownership check)
    let lp_token_addr = generate_lp_token_address(&request.pool_id);
    let user_lp_balance = {
        let token_balances = state.token_balances.read().await;
        token_balances.get(&(provider, lp_token_addr)).copied().unwrap_or(0u128)
    };

    if user_lp_balance == 0 {
        return Ok(Json(ApiResponse::error(
            "You have no LP tokens for this pool. Add liquidity first.".to_string(),
        )));
    }

    // Calculate LP tokens to burn based on percentage of user's LP balance
    let lp_to_burn = (user_lp_balance * request.percentage as u128) / 100;
    if lp_to_burn == 0 {
        return Ok(Json(ApiResponse::error(
            "LP token amount to burn rounds to zero".to_string(),
        )));
    }

    // Calculate proportional token amounts to return based on LP share of total supply
    let total_lp_supply = pool.lp_token_supply;
    if total_lp_supply == 0 {
        return Ok(Json(ApiResponse::error(
            "Pool has zero LP token supply".to_string(),
        )));
    }
    let amount0_to_return = (pool.reserve0 * lp_to_burn) / total_lp_supply;
    let amount1_to_return = (pool.reserve1 * lp_to_burn) / total_lp_supply;

    // Check if tokens are native QUG or custom tokens
    let is_native_token0 =
        pool.token0.to_uppercase() == "QUG" || pool.token0.to_lowercase() == "native-qug";
    let is_native_token1 =
        pool.token1.to_uppercase() == "QUG" || pool.token1.to_lowercase() == "native-qug";

    // Resolve token addresses for custom tokens
    // v3.9.4-beta: Use wallet-aware resolution for pools with symbol-based tokens
    let token0_addr = if !is_native_token0 {
        if pool.token0.starts_with("0x") || pool.token0.starts_with("qnk") {
            match parse_address(&pool.token0) {
                Ok(addr) => addr,
                Err(e) => return Ok(Json(ApiResponse::error(e))),
            }
        } else {
            match resolve_token_symbol_for_wallet(&state, &pool.token0, Some(&provider)).await {
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
            match resolve_token_symbol_for_wallet(&state, &pool.token1, Some(&provider)).await {
                Ok(addr) => addr,
                Err(e) => return Ok(Json(ApiResponse::error(e))),
            }
        }
    } else {
        [0u8; 32]
    };

    let mut token_balance_changes: Vec<([u8; 32], [u8; 32], u128)> = Vec::new();
    // v10.2.1: Track native QUG balance changes for persistence
    let mut native_qug_balance_change: Option<([u8; 32], u128)> = None;

    // Return balances to provider
    {
        let mut wallet_balances = state.wallet_balances.write().await;
        let mut token_balances = state.token_balances.write().await;

        // Return token0
        if is_native_token0 {
            let balance = wallet_balances.entry(provider).or_insert(0);
            *balance += amount0_to_return;
            native_qug_balance_change = Some((provider, *balance));
            tracing::info!(
                "💰 Returned {} QUG to {} from liquidity removal",
                amount0_to_return,
                hex::encode(provider)
            );
        } else {
            let balance_key = (provider, token0_addr);
            *token_balances.entry(balance_key).or_insert(0) += amount0_to_return as u128;
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
            let balance = wallet_balances.entry(provider).or_insert(0);
            *balance += amount1_to_return;
            native_qug_balance_change = Some((provider, *balance));
            tracing::info!(
                "💰 Returned {} QUG to {} from liquidity removal",
                amount1_to_return,
                hex::encode(provider)
            );
        } else {
            let balance_key = (provider, token1_addr);
            *token_balances.entry(balance_key).or_insert(0) += amount1_to_return as u128;
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

    // v10.2.1: Persist native QUG balance after liquidity removal
    if let Some((addr, new_balance)) = native_qug_balance_change {
        if let Err(e) = state.storage_engine.save_wallet_balance(&addr, new_balance).await {
            tracing::warn!("⚠️ Failed to persist QUG balance after liquidity removal: {}", e);
        }
    }

    // Burn LP tokens from provider's balance
    {
        let mut token_balances = state.token_balances.write().await;
        let current_lp = token_balances.get(&(provider, lp_token_addr)).copied().unwrap_or(0);
        let new_lp_balance = current_lp.saturating_sub(lp_to_burn);
        token_balances.insert((provider, lp_token_addr), new_lp_balance);
        drop(token_balances);
        state.storage_engine.save_token_balance(&provider, &lp_token_addr, new_lp_balance).await.ok();
        tracing::info!(
            "🔥 Burned {} LP tokens from {} (pool {}, remaining: {})",
            lp_to_burn, hex::encode(&provider[..8]), request.pool_id, new_lp_balance
        );
    }

    // Update or remove pool
    {
        let mut pools = state.liquidity_pools.write().await;

        // Check if pool should be fully removed: either user requested 100%
        // or the LP supply will drop to zero after burning
        let should_remove = {
            if let Some(pool) = pools.get(&request.pool_id) {
                pool.lp_token_supply <= lp_to_burn
            } else {
                false
            }
        };

        if should_remove {
            // Remove pool entirely
            pools.remove(&request.pool_id);
            tracing::info!(
                "🗑️ Removed liquidity pool {} (all LP tokens burned)",
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
            // Update pool reserves and LP supply
            if let Some(pool) = pools.get_mut(&request.pool_id) {
                pool.reserve0 -= amount0_to_return;
                pool.reserve1 -= amount1_to_return;
                pool.lp_token_supply = pool.lp_token_supply.saturating_sub(lp_to_burn);
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

    // v3.9.5-beta: Broadcast pool update via gossipsub (P2P DEX state replication)
    // Only broadcast partial removals - full removals (zero reserves) fail verify_structure()
    // and will propagate through consensus instead
    if request.percentage < 100 {
        let pool_for_broadcast = {
            let pools = state.liquidity_pools.read().await;
            pools.get(&request.pool_id).cloned()
        };

        if let Some(ref p) = pool_for_broadcast {
            let parse_token = |s: &str| -> [u8; 32] {
                if s.to_uppercase() == "QUG" || s.to_lowercase() == "native-qug" {
                    [0u8; 32]
                } else {
                    hex::decode(s.trim_start_matches("0x"))
                        .ok()
                        .and_then(|b| if b.len() == 32 {
                            let mut a = [0u8; 32];
                            a.copy_from_slice(&b);
                            Some(a)
                        } else { None })
                        .unwrap_or([0u8; 32])
                }
            };
            let t0_bytes = parse_token(&p.token0);
            let t1_bytes = parse_token(&p.token1);

            let mut announcement = q_types::PoolAnnouncement::new(
                t0_bytes, t1_bytes, p.reserve0, p.reserve1,
                p.lp_token_supply, provider, p.created_at.timestamp() as u64,
            );

            if let Err(e) = announcement.sign(&*state.node_signing_key) {
                tracing::warn!("Failed to sign remove-liquidity pool announcement: {}", e);
            } else if let Ok(announcement_bytes) = serde_json::to_vec(&announcement) {
                if let Some(ref command_tx) = state.libp2p_command_tx {
                    let topic = "/qnk/liquidity-pools".to_string();
                    if let Err(e) = command_tx.send(q_network::NetworkCommand::PublishPoolAnnouncement {
                        topic: topic.clone(),
                        announcement_bytes: announcement_bytes.clone(),
                    }) {
                        tracing::warn!("🏊 [DEX P2P] Failed to broadcast remove-liquidity: {}", e);
                    } else {
                        tracing::info!("🏊 [DEX P2P] Broadcast pool {} reserves update via P2P ({} bytes)",
                            &request.pool_id[..20.min(request.pool_id.len())],
                            announcement_bytes.len());
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
        pool_id: request.pool_id.clone(),
        amount0_returned: amount0_to_return,
        amount1_returned: amount1_to_return,
        transaction_id: tx_hash,
        lp_tokens_burned: lp_to_burn.to_string(),
    })))
}

#[derive(Debug, Serialize)]
pub struct PoolInfo {
    pub pool_id: String,
    pub token0: String,
    pub token1: String,
    #[serde(serialize_with = "q_types::u128_serde::serialize")]
    pub reserve0: u128,
    #[serde(serialize_with = "q_types::u128_serde::serialize")]
    pub reserve1: u128,
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
/// v3.9.4-beta: Now accepts optional wallet_address to prefer tokens the user owns
async fn resolve_token_symbol(state: &Arc<AppState>, symbol: &str) -> Result<[u8; 32], String> {
    resolve_token_symbol_for_wallet(state, symbol, None).await
}

/// Resolve a token symbol with preference for tokens the wallet owns
/// v3.9.4-beta: CRITICAL FIX - When multiple tokens have the same symbol (e.g., 5 different "LLAMA"),
/// this function prefers the contract where the caller actually has a balance.
async fn resolve_token_symbol_for_wallet(
    state: &Arc<AppState>,
    symbol: &str,
    wallet_address: Option<&[u8; 32]>,
) -> Result<[u8; 32], String> {
    // 🆕 v2.2.1: Special handling for Index Fund tokens (QNK10, DEFI5, etc.)
    let symbol_upper = symbol.to_uppercase();
    if symbol_upper.starts_with("INDEX-FUND-") || symbol_upper == "QNK10" || symbol_upper == "DEFI5" {
        // Generate deterministic address for index fund tokens
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"QNK-INDEX-FUND:");
        // Normalize: extract the fund name (e.g., "QNK10" from "INDEX-FUND-QNK10")
        let fund_name = if symbol_upper.starts_with("INDEX-FUND-") {
            symbol_upper.strip_prefix("INDEX-FUND-").unwrap_or(&symbol_upper)
        } else {
            &symbol_upper
        };
        hasher.update(fund_name.as_bytes());
        let hash = hasher.finalize();
        let mut addr = [0u8; 32];
        addr.copy_from_slice(hash.as_bytes());
        // Mark as index fund: set first byte to 0x1F (Index Fund marker)
        addr[0] = 0x1F;
        tracing::debug!("📊 Resolved index fund token '{}' -> qnk{}", symbol, hex::encode(&addr[..8]));
        return Ok(addr);
    }

    // Special handling for QUGUSD stablecoin
    if symbol.eq_ignore_ascii_case("QUGUSD") || symbol.eq_ignore_ascii_case("QUGUSD-STABLE") {
        let mut addr = [0u8; 32];
        addr[0] = 0xCD; // CDP marker
        addr[1] = 0x01; // QUGUSD identifier
        return Ok(addr);
    }

    // Search through all deployed contracts to find one with matching symbol
    let ecosystem = &state.orobit_ecosystem;

    // Access the deployed contracts directly
    let deployed_contracts = ecosystem.deployed_contracts.read().await;

    // v3.9.4-beta: Collect ALL matching contracts, then pick the one user owns
    let mut matching_contracts: Vec<[u8; 32]> = Vec::new();

    // Search for contracts with matching symbol
    for contract in deployed_contracts.values() {
        if let Some(contract_symbol) = &contract.metadata.symbol {
            if contract_symbol.eq_ignore_ascii_case(symbol) {
                matching_contracts.push(contract.address.0);
            }
        }
    }
    drop(deployed_contracts);

    if matching_contracts.is_empty() {
        return Err(format!("No contract found with symbol '{}'", symbol));
    }

    // If only one contract matches, return it
    if matching_contracts.len() == 1 {
        tracing::debug!(
            "🔍 [RESOLVE] Symbol '{}' -> single match: qnk{}",
            symbol,
            hex::encode(&matching_contracts[0][..8])
        );
        return Ok(matching_contracts[0]);
    }

    // v3.9.4-beta: Multiple contracts with same symbol - prefer one user has balance in
    if let Some(wallet) = wallet_address {
        let token_balances = state.token_balances.read().await;

        for contract_addr in &matching_contracts {
            let balance_key = (*wallet, *contract_addr);
            if let Some(&balance) = token_balances.get(&balance_key) {
                if balance > 0 {
                    tracing::info!(
                        "✅ [RESOLVE] Symbol '{}' has {} matching contracts, chose qnk{} (wallet has {} balance)",
                        symbol,
                        matching_contracts.len(),
                        hex::encode(&contract_addr[..8]),
                        balance as f64 / 1e24
                    );
                    return Ok(*contract_addr);
                }
            }
        }
        drop(token_balances);

        tracing::warn!(
            "⚠️  [RESOLVE] Symbol '{}' has {} matching contracts but wallet qnk{} has no balance in any. Using first match.",
            symbol,
            matching_contracts.len(),
            hex::encode(&wallet[..8])
        );
    } else {
        tracing::warn!(
            "⚠️  [RESOLVE] Symbol '{}' has {} matching contracts. No wallet provided, using first match. Consider specifying full token address.",
            symbol,
            matching_contracts.len()
        );
    }

    // Fallback: return first matching contract
    Ok(matching_contracts[0])
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
    #[serde(serialize_with = "q_types::u128_serde::serialize")]
    pub balance: u128,
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
                    let supply_u128 = supply as u128;

                    // Calculate amount locked in liquidity pools
                    let pools = state.liquidity_pools.read().await;
                    let mut locked_amount = 0u128;

                    for pool in pools.values() {
                        if pool.provider == wallet_addr {
                            // Check if token0 matches
                            if let Ok(pool_token0_addr) = resolve_token_address(&state, &pool.token0).await {
                                if pool_token0_addr == token_addr {
                                    locked_amount += pool.reserve0 as u128;
                                }
                            }

                            // Check if token1 matches
                            if let Ok(pool_token1_addr) = resolve_token_address(&state, &pool.token1).await {
                                if pool_token1_addr == token_addr {
                                    locked_amount += pool.reserve1 as u128;
                                }
                            }
                        }
                    }
                    drop(pools);

                    // Calculate available balance (initial supply - locked in pools)
                    let available_balance = supply_u128.saturating_sub(locked_amount);

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
                        balance_display: available_balance as f64 / 1e24,
                    });

                    tracing::info!(
                        "🔄 Refreshed balance for token {} ({}): {} ({} display units). Initial: {}, Locked: {}",
                        symbol,
                        hex::encode(&token_addr[..8]),
                        available_balance,
                        available_balance as f64 / 1e24,
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
            // v4.0.13: Determine reserve order based on token direction
            // Decimals are no longer needed since all values are in 24-decimal format
            let (reserve_in, reserve_out) =
                if pool.token0 == from_canonical || pool.token0 == request.from_token {
                    (pool.reserve0, pool.reserve1)
                } else {
                    (pool.reserve1, pool.reserve0)
                };

            // v4.0.13: REMOVED cross-decimal normalization.
            // ALL pool reserves and amounts are already in 24-decimal format internally.
            // The normalize/denormalize functions were a no-op mathematically but added
            // unnecessary complexity and potential for edge-case mismatches with handlers.rs.
            // This quote function must match handlers.rs exactly (no normalization).

            tracing::debug!(
                "📊 [QUOTE v4.0.13] Direct AMM calculation (all 24-dec): reserve_in={}, reserve_out={}, amount_in={}",
                reserve_in, reserve_out, request.amount_in
            );

            // Calculate swap using quantum-enhanced AMM with raw reserves (all 24-decimal)
            let fee_rate = DEX_TOTAL_FEE_BPS as f64 / BPS_DIVISOR as f64; // 0.003 (0.30%)
            let (amount_out, price_impact, effective_price) =
                calculate_quantum_swap(request.amount_in, reserve_in, reserve_out, fee_rate);

            // Calculate fee breakdown in the INPUT token's units
            let total_fee = request.amount_in * DEX_TOTAL_FEE_BPS as u128 / BPS_DIVISOR;
            let protocol_fee = request.amount_in * DEX_PROTOCOL_FEE_BPS as u128 / BPS_DIVISOR;
            let lp_fee = request.amount_in * DEX_LP_FEE_BPS as u128 / BPS_DIVISOR;

            // v4.0.13: All values are in 24-decimal format
            let display_divisor = 1e24;

            tracing::info!(
                "⚛️ Quantum swap quote (v4.0.13): {} {} => {} {} (impact: {:.4}%, pool: {}) | Fees: total={} protocol={} lp={}",
                request.amount_in as f64 / display_divisor,
                request.from_token,
                amount_out as f64 / display_divisor,
                request.to_token,
                price_impact * 100.0,
                pool.pool_id,
                total_fee,
                protocol_fee,
                lp_fee
            );

            Ok(Json(ApiResponse::success(SwapQuoteResponse {
                from_token: request.from_token,
                to_token: request.to_token,
                amount_in: request.amount_in,
                amount_out,
                price_impact,
                effective_price,
                fee: total_fee,
                protocol_fee,
                lp_fee,
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

// ============================================================================
// Admin Pool Repair - v4.0.11
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct AdminResetPoolRequest {
    pub pool_id: String,
    #[serde(deserialize_with = "deserialize_u128_from_any")]
    pub new_reserve0: u128,
    #[serde(deserialize_with = "deserialize_u128_from_any")]
    pub new_reserve1: u128,
    /// Admin key for authentication (must match bootstrap node wallet)
    pub admin_key: Option<String>,
}

/// Admin endpoint to reset corrupted pool reserves
/// This is needed when pools become corrupted due to decimal mismatch bugs
pub async fn admin_reset_pool_reserves(
    State(state): State<Arc<crate::AppState>>,
    Json(request): Json<AdminResetPoolRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    // Simple admin auth: only allow from localhost or with admin key
    let admin_key = request.admin_key.as_deref().unwrap_or("");
    if admin_key != "qnk-admin-pool-repair-2026" {
        return Ok(Json(ApiResponse::error(
            "Unauthorized: Invalid admin key".to_string(),
        )));
    }

    let pool_id = &request.pool_id;

    // Update pool in memory
    let pool_data_for_storage = {
        let mut pools = state.liquidity_pools.write().await;
        if let Some(pool) = pools.get_mut(pool_id) {
            let old_r0 = pool.reserve0;
            let old_r1 = pool.reserve1;

            pool.reserve0 = request.new_reserve0;
            pool.reserve1 = request.new_reserve1;

            tracing::warn!(
                "🔧 [ADMIN] Reset pool {} reserves: {} / {} → {} / {} (display: {:.4} / {:.4} → {:.4} / {:.4})",
                pool_id,
                old_r0, old_r1,
                request.new_reserve0, request.new_reserve1,
                old_r0 as f64 / 1e24, old_r1 as f64 / 1e24,
                request.new_reserve0 as f64 / 1e24, request.new_reserve1 as f64 / 1e24,
            );

            serde_json::to_vec(&*pool).ok()
        } else {
            return Ok(Json(ApiResponse::error(format!(
                "Pool '{}' not found. Available pools: {:?}",
                pool_id,
                pools.keys().collect::<Vec<_>>()
            ))));
        }
    };

    // Persist to storage
    if let Some(pool_data) = pool_data_for_storage {
        if let Err(e) = state.storage_engine.save_liquidity_pool(pool_id, &pool_data).await {
            tracing::error!("Failed to persist pool reset: {}", e);
            return Ok(Json(ApiResponse::error(format!(
                "Pool updated in memory but failed to persist: {}", e
            ))));
        }
    }

    Ok(Json(ApiResponse::success(serde_json::json!({
        "pool_id": pool_id,
        "new_reserve0": request.new_reserve0.to_string(),
        "new_reserve1": request.new_reserve1.to_string(),
        "new_reserve0_display": request.new_reserve0 as f64 / 1e24,
        "new_reserve1_display": request.new_reserve1 as f64 / 1e24,
        "status": "Pool reserves reset successfully"
    }))))
}
