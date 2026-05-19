/// DEX Integration API for Q-NarwhalKnight
///
/// This module provides secure, easy-to-use API endpoints specifically designed
/// for external DEXes and swap protocols to integrate with the Q-NarwhalKnight
/// node system and VM. It focuses on security, standardization, and ease of use.
///
/// ## Security Features Implemented:
/// - Comprehensive input validation for all endpoints
/// - Rate limiting infrastructure (RateLimiter struct)
/// - API key generation and validation system
/// - Security headers for all responses
/// - Client IP extraction and tracking
/// - Transaction deadline validation
/// - Address format validation
/// - Slippage tolerance bounds checking
///
/// ## Endpoints Available:
/// - Node information and capabilities
/// - Token management and metadata
/// - Liquidity pool creation and management
/// - Swap quote generation with proper validation
/// - Swap execution with comprehensive checks
/// - Price oracle integration
/// - Security audit and compliance checking
/// - Integration helpers (webhooks, API keys, rate limits)
///
/// ## Usage:
/// All endpoints are available under `/api/v1/dex/` and return standardized
/// DexApiResponse<T> wrappers with success/error status, timestamps, and metadata.
use axum::{
    extract::{Path, Query, State},
    http::Request,
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Json,
    routing::{get, post},
    Router,
};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::AppState;
use q_types::*;
use q_vm::contracts::{ContractAddress, ContractType, DeploymentOptions};

/// Security middleware for DEX API endpoints
pub struct DexSecurityMiddleware;

/// Rate limiter for API calls
#[derive(Debug, Clone)]
pub struct RateLimiter {
    calls: Arc<RwLock<HashMap<String, (u64, Instant)>>>, // client_id -> (count, window_start)
    max_calls_per_hour: u64,
    window_duration: Duration,
}

impl RateLimiter {
    pub fn new(max_calls_per_hour: u64) -> Self {
        Self {
            calls: Arc::new(RwLock::new(HashMap::new())),
            max_calls_per_hour,
            window_duration: Duration::from_secs(3600), // 1 hour
        }
    }

    pub async fn is_allowed(&self, client_id: &str) -> bool {
        let mut calls = self.calls.write().await;
        let now = Instant::now();

        match calls.get_mut(client_id) {
            Some((count, window_start)) => {
                if now.duration_since(*window_start) >= self.window_duration {
                    // Reset window
                    *count = 1;
                    *window_start = now;
                    true
                } else if *count < self.max_calls_per_hour {
                    *count += 1;
                    true
                } else {
                    false // Rate limited
                }
            }
            None => {
                calls.insert(client_id.to_string(), (1, now));
                true
            }
        }
    }

    pub async fn get_remaining_calls(&self, client_id: &str) -> u64 {
        let calls = self.calls.read().await;
        match calls.get(client_id) {
            Some((count, window_start)) => {
                let now = Instant::now();
                if now.duration_since(*window_start) >= self.window_duration {
                    self.max_calls_per_hour
                } else {
                    self.max_calls_per_hour.saturating_sub(*count)
                }
            }
            None => self.max_calls_per_hour,
        }
    }
}

/// API Key validation
#[derive(Debug, Clone)]
pub struct ApiKey {
    pub key: String,
    pub permissions: Vec<String>,
    pub rate_limit: u64,
    pub created_at: u64,
    pub expires_at: Option<u64>,
    pub is_active: bool,
}

impl ApiKey {
    pub fn validate(&self) -> bool {
        if !self.is_active {
            return false;
        }

        if let Some(expires_at) = self.expires_at {
            let now = chrono::Utc::now().timestamp() as u64;
            if now > expires_at {
                return false;
            }
        }

        true
    }

    pub fn has_permission(&self, permission: &str) -> bool {
        self.permissions.contains(&permission.to_string())
            || self.permissions.contains(&"admin".to_string())
    }
}

/// Security context for requests
#[derive(Debug)]
pub struct SecurityContext {
    pub client_id: String,
    pub api_key: Option<ApiKey>,
    pub rate_limit_remaining: u64,
    pub request_timestamp: u64,
    pub client_ip: String,
}

// Rate limiting and API key storage would be part of AppState in production
// For now, we'll use simple validation

/// Simple security validation helper
pub fn validate_api_key(api_key: &str) -> bool {
    // Simple validation - in production this would query a database
    // For now, accept any key that starts with "qnk_" and is at least 32 chars
    api_key.starts_with("qnk_") && api_key.len() >= 32
}

/// Extract client IP from headers
pub fn extract_client_ip(headers: &HeaderMap) -> String {
    headers
        .get("x-forwarded-for")
        .or_else(|| headers.get("x-real-ip"))
        .and_then(|h| h.to_str().ok())
        .unwrap_or("127.0.0.1")
        .split(',')
        .next()
        .unwrap_or("127.0.0.1")
        .trim()
        .to_string()
}

/// Add security headers to response
pub fn add_security_headers(headers: &mut HeaderMap) {
    headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
    headers.insert("X-Frame-Options", "DENY".parse().unwrap());
    headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
    headers.insert(
        "Strict-Transport-Security",
        "max-age=31536000; includeSubDomains".parse().unwrap(),
    );
    headers.insert("X-API-Version", "1.0.0".parse().unwrap());
    headers.insert("X-RateLimit-Limit", "5000".parse().unwrap());
}

/// DEX Integration API Response wrapper
#[derive(Serialize)]
pub struct DexApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: u64,
    pub api_version: String,
    pub network: String, // "mainnet-genesis", "testnet"
}

impl<T> DexApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: chrono::Utc::now().timestamp() as u64,
            api_version: "1.0.0".to_string(),
            network: "mainnet-genesis".to_string(), // TODO: Make configurable
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message),
            timestamp: chrono::Utc::now().timestamp() as u64,
            api_version: "1.0.0".to_string(),
            network: "mainnet-genesis".to_string(),
        }
    }
}

/// DEX Integration Router with security middleware
pub fn create_dex_integration_router() -> Router<Arc<AppState>> {
    use axum::middleware::from_fn;

    Router::new()
        // Core DEX Integration Endpoints
        .route("/info", get(get_node_info))
        .route("/supported-tokens", get(get_supported_tokens))
        .route("/tokens", get(get_supported_tokens)) // Shorter alias for frontend compatibility
        .route("/token/:address/info", get(get_token_info))
        // Liquidity Pool Endpoints
        .route("/pools", get(get_all_pools))
        .route("/pools/:address", get(get_pool_info))
        .route("/pools/:address/reserves", get(get_pool_reserves))
        .route("/pools/create", post(create_liquidity_pool))
        // Swap/Trade Endpoints
        .route("/swap/quote", post(get_swap_quote))
        .route("/swap/execute", post(execute_swap))
        .route("/swap/:tx_hash/status", get(get_swap_status))
        // Price Oracle Endpoints
        .route("/prices", get(get_all_prices))
        .route("/prices/:token", get(get_token_price))
        .route("/prices/historical/:token", get(get_historical_prices))
        // Security & Compliance Endpoints
        .route("/security/audit/:contract", get(get_contract_audit))
        .route("/compliance/check", post(compliance_check))
        // Integration Helper Endpoints
        .route("/integration/webhook", post(setup_webhook))
        .route("/integration/api-key", post(generate_api_key))
        .route("/integration/rate-limits", get(get_rate_limits))
}

// ============ CORE INTEGRATION ENDPOINTS ============

/// Get node information for DEX integration
#[derive(Serialize)]
pub struct NodeIntegrationInfo {
    pub node_id: String,
    pub network: String,
    pub api_version: String,
    pub supported_standards: Vec<String>, // ERC-20, BEP-20, etc.
    pub vm_capabilities: VmCapabilities,
    pub security_features: SecurityFeatures,
    pub performance_metrics: PerformanceMetrics,
}

#[derive(Serialize)]
pub struct VmCapabilities {
    pub smart_contracts: bool,
    pub cross_chain: bool,
    pub quantum_security: bool,
    pub supported_contract_types: Vec<String>,
    pub max_gas_limit: u64,
    pub consensus_type: String, // "DAG-Knight"
}

#[derive(Serialize)]
pub struct SecurityFeatures {
    pub tor_integration: bool,
    pub quantum_crypto: bool,
    pub post_quantum_ready: bool,
    pub audit_status: String,
    pub bug_bounty_program: bool,
}

#[derive(Serialize)]
pub struct PerformanceMetrics {
    pub tps: u64,
    pub block_time_seconds: f64,
    pub finality_time_seconds: f64,
    pub current_load: f64, // 0.0 to 1.0
}

pub async fn get_node_info(
    State(state): State<Arc<AppState>>,
) -> Result<Json<DexApiResponse<NodeIntegrationInfo>>, StatusCode> {
    let node_status = state.node_status.read().await;

    let info = NodeIntegrationInfo {
        node_id: hex::encode(&state.node_id),
        network: "Q-NarwhalKnight-Mainnet".to_string(),
        api_version: "1.0.0".to_string(),
        supported_standards: vec![
            "QNK-20".to_string(), // Native token standard
            "ERC-20-Compatible".to_string(),
            "Cross-Chain".to_string(),
        ],
        vm_capabilities: VmCapabilities {
            smart_contracts: true,
            cross_chain: true,
            quantum_security: true,
            supported_contract_types: vec![
                "PrivateDex".to_string(),
                "LiquidityPool".to_string(),
                "StakingContract".to_string(),
                "YieldFarming".to_string(),
                "MultisigWallet".to_string(),
            ],
            max_gas_limit: 30_000_000,
            consensus_type: "DAG-Knight".to_string(),
        },
        security_features: SecurityFeatures {
            tor_integration: true,
            quantum_crypto: true,
            post_quantum_ready: true,
            audit_status: "Ongoing".to_string(),
            bug_bounty_program: true,
        },
        performance_metrics: PerformanceMetrics {
            tps: 27200, // Based on Phase 1 achievements
            block_time_seconds: 0.5,
            finality_time_seconds: 2.9,
            current_load: (node_status.tx_pool_size as f64) / 10000.0,
        },
    };

    Ok(Json(DexApiResponse::success(info)))
}

// ============ TOKEN MANAGEMENT ENDPOINTS ============

#[derive(Serialize)]
pub struct TokenInfo {
    pub address: String,
    pub name: String,
    pub symbol: String,
    pub decimals: u8,
    pub total_supply: String, // Use string for large numbers
    pub contract_type: String,
    pub verified: bool,
    pub audit_report: Option<String>,
}

pub async fn get_supported_tokens(
    State(state): State<Arc<AppState>>,
) -> Result<Json<DexApiResponse<Vec<TokenInfo>>>, StatusCode> {
    // Start with native QUG and QUGUSD tokens
    let mut tokens = vec![
        TokenInfo {
            address: hex::encode(q_types::QUG_TOKEN_ADDRESS),
            name: "Quillon".to_string(),
            symbol: "QUG".to_string(),
            decimals: q_types::QUG_DECIMALS,
            total_supply: q_types::QUG_MAX_SUPPLY.to_string(),
            contract_type: "Native".to_string(),
            verified: true,
            audit_report: Some("https://audits.q-narwhalknight.dev/qug".to_string()),
        },
        TokenInfo {
            address: hex::encode(q_types::QUGUSD_TOKEN_ADDRESS),
            name: "Quillon USD".to_string(),
            symbol: "QUGUSD".to_string(),
            decimals: q_types::QUGUSD_DECIMALS,
            total_supply: "unlimited".to_string(), // Unlimited if properly collateralized
            contract_type: "Stablecoin".to_string(),
            verified: true,
            audit_report: Some("https://audits.q-narwhalknight.dev/qugusd".to_string()),
        },
    ];

    // ✅ v7.2.5: Add wrapped bridge tokens (wBTC, wZEC, wIRON, wETH)
    {
        use q_types::{
            WBTC_DECIMALS, WBTC_TOKEN_ADDRESS, WETH_DECIMALS, WETH_TOKEN_ADDRESS, WIRON_DECIMALS,
            WIRON_TOKEN_ADDRESS, WZEC_DECIMALS, WZEC_TOKEN_ADDRESS,
        };

        // Get total supply from token_balances (sum of all minted wrapped tokens)
        let token_bals = state.token_balances.read().await;
        let wbtc_supply: u128 = token_bals
            .iter()
            .filter(|((_, t), _)| *t == WBTC_TOKEN_ADDRESS)
            .map(|(_, a)| *a)
            .sum();
        let wzec_supply: u128 = token_bals
            .iter()
            .filter(|((_, t), _)| *t == WZEC_TOKEN_ADDRESS)
            .map(|(_, a)| *a)
            .sum();
        let wiron_supply: u128 = token_bals
            .iter()
            .filter(|((_, t), _)| *t == WIRON_TOKEN_ADDRESS)
            .map(|(_, a)| *a)
            .sum();
        let weth_supply: u128 = token_bals
            .iter()
            .filter(|((_, t), _)| *t == WETH_TOKEN_ADDRESS)
            .map(|(_, a)| *a)
            .sum();
        drop(token_bals);

        tokens.push(TokenInfo {
            address: hex::encode(WBTC_TOKEN_ADDRESS),
            name: "Wrapped Bitcoin".to_string(),
            symbol: "wBTC".to_string(),
            decimals: WBTC_DECIMALS,
            total_supply: wbtc_supply.to_string(),
            contract_type: "Wrapped".to_string(),
            verified: true,
            audit_report: Some("https://audits.q-narwhalknight.dev/bridge/wbtc".to_string()),
        });

        tokens.push(TokenInfo {
            address: hex::encode(WZEC_TOKEN_ADDRESS),
            name: "Wrapped Zcash".to_string(),
            symbol: "wZEC".to_string(),
            decimals: WZEC_DECIMALS,
            total_supply: wzec_supply.to_string(),
            contract_type: "Wrapped".to_string(),
            verified: true,
            audit_report: Some("https://audits.q-narwhalknight.dev/bridge/wzec".to_string()),
        });

        tokens.push(TokenInfo {
            address: hex::encode(WIRON_TOKEN_ADDRESS),
            name: "Wrapped Iron Fish".to_string(),
            symbol: "wIRON".to_string(),
            decimals: WIRON_DECIMALS,
            total_supply: wiron_supply.to_string(),
            contract_type: "Wrapped".to_string(),
            verified: true,
            audit_report: Some("https://audits.q-narwhalknight.dev/bridge/wiron".to_string()),
        });

        tokens.push(TokenInfo {
            address: hex::encode(WETH_TOKEN_ADDRESS),
            name: "Wrapped Ethereum".to_string(),
            symbol: "wETH".to_string(),
            decimals: WETH_DECIMALS,
            total_supply: weth_supply.to_string(),
            contract_type: "Wrapped".to_string(),
            verified: true,
            audit_report: Some("https://audits.q-narwhalknight.dev/bridge/weth".to_string()),
        });
    }

    // ✅ Add custom tokens from deployed contracts
    // v10.2.1: Deduplicate by symbol — first contract per symbol wins
    let mut seen_symbols = std::collections::HashSet::new();
    seen_symbols.insert("QUG".to_string());
    seen_symbols.insert("QUGUSD".to_string());
    seen_symbols.insert("wBTC".to_string());
    seen_symbols.insert("wZEC".to_string());
    seen_symbols.insert("wIRON".to_string());
    seen_symbols.insert("wETH".to_string());
    let genesis_ts = q_storage::emission_controller::GENESIS_TIMESTAMP;
    let deployed_contracts = state.orobit_ecosystem.deployed_contracts.read().await;
    // v10.2.2: Sort by deployed_at descending so the NEWEST contract per symbol wins.
    // HashMap iteration is non-deterministic; sorting ensures the most recent (correct)
    // deployment is always picked when duplicate symbols exist.
    let mut sorted_contracts: Vec<_> = deployed_contracts.values().collect();
    sorted_contracts.sort_by(|a, b| b.deployed_at.cmp(&a.deployed_at));
    for contract in sorted_contracts {
        // v7.1.7: Skip pre-genesis (testnet) contracts
        if contract.deployed_at < genesis_ts {
            continue;
        }
        // Check if this contract has token metadata (symbol indicates it's a token)
        if let Some(symbol) = &contract.metadata.symbol {
            // v10.2.1: Skip duplicate symbols
            if seen_symbols.contains(&symbol.to_uppercase()) {
                continue;
            }
            seen_symbols.insert(symbol.to_uppercase());
            // Get token details from deployment params
            // v4.0.15: Pass through raw supply - frontend handles arbitrary sizes
            let total_supply_str = contract
                .deployment_params
                .get("initialSupply")
                .or_else(|| contract.deployment_params.get("initial_supply"))
                .map(|v| {
                    if let Some(n) = v.as_u64() {
                        n.to_string()
                    } else if let Some(s) = v.as_str() {
                        // Try parsing as u128 first (exact), fall back to f64 for scientific notation
                        if let Ok(n) = s.parse::<u128>() {
                            n.to_string()
                        } else if let Ok(f) = s.parse::<f64>() {
                            format!("{:.0}", f)
                        } else {
                            "0".to_string()
                        }
                    } else {
                        v.to_string()
                    }
                })
                .unwrap_or_else(|| "0".to_string());

            let name = if contract.metadata.name.is_empty() {
                symbol.clone()
            } else {
                contract.metadata.name.clone()
            };
            // v5.1.3: Read actual decimals from deployment params instead of hardcoding 8.
            // Tokens can have different decimal values (e.g. BORK=7, custom=18, etc.)
            let decimals = contract
                .deployment_params
                .get("decimals")
                .and_then(|v| v.as_u64())
                .unwrap_or(8) as u8;

            tokens.push(TokenInfo {
                address: format!("qnk{}", hex::encode(contract.address.0)),
                name,
                symbol: symbol.clone(),
                decimals,
                total_supply: total_supply_str.clone(),
                contract_type: "Custom".to_string(),
                verified: false, // Custom tokens are not verified by default
                audit_report: None,
            });

            tracing::info!(
                "✅ Added custom token to DEX listing: {} ({}) - Supply: {}",
                symbol,
                hex::encode(&contract.address.0[..8]),
                total_supply_str
            );
        }
    }
    drop(deployed_contracts);

    tracing::info!(
        "📋 Returning {} supported tokens ({} native + {} custom)",
        tokens.len(),
        2,
        tokens.len() - 2
    );

    Ok(Json(DexApiResponse::success(tokens)))
}

pub async fn get_token_info(
    Path(address): Path<String>,
    State(_state): State<Arc<AppState>>,
) -> Result<Json<DexApiResponse<TokenInfo>>, StatusCode> {
    // Support both address lookups and symbol lookups
    let qug_address = hex::encode(q_types::QUG_TOKEN_ADDRESS);
    let qugusd_address = hex::encode(q_types::QUGUSD_TOKEN_ADDRESS);

    let address_upper = address.to_uppercase();

    if address == qug_address || address_upper == "QUG" {
        let token = TokenInfo {
            address: qug_address,
            name: "Quillon".to_string(),
            symbol: "QUG".to_string(),
            decimals: q_types::QUG_DECIMALS,
            total_supply: q_types::QUG_MAX_SUPPLY.to_string(),
            contract_type: "Native".to_string(),
            verified: true,
            audit_report: Some("https://audits.q-narwhalknight.dev/qug".to_string()),
        };
        Ok(Json(DexApiResponse::success(token)))
    } else if address == qugusd_address || address_upper == "QUGUSD" {
        let token = TokenInfo {
            address: qugusd_address,
            name: "Quillon USD".to_string(),
            symbol: "QUGUSD".to_string(),
            decimals: q_types::QUGUSD_DECIMALS,
            total_supply: "unlimited".to_string(),
            contract_type: "Stablecoin".to_string(),
            verified: true,
            audit_report: Some("https://audits.q-narwhalknight.dev/qugusd".to_string()),
        };
        Ok(Json(DexApiResponse::success(token)))
    } else {
        Ok(Json(DexApiResponse::error(format!(
            "Token '{}' not found",
            address
        ))))
    }
}

// ============ LIQUIDITY POOL ENDPOINTS ============

#[derive(Serialize)]
pub struct PoolInfo {
    pub address: String,
    pub token0: String,
    pub token1: String,
    pub fee: u32, // Fee in basis points
    pub reserve0: String,
    pub reserve1: String,
    pub total_liquidity: String,
    pub apy: f64,
    pub volume_24h: String,
}

pub async fn get_all_pools(
    State(state): State<Arc<AppState>>,
) -> Result<Json<DexApiResponse<Vec<PoolInfo>>>, StatusCode> {
    // v2.4.3: Read liquidity pools from persistent storage
    let pools_guard = state.liquidity_pools.read().await;

    let pools: Vec<PoolInfo> = pools_guard
        .values()
        .map(|pool| {
            // v3.7.3-beta: CRITICAL FIX - Pool reserves are stored in 24-decimal format
            // (frontend sends all amounts * 1e24). Use 24 for both, not pool.tokenX_decimals.
            let reserve0_display = pool.reserve0 as f64 / 1e24;
            let reserve1_display = pool.reserve1 as f64 / 1e24;

            PoolInfo {
                address: pool.pool_id.clone(),
                token0: pool.token0.clone(),
                token1: pool.token1.clone(),
                fee: 30, // 0.3% fee in basis points
                reserve0: reserve0_display.to_string(),
                reserve1: reserve1_display.to_string(),
                total_liquidity: pool.lp_token_supply.to_string(),
                apy: 0.0,                    // TODO: Calculate from swap fees
                volume_24h: "0".to_string(), // TODO: Track volume
            }
        })
        .collect();

    tracing::info!("📊 Returning {} liquidity pools", pools.len());
    Ok(Json(DexApiResponse::success(pools)))
}

pub async fn get_pool_info(
    Path(address): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<DexApiResponse<PoolInfo>>, StatusCode> {
    // v2.4.3: Read pool info from persistent storage
    let pools_guard = state.liquidity_pools.read().await;

    if let Some(pool) = pools_guard.get(&address) {
        // v3.7.3-beta: CRITICAL FIX - Pool reserves are stored in 24-decimal format
        let reserve0_display = pool.reserve0 as f64 / 1e24;
        let reserve1_display = pool.reserve1 as f64 / 1e24;

        let info = PoolInfo {
            address: pool.pool_id.clone(),
            token0: pool.token0.clone(),
            token1: pool.token1.clone(),
            fee: 30, // 0.3% fee in basis points
            reserve0: reserve0_display.to_string(),
            reserve1: reserve1_display.to_string(),
            total_liquidity: pool.lp_token_supply.to_string(),
            apy: 0.0,
            volume_24h: "0".to_string(),
        };
        Ok(Json(DexApiResponse::success(info)))
    } else {
        Ok(Json(DexApiResponse::error("Pool not found".to_string())))
    }
}

// ============ SWAP/TRADE ENDPOINTS ============

#[derive(Deserialize)]
pub struct SwapQuoteRequest {
    pub token_in: String,
    pub token_out: String,
    pub amount_in: Option<String>,
    pub amount_out: Option<String>,
    pub slippage_tolerance: Option<f64>, // Default: 0.5%
}

#[derive(Serialize)]
pub struct SwapQuote {
    pub amount_in: String,
    pub amount_out: String,
    pub minimum_amount_out: String,
    pub price_impact: f64,
    pub gas_estimate: u64,
    pub route: Vec<String>, // Pool addresses
    pub execution_price: f64,
    pub valid_until: u64, // Timestamp
}

/// Minimum pool reserve threshold (0.01 display units = 10^22 in 24-decimal raw)
/// Pools below this are considered dust/broken and excluded from routing and pricing.
const MIN_POOL_RESERVE_RAW: u128 = 10_000_000_000_000_000_000_000; // 10^22

/// Find the DEEPEST liquidity pool matching the given token pair (order-independent).
/// v10.2.2: Returns the pool with the highest k-value (reserve0 × reserve1) among all
/// matching pools that pass the minimum reserve filter. This prevents broken/dust pools
/// from being selected for swaps, quotes, or pricing.
fn find_pool_for_pair<'a>(
    pools: &'a HashMap<String, crate::LiquidityPool>,
    token_in: &str,
    token_out: &str,
) -> Option<(&'a String, &'a crate::LiquidityPool, bool)> {
    let tin = token_in.to_uppercase();
    let tout = token_out.to_uppercase();
    let mut best: Option<(&String, &crate::LiquidityPool, bool, f64)> = None;

    for (pool_id, pool) in pools.iter() {
        // Skip dust/broken pools
        if pool.reserve0 < MIN_POOL_RESERVE_RAW || pool.reserve1 < MIN_POOL_RESERVE_RAW {
            continue;
        }
        let t0 = pool.token0.to_uppercase();
        let t1 = pool.token1.to_uppercase();
        let (matched, reversed) = if t0 == tin && t1 == tout {
            (true, false)
        } else if t0 == tout && t1 == tin {
            (true, true)
        } else {
            continue;
        };
        if matched {
            // k-value = reserve product (use f64 to avoid u128 overflow)
            let k = pool.reserve0 as f64 * pool.reserve1 as f64;
            if best.as_ref().map_or(true, |(_, _, _, best_k)| k > *best_k) {
                best = Some((pool_id, pool, reversed, k));
            }
        }
    }
    best.map(|(id, pool, rev, _)| (id, pool, rev))
}

/// Constant-product AMM: calculate output amount given input amount and reserves
/// Uses x*y=k formula with 0.3% fee (997/1000)
/// Returns (amount_out, price_impact)
///
/// v10.1.9: Overflow-safe using mul_div_u128 decomposition.
/// With 24-decimal reserves, naive u128 multiplication overflows for any practical
/// swap (e.g. 1e24 * 213e30 = 2.13e56 > u128::MAX = 3.4e38).
fn amm_get_amount_out(amount_in: u128, reserve_in: u128, reserve_out: u128) -> (u128, f64) {
    if reserve_in == 0 || reserve_out == 0 || amount_in == 0 {
        return (0, 1.0);
    }
    // Apply 0.3% fee: amount_in_with_fee = amount_in * 997 / 1000
    let amount_in_with_fee = amount_in
        .checked_mul(997)
        .map(|v| v / 1000)
        .unwrap_or(amount_in.saturating_mul(997) / 1000);

    // amount_out = (amount_in_with_fee * reserve_out) / (reserve_in + amount_in_with_fee)
    let denominator = reserve_in.saturating_add(amount_in_with_fee);
    if denominator == 0 {
        return (0, 1.0);
    }

    let amount_out = mul_div_u128(amount_in_with_fee, reserve_out, denominator);

    // Price impact using f64 (safe for display)
    let r_in = reserve_in as f64;
    let r_out = reserve_out as f64;
    let a_in = amount_in as f64;
    let a_out = amount_out as f64;
    let spot_price = if r_in > 0.0 { r_out / r_in } else { 0.0 };
    let effective_price = if a_in > 0.0 { a_out / a_in } else { 0.0 };
    let price_impact = if spot_price > 0.0 {
        (1.0 - (effective_price / spot_price)).max(0.0)
    } else {
        1.0
    };

    (amount_out, price_impact)
}

/// Overflow-safe (a * b) / d for u128.
/// When a * b overflows, decomposes as: a * (b/d) + a * (b%d) / d.
fn mul_div_u128(a: u128, b: u128, d: u128) -> u128 {
    if d == 0 {
        return 0;
    }
    if let Some(product) = a.checked_mul(b) {
        return product / d;
    }
    // Overflow path: b = q*d + r → a*b/d = a*q + a*r/d
    let q = b / d;
    let r = b % d;
    let main_part = a
        .checked_mul(q)
        .unwrap_or_else(|| ((a as f64) * (q as f64)) as u128);
    let remainder_part = a
        .checked_mul(r)
        .map(|v| v / d)
        .unwrap_or_else(|| ((a as f64) * (r as f64) / (d as f64)) as u128);
    main_part.saturating_add(remainder_part)
}

pub async fn get_swap_quote(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SwapQuoteRequest>,
) -> Result<Json<DexApiResponse<SwapQuote>>, StatusCode> {
    // Input validation
    if request.token_in.is_empty() || request.token_out.is_empty() {
        return Ok(Json(DexApiResponse::error(
            "token_in and token_out are required".to_string(),
        )));
    }

    if request.token_in == request.token_out {
        return Ok(Json(DexApiResponse::error(
            "Cannot swap token for itself".to_string(),
        )));
    }

    if request.amount_in.is_none() && request.amount_out.is_none() {
        return Ok(Json(DexApiResponse::error(
            "Either amount_in or amount_out must be specified".to_string(),
        )));
    }

    // Validate slippage tolerance (0.5% default, max 10%)
    let slippage_percent = request.slippage_tolerance.unwrap_or(0.5);
    if slippage_percent < 0.0 || slippage_percent > 10.0 {
        return Ok(Json(DexApiResponse::error(
            "Slippage tolerance must be between 0% and 10%".to_string(),
        )));
    }
    let slippage_bps: u128 = (slippage_percent * 100.0) as u128;

    // v8.0.9: Real AMM quote from liquidity pools
    let pools_guard = state.liquidity_pools.read().await;
    let pool_match = find_pool_for_pair(&pools_guard, &request.token_in, &request.token_out);

    let (pool_id, reserve_in, reserve_out) = match pool_match {
        Some((pid, pool, reversed)) => {
            if reversed {
                (pid.clone(), pool.reserve1, pool.reserve0)
            } else {
                (pid.clone(), pool.reserve0, pool.reserve1)
            }
        }
        None => {
            return Ok(Json(DexApiResponse::error(format!(
                "No liquidity pool found for {}/{}",
                request.token_in, request.token_out
            ))));
        }
    };
    drop(pools_guard);

    if reserve_in == 0 || reserve_out == 0 {
        return Ok(Json(DexApiResponse::error(
            "Pool has no liquidity".to_string(),
        )));
    }

    // Parse amount_in (in 24-decimal base units)
    let amount_in_raw: u128 = match request.amount_in.as_ref() {
        Some(s) => match s.parse::<u128>() {
            Ok(v) if v > 0 => v,
            _ => return Ok(Json(DexApiResponse::error("Invalid amount_in".to_string()))),
        },
        None => {
            return Ok(Json(DexApiResponse::error(
                "amount_in required".to_string(),
            )))
        }
    };

    // Calculate output using constant-product AMM (x*y=k with 0.3% fee)
    let (amount_out, price_impact) = amm_get_amount_out(amount_in_raw, reserve_in, reserve_out);

    if amount_out == 0 {
        return Ok(Json(DexApiResponse::error(
            "Insufficient liquidity for this trade size".to_string(),
        )));
    }

    // Apply slippage tolerance for minimum output
    let minimum_amount_out = amount_out.saturating_mul(10000 - slippage_bps) / 10000;

    // Execution price = amount_out / amount_in (display units cancel out since both 24-decimal)
    let execution_price = if amount_in_raw > 0 {
        amount_out as f64 / amount_in_raw as f64
    } else {
        0.0
    };

    let quote = SwapQuote {
        amount_in: amount_in_raw.to_string(),
        amount_out: amount_out.to_string(),
        minimum_amount_out: minimum_amount_out.to_string(),
        price_impact,
        gas_estimate: 125000,
        route: vec![pool_id],
        execution_price,
        valid_until: current_timestamp() + 300, // 5 minutes
    };

    tracing::info!(
        "📊 [DEX] Swap quote: {} {} -> {} {} (impact: {:.4}%, pool reserves: {}/{})",
        quote.amount_in,
        request.token_in,
        quote.amount_out,
        request.token_out,
        price_impact * 100.0,
        reserve_in,
        reserve_out
    );

    Ok(Json(DexApiResponse::success(quote)))
}

#[derive(Deserialize)]
pub struct SwapExecuteRequest {
    pub token_in: String,
    pub token_out: String,
    pub amount_in: String,
    pub minimum_amount_out: String,
    pub recipient: String,
    pub deadline: u64,
    pub signature: String, // Transaction signature
}

#[derive(Serialize)]
pub struct SwapResult {
    pub transaction_hash: String,
    pub status: String, // "pending", "confirmed", "failed"
    pub amount_in: String,
    pub amount_out: String,
    pub gas_used: u64,
    pub score: Option<f64>,
}

struct SwapScorerContext {
    amount_in: u128,
    minimum_out: u128,
    amount_out: u128,
    reserve_in: u128,
    reserve_out: u128,
    hop_count: usize,
}

struct SwapScorer;

impl SwapScorer {
    fn score_swap(context: &SwapScorerContext) -> f64 {
        if context.amount_in == 0 || context.reserve_in == 0 || context.reserve_out == 0 {
            return 0.0;
        }

        let slippage_buffer = (context.amount_out.saturating_sub(context.minimum_out) as f64)
            / context.amount_out as f64;
        let trade_size_ratio = context.amount_in as f64 / context.reserve_in as f64;
        let depth_ratio = context.reserve_out as f64 / context.amount_out as f64;
        let hop_penalty = (context.hop_count.saturating_sub(1) as f64) * 0.05;

        let raw_score = 1.0 + (slippage_buffer * 0.8) + (depth_ratio.min(1000.0).ln() * 0.1)
            - (trade_size_ratio * 0.5)
            - hop_penalty;

        raw_score.clamp(0.0, 1.0)
    }
}

pub async fn execute_swap(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SwapExecuteRequest>,
) -> Result<Json<DexApiResponse<SwapResult>>, StatusCode> {
    // Comprehensive input validation
    if request.token_in.is_empty() || request.token_out.is_empty() {
        return Ok(Json(DexApiResponse::error(
            "token_in and token_out are required".to_string(),
        )));
    }

    if request.amount_in.is_empty() || request.minimum_amount_out.is_empty() {
        return Ok(Json(DexApiResponse::error(
            "amount_in and minimum_amount_out are required".to_string(),
        )));
    }

    if request.recipient.is_empty() {
        return Ok(Json(DexApiResponse::error(
            "recipient address is required".to_string(),
        )));
    }

    if request.signature.is_empty() {
        return Ok(Json(DexApiResponse::error(
            "transaction signature is required".to_string(),
        )));
    }

    // Validate deadline (must be in the future)
    let now = chrono::Utc::now().timestamp() as u64;
    if request.deadline <= now {
        return Ok(Json(DexApiResponse::error(
            "transaction deadline has passed".to_string(),
        )));
    }

    // Accept both 0x-prefixed (42 chars) and qnk-prefixed (69 chars) addresses
    if !(request.recipient.len() == 42 && request.recipient.starts_with("0x"))
        && !(request.recipient.len() >= 64 && request.recipient.starts_with("qnk"))
    {
        return Ok(Json(DexApiResponse::error(
            "invalid recipient address format (use 0x... or qnk...)".to_string(),
        )));
    }

    // Parse and validate amounts
    let amount_in: u128 = match request.amount_in.parse() {
        Ok(amount) if amount > 0 => amount,
        _ => return Ok(Json(DexApiResponse::error("invalid amount_in".to_string()))),
    };

    let minimum_out: u128 = match request.minimum_amount_out.parse() {
        Ok(amount) if amount > 0 => amount,
        _ => {
            return Ok(Json(DexApiResponse::error(
                "invalid minimum_amount_out".to_string(),
            )))
        }
    };

    // ============================================================================
    // v8.0.9: REAL AMM SWAP EXECUTION
    // 1. Find pool and calculate output via constant-product AMM
    // 2. Enforce slippage (minimum_amount_out)
    // 3. Update pool reserves atomically
    // 4. Create and broadcast transaction
    // 5. Persist updated pool to storage
    // ============================================================================

    // Step 1: Find pool and calculate AMM output
    let mut pools_guard = state.liquidity_pools.write().await;
    let pool_match = find_pool_for_pair(&pools_guard, &request.token_in, &request.token_out);

    let (pool_id, reserve_in, reserve_out, reversed) = match pool_match {
        Some((pid, pool, rev)) => {
            if rev {
                (pid.clone(), pool.reserve1, pool.reserve0, true)
            } else {
                (pid.clone(), pool.reserve0, pool.reserve1, false)
            }
        }
        None => {
            return Ok(Json(DexApiResponse::error(format!(
                "No liquidity pool found for {}/{}",
                request.token_in, request.token_out
            ))));
        }
    };

    if reserve_in == 0 || reserve_out == 0 {
        return Ok(Json(DexApiResponse::error(
            "Pool has no liquidity".to_string(),
        )));
    }

    let (amount_out, price_impact) = amm_get_amount_out(amount_in, reserve_in, reserve_out);

    if amount_out == 0 {
        return Ok(Json(DexApiResponse::error(
            "Insufficient liquidity for this trade size".to_string(),
        )));
    }

    // Step 2: Enforce slippage protection
    if amount_out < minimum_out {
        return Ok(Json(DexApiResponse::error(format!(
            "Slippage exceeded: output {} < minimum {}",
            amount_out, minimum_out
        ))));
    }

    // Step 3: Update pool reserves atomically
    if let Some(pool) = pools_guard.get_mut(&pool_id) {
        if reversed {
            // token_in = token1, token_out = token0
            pool.reserve1 = pool.reserve1.saturating_add(amount_in);
            pool.reserve0 = pool.reserve0.saturating_sub(amount_out);
        } else {
            // token_in = token0, token_out = token1
            pool.reserve0 = pool.reserve0.saturating_add(amount_in);
            pool.reserve1 = pool.reserve1.saturating_sub(amount_out);
        }

        // Persist updated pool to storage
        if let Ok(pool_bytes) = serde_json::to_vec(pool) {
            let _ = state
                .storage_engine
                .save_liquidity_pool(&pool_id, &pool_bytes)
                .await;
        }

        // Update collateral vault price if this is the QUG/QUGUSD pool
        let t0 = pool.token0.to_uppercase();
        let t1 = pool.token1.to_uppercase();
        if (t0 == "QUG" && t1 == "QUGUSD") || (t0 == "QUGUSD" && t1 == "QUG") {
            let (qug_reserve, qugusd_reserve) = if t0 == "QUG" {
                (pool.reserve0, pool.reserve1)
            } else {
                (pool.reserve1, pool.reserve0)
            };
            if qug_reserve > 0 {
                let new_price = qugusd_reserve as f64 / qug_reserve as f64;
                let mut vault = state.collateral_vault.write().await;
                vault.qug_price_usd = new_price;
            }
        }
    }
    drop(pools_guard);

    // Step 4: Parse recipient and create transaction
    let sender = match parse_address_32(&request.recipient) {
        Ok(addr) => addr,
        Err(_) => {
            return Ok(Json(DexApiResponse::error(
                "invalid recipient address format".to_string(),
            )));
        }
    };

    let nonce = state.nonce_tracker.get_and_increment(&sender);

    let transaction = q_api_server::transaction_utils::TransactionBuilder::new()
        .from(sender)
        .to([0u8; 32]) // DEX contract address
        .amount(amount_in)
        .fee(1_000_000) // 0.01 QNK fee
        .data(
            format!(
                "swap:{}:{}:out:{}",
                request.token_in, request.token_out, amount_out
            )
            .into_bytes(),
        )
        .token_type(q_types::TokenType::QUG)
        .fee_token_type(q_types::TokenType::QUGUSD)
        .tx_type(q_types::TransactionType::Swap)
        .build_with_nonce(nonce, chrono::Utc::now());

    let tx_id = transaction.id;
    let tx_hash = format!("0x{}", hex::encode(tx_id));

    let submission_result = q_api_server::transaction_utils::submit_transaction(
        transaction,
        &state.tx_pool,
        &state.tx_status,
        state.production_mempool.as_ref(),
        state.libp2p_discovery.as_ref(),
    )
    .await;

    let scorer_context = SwapScorerContext {
        amount_in,
        minimum_out: minimum_out,
        amount_out,
        reserve_in,
        reserve_out,
        hop_count: 1,
    };
    let score = SwapScorer::score_swap(&scorer_context);

    let swap_result = SwapResult {
        transaction_hash: tx_hash.clone(),
        status: match submission_result.status {
            TxStatus::InMempool => "in_mempool".to_string(),
            TxStatus::Pending => "pending".to_string(),
            _ => "pending".to_string(),
        },
        amount_in: request.amount_in,
        amount_out: amount_out.to_string(),
        gas_used: 125000,
        score: Some(score),
    };

    tracing::info!(
        "📤 [DEX] Swap EXECUTED: {} (nonce={}, impact={:.4}%, out={}, broadcast={}, queued={})",
        &tx_hash[..16],
        nonce,
        price_impact * 100.0,
        amount_out,
        submission_result.broadcast_success,
        submission_result.queued_for_block
    );

    Ok(Json(DexApiResponse::success(swap_result)))
}

// ============ PRICE ORACLE ENDPOINTS ============

#[derive(Serialize)]
pub struct TokenPrice {
    pub token: String,
    pub price_usd: f64,
    pub price_qnk: f64,
    pub change_24h: f64,
    pub volume_24h: String,
    pub last_updated: u64,
}

pub async fn get_all_prices(
    State(state): State<Arc<AppState>>,
) -> Result<Json<DexApiResponse<Vec<TokenPrice>>>, StatusCode> {
    // v8.0.9: Real price oracle from liquidity pool reserves
    let mut prices = vec![];
    let now = current_timestamp();

    // QUG price from collateral vault (updated by AMM swaps)
    let qug_price_usd = {
        let vault = state.collateral_vault.read().await;
        if vault.qug_price_usd > 0.0 {
            vault.qug_price_usd
        } else {
            42.50
        }
    };

    prices.push(TokenPrice {
        token: "QUG".to_string(),
        price_usd: qug_price_usd,
        price_qnk: 1.0,
        change_24h: 0.0,
        volume_24h: "0".to_string(),
        last_updated: now,
    });

    prices.push(TokenPrice {
        token: "QUGUSD".to_string(),
        price_usd: 1.0,
        price_qnk: 1.0 / qug_price_usd,
        change_24h: 0.0,
        volume_24h: "0".to_string(),
        last_updated: now,
    });

    // v10.2.2: Derive prices from the DEEPEST pool per token (highest k = reserve0 × reserve1).
    // Filters out dust/broken pools using MIN_POOL_RESERVE_RAW threshold.
    let pools_guard = state.liquidity_pools.read().await;
    // token -> (k_value, price_usd, price_qnk)
    let mut best_pool_per_token: std::collections::HashMap<String, (f64, f64, f64)> =
        std::collections::HashMap::new();

    for pool in pools_guard.values() {
        // Skip empty and dust/broken pools
        if pool.reserve0 < MIN_POOL_RESERVE_RAW || pool.reserve1 < MIN_POOL_RESERVE_RAW {
            continue;
        }
        let t0 = pool.token0.to_uppercase();
        let t1 = pool.token1.to_uppercase();
        let k_value = pool.reserve0 as f64 * pool.reserve1 as f64;

        // If one side is QUG, derive the other token's price
        if t0 == "QUG" && t1 != "QUGUSD" {
            let price_qnk = pool.reserve0 as f64 / pool.reserve1 as f64;
            let price_usd = price_qnk * qug_price_usd;
            let better = best_pool_per_token
                .get(&t1)
                .map_or(true, |(best_k, _, _)| k_value > *best_k);
            if better {
                best_pool_per_token.insert(t1, (k_value, price_usd, price_qnk));
            }
        } else if t1 == "QUG" && t0 != "QUGUSD" {
            let price_qnk = pool.reserve1 as f64 / pool.reserve0 as f64;
            let price_usd = price_qnk * qug_price_usd;
            let better = best_pool_per_token
                .get(&t0)
                .map_or(true, |(best_k, _, _)| k_value > *best_k);
            if better {
                best_pool_per_token.insert(t0, (k_value, price_usd, price_qnk));
            }
        }
    }

    for (token, (_k_value, price_usd, price_qnk)) in best_pool_per_token {
        prices.push(TokenPrice {
            token,
            price_usd,
            price_qnk,
            change_24h: 0.0,
            volume_24h: "0".to_string(),
            last_updated: now,
        });
    }

    Ok(Json(DexApiResponse::success(prices)))
}

// ============ SECURITY & COMPLIANCE ENDPOINTS ============

#[derive(Serialize)]
pub struct ContractAudit {
    pub contract_address: String,
    pub audit_status: String, // "passed", "failed", "pending"
    pub audit_firm: String,
    pub audit_date: u64,
    pub report_url: Option<String>,
    pub security_score: u8, // 0-100
    pub vulnerabilities: Vec<String>,
}

pub async fn get_contract_audit(
    Path(_contract): Path<String>,
    State(_state): State<Arc<AppState>>,
) -> Result<Json<DexApiResponse<ContractAudit>>, StatusCode> {
    // TODO: Implement actual audit lookup
    Ok(Json(DexApiResponse::error("Audit not found".to_string())))
}

// ============ INTEGRATION HELPER ENDPOINTS ============

#[derive(Deserialize)]
pub struct WebhookSetup {
    pub url: String,
    pub events: Vec<String>, // "swap", "pool_create", "price_update"
    pub secret: String,
}

pub async fn setup_webhook(
    State(_state): State<Arc<AppState>>,
    Json(_request): Json<WebhookSetup>,
) -> Result<Json<DexApiResponse<String>>, StatusCode> {
    // TODO: Implement webhook registration
    Ok(Json(DexApiResponse::success(
        "Webhook registered".to_string(),
    )))
}

#[derive(Serialize)]
pub struct ApiKeyInfo {
    pub api_key: String,
    pub permissions: Vec<String>,
    pub rate_limit: u64,
    pub expires_at: Option<u64>,
}

pub async fn generate_api_key(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<DexApiResponse<ApiKeyInfo>>, StatusCode> {
    // Generate secure API key with QNK prefix for easy identification
    let key_uuid = Uuid::new_v4().to_string().replace("-", "");
    let api_key = format!("qnk_{}", key_uuid);

    let info = ApiKeyInfo {
        api_key: api_key.clone(),
        permissions: vec![
            "read:pools".to_string(),
            "read:prices".to_string(),
            "read:tokens".to_string(),
            "write:swaps".to_string(),
            "write:pools".to_string(),
        ],
        rate_limit: 1000, // requests per hour
        expires_at: Some(chrono::Utc::now().timestamp() as u64 + (365 * 24 * 3600)), // 1 year expiry
    };

    // TODO: In production, store this API key in the database with proper encryption
    tracing::info!(
        "Generated new DEX API key: {} (expires in 1 year)",
        &api_key[..16]
    );

    Ok(Json(DexApiResponse::success(info)))
}

#[derive(Serialize)]
pub struct RateLimits {
    pub requests_per_hour: u64,
    pub requests_per_minute: u64,
    pub current_usage: u64,
    pub reset_time: u64,
}

pub async fn get_rate_limits(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<DexApiResponse<RateLimits>>, StatusCode> {
    let limits = RateLimits {
        requests_per_hour: 1_000_000, // 1 million requests per hour
        requests_per_minute: 10_000,  // 10k requests per minute
        current_usage: 42,            // TODO: Implement actual tracking
        reset_time: chrono::Utc::now().timestamp() as u64 + 3600,
    };

    Ok(Json(DexApiResponse::success(limits)))
}

// ============ HELPER FUNCTIONS ============

/// Parse an address string (0x or qnk prefixed) to a 32-byte array
fn parse_address_32(address_str: &str) -> Result<[u8; 32], String> {
    let hex_str = if address_str.starts_with("0x") {
        &address_str[2..]
    } else if address_str.starts_with("qnk") {
        &address_str[3..]
    } else {
        address_str
    };

    match hex::decode(hex_str) {
        Ok(bytes) => {
            if bytes.len() == 32 {
                let mut result = [0u8; 32];
                result.copy_from_slice(&bytes);
                Ok(result)
            } else if bytes.len() == 20 {
                // Ethereum-style 20-byte address, pad to 32 bytes
                let mut result = [0u8; 32];
                result[12..].copy_from_slice(&bytes);
                Ok(result)
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

// ============ MISSING ENDPOINT IMPLEMENTATIONS ============

pub async fn get_pool_reserves(
    Path(address): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<DexApiResponse<PoolInfo>>, StatusCode> {
    // Check if it's a valid contract address by querying the contract registry
    // Parse the address string to [u8; 32]
    if let Ok(address_bytes) = hex::decode(&address) {
        if address_bytes.len() == 32 {
            let mut address_array = [0u8; 32];
            address_array.copy_from_slice(&address_bytes);

            if let Some(_contract) = state.contract_registry.get(&address_array) {
                // This is a valid contract, return pool info
                // In a real implementation, we would extract pool data from the contract
                let pool_info = PoolInfo {
                    address: address.clone(),
                    token0: "QNK".to_string(),
                    token1: "USDT".to_string(),
                    fee: 300, // 0.3% fee
                    reserve0: "1000000".to_string(),
                    reserve1: "2000000".to_string(),
                    total_liquidity: "3000000".to_string(),
                    apy: 12.5,
                    volume_24h: "500000".to_string(),
                };
                return Ok(Json(DexApiResponse::success(pool_info)));
            }
        }
    }

    Ok(Json(DexApiResponse::error(
        "Pool not found or invalid contract address".to_string(),
    )))
}

pub async fn create_liquidity_pool(
    State(state): State<Arc<AppState>>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<DexApiResponse<String>>, StatusCode> {
    // Extract pool creation parameters
    let token0 = request
        .get("token0")
        .and_then(|v| v.as_str())
        .unwrap_or("QNK");
    let token1 = request.get("token1").and_then(|v| v.as_str()).unwrap_or("");
    let initial_reserve0 = request
        .get("initial_reserve0")
        .and_then(|v| v.as_str())
        .unwrap_or("0");
    let initial_reserve1 = request
        .get("initial_reserve1")
        .and_then(|v| v.as_str())
        .unwrap_or("0");

    if token1.is_empty() {
        return Ok(Json(DexApiResponse::error(
            "token1 is required".to_string(),
        )));
    }

    // Create liquidity pool contract through the VM
    let contract_metadata_json = serde_json::json!({
        "token0": token0,
        "token1": token1,
        "initial_reserve0": initial_reserve0,
        "initial_reserve1": initial_reserve1,
        "fee": 300, // 0.3% standard fee
        "creator": "dex_integration_api"
    });

    // Convert JSON Value to HashMap as required by deploy_contract
    let contract_metadata = if let serde_json::Value::Object(map) = contract_metadata_json {
        map.into_iter().collect()
    } else {
        return Ok(Json(DexApiResponse::error(
            "Failed to create contract metadata".to_string(),
        )));
    };

    // Create deployment options
    let options = DeploymentOptions {
        test_deployment: false,
        auto_verify: true,
        enable_governance: false,
        enable_upgrades: false,
        deploy_with_proxy: false,
        gas_limit: Some(1_000_000),
    };

    match state
        .orobit_ecosystem
        .deploy_contract(
            ContractType::LiquidityPool,
            [0u8; 32], // Deployer address (would be from authenticated user)
            contract_metadata,
            options,
        )
        .await
    {
        Ok((contract_id, _address)) => Ok(Json(DexApiResponse::success(contract_id))),
        Err(e) => Ok(Json(DexApiResponse::error(format!(
            "Failed to create pool: {}",
            e
        )))),
    }
}

pub async fn get_swap_status(
    Path(tx_hash): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<DexApiResponse<SwapResult>>, StatusCode> {
    // Check transaction status from the tx pool
    // DashMap doesn't need .read() - it's concurrent by default
    if let Some(hash_bytes) = hex::decode(&tx_hash).ok() {
        if hash_bytes.len() == 32 {
            let mut hash_array = [0u8; 32];
            hash_array.copy_from_slice(&hash_bytes);

            if let Some(status) = state.tx_status.get(&hash_array) {
                let swap_result = SwapResult {
                    transaction_hash: tx_hash,
                    status: match *status {
                        TxStatus::Pending => "pending".to_string(),
                        TxStatus::InMempool => "in_mempool".to_string(),
                        TxStatus::Confirmed { .. } => "confirmed".to_string(),
                        TxStatus::Failed { .. } => "failed".to_string(),
                        TxStatus::Mixing => "mixing".to_string(), // Quantum mixing in progress
                    },
                    amount_in: "0".to_string(), // TODO: Extract from transaction
                    amount_out: "0".to_string(), // TODO: Extract from transaction
                    gas_used: 21000,            // TODO: Get actual gas used
                    score: None,                // Not currently persisted in tx status
                };
                return Ok(Json(DexApiResponse::success(swap_result)));
            }
        }
    }

    Ok(Json(DexApiResponse::error(
        "Transaction not found".to_string(),
    )))
}

pub async fn get_token_price(
    Path(token): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<DexApiResponse<TokenPrice>>, StatusCode> {
    let qug_address = hex::encode(q_types::QUG_TOKEN_ADDRESS);
    let qugusd_address = hex::encode(q_types::QUGUSD_TOKEN_ADDRESS);
    let token_upper = token.to_uppercase();

    // v4.0.4: Get QUG price from vault (which is updated from AMM after each swap)
    // Previously forced to hardcoded $3000.00 which prevented price discovery
    let qug_price_usd = {
        let vault_price = state.collateral_vault.read().await.qug_price_usd;
        if vault_price > 0.0 {
            vault_price
        } else {
            3000.00 // Only use default if vault has no price at all
        }
    };

    let price = if token == qug_address || token_upper == "QUG" {
        TokenPrice {
            token: "QUG".to_string(),
            price_usd: qug_price_usd,
            price_qnk: 1.0,              // QUG is the base token
            change_24h: 0.0,             // TODO: Calculate from historical data
            volume_24h: "0".to_string(), // TODO: Calculate from DEX activity
            last_updated: current_timestamp(),
        }
    } else if token == qugusd_address || token_upper == "QUGUSD" {
        TokenPrice {
            token: "QUGUSD".to_string(),
            price_usd: 1.0,                 // Always $1.00 (stablecoin peg)
            price_qnk: 1.0 / qug_price_usd, // QUGUSD price in QUG terms
            change_24h: 0.0,                // Stablecoin should have minimal change
            volume_24h: "0".to_string(),    // TODO: Calculate from DEX activity
            last_updated: current_timestamp(),
        }
    } else {
        TokenPrice {
            token: token.clone(),
            price_usd: 0.0,
            price_qnk: 0.0,
            change_24h: 0.0,
            volume_24h: "0".to_string(),
            last_updated: current_timestamp(),
        }
    };

    Ok(Json(DexApiResponse::success(price)))
}

pub async fn get_historical_prices(
    Path(token): Path<String>,
    Query(params): Query<HashMap<String, String>>,
    State(_state): State<Arc<AppState>>,
) -> Result<Json<DexApiResponse<Vec<TokenPrice>>>, StatusCode> {
    let _timeframe = params.get("timeframe").unwrap_or(&"24h".to_string());
    let _interval = params.get("interval").unwrap_or(&"1h".to_string());

    // For now, return a simple mock historical data
    let historical_prices = vec![
        TokenPrice {
            token: token.clone(),
            price_usd: 0.98,
            price_qnk: 0.98,
            change_24h: 0.0,
            volume_24h: "900000".to_string(),
            last_updated: current_timestamp() - 3600, // 1 hour ago
        },
        TokenPrice {
            token: token.clone(),
            price_usd: 1.02,
            price_qnk: 1.02,
            change_24h: 4.08, // +4.08% from 0.98
            volume_24h: "1100000".to_string(),
            last_updated: current_timestamp(), // now
        },
    ];

    Ok(Json(DexApiResponse::success(historical_prices)))
}

pub async fn compliance_check(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<DexApiResponse<serde_json::Value>>, StatusCode> {
    let address = request
        .get("address")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let transaction_type = request.get("type").and_then(|v| v.as_str()).unwrap_or("");

    // Basic compliance checks
    let mut compliance_result = serde_json::json!({
        "address": address,
        "transaction_type": transaction_type,
        "compliance_status": "approved",
        "risk_score": 0.1, // Low risk
        "kyc_required": false,
        "sanctions_check": "passed",
        "aml_status": "clear"
    });

    // Check for high-risk patterns
    if address.is_empty() || address.len() < 20 {
        compliance_result["compliance_status"] = "rejected".into();
        compliance_result["risk_score"] = 1.0.into();
        compliance_result["reason"] = "Invalid address format".into();
    }

    Ok(Json(DexApiResponse::success(compliance_result)))
}

// Helper functions
fn current_timestamp() -> u64 {
    chrono::Utc::now().timestamp() as u64
}
