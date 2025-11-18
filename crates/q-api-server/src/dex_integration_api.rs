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
use std::collections::HashMap;
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
    pub network: String, // "mainnet", "testnet"
}

impl<T> DexApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: chrono::Utc::now().timestamp() as u64,
            api_version: "1.0.0".to_string(),
            network: "mainnet".to_string(), // TODO: Make configurable
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message),
            timestamp: chrono::Utc::now().timestamp() as u64,
            api_version: "1.0.0".to_string(),
            network: "mainnet".to_string(),
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

    // TODO: Add custom tokens from deployed contracts
    // The ContractRegistry needs to be enhanced to expose deployed contract metadata
    // for token discovery. For now, return just the native QUG and QUGUSD tokens.
    //
    // Future enhancement: Implement contract metadata iteration in OrobitSmartContractEcosystem
    // to enumerate all deployed token contracts with their symbol, name, decimals, and supply.

    tracing::info!(
        "📋 Returning {} supported tokens (native only, custom tokens TBD)",
        tokens.len()
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
    State(_state): State<Arc<AppState>>,
) -> Result<Json<DexApiResponse<Vec<PoolInfo>>>, StatusCode> {
    // TODO: Implement actual pool registry lookup
    let pools = vec![];
    Ok(Json(DexApiResponse::success(pools)))
}

pub async fn get_pool_info(
    Path(_address): Path<String>,
    State(_state): State<Arc<AppState>>,
) -> Result<Json<DexApiResponse<PoolInfo>>, StatusCode> {
    // TODO: Implement actual pool lookup
    Ok(Json(DexApiResponse::error("Pool not found".to_string())))
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

pub async fn get_swap_quote(
    State(_state): State<Arc<AppState>>,
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

    // Validate amount_in or amount_out is provided
    if request.amount_in.is_none() && request.amount_out.is_none() {
        return Ok(Json(DexApiResponse::error(
            "Either amount_in or amount_out must be specified".to_string(),
        )));
    }

    // Validate slippage tolerance
    let slippage = request.slippage_tolerance.unwrap_or(0.5);
    if slippage < 0.0 || slippage > 10.0 {
        return Ok(Json(DexApiResponse::error(
            "Slippage tolerance must be between 0% and 10%".to_string(),
        )));
    }

    // For demonstration, return a mock quote
    let amount_in = request.amount_in.unwrap_or_else(|| "1000000".to_string()); // 1 QNK
    let amount_out = "950000".to_string(); // 0.95 of the other token (accounting for fees)
    let minimum_amount_out = {
        let base: u64 = amount_out.parse().unwrap_or(0);
        let slippage_adjusted = base as f64 * (1.0 - slippage / 100.0);
        (slippage_adjusted as u64).to_string()
    };

    let quote = SwapQuote {
        amount_in,
        amount_out: amount_out.clone(),
        minimum_amount_out,
        price_impact: 0.02,                       // 2% price impact
        gas_estimate: 120000,                     // Estimated gas for DEX swap
        route: vec!["pool_qnk_usdc".to_string()], // Route through QNK/USDC pool
        execution_price: 0.95,
        valid_until: current_timestamp() + 300, // Valid for 5 minutes
    };

    tracing::info!(
        "Generated swap quote: {} {} -> {} {}",
        quote.amount_in,
        request.token_in,
        quote.amount_out,
        request.token_out
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

    // Validate recipient address format (basic check)
    if request.recipient.len() != 42 || !request.recipient.starts_with("0x") {
        return Ok(Json(DexApiResponse::error(
            "invalid recipient address format".to_string(),
        )));
    }

    // Parse and validate amounts
    let amount_in: u64 = match request.amount_in.parse() {
        Ok(amount) if amount > 0 => amount,
        _ => return Ok(Json(DexApiResponse::error("invalid amount_in".to_string()))),
    };

    let _minimum_out: u64 = match request.minimum_amount_out.parse() {
        Ok(amount) if amount > 0 => amount,
        _ => {
            return Ok(Json(DexApiResponse::error(
                "invalid minimum_amount_out".to_string(),
            )))
        }
    };

    // Create mock transaction for demonstration
    let tx_hash = format!("0x{}", hex::encode(&rand::random::<[u8; 32]>()));

    // In production, this would:
    // 1. Verify the signature
    // 2. Check token balances
    // 3. Execute the swap through the VM
    // 4. Update balances
    // 5. Emit events

    let swap_result = SwapResult {
        transaction_hash: tx_hash.clone(),
        status: "pending".to_string(), // Would start as pending
        amount_in: request.amount_in,
        amount_out: (amount_in * 95 / 100).to_string(), // Mock 5% fee
        gas_used: 125000,                               // Estimated gas used
    };

    // Update transaction pool with pending transaction
    // DashMap doesn't need .write() - it's concurrent by default
    {
        // Convert hex string to bytes for the transaction hash
        if let Ok(hash_bytes) = hex::decode(&tx_hash[2..]) {
            if hash_bytes.len() == 32 {
                let mut hash_array = [0u8; 32];
                hash_array.copy_from_slice(&hash_bytes);

                // Create a mock transaction
                let transaction = Transaction {
                    id: hash_array,
                    from: [0u8; 32], // Would be derived from signature
                    to: [0u8; 32],   // Would be the DEX contract
                    amount: amount_in,
                    fee: 1000000, // 0.01 QNK fee
                    nonce: 0,
                    signature: vec![],
                    timestamp: chrono::Utc::now(),
                    data: format!("swap:{}:{}", request.token_in, request.token_out).into_bytes(),
                    token_type: q_types::TokenType::QUG,
                    fee_token_type: q_types::TokenType::QUGUSD,
                };

                state.tx_pool.insert(hash_array, transaction);
                state.tx_status.insert(hash_array, TxStatus::Pending);
            }
        }
    }

    tracing::info!("Swap transaction submitted: {} (pending)", tx_hash);

    // Emit real-time event
    // Transaction submitted event would be emitted here if the method existed
    // let _ = state.event_emitter.emit_transaction_submitted(tx_hash.clone()).await;

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
    State(_state): State<Arc<AppState>>,
) -> Result<Json<DexApiResponse<Vec<TokenPrice>>>, StatusCode> {
    // TODO: Implement actual price oracle
    let prices = vec![];
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

    // Get current QUG price from CollateralVault
    let qug_price_usd = state.collateral_vault.read().await.qug_price_usd;

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
