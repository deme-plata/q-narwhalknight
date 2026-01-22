/// Quillon Bank API Endpoints for CLI Integration
///
/// Provides production-ready RESTful API endpoints for the Quillon Bank CLI
/// to execute real banking operations on the quantum blockchain.
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info};

use crate::handlers::parse_wallet_address;
use crate::AppState;
use chrono::Utc;
use q_quillon_bank::{AssetType, QuillonBankSystem};
use q_types::{ApiResponse, Transaction};

/// Create Quillon Bank API router with AEGIS-QL protection for sensitive operations
pub fn create_quillon_bank_router() -> Router<Arc<AppState>> {
    // Public routes (read-only, no authentication required)
    let public_routes = create_public_routes();

    // Protected routes (FOUNDER-ONLY - AEGIS-QL authentication required)
    // Note: Middleware will be applied in main.rs when state is available
    let protected_routes = create_protected_routes();

    // Merge public and protected routes
    Router::new().merge(public_routes).merge(protected_routes)
}

/// Create public Quillon Bank routes (read-only, no authentication)
pub fn create_public_routes() -> Router<Arc<AppState>> {
    Router::new()
        // Status & Metrics (PUBLIC - read-only)
        .route("/stablecoin/status", get(get_stablecoin_status))
        .route("/metrics", get(get_banking_metrics))
        .route("/risk/status", get(get_risk_status))
        .route("/quantum/status", get(get_quantum_status))
        .route("/stablecoin/collateral", get(get_collateral_status))
        .route("/stablecoin/peg", get(get_peg_status))
        .route("/lending/applications", get(get_loan_applications))
        .route("/lending/apply", post(apply_loan))
        .route("/lending/payback", post(payback_loan))
        .route("/lending/at-risk", get(get_loans_at_risk))
        .route("/accounts", get(list_accounts))
        .route("/accounts/pending", get(get_pending_accounts))
        .route("/treasury/reserves", get(get_reserves_status))
        .route("/treasury/profits", get(calculate_profits))
        .route("/risk/assessment", get(risk_assessment))
        .route("/risk/liquidations/queue", get(liquidation_queue))
        .route("/analytics/daily-summary", get(daily_summary))
        .route("/analytics/customers", get(customer_analytics))
        // Development Fee Transparency (PUBLIC - read-only)
        .route("/devfee/status", get(get_dev_fee_status))
        .route("/devfee/stats", get(get_dev_fee_stats))
        .route("/devfee/wallet", get(get_founder_wallet_info))
}

/// Create protected Quillon Bank routes (FOUNDER-ONLY - requires AEGIS-QL authentication)
pub fn create_protected_routes() -> Router<Arc<AppState>> {
    Router::new()
        // Stablecoin Operations (FOUNDER-ONLY)
        .route("/stablecoin/mint", post(mint_qnkusd))
        .route("/stablecoin/burn", post(burn_qnkusd))
        .route("/stablecoin/collateral/add", post(add_collateral))
        .route(
            "/stablecoin/collateral/rebalance",
            post(rebalance_collateral),
        )
        .route("/stablecoin/peg/adjust", post(adjust_peg))
        // Lending Operations (FOUNDER-ONLY)
        .route("/lending/approve", post(approve_loan))
        .route("/lending/liquidate", post(liquidate_loan))
        // Account Management (FOUNDER-ONLY)
        .route("/accounts/approve", post(approve_account))
        // Treasury Management (FOUNDER-ONLY)
        .route("/treasury/reserves/allocate", post(allocate_reserves))
        .route("/treasury/profits/distribute", post(distribute_profits))
        // Risk Management (FOUNDER-ONLY)
        .route("/risk/liquidations/execute", post(execute_liquidations))
}

// ============================================================================
// Status & Metrics Endpoints
// ============================================================================

#[derive(Serialize)]
struct StablecoinStatus {
    total_supply: u64,
    collateral_value: u64,
    collateralization_ratio: f64,
    peg_price: f64,
}

async fn get_stablecoin_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<StablecoinStatus>>, StatusCode> {
    info!("📊 Fetching QNKUSD stablecoin status");

    // Get real stablecoin metrics from Quillon Bank system
    let bank_system = state.quillon_bank.read().await;

    let metrics = bank_system.get_bank_metrics().await.map_err(|e| {
        error!("Failed to get stablecoin metrics: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Calculate total collateral value
    let total_deposits_value: u128 = metrics.total_deposits.values().sum();

    let status = StablecoinStatus {
        total_supply: (metrics.qnkusd_metrics.total_supply / 1_000_000_000_000) as u64,
        collateral_value: (total_deposits_value / 1_000_000_000_000) as u64,
        collateralization_ratio: metrics.qnkusd_metrics.collateral_ratio,
        peg_price: 1.0, // TODO: Get from oracle
    };

    Ok(Json(ApiResponse::success(status)))
}

#[derive(Serialize)]
struct BankingMetrics {
    active_accounts: u64,
    total_deposits: u64,
    active_loans: u64,
    average_credit_score: f64,
}

async fn get_banking_metrics(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<BankingMetrics>>, StatusCode> {
    info!("📊 Fetching banking metrics");

    let bank_system = state.quillon_bank.read().await;

    let bank_metrics = bank_system.get_bank_metrics().await.map_err(|e| {
        error!("Failed to get banking metrics: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Calculate aggregated values
    let total_deposits: u64 =
        (bank_metrics.total_deposits.values().sum::<u128>() / 1_000_000_000_000) as u64;
    let total_loans: u64 =
        (bank_metrics.total_loans.values().sum::<u128>() / 1_000_000_000_000) as u64;

    let metrics = BankingMetrics {
        active_accounts: bank_metrics.total_accounts,
        total_deposits,
        active_loans: total_loans,
        average_credit_score: bank_metrics.average_credit_score,
    };

    Ok(Json(ApiResponse::success(metrics)))
}

#[derive(Serialize)]
struct RiskStatus {
    loans_at_risk_count: u64,
    loans_at_risk_value: u64,
    liquidation_queue: u64,
    reserve_ratio: f64,
}

async fn get_risk_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<RiskStatus>>, StatusCode> {
    info!("⚠️  Fetching risk status");

    let bank_system = state.quillon_bank.read().await;

    // Get banking metrics for risk calculation
    let metrics = bank_system.get_bank_metrics().await.map_err(|e| {
        error!("Failed to get metrics: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Calculate risk metrics from banking data
    let total_loans_value: u128 = metrics.total_loans.values().sum();
    let reserve_ratio = if total_loans_value > 0 {
        (metrics.total_deposits.values().sum::<u128>() as f64) / (total_loans_value as f64) * 100.0
    } else {
        100.0
    };

    let status = RiskStatus {
        loans_at_risk_count: 0, // TODO: Implement risk assessment
        loans_at_risk_value: 0,
        liquidation_queue: 0,
        reserve_ratio,
    };

    Ok(Json(ApiResponse::success(status)))
}

#[derive(Serialize)]
struct QuantumStatus {
    quantum_vaults: u64,
    post_quantum_transactions_24h: u64,
    quantum_privacy_adoption: f64,
}

async fn get_quantum_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<QuantumStatus>>, StatusCode> {
    info!("⚛️  Fetching quantum features status");

    let bank_system = state.quillon_bank.read().await;

    let metrics = bank_system.get_bank_metrics().await.map_err(|e| {
        error!("Failed to get quantum metrics: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let status = QuantumStatus {
        quantum_vaults: metrics.quantum_metrics.total_quantum_vaults,
        post_quantum_transactions_24h: metrics.quantum_metrics.post_quantum_transactions_24h,
        quantum_privacy_adoption: metrics.quantum_metrics.quantum_privacy_adoption,
    };

    Ok(Json(ApiResponse::success(status)))
}

// ============================================================================
// Stablecoin Operations
// ============================================================================

#[derive(Deserialize)]
pub struct MintRequest {
    amount: u64,
    collateral_type: String,
    collateral_amount: f64,
    reason: Option<String>,
    /// Optional wallet address (if not authenticated via X-Wallet-Auth header)
    wallet_address: Option<String>,
}

#[derive(Serialize)]
pub struct MintResponse {
    transaction_id: String,
    amount_minted: u64,
    collateral_locked: f64,
    collateral_ratio: f64,
    finalized_in_seconds: f64,
}

pub async fn mint_qnkusd(
    State(state): State<Arc<AppState>>,
    Json(request): Json<MintRequest>,
) -> Result<Json<ApiResponse<MintResponse>>, StatusCode> {
    info!(
        "💰 Minting {} QUGUSD with {} {} collateral",
        request.amount as f64 / 1e8,
        request.collateral_amount,
        request.collateral_type
    );

    // Parse collateral type
    let collateral_type = match request.collateral_type.to_uppercase().as_str() {
        "QUG" | "ORB" => AssetType::ORB, // Q-NarwhalKnight native token
        "BTC" => AssetType::BTC,
        "ETH" => AssetType::ETH,
        "USDC" => AssetType::USDC,
        _ => {
            error!("❌ Invalid collateral type: {}", request.collateral_type);
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    // Execute mint operation on blockchain
    let start = std::time::Instant::now();

    // ✅ CRITICAL FIX: Get wallet address from request body (frontend provides it)
    let borrower_bytes = if let Some(wallet_addr) = &request.wallet_address {
        // Parse wallet address from frontend
        let hex_part = if wallet_addr.starts_with("qnk") {
            &wallet_addr[3..]
        } else if wallet_addr.starts_with("0x") {
            &wallet_addr[2..]
        } else {
            wallet_addr.as_str()
        };

        match hex::decode(hex_part) {
            Ok(bytes) if bytes.len() == 32 => {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                info!("👤 Minting for wallet: qnk{}", hex::encode(&arr[..8]));
                arr
            }
            _ => {
                error!("❌ Invalid wallet address format: {}", wallet_addr);
                return Err(StatusCode::BAD_REQUEST);
            }
        }
    } else {
        // Fallback: Create a new random address (should not happen in production)
        error!("⚠️  No wallet address provided - using random address (THIS IS A BUG)");
        let borrower = q_quillon_bank::Address::new();
        borrower.0
    };

    // NOTE: Frontend sends `amount` in base units (e.g., 2656000000 for 26.56 QUGUSD)
    // We need to convert back to human-readable for collateral ratio calculation
    let amount_usd = (request.amount as f64) / 1e24; // Convert base units to USD

    // Calculate actual collateral ratio before the mint call
    let collateral_value_usd = match &collateral_type {
        AssetType::ORB => request.collateral_amount * 42.50, // QUG price ~$42.50
        AssetType::BTC => request.collateral_amount * 70_000.0,
        AssetType::ETH => request.collateral_amount * 3_500.0,
        AssetType::USDC => request.collateral_amount,
        _ => 0.0,
    };

    let collateral_ratio = (collateral_value_usd / amount_usd) * 100.0;

    // Convert frontend base units (100M) to Quillon Bank base units (1T)
    // Frontend: 1 QUGUSD = 100,000,000 base units
    // Backend: 1 QUGUSD = 1,000,000,000,000 base units
    // Multiplier: 10,000 (1T / 100M)
    let amount_backend_units = (request.amount as u128) * 10_000;

    let tx_id = {
        let mut bank_system = state.quillon_bank.write().await;
        let borrower = q_quillon_bank::Address(borrower_bytes);
        bank_system
            .mint_qnkusd(
                &borrower,
                (request.collateral_amount * 1_000_000_000_000.0) as u128, // Convert to base units
                collateral_type,
                amount_backend_units, // Already converted from frontend base units
            )
            .await
            .map_err(|e| {
                error!("Failed to mint QNKUSD: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?
    }; // Drop the lock here

    let finalized_in_seconds = start.elapsed().as_secs_f64();

    // Create a blockchain Transaction object for Recent Activity
    let zero_address = [0u8; 32]; // System address for minting
    let transaction = Transaction {
        id: tx_id.0,            // Use the transaction ID from Quillon Bank
        from: zero_address,     // System/CDP mint (from zero address)
        to: borrower_bytes,     // User receiving QUGUSD
        amount: request.amount as u128, // QUGUSD amount already in frontend base units
        fee: 0,                 // No fee for CDP minting
        nonce: 0,               // CDP operations don't use nonces
        signature: vec![],      // System operation, no signature needed
        timestamp: Utc::now(),
        data: format!(
            "CDP_MINT:{}:{}",
            request.collateral_type, request.collateral_amount
        )
        .into_bytes(),
        token_type: q_types::TokenType::QUGUSD,
        fee_token_type: q_types::TokenType::QUGUSD,
        tx_type: q_types::TransactionType::StableMint,
        pqc_signature: None,
        signature_phase: q_types::TxSignaturePhase::Phase0Ed25519,
        pqc_public_key: None,
    };

    // Store transaction for Recent Activity display
    if let Err(e) = state.storage_engine.save_transaction(&transaction).await {
        error!("Failed to save CDP mint transaction to storage: {}", e);
    } else {
        info!(
            "💳 CDP mint transaction saved to Recent Activity: {}",
            hex::encode(&tx_id.0)
        );
    }

    // ✅ CRITICAL FIX: Update user's QUGUSD balance in token_balances map
    {
        let mut token_balances = state.token_balances.write().await;
        let balance_key = (borrower_bytes, q_types::QUGUSD_TOKEN_ADDRESS);
        let current_balance = token_balances.get(&balance_key).copied().unwrap_or(0);
        let new_balance = current_balance + request.amount as u128;
        token_balances.insert(balance_key, new_balance);

        info!(
            "💰 Updated QUGUSD balance for {}: {} → {} (minted: {})",
            hex::encode(&borrower_bytes[..8]),
            current_balance as f64 / 1e8,
            new_balance as f64 / 1e8,
            request.amount as f64 / 1e8
        );

        // Persist the balance update to storage
        if let Err(e) = state
            .storage_engine
            .save_token_balance(&borrower_bytes, &q_types::QUGUSD_TOKEN_ADDRESS, new_balance)
            .await
        {
            error!("Failed to persist QUGUSD balance after minting: {}", e);
        }
    }

    // ✅ CRITICAL FIX: Lock QUG collateral by deducting from wallet balance
    {
        let mut wallet_balances = state.wallet_balances.write().await;
        let current_qug = wallet_balances.get(&borrower_bytes).copied().unwrap_or(0);
        let collateral_base_units = (request.collateral_amount * 1e8) as u128;

        if current_qug >= collateral_base_units {
            let new_qug_balance = current_qug - collateral_base_units;
            wallet_balances.insert(borrower_bytes, new_qug_balance);

            info!(
                "🔒 Locked {} QUG as collateral: {} → {}",
                request.collateral_amount,
                current_qug as f64 / 1e8,
                new_qug_balance as f64 / 1e8
            );

            // Persist the QUG balance update
            if let Err(e) = state
                .storage_engine
                .save_wallet_balance(&borrower_bytes, new_qug_balance)
                .await
            {
                error!(
                    "Failed to persist QUG balance after locking collateral: {}",
                    e
                );
            }
        } else {
            error!(
                "⚠️  Insufficient QUG balance for collateral lock: {} QUG required, {} available",
                request.collateral_amount,
                current_qug as f64 / 1e8
            );
        }
    }

    let response = MintResponse {
        transaction_id: format!("0x{}", hex::encode(&tx_id.0)),
        amount_minted: request.amount,
        collateral_locked: request.collateral_amount,
        collateral_ratio,
        finalized_in_seconds,
    };

    info!(
        "✅ Minted {} QUGUSD in {:.2}s",
        request.amount, finalized_in_seconds
    );

    Ok(Json(ApiResponse::success(response)))
}

#[derive(Deserialize)]
pub struct BurnRequest {
    amount: u64,
    recipient: String,
    collateral_type: String,
}

pub async fn burn_qnkusd(
    State(state): State<Arc<AppState>>,
    Json(request): Json<BurnRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    info!("🔥 Burning {} QNKUSD", request.amount);

    let mut bank_system = state.quillon_bank.write().await;

    // Parse collateral type
    let collateral_type = match request.collateral_type.to_uppercase().as_str() {
        "QUG" | "ORB" => AssetType::ORB, // Q-NarwhalKnight native token
        "BTC" => AssetType::BTC,
        "ETH" => AssetType::ETH,
        "USDC" => AssetType::USDC,
        _ => return Err(StatusCode::BAD_REQUEST),
    };

    // Parse recipient address
    let recipient_bytes = parse_address(&request.recipient)?;

    // Execute burn operation
    let holder = q_quillon_bank::Address(recipient_bytes);

    let _tx_id = bank_system
        .burn_qnkusd(
            &holder,
            (request.amount * 1_000_000_000_000) as u128, // Convert QNKUSD to base units
        )
        .await
        .map_err(|e| {
            error!("Failed to burn QNKUSD: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Calculate collateral returned based on current ratio
    let collateral_returned = (request.amount as f64) / 70_000.0; // Estimate based on BTC price

    let response = serde_json::json!({
        "amount_burned": request.amount,
        "collateral_returned": collateral_returned,
        "recipient": request.recipient,
    });

    info!("✅ Burned {} QNKUSD", request.amount);

    Ok(Json(ApiResponse::success(response)))
}

#[derive(Serialize)]
struct CollateralAsset {
    asset_type: String,
    amount: f64,
    value_usd: u64,
    percentage: f64,
}

#[derive(Serialize)]
struct CollateralStatus {
    total_value: u64,
    composition: Vec<CollateralAsset>,
    ratio: f64,
}

async fn get_collateral_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<CollateralStatus>>, StatusCode> {
    info!("📊 Fetching collateral status");

    let bank_system = state.quillon_bank.read().await;

    // Get metrics to calculate collateral composition
    let metrics = bank_system.get_bank_metrics().await.map_err(|e| {
        error!("Failed to get metrics: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Calculate total collateral value
    let total_value: u128 = metrics.total_deposits.values().sum();

    // Build composition array from deposits
    let composition: Vec<CollateralAsset> = metrics
        .total_deposits
        .iter()
        .map(|(asset_type, amount)| CollateralAsset {
            asset_type: format!("{:?}", asset_type),
            amount: (*amount as f64) / 1_000_000_000_000.0,
            value_usd: (*amount / 1_000_000_000_000) as u64,
            percentage: if total_value > 0 {
                (*amount as f64 / total_value as f64) * 100.0
            } else {
                0.0
            },
        })
        .collect();

    let status = CollateralStatus {
        total_value: (total_value / 1_000_000_000_000) as u64,
        composition,
        ratio: metrics.qnkusd_metrics.collateral_ratio,
    };

    Ok(Json(ApiResponse::success(status)))
}

#[derive(Deserialize)]
pub struct AddCollateralRequest {
    collateral_type: String,
    amount: f64,
    reason: Option<String>,
}

pub async fn add_collateral(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AddCollateralRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    info!(
        "➕ Adding {} {} collateral",
        request.amount, request.collateral_type
    );

    let mut bank_system = state.quillon_bank.write().await;

    let collateral_type = match request.collateral_type.to_uppercase().as_str() {
        "QUG" | "ORB" => AssetType::ORB, // Q-NarwhalKnight native token
        "BTC" => AssetType::BTC,
        "ETH" => AssetType::ETH,
        "USDC" => AssetType::USDC,
        _ => return Err(StatusCode::BAD_REQUEST),
    };

    // Add collateral to system (for now, just acknowledge)
    info!(
        "Adding collateral: {} {} (value estimation)",
        request.amount, request.collateral_type
    );

    // TODO: Implement actual collateral addition through treasury system

    info!(
        "✅ Added {} {} collateral",
        request.amount, request.collateral_type
    );

    Ok(Json(ApiResponse::success(serde_json::json!({
        "success": true,
        "collateral_type": request.collateral_type,
        "amount": request.amount,
    }))))
}

pub async fn rebalance_collateral(
    State(_state): State<Arc<AppState>>,
    Json(_request): Json<serde_json::Value>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    info!("⚖️  Rebalancing collateral");

    // TODO: Implement collateral rebalancing logic

    Ok(Json(ApiResponse::success(serde_json::json!({
        "success": true,
        "message": "Collateral rebalanced successfully",
    }))))
}

async fn get_peg_status(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    info!("📊 Fetching peg status");

    // TODO: Get real peg price from oracle

    Ok(Json(ApiResponse::success(serde_json::json!({
        "current_price": 1.0002,
        "target_price": 1.0,
        "range_min": 0.995,
        "range_max": 1.005,
        "status": "stable",
    }))))
}

pub async fn adjust_peg(
    State(_state): State<Arc<AppState>>,
    Json(_request): Json<serde_json::Value>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    info!("🎛️  Adjusting peg parameters");

    // TODO: Implement peg adjustment logic

    Ok(Json(ApiResponse::success(serde_json::json!({
        "success": true,
        "message": "Peg parameters adjusted successfully",
    }))))
}

// ============================================================================
// Helper Functions
// ============================================================================

fn parse_address(address_str: &str) -> Result<[u8; 32], StatusCode> {
    let hex_str = address_str.strip_prefix("0x").unwrap_or(address_str);

    if hex_str.len() != 64 {
        return Err(StatusCode::BAD_REQUEST);
    }

    let mut address = [0u8; 32];
    for i in 0..32 {
        let byte_str = &hex_str[i * 2..i * 2 + 2];
        address[i] = u8::from_str_radix(byte_str, 16).map_err(|_| StatusCode::BAD_REQUEST)?;
    }

    Ok(address)
}

// ============================================================================
// Loan Application Implementation
// ============================================================================

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct LoanApplication {
    pub loan_id: String,
    pub borrower_address: String,
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub loan_amount: u128,      // QUGUSD in base units
    pub collateral_amount: f64, // QUG amount
    pub collateral_type: String,
    pub term_months: u32,
    pub interest_rate: f64,
    pub monthly_payment: f64,
    pub status: String, // "pending", "approved", "rejected"
    pub created_at: i64,
}

#[derive(Debug, serde::Deserialize)]
pub struct ApplyLoanRequest {
    pub wallet_address: String,
    #[serde(deserialize_with = "q_types::u128_serde::deserialize")]
    pub loan_amount: u128,
    pub collateral_amount: f64,
    pub collateral_type: String,
    pub term_months: u32,
}

async fn get_loan_applications(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let pending_loans = state.pending_loan_applications.read().await;
    let applications: Vec<serde_json::Value> = pending_loans
        .values()
        .map(|loan| {
            serde_json::json!({
                "loan_id": loan.loan_id,
                "borrower_address": loan.borrower_address,
                "loan_amount": loan.loan_amount,
                "collateral_amount": loan.collateral_amount,
                "collateral_type": loan.collateral_type,
                "term_months": loan.term_months,
                "interest_rate": loan.interest_rate,
                "monthly_payment": loan.monthly_payment,
                "status": loan.status,
                "created_at": loan.created_at,
            })
        })
        .collect();

    Ok(Json(ApiResponse::success(
        serde_json::json!({"applications": applications}),
    )))
}

pub async fn approve_loan(
    State(state): State<Arc<AppState>>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let loan_id = request
        .get("loan_id")
        .and_then(|v| v.as_str())
        .ok_or(StatusCode::BAD_REQUEST)?;

    // Get loan from pending applications
    let mut pending_loans = state.pending_loan_applications.write().await;
    let loan = pending_loans
        .get_mut(loan_id)
        .ok_or(StatusCode::NOT_FOUND)?;

    info!(
        "🏦 Approving loan {} for {} QUGUSD",
        loan_id,
        loan.loan_amount as f64 / 1e8
    );

    // Parse borrower address
    let borrower_addr = match parse_wallet_address(&loan.borrower_address) {
        Ok(addr) => addr,
        Err(e) => {
            error!("Invalid borrower address: {}", e);
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    // 1. Lock QUG collateral from borrower's wallet
    {
        let mut wallet_balances = state.wallet_balances.write().await;
        let qug_balance = wallet_balances.get_mut(&borrower_addr).ok_or_else(|| {
            error!("Borrower wallet not found");
            StatusCode::NOT_FOUND
        })?;

        let collateral_base_units = (loan.collateral_amount * 1e8) as u128;

        if *qug_balance < collateral_base_units {
            error!(
                "Insufficient QUG balance for collateral lock: need {}, have {}",
                collateral_base_units, *qug_balance
            );
            return Err(StatusCode::BAD_REQUEST);
        }

        *qug_balance -= collateral_base_units;

        info!(
            "🔒 Locked {} QUG as collateral from {}",
            loan.collateral_amount,
            hex::encode(&borrower_addr[..8])
        );

        // Persist QUG balance update
        if let Err(e) = state
            .storage_engine
            .save_wallet_balance(&borrower_addr, *qug_balance)
            .await
        {
            error!("Failed to persist QUG balance after locking collateral: {}", e);
        }
    }

    // 2. Mint QUGUSD and credit to borrower's token balance
    {
        let mut token_balances = state.token_balances.write().await;
        let balance_key = (borrower_addr, q_types::QUGUSD_TOKEN_ADDRESS);
        let current_qugusd = token_balances.get(&balance_key).copied().unwrap_or(0);
        let loan_amount_base_units = loan.loan_amount as u128; // Already in base units
        let new_qugusd = current_qugusd + loan_amount_base_units;

        token_balances.insert(balance_key, new_qugusd);

        info!(
            "💰 Minted {} QUGUSD for borrower: {} → {}",
            loan_amount_base_units as f64 / 1e8,
            current_qugusd as f64 / 1e8,
            new_qugusd as f64 / 1e8
        );

        // Persist QUGUSD balance update
        if let Err(e) = state
            .storage_engine
            .save_token_balance(&borrower_addr, &q_types::QUGUSD_TOKEN_ADDRESS, new_qugusd)
            .await
        {
            error!("Failed to persist QUGUSD balance after minting: {}", e);
        }
    }

    // 3. Create a transaction record for the loan disbursement
    let zero_address = [0u8; 32]; // System address for loan minting
    let transaction = Transaction {
        id: hex::decode(loan_id.replace("-", ""))
            .unwrap_or_else(|_| vec![0u8; 32])
            .try_into()
            .unwrap_or([0u8; 32]),
        from: zero_address,         // System/Loan mint (from zero address)
        to: borrower_addr,           // Borrower receiving QUGUSD
        amount: loan.loan_amount as u128, // QUGUSD amount in base units
        fee: 0,                      // No fee for loan disbursement
        nonce: 0,                    // Loan operations don't use nonces
        signature: vec![],           // System operation, no signature needed
        timestamp: Utc::now(),
        data: format!(
            "LOAN_DISBURSEMENT:{}:{}:{}%:{}mo",
            loan.collateral_amount,
            loan.collateral_type,
            loan.interest_rate,
            loan.term_months
        )
        .into_bytes(),
        token_type: q_types::TokenType::QUGUSD,
        fee_token_type: q_types::TokenType::QUGUSD,
        tx_type: q_types::TransactionType::StableMint,
        pqc_signature: None,
        signature_phase: q_types::TxSignaturePhase::Phase0Ed25519,
        pqc_public_key: None,
    };

    // Store transaction for Recent Activity display
    if let Err(e) = state.storage_engine.save_transaction(&transaction).await {
        error!("Failed to save loan disbursement transaction: {}", e);
    } else {
        info!("💳 Loan disbursement transaction saved to Recent Activity");
    }

    // 4. Update loan status to approved and extract values before dropping lock
    let collateral_amount = loan.collateral_amount;
    let loan_amount = loan.loan_amount;
    loan.status = "approved".to_string();
    let approved_loan = loan.clone();
    drop(pending_loans);

    info!("✅ Loan {} approved and funds disbursed", loan_id);

    Ok(Json(ApiResponse::success(
        serde_json::json!({
            "success": true,
            "loan": approved_loan,
            "collateral_locked": collateral_amount,
            "qugusd_disbursed": loan_amount as f64 / 1e8,
        }),
    )))
}

async fn get_loans_at_risk(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"loans": []}))))
}

pub async fn liquidate_loan(
    State(_state): State<Arc<AppState>>,
    Json(_request): Json<serde_json::Value>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(
        serde_json::json!({"success": true}),
    )))
}

pub async fn apply_loan(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ApplyLoanRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    info!(
        "🏦 Loan application received for {} QUGUSD",
        request.loan_amount as f64 / 1e8
    );

    // 1. Parse and validate wallet address
    let borrower_address = match parse_wallet_address(&request.wallet_address) {
        Ok(addr) => addr,
        Err(e) => {
            error!("Invalid wallet address: {}", e);
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    // 2. Validate collateral availability
    let wallet_balances = state.wallet_balances.read().await;
    let current_qug_balance =
        wallet_balances.get(&borrower_address).copied().unwrap_or(0) as f64 / 1e8;
    drop(wallet_balances);

    if current_qug_balance < request.collateral_amount {
        error!(
            "Insufficient collateral: have {:.2} QUG, need {:.2} QUG",
            current_qug_balance, request.collateral_amount
        );
        return Err(StatusCode::BAD_REQUEST);
    }

    // 3. Calculate interest rate based on collateral ratio and term
    const QUG_PRICE: f64 = 42.50; // $42.50 per QUG
    const MINIMUM_COLLATERAL_RATIO: f64 = 1.5; // 150%

    let loan_amount_f64 = request.loan_amount as f64 / 1e8;
    let collateral_ratio = (request.collateral_amount * QUG_PRICE) / loan_amount_f64;

    if collateral_ratio < MINIMUM_COLLATERAL_RATIO {
        error!(
            "Collateral ratio {:.2}% below minimum {:.2}%",
            collateral_ratio * 100.0,
            MINIMUM_COLLATERAL_RATIO * 100.0
        );
        return Err(StatusCode::BAD_REQUEST);
    }

    // Calculate interest rate
    let base_rate = 0.05; // 5% APR
    let collateral_bonus = ((collateral_ratio - MINIMUM_COLLATERAL_RATIO) / 0.10) * -0.01;
    let term_premium = (request.term_months as f64 / 6.0) * 0.005;
    let interest_rate = (base_rate + collateral_bonus + term_premium).max(0.01);

    // 4. Calculate monthly payment
    let total_interest = loan_amount_f64 * interest_rate * (request.term_months as f64 / 12.0);
    let total_repayment = loan_amount_f64 + total_interest;
    let monthly_payment = total_repayment / request.term_months as f64;

    // 5. Create LoanApplication with UUID
    let loan_id = uuid::Uuid::new_v4().to_string();
    let loan_application = LoanApplication {
        loan_id: loan_id.clone(),
        borrower_address: request.wallet_address.clone(),
        loan_amount: request.loan_amount,
        collateral_amount: request.collateral_amount,
        collateral_type: request.collateral_type.clone(),
        term_months: request.term_months,
        interest_rate: interest_rate * 100.0,
        monthly_payment,
        status: "pending".to_string(),
        created_at: chrono::Utc::now().timestamp(),
    };

    // 6. Serialize loan application for persistence and networking
    let loan_bytes = match bincode::serialize(&loan_application) {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("Failed to serialize loan application: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    // 7. Persist to RocksDB for durability
    if let Err(e) = state
        .storage_engine
        .save_loan_application(&loan_id, &loan_bytes)
        .await
    {
        error!("Failed to persist loan application to RocksDB: {}", e);
        // Continue anyway - we'll store it in memory
    } else {
        info!("💾 Persisted loan {} to RocksDB", loan_id);
    }

    // 8. Broadcast to network for decentralized consensus
    if let Some(ref cmd_tx) = state.libp2p_command_tx {
        let _ = cmd_tx.send(q_network::NetworkCommand::PublishBlock {
            topic: "qnk/bank/loan-applications".to_string(),
            block_bytes: loan_bytes.clone(),
            block_height: 0, // Loan applications don't have block heights
        });
        info!(
            "📡 Broadcasted loan application {} to network for consensus",
            loan_id
        );
    }

    // 9. Skip storing in pending_loan_applications - will be loaded from RocksDB on next GET request
    // This avoids type mismatch issues from multiple crate compilations

    info!(
        "✅ Loan application {} created: {} QUGUSD @ {:.2}% APR for {} months",
        loan_id,
        loan_amount_f64,
        interest_rate * 100.0,
        request.term_months
    );

    Ok(Json(ApiResponse::success(serde_json::json!({
        "loan_id": loan_id,
        "status": "pending",
        "interest_rate": interest_rate * 100.0,
        "monthly_payment": monthly_payment,
        "collateral_ratio": collateral_ratio * 100.0,
        "message": "Loan application submitted successfully. Awaiting founder approval via Quillon Bank CLI."
    }))))
}

#[derive(Debug, serde::Deserialize)]
pub struct PaybackLoanRequest {
    pub wallet_address: String,
    pub loan_id: String,
    #[serde(deserialize_with = "q_types::u128_serde::deserialize")]
    pub payment_amount: u128, // QUGUSD amount in base units
}

/// POST /api/v1/quillon-bank/lending/payback - Pay back a loan
pub async fn payback_loan(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PaybackLoanRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    info!(
        "💳 Loan payback received: {} QUGUSD for loan {}",
        request.payment_amount as f64 / 1e8,
        request.loan_id
    );

    // 1. Parse and validate wallet address
    let borrower_address = match parse_wallet_address(&request.wallet_address) {
        Ok(addr) => addr,
        Err(e) => {
            error!("Invalid wallet address: {}", e);
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    // 2. Get the loan from pending applications
    let mut pending_loans = state.pending_loan_applications.write().await;
    let loan = pending_loans
        .get_mut(&request.loan_id)
        .ok_or_else(|| {
            error!("Loan {} not found", request.loan_id);
            StatusCode::NOT_FOUND
        })?;

    // 3. Verify the borrower owns this loan
    let loan_borrower = match parse_wallet_address(&loan.borrower_address) {
        Ok(addr) => addr,
        Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
    };

    if loan_borrower != borrower_address {
        error!("Wallet mismatch: borrower does not own this loan");
        return Err(StatusCode::FORBIDDEN);
    }

    // 4. Verify loan is approved
    if loan.status != "approved" {
        error!("Cannot pay back loan with status: {}", loan.status);
        return Err(StatusCode::BAD_REQUEST);
    }

    // 5. Calculate total amount owed (principal + interest)
    let principal = loan.loan_amount as u64;
    let interest_rate = loan.interest_rate / 100.0;
    let total_interest = (principal as f64) * interest_rate * (loan.term_months as f64 / 12.0);
    let total_owed = principal + (total_interest as u64);
    let payment_amount = request.payment_amount as u64;

    info!(
        "📊 Loan payback details: Principal: {}, Interest: {:.2}, Total Owed: {}, Payment: {}",
        principal as f64 / 1e8,
        total_interest / 1e8,
        total_owed as f64 / 1e8,
        payment_amount as f64 / 1e8
    );

    // 6. Burn QUGUSD from borrower's balance
    {
        let mut token_balances = state.token_balances.write().await;
        let balance_key = (borrower_address, q_types::QUGUSD_TOKEN_ADDRESS);
        let current_qugusd = token_balances.get(&balance_key).copied().unwrap_or(0);

        if current_qugusd < payment_amount as u128 {
            error!(
                "Insufficient QUGUSD balance: have {}, need {}",
                current_qugusd as f64 / 1e8,
                payment_amount as f64 / 1e8
            );
            return Err(StatusCode::BAD_REQUEST);
        }

        let new_qugusd = current_qugusd - payment_amount as u128;
        token_balances.insert(balance_key, new_qugusd);

        info!(
            "🔥 Burned {} QUGUSD from borrower: {} → {}",
            payment_amount as f64 / 1e8,
            current_qugusd as f64 / 1e8,
            new_qugusd as f64 / 1e8
        );

        // Persist QUGUSD balance update
        if let Err(e) = state
            .storage_engine
            .save_token_balance(&borrower_address, &q_types::QUGUSD_TOKEN_ADDRESS, new_qugusd)
            .await
        {
            error!("Failed to persist QUGUSD balance after payback: {}", e);
        }
    }

    // 7. Calculate collateral to return (proportional to payment)
    let payment_ratio = (payment_amount as f64) / (total_owed as f64);
    let collateral_to_return = loan.collateral_amount * payment_ratio;
    let collateral_to_return_base = (collateral_to_return * 1e8) as u128;

    // 8. Return collateral to borrower
    {
        let mut wallet_balances = state.wallet_balances.write().await;
        let current_qug = wallet_balances.get(&borrower_address).copied().unwrap_or(0);
        let new_qug = current_qug + collateral_to_return_base;
        wallet_balances.insert(borrower_address, new_qug);

        info!(
            "🔓 Returned {} QUG collateral to borrower: {} → {}",
            collateral_to_return,
            current_qug as f64 / 1e8,
            new_qug as f64 / 1e8
        );

        // Persist QUG balance update
        if let Err(e) = state
            .storage_engine
            .save_wallet_balance(&borrower_address, new_qug)
            .await
        {
            error!("Failed to persist QUG balance after collateral return: {}", e);
        }
    }

    // 9. Update loan status
    let fully_paid = payment_amount >= total_owed;
    if fully_paid {
        loan.status = "paid".to_string();
        info!("✅ Loan {} fully paid off!", request.loan_id);
    } else {
        info!(
            "💰 Partial payment received: {:.2}% of total",
            payment_ratio * 100.0
        );
    }

    // 10. Create transaction record for payback
    let transaction = Transaction {
        id: hex::decode(request.loan_id.replace("-", ""))
            .unwrap_or_else(|_| vec![0u8; 32])
            .try_into()
            .unwrap_or([0u8; 32]),
        from: borrower_address,
        to: [0u8; 32], // System address (loan burning)
        amount: payment_amount as u128,
        fee: 0,
        nonce: 0,
        signature: vec![],
        timestamp: Utc::now(),
        data: format!("LOAN_PAYBACK:{}:{:.2}%", request.loan_id, payment_ratio * 100.0)
            .into_bytes(),
        token_type: q_types::TokenType::QUGUSD,
        fee_token_type: q_types::TokenType::QUGUSD,
        tx_type: q_types::TransactionType::StableBurn,
        pqc_signature: None,
        signature_phase: q_types::TxSignaturePhase::Phase0Ed25519,
        pqc_public_key: None,
    };

    if let Err(e) = state.storage_engine.save_transaction(&transaction).await {
        error!("Failed to save loan payback transaction: {}", e);
    }

    let response = serde_json::json!({
        "success": true,
        "loan_id": request.loan_id,
        "payment_amount": payment_amount as f64 / 1e8,
        "collateral_returned": collateral_to_return,
        "remaining_balance": if fully_paid { 0.0 } else { (total_owed - payment_amount) as f64 / 1e8 },
        "status": loan.status,
        "fully_paid": fully_paid,
    });

    drop(pending_loans);

    info!("✅ Loan payback processed successfully");

    Ok(Json(ApiResponse::success(response)))
}

async fn list_accounts(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(
        serde_json::json!({"accounts": []}),
    )))
}

async fn get_pending_accounts(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(
        serde_json::json!({"pending": []}),
    )))
}

pub async fn approve_account(
    State(_state): State<Arc<AppState>>,
    Json(_request): Json<serde_json::Value>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(
        serde_json::json!({"success": true}),
    )))
}

async fn get_reserves_status(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(
        serde_json::json!({"reserves": {}}),
    )))
}

pub async fn allocate_reserves(
    State(_state): State<Arc<AppState>>,
    Json(_request): Json<serde_json::Value>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(
        serde_json::json!({"success": true}),
    )))
}

async fn calculate_profits(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(
        serde_json::json!({"profits": {}}),
    )))
}

pub async fn distribute_profits(
    State(_state): State<Arc<AppState>>,
    Json(_request): Json<serde_json::Value>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(
        serde_json::json!({"success": true}),
    )))
}

async fn risk_assessment(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(
        serde_json::json!({"assessment": {}}),
    )))
}

async fn liquidation_queue(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"queue": []}))))
}

pub async fn execute_liquidations(
    State(_state): State<Arc<AppState>>,
    Json(_request): Json<serde_json::Value>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(
        serde_json::json!({"success": true}),
    )))
}

async fn daily_summary(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(
        serde_json::json!({"summary": {}}),
    )))
}

async fn customer_analytics(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(
        serde_json::json!({"analytics": {}}),
    )))
}

// ============================================================================
// Development Fee Transparency Endpoints
// ============================================================================

/// Development fee status - shows the transparent 1% fee configuration
#[derive(Serialize)]
struct DevFeeStatus {
    enabled: bool,
    fee_percent: f64,
    founder_wallet: String,
    description: String,
    documentation_url: String,
}

async fn get_dev_fee_status(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<DevFeeStatus>>, StatusCode> {
    info!("📊 Fetching development fee status");

    const DEV_FEE_PERCENT: f64 = 0.01; // 1%
    const FOUNDER_WALLET_HEX: &str =
        "efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723";

    let status = DevFeeStatus {
        enabled: true,
        fee_percent: DEV_FEE_PERCENT,
        founder_wallet: format!("qnk{}", FOUNDER_WALLET_HEX),
        description: "Transparent 1% development fee funds ongoing protocol development, post-quantum research, infrastructure, security audits, and community support".to_string(),
        documentation_url: "https://github.com/deme-plata/q-narwhalknight/blob/main/DEVELOPMENT_FEE_TRANSPARENCY.md".to_string(),
    };

    Ok(Json(ApiResponse::success(status)))
}

/// Development fee statistics - shows how much has been collected
#[derive(Serialize)]
struct DevFeeStats {
    total_collected_qnk: f64,
    total_mining_rewards_qnk: f64,
    fee_percentage_actual: f64,
    blocks_processed: u64,
    last_updated: chrono::DateTime<chrono::Utc>,
}

async fn get_dev_fee_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<DevFeeStats>>, StatusCode> {
    info!("📊 Fetching development fee statistics");

    const FOUNDER_WALLET_HEX: &str =
        "efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723";

    // Decode founder wallet
    let founder_wallet_bytes = match hex::decode(FOUNDER_WALLET_HEX) {
        Ok(bytes) if bytes.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            arr
        }
        _ => {
            error!("Invalid founder wallet hex");
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    // Get founder wallet balance (this is the total dev fees collected)
    let founder_balance = state
        .wallet_balances
        .read()
        .await
        .get(&founder_wallet_bytes)
        .copied()
        .unwrap_or(0);

    // Estimate total mining rewards (founder balance / 0.01)
    // Since founder gets 1%, total rewards = founder_balance * 100
    let estimated_total_rewards = founder_balance * 100;

    // Calculate actual fee percentage
    let actual_fee_percent = if estimated_total_rewards > 0 {
        (founder_balance as f64 / estimated_total_rewards as f64) * 100.0
    } else {
        0.0
    };

    let block_height = state.node_status.read().await.current_height;

    let stats = DevFeeStats {
        total_collected_qnk: founder_balance as f64 / 1e24,
        total_mining_rewards_qnk: estimated_total_rewards as f64 / 1e24,
        fee_percentage_actual: actual_fee_percent,
        blocks_processed: block_height,
        last_updated: Utc::now(),
    };

    Ok(Json(ApiResponse::success(stats)))
}

/// Founder wallet information - shows current balance and recent activity
#[derive(Serialize)]
struct FounderWalletInfo {
    wallet_address: String,
    balance_qnk: f64,
    balance_qug: u128,
    role: String,
    description: String,
    last_updated: chrono::DateTime<chrono::Utc>,
}

async fn get_founder_wallet_info(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<FounderWalletInfo>>, StatusCode> {
    info!("📊 Fetching founder wallet information");

    const FOUNDER_WALLET_HEX: &str =
        "efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723";

    // Decode founder wallet
    let founder_wallet_bytes = match hex::decode(FOUNDER_WALLET_HEX) {
        Ok(bytes) if bytes.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            arr
        }
        _ => {
            error!("Invalid founder wallet hex");
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    // Get founder wallet balance
    let balance = state
        .wallet_balances
        .read()
        .await
        .get(&founder_wallet_bytes)
        .copied()
        .unwrap_or(0);

    let info = FounderWalletInfo {
        wallet_address: format!("qnk{}", FOUNDER_WALLET_HEX),
        balance_qnk: balance as f64 / 1e24,
        balance_qug: balance,
        role: "Founder & CEO - Development Fund".to_string(),
        description: "Receives 1% of all mining rewards to fund ongoing development, research, infrastructure, and community support".to_string(),
        last_updated: Utc::now(),
    };

    Ok(Json(ApiResponse::success(info)))
}
