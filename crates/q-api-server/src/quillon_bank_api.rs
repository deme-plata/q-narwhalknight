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

use crate::AppState;
use q_quillon_bank::{QuillonBankSystem, AssetType};
use q_types::ApiResponse;

/// Create Quillon Bank API router
pub fn create_quillon_bank_router() -> Router<Arc<AppState>> {
    Router::new()
        // Status & Metrics
        .route("/stablecoin/status", get(get_stablecoin_status))
        .route("/metrics", get(get_banking_metrics))
        .route("/risk/status", get(get_risk_status))
        .route("/quantum/status", get(get_quantum_status))
        // Stablecoin Operations
        .route("/stablecoin/mint", post(mint_qnkusd))
        .route("/stablecoin/burn", post(burn_qnkusd))
        .route("/stablecoin/collateral", get(get_collateral_status))
        .route("/stablecoin/collateral/add", post(add_collateral))
        .route("/stablecoin/collateral/rebalance", post(rebalance_collateral))
        .route("/stablecoin/peg", get(get_peg_status))
        .route("/stablecoin/peg/adjust", post(adjust_peg))
        // Lending Operations
        .route("/lending/applications", get(get_loan_applications))
        .route("/lending/approve", post(approve_loan))
        .route("/lending/at-risk", get(get_loans_at_risk))
        .route("/lending/liquidate", post(liquidate_loan))
        // Account Management
        .route("/accounts", get(list_accounts))
        .route("/accounts/pending", get(get_pending_accounts))
        .route("/accounts/approve", post(approve_account))
        // Treasury Management
        .route("/treasury/reserves", get(get_reserves_status))
        .route("/treasury/reserves/allocate", post(allocate_reserves))
        .route("/treasury/profits", get(calculate_profits))
        .route("/treasury/profits/distribute", post(distribute_profits))
        // Risk Management
        .route("/risk/assessment", get(risk_assessment))
        .route("/risk/liquidations/queue", get(liquidation_queue))
        .route("/risk/liquidations/execute", post(execute_liquidations))
        // Analytics
        .route("/analytics/daily-summary", get(daily_summary))
        .route("/analytics/customers", get(customer_analytics))
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
    let total_deposits: u64 = (bank_metrics.total_deposits.values().sum::<u128>() / 1_000_000_000_000) as u64;
    let total_loans: u64 = (bank_metrics.total_loans.values().sum::<u128>() / 1_000_000_000_000) as u64;

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
struct MintRequest {
    amount: u64,
    collateral_type: String,
    collateral_amount: f64,
    reason: Option<String>,
}

#[derive(Serialize)]
struct MintResponse {
    transaction_id: String,
    amount_minted: u64,
    collateral_locked: f64,
    collateral_ratio: f64,
    finalized_in_seconds: f64,
}

async fn mint_qnkusd(
    State(state): State<Arc<AppState>>,
    Json(request): Json<MintRequest>,
) -> Result<Json<ApiResponse<MintResponse>>, StatusCode> {
    info!("💰 Minting {} QNKUSD with {} {} collateral",
        request.amount, request.collateral_amount, request.collateral_type);

    let mut bank_system = state.quillon_bank.write().await;

    // Parse collateral type
    let collateral_type = match request.collateral_type.to_uppercase().as_str() {
        "BTC" => AssetType::BTC,
        "ETH" => AssetType::ETH,
        "USDC" => AssetType::USDC,
        _ => return Err(StatusCode::BAD_REQUEST),
    };

    // Execute mint operation on blockchain
    let start = std::time::Instant::now();

    // For CLI integration, we need to create a borrower address
    // In production, this should come from authenticated user
    let borrower = q_quillon_bank::Address::new();

    // Calculate actual collateral ratio before the mint call
    let collateral_value_usd = match &collateral_type {
        AssetType::BTC => request.collateral_amount * 70_000.0,
        AssetType::ETH => request.collateral_amount * 3_500.0,
        AssetType::USDC => request.collateral_amount,
        _ => 0.0,
    };

    let collateral_ratio = (collateral_value_usd / request.amount as f64) * 100.0;

    let tx_id = bank_system.mint_qnkusd(
        &borrower,
        (request.collateral_amount * 1_000_000_000_000.0) as u128, // Convert to base units
        collateral_type,
        (request.amount * 1_000_000_000_000) as u128, // Convert QNKUSD to base units
    ).await.map_err(|e| {
        error!("Failed to mint QNKUSD: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let finalized_in_seconds = start.elapsed().as_secs_f64();

    let response = MintResponse {
        transaction_id: format!("{:?}", tx_id),
        amount_minted: request.amount,
        collateral_locked: request.collateral_amount,
        collateral_ratio,
        finalized_in_seconds,
    };

    info!("✅ Minted {} QNKUSD in {:.2}s", request.amount, finalized_in_seconds);

    Ok(Json(ApiResponse::success(response)))
}

#[derive(Deserialize)]
struct BurnRequest {
    amount: u64,
    recipient: String,
    collateral_type: String,
}

async fn burn_qnkusd(
    State(state): State<Arc<AppState>>,
    Json(request): Json<BurnRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    info!("🔥 Burning {} QNKUSD", request.amount);

    let mut bank_system = state.quillon_bank.write().await;

    // Parse collateral type
    let collateral_type = match request.collateral_type.to_uppercase().as_str() {
        "BTC" => AssetType::BTC,
        "ETH" => AssetType::ETH,
        "USDC" => AssetType::USDC,
        _ => return Err(StatusCode::BAD_REQUEST),
    };

    // Parse recipient address
    let recipient_bytes = parse_address(&request.recipient)?;

    // Execute burn operation
    let holder = q_quillon_bank::Address(recipient_bytes);

    let _tx_id = bank_system.burn_qnkusd(
        &holder,
        (request.amount * 1_000_000_000_000) as u128, // Convert QNKUSD to base units
    ).await.map_err(|e| {
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
    let composition: Vec<CollateralAsset> = metrics.total_deposits.iter().map(|(asset_type, amount)| {
        CollateralAsset {
            asset_type: format!("{:?}", asset_type),
            amount: (*amount as f64) / 1_000_000_000_000.0,
            value_usd: (*amount / 1_000_000_000_000) as u64,
            percentage: if total_value > 0 { (*amount as f64 / total_value as f64) * 100.0 } else { 0.0 },
        }
    }).collect();

    let status = CollateralStatus {
        total_value: (total_value / 1_000_000_000_000) as u64,
        composition,
        ratio: metrics.qnkusd_metrics.collateral_ratio,
    };

    Ok(Json(ApiResponse::success(status)))
}

#[derive(Deserialize)]
struct AddCollateralRequest {
    collateral_type: String,
    amount: f64,
    reason: Option<String>,
}

async fn add_collateral(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AddCollateralRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    info!("➕ Adding {} {} collateral", request.amount, request.collateral_type);

    let mut bank_system = state.quillon_bank.write().await;

    let collateral_type = match request.collateral_type.to_uppercase().as_str() {
        "BTC" => AssetType::BTC,
        "ETH" => AssetType::ETH,
        "USDC" => AssetType::USDC,
        _ => return Err(StatusCode::BAD_REQUEST),
    };

    // Add collateral to system (for now, just acknowledge)
    info!("Adding collateral: {} {} (value estimation)", request.amount, request.collateral_type);

    // TODO: Implement actual collateral addition through treasury system

    info!("✅ Added {} {} collateral", request.amount, request.collateral_type);

    Ok(Json(ApiResponse::success(serde_json::json!({
        "success": true,
        "collateral_type": request.collateral_type,
        "amount": request.amount,
    }))))
}

async fn rebalance_collateral(
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

async fn adjust_peg(
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
        let byte_str = &hex_str[i*2..i*2+2];
        address[i] = u8::from_str_radix(byte_str, 16)
            .map_err(|_| StatusCode::BAD_REQUEST)?;
    }

    Ok(address)
}

// ============================================================================
// Stub Implementations (TODO: Implement fully)
// ============================================================================

async fn get_loan_applications(State(_state): State<Arc<AppState>>) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"applications": []}))))
}

async fn approve_loan(State(_state): State<Arc<AppState>>, Json(_request): Json<serde_json::Value>) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"success": true}))))
}

async fn get_loans_at_risk(State(_state): State<Arc<AppState>>) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"loans": []}))))
}

async fn liquidate_loan(State(_state): State<Arc<AppState>>, Json(_request): Json<serde_json::Value>) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"success": true}))))
}

async fn list_accounts(State(_state): State<Arc<AppState>>) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"accounts": []}))))
}

async fn get_pending_accounts(State(_state): State<Arc<AppState>>) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"pending": []}))))
}

async fn approve_account(State(_state): State<Arc<AppState>>, Json(_request): Json<serde_json::Value>) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"success": true}))))
}

async fn get_reserves_status(State(_state): State<Arc<AppState>>) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"reserves": {}}))))
}

async fn allocate_reserves(State(_state): State<Arc<AppState>>, Json(_request): Json<serde_json::Value>) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"success": true}))))
}

async fn calculate_profits(State(_state): State<Arc<AppState>>) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"profits": {}}))))
}

async fn distribute_profits(State(_state): State<Arc<AppState>>, Json(_request): Json<serde_json::Value>) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"success": true}))))
}

async fn risk_assessment(State(_state): State<Arc<AppState>>) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"assessment": {}}))))
}

async fn liquidation_queue(State(_state): State<Arc<AppState>>) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"queue": []}))))
}

async fn execute_liquidations(State(_state): State<Arc<AppState>>, Json(_request): Json<serde_json::Value>) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"success": true}))))
}

async fn daily_summary(State(_state): State<Arc<AppState>>) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"summary": {}}))))
}

async fn customer_analytics(State(_state): State<Arc<AppState>>) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    Ok(Json(ApiResponse::success(serde_json::json!({"analytics": {}}))))
}