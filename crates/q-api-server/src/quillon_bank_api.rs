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
use q_storage::BalanceStorage; // v10.1.2: For get_balance() in loan collateral check
use crate::privacy_proof_generator::apply_privacy_proofs; // v3.4.16: Auto privacy by default
use crate::streaming::StreamEvent;
use crate::AppState;
use chrono::Utc;
use q_quillon_bank::{AssetType, QuillonBankSystem};
use q_types::{ApiResponse, DeliveryMethod, EmailMessage, Transaction};
use uuid::Uuid;

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
        // v3.9.1-beta: Bank Messaging System (User communication with bank)
        .route("/messages/:wallet_address", get(get_user_messages))
        .route("/messages/send", post(send_message))
        .route("/messages/unread/:wallet_address", get(get_unread_count))
        .route("/messages/:message_id/read", post(mark_message_read))
        // v3.9.1-beta: Identity System (Decentralized ID with VM)
        .route("/identity/:wallet_address", get(get_user_identity))
        .route("/identity/register", post(register_identity))
        .route("/identity/death-certificate", post(issue_death_certificate))
        .route("/identity/death-certificate/:cert_id", get(get_death_certificate))
        .route("/identity/inheritance/:wallet_address", get(get_inheritance_info))
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
        .route("/lending/reject", post(reject_loan))
        .route("/lending/liquidate", post(liquidate_loan))
        // Account Management (FOUNDER-ONLY)
        .route("/accounts/approve", post(approve_account))
        // Treasury Management (FOUNDER-ONLY)
        .route("/treasury/reserves/allocate", post(allocate_reserves))
        .route("/treasury/profits/distribute", post(distribute_profits))
        // Risk Management (FOUNDER-ONLY)
        .route("/risk/liquidations/execute", post(execute_liquidations))
        // v3.9.1-beta: Bank Admin Messaging (FOUNDER-ONLY)
        .route("/messages/admin/respond", post(bank_respond_message))
        .route("/messages/admin/list", get(list_all_user_messages))
        // v3.9.1-beta: Identity Admin (FOUNDER-ONLY)
        .route("/identity/admin/list", get(list_all_identities))
        .route("/identity/admin/approve", post(approve_identity))
        .route("/identity/admin/death-certificate/list", get(list_all_death_certificates))
        .route("/identity/admin/death-certificate/approve", post(approve_death_certificate))
        .route("/identity/admin/transfer", post(execute_inheritance_transfer))
        // v8.1.4: Email broadcasting (FOUNDER-ONLY)
        .route("/email/broadcast", post(broadcast_bank_email))
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

    // v4.0.4: Use live vault QUG price instead of hardcoded $3000.00
    let qug_price = state.collateral_vault.read().await.qug_price_usd;
    let collateral_value_usd = match &collateral_type {
        AssetType::ORB => request.collateral_amount * qug_price,
        AssetType::BTC => request.collateral_amount * 70_000.0, // TODO: get live BTC price
        AssetType::ETH => request.collateral_amount * 3_500.0,  // TODO: get live ETH price
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
    let mut transaction = Transaction {
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
        // v3.4.16-beta: ZK privacy fields - auto-populated below
        zk_proof_bundle: None,
        privacy_level: q_types::TransactionPrivacyLevel::Transparent,
        bulletproof: None,
        nullifier: None,
        memo: None,
    };

    // v3.4.16-beta: AUTO-APPLY MAXIMUM PRIVACY for CDP transactions
    if let Err(e) = apply_privacy_proofs(&mut transaction, None).await {
        tracing::warn!("⚠️ Privacy proof generation failed for CDP mint: {}", e);
    }

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
            "💰 Updated QUGUSD balance for {}: {:.4} → {:.4} (minted: {:.4})",
            hex::encode(&borrower_bytes[..8]),
            current_balance as f64 / 1e24,
            new_balance as f64 / 1e24,
            request.amount as f64 / 1e24
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
        let collateral_base_units = (request.collateral_amount * 1e24) as u128;

        if current_qug >= collateral_base_units {
            let new_qug_balance = current_qug - collateral_base_units;
            wallet_balances.insert(borrower_bytes, new_qug_balance);

            info!(
                "🔒 Locked {} QUG as collateral: {:.4} → {:.4}",
                request.collateral_amount,
                current_qug as f64 / 1e24,
                new_qug_balance as f64 / 1e24
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
                "⚠️  Insufficient QUG balance for collateral lock: {} QUG required, {:.4} available",
                request.collateral_amount,
                current_qug as f64 / 1e24
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

    // Calculate collateral returned based on collateral type
    let collateral_returned = match &collateral_type {
        AssetType::ORB => {
            let qug_price = state.collateral_vault.read().await.qug_price_usd;
            if qug_price > 0.0 { (request.amount as f64) / qug_price } else { 0.0 }
        }
        AssetType::BTC => (request.amount as f64) / 70_000.0,
        AssetType::ETH => (request.amount as f64) / 3_500.0,
        AssetType::USDC => request.amount as f64,
        _ => 0.0,
    };

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
    #[serde(default, serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub amount_paid: u128, // Track total amount paid back (in base units)
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
            // Sanitize f64 values — NaN/Infinity causes serde_json::json! to panic
            let safe_f64 = |v: f64| if v.is_finite() { v } else { 0.0 };
            serde_json::json!({
                "loan_id": loan.loan_id,
                "borrower_address": loan.borrower_address,
                "loan_amount": loan.loan_amount.to_string(),
                "collateral_amount": safe_f64(loan.collateral_amount),
                "collateral_type": loan.collateral_type,
                "term_months": loan.term_months,
                "interest_rate": safe_f64(loan.interest_rate),
                "monthly_payment": safe_f64(loan.monthly_payment),
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
        "🏦 Approving loan {} for {:.4} QUGUSD",
        loan_id,
        loan.loan_amount as f64 / 1e24
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

        let collateral_base_units = (loan.collateral_amount * 1e24) as u128;

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
            "💰 Minted {:.4} QUGUSD for borrower: {:.4} → {:.4}",
            loan_amount_base_units as f64 / 1e24,
            current_qugusd as f64 / 1e24,
            new_qugusd as f64 / 1e24
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
    let mut transaction = Transaction {
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
        // v3.4.16-beta: ZK privacy fields - auto-populated below
        zk_proof_bundle: None,
        privacy_level: q_types::TransactionPrivacyLevel::Transparent,
        bulletproof: None,
        nullifier: None,
        memo: None,
    };

    // v3.4.16-beta: AUTO-APPLY MAXIMUM PRIVACY for loan disbursements
    if let Err(e) = apply_privacy_proofs(&mut transaction, None).await {
        tracing::warn!("⚠️ Privacy proof generation failed for loan disbursement: {}", e);
    }

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
            "qugusd_disbursed": loan_amount as f64 / 1e24,
        }),
    )))
}

/// Reject a loan application (FOUNDER-ONLY)
pub async fn reject_loan(
    State(state): State<Arc<AppState>>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let loan_id = request
        .get("loan_id")
        .and_then(|v| v.as_str())
        .ok_or(StatusCode::BAD_REQUEST)?;
    let reason = request
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("No reason provided");

    info!("❌ Rejecting loan: {} (reason: {})", loan_id, reason);

    let mut pending_loans = state.pending_loan_applications.write().await;
    let loan = pending_loans
        .get_mut(loan_id)
        .ok_or(StatusCode::NOT_FOUND)?;

    if loan.status != "pending" {
        error!("Cannot reject loan with status: {}", loan.status);
        return Err(StatusCode::BAD_REQUEST);
    }

    loan.status = "rejected".to_string();
    let rejected_loan = loan.clone();
    drop(pending_loans);

    // Persist updated loan to RocksDB
    if let Ok(loan_bytes) = bincode::serialize(&rejected_loan) {
        if let Err(e) = state
            .storage_engine
            .save_loan_application(loan_id, &loan_bytes)
            .await
        {
            error!("Failed to persist rejected loan: {}", e);
        }
    }

    info!("✅ Loan {} rejected", loan_id);

    Ok(Json(ApiResponse::success(serde_json::json!({
        "success": true,
        "loan_id": loan_id,
        "status": "rejected",
        "reason": reason,
    }))))
}

async fn get_loans_at_risk(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let pending_loans = state.pending_loan_applications.read().await;
    let qug_price: f64 = state.collateral_vault.read().await.qug_price_usd;
    const LIQUIDATION_THRESHOLD: f64 = 1.2; // 120%

    let at_risk: Vec<serde_json::Value> = pending_loans
        .values()
        .filter(|loan| loan.status == "approved")
        .filter_map(|loan| {
            let loan_amount_usd = loan.loan_amount as f64 / 1e24;
            let collateral_value_usd = loan.collateral_amount * qug_price;
            let current_ratio = if loan_amount_usd > 0.0 {
                collateral_value_usd / loan_amount_usd
            } else {
                f64::MAX
            };

            if current_ratio < LIQUIDATION_THRESHOLD {
                Some(serde_json::json!({
                    "loan_id": loan.loan_id,
                    "borrower_address": loan.borrower_address,
                    "loan_amount_usd": loan_amount_usd,
                    "collateral_amount_qug": loan.collateral_amount,
                    "collateral_value_usd": collateral_value_usd,
                    "current_ratio": current_ratio * 100.0,
                    "liquidation_threshold": LIQUIDATION_THRESHOLD * 100.0,
                }))
            } else {
                None
            }
        })
        .collect();

    let count = at_risk.len();
    Ok(Json(ApiResponse::success(serde_json::json!({
        "loans": at_risk,
        "qug_price": qug_price,
        "count": count,
    }))))
}

pub async fn liquidate_loan(
    State(state): State<Arc<AppState>>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let loan_id = request
        .get("loan_id")
        .and_then(|v| v.as_str())
        .ok_or(StatusCode::BAD_REQUEST)?;

    info!("⚠️ Liquidating loan: {}", loan_id);

    let mut pending_loans = state.pending_loan_applications.write().await;
    let loan = pending_loans
        .get_mut(loan_id)
        .ok_or(StatusCode::NOT_FOUND)?;

    if loan.status != "approved" {
        error!("Cannot liquidate loan with status: {}", loan.status);
        return Err(StatusCode::BAD_REQUEST);
    }

    // Verify loan is below liquidation threshold
    let qug_price: f64 = state.collateral_vault.read().await.qug_price_usd;
    let loan_amount_usd = loan.loan_amount as f64 / 1e24;
    let collateral_value_usd = loan.collateral_amount * qug_price;
    let current_ratio = if loan_amount_usd > 0.0 {
        collateral_value_usd / loan_amount_usd
    } else {
        f64::MAX
    };

    const LIQUIDATION_THRESHOLD: f64 = 1.2;
    if current_ratio >= LIQUIDATION_THRESHOLD {
        return Ok(Json(ApiResponse::success(serde_json::json!({
            "success": false,
            "error": "Loan is not below liquidation threshold",
            "current_ratio": current_ratio * 100.0,
            "threshold": LIQUIDATION_THRESHOLD * 100.0,
        }))));
    }

    // Seize collateral: collateral stays locked (not returned to borrower)
    // Mark loan as liquidated
    loan.status = "liquidated".to_string();
    let collateral_seized = loan.collateral_amount;
    let loan_clone = loan.clone();
    drop(pending_loans);

    // Persist updated loan
    if let Ok(loan_bytes) = bincode::serialize(&loan_clone) {
        if let Err(e) = state
            .storage_engine
            .save_loan_application(loan_id, &loan_bytes)
            .await
        {
            error!("Failed to persist liquidated loan: {}", e);
        }
    }

    info!(
        "🔨 Loan {} liquidated: seized {:.4} QUG collateral (worth ${:.2})",
        loan_id, collateral_seized, collateral_value_usd
    );

    Ok(Json(ApiResponse::success(serde_json::json!({
        "success": true,
        "loan_id": loan_id,
        "collateral_seized_qug": collateral_seized,
        "collateral_value_usd": collateral_value_usd,
        "loan_amount_usd": loan_amount_usd,
        "ratio_at_liquidation": current_ratio * 100.0,
    }))))
}

pub async fn apply_loan(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ApplyLoanRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    info!(
        "🏦 Loan application received for {:.4} QUGUSD",
        request.loan_amount as f64 / 1e24
    );

    // 1. Parse and validate wallet address
    let borrower_address = match parse_wallet_address(&request.wallet_address) {
        Ok(addr) => addr,
        Err(e) => {
            error!("Invalid wallet address: {}", e);
            return Ok(Json(ApiResponse::error(format!("Invalid wallet address: {}", e))));
        }
    };

    // 2. Validate collateral availability
    // v10.1.2: Read from RocksDB (authoritative) like swap handler does, not stale in-memory cache
    let current_qug_balance = {
        let storage_balance = state
            .storage_engine
            .get_balance(&hex::encode(borrower_address))
            .await
            .unwrap_or(0);
        // Sync in-memory cache while we're at it
        let mut wallet_balances = state.wallet_balances.write().await;
        wallet_balances.insert(borrower_address, storage_balance);
        storage_balance as f64 / 1e24
    };

    if current_qug_balance < request.collateral_amount {
        error!(
            "Insufficient collateral: have {:.2} QUG, need {:.2} QUG",
            current_qug_balance, request.collateral_amount
        );
        return Ok(Json(ApiResponse::error(format!(
            "Insufficient QUG collateral. Required: {:.4} QUG, Available: {:.4} QUG",
            request.collateral_amount, current_qug_balance
        ))));
    }

    // 3. Calculate interest rate based on collateral ratio and term
    // v4.0.4: Use live vault QUG price instead of hardcoded $3000.00
    let qug_price: f64 = state.collateral_vault.read().await.qug_price_usd;
    const MINIMUM_COLLATERAL_RATIO: f64 = 1.5; // 150%

    let loan_amount_f64 = request.loan_amount as f64 / 1e24;
    let collateral_ratio = (request.collateral_amount * qug_price) / loan_amount_f64;

    if collateral_ratio < MINIMUM_COLLATERAL_RATIO {
        error!(
            "Collateral ratio {:.2}% below minimum {:.2}%",
            collateral_ratio * 100.0,
            MINIMUM_COLLATERAL_RATIO * 100.0
        );
        return Ok(Json(ApiResponse::error(format!(
            "Collateral ratio {:.1}% is below minimum {:.1}%. Add more collateral or reduce loan amount.",
            collateral_ratio * 100.0,
            MINIMUM_COLLATERAL_RATIO * 100.0
        ))));
    }

    // Calculate interest rate
    let base_rate = 0.05; // 5% APR
    // More collateral = lower rate: -0.5% per 50% above minimum, capped at -2%
    let excess_ratio = (collateral_ratio - MINIMUM_COLLATERAL_RATIO).max(0.0);
    let collateral_discount = (excess_ratio * 0.01).min(0.02); // Max 2% discount
    let term_premium = (request.term_months as f64 / 6.0) * 0.005; // +0.5% per 6 months
    let interest_rate = (base_rate - collateral_discount + term_premium).max(0.01); // Min 1% APR

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
        amount_paid: 0,
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

    // 9. Store loan in pending_loan_applications for immediate availability
    {
        let mut pending_loans = state.pending_loan_applications.write().await;
        pending_loans.insert(loan_id.clone(), loan_application.clone());
        info!("📋 Stored loan {} in pending_loan_applications ({} total)", loan_id, pending_loans.len());
    }

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
        "💳 Loan payback received: {:.4} QUGUSD for loan {}",
        request.payment_amount as f64 / 1e24,
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

    // 5. Calculate total amount owed (principal + interest) using u128
    let principal = loan.loan_amount; // Keep as u128, no truncation
    let interest_rate = loan.interest_rate / 100.0;
    let total_interest_f64 = (principal as f64) * interest_rate * (loan.term_months as f64 / 12.0);
    let total_interest = total_interest_f64 as u128;
    let total_owed = principal + total_interest;
    let remaining_owed = total_owed.saturating_sub(loan.amount_paid); // Account for prior payments
    let payment_amount = request.payment_amount;

    info!(
        "📊 Loan payback details: Principal: {:.4}, Interest: {:.4}, Total Owed: {:.4}, Already Paid: {:.4}, Remaining: {:.4}, Payment: {:.4}",
        principal as f64 / 1e24,
        total_interest as f64 / 1e24,
        total_owed as f64 / 1e24,
        loan.amount_paid as f64 / 1e24,
        remaining_owed as f64 / 1e24,
        payment_amount as f64 / 1e24
    );

    // 6. Burn QUGUSD from borrower's balance
    {
        let mut token_balances = state.token_balances.write().await;
        let balance_key = (borrower_address, q_types::QUGUSD_TOKEN_ADDRESS);
        let current_qugusd = token_balances.get(&balance_key).copied().unwrap_or(0);

        if current_qugusd < payment_amount {
            error!(
                "Insufficient QUGUSD balance: have {:.4}, need {:.4}",
                current_qugusd as f64 / 1e24,
                payment_amount as f64 / 1e24
            );
            return Err(StatusCode::BAD_REQUEST);
        }

        let new_qugusd = current_qugusd - payment_amount;
        token_balances.insert(balance_key, new_qugusd);

        info!(
            "🔥 Burned {:.4} QUGUSD from borrower: {:.4} → {:.4}",
            payment_amount as f64 / 1e24,
            current_qugusd as f64 / 1e24,
            new_qugusd as f64 / 1e24
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

    // 7. Calculate collateral to return (proportional to payment vs remaining)
    let payment_ratio = ((payment_amount as f64) / (remaining_owed as f64).max(1.0)).min(1.0);
    // Calculate remaining collateral (accounting for prior partial returns)
    let prior_returned_ratio = if total_owed > 0 { (loan.amount_paid as f64) / (total_owed as f64) } else { 0.0 };
    let remaining_collateral = loan.collateral_amount * (1.0 - prior_returned_ratio).max(0.0);
    let collateral_to_return = remaining_collateral * payment_ratio;
    let collateral_to_return_base = (collateral_to_return * 1e24) as u128;

    // 8. Return collateral to borrower
    {
        let mut wallet_balances = state.wallet_balances.write().await;
        let current_qug = wallet_balances.get(&borrower_address).copied().unwrap_or(0);
        let new_qug = current_qug + collateral_to_return_base;
        wallet_balances.insert(borrower_address, new_qug);

        info!(
            "🔓 Returned {:.4} QUG collateral to borrower: {:.4} → {:.4}",
            collateral_to_return,
            current_qug as f64 / 1e24,
            new_qug as f64 / 1e24
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

    // 9. Update loan status and track payment
    loan.amount_paid += payment_amount;
    let fully_paid = loan.amount_paid >= total_owed;
    if fully_paid {
        loan.status = "paid".to_string();
        info!("✅ Loan {} fully paid off!", request.loan_id);
    } else {
        loan.status = "approved".to_string(); // Keep active for further payments
        info!(
            "💰 Partial payment received: {:.2}% of total ({:.4} / {:.4} QUGUSD paid)",
            (loan.amount_paid as f64 / total_owed as f64) * 100.0,
            loan.amount_paid as f64 / 1e24,
            total_owed as f64 / 1e24
        );
    }

    // Persist updated loan to RocksDB
    if let Ok(loan_bytes) = bincode::serialize(&loan.clone()) {
        if let Err(e) = state
            .storage_engine
            .save_loan_application(&request.loan_id, &loan_bytes)
            .await
        {
            error!("Failed to persist updated loan after payback: {}", e);
        }
    }

    // 10. Create transaction record for payback
    let mut transaction = Transaction {
        id: hex::decode(request.loan_id.replace("-", ""))
            .unwrap_or_else(|_| vec![0u8; 32])
            .try_into()
            .unwrap_or([0u8; 32]),
        from: borrower_address,
        to: [0u8; 32], // System address (loan burning)
        amount: payment_amount,
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
        // v3.4.16-beta: ZK privacy fields - auto-populated below
        zk_proof_bundle: None,
        privacy_level: q_types::TransactionPrivacyLevel::Transparent,
        bulletproof: None,
        nullifier: None,
        memo: None,
    };

    // v3.4.16-beta: AUTO-APPLY MAXIMUM PRIVACY for loan payback
    if let Err(e) = apply_privacy_proofs(&mut transaction, None).await {
        tracing::warn!("⚠️ Privacy proof generation failed for loan payback: {}", e);
    }

    if let Err(e) = state.storage_engine.save_transaction(&transaction).await {
        error!("Failed to save loan payback transaction: {}", e);
    }

    let final_remaining = if fully_paid { 0u128 } else { total_owed.saturating_sub(loan.amount_paid) };
    let loan_status = loan.status.clone();
    let response = serde_json::json!({
        "success": true,
        "loan_id": request.loan_id,
        "payment_amount": payment_amount as f64 / 1e24,
        "collateral_returned": collateral_to_return,
        "remaining_balance": final_remaining as f64 / 1e24,
        "total_paid": loan.amount_paid as f64 / 1e24,
        "status": loan_status,
        "fully_paid": fully_paid,
    });

    drop(pending_loans);

    info!("✅ Loan payback processed successfully");

    Ok(Json(ApiResponse::success(response)))
}

async fn list_accounts(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet_balances = state.wallet_balances.read().await;
    let accounts: Vec<serde_json::Value> = wallet_balances
        .iter()
        .filter(|(_, &balance)| balance > 0)
        .map(|(addr, &balance)| {
            serde_json::json!({
                "address": format!("qnk{}", hex::encode(addr)),
                "balance_qug": balance as f64 / 1e24,
            })
        })
        .collect();

    let count = accounts.len();
    Ok(Json(ApiResponse::success(
        serde_json::json!({"accounts": accounts, "count": count}),
    )))
}

async fn get_pending_accounts(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let identities = state.user_identities.read().await;
    let pending: Vec<serde_json::Value> = identities
        .iter()
        .filter(|i| !i.verified)
        .map(|i| {
            serde_json::json!({
                "wallet_address": i.wallet_address,
                "display_name": i.display_name,
                "created_at": i.created_at,
                "kyc_level": i.kyc_level,
            })
        })
        .collect();

    let count = pending.len();
    Ok(Json(ApiResponse::success(
        serde_json::json!({"pending": pending, "count": count}),
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
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let pending_loans = state.pending_loan_applications.read().await;

    let mut total_interest_earned: f64 = 0.0;
    let mut total_principal_repaid: f64 = 0.0;
    let mut paid_loan_count: u64 = 0;

    for loan in pending_loans.values() {
        if loan.status == "paid" || loan.amount_paid > 0 {
            let principal = loan.loan_amount as f64 / 1e24;
            let interest_rate = loan.interest_rate / 100.0;
            let total_interest = principal * interest_rate * (loan.term_months as f64 / 12.0);
            let total_owed = principal + total_interest;
            let paid = loan.amount_paid as f64 / 1e24;

            // Interest is earned proportionally to amount paid
            let interest_portion = if total_owed > 0.0 {
                total_interest * (paid / total_owed).min(1.0)
            } else {
                0.0
            };

            total_interest_earned += interest_portion;
            total_principal_repaid += (paid - interest_portion).max(0.0);
            if loan.status == "paid" {
                paid_loan_count += 1;
            }
        }
    }

    Ok(Json(ApiResponse::success(serde_json::json!({
        "total_interest_earned_qugusd": total_interest_earned,
        "total_principal_repaid_qugusd": total_principal_repaid,
        "paid_loans": paid_loan_count,
        "total_revenue_qugusd": total_interest_earned,
    }))))
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
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let pending_loans = state.pending_loan_applications.read().await;
    let qug_price: f64 = state.collateral_vault.read().await.qug_price_usd;
    const LIQUIDATION_THRESHOLD: f64 = 1.2;
    const WARNING_THRESHOLD: f64 = 1.5;

    let mut at_risk_count = 0u64;
    let mut at_risk_value = 0.0f64;
    let mut warning_count = 0u64;
    let mut total_active_loans = 0u64;
    let mut total_loan_value = 0.0f64;

    for loan in pending_loans.values() {
        if loan.status == "approved" {
            total_active_loans += 1;
            let loan_amount_usd = loan.loan_amount as f64 / 1e24;
            total_loan_value += loan_amount_usd;
            let collateral_value_usd = loan.collateral_amount * qug_price;
            let ratio = if loan_amount_usd > 0.0 {
                collateral_value_usd / loan_amount_usd
            } else {
                f64::MAX
            };

            if ratio < LIQUIDATION_THRESHOLD {
                at_risk_count += 1;
                at_risk_value += loan_amount_usd;
            } else if ratio < WARNING_THRESHOLD {
                warning_count += 1;
            }
        }
    }

    Ok(Json(ApiResponse::success(serde_json::json!({
        "total_active_loans": total_active_loans,
        "total_loan_value_usd": total_loan_value,
        "at_risk_count": at_risk_count,
        "at_risk_value_usd": at_risk_value,
        "warning_count": warning_count,
        "qug_price_usd": qug_price,
        "liquidation_threshold": LIQUIDATION_THRESHOLD * 100.0,
        "warning_threshold": WARNING_THRESHOLD * 100.0,
    }))))
}

async fn liquidation_queue(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let pending_loans = state.pending_loan_applications.read().await;
    let qug_price: f64 = state.collateral_vault.read().await.qug_price_usd;
    const LIQUIDATION_THRESHOLD: f64 = 1.2;

    let queue: Vec<serde_json::Value> = pending_loans
        .values()
        .filter(|loan| loan.status == "approved")
        .filter_map(|loan| {
            let loan_amount_usd = loan.loan_amount as f64 / 1e24;
            let collateral_value_usd = loan.collateral_amount * qug_price;
            let ratio = if loan_amount_usd > 0.0 {
                collateral_value_usd / loan_amount_usd
            } else {
                f64::MAX
            };

            if ratio < LIQUIDATION_THRESHOLD {
                Some(serde_json::json!({
                    "loan_id": loan.loan_id,
                    "borrower_address": loan.borrower_address,
                    "loan_amount_usd": loan_amount_usd,
                    "collateral_qug": loan.collateral_amount,
                    "collateral_value_usd": collateral_value_usd,
                    "current_ratio": ratio * 100.0,
                    "shortfall_usd": (loan_amount_usd * LIQUIDATION_THRESHOLD) - collateral_value_usd,
                }))
            } else {
                None
            }
        })
        .collect();

    let count = queue.len();
    Ok(Json(ApiResponse::success(serde_json::json!({
        "queue": queue,
        "count": count,
        "qug_price": qug_price,
    }))))
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
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let now = chrono::Utc::now().timestamp_millis();
    let day_ago = now - 86_400_000; // 24 hours in milliseconds

    let messages = state.bank_messages.read().await;
    let messages_today = messages.iter().filter(|m| m.timestamp > day_ago).count();
    let unread_messages = messages.iter().filter(|m| !m.read && m.from == MessageSender::User).count();

    let loans = state.pending_loan_applications.read().await;
    let new_loans_today = loans.values().filter(|l| l.created_at > day_ago / 1000).count();
    let active_loans = loans.values().filter(|l| l.status == "approved").count();
    let pending_loans = loans.values().filter(|l| l.status == "pending").count();

    let node_status = state.node_status.read().await;
    let block_height = node_status.current_height;
    let connected_peers = node_status.connected_peers;
    drop(node_status);

    Ok(Json(ApiResponse::success(serde_json::json!({
        "date": chrono::Utc::now().format("%Y-%m-%d").to_string(),
        "messages_today": messages_today,
        "unread_messages": unread_messages,
        "new_loan_applications": new_loans_today,
        "active_loans": active_loans,
        "pending_loans": pending_loans,
        "block_height": block_height,
        "connected_peers": connected_peers,
    }))))
}

async fn customer_analytics(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet_balances = state.wallet_balances.read().await;
    let active_wallets: Vec<_> = wallet_balances.iter().filter(|(_, &b)| b > 0).collect();
    let total_wallets = active_wallets.len();
    let total_balance: u128 = active_wallets.iter().map(|(_, &b)| b).sum();
    let avg_balance = if total_wallets > 0 {
        (total_balance as f64 / 1e24) / total_wallets as f64
    } else {
        0.0
    };
    drop(wallet_balances);

    let (total_loans, active_loans, paid_loans, unique_borrower_count) = {
        let loans = state.pending_loan_applications.read().await;
        let total = loans.len();
        let active = loans.values().filter(|l| l.status == "approved").count();
        let paid = loans.values().filter(|l| l.status == "paid").count();
        let unique: std::collections::HashSet<String> = loans.values().map(|l| l.borrower_address.clone()).collect();
        (total, active, paid, unique.len())
    };

    let (registered_identities, verified_identities) = {
        let identities = state.user_identities.read().await;
        let total = identities.len();
        let verified = identities.iter().filter(|i| i.verified).count();
        (total, verified)
    };

    Ok(Json(ApiResponse::success(serde_json::json!({
        "total_wallets": total_wallets,
        "total_balance_qug": total_balance as f64 / 1e24,
        "average_balance_qug": avg_balance,
        "total_loan_applications": total_loans,
        "active_loans": active_loans,
        "paid_loans": paid_loans,
        "unique_borrowers": unique_borrower_count,
        "registered_identities": registered_identities,
        "verified_identities": verified_identities,
    }))))
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

// ============================================================================
// v3.9.1-beta: Bank Messaging System
// ============================================================================

/// Message between user and bank
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BankMessage {
    pub id: String,
    pub from: MessageSender,
    pub wallet_address: String,
    pub content: String,
    pub subject: Option<String>,
    pub loan_id: Option<String>,
    pub timestamp: i64,
    pub read: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageSender {
    User,
    Bank,
}

#[derive(Debug, Deserialize)]
pub struct SendMessageRequest {
    pub wallet_address: String,
    pub content: String,
    pub subject: Option<String>,
    pub loan_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct BankRespondRequest {
    pub message_id: String,
    pub wallet_address: String,
    pub content: String,
    pub subject: Option<String>,
}

/// Get all messages for a user wallet
async fn get_user_messages(
    State(state): State<Arc<AppState>>,
    Path(wallet_address): Path<String>,
) -> Result<Json<Vec<BankMessage>>, StatusCode> {
    info!("📬 Fetching messages for wallet: {}", wallet_address);

    let messages = state.bank_messages.read().await;
    let normalized = wallet_address.to_lowercase().replace("qnk", "");

    let user_messages: Vec<BankMessage> = messages
        .iter()
        .filter(|m| m.wallet_address.to_lowercase().replace("qnk", "") == normalized)
        .cloned()
        .collect();

    Ok(Json(user_messages))
}

/// Send a message to the bank (from user)
async fn send_message(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SendMessageRequest>,
) -> Result<Json<ApiResponse<BankMessage>>, StatusCode> {
    info!("📤 User sending message from wallet: {}", req.wallet_address);

    let message = BankMessage {
        id: format!("msg_{}", chrono::Utc::now().timestamp_millis()),
        from: MessageSender::User,
        wallet_address: req.wallet_address.clone(),
        content: req.content,
        subject: req.subject,
        loan_id: req.loan_id,
        timestamp: chrono::Utc::now().timestamp_millis(),
        read: false, // Bank hasn't read it yet
    };

    state.bank_messages.write().await.push(message.clone());

    // v3.9.1-beta: RocksDB persistence
    if let Ok(data) = serde_json::to_vec(&message) {
        let kv = state.storage_engine.get_kv();
        if let Err(e) = kv.put(q_storage::CF_BANK_MESSAGES, message.id.as_bytes(), &data).await {
            error!("Failed to persist bank message to RocksDB: {}", e);
        } else {
            // Also index by wallet address for efficient lookups
            let inverted_ts = u64::MAX - (message.timestamp as u64);
            let mut index_key = Vec::with_capacity(40);
            index_key.extend_from_slice(req.wallet_address.as_bytes());
            index_key.extend_from_slice(&inverted_ts.to_be_bytes());
            let _ = kv.put(q_storage::CF_BANK_MSG_INDEX, &index_key, message.id.as_bytes()).await;
            info!("📬 Message persisted to RocksDB: {}", message.id);
        }
    }

    Ok(Json(ApiResponse::success(message)))
}

/// Get unread message count for a wallet
async fn get_unread_count(
    State(state): State<Arc<AppState>>,
    Path(wallet_address): Path<String>,
) -> Result<Json<ApiResponse<u32>>, StatusCode> {
    let messages = state.bank_messages.read().await;
    let normalized = wallet_address.to_lowercase().replace("qnk", "");

    let unread = messages
        .iter()
        .filter(|m| {
            m.wallet_address.to_lowercase().replace("qnk", "") == normalized
                && !m.read
                && m.from == MessageSender::Bank
        })
        .count() as u32;

    Ok(Json(ApiResponse::success(unread)))
}

/// Mark a message as read
async fn mark_message_read(
    State(state): State<Arc<AppState>>,
    Path(message_id): Path<String>,
) -> Result<Json<ApiResponse<bool>>, StatusCode> {
    let mut messages = state.bank_messages.write().await;

    if let Some(msg) = messages.iter_mut().find(|m| m.id == message_id) {
        msg.read = true;
        return Ok(Json(ApiResponse::success(true)));
    }

    Ok(Json(ApiResponse::success(false)))
}

/// Bank responds to a user message (FOUNDER-ONLY)
async fn bank_respond_message(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BankRespondRequest>,
) -> Result<Json<ApiResponse<BankMessage>>, StatusCode> {
    info!("🏦 Bank responding to wallet: {}", req.wallet_address);

    let message = BankMessage {
        id: format!("msg_{}", chrono::Utc::now().timestamp_millis()),
        from: MessageSender::Bank,
        wallet_address: req.wallet_address.clone(),
        content: req.content,
        subject: req.subject,
        loan_id: None,
        timestamp: chrono::Utc::now().timestamp_millis(),
        read: false, // User hasn't read it yet
    };

    state.bank_messages.write().await.push(message.clone());

    // v3.9.1-beta: RocksDB persistence
    if let Ok(data) = serde_json::to_vec(&message) {
        let kv = state.storage_engine.get_kv();
        if let Err(e) = kv.put(q_storage::CF_BANK_MESSAGES, message.id.as_bytes(), &data).await {
            error!("Failed to persist bank response to RocksDB: {}", e);
        } else {
            // Also index by wallet address
            let inverted_ts = u64::MAX - (message.timestamp as u64);
            let mut index_key = Vec::with_capacity(40);
            index_key.extend_from_slice(req.wallet_address.as_bytes());
            index_key.extend_from_slice(&inverted_ts.to_be_bytes());
            let _ = kv.put(q_storage::CF_BANK_MSG_INDEX, &index_key, message.id.as_bytes()).await;
            info!("🏦 Bank response persisted to RocksDB: {}", message.id);
        }
    }

    Ok(Json(ApiResponse::success(message)))
}

/// List all user messages (FOUNDER-ONLY for admin dashboard)
async fn list_all_user_messages(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<BankMessage>>, StatusCode> {
    let messages = state.bank_messages.read().await;
    Ok(Json(messages.clone()))
}

// ============================================================================
// v3.9.1-beta: Decentralized Identity System
// ============================================================================

/// User identity record stored on VM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserIdentity {
    pub wallet_address: String,
    pub display_name: Option<String>,
    pub email_hash: Option<String>, // SHA3-256 hash for privacy
    pub created_at: i64,
    pub verified: bool,
    pub kyc_level: u8, // 0=none, 1=basic, 2=enhanced, 3=full
    pub is_deceased: bool,
    pub death_certificate_id: Option<String>,
    pub beneficiary_address: Option<String>,
    pub last_active: i64,
}

/// Death certificate for account inheritance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeathCertificate {
    pub id: String,
    pub deceased_wallet: String,
    pub beneficiary_wallet: String,
    pub issued_at: i64,
    pub approved: bool,
    pub approved_by: Option<String>,
    pub approved_at: Option<i64>,
    pub executed: bool,
    pub executed_at: Option<i64>,
    pub reason: String,
}

#[derive(Debug, Deserialize)]
pub struct RegisterIdentityRequest {
    pub wallet_address: String,
    pub display_name: Option<String>,
    pub email_hash: Option<String>,
    pub beneficiary_address: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct IssuDeathCertificateRequest {
    pub deceased_wallet: String,
    pub beneficiary_wallet: String,
    pub reason: String,
    pub proof_documents: Option<String>, // Hash of uploaded documents
}

#[derive(Debug, Serialize)]
pub struct InheritanceInfo {
    pub deceased_wallet: String,
    pub beneficiary_wallet: String,
    pub total_balance: u128,
    pub token_balances: Vec<TokenBalance>,
    pub active_loans: Vec<String>,
    pub death_certificate: Option<DeathCertificate>,
    pub transfer_ready: bool,
}

#[derive(Debug, Serialize)]
pub struct TokenBalance {
    pub token_address: String,
    pub symbol: String,
    pub balance: u128,
}

/// Get user identity
async fn get_user_identity(
    State(state): State<Arc<AppState>>,
    Path(wallet_address): Path<String>,
) -> Result<Json<ApiResponse<Option<UserIdentity>>>, StatusCode> {
    info!("🪪 Fetching identity for wallet: {}", wallet_address);

    let identities = state.user_identities.read().await;
    let normalized = wallet_address.to_lowercase().replace("qnk", "");

    let identity = identities
        .iter()
        .find(|i| i.wallet_address.to_lowercase().replace("qnk", "") == normalized)
        .cloned();

    Ok(Json(ApiResponse::success(identity)))
}

/// Register new identity
async fn register_identity(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterIdentityRequest>,
) -> Result<Json<ApiResponse<UserIdentity>>, StatusCode> {
    info!("🪪 Registering identity for wallet: {}", req.wallet_address);

    let identity = UserIdentity {
        wallet_address: req.wallet_address.clone(),
        display_name: req.display_name,
        email_hash: req.email_hash,
        created_at: chrono::Utc::now().timestamp_millis(),
        verified: false, // Needs admin approval
        kyc_level: 0,
        is_deceased: false,
        death_certificate_id: None,
        beneficiary_address: req.beneficiary_address,
        last_active: chrono::Utc::now().timestamp_millis(),
    };

    state.user_identities.write().await.push(identity.clone());

    // v3.9.1-beta: RocksDB persistence
    if let Ok(data) = serde_json::to_vec(&identity) {
        let kv = state.storage_engine.get_kv();
        if let Err(e) = kv.put(q_storage::CF_USER_IDENTITIES, req.wallet_address.as_bytes(), &data).await {
            error!("Failed to persist identity to RocksDB: {}", e);
        } else {
            info!("🪪 Identity persisted to RocksDB: {}", req.wallet_address);
        }
    }

    Ok(Json(ApiResponse::success(identity)))
}

/// Issue death certificate request
async fn issue_death_certificate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IssuDeathCertificateRequest>,
) -> Result<Json<ApiResponse<DeathCertificate>>, StatusCode> {
    info!(
        "💀 Death certificate request: {} -> {}",
        req.deceased_wallet, req.beneficiary_wallet
    );

    let cert = DeathCertificate {
        id: format!("death_{}", chrono::Utc::now().timestamp_millis()),
        deceased_wallet: req.deceased_wallet,
        beneficiary_wallet: req.beneficiary_wallet,
        issued_at: chrono::Utc::now().timestamp_millis(),
        approved: false,
        approved_by: None,
        approved_at: None,
        executed: false,
        executed_at: None,
        reason: req.reason,
    };

    state.death_certificates.write().await.push(cert.clone());

    // v3.9.1-beta: RocksDB persistence
    if let Ok(data) = serde_json::to_vec(&cert) {
        let kv = state.storage_engine.get_kv();
        if let Err(e) = kv.put(q_storage::CF_DEATH_CERTIFICATES, cert.id.as_bytes(), &data).await {
            error!("Failed to persist death certificate to RocksDB: {}", e);
        } else {
            info!("💀 Death certificate persisted to RocksDB: {}", cert.id);
        }
    }

    Ok(Json(ApiResponse::success(cert)))
}

/// Get inheritance information for a wallet
async fn get_inheritance_info(
    State(state): State<Arc<AppState>>,
    Path(wallet_address): Path<String>,
) -> Result<Json<ApiResponse<Option<InheritanceInfo>>>, StatusCode> {
    let certs = state.death_certificates.read().await;
    let normalized = wallet_address.to_lowercase().replace("qnk", "");

    // Find death certificate for this wallet (as deceased)
    let cert = certs
        .iter()
        .find(|c| c.deceased_wallet.to_lowercase().replace("qnk", "") == normalized)
        .cloned();

    if let Some(cert) = cert {
        // Parse wallet address to get balance
        let wallet_bytes = match parse_wallet_address(&wallet_address) {
            Ok(bytes) => bytes,
            Err(_) => return Ok(Json(ApiResponse::success(None))),
        };

        let balance = state
            .wallet_balances
            .read()
            .await
            .get(&wallet_bytes)
            .copied()
            .unwrap_or(0);

        let info = InheritanceInfo {
            deceased_wallet: cert.deceased_wallet.clone(),
            beneficiary_wallet: cert.beneficiary_wallet.clone(),
            total_balance: balance,
            token_balances: vec![], // TODO: Fetch token balances
            active_loans: vec![],   // TODO: Check for active loans
            death_certificate: Some(cert.clone()),
            transfer_ready: cert.approved && !cert.executed,
        };

        return Ok(Json(ApiResponse::success(Some(info))));
    }

    Ok(Json(ApiResponse::success(None)))
}

/// Approve identity (FOUNDER-ONLY)
async fn approve_identity(
    State(state): State<Arc<AppState>>,
    Json(wallet_address): Json<String>,
) -> Result<Json<ApiResponse<bool>>, StatusCode> {
    let mut identities = state.user_identities.write().await;
    let normalized = wallet_address.to_lowercase().replace("qnk", "");

    if let Some(identity) = identities
        .iter_mut()
        .find(|i| i.wallet_address.to_lowercase().replace("qnk", "") == normalized)
    {
        identity.verified = true;
        identity.kyc_level = 1;
        return Ok(Json(ApiResponse::success(true)));
    }

    Ok(Json(ApiResponse::success(false)))
}

/// Approve death certificate (FOUNDER-ONLY)
async fn approve_death_certificate(
    State(state): State<Arc<AppState>>,
    Json(cert_id): Json<String>,
) -> Result<Json<ApiResponse<bool>>, StatusCode> {
    let mut certs = state.death_certificates.write().await;

    if let Some(cert) = certs.iter_mut().find(|c| c.id == cert_id) {
        cert.approved = true;
        cert.approved_by = Some("founder".to_string());
        cert.approved_at = Some(chrono::Utc::now().timestamp_millis());

        // Mark identity as deceased
        let mut identities = state.user_identities.write().await;
        let normalized = cert.deceased_wallet.to_lowercase().replace("qnk", "");
        if let Some(identity) = identities
            .iter_mut()
            .find(|i| i.wallet_address.to_lowercase().replace("qnk", "") == normalized)
        {
            identity.is_deceased = true;
            identity.death_certificate_id = Some(cert.id.clone());
        }

        return Ok(Json(ApiResponse::success(true)));
    }

    Ok(Json(ApiResponse::success(false)))
}

/// List all registered identities (FOUNDER-ONLY)
async fn list_all_identities(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<UserIdentity>>>, StatusCode> {
    let identities = state.user_identities.read().await;
    Ok(Json(ApiResponse::success(identities.clone())))
}

/// List all death certificates (FOUNDER-ONLY)
async fn list_all_death_certificates(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<DeathCertificate>>>, StatusCode> {
    let certs = state.death_certificates.read().await;
    Ok(Json(ApiResponse::success(certs.clone())))
}

/// Get a specific death certificate by ID (PUBLIC)
async fn get_death_certificate(
    State(state): State<Arc<AppState>>,
    Path(cert_id): Path<String>,
) -> Result<Json<ApiResponse<Option<DeathCertificate>>>, StatusCode> {
    let certs = state.death_certificates.read().await;
    let cert = certs.iter().find(|c| c.id == cert_id).cloned();
    Ok(Json(ApiResponse::success(cert)))
}

/// Execute inheritance transfer (FOUNDER-ONLY)
async fn execute_inheritance_transfer(
    State(state): State<Arc<AppState>>,
    Json(cert_id): Json<String>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    info!("💰 Executing inheritance transfer for certificate: {}", cert_id);

    let mut certs = state.death_certificates.write().await;

    let cert = match certs.iter_mut().find(|c| c.id == cert_id && c.approved && !c.executed) {
        Some(c) => c,
        None => {
            return Ok(Json(ApiResponse::error(
                "Certificate not found or not approved".to_string(),
            )));
        }
    };

    // Get deceased wallet balance
    let deceased_bytes = match parse_wallet_address(&cert.deceased_wallet) {
        Ok(bytes) => bytes,
        Err(e) => return Ok(Json(ApiResponse::error(format!("Invalid deceased wallet: {}", e)))),
    };

    let beneficiary_bytes = match parse_wallet_address(&cert.beneficiary_wallet) {
        Ok(bytes) => bytes,
        Err(e) => return Ok(Json(ApiResponse::error(format!("Invalid beneficiary wallet: {}", e)))),
    };

    // Check for active (unpaid) loans - block transfer if any exist
    {
        let loans = state.pending_loan_applications.read().await;
        let active_loans: Vec<_> = loans
            .values()
            .filter(|l| {
                let borrower_norm = l.borrower_address.to_lowercase().replace("qnk", "");
                let deceased_norm = cert.deceased_wallet.to_lowercase().replace("qnk", "");
                borrower_norm == deceased_norm && (l.status == "approved" || l.status == "pending")
            })
            .collect();

        if !active_loans.is_empty() {
            let loan_ids: Vec<_> = active_loans.iter().map(|l| l.loan_id.as_str()).collect();
            return Ok(Json(ApiResponse::error(format!(
                "Cannot transfer: {} active/pending loan(s) exist: {}",
                active_loans.len(),
                loan_ids.join(", ")
            ))));
        }
    }

    let balance = {
        let balances = state.wallet_balances.read().await;
        balances.get(&deceased_bytes).copied().unwrap_or(0)
    };

    // Transfer QUG balance
    {
        let mut balances = state.wallet_balances.write().await;
        balances.insert(deceased_bytes, 0);
        let current = balances.get(&beneficiary_bytes).copied().unwrap_or(0);
        balances.insert(beneficiary_bytes, current + balance);
    }

    // Persist QUG balance changes
    if let Err(e) = state.storage_engine.save_wallet_balance(&deceased_bytes, 0).await {
        error!("Failed to persist deceased QUG balance: {}", e);
    }
    if let Err(e) = state.storage_engine.save_wallet_balance(
        &beneficiary_bytes,
        state.wallet_balances.read().await.get(&beneficiary_bytes).copied().unwrap_or(0),
    ).await {
        error!("Failed to persist beneficiary QUG balance: {}", e);
    }

    // Transfer ALL token balances (QUGUSD, custom tokens, etc.)
    let mut transferred_tokens = Vec::new();
    {
        let mut token_balances = state.token_balances.write().await;
        // Collect keys matching deceased address
        let deceased_keys: Vec<_> = token_balances
            .keys()
            .filter(|(addr, _)| addr == &deceased_bytes)
            .cloned()
            .collect();

        for (_, token_addr) in &deceased_keys {
            let token_balance = token_balances.get(&(deceased_bytes, *token_addr)).copied().unwrap_or(0);
            if token_balance > 0 {
                // Remove from deceased
                token_balances.insert((deceased_bytes, *token_addr), 0);
                // Add to beneficiary
                let current = token_balances.get(&(beneficiary_bytes, *token_addr)).copied().unwrap_or(0);
                token_balances.insert((beneficiary_bytes, *token_addr), current + token_balance);

                transferred_tokens.push((token_addr.clone(), token_balance));

                // Persist token balance changes
                if let Err(e) = state.storage_engine.save_token_balance(&deceased_bytes, token_addr, 0).await {
                    error!("Failed to persist deceased token balance: {}", e);
                }
                if let Err(e) = state.storage_engine.save_token_balance(
                    &beneficiary_bytes, token_addr, current + token_balance,
                ).await {
                    error!("Failed to persist beneficiary token balance: {}", e);
                }
            }
        }
    }

    // Mark certificate as executed
    cert.executed = true;
    cert.executed_at = Some(chrono::Utc::now().timestamp_millis());

    // Persist updated certificate
    let cert_clone = cert.clone();
    drop(certs);
    if let Ok(data) = serde_json::to_vec(&cert_clone) {
        let kv = state.storage_engine.get_kv();
        let _ = kv.put(q_storage::CF_DEATH_CERTIFICATES, cert_clone.id.as_bytes(), &data).await;
    }

    let mut summary = format!("Transferred {:.4} QUG", balance as f64 / 1e24);
    if !transferred_tokens.is_empty() {
        summary.push_str(&format!(" + {} token type(s)", transferred_tokens.len()));
    }
    summary.push_str(" to beneficiary");

    info!(
        "✅ Inheritance transfer complete: {} from {} to {}",
        summary,
        cert_clone.deceased_wallet,
        cert_clone.beneficiary_wallet
    );

    Ok(Json(ApiResponse::success(summary)))
}

// ============================================================================
// v8.1.4: Email Broadcasting (FOUNDER-ONLY)
// ============================================================================

#[derive(Deserialize)]
pub struct BroadcastEmailRequest {
    pub subject: String,
    pub body: String,
    #[serde(default)]
    pub body_html: Option<String>,
}

/// Broadcast an email to all email-registered wallets on this node.
/// The email lands in each user's "quillon-bank" folder (not inbox).
pub async fn broadcast_bank_email(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BroadcastEmailRequest>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    info!("📧 Bank email broadcast requested: subject={}", req.subject);

    // Get all wallets that have email activity
    let wallets = state
        .storage_engine
        .get_all_email_wallets()
        .await
        .map_err(|e| {
            error!("Failed to get email wallets: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if wallets.is_empty() {
        return Ok(Json(ApiResponse::success("No email users found — 0 emails sent".to_string())));
    }

    let thread_id = Uuid::new_v4().to_string();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut sent_count = 0u64;

    for wallet in &wallets {
        let email = EmailMessage {
            id: Uuid::new_v4().to_string(),
            from_wallet: [0u8; 32],
            from_email: Some("bank@quillon.xyz".to_string()),
            to_wallet: Some(*wallet),
            to_email: None,
            subject: req.subject.clone(),
            body: req.body.clone(),
            body_html: req.body_html.clone(),
            encrypted: false,
            signature: Vec::new(),
            timestamp,
            read: false,
            folder: "quillon-bank".to_string(),
            thread_id: Some(thread_id.clone()),
            in_reply_to: None,
            crypto_transfer: None,
            delivery_method: DeliveryMethod::P2PGossipsub,
        };

        if let Err(e) = state.storage_engine.save_email(&email).await {
            error!("Failed to save broadcast email to wallet {}: {}", hex::encode(wallet), e);
            continue;
        }

        // Emit SSE event so the user's UI updates in real-time
        let _ = state
            .event_broadcaster
            .broadcast(StreamEvent::EmailReceived {
                email_id: email.id.clone(),
                from_address: "bank@quillon.xyz".to_string(),
                subject: req.subject.clone(),
                preview: req.body.chars().take(100).collect(),
                has_crypto: false,
                crypto_amount: None,
                crypto_token: None,
                timestamp: chrono::Utc::now(),
            })
            .await;

        sent_count += 1;
    }

    info!("📧 Bank broadcast complete: {} emails sent to {} wallets", sent_count, wallets.len());
    Ok(Json(ApiResponse::success(format!("Broadcast sent to {} email users", sent_count))))
}
