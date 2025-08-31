use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};
use q_types::*;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{AppState, StreamEvent};

/// Health check endpoint
pub async fn health_check() -> Result<Json<ApiResponse<String>>, StatusCode> {
    Ok(Json(ApiResponse::success("OK".to_string())))
}

/// Prometheus metrics endpoint
pub async fn metrics(State(_state): State<Arc<AppState>>) -> Result<String, StatusCode> {
    // TODO: Implement proper Prometheus metrics
    Ok("# Q-NarwhalKnight metrics\n# Coming soon...".to_string())
}

/// Node status endpoint
pub async fn node_status(State(state): State<Arc<AppState>>) -> Result<Json<ApiResponse<NodeStatus>>, StatusCode> {
    let status = state.node_status.read().await.clone();
    Ok(Json(ApiResponse::success(status)))
}

/// Create a new wallet
pub async fn create_wallet(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateWalletRequest>,
) -> Result<Json<ApiResponse<WalletInfo>>, StatusCode> {
    debug!("Creating new wallet");

    match state
        .wallet_manager
        .create_wallet(request.password.as_deref())
        .await
    {
        Ok(wallet) => {
            info!("Created wallet with ID: {}", wallet.id);
            Ok(Json(ApiResponse::success(wallet)))
        }
        Err(e) => {
            error!("Failed to create wallet: {}", e);
            Ok(Json(ApiResponse::error(format!(
                "Failed to create wallet: {}",
                e
            ))))
        }
    }
}

/// Get wallet information
pub async fn get_wallet(
    State(state): State<Arc<AppState>>,
    Path(wallet_id): Path<Uuid>,
) -> Result<Json<ApiResponse<WalletInfo>>, StatusCode> {
    debug!("Getting wallet info for ID: {}", wallet_id);

    match state.wallet_manager.get_wallet(&wallet_id).await {
        Ok(Some(wallet)) => Ok(Json(ApiResponse::success(wallet))),
        Ok(None) => Ok(Json(ApiResponse::error("Wallet not found".to_string()))),
        Err(e) => {
            error!("Failed to get wallet: {}", e);
            Ok(Json(ApiResponse::error(format!(
                "Failed to get wallet: {}",
                e
            ))))
        }
    }
}

/// List all wallets
pub async fn list_wallets(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<WalletInfo>>>, StatusCode> {
    debug!("Listing all wallets");

    match state.wallet_manager.list_wallets().await {
        Ok(wallets) => Ok(Json(ApiResponse::success(wallets))),
        Err(e) => {
            error!("Failed to list wallets: {}", e);
            Ok(Json(ApiResponse::error(format!(
                "Failed to list wallets: {}",
                e
            ))))
        }
    }
}

/// Sign a transaction
pub async fn sign_transaction(
    State(state): State<Arc<AppState>>,
    Path(wallet_id): Path<Uuid>,
    Json(request): Json<SignTransactionRequest>,
) -> Result<Json<ApiResponse<Transaction>>, StatusCode> {
    debug!("Signing transaction for wallet: {}", wallet_id);

    // Create transaction
    let transaction = match state
        .wallet_manager
        .create_transaction(&wallet_id, request.to, request.amount, request.fee)
        .await
    {
        Ok(tx) => tx,
        Err(e) => {
            error!("Failed to create transaction: {}", e);
            return Ok(Json(ApiResponse::error(format!(
                "Failed to create transaction: {}",
                e
            ))));
        }
    };

    // Sign transaction
    match state
        .wallet_manager
        .sign_transaction(&wallet_id, transaction, Some(&request.password))
        .await
    {
        Ok(signed_tx) => {
            info!("Signed transaction for wallet: {}", wallet_id);
            Ok(Json(ApiResponse::success(signed_tx)))
        }
        Err(e) => {
            error!("Failed to sign transaction: {}", e);
            Ok(Json(ApiResponse::error(format!(
                "Failed to sign transaction: {}",
                e
            ))))
        }
    }
}

/// Submit a transaction to the mempool
pub async fn submit_transaction(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SubmitTransactionRequest>,
) -> Result<Json<ApiResponse<TxHash>>, StatusCode> {
    debug!("Submitting transaction");

    let tx_hash = request.transaction.hash();
    
    // Add to transaction pool
    {
        let mut tx_pool = state.tx_pool.write().await;
        tx_pool.insert(tx_hash, request.transaction.clone());
    }

    // Update transaction status
    {
        let mut tx_status = state.tx_status.write().await;
        tx_status.insert(tx_hash, TxStatus::InMempool);
    }

    // Emit real-time event
    let event = StreamEvent::TransactionSubmitted {
        transaction: request.transaction,
        timestamp: chrono::Utc::now(),
    };
    
    if let Err(e) = state.event_emitter.emit_immediate(event).await {
        warn!("Failed to emit transaction submitted event: {}", e);
    }

    // TODO: Actually broadcast to P2P network and process through consensus

    info!("Submitted transaction with hash: {:?}", tx_hash);
    Ok(Json(ApiResponse::success(tx_hash)))
}

/// Get transaction status
pub async fn get_transaction(
    State(state): State<Arc<AppState>>,
    Path(tx_hash_str): Path<String>,
) -> Result<Json<ApiResponse<TxStatus>>, StatusCode> {
    debug!("Getting transaction status for: {}", tx_hash_str);

    // Parse transaction hash from hex string
    let tx_hash = match hex::decode(&tx_hash_str) {
        Ok(bytes) if bytes.len() == 32 => {
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&bytes);
            hash
        }
        _ => {
            return Ok(Json(ApiResponse::error(
                "Invalid transaction hash format".to_string(),
            )));
        }
    };

    let tx_status = state.tx_status.read().await;
    match tx_status.get(&tx_hash) {
        Some(status) => Ok(Json(ApiResponse::success(status.clone()))),
        None => Ok(Json(ApiResponse::error(
            "Transaction not found".to_string(),
        ))),
    }
}

/// Get block by height
pub async fn get_block(
    State(state): State<Arc<AppState>>,
    Path(height): Path<Height>,
) -> Result<Json<ApiResponse<Vec<Transaction>>>, StatusCode> {
    debug!("Getting block at height: {}", height);

    let blocks = state.blocks.read().await;
    match blocks.get(&height) {
        Some(transactions) => Ok(Json(ApiResponse::success(transactions.clone()))),
        None => Ok(Json(ApiResponse::error("Block not found".to_string()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;
    use axum::http::StatusCode;
    use axum_test::TestServer;

    async fn create_test_server() -> TestServer {
        let config = Config::default();
        let state = Arc::new(AppState::new(config).await.unwrap());
        
        let app = axum::Router::new()
            .route("/health", axum::routing::get(health_check))
            .route("/api/v1/wallets", axum::routing::post(create_wallet))
            .route("/api/v1/wallets", axum::routing::get(list_wallets))
            .with_state(state);

        TestServer::new(app).unwrap()
    }

    #[tokio::test]
    async fn test_health_check() {
        let server = create_test_server().await;
        let response = server.get("/health").await;
        assert_eq!(response.status_code(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_create_wallet() {
        let server = create_test_server().await;
        
        let request = CreateWalletRequest {
            password: Some("test123".to_string()),
            mnemonic: None,
        };

        let response = server
            .post("/api/v1/wallets")
            .json(&request)
            .await;
        
        assert_eq!(response.status_code(), StatusCode::OK);
        
        let body: ApiResponse<WalletInfo> = response.json();
        assert!(body.success);
        assert!(body.data.is_some());
    }
}