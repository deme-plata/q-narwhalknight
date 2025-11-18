//! Zcash Wallet API - HTTP Endpoints
//!
//! RESTful API for Zcash (ZEC) wallet operations:
//! - GET /api/zcash/status - Node sync status
//! - POST /api/zcash/address - Generate new address
//! - GET /api/zcash/balance - Get wallet balance
//! - POST /api/zcash/send - Send shielded transaction
//! - GET /api/zcash/transactions - Transaction history

use crate::zcash_rpc::{ZcashAddressType, ZcashRpcClient};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info};

/// Shared Zcash client state
pub type ZcashState = Arc<ZcashRpcClient>;

/// API Response wrapper
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    Success { data: T },
    Error { error: String },
}

impl<T> IntoResponse for ApiResponse<T>
where
    T: Serialize,
{
    fn into_response(self) -> Response {
        match serde_json::to_string(&self) {
            Ok(json) => (StatusCode::OK, json).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("{{\"error\": \"Serialization failed: {}\"}}", e),
            )
                .into_response(),
        }
    }
}

/// Node status response
#[derive(Debug, Serialize)]
pub struct NodeStatusResponse {
    pub is_synced: bool,
    pub current_height: u64,
    pub sync_progress: f64,
}

/// Address generation request
#[derive(Debug, Deserialize)]
pub struct GenerateAddressRequest {
    #[serde(default = "default_address_type")]
    pub address_type: String, // "transparent" | "sapling" | "orchard"
}

fn default_address_type() -> String {
    "sapling".to_string() // Default to shielded Sapling addresses
}

/// Address generation response
#[derive(Debug, Serialize)]
pub struct AddressResponse {
    pub address: String,
    pub address_type: String,
}

/// Balance response (all address types)
#[derive(Debug, Serialize)]
pub struct BalanceResponse {
    pub transparent: f64,
    pub sapling: f64,
    pub orchard: f64,
    pub total: f64,
}

/// Send transaction request
#[derive(Debug, Deserialize)]
pub struct SendRequest {
    pub from_address: String,
    pub to_address: String,
    pub amount: f64,
    pub memo: Option<String>,
}

/// Send transaction response
#[derive(Debug, Serialize)]
pub struct SendResponse {
    pub txid: String,
    pub amount: f64,
}

/// Transaction history response
#[derive(Debug, Serialize)]
pub struct TransactionHistoryResponse {
    pub transactions: Vec<crate::zcash_rpc::ZcashTransaction>,
    pub count: usize,
}

/// GET /api/zcash/status - Check Zebra node sync status
pub async fn get_node_status(
    State(client): State<ZcashState>,
) -> Result<ApiResponse<NodeStatusResponse>, StatusCode> {
    info!("📊 [ZCASH API] GET /api/zcash/status");

    match client.check_node_status().await {
        Ok((is_synced, height, progress)) => Ok(ApiResponse::Success {
            data: NodeStatusResponse {
                is_synced,
                current_height: height,
                sync_progress: progress,
            },
        }),
        Err(e) => {
            error!("❌ [ZCASH API] Node status check failed: {}", e);
            Ok(ApiResponse::Error {
                error: format!("Node status check failed: {}", e),
            })
        }
    }
}

/// POST /api/zcash/address - Generate new Zcash address
pub async fn generate_address(
    State(client): State<ZcashState>,
    Json(req): Json<GenerateAddressRequest>,
) -> Result<ApiResponse<AddressResponse>, StatusCode> {
    info!(
        "🔑 [ZCASH API] POST /api/zcash/address - type: {}",
        req.address_type
    );

    let addr_type = match req.address_type.to_lowercase().as_str() {
        "transparent" | "t" => ZcashAddressType::Transparent,
        "sapling" | "z" | "shielded" => ZcashAddressType::Sapling,
        "orchard" | "o" | "unified" => ZcashAddressType::Orchard,
        _ => {
            return Ok(ApiResponse::Error {
                error: format!(
                    "Invalid address type: '{}'. Use 'transparent', 'sapling', or 'orchard'",
                    req.address_type
                ),
            });
        }
    };

    match client.generate_address(addr_type.clone()).await {
        Ok(address) => Ok(ApiResponse::Success {
            data: AddressResponse {
                address,
                address_type: req.address_type,
            },
        }),
        Err(e) => {
            error!("❌ [ZCASH API] Address generation failed: {}", e);
            Ok(ApiResponse::Error {
                error: format!("Address generation failed: {}", e),
            })
        }
    }
}

/// GET /api/zcash/balance - Get wallet balance (all types)
pub async fn get_balance(
    State(client): State<ZcashState>,
) -> Result<ApiResponse<BalanceResponse>, StatusCode> {
    info!("💰 [ZCASH API] GET /api/zcash/balance");

    match client.get_balance().await {
        Ok(balance) => Ok(ApiResponse::Success {
            data: BalanceResponse {
                transparent: balance.transparent,
                sapling: balance.sapling,
                orchard: balance.orchard,
                total: balance.total,
            },
        }),
        Err(e) => {
            error!("❌ [ZCASH API] Balance query failed: {}", e);
            Ok(ApiResponse::Error {
                error: format!("Balance query failed: {}", e),
            })
        }
    }
}

/// POST /api/zcash/send - Send shielded transaction
pub async fn send_transaction(
    State(client): State<ZcashState>,
    Json(req): Json<SendRequest>,
) -> Result<ApiResponse<SendResponse>, StatusCode> {
    info!(
        "📤 [ZCASH API] POST /api/zcash/send - {} ZEC to {}",
        req.amount, req.to_address
    );

    // Validate amount
    if req.amount <= 0.0 {
        return Ok(ApiResponse::Error {
            error: "Amount must be greater than 0".to_string(),
        });
    }

    match client
        .send_shielded(&req.from_address, &req.to_address, req.amount, req.memo)
        .await
    {
        Ok(txid) => {
            info!("✅ [ZCASH API] Transaction sent: {}", txid);
            Ok(ApiResponse::Success {
                data: SendResponse {
                    txid,
                    amount: req.amount,
                },
            })
        }
        Err(e) => {
            error!("❌ [ZCASH API] Transaction failed: {}", e);
            Ok(ApiResponse::Error {
                error: format!("Transaction failed: {}", e),
            })
        }
    }
}

/// GET /api/zcash/transactions?count=10 - Get transaction history
pub async fn get_transactions(
    State(client): State<ZcashState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<ApiResponse<TransactionHistoryResponse>, StatusCode> {
    let count = params
        .get("count")
        .and_then(|c| c.parse::<usize>().ok())
        .unwrap_or(10);

    info!("📜 [ZCASH API] GET /api/zcash/transactions?count={}", count);

    match client.get_transaction_history(count).await {
        Ok(transactions) => Ok(ApiResponse::Success {
            data: TransactionHistoryResponse {
                count: transactions.len(),
                transactions,
            },
        }),
        Err(e) => {
            error!("❌ [ZCASH API] Transaction history failed: {}", e);
            Ok(ApiResponse::Error {
                error: format!("Transaction history failed: {}", e),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_address_type() {
        let default_type = default_address_type();
        assert_eq!(default_type, "sapling");
    }

    #[test]
    fn test_api_response_serialization() {
        let response: ApiResponse<String> = ApiResponse::Success {
            data: "test".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"data\":\"test\""));
    }
}
