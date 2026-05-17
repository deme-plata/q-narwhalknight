//! AEGIS-QL Authentication Middleware for Quillon Bank API
//!
//! Provides post-quantum signature verification for all sensitive banking operations.
//! Only requests signed by the founder wallet are authorized.

use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use q_aegis_ql::{AegisError, AegisQL, PublicKey as AegisPublicKey, Signature as AegisSignature};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// Founder wallet address (hardcoded for security)
pub const FOUNDER_WALLET: &str = "efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723";

/// Maximum allowed time difference for timestamp validation (5 minutes)
const MAX_TIMESTAMP_DIFF: i64 = 300; // seconds

/// AEGIS-QL access control state
pub struct AegisAuthState {
    /// Founder's AEGIS-QL public key for signature verification
    pub founder_public_key: AegisPublicKey,
    /// Founder wallet address (derived from public key)
    pub founder_wallet: [u8; 32],
}

impl AegisAuthState {
    /// Create new AEGIS-QL auth state with founder's public key
    pub fn new(founder_public_key: AegisPublicKey) -> Self {
        // Derive wallet from public key (must match FOUNDER_WALLET constant)
        let wallet = derive_wallet_from_pubkey(&founder_public_key);

        Self {
            founder_public_key,
            founder_wallet: wallet,
        }
    }

    /// Verify wallet address matches founder
    pub fn is_founder(&self, wallet: &[u8; 32]) -> bool {
        wallet == &self.founder_wallet
    }

    /// Verify AEGIS-QL signature
    pub fn verify_signature(
        &self,
        message: &[u8],
        signature: &AegisSignature,
    ) -> Result<bool, AegisError> {
        let aegis = AegisQL::new();
        aegis.verify(message, signature, &self.founder_public_key)
    }
}

/// Derive wallet address from AEGIS-QL public key (SHA3-256)
fn derive_wallet_from_pubkey(public_key: &AegisPublicKey) -> [u8; 32] {
    use sha3::{Digest, Sha3_256};

    let mut hasher = Sha3_256::new();

    // Hash polynomial a
    for coeff in &public_key.a {
        hasher.update(&coeff.to_le_bytes());
    }

    // Hash polynomial t
    for coeff in &public_key.t {
        hasher.update(&coeff.to_le_bytes());
    }

    let hash = hasher.finalize();
    let mut wallet = [0u8; 32];
    wallet.copy_from_slice(&hash);
    wallet
}

/// Extract AEGIS-QL authentication headers from request
#[derive(Debug)]
struct AuthHeaders {
    wallet_address: [u8; 32],
    signature: AegisSignature,
    timestamp: i64,
    operation: String,
}

impl AuthHeaders {
    /// Parse authentication headers from HTTP request
    fn from_headers(headers: &HeaderMap) -> Result<Self, StatusCode> {
        // Extract wallet address
        let wallet_hex = headers
            .get("X-Wallet-Address")
            .ok_or(StatusCode::UNAUTHORIZED)?
            .to_str()
            .map_err(|_| StatusCode::BAD_REQUEST)?;

        let wallet_bytes = hex::decode(wallet_hex).map_err(|_| StatusCode::BAD_REQUEST)?;

        if wallet_bytes.len() != 32 {
            error!("Invalid wallet address length: {}", wallet_bytes.len());
            return Err(StatusCode::BAD_REQUEST);
        }

        let mut wallet_address = [0u8; 32];
        wallet_address.copy_from_slice(&wallet_bytes);

        // Extract AEGIS-QL signature
        let signature_hex = headers
            .get("X-AEGIS-Signature")
            .ok_or(StatusCode::UNAUTHORIZED)?
            .to_str()
            .map_err(|_| StatusCode::BAD_REQUEST)?;

        let signature_bytes = hex::decode(signature_hex).map_err(|_| StatusCode::BAD_REQUEST)?;

        let signature = AegisSignature::from_bytes(&signature_bytes).map_err(|_| {
            error!("Failed to deserialize AEGIS-QL signature");
            StatusCode::BAD_REQUEST
        })?;

        // Extract timestamp
        let timestamp_str = headers
            .get("X-Timestamp")
            .ok_or(StatusCode::UNAUTHORIZED)?
            .to_str()
            .map_err(|_| StatusCode::BAD_REQUEST)?;

        let timestamp = timestamp_str
            .parse::<i64>()
            .map_err(|_| StatusCode::BAD_REQUEST)?;

        // Extract operation
        let operation = headers
            .get("X-Operation")
            .ok_or(StatusCode::UNAUTHORIZED)?
            .to_str()
            .map_err(|_| StatusCode::BAD_REQUEST)?
            .to_string();

        Ok(Self {
            wallet_address,
            signature,
            timestamp,
            operation,
        })
    }

    /// Validate timestamp is within acceptable window
    fn validate_timestamp(&self) -> Result<(), StatusCode> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .as_secs() as i64;

        let time_diff = (now - self.timestamp).abs();

        if time_diff > MAX_TIMESTAMP_DIFF {
            warn!(
                "Timestamp validation failed: diff={}s (max={}s)",
                time_diff, MAX_TIMESTAMP_DIFF
            );
            return Err(StatusCode::UNAUTHORIZED);
        }

        Ok(())
    }

    /// Reconstruct signed message for verification
    fn reconstruct_message(&self) -> String {
        format!("QUILLON_BANK:{}:{}", self.operation, self.timestamp)
    }
}

/// Axum middleware to verify AEGIS-QL signatures for founder-only operations
pub async fn verify_founder_signature(
    State(auth_state): State<Arc<RwLock<AegisAuthState>>>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    info!("🔐 AEGIS-QL authentication check initiated");

    // Localhost bypass: CLI running on the same machine can use X-Admin-Local header.
    // SECURITY (issue #58): fail closed when ConnectInfo is missing. The previous
    // `.unwrap_or(true)` defaulted to "local" if no SocketAddr was attached, which is
    // exploitable behind any proxy that strips connection metadata (or in any tower-only
    // pipeline that constructs Requests without ConnectInfo). For Unix-socket deployments,
    // set `Q_ENABLE_LOCAL_ADMIN_BYPASS=1` so the bypass can engage; otherwise the bypass
    // is rejected and the request must pass normal AEGIS founder authentication.
    if request.headers().get("X-Admin-Local").map(|v| v.as_bytes()) == Some(b"true") {
        let is_local = request
            .extensions()
            .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
            .map(|ci| ci.0.ip().is_loopback())
            .unwrap_or_else(|| {
                // Fail closed unless explicitly opted in.
                std::env::var("Q_ENABLE_LOCAL_ADMIN_BYPASS").ok().as_deref() == Some("1")
            });

        if is_local {
            info!("✅ Localhost admin bypass - X-Admin-Local header from loopback address");
            return Ok(next.run(request).await);
        } else {
            warn!("❌ X-Admin-Local header rejected — non-loopback address or missing ConnectInfo (set Q_ENABLE_LOCAL_ADMIN_BYPASS=1 for unix-socket deployments)");
        }
    }

    // Extract authentication headers
    let auth_headers = AuthHeaders::from_headers(request.headers())?;

    info!(
        "📍 Wallet: {}",
        hex::encode(&auth_headers.wallet_address[..8])
    );
    info!("⏱️  Timestamp: {}", auth_headers.timestamp);
    info!("🎯 Operation: {}", auth_headers.operation);

    // Validate timestamp (prevent replay attacks)
    auth_headers.validate_timestamp()?;
    info!("✅ Timestamp valid (within 5-minute window)");

    // Get auth state
    let auth_state_guard = auth_state.read().await;

    // Verify wallet is founder
    if !auth_state_guard.is_founder(&auth_headers.wallet_address) {
        warn!(
            "❌ Unauthorized wallet: {} (expected: {})",
            hex::encode(&auth_headers.wallet_address),
            FOUNDER_WALLET
        );
        return Err(StatusCode::FORBIDDEN);
    }
    info!("✅ Wallet verified as FOUNDER");

    // Reconstruct message that was signed
    let message = auth_headers.reconstruct_message();

    // Verify AEGIS-QL signature
    let is_valid = auth_state_guard
        .verify_signature(message.as_bytes(), &auth_headers.signature)
        .map_err(|e| {
            error!("AEGIS-QL signature verification error: {:?}", e);
            StatusCode::UNAUTHORIZED
        })?;

    if !is_valid {
        warn!("❌ Invalid AEGIS-QL signature");
        return Err(StatusCode::UNAUTHORIZED);
    }

    info!("✅ AEGIS-QL signature verified (post-quantum secure)");
    info!("🎉 Authentication successful - proceeding with operation");

    // Authentication successful - proceed with request
    Ok(next.run(request).await)
}

/// Create protected router with AEGIS-QL authentication
pub fn create_protected_routes() -> axum::Router<Arc<crate::AppState>> {
    use crate::quillon_bank_api;
    use axum::routing::{get, post};

    axum::Router::new()
        // Stablecoin operations (founder-only)
        .route("/stablecoin/mint", post(quillon_bank_api::mint_qnkusd))
        .route("/stablecoin/burn", post(quillon_bank_api::burn_qnkusd))
        .route(
            "/stablecoin/collateral/add",
            post(quillon_bank_api::add_collateral),
        )
        .route(
            "/stablecoin/collateral/rebalance",
            post(quillon_bank_api::rebalance_collateral),
        )
        .route("/stablecoin/peg/adjust", post(quillon_bank_api::adjust_peg))
        // Lending operations (founder-only)
        .route("/lending/approve", post(quillon_bank_api::approve_loan))
        .route("/lending/reject", post(quillon_bank_api::reject_loan))
        .route("/lending/liquidate", post(quillon_bank_api::liquidate_loan))
        // Treasury operations (founder-only)
        .route(
            "/treasury/reserves/allocate",
            post(quillon_bank_api::allocate_reserves),
        )
        .route(
            "/treasury/profits/distribute",
            post(quillon_bank_api::distribute_profits),
        )
        // Risk management (founder-only)
        .route(
            "/risk/liquidations/execute",
            post(quillon_bank_api::execute_liquidations),
        )
        // v8.1.4: Email broadcasting (founder-only)
        .route(
            "/email/broadcast",
            post(quillon_bank_api::broadcast_bank_email),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_founder_wallet_constant() {
        // Verify founder wallet is correctly formatted
        assert_eq!(FOUNDER_WALLET.len(), 64); // 32 bytes = 64 hex chars
        assert!(hex::decode(FOUNDER_WALLET).is_ok());
    }

    #[test]
    fn test_timestamp_validation() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        // Valid timestamp (current time)
        let auth = AuthHeaders {
            wallet_address: [0u8; 32],
            signature: AegisSignature {
                z: vec![],
                c: [0u8; 32],
            },
            timestamp: now,
            operation: "TEST".to_string(),
        };
        assert!(auth.validate_timestamp().is_ok());

        // Invalid timestamp (too old)
        let auth_old = AuthHeaders {
            wallet_address: [0u8; 32],
            signature: AegisSignature {
                z: vec![],
                c: [0u8; 32],
            },
            timestamp: now - 400, // 6+ minutes old
            operation: "TEST".to_string(),
        };
        assert!(auth_old.validate_timestamp().is_err());
    }

    #[test]
    fn test_message_reconstruction() {
        let auth = AuthHeaders {
            wallet_address: [0u8; 32],
            signature: AegisSignature {
                z: vec![],
                c: [0u8; 32],
            },
            timestamp: 1729983456,
            operation: "MINT_QUGUSD".to_string(),
        };

        let message = auth.reconstruct_message();
        assert_eq!(message, "QUILLON_BANK:MINT_QUGUSD:1729983456");
    }
}
