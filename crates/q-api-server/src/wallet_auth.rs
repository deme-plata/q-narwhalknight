//! Wallet Authentication Middleware
//!
//! Crypto-agile signature-based authentication supporting:
//! - Phase Q0: Ed25519 (classical)
//! - Phase Q1: Ed25519 + Dilithium5 (hybrid)
//! - Phase Q2: Dilithium5 (post-quantum)
//! - Critical ops: Dilithium5 + SPHINCS+ (ultra-secure)

use axum::{
    async_trait,
    extract::{FromRequestParts, OriginalUri, State},
    http::{request::Parts, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature as DalekSignature, Verifier, VerifyingKey};
use hmac::{Hmac, Mac};
use q_aegis_ql::{AegisQL, PublicKey as AegisPublicKey, Signature as AegisSignature};
use q_types::{Address, ApiResponse};
use q_wallet::{
    dilithium_wallet::Dilithium5KeyPair,
    sphincs_wallet::{OperationType, SphincsPlusKeyPair},
};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use sha3::{Digest, Sha3_256};
use std::sync::Arc;

type HmacSha256 = Hmac<Sha256>;

/// Cryptographic scheme used for authentication
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuthScheme {
    /// Phase Q0: Ed25519 only (64-byte signature)
    Ed25519,
    /// Phase Q1: Ed25519 + Dilithium5 (dual signature)
    Hybrid,
    /// Phase Q2: Dilithium5 only (~4.6 KB signature)
    Dilithium5,
    /// Critical operations: Dilithium5 + SPHINCS+ (~55 KB total)
    UltraSecure,
    /// AEGIS-QL: Fast post-quantum lattice-based crypto (~2 KB signature)
    AegisQL,
    /// AEGIS-QL Hybrid: Ed25519 + AEGIS-QL (dual signature)
    AegisQLHybrid,
}

impl AuthScheme {
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Ed25519 => "Classical Ed25519 (Phase Q0)",
            Self::Hybrid => "Hybrid Ed25519+Dilithium5 (Phase Q1)",
            Self::Dilithium5 => "Post-Quantum Dilithium5 (Phase Q2)",
            Self::UltraSecure => "Ultra-Secure Dilithium5+SPHINCS+ (Critical)",
            Self::AegisQL => "AEGIS-QL Post-Quantum Lattice-Based (Fast)",
            Self::AegisQLHybrid => "Hybrid Ed25519+AEGIS-QL (Performance)",
        }
    }
}

/// Authentication header containing signature proof
#[derive(Debug, Deserialize)]
pub struct AuthHeader {
    /// Wallet address (qnk prefix or hex)
    pub address: String,
    /// Unix timestamp of the request (to prevent replay attacks)
    pub timestamp: i64,
    /// Cryptographic scheme used
    #[serde(default = "default_scheme")]
    pub scheme: AuthScheme,
    /// Ed25519 signature (if scheme uses Ed25519)
    pub signature: Option<String>,
    /// Dilithium5 signature (if scheme uses Dilithium5)
    pub dilithium5_signature: Option<String>,
    /// Dilithium5 public key (required for Dilithium5 verification)
    pub dilithium5_public_key: Option<String>,
    /// SPHINCS+ signature (if UltraSecure scheme)
    pub sphincs_signature: Option<String>,
    /// SPHINCS+ public key (required for SPHINCS+ verification)
    pub sphincs_public_key: Option<String>,
    /// Operation type (determines if SPHINCS+ is required)
    #[serde(default)]
    pub operation_type: Option<OperationType>,
    /// AEGIS-QL signature (if scheme uses AEGIS-QL)
    pub aegis_signature: Option<String>,
    /// AEGIS-QL public key (required for AEGIS-QL verification)
    pub aegis_public_key: Option<String>,
}

fn default_scheme() -> AuthScheme {
    AuthScheme::Ed25519
}

/// Authenticated wallet - proves the request comes from the wallet owner
#[derive(Debug, Clone)]
pub struct AuthenticatedWallet {
    pub address: Address,
    pub timestamp: DateTime<Utc>,
    pub scheme: AuthScheme,
}

#[derive(Debug, Serialize)]
pub struct AuthError {
    error: String,
    message: String,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let status = StatusCode::UNAUTHORIZED;
        let body = Json(ApiResponse::<()>::error(self.message));
        (status, body).into_response()
    }
}

#[async_trait]
impl FromRequestParts<std::sync::Arc<crate::AppState>> for AuthenticatedWallet {
    type Rejection = AuthError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &std::sync::Arc<crate::AppState>,
    ) -> Result<Self, Self::Rejection> {
        let request_path = parts.uri.path();

        // v8.0.1: Try Bearer token first (OAuth2 flow), then fall back to X-Wallet-Auth
        if let Some(bearer_result) = try_bearer_auth(parts, state).await {
            return bearer_result;
        }

        // v8.9.x: Try AIOC service auth (AI Operations Center on localhost)
        if let Some(aioc_result) = try_aioc_service_auth(parts, state).await {
            return aioc_result;
        }

        // Extract cryptographic authentication header
        let auth_header = parts
            .headers
            .get("X-Wallet-Auth")
            .ok_or_else(|| {
                if request_path.contains("/transactions/") && !request_path.contains("/transactions/send") {
                    tracing::debug!("🔐 [AUTH] No X-Wallet-Auth header for transaction lookup: {}", request_path);
                }
                AuthError {
                    error: "missing_auth".to_string(),
                    message: "Missing X-Wallet-Auth or Authorization: Bearer header.".to_string(),
                }
            })?
            .to_str()
            .map_err(|_| AuthError {
                error: "invalid_auth_header".to_string(),
                message: "Invalid X-Wallet-Auth header format".to_string(),
            })?;

        // v3.4.5: Debug log for transaction lookup auth attempts
        if request_path.contains("/transactions/") && !request_path.contains("/transactions/send") {
            tracing::debug!("🔐 [AUTH] X-Wallet-Auth received for transaction lookup: {}", request_path);
        }

        // Parse JSON authentication header
        let auth: AuthHeader = serde_json::from_str(auth_header).map_err(|e| {
            tracing::warn!("🔐 [AUTH] Invalid JSON in X-Wallet-Auth header: {}", e);
            AuthError {
                error: "invalid_auth_json".to_string(),
                message: format!("Invalid authentication JSON: {}", e),
            }
        })?;

        // AUTH DEBUG logging removed - was spamming logs

        // Check timestamp to prevent replay attacks (max 5 minutes old)
        let now = Utc::now().timestamp();
        let age = now - auth.timestamp;
        if age.abs() > 300 {
            // 5 minutes
            return Err(AuthError {
                error: "expired_auth".to_string(),
                message:
                    "Authentication expired. Timestamp must be within 5 minutes of current time."
                        .to_string(),
            });
        }

        // Parse wallet address
        let hex_part = if auth.address.starts_with("qnk") {
            &auth.address[3..]
        } else {
            &auth.address
        };

        let address_bytes = hex::decode(hex_part).map_err(|_| AuthError {
            error: "invalid_address".to_string(),
            message: "Invalid wallet address format".to_string(),
        })?;

        if address_bytes.len() != 32 {
            return Err(AuthError {
                error: "invalid_address_length".to_string(),
                message: "Address must be 32 bytes".to_string(),
            });
        }

        let mut address = [0u8; 32];
        address.copy_from_slice(&address_bytes);

        // Generate authentication challenge message
        // Message format: SHA3-256(address + timestamp + request_path)
        // v8.2.8: Use OriginalUri to get the full path BEFORE axum nest() strips the prefix.
        // Without this, nested routes like /api/v1/email/send get stripped to /send,
        // causing a path mismatch with the frontend which signs the full path.
        let backend_path = parts
            .extensions
            .get::<OriginalUri>()
            .map(|ou| ou.path().to_string())
            .unwrap_or_else(|| parts.uri.path().to_string());
        let mut hasher = Sha3_256::new();
        hasher.update(&address);
        hasher.update(&auth.timestamp.to_le_bytes());
        hasher.update(backend_path.as_bytes());
        let message = hasher.finalize();

        // v3.4.5: Debug log for transaction lookup path verification
        if backend_path.contains("/transactions/") && !backend_path.contains("/transactions/send") {
            tracing::debug!(
                "🔐 [AUTH] Path verification: backend_path={} wallet={} timestamp={}",
                backend_path,
                &auth.address,
                auth.timestamp
            );
        }

        // Verify signature(s) based on scheme
        match auth.scheme {
            AuthScheme::Ed25519 => {
                verify_ed25519(&auth, &address, &message)?;
            }
            AuthScheme::Hybrid => {
                // BOTH Ed25519 AND Dilithium5 must verify
                verify_ed25519(&auth, &address, &message)?;
                verify_dilithium5(&auth, &address, &message)?;
            }
            AuthScheme::Dilithium5 => {
                verify_dilithium5(&auth, &address, &message)?;
            }
            AuthScheme::UltraSecure => {
                // ALL THREE signatures must verify: Ed25519, Dilithium5, SPHINCS+
                verify_dilithium5(&auth, &address, &message)?;
                verify_sphincs_plus(&auth, &address, &message)?;
            }
            AuthScheme::AegisQL => {
                verify_aegis_ql(&auth, &address, &message)?;
            }
            AuthScheme::AegisQLHybrid => {
                // BOTH Ed25519 AND AEGIS-QL must verify
                verify_ed25519(&auth, &address, &message)?;
                verify_aegis_ql(&auth, &address, &message)?;
            }
        }

        // Authentication successful!
        Ok(AuthenticatedWallet {
            address,
            timestamp: DateTime::from_timestamp(auth.timestamp, 0).unwrap_or_else(Utc::now),
            scheme: auth.scheme,
        })
    }
}

/// Verify Ed25519 signature
fn verify_ed25519(auth: &AuthHeader, address: &Address, message: &[u8]) -> Result<(), AuthError> {
    let signature_hex = auth.signature.as_ref().ok_or_else(|| AuthError {
        error: "missing_ed25519_signature".to_string(),
        message: "Ed25519 signature required for this scheme".to_string(),
    })?;

    // AUTH DEBUG logging removed - was spamming logs

    let sig_bytes = hex::decode(signature_hex).map_err(|_| AuthError {
        error: "invalid_signature".to_string(),
        message: "Invalid Ed25519 signature format".to_string(),
    })?;

    if sig_bytes.len() != 64 {
        return Err(AuthError {
            error: "invalid_signature_length".to_string(),
            message: "Ed25519 signature must be 64 bytes".to_string(),
        });
    }

    let public_key = VerifyingKey::from_bytes(address).map_err(|_| AuthError {
        error: "invalid_public_key".to_string(),
        message: "Invalid Ed25519 public key in address".to_string(),
    })?;

    let signature =
        DalekSignature::from_bytes(&sig_bytes[..64].try_into().map_err(|_| AuthError {
            error: "signature_conversion_failed".to_string(),
            message: "Failed to convert Ed25519 signature bytes".to_string(),
        })?);

    public_key
        .verify(message, &signature)
        .map_err(|e| {
            // v8.1.3: Upgraded to warn! for visibility in production logs
            tracing::warn!(
                "🔐 [AUTH FAIL] Ed25519 signature verification failed for address {}: {:?}",
                hex::encode(address),
                e
            );
            AuthError {
                error: "invalid_signature".to_string(),
                message: "Ed25519 signature verification failed".to_string(),
            }
        })?;

    Ok(())
}

/// Verify Dilithium5 post-quantum signature
fn verify_dilithium5(
    auth: &AuthHeader,
    address: &Address,
    message: &[u8],
) -> Result<(), AuthError> {
    let signature_hex = auth
        .dilithium5_signature
        .as_ref()
        .ok_or_else(|| AuthError {
            error: "missing_dilithium5_signature".to_string(),
            message: "Dilithium5 signature required for this scheme".to_string(),
        })?;

    let public_key_hex = auth
        .dilithium5_public_key
        .as_ref()
        .ok_or_else(|| AuthError {
            error: "missing_dilithium5_public_key".to_string(),
            message: "Dilithium5 public key required for verification".to_string(),
        })?;

    let sig_bytes = hex::decode(signature_hex).map_err(|_| AuthError {
        error: "invalid_dilithium5_signature".to_string(),
        message: "Invalid Dilithium5 signature format".to_string(),
    })?;

    let public_key_bytes = hex::decode(public_key_hex).map_err(|_| AuthError {
        error: "invalid_dilithium5_public_key".to_string(),
        message: "Invalid Dilithium5 public key format".to_string(),
    })?;

    // Verify that the public key derives to the provided address
    let derived_address = Dilithium5KeyPair::derive_address(&public_key_bytes);
    if &derived_address != address {
        return Err(AuthError {
            error: "address_mismatch".to_string(),
            message: "Dilithium5 public key does not match wallet address".to_string(),
        });
    }

    // Verify the Dilithium5 signature
    let is_valid =
        Dilithium5KeyPair::verify(message, &sig_bytes, &public_key_bytes).map_err(|e| {
            AuthError {
                error: "dilithium5_verification_failed".to_string(),
                message: format!("Dilithium5 verification error: {}", e),
            }
        })?;

    if !is_valid {
        return Err(AuthError {
            error: "invalid_dilithium5_signature".to_string(),
            message: "Dilithium5 signature verification failed".to_string(),
        });
    }

    Ok(())
}

/// Verify SPHINCS+ ultra-conservative signature (for critical operations)
fn verify_sphincs_plus(
    auth: &AuthHeader,
    address: &Address,
    message: &[u8],
) -> Result<(), AuthError> {
    let signature_hex = auth.sphincs_signature.as_ref().ok_or_else(|| AuthError {
        error: "missing_sphincs_signature".to_string(),
        message: "SPHINCS+ signature required for ultra-secure scheme".to_string(),
    })?;

    let public_key_hex = auth.sphincs_public_key.as_ref().ok_or_else(|| AuthError {
        error: "missing_sphincs_public_key".to_string(),
        message: "SPHINCS+ public key required for verification".to_string(),
    })?;

    let sig_bytes = hex::decode(signature_hex).map_err(|_| AuthError {
        error: "invalid_sphincs_signature".to_string(),
        message: "Invalid SPHINCS+ signature format".to_string(),
    })?;

    let public_key_bytes = hex::decode(public_key_hex).map_err(|_| AuthError {
        error: "invalid_sphincs_public_key".to_string(),
        message: "Invalid SPHINCS+ public key format".to_string(),
    })?;

    // Verify that the public key derives to the provided address
    let derived_address = SphincsPlusKeyPair::derive_address(&public_key_bytes);
    if &derived_address != address {
        return Err(AuthError {
            error: "address_mismatch".to_string(),
            message: "SPHINCS+ public key does not match wallet address".to_string(),
        });
    }

    // Verify the SPHINCS+ signature
    let is_valid =
        SphincsPlusKeyPair::verify(message, &sig_bytes, &public_key_bytes).map_err(|e| {
            AuthError {
                error: "sphincs_verification_failed".to_string(),
                message: format!("SPHINCS+ verification error: {}", e),
            }
        })?;

    if !is_valid {
        return Err(AuthError {
            error: "invalid_sphincs_signature".to_string(),
            message: "SPHINCS+ signature verification failed".to_string(),
        });
    }

    Ok(())
}

/// Verify AEGIS-QL post-quantum lattice-based signature
fn verify_aegis_ql(auth: &AuthHeader, address: &Address, message: &[u8]) -> Result<(), AuthError> {
    let signature_json = auth.aegis_signature.as_ref().ok_or_else(|| AuthError {
        error: "missing_aegis_signature".to_string(),
        message: "AEGIS-QL signature required for this scheme".to_string(),
    })?;

    let public_key_json = auth.aegis_public_key.as_ref().ok_or_else(|| AuthError {
        error: "missing_aegis_public_key".to_string(),
        message: "AEGIS-QL public key required for verification".to_string(),
    })?;

    // Deserialize AEGIS-QL signature from JSON
    let signature: AegisSignature =
        serde_json::from_str(signature_json).map_err(|e| AuthError {
            error: "invalid_aegis_signature".to_string(),
            message: format!("Invalid AEGIS-QL signature format: {}", e),
        })?;

    // Deserialize AEGIS-QL public key from JSON
    let public_key: AegisPublicKey =
        serde_json::from_str(public_key_json).map_err(|e| AuthError {
            error: "invalid_aegis_public_key".to_string(),
            message: format!("Invalid AEGIS-QL public key format: {}", e),
        })?;

    // Verify the AEGIS-QL signature
    let aegis = AegisQL::new();
    let is_valid = aegis
        .verify(message, &signature, &public_key)
        .map_err(|e| AuthError {
            error: "aegis_verification_failed".to_string(),
            message: format!("AEGIS-QL verification error: {:?}", e),
        })?;

    if !is_valid {
        return Err(AuthError {
            error: "invalid_aegis_signature".to_string(),
            message: "AEGIS-QL signature verification failed".to_string(),
        });
    }

    Ok(())
}

/// v8.0.1: Try Bearer token authentication (OAuth2 flow).
/// Returns Some(Ok(...)) if Bearer auth succeeded, Some(Err(...)) if token was invalid,
/// None if no Bearer token was present (fall through to X-Wallet-Auth).
async fn try_bearer_auth(
    parts: &Parts,
    state: &std::sync::Arc<crate::AppState>,
) -> Option<Result<AuthenticatedWallet, AuthError>> {
    let auth_header = parts.headers.get("Authorization")?.to_str().ok()?;
    let token = auth_header.strip_prefix("Bearer ")?;
    if token.is_empty() {
        return None;
    }

    // Look up the access token in OAuth2 storage
    let access_token = match state.oauth2_storage.get_access_token(token).await {
        Some(t) => t,
        None => {
            return Some(Err(AuthError {
                error: "invalid_bearer_token".to_string(),
                message: "Bearer token not found or revoked.".to_string(),
            }));
        }
    };

    // Check expiration
    if access_token.expires_at < chrono::Utc::now() {
        return Some(Err(AuthError {
            error: "expired_bearer_token".to_string(),
            message: "Bearer token has expired. Please re-authenticate.".to_string(),
        }));
    }

    // Extract wallet address from token
    let addr_str = &access_token.wallet_address;
    let hex_part = if addr_str.starts_with("qnk") {
        &addr_str[3..]
    } else {
        addr_str
    };

    let address_bytes = match hex::decode(hex_part) {
        Ok(b) if b.len() == 32 => b,
        _ => {
            return Some(Err(AuthError {
                error: "invalid_token_address".to_string(),
                message: "Bearer token has invalid wallet address.".to_string(),
            }));
        }
    };

    let mut address = [0u8; 32];
    address.copy_from_slice(&address_bytes);

    tracing::debug!(
        "🔐 [AUTH] Bearer token accepted for wallet qnk{}... (scopes: {:?})",
        &hex_part[..8.min(hex_part.len())],
        access_token.scopes
    );

    Some(Ok(AuthenticatedWallet {
        address,
        timestamp: chrono::Utc::now(),
        scheme: AuthScheme::Ed25519, // Bearer tokens don't have a crypto scheme
    }))
}

/// Generate authentication challenge for wallet
pub fn generate_auth_challenge(address: &Address, path: &str, timestamp: i64) -> Vec<u8> {
    let mut hasher = Sha3_256::new();
    hasher.update(address);
    hasher.update(&timestamp.to_le_bytes());
    hasher.update(path.as_bytes());
    hasher.finalize().to_vec()
}

/// v8.9.x: AIOC Service Authentication header
#[derive(Debug, Deserialize)]
struct AiocServiceHeader {
    service: String,
    wallet_address: String,
    hmac: String,
    timestamp: i64,
}

/// v8.9.x: Try AIOC service authentication.
/// Allows the AI Operations Center (running on localhost) to call authenticated
/// endpoints on behalf of a logged-in wallet user via HMAC-SHA256 shared secret.
///
/// Returns Some(Ok(...)) if AIOC auth succeeded, Some(Err(...)) if header was present but invalid,
/// None if no X-AIOC-Service-Auth header was present (fall through to X-Wallet-Auth).
async fn try_aioc_service_auth(
    parts: &Parts,
    state: &Arc<crate::AppState>,
) -> Option<Result<AuthenticatedWallet, AuthError>> {
    let raw_header = parts.headers.get("X-AIOC-Service-Auth");
    if raw_header.is_none() {
        return None; // No AIOC header at all — fall through silently
    }
    tracing::info!("🤖 [AIOC AUTH] Header found, attempting authentication...");
    let header_value = raw_header?.to_str().ok()?;
    if header_value.is_empty() {
        tracing::warn!("🤖 [AIOC AUTH] Header present but empty");
        return None;
    }

    // Parse the JSON header
    let aioc_header: AiocServiceHeader = match serde_json::from_str(header_value) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!("🤖 [AIOC AUTH] Invalid JSON in X-AIOC-Service-Auth: {}", e);
            return Some(Err(AuthError {
                error: "invalid_aioc_header".to_string(),
                message: format!("Invalid AIOC service auth JSON: {}", e),
            }));
        }
    };

    // Verify service name
    if aioc_header.service != "aioc" {
        return Some(Err(AuthError {
            error: "invalid_aioc_service".to_string(),
            message: "Invalid service identifier".to_string(),
        }));
    }

    // Verify timestamp within 5 minutes (replay protection)
    let now = Utc::now().timestamp();
    let age = now - aioc_header.timestamp;
    if age.abs() > 300 {
        tracing::warn!(
            "🤖 [AIOC AUTH] Expired timestamp: age={}s for wallet {}",
            age,
            &aioc_header.wallet_address
        );
        return Some(Err(AuthError {
            error: "expired_aioc_auth".to_string(),
            message: "AIOC service auth expired. Timestamp must be within 5 minutes.".to_string(),
        }));
    }

    // Verify request comes from localhost
    let connect_info = parts
        .extensions
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>();
    let is_localhost = connect_info
        .map(|ci| {
            let ip = ci.0.ip();
            tracing::debug!("🤖 [AIOC AUTH] Request from IP: {}", ip);
            ip.is_loopback()
        })
        .unwrap_or_else(|| {
            tracing::warn!("🤖 [AIOC AUTH] ConnectInfo not available — allowing (HMAC verified)");
            true // If ConnectInfo unavailable, trust HMAC alone
        });

    if !is_localhost {
        tracing::warn!(
            "🤖 [AIOC AUTH] Rejected non-localhost request from {:?} for wallet {}",
            connect_info.map(|ci| ci.0),
            &aioc_header.wallet_address
        );
        return Some(Err(AuthError {
            error: "aioc_not_localhost".to_string(),
            message: "AIOC service auth is only allowed from localhost".to_string(),
        }));
    }

    // Load shared secret from environment
    let secret = match std::env::var("Q_AIOC_SERVICE_SECRET") {
        Ok(s) if !s.is_empty() => s,
        _ => {
            tracing::warn!("🤖 [AIOC AUTH] Q_AIOC_SERVICE_SECRET not configured");
            return Some(Err(AuthError {
                error: "aioc_not_configured".to_string(),
                message: "AIOC service auth not configured on this node".to_string(),
            }));
        }
    };

    // Verify HMAC-SHA256(wallet_address + timestamp, secret)
    let mut mac = match HmacSha256::new_from_slice(secret.as_bytes()) {
        Ok(m) => m,
        Err(_) => {
            return Some(Err(AuthError {
                error: "aioc_hmac_error".to_string(),
                message: "HMAC initialization failed".to_string(),
            }));
        }
    };
    mac.update(aioc_header.wallet_address.as_bytes());
    mac.update(&aioc_header.timestamp.to_le_bytes());

    let expected_hmac = hex::decode(&aioc_header.hmac).unwrap_or_default();
    if mac.verify_slice(&expected_hmac).is_err() {
        tracing::warn!(
            "🤖 [AIOC AUTH] HMAC verification failed for wallet {}",
            &aioc_header.wallet_address
        );
        return Some(Err(AuthError {
            error: "invalid_aioc_hmac".to_string(),
            message: "AIOC service auth HMAC verification failed".to_string(),
        }));
    }

    // Parse wallet address
    let addr_str = &aioc_header.wallet_address;
    let hex_part = if addr_str.starts_with("qnk") {
        &addr_str[3..]
    } else {
        addr_str
    };

    let address_bytes = match hex::decode(hex_part) {
        Ok(b) if b.len() == 32 => b,
        _ => {
            return Some(Err(AuthError {
                error: "invalid_aioc_address".to_string(),
                message: "Invalid wallet address in AIOC service auth".to_string(),
            }));
        }
    };

    let mut address = [0u8; 32];
    address.copy_from_slice(&address_bytes);

    tracing::info!(
        "🤖 [AIOC AUTH] Service auth accepted for wallet qnk{}...",
        &hex_part[..8.min(hex_part.len())]
    );

    Some(Ok(AuthenticatedWallet {
        address,
        timestamp: DateTime::from_timestamp(aioc_header.timestamp, 0).unwrap_or_else(Utc::now),
        scheme: AuthScheme::Ed25519, // AIOC service auth doesn't have a crypto scheme
    }))
}

/// Validate an `X-Wallet-Auth`-style JSON auth header passed as a URL query parameter.
///
/// Used for endpoints where the client cannot set HTTP headers (SSE EventSource,
/// WebSocket). The frontend signs the canonical path (e.g. "/api/v1/events" or
/// "/ws/chat/signal") with SHA3-256(address || ts_le_i64 || path) challenge,
/// identical to the standard `X-Wallet-Auth` flow.
///
/// Returns the validated 32-byte wallet address on success.
pub fn validate_wallet_auth_query(auth_json: &str, expected_path: &str) -> Result<[u8; 32], &'static str> {
    let auth: AuthHeader = serde_json::from_str(auth_json).map_err(|_| "invalid_auth_json")?;

    // Replay-attack prevention: reject headers older than 5 minutes.
    // Surface the drift so callers can tell "client clock is off" from "stale token".
    let now = chrono::Utc::now().timestamp();
    let drift = now - auth.timestamp;
    if drift.abs() > 300 {
        tracing::warn!(
            "auth-via-query: expired_auth — drift={}s (server_now={}, client_ts={})",
            drift, now, auth.timestamp
        );
        return Err("expired_auth");
    }

    // Parse address — strip qnk prefix to get raw 32-byte public key
    let hex_part = if auth.address.starts_with("qnk") { &auth.address[3..] } else { &auth.address };
    let addr_bytes = hex::decode(hex_part).map_err(|_| "invalid_address")?;
    if addr_bytes.len() != 32 { return Err("invalid_address_length"); }
    let mut address = [0u8; 32];
    address.copy_from_slice(&addr_bytes);

    // Reconstruct the challenge: identical to generate_auth_challenge()
    let challenge = generate_auth_challenge(&address, expected_path, auth.timestamp);

    // Verify Ed25519 signature (only Ed25519 supported for WebSocket auth)
    let sig_hex = auth.signature.as_deref().ok_or("missing_signature")?;
    let sig_bytes = hex::decode(sig_hex).map_err(|_| "invalid_signature_hex")?;
    if sig_bytes.len() != 64 { return Err("invalid_signature_length"); }

    let sig_arr: [u8; 64] = sig_bytes.try_into().map_err(|_| "signature_conversion_failed")?;
    let sig = DalekSignature::from_bytes(&sig_arr);
    let vk = VerifyingKey::from_bytes(&address).map_err(|_| "invalid_public_key")?;
    vk.verify(&challenge, &sig).map_err(|_| "signature_verification_failed")?;

    Ok(address)
}

/// Backwards-compat shim — preserved so existing /ws/chat/signal call sites continue to work.
pub fn validate_signaling_auth_query(auth_json: &str, expected_peer_id: &str) -> Result<String, &'static str> {
    let strip_prefix = |s: &str| s.strip_prefix("qnk").unwrap_or(s).to_lowercase();
    let addr_bytes = validate_wallet_auth_query(auth_json, "/ws/chat/signal")?;
    let addr_str = format!("qnk{}", hex::encode(addr_bytes));
    if strip_prefix(&addr_str) != strip_prefix(expected_peer_id) {
        return Err("peer_id_mismatch");
    }
    Ok(addr_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    #[test]
    fn test_auth_challenge_generation() {
        let address = [1u8; 32];
        let path = "/api/v1/wallets/test/balance";
        let timestamp = 1234567890;

        let challenge = generate_auth_challenge(&address, path, timestamp);
        assert_eq!(challenge.len(), 32); // SHA3-256 output
    }

    #[test]
    fn test_validate_wallet_auth_query_valid() {
        let secret = [7u8; 32];
        let signing_key = SigningKey::from_bytes(&secret);
        let address = signing_key.verifying_key().to_bytes();
        let timestamp = chrono::Utc::now().timestamp();
        let challenge = generate_auth_challenge(&address, "/api/v1/events", timestamp);
        let sig = signing_key.sign(&challenge);
        let auth_json = serde_json::json!({
            "address": format!("qnk{}", hex::encode(address)),
            "timestamp": timestamp,
            "scheme": "Ed25519",
            "signature": hex::encode(sig.to_bytes()),
        })
        .to_string();

        let out = validate_wallet_auth_query(&auth_json, "/api/v1/events").expect("should validate");
        assert_eq!(out, address);
    }

    #[test]
    fn test_validate_wallet_auth_query_expired() {
        let auth_json = serde_json::json!({
            "address": format!("qnk{}", "11".repeat(32)),
            "timestamp": chrono::Utc::now().timestamp() - 301,
            "scheme": "Ed25519",
            "signature": "00".repeat(64),
        })
        .to_string();
        let err = validate_wallet_auth_query(&auth_json, "/api/v1/events").unwrap_err();
        assert_eq!(err, "expired_auth");
    }

    #[test]
    fn test_validate_wallet_auth_query_wrong_signature() {
        let secret = [8u8; 32];
        let signing_key = SigningKey::from_bytes(&secret);
        let address = signing_key.verifying_key().to_bytes();
        let timestamp = chrono::Utc::now().timestamp();
        let challenge = generate_auth_challenge(&address, "/api/v1/events", timestamp);
        let mut sig = signing_key.sign(&challenge).to_bytes();
        sig[0] ^= 0x01;
        let auth_json = serde_json::json!({
            "address": format!("qnk{}", hex::encode(address)),
            "timestamp": timestamp,
            "scheme": "Ed25519",
            "signature": hex::encode(sig),
        })
        .to_string();
        let err = validate_wallet_auth_query(&auth_json, "/api/v1/events").unwrap_err();
        assert_eq!(err, "signature_verification_failed");
    }

    #[test]
    fn test_validate_wallet_auth_query_wrong_path() {
        let secret = [9u8; 32];
        let signing_key = SigningKey::from_bytes(&secret);
        let address = signing_key.verifying_key().to_bytes();
        let timestamp = chrono::Utc::now().timestamp();
        let challenge = generate_auth_challenge(&address, "/ws/chat/signal", timestamp);
        let sig = signing_key.sign(&challenge);
        let auth_json = serde_json::json!({
            "address": format!("qnk{}", hex::encode(address)),
            "timestamp": timestamp,
            "scheme": "Ed25519",
            "signature": hex::encode(sig.to_bytes()),
        })
        .to_string();
        let err = validate_wallet_auth_query(&auth_json, "/api/v1/events").unwrap_err();
        assert_eq!(err, "signature_verification_failed");
    }
}
