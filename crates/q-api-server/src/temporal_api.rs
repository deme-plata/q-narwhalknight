//! TemporalShield API endpoints
//!
//! Provides REST API for protecting messages with TemporalShield-STARK.
//! All proofs are generated with NO TRUSTED SETUP.
//!
//! ## Security Properties
//! - OTP encryption (information-theoretic secrecy)
//! - Shamir (k,n) threshold sharing
//! - ML-KEM-1024 (post-quantum key encapsulation)
//! - zk-STARK proofs (Winterfell - NO TRUSTED SETUP)

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::AppState;

// Import TemporalShield components
use q_temporal_shield::{
    TemporalShield, TemporalShieldConfig, TemporalEnvelope,
    TrusteePublicKey,
};

/// Request to protect a message
#[derive(Debug, Deserialize)]
pub struct ProtectRequest {
    /// Message to protect (base64 encoded)
    pub message: String,
    /// Threshold k (minimum shares needed to reconstruct)
    pub threshold: usize,
    /// Total trustees n
    pub total_trustees: usize,
    /// Trustee public keys (base64 encoded)
    pub trustee_public_keys: Vec<String>,
}

/// Response with protected envelope
#[derive(Debug, Serialize)]
pub struct ProtectResponse {
    /// Envelope ID for tracking
    pub envelope_id: String,
    /// Encrypted ciphertext (base64)
    pub ciphertext: String,
    /// Key commitment (hex)
    pub key_commitment: String,
    /// Share commitments (hex)
    pub share_commitments: Vec<String>,
    /// Encrypted shares for each trustee (base64)
    pub encrypted_shares: Vec<String>,
    /// STARK proof (base64) - NO TRUSTED SETUP
    pub stark_proof: String,
    /// Proof size in bytes
    pub proof_size_bytes: usize,
    /// Message: NO TRUSTED SETUP
    pub security_note: String,
}

/// Request to verify an envelope
#[derive(Debug, Deserialize)]
pub struct VerifyRequest {
    /// STARK proof (base64)
    pub stark_proof: String,
    /// Key commitment (hex)
    pub key_commitment: String,
    /// Share commitments (hex)
    pub share_commitments: Vec<String>,
    /// Threshold k
    pub threshold: usize,
    /// Total trustees n
    pub total_trustees: usize,
}

/// Verification response
#[derive(Debug, Serialize)]
pub struct VerifyResponse {
    /// Whether the proof is valid
    pub valid: bool,
    /// Verification message
    pub message: String,
    /// Security note
    pub security_note: String,
}

/// Request to reconstruct a message
#[derive(Debug, Deserialize)]
pub struct ReconstructRequest {
    /// Encrypted ciphertext (base64)
    pub ciphertext: String,
    /// Key commitment (hex)
    pub key_commitment: String,
    /// Share commitments (hex)
    pub share_commitments: Vec<String>,
    /// Decrypted shares (index, base64 share)
    pub decrypted_shares: Vec<(usize, String)>,
    /// Threshold k
    pub threshold: usize,
    /// Total trustees n
    pub total_trustees: usize,
    /// Message size
    pub message_size: usize,
}

/// Reconstruction response
#[derive(Debug, Serialize)]
pub struct ReconstructResponse {
    /// Recovered message (base64)
    pub message: String,
    /// Success status
    pub success: bool,
}

/// Generate a new trustee key pair
#[derive(Debug, Serialize)]
pub struct TrusteeKeyResponse {
    /// Trustee ID (hex)
    pub trustee_id: String,
    /// Public key (base64)
    pub public_key: String,
    /// Private key (base64) - SENSITIVE!
    pub private_key: String,
    /// Warning message
    pub warning: String,
}

/// Protect a message with TemporalShield-STARK
///
/// NO TRUSTED SETUP - All proofs are transparent and post-quantum secure.
pub async fn protect_message(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<ProtectRequest>,
) -> impl IntoResponse {
    // Decode message
    let message = match BASE64.decode(&req.message) {
        Ok(m) => m,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "Invalid base64 message"
                })),
            )
        }
    };

    // Validate parameters
    if req.threshold == 0 || req.threshold > req.total_trustees {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "Invalid threshold: must be 0 < k <= n"
            })),
        );
    }

    if req.trustee_public_keys.len() != req.total_trustees {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("Expected {} trustee keys, got {}", req.total_trustees, req.trustee_public_keys.len())
            })),
        );
    }

    // Decode trustee public keys from base64
    let trustees: Result<Vec<TrusteePublicKey>, _> = req.trustee_public_keys
        .iter()
        .map(|pk_b64| {
            let bytes = BASE64.decode(pk_b64)?;
            TrusteePublicKey::from_bytes(&bytes)
                .map_err(|e| base64::DecodeError::InvalidLength(bytes.len()))
        })
        .collect();

    let trustees = match trustees {
        Ok(t) => t,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "Invalid trustee public key format"
                })),
            )
        }
    };

    // Create TemporalShield with config
    let config = match TemporalShieldConfig::custom(req.threshold, req.total_trustees, 128) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("Invalid config: {:?}", e)
                })),
            )
        }
    };
    let shield = TemporalShield::new(config);

    // Protect message (OTP + Shamir + ML-KEM + STARK)
    let envelope = match shield.protect(&message, &trustees) {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Protection failed: {:?}", e)
                })),
            )
        }
    };

    // Serialize envelope
    let envelope_bytes = match envelope.to_bytes() {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Envelope serialization failed: {:?}", e)
                })),
            )
        }
    };

    let response = ProtectResponse {
        envelope_id: format!("ts-{}", uuid::Uuid::new_v4()),
        ciphertext: BASE64.encode(&envelope.ciphertext),
        key_commitment: hex::encode(envelope.key_commitment),
        share_commitments: envelope.share_commitments.iter().map(hex::encode).collect(),
        encrypted_shares: envelope.encrypted_shares.iter()
            .map(|s| BASE64.encode(&s.encrypted_data))
            .collect(),
        stark_proof: BASE64.encode(&envelope.stark_proof),
        proof_size_bytes: envelope.stark_proof.len(),
        security_note: "NO TRUSTED SETUP - Proof uses zk-STARKs with Fiat-Shamir transformation".to_string(),
    };

    (StatusCode::OK, Json(serde_json::to_value(response).unwrap()))
}

/// Verify a TemporalShield envelope
///
/// NO TRUSTED SETUP - Verification is transparent and deterministic.
pub async fn verify_envelope(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<VerifyRequest>,
) -> impl IntoResponse {
    // Decode proof
    let proof_bytes = match BASE64.decode(&req.stark_proof) {
        Ok(p) => p,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "Invalid base64 proof"
                })),
            )
        }
    };

    // Decode key commitment
    let key_commitment = match hex::decode(&req.key_commitment) {
        Ok(k) if k.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&k);
            arr
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "Invalid key commitment (expected 64 hex chars)"
                })),
            )
        }
    };

    // Decode share commitments
    let share_commitments: Result<Vec<[u8; 32]>, _> = req.share_commitments
        .iter()
        .map(|s| {
            let bytes = hex::decode(s)?;
            if bytes.len() != 32 {
                return Err(hex::FromHexError::InvalidStringLength);
            }
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            Ok(arr)
        })
        .collect();

    let share_commitments = match share_commitments {
        Ok(sc) => sc,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "Invalid share commitment format"
                })),
            )
        }
    };

    // Verify using the STARK verifier (NO TRUSTED SETUP)
    let result = q_temporal_shield::stark::verify_proof(
        proof_bytes,
        &key_commitment,
        &share_commitments,
        req.threshold,
        req.total_trustees,
    );

    let (valid, message) = match result {
        Ok(()) => (true, "STARK proof verified successfully".to_string()),
        Err(e) => (false, format!("Verification failed: {:?}", e)),
    };

    let response = VerifyResponse {
        valid,
        message,
        security_note: "Verification uses NO TRUSTED SETUP - all randomness derived from public transcript".to_string(),
    };

    (StatusCode::OK, Json(serde_json::to_value(response).unwrap()))
}

/// Reconstruct a message from decrypted shares
pub async fn reconstruct_message(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<ReconstructRequest>,
) -> impl IntoResponse {
    // Validate we have enough shares
    if req.decrypted_shares.len() < req.threshold {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("Insufficient shares: have {}, need {}", req.decrypted_shares.len(), req.threshold)
            })),
        );
    }

    // Decode ciphertext
    let ciphertext = match BASE64.decode(&req.ciphertext) {
        Ok(c) => c,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "Invalid base64 ciphertext"
                })),
            )
        }
    };

    // Decode key commitment
    let key_commitment = match hex::decode(&req.key_commitment) {
        Ok(k) if k.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&k);
            arr
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "Invalid key commitment"
                })),
            )
        }
    };

    // Decode share commitments
    let share_commitments: Result<Vec<[u8; 32]>, _> = req.share_commitments
        .iter()
        .map(|s| {
            let bytes = hex::decode(s)?;
            if bytes.len() != 32 {
                return Err(hex::FromHexError::InvalidStringLength);
            }
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            Ok(arr)
        })
        .collect();

    let share_commitments = match share_commitments {
        Ok(sc) => sc,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "Invalid share commitment format"
                })),
            )
        }
    };

    // Decode the decrypted shares
    let decrypted_shares: Result<Vec<(usize, Vec<u8>)>, base64::DecodeError> = req.decrypted_shares
        .iter()
        .map(|(idx, share_b64)| {
            let share_data = BASE64.decode(share_b64)?;
            Ok((*idx, share_data))
        })
        .collect();

    let decrypted_shares = match decrypted_shares {
        Ok(ds) => ds,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "Invalid share data format"
                })),
            )
        }
    };

    // Create config and shield
    let config = match TemporalShieldConfig::custom(req.threshold, req.total_trustees, 128) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("Invalid config: {:?}", e)
                })),
            )
        }
    };
    let shield = TemporalShield::new(config);

    // Create a minimal envelope for reconstruction using the constructor
    // Note: In production, the full envelope would be stored/provided
    let envelope = TemporalEnvelope::new(
        ciphertext,
        key_commitment,
        share_commitments,
        vec![], // encrypted_shares not needed for reconstruction
        vec![], // stark_proof already verified
        req.threshold,
        req.total_trustees,
        [0u8; 32], // config_hash
        1,         // num_chunks
    );

    // Reconstruct the message
    let result = shield.reconstruct(&envelope, &decrypted_shares);

    match result {
        Ok(message) => {
            let response = ReconstructResponse {
                message: BASE64.encode(&message),
                success: true,
            };
            (StatusCode::OK, Json(serde_json::to_value(response).unwrap()))
        }
        Err(e) => {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("Reconstruction failed: {:?}", e),
                    "success": false
                })),
            )
        }
    }
}

/// Generate a new trustee key pair
pub async fn generate_trustee_key(
    State(_state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Generate a real ML-KEM + Dilithium keypair
    let keypair = match TrusteePublicKey::generate(None) {
        Ok(kp) => kp,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Key generation failed: {:?}", e)
                })),
            )
        }
    };

    // Serialize public key
    let public_key_bytes = match keypair.public_key.to_bytes() {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Public key serialization failed: {:?}", e)
                })),
            )
        }
    };

    // Combine private key components for serialization
    // Note: The private key contains the ML-KEM secret key and Dilithium secret key
    // We concatenate them with length prefixes for later parsing
    let mut private_key_bytes = Vec::new();
    private_key_bytes.extend_from_slice(&(keypair.private_key.kem_secret_key.len() as u32).to_le_bytes());
    private_key_bytes.extend_from_slice(&keypair.private_key.kem_secret_key);
    if let Some(ref sig_sk) = keypair.private_key.signature_secret_key {
        private_key_bytes.extend_from_slice(&(sig_sk.len() as u32).to_le_bytes());
        private_key_bytes.extend_from_slice(sig_sk);
    }

    let response = TrusteeKeyResponse {
        trustee_id: hex::encode(keypair.public_key.id),
        public_key: BASE64.encode(&public_key_bytes),
        private_key: BASE64.encode(&private_key_bytes),
        warning: "SENSITIVE: Store private key securely. Consider using HSM for production.".to_string(),
    };

    (StatusCode::OK, Json(serde_json::to_value(response).unwrap()))
}

/// Get TemporalShield security information
pub async fn get_security_info() -> impl IntoResponse {
    let info = serde_json::json!({
        "name": "TemporalShield-STARK",
        "version": "0.1.0",
        "description": "Post-quantum secure secret sharing with zk-STARK proofs",
        "security_properties": {
            "trusted_setup": "NONE - Uses zk-STARKs, not zk-SNARKs",
            "post_quantum": true,
            "encryption": "OTP (information-theoretic secrecy) + ML-KEM-1024",
            "secret_sharing": "Shamir (k,n) threshold scheme",
            "proof_system": "zk-STARK (Winterfell library)",
            "hash_function": "BLAKE3 (256-bit)"
        },
        "recommended_parameters": {
            "threshold_3_of_5": {
                "k": 3,
                "n": 5,
                "security_level": "128-bit"
            },
            "threshold_5_of_9": {
                "k": 5,
                "n": 9,
                "security_level": "128-bit"
            }
        },
        "use_cases": [
            "Private transaction memos",
            "Validator key backup",
            "AI chat history archive",
            "Oracle commit-reveal schemes"
        ]
    });

    (StatusCode::OK, Json(info))
}

/// Configure routes for TemporalShield API
pub fn routes() -> axum::Router<Arc<AppState>> {
    use axum::routing::{get, post};

    axum::Router::new()
        .route("/temporal/protect", post(protect_message))
        .route("/temporal/verify", post(verify_envelope))
        .route("/temporal/reconstruct", post(reconstruct_message))
        .route("/temporal/generate-trustee-key", post(generate_trustee_key))
        .route("/temporal/info", get(get_security_info))
}
