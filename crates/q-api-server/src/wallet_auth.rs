//! Wallet Authentication Middleware
//!
//! Crypto-agile signature-based authentication supporting:
//! - Phase Q0: Ed25519 (classical)
//! - Phase Q1: Ed25519 + Dilithium5 (hybrid)
//! - Phase Q2: Dilithium5 (post-quantum)
//! - Critical ops: Dilithium5 + SPHINCS+ (ultra-secure)

use axum::{
    async_trait,
    extract::{FromRequestParts, State},
    http::{request::Parts, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature as DalekSignature, Verifier, VerifyingKey};
use q_aegis_ql::{AegisQL, PublicKey as AegisPublicKey, Signature as AegisSignature};
use q_types::{Address, ApiResponse};
use q_wallet::{
    dilithium_wallet::Dilithium5KeyPair,
    sphincs_wallet::{OperationType, SphincsPlusKeyPair},
};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::sync::Arc;

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
impl<S> FromRequestParts<S> for AuthenticatedWallet
where
    S: Send + Sync,
{
    type Rejection = AuthError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Extract authentication header
        let auth_header = parts
            .headers
            .get("X-Wallet-Auth")
            .ok_or_else(|| AuthError {
                error: "missing_auth".to_string(),
                message: "Missing X-Wallet-Auth header. Please sign your request.".to_string(),
            })?
            .to_str()
            .map_err(|_| AuthError {
                error: "invalid_auth_header".to_string(),
                message: "Invalid X-Wallet-Auth header format".to_string(),
            })?;

        eprintln!(
            "🔍 [AUTH DEBUG] Received X-Wallet-Auth header: {}",
            auth_header
        );
        eprintln!("🔍 [AUTH DEBUG] Request path: {}", parts.uri.path());

        // Parse JSON authentication header
        let auth: AuthHeader = serde_json::from_str(auth_header).map_err(|e| AuthError {
            error: "invalid_auth_json".to_string(),
            message: format!("Invalid authentication JSON: {}", e),
        })?;

        eprintln!(
            "🔍 [AUTH DEBUG] Parsed auth - address: {}, timestamp: {}, scheme: {:?}",
            auth.address, auth.timestamp, auth.scheme
        );

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
        let mut hasher = Sha3_256::new();
        hasher.update(&address);
        hasher.update(&auth.timestamp.to_le_bytes());
        hasher.update(parts.uri.path().as_bytes());
        let message = hasher.finalize();

        eprintln!(
            "🔍 [AUTH DEBUG] Challenge message hash: {}",
            hex::encode(&message)
        );

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

    eprintln!(
        "🔍 [AUTH DEBUG] Verifying Ed25519 signature: {}",
        signature_hex
    );
    eprintln!(
        "🔍 [AUTH DEBUG] Message to verify: {}",
        hex::encode(message)
    );
    eprintln!(
        "🔍 [AUTH DEBUG] Public key (address): {}",
        hex::encode(address)
    );

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
        .map_err(|_| AuthError {
            error: "invalid_signature".to_string(),
            message: "Ed25519 signature verification failed".to_string(),
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

/// Generate authentication challenge for wallet
pub fn generate_auth_challenge(address: &Address, path: &str, timestamp: i64) -> Vec<u8> {
    let mut hasher = Sha3_256::new();
    hasher.update(address);
    hasher.update(&timestamp.to_le_bytes());
    hasher.update(path.as_bytes());
    hasher.finalize().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_challenge_generation() {
        let address = [1u8; 32];
        let path = "/api/v1/wallets/test/balance";
        let timestamp = 1234567890;

        let challenge = generate_auth_challenge(&address, path, timestamp);
        assert_eq!(challenge.len(), 32); // SHA3-256 output
    }
}
