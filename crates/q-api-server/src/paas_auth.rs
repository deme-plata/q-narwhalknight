// PaaS Authentication Module
// Implements hybrid cryptographic authentication for Privacy-as-a-Service endpoints
//
// Security Model:
// - Phase 0: ECDSA (secp256k1) signatures for backward compatibility
// - Phase 1: Hybrid ECDSA + Dilithium5 signatures for quantum resistance
// - Phase 2+: Pure post-quantum signatures (Dilithium5)
//
// Authentication Flow:
// 1. Client signs request with wallet private key
// 2. Server verifies signature and extracts public key
// 3. Public key hash becomes wallet address for balance checks
// 4. Rate limiting applied based on account tier

use axum::{extract::Request, http::StatusCode, middleware::Next, response::Response};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// Authentication token passed in X-Auth-Token header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaaSAuthToken {
    /// Wallet address (32-byte public key hash)
    pub wallet_address: [u8; 32],

    /// Timestamp of signature (Unix epoch milliseconds)
    pub timestamp: u64,

    /// Signature type: "ecdsa", "dilithium5", or "hybrid"
    pub signature_type: String,

    /// ECDSA signature (65 bytes: r(32) + s(32) + v(1))
    pub ecdsa_signature: Option<Vec<u8>>,

    /// Dilithium5 signature (4627 bytes)
    pub dilithium5_signature: Option<Vec<u8>>,

    /// Dilithium5 public key (2592 bytes) - required for signature verification
    /// For ECDSA, the public key is recovered from the signature
    /// For Dilithium5, the public key must be provided
    pub dilithium5_public_key: Option<Vec<u8>>,

    /// Message that was signed (typically: timestamp + endpoint + body_hash)
    pub signed_message: Vec<u8>,
}

/// Account tier determines rate limits and feature access
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccountTier {
    /// Free tier: 100 requests/minute
    Free,

    /// Standard tier: 1,000 requests/minute
    Standard,

    /// Premium tier: 10,000 requests/minute
    Premium,

    /// Enterprise tier: Unlimited requests
    Enterprise,
}

impl AccountTier {
    pub fn rate_limit_per_minute(&self) -> Option<u32> {
        match self {
            AccountTier::Free => Some(100),
            AccountTier::Standard => Some(1000),
            AccountTier::Premium => Some(10000),
            AccountTier::Enterprise => None, // Unlimited
        }
    }

    // v3.0.6-beta: Updated for u128 with 24 decimals (1 QNK = 10^24 base units)
    pub fn from_balance_qnk(balance: u128) -> Self {
        const ONE_QNK: u128 = 1_000_000_000_000_000_000_000_000; // 10^24
        let balance_whole = balance / ONE_QNK; // Convert to whole QNK

        match balance_whole {
            0..=100 => AccountTier::Free,
            101..=1000 => AccountTier::Standard,
            1001..=10000 => AccountTier::Premium,
            _ => AccountTier::Enterprise,
        }
    }
}

/// Authenticated request context
#[derive(Debug, Clone)]
pub struct AuthContext {
    pub wallet_address: [u8; 32],
    pub account_tier: AccountTier,
    pub signature_type: String,
    pub authenticated_at: u64,
}

/// Rate limiting state
#[derive(Debug, Clone)]
struct RateLimitEntry {
    requests: u32,
    window_start: u64, // Unix timestamp in seconds
}

/// PaaS Authentication Manager
pub struct PaaSAuthManager {
    /// Rate limiting state per wallet address
    rate_limits: Arc<RwLock<HashMap<[u8; 32], RateLimitEntry>>>,

    /// Account tier overrides (for premium/enterprise customers)
    tier_overrides: Arc<RwLock<HashMap<[u8; 32], AccountTier>>>,
}

impl PaaSAuthManager {
    pub fn new() -> Self {
        Self {
            rate_limits: Arc::new(RwLock::new(HashMap::new())),
            tier_overrides: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Verify authentication token and extract wallet address
    pub async fn verify_auth_token(&self, token: &PaaSAuthToken) -> Result<AuthContext, String> {
        // 1. Check timestamp freshness (must be within 5 minutes)
        let current_time = chrono::Utc::now().timestamp_millis() as u64;
        let time_diff = current_time.saturating_sub(token.timestamp);

        if time_diff > 300_000 {
            // 5 minutes
            return Err("Authentication token expired (>5 minutes old)".to_string());
        }

        if token.timestamp > current_time + 60_000 {
            // 1 minute future tolerance
            return Err("Authentication token timestamp is in the future".to_string());
        }

        // 2. Verify signature based on type
        let verified = match token.signature_type.as_str() {
            "ecdsa" => self.verify_ecdsa_signature(token).await?,
            "dilithium5" => self.verify_dilithium5_signature(token).await?,
            "hybrid" => {
                // Both signatures must be valid
                let ecdsa_valid = self.verify_ecdsa_signature(token).await?;
                let dilithium5_valid = self.verify_dilithium5_signature(token).await?;
                ecdsa_valid && dilithium5_valid
            }
            _ => {
                return Err(format!(
                    "Unsupported signature type: {}",
                    token.signature_type
                ))
            }
        };

        if !verified {
            return Err("Signature verification failed".to_string());
        }

        // 3. Determine account tier (from overrides or balance)
        let tier_overrides = self.tier_overrides.read().await;
        let account_tier = tier_overrides
            .get(&token.wallet_address)
            .copied()
            .unwrap_or(AccountTier::Free); // Default to free tier

        info!(
            "✅ Authentication successful for wallet {} (tier: {:?})",
            hex::encode(&token.wallet_address[..8]),
            account_tier
        );

        Ok(AuthContext {
            wallet_address: token.wallet_address,
            account_tier,
            signature_type: token.signature_type.clone(),
            authenticated_at: current_time,
        })
    }

    /// Verify ECDSA signature (secp256k1)
    async fn verify_ecdsa_signature(&self, token: &PaaSAuthToken) -> Result<bool, String> {
        let signature_bytes = token
            .ecdsa_signature
            .as_ref()
            .ok_or("Missing ECDSA signature")?;

        if signature_bytes.len() != 65 {
            return Err("Invalid ECDSA signature length (expected 65 bytes)".to_string());
        }

        // Extract r, s, v components
        let r = &signature_bytes[0..32];
        let s = &signature_bytes[32..64];
        let recovery_id = signature_bytes[64];

        // Verify recovery ID is valid (0-3)
        if recovery_id > 3 {
            return Err("Invalid ECDSA recovery ID".to_string());
        }

        // Hash the signed message (this is the message hash that was signed)
        let message_hash = Sha256::digest(&token.signed_message);

        // Create secp256k1 context
        use secp256k1::{
            ecdsa::{RecoverableSignature, RecoveryId},
            Message, Secp256k1,
        };

        let secp = Secp256k1::new();

        // Parse recovery ID
        let rec_id = RecoveryId::from_i32(recovery_id as i32)
            .map_err(|e| format!("Invalid recovery ID: {}", e))?;

        // Construct recoverable signature from r, s, and recovery ID
        let mut sig_data = [0u8; 64];
        sig_data[0..32].copy_from_slice(r);
        sig_data[32..64].copy_from_slice(s);

        let signature = RecoverableSignature::from_compact(&sig_data, rec_id)
            .map_err(|e| format!("Invalid signature format: {}", e))?;

        // Parse message hash
        let message = Message::from_digest_slice(&message_hash)
            .map_err(|e| format!("Invalid message hash: {}", e))?;

        // Recover public key from signature
        let recovered_pubkey = secp
            .recover_ecdsa(&message, &signature)
            .map_err(|e| format!("Failed to recover public key: {}", e))?;

        // Serialize recovered public key and hash it to get wallet address
        let pubkey_bytes = recovered_pubkey.serialize_uncompressed();
        let pubkey_hash = Sha256::digest(&pubkey_bytes[1..]); // Skip 0x04 prefix

        // Compare with expected wallet address
        if pubkey_hash.as_slice() != &token.wallet_address[..] {
            warn!(
                "🔐 ECDSA verification failed: wallet mismatch (expected: {}, recovered: {})",
                hex::encode(&token.wallet_address[..8]),
                hex::encode(&pubkey_hash[..8])
            );
            return Ok(false);
        }

        info!(
            "✅ ECDSA signature verified successfully for wallet {}",
            hex::encode(&token.wallet_address[..8])
        );

        Ok(true)
    }

    /// Verify Dilithium5 signature
    async fn verify_dilithium5_signature(&self, token: &PaaSAuthToken) -> Result<bool, String> {
        let signature_bytes = token
            .dilithium5_signature
            .as_ref()
            .ok_or("Missing Dilithium5 signature")?;

        let public_key_bytes = token
            .dilithium5_public_key
            .as_ref()
            .ok_or("Missing Dilithium5 public key")?;

        // Dilithium5 signatures are 4627 bytes
        if signature_bytes.len() != 4627 {
            return Err(format!(
                "Invalid Dilithium5 signature length (expected 4627, got {})",
                signature_bytes.len()
            ));
        }

        // Dilithium5 public keys are 2592 bytes
        if public_key_bytes.len() != 2592 {
            return Err(format!(
                "Invalid Dilithium5 public key length (expected 2592, got {})",
                public_key_bytes.len()
            ));
        }

        // Use pqcrypto-dilithium for verification
        use pqcrypto_dilithium::dilithium5;
        use pqcrypto_traits::sign::{DetachedSignature as _, PublicKey as _};

        // Parse public key
        let public_key = dilithium5::PublicKey::from_bytes(public_key_bytes)
            .map_err(|e| format!("Invalid Dilithium5 public key: {:?}", e))?;

        // Parse signature
        let signature = dilithium5::DetachedSignature::from_bytes(signature_bytes)
            .map_err(|e| format!("Invalid Dilithium5 signature: {:?}", e))?;

        // Verify signature
        let verification_result =
            dilithium5::verify_detached_signature(&signature, &token.signed_message, &public_key);

        match verification_result {
            Ok(_) => {
                // Verify that the public key hash matches the wallet address
                let pubkey_hash = Sha256::digest(public_key_bytes);

                if pubkey_hash.as_slice() != &token.wallet_address[..] {
                    warn!(
                        "🔐 Dilithium5 verification failed: wallet mismatch (expected: {}, computed: {})",
                        hex::encode(&token.wallet_address[..8]),
                        hex::encode(&pubkey_hash[..8])
                    );
                    return Ok(false);
                }

                info!(
                    "✅ Dilithium5 signature verified successfully for wallet {}",
                    hex::encode(&token.wallet_address[..8])
                );
                Ok(true)
            }
            Err(e) => {
                warn!("🔐 Dilithium5 signature verification failed: {:?}", e);
                Ok(false)
            }
        }
    }

    /// Check rate limit for wallet address
    pub async fn check_rate_limit(
        &self,
        wallet_address: &[u8; 32],
        tier: AccountTier,
    ) -> Result<(), String> {
        // Enterprise tier has no rate limits
        if tier == AccountTier::Enterprise {
            return Ok(());
        }

        let rate_limit = tier.rate_limit_per_minute().unwrap();
        let current_time = chrono::Utc::now().timestamp() as u64;
        let current_window = current_time / 60; // 1-minute windows

        let mut rate_limits = self.rate_limits.write().await;
        let entry = rate_limits
            .entry(*wallet_address)
            .or_insert(RateLimitEntry {
                requests: 0,
                window_start: current_window,
            });

        // Reset counter if we're in a new time window
        if entry.window_start < current_window {
            entry.requests = 0;
            entry.window_start = current_window;
        }

        // Check if rate limit exceeded
        if entry.requests >= rate_limit {
            warn!(
                "🚫 Rate limit exceeded for wallet {} (tier: {:?}, limit: {}/min)",
                hex::encode(&wallet_address[..8]),
                tier,
                rate_limit
            );
            return Err(format!(
                "Rate limit exceeded: {} requests/minute (tier: {:?})",
                rate_limit, tier
            ));
        }

        // Increment request counter
        entry.requests += 1;

        info!(
            "✅ Rate limit check passed for wallet {} ({}/{} requests this minute)",
            hex::encode(&wallet_address[..8]),
            entry.requests,
            rate_limit
        );

        Ok(())
    }

    /// Set account tier override (for premium/enterprise customers)
    pub async fn set_account_tier(&self, wallet_address: [u8; 32], tier: AccountTier) {
        let mut tier_overrides = self.tier_overrides.write().await;
        tier_overrides.insert(wallet_address, tier);

        info!(
            "🎯 Account tier set for wallet {}: {:?}",
            hex::encode(&wallet_address[..8]),
            tier
        );
    }
}

/// Axum middleware for PaaS authentication
pub async fn paas_auth_middleware(
    auth_manager: Arc<PaaSAuthManager>,
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract X-Auth-Token header
    let auth_header = request
        .headers()
        .get("X-Auth-Token")
        .and_then(|h| h.to_str().ok())
        .ok_or_else(|| {
            error!("❌ Missing X-Auth-Token header");
            StatusCode::UNAUTHORIZED
        })?;

    // Parse authentication token (JSON format)
    let auth_token: PaaSAuthToken = serde_json::from_str(auth_header).map_err(|e| {
        error!("❌ Invalid authentication token format: {}", e);
        StatusCode::UNAUTHORIZED
    })?;

    // Verify authentication
    let auth_context = auth_manager
        .verify_auth_token(&auth_token)
        .await
        .map_err(|e| {
            error!("❌ Authentication failed: {}", e);
            StatusCode::UNAUTHORIZED
        })?;

    // Check rate limit
    auth_manager
        .check_rate_limit(&auth_context.wallet_address, auth_context.account_tier)
        .await
        .map_err(|e| {
            error!("❌ Rate limit check failed: {}", e);
            StatusCode::TOO_MANY_REQUESTS
        })?;

    // Store auth context in request extensions for handlers to access
    request.extensions_mut().insert(auth_context);

    // Continue to next middleware/handler
    Ok(next.run(request).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_account_tier_from_balance() {
        assert_eq!(
            AccountTier::from_balance_qnk(50_00000000),
            AccountTier::Free
        );
        assert_eq!(
            AccountTier::from_balance_qnk(500_00000000),
            AccountTier::Standard
        );
        assert_eq!(
            AccountTier::from_balance_qnk(5000_00000000),
            AccountTier::Premium
        );
        assert_eq!(
            AccountTier::from_balance_qnk(50000_00000000),
            AccountTier::Enterprise
        );
    }

    #[tokio::test]
    async fn test_rate_limit_basic() {
        let auth_manager = PaaSAuthManager::new();
        let wallet = [1u8; 32];

        // Free tier: 100 requests/minute
        for i in 0..100 {
            let result = auth_manager
                .check_rate_limit(&wallet, AccountTier::Free)
                .await;
            assert!(result.is_ok(), "Request {} should succeed", i);
        }

        // 101st request should fail
        let result = auth_manager
            .check_rate_limit(&wallet, AccountTier::Free)
            .await;
        assert!(result.is_err(), "101st request should exceed rate limit");
    }

    #[tokio::test]
    async fn test_enterprise_unlimited() {
        let auth_manager = PaaSAuthManager::new();
        let wallet = [2u8; 32];

        // Enterprise tier: unlimited requests
        for i in 0..10000 {
            let result = auth_manager
                .check_rate_limit(&wallet, AccountTier::Enterprise)
                .await;
            assert!(result.is_ok(), "Enterprise request {} should succeed", i);
        }
    }
}
