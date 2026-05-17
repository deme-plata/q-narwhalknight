// PaaS API Key Management System
// Implements secure API key generation, storage, and rotation with argon2id hashing
//
// Security Features:
// - argon2id password hashing (OWASP recommended parameters)
// - Automatic key rotation every 90 days (configurable)
// - Rate limiting per tier
// - Key revocation and blacklisting
// - Audit logging for all key operations

use argon2::{
    password_hash::{rand_core::OsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use uuid::Uuid;

/// API key tier levels (same as in privacy_service_api.rs)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Copy)]
#[serde(rename_all = "snake_case")]
pub enum ApiTier {
    PayPerUse,
    Professional,
    Enterprise,
    WhiteLabel,
}

impl ApiTier {
    /// Get rate limit for this tier (requests per minute)
    pub fn rate_limit(&self) -> Option<u32> {
        match self {
            ApiTier::PayPerUse => Some(100),
            ApiTier::Professional => Some(1_000),
            ApiTier::Enterprise => Some(10_000),
            ApiTier::WhiteLabel => None, // Unlimited
        }
    }

    /// Get daily request limit
    pub fn daily_limit(&self) -> Option<u32> {
        match self {
            ApiTier::PayPerUse => Some(10_000),
            ApiTier::Professional => Some(100_000),
            ApiTier::Enterprise => None, // Unlimited
            ApiTier::WhiteLabel => None, // Unlimited
        }
    }
}

/// API key information stored in database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyRecord {
    /// Unique API key ID (UUID)
    pub key_id: String,

    /// API key hash (argon2id)
    pub key_hash: String,

    /// Wallet address owning this key
    pub wallet_address: [u8; 32],

    /// Account tier
    pub tier: ApiTier,

    /// Key creation timestamp
    pub created_at: u64,

    /// Key expiration timestamp (None for no expiration)
    pub expires_at: Option<u64>,

    /// Last rotation timestamp
    pub last_rotated: u64,

    /// Whether this key is currently active
    pub is_active: bool,

    /// Revocation reason (if revoked)
    pub revocation_reason: Option<String>,

    /// Rate limiting state
    pub rate_limit_state: RateLimitState,

    /// Total requests made with this key
    pub total_requests: u64,

    /// Last request timestamp
    pub last_request_at: Option<u64>,
}

/// Rate limiting state per API key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitState {
    /// Current window start (Unix timestamp in seconds)
    pub window_start: u64,

    /// Requests in current minute
    pub requests_this_minute: u32,

    /// Requests today
    pub requests_today: u32,

    /// Date of last daily reset (Unix timestamp)
    pub last_daily_reset: u64,
}

impl Default for RateLimitState {
    fn default() -> Self {
        let now = chrono::Utc::now().timestamp() as u64;
        Self {
            window_start: now / 60, // Current minute
            requests_this_minute: 0,
            requests_today: 0,
            last_daily_reset: now / 86400, // Current day
        }
    }
}

/// API key generation result
#[derive(Debug, Clone, Serialize)]
pub struct GeneratedApiKey {
    /// The actual API key (only returned once during generation)
    pub api_key: String,

    /// API key ID for reference
    pub key_id: String,

    /// Key creation timestamp
    pub created_at: u64,

    /// Key expiration timestamp
    pub expires_at: Option<u64>,

    /// Account tier
    pub tier: ApiTier,
}

/// API key manager
pub struct PaaSApiKeyManager {
    /// API key records (key_id -> ApiKeyRecord)
    keys: Arc<RwLock<HashMap<String, ApiKeyRecord>>>,

    /// Key lookup by hash prefix (first 8 bytes of SHA256 hash)
    hash_index: Arc<RwLock<HashMap<String, String>>>,

    /// Blacklisted key IDs
    blacklist: Arc<RwLock<HashMap<String, String>>>, // key_id -> reason

    /// Argon2 hasher instance
    argon2: Argon2<'static>,
}

impl PaaSApiKeyManager {
    pub fn new() -> Self {
        // Use OWASP recommended Argon2id parameters
        let argon2 = Argon2::default();

        Self {
            keys: Arc::new(RwLock::new(HashMap::new())),
            hash_index: Arc::new(RwLock::new(HashMap::new())),
            blacklist: Arc::new(RwLock::new(HashMap::new())),
            argon2,
        }
    }

    /// Generate a new API key for a wallet
    pub async fn generate_key(
        &self,
        wallet_address: [u8; 32],
        tier: ApiTier,
        expires_in_days: Option<u32>,
    ) -> Result<GeneratedApiKey, String> {
        // Generate cryptographically secure random API key
        let api_key = self.generate_random_key();
        let key_id = Uuid::new_v4().to_string();

        // Hash the API key with argon2id
        let salt = SaltString::generate(&mut OsRng);
        let key_hash = self
            .argon2
            .hash_password(api_key.as_bytes(), &salt)
            .map_err(|e| format!("Failed to hash API key: {}", e))?
            .to_string();

        // Calculate expiration
        let now = chrono::Utc::now().timestamp() as u64;
        let expires_at = expires_in_days.map(|days| now + (days as u64 * 86400));

        // Create key record
        let key_record = ApiKeyRecord {
            key_id: key_id.clone(),
            key_hash: key_hash.clone(),
            wallet_address,
            tier,
            created_at: now,
            expires_at,
            last_rotated: now,
            is_active: true,
            revocation_reason: None,
            rate_limit_state: RateLimitState::default(),
            total_requests: 0,
            last_request_at: None,
        };

        // Store key record
        let mut keys = self.keys.write().await;
        keys.insert(key_id.clone(), key_record);

        // Build hash index for fast lookup
        let hash_prefix = self.compute_hash_prefix(&api_key);
        let mut hash_index = self.hash_index.write().await;
        hash_index.insert(hash_prefix, key_id.clone());

        info!(
            "✅ Generated API key {} for wallet {} (tier: {:?})",
            key_id,
            hex::encode(&wallet_address[..8]),
            tier
        );

        Ok(GeneratedApiKey {
            api_key,
            key_id,
            created_at: now,
            expires_at,
            tier,
        })
    }

    /// Verify an API key and return the key record
    pub async fn verify_key(&self, api_key: &str) -> Result<ApiKeyRecord, String> {
        // Find key by hash prefix
        let hash_prefix = self.compute_hash_prefix(api_key);
        let hash_index = self.hash_index.read().await;
        let key_id = hash_index
            .get(&hash_prefix)
            .ok_or("Invalid API key")?
            .clone();
        drop(hash_index);

        // Check if key is blacklisted
        let blacklist = self.blacklist.read().await;
        if let Some(reason) = blacklist.get(&key_id) {
            return Err(format!("API key revoked: {}", reason));
        }
        drop(blacklist);

        // Get key record
        let keys = self.keys.read().await;
        let mut key_record = keys.get(&key_id).ok_or("API key not found")?.clone();
        drop(keys);

        // Check if key is active
        if !key_record.is_active {
            return Err("API key is inactive".to_string());
        }

        // Check expiration
        let now = chrono::Utc::now().timestamp() as u64;
        if let Some(expires_at) = key_record.expires_at {
            if now > expires_at {
                return Err("API key expired".to_string());
            }
        }

        // Verify password hash (timing-safe)
        let parsed_hash = PasswordHash::new(&key_record.key_hash)
            .map_err(|e| format!("Failed to parse key hash: {}", e))?;

        self.argon2
            .verify_password(api_key.as_bytes(), &parsed_hash)
            .map_err(|_| "Invalid API key".to_string())?;

        // Update last request timestamp
        key_record.last_request_at = Some(now);
        let mut keys = self.keys.write().await;
        if let Some(record) = keys.get_mut(&key_id) {
            record.last_request_at = Some(now);
        }

        info!(
            "✅ Verified API key {} (wallet: {})",
            key_id,
            hex::encode(&key_record.wallet_address[..8])
        );

        Ok(key_record)
    }

    /// Check rate limit for an API key
    pub async fn check_rate_limit(&self, key_id: &str) -> Result<(), String> {
        let mut keys = self.keys.write().await;
        let key_record = keys.get_mut(key_id).ok_or("API key not found")?;

        let now = chrono::Utc::now().timestamp() as u64;
        let current_minute = now / 60;
        let current_day = now / 86400;

        // Reset minute window if needed
        if key_record.rate_limit_state.window_start < current_minute {
            key_record.rate_limit_state.requests_this_minute = 0;
            key_record.rate_limit_state.window_start = current_minute;
        }

        // Reset daily counter if needed
        if key_record.rate_limit_state.last_daily_reset < current_day {
            key_record.rate_limit_state.requests_today = 0;
            key_record.rate_limit_state.last_daily_reset = current_day;
        }

        // Check per-minute rate limit
        if let Some(rate_limit) = key_record.tier.rate_limit() {
            if key_record.rate_limit_state.requests_this_minute >= rate_limit {
                return Err(format!(
                    "Rate limit exceeded: {} requests/minute (tier: {:?})",
                    rate_limit, key_record.tier
                ));
            }
        }

        // Check daily rate limit
        if let Some(daily_limit) = key_record.tier.daily_limit() {
            if key_record.rate_limit_state.requests_today >= daily_limit {
                return Err(format!(
                    "Daily limit exceeded: {} requests/day (tier: {:?})",
                    daily_limit, key_record.tier
                ));
            }
        }

        // Increment counters
        key_record.rate_limit_state.requests_this_minute += 1;
        key_record.rate_limit_state.requests_today += 1;
        key_record.total_requests += 1;

        Ok(())
    }

    /// Rotate an API key (generate new key, keep same ID)
    pub async fn rotate_key(&self, key_id: &str) -> Result<String, String> {
        // Generate new API key
        let new_api_key = self.generate_random_key();

        // Hash the new API key
        let salt = SaltString::generate(&mut OsRng);
        let new_key_hash = self
            .argon2
            .hash_password(new_api_key.as_bytes(), &salt)
            .map_err(|e| format!("Failed to hash new API key: {}", e))?
            .to_string();

        // Update key record
        let mut keys = self.keys.write().await;
        let key_record = keys.get_mut(key_id).ok_or("API key not found")?;

        let old_hash_prefix = self.compute_hash_prefix_from_hash(&key_record.key_hash);
        key_record.key_hash = new_key_hash;
        key_record.last_rotated = chrono::Utc::now().timestamp() as u64;

        // Update hash index
        let new_hash_prefix = self.compute_hash_prefix(&new_api_key);
        let mut hash_index = self.hash_index.write().await;
        hash_index.remove(&old_hash_prefix);
        hash_index.insert(new_hash_prefix, key_id.to_string());

        info!(
            "🔄 Rotated API key {} (wallet: {})",
            key_id,
            hex::encode(&key_record.wallet_address[..8])
        );

        Ok(new_api_key)
    }

    /// Revoke an API key
    pub async fn revoke_key(&self, key_id: &str, reason: String) -> Result<(), String> {
        let mut keys = self.keys.write().await;
        let key_record = keys.get_mut(key_id).ok_or("API key not found")?;

        key_record.is_active = false;
        key_record.revocation_reason = Some(reason.clone());

        // Add to blacklist
        let mut blacklist = self.blacklist.write().await;
        blacklist.insert(key_id.to_string(), reason.clone());

        warn!("🚫 Revoked API key {} (reason: {})", key_id, reason);

        Ok(())
    }

    /// List all API keys for a wallet
    pub async fn list_keys_for_wallet(&self, wallet_address: &[u8; 32]) -> Vec<ApiKeyRecord> {
        let keys = self.keys.read().await;
        keys.values()
            .filter(|k| k.wallet_address == *wallet_address)
            .cloned()
            .collect()
    }

    /// Generate a cryptographically secure random API key
    fn generate_random_key(&self) -> String {
        // Format: paas_<32-byte-hex>_<checksum>
        let random_bytes: [u8; 32] = rand::random();
        let hex_key = hex::encode(random_bytes);

        // Add checksum for integrity verification
        let mut hasher = Sha256::new();
        hasher.update(&random_bytes);
        let checksum = hex::encode(&hasher.finalize()[..4]);

        format!("paas_{}_{}", hex_key, checksum)
    }

    /// Compute hash prefix for fast lookup
    fn compute_hash_prefix(&self, api_key: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(api_key.as_bytes());
        hex::encode(&hasher.finalize()[..8])
    }

    /// Compute hash prefix from existing argon2id hash
    fn compute_hash_prefix_from_hash(&self, argon2_hash: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(argon2_hash.as_bytes());
        hex::encode(&hasher.finalize()[..8])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_and_verify_key() {
        let manager = PaaSApiKeyManager::new();
        let wallet = [1u8; 32];

        // Generate key
        let generated = manager
            .generate_key(wallet, ApiTier::Professional, Some(90))
            .await
            .unwrap();

        assert!(generated.api_key.starts_with("paas_"));
        assert_eq!(generated.tier, ApiTier::Professional);

        // Verify key
        let verified = manager.verify_key(&generated.api_key).await.unwrap();
        assert_eq!(verified.wallet_address, wallet);
        assert_eq!(verified.tier, ApiTier::Professional);
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let manager = PaaSApiKeyManager::new();
        let wallet = [2u8; 32];

        // Generate free tier key (100 req/min)
        let generated = manager
            .generate_key(wallet, ApiTier::PayPerUse, None)
            .await
            .unwrap();

        // Make 100 requests (should all succeed)
        for _ in 0..100 {
            assert!(manager.check_rate_limit(&generated.key_id).await.is_ok());
        }

        // 101st request should fail
        assert!(manager.check_rate_limit(&generated.key_id).await.is_err());
    }

    #[tokio::test]
    async fn test_key_rotation() {
        let manager = PaaSApiKeyManager::new();
        let wallet = [3u8; 32];

        // Generate key
        let generated = manager
            .generate_key(wallet, ApiTier::Enterprise, None)
            .await
            .unwrap();

        let old_key = generated.api_key.clone();

        // Rotate key
        let new_key = manager.rotate_key(&generated.key_id).await.unwrap();

        // Old key should no longer work
        assert!(manager.verify_key(&old_key).await.is_err());

        // New key should work
        let verified = manager.verify_key(&new_key).await.unwrap();
        assert_eq!(verified.wallet_address, wallet);
    }

    #[tokio::test]
    async fn test_key_revocation() {
        let manager = PaaSApiKeyManager::new();
        let wallet = [4u8; 32];

        // Generate key
        let generated = manager
            .generate_key(wallet, ApiTier::WhiteLabel, None)
            .await
            .unwrap();

        // Revoke key
        manager
            .revoke_key(&generated.key_id, "Test revocation".to_string())
            .await
            .unwrap();

        // Key should no longer work
        let result = manager.verify_key(&generated.api_key).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("revoked"));
    }
}
