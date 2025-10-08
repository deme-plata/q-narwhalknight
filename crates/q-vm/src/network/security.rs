/// Security Module for VM Network Bridge
///
/// Implements cryptographic authentication, authorization, rate limiting,
/// and bytecode validation to prevent attacks on the VM network.

use anyhow::Result;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::vm::VmError;

/// Maximum message size (1MB)
pub const MAX_MESSAGE_SIZE: usize = 1_048_576;

/// Maximum bytecode size (24KB - Ethereum limit)
pub const MAX_BYTECODE_SIZE: usize = 24_576;

/// Maximum contract call arguments size (1MB)
pub const MAX_ARGS_SIZE: usize = 1_000_000;

/// Message timestamp tolerance (30 seconds)
pub const TIMESTAMP_TOLERANCE_SECS: u64 = 30;

/// Request cleanup interval (60 seconds)
pub const CLEANUP_INTERVAL_SECS: u64 = 60;

/// Cryptographically signed message wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedVmMessage<T> {
    /// The actual message payload
    pub message: T,

    /// Ed25519 signature (64 bytes)
    #[serde(with = "serde_big_array::BigArray")]
    pub signature: [u8; 64],

    /// Public key of signer (32 bytes)
    pub public_key: [u8; 32],

    /// Unix timestamp (for replay protection)
    pub timestamp: u64,

    /// Unique nonce (for replay protection)
    pub nonce: u64,
}

impl<T: Serialize> SignedVmMessage<T> {
    /// Create and sign a new message
    pub fn sign(message: T, signing_key: &SigningKey) -> Result<Self, VmError> {
        let nonce = rand::random::<u64>();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Serialize message + timestamp + nonce for signing
        let msg_bytes = bincode::serialize(&(&message, timestamp, nonce))
            .map_err(|e| VmError::SerializationError(e.to_string()))?;

        let signature = signing_key.sign(&msg_bytes);
        let public_key = signing_key.verifying_key().to_bytes();

        Ok(Self {
            message,
            signature: signature.to_bytes(),
            public_key,
            timestamp,
            nonce,
        })
    }

    /// Verify message signature and freshness
    pub fn verify(&self) -> Result<(), VmError> {
        // 1. Check timestamp (reject if too old or in future)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let age = now.saturating_sub(self.timestamp);
        if age > TIMESTAMP_TOLERANCE_SECS {
            return Err(VmError::InvalidTransaction(
                format!("Stale message: {}s old", age)
            ));
        }

        // Reject future timestamps (clock skew attack)
        if self.timestamp > now + 5 {
            return Err(VmError::InvalidTransaction(
                "Message timestamp in future".into()
            ));
        }

        // 2. Verify signature
        let verifying_key = VerifyingKey::from_bytes(&self.public_key)
            .map_err(|e| VmError::InvalidTransaction(format!("Invalid public key: {}", e)))?;

        let signature = Signature::from_bytes(&self.signature);

        let msg_bytes = bincode::serialize(&(&self.message, self.timestamp, self.nonce))
            .map_err(|e| VmError::SerializationError(e.to_string()))?;

        verifying_key.verify(&msg_bytes, &signature)
            .map_err(|_| VmError::InvalidTransaction("Invalid signature".into()))?;

        Ok(())
    }

    /// Get the public key as a verifying key
    pub fn get_public_key(&self) -> Result<VerifyingKey, VmError> {
        VerifyingKey::from_bytes(&self.public_key)
            .map_err(|e| VmError::InvalidTransaction(format!("Invalid public key: {}", e)))
    }
}

/// Rate limiter for peers using token bucket algorithm
pub struct PeerRateLimiter {
    /// Rate limit: requests per second per peer
    requests_per_second: u32,

    /// Track last request time and token count per peer
    peer_states: Arc<RwLock<HashMap<[u8; 32], (Instant, u32)>>>,
}

impl PeerRateLimiter {
    pub fn new(requests_per_second: u32) -> Self {
        Self {
            requests_per_second,
            peer_states: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check if peer is allowed to make a request (token bucket)
    pub async fn check_rate_limit(&self, peer_pubkey: &[u8; 32]) -> Result<(), VmError> {
        let mut states = self.peer_states.write().await;
        let now = Instant::now();

        let (last_time, tokens) = states.entry(*peer_pubkey).or_insert((now, self.requests_per_second));

        // Refill tokens based on time elapsed
        let elapsed = now.duration_since(*last_time).as_secs_f64();
        let new_tokens = (*tokens as f64 + elapsed * self.requests_per_second as f64)
            .min(self.requests_per_second as f64) as u32;

        if new_tokens > 0 {
            // Allow request, consume token
            *states.get_mut(peer_pubkey).unwrap() = (now, new_tokens - 1);
            Ok(())
        } else {
            Err(VmError::ExecutionError("Rate limit exceeded".into()))
        }
    }

    /// Cleanup old peer states (prevent memory leak)
    pub async fn cleanup_old_states(&self) {
        let mut states = self.peer_states.write().await;
        let now = Instant::now();
        let cutoff = Duration::from_secs(300); // 5 minutes

        states.retain(|_, (last_time, _)| now.duration_since(*last_time) < cutoff);
    }
}

/// Global resource quota manager
pub struct ResourceQuotaManager {
    /// Total available gas units
    total_gas_pool: Arc<tokio::sync::Semaphore>,

    /// Maximum gas per request
    max_gas_per_request: u64,

    /// Track gas usage per peer
    peer_gas_usage: Arc<RwLock<HashMap<[u8; 32], u64>>>,
}

impl ResourceQuotaManager {
    pub fn new(total_gas: u64, max_gas_per_request: u64) -> Self {
        Self {
            total_gas_pool: Arc::new(tokio::sync::Semaphore::new(total_gas as usize)),
            max_gas_per_request,
            peer_gas_usage: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Try to acquire gas quota for execution
    pub async fn acquire_gas(&self, gas_amount: u64, peer_pubkey: &[u8; 32]) -> Result<GasQuotaPermit, VmError> {
        // Check per-request limit
        if gas_amount > self.max_gas_per_request {
            return Err(VmError::InvalidTransaction(
                format!("Gas limit {} exceeds maximum {}", gas_amount, self.max_gas_per_request)
            ));
        }

        // Try to acquire from global pool with timeout
        let permit = tokio::time::timeout(
            Duration::from_millis(100),
            self.total_gas_pool.clone().acquire_many_owned(gas_amount as u32)
        )
        .await
        .map_err(|_| VmError::ExecutionError("Gas pool timeout".into()))?
        .map_err(|_| VmError::OutOfGas)?;

        // Track usage per peer
        {
            let mut usage = self.peer_gas_usage.write().await;
            *usage.entry(*peer_pubkey).or_insert(0) += gas_amount;
        }

        Ok(GasQuotaPermit {
            _permit: permit,
            gas_amount,
            peer_pubkey: *peer_pubkey,
            usage_tracker: self.peer_gas_usage.clone(),
        })
    }

    /// Get gas usage statistics
    pub async fn get_stats(&self) -> ResourceQuotaStats {
        let usage = self.peer_gas_usage.read().await;
        let total_used: u64 = usage.values().sum();

        ResourceQuotaStats {
            total_gas_used: total_used,
            peers_active: usage.len(),
            available_permits: self.total_gas_pool.available_permits() as u64,
        }
    }
}

/// RAII guard for gas quota - automatically releases on drop
pub struct GasQuotaPermit {
    _permit: tokio::sync::OwnedSemaphorePermit,
    gas_amount: u64,
    peer_pubkey: [u8; 32],
    usage_tracker: Arc<RwLock<HashMap<[u8; 32], u64>>>,
}

impl Drop for GasQuotaPermit {
    fn drop(&mut self) {
        // Update usage tracking (permit is auto-released)
        let tracker = self.usage_tracker.clone();
        let gas = self.gas_amount;
        let peer = self.peer_pubkey;

        tokio::spawn(async move {
            let mut usage = tracker.write().await;
            if let Some(current) = usage.get_mut(&peer) {
                *current = current.saturating_sub(gas);
            }
        });
    }
}

#[derive(Debug, Clone)]
pub struct ResourceQuotaStats {
    pub total_gas_used: u64,
    pub peers_active: usize,
    pub available_permits: u64,
}

/// Bytecode validator and static analyzer
pub struct BytecodeValidator {
    /// Maximum bytecode size
    max_size: usize,

    /// Blacklisted WASM operations
    blacklisted_ops: HashSet<String>,
}

impl BytecodeValidator {
    pub fn new() -> Self {
        let mut blacklisted_ops = HashSet::new();
        // Add dangerous operations
        blacklisted_ops.insert("call_indirect".to_string());
        // Note: More ops can be blacklisted based on security policy

        Self {
            max_size: MAX_BYTECODE_SIZE,
            blacklisted_ops,
        }
    }

    /// Validate bytecode before deployment
    pub fn validate(&self, bytecode: &[u8]) -> Result<(), VmError> {
        // 1. Size check
        if bytecode.len() > self.max_size {
            return Err(VmError::InvalidTransaction(
                format!("Bytecode too large: {} > {}", bytecode.len(), self.max_size)
            ));
        }

        // 2. WASM format validation
        wasmparser::validate(bytecode)
            .map_err(|e| VmError::CompilationError(format!("Invalid WASM: {}", e)))?;

        // 3. Static analysis for dangerous operations
        self.analyze_safety(bytecode)?;

        Ok(())
    }

    /// Analyze bytecode for dangerous operations
    fn analyze_safety(&self, bytecode: &[u8]) -> Result<(), VmError> {
        use wasmparser::{Payload, Operator};

        for payload in wasmparser::Parser::new(0).parse_all(bytecode) {
            match payload.map_err(|e| VmError::CompilationError(e.to_string()))? {
                Payload::CodeSectionEntry(body) => {
                    let mut reader = body.get_operators_reader()
                        .map_err(|e| VmError::CompilationError(e.to_string()))?;

                    while !reader.eof() {
                        let op = reader.read()
                            .map_err(|e| VmError::CompilationError(e.to_string()))?;

                        // Check for blacklisted operations
                        match op {
                            Operator::CallIndirect { .. } => {
                                if self.blacklisted_ops.contains("call_indirect") {
                                    return Err(VmError::CompilationError(
                                        "Indirect calls not allowed (security policy)".into()
                                    ));
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Authorization checker for contract access
pub struct AccessController {
    /// Authorized peers (whitelist)
    authorized_peers: Arc<RwLock<HashSet<[u8; 32]>>>,

    /// Banned peers (blacklist)
    banned_peers: Arc<RwLock<HashSet<[u8; 32]>>>,

    /// Contract-specific permissions
    contract_permissions: Arc<RwLock<HashMap<String, HashSet<[u8; 32]>>>>,
}

impl AccessController {
    pub fn new() -> Self {
        Self {
            authorized_peers: Arc::new(RwLock::new(HashSet::new())),
            banned_peers: Arc::new(RwLock::new(HashSet::new())),
            contract_permissions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check if peer is authorized for general access
    pub async fn is_peer_authorized(&self, peer_pubkey: &[u8; 32]) -> bool {
        let banned = self.banned_peers.read().await;
        if banned.contains(peer_pubkey) {
            return false;
        }

        // If whitelist is empty, allow all non-banned peers
        let authorized = self.authorized_peers.read().await;
        if authorized.is_empty() {
            return true;
        }

        authorized.contains(peer_pubkey)
    }

    /// Check if peer can access specific contract
    pub async fn is_authorized_for_contract(
        &self,
        peer_pubkey: &[u8; 32],
        contract_address: &str,
    ) -> bool {
        // Check general authorization first
        if !self.is_peer_authorized(peer_pubkey).await {
            return false;
        }

        // Check contract-specific permissions
        let permissions = self.contract_permissions.read().await;
        if let Some(allowed_peers) = permissions.get(contract_address) {
            allowed_peers.contains(peer_pubkey)
        } else {
            // No specific permissions = allow all authorized peers
            true
        }
    }

    /// Ban a peer
    pub async fn ban_peer(&self, peer_pubkey: [u8; 32]) {
        warn!("Banning peer: {}", hex::encode(peer_pubkey));
        let mut banned = self.banned_peers.write().await;
        banned.insert(peer_pubkey);
    }

    /// Authorize a peer
    pub async fn authorize_peer(&self, peer_pubkey: [u8; 32]) {
        info!("Authorizing peer: {}", hex::encode(peer_pubkey));
        let mut authorized = self.authorized_peers.write().await;
        authorized.insert(peer_pubkey);
    }

    /// Grant contract-specific permission
    pub async fn grant_contract_permission(&self, contract_address: String, peer_pubkey: [u8; 32]) {
        let mut permissions = self.contract_permissions.write().await;
        permissions.entry(contract_address).or_insert_with(HashSet::new).insert(peer_pubkey);
    }
}

/// Nonce tracker for replay attack prevention
pub struct NonceTracker {
    /// Track used nonces per peer
    used_nonces: Arc<RwLock<HashMap<[u8; 32], HashSet<u64>>>>,
}

impl NonceTracker {
    pub fn new() -> Self {
        Self {
            used_nonces: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check and mark nonce as used
    pub async fn check_and_mark_nonce(&self, peer_pubkey: &[u8; 32], nonce: u64) -> Result<(), VmError> {
        let mut nonces = self.used_nonces.write().await;
        let peer_nonces = nonces.entry(*peer_pubkey).or_insert_with(HashSet::new);

        if peer_nonces.contains(&nonce) {
            return Err(VmError::InvalidTransaction("Nonce already used (replay attack?)".into()));
        }

        peer_nonces.insert(nonce);
        Ok(())
    }

    /// Cleanup old nonces (prevent memory leak)
    pub async fn cleanup(&self) {
        let mut nonces = self.used_nonces.write().await;

        // Keep only last 10000 nonces per peer
        for (_peer, peer_nonces) in nonces.iter_mut() {
            if peer_nonces.len() > 10000 {
                // Remove oldest half
                let to_remove: Vec<u64> = peer_nonces.iter()
                    .copied()
                    .take(5000)
                    .collect();

                for nonce in to_remove {
                    peer_nonces.remove(&nonce);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_signing_and_verification() {
        let signing_key = SigningKey::generate(&mut rand::thread_rng());

        let message = "test message";
        let signed = SignedVmMessage::sign(message, &signing_key).unwrap();

        assert!(signed.verify().is_ok());
    }

    #[test]
    fn test_signature_tampering_detection() {
        let signing_key = SigningKey::generate(&mut rand::thread_rng());

        let message = "original message";
        let mut signed = SignedVmMessage::sign(message, &signing_key).unwrap();

        // Tamper with signature
        signed.signature[0] ^= 1;

        assert!(signed.verify().is_err());
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let limiter = PeerRateLimiter::new(2); // 2 req/sec
        let peer = [1u8; 32];

        // First two requests should succeed
        assert!(limiter.check_rate_limit(&peer).await.is_ok());
        assert!(limiter.check_rate_limit(&peer).await.is_ok());

        // Third should fail
        assert!(limiter.check_rate_limit(&peer).await.is_err());
    }

    #[tokio::test]
    async fn test_gas_quota() {
        let quota = ResourceQuotaManager::new(1000, 500);
        let peer = [1u8; 32];

        // Acquire within limit
        let permit1 = quota.acquire_gas(400, &peer).await;
        assert!(permit1.is_ok());

        // Exceed per-request limit
        let permit2 = quota.acquire_gas(600, &peer).await;
        assert!(permit2.is_err());
    }

    #[test]
    fn test_bytecode_validation_size_limit() {
        let validator = BytecodeValidator::new();

        // Too large
        let large_bytecode = vec![0u8; MAX_BYTECODE_SIZE + 1];
        assert!(validator.validate(&large_bytecode).is_err());

        // Within limit (but invalid WASM)
        let small_bytecode = vec![0u8; 100];
        assert!(validator.validate(&small_bytecode).is_err()); // Invalid WASM format
    }

    #[tokio::test]
    async fn test_access_control() {
        let controller = AccessController::new();
        let peer = [1u8; 32];

        // Initially allowed (no whitelist)
        assert!(controller.is_peer_authorized(&peer).await);

        // Ban peer
        controller.ban_peer(peer).await;
        assert!(!controller.is_peer_authorized(&peer).await);
    }

    #[tokio::test]
    async fn test_nonce_replay_protection() {
        let tracker = NonceTracker::new();
        let peer = [1u8; 32];
        let nonce = 12345;

        // First use should succeed
        assert!(tracker.check_and_mark_nonce(&peer, nonce).await.is_ok());

        // Replay should fail
        assert!(tracker.check_and_mark_nonce(&peer, nonce).await.is_err());
    }
}
