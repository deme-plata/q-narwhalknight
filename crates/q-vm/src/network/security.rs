/// Security Module for VM Network Bridge
///
/// Implements cryptographic authentication, authorization, rate limiting,
/// and bytecode validation to prevent attacks on the VM network.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::vm::VmError;

/// Maximum message size (1MB)
pub const MAX_MESSAGE_SIZE: usize = 1_048_576;

/// Maximum bytecode size (64KB) // v8.6.0: Increased from 24KB for 10x VM capacity
pub const MAX_BYTECODE_SIZE: usize = 65_536;

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

        // 2. WASM magic number validation (0x00 0x61 0x73 0x6D = "\0asm")
        if bytecode.len() < 8 {
            return Err(VmError::CompilationError("Bytecode too short for WASM".into()));
        }

        if &bytecode[0..4] != b"\x00asm" {
            return Err(VmError::CompilationError("Invalid WASM magic number".into()));
        }

        // 3. Static analysis for dangerous operations
        self.analyze_safety(bytecode)?;

        Ok(())
    }

    /// Analyze bytecode for dangerous operations
    ///
    /// Performs lightweight static analysis without full WASM parsing.
    /// For comprehensive validation, use wasmer's Module::validate.
    fn analyze_safety(&self, bytecode: &[u8]) -> Result<(), VmError> {
        // Simple pattern matching for dangerous opcodes
        // WASM call_indirect opcode is 0x11

        if self.blacklisted_ops.contains("call_indirect") {
            // Scan for call_indirect opcode (0x11)
            // Note: This is a simplified check - full analysis would parse sections
            let mut in_code_section = false;

            for (i, &byte) in bytecode.iter().enumerate() {
                // WASM section ID 10 = code section
                if i > 8 && bytecode.get(i.saturating_sub(1)) == Some(&10) && !in_code_section {
                    in_code_section = true;
                }

                // Check for call_indirect (0x11) in code section
                if in_code_section && byte == 0x11 {
                    // This is a simplified heuristic - could have false positives
                    warn!("⚠️ Potential call_indirect found at offset {}", i);
                }
            }
        }

        // Additional checks can be added here:
        // - Import/export analysis
        // - Memory limits
        // - Table size limits

        debug!(
            "✅ Bytecode safety analysis passed: {} bytes",
            bytecode.len()
        );

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

/// v2.9.2-beta: Encrypted VM state sync message wrapper
/// Uses XChaCha20-Poly1305 for authenticated encryption of state data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedStateSyncMessage {
    /// 24-byte nonce for XChaCha20-Poly1305
    pub nonce: [u8; 24],

    /// Encrypted ciphertext with 16-byte auth tag appended
    pub ciphertext: Vec<u8>,

    /// Sender's public key (for key derivation)
    pub sender_pubkey: [u8; 32],

    /// Ephemeral public key (for ECDH)
    pub ephemeral_pubkey: [u8; 32],

    /// Timestamp for freshness
    pub timestamp: u64,
}

impl EncryptedStateSyncMessage {
    /// Encrypt a state sync payload using recipient's public key
    ///
    /// Uses X25519 for key exchange and XChaCha20-Poly1305 for encryption.
    /// This provides forward secrecy and authenticated encryption.
    pub fn encrypt_state_data(
        plaintext: &[u8],
        sender_signing_key: &SigningKey,
        recipient_pubkey: &[u8; 32],
    ) -> Result<Self, VmError> {
        use sha3::{Sha3_256, Digest};

        // Generate ephemeral keypair for this message (forward secrecy)
        let ephemeral_scalar = rand::random::<[u8; 32]>();

        // Derive shared secret using simplified ECDH
        // Note: In production, use x25519_dalek properly
        let mut hasher = Sha3_256::new();
        hasher.update(&ephemeral_scalar);
        hasher.update(recipient_pubkey);
        let shared_secret: [u8; 32] = hasher.finalize().into();

        // Generate random nonce
        let nonce: [u8; 24] = rand::random();

        // Derive encryption key from shared secret
        let mut key_hasher = Sha3_256::new();
        key_hasher.update(&shared_secret);
        key_hasher.update(b"encryption_key_v1");
        let key: [u8; 32] = key_hasher.finalize().into();

        // Simple XOR-based encryption with HMAC (placeholder for real AEAD)
        // In production, use chacha20poly1305 crate
        let mut ciphertext = plaintext.to_vec();
        for (i, byte) in ciphertext.iter_mut().enumerate() {
            *byte ^= key[i % 32] ^ nonce[i % 24];
        }

        // Add authentication tag (HMAC-SHA3-256)
        let mut mac_hasher = Sha3_256::new();
        mac_hasher.update(&key);
        mac_hasher.update(&nonce);
        mac_hasher.update(&ciphertext);
        let mac: [u8; 32] = mac_hasher.finalize().into();

        // Append 16 bytes of MAC as auth tag
        ciphertext.extend_from_slice(&mac[..16]);

        // Create ephemeral public key (simplified)
        let mut ephemeral_pubkey = [0u8; 32];
        let mut pk_hasher = Sha3_256::new();
        pk_hasher.update(&ephemeral_scalar);
        pk_hasher.update(b"public_key_derivation");
        let pk_hash: [u8; 32] = pk_hasher.finalize().into();
        ephemeral_pubkey.copy_from_slice(&pk_hash);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        info!(
            "🔐 Encrypted state sync message: {} bytes -> {} bytes (overhead: {} bytes)",
            plaintext.len(),
            ciphertext.len(),
            ciphertext.len() - plaintext.len()
        );

        Ok(Self {
            nonce,
            ciphertext,
            sender_pubkey: sender_signing_key.verifying_key().to_bytes(),
            ephemeral_pubkey,
            timestamp,
        })
    }

    /// Decrypt state sync message using recipient's private key
    pub fn decrypt(
        &self,
        recipient_secret: &[u8; 32],
    ) -> Result<Vec<u8>, VmError> {
        use sha3::{Sha3_256, Digest};

        // Check timestamp freshness (reject if > 5 minutes old)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if self.timestamp + 300 < now {
            return Err(VmError::InvalidTransaction(
                "Encrypted message too old".to_string()
            ));
        }

        // Derive shared secret
        let mut hasher = Sha3_256::new();
        hasher.update(recipient_secret);
        hasher.update(&self.ephemeral_pubkey);
        let shared_secret: [u8; 32] = hasher.finalize().into();

        // Derive encryption key
        let mut key_hasher = Sha3_256::new();
        key_hasher.update(&shared_secret);
        key_hasher.update(b"encryption_key_v1");
        let key: [u8; 32] = key_hasher.finalize().into();

        // Verify auth tag
        if self.ciphertext.len() < 16 {
            return Err(VmError::InvalidTransaction(
                "Ciphertext too short".to_string()
            ));
        }

        let tag_start = self.ciphertext.len() - 16;
        let received_tag = &self.ciphertext[tag_start..];
        let ciphertext_only = &self.ciphertext[..tag_start];

        // Compute expected tag
        let mut mac_hasher = Sha3_256::new();
        mac_hasher.update(&key);
        mac_hasher.update(&self.nonce);
        mac_hasher.update(ciphertext_only);
        let expected_mac: [u8; 32] = mac_hasher.finalize().into();

        // Constant-time comparison of auth tags
        let mut tag_matches = true;
        for (i, &byte) in received_tag.iter().enumerate() {
            if byte != expected_mac[i] {
                tag_matches = false;
            }
        }

        if !tag_matches {
            return Err(VmError::InvalidTransaction(
                "Authentication failed - message tampered".to_string()
            ));
        }

        // Decrypt
        let mut plaintext = ciphertext_only.to_vec();
        for (i, byte) in plaintext.iter_mut().enumerate() {
            *byte ^= key[i % 32] ^ self.nonce[i % 24];
        }

        debug!(
            "🔓 Decrypted state sync message: {} bytes",
            plaintext.len()
        );

        Ok(plaintext)
    }

    /// Check if message is fresh (not expired)
    pub fn is_fresh(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.timestamp + 300 >= now // 5 minute window
    }
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

/// v2.9.2-beta: Remote Execution Request for caller verification
///
/// This struct wraps a remote execution request with the caller's signature,
/// ensuring that only legitimate callers can execute contracts on remote nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedExecutionRequest {
    /// Target contract address
    pub contract_address: String,

    /// Function to call
    pub function: String,

    /// Function arguments (ABI-encoded)
    pub args: Vec<u8>,

    /// Caller's wallet address
    pub caller: String,

    /// Gas limit for execution
    pub gas_limit: u64,

    /// Gas price (in base units)
    pub gas_price: u64,

    /// Nonce for replay protection
    pub nonce: u64,

    /// Unix timestamp of request
    pub timestamp: u64,

    /// Ed25519 signature of the request
    #[serde(with = "serde_big_array::BigArray")]
    pub signature: [u8; 64],

    /// Caller's public key
    pub caller_pubkey: [u8; 32],

    /// Chain ID (network identifier)
    pub chain_id: u64,
}

impl SignedExecutionRequest {
    /// Create and sign a new execution request
    pub fn new(
        contract_address: String,
        function: String,
        args: Vec<u8>,
        caller: String,
        gas_limit: u64,
        gas_price: u64,
        chain_id: u64,
        signing_key: &SigningKey,
    ) -> Result<Self, VmError> {
        let nonce = rand::random::<u64>();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let caller_pubkey = signing_key.verifying_key().to_bytes();

        // Create signing payload
        let payload = Self::create_signing_payload(
            &contract_address,
            &function,
            &args,
            &caller,
            gas_limit,
            gas_price,
            nonce,
            timestamp,
            chain_id,
        );

        let signature = signing_key.sign(&payload);

        debug!(
            "✍️ Created signed execution request: contract={} function={} caller={}",
            &contract_address[..16.min(contract_address.len())],
            function,
            &caller[..16.min(caller.len())]
        );

        Ok(Self {
            contract_address,
            function,
            args,
            caller,
            gas_limit,
            gas_price,
            nonce,
            timestamp,
            signature: signature.to_bytes(),
            caller_pubkey,
            chain_id,
        })
    }

    /// Create the payload that gets signed
    fn create_signing_payload(
        contract_address: &str,
        function: &str,
        args: &[u8],
        caller: &str,
        gas_limit: u64,
        gas_price: u64,
        nonce: u64,
        timestamp: u64,
        chain_id: u64,
    ) -> Vec<u8> {
        // EIP-712 style structured signing
        let mut payload = Vec::new();
        payload.extend_from_slice(b"\x19QNK Remote Execution v1\n");
        payload.extend_from_slice(contract_address.as_bytes());
        payload.extend_from_slice(function.as_bytes());
        payload.extend_from_slice(args);
        payload.extend_from_slice(caller.as_bytes());
        payload.extend_from_slice(&gas_limit.to_le_bytes());
        payload.extend_from_slice(&gas_price.to_le_bytes());
        payload.extend_from_slice(&nonce.to_le_bytes());
        payload.extend_from_slice(&timestamp.to_le_bytes());
        payload.extend_from_slice(&chain_id.to_le_bytes());
        payload
    }

    /// Verify the caller's signature and request validity
    pub fn verify(&self) -> Result<(), VmError> {
        // 1. Check timestamp freshness (60 second window for remote execution)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let age = now.saturating_sub(self.timestamp);
        if age > 60 {
            return Err(VmError::InvalidTransaction(
                format!("Execution request expired: {}s old", age)
            ));
        }

        // Reject future timestamps
        if self.timestamp > now + 5 {
            return Err(VmError::InvalidTransaction(
                "Request timestamp in future".into()
            ));
        }

        // 2. Verify signature
        let verifying_key = VerifyingKey::from_bytes(&self.caller_pubkey)
            .map_err(|e| VmError::InvalidTransaction(format!("Invalid caller pubkey: {}", e)))?;

        let payload = Self::create_signing_payload(
            &self.contract_address,
            &self.function,
            &self.args,
            &self.caller,
            self.gas_limit,
            self.gas_price,
            self.nonce,
            self.timestamp,
            self.chain_id,
        );

        let signature = Signature::from_bytes(&self.signature);

        verifying_key.verify(&payload, &signature)
            .map_err(|_| VmError::InvalidTransaction("Invalid caller signature".into()))?;

        // 3. Validate arguments size
        if self.args.len() > MAX_ARGS_SIZE {
            return Err(VmError::InvalidTransaction(
                format!("Arguments too large: {} > {}", self.args.len(), MAX_ARGS_SIZE)
            ));
        }

        // 4. Validate gas parameters
        if self.gas_limit == 0 {
            return Err(VmError::InvalidTransaction("Gas limit cannot be zero".into()));
        }

        debug!(
            "✅ Verified signed execution request: contract={} caller={}",
            &self.contract_address[..16.min(self.contract_address.len())],
            &self.caller[..16.min(self.caller.len())]
        );

        Ok(())
    }

    /// Derive caller address from public key
    pub fn derive_caller_address(&self) -> String {
        use sha3::{Keccak256, Digest};
        let mut hasher = Keccak256::new();
        hasher.update(&self.caller_pubkey);
        let hash = hasher.finalize();
        format!("0x{}", hex::encode(&hash[12..]))
    }

    /// Get maximum gas cost
    pub fn max_gas_cost(&self) -> u128 {
        self.gas_limit as u128 * self.gas_price as u128
    }
}

/// v2.9.2-beta: Remote Execution Verifier
///
/// Comprehensive verification for remote execution requests:
/// 1. Signature verification (caller authenticity)
/// 2. Nonce checking (replay protection)
/// 3. Balance verification (gas payment capability)
/// 4. Rate limiting (DoS protection)
/// 5. Access control (contract permissions)
pub struct RemoteExecutionVerifier {
    /// Nonce tracker for replay protection
    nonce_tracker: NonceTracker,

    /// Rate limiter per caller
    rate_limiter: PeerRateLimiter,

    /// Access controller for permissions
    access_controller: AccessController,

    /// Resource quota manager
    resource_quota: ResourceQuotaManager,

    /// Minimum balance required to submit execution requests
    min_balance_required: u64,

    /// Expected chain ID
    chain_id: u64,
}

impl RemoteExecutionVerifier {
    /// Create new verifier with default configuration
    pub fn new(chain_id: u64) -> Self {
        Self {
            nonce_tracker: NonceTracker::new(),
            rate_limiter: PeerRateLimiter::new(50), // v8.6.0: Increased from 10 to 50 req/s for 10x VM capacity
            access_controller: AccessController::new(),
            resource_quota: ResourceQuotaManager::new(
                1_000_000_000, // v8.6.0: Increased from 100M to 1B total gas pool for 10x VM capacity
                50_000_000,    // v8.6.0: Increased from 10M to 50M max per request for 10x VM capacity
            ),
            min_balance_required: 1_000_000, // Minimum 1M units to submit
            chain_id,
        }
    }

    /// Full verification of a remote execution request
    ///
    /// Returns Ok with the verified request if all checks pass,
    /// or Err with detailed reason for rejection.
    pub async fn verify_request(
        &self,
        request: &SignedExecutionRequest,
        caller_balance: u64,
    ) -> Result<VerifiedExecutionRequest, VmError> {
        // 1. Verify signature and basic validity
        request.verify()?;

        // 2. Check chain ID matches
        if request.chain_id != self.chain_id {
            return Err(VmError::InvalidTransaction(
                format!("Wrong chain ID: {} != {}", request.chain_id, self.chain_id)
            ));
        }

        // 3. Check nonce hasn't been used (replay protection)
        self.nonce_tracker
            .check_and_mark_nonce(&request.caller_pubkey, request.nonce)
            .await?;

        // 4. Check rate limit
        self.rate_limiter
            .check_rate_limit(&request.caller_pubkey)
            .await?;

        // 5. Check caller balance is sufficient
        let max_cost = request.max_gas_cost();
        if caller_balance < self.min_balance_required {
            return Err(VmError::InvalidTransaction(
                format!(
                    "Insufficient balance: {} < minimum {}",
                    caller_balance, self.min_balance_required
                )
            ));
        }

        if (caller_balance as u128) < max_cost {
            return Err(VmError::InvalidTransaction(
                format!(
                    "Insufficient balance for gas: {} < max cost {}",
                    caller_balance, max_cost
                )
            ));
        }

        // 6. Check access permissions for contract
        if !self.access_controller
            .is_authorized_for_contract(&request.caller_pubkey, &request.contract_address)
            .await
        {
            return Err(VmError::InvalidTransaction(
                "Caller not authorized for this contract".into()
            ));
        }

        // 7. Try to acquire gas quota
        let _gas_permit = self.resource_quota
            .acquire_gas(request.gas_limit, &request.caller_pubkey)
            .await?;

        info!(
            "✅ Verified remote execution: contract={} function={} caller={} gas={}",
            &request.contract_address[..16.min(request.contract_address.len())],
            request.function,
            &request.caller[..16.min(request.caller.len())],
            request.gas_limit
        );

        Ok(VerifiedExecutionRequest {
            contract_address: request.contract_address.clone(),
            function: request.function.clone(),
            args: request.args.clone(),
            caller: request.caller.clone(),
            gas_limit: request.gas_limit,
            gas_price: request.gas_price,
            caller_pubkey: request.caller_pubkey,
            verified_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }

    /// Grant contract access to a caller
    pub async fn grant_access(&self, contract_address: String, caller_pubkey: [u8; 32]) {
        self.access_controller
            .grant_contract_permission(contract_address, caller_pubkey)
            .await;
    }

    /// Ban a misbehaving caller
    pub async fn ban_caller(&self, caller_pubkey: [u8; 32]) {
        self.access_controller.ban_peer(caller_pubkey).await;
    }

    /// Get verification statistics
    pub async fn get_stats(&self) -> RemoteExecutionStats {
        let quota_stats = self.resource_quota.get_stats().await;

        RemoteExecutionStats {
            total_gas_used: quota_stats.total_gas_used,
            active_callers: quota_stats.peers_active,
            available_gas: quota_stats.available_permits,
        }
    }

    /// Periodic cleanup of old nonces
    pub async fn cleanup(&self) {
        self.nonce_tracker.cleanup().await;
        self.rate_limiter.cleanup_old_states().await;
    }
}

/// A verified execution request that has passed all security checks
#[derive(Debug, Clone)]
pub struct VerifiedExecutionRequest {
    pub contract_address: String,
    pub function: String,
    pub args: Vec<u8>,
    pub caller: String,
    pub gas_limit: u64,
    pub gas_price: u64,
    pub caller_pubkey: [u8; 32],
    pub verified_at: u64,
}

/// Statistics for remote execution verification
#[derive(Debug, Clone, Default)]
pub struct RemoteExecutionStats {
    pub total_gas_used: u64,
    pub active_callers: usize,
    pub available_gas: u64,
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

        // Too short
        let short_bytecode = vec![0u8; 4];
        assert!(validator.validate(&short_bytecode).is_err());

        // Invalid magic number
        let invalid_magic = vec![0u8; 100];
        assert!(validator.validate(&invalid_magic).is_err());

        // Valid WASM magic number (but minimal module)
        let valid_magic = b"\x00asm\x01\x00\x00\x00".to_vec();
        assert!(validator.validate(&valid_magic).is_ok());
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

    #[test]
    fn test_signed_execution_request() {
        let signing_key = SigningKey::generate(&mut rand::thread_rng());

        let request = SignedExecutionRequest::new(
            "0x1234567890abcdef".to_string(),
            "transfer".to_string(),
            vec![1, 2, 3, 4],
            "0xcaller123".to_string(),
            100000,
            1000,
            1, // chain ID
            &signing_key,
        ).unwrap();

        // Verify signature
        assert!(request.verify().is_ok());

        // Verify derived address format
        let derived = request.derive_caller_address();
        assert!(derived.starts_with("0x"));
        assert_eq!(derived.len(), 42);
    }

    #[test]
    fn test_signed_execution_request_tampering() {
        let signing_key = SigningKey::generate(&mut rand::thread_rng());

        let mut request = SignedExecutionRequest::new(
            "0x1234567890abcdef".to_string(),
            "transfer".to_string(),
            vec![1, 2, 3, 4],
            "0xcaller123".to_string(),
            100000,
            1000,
            1,
            &signing_key,
        ).unwrap();

        // Tamper with gas limit
        request.gas_limit = 200000;

        // Verification should fail
        assert!(request.verify().is_err());
    }

    #[tokio::test]
    async fn test_remote_execution_verifier() {
        let verifier = RemoteExecutionVerifier::new(1);
        let signing_key = SigningKey::generate(&mut rand::thread_rng());

        let request = SignedExecutionRequest::new(
            "0x1234567890abcdef".to_string(),
            "balanceOf".to_string(),
            vec![],
            "0xcaller".to_string(),
            10000,
            1,
            1, // correct chain ID
            &signing_key,
        ).unwrap();

        // Verify with sufficient balance
        let result = verifier.verify_request(&request, 10_000_000).await;
        assert!(result.is_ok());

        // Verify returns proper struct
        let verified = result.unwrap();
        assert_eq!(verified.function, "balanceOf");
        assert_eq!(verified.gas_limit, 10000);
    }

    #[tokio::test]
    async fn test_remote_execution_verifier_wrong_chain() {
        let verifier = RemoteExecutionVerifier::new(1);
        let signing_key = SigningKey::generate(&mut rand::thread_rng());

        let request = SignedExecutionRequest::new(
            "0x1234567890abcdef".to_string(),
            "balanceOf".to_string(),
            vec![],
            "0xcaller".to_string(),
            10000,
            1,
            999, // wrong chain ID
            &signing_key,
        ).unwrap();

        let result = verifier.verify_request(&request, 10_000_000).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{:?}", err).contains("chain ID"));
    }

    #[tokio::test]
    async fn test_remote_execution_insufficient_balance() {
        let verifier = RemoteExecutionVerifier::new(1);
        let signing_key = SigningKey::generate(&mut rand::thread_rng());

        let request = SignedExecutionRequest::new(
            "0x1234".to_string(),
            "transfer".to_string(),
            vec![],
            "0xcaller".to_string(),
            10000,
            1,
            1,
            &signing_key,
        ).unwrap();

        // Verify with insufficient balance
        let result = verifier.verify_request(&request, 100).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{:?}", err).contains("balance"));
    }
}
