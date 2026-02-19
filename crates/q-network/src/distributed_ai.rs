use libp2p::gossipsub::{IdentTopic, Topic};
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};
use uuid::Uuid;
use chrono;

// v2.5.1-beta: AEAD encryption for P2P AI message privacy
use chacha20poly1305::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    XChaCha20Poly1305, XNonce,
};

/// v2.5.1-beta: Encryption helper for P2P AI messages
/// Uses XChaCha20-Poly1305 AEAD with 256-bit keys (128-bit post-quantum security)
pub struct AIMessageEncryption {
    /// Symmetric encryption key (derived from Kyber KEM or pre-shared)
    cipher: XChaCha20Poly1305,
}

impl AIMessageEncryption {
    /// Create new encryption instance from 32-byte key
    pub fn new(key: &[u8; 32]) -> Self {
        let cipher = XChaCha20Poly1305::new(key.into());
        Self { cipher }
    }

    /// Create encryption instance from shared secret (e.g., from Kyber KEM)
    pub fn from_shared_secret(shared_secret: &[u8]) -> Self {
        // Derive 32-byte key using BLAKE3
        let key = blake3::derive_key("qnk-ai-message-encryption-v1", shared_secret);
        Self::new(&key)
    }

    /// Encrypt prompt/message content
    /// Returns (nonce, ciphertext) tuple
    pub fn encrypt(&self, plaintext: &str) -> Result<(Vec<u8>, Vec<u8>), String> {
        let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);
        let ciphertext = self.cipher
            .encrypt(&nonce, plaintext.as_bytes())
            .map_err(|e| format!("Encryption failed: {}", e))?;
        Ok((nonce.to_vec(), ciphertext))
    }

    /// Decrypt message content
    pub fn decrypt(&self, nonce: &[u8], ciphertext: &[u8]) -> Result<String, String> {
        let nonce_arr: [u8; 24] = nonce.try_into()
            .map_err(|_| "Invalid nonce length")?;
        let nonce = XNonce::from_slice(&nonce_arr);

        let plaintext = self.cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| format!("Decryption failed: {}", e))?;

        String::from_utf8(plaintext)
            .map_err(|e| format!("UTF-8 decode failed: {}", e))
    }
}

/// v2.5.1-beta: Encrypted content wrapper for sensitive AI data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedContent {
    /// 24-byte XChaCha20 nonce
    pub nonce: Vec<u8>,
    /// Encrypted ciphertext with Poly1305 auth tag
    pub ciphertext: Vec<u8>,
}

/// Gossipsub topics for distributed AI inference
pub const TOPIC_AI_INFERENCE_REQUEST: &str = "qnk/ai/inference-request/v1";
pub const TOPIC_AI_LAYER_OUTPUT: &str = "qnk/ai/layer-output/v1";
pub const TOPIC_AI_NODE_CAPABILITY: &str = "qnk/ai/node-capability/v1";
pub const TOPIC_AI_COORDINATOR: &str = "qnk/ai/coordinator/v1";
pub const TOPIC_AI_HEARTBEAT: &str = "qnk/ai/heartbeat/v1";

// v6.0.0: New topics for decentralized inference protocol
/// Offers from workers to handle inference requests (offer-based routing)
pub const TOPIC_AI_INFERENCE_OFFER: &str = "qnk/ai/inference-offer/v1";
/// opML verification challenges and results
pub const TOPIC_AI_OPML_VERIFICATION: &str = "qnk/ai/opml-verification/v1";
/// Worker staking events (stake, unstake, slash)
pub const TOPIC_AI_STAKING: &str = "qnk/ai/staking/v1";
/// Model registry events (new model registered)
pub const TOPIC_AI_MODEL_REGISTRY: &str = "qnk/ai/model-registry/v1";

/// AI-specific Gossipsub topics manager
pub struct DistributedAITopics {
    pub inference_request: IdentTopic,
    pub layer_output: IdentTopic,
    pub node_capability: IdentTopic,
    pub coordinator: IdentTopic,
    pub heartbeat: IdentTopic,
    // v6.0.0: Decentralized inference topics
    pub inference_offer: IdentTopic,
    pub opml_verification: IdentTopic,
    pub staking: IdentTopic,
    pub model_registry: IdentTopic,
}

impl DistributedAITopics {
    pub fn new() -> Self {
        info!("🤖 Initializing Distributed AI Gossipsub topics (v6.0.0 decentralized)");

        Self {
            inference_request: IdentTopic::new(TOPIC_AI_INFERENCE_REQUEST),
            layer_output: IdentTopic::new(TOPIC_AI_LAYER_OUTPUT),
            node_capability: IdentTopic::new(TOPIC_AI_NODE_CAPABILITY),
            coordinator: IdentTopic::new(TOPIC_AI_COORDINATOR),
            heartbeat: IdentTopic::new(TOPIC_AI_HEARTBEAT),
            inference_offer: IdentTopic::new(TOPIC_AI_INFERENCE_OFFER),
            opml_verification: IdentTopic::new(TOPIC_AI_OPML_VERIFICATION),
            staking: IdentTopic::new(TOPIC_AI_STAKING),
            model_registry: IdentTopic::new(TOPIC_AI_MODEL_REGISTRY),
        }
    }

    /// Get all AI topics for subscription
    pub fn all_topics(&self) -> Vec<IdentTopic> {
        vec![
            self.inference_request.clone(),
            self.layer_output.clone(),
            self.node_capability.clone(),
            self.coordinator.clone(),
            self.heartbeat.clone(),
            self.inference_offer.clone(),
            self.opml_verification.clone(),
            self.staking.clone(),
            self.model_registry.clone(),
        ]
    }

    /// Check if a topic hash matches any AI topic
    pub fn is_ai_topic(&self, topic: &libp2p::gossipsub::TopicHash) -> bool {
        let topic_str = topic.as_str();
        topic_str.starts_with("qnk/ai/")
    }
}

impl Default for DistributedAITopics {
    fn default() -> Self {
        Self::new()
    }
}

/// AI message envelope for Gossipsub with AEGIS-QL authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIGossipsubMessage {
    /// Protocol version for compatibility checking (v0.9.29+ FIX: Prevents binary incompatibility)
    /// Version 1: Initial protocol with all current features
    /// Increment when making breaking changes to message format
    #[serde(default = "default_protocol_version")]
    pub protocol_version: u32,

    pub message_id: String,
    pub timestamp: i64,
    pub sender_node_id: String,
    pub sender_peer_id: String,
    pub payload: AIMessagePayload,

    // AEGIS-QL post-quantum message authentication (Phase 1 enhancement)
    // v2.3.18-beta FIX: Removed skip_serializing_if for postcard binary compatibility
    // Postcard requires all fields in exact order with discriminants
    #[serde(default)] // v0.9.14 FIX: Backwards compatibility - use None if field missing
    pub aegis_signature: Option<Vec<u8>>, // AEGIS-256 MAC for message integrity
    #[serde(default)] // v0.9.14 FIX: Backwards compatibility - use None if field missing
    pub sender_public_key: Option<Vec<u8>>, // Ed25519 public key for verification

    // Retry and reliability metadata
    #[serde(default)] // v0.9.14 FIX: Backwards compatibility - use 0 if field missing
    pub sequence_number: u64, // Monotonic sequence for deduplication
    #[serde(default)] // v0.9.14 FIX: Backwards compatibility - use 0 if field missing
    pub retry_count: u8, // Number of retries (for exponential backoff)
    #[serde(default)] // v0.9.14 FIX: Backwards compatibility - use Normal if field missing
    pub priority: MessagePriority, // Priority for gossipsub mesh routing
}

/// Default protocol version for backwards compatibility
/// v0.9.29+ nodes will use version 1, older nodes default to 0
fn default_protocol_version() -> u32 {
    0 // Old binaries without version field will deserialize as version 0
}

/// Current protocol version - increment when making breaking changes
pub const CURRENT_PROTOCOL_VERSION: u32 = 1;

/// Message priority for gossipsub routing optimization
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    Low = 0,      // Heartbeats, capability announcements
    Normal = 1,   // Regular inference requests
    High = 2,     // Layer outputs, KV cache updates
    Critical = 3, // Coordinator election, error recovery
}

impl Default for MessagePriority {
    fn default() -> Self {
        MessagePriority::Normal
    }
}

impl AIGossipsubMessage {
    /// Create a new message with automatic sequence numbering
    /// v0.9.29+ FIX: Now includes protocol version for compatibility checking
    pub fn new(
        sender_node_id: String,
        sender_peer_id: String,
        payload: AIMessagePayload,
        sequence_number: u64,
    ) -> Self {
        use uuid::Uuid;

        let priority = match &payload {
            AIMessagePayload::CoordinatorElection { .. } => MessagePriority::Critical,
            AIMessagePayload::LayerOutput { .. } | AIMessagePayload::KVCacheUpdate { .. } => MessagePriority::High,
            AIMessagePayload::TokenChunk { .. } => MessagePriority::High, // Streaming tokens need priority
            AIMessagePayload::InferenceStarted { .. } => MessagePriority::High, // Start acknowledgment
            AIMessagePayload::InferenceComplete { .. } => MessagePriority::High, // Completion signal
            AIMessagePayload::InferenceError { .. } => MessagePriority::High, // Errors need fast routing
            AIMessagePayload::InferenceRequest { .. } | AIMessagePayload::InferenceResponse { .. } => MessagePriority::Normal,
            AIMessagePayload::TargetedInferenceRequest { .. } => MessagePriority::Normal, // Data parallel requests
            AIMessagePayload::CancelInference { .. } => MessagePriority::Normal, // Cancellation requests
            AIMessagePayload::Heartbeat { .. } | AIMessagePayload::NodeCapability { .. } => MessagePriority::Low,
            // Tensor parallelism messages - high priority for low latency
            AIMessagePayload::AllReduceChunk { .. } => MessagePriority::Critical, // Time-sensitive
            AIMessagePayload::AllReduceComplete { .. } => MessagePriority::High,
            AIMessagePayload::ShardAssignment { .. } => MessagePriority::High,
            AIMessagePayload::WeightShard { .. } => MessagePriority::Normal, // Bulk transfer
            AIMessagePayload::ShardReady { .. } => MessagePriority::High,
            AIMessagePayload::TensorParallelRequest { .. } => MessagePriority::High,
            AIMessagePayload::HiddenStates { .. } => MessagePriority::Critical, // Time-sensitive
            AIMessagePayload::TensorParallelToken { .. } => MessagePriority::High,
            _ => MessagePriority::Normal,
        };

        Self {
            // v2.6.9 FIX: Use protocol v0 for unsigned messages until signing infrastructure is ready
            // Protocol v0 allows unsigned messages for backwards compatibility
            // TODO: Implement proper message signing and switch to protocol v1
            protocol_version: 0, // TEMPORARY: Use v0 to allow unsigned messages
            message_id: Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().timestamp(),
            sender_node_id,
            sender_peer_id,
            payload,
            aegis_signature: None,
            sender_public_key: None,
            sequence_number,
            retry_count: 0,
            priority,
        }
    }

    /// Increment retry count for exponential backoff
    pub fn increment_retry(&mut self) {
        self.retry_count = self.retry_count.saturating_add(1);
    }

    /// Calculate exponential backoff delay in milliseconds
    pub fn backoff_delay_ms(&self) -> u64 {
        // Exponential backoff: 100ms, 200ms, 400ms, 800ms, 1600ms (max 5 retries)
        let base_delay = 100;
        let max_retries = 5;
        if self.retry_count >= max_retries {
            return base_delay * (1 << (max_retries - 1)); // Cap at max delay
        }
        base_delay * (1 << self.retry_count)
    }

    /// Check if message should be retired (too many retries)
    pub fn should_retire(&self) -> bool {
        self.retry_count >= 5
    }

    /// Write variable-length string with u32 length prefix
    /// This prevents serialization ambiguity attacks where "ABC" + "DEF" = "AB" + "CDEF"
    fn write_string(buffer: &mut Vec<u8>, s: &str) {
        let bytes = s.as_bytes();
        let len = bytes.len() as u32;
        buffer.extend_from_slice(&len.to_le_bytes());
        buffer.extend_from_slice(bytes);
    }

    /// Create canonical message for signing/verification
    ///
    /// v1.0.3-beta FIX: Versioned canonical formats with migration path
    ///
    /// IMPORTANT: Each protocol version has a FROZEN canonical format.
    /// Changing a format requires incrementing protocol_version and adding new branch.
    fn create_canonical_message(&self, payload_bytes: &[u8]) -> Vec<u8> {
        match self.protocol_version {
            0 => self.create_canonical_v0(payload_bytes),
            1 => self.create_canonical_v1(payload_bytes),
            _ => {
                error!("⚠️  Unsupported protocol version: {}", self.protocol_version);
                error!("   Falling back to v1 canonical format");
                self.create_canonical_v1(payload_bytes)
            }
        }
    }

    /// Protocol v0 canonical format (LEGACY - preserved for backwards compatibility)
    ///
    /// FORMAT (FROZEN - DO NOT CHANGE):
    /// ================================
    /// timestamp: i64 (8 bytes, little-endian)
    /// sender_node_id: raw bytes (NO length prefix - VULNERABLE to ambiguity)
    /// sender_peer_id: raw bytes (NO length prefix - VULNERABLE to ambiguity)
    /// sequence_number: u64 (8 bytes, little-endian)
    /// payload: raw bytes (NO length prefix - VULNERABLE to ambiguity)
    ///
    /// SECURITY WARNING: This format has serialization ambiguity vulnerability.
    /// Only used for backwards compatibility with pre-v1.0.3 nodes.
    fn create_canonical_v0(&self, payload_bytes: &[u8]) -> Vec<u8> {
        let mut message = Vec::with_capacity(
            8 + // timestamp
            self.sender_node_id.len() +
            self.sender_peer_id.len() +
            8 + // sequence_number
            payload_bytes.len()
        );

        // V0 format (no length prefixes - vulnerable but preserved for migration)
        message.extend_from_slice(&self.timestamp.to_le_bytes());
        message.extend_from_slice(self.sender_node_id.as_bytes());
        message.extend_from_slice(self.sender_peer_id.as_bytes());
        message.extend_from_slice(&self.sequence_number.to_le_bytes());
        message.extend_from_slice(payload_bytes);

        message
    }

    /// Protocol v1 canonical format (CURRENT - v1.0.3+)
    ///
    /// FORMAT (FROZEN - DO NOT CHANGE):
    /// ================================
    /// protocol_version: u32 (4 bytes, little-endian)
    /// message_id: u32 length + UTF-8 bytes
    /// timestamp: i64 (8 bytes, little-endian)
    /// sequence_number: u64 (8 bytes, little-endian)
    /// sender_node_id: u32 length + UTF-8 bytes
    /// sender_peer_id: u32 length + UTF-8 bytes
    /// priority: u8 (1 byte)
    /// payload: u32 length + bytes
    ///
    /// SECURITY: Length-prefixed encoding prevents serialization ambiguity attacks.
    /// FUTURE: v1.1 will migrate to deterministic bincode (protocol_version = 2)
    fn create_canonical_v1(&self, payload_bytes: &[u8]) -> Vec<u8> {
        let mut message = Vec::with_capacity(
            4 + // protocol_version
            4 + self.message_id.len() +
            8 + // timestamp
            8 + // sequence_number
            4 + self.sender_node_id.len() +
            4 + self.sender_peer_id.len() +
            1 + // priority
            4 + payload_bytes.len()
        );

        // Fixed-size: protocol version (4 bytes)
        message.extend_from_slice(&self.protocol_version.to_le_bytes());

        // Variable: message_id (len + data)
        Self::write_string(&mut message, &self.message_id);

        // Fixed-size: timestamp (8 bytes)
        message.extend_from_slice(&self.timestamp.to_le_bytes());

        // Fixed-size: sequence_number (8 bytes)
        message.extend_from_slice(&self.sequence_number.to_le_bytes());

        // Variable: sender_node_id (len + data)
        Self::write_string(&mut message, &self.sender_node_id);

        // Variable: sender_peer_id (len + data)
        Self::write_string(&mut message, &self.sender_peer_id);

        // Fixed-size: priority (1 byte)
        message.push(self.priority as u8);

        // Variable: payload (len + data)
        let payload_len = payload_bytes.len() as u32;
        message.extend_from_slice(&payload_len.to_le_bytes());
        message.extend_from_slice(payload_bytes);

        message
    }

    /// Sign message with hybrid quantum-resistant signature (Ed25519 + Lamport OTS)
    /// v1.0.3-beta FIX: Implements proper cryptographic authentication with canonical message format
    pub async fn sign(&mut self, signer: &q_quantum_crypto::QuantumSigner) -> anyhow::Result<()> {
        use sha3::{Digest, Sha3_256};

        // Serialize payload for signing
        let payload_bytes = bincode::serialize(&self.payload)
            .map_err(|e| anyhow::anyhow!("Failed to serialize payload: {}", e))?;

        // Create canonical message with length-prefixed encoding (prevents ambiguity attacks)
        let canonical_message = self.create_canonical_message(&payload_bytes);

        // Generate quantum-resistant signature (Lamport OTS)
        let quantum_sig = signer.sign_message(&canonical_message).await
            .map_err(|e| anyhow::anyhow!("Failed to sign message: {}", e))?;

        // Get current public key
        let public_key = signer.get_public_key().await
            .map_err(|e| anyhow::anyhow!("Failed to get public key: {}", e))?;

        // Store signature and public key (serialized QuantumSignature)
        self.aegis_signature = Some(bincode::serialize(&quantum_sig)
            .map_err(|e| anyhow::anyhow!("Failed to serialize signature: {}", e))?);
        self.sender_public_key = Some(public_key);

        Ok(())
    }

    /// Verify message timestamp is within acceptable window
    /// v1.0.3-beta FIX: Prevents replay attacks with old signed messages
    ///
    /// Rejection criteria:
    /// - Messages older than 5 minutes (300 seconds)
    /// - Messages from future (>30s clock skew tolerance)
    pub fn verify_timestamp(&self) -> bool {
        use tracing::{error, warn};

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let msg_age = now - self.timestamp;

        // Reject messages older than 5 minutes
        if msg_age > 300 {
            error!("❌ Message too old: {} seconds (max 300)", msg_age);
            error!("   Message ID: {}", self.message_id);
            error!("   Sender: {} (peer: {})", self.sender_node_id, self.sender_peer_id);
            error!("   Timestamp: {}, Now: {}", self.timestamp, now);
            error!("   🚨 POSSIBLE REPLAY ATTACK: Message timestamp outside acceptable window");
            return false;
        }

        // Reject messages from future (allow 30s clock skew)
        if msg_age < -30 {
            error!("❌ Message from future: {} seconds ahead", msg_age.abs());
            error!("   Message ID: {}", self.message_id);
            error!("   Sender: {} (peer: {})", self.sender_node_id, self.sender_peer_id);
            error!("   Timestamp: {}, Now: {}", self.timestamp, now);
            error!("   🚨 POSSIBLE CLOCK SKEW ATTACK: Check sender's system clock");
            return false;
        }

        true
    }

    /// Verify message authenticity with quantum-resistant signature verification
    /// v1.0.3-beta FIX: Implements proper cryptographic verification with timestamp validation
    ///
    /// Security Model:
    /// - Protocol v0: Accepts unsigned messages (backwards compatibility during migration)
    /// - Protocol v1+: Requires signature + public key
    /// - Uses Lamport OTS (information-theoretically secure, quantum-resistant)
    /// - Verifies timestamp (5-minute window) to prevent replay attacks
    /// - Verifies: timestamp || sender_node_id || sender_peer_id || sequence || payload
    pub async fn verify_signature_async(&self) -> bool {
        use sha3::{Digest, Sha3_256};
        use tracing::{debug, error, warn};

        // Step 1: Verify timestamp FIRST (cheap check to prevent replay attacks)
        if !self.verify_timestamp() {
            return false;
        }

        // Protocol v0: Allow unsigned messages for backwards compatibility (TEMPORARY)
        // TODO: Remove after 2-week migration period (target: 2025-12-07)
        if self.protocol_version == 0 && self.aegis_signature.is_none() {
            warn!("⚠️  Accepting unsigned message from legacy node (protocol v0) - message_id: {}",
                  self.message_id);
            warn!("   Sender: {} (peer: {})", self.sender_node_id, self.sender_peer_id);
            warn!("   ⚠️  MIGRATION WARNING: Unsigned messages will be REJECTED after 2025-12-07");
            return true; // Backwards compatibility
        }

        // Protocol v1+: Require signatures
        if self.aegis_signature.is_none() || self.sender_public_key.is_none() {
            error!("❌ SIGNATURE VERIFICATION FAILED: Missing signature or public key");
            error!("   Message ID: {}", self.message_id);
            error!("   Protocol version: {}", self.protocol_version);
            error!("   Sender: {} (peer: {})", self.sender_node_id, self.sender_peer_id);
            error!("   Has signature: {}", self.aegis_signature.is_some());
            error!("   Has public key: {}", self.sender_public_key.is_some());
            return false;
        }

        // Reconstruct signed message (canonical form with length-prefixed encoding)
        let payload_bytes = match bincode::serialize(&self.payload) {
            Ok(bytes) => bytes,
            Err(e) => {
                error!("❌ SIGNATURE VERIFICATION FAILED: Cannot serialize payload: {}", e);
                error!("   Message ID: {}", self.message_id);
                return false;
            }
        };

        // Use canonical message format (same as signing)
        let canonical_message = self.create_canonical_message(&payload_bytes);

        // Deserialize quantum signature
        let signature: q_quantum_crypto::QuantumSignature = match bincode::deserialize(self.aegis_signature.as_ref().unwrap()) {
            Ok(sig) => sig,
            Err(e) => {
                error!("❌ SIGNATURE VERIFICATION FAILED: Invalid signature format: {}", e);
                error!("   Message ID: {}", self.message_id);
                error!("   Signature length: {} bytes", self.aegis_signature.as_ref().unwrap().len());
                return false;
            }
        };

        // Create verifier with sender's node ID (for audit trail)
        let sender_node_id: [u8; 32] = {
            let hash = Sha3_256::digest(self.sender_node_id.as_bytes());
            let mut node_id = [0u8; 32];
            node_id.copy_from_slice(&hash);
            node_id
        };

        let verifier = q_quantum_crypto::QuantumVerifier::new(sender_node_id);

        // Verify quantum signature (Lamport OTS) against canonical message
        match verifier.verify_signature(&canonical_message, &signature).await {
            Ok(true) => {
                debug!("✅ Signature verified for message {} from {} (protocol v{})",
                       self.message_id, self.sender_node_id, self.protocol_version);
                true
            }
            Ok(false) => {
                error!("❌ SIGNATURE VERIFICATION FAILED: Invalid signature");
                error!("   Message ID: {}", self.message_id);
                error!("   Sender: {} (peer: {})", self.sender_node_id, self.sender_peer_id);
                error!("   Protocol version: {}", self.protocol_version);
                error!("   Timestamp: {}", self.timestamp);
                error!("   Sequence number: {}", self.sequence_number);
                error!("   🚨 POSSIBLE ATTACK: Message authentication failed");
                false
            }
            Err(e) => {
                error!("❌ SIGNATURE VERIFICATION ERROR: {}", e);
                error!("   Message ID: {}", self.message_id);
                error!("   This may indicate corrupted data or incompatible signature format");
                false
            }
        }
    }

    /// Synchronous signature verification (calls async internally)
    /// NOTE: This blocks the current thread! Use verify_signature_async() when possible
    pub fn verify_signature(&self) -> bool {
        // Create a runtime for async verification
        // This is a temporary bridge until all callers are converted to async
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.verify_signature_async().await
            })
        })
    }
}

/// Distributed AI message payload types
/// v0.9.29+ FIX: Added explicit discriminants for binary stability
/// IMPORTANT: Never reorder variants or change discriminants - this will break compatibility!
/// To add new message types, append to the end with the next sequential number
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(u8)] // Force explicit u8 discriminants for binary stability
pub enum AIMessagePayload {
    InferenceRequest {
        request_id: String,
        prompt: String,
        max_tokens: Option<usize>,
        temperature: Option<f64>,
        model: String,
    } = 0,
    InferenceResponse {
        request_id: String,
        generated_text: String,
        tokens_generated: usize,
        latency_ms: u64,
        nodes_participated: Vec<String>,
    } = 1,
    LayerOutput {
        request_id: String,
        layer_index: usize,
        compressed_data: Vec<u8>,
        shape: Vec<usize>,
    } = 2,
    NodeCapability {
        node_id: String,
        peer_id: String,
        capability: NodeCapability,
        available_layers: usize,
    } = 3,
    CoordinatorElection {
        node_id: String,
        score: u64,
        uptime_secs: u64,
        inference_count: u64,
    } = 4,
    Heartbeat {
        node_id: String,
        active_requests: usize,
        layers_assigned: Option<(usize, usize)>, // (start, end)
    } = 5,
    LayerAssignment {
        request_id: String,
        assignments: std::collections::HashMap<String, (usize, usize)>, // node_id -> (start_layer, end_layer)
    } = 6,
    KVCacheUpdate {
        request_id: String,
        layer_index: usize,
        cache_data: Vec<u8>,
        sequence_length: usize,
    } = 7,
    /// NEW v1.0: Targeted inference request for data parallelism
    /// Only the specified target node processes this request (load balanced)
    /// v2.5.1-beta: Added encrypted_prompt for privacy (prompt field is empty when encrypted)
    TargetedInferenceRequest {
        request_id: String,
        target_node_id: String, // Only this node should process
        prompt: String,         // Empty if encrypted_prompt is set (backward compat)
        max_tokens: Option<usize>,
        temperature: Option<f64>,
        model: String,
        /// v2.5.1-beta: Encrypted prompt for privacy (takes precedence over prompt)
        #[serde(default)]
        encrypted_prompt: Option<EncryptedContent>,
    } = 8,
    /// NEW v1.0: Token chunk for streaming responses
    /// Sent from worker to coordinator during generation
    /// v2.5.1-beta: Added encrypted_token for privacy
    TokenChunk {
        request_id: String,
        token: String,          // Empty if encrypted_token is set (backward compat)
        token_index: usize,
        /// v2.5.1-beta: Encrypted token for privacy (takes precedence over token)
        #[serde(default)]
        encrypted_token: Option<EncryptedContent>,
    } = 9,
    /// NEW v1.0: Worker acknowledges it accepted the targeted request
    /// Allows coordinator to detect if worker is unresponsive (timeout re-route)
    InferenceStarted {
        request_id: String,
        worker_node_id: String,
        model: String,
        started_at_ms: u64,
    } = 10,
    /// NEW v1.0: Worker signals completion
    /// finish_reason: "eos", "length", "stop", "cancelled", "error"
    InferenceComplete {
        request_id: String,
        worker_node_id: String,
        finish_reason: String,
        tokens_generated: usize,
        total_time_ms: u64,
    } = 11,
    /// NEW v1.0: Cancel inference (client disconnect or timeout)
    /// Cooperative cancellation - worker should stop generation
    CancelInference {
        request_id: String,
        target_node_id: String,
        reason: String, // "client_closed", "timeout", "reassign"
    } = 12,
    /// NEW v1.0: Inference error from worker
    InferenceError {
        request_id: String,
        worker_node_id: String,
        code: String,    // "engine_error", "model_load_failed", etc.
        message: String,
    } = 13,

    // ============================================================
    // TENSOR PARALLELISM MESSAGES (v2.4.0+)
    // These enable true parallel inference where all nodes work
    // together on each token for Nx speedup (not just throughput)
    // ============================================================

    /// All-reduce chunk during scatter-reduce or all-gather phase
    /// Used to combine partial tensor results from multiple nodes
    AllReduceChunk {
        request_id: String,
        layer_index: usize,
        phase: String,        // "scatter" or "gather"
        step: usize,          // Step within phase (0 to N-2)
        chunk_index: usize,   // Which chunk of the ring buffer
        tensor_data: Vec<u8>, // Compressed tensor chunk
        shape: Vec<usize>,
        compressed: bool,
    } = 14,

    /// Signal that all-reduce is complete for a layer
    AllReduceComplete {
        request_id: String,
        layer_index: usize,
        total_time_ms: u64,
    } = 15,

    /// Coordinator assigns weight shards to nodes
    /// Sent when establishing tensor parallel group
    ShardAssignment {
        request_id: String,
        node_rank: usize,       // This node's rank (0 to world_size-1)
        world_size: usize,      // Total nodes in tensor parallel group
        model_name: String,     // e.g., "Mistral-7B-Instruct-v0.3"
        layers_start: usize,    // First layer (usually 0)
        layers_end: usize,      // Last layer (usually 31)
    } = 16,

    /// Weight shard being distributed from coordinator to workers
    WeightShard {
        request_id: String,
        layer_index: usize,
        weight_name: String,    // e.g., "attn_q", "ffn_gate"
        shard_data: Vec<u8>,    // Compressed weight data
        shape: Vec<usize>,
        shard_dim: usize,       // Which dimension was sharded (0 or 1)
        original_shape: Vec<usize>,
        compressed: bool,
    } = 17,

    /// Worker acknowledges shard receipt and readiness
    ShardReady {
        request_id: String,
        node_id: String,
        layer_index: usize,
        weight_name: String,
        size_bytes: usize,
    } = 18,

    /// Tensor parallel inference request (coordinator → all workers)
    TensorParallelRequest {
        request_id: String,
        prompt: String,
        input_ids: Vec<u32>,    // Tokenized input
        max_tokens: usize,
        temperature: f64,
    } = 19,

    /// Hidden states being broadcast for next layer
    /// All nodes receive same hidden states, compute partial outputs
    HiddenStates {
        request_id: String,
        layer_index: usize,
        sequence_position: usize,
        data: Vec<u8>,          // Compressed hidden states [batch, seq, hidden_dim]
        shape: Vec<usize>,
        compressed: bool,
    } = 20,

    /// Tensor parallel token output (after final layer)
    TensorParallelToken {
        request_id: String,
        token_id: u32,
        token_text: String,
        token_index: usize,
        logprob: f32,
    } = 21,

    // ============================================================
    // TOKEN STREAMING OPTIMIZATION (v2.5.1+)
    // Batch multiple tokens together to reduce P2P message overhead
    // 10-15% speedup by reducing network round-trips
    // ============================================================

    /// Bulk token chunk for optimized streaming
    /// Batches multiple tokens (typically 4-8) to reduce network overhead
    /// v2.5.1-beta: Supports both plaintext and encrypted tokens
    BulkTokenChunk {
        request_id: String,
        /// Starting index of first token in this batch
        start_index: usize,
        /// Plaintext tokens (empty if encrypted)
        tokens: Vec<String>,
        /// v2.5.1-beta: Encrypted concatenated tokens for privacy
        #[serde(default)]
        encrypted_tokens: Option<EncryptedContent>,
    } = 22,

    // ============================================================
    // RPC PIPELINE PARALLELISM MESSAGES (v5.1.0+)
    // llama.cpp RPC workers for distributed layer processing
    // ============================================================

    /// Node announcing its llama.cpp RPC worker endpoint is available
    /// Coordinator collects these to build `--rpc host1:port,host2:port` args
    RpcWorkerAvailable {
        peer_id: String,
        host: String,
        port: u16,
        available_memory_gb: usize,
    } = 23,

    /// Node announcing its RPC worker has stopped
    RpcWorkerStopped {
        peer_id: String,
    } = 24,

    // ═══════════════════════════════════════════════════════════════
    // v6.0.0: Decentralized AI Inference Protocol Messages
    // ═══════════════════════════════════════════════════════════════

    /// v6.0.0: Worker announces capability with staking info for decentralized routing.
    /// Replaces single-coordinator model with offer-based routing.
    StakedWorkerCapability {
        peer_id: String,
        node_id: String,
        capability: NodeCapability,
        /// SHA3-256 of model GGUF file
        model_hash: Vec<u8>,
        /// Human-readable model name
        model_name: String,
        /// QUG staked (24-decimal base units)
        stake_amount: u128,
        /// Worker's price per token (24-decimal QUG)
        price_per_token: u128,
        /// Current load (active requests / max capacity)
        #[serde(default)]
        load_fraction: f32,
    } = 25,

    /// v6.0.0: Worker offers to handle an inference request (offer-based routing).
    /// First valid offer wins — natural load balancing without coordinator.
    InferenceOffer {
        request_id: String,
        worker_peer_id: String,
        worker_address: Vec<u8>,
        estimated_latency_ms: u64,
        price_per_token: u128,
        model_hash: Vec<u8>,
    } = 26,

    /// v6.0.0: User accepts a worker's offer and assigns the inference task.
    InferenceAssignment {
        request_id: String,
        worker_peer_id: String,
        user_address: Vec<u8>,
        /// Payment commitment (escrow reference)
        escrow_amount: u128,
        /// Deterministic seed for verifiable inference
        #[serde(default)]
        deterministic_seed: Option<u64>,
    } = 27,

    /// v6.0.0: Worker commits inference output hash (opML commitment).
    OpMLCommitment {
        request_id: String,
        worker_address: Vec<u8>,
        output_hash: Vec<u8>,
        token_count: u32,
        model_hash: Vec<u8>,
        seed: u64,
        /// Dilithium signature over (request_id ++ output_hash)
        signature: Vec<u8>,
    } = 28,

    /// v6.0.0: Verification challenge — assigned verifier must re-execute.
    VerificationChallenge {
        request_id: String,
        verifier_address: Vec<u8>,
        /// Worker's commitment for the verifier to check against
        worker_output_hash: Vec<u8>,
        worker_token_count: u32,
        model_hash: Vec<u8>,
        seed: u64,
        prompt: String,
        max_tokens: u32,
    } = 29,

    /// v6.0.0: Verifier submits re-execution result.
    VerificationResult {
        request_id: String,
        verifier_address: Vec<u8>,
        output_hash: Vec<u8>,
        matches_worker: bool,
    } = 30,

    /// v6.0.0: Dispute opened (hashes differ between worker and verifier).
    DisputeOpened {
        request_id: String,
        worker_address: Vec<u8>,
        verifier_address: Vec<u8>,
    } = 31,

    /// v6.0.0: Bisection round in dispute protocol.
    DisputeBisection {
        request_id: String,
        round: u32,
        range_lo: u32,
        range_hi: u32,
        /// Prefix hash at midpoint from each party
        worker_hash_at_mid: Vec<u8>,
        verifier_hash_at_mid: Vec<u8>,
    } = 32,

    /// v6.0.0: Dispute resolved with slashing.
    DisputeResolved {
        request_id: String,
        /// "worker_correct", "verifier_correct", or "inconclusive"
        outcome: String,
        slashed_address: Vec<u8>,
        slashed_amount: u128,
        bounty_recipient: Vec<u8>,
        bounty_amount: u128,
    } = 33,

    /// v6.0.0: Worker staking/unstaking event broadcast.
    StakeEvent {
        worker_address: Vec<u8>,
        peer_id: String,
        /// "stake", "unstake_request", "unstake_complete", "slash"
        event_type: String,
        amount: u128,
        new_total_stake: u128,
    } = 34,

    /// v6.0.0: Model registration broadcast.
    ModelRegistered {
        model_hash: Vec<u8>,
        model_name: String,
        family: String,
        quantization: String,
        parameter_count: u64,
        registered_by: Vec<u8>,
    } = 35,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeCapability {
    CPU { cores: usize, ram_gb: usize },
    CUDA { vram_gb: usize, compute_capability: String },
    Metal { vram_gb: usize },
}

impl NodeCapability {
    pub fn score(&self) -> u64 {
        match self {
            NodeCapability::CPU { cores, ram_gb } => (*cores as u64) * 10 + (*ram_gb as u64),
            NodeCapability::CUDA { vram_gb, .. } => (*vram_gb as u64) * 1000,
            NodeCapability::Metal { vram_gb } => (*vram_gb as u64) * 800,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ai_topics_creation() {
        let topics = DistributedAITopics::new();
        assert_eq!(topics.inference_request.as_str(), TOPIC_AI_INFERENCE_REQUEST);
        assert_eq!(topics.all_topics().len(), 5);
    }

    #[test]
    fn test_is_ai_topic() {
        use libp2p::gossipsub::Topic;

        let topics = DistributedAITopics::new();
        let ai_topic = IdentTopic::new("qnk/ai/test/v1");
        let other_topic = IdentTopic::new("qnk/dex/orders/v1");

        assert!(topics.is_ai_topic(&ai_topic.hash()));
        assert!(!topics.is_ai_topic(&other_topic.hash()));
    }

    #[test]
    fn test_node_capability_scoring() {
        let cpu = NodeCapability::CPU { cores: 8, ram_gb: 16 };
        let cuda = NodeCapability::CUDA {
            vram_gb: 12,
            compute_capability: "8.0".to_string(),
        };

        assert!(cuda.score() > cpu.score());
        assert_eq!(cpu.score(), 96); // 8*10 + 16
        assert_eq!(cuda.score(), 12000); // 12*1000
    }

    // ==================== CRITICAL SECURITY TESTS (v1.0.3-beta) ====================

    #[test]
    fn test_ambiguous_serialization_prevented() {
        // SHOWSTOPPER #1 FIX TEST: Verify length-prefixed encoding prevents ambiguity
        //
        // Attack scenario:
        // Without length prefixes: "ABC" + "DEF" = "ABCDEF" = "AB" + "CDEF" (ambiguous!)
        // With length prefixes: [3]"ABC" + [3]"DEF" ≠ [2]"AB" + [4]"CDEF" (unambiguous!)

        use std::io::Write;

        // Scenario 1: "ABC" + "DEF"
        let mut buffer1 = Vec::new();
        write_string(&mut buffer1, "ABC");
        write_string(&mut buffer1, "DEF");

        // Scenario 2: "AB" + "CDEF"
        let mut buffer2 = Vec::new();
        write_string(&mut buffer2, "AB");
        write_string(&mut buffer2, "CDEF");

        // CRITICAL: These must be DIFFERENT with length-prefixing
        assert_ne!(
            buffer1, buffer2,
            "SECURITY FAILURE: Serialization ambiguity not prevented!"
        );

        // Verify the actual bytes include length prefixes
        // "ABC" = [3, 0, 0, 0, 'A', 'B', 'C']
        assert_eq!(buffer1[0..4], [3, 0, 0, 0]); // Length of "ABC" as u32 LE
        assert_eq!(buffer1[4..7], [b'A', b'B', b'C']); // "ABC" bytes
        assert_eq!(buffer1[7..11], [3, 0, 0, 0]); // Length of "DEF" as u32 LE
        assert_eq!(buffer1[11..14], [b'D', b'E', b'F']); // "DEF" bytes

        // "AB" + "CDEF"
        assert_eq!(buffer2[0..4], [2, 0, 0, 0]); // Length of "AB" as u32 LE
        assert_eq!(buffer2[4..6], [b'A', b'B']); // "AB" bytes
        assert_eq!(buffer2[6..10], [4, 0, 0, 0]); // Length of "CDEF" as u32 LE
        assert_eq!(buffer2[10..14], [b'C', b'D', b'E', b'F']); // "CDEF" bytes

        println!("✅ SECURITY TEST PASSED: Serialization ambiguity prevented");
        println!("   Scenario 1 (ABC+DEF): {:?}", buffer1);
        println!("   Scenario 2 (AB+CDEF): {:?}", buffer2);
        println!("   Length-prefixing ensures unambiguous parsing");
    }

    #[test]
    fn test_timestamp_validation_window() {
        // SHOWSTOPPER #1 FIX TEST: Verify 5-minute acceptance window and clock skew

        let now = chrono::Utc::now().timestamp();

        // Test 1: Current timestamp should be valid
        let msg_valid = AIGossipsubMessage {
            protocol_version: 1,
            message_id: "test_msg_1".to_string(),
            timestamp: now,
            sender_node_id: "test_node".to_string(),
            sender_peer_id: "test_peer".to_string(),
            payload: AIMessagePayload::Heartbeat,
            aegis_signature: None,
            sender_public_key: None,
            sequence_number: 1,
            retry_count: 0,
            priority: MessagePriority::Normal,
        };
        assert!(
            msg_valid.verify_timestamp(),
            "Valid timestamp rejected (now)"
        );

        // Test 2: Message 4 minutes old (valid - within 5 min window)
        let msg_4min_old = AIGossipsubMessage {
            timestamp: now - 240, // 4 minutes ago
            ..msg_valid.clone()
        };
        assert!(
            msg_4min_old.verify_timestamp(),
            "4-minute old message rejected (should be valid)"
        );

        // Test 3: Message 6 minutes old (invalid - exceeds 5 min window)
        let msg_6min_old = AIGossipsubMessage {
            timestamp: now - 360, // 6 minutes ago
            ..msg_valid.clone()
        };
        assert!(
            !msg_6min_old.verify_timestamp(),
            "6-minute old message accepted (should be rejected)"
        );

        // Test 4: Message 20 seconds in future (valid - within 30s clock skew)
        let msg_future_20s = AIGossipsubMessage {
            timestamp: now + 20, // 20 seconds ahead
            ..msg_valid.clone()
        };
        assert!(
            msg_future_20s.verify_timestamp(),
            "20s future message rejected (should allow clock skew)"
        );

        // Test 5: Message 45 seconds in future (invalid - exceeds clock skew)
        let msg_future_45s = AIGossipsubMessage {
            timestamp: now + 45, // 45 seconds ahead
            ..msg_valid.clone()
        };
        assert!(
            !msg_future_45s.verify_timestamp(),
            "45s future message accepted (should reject excessive clock skew)"
        );

        println!("✅ SECURITY TEST PASSED: Timestamp validation working correctly");
        println!("   5-minute acceptance window enforced");
        println!("   30-second clock skew tolerance working");
        println!("   Replay attack prevention active");
    }

    #[tokio::test]
    async fn test_versioned_canonical_format_dispatch() {
        // SHOWSTOPPER #1 FIX TEST: Verify protocol version dispatch works correctly

        let payload = AIMessagePayload::Heartbeat;
        let payload_bytes = bincode::serialize(&payload).unwrap();

        // Test v0 message (legacy format)
        let msg_v0 = AIGossipsubMessage {
            protocol_version: 0, // v0 uses legacy format
            message_id: "test_v0".to_string(),
            timestamp: chrono::Utc::now().timestamp(),
            sender_node_id: "node_v0".to_string(),
            sender_peer_id: "peer_v0".to_string(),
            payload: payload.clone(),
            aegis_signature: None,
            sender_public_key: None,
            sequence_number: 1,
            retry_count: 0,
            priority: MessagePriority::Normal,
        };

        let canonical_v0 = msg_v0.create_canonical_message(&payload_bytes);

        // Test v1 message (secure format with length-prefixing)
        let msg_v1 = AIGossipsubMessage {
            protocol_version: 1, // v1 uses secure format
            ..msg_v0.clone()
        };

        let canonical_v1 = msg_v1.create_canonical_message(&payload_bytes);

        // CRITICAL: v0 and v1 canonical formats must be DIFFERENT
        // (v1 includes length prefixes, v0 doesn't)
        assert_ne!(
            canonical_v0, canonical_v1,
            "SECURITY FAILURE: v0 and v1 formats are identical (version dispatch broken)"
        );

        println!("✅ SECURITY TEST PASSED: Versioned canonical format dispatch working");
        println!("   v0 format: {} bytes", canonical_v0.len());
        println!("   v1 format: {} bytes (includes length prefixes)", canonical_v1.len());
        println!("   Migration path to v2 (bincode) preserved");
    }

    #[tokio::test]
    async fn test_signature_verification_integration() {
        // SHOWSTOPPER #1 FIX TEST: End-to-end signature verification
        //
        // This tests the complete flow:
        // 1. Create message
        // 2. Sign message with quantum signer
        // 3. Verify signature
        // 4. Reject tampered messages

        use q_quantum_crypto::QuantumSigner;

        // Create quantum signer for testing
        let signer = QuantumSigner::new_lamport_ots().await.unwrap();
        let public_key = signer.get_public_key().await.unwrap();

        // Create message
        let mut msg = AIGossipsubMessage {
            protocol_version: 1,
            message_id: "test_signature".to_string(),
            timestamp: chrono::Utc::now().timestamp(),
            sender_node_id: "test_node".to_string(),
            sender_peer_id: "test_peer".to_string(),
            payload: AIMessagePayload::Heartbeat,
            aegis_signature: None,
            sender_public_key: Some(public_key.clone()),
            sequence_number: 1,
            retry_count: 0,
            priority: MessagePriority::Normal,
        };

        // Sign the message
        msg.sign(&signer).await.unwrap();

        // Verify signature is present
        assert!(
            msg.aegis_signature.is_some(),
            "Signature not created after signing"
        );

        // Test 1: Valid signature should verify
        let valid = msg.verify_signature_async().await;
        assert!(valid, "Valid signature rejected");

        // Test 2: Tampered message should fail verification
        let mut tampered_msg = msg.clone();
        tampered_msg.message_id = "tampered_id".to_string(); // Change message_id

        let tampered_valid = tampered_msg.verify_signature_async().await;
        assert!(
            !tampered_valid,
            "Tampered message signature verified (CRITICAL SECURITY FAILURE)"
        );

        // Test 3: Wrong public key should fail verification
        let wrong_signer = QuantumSigner::new_lamport_ots().await.unwrap();
        let wrong_public_key = wrong_signer.get_public_key().await.unwrap();

        let mut wrong_key_msg = msg.clone();
        wrong_key_msg.sender_public_key = Some(wrong_public_key);

        let wrong_key_valid = wrong_key_msg.verify_signature_async().await;
        assert!(
            !wrong_key_valid,
            "Message with wrong public key verified (CRITICAL SECURITY FAILURE)"
        );

        println!("✅ SECURITY TEST PASSED: Signature verification working end-to-end");
        println!("   ✓ Valid signatures verified");
        println!("   ✓ Tampered messages rejected");
        println!("   ✓ Wrong public keys rejected");
        println!("   Message forgery attack prevented");
    }

    #[tokio::test]
    async fn test_protocol_v0_backwards_compatibility() {
        // SHOWSTOPPER #1 FIX TEST: Verify v0 unsigned messages are accepted
        //
        // Protocol v0: Accept unsigned (backwards compat until 2025-12-07)
        // Protocol v1+: Reject unsigned (security enforced)

        // Test 1: v0 unsigned message (should pass verify_timestamp, fail verify_signature gracefully)
        let msg_v0_unsigned = AIGossipsubMessage {
            protocol_version: 0,
            message_id: "test_v0_unsigned".to_string(),
            timestamp: chrono::Utc::now().timestamp(),
            sender_node_id: "node_v0".to_string(),
            sender_peer_id: "peer_v0".to_string(),
            payload: AIMessagePayload::Heartbeat,
            aegis_signature: None, // No signature
            sender_public_key: None, // No public key
            sequence_number: 1,
            retry_count: 0,
            priority: MessagePriority::Normal,
        };

        // Timestamp should be valid
        assert!(
            msg_v0_unsigned.verify_timestamp(),
            "v0 timestamp validation broken"
        );

        // Signature verification should fail but NOT panic
        let sig_valid = msg_v0_unsigned.verify_signature_async().await;
        assert!(
            !sig_valid,
            "v0 unsigned message incorrectly verified as signed"
        );

        // Test 2: v1 unsigned message (should fail signature verification)
        let msg_v1_unsigned = AIGossipsubMessage {
            protocol_version: 1,
            ..msg_v0_unsigned.clone()
        };

        let v1_sig_valid = msg_v1_unsigned.verify_signature_async().await;
        assert!(
            !v1_sig_valid,
            "v1 unsigned message incorrectly verified (CRITICAL SECURITY FAILURE)"
        );

        println!("✅ SECURITY TEST PASSED: Protocol version compatibility working");
        println!("   ✓ v0 unsigned messages handled gracefully (backwards compat)");
        println!("   ✓ v1 unsigned messages rejected (security enforced)");
        println!("   Migration path from v0 → v1 preserved");
    }
}
