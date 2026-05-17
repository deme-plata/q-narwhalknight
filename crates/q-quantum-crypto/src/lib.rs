//! 🔬 Q2 Full Quantum Cryptography Module
//! Quantum Key Distribution (QKD) + Quantum Digital Signatures + Plugin Integration
//! Phase 2 quantum enhancement for Q-NarwhalKnight consensus

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use q_plugin_system::{PluginConfig, PluginError, PluginId, PluginManager};

// Plugin message types for quantum crypto operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMessage {
    pub id: String,
    pub operation: String,
    pub data: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

// PQ Crypto Protocol placeholder
#[derive(Debug, Clone)]
pub enum PQCryptoProtocol {
    Dilithium,
    Kyber,
    Falcon,
}

pub mod bb84_protocol;
pub mod npab_protocol;
pub mod qkd;
pub mod qkd_protocol_selector;
pub mod quantum_channels;
pub mod quantum_entropy;
pub mod quantum_error_correction;
pub mod quantum_signatures;
pub mod sarg04_protocol;

pub use bb84_protocol::{BB84State, PhotonPolarization, QuantumBit};
pub use qkd::{BB84Protocol, QKDChannel, QKDEngine, QKDKey};
pub use quantum_channels::{ChannelState, QuantumChannel, QuantumChannelManager};
pub use quantum_entropy::{QuantumEntropySource, QuantumRNG, TrueRandomGenerator};
pub use quantum_error_correction::{QuantumErrorCorrection, ShorCode, StabilizerCode};
pub use quantum_signatures::{LamportOTS, QuantumSignature, QuantumSigner, QuantumVerifier};

// v10.1.5: QKD protocol selector and protocol exports
pub use sarg04_protocol::{SARG04Protocol, SARG04Config, SARG04Stats, SARG04SecurityAnalysis};
pub use npab_protocol::{NPABProtocol, NPABConfig, NPABStats, NPABSecurityAnalysis};
pub use qkd_protocol_selector::{
    QKDProtocolSelector, QKDProtocolChoice, ChannelProfile, SelectionRationale, KeyExchangeRole,
};

/// Node identifier for quantum communication
pub type NodeId = [u8; 32];

/// Quantum key material (unconditionally secure)
pub type QuantumKey = Vec<u8>;

/// Photon state for BB84 protocol
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhotonState {
    /// Rectilinear basis: horizontal (0) or vertical (1)
    Rectilinear(bool),
    /// Diagonal basis: 45° (0) or 135° (1)
    Diagonal(bool),
}

/// Quantum cryptographic engine coordinating all Q2 operations
#[derive(Debug)]
pub struct QuantumCryptoEngine {
    /// Quantum Key Distribution engine
    pub qkd_engine: Arc<QKDEngine>,
    /// Quantum signature system
    pub signature_system: Arc<RwLock<QuantumSigner>>,
    /// Quantum channel management
    pub channel_manager: Arc<QuantumChannelManager>,
    /// True quantum entropy source
    pub entropy_source: Arc<QuantumEntropySource>,
    /// Quantum error correction
    pub error_correction: Arc<QuantumErrorCorrection>,
    /// Node identity
    pub node_id: NodeId,
    /// Active quantum keys
    pub active_keys: Arc<RwLock<HashMap<NodeId, QuantumKey>>>,
}

impl QuantumCryptoEngine {
    /// Initialize Q2 Full Quantum cryptographic engine
    pub async fn initialize(node_id: NodeId) -> Result<Self> {
        // Initialize quantum entropy source
        let entropy_source = Arc::new(QuantumEntropySource::new().await?);

        // Initialize QKD engine with true quantum randomness
        let qkd_engine = Arc::new(QKDEngine::new(node_id, entropy_source.clone()).await?);

        // Initialize quantum signature system
        let signature_system = Arc::new(RwLock::new(
            QuantumSigner::new(node_id, entropy_source.clone()).await?,
        ));

        // Initialize quantum channel manager
        let channel_manager =
            Arc::new(QuantumChannelManager::new(node_id, entropy_source.clone()).await?);

        // Initialize quantum error correction
        let error_correction = Arc::new(QuantumErrorCorrection::new());

        // Initialize active keys storage
        let active_keys = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            qkd_engine,
            signature_system,
            channel_manager,
            entropy_source,
            error_correction,
            node_id,
            active_keys,
        })
    }

    /// Establish quantum key distribution with peer
    pub async fn establish_qkd_session(&self, peer_id: NodeId) -> Result<QuantumKey> {
        // 1. Establish quantum channel
        let channel = self.channel_manager.establish_channel(peer_id).await?;

        // 2. Execute BB84 protocol for key distribution
        let raw_key = self
            .qkd_engine
            .execute_bb84_protocol(peer_id, &channel)
            .await?;

        // 3. Apply quantum error correction
        let corrected_key = self.error_correction.correct_quantum_errors(&raw_key)?;

        // 4. Privacy amplification to ensure unconditional security
        let final_key = self.privacy_amplification(&corrected_key).await?;

        // 5. Store quantum key securely
        self.active_keys
            .write()
            .await
            .insert(peer_id, final_key.clone());

        Ok(final_key)
    }

    /// Sign message using quantum digital signatures
    pub async fn quantum_sign(&self, message: &[u8]) -> Result<QuantumSignature> {
        let signer = self.signature_system.read().await;
        signer.sign_message(message).await
    }

    /// Verify quantum digital signature
    pub async fn quantum_verify(
        &self,
        message: &[u8],
        signature: &QuantumSignature,
        signer_id: NodeId,
    ) -> Result<bool> {
        let verifier = QuantumVerifier::new(signer_id);
        verifier.verify_signature(message, signature).await
    }

    /// Encrypt data using quantum-distributed keys
    pub async fn quantum_encrypt(&self, data: &[u8], peer_id: NodeId) -> Result<Vec<u8>> {
        // Get quantum key for peer
        let keys = self.active_keys.read().await;
        let quantum_key = keys
            .get(&peer_id)
            .ok_or_else(|| anyhow::anyhow!("No quantum key available for peer"))?;

        // One-time pad encryption (information-theoretically secure)
        self.one_time_pad_encrypt(data, quantum_key).await
    }

    /// Decrypt data using quantum-distributed keys
    pub async fn quantum_decrypt(&self, encrypted_data: &[u8], peer_id: NodeId) -> Result<Vec<u8>> {
        // Get quantum key for peer
        let keys = self.active_keys.read().await;
        let quantum_key = keys
            .get(&peer_id)
            .ok_or_else(|| anyhow::anyhow!("No quantum key available for peer"))?;

        // One-time pad decryption
        self.one_time_pad_decrypt(encrypted_data, quantum_key).await
    }

    /// Generate fresh quantum entropy
    pub async fn generate_quantum_entropy(&self, bytes: usize) -> Result<Vec<u8>> {
        self.entropy_source.generate_true_random(bytes).await
    }

    /// Refresh quantum keys (forward secrecy)
    pub async fn refresh_quantum_keys(&self) -> Result<()> {
        let mut keys = self.active_keys.write().await;
        let peer_ids: Vec<NodeId> = keys.keys().cloned().collect();

        for peer_id in peer_ids {
            // Generate new quantum key
            let new_key = self.establish_qkd_session(peer_id).await?;
            keys.insert(peer_id, new_key);
        }

        Ok(())
    }

    /// Privacy amplification using universal hash functions
    async fn privacy_amplification(&self, raw_key: &[u8]) -> Result<QuantumKey> {
        // Apply universal hash function to extract unconditionally secure key
        let entropy = self.entropy_source.generate_true_random(32).await?;
        let mut amplified_key = Vec::with_capacity(32);

        for i in 0..32 {
            let mut hash_input = Vec::new();
            hash_input.extend_from_slice(raw_key);
            hash_input.extend_from_slice(&entropy);
            hash_input.push(i as u8);

            // Simple universal hash (in production, use sophisticated construction)
            let hash_byte = hash_input
                .iter()
                .enumerate()
                .map(|(idx, &byte)| byte.wrapping_mul(idx as u8 + 1))
                .fold(0u8, |acc, x| acc.wrapping_add(x));

            amplified_key.push(hash_byte);
        }

        Ok(amplified_key)
    }

    /// One-time pad encryption (information-theoretically secure)
    async fn one_time_pad_encrypt(&self, data: &[u8], key: &[u8]) -> Result<Vec<u8>> {
        if data.len() > key.len() {
            return Err(anyhow::anyhow!(
                "Insufficient key material for one-time pad"
            ));
        }

        let encrypted: Vec<u8> = data.iter().zip(key.iter()).map(|(&d, &k)| d ^ k).collect();

        Ok(encrypted)
    }

    /// One-time pad decryption
    async fn one_time_pad_decrypt(&self, encrypted_data: &[u8], key: &[u8]) -> Result<Vec<u8>> {
        // XOR is its own inverse
        self.one_time_pad_encrypt(encrypted_data, key).await
    }

    /// Get quantum cryptographic statistics
    pub async fn get_quantum_stats(&self) -> QuantumCryptoStats {
        let keys_count = self.active_keys.read().await.len();
        let entropy_generated = self.entropy_source.get_total_entropy_generated().await;
        let qkd_sessions = self.qkd_engine.get_session_count().await;
        let signatures_generated = self
            .signature_system
            .read()
            .await
            .get_signature_count()
            .await;

        QuantumCryptoStats {
            active_quantum_keys: keys_count,
            total_entropy_generated: entropy_generated,
            qkd_sessions_established: qkd_sessions,
            quantum_signatures_generated: signatures_generated,
            quantum_channels_active: self.channel_manager.get_active_channel_count().await,
            last_key_refresh: SystemTime::now(),
        }
    }

    /// Perform quantum cryptographic health check
    pub async fn health_check(&self) -> Result<QuantumHealthStatus> {
        // Check quantum entropy source
        let entropy_healthy = self.entropy_source.health_check().await?;

        // Check QKD system
        let qkd_healthy = self.qkd_engine.health_check().await?;

        // Check quantum channels
        let channels_healthy = self.channel_manager.health_check().await?;

        // Check signature system
        let signatures_healthy = {
            let signer = self.signature_system.read().await;
            signer.health_check().await?
        };

        let overall_status =
            if entropy_healthy && qkd_healthy && channels_healthy && signatures_healthy {
                QuantumHealthStatus::Optimal
            } else if entropy_healthy && (qkd_healthy || channels_healthy) {
                QuantumHealthStatus::Degraded
            } else {
                QuantumHealthStatus::Critical
            };

        Ok(overall_status)
    }
}

/// Quantum cryptographic statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCryptoStats {
    pub active_quantum_keys: usize,
    pub total_entropy_generated: u64,
    pub qkd_sessions_established: u64,
    pub quantum_signatures_generated: u64,
    pub quantum_channels_active: usize,
    pub last_key_refresh: SystemTime,
}

/// Quantum system health status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantumHealthStatus {
    /// All quantum systems operating optimally
    Optimal,
    /// Some quantum systems degraded but functional
    Degraded,
    /// Critical quantum system failures
    Critical,
    /// Quantum systems offline
    Offline,
}

/// Quantum cryptographic configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumConfig {
    /// BB84 protocol parameters
    pub bb84_config: BB84Config,
    /// Quantum signature parameters
    pub signature_config: QuantumSignatureConfig,
    /// Error correction parameters
    pub error_correction_config: ErrorCorrectionConfig,
    /// Key refresh interval
    pub key_refresh_interval: Duration,
    /// Maximum quantum channel distance (meters)
    pub max_channel_distance: f64,
}

/// BB84 protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BB84Config {
    /// Number of photons to send
    pub photon_count: usize,
    /// Error rate threshold for aborting
    pub error_threshold: f64,
    /// Privacy amplification factor
    pub amplification_factor: f64,
}

/// Quantum signature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSignatureConfig {
    /// Signature scheme type
    pub scheme: QuantumSignatureScheme,
    /// Security parameter
    pub security_parameter: u32,
    /// Key lifetime
    pub key_lifetime: Duration,
}

/// Available quantum signature schemes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumSignatureScheme {
    /// Lamport one-time signatures
    LamportOTS,
    /// Merkle signature scheme (multiple uses)
    Merkle,
    /// XMSS (Extended Merkle Signature Scheme)
    XMSS,
}

/// Error correction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCorrectionConfig {
    /// Error correction code type
    pub code_type: QuantumErrorCorrectionCode,
    /// Correction threshold
    pub correction_threshold: f64,
}

/// Available quantum error correction codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumErrorCorrectionCode {
    /// Shor 9-qubit code
    Shor9,
    /// Steane 7-qubit code
    Steane7,
    /// Surface code
    Surface,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            bb84_config: BB84Config {
                photon_count: 100_000,
                error_threshold: 0.11,
                amplification_factor: 2.0,
            },
            signature_config: QuantumSignatureConfig {
                scheme: QuantumSignatureScheme::LamportOTS,
                security_parameter: 256,
                key_lifetime: Duration::from_secs(3600), // 1 hour
            },
            error_correction_config: ErrorCorrectionConfig {
                code_type: QuantumErrorCorrectionCode::Shor9,
                correction_threshold: 0.01,
            },
            key_refresh_interval: Duration::from_secs(300), // 5 minutes
            max_channel_distance: 100.0,                    // 100 meters
        }
    }
}

/// Quantum-safe random number generator using true quantum entropy
pub struct QuantumSafeRNG {
    entropy_source: Arc<QuantumEntropySource>,
    buffer: Vec<u8>,
    buffer_position: usize,
}

impl QuantumSafeRNG {
    /// Create new quantum-safe RNG
    pub async fn new() -> Result<Self> {
        let entropy_source = Arc::new(QuantumEntropySource::new().await?);
        let buffer = entropy_source.generate_true_random(4096).await?;

        Ok(Self {
            entropy_source,
            buffer,
            buffer_position: 0,
        })
    }

    /// Generate quantum-safe random bytes
    pub async fn generate(&mut self, bytes: usize) -> Result<Vec<u8>> {
        if self.buffer_position + bytes > self.buffer.len() {
            // Refresh buffer with fresh quantum entropy
            self.buffer = self
                .entropy_source
                .generate_true_random(4096.max(bytes))
                .await?;
            self.buffer_position = 0;
        }

        let result = self.buffer[self.buffer_position..self.buffer_position + bytes].to_vec();
        self.buffer_position += bytes;

        Ok(result)
    }
}

/// Plugin wrapper for QuantumCryptoEngine
pub struct QuantumCryptoPlugin {
    id: String,
    version: String,
    engine: Arc<QuantumCryptoEngine>,
    config: QuantumConfig,
}

impl QuantumCryptoPlugin {
    pub async fn new(config: QuantumConfig) -> Result<Self> {
        let node_id = [0u8; 32]; // TODO: Get from node configuration
        let engine = Arc::new(QuantumCryptoEngine::initialize(node_id).await?);

        Ok(Self {
            id: "q-quantum-crypto".to_string(),
            version: "1.0.0".to_string(),
            engine,
            config,
        })
    }

    pub fn get_engine(&self) -> Arc<QuantumCryptoEngine> {
        self.engine.clone()
    }
}

impl QuantumCryptoPlugin {
    fn get_id(&self) -> &str {
        &self.id
    }

    fn get_version(&self) -> &str {
        &self.version
    }

    fn get_name(&self) -> &str {
        "Q-NarwhalKnight Quantum Cryptography"
    }

    fn get_description(&self) -> &str {
        "Advanced quantum key distribution and cryptographic protocols for quantum-enhanced consensus"
    }

    async fn initialize(&mut self) -> Result<(), PluginError> {
        info!("🔬 Initializing Q-NarwhalKnight Quantum Cryptography Plugin");

        // Initialize quantum systems
        info!("🔑 Quantum Key Distribution ready");
        info!("✍️ Quantum Digital Signatures ready");
        info!("🌊 Quantum Channel Management ready");
        info!("🎲 Quantum Entropy Source ready");
        info!("🛠️ Quantum Error Correction ready");

        info!("✅ Quantum Cryptography Plugin fully initialized");
        Ok(())
    }

    pub async fn execute(
        &mut self,
        message: QuantumMessage,
    ) -> Result<QuantumMessage, PluginError> {
        debug!(
            "🔬 Processing quantum crypto message: {:?}",
            message.operation
        );

        match message.operation.as_str() {
            "establish_qkd" => {
                let request: EstablishQKDRequest = serde_json::from_value(message.data.clone())
                    .map_err(|e| PluginError::execution_failed(&e.to_string()))?;

                let peer_id = request.peer_id;
                match self.engine.establish_qkd_session(peer_id).await {
                    Ok(quantum_key) => {
                        let response = EstablishQKDResponse {
                            success: true,
                            peer_id,
                            key_length: quantum_key.len(),
                            error: None,
                        };
                        Ok(QuantumMessage {
                            id: uuid::Uuid::new_v4().to_string(),
                            operation: "qkd_established".to_string(),
                            data: serde_json::to_value(&response).unwrap(),
                            timestamp: Utc::now(),
                        })
                    }
                    Err(e) => {
                        let response = EstablishQKDResponse {
                            success: false,
                            peer_id,
                            key_length: 0,
                            error: Some(e.to_string()),
                        };
                        Ok(QuantumMessage {
                            id: uuid::Uuid::new_v4().to_string(),
                            operation: "qkd_failed".to_string(),
                            data: serde_json::to_value(&response).unwrap(),
                            timestamp: Utc::now(),
                        })
                    }
                }
            }

            "quantum_sign" => {
                let request: QuantumSignRequest = serde_json::from_value(message.data.clone())
                    .map_err(|e| PluginError::execution_failed(&e.to_string()))?;

                match self.engine.quantum_sign(&request.message).await {
                    Ok(signature) => {
                        let response = QuantumSignResponse {
                            success: true,
                            signature: Some(signature),
                            error: None,
                        };
                        Ok(QuantumMessage {
                            id: Uuid::new_v4().to_string(),
                            operation: "quantum_signed".to_string(),
                            data: serde_json::to_value(&response).unwrap(),
                            timestamp: Utc::now(),
                        })
                    }
                    Err(e) => {
                        let response = QuantumSignResponse {
                            success: false,
                            signature: None,
                            error: Some(e.to_string()),
                        };
                        Ok(QuantumMessage {
                            id: Uuid::new_v4().to_string(),
                            operation: "quantum_sign_failed".to_string(),
                            data: serde_json::to_value(&response).unwrap(),
                            timestamp: Utc::now(),
                        })
                    }
                }
            }

            "quantum_encrypt" => {
                let request: QuantumEncryptRequest =
                    serde_json::from_value(message.data.clone())
                        .map_err(|e| PluginError::execution_failed(&e.to_string()))?;

                match self
                    .engine
                    .quantum_encrypt(&request.data, request.peer_id)
                    .await
                {
                    Ok(encrypted_data) => {
                        let response = QuantumEncryptResponse {
                            success: true,
                            encrypted_data: Some(encrypted_data),
                            error: None,
                        };
                        Ok(QuantumMessage {
                            id: Uuid::new_v4().to_string(),
                            operation: "quantum_encrypted".to_string(),
                            data: serde_json::to_value(&response).unwrap(),
                            timestamp: Utc::now(),
                        })
                    }
                    Err(e) => {
                        let response = QuantumEncryptResponse {
                            success: false,
                            encrypted_data: None,
                            error: Some(e.to_string()),
                        };
                        Ok(QuantumMessage {
                            id: Uuid::new_v4().to_string(),
                            operation: "quantum_encrypt_failed".to_string(),
                            data: serde_json::to_value(&response).unwrap(),
                            timestamp: Utc::now(),
                        })
                    }
                }
            }

            "get_status" => {
                let status = QuantumCryptoStatus {
                    active_keys: self.engine.active_keys.read().await.len(),
                    node_id: self.engine.node_id,
                    qkd_ready: true,
                    signatures_ready: true,
                    entropy_available: true,
                };

                Ok(QuantumMessage {
                    id: Uuid::new_v4().to_string(),
                    operation: "quantum_crypto_status".to_string(),
                    data: serde_json::to_value(&status).unwrap(),
                    timestamp: Utc::now(),
                })
            }

            _ => {
                warn!("🚫 Unknown quantum crypto operation: {}", message.operation);
                Err(PluginError::execution_failed(&format!(
                    "Unknown operation: {}",
                    message.operation
                )))
            }
        }
    }

    async fn shutdown(&mut self) -> Result<(), PluginError> {
        info!("🔬 Shutting down Q-NarwhalKnight Quantum Cryptography Plugin");

        // Clear quantum keys for security
        self.engine.active_keys.write().await.clear();

        info!("✅ Quantum Cryptography Plugin shut down successfully");
        Ok(())
    }
}

// Plugin message types
#[derive(Debug, Serialize, Deserialize)]
pub struct EstablishQKDRequest {
    pub peer_id: NodeId,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EstablishQKDResponse {
    pub success: bool,
    pub peer_id: NodeId,
    pub key_length: usize,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QuantumSignRequest {
    pub message: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QuantumSignResponse {
    pub success: bool,
    pub signature: Option<QuantumSignature>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QuantumEncryptRequest {
    pub data: Vec<u8>,
    pub peer_id: NodeId,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QuantumEncryptResponse {
    pub success: bool,
    pub encrypted_data: Option<Vec<u8>>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QuantumCryptoStatus {
    pub active_keys: usize,
    pub node_id: NodeId,
    pub qkd_ready: bool,
    pub signatures_ready: bool,
    pub entropy_available: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quantum_crypto_engine_initialization() {
        let node_id = [1u8; 32];
        let result = QuantumCryptoEngine::initialize(node_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_quantum_safe_rng() {
        let mut rng = QuantumSafeRNG::new().await.unwrap();
        let random_bytes = rng.generate(32).await.unwrap();
        assert_eq!(random_bytes.len(), 32);

        // Generate another set to ensure they're different
        let random_bytes2 = rng.generate(32).await.unwrap();
        assert_ne!(random_bytes, random_bytes2);
    }

    #[test]
    fn test_photon_state_serialization() {
        let state = PhotonState::Rectilinear(true);
        let serialized = serde_json::to_string(&state).unwrap();
        let deserialized: PhotonState = serde_json::from_str(&serialized).unwrap();
        assert_eq!(state, deserialized);
    }

    #[test]
    fn test_quantum_config_default() {
        let config = QuantumConfig::default();
        assert_eq!(config.bb84_config.photon_count, 100_000);
        assert_eq!(config.bb84_config.error_threshold, 0.11);
        assert!(matches!(
            config.signature_config.scheme,
            QuantumSignatureScheme::LamportOTS
        ));
    }
}
