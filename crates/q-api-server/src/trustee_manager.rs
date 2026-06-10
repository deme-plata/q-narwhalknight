//! Trustee Manager - HSM-backed key management for TemporalShield
//!
//! Provides centralized management of trustee keys with HSM integration.
//! System trustees are pre-generated and stored in HSM slots.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use q_temporal_shield::{
    TrusteePublicKey,
    trustee::hsm::{HsmSimulator, DistributedHsmNetwork},
};
use tracing::{info, warn, error};

/// Rotation policy for trustee keys
#[derive(Debug, Clone)]
pub struct RotationPolicy {
    /// Rotation interval in seconds (0 = never rotate)
    pub interval_secs: u64,
    /// Number of epochs between rotations
    pub epochs_between_rotations: u64,
    /// Last rotation timestamp
    pub last_rotation: u64,
}

impl Default for RotationPolicy {
    fn default() -> Self {
        Self {
            interval_secs: 86400 * 30, // 30 days
            epochs_between_rotations: 1000,
            last_rotation: 0,
        }
    }
}

/// Trustee configuration for different use cases
#[derive(Debug, Clone)]
pub struct TrusteeConfig {
    /// Threshold (k) - minimum shares needed
    pub threshold: usize,
    /// Total trustees (n)
    pub total: usize,
    /// Human-readable name
    pub name: String,
}

impl TrusteeConfig {
    pub fn memo() -> Self {
        Self { threshold: 3, total: 5, name: "Private TX Memos".to_string() }
    }

    pub fn validator() -> Self {
        Self { threshold: 5, total: 9, name: "Validator Key Backup".to_string() }
    }

    pub fn chat() -> Self {
        Self { threshold: 3, total: 5, name: "AI Chat Archive".to_string() }
    }

    pub fn oracle() -> Self {
        Self { threshold: 2, total: 3, name: "Oracle Commit-Reveal".to_string() }
    }

    pub fn bank() -> Self {
        Self { threshold: 3, total: 5, name: "Bank Audit Trail".to_string() }
    }

    pub fn ai_tensors() -> Self {
        Self { threshold: 3, total: 5, name: "Distributed AI Tensors".to_string() }
    }
}

/// System trustee stored in HSM
#[derive(Clone)]
pub struct SystemTrustee {
    pub id: [u8; 32],
    pub public_key: TrusteePublicKey,
    pub slot_id: usize,
}

/// Centralized trustee manager with HSM integration
pub struct TrusteeManager {
    /// Primary HSM for system trustee keys
    hsm: HsmSimulator,

    /// Distributed HSM network (optional, for production)
    distributed_network: Option<DistributedHsmNetwork>,

    /// System trustees by purpose
    memo_trustees: Vec<SystemTrustee>,
    validator_trustees: Vec<SystemTrustee>,
    chat_trustees: Vec<SystemTrustee>,
    oracle_trustees: Vec<SystemTrustee>,
    bank_trustees: Vec<SystemTrustee>,
    ai_tensor_trustees: Vec<SystemTrustee>,

    /// User-specific trustee configurations
    user_trustees: Arc<RwLock<HashMap<String, Vec<TrusteePublicKey>>>>,

    /// Rotation policy
    rotation_policy: RotationPolicy,

    /// Initialized flag
    initialized: bool,
}

impl TrusteeManager {
    /// Create a new TrusteeManager with HSM simulator
    pub fn new() -> Result<Self, String> {
        let hsm = HsmSimulator::new()
            .map_err(|e| format!("Failed to create HSM: {:?}", e))?;

        Ok(Self {
            hsm,
            distributed_network: None,
            memo_trustees: Vec::new(),
            validator_trustees: Vec::new(),
            chat_trustees: Vec::new(),
            oracle_trustees: Vec::new(),
            bank_trustees: Vec::new(),
            ai_tensor_trustees: Vec::new(),
            user_trustees: Arc::new(RwLock::new(HashMap::new())),
            rotation_policy: RotationPolicy::default(),
            initialized: false,
        })
    }

    /// Initialize system trustees (generates keys and stores in HSM)
    pub fn initialize(&mut self) -> Result<(), String> {
        if self.initialized {
            return Ok(());
        }

        info!("Initializing TrusteeManager with HSM-backed keys...");

        // Generate trustees for each use case
        self.memo_trustees = self.generate_trustees("memo", TrusteeConfig::memo().total)?;
        self.validator_trustees = self.generate_trustees("validator", TrusteeConfig::validator().total)?;
        self.chat_trustees = self.generate_trustees("chat", TrusteeConfig::chat().total)?;
        self.oracle_trustees = self.generate_trustees("oracle", TrusteeConfig::oracle().total)?;
        self.bank_trustees = self.generate_trustees("bank", TrusteeConfig::bank().total)?;
        self.ai_tensor_trustees = self.generate_trustees("ai_tensor", TrusteeConfig::ai_tensors().total)?;

        self.initialized = true;
        info!("TrusteeManager initialized with {} total HSM-backed keys",
            self.memo_trustees.len() +
            self.validator_trustees.len() +
            self.chat_trustees.len() +
            self.oracle_trustees.len() +
            self.bank_trustees.len() +
            self.ai_tensor_trustees.len()
        );

        Ok(())
    }

    /// Generate trustees for a specific purpose
    fn generate_trustees(&self, prefix: &str, count: usize) -> Result<Vec<SystemTrustee>, String> {
        let mut trustees = Vec::with_capacity(count);

        for i in 0..count {
            let name = format!("{}_{}", prefix, i);

            // Generate keypair
            let keypair = TrusteePublicKey::generate(Some(name.clone()))
                .map_err(|e| format!("Failed to generate trustee {}: {:?}", name, e))?;

            // Seal private key in HSM
            let key_id = keypair.public_key.id;

            // Serialize private key for HSM storage
            let mut private_key_bytes = Vec::new();
            private_key_bytes.extend_from_slice(&(keypair.private_key.kem_secret_key.len() as u32).to_le_bytes());
            private_key_bytes.extend_from_slice(&keypair.private_key.kem_secret_key);
            if let Some(ref sig_sk) = keypair.private_key.signature_secret_key {
                private_key_bytes.push(1); // has signature key
                private_key_bytes.extend_from_slice(&(sig_sk.len() as u32).to_le_bytes());
                private_key_bytes.extend_from_slice(sig_sk);
            } else {
                private_key_bytes.push(0); // no signature key
            }

            self.hsm.seal_key(key_id, &private_key_bytes)
                .map_err(|e| format!("Failed to seal key in HSM: {:?}", e))?;

            trustees.push(SystemTrustee {
                id: key_id,
                public_key: keypair.public_key,
                slot_id: i,
            });
        }

        info!("Generated {} {} trustees with HSM-sealed keys", count, prefix);
        Ok(trustees)
    }

    /// Get memo trustees public keys (3-of-5 threshold)
    pub fn get_memo_trustees(&self) -> Vec<TrusteePublicKey> {
        self.memo_trustees.iter()
            .map(|t| t.public_key.clone())
            .collect()
    }

    /// Get validator backup trustees public keys (5-of-9 threshold)
    pub fn get_validator_trustees(&self) -> Vec<TrusteePublicKey> {
        self.validator_trustees.iter()
            .map(|t| t.public_key.clone())
            .collect()
    }

    /// Get chat archive trustees public keys (3-of-5 threshold)
    pub fn get_chat_trustees(&self) -> Vec<TrusteePublicKey> {
        self.chat_trustees.iter()
            .map(|t| t.public_key.clone())
            .collect()
    }

    /// Get oracle commit-reveal trustees public keys (2-of-3 threshold)
    pub fn get_oracle_trustees(&self) -> Vec<TrusteePublicKey> {
        self.oracle_trustees.iter()
            .map(|t| t.public_key.clone())
            .collect()
    }

    /// Get bank audit trustees public keys (3-of-5 threshold)
    pub fn get_bank_trustees(&self) -> Vec<TrusteePublicKey> {
        self.bank_trustees.iter()
            .map(|t| t.public_key.clone())
            .collect()
    }

    /// Get AI tensor trustees public keys (3-of-5 threshold)
    pub fn get_ai_tensor_trustees(&self) -> Vec<TrusteePublicKey> {
        self.ai_tensor_trustees.iter()
            .map(|t| t.public_key.clone())
            .collect()
    }

    /// Decrypt a share using HSM (key never leaves HSM unencrypted)
    pub fn decrypt_share_via_hsm(
        &self,
        trustee_id: [u8; 32],
        kem_ciphertext: &[u8],
    ) -> Result<[u8; 32], String> {
        // Verify we have this trustee's key
        if !self.hsm.has_key(&trustee_id) {
            return Err(format!("Trustee key not found in HSM: {}", hex::encode(trustee_id)));
        }

        // Decrypt inside HSM
        self.hsm.decrypt_inside_hsm(trustee_id, kem_ciphertext)
            .map_err(|e| format!("HSM decryption failed: {:?}", e))
    }

    /// Register user-specific trustees (for custom thresholds)
    pub fn register_user_trustees(
        &self,
        user_address: &str,
        trustees: Vec<TrusteePublicKey>,
    ) -> Result<(), String> {
        let mut user_trustees = self.user_trustees.write()
            .map_err(|_| "Failed to acquire write lock")?;
        user_trustees.insert(user_address.to_string(), trustees);
        Ok(())
    }

    /// Get user-specific trustees (falls back to system trustees)
    pub fn get_user_memo_trustees(&self, user_address: &str) -> Vec<TrusteePublicKey> {
        if let Ok(user_trustees) = self.user_trustees.read() {
            if let Some(trustees) = user_trustees.get(user_address) {
                return trustees.clone();
            }
        }
        self.get_memo_trustees()
    }

    /// Check if key rotation is needed
    pub fn should_rotate(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        now - self.rotation_policy.last_rotation > self.rotation_policy.interval_secs
    }

    /// Get HSM audit log
    pub fn get_audit_log(&self) -> Vec<q_temporal_shield::trustee::hsm::HsmAuditEntry> {
        self.hsm.get_audit_log()
    }

    /// Get total number of HSM-backed keys
    pub fn total_keys(&self) -> usize {
        self.hsm.key_count()
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

impl Default for TrusteeManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default TrusteeManager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trustee_manager_initialization() {
        let mut manager = TrusteeManager::new().unwrap();
        assert!(!manager.is_initialized());

        manager.initialize().unwrap();
        assert!(manager.is_initialized());

        // Check all trustee groups are populated
        assert_eq!(manager.get_memo_trustees().len(), 5);
        assert_eq!(manager.get_validator_trustees().len(), 9);
        assert_eq!(manager.get_chat_trustees().len(), 5);
        assert_eq!(manager.get_oracle_trustees().len(), 3);
        assert_eq!(manager.get_bank_trustees().len(), 5);
        assert_eq!(manager.get_ai_tensor_trustees().len(), 5);
    }

    #[test]
    fn test_trustee_configs() {
        assert_eq!(TrusteeConfig::memo().threshold, 3);
        assert_eq!(TrusteeConfig::memo().total, 5);

        assert_eq!(TrusteeConfig::validator().threshold, 5);
        assert_eq!(TrusteeConfig::validator().total, 9);

        assert_eq!(TrusteeConfig::oracle().threshold, 2);
        assert_eq!(TrusteeConfig::oracle().total, 3);
    }

    #[test]
    fn test_hsm_key_count() {
        let mut manager = TrusteeManager::new().unwrap();
        assert_eq!(manager.total_keys(), 0);

        manager.initialize().unwrap();
        // 5 + 9 + 5 + 3 + 5 + 5 = 32 total keys
        assert_eq!(manager.total_keys(), 32);
    }
}
