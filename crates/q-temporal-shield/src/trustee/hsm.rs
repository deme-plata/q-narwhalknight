//! Hardware Security Module (HSM) simulator
//!
//! In production, this would interface with actual HSMs.
//! For development/testing, we simulate HSM behavior.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::crypto::{aead, hash, rand};
use crate::error::{TemporalError, TemporalResult};

/// HSM simulator for development and testing
pub struct HsmSimulator {
    /// Master key (in real HSM, this never leaves hardware)
    master_key: [u8; 32],

    /// Sealed keys: key_id -> encrypted_key
    sealed_keys: Arc<RwLock<HashMap<[u8; 32], Vec<u8>>>>,

    /// Audit log
    audit_log: Arc<RwLock<Vec<HsmAuditEntry>>>,
}

/// Audit log entry
#[derive(Debug, Clone)]
pub struct HsmAuditEntry {
    pub timestamp: u64,
    pub operation: HsmOperation,
    pub key_id: [u8; 32],
    pub success: bool,
}

/// HSM operations
#[derive(Debug, Clone)]
pub enum HsmOperation {
    SealKey,
    UnsealKey,
    Decrypt,
    Sign,
}

impl HsmSimulator {
    /// Create a new HSM simulator with a random master key
    pub fn new() -> TemporalResult<Self> {
        let master_key = rand::random_32()?;

        Ok(Self {
            master_key,
            sealed_keys: Arc::new(RwLock::new(HashMap::new())),
            audit_log: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Create HSM with a specific master key (for testing)
    pub fn with_master_key(master_key: [u8; 32]) -> Self {
        Self {
            master_key,
            sealed_keys: Arc::new(RwLock::new(HashMap::new())),
            audit_log: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Seal a private key (key never leaves HSM in unencrypted form)
    pub fn seal_key(&self, key_id: [u8; 32], private_key: &[u8]) -> TemporalResult<()> {
        // Generate nonce
        let nonce = rand::random_nonce()?;

        // Encrypt with master key
        let mut encrypted = nonce.to_vec();
        let ciphertext = aead::encrypt(&self.master_key, &nonce, private_key)?;
        encrypted.extend(ciphertext);

        // Store
        let mut keys = self.sealed_keys.write()
            .map_err(|_| TemporalError::HsmError("Lock poisoned".to_string()))?;
        keys.insert(key_id, encrypted);

        // Audit
        self.log_operation(HsmOperation::SealKey, key_id, true);

        Ok(())
    }

    /// Unseal a key (returns the key - use with caution!)
    pub fn unseal_key(&self, key_id: [u8; 32]) -> TemporalResult<Vec<u8>> {
        let keys = self.sealed_keys.read()
            .map_err(|_| TemporalError::HsmError("Lock poisoned".to_string()))?;

        let encrypted = keys.get(&key_id)
            .ok_or(TemporalError::KeyNotFound(key_id))?;

        // Extract nonce and ciphertext
        if encrypted.len() < 24 {
            return Err(TemporalError::HsmError("Invalid sealed key format".to_string()));
        }

        let (nonce_bytes, ciphertext) = encrypted.split_at(24);
        let mut nonce = [0u8; 24];
        nonce.copy_from_slice(nonce_bytes);

        // Decrypt
        let private_key = aead::decrypt(&self.master_key, &nonce, ciphertext)?;

        // Audit
        self.log_operation(HsmOperation::UnsealKey, key_id, true);

        Ok(private_key)
    }

    /// Perform decryption inside HSM (key never leaves)
    pub fn decrypt_inside_hsm(
        &self,
        key_id: [u8; 32],
        kem_ciphertext: &[u8],
    ) -> TemporalResult<[u8; 32]> {
        // Get the sealed key and decrypt it internally
        let private_key = self.unseal_key(key_id)?;

        // Perform KEM decapsulation
        let shared_secret = crate::crypto::kem::MlKem1024::decapsulate(kem_ciphertext, &private_key)?;

        // Audit
        self.log_operation(HsmOperation::Decrypt, key_id, true);

        Ok(shared_secret)
    }

    /// Check if a key is sealed in this HSM
    pub fn has_key(&self, key_id: &[u8; 32]) -> bool {
        self.sealed_keys.read()
            .map(|keys| keys.contains_key(key_id))
            .unwrap_or(false)
    }

    /// Get the number of sealed keys
    pub fn key_count(&self) -> usize {
        self.sealed_keys.read()
            .map(|keys| keys.len())
            .unwrap_or(0)
    }

    /// Get audit log
    pub fn get_audit_log(&self) -> Vec<HsmAuditEntry> {
        self.audit_log.read()
            .map(|log| log.clone())
            .unwrap_or_default()
    }

    /// Clear audit log
    pub fn clear_audit_log(&self) {
        if let Ok(mut log) = self.audit_log.write() {
            log.clear();
        }
    }

    fn log_operation(&self, operation: HsmOperation, key_id: [u8; 32], success: bool) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let entry = HsmAuditEntry {
            timestamp,
            operation,
            key_id,
            success,
        };

        if let Ok(mut log) = self.audit_log.write() {
            log.push(entry);
        }
    }
}

impl Default for HsmSimulator {
    fn default() -> Self {
        Self::new().expect("Failed to create HSM simulator")
    }
}

/// Multi-HSM setup for distributed trustee keys
pub struct DistributedHsmNetwork {
    /// HSMs indexed by trustee ID
    hsms: HashMap<[u8; 32], HsmSimulator>,
}

impl DistributedHsmNetwork {
    /// Create a new distributed HSM network
    pub fn new() -> Self {
        Self {
            hsms: HashMap::new(),
        }
    }

    /// Add an HSM for a trustee
    pub fn add_hsm(&mut self, trustee_id: [u8; 32]) -> TemporalResult<()> {
        let hsm = HsmSimulator::new()?;
        self.hsms.insert(trustee_id, hsm);
        Ok(())
    }

    /// Get HSM for a trustee
    pub fn get_hsm(&self, trustee_id: &[u8; 32]) -> Option<&HsmSimulator> {
        self.hsms.get(trustee_id)
    }

    /// Get mutable HSM for a trustee
    pub fn get_hsm_mut(&mut self, trustee_id: &[u8; 32]) -> Option<&mut HsmSimulator> {
        self.hsms.get_mut(trustee_id)
    }

    /// Number of HSMs in the network
    pub fn hsm_count(&self) -> usize {
        self.hsms.len()
    }
}

impl Default for DistributedHsmNetwork {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hsm_seal_unseal() {
        let hsm = HsmSimulator::new().unwrap();
        let key_id = [1u8; 32];
        let private_key = b"test private key data";

        hsm.seal_key(key_id, private_key).unwrap();
        assert!(hsm.has_key(&key_id));

        let recovered = hsm.unseal_key(key_id).unwrap();
        assert_eq!(recovered, private_key);
    }

    #[test]
    fn test_hsm_key_not_found() {
        let hsm = HsmSimulator::new().unwrap();
        let key_id = [1u8; 32];

        let result = hsm.unseal_key(key_id);
        assert!(result.is_err());
    }

    #[test]
    fn test_hsm_audit_log() {
        let hsm = HsmSimulator::new().unwrap();
        let key_id = [1u8; 32];
        let private_key = b"test key";

        hsm.seal_key(key_id, private_key).unwrap();
        hsm.unseal_key(key_id).unwrap();

        let log = hsm.get_audit_log();
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_distributed_hsm_network() {
        let mut network = DistributedHsmNetwork::new();

        let trustee1 = [1u8; 32];
        let trustee2 = [2u8; 32];

        network.add_hsm(trustee1).unwrap();
        network.add_hsm(trustee2).unwrap();

        assert_eq!(network.hsm_count(), 2);
        assert!(network.get_hsm(&trustee1).is_some());
        assert!(network.get_hsm(&trustee2).is_some());
    }
}
