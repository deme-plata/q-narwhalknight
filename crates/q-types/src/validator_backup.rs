//! Validator Key Backup - TemporalShield protection for validator keypairs
//!
//! Provides secure backup/restore for validator keys using (5,9) threshold
//! secret sharing with post-quantum encryption (ML-KEM-1024) and zk-STARK proofs.
//!
//! ## Security Properties
//! - Information-theoretic secrecy via OTP encryption
//! - Post-quantum security via ML-KEM-1024
//! - Threshold (5,9) requires 5 of 9 trustees to restore
//! - zk-STARK proofs (NO TRUSTED SETUP) verify backup integrity
//!
//! ## Trustee Distribution (Recommended)
//! - 3x HSMs (geographically distributed)
//! - 2x Cold storage (air-gapped)
//! - 2x Trusted team members
//! - 2x Legal/compliance escrow

use serde::{Deserialize, Serialize};
use crate::NodeId;

/// Backup threshold parameters
pub const VALIDATOR_BACKUP_THRESHOLD: usize = 5;
pub const VALIDATOR_BACKUP_TOTAL: usize = 9;

/// Backup metadata for tracking and verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    /// Node ID being backed up
    pub node_id: NodeId,
    /// Backup creation timestamp
    pub created_at: u64,
    /// Threshold (k) - minimum shares needed
    pub threshold: usize,
    /// Total trustees (n)
    pub total_trustees: usize,
    /// Public key fingerprint for verification
    pub fingerprint: [u8; 32],
    /// Backup version
    pub version: u32,
    /// Human-readable label
    pub label: Option<String>,
}

impl BackupMetadata {
    /// Create new backup metadata
    pub fn new(node_id: NodeId, fingerprint: [u8; 32], label: Option<String>) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            node_id,
            created_at: timestamp,
            threshold: VALIDATOR_BACKUP_THRESHOLD,
            total_trustees: VALIDATOR_BACKUP_TOTAL,
            fingerprint,
            version: 1,
            label,
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self)
            .map_err(|e| format!("Serialization failed: {}", e))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        bincode::deserialize(bytes)
            .map_err(|e| format!("Deserialization failed: {}", e))
    }
}

/// Serializable backup envelope containing protected keypair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorKeyBackup {
    /// Unique backup ID (hash of envelope + metadata)
    pub backup_id: [u8; 32],
    /// Backup metadata
    pub metadata: BackupMetadata,
    /// Protected keypair data (TemporalEnvelope bytes)
    pub protected_data: Vec<u8>,
    /// Key commitment for verification
    pub key_commitment: [u8; 32],
    /// Share commitments for verification
    pub share_commitments: Vec<[u8; 32]>,
    /// STARK proof for integrity verification
    pub stark_proof: Vec<u8>,
}

impl ValidatorKeyBackup {
    /// Create a new backup structure (to be filled by TemporalShield)
    pub fn new(
        backup_id: [u8; 32],
        metadata: BackupMetadata,
        protected_data: Vec<u8>,
        key_commitment: [u8; 32],
        share_commitments: Vec<[u8; 32]>,
        stark_proof: Vec<u8>,
    ) -> Self {
        Self {
            backup_id,
            metadata,
            protected_data,
            key_commitment,
            share_commitments,
            stark_proof,
        }
    }

    /// Verify structural integrity (doesn't verify STARK proof)
    pub fn check_structure(&self) -> bool {
        self.share_commitments.len() == self.metadata.total_trustees
            && !self.protected_data.is_empty()
            && !self.stark_proof.is_empty()
    }

    /// Serialize to bytes for storage
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self)
            .map_err(|e| format!("Backup serialization failed: {}", e))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        bincode::deserialize(bytes)
            .map_err(|e| format!("Backup deserialization failed: {}", e))
    }

    /// Get backup ID as hex string
    pub fn id_hex(&self) -> String {
        hex::encode(self.backup_id)
    }

    /// Get fingerprint from metadata
    pub fn fingerprint(&self) -> [u8; 32] {
        self.metadata.fingerprint
    }

    /// Check if backup can be restored with given number of shares
    pub fn can_restore(&self, available_shares: usize) -> bool {
        available_shares >= self.metadata.threshold
    }
}

/// Result of a backup restore operation
#[derive(Debug, Clone)]
pub struct RestoreResult {
    /// Whether restore was successful
    pub success: bool,
    /// Node ID of restored keypair
    pub node_id: Option<NodeId>,
    /// Number of shares used
    pub shares_used: usize,
    /// Error message if failed
    pub error: Option<String>,
}

impl RestoreResult {
    pub fn success(node_id: NodeId, shares_used: usize) -> Self {
        Self {
            success: true,
            node_id: Some(node_id),
            shares_used,
            error: None,
        }
    }

    pub fn failure(error: String) -> Self {
        Self {
            success: false,
            node_id: None,
            shares_used: 0,
            error: Some(error),
        }
    }
}

/// Backup status for tracking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BackupStatus {
    /// Backup is complete and verified
    Verified,
    /// Backup exists but STARK proof not verified
    Unverified,
    /// Backup is incomplete (missing shares)
    Incomplete,
    /// Backup has been revoked
    Revoked,
}

/// Registry entry for tracking backups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRegistryEntry {
    /// Backup ID
    pub backup_id: [u8; 32],
    /// Node ID
    pub node_id: NodeId,
    /// Creation timestamp
    pub created_at: u64,
    /// Status
    pub status: BackupStatus,
    /// Fingerprint for quick lookup
    pub fingerprint: [u8; 32],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backup_metadata() {
        let node_id = [1u8; 32];
        let fingerprint = [2u8; 32];
        let metadata = BackupMetadata::new(node_id, fingerprint, Some("Test backup".to_string()));

        assert_eq!(metadata.node_id, node_id);
        assert_eq!(metadata.fingerprint, fingerprint);
        assert_eq!(metadata.threshold, VALIDATOR_BACKUP_THRESHOLD);
        assert_eq!(metadata.total_trustees, VALIDATOR_BACKUP_TOTAL);
    }

    #[test]
    fn test_backup_serialization() {
        let node_id = [1u8; 32];
        let fingerprint = [2u8; 32];
        let metadata = BackupMetadata::new(node_id, fingerprint, None);

        let backup = ValidatorKeyBackup::new(
            [3u8; 32],
            metadata,
            vec![0u8; 100],
            [4u8; 32],
            vec![[5u8; 32]; 9],
            vec![0u8; 50],
        );

        assert!(backup.check_structure());

        let bytes = backup.to_bytes().unwrap();
        let restored = ValidatorKeyBackup::from_bytes(&bytes).unwrap();

        assert_eq!(backup.backup_id, restored.backup_id);
        assert_eq!(backup.metadata.node_id, restored.metadata.node_id);
    }

    #[test]
    fn test_can_restore() {
        let metadata = BackupMetadata::new([1u8; 32], [2u8; 32], None);
        let backup = ValidatorKeyBackup::new(
            [3u8; 32],
            metadata,
            vec![0u8; 100],
            [4u8; 32],
            vec![[5u8; 32]; 9],
            vec![0u8; 50],
        );

        assert!(!backup.can_restore(4)); // Need 5
        assert!(backup.can_restore(5));  // Exactly threshold
        assert!(backup.can_restore(9));  // All shares
    }
}
