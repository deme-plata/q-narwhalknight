//! v8.5.0: P2P Update Announcement Protocol
//!
//! Enables cryptographically signed software update announcements via gossipsub.
//! Bootstrap nodes (Beta, Gamma, Delta) sign announcements with Ed25519.
//! Remote nodes verify quorum (2-of-3 trusted signers) before auto-updating.
//!
//! Security model:
//! - Hardcoded trusted signer public keys (compile-time, not configurable)
//! - Quorum requirement prevents single-server compromise from pushing malicious updates
//! - Dual-hash verification (SHA-256 + BLAKE3) for download integrity
//! - 24-hour announcement expiry prevents replay attacks

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

/// Minimum number of distinct trusted signers required before a node trusts an update announcement.
/// With 3 bootstrap nodes (Beta, Gamma, Delta), quorum of 2 means a single compromised server
/// cannot push a malicious update.
pub const MIN_UPDATE_QUORUM: usize = 2;

/// Maximum age of an announcement before it's rejected (prevents replay attacks).
/// 24 hours in seconds.
pub const MAX_ANNOUNCEMENT_AGE_SECS: u64 = 86400;

/// Trusted update signer public keys (hex-encoded Ed25519 verifying keys).
/// These are the node_signing_key public keys of the bootstrap servers.
/// To update: extract from each server's persisted signing key file.
///
/// IMPORTANT: These must be populated with real keys before mainnet deployment.
/// During development/testing, the announce endpoint will use the node's own signing key,
/// and other bootstrap nodes will co-sign if they're running the same version.
pub const TRUSTED_UPDATE_SIGNERS: &[&str] = &[
    // Server Beta (185.182.185.227) — primary bootstrap
    "af7a81d8d377869ed17fd7d4d3981b45b9221fe72a6ef025e09fd2a87fad0f17",
    // Server Gamma (109.205.176.60) — backup bootstrap
    "0aa50aa1ee7536af32747c954c112966cb2b303a4370f9f8f0b5f0093aeba5c0",
    // Server Delta (5.79.79.158) — 4th bootstrap
    "425d5cd0fde72f6702b79fcd96157364f132be5c57800638a4b9d0fa370a8264",
];

/// A signed announcement that a new software version is available.
/// Published to gossipsub topic `/qnk/{network}/update-announcements`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateAnnouncement {
    /// Semantic version of the new binary (e.g., "8.5.0")
    pub version: String,

    /// SHA-256 hex digest of the binary
    pub sha256_checksum: String,

    /// BLAKE3 hex digest of the binary (fast verification)
    pub blake3_checksum: String,

    /// Binary file size in bytes
    pub binary_size: u64,

    /// HTTPS download URL (e.g., "https://quillon.xyz/downloads/q-api-server-v8.5.0")
    pub download_url: String,

    /// Network ID this update applies to (e.g., "mainnet2026.1.1")
    pub network_id: String,

    /// Minimum version that can auto-update to this version (optional skip protection)
    #[serde(default)]
    pub min_update_from_version: Option<String>,

    /// Unix timestamp when announcement was created
    pub timestamp: u64,

    /// Ed25519 signature over `signable_bytes()` (hex-encoded)
    pub signature: String,

    /// Ed25519 public key of the signer (hex-encoded)
    pub signer_pubkey: String,

    /// libp2p PeerId of the signing node
    pub signer_peer_id: String,

    /// If true, this is a mandatory security update (nodes should apply immediately)
    #[serde(default)]
    pub mandatory: bool,

    /// Human-readable release notes
    #[serde(default)]
    pub release_notes: String,
}

impl UpdateAnnouncement {
    /// Compute the canonical byte representation for signing.
    /// Excludes signature and signer fields to allow verification.
    /// Prefixed with `QNK-UPDATE-v1:` for domain separation.
    pub fn signable_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(512);
        buf.extend_from_slice(b"QNK-UPDATE-v1:");
        buf.extend_from_slice(self.version.as_bytes());
        buf.push(b':');
        buf.extend_from_slice(self.sha256_checksum.as_bytes());
        buf.push(b':');
        buf.extend_from_slice(self.blake3_checksum.as_bytes());
        buf.push(b':');
        buf.extend_from_slice(self.binary_size.to_le_bytes().as_slice());
        buf.push(b':');
        buf.extend_from_slice(self.download_url.as_bytes());
        buf.push(b':');
        buf.extend_from_slice(self.network_id.as_bytes());
        buf.push(b':');
        buf.extend_from_slice(self.timestamp.to_le_bytes().as_slice());
        buf.push(b':');
        buf.push(if self.mandatory { 1 } else { 0 });
        buf
    }

    /// Sign this announcement with the given Ed25519 signing key.
    /// Sets `signature` and `signer_pubkey` fields.
    pub fn sign(&mut self, signing_key: &SigningKey) {
        let verifying_key = signing_key.verifying_key();
        self.signer_pubkey = hex::encode(verifying_key.as_bytes());
        let sig = signing_key.sign(&self.signable_bytes());
        self.signature = hex::encode(sig.to_bytes());
    }

    /// Verify the Ed25519 signature on this announcement.
    /// Returns the verifying key on success.
    pub fn verify_signature(&self) -> Result<VerifyingKey, String> {
        let pubkey_bytes = hex::decode(&self.signer_pubkey)
            .map_err(|e| format!("invalid signer_pubkey hex: {}", e))?;
        let pubkey_array: [u8; 32] = pubkey_bytes
            .try_into()
            .map_err(|_| "signer_pubkey must be 32 bytes".to_string())?;
        let verifying_key = VerifyingKey::from_bytes(&pubkey_array)
            .map_err(|e| format!("invalid Ed25519 public key: {}", e))?;

        let sig_bytes = hex::decode(&self.signature)
            .map_err(|e| format!("invalid signature hex: {}", e))?;
        let sig_array: [u8; 64] = sig_bytes
            .try_into()
            .map_err(|_| "signature must be 64 bytes".to_string())?;
        let signature = Signature::from_bytes(&sig_array);

        verifying_key
            .verify(&self.signable_bytes(), &signature)
            .map_err(|e| format!("signature verification failed: {}", e))?;

        Ok(verifying_key)
    }

    /// Check if the signer's public key is in the trusted signers list.
    pub fn is_trusted_signer(&self) -> bool {
        TRUSTED_UPDATE_SIGNERS
            .iter()
            .any(|&trusted| trusted == self.signer_pubkey)
    }

    /// Check if this announcement has expired (older than MAX_ANNOUNCEMENT_AGE_SECS).
    pub fn is_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(self.timestamp) > MAX_ANNOUNCEMENT_AGE_SECS
    }

    /// Create a new unsigned announcement.
    pub fn new(
        version: String,
        sha256_checksum: String,
        blake3_checksum: String,
        binary_size: u64,
        download_url: String,
        network_id: String,
        peer_id: String,
        mandatory: bool,
        release_notes: String,
    ) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            version,
            sha256_checksum,
            blake3_checksum,
            binary_size,
            download_url,
            network_id,
            min_update_from_version: None,
            timestamp,
            signature: String::new(),
            signer_pubkey: String::new(),
            signer_peer_id: peer_id,
            mandatory,
            release_notes,
        }
    }
}

/// Accumulator for collecting co-signed announcements and checking quorum.
#[derive(Debug, Default)]
pub struct UpdateQuorum {
    /// Map of (version, sha256) -> list of distinct signer pubkeys that signed it
    entries: std::collections::HashMap<(String, String), Vec<String>>,
}

impl UpdateQuorum {
    pub fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
        }
    }

    /// Record a verified announcement. Returns true if quorum is now reached.
    pub fn record(&mut self, announcement: &UpdateAnnouncement) -> bool {
        let key = (
            announcement.version.clone(),
            announcement.sha256_checksum.clone(),
        );
        let signers = self.entries.entry(key).or_default();

        // Don't double-count the same signer
        if !signers.contains(&announcement.signer_pubkey) {
            signers.push(announcement.signer_pubkey.clone());
        }

        signers.len() >= MIN_UPDATE_QUORUM
    }

    /// Check if quorum has been reached for a specific (version, sha256) pair.
    pub fn has_quorum(&self, version: &str, sha256: &str) -> bool {
        self.entries
            .get(&(version.to_string(), sha256.to_string()))
            .map(|signers| signers.len() >= MIN_UPDATE_QUORUM)
            .unwrap_or(false)
    }

    /// Get the number of distinct signers for a (version, sha256) pair.
    pub fn signer_count(&self, version: &str, sha256: &str) -> usize {
        self.entries
            .get(&(version.to_string(), sha256.to_string()))
            .map(|signers| signers.len())
            .unwrap_or(0)
    }

    /// Remove entries for versions older than the given version.
    /// Prevents unbounded growth of the quorum accumulator.
    pub fn prune_older_than(&mut self, version: &str) {
        self.entries
            .retain(|(v, _), _| v.as_str() >= version);
    }
}

/// Parse trusted signer hex strings into VerifyingKeys.
/// Returns only the keys that parse successfully (logs warnings for failures).
pub fn load_trusted_signers() -> Vec<VerifyingKey> {
    TRUSTED_UPDATE_SIGNERS
        .iter()
        .filter_map(|hex_key| {
            let bytes = hex::decode(hex_key).ok()?;
            let array: [u8; 32] = bytes.try_into().ok()?;
            VerifyingKey::from_bytes(&array).ok()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic signing key for tests (no rand / OsRng dependency).
    fn signing_key_from_index(i: u32) -> SigningKey {
        let mut seed = [0u8; 32];
        seed[0..4].copy_from_slice(&i.to_le_bytes());
        SigningKey::from_bytes(&seed)
    }

    #[test]
    fn test_sign_and_verify() {
        let signing_key = signing_key_from_index(0);
        let mut announcement = UpdateAnnouncement::new(
            "8.5.0".to_string(),
            "abcd1234".repeat(8),
            "ef567890".repeat(8),
            12345678,
            "https://quillon.xyz/downloads/q-api-server-v8.5.0".to_string(),
            "mainnet2026.1.1".to_string(),
            "12D3KooWTest".to_string(),
            false,
            "Test release".to_string(),
        );

        announcement.sign(&signing_key);
        assert!(!announcement.signature.is_empty());
        assert!(!announcement.signer_pubkey.is_empty());

        let result = announcement.verify_signature();
        assert!(result.is_ok(), "Signature verification failed: {:?}", result);
    }

    #[test]
    fn test_tampered_announcement_fails_verification() {
        let signing_key = signing_key_from_index(1);
        let mut announcement = UpdateAnnouncement::new(
            "8.5.0".to_string(),
            "abcd1234".repeat(8),
            "ef567890".repeat(8),
            12345678,
            "https://quillon.xyz/downloads/q-api-server-v8.5.0".to_string(),
            "mainnet2026.1.1".to_string(),
            "12D3KooWTest".to_string(),
            false,
            "Test release".to_string(),
        );

        announcement.sign(&signing_key);

        // Tamper with the version
        announcement.version = "9.9.9".to_string();

        let result = announcement.verify_signature();
        assert!(result.is_err(), "Tampered announcement should fail verification");
    }

    #[test]
    fn test_quorum_accumulation() {
        let mut quorum = UpdateQuorum::new();

        let key1 = signing_key_from_index(2);
        let key2 = signing_key_from_index(3);

        let mut ann1 = UpdateAnnouncement::new(
            "8.5.0".to_string(),
            "checksum_a".to_string(),
            "blake3_a".to_string(),
            100,
            "https://example.com".to_string(),
            "mainnet".to_string(),
            "peer1".to_string(),
            false,
            String::new(),
        );
        ann1.sign(&key1);

        // First signer — not yet quorum
        assert!(!quorum.record(&ann1));
        assert_eq!(quorum.signer_count("8.5.0", "checksum_a"), 1);

        // Same signer again — still no quorum
        assert!(!quorum.record(&ann1));
        assert_eq!(quorum.signer_count("8.5.0", "checksum_a"), 1);

        // Second signer — quorum reached
        let mut ann2 = ann1.clone();
        ann2.sign(&key2);
        assert!(quorum.record(&ann2));
        assert_eq!(quorum.signer_count("8.5.0", "checksum_a"), 2);
        assert!(quorum.has_quorum("8.5.0", "checksum_a"));
    }

    #[test]
    fn test_different_checksum_no_quorum() {
        let mut quorum = UpdateQuorum::new();

        let key1 = signing_key_from_index(4);
        let key2 = signing_key_from_index(5);

        let mut ann1 = UpdateAnnouncement::new(
            "8.5.0".to_string(),
            "checksum_a".to_string(),
            "blake3_a".to_string(),
            100,
            "https://example.com".to_string(),
            "mainnet".to_string(),
            "peer1".to_string(),
            false,
            String::new(),
        );
        ann1.sign(&key1);
        quorum.record(&ann1);

        // Different SHA256 — should NOT contribute to quorum for checksum_a
        let mut ann2 = UpdateAnnouncement::new(
            "8.5.0".to_string(),
            "checksum_b".to_string(), // Different!
            "blake3_a".to_string(),
            100,
            "https://example.com".to_string(),
            "mainnet".to_string(),
            "peer2".to_string(),
            false,
            String::new(),
        );
        ann2.sign(&key2);
        quorum.record(&ann2);

        assert!(!quorum.has_quorum("8.5.0", "checksum_a"));
        assert!(!quorum.has_quorum("8.5.0", "checksum_b"));
    }

    #[test]
    fn test_announcement_expiry() {
        let mut ann = UpdateAnnouncement::new(
            "8.5.0".to_string(),
            "check".to_string(),
            "blake".to_string(),
            100,
            "https://example.com".to_string(),
            "mainnet".to_string(),
            "peer1".to_string(),
            false,
            String::new(),
        );

        // Fresh announcement should not be expired
        assert!(!ann.is_expired());

        // Set timestamp to 25 hours ago
        ann.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - 90_000;
        assert!(ann.is_expired());
    }

    #[test]
    fn test_signable_bytes_deterministic() {
        let ann = UpdateAnnouncement::new(
            "8.5.0".to_string(),
            "abc".to_string(),
            "def".to_string(),
            42,
            "https://example.com".to_string(),
            "mainnet".to_string(),
            "peer1".to_string(),
            true,
            "notes".to_string(),
        );

        let bytes1 = ann.signable_bytes();
        let bytes2 = ann.signable_bytes();
        assert_eq!(bytes1, bytes2, "signable_bytes must be deterministic");
        assert!(bytes1.starts_with(b"QNK-UPDATE-v1:"));
    }
}
