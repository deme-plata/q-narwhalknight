//! Plugin Verification Module
//!
//! Provides cryptographic verification for plugin manifests and WASM bytecode.
//! Ensures all plugins are verified before installation:
//! - Manifest signature verification (Ed25519)
//! - WASM bytecode hash verification (SHA3-256)
//! - Trusted author list management

use crate::network::protocol::PluginManifest;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Result of manifest verification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestVerificationResult {
    /// Whether the manifest is valid
    pub valid: bool,
    /// Whether the author is trusted
    pub author_trusted: bool,
    /// Verification error message if invalid
    pub error: Option<String>,
    /// Timestamp of verification
    pub verified_at: u64,
}

/// Result of WASM bytecode verification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WasmVerificationResult {
    /// Whether the WASM hash matches the manifest
    pub hash_valid: bool,
    /// Computed hash of the WASM bytecode (hex-encoded)
    pub computed_hash: String,
    /// Expected hash from the manifest (hex-encoded)
    pub expected_hash: String,
    /// Size of the WASM bytecode in bytes
    pub size: u64,
}

/// Errors that can occur during plugin verification
#[derive(Debug, Clone, thiserror::Error, Serialize, Deserialize)]
pub enum PluginVerificationError {
    #[error("Invalid signature: {0}")]
    InvalidSignature(String),

    #[error("Invalid public key: {0}")]
    InvalidPublicKey(String),

    #[error("Hash mismatch: expected {expected}, got {computed}")]
    HashMismatch { expected: String, computed: String },

    #[error("Untrusted author: {0}")]
    UntrustedAuthor(String),

    #[error("Manifest validation failed: {0}")]
    ManifestValidation(String),

    #[error("WASM validation failed: {0}")]
    WasmValidation(String),
}

/// List of trusted plugin authors (public keys)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrustedAuthorList {
    /// Set of trusted author public keys (hex-encoded Ed25519 public keys)
    pub trusted_keys: HashSet<String>,
    /// Whether to allow plugins from untrusted authors
    pub allow_untrusted: bool,
    /// Description of the trust list
    pub description: String,
    /// Last updated timestamp
    pub updated_at: u64,
}

impl TrustedAuthorList {
    /// Create a new empty trusted author list
    pub fn new(allow_untrusted: bool) -> Self {
        Self {
            trusted_keys: HashSet::new(),
            allow_untrusted,
            description: "Plugin trusted author list".to_string(),
            updated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Add a trusted author
    pub fn add_trusted(&mut self, pubkey: String) {
        self.trusted_keys.insert(pubkey);
        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    /// Remove a trusted author
    pub fn remove_trusted(&mut self, pubkey: &str) -> bool {
        let removed = self.trusted_keys.remove(pubkey);
        if removed {
            self.updated_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }
        removed
    }

    /// Check if an author is trusted
    pub fn is_trusted(&self, pubkey: &str) -> bool {
        self.trusted_keys.contains(pubkey)
    }

    /// Check if a plugin from this author should be allowed
    pub fn should_allow(&self, pubkey: &str) -> bool {
        self.allow_untrusted || self.is_trusted(pubkey)
    }
}

/// Plugin verifier for manifest and WASM verification
pub struct PluginVerifier {
    /// Trusted author list
    trusted_authors: Arc<RwLock<TrustedAuthorList>>,
    /// Whether to require trusted authors
    require_trusted: bool,
}

impl PluginVerifier {
    /// Create a new plugin verifier
    pub fn new(require_trusted: bool) -> Self {
        Self {
            trusted_authors: Arc::new(RwLock::new(TrustedAuthorList::new(!require_trusted))),
            require_trusted,
        }
    }

    /// Create a verifier with a trusted author list
    pub fn with_trusted_authors(trusted_authors: TrustedAuthorList) -> Self {
        Self {
            require_trusted: !trusted_authors.allow_untrusted,
            trusted_authors: Arc::new(RwLock::new(trusted_authors)),
        }
    }

    /// Verify a plugin manifest signature
    pub async fn verify_manifest(
        &self,
        manifest: &PluginManifest,
    ) -> Result<ManifestVerificationResult, PluginVerificationError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Validate manifest fields
        if manifest.plugin_id.is_empty() {
            return Err(PluginVerificationError::ManifestValidation(
                "Plugin ID is empty".to_string(),
            ));
        }

        if manifest.version.is_empty() {
            return Err(PluginVerificationError::ManifestValidation(
                "Version is empty".to_string(),
            ));
        }

        // Check if signature is present
        if manifest.signature.is_empty() {
            return Ok(ManifestVerificationResult {
                valid: false,
                author_trusted: false,
                error: Some("No signature present".to_string()),
                verified_at: now,
            });
        }

        // Decode and verify public key
        let pubkey_bytes = hex::decode(&manifest.author_pubkey).map_err(|e| {
            PluginVerificationError::InvalidPublicKey(format!("Invalid hex encoding: {}", e))
        })?;

        if pubkey_bytes.len() != 32 {
            return Err(PluginVerificationError::InvalidPublicKey(format!(
                "Invalid public key length: expected 32, got {}",
                pubkey_bytes.len()
            )));
        }

        let pubkey_array: [u8; 32] = pubkey_bytes.try_into().map_err(|_| {
            PluginVerificationError::InvalidPublicKey("Failed to convert public key".to_string())
        })?;

        let verifying_key = VerifyingKey::from_bytes(&pubkey_array).map_err(|e| {
            PluginVerificationError::InvalidPublicKey(format!("Invalid Ed25519 key: {}", e))
        })?;

        // Decode signature
        if manifest.signature.len() != 64 {
            return Err(PluginVerificationError::InvalidSignature(format!(
                "Invalid signature length: expected 64, got {}",
                manifest.signature.len()
            )));
        }

        let sig_array: [u8; 64] = manifest.signature.clone().try_into().map_err(|_| {
            PluginVerificationError::InvalidSignature("Failed to convert signature".to_string())
        })?;

        let signature = Signature::from_bytes(&sig_array);

        // Get the signing bytes
        let signing_bytes = manifest.signing_bytes();

        // Verify signature
        let valid = verifying_key.verify(&signing_bytes, &signature).is_ok();

        if !valid {
            debug!(
                "Manifest signature verification failed for {}",
                manifest.unique_id()
            );
        }

        // Check if author is trusted
        let trusted_list = self.trusted_authors.read().await;
        let author_trusted = trusted_list.is_trusted(&manifest.author_pubkey);
        let should_allow = trusted_list.should_allow(&manifest.author_pubkey);

        if !should_allow && self.require_trusted {
            return Err(PluginVerificationError::UntrustedAuthor(
                manifest.author_pubkey.clone(),
            ));
        }

        Ok(ManifestVerificationResult {
            valid,
            author_trusted,
            error: if valid { None } else { Some("Signature verification failed".to_string()) },
            verified_at: now,
        })
    }

    /// Verify WASM bytecode against manifest hash
    pub fn verify_wasm(
        &self,
        manifest: &PluginManifest,
        wasm_bytes: &[u8],
    ) -> Result<WasmVerificationResult, PluginVerificationError> {
        // Compute hash of WASM bytecode
        let mut hasher = Sha3_256::new();
        hasher.update(wasm_bytes);
        let computed_hash = hex::encode(hasher.finalize());

        let hash_valid = computed_hash == manifest.wasm_hash;

        if !hash_valid {
            warn!(
                "WASM hash mismatch for {}: expected {}, got {}",
                manifest.unique_id(),
                manifest.wasm_hash,
                computed_hash
            );
        }

        // Verify size matches
        if wasm_bytes.len() as u64 != manifest.wasm_size {
            return Err(PluginVerificationError::WasmValidation(format!(
                "Size mismatch: expected {}, got {}",
                manifest.wasm_size,
                wasm_bytes.len()
            )));
        }

        Ok(WasmVerificationResult {
            hash_valid,
            computed_hash,
            expected_hash: manifest.wasm_hash.clone(),
            size: wasm_bytes.len() as u64,
        })
    }

    /// Add a trusted author
    pub async fn add_trusted_author(&self, pubkey: String) {
        let mut trusted = self.trusted_authors.write().await;
        trusted.add_trusted(pubkey);
    }

    /// Remove a trusted author
    pub async fn remove_trusted_author(&self, pubkey: &str) -> bool {
        let mut trusted = self.trusted_authors.write().await;
        trusted.remove_trusted(pubkey)
    }

    /// Check if an author is trusted
    pub async fn is_author_trusted(&self, pubkey: &str) -> bool {
        let trusted = self.trusted_authors.read().await;
        trusted.is_trusted(pubkey)
    }

    /// Get the trusted author list
    pub async fn get_trusted_authors(&self) -> TrustedAuthorList {
        self.trusted_authors.read().await.clone()
    }

    /// Set the trusted author list
    pub async fn set_trusted_authors(&self, list: TrustedAuthorList) {
        let mut trusted = self.trusted_authors.write().await;
        *trusted = list;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::protocol::PluginManifestPermissions;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn create_signed_manifest() -> (PluginManifest, SigningKey) {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        let author_pubkey = hex::encode(verifying_key.to_bytes());

        let mut manifest = PluginManifest {
            plugin_id: "test.plugin".to_string(),
            version: "1.0.0".to_string(),
            name: "Test Plugin".to_string(),
            description: "A test plugin".to_string(),
            author_pubkey,
            wasm_hash: "0".repeat(64),
            wasm_size: 1024,
            min_node_version: "0.1.0".to_string(),
            permissions: PluginManifestPermissions::default(),
            dependencies: vec![],
            published_at: 1700000000,
            signature: vec![],
        };

        // Sign the manifest
        use ed25519_dalek::Signer;
        let signing_bytes = manifest.signing_bytes();
        let signature = signing_key.sign(&signing_bytes);
        manifest.signature = signature.to_bytes().to_vec();

        (manifest, signing_key)
    }

    #[tokio::test]
    async fn test_verify_valid_manifest() {
        let (manifest, _) = create_signed_manifest();
        let verifier = PluginVerifier::new(false);

        let result = verifier.verify_manifest(&manifest).await.unwrap();
        assert!(result.valid);
    }

    #[tokio::test]
    async fn test_verify_invalid_signature() {
        let (mut manifest, _) = create_signed_manifest();
        manifest.signature[0] ^= 0xFF; // Corrupt signature

        let verifier = PluginVerifier::new(false);
        let result = verifier.verify_manifest(&manifest).await.unwrap();
        assert!(!result.valid);
    }

    #[tokio::test]
    async fn test_verify_tampered_manifest() {
        let (mut manifest, _) = create_signed_manifest();
        manifest.name = "Tampered Name".to_string(); // Modify after signing

        let verifier = PluginVerifier::new(false);
        let result = verifier.verify_manifest(&manifest).await.unwrap();
        assert!(!result.valid);
    }

    #[tokio::test]
    async fn test_trusted_author_check() {
        let (manifest, _) = create_signed_manifest();
        let verifier = PluginVerifier::new(false);

        // Initially not trusted
        let result = verifier.verify_manifest(&manifest).await.unwrap();
        assert!(!result.author_trusted);

        // Add to trusted list
        verifier.add_trusted_author(manifest.author_pubkey.clone()).await;

        let result = verifier.verify_manifest(&manifest).await.unwrap();
        assert!(result.author_trusted);
    }

    #[tokio::test]
    async fn test_require_trusted_author() {
        let (manifest, _) = create_signed_manifest();
        let verifier = PluginVerifier::new(true); // Require trusted

        // Should fail without trusted author
        let result = verifier.verify_manifest(&manifest).await;
        assert!(matches!(result, Err(PluginVerificationError::UntrustedAuthor(_))));

        // Add to trusted list
        verifier.add_trusted_author(manifest.author_pubkey.clone()).await;

        // Should pass now
        let result = verifier.verify_manifest(&manifest).await.unwrap();
        assert!(result.valid);
        assert!(result.author_trusted);
    }

    #[test]
    fn test_wasm_verification_valid() {
        let wasm_bytes = b"test wasm bytecode";
        let mut hasher = Sha3_256::new();
        hasher.update(wasm_bytes);
        let wasm_hash = hex::encode(hasher.finalize());

        let manifest = PluginManifest {
            plugin_id: "test".to_string(),
            version: "1.0.0".to_string(),
            name: "Test".to_string(),
            description: "".to_string(),
            author_pubkey: "0".repeat(64),
            wasm_hash,
            wasm_size: wasm_bytes.len() as u64,
            min_node_version: "".to_string(),
            permissions: PluginManifestPermissions::default(),
            dependencies: vec![],
            published_at: 0,
            signature: vec![],
        };

        let verifier = PluginVerifier::new(false);
        let result = verifier.verify_wasm(&manifest, wasm_bytes).unwrap();
        assert!(result.hash_valid);
    }

    #[test]
    fn test_wasm_verification_hash_mismatch() {
        let wasm_bytes = b"test wasm bytecode";

        let manifest = PluginManifest {
            plugin_id: "test".to_string(),
            version: "1.0.0".to_string(),
            name: "Test".to_string(),
            description: "".to_string(),
            author_pubkey: "0".repeat(64),
            wasm_hash: "0".repeat(64), // Wrong hash
            wasm_size: wasm_bytes.len() as u64,
            min_node_version: "".to_string(),
            permissions: PluginManifestPermissions::default(),
            dependencies: vec![],
            published_at: 0,
            signature: vec![],
        };

        let verifier = PluginVerifier::new(false);
        let result = verifier.verify_wasm(&manifest, wasm_bytes).unwrap();
        assert!(!result.hash_valid);
    }

    #[test]
    fn test_wasm_verification_size_mismatch() {
        let wasm_bytes = b"test wasm bytecode";
        let mut hasher = Sha3_256::new();
        hasher.update(wasm_bytes);
        let wasm_hash = hex::encode(hasher.finalize());

        let manifest = PluginManifest {
            plugin_id: "test".to_string(),
            version: "1.0.0".to_string(),
            name: "Test".to_string(),
            description: "".to_string(),
            author_pubkey: "0".repeat(64),
            wasm_hash,
            wasm_size: 9999, // Wrong size
            min_node_version: "".to_string(),
            permissions: PluginManifestPermissions::default(),
            dependencies: vec![],
            published_at: 0,
            signature: vec![],
        };

        let verifier = PluginVerifier::new(false);
        let result = verifier.verify_wasm(&manifest, wasm_bytes);
        assert!(matches!(result, Err(PluginVerificationError::WasmValidation(_))));
    }

    #[test]
    fn test_trusted_author_list() {
        let mut list = TrustedAuthorList::new(false);
        assert!(!list.allow_untrusted);

        let key = "abc123".to_string();
        list.add_trusted(key.clone());
        assert!(list.is_trusted(&key));
        assert!(list.should_allow(&key));
        assert!(!list.should_allow("unknown"));

        list.allow_untrusted = true;
        assert!(list.should_allow("unknown"));

        list.remove_trusted(&key);
        assert!(!list.is_trusted(&key));
    }
}
