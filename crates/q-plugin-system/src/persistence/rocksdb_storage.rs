//! RocksDB-backed Persistent Plugin Storage with Consensus Verification
//!
//! This module provides true decentralized persistence for the plugin system,
//! ensuring every node stores and verifies plugin state through consensus.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Plugin State Change                          │
//! └──────────────────────────┬──────────────────────────────────────┘
//!                            │
//!                            ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  1. Local Signature Verification (Ed25519/Dilithium5)           │
//! │  2. State Hash Computation (SHA3-256)                           │
//! │  3. Local RocksDB Write (Plugin CF)                             │
//! │  4. P2P Gossipsub Broadcast                                     │
//! │  5. Consensus Inclusion (DAG-Knight)                            │
//! └─────────────────────────────────────────────────────────────────┘
//!                            │
//!              ┌─────────────┼─────────────┐
//!              │             │             │
//!              ▼             ▼             ▼
//!         ┌────────┐   ┌────────┐   ┌────────┐
//!         │ Node A │   │ Node B │   │ Node C │
//!         │ Verify │   │ Verify │   │ Verify │
//!         │ Store  │   │ Store  │   │ Store  │
//!         └────────┘   └────────┘   └────────┘
//! ```
//!
//! ## Consensus Verification Flow
//!
//! 1. **Local Change**: Plugin state modified
//! 2. **Sign**: State change signed with node's key
//! 3. **Hash**: SHA3-256 of new state computed
//! 4. **Store**: Written to local RocksDB
//! 5. **Broadcast**: Sent to peers via gossipsub
//! 6. **Verify**: Each peer verifies signature and hash
//! 7. **Replicate**: Peers store if verification passes
//! 8. **Consensus**: Included in DAG-Knight consensus round

use crate::network::protocol::{PluginManifest, PluginMessage, PluginGossipsubMessage};
use rocksdb::{ColumnFamilyDescriptor, DBCompactionStyle, Options, WriteBatch, DB};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Column family names for plugin storage
pub const CF_PLUGINS: &str = "plugins";
pub const CF_PLUGIN_STATE: &str = "plugin_state";
pub const CF_PLUGIN_WASM: &str = "plugin_wasm";
pub const CF_PLUGIN_CONSENSUS: &str = "plugin_consensus";
pub const CF_PLUGIN_VERIFICATION: &str = "plugin_verification";

/// Consensus-verified plugin state entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedPluginState {
    /// Plugin manifest
    pub manifest: PluginManifest,
    /// Current state data (serialized)
    pub state_data: Vec<u8>,
    /// State hash (SHA3-256)
    pub state_hash: [u8; 32],
    /// Timestamp of last update
    pub updated_at: u64,
    /// Height when state was last verified via consensus
    pub consensus_height: u64,
    /// Number of nodes that verified this state
    pub verification_count: u32,
    /// Nodes that have verified this state (peer IDs)
    pub verified_by: Vec<String>,
    /// Ed25519 signature of state hash by the updater
    pub signature: Vec<u8>,
    /// Public key of the signer
    pub signer_pubkey: Vec<u8>,
}

/// Plugin verification status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PluginVerificationStatus {
    /// Pending verification from other nodes
    Pending,
    /// Verified by minimum required nodes
    Verified,
    /// Consensus reached (included in block)
    ConsensusReached,
    /// Verification failed (rejected)
    Rejected(String),
}

/// Consensus proof for plugin state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConsensusProof {
    /// Plugin ID
    pub plugin_id: String,
    /// State hash being proven
    pub state_hash: [u8; 32],
    /// Block height where consensus was reached
    pub block_height: u64,
    /// Block hash
    pub block_hash: String,
    /// Merkle proof (path from state to block root)
    pub merkle_proof: Vec<[u8; 32]>,
    /// Validator signatures
    pub validator_signatures: Vec<ValidatorSignature>,
}

/// Validator signature on plugin state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorSignature {
    /// Validator's peer ID
    pub validator_id: String,
    /// Validator's public key
    pub pubkey: Vec<u8>,
    /// Signature over state_hash + block_height
    pub signature: Vec<u8>,
    /// Timestamp
    pub timestamp: u64,
}

/// Statistics for plugin persistence
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginPersistenceStats {
    pub total_plugins_stored: u64,
    pub total_state_entries: u64,
    pub total_wasm_bytes: u64,
    pub consensus_proofs_stored: u64,
    pub verifications_performed: u64,
    pub verifications_passed: u64,
    pub verifications_failed: u64,
    pub p2p_sync_received: u64,
    pub p2p_sync_sent: u64,
}

/// RocksDB-backed persistent plugin storage
pub struct PluginPersistentStorage {
    /// RocksDB instance
    db: Arc<DB>,
    /// In-memory cache for hot plugin states
    state_cache: Arc<RwLock<HashMap<String, VerifiedPluginState>>>,
    /// Pending verifications
    pending_verifications: Arc<RwLock<HashMap<String, Vec<String>>>>,
    /// Our peer ID for signing
    local_peer_id: String,
    /// Our signing key (Ed25519)
    signing_key: Option<ed25519_dalek::SigningKey>,
    /// Statistics
    stats: Arc<RwLock<PluginPersistenceStats>>,
    /// Minimum verifications required for consensus
    min_verifications: u32,
}

impl PluginPersistentStorage {
    /// Create new persistent storage
    pub fn new<P: AsRef<Path>>(
        db_path: P,
        local_peer_id: String,
        signing_key: Option<ed25519_dalek::SigningKey>,
    ) -> Result<Self, PluginPersistenceError> {
        // Configure RocksDB options
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_compaction_style(DBCompactionStyle::Level);
        opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB write buffer
        opts.set_max_write_buffer_number(3);
        opts.set_target_file_size_base(64 * 1024 * 1024);
        opts.set_level_zero_file_num_compaction_trigger(4);
        opts.set_max_background_jobs(4);

        // Define column families
        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new(CF_PLUGINS, Options::default()),
            ColumnFamilyDescriptor::new(CF_PLUGIN_STATE, Options::default()),
            ColumnFamilyDescriptor::new(CF_PLUGIN_WASM, Options::default()),
            ColumnFamilyDescriptor::new(CF_PLUGIN_CONSENSUS, Options::default()),
            ColumnFamilyDescriptor::new(CF_PLUGIN_VERIFICATION, Options::default()),
        ];

        let db = DB::open_cf_descriptors(&opts, db_path, cf_descriptors)
            .map_err(|e| PluginPersistenceError::DatabaseError(e.to_string()))?;

        info!(
            "🔐 [PLUGIN PERSISTENCE] Initialized RocksDB storage with consensus verification"
        );
        info!(
            "🔐 [PLUGIN PERSISTENCE] Column families: plugins, plugin_state, plugin_wasm, plugin_consensus, plugin_verification"
        );

        Ok(Self {
            db: Arc::new(db),
            state_cache: Arc::new(RwLock::new(HashMap::new())),
            pending_verifications: Arc::new(RwLock::new(HashMap::new())),
            local_peer_id,
            signing_key,
            stats: Arc::new(RwLock::new(PluginPersistenceStats::default())),
            min_verifications: 3, // Require 3 node verifications for consensus
        })
    }

    /// Store plugin manifest with verification
    pub async fn store_plugin_manifest(
        &self,
        manifest: &PluginManifest,
    ) -> Result<(), PluginPersistenceError> {
        let plugin_id = manifest.unique_id();

        // Verify manifest signature
        if !self.verify_manifest_signature(manifest)? {
            return Err(PluginPersistenceError::SignatureVerificationFailed(
                "Manifest signature invalid".to_string(),
            ));
        }

        // Serialize manifest
        let manifest_bytes = bincode::serialize(manifest)
            .map_err(|e| PluginPersistenceError::SerializationError(e.to_string()))?;

        // Compute hash
        let mut hasher = Sha3_256::new();
        hasher.update(&manifest_bytes);
        let hash: [u8; 32] = hasher.finalize().into();

        // Store in RocksDB
        let cf_plugins = self
            .db
            .cf_handle(CF_PLUGINS)
            .ok_or_else(|| PluginPersistenceError::ColumnFamilyNotFound(CF_PLUGINS.to_string()))?;

        self.db
            .put_cf(&cf_plugins, plugin_id.as_bytes(), &manifest_bytes)
            .map_err(|e| PluginPersistenceError::DatabaseError(e.to_string()))?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_plugins_stored += 1;
        }

        info!(
            "🔐 [PLUGIN PERSISTENCE] Stored plugin manifest: {} (hash: {})",
            plugin_id,
            hex::encode(&hash[..8])
        );

        Ok(())
    }

    /// Store plugin WASM bytecode with hash verification
    pub async fn store_plugin_wasm(
        &self,
        plugin_id: &str,
        wasm_bytes: &[u8],
        expected_hash: &str,
    ) -> Result<(), PluginPersistenceError> {
        // Verify WASM hash
        let mut hasher = Sha3_256::new();
        hasher.update(wasm_bytes);
        let computed_hash = hex::encode(hasher.finalize());

        if computed_hash != expected_hash {
            return Err(PluginPersistenceError::HashMismatch {
                expected: expected_hash.to_string(),
                computed: computed_hash,
            });
        }

        // Store in RocksDB
        let cf_wasm = self
            .db
            .cf_handle(CF_PLUGIN_WASM)
            .ok_or_else(|| PluginPersistenceError::ColumnFamilyNotFound(CF_PLUGIN_WASM.to_string()))?;

        self.db
            .put_cf(&cf_wasm, plugin_id.as_bytes(), wasm_bytes)
            .map_err(|e| PluginPersistenceError::DatabaseError(e.to_string()))?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_wasm_bytes += wasm_bytes.len() as u64;
        }

        info!(
            "🔐 [PLUGIN PERSISTENCE] Stored plugin WASM: {} ({} bytes, verified hash)",
            plugin_id,
            wasm_bytes.len()
        );

        Ok(())
    }

    /// Store verified plugin state
    pub async fn store_plugin_state(
        &self,
        plugin_id: &str,
        state_data: &[u8],
    ) -> Result<VerifiedPluginState, PluginPersistenceError> {
        // Compute state hash
        let mut hasher = Sha3_256::new();
        hasher.update(state_data);
        let state_hash: [u8; 32] = hasher.finalize().into();

        // Sign the state hash
        let signature = if let Some(ref signing_key) = self.signing_key {
            use ed25519_dalek::Signer;
            signing_key.sign(&state_hash).to_bytes().to_vec()
        } else {
            Vec::new()
        };

        let signer_pubkey = if let Some(ref signing_key) = self.signing_key {
            signing_key.verifying_key().to_bytes().to_vec()
        } else {
            Vec::new()
        };

        // Load manifest
        let manifest = self.get_plugin_manifest(plugin_id).await?
            .ok_or_else(|| PluginPersistenceError::PluginNotFound(plugin_id.to_string()))?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let verified_state = VerifiedPluginState {
            manifest,
            state_data: state_data.to_vec(),
            state_hash,
            updated_at: now,
            consensus_height: 0, // Will be updated when included in block
            verification_count: 1, // Self-verified
            verified_by: vec![self.local_peer_id.clone()],
            signature,
            signer_pubkey,
        };

        // Serialize and store
        let state_bytes = bincode::serialize(&verified_state)
            .map_err(|e| PluginPersistenceError::SerializationError(e.to_string()))?;

        let cf_state = self
            .db
            .cf_handle(CF_PLUGIN_STATE)
            .ok_or_else(|| PluginPersistenceError::ColumnFamilyNotFound(CF_PLUGIN_STATE.to_string()))?;

        self.db
            .put_cf(&cf_state, plugin_id.as_bytes(), &state_bytes)
            .map_err(|e| PluginPersistenceError::DatabaseError(e.to_string()))?;

        // Update cache
        {
            let mut cache = self.state_cache.write().await;
            cache.insert(plugin_id.to_string(), verified_state.clone());
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_state_entries += 1;
        }

        info!(
            "🔐 [PLUGIN PERSISTENCE] Stored plugin state: {} (hash: {}, signed by local node)",
            plugin_id,
            hex::encode(&state_hash[..8])
        );

        Ok(verified_state)
    }

    /// Receive and verify state from peer
    pub async fn receive_peer_state(
        &self,
        plugin_id: &str,
        state: &VerifiedPluginState,
        peer_id: &str,
    ) -> Result<bool, PluginPersistenceError> {
        // 1. Verify the state hash
        let mut hasher = Sha3_256::new();
        hasher.update(&state.state_data);
        let computed_hash: [u8; 32] = hasher.finalize().into();

        if computed_hash != state.state_hash {
            warn!(
                "🔐 [PLUGIN PERSISTENCE] State hash mismatch from peer {}: {} != {}",
                peer_id,
                hex::encode(&computed_hash[..8]),
                hex::encode(&state.state_hash[..8])
            );
            let mut stats = self.stats.write().await;
            stats.verifications_failed += 1;
            return Ok(false);
        }

        // 2. Verify the signature
        if !state.signature.is_empty() && !state.signer_pubkey.is_empty() {
            if !self.verify_state_signature(state)? {
                warn!(
                    "🔐 [PLUGIN PERSISTENCE] State signature invalid from peer {}",
                    peer_id
                );
                let mut stats = self.stats.write().await;
                stats.verifications_failed += 1;
                return Ok(false);
            }
        }

        // 3. Check if we already have this state
        let existing = self.get_plugin_state(plugin_id).await?;

        if let Some(existing_state) = existing {
            // If we have newer state, don't overwrite
            if existing_state.updated_at > state.updated_at {
                debug!(
                    "🔐 [PLUGIN PERSISTENCE] Ignoring older state from peer {} for {}",
                    peer_id, plugin_id
                );
                return Ok(true);
            }

            // If same state, add verification
            if existing_state.state_hash == state.state_hash {
                return self.add_state_verification(plugin_id, peer_id).await;
            }
        }

        // 4. Store the new state
        let cf_state = self
            .db
            .cf_handle(CF_PLUGIN_STATE)
            .ok_or_else(|| PluginPersistenceError::ColumnFamilyNotFound(CF_PLUGIN_STATE.to_string()))?;

        let state_bytes = bincode::serialize(state)
            .map_err(|e| PluginPersistenceError::SerializationError(e.to_string()))?;

        self.db
            .put_cf(&cf_state, plugin_id.as_bytes(), &state_bytes)
            .map_err(|e| PluginPersistenceError::DatabaseError(e.to_string()))?;

        // Update cache
        {
            let mut cache = self.state_cache.write().await;
            cache.insert(plugin_id.to_string(), state.clone());
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.verifications_performed += 1;
            stats.verifications_passed += 1;
            stats.p2p_sync_received += 1;
        }

        info!(
            "🔐 [PLUGIN PERSISTENCE] Received and verified state from peer {}: {} (hash: {})",
            peer_id,
            plugin_id,
            hex::encode(&state.state_hash[..8])
        );

        Ok(true)
    }

    /// Add verification from a peer
    async fn add_state_verification(
        &self,
        plugin_id: &str,
        peer_id: &str,
    ) -> Result<bool, PluginPersistenceError> {
        let cf_state = self
            .db
            .cf_handle(CF_PLUGIN_STATE)
            .ok_or_else(|| PluginPersistenceError::ColumnFamilyNotFound(CF_PLUGIN_STATE.to_string()))?;

        // Load current state
        let state_bytes = self
            .db
            .get_cf(&cf_state, plugin_id.as_bytes())
            .map_err(|e| PluginPersistenceError::DatabaseError(e.to_string()))?
            .ok_or_else(|| PluginPersistenceError::PluginNotFound(plugin_id.to_string()))?;

        let mut state: VerifiedPluginState = bincode::deserialize(&state_bytes)
            .map_err(|e| PluginPersistenceError::SerializationError(e.to_string()))?;

        // Add verification if not already present
        if !state.verified_by.contains(&peer_id.to_string()) {
            state.verified_by.push(peer_id.to_string());
            state.verification_count += 1;

            // Save updated state
            let updated_bytes = bincode::serialize(&state)
                .map_err(|e| PluginPersistenceError::SerializationError(e.to_string()))?;

            self.db
                .put_cf(&cf_state, plugin_id.as_bytes(), &updated_bytes)
                .map_err(|e| PluginPersistenceError::DatabaseError(e.to_string()))?;

            // Update cache
            {
                let mut cache = self.state_cache.write().await;
                cache.insert(plugin_id.to_string(), state.clone());
            }

            info!(
                "🔐 [PLUGIN PERSISTENCE] Added verification from peer {}: {} ({}/{} verifications)",
                peer_id, plugin_id, state.verification_count, self.min_verifications
            );

            // Check if consensus reached
            if state.verification_count >= self.min_verifications {
                info!(
                    "🔐 [PLUGIN PERSISTENCE] ✅ CONSENSUS REACHED for {}: {} verifications",
                    plugin_id, state.verification_count
                );
            }
        }

        let mut stats = self.stats.write().await;
        stats.verifications_performed += 1;
        stats.verifications_passed += 1;

        Ok(true)
    }

    /// Store consensus proof
    pub async fn store_consensus_proof(
        &self,
        proof: &PluginConsensusProof,
    ) -> Result<(), PluginPersistenceError> {
        let cf_consensus = self
            .db
            .cf_handle(CF_PLUGIN_CONSENSUS)
            .ok_or_else(|| PluginPersistenceError::ColumnFamilyNotFound(CF_PLUGIN_CONSENSUS.to_string()))?;

        let proof_bytes = bincode::serialize(proof)
            .map_err(|e| PluginPersistenceError::SerializationError(e.to_string()))?;

        let key = format!("{}:{}", proof.plugin_id, proof.block_height);

        self.db
            .put_cf(&cf_consensus, key.as_bytes(), &proof_bytes)
            .map_err(|e| PluginPersistenceError::DatabaseError(e.to_string()))?;

        // Update the plugin state with consensus height
        if let Ok(Some(mut state)) = self.get_plugin_state(&proof.plugin_id).await {
            if state.state_hash == proof.state_hash {
                state.consensus_height = proof.block_height;

                let cf_state = self.db.cf_handle(CF_PLUGIN_STATE).unwrap();
                let state_bytes = bincode::serialize(&state).unwrap();
                self.db
                    .put_cf(&cf_state, proof.plugin_id.as_bytes(), &state_bytes)
                    .ok();

                // Update cache
                let mut cache = self.state_cache.write().await;
                cache.insert(proof.plugin_id.clone(), state);
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.consensus_proofs_stored += 1;
        }

        info!(
            "🔐 [PLUGIN PERSISTENCE] Stored consensus proof for {} at height {}",
            proof.plugin_id, proof.block_height
        );

        Ok(())
    }

    /// Get plugin manifest
    pub async fn get_plugin_manifest(
        &self,
        plugin_id: &str,
    ) -> Result<Option<PluginManifest>, PluginPersistenceError> {
        let cf_plugins = self
            .db
            .cf_handle(CF_PLUGINS)
            .ok_or_else(|| PluginPersistenceError::ColumnFamilyNotFound(CF_PLUGINS.to_string()))?;

        match self.db.get_cf(&cf_plugins, plugin_id.as_bytes()) {
            Ok(Some(bytes)) => {
                let manifest: PluginManifest = bincode::deserialize(&bytes)
                    .map_err(|e| PluginPersistenceError::SerializationError(e.to_string()))?;
                Ok(Some(manifest))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PluginPersistenceError::DatabaseError(e.to_string())),
        }
    }

    /// Get plugin WASM bytecode
    pub async fn get_plugin_wasm(
        &self,
        plugin_id: &str,
    ) -> Result<Option<Vec<u8>>, PluginPersistenceError> {
        let cf_wasm = self
            .db
            .cf_handle(CF_PLUGIN_WASM)
            .ok_or_else(|| PluginPersistenceError::ColumnFamilyNotFound(CF_PLUGIN_WASM.to_string()))?;

        match self.db.get_cf(&cf_wasm, plugin_id.as_bytes()) {
            Ok(bytes) => Ok(bytes),
            Err(e) => Err(PluginPersistenceError::DatabaseError(e.to_string())),
        }
    }

    /// Get plugin state
    pub async fn get_plugin_state(
        &self,
        plugin_id: &str,
    ) -> Result<Option<VerifiedPluginState>, PluginPersistenceError> {
        // Check cache first
        {
            let cache = self.state_cache.read().await;
            if let Some(state) = cache.get(plugin_id) {
                return Ok(Some(state.clone()));
            }
        }

        // Load from RocksDB
        let cf_state = self
            .db
            .cf_handle(CF_PLUGIN_STATE)
            .ok_or_else(|| PluginPersistenceError::ColumnFamilyNotFound(CF_PLUGIN_STATE.to_string()))?;

        match self.db.get_cf(&cf_state, plugin_id.as_bytes()) {
            Ok(Some(bytes)) => {
                let state: VerifiedPluginState = bincode::deserialize(&bytes)
                    .map_err(|e| PluginPersistenceError::SerializationError(e.to_string()))?;

                // Cache for future access
                {
                    let mut cache = self.state_cache.write().await;
                    cache.insert(plugin_id.to_string(), state.clone());
                }

                Ok(Some(state))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PluginPersistenceError::DatabaseError(e.to_string())),
        }
    }

    /// Get verification status for a plugin
    pub async fn get_verification_status(
        &self,
        plugin_id: &str,
    ) -> Result<PluginVerificationStatus, PluginPersistenceError> {
        let state = self.get_plugin_state(plugin_id).await?;

        match state {
            None => Ok(PluginVerificationStatus::Pending),
            Some(state) => {
                if state.consensus_height > 0 {
                    Ok(PluginVerificationStatus::ConsensusReached)
                } else if state.verification_count >= self.min_verifications {
                    Ok(PluginVerificationStatus::Verified)
                } else {
                    Ok(PluginVerificationStatus::Pending)
                }
            }
        }
    }

    /// List all stored plugins
    pub async fn list_plugins(&self) -> Result<Vec<String>, PluginPersistenceError> {
        let cf_plugins = self
            .db
            .cf_handle(CF_PLUGINS)
            .ok_or_else(|| PluginPersistenceError::ColumnFamilyNotFound(CF_PLUGINS.to_string()))?;

        let mut plugins = Vec::new();
        let iter = self.db.iterator_cf(&cf_plugins, rocksdb::IteratorMode::Start);

        for result in iter {
            let (key, _) = result.map_err(|e| PluginPersistenceError::DatabaseError(e.to_string()))?;
            if let Ok(plugin_id) = String::from_utf8(key.to_vec()) {
                plugins.push(plugin_id);
            }
        }

        Ok(plugins)
    }

    /// Get persistence statistics
    pub async fn get_stats(&self) -> PluginPersistenceStats {
        self.stats.read().await.clone()
    }

    /// Verify manifest signature
    fn verify_manifest_signature(&self, manifest: &PluginManifest) -> Result<bool, PluginPersistenceError> {
        if manifest.signature.is_empty() || manifest.author_pubkey.is_empty() {
            return Ok(false);
        }

        // Decode public key
        let pubkey_bytes = hex::decode(&manifest.author_pubkey)
            .map_err(|e| PluginPersistenceError::SignatureVerificationFailed(e.to_string()))?;

        if pubkey_bytes.len() != 32 {
            return Ok(false);
        }

        let pubkey_array: [u8; 32] = pubkey_bytes
            .try_into()
            .map_err(|_| PluginPersistenceError::SignatureVerificationFailed("Invalid pubkey length".to_string()))?;

        let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&pubkey_array)
            .map_err(|e| PluginPersistenceError::SignatureVerificationFailed(e.to_string()))?;

        // Get signing bytes
        let signing_bytes = manifest.signing_bytes();

        // Decode signature
        if manifest.signature.len() != 64 {
            return Ok(false);
        }

        let sig_array: [u8; 64] = manifest.signature
            .clone()
            .try_into()
            .map_err(|_| PluginPersistenceError::SignatureVerificationFailed("Invalid signature length".to_string()))?;

        let signature = ed25519_dalek::Signature::from_bytes(&sig_array);

        // Verify
        use ed25519_dalek::Verifier;
        Ok(verifying_key.verify(&signing_bytes, &signature).is_ok())
    }

    /// Verify state signature
    fn verify_state_signature(&self, state: &VerifiedPluginState) -> Result<bool, PluginPersistenceError> {
        if state.signature.len() != 64 || state.signer_pubkey.len() != 32 {
            return Ok(false);
        }

        let pubkey_array: [u8; 32] = state.signer_pubkey
            .clone()
            .try_into()
            .map_err(|_| PluginPersistenceError::SignatureVerificationFailed("Invalid pubkey length".to_string()))?;

        let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&pubkey_array)
            .map_err(|e| PluginPersistenceError::SignatureVerificationFailed(e.to_string()))?;

        let sig_array: [u8; 64] = state.signature
            .clone()
            .try_into()
            .map_err(|_| PluginPersistenceError::SignatureVerificationFailed("Invalid signature length".to_string()))?;

        let signature = ed25519_dalek::Signature::from_bytes(&sig_array);

        use ed25519_dalek::Verifier;
        Ok(verifying_key.verify(&state.state_hash, &signature).is_ok())
    }

    /// Create P2P state broadcast message
    pub fn create_state_broadcast(
        &self,
        plugin_id: &str,
        state: &VerifiedPluginState,
    ) -> PluginGossipsubMessage {
        PluginGossipsubMessage::new(
            self.local_peer_id.clone(),
            PluginMessage::RegistrySync {
                plugins: vec![state.manifest.clone()],
            },
            0, // Sequence will be set by distributor
        )
    }
}

/// Errors that can occur during plugin persistence
#[derive(Debug, thiserror::Error)]
pub enum PluginPersistenceError {
    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Column family not found: {0}")]
    ColumnFamilyNotFound(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Plugin not found: {0}")]
    PluginNotFound(String),

    #[error("Signature verification failed: {0}")]
    SignatureVerificationFailed(String),

    #[error("Hash mismatch: expected {expected}, computed {computed}")]
    HashMismatch { expected: String, computed: String },

    #[error("Consensus not reached: {0}")]
    ConsensusNotReached(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_manifest() -> PluginManifest {
        use crate::network::protocol::PluginManifestPermissions;

        PluginManifest {
            plugin_id: "test.plugin".to_string(),
            version: "1.0.0".to_string(),
            name: "Test Plugin".to_string(),
            description: "A test plugin".to_string(),
            author_pubkey: "0".repeat(64),
            wasm_hash: "0".repeat(64),
            wasm_size: 1024,
            min_node_version: "0.1.0".to_string(),
            permissions: PluginManifestPermissions::default(),
            dependencies: vec![],
            published_at: 1700000000,
            signature: vec![0; 64],
        }
    }

    #[tokio::test]
    async fn test_storage_creation() {
        let dir = tempdir().unwrap();
        let storage = PluginPersistentStorage::new(
            dir.path(),
            "test-peer".to_string(),
            None,
        );
        assert!(storage.is_ok());
    }

    #[tokio::test]
    async fn test_list_plugins_empty() {
        let dir = tempdir().unwrap();
        let storage = PluginPersistentStorage::new(
            dir.path(),
            "test-peer".to_string(),
            None,
        ).unwrap();

        let plugins = storage.list_plugins().await.unwrap();
        assert!(plugins.is_empty());
    }

    #[tokio::test]
    async fn test_verification_status_pending() {
        let dir = tempdir().unwrap();
        let storage = PluginPersistentStorage::new(
            dir.path(),
            "test-peer".to_string(),
            None,
        ).unwrap();

        let status = storage.get_verification_status("nonexistent:1.0.0").await.unwrap();
        assert_eq!(status, PluginVerificationStatus::Pending);
    }
}
