//! Plugin Distribution Manager
//!
//! Manages the distribution of plugins across the P2P network.
//! Handles announcing new plugins, requesting plugins from peers,
//! and caching downloaded plugins locally.
//!
//! ## Architecture
//!
//! The `PluginDistributor` maintains:
//! - **Local plugin cache**: WASM bytecode of installed plugins
//! - **Pending downloads**: Tracks in-progress plugin downloads
//! - **Registry**: Known plugin manifests from the network
//!
//! ## Chunked Transfer
//!
//! Large WASM files are split into chunks for efficient transfer:
//! - Default chunk size: 256 KB
//! - Each chunk is verified with SHA3-256
//! - Out-of-order delivery is supported
//! - Resumable downloads on reconnection

use super::protocol::{
    PluginChunkData, PluginGossipsubMessage, PluginManifest, PluginMessage,
    PLUGIN_PROTOCOL_VERSION,
};
use super::verification::{PluginVerificationError, PluginVerifier};
use super::{DEFAULT_CHUNK_SIZE, MAX_CHUNKS_PER_PLUGIN, MAX_PLUGIN_SIZE};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Configuration for the plugin distributor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDistributorConfig {
    /// Network identifier (e.g., "testnet-phase19")
    pub network_id: String,
    /// Chunk size in bytes (default: 256 KB)
    pub chunk_size: usize,
    /// Maximum concurrent downloads
    pub max_concurrent_downloads: usize,
    /// Download timeout in seconds
    pub download_timeout_secs: u64,
    /// Maximum retries per chunk
    pub max_chunk_retries: u32,
    /// Enable automatic registry sync
    pub auto_registry_sync: bool,
    /// Registry sync interval in seconds
    pub registry_sync_interval_secs: u64,
}

impl Default for PluginDistributorConfig {
    fn default() -> Self {
        Self {
            network_id: "testnet-phase19".to_string(),
            chunk_size: DEFAULT_CHUNK_SIZE,
            max_concurrent_downloads: 5,
            download_timeout_secs: 300, // 5 minutes
            max_chunk_retries: 3,
            auto_registry_sync: true,
            registry_sync_interval_secs: 60, // 1 minute
        }
    }
}

/// Download state for a plugin being downloaded
#[derive(Debug, Clone)]
pub struct DownloadState {
    /// Plugin manifest
    pub manifest: PluginManifest,
    /// Total chunks expected
    pub total_chunks: u32,
    /// Received chunks (chunk_index -> data)
    pub received_chunks: HashMap<u32, Vec<u8>>,
    /// Chunk hashes for verification
    pub chunk_hashes: HashMap<u32, [u8; 32]>,
    /// Download start timestamp
    pub started_at: u64,
    /// Last activity timestamp
    pub last_activity: u64,
    /// Retry counts per chunk
    pub chunk_retries: HashMap<u32, u32>,
    /// Peer IDs that have this plugin
    pub known_sources: Vec<String>,
}

impl DownloadState {
    /// Create a new download state from an announcement
    pub fn new(manifest: PluginManifest, total_chunks: u32, source_peer: String) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            manifest,
            total_chunks,
            received_chunks: HashMap::new(),
            chunk_hashes: HashMap::new(),
            started_at: now,
            last_activity: now,
            chunk_retries: HashMap::new(),
            known_sources: vec![source_peer],
        }
    }

    /// Check if all chunks have been received
    pub fn is_complete(&self) -> bool {
        self.received_chunks.len() == self.total_chunks as usize
    }

    /// Get the list of missing chunk indices
    pub fn missing_chunks(&self) -> Vec<u32> {
        (0..self.total_chunks)
            .filter(|i| !self.received_chunks.contains_key(i))
            .collect()
    }

    /// Add a received chunk
    pub fn add_chunk(&mut self, chunk_index: u32, data: Vec<u8>, hash: [u8; 32]) -> bool {
        if chunk_index >= self.total_chunks {
            return false;
        }

        // Verify chunk hash
        let mut hasher = Sha3_256::new();
        hasher.update(&data);
        let computed_hash: [u8; 32] = hasher.finalize().into();

        if computed_hash != hash {
            warn!(
                "Chunk {} hash mismatch for plugin {}",
                chunk_index,
                self.manifest.unique_id()
            );
            return false;
        }

        self.received_chunks.insert(chunk_index, data);
        self.chunk_hashes.insert(chunk_index, hash);
        self.last_activity = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        true
    }

    /// Assemble the complete WASM bytecode from chunks
    pub fn assemble(&self) -> Option<Vec<u8>> {
        if !self.is_complete() {
            return None;
        }

        let mut wasm_bytes = Vec::with_capacity(self.manifest.wasm_size as usize);

        for i in 0..self.total_chunks {
            if let Some(chunk) = self.received_chunks.get(&i) {
                wasm_bytes.extend_from_slice(chunk);
            } else {
                return None;
            }
        }

        // Truncate to exact size (last chunk may have padding)
        wasm_bytes.truncate(self.manifest.wasm_size as usize);

        Some(wasm_bytes)
    }
}

/// Progress information for a download
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    /// Plugin ID
    pub plugin_id: String,
    /// Total chunks
    pub total_chunks: u32,
    /// Received chunks
    pub received_chunks: u32,
    /// Progress percentage (0-100)
    pub progress_percent: f32,
    /// Bytes downloaded
    pub bytes_downloaded: u64,
    /// Total bytes expected
    pub total_bytes: u64,
    /// Download state
    pub state: DownloadProgressState,
}

/// Download progress state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DownloadProgressState {
    /// Waiting for chunks
    Downloading,
    /// Verifying downloaded data
    Verifying,
    /// Download complete
    Complete,
    /// Download failed
    Failed(String),
}

/// Statistics for the plugin distributor
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginDistributorStats {
    /// Total plugins in local cache
    pub cached_plugins: u64,
    /// Total bytes in cache
    pub cache_size_bytes: u64,
    /// Active downloads
    pub active_downloads: u64,
    /// Completed downloads
    pub completed_downloads: u64,
    /// Failed downloads
    pub failed_downloads: u64,
    /// Chunks sent to peers
    pub chunks_sent: u64,
    /// Chunks received from peers
    pub chunks_received: u64,
    /// Plugin announcements sent
    pub announcements_sent: u64,
    /// Plugin announcements received
    pub announcements_received: u64,
    /// Registry syncs performed
    pub registry_syncs: u64,
}

/// Plugin distributor - manages P2P plugin distribution
pub struct PluginDistributor {
    /// Configuration
    config: PluginDistributorConfig,
    /// Local plugin cache: plugin_id -> WASM bytecode
    local_plugins: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    /// Known plugin manifests
    registry: Arc<RwLock<HashMap<String, PluginManifest>>>,
    /// Pending downloads
    pending_downloads: Arc<RwLock<HashMap<String, DownloadState>>>,
    /// Plugin verifier
    verifier: Arc<PluginVerifier>,
    /// Gossipsub topic for this network
    gossipsub_topic: String,
    /// Message sequence number
    sequence_number: AtomicU64,
    /// Our peer ID
    local_peer_id: String,
    /// Statistics
    stats: Arc<RwLock<PluginDistributorStats>>,
}

impl PluginDistributor {
    /// Create a new plugin distributor
    pub fn new(
        config: PluginDistributorConfig,
        local_peer_id: String,
        verifier: Arc<PluginVerifier>,
    ) -> Self {
        let gossipsub_topic = super::plugin_topic(&config.network_id);

        info!(
            "Initializing PluginDistributor for network '{}' on topic '{}'",
            config.network_id, gossipsub_topic
        );

        Self {
            config,
            local_plugins: Arc::new(RwLock::new(HashMap::new())),
            registry: Arc::new(RwLock::new(HashMap::new())),
            pending_downloads: Arc::new(RwLock::new(HashMap::new())),
            verifier,
            gossipsub_topic,
            sequence_number: AtomicU64::new(0),
            local_peer_id,
            stats: Arc::new(RwLock::new(PluginDistributorStats::default())),
        }
    }

    /// Get the gossipsub topic for plugin messages
    pub fn gossipsub_topic(&self) -> &str {
        &self.gossipsub_topic
    }

    /// Get the next sequence number
    fn next_sequence(&self) -> u64 {
        self.sequence_number.fetch_add(1, Ordering::SeqCst)
    }

    /// Announce a new plugin to the network
    ///
    /// # Arguments
    /// * `manifest` - The plugin manifest
    /// * `wasm_bytes` - The WASM bytecode
    ///
    /// # Returns
    /// The gossipsub message to broadcast, or an error
    pub async fn announce_plugin(
        &self,
        manifest: PluginManifest,
        wasm_bytes: Vec<u8>,
    ) -> Result<PluginGossipsubMessage, PluginDistributionError> {
        let plugin_id = manifest.unique_id();

        // Verify the manifest signature
        let manifest_result = self.verifier.verify_manifest(&manifest).await
            .map_err(PluginDistributionError::VerificationError)?;
        if !manifest_result.valid {
            return Err(PluginDistributionError::InvalidManifest(
                "Manifest signature verification failed".to_string(),
            ));
        }

        // Verify WASM hash
        let wasm_result = self.verifier.verify_wasm(&manifest, &wasm_bytes)
            .map_err(PluginDistributionError::VerificationError)?;
        if !wasm_result.hash_valid {
            return Err(PluginDistributionError::HashMismatch(
                "WASM hash does not match manifest".to_string(),
            ));
        }

        // Check size limits
        if wasm_bytes.len() > MAX_PLUGIN_SIZE {
            return Err(PluginDistributionError::PluginTooLarge(
                wasm_bytes.len(),
                MAX_PLUGIN_SIZE,
            ));
        }

        // Store in local cache
        {
            let mut cache = self.local_plugins.write().await;
            cache.insert(plugin_id.clone(), wasm_bytes.clone());
        }

        // Add to registry
        {
            let mut registry = self.registry.write().await;
            registry.insert(plugin_id.clone(), manifest.clone());
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.cached_plugins += 1;
            stats.cache_size_bytes += wasm_bytes.len() as u64;
            stats.announcements_sent += 1;
        }

        let chunk_count = manifest.chunk_count(self.config.chunk_size);

        info!(
            "Announcing plugin '{}' ({} bytes, {} chunks)",
            plugin_id,
            wasm_bytes.len(),
            chunk_count
        );

        // Create announcement message
        let message = PluginGossipsubMessage::new(
            self.local_peer_id.clone(),
            PluginMessage::Announce {
                manifest,
                chunk_count,
            },
            self.next_sequence(),
        );

        Ok(message)
    }

    /// Request a plugin from the network
    ///
    /// # Arguments
    /// * `plugin_id` - The plugin identifier (plugin_id:version)
    ///
    /// # Returns
    /// The gossipsub message to broadcast
    pub async fn request_plugin(
        &self,
        plugin_id: &str,
    ) -> Result<PluginGossipsubMessage, PluginDistributionError> {
        // Check if already cached
        {
            let cache = self.local_plugins.read().await;
            if cache.contains_key(plugin_id) {
                return Err(PluginDistributionError::AlreadyCached(plugin_id.to_string()));
            }
        }

        // Check if already downloading
        {
            let downloads = self.pending_downloads.read().await;
            if downloads.contains_key(plugin_id) {
                // Request the first missing chunk
                if let Some(download) = downloads.get(plugin_id) {
                    if let Some(missing) = download.missing_chunks().first() {
                        return Ok(PluginGossipsubMessage::new(
                            self.local_peer_id.clone(),
                            PluginMessage::RequestPlugin {
                                plugin_id: plugin_id.to_string(),
                                chunk_index: *missing,
                            },
                            self.next_sequence(),
                        ));
                    }
                }
                return Err(PluginDistributionError::DownloadInProgress(
                    plugin_id.to_string(),
                ));
            }
        }

        // First, query availability to get manifest
        info!("Querying availability for plugin '{}'", plugin_id);

        Ok(PluginGossipsubMessage::new(
            self.local_peer_id.clone(),
            PluginMessage::QueryAvailability {
                plugin_id: plugin_id.to_string(),
            },
            self.next_sequence(),
        ))
    }

    /// Handle an incoming plugin message from a peer
    ///
    /// # Arguments
    /// * `peer_id` - The sender's peer ID
    /// * `message` - The received message
    ///
    /// # Returns
    /// Optional response message to send, or an error
    pub async fn handle_message(
        &self,
        peer_id: &str,
        message: PluginGossipsubMessage,
    ) -> Result<Option<PluginGossipsubMessage>, PluginDistributionError> {
        // Validate timestamp
        if !message.is_timestamp_valid() {
            return Err(PluginDistributionError::InvalidTimestamp);
        }

        // Validate protocol version
        if message.protocol_version > PLUGIN_PROTOCOL_VERSION {
            warn!(
                "Received message with newer protocol version {} from peer {}",
                message.protocol_version, peer_id
            );
        }

        match message.payload {
            PluginMessage::Announce {
                manifest,
                chunk_count,
            } => {
                self.handle_announce(peer_id, manifest, chunk_count).await
            }

            PluginMessage::RequestPlugin {
                plugin_id,
                chunk_index,
            } => self.handle_request_plugin(&plugin_id, chunk_index).await,

            PluginMessage::PluginChunk {
                plugin_id,
                chunk_index,
                total_chunks,
                data,
                chunk_hash,
            } => {
                self.handle_plugin_chunk(peer_id, &plugin_id, chunk_index, total_chunks, data, chunk_hash)
                    .await
            }

            PluginMessage::RegistrySync { plugins } => {
                self.handle_registry_sync(plugins).await
            }

            PluginMessage::RequestRegistrySync {
                since_timestamp,
                max_count,
            } => {
                self.handle_request_registry_sync(since_timestamp, max_count)
                    .await
            }

            PluginMessage::QueryAvailability { plugin_id } => {
                self.handle_query_availability(&plugin_id).await
            }

            PluginMessage::AvailabilityResponse {
                plugin_id,
                available,
                manifest,
            } => {
                self.handle_availability_response(peer_id, &plugin_id, available, manifest)
                    .await
            }
        }
    }

    /// Handle a plugin announcement
    async fn handle_announce(
        &self,
        peer_id: &str,
        manifest: PluginManifest,
        chunk_count: u32,
    ) -> Result<Option<PluginGossipsubMessage>, PluginDistributionError> {
        let plugin_id = manifest.unique_id();

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.announcements_received += 1;
        }

        // Verify manifest signature
        match self.verifier.verify_manifest(&manifest).await {
            Ok(result) if result.valid => {}
            Ok(_) => {
                warn!(
                    "Ignoring plugin '{}' with invalid signature from peer {}",
                    plugin_id, peer_id
                );
                return Ok(None);
            }
            Err(e) => {
                warn!(
                    "Error verifying manifest for plugin '{}' from peer {}: {}",
                    plugin_id, peer_id, e
                );
                return Ok(None);
            }
        }

        // Check if we already have this plugin
        {
            let cache = self.local_plugins.read().await;
            if cache.contains_key(&plugin_id) {
                debug!("Already have plugin '{}', ignoring announcement", plugin_id);
                return Ok(None);
            }
        }

        // Check if we're already downloading
        {
            let mut downloads = self.pending_downloads.write().await;
            if let Some(download) = downloads.get_mut(&plugin_id) {
                // Add this peer as a source
                if !download.known_sources.contains(&peer_id.to_string()) {
                    download.known_sources.push(peer_id.to_string());
                }
                return Ok(None);
            }
        }

        // Add to registry
        {
            let mut registry = self.registry.write().await;
            registry.insert(plugin_id.clone(), manifest.clone());
        }

        info!(
            "Received announcement for plugin '{}' from peer {} ({} chunks)",
            plugin_id, peer_id, chunk_count
        );

        // Start downloading by requesting chunk 0
        {
            let mut downloads = self.pending_downloads.write().await;
            downloads.insert(
                plugin_id.clone(),
                DownloadState::new(manifest, chunk_count, peer_id.to_string()),
            );

            let mut stats = self.stats.write().await;
            stats.active_downloads += 1;
        }

        // Request first chunk
        Ok(Some(PluginGossipsubMessage::new(
            self.local_peer_id.clone(),
            PluginMessage::RequestPlugin {
                plugin_id,
                chunk_index: 0,
            },
            self.next_sequence(),
        )))
    }

    /// Handle a plugin request
    async fn handle_request_plugin(
        &self,
        plugin_id: &str,
        chunk_index: u32,
    ) -> Result<Option<PluginGossipsubMessage>, PluginDistributionError> {
        // Check if we have this plugin
        let wasm_bytes = {
            let cache = self.local_plugins.read().await;
            match cache.get(plugin_id) {
                Some(bytes) => bytes.clone(),
                None => {
                    debug!(
                        "Received request for unknown plugin '{}', chunk {}",
                        plugin_id, chunk_index
                    );
                    return Ok(None);
                }
            }
        };

        // Calculate chunk data
        let chunk_size = self.config.chunk_size;
        let total_chunks = (wasm_bytes.len() + chunk_size - 1) / chunk_size;

        if chunk_index as usize >= total_chunks {
            warn!(
                "Received request for invalid chunk {} of plugin '{}' (total: {})",
                chunk_index, plugin_id, total_chunks
            );
            return Ok(None);
        }

        let start = chunk_index as usize * chunk_size;
        let end = std::cmp::min(start + chunk_size, wasm_bytes.len());
        let data = wasm_bytes[start..end].to_vec();

        // Compute chunk hash
        let mut hasher = Sha3_256::new();
        hasher.update(&data);
        let chunk_hash: [u8; 32] = hasher.finalize().into();

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.chunks_sent += 1;
        }

        debug!(
            "Sending chunk {}/{} for plugin '{}' ({} bytes)",
            chunk_index + 1,
            total_chunks,
            plugin_id,
            data.len()
        );

        Ok(Some(PluginGossipsubMessage::new(
            self.local_peer_id.clone(),
            PluginMessage::PluginChunk {
                plugin_id: plugin_id.to_string(),
                chunk_index,
                total_chunks: total_chunks as u32,
                data,
                chunk_hash,
            },
            self.next_sequence(),
        )))
    }

    /// Handle a received plugin chunk
    async fn handle_plugin_chunk(
        &self,
        peer_id: &str,
        plugin_id: &str,
        chunk_index: u32,
        total_chunks: u32,
        data: Vec<u8>,
        chunk_hash: [u8; 32],
    ) -> Result<Option<PluginGossipsubMessage>, PluginDistributionError> {
        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.chunks_received += 1;
        }

        // Check if we're downloading this plugin
        let (is_complete, missing_chunk) = {
            let mut downloads = self.pending_downloads.write().await;

            let download = match downloads.get_mut(plugin_id) {
                Some(d) => d,
                None => {
                    debug!(
                        "Received chunk for unknown download '{}', ignoring",
                        plugin_id
                    );
                    return Ok(None);
                }
            };

            // Add the chunk
            if !download.add_chunk(chunk_index, data, chunk_hash) {
                warn!(
                    "Failed to add chunk {} for plugin '{}' from peer {}",
                    chunk_index, plugin_id, peer_id
                );
                return Ok(None);
            }

            debug!(
                "Received chunk {}/{} for plugin '{}' from peer {}",
                chunk_index + 1,
                total_chunks,
                plugin_id,
                peer_id
            );

            let is_complete = download.is_complete();
            let missing = download.missing_chunks().first().copied();

            (is_complete, missing)
        };

        if is_complete {
            // Finalize the download
            return self.finalize_download(plugin_id).await;
        }

        // Request next missing chunk
        if let Some(next_chunk) = missing_chunk {
            return Ok(Some(PluginGossipsubMessage::new(
                self.local_peer_id.clone(),
                PluginMessage::RequestPlugin {
                    plugin_id: plugin_id.to_string(),
                    chunk_index: next_chunk,
                },
                self.next_sequence(),
            )));
        }

        Ok(None)
    }

    /// Finalize a completed download
    async fn finalize_download(
        &self,
        plugin_id: &str,
    ) -> Result<Option<PluginGossipsubMessage>, PluginDistributionError> {
        let download = {
            let mut downloads = self.pending_downloads.write().await;
            downloads.remove(plugin_id)
        };

        let download = match download {
            Some(d) => d,
            None => return Ok(None),
        };

        // Assemble WASM bytes
        let wasm_bytes = match download.assemble() {
            Some(bytes) => bytes,
            None => {
                error!("Failed to assemble plugin '{}' from chunks", plugin_id);
                let mut stats = self.stats.write().await;
                stats.active_downloads -= 1;
                stats.failed_downloads += 1;
                return Ok(None);
            }
        };

        // Verify WASM hash
        let wasm_result = self
            .verifier
            .verify_wasm(&download.manifest, &wasm_bytes)
            .map_err(PluginDistributionError::VerificationError)?;

        if !wasm_result.hash_valid {
            error!(
                "WASM hash mismatch for plugin '{}' after download",
                plugin_id
            );
            let mut stats = self.stats.write().await;
            stats.active_downloads -= 1;
            stats.failed_downloads += 1;
            return Err(PluginDistributionError::HashMismatch(
                "Assembled WASM hash does not match manifest".to_string(),
            ));
        }

        // Store in local cache
        {
            let mut cache = self.local_plugins.write().await;
            cache.insert(plugin_id.to_string(), wasm_bytes.clone());
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.active_downloads -= 1;
            stats.completed_downloads += 1;
            stats.cached_plugins += 1;
            stats.cache_size_bytes += wasm_bytes.len() as u64;
        }

        info!(
            "Successfully downloaded plugin '{}' ({} bytes)",
            plugin_id,
            wasm_bytes.len()
        );

        Ok(None)
    }

    /// Handle a registry sync message
    async fn handle_registry_sync(
        &self,
        plugins: Vec<PluginManifest>,
    ) -> Result<Option<PluginGossipsubMessage>, PluginDistributionError> {
        let mut added = 0;

        for manifest in plugins {
            // Verify signature
            let verify_result = self.verifier.verify_manifest(&manifest).await;
            let is_valid = verify_result.map(|r| r.valid).unwrap_or(false);
            if !is_valid {
                continue;
            }

            let plugin_id = manifest.unique_id();

            // Add to registry if not already present
            let mut registry = self.registry.write().await;
            if !registry.contains_key(&plugin_id) {
                registry.insert(plugin_id, manifest);
                added += 1;
            }
        }

        if added > 0 {
            let mut stats = self.stats.write().await;
            stats.registry_syncs += 1;
            info!("Added {} plugins to registry from sync", added);
        }

        Ok(None)
    }

    /// Handle a request for registry sync
    async fn handle_request_registry_sync(
        &self,
        since_timestamp: Option<u64>,
        max_count: Option<u32>,
    ) -> Result<Option<PluginGossipsubMessage>, PluginDistributionError> {
        let registry = self.registry.read().await;

        let mut plugins: Vec<PluginManifest> = registry
            .values()
            .filter(|m| {
                if let Some(since) = since_timestamp {
                    m.published_at > since
                } else {
                    true
                }
            })
            .cloned()
            .collect();

        // Sort by published_at descending
        plugins.sort_by(|a, b| b.published_at.cmp(&a.published_at));

        // Apply max count limit
        if let Some(max) = max_count {
            plugins.truncate(max as usize);
        }

        if plugins.is_empty() {
            return Ok(None);
        }

        Ok(Some(PluginGossipsubMessage::new(
            self.local_peer_id.clone(),
            PluginMessage::RegistrySync { plugins },
            self.next_sequence(),
        )))
    }

    /// Handle an availability query
    async fn handle_query_availability(
        &self,
        plugin_id: &str,
    ) -> Result<Option<PluginGossipsubMessage>, PluginDistributionError> {
        let cache = self.local_plugins.read().await;
        let registry = self.registry.read().await;

        let available = cache.contains_key(plugin_id);
        let manifest = if available {
            registry.get(plugin_id).cloned()
        } else {
            None
        };

        Ok(Some(PluginGossipsubMessage::new(
            self.local_peer_id.clone(),
            PluginMessage::AvailabilityResponse {
                plugin_id: plugin_id.to_string(),
                available,
                manifest,
            },
            self.next_sequence(),
        )))
    }

    /// Handle an availability response
    async fn handle_availability_response(
        &self,
        peer_id: &str,
        plugin_id: &str,
        available: bool,
        manifest: Option<PluginManifest>,
    ) -> Result<Option<PluginGossipsubMessage>, PluginDistributionError> {
        if !available {
            return Ok(None);
        }

        let manifest = match manifest {
            Some(m) => m,
            None => return Ok(None),
        };

        // Verify manifest
        let verify_result = self.verifier.verify_manifest(&manifest).await;
        let is_valid = verify_result.map(|r| r.valid).unwrap_or(false);
        if !is_valid {
            warn!(
                "Received invalid manifest for '{}' from peer {}",
                plugin_id, peer_id
            );
            return Ok(None);
        }

        // Check if we already have this plugin
        {
            let cache = self.local_plugins.read().await;
            if cache.contains_key(plugin_id) {
                return Ok(None);
            }
        }

        // Check if already downloading
        {
            let downloads = self.pending_downloads.read().await;
            if downloads.contains_key(plugin_id) {
                return Ok(None);
            }
        }

        // Start download
        let chunk_count = manifest.chunk_count(self.config.chunk_size);

        info!(
            "Starting download of plugin '{}' from peer {} ({} chunks)",
            plugin_id, peer_id, chunk_count
        );

        {
            let mut downloads = self.pending_downloads.write().await;
            downloads.insert(
                plugin_id.to_string(),
                DownloadState::new(manifest, chunk_count, peer_id.to_string()),
            );

            let mut stats = self.stats.write().await;
            stats.active_downloads += 1;
        }

        // Request first chunk
        Ok(Some(PluginGossipsubMessage::new(
            self.local_peer_id.clone(),
            PluginMessage::RequestPlugin {
                plugin_id: plugin_id.to_string(),
                chunk_index: 0,
            },
            self.next_sequence(),
        )))
    }

    /// Get a cached plugin's WASM bytecode
    pub async fn get_cached_plugin(&self, plugin_id: &str) -> Option<Vec<u8>> {
        let cache = self.local_plugins.read().await;
        cache.get(plugin_id).cloned()
    }

    /// Check if a plugin is cached locally
    pub async fn is_plugin_cached(&self, plugin_id: &str) -> bool {
        let cache = self.local_plugins.read().await;
        cache.contains_key(plugin_id)
    }

    /// Get the manifest for a known plugin
    pub async fn get_manifest(&self, plugin_id: &str) -> Option<PluginManifest> {
        let registry = self.registry.read().await;
        registry.get(plugin_id).cloned()
    }

    /// Get all known plugin manifests
    pub async fn get_all_manifests(&self) -> Vec<PluginManifest> {
        let registry = self.registry.read().await;
        registry.values().cloned().collect()
    }

    /// Get download progress for a plugin
    pub async fn get_download_progress(&self, plugin_id: &str) -> Option<DownloadProgress> {
        let downloads = self.pending_downloads.read().await;

        let download = downloads.get(plugin_id)?;

        let received = download.received_chunks.len() as u32;
        let total = download.total_chunks;
        let bytes_downloaded: u64 = download
            .received_chunks
            .values()
            .map(|c| c.len() as u64)
            .sum();

        Some(DownloadProgress {
            plugin_id: plugin_id.to_string(),
            total_chunks: total,
            received_chunks: received,
            progress_percent: if total > 0 {
                (received as f32 / total as f32) * 100.0
            } else {
                0.0
            },
            bytes_downloaded,
            total_bytes: download.manifest.wasm_size,
            state: if download.is_complete() {
                DownloadProgressState::Verifying
            } else {
                DownloadProgressState::Downloading
            },
        })
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> PluginDistributorStats {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// Remove a plugin from the local cache
    pub async fn remove_plugin(&self, plugin_id: &str) -> bool {
        let removed = {
            let mut cache = self.local_plugins.write().await;
            cache.remove(plugin_id).is_some()
        };

        if removed {
            let mut stats = self.stats.write().await;
            stats.cached_plugins = stats.cached_plugins.saturating_sub(1);
        }

        removed
    }

    /// Cancel a pending download
    pub async fn cancel_download(&self, plugin_id: &str) -> bool {
        let removed = {
            let mut downloads = self.pending_downloads.write().await;
            downloads.remove(plugin_id).is_some()
        };

        if removed {
            let mut stats = self.stats.write().await;
            stats.active_downloads = stats.active_downloads.saturating_sub(1);
        }

        removed
    }
}

/// Errors that can occur during plugin distribution
#[derive(Debug, thiserror::Error)]
pub enum PluginDistributionError {
    #[error("Invalid manifest: {0}")]
    InvalidManifest(String),

    #[error("Hash mismatch: {0}")]
    HashMismatch(String),

    #[error("Plugin too large: {0} bytes (max: {1})")]
    PluginTooLarge(usize, usize),

    #[error("Plugin already cached: {0}")]
    AlreadyCached(String),

    #[error("Download already in progress: {0}")]
    DownloadInProgress(String),

    #[error("Invalid timestamp")]
    InvalidTimestamp,

    #[error("Verification error: {0}")]
    VerificationError(#[from] PluginVerificationError),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_manifest() -> PluginManifest {
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

    #[test]
    fn test_download_state_chunks() {
        let manifest = create_test_manifest();
        let mut state = DownloadState::new(manifest, 4, "peer1".to_string());

        assert!(!state.is_complete());
        assert_eq!(state.missing_chunks(), vec![0, 1, 2, 3]);

        // Add chunk 0 with valid hash
        let data0 = vec![1, 2, 3];
        let mut hasher = Sha3_256::new();
        hasher.update(&data0);
        let hash0: [u8; 32] = hasher.finalize().into();
        assert!(state.add_chunk(0, data0, hash0));

        assert!(!state.is_complete());
        assert_eq!(state.missing_chunks(), vec![1, 2, 3]);

        // Add remaining chunks
        for i in 1..4 {
            let data = vec![i as u8; 10];
            let mut hasher = Sha3_256::new();
            hasher.update(&data);
            let hash: [u8; 32] = hasher.finalize().into();
            assert!(state.add_chunk(i, data, hash));
        }

        assert!(state.is_complete());
        assert!(state.missing_chunks().is_empty());
    }

    #[test]
    fn test_download_state_invalid_chunk() {
        let manifest = create_test_manifest();
        let mut state = DownloadState::new(manifest, 2, "peer1".to_string());

        // Try to add chunk with wrong hash
        let data = vec![1, 2, 3];
        let wrong_hash = [0u8; 32]; // Wrong hash
        assert!(!state.add_chunk(0, data, wrong_hash));
        assert!(!state.is_complete());
    }

    #[test]
    fn test_download_state_assemble() {
        let mut manifest = create_test_manifest();
        manifest.wasm_size = 15; // 3 chunks of 5 bytes each

        let mut state = DownloadState::new(manifest, 3, "peer1".to_string());

        // Add 3 chunks
        for i in 0..3 {
            let data = vec![(i + 1) as u8; 5];
            let mut hasher = Sha3_256::new();
            hasher.update(&data);
            let hash: [u8; 32] = hasher.finalize().into();
            state.add_chunk(i, data, hash);
        }

        let assembled = state.assemble().unwrap();
        assert_eq!(assembled.len(), 15);
        assert_eq!(&assembled[0..5], &[1, 1, 1, 1, 1]);
        assert_eq!(&assembled[5..10], &[2, 2, 2, 2, 2]);
        assert_eq!(&assembled[10..15], &[3, 3, 3, 3, 3]);
    }

    #[tokio::test]
    async fn test_plugin_distributor_creation() {
        let config = PluginDistributorConfig::default();
        let verifier = Arc::new(PluginVerifier::new(None));
        let distributor = PluginDistributor::new(config, "peer123".to_string(), verifier);

        assert_eq!(
            distributor.gossipsub_topic(),
            "/qnk/testnet-phase19/plugins/v1"
        );
    }

    #[tokio::test]
    async fn test_is_plugin_cached() {
        let config = PluginDistributorConfig::default();
        let verifier = Arc::new(PluginVerifier::new(None));
        let distributor = PluginDistributor::new(config, "peer123".to_string(), verifier);

        // Nothing cached initially
        assert!(!distributor.is_plugin_cached("test:1.0.0").await);

        // Manually insert into cache for testing
        {
            let mut cache = distributor.local_plugins.write().await;
            cache.insert("test:1.0.0".to_string(), vec![1, 2, 3]);
        }

        assert!(distributor.is_plugin_cached("test:1.0.0").await);
    }
}
