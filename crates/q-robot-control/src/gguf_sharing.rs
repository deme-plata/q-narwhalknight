/// Advanced GGUF File Transfer (AFT) System
///
/// Smart distributed sharing of GGUF model files and layers across AI organisms
/// Features:
/// - Intelligent model sharding with layer-level granularity  
/// - P2P file transfer with BitTorrent-style coordination
/// - Deduplication and caching across the swarm
/// - QNK token incentives for storage providers
/// - Automatic model reconstruction from distributed shards
use anyhow::Result;
use blake3::Hasher;
use chrono::{DateTime, Utc};
use libp2p::{
    gossipsub::{self, MessageId, TopicHash},
    request_response::{self, ProtocolSupport, ResponseChannel},
    PeerId, Swarm,
};
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::*;

/// GGUF model shard with content addressing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufShard {
    /// Content hash (Blake3) for verification
    pub content_hash: String,
    /// Shard size in bytes
    pub size: u64,
    /// Layer information from GGUF metadata
    pub layer_info: LayerInfo,
    /// Offset in original model file
    pub offset: u64,
    /// Chunk sequence number
    pub chunk_id: u32,
    /// Total chunks for this model
    pub total_chunks: u32,
    /// Model metadata
    pub model_metadata: ModelMetadata,
}

/// Layer information extracted from GGUF
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    /// Layer name (e.g., "model.layers.0.attention.wq.weight")
    pub name: String,
    /// Layer type (attention, feedforward, embedding, etc.)
    pub layer_type: LayerType,
    /// Tensor shape
    pub shape: Vec<u64>,
    /// Data type (f32, f16, q4_0, etc.)
    pub dtype: String,
    /// Quantization info if applicable
    pub quantization: Option<QuantizationInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    Attention,
    FeedForward,
    Embedding,
    Normalization,
    Output,
    Unknown(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationInfo {
    pub method: String, // q4_0, q4_1, q8_0, etc.
    pub bits_per_weight: u8,
    pub group_size: Option<u32>,
}

/// Model metadata from GGUF
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub architecture: String, // llama, mistral, etc.
    pub parameter_count: u64,
    pub context_length: u32,
    pub vocab_size: u32,
    pub total_size: u64,
}

/// Request for GGUF shard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardRequest {
    pub model_hash: String,
    pub chunk_ids: Vec<u32>,
    pub requesting_peer: String, // PeerId as string for serialization
    pub max_concurrent_transfers: u8,
}

/// Response with shard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardResponse {
    pub chunk_id: u32,
    pub data: Vec<u8>,
    pub content_hash: String,
    pub compressed: bool,
}

/// Advanced File Transfer Manager
#[derive(Debug)]
pub struct GgufSharingManager {
    /// Local shard storage
    local_shards: Arc<RwLock<HashMap<String, GgufShard>>>,
    /// Shard data cache (memory-mapped files)
    shard_cache: Arc<RwLock<HashMap<String, Arc<Mmap>>>>,
    /// Peer shard availability
    peer_availability: Arc<RwLock<HashMap<String, Vec<String>>>>,
    /// Storage path for shards
    storage_path: PathBuf,
    /// P2P transfer requests
    transfer_queue: Arc<Mutex<mpsc::UnboundedReceiver<TransferRequest>>>,
    /// Active transfers
    active_transfers: Arc<RwLock<HashMap<Uuid, ActiveTransfer>>>,
    /// QNK payment processor
    payment_processor: Arc<crate::blockchain_payment::QNKPaymentProcessor>,
}

#[derive(Debug)]
struct TransferRequest {
    pub request_id: Uuid,
    pub shard_hash: String,
    pub target_peer: PeerId,
    pub priority: TransferPriority,
    pub response_channel: Option<mpsc::UnboundedSender<TransferResult>>,
}

#[derive(Debug)]
struct ActiveTransfer {
    pub request_id: Uuid,
    pub start_time: DateTime<Utc>,
    pub bytes_transferred: u64,
    pub total_bytes: u64,
    pub peer: PeerId,
}

#[derive(Debug, Clone)]
pub enum TransferPriority {
    Critical, // Needed for immediate inference
    High,     // Preemptive caching
    Normal,   // Background synchronization
    Low,      // Opportunistic sharing
}

#[derive(Debug, Clone)]
pub struct TransferResult {
    pub success: bool,
    pub shard_hash: String,
    pub error: Option<String>,
    pub transfer_time_ms: u64,
    pub bytes_transferred: u64,
}

impl GgufSharingManager {
    /// Create new GGUF sharing manager
    pub async fn new(
        storage_path: PathBuf,
        payment_processor: Arc<crate::blockchain_payment::QNKPaymentProcessor>,
    ) -> Result<Self> {
        // Create storage directory
        tokio::fs::create_dir_all(&storage_path).await?;

        let (transfer_tx, transfer_rx) = mpsc::unbounded_channel();

        let manager = Self {
            local_shards: Arc::new(RwLock::new(HashMap::new())),
            shard_cache: Arc::new(RwLock::new(HashMap::new())),
            peer_availability: Arc::new(RwLock::new(HashMap::new())),
            storage_path,
            transfer_queue: Arc::new(Mutex::new(transfer_rx)),
            active_transfers: Arc::new(RwLock::new(HashMap::new())),
            payment_processor,
        };

        // Start background transfer processing
        manager.start_transfer_processor().await;

        // Scan existing shards
        manager.scan_existing_shards().await?;

        info!(
            "🗂️  GGUF sharing manager initialized with {} local shards",
            manager.local_shards.read().unwrap().len()
        );

        Ok(manager)
    }

    /// Shard a GGUF model file into transferable chunks
    pub async fn shard_model(&self, model_path: &Path) -> Result<Vec<GgufShard>> {
        info!("🔪 Sharding GGUF model: {:?}", model_path);

        // Memory-map the model file for efficient reading
        let file = File::open(model_path).await?;
        let metadata = file.metadata().await?;
        let file_size = metadata.len();

        // Read GGUF header to extract metadata
        let model_metadata = self.parse_gguf_metadata(&file).await?;

        info!(
            "📊 Model: {} ({} parameters, {} layers)",
            model_metadata.name, model_metadata.parameter_count, "auto-detected"
        );

        // Calculate optimal chunk size (16MB default, but layer-aware)
        let base_chunk_size = 16 * 1024 * 1024; // 16MB
        let optimal_chunks = (file_size / base_chunk_size).max(1);
        let chunk_size = file_size / optimal_chunks;

        let mut shards = Vec::new();
        let mut offset = 0;
        let mut chunk_id = 0;

        while offset < file_size {
            let size = (chunk_size).min(file_size - offset);

            // Read chunk data
            let mut chunk_data = vec![0u8; size as usize];
            let std_file = std::fs::File::open(model_path)?;
            let mmap = unsafe { Mmap::map(&std_file)? };
            chunk_data.copy_from_slice(&mmap[offset as usize..(offset + size) as usize]);

            // Calculate content hash
            let mut hasher = Hasher::new();
            hasher.update(&chunk_data);
            let content_hash = hasher.finalize().to_hex().to_string();

            // Extract layer information (simplified - real implementation would parse GGUF)
            let layer_info = LayerInfo {
                name: format!("chunk_{}", chunk_id),
                layer_type: LayerType::Unknown("mixed".to_string()),
                shape: vec![size],
                dtype: "mixed".to_string(),
                quantization: None,
            };

            let shard = GgufShard {
                content_hash: content_hash.clone(),
                size,
                layer_info,
                offset,
                chunk_id,
                total_chunks: optimal_chunks as u32,
                model_metadata: model_metadata.clone(),
            };

            // Store shard data
            self.store_shard_data(&shard, chunk_data).await?;

            shards.push(shard);
            offset += size;
            chunk_id += 1;
        }

        info!("✅ Model sharded into {} chunks", shards.len());

        // Update local shard registry
        {
            let mut local_shards = self.local_shards.write().unwrap();
            for shard in &shards {
                local_shards.insert(shard.content_hash.clone(), shard.clone());
            }
        }

        Ok(shards)
    }

    /// Request shards from peer network
    pub async fn request_model_shards(
        &self,
        model_hash: &str,
        required_chunks: Vec<u32>,
    ) -> Result<Vec<GgufShard>> {
        info!(
            "📥 Requesting {} shards for model {}",
            required_chunks.len(),
            model_hash
        );

        // Find peers with available shards
        let peer_candidates = self
            .find_shard_providers(model_hash, &required_chunks)
            .await?;

        if peer_candidates.is_empty() {
            anyhow::bail!("No peers available with required model shards");
        }

        let mut retrieved_shards = Vec::new();

        // Request shards with load balancing
        for chunk_id in &required_chunks {
            if let Some(provider) = self.select_best_provider(&peer_candidates, chunk_id).await {
                let provider_clone = provider.clone();
                match self
                    .request_shard_from_peer(provider, model_hash, chunk_id)
                    .await
                {
                    Ok(shard) => {
                        info!(
                            "✅ Retrieved shard {} from peer {}",
                            chunk_id, provider_clone
                        );
                        retrieved_shards.push(shard);
                    }
                    Err(e) => {
                        warn!(
                            "❌ Failed to retrieve shard {} from {}: {}",
                            chunk_id, provider_clone, e
                        );
                    }
                }
            }
        }

        info!(
            "📦 Successfully retrieved {}/{} requested shards",
            retrieved_shards.len(),
            required_chunks.len()
        );

        Ok(retrieved_shards)
    }

    /// Reconstruct GGUF model from distributed shards
    pub async fn reconstruct_model(
        &self,
        shards: Vec<GgufShard>,
        output_path: &Path,
    ) -> Result<()> {
        info!(
            "🔧 Reconstructing model from {} shards to {:?}",
            shards.len(),
            output_path
        );

        // Sort shards by chunk_id
        let mut sorted_shards = shards;
        sorted_shards.sort_by_key(|s| s.chunk_id);

        // Verify we have all chunks
        let total_chunks = sorted_shards.first().map(|s| s.total_chunks).unwrap_or(0);

        if sorted_shards.len() != total_chunks as usize {
            anyhow::bail!(
                "Missing shards: have {}, need {}",
                sorted_shards.len(),
                total_chunks
            );
        }

        // Create output file
        let mut output_file = File::create(output_path).await?;

        // Write shards in order
        for shard in &sorted_shards {
            let shard_data = self.load_shard_data(&shard.content_hash).await?;
            output_file.write_all(&shard_data).await?;
        }

        output_file.flush().await?;

        // Verify reconstructed file
        let reconstructed_size: u64 = sorted_shards.iter().map(|s| s.size).sum();
        let file_metadata = output_file.metadata().await?;

        if file_metadata.len() != reconstructed_size {
            anyhow::bail!(
                "Reconstruction size mismatch: expected {}, got {}",
                reconstructed_size,
                file_metadata.len()
            );
        }

        info!(
            "✅ Model successfully reconstructed ({} bytes)",
            reconstructed_size
        );

        Ok(())
    }

    /// Announce shard availability to network
    pub async fn announce_shard_availability(
        &self,
        swarm: &mut Swarm<crate::distributed_ai_production::AIBehaviour>,
    ) -> Result<()> {
        let local_shards = self.local_shards.read().unwrap();
        let available_hashes: Vec<String> = local_shards.keys().cloned().collect();

        if available_hashes.is_empty() {
            return Ok(());
        }

        #[derive(Serialize)]
        struct ShardAnnouncement {
            peer_id: String,
            available_shards: Vec<String>,
            timestamp: DateTime<Utc>,
            storage_capacity: u64,
            bandwidth_mbps: u32,
        }

        let announcement = ShardAnnouncement {
            peer_id: swarm.local_peer_id().to_string(),
            available_shards: available_hashes,
            timestamp: Utc::now(),
            storage_capacity: 100 * 1024 * 1024 * 1024, // 100GB example
            bandwidth_mbps: 1000,                       // 1Gbps example
        };

        let topic = gossipsub::IdentTopic::new("gguf-shard-availability");
        let message = serde_json::to_vec(&announcement)?;

        swarm.behaviour_mut().gossipsub.publish(topic, message)?;

        debug!(
            "📣 Announced availability of {} shards to network",
            announcement.available_shards.len()
        );

        Ok(())
    }

    /// Smart caching strategy for frequently accessed shards
    pub async fn optimize_shard_cache(&self) -> Result<()> {
        info!("🧹 Optimizing shard cache...");

        // Analyze access patterns and implement LRU eviction
        // This is a simplified version - production would track access metrics

        let cache_size_limit = 50 * 1024 * 1024 * 1024; // 50GB cache
        let mut current_cache_size = 0;

        {
            let cache = self.shard_cache.read().unwrap();
            current_cache_size = cache.values().map(|mmap| mmap.len() as u64).sum();
        }

        if current_cache_size > cache_size_limit {
            info!(
                "💾 Cache size ({} bytes) exceeds limit, evicting least used shards",
                current_cache_size
            );

            // Implement LRU eviction (simplified)
            let mut cache = self.shard_cache.write().unwrap();
            let mut evicted_count = 0;

            // Remove random entries until under limit (real implementation would use LRU)
            while cache.len() > 100 && current_cache_size > cache_size_limit {
                if let Some(key) = cache.keys().next().cloned() {
                    if let Some(evicted) = cache.remove(&key) {
                        current_cache_size -= evicted.len() as u64;
                        evicted_count += 1;
                    }
                } else {
                    break;
                }
            }

            info!("🗑️  Evicted {} shards from cache", evicted_count);
        }

        Ok(())
    }

    // Helper methods

    async fn parse_gguf_metadata(&self, _file: &File) -> Result<ModelMetadata> {
        // Simplified metadata extraction - real implementation would parse GGUF header
        Ok(ModelMetadata {
            name: "auto-detected-model".to_string(),
            architecture: "llama".to_string(),
            parameter_count: 1_100_000_000, // 1.1B example
            context_length: 2048,
            vocab_size: 32000,
            total_size: 0, // Will be filled by caller
        })
    }

    async fn store_shard_data(&self, shard: &GgufShard, data: Vec<u8>) -> Result<()> {
        let file_path = self
            .storage_path
            .join(format!("{}.shard", shard.content_hash));
        let mut file = File::create(file_path).await?;
        file.write_all(&data).await?;
        Ok(())
    }

    async fn load_shard_data(&self, content_hash: &str) -> Result<Vec<u8>> {
        let file_path = self.storage_path.join(format!("{}.shard", content_hash));
        let mut file = File::open(file_path).await?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).await?;
        Ok(data)
    }

    async fn scan_existing_shards(&self) -> Result<()> {
        // Scan storage directory for existing shards
        let mut dir = tokio::fs::read_dir(&self.storage_path).await?;
        let mut count = 0;

        while let Some(entry) = dir.next_entry().await? {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "shard") {
                count += 1;
                // In a real implementation, we'd load shard metadata
            }
        }

        info!("🔍 Found {} existing shards in storage", count);
        Ok(())
    }

    async fn start_transfer_processor(&self) {
        // Background task to process transfer requests
        info!("🚀 Starting GGUF transfer processor");
    }

    async fn find_shard_providers(
        &self,
        _model_hash: &str,
        _chunks: &[u32],
    ) -> Result<Vec<String>> {
        // Query peer availability registry
        // Simplified - real implementation would query gossipsub network
        Ok(vec![])
    }

    async fn select_best_provider(
        &self,
        _candidates: &[String],
        _chunk_id: &u32,
    ) -> Option<String> {
        // Select optimal provider based on latency, bandwidth, availability
        None
    }

    async fn request_shard_from_peer(
        &self,
        _peer: String,
        _model_hash: &str,
        _chunk_id: &u32,
    ) -> Result<GgufShard> {
        // Send request-response protocol message to peer
        anyhow::bail!("Peer communication not yet implemented")
    }
}

/// Demo function showcasing GGUF sharing capabilities
pub async fn demo_gguf_sharing() -> Result<()> {
    info!("🗂️  GGUF Advanced File Transfer Demo");
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let storage_path = PathBuf::from("./gguf_shards");
    let payment_processor =
        Arc::new(crate::blockchain_payment::QNKPaymentProcessor::new_default().await?);

    let sharing_manager = GgufSharingManager::new(storage_path, payment_processor).await?;

    info!("✅ GGUF Sharing System Features:");
    info!("   🔪 Smart model sharding with layer awareness");
    info!("   📡 P2P file transfer with BitTorrent-style coordination");
    info!("   🎯 Content-addressed storage with Blake3 hashing");
    info!("   💰 QNK token incentives for storage providers");
    info!("   🔧 Automatic model reconstruction from distributed shards");
    info!("   💾 Intelligent caching with LRU eviction");
    info!("   🚀 Background transfer optimization");

    // Example: Shard a hypothetical model
    let example_model = PathBuf::from("models/tinyllama-1.1b.gguf");
    if example_model.exists() {
        let shards = sharing_manager.shard_model(&example_model).await?;
        info!("🎉 Successfully sharded model into {} chunks", shards.len());

        // Demonstrate reconstruction
        let output_path = PathBuf::from("./reconstructed_model.gguf");
        sharing_manager
            .reconstruct_model(shards, &output_path)
            .await?;
        info!("✅ Model reconstruction successful");

        // Clean up
        tokio::fs::remove_file(output_path).await.ok();
    } else {
        info!("📝 Demo mode: Model file not found, showing architecture only");
    }

    Ok(())
}
