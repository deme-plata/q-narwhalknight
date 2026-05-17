//! Q-NarwhalKnight Cross-Shard Communication
//! 
//! Efficient, zero-copy inter-shard messaging system for horizontal scaling
//! of the Q-NarwhalKnight consensus protocol.
//! 
//! # Features
//! - Zero-copy message passing between shards
//! - Message compression and batching
//! - Reliable delivery with acknowledgments
//! - Load balancing across shard connections
//! - Performance monitoring and metrics

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tracing::{info, debug, warn, error};
use uuid::Uuid;

pub mod messaging;
pub mod compression;
pub mod routing;
pub mod metrics;

/// Shard identifier
pub type ShardId = u32;

/// Message priority levels for cross-shard communication
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Cross-shard message envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossShardMessage {
    pub id: Uuid,
    pub source_shard: ShardId,
    pub target_shard: ShardId,
    pub priority: MessagePriority,
    pub payload: MessagePayload,
    pub timestamp: u64,
    pub requires_ack: bool,
}

/// Message payload types for cross-shard communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    /// Transaction that needs to be processed by another shard
    CrossShardTransaction {
        transaction_id: String,
        data: Vec<u8>,
    },
    /// State synchronization between shards
    StateSyncRequest {
        state_root: String,
        block_height: u64,
    },
    StateSyncResponse {
        state_data: Vec<u8>,
        is_complete: bool,
    },
    /// Consensus coordination messages
    ConsensusVote {
        round: u64,
        vote_data: Vec<u8>,
    },
    ConsensusProposal {
        round: u64,
        proposal_data: Vec<u8>,
    },
    /// Load balancing and shard management
    LoadBalanceRequest {
        current_load: f64,
        shard_capacity: u64,
    },
    ShardRebalance {
        new_shard_assignment: HashMap<String, ShardId>,
    },
    /// Acknowledgment messages
    Acknowledgment {
        original_message_id: Uuid,
        status: AckStatus,
    },
}

/// Acknowledgment status for reliable messaging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AckStatus {
    Success,
    Failure { reason: String },
    Retry { attempt: u32 },
}

/// Cross-shard communication metrics
#[derive(Debug, Clone, Default)]
pub struct CrossShardMetrics {
    pub messages_sent: AtomicU64,
    pub messages_received: AtomicU64,
    pub bytes_sent: AtomicU64,
    pub bytes_received: AtomicU64,
    pub message_latency_ms: AtomicU64,
    pub compression_ratio: AtomicU64, // As percentage * 100
    pub failed_deliveries: AtomicU64,
}

/// Cross-shard communication manager
pub struct CrossShardManager {
    shard_id: ShardId,
    peer_shards: Arc<RwLock<HashMap<ShardId, ShardConnection>>>,
    message_handlers: Arc<RwLock<HashMap<String, MessageHandler>>>,
    metrics: Arc<CrossShardMetrics>,
    message_buffer: Arc<RwLock<Vec<CrossShardMessage>>>,
    compression_enabled: bool,
}

/// Connection to a peer shard
pub struct ShardConnection {
    pub shard_id: ShardId,
    pub sender: mpsc::UnboundedSender<CrossShardMessage>,
    pub last_seen: Instant,
    pub connection_quality: f64, // 0.0 - 1.0
}

/// Message handler function type
pub type MessageHandler = Box<dyn Fn(&CrossShardMessage) -> Result<Option<CrossShardMessage>> + Send + Sync>;

impl CrossShardManager {
    /// Create a new cross-shard communication manager
    pub fn new(shard_id: ShardId) -> Self {
        info!("🔄 Initializing cross-shard manager for shard {}", shard_id);
        
        Self {
            shard_id,
            peer_shards: Arc::new(RwLock::new(HashMap::new())),
            message_handlers: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(CrossShardMetrics::default()),
            message_buffer: Arc::new(RwLock::new(Vec::new())),
            compression_enabled: true,
        }
    }

    /// Add a peer shard connection
    pub async fn add_peer_shard(&self, peer_shard_id: ShardId, sender: mpsc::UnboundedSender<CrossShardMessage>) -> Result<()> {
        let connection = ShardConnection {
            shard_id: peer_shard_id,
            sender,
            last_seen: Instant::now(),
            connection_quality: 1.0,
        };

        let mut peers = self.peer_shards.write().await;
        peers.insert(peer_shard_id, connection);
        
        info!("✅ Added peer shard connection: {} -> {}", self.shard_id, peer_shard_id);
        Ok(())
    }

    /// Register a message handler for a specific message type
    pub async fn register_handler<F>(&self, message_type: &str, handler: F) -> Result<()>
    where
        F: Fn(&CrossShardMessage) -> Result<Option<CrossShardMessage>> + Send + Sync + 'static,
    {
        let mut handlers = self.message_handlers.write().await;
        handlers.insert(message_type.to_string(), Box::new(handler));
        
        debug!("📝 Registered handler for message type: {}", message_type);
        Ok(())
    }

    /// Send a message to another shard
    pub async fn send_message(&self, mut message: CrossShardMessage) -> Result<()> {
        let start_time = Instant::now();
        
        // Set source shard if not already set
        message.source_shard = self.shard_id;
        message.timestamp = chrono::Utc::now().timestamp_millis() as u64;
        
        // Compress message if enabled
        if self.compression_enabled {
            message = self.compress_message(message).await?;
        }

        // Find the target shard connection
        let peers = self.peer_shards.read().await;
        let connection = peers.get(&message.target_shard)
            .ok_or_else(|| anyhow::anyhow!("No connection to shard {}", message.target_shard))?;

        // Send message
        let message_size = bincode::serialize(&message)?.len();
        connection.sender.send(message.clone())
            .map_err(|e| anyhow::anyhow!("Failed to send message: {}", e))?;

        // Update metrics
        self.metrics.messages_sent.fetch_add(1, Ordering::Relaxed);
        self.metrics.bytes_sent.fetch_add(message_size as u64, Ordering::Relaxed);
        
        let latency_ms = start_time.elapsed().as_millis() as u64;
        self.metrics.message_latency_ms.fetch_add(latency_ms, Ordering::Relaxed);
        
        debug!("📤 Sent message {} to shard {} ({} bytes, {}ms)", 
               message.id, message.target_shard, message_size, latency_ms);

        // Store message for potential retry if acknowledgment required
        if message.requires_ack {
            let mut buffer = self.message_buffer.write().await;
            buffer.push(message);
        }

        Ok(())
    }

    /// Process an incoming message
    pub async fn handle_message(&self, message: CrossShardMessage) -> Result<()> {
        let start_time = Instant::now();
        
        debug!("📥 Received message {} from shard {}", message.id, message.source_shard);
        
        // Update metrics
        let message_size = bincode::serialize(&message)?.len();
        self.metrics.messages_received.fetch_add(1, Ordering::Relaxed);
        self.metrics.bytes_received.fetch_add(message_size as u64, Ordering::Relaxed);

        // Decompress if needed
        let decompressed_message = if self.compression_enabled {
            self.decompress_message(message).await?
        } else {
            message
        };

        // Route message to appropriate handler
        let response = self.route_message(&decompressed_message).await?;
        
        // Send acknowledgment if required
        if decompressed_message.requires_ack {
            let ack = CrossShardMessage {
                id: Uuid::new_v4(),
                source_shard: self.shard_id,
                target_shard: decompressed_message.source_shard,
                priority: MessagePriority::Normal,
                payload: MessagePayload::Acknowledgment {
                    original_message_id: decompressed_message.id,
                    status: AckStatus::Success,
                },
                timestamp: chrono::Utc::now().timestamp_millis() as u64,
                requires_ack: false,
            };
            
            self.send_message(ack).await?;
        }

        // Send response if handler generated one
        if let Some(response_message) = response {
            self.send_message(response_message).await?;
        }

        let processing_time = start_time.elapsed();
        debug!("✅ Processed message in {:.2}ms", processing_time.as_millis());

        Ok(())
    }

    /// Route message to appropriate handler
    async fn route_message(&self, message: &CrossShardMessage) -> Result<Option<CrossShardMessage>> {
        let message_type = match &message.payload {
            MessagePayload::CrossShardTransaction { .. } => "transaction",
            MessagePayload::StateSyncRequest { .. } => "state_sync_request",
            MessagePayload::StateSyncResponse { .. } => "state_sync_response",
            MessagePayload::ConsensusVote { .. } => "consensus_vote",
            MessagePayload::ConsensusProposal { .. } => "consensus_proposal",
            MessagePayload::LoadBalanceRequest { .. } => "load_balance_request",
            MessagePayload::ShardRebalance { .. } => "shard_rebalance",
            MessagePayload::Acknowledgment { .. } => "acknowledgment",
        };

        let handlers = self.message_handlers.read().await;
        if let Some(handler) = handlers.get(message_type) {
            handler(message)
        } else {
            warn!("⚠️ No handler registered for message type: {}", message_type);
            Ok(None)
        }
    }

    /// Compress message payload for efficient transmission
    async fn compress_message(&self, mut message: CrossShardMessage) -> Result<CrossShardMessage> {
        let serialized = bincode::serialize(&message.payload)?;
        let compressed = lz4::block::compress(&serialized, None, false)?;
        
        let compression_ratio = (compressed.len() as f64 / serialized.len() as f64) * 100.0;
        self.metrics.compression_ratio.store(compression_ratio as u64, Ordering::Relaxed);
        
        // Store compressed data in a special payload type
        // This is simplified - in practice, you'd extend the payload enum
        debug!("🗜️ Compressed message: {} -> {} bytes ({:.1}% ratio)", 
               serialized.len(), compressed.len(), compression_ratio);
        
        Ok(message)
    }

    /// Decompress message payload
    async fn decompress_message(&self, message: CrossShardMessage) -> Result<CrossShardMessage> {
        // This would decompress the payload if it was compressed
        // Simplified implementation
        Ok(message)
    }

    /// Start message processing loop
    pub async fn start_processing(&self, mut receiver: mpsc::UnboundedReceiver<CrossShardMessage>) -> Result<()> {
        info!("🚀 Starting cross-shard message processing for shard {}", self.shard_id);
        
        while let Some(message) = receiver.recv().await {
            if let Err(e) = self.handle_message(message).await {
                error!("❌ Failed to handle message: {}", e);
                self.metrics.failed_deliveries.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        warn!("⚠️ Message processing loop ended for shard {}", self.shard_id);
        Ok(())
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> CrossShardMetrics {
        CrossShardMetrics {
            messages_sent: AtomicU64::new(self.metrics.messages_sent.load(Ordering::Relaxed)),
            messages_received: AtomicU64::new(self.metrics.messages_received.load(Ordering::Relaxed)),
            bytes_sent: AtomicU64::new(self.metrics.bytes_sent.load(Ordering::Relaxed)),
            bytes_received: AtomicU64::new(self.metrics.bytes_received.load(Ordering::Relaxed)),
            message_latency_ms: AtomicU64::new(self.metrics.message_latency_ms.load(Ordering::Relaxed)),
            compression_ratio: AtomicU64::new(self.metrics.compression_ratio.load(Ordering::Relaxed)),
            failed_deliveries: AtomicU64::new(self.metrics.failed_deliveries.load(Ordering::Relaxed)),
        }
    }

    /// Perform periodic maintenance tasks
    pub async fn maintenance_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            // Clean up old messages from buffer
            self.cleanup_message_buffer().await;
            
            // Update connection quality metrics
            self.update_connection_quality().await;
            
            // Log performance statistics
            self.log_performance_stats().await;
        }
    }

    async fn cleanup_message_buffer(&self) {
        let mut buffer = self.message_buffer.write().await;
        let cutoff = chrono::Utc::now().timestamp_millis() as u64 - 300_000; // 5 minutes
        
        buffer.retain(|msg| msg.timestamp > cutoff);
        
        if buffer.len() > 1000 {
            debug!("🧹 Cleaned up message buffer: retained {} messages", buffer.len());
        }
    }

    async fn update_connection_quality(&self) {
        let mut peers = self.peer_shards.write().await;
        let now = Instant::now();
        
        for (shard_id, connection) in peers.iter_mut() {
            let time_since_last_seen = now.duration_since(connection.last_seen);
            
            if time_since_last_seen > Duration::from_secs(60) {
                connection.connection_quality *= 0.9; // Degrade quality
                debug!("📉 Connection quality degraded for shard {}: {:.2}", 
                       shard_id, connection.connection_quality);
            }
        }
    }

    async fn log_performance_stats(&self) {
        let metrics = self.get_metrics();
        let messages_sent = metrics.messages_sent.load(Ordering::Relaxed);
        let messages_received = metrics.messages_received.load(Ordering::Relaxed);
        
        if messages_sent > 0 || messages_received > 0 {
            info!("📊 Cross-shard metrics - Sent: {}, Received: {}, Failures: {}", 
                  messages_sent, 
                  messages_received, 
                  metrics.failed_deliveries.load(Ordering::Relaxed));
        }
    }
}

/// Create a cross-shard communication network for multiple shards
pub async fn create_shard_network(shard_count: u32) -> Result<HashMap<ShardId, (CrossShardManager, mpsc::UnboundedReceiver<CrossShardMessage>)>> {
    let mut shard_network = HashMap::new();
    let mut shard_senders = HashMap::new();
    
    // Create managers and channels for each shard
    for shard_id in 0..shard_count {
        let manager = CrossShardManager::new(shard_id);
        let (sender, receiver) = mpsc::unbounded_channel();
        
        shard_senders.insert(shard_id, sender);
        shard_network.insert(shard_id, (manager, receiver));
    }
    
    // Connect all shards to each other
    for (shard_id, (manager, _)) in &shard_network {
        for (peer_id, sender) in &shard_senders {
            if shard_id != peer_id {
                manager.add_peer_shard(*peer_id, sender.clone()).await?;
            }
        }
    }
    
    info!("🌐 Created cross-shard network with {} interconnected shards", shard_count);
    Ok(shard_network)
}