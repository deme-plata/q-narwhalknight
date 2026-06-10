//! Narwhal Mempool Integration for VM
//! Provides reliable broadcast and ordering for VM transactions

use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use tokio::sync::{RwLock, broadcast};
use anyhow::Result;
use std::time::{Instant, Duration};

use crate::types::VMTransaction;
use super::{VMIntegration, IntegrationConfig, IntegrationResult, IntegrationStatus, 
           IntegrationMetrics, CryptographicPhase};

/// Mock Narwhal core for mempool operations
#[derive(Debug)]
pub struct MockNarwhalCore {
    node_id: String,
    mempool: Arc<RwLock<VecDeque<VMTransaction>>>,
    broadcast_channel: broadcast::Sender<VMTransaction>,
    validator_set: Arc<RwLock<HashMap<String, ValidatorInfo>>>,
    batch_timer: Arc<RwLock<Option<Instant>>>,
}

#[derive(Debug, Clone)]
pub struct ValidatorInfo {
    pub id: String,
    pub address: String,
    pub stake: u64,
    pub is_online: bool,
    pub last_seen: Instant,
}

impl MockNarwhalCore {
    pub fn new(node_id: String) -> Self {
        let (tx, _) = broadcast::channel(1000);
        
        Self {
            node_id,
            mempool: Arc::new(RwLock::new(VecDeque::new())),
            broadcast_channel: tx,
            validator_set: Arc::new(RwLock::new(HashMap::new())),
            batch_timer: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Submit transaction to mempool
    pub async fn submit_transaction(&self, tx: VMTransaction) -> Result<()> {
        // Validate transaction format
        if !self.validate_transaction_format(&tx).await? {
            return Err(anyhow::anyhow!("Invalid transaction format"));
        }
        
        // Add to mempool
        self.mempool.write().await.push_back(tx.clone());
        
        // Broadcast to validators
        if let Err(_) = self.broadcast_channel.send(tx) {
            tracing::warn!("No receivers for transaction broadcast");
        }
        
        // Start batch timer if needed
        self.start_batch_timer_if_needed().await;
        
        Ok(())
    }
    
    /// Get batch of transactions for consensus
    pub async fn get_transaction_batch(&self, max_size: usize) -> Result<Vec<VMTransaction>> {
        let mut mempool = self.mempool.write().await;
        let mut batch = Vec::new();
        
        for _ in 0..max_size {
            if let Some(tx) = mempool.pop_front() {
                batch.push(tx);
            } else {
                break;
            }
        }
        
        // Reset batch timer
        *self.batch_timer.write().await = None;
        
        Ok(batch)
    }
    
    /// Validate transaction format for mempool acceptance
    async fn validate_transaction_format(&self, tx: &VMTransaction) -> Result<bool> {
        // Basic format validation
        if tx.gas_limit == 0 || tx.gas_price == 0 {
            return Ok(false);
        }
        
        if tx.signature.is_empty() {
            // In production, would verify cryptographic signature
            tracing::debug!("Transaction missing signature (allowed in test mode)");
        }
        
        Ok(true)
    }
    
    /// Start batch timer for batching transactions
    async fn start_batch_timer_if_needed(&self) {
        let mut timer = self.batch_timer.write().await;
        if timer.is_none() {
            *timer = Some(Instant::now());
        }
    }
    
    /// Check if batch should be created based on time or size
    pub async fn should_create_batch(&self, batch_timeout: Duration, max_batch_size: usize) -> bool {
        let mempool = self.mempool.read().await;
        let timer = self.batch_timer.read().await;
        
        // Size-based batching
        if mempool.len() >= max_batch_size {
            return true;
        }
        
        // Time-based batching
        if let Some(start_time) = *timer {
            if start_time.elapsed() >= batch_timeout && !mempool.is_empty() {
                return true;
            }
        }
        
        false
    }
    
    /// Get mempool status
    pub async fn get_mempool_status(&self) -> MempoolStatus {
        let mempool = self.mempool.read().await;
        let validators = self.validator_set.read().await;
        
        MempoolStatus {
            pending_transactions: mempool.len() as u64,
            total_validators: validators.len() as u32,
            online_validators: validators.values()
                .filter(|v| v.is_online)
                .count() as u32,
            last_batch_time: *self.batch_timer.read().await,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MempoolStatus {
    pub pending_transactions: u64,
    pub total_validators: u32,
    pub online_validators: u32,
    pub last_batch_time: Option<Instant>,
}

/// Narwhal broadcast integration with VM
pub struct NarwhalBroadcastIntegration {
    narwhal: Arc<MockNarwhalCore>,
    metrics: Arc<RwLock<IntegrationMetrics>>,
    config: Arc<RwLock<Option<IntegrationConfig>>>,
    transaction_receiver: Arc<RwLock<Option<broadcast::Receiver<VMTransaction>>>>,
}

impl NarwhalBroadcastIntegration {
    pub fn new(node_id: String) -> Self {
        let narwhal = Arc::new(MockNarwhalCore::new(node_id));
        
        Self {
            narwhal,
            metrics: Arc::new(RwLock::new(IntegrationMetrics::default())),
            config: Arc::new(RwLock::new(None)),
            transaction_receiver: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Submit transaction to reliable broadcast
    pub async fn broadcast_transaction(&self, tx: VMTransaction) -> Result<()> {
        let start_time = Instant::now();
        
        // Submit to mempool
        self.narwhal.submit_transaction(tx.clone()).await?;
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.mempool_broadcasts += 1;
            metrics.total_transactions += 1;
        }
        
        tracing::info!(
            "Broadcasted transaction {} in {:?}",
            tx.id,
            start_time.elapsed()
        );
        
        Ok(())
    }
    
    /// Get batch of transactions ready for consensus
    pub async fn get_consensus_batch(&self) -> Result<Vec<VMTransaction>> {
        let config = self.config.read().await;
        let batch_size = config.as_ref()
            .map(|c| c.mempool_batch_size)
            .unwrap_or(100);
        
        let batch = self.narwhal.get_transaction_batch(batch_size).await?;
        
        tracing::debug!("Created consensus batch with {} transactions", batch.len());
        
        Ok(batch)
    }
    
    /// Start transaction listening service
    pub async fn start_transaction_listener(&self) -> Result<()> {
        let mut receiver = self.narwhal.broadcast_channel.subscribe();
        *self.transaction_receiver.write().await = Some(receiver);
        
        tracing::info!("Started Narwhal transaction listener");
        Ok(())
    }
    
    /// Process incoming transaction from broadcast
    pub async fn process_incoming_transaction(&self) -> Result<Option<VMTransaction>> {
        let mut receiver_guard = self.transaction_receiver.write().await;
        if let Some(receiver) = receiver_guard.as_mut() {
            match receiver.try_recv() {
                Ok(tx) => {
                    tracing::debug!("Received broadcasted transaction: {}", tx.id);
                    return Ok(Some(tx));
                }
                Err(broadcast::error::TryRecvError::Empty) => return Ok(None),
                Err(broadcast::error::TryRecvError::Lagged(_)) => {
                    tracing::warn!("Transaction receiver lagged behind");
                    return Ok(None);
                }
                Err(broadcast::error::TryRecvError::Closed) => {
                    tracing::error!("Transaction broadcast channel closed");
                    return Err(anyhow::anyhow!("Broadcast channel closed"));
                }
            }
        }
        Ok(None)
    }
    
    /// Check if ready to create consensus batch
    pub async fn is_batch_ready(&self) -> Result<bool> {
        let config = self.config.read().await;
        let timeout = Duration::from_millis(
            config.as_ref()
                .map(|c| c.consensus_timeout_ms)
                .unwrap_or(1000)
        );
        let max_size = config.as_ref()
            .map(|c| c.mempool_batch_size)
            .unwrap_or(100);
        
        Ok(self.narwhal.should_create_batch(timeout, max_size).await)
    }
    
    /// Get detailed mempool metrics
    pub async fn get_mempool_metrics(&self) -> Result<MempoolMetrics> {
        let status = self.narwhal.get_mempool_status().await;
        let integration_metrics = self.metrics.read().await;
        
        Ok(MempoolMetrics {
            pending_transactions: status.pending_transactions,
            total_broadcasts: integration_metrics.mempool_broadcasts,
            validator_count: status.total_validators,
            online_validators: status.online_validators,
            batch_creation_rate: if integration_metrics.mempool_broadcasts > 0 {
                integration_metrics.mempool_broadcasts as f64 / 60.0 // per minute
            } else {
                0.0
            },
            average_batch_size: if integration_metrics.mempool_broadcasts > 0 {
                integration_metrics.total_transactions as f64 / integration_metrics.mempool_broadcasts as f64
            } else {
                0.0
            },
        })
    }
}

#[derive(Debug, Clone)]
pub struct MempoolMetrics {
    pub pending_transactions: u64,
    pub total_broadcasts: u64,
    pub validator_count: u32,
    pub online_validators: u32,
    pub batch_creation_rate: f64,
    pub average_batch_size: f64,
}

#[async_trait::async_trait]
impl VMIntegration for NarwhalBroadcastIntegration {
    async fn initialize(&self, config: &IntegrationConfig) -> Result<()> {
        *self.config.write().await = Some(config.clone());
        
        // Start transaction listener
        self.start_transaction_listener().await?;
        
        tracing::info!(
            "Initialized Narwhal broadcast integration for node {} with batch size {}",
            config.node_id, config.mempool_batch_size
        );
        
        Ok(())
    }
    
    async fn process_transaction(&self, tx: &VMTransaction) -> Result<IntegrationResult> {
        let start_time = Instant::now();
        
        // Broadcast transaction
        self.broadcast_transaction(tx.clone()).await?;
        
        // Create mock integration result
        let metrics = self.metrics.read().await.clone();
        let config = self.config.read().await;
        
        Ok(IntegrationResult {
            success: true,
            transaction_hash: tx.id.clone(),
            execution_result: crate::vm::ExecutionResult {
                success: true,
                return_data: vec![],
                gas_used: 21000,
                logs: vec!["Transaction broadcasted successfully".to_string()],
                error: None,
            },
            consensus_round: 0, // Will be filled by consensus
            vdf_output: None,
            crypto_phase: config.as_ref()
                .map(|c| c.phase)
                .unwrap_or(CryptographicPhase::Phase1),
            processing_time_ms: start_time.elapsed().as_millis() as u64,
            integration_metrics: metrics,
        })
    }
    
    async fn get_status(&self) -> Result<IntegrationStatus> {
        let mempool_status = self.narwhal.get_mempool_status().await;
        let config = self.config.read().await;
        
        Ok(IntegrationStatus {
            is_healthy: true,
            consensus_status: "Not integrated".to_string(),
            mempool_status: format!(
                "{} pending transactions, {}/{} validators online",
                mempool_status.pending_transactions,
                mempool_status.online_validators,
                mempool_status.total_validators
            ),
            vdf_status: "Not integrated".to_string(),
            crypto_status: format!("Phase {:?}", 
                                 config.as_ref().map(|c| c.phase)
                                       .unwrap_or(CryptographicPhase::Phase1)),
            current_phase: config.as_ref().map(|c| c.phase)
                               .unwrap_or(CryptographicPhase::Phase1),
            active_connections: mempool_status.online_validators,
            pending_transactions: mempool_status.pending_transactions,
        })
    }
    
    async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutting down Narwhal broadcast integration");
        Ok(())
    }
}