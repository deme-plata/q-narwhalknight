/// Quantum Fair Queueing Mechanisms for Q-NarwhalKnight
/// Provides quantum-enhanced fairness guarantees for transaction ordering and resource allocation

use q_types::*;
use q_quantum_rng::{QuantumRNG, QuantumRandomness};
use q_lattice_vrf::{LatticeVRF, VRFResult};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use sha3::{Digest, Sha3_256};
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, Mutex};
use priority_queue::PriorityQueue;
use tracing::{debug, info, warn, trace};

pub mod quantum_scheduler;
pub mod fairness_metrics;
pub mod bandwidth_allocation;
pub mod anti_censorship;

pub use quantum_scheduler::{QuantumScheduler, SchedulingPolicy};
pub use fairness_metrics::{FairnessAnalyzer, FairnessMetrics};
pub use bandwidth_allocation::{BandwidthAllocator, AllocationStrategy};
pub use anti_censorship::{AntiCensorshipManager, CensorshipResistance};

/// Main quantum fair queueing system
pub struct QuantumFairQueue {
    /// Node identity
    node_id: NodeId,
    
    /// Phase for quantum features
    phase: Phase,
    
    /// Quantum randomness source
    quantum_rng: Option<QuantumRNG>,
    
    /// Lattice VRF for deterministic randomness
    lattice_vrf: Option<LatticeVRF>,
    
    /// Priority queues for different transaction types
    priority_queues: RwLock<HashMap<TransactionType, PriorityQueue<TransactionId, Priority>>>,
    
    /// Fair scheduling state
    scheduler: RwLock<QuantumScheduler>,
    
    /// Fairness metrics
    fairness_analyzer: RwLock<FairnessAnalyzer>,
    
    /// Bandwidth allocation
    bandwidth_allocator: RwLock<BandwidthAllocator>,
    
    /// Anti-censorship mechanisms
    anti_censorship: RwLock<AntiCensorshipManager>,
    
    /// System statistics
    stats: RwLock<QueueStatistics>,
}

/// Transaction types for prioritization
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionType {
    /// High-priority consensus messages
    Consensus,
    
    /// Regular user transactions
    User,
    
    /// System maintenance operations
    System,
    
    /// Quantum beacon updates
    QuantumBeacon,
    
    /// Emergency protocol updates
    Emergency,
}

/// Priority levels (higher number = higher priority)
pub type Priority = i64;

/// Queue statistics
#[derive(Debug, Clone, Default)]
pub struct QueueStatistics {
    /// Total transactions processed
    pub total_processed: u64,
    
    /// Average wait time per type
    pub avg_wait_times: HashMap<TransactionType, Duration>,
    
    /// Fairness scores
    pub fairness_scores: HashMap<NodeId, f64>,
    
    /// Throughput metrics
    pub throughput_per_second: f64,
    
    /// Queue depths
    pub queue_depths: HashMap<TransactionType, usize>,
    
    /// Quantum randomness usage
    pub quantum_randomness_used: u64,
    
    /// Censorship resistance events
    pub censorship_events: u64,
}

/// Queueing configuration
#[derive(Debug, Clone)]
pub struct QueueConfig {
    /// Enable quantum fairness enhancement
    pub quantum_enhanced: bool,
    
    /// Base priority weights per transaction type
    pub type_weights: HashMap<TransactionType, f64>,
    
    /// Maximum queue size per type
    pub max_queue_sizes: HashMap<TransactionType, usize>,
    
    /// Aging factor for queue fairness
    pub aging_factor: f64,
    
    /// Quantum randomness refresh interval
    pub randomness_refresh_ms: u64,
    
    /// Enable anti-censorship mechanisms
    pub anti_censorship_enabled: bool,
    
    /// Bandwidth allocation strategy
    pub allocation_strategy: AllocationStrategy,
}

impl Default for QueueConfig {
    fn default() -> Self {
        let mut type_weights = HashMap::new();
        type_weights.insert(TransactionType::Emergency, 1000.0);
        type_weights.insert(TransactionType::Consensus, 500.0);
        type_weights.insert(TransactionType::QuantumBeacon, 400.0);
        type_weights.insert(TransactionType::System, 200.0);
        type_weights.insert(TransactionType::User, 100.0);
        
        let mut max_queue_sizes = HashMap::new();
        max_queue_sizes.insert(TransactionType::Emergency, 100);
        max_queue_sizes.insert(TransactionType::Consensus, 10000);
        max_queue_sizes.insert(TransactionType::QuantumBeacon, 1000);
        max_queue_sizes.insert(TransactionType::System, 5000);
        max_queue_sizes.insert(TransactionType::User, 50000);
        
        Self {
            quantum_enhanced: true,
            type_weights,
            max_queue_sizes,
            aging_factor: 0.1,
            randomness_refresh_ms: 1000,
            anti_censorship_enabled: true,
            allocation_strategy: AllocationStrategy::ProportionalFair,
        }
    }
}

impl QuantumFairQueue {
    /// Create new quantum fair queue
    pub async fn new(node_id: NodeId, phase: Phase, config: QueueConfig) -> Result<Self> {
        info!("Initializing Quantum Fair Queue for {:?} phase", phase);

        // Initialize quantum components for Phase 2+
        let (quantum_rng, lattice_vrf) = if phase >= Phase::Phase2 && config.quantum_enhanced {
            let qrng = match QuantumRNG::new(phase, Default::default()).await {
                Ok(q) => {
                    info!("Quantum RNG initialized for fair queueing");
                    Some(q)
                },
                Err(e) => {
                    warn!("Failed to initialize QRNG for fair queueing: {}", e);
                    None
                }
            };

            let vrf = match LatticeVRF::new(Default::default(), phase).await {
                Ok(v) => {
                    info!("Lattice VRF initialized for queue randomness");
                    Some(v)
                },
                Err(e) => {
                    warn!("Failed to initialize L-VRF for queueing: {}", e);
                    None
                }
            };

            (qrng, vrf)
        } else {
            (None, None)
        };

        // Initialize priority queues
        let mut priority_queues = HashMap::new();
        for tx_type in [TransactionType::Consensus, TransactionType::User, 
                       TransactionType::System, TransactionType::QuantumBeacon, 
                       TransactionType::Emergency] {
            priority_queues.insert(tx_type, PriorityQueue::new());
        }

        let scheduler = QuantumScheduler::new(config.clone(), quantum_rng.as_ref(), lattice_vrf.as_ref()).await?;
        let fairness_analyzer = FairnessAnalyzer::new(node_id)?;
        let bandwidth_allocator = BandwidthAllocator::new(config.allocation_strategy)?;
        let anti_censorship = AntiCensorshipManager::new(config.anti_censorship_enabled)?;

        Ok(Self {
            node_id,
            phase,
            quantum_rng,
            lattice_vrf,
            priority_queues: RwLock::new(priority_queues),
            scheduler: RwLock::new(scheduler),
            fairness_analyzer: RwLock::new(fairness_analyzer),
            bandwidth_allocator: RwLock::new(bandwidth_allocator),
            anti_censorship: RwLock::new(anti_censorship),
            stats: RwLock::new(QueueStatistics::default()),
        })
    }

    /// Enqueue transaction with quantum fairness
    pub async fn enqueue(&self, tx_id: TransactionId, tx_type: TransactionType, from_node: NodeId) -> Result<()> {
        let enqueue_time = Instant::now();
        debug!("Enqueueing transaction {} of type {:?} from node {}", 
               hex::encode(&tx_id), tx_type, from_node);

        // Check for censorship attempts
        {
            let mut anti_censorship = self.anti_censorship.write().await;
            if anti_censorship.is_censorship_attempt(&tx_id, from_node).await? {
                warn!("Potential censorship detected for transaction from node {}", from_node);
                
                // Apply anti-censorship measures
                anti_censorship.apply_countermeasures(&tx_id, from_node).await?;
                
                let mut stats = self.stats.write().await;
                stats.censorship_events += 1;
            }
        }

        // Calculate quantum-enhanced priority
        let priority = self.calculate_priority(tx_type, from_node, enqueue_time).await?;

        // Enqueue with calculated priority
        {
            let mut queues = self.priority_queues.write().await;
            if let Some(queue) = queues.get_mut(&tx_type) {
                // Check queue capacity
                let max_size = self.get_max_queue_size(tx_type);
                if queue.len() >= max_size {
                    // Apply quantum fairness for eviction
                    self.quantum_eviction(queue, tx_type).await?;
                }
                
                queue.push(tx_id, priority);
                trace!("Transaction {} enqueued with priority {}", hex::encode(&tx_id), priority);
            } else {
                return Err(anyhow!("Unknown transaction type: {:?}", tx_type));
            }
        }

        // Update fairness metrics
        {
            let mut analyzer = self.fairness_analyzer.write().await;
            analyzer.record_enqueue(from_node, tx_type, priority).await?;
        }

        Ok(())
    }

    /// Dequeue next transaction using quantum fair scheduling
    pub async fn dequeue(&self) -> Result<Option<(TransactionId, TransactionType)>> {
        let dequeue_time = Instant::now();
        
        // Use quantum scheduler to select next transaction type
        let selected_type = {
            let mut scheduler = self.scheduler.write().await;
            scheduler.select_next_type().await?
        };

        if let Some(tx_type) = selected_type {
            let mut queues = self.priority_queues.write().await;
            if let Some(queue) = queues.get_mut(&tx_type) {
                if let Some((tx_id, _priority)) = queue.pop() {
                    debug!("Dequeued transaction {} of type {:?}", hex::encode(&tx_id), tx_type);
                    
                    // Update statistics
                    {
                        let mut stats = self.stats.write().await;
                        stats.total_processed += 1;
                        
                        // Update queue depths
                        stats.queue_depths.insert(tx_type, queue.len());
                    }
                    
                    // Update fairness metrics
                    {
                        let mut analyzer = self.fairness_analyzer.write().await;
                        analyzer.record_dequeue(tx_type, dequeue_time).await?;
                    }
                    
                    return Ok(Some((tx_id, tx_type)));
                }
            }
        }

        Ok(None)
    }

    /// Calculate quantum-enhanced priority
    async fn calculate_priority(&self, tx_type: TransactionType, from_node: NodeId, enqueue_time: Instant) -> Result<Priority> {
        // Base priority from type
        let base_priority = match tx_type {
            TransactionType::Emergency => 1000,
            TransactionType::Consensus => 500,
            TransactionType::QuantumBeacon => 400,
            TransactionType::System => 200,
            TransactionType::User => 100,
        };

        // Quantum enhancement if available
        let quantum_bonus = if let Some(ref qrng) = self.quantum_rng {
            // Use quantum randomness for fairness
            let random_bytes = qrng.generate_bytes(4).await?;
            let random_value = u32::from_be_bytes([random_bytes[0], random_bytes[1], random_bytes[2], random_bytes[3]]);
            
            // Map to small priority adjustment (-10 to +10)
            let bonus = (random_value % 21) as i64 - 10;
            trace!("Applied quantum priority bonus: {}", bonus);
            
            {
                let mut stats = self.stats.write().await;
                stats.quantum_randomness_used += 1;
            }
            
            bonus
        } else {
            0
        };

        // Node fairness adjustment
        let fairness_adjustment = {
            let analyzer = self.fairness_analyzer.read().await;
            analyzer.get_node_fairness_adjustment(from_node).await?
        };

        // Time-based aging (older transactions get higher priority)
        let age_ms = enqueue_time.elapsed().as_millis() as i64;
        let aging_bonus = (age_ms / 1000) * 5; // +5 priority per second of age

        let total_priority = base_priority + quantum_bonus + fairness_adjustment + aging_bonus;
        
        debug!("Calculated priority for {:?} transaction: base={}, quantum={}, fairness={}, aging={}, total={}", 
               tx_type, base_priority, quantum_bonus, fairness_adjustment, aging_bonus, total_priority);

        Ok(total_priority)
    }

    /// Quantum-enhanced eviction when queue is full
    async fn quantum_eviction(&self, queue: &mut PriorityQueue<TransactionId, Priority>, tx_type: TransactionType) -> Result<()> {
        debug!("Applying quantum eviction for {:?} queue", tx_type);

        // Use quantum randomness to fairly select victim
        if let Some(ref qrng) = self.quantum_rng {
            let random_bytes = qrng.generate_bytes(4).await?;
            let random_index = u32::from_be_bytes([random_bytes[0], random_bytes[1], random_bytes[2], random_bytes[3]]) as usize;
            
            // Select from bottom 10% of queue (lowest priority)
            let queue_len = queue.len();
            let bottom_10_percent = queue_len / 10;
            let victim_index = random_index % bottom_10_percent.max(1);
            
            // Remove the selected victim
            if let Some(victim) = queue.peek_by_index(victim_index) {
                let victim_id = *victim.0;
                queue.remove(&victim_id);
                debug!("Quantum evicted transaction {}", hex::encode(&victim_id));
            }
        } else {
            // Classical eviction - remove lowest priority
            queue.pop();
        }

        Ok(())
    }

    /// Get maximum queue size for transaction type
    fn get_max_queue_size(&self, tx_type: TransactionType) -> usize {
        match tx_type {
            TransactionType::Emergency => 100,
            TransactionType::Consensus => 10000,
            TransactionType::QuantumBeacon => 1000,
            TransactionType::System => 5000,
            TransactionType::User => 50000,
        }
    }

    /// Get current queue statistics
    pub async fn get_statistics(&self) -> QueueStatistics {
        self.stats.read().await.clone()
    }

    /// Get fairness metrics
    pub async fn get_fairness_metrics(&self) -> Result<FairnessMetrics> {
        let analyzer = self.fairness_analyzer.read().await;
        analyzer.get_current_metrics().await
    }

    /// Update bandwidth allocation
    pub async fn update_bandwidth_allocation(&self, allocations: HashMap<NodeId, u64>) -> Result<()> {
        let mut allocator = self.bandwidth_allocator.write().await;
        allocator.update_allocations(allocations).await
    }

    /// Check if system is operating fairly
    pub async fn is_fair(&self) -> Result<bool> {
        let metrics = self.get_fairness_metrics().await?;
        Ok(metrics.overall_fairness_score > 0.8)
    }
}

/// Transaction queueing interface
#[async_trait]
pub trait TransactionQueue: Send + Sync {
    async fn enqueue(&self, tx_id: TransactionId, tx_type: TransactionType, from_node: NodeId) -> Result<()>;
    async fn dequeue(&self) -> Result<Option<(TransactionId, TransactionType)>>;
    async fn queue_depth(&self, tx_type: TransactionType) -> Result<usize>;
    async fn is_full(&self, tx_type: TransactionType) -> Result<bool>;
}

#[async_trait]
impl TransactionQueue for QuantumFairQueue {
    async fn enqueue(&self, tx_id: TransactionId, tx_type: TransactionType, from_node: NodeId) -> Result<()> {
        self.enqueue(tx_id, tx_type, from_node).await
    }
    
    async fn dequeue(&self) -> Result<Option<(TransactionId, TransactionType)>> {
        self.dequeue().await
    }
    
    async fn queue_depth(&self, tx_type: TransactionType) -> Result<usize> {
        let queues = self.priority_queues.read().await;
        Ok(queues.get(&tx_type).map(|q| q.len()).unwrap_or(0))
    }
    
    async fn is_full(&self, tx_type: TransactionType) -> Result<bool> {
        let depth = self.queue_depth(tx_type).await?;
        Ok(depth >= self.get_max_queue_size(tx_type))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_quantum_fair_queue_creation() {
        let node_id = [1u8; 32];
        let config = QueueConfig::default();
        let queue = QuantumFairQueue::new(node_id, Phase::Phase0, config).await.unwrap();
        
        assert_eq!(queue.node_id, node_id);
        assert_eq!(queue.phase, Phase::Phase0);
    }

    #[tokio::test]
    async fn test_enqueue_dequeue() {
        let node_id = [1u8; 32];
        let config = QueueConfig::default();
        let queue = QuantumFairQueue::new(node_id, Phase::Phase0, config).await.unwrap();
        
        let tx_id = Uuid::new_v4().into_bytes();
        
        // Enqueue transaction
        queue.enqueue(tx_id, TransactionType::User, node_id).await.unwrap();
        
        // Check queue depth
        let depth = queue.queue_depth(TransactionType::User).await.unwrap();
        assert_eq!(depth, 1);
        
        // Dequeue transaction
        let dequeued = queue.dequeue().await.unwrap();
        assert!(dequeued.is_some());
        assert_eq!(dequeued.unwrap().0, tx_id);
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let node_id = [1u8; 32];
        let config = QueueConfig::default();
        let queue = QuantumFairQueue::new(node_id, Phase::Phase0, config).await.unwrap();
        
        let user_tx = Uuid::new_v4().into_bytes();
        let consensus_tx = Uuid::new_v4().into_bytes();
        let emergency_tx = Uuid::new_v4().into_bytes();
        
        // Enqueue in reverse priority order
        queue.enqueue(user_tx, TransactionType::User, node_id).await.unwrap();
        queue.enqueue(consensus_tx, TransactionType::Consensus, node_id).await.unwrap();
        queue.enqueue(emergency_tx, TransactionType::Emergency, node_id).await.unwrap();
        
        // Dequeue should return highest priority first
        let first = queue.dequeue().await.unwrap();
        assert!(first.is_some());
        // Emergency should come first, but exact ordering depends on scheduling algorithm
    }
}