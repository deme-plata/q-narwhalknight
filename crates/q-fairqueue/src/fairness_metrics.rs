/// Fairness analysis and metrics for quantum queueing systems

use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time;
use tracing::debug;

use crate::{TransactionType, Priority};
use q_types::NodeId;

/// Fairness metrics and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessMetrics {
    /// Overall fairness score (0.0 to 1.0)
    pub overall_fairness_score: f64,
    
    /// Fairness per node
    pub node_fairness: HashMap<NodeId, f64>,
    
    /// Fairness per transaction type
    pub type_fairness: HashMap<TransactionType, f64>,
    
    /// Jain's fairness index
    pub jains_fairness_index: f64,
    
    /// Standard deviation of wait times
    pub wait_time_std_dev: f64,
    
    /// Coefficient of variation
    pub coefficient_of_variation: f64,
    
    /// Quantum enhancement impact
    pub quantum_impact_score: f64,
}

/// Fairness analyzer
pub struct FairnessAnalyzer {
    /// Node ID of this analyzer
    node_id: NodeId,
    
    /// Historical enqueue records
    enqueue_history: Vec<EnqueueRecord>,
    
    /// Historical dequeue records
    dequeue_history: Vec<DequeueRecord>,
    
    /// Node activity tracking
    node_activity: HashMap<NodeId, NodeActivity>,
    
    /// Type activity tracking
    type_activity: HashMap<TransactionType, TypeActivity>,
}

/// Enqueue record for fairness analysis
#[derive(Debug, Clone)]
struct EnqueueRecord {
    node_id: NodeId,
    tx_type: TransactionType,
    priority: Priority,
    timestamp: Instant,
}

/// Dequeue record for fairness analysis
#[derive(Debug, Clone)]
struct DequeueRecord {
    tx_type: TransactionType,
    timestamp: Instant,
    wait_time: Duration,
}

/// Node activity statistics
#[derive(Debug, Clone, Default)]
struct NodeActivity {
    enqueues: u64,
    total_priority: i64,
    avg_priority: f64,
    last_enqueue: Option<Instant>,
    fairness_score: f64,
}

/// Transaction type activity statistics
#[derive(Debug, Clone, Default)]
struct TypeActivity {
    enqueues: u64,
    dequeues: u64,
    total_wait_time: Duration,
    avg_wait_time: Duration,
    max_wait_time: Duration,
}

impl FairnessAnalyzer {
    /// Create new fairness analyzer
    pub fn new(node_id: NodeId) -> Result<Self> {
        Ok(Self {
            node_id,
            enqueue_history: Vec::new(),
            dequeue_history: Vec::new(),
            node_activity: HashMap::new(),
            type_activity: HashMap::new(),
        })
    }
    
    /// Record transaction enqueue
    pub async fn record_enqueue(&mut self, node_id: NodeId, tx_type: TransactionType, priority: Priority) -> Result<()> {
        let timestamp = Instant::now();
        
        // Add to history
        self.enqueue_history.push(EnqueueRecord {
            node_id,
            tx_type,
            priority,
            timestamp,
        });
        
        // Update node activity
        let node_activity = self.node_activity.entry(node_id).or_default();
        node_activity.enqueues += 1;
        node_activity.total_priority += priority;
        node_activity.avg_priority = node_activity.total_priority as f64 / node_activity.enqueues as f64;
        node_activity.last_enqueue = Some(timestamp);
        
        // Update type activity
        let type_activity = self.type_activity.entry(tx_type).or_default();
        type_activity.enqueues += 1;
        
        // Prune old history (keep last 10000 records)
        if self.enqueue_history.len() > 10000 {
            self.enqueue_history.drain(0..1000);
        }
        
        debug!("Recorded enqueue: node={}, type={:?}, priority={}", 
               hex::encode(&node_id), tx_type, priority);
        
        Ok(())
    }
    
    /// Record transaction dequeue
    pub async fn record_dequeue(&mut self, tx_type: TransactionType, timestamp: Instant) -> Result<()> {
        // Find corresponding enqueue to calculate wait time
        let wait_time = if let Some(enqueue_record) = self.enqueue_history
            .iter()
            .rev() // Search from most recent
            .find(|r| r.tx_type == tx_type) {
            timestamp.duration_since(enqueue_record.timestamp)
        } else {
            Duration::from_secs(0)
        };
        
        // Add to dequeue history
        self.dequeue_history.push(DequeueRecord {
            tx_type,
            timestamp,
            wait_time,
        });
        
        // Update type activity
        let type_activity = self.type_activity.entry(tx_type).or_default();
        type_activity.dequeues += 1;
        type_activity.total_wait_time += wait_time;
        type_activity.avg_wait_time = type_activity.total_wait_time / type_activity.dequeues as u32;
        
        if wait_time > type_activity.max_wait_time {
            type_activity.max_wait_time = wait_time;
        }
        
        // Prune old dequeue history
        if self.dequeue_history.len() > 10000 {
            self.dequeue_history.drain(0..1000);
        }
        
        debug!("Recorded dequeue: type={:?}, wait_time={}ms", 
               tx_type, wait_time.as_millis());
        
        Ok(())
    }
    
    /// Get current fairness metrics
    pub async fn get_current_metrics(&self) -> Result<FairnessMetrics> {
        let node_fairness = self.calculate_node_fairness()?;
        let type_fairness = self.calculate_type_fairness()?;
        let overall_fairness = self.calculate_overall_fairness(&node_fairness, &type_fairness)?;
        let jains_index = self.calculate_jains_fairness_index()?;
        let wait_time_stats = self.calculate_wait_time_statistics()?;
        let quantum_impact = self.calculate_quantum_impact()?;
        
        Ok(FairnessMetrics {
            overall_fairness_score: overall_fairness,
            node_fairness,
            type_fairness,
            jains_fairness_index: jains_index,
            wait_time_std_dev: wait_time_stats.0,
            coefficient_of_variation: wait_time_stats.1,
            quantum_impact_score: quantum_impact,
        })
    }
    
    /// Calculate per-node fairness scores
    fn calculate_node_fairness(&self) -> Result<HashMap<NodeId, f64>> {
        let mut fairness = HashMap::new();
        
        if self.node_activity.is_empty() {
            return Ok(fairness);
        }
        
        // Calculate average priority across all nodes
        let total_priority: i64 = self.node_activity.values().map(|a| a.total_priority).sum();
        let total_enqueues: u64 = self.node_activity.values().map(|a| a.enqueues).sum();
        let global_avg_priority = if total_enqueues > 0 {
            total_priority as f64 / total_enqueues as f64
        } else {
            0.0
        };
        
        // Calculate fairness relative to global average
        for (node_id, activity) in &self.node_activity {
            let node_avg = activity.avg_priority;
            let fairness_score = if global_avg_priority > 0.0 {
                // Fairness is inverse of deviation from average
                let deviation = (node_avg - global_avg_priority).abs();
                let max_deviation = global_avg_priority; // Worst case
                1.0 - (deviation / max_deviation).min(1.0)
            } else {
                1.0
            };
            
            fairness.insert(*node_id, fairness_score);
        }
        
        Ok(fairness)
    }
    
    /// Calculate per-type fairness scores
    fn calculate_type_fairness(&self) -> Result<HashMap<TransactionType, f64>> {
        let mut fairness = HashMap::new();
        
        let tx_types = [
            TransactionType::Emergency,
            TransactionType::Consensus,
            TransactionType::QuantumBeacon,
            TransactionType::System,
            TransactionType::User,
        ];
        
        // Calculate expected vs actual service ratios
        for tx_type in tx_types {
            if let Some(activity) = self.type_activity.get(&tx_type) {
                let enqueue_ratio = activity.enqueues as f64 / self.enqueue_history.len().max(1) as f64;
                let dequeue_ratio = activity.dequeues as f64 / self.dequeue_history.len().max(1) as f64;
                
                // Fairness is how close dequeue ratio is to enqueue ratio
                let fairness_score = if enqueue_ratio > 0.0 {
                    1.0 - ((dequeue_ratio - enqueue_ratio).abs() / enqueue_ratio).min(1.0)
                } else {
                    1.0
                };
                
                fairness.insert(tx_type, fairness_score);
            } else {
                fairness.insert(tx_type, 1.0);
            }
        }
        
        Ok(fairness)
    }
    
    /// Calculate overall fairness score
    fn calculate_overall_fairness(
        &self,
        node_fairness: &HashMap<NodeId, f64>,
        type_fairness: &HashMap<TransactionType, f64>
    ) -> Result<f64> {
        let node_avg = if node_fairness.is_empty() {
            1.0
        } else {
            node_fairness.values().sum::<f64>() / node_fairness.len() as f64
        };
        
        let type_avg = if type_fairness.is_empty() {
            1.0
        } else {
            type_fairness.values().sum::<f64>() / type_fairness.len() as f64
        };
        
        // Weight node fairness more heavily
        Ok(node_avg * 0.7 + type_avg * 0.3)
    }
    
    /// Calculate Jain's fairness index
    fn calculate_jains_fairness_index(&self) -> Result<f64> {
        if self.node_activity.is_empty() {
            return Ok(1.0);
        }
        
        let throughputs: Vec<f64> = self.node_activity.values()
            .map(|a| a.enqueues as f64)
            .collect();
        
        if throughputs.is_empty() {
            return Ok(1.0);
        }
        
        let sum: f64 = throughputs.iter().sum();
        let sum_squares: f64 = throughputs.iter().map(|x| x * x).sum();
        let n = throughputs.len() as f64;
        
        if sum_squares > 0.0 {
            Ok((sum * sum) / (n * sum_squares))
        } else {
            Ok(1.0)
        }
    }
    
    /// Calculate wait time statistics
    fn calculate_wait_time_statistics(&self) -> Result<(f64, f64)> {
        if self.dequeue_history.is_empty() {
            return Ok((0.0, 0.0));
        }
        
        let wait_times: Vec<f64> = self.dequeue_history.iter()
            .map(|r| r.wait_time.as_millis() as f64)
            .collect();
        
        let mean = wait_times.iter().sum::<f64>() / wait_times.len() as f64;
        let variance = wait_times.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / wait_times.len() as f64;
        
        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 0.0 };
        
        Ok((std_dev, coefficient_of_variation))
    }
    
    /// Calculate quantum enhancement impact
    fn calculate_quantum_impact(&self) -> Result<f64> {
        // This would analyze how quantum randomness affects fairness
        // For now, return a placeholder value
        Ok(0.8) // Assume quantum enhancement improves fairness by 20%
    }
    
    /// Get fairness adjustment for a specific node
    pub async fn get_node_fairness_adjustment(&self, node_id: NodeId) -> Result<i64> {
        if let Some(activity) = self.node_activity.get(&node_id) {
            // Nodes with lower historical priority get positive adjustment
            let global_avg = self.calculate_global_avg_priority()?;
            let adjustment = ((global_avg - activity.avg_priority) * 0.1) as i64;
            Ok(adjustment.max(-50).min(50)) // Limit adjustment range
        } else {
            Ok(0)
        }
    }
    
    /// Calculate global average priority
    fn calculate_global_avg_priority(&self) -> Result<f64> {
        if self.node_activity.is_empty() {
            return Ok(0.0);
        }
        
        let total_priority: i64 = self.node_activity.values().map(|a| a.total_priority).sum();
        let total_enqueues: u64 = self.node_activity.values().map(|a| a.enqueues).sum();
        
        Ok(if total_enqueues > 0 {
            total_priority as f64 / total_enqueues as f64
        } else {
            0.0
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fairness_analyzer_creation() {
        let node_id = [1u8; 32];
        let analyzer = FairnessAnalyzer::new(node_id).unwrap();
        
        assert_eq!(analyzer.node_id, node_id);
        assert!(analyzer.enqueue_history.is_empty());
    }

    #[tokio::test]
    async fn test_enqueue_recording() {
        let node_id = [1u8; 32];
        let mut analyzer = FairnessAnalyzer::new(node_id).unwrap();
        
        analyzer.record_enqueue(node_id, TransactionType::User, 100).await.unwrap();
        
        assert_eq!(analyzer.enqueue_history.len(), 1);
        assert!(analyzer.node_activity.contains_key(&node_id));
    }

    #[tokio::test]
    async fn test_fairness_metrics() {
        let node_id = [1u8; 32];
        let mut analyzer = FairnessAnalyzer::new(node_id).unwrap();
        
        // Record some activity
        analyzer.record_enqueue(node_id, TransactionType::User, 100).await.unwrap();
        analyzer.record_dequeue(TransactionType::User, Instant::now()).await.unwrap();
        
        let metrics = analyzer.get_current_metrics().await.unwrap();
        
        assert!(metrics.overall_fairness_score >= 0.0);
        assert!(metrics.overall_fairness_score <= 1.0);
        assert!(metrics.jains_fairness_index >= 0.0);
    }
}