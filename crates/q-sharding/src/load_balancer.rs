// Q-NarwhalKnight Dynamic Load Balancer
// Intelligent distribution of workload across shards

use crate::{ShardConfig, ShardId, ShardMetrics};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;

/// Dynamic load balancer for distributing work across shards
#[derive(Debug)]
pub struct LoadBalancer {
    config: ShardConfig,
    shard_loads: Arc<RwLock<HashMap<u32, ShardLoadState>>>,
    load_history: Arc<RwLock<VecDeque<LoadSnapshot>>>,
    balancing_strategy: LoadBalancingStrategy,
    last_rebalance: Arc<RwLock<Instant>>,
    performance_predictor: PerformancePredictor,
}

/// Current load state for a shard
#[derive(Debug, Clone)]
struct ShardLoadState {
    shard_id: u32,
    current_tps: f64,
    queue_depth: usize,
    cpu_utilization: f64,
    memory_usage_mb: f64,
    last_updated: Instant,
    load_score: f64, // Composite load metric
    is_healthy: bool,
}

/// Historical load snapshot for trend analysis
#[derive(Debug, Clone)]
struct LoadSnapshot {
    timestamp: SystemTime,
    shard_loads: HashMap<u32, f64>,
    total_system_load: f64,
}

/// Load balancing strategies
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least loaded shard selection
    LeastLoaded,
    /// Weighted round-robin based on shard capacity
    WeightedRoundRobin,
    /// Predictive load balancing using ML
    Predictive,
    /// Custom hybrid strategy
    Hybrid,
}

/// Performance predictor using historical data
#[derive(Debug)]
struct PerformancePredictor {
    prediction_window: Duration,
    historical_accuracy: f64,
    trend_weights: HashMap<String, f64>, // feature -> weight
}

impl Default for PerformancePredictor {
    fn default() -> Self {
        let mut trend_weights = HashMap::new();
        trend_weights.insert("cpu_trend".to_string(), 0.3);
        trend_weights.insert("memory_trend".to_string(), 0.2);
        trend_weights.insert("tps_trend".to_string(), 0.4);
        trend_weights.insert("queue_trend".to_string(), 0.1);

        Self {
            prediction_window: Duration::from_secs(60),
            historical_accuracy: 0.85,
            trend_weights,
        }
    }
}

impl LoadBalancer {
    /// Create new load balancer
    pub async fn new(config: ShardConfig) -> Result<Self> {
        tracing::info!(
            "Creating load balancer for {} shards",
            config.consensus_shards
        );

        let shard_loads = Arc::new(RwLock::new(HashMap::new()));
        let load_history = Arc::new(RwLock::new(VecDeque::new()));
        let last_rebalance = Arc::new(RwLock::new(Instant::now()));

        // Initialize shard load states
        {
            let mut loads = shard_loads.write().await;
            for shard_id in 0..config.consensus_shards {
                loads.insert(
                    shard_id,
                    ShardLoadState {
                        shard_id,
                        current_tps: 0.0,
                        queue_depth: 0,
                        cpu_utilization: 0.0,
                        memory_usage_mb: 0.0,
                        last_updated: Instant::now(),
                        load_score: 0.0,
                        is_healthy: true,
                    },
                );
            }
        }

        Ok(Self {
            config,
            shard_loads,
            load_history,
            balancing_strategy: LoadBalancingStrategy::LeastLoaded,
            last_rebalance,
            performance_predictor: PerformancePredictor::default(),
        })
    }

    /// Update load information for a specific shard
    pub async fn update_shard_load(&self, shard_metrics: ShardMetrics) -> Result<()> {
        let mut loads = self.shard_loads.write().await;

        if let ShardId::Consensus(shard_id) = shard_metrics.shard_id {
            let load_score = self.calculate_load_score(&shard_metrics);
            let is_healthy = self.assess_shard_health(&shard_metrics);

            loads.insert(
                shard_id,
                ShardLoadState {
                    shard_id,
                    current_tps: shard_metrics.transactions_per_second,
                    queue_depth: shard_metrics.queue_depth,
                    cpu_utilization: shard_metrics.cpu_utilization,
                    memory_usage_mb: shard_metrics.memory_usage_mb,
                    last_updated: Instant::now(),
                    load_score,
                    is_healthy,
                },
            );

            tracing::debug!(
                "Updated load for shard {}: TPS={:.1}, Load Score={:.2}, Healthy={}",
                shard_id,
                shard_metrics.transactions_per_second,
                load_score,
                is_healthy
            );
        }

        Ok(())
    }

    /// Calculate composite load score for a shard
    fn calculate_load_score(&self, metrics: &ShardMetrics) -> f64 {
        // Weighted composite score: higher score = higher load
        let cpu_weight = 0.4;
        let memory_weight = 0.2;
        let queue_weight = 0.3;
        let tps_weight = 0.1;

        let cpu_score = metrics.cpu_utilization;
        let memory_score = (metrics.memory_usage_mb / 1000.0).min(1.0); // Normalize to 0-1
        let queue_score = (metrics.queue_depth as f64 / 1000.0).min(1.0); // Normalize to 0-1
        let tps_score = (metrics.transactions_per_second / 10000.0).min(1.0); // Normalize to 0-1

        cpu_weight * cpu_score
            + memory_weight * memory_score
            + queue_weight * queue_score
            + tps_weight * tps_score
    }

    /// Assess if a shard is healthy
    fn assess_shard_health(&self, metrics: &ShardMetrics) -> bool {
        // Shard is unhealthy if any resource is over threshold
        let cpu_threshold = 0.9; // 90% CPU
        let memory_threshold = 2000.0; // 2GB memory
        let queue_threshold = 5000; // 5000 pending transactions

        metrics.cpu_utilization < cpu_threshold
            && metrics.memory_usage_mb < memory_threshold
            && metrics.queue_depth < queue_threshold
    }

    /// Select the best shard for new workload
    pub async fn select_least_loaded_shard(&self) -> Result<u32> {
        let loads = self.shard_loads.read().await;

        match self.balancing_strategy {
            LoadBalancingStrategy::LeastLoaded => self.select_least_loaded_internal(&loads).await,

            LoadBalancingStrategy::Predictive => self.select_predictive_shard(&loads).await,

            LoadBalancingStrategy::RoundRobin => {
                // Simple round-robin among healthy shards
                let healthy_shards: Vec<u32> = loads
                    .iter()
                    .filter(|(_, state)| state.is_healthy)
                    .map(|(id, _)| *id)
                    .collect();

                if healthy_shards.is_empty() {
                    // Fall back to any shard
                    loads
                        .keys()
                        .next()
                        .copied()
                        .ok_or_else(|| anyhow::anyhow!("No shards available"))
                } else {
                    let now = SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
                    let index = (now.as_secs() % healthy_shards.len() as u64) as usize;
                    Ok(healthy_shards[index])
                }
            }

            LoadBalancingStrategy::WeightedRoundRobin => self.select_weighted_shard(&loads).await,

            LoadBalancingStrategy::Hybrid => {
                // Use least loaded for simplicity, but could combine strategies
                self.select_least_loaded_internal(&loads).await
            }
        }
    }

    /// Select shard using predictive algorithm
    async fn select_predictive_shard(&self, loads: &HashMap<u32, ShardLoadState>) -> Result<u32> {
        let mut best_shard = None;
        let mut best_predicted_load = f64::MAX;

        for (shard_id, load_state) in loads.iter() {
            if !load_state.is_healthy {
                continue;
            }

            // Predict future load based on trends
            let predicted_load = self
                .predict_future_load(*shard_id)
                .await
                .unwrap_or(load_state.load_score);

            if predicted_load < best_predicted_load {
                best_predicted_load = predicted_load;
                best_shard = Some(*shard_id);
            }
        }

        best_shard
            .ok_or_else(|| anyhow::anyhow!("No healthy shards available for predictive selection"))
    }

    /// Select shard using weighted round-robin
    async fn select_weighted_shard(&self, loads: &HashMap<u32, ShardLoadState>) -> Result<u32> {
        // Calculate weights based on inverse load scores
        let mut weights: Vec<(u32, f64)> = loads
            .iter()
            .filter(|(_, state)| state.is_healthy)
            .map(|(id, state)| {
                let weight = if state.load_score > 0.0 {
                    1.0 / state.load_score
                } else {
                    1.0
                };
                (*id, weight)
            })
            .collect();

        if weights.is_empty() {
            return Err(anyhow::anyhow!("No healthy shards available"));
        }

        // Simple weighted selection (could be more sophisticated)
        weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(weights[0].0)
    }

    /// Internal least-loaded selection to avoid recursion
    async fn select_least_loaded_internal(
        &self,
        loads: &HashMap<u32, ShardLoadState>,
    ) -> Result<u32> {
        // Find shard with lowest load score among healthy shards
        let mut best_shard = None;
        let mut lowest_load = f64::MAX;

        for (shard_id, load_state) in loads.iter() {
            if load_state.is_healthy && load_state.load_score < lowest_load {
                lowest_load = load_state.load_score;
                best_shard = Some(*shard_id);
            }
        }

        // If no healthy shard found, pick the least loaded unhealthy one
        if best_shard.is_none() {
            for (shard_id, load_state) in loads.iter() {
                if load_state.load_score < lowest_load {
                    lowest_load = load_state.load_score;
                    best_shard = Some(*shard_id);
                }
            }
        }

        best_shard.ok_or_else(|| anyhow::anyhow!("No shards available"))
    }

    /// Predict future load for a shard based on historical trends
    async fn predict_future_load(&self, shard_id: u32) -> Result<f64> {
        let history = self.load_history.read().await;

        if history.len() < 3 {
            // Not enough historical data
            return Err(anyhow::anyhow!(
                "Insufficient historical data for prediction"
            ));
        }

        // Simple linear trend prediction
        let recent_loads: Vec<f64> = history
            .iter()
            .rev()
            .take(5) // Use last 5 snapshots
            .filter_map(|snapshot| snapshot.shard_loads.get(&shard_id))
            .cloned()
            .collect();

        if recent_loads.len() < 2 {
            return Err(anyhow::anyhow!("Insufficient shard data for prediction"));
        }

        // Calculate trend
        let mut trend = 0.0;
        for i in 1..recent_loads.len() {
            trend += recent_loads[i] - recent_loads[i - 1];
        }
        trend /= (recent_loads.len() - 1) as f64;

        // Predict next load
        let current_load = recent_loads[0];
        let predicted_load = current_load + trend;

        Ok(predicted_load.max(0.0)) // Ensure non-negative prediction
    }

    /// Record current system load snapshot for historical analysis
    pub async fn record_load_snapshot(&self) -> Result<()> {
        let loads = self.shard_loads.read().await;
        let mut shard_loads = HashMap::new();
        let mut total_load = 0.0;

        for (shard_id, load_state) in loads.iter() {
            shard_loads.insert(*shard_id, load_state.load_score);
            total_load += load_state.load_score;
        }

        let snapshot = LoadSnapshot {
            timestamp: SystemTime::now(),
            shard_loads,
            total_system_load: total_load,
        };

        let mut history = self.load_history.write().await;
        history.push_back(snapshot);

        // Keep only recent history (last 100 snapshots)
        while history.len() > 100 {
            history.pop_front();
        }

        Ok(())
    }

    /// Check if system needs rebalancing
    pub async fn needs_rebalancing(&self) -> Result<bool> {
        let loads = self.shard_loads.read().await;
        let last_rebalance = self.last_rebalance.read().await;

        // Don't rebalance too frequently
        if last_rebalance.elapsed() < Duration::from_secs(60) {
            return Ok(false);
        }

        // Check load imbalance
        let load_scores: Vec<f64> = loads.values().map(|s| s.load_score).collect();

        if load_scores.is_empty() {
            return Ok(false);
        }

        let max_load = load_scores.iter().cloned().fold(0.0, f64::max);
        let min_load = load_scores.iter().cloned().fold(f64::MAX, f64::min);

        // Rebalance if load difference exceeds threshold
        let load_imbalance = max_load - min_load;
        let rebalance_threshold = self.config.rebalance_threshold;

        Ok(load_imbalance > rebalance_threshold)
    }

    /// Get current load balancing metrics
    pub async fn get_load_metrics(&self) -> Result<LoadBalancingMetrics> {
        let loads = self.shard_loads.read().await;
        let history = self.load_history.read().await;

        let total_shards = loads.len();
        let healthy_shards = loads.values().filter(|s| s.is_healthy).count();
        let unhealthy_shards = total_shards - healthy_shards;

        let load_scores: Vec<f64> = loads.values().map(|s| s.load_score).collect();
        let average_load = if !load_scores.is_empty() {
            load_scores.iter().sum::<f64>() / load_scores.len() as f64
        } else {
            0.0
        };

        let max_load = load_scores.iter().cloned().fold(0.0, f64::max);
        let min_load = load_scores.iter().cloned().fold(f64::MAX, f64::min);
        let load_variance = max_load - min_load;

        Ok(LoadBalancingMetrics {
            total_shards,
            healthy_shards,
            unhealthy_shards,
            average_load,
            max_load,
            min_load,
            load_variance,
            prediction_accuracy: self.performance_predictor.historical_accuracy,
            snapshots_recorded: history.len(),
        })
    }

    /// Mark rebalancing as completed
    pub async fn mark_rebalanced(&self) {
        let mut last_rebalance = self.last_rebalance.write().await;
        *last_rebalance = Instant::now();
    }

    /// Get recommended shard for migration during rebalancing
    pub async fn get_migration_target(&self, source_shard: u32) -> Result<Option<u32>> {
        let loads = self.shard_loads.read().await;

        if let Some(source_load) = loads.get(&source_shard) {
            // Find the shard with lowest load that's different from source
            let mut best_target = None;
            let mut lowest_load = f64::MAX;

            for (target_shard, load_state) in loads.iter() {
                if *target_shard != source_shard
                    && load_state.is_healthy
                    && load_state.load_score < lowest_load
                    && load_state.load_score < source_load.load_score * 0.8
                {
                    // Target must be significantly less loaded

                    lowest_load = load_state.load_score;
                    best_target = Some(*target_shard);
                }
            }

            Ok(best_target)
        } else {
            Ok(None)
        }
    }
}

/// Load balancing performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingMetrics {
    pub total_shards: usize,
    pub healthy_shards: usize,
    pub unhealthy_shards: usize,
    pub average_load: f64,
    pub max_load: f64,
    pub min_load: f64,
    pub load_variance: f64,
    pub prediction_accuracy: f64,
    pub snapshots_recorded: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_load_balancer_creation() {
        let config = ShardConfig::default();
        let balancer = LoadBalancer::new(config).await;
        assert!(balancer.is_ok());
    }

    #[tokio::test]
    async fn test_load_score_calculation() {
        let config = ShardConfig::default();
        let balancer = LoadBalancer::new(config).await.unwrap();

        let metrics = ShardMetrics {
            shard_id: ShardId::Consensus(0),
            transactions_per_second: 1000.0,
            average_latency_ms: 10.0,
            cpu_utilization: 0.5,
            memory_usage_mb: 500.0,
            active_connections: 10,
            queue_depth: 100,
        };

        let load_score = balancer.calculate_load_score(&metrics);
        assert!(load_score >= 0.0 && load_score <= 1.0);
    }

    #[tokio::test]
    async fn test_shard_selection() {
        let config = ShardConfig::default();
        let balancer = LoadBalancer::new(config).await.unwrap();

        let selected_shard = balancer.select_least_loaded_shard().await;
        assert!(selected_shard.is_ok());

        let shard_id = selected_shard.unwrap();
        assert!(shard_id < config.consensus_shards);
    }

    #[tokio::test]
    async fn test_load_metrics() {
        let config = ShardConfig::default();
        let balancer = LoadBalancer::new(config).await.unwrap();

        let metrics = balancer.get_load_metrics().await;
        assert!(metrics.is_ok());

        let metrics = metrics.unwrap();
        assert_eq!(metrics.total_shards, config.consensus_shards as usize);
    }
}
