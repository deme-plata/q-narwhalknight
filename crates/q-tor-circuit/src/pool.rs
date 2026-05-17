/// Circuit Pool: Manages dedicated circuit pools for Q-NarwhalKnight
/// Maintains 4 circuits per validator with specific purposes
use anyhow::{Context, Result};
use q_types::{NodeId, Phase};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

use super::{CircuitInfo, CircuitPurpose};

/// Circuit pool status
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PoolStatus {
    /// Pool is initializing
    Initializing,
    /// Pool is healthy with all circuits active
    Healthy,
    /// Pool is degraded with some circuits down
    Degraded,
    /// Pool is critical with most circuits down  
    Critical,
    /// Pool is shut down
    Shutdown,
}

/// Circuit health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitHealth {
    pub circuit_id: u64,
    pub purpose: CircuitPurpose,
    pub status: CircuitStatus,
    pub latency_ms: Option<u64>,
    pub error_rate: f64,
    pub last_error: Option<String>,
    pub uptime: Duration,
}

/// Individual circuit status
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CircuitStatus {
    /// Circuit is being built
    Building,
    /// Circuit is active and ready
    Active,
    /// Circuit has connectivity issues
    Degraded,
    /// Circuit has failed
    Failed,
    /// Circuit is being torn down
    Closing,
}

/// Circuit pool manager
pub struct CircuitPool {
    /// Map of circuit purpose to circuit IDs
    circuits: Arc<RwLock<HashMap<CircuitPurpose, Vec<u64>>>>,
    /// Circuit health information
    health: Arc<RwLock<HashMap<u64, CircuitHealth>>>,
    /// Pool configuration
    config: PoolConfig,
    /// Node ID for this validator
    node_id: NodeId,
    /// Current phase
    phase: Phase,
    /// Pool status
    status: Arc<Mutex<PoolStatus>>,
    /// Circuit creation counter
    circuit_counter: Arc<Mutex<u64>>,
}

/// Pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum circuits per purpose
    pub max_circuits_per_purpose: usize,
    /// Minimum healthy circuits per purpose
    pub min_healthy_circuits: usize,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Circuit timeout for creation
    pub circuit_creation_timeout: Duration,
    /// Maximum error rate before marking circuit failed
    pub max_error_rate: f64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            // v8.6.0: increased from 2 to 3 — more circuits per purpose improves
            // load balancing and reduces per-circuit throughput bottleneck
            max_circuits_per_purpose: 3,
            min_healthy_circuits: 1,
            health_check_interval: Duration::from_secs(30),
            // v8.6.0: reduced from 60s to 30s — if a circuit can't be built in 30s,
            // the relay is likely overloaded; fail fast and try another path
            circuit_creation_timeout: Duration::from_secs(30),
            max_error_rate: 0.1, // 10% error rate threshold
        }
    }
}

impl CircuitPool {
    /// Create new circuit pool
    pub async fn new(total_circuits: usize, node_id: NodeId) -> Result<Self> {
        info!(
            "🏊 Creating circuit pool with {} total circuits for validator {}",
            total_circuits,
            hex::encode(&node_id[..4])
        );

        let config = PoolConfig::default();

        let pool = Self {
            circuits: Arc::new(RwLock::new(HashMap::new())),
            health: Arc::new(RwLock::new(HashMap::new())),
            config,
            node_id,
            phase: Phase::Phase1, // Default to Phase 1 for post-quantum
            status: Arc::new(Mutex::new(PoolStatus::Initializing)),
            circuit_counter: Arc::new(Mutex::new(1000)), // Start from 1000
        };

        // Initialize circuit pools for each purpose
        pool.initialize_pools(total_circuits).await?;

        // Start health monitor
        pool.start_health_monitor().await;

        {
            let mut status = pool.status.lock().await;
            *status = PoolStatus::Healthy;
        }

        info!("✅ Circuit pool initialized successfully");
        Ok(pool)
    }

    /// Initialize circuit pools for each purpose
    async fn initialize_pools(&self, total_circuits: usize) -> Result<()> {
        let purposes = [
            CircuitPurpose::Control,
            CircuitPurpose::BlockGossip,
            CircuitPurpose::AckGossip,
            CircuitPurpose::QuantumBeacon,
        ];

        let circuits_per_purpose = total_circuits / purposes.len();
        let mut circuits_map = self.circuits.write().await;
        let mut health_map = self.health.write().await;

        for purpose in &purposes {
            let count = if *purpose == CircuitPurpose::Control {
                1 // Only one control circuit
            } else {
                circuits_per_purpose.max(1)
            };

            let mut purpose_circuits = Vec::new();

            for _ in 0..count {
                let circuit_id = self.generate_circuit_id().await;

                // Create circuit health entry
                let health = CircuitHealth {
                    circuit_id,
                    purpose: *purpose,
                    status: CircuitStatus::Building,
                    latency_ms: None,
                    error_rate: 0.0,
                    last_error: None,
                    uptime: Duration::from_secs(0),
                };

                purpose_circuits.push(circuit_id);
                health_map.insert(circuit_id, health);

                debug!("🛠️ Initialized {:?} circuit {}", purpose, circuit_id);
            }

            circuits_map.insert(*purpose, purpose_circuits);
            info!("📋 Created {} circuits for {:?}", count, purpose);
        }

        Ok(())
    }

    /// Generate unique circuit ID
    async fn generate_circuit_id(&self) -> u64 {
        let mut counter = self.circuit_counter.lock().await;
        *counter += 1;
        *counter
    }

    /// Register a new circuit with the pool
    pub async fn register_circuit(
        &mut self,
        circuit_id: u64,
        purpose: CircuitPurpose,
    ) -> Result<()> {
        debug!(
            "📝 Registering circuit {} for purpose {:?}",
            circuit_id, purpose
        );

        // Update circuit health to active
        {
            let mut health_map = self.health.write().await;
            if let Some(health) = health_map.get_mut(&circuit_id) {
                health.status = CircuitStatus::Active;
                info!("✅ Circuit {} is now active", circuit_id);
            } else {
                warn!("⚠️ Attempted to register unknown circuit {}", circuit_id);
            }
        }

        Ok(())
    }

    /// Get healthy circuits for a specific purpose
    pub async fn get_healthy_circuits(&self, purpose: CircuitPurpose) -> Result<Vec<u64>> {
        let circuits_map = self.circuits.read().await;
        let health_map = self.health.read().await;

        if let Some(circuit_ids) = circuits_map.get(&purpose) {
            let healthy_circuits: Vec<u64> = circuit_ids
                .iter()
                .filter_map(|&id| {
                    health_map.get(&id).and_then(|health| {
                        if health.status == CircuitStatus::Active {
                            Some(id)
                        } else {
                            None
                        }
                    })
                })
                .collect();

            if healthy_circuits.is_empty() {
                anyhow::bail!("No healthy circuits available for purpose {:?}", purpose);
            }

            Ok(healthy_circuits)
        } else {
            anyhow::bail!("No circuits configured for purpose {:?}", purpose);
        }
    }

    /// Get the best circuit for a purpose (lowest latency, lowest load)
    pub async fn get_best_circuit(&self, purpose: CircuitPurpose) -> Result<u64> {
        let healthy_circuits = self.get_healthy_circuits(purpose).await?;
        let health_map = self.health.read().await;

        let best_circuit = healthy_circuits
            .iter()
            .min_by(|&&a, &&b| {
                let health_a = health_map.get(&a).unwrap();
                let health_b = health_map.get(&b).unwrap();

                // Primary: lowest error rate
                match health_a.error_rate.partial_cmp(&health_b.error_rate) {
                    Some(std::cmp::Ordering::Equal) => {
                        // Secondary: lowest latency
                        match (health_a.latency_ms, health_b.latency_ms) {
                            (Some(lat_a), Some(lat_b)) => lat_a.cmp(&lat_b),
                            (Some(_), None) => std::cmp::Ordering::Less,
                            (None, Some(_)) => std::cmp::Ordering::Greater,
                            (None, None) => std::cmp::Ordering::Equal,
                        }
                    }
                    Some(ordering) => ordering,
                    None => std::cmp::Ordering::Equal,
                }
            })
            .copied();

        match best_circuit {
            Some(circuit_id) => {
                debug!(
                    "🎯 Selected circuit {} as best for {:?}",
                    circuit_id, purpose
                );
                Ok(circuit_id)
            }
            None => {
                anyhow::bail!("No suitable circuit found for purpose {:?}", purpose);
            }
        }
    }

    /// Update circuit health metrics
    pub async fn update_circuit_health(
        &self,
        circuit_id: u64,
        latency: Option<Duration>,
        success: bool,
        error: Option<String>,
    ) -> Result<()> {
        let mut health_map = self.health.write().await;

        if let Some(health) = health_map.get_mut(&circuit_id) {
            // Update latency
            if let Some(latency) = latency {
                health.latency_ms = Some(latency.as_millis() as u64);
            }

            // Update error rate using exponential moving average
            let error_weight = 0.1; // 10% weight for new errors
            if success {
                health.error_rate = health.error_rate * (1.0 - error_weight);
            } else {
                health.error_rate = health.error_rate * (1.0 - error_weight) + error_weight;
                health.last_error = error;

                warn!(
                    "⚠️ Circuit {} error rate now: {:.2}%",
                    circuit_id,
                    health.error_rate * 100.0
                );
            }

            // Check if circuit should be marked as failed
            if health.error_rate > self.config.max_error_rate {
                health.status = CircuitStatus::Failed;
                error!(
                    "❌ Circuit {} marked as failed (error rate: {:.2}%)",
                    circuit_id,
                    health.error_rate * 100.0
                );
            }

            debug!(
                "📊 Updated health for circuit {}: latency={:?}ms, error_rate={:.2}%",
                circuit_id,
                health.latency_ms,
                health.error_rate * 100.0
            );
        } else {
            warn!(
                "⚠️ Attempted to update health for unknown circuit {}",
                circuit_id
            );
        }

        Ok(())
    }

    /// Start health monitoring background task
    async fn start_health_monitor(&self) {
        let health = self.health.clone();
        let status = self.status.clone();
        let interval = self.config.health_check_interval;

        tokio::spawn(async move {
            let mut health_check_interval = tokio::time::interval(interval);

            loop {
                health_check_interval.tick().await;

                let current_status = {
                    let health_map = health.read().await;
                    Self::calculate_pool_status(&health_map)
                };

                {
                    let mut status_guard = status.lock().await;
                    if *status_guard != current_status {
                        info!(
                            "🔄 Pool status changed: {:?} -> {:?}",
                            *status_guard, current_status
                        );
                        *status_guard = current_status;
                    }
                }

                // Log health summary every few minutes
                Self::log_health_summary(&health).await;
            }
        });
    }

    /// Calculate overall pool status based on circuit health
    fn calculate_pool_status(health_map: &HashMap<u64, CircuitHealth>) -> PoolStatus {
        let total_circuits = health_map.len();
        if total_circuits == 0 {
            return PoolStatus::Shutdown;
        }

        let active_circuits = health_map
            .values()
            .filter(|h| h.status == CircuitStatus::Active)
            .count();

        let failed_circuits = health_map
            .values()
            .filter(|h| h.status == CircuitStatus::Failed)
            .count();

        let healthy_ratio = active_circuits as f64 / total_circuits as f64;

        match healthy_ratio {
            r if r >= 0.8 => PoolStatus::Healthy,
            r if r >= 0.5 => PoolStatus::Degraded,
            _ => PoolStatus::Critical,
        }
    }

    /// Log health summary periodically
    async fn log_health_summary(health: &Arc<RwLock<HashMap<u64, CircuitHealth>>>) {
        let health_map = health.read().await;
        let total = health_map.len();
        let active = health_map
            .values()
            .filter(|h| h.status == CircuitStatus::Active)
            .count();
        let failed = health_map
            .values()
            .filter(|h| h.status == CircuitStatus::Failed)
            .count();

        info!(
            "📊 Circuit Health Summary: {}/{} active, {} failed",
            active, total, failed
        );
    }

    /// Get pool status
    pub async fn get_pool_status(&self) -> PoolStatus {
        let status = self.status.lock().await;
        *status
    }

    /// Get detailed pool statistics
    pub async fn get_pool_stats(&self) -> PoolStats {
        let circuits_map = self.circuits.read().await;
        let health_map = self.health.read().await;

        let mut purpose_stats = HashMap::new();
        let mut total_active = 0;
        let mut total_failed = 0;
        let mut total_latency = 0u64;
        let mut latency_count = 0;

        for (purpose, circuit_ids) in circuits_map.iter() {
            let mut active_count = 0;
            let mut failed_count = 0;
            let mut purpose_latency = 0u64;
            let mut purpose_latency_count = 0;

            for &circuit_id in circuit_ids {
                if let Some(health) = health_map.get(&circuit_id) {
                    match health.status {
                        CircuitStatus::Active => active_count += 1,
                        CircuitStatus::Failed => failed_count += 1,
                        _ => {}
                    }

                    if let Some(latency) = health.latency_ms {
                        purpose_latency += latency;
                        purpose_latency_count += 1;
                    }
                }
            }

            total_active += active_count;
            total_failed += failed_count;
            total_latency += purpose_latency;
            latency_count += purpose_latency_count;

            let avg_latency = if purpose_latency_count > 0 {
                Some(Duration::from_millis(
                    purpose_latency / purpose_latency_count as u64,
                ))
            } else {
                None
            };

            purpose_stats.insert(
                *purpose,
                PurposeStats {
                    total_circuits: circuit_ids.len(),
                    active_circuits: active_count,
                    failed_circuits: failed_count,
                    average_latency: avg_latency,
                },
            );
        }

        let overall_avg_latency = if latency_count > 0 {
            Some(Duration::from_millis(total_latency / latency_count as u64))
        } else {
            None
        };

        PoolStats {
            total_circuits: health_map.len(),
            active_circuits: total_active,
            failed_circuits: total_failed,
            average_latency: overall_avg_latency,
            pool_status: self.get_pool_status().await,
            purpose_breakdown: purpose_stats,
        }
    }

    /// Shutdown the circuit pool
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("🛑 Shutting down circuit pool");

        {
            let mut status = self.status.lock().await;
            *status = PoolStatus::Shutdown;
        }

        // Mark all circuits as closing
        {
            let mut health_map = self.health.write().await;
            for health in health_map.values_mut() {
                health.status = CircuitStatus::Closing;
            }
        }

        info!("✅ Circuit pool shutdown complete");
        Ok(())
    }
}

/// Pool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStats {
    pub total_circuits: usize,
    pub active_circuits: usize,
    pub failed_circuits: usize,
    pub average_latency: Option<Duration>,
    pub pool_status: PoolStatus,
    pub purpose_breakdown: HashMap<CircuitPurpose, PurposeStats>,
}

/// Statistics for circuits of specific purpose
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurposeStats {
    pub total_circuits: usize,
    pub active_circuits: usize,
    pub failed_circuits: usize,
    pub average_latency: Option<Duration>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_circuit_pool_creation() {
        let node_id = [1u8; 32];
        let pool = CircuitPool::new(4, node_id).await;
        assert!(pool.is_ok());

        let pool = pool.unwrap();
        let stats = pool.get_pool_stats().await;
        assert_eq!(stats.total_circuits, 4);
    }

    #[tokio::test]
    async fn test_circuit_registration() {
        let node_id = [1u8; 32];
        let mut pool = CircuitPool::new(4, node_id).await.unwrap();

        let result = pool.register_circuit(1001, CircuitPurpose::Control).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_health_update() {
        let node_id = [1u8; 32];
        let pool = CircuitPool::new(4, node_id).await.unwrap();

        let result = pool
            .update_circuit_health(1001, Some(Duration::from_millis(150)), true, None)
            .await;

        // Should succeed even if circuit doesn't exist (just warns)
        assert!(result.is_ok());
    }

    #[test]
    fn test_pool_status_calculation() {
        let mut health_map = HashMap::new();

        // All active
        for i in 0..4 {
            health_map.insert(
                i,
                CircuitHealth {
                    circuit_id: i,
                    purpose: CircuitPurpose::Control,
                    status: CircuitStatus::Active,
                    latency_ms: Some(100),
                    error_rate: 0.01,
                    last_error: None,
                    uptime: Duration::from_secs(3600),
                },
            );
        }

        let status = CircuitPool::calculate_pool_status(&health_map);
        assert_eq!(status, PoolStatus::Healthy);

        // Half failed
        health_map.get_mut(&2).unwrap().status = CircuitStatus::Failed;
        health_map.get_mut(&3).unwrap().status = CircuitStatus::Failed;

        let status = CircuitPool::calculate_pool_status(&health_map);
        assert_eq!(status, PoolStatus::Degraded);
    }
}
