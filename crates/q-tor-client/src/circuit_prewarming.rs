/// Proactive Circuit Prewarming for Q-NarwhalKnight Tor Layer
///
/// This module provides background circuit management that:
/// - Monitors circuit expiration times
/// - Creates replacement circuits before they expire
/// - Eliminates cold-start latency for first connections
/// - Maintains healthy circuit pool for all operation types
///
/// Benefits:
/// - Zero latency on first request to any operation type
/// - Seamless circuit rotation without connection interruption
/// - Automatic recovery from circuit failures
/// - Health monitoring and reporting

use crate::dedicated_circuits::{
    DedicatedCircuitConfig, DedicatedCircuitManager, IsolatedCircuitStats, ManagerStats,
    OperationType,
};
use anyhow::Result;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime},
};
use tokio::{
    sync::{mpsc, RwLock},
    task::JoinHandle,
    time::interval,
};
use tracing::{debug, error, info, warn};

/// Configuration for circuit prewarming
#[derive(Debug, Clone)]
pub struct PrewarmingConfig {
    /// How far ahead to prewarm circuits (before rotation)
    pub prewarm_ahead: Duration,
    /// Interval for checking circuit health
    pub health_check_interval: Duration,
    /// Maximum consecutive failures before alerting
    pub max_consecutive_failures: u32,
    /// Enable adaptive prewarming (based on usage patterns)
    pub adaptive_prewarming: bool,
    /// Minimum healthy circuits per operation type
    pub min_healthy_circuits: usize,
}

impl Default for PrewarmingConfig {
    fn default() -> Self {
        Self {
            prewarm_ahead: Duration::from_secs(60), // Prewarm 1 minute before expiration
            health_check_interval: Duration::from_secs(30), // Check every 30 seconds
            max_consecutive_failures: 3,
            adaptive_prewarming: true,
            min_healthy_circuits: 1,
        }
    }
}

/// Health status of a circuit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitHealth {
    /// Circuit is healthy and ready
    Healthy,
    /// Circuit is expiring soon
    ExpiringSoon,
    /// Circuit has failed
    Failed,
    /// Circuit doesn't exist yet
    NotCreated,
}

/// Circuit health report for monitoring
#[derive(Debug, Clone)]
pub struct CircuitHealthReport {
    pub operation_type: OperationType,
    pub health: CircuitHealth,
    pub time_until_rotation: Option<Duration>,
    pub requests_served: u64,
    pub average_latency_ms: f64,
    pub last_checked: Instant,
}

/// Prewarming statistics
#[derive(Debug, Clone, Default)]
pub struct PrewarmingStats {
    pub circuits_prewarmed: u64,
    pub prewarm_failures: u64,
    pub health_checks_performed: u64,
    pub unhealthy_circuits_detected: u64,
    pub auto_recoveries: u64,
    pub uptime: Duration,
    pub started_at: Option<SystemTime>,
}

/// Circuit Prewarming Manager
///
/// Runs background tasks to:
/// 1. Monitor circuit expiration
/// 2. Create replacement circuits before they expire
/// 3. Perform health checks
/// 4. Report circuit health status
pub struct CircuitPrewarmingManager {
    /// Reference to the DedicatedCircuitManager
    circuit_manager: Arc<DedicatedCircuitManager>,
    /// Configuration
    config: PrewarmingConfig,
    /// Prewarming statistics
    stats: Arc<RwLock<PrewarmingStats>>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// Health reports per operation type
    health_reports: Arc<RwLock<HashMap<OperationType, CircuitHealthReport>>>,
    /// Background task handles
    task_handles: RwLock<Vec<JoinHandle<()>>>,
    /// Last health check time
    last_health_check: Arc<RwLock<Instant>>,
    /// Consecutive failures counter
    consecutive_failures: Arc<AtomicU64>,
}

impl CircuitPrewarmingManager {
    /// Create a new prewarming manager
    pub fn new(
        circuit_manager: Arc<DedicatedCircuitManager>,
        config: PrewarmingConfig,
    ) -> Self {
        info!("🔥 Creating Circuit Prewarming Manager");
        info!("   Prewarm ahead: {}s", config.prewarm_ahead.as_secs());
        info!(
            "   Health check interval: {}s",
            config.health_check_interval.as_secs()
        );

        Self {
            circuit_manager,
            config,
            stats: Arc::new(RwLock::new(PrewarmingStats {
                started_at: Some(SystemTime::now()),
                ..Default::default()
            })),
            shutdown: Arc::new(AtomicBool::new(false)),
            health_reports: Arc::new(RwLock::new(HashMap::new())),
            task_handles: RwLock::new(Vec::new()),
            last_health_check: Arc::new(RwLock::new(Instant::now())),
            consecutive_failures: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Start the prewarming background tasks
    pub async fn start(&self) -> Result<()> {
        info!("🚀 Starting Circuit Prewarming Manager background tasks");

        // Start health check task
        let health_check_handle = self.spawn_health_check_task().await;

        // Start prewarming task
        let prewarm_handle = self.spawn_prewarm_task().await;

        // Store handles
        {
            let mut handles = self.task_handles.write().await;
            handles.push(health_check_handle);
            handles.push(prewarm_handle);
        }

        // Perform initial prewarm
        self.initial_prewarm().await?;

        info!("✅ Circuit Prewarming Manager started");
        Ok(())
    }

    /// Perform initial prewarming for all operation types
    async fn initial_prewarm(&self) -> Result<()> {
        info!("🔥 Performing initial circuit prewarming...");

        let operation_types = Self::all_operation_types();

        for op_type in operation_types {
            match self.prewarm_circuit(op_type).await {
                Ok(()) => {
                    debug!("✅ Prewarmed circuit for {}", op_type.name());
                }
                Err(e) => {
                    warn!("⚠️ Failed to prewarm circuit for {}: {}", op_type.name(), e);
                }
            }
        }

        info!("✅ Initial prewarming complete");
        Ok(())
    }

    /// Spawn the health check background task
    async fn spawn_health_check_task(&self) -> JoinHandle<()> {
        let circuit_manager = Arc::clone(&self.circuit_manager);
        let shutdown = Arc::clone(&self.shutdown);
        let health_reports = Arc::clone(&self.health_reports);
        let stats = Arc::clone(&self.stats);
        let config = self.config.clone();
        let last_health_check = Arc::clone(&self.last_health_check);
        let consecutive_failures = Arc::clone(&self.consecutive_failures);

        tokio::spawn(async move {
            let mut interval = interval(config.health_check_interval);

            loop {
                interval.tick().await;

                if shutdown.load(Ordering::Relaxed) {
                    info!("🛑 Health check task shutting down");
                    break;
                }

                // Update last health check time
                {
                    let mut last = last_health_check.write().await;
                    *last = Instant::now();
                }

                // Check health of all operation types
                let all_stats = circuit_manager.get_all_stats().await;
                let mut new_reports = HashMap::new();

                for op_type in Self::all_operation_types() {
                    let health = if let Some(circuit_stats) = all_stats.get(&op_type) {
                        Self::calculate_health(&config, op_type, circuit_stats)
                    } else {
                        CircuitHealth::NotCreated
                    };

                    let report = CircuitHealthReport {
                        operation_type: op_type,
                        health,
                        time_until_rotation: all_stats.get(&op_type).map(|s| {
                            let elapsed = s.last_rotation.elapsed();
                            let rotation_interval = op_type.rotation_interval();
                            if elapsed < rotation_interval {
                                rotation_interval - elapsed
                            } else {
                                Duration::ZERO
                            }
                        }),
                        requests_served: all_stats
                            .get(&op_type)
                            .map(|s| s.requests_served)
                            .unwrap_or(0),
                        average_latency_ms: all_stats
                            .get(&op_type)
                            .map(|s| s.average_latency_ms)
                            .unwrap_or(0.0),
                        last_checked: Instant::now(),
                    };

                    // Track unhealthy circuits
                    if health != CircuitHealth::Healthy && health != CircuitHealth::NotCreated {
                        let mut s = stats.write().await;
                        s.unhealthy_circuits_detected += 1;
                    }

                    new_reports.insert(op_type, report);
                }

                // Update health reports
                {
                    let mut reports = health_reports.write().await;
                    *reports = new_reports;
                }

                // Update stats
                {
                    let mut s = stats.write().await;
                    s.health_checks_performed += 1;
                    if let Some(started_at) = s.started_at {
                        s.uptime = started_at.elapsed().unwrap_or_default();
                    }
                }

                debug!("🔍 Health check complete");
            }
        })
    }

    /// Spawn the prewarming background task
    async fn spawn_prewarm_task(&self) -> JoinHandle<()> {
        let circuit_manager = Arc::clone(&self.circuit_manager);
        let shutdown = Arc::clone(&self.shutdown);
        let health_reports = Arc::clone(&self.health_reports);
        let stats = Arc::clone(&self.stats);
        let config = self.config.clone();
        let consecutive_failures = Arc::clone(&self.consecutive_failures);

        tokio::spawn(async move {
            // Check more frequently than health checks
            let check_interval = config.health_check_interval / 2;
            let mut interval = interval(check_interval);

            loop {
                interval.tick().await;

                if shutdown.load(Ordering::Relaxed) {
                    info!("🛑 Prewarm task shutting down");
                    break;
                }

                // Check which circuits need prewarming
                let reports = health_reports.read().await;

                for (op_type, report) in reports.iter() {
                    // Prewarm if expiring soon or failed
                    let should_prewarm = match report.health {
                        CircuitHealth::ExpiringSoon => true,
                        CircuitHealth::Failed => true,
                        CircuitHealth::NotCreated => true,
                        CircuitHealth::Healthy => {
                            // Also check if time until rotation is less than prewarm_ahead
                            report
                                .time_until_rotation
                                .map(|t| t < config.prewarm_ahead)
                                .unwrap_or(false)
                        }
                    };

                    if should_prewarm {
                        debug!(
                            "🔥 Prewarming circuit for {} (health: {:?})",
                            op_type.name(),
                            report.health
                        );

                        // Trigger circuit rotation to get fresh circuit
                        match circuit_manager.get_client(*op_type).await {
                            Ok(_) => {
                                let mut s = stats.write().await;
                                s.circuits_prewarmed += 1;
                                consecutive_failures.store(0, Ordering::Relaxed);

                                if report.health == CircuitHealth::Failed {
                                    s.auto_recoveries += 1;
                                    info!("🔧 Auto-recovered circuit for {}", op_type.name());
                                }

                                debug!("✅ Prewarmed circuit for {}", op_type.name());
                            }
                            Err(e) => {
                                let mut s = stats.write().await;
                                s.prewarm_failures += 1;
                                let failures =
                                    consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;

                                warn!(
                                    "⚠️ Failed to prewarm circuit for {}: {}",
                                    op_type.name(),
                                    e
                                );

                                if failures >= config.max_consecutive_failures as u64 {
                                    error!(
                                        "❌ {} consecutive prewarm failures! Tor network may be unavailable",
                                        failures
                                    );
                                }
                            }
                        }
                    }
                }
            }
        })
    }

    /// Prewarm a specific circuit type
    async fn prewarm_circuit(&self, op_type: OperationType) -> Result<()> {
        // Get or create client (this triggers circuit creation)
        let _client = self.circuit_manager.get_client(op_type).await?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.circuits_prewarmed += 1;
        }

        Ok(())
    }

    /// Calculate health status for a circuit
    fn calculate_health(
        config: &PrewarmingConfig,
        op_type: OperationType,
        stats: &IsolatedCircuitStats,
    ) -> CircuitHealth {
        let rotation_interval = op_type.rotation_interval();
        let elapsed = stats.last_rotation.elapsed();

        // Check if circuit is expiring soon
        if elapsed + config.prewarm_ahead >= rotation_interval {
            return CircuitHealth::ExpiringSoon;
        }

        // Check failure rate
        if stats.failures > 3 && stats.requests_served > 0 {
            let failure_rate = stats.failures as f64 / stats.requests_served as f64;
            if failure_rate > 0.1 {
                // More than 10% failures
                return CircuitHealth::Failed;
            }
        }

        // Check latency (if it's too high, circuit may be degraded)
        if stats.average_latency_ms > 5000.0 && stats.requests_served > 5 {
            return CircuitHealth::Failed;
        }

        CircuitHealth::Healthy
    }

    /// Get all operation types
    fn all_operation_types() -> Vec<OperationType> {
        vec![
            OperationType::BlockPropagation,
            OperationType::PeerDiscovery,
            OperationType::TransactionSubmission,
            OperationType::P2PSync,
            OperationType::ValidatorCommunication,
            OperationType::AIInference,
            OperationType::QuantumEntropy,
            OperationType::General,
        ]
    }

    /// Get current health report for all circuits
    pub async fn get_health_report(&self) -> HashMap<OperationType, CircuitHealthReport> {
        self.health_reports.read().await.clone()
    }

    /// Get health status summary
    pub async fn get_health_summary(&self) -> CircuitHealthSummary {
        let reports = self.health_reports.read().await;

        let mut healthy = 0;
        let mut expiring = 0;
        let mut failed = 0;
        let mut not_created = 0;

        for report in reports.values() {
            match report.health {
                CircuitHealth::Healthy => healthy += 1,
                CircuitHealth::ExpiringSoon => expiring += 1,
                CircuitHealth::Failed => failed += 1,
                CircuitHealth::NotCreated => not_created += 1,
            }
        }

        CircuitHealthSummary {
            total: reports.len(),
            healthy,
            expiring_soon: expiring,
            failed,
            not_created,
            overall_health: if failed > 0 {
                OverallHealth::Degraded
            } else if expiring > healthy {
                OverallHealth::Warning
            } else if healthy == 0 && not_created > 0 {
                OverallHealth::Initializing
            } else {
                OverallHealth::Good
            },
        }
    }

    /// Get prewarming statistics
    pub async fn get_stats(&self) -> PrewarmingStats {
        self.stats.read().await.clone()
    }

    /// Force prewarm all circuits
    pub async fn force_prewarm_all(&self) -> Result<()> {
        info!("🔥 Force prewarming all circuits");

        for op_type in Self::all_operation_types() {
            if let Err(e) = self.prewarm_circuit(op_type).await {
                warn!("⚠️ Failed to force prewarm {}: {}", op_type.name(), e);
            }
        }

        Ok(())
    }

    /// Shutdown the prewarming manager
    pub async fn shutdown(&self) -> Result<()> {
        info!("🛑 Shutting down Circuit Prewarming Manager");

        // Signal shutdown
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for tasks to complete
        let mut handles = self.task_handles.write().await;
        for handle in handles.drain(..) {
            if let Err(e) = handle.await {
                warn!("⚠️ Task shutdown error: {}", e);
            }
        }

        info!("✅ Circuit Prewarming Manager shut down");
        Ok(())
    }

    /// Check if prewarming manager is running
    pub fn is_running(&self) -> bool {
        !self.shutdown.load(Ordering::Relaxed)
    }
}

/// Circuit health summary
#[derive(Debug, Clone)]
pub struct CircuitHealthSummary {
    pub total: usize,
    pub healthy: usize,
    pub expiring_soon: usize,
    pub failed: usize,
    pub not_created: usize,
    pub overall_health: OverallHealth,
}

/// Overall health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverallHealth {
    Good,
    Warning,
    Degraded,
    Initializing,
}

impl std::fmt::Display for OverallHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OverallHealth::Good => write!(f, "Good"),
            OverallHealth::Warning => write!(f, "Warning"),
            OverallHealth::Degraded => write!(f, "Degraded"),
            OverallHealth::Initializing => write!(f, "Initializing"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prewarming_config_defaults() {
        let config = PrewarmingConfig::default();
        assert_eq!(config.prewarm_ahead, Duration::from_secs(60));
        assert_eq!(config.health_check_interval, Duration::from_secs(30));
        assert_eq!(config.max_consecutive_failures, 3);
        assert!(config.adaptive_prewarming);
    }

    #[test]
    fn test_circuit_health_display() {
        assert_eq!(format!("{}", OverallHealth::Good), "Good");
        assert_eq!(format!("{}", OverallHealth::Degraded), "Degraded");
    }

    #[test]
    fn test_all_operation_types() {
        let types = CircuitPrewarmingManager::all_operation_types();
        assert_eq!(types.len(), 8);
        assert!(types.contains(&OperationType::BlockPropagation));
        assert!(types.contains(&OperationType::ValidatorCommunication));
    }
}
