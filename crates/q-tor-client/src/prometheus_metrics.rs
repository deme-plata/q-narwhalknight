/// Prometheus metrics integration for Q-Tor-Client
/// Provides comprehensive monitoring of Tor circuits, performance, and privacy metrics
use anyhow::{Context, Result};
use prometheus::{
    register_counter, register_counter_vec, register_gauge, register_gauge_vec, register_histogram,
    register_histogram_vec, Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramVec,
    Registry,
};
use serde::{Deserialize, Serialize};
use std::{
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::{
    dandelion::DandelionStatistics,
    quantum_seeding::{EntropyQuality, RandomnessTest},
};

/// Prometheus metrics collector for Tor operations
#[derive(Debug)]
pub struct TorPrometheusMetrics {
    /// Metrics registry
    registry: Arc<Registry>,

    // Circuit metrics
    /// Number of active Tor circuits
    active_circuits: Gauge,
    /// Circuit creation attempts
    circuit_creation_total: Counter,
    /// Circuit failures
    circuit_failures_total: CounterVec,
    /// Circuit latency histogram
    circuit_latency_seconds: HistogramVec,

    // Connection metrics
    /// Total connections made
    connections_total: Counter,
    /// Connection failures
    connection_failures_total: CounterVec,
    /// Connection duration
    connection_duration_seconds: Histogram,
    /// Bytes sent/received
    bytes_transferred_total: CounterVec,

    // Dandelion++ metrics
    /// Dandelion++ transactions
    dandelion_transactions_total: CounterVec,
    /// Stem phase forwards
    dandelion_stem_forwards_total: Counter,
    /// Fluff phase broadcasts
    dandelion_fluff_broadcasts_total: Counter,
    /// Stem to fluff transitions
    dandelion_stem_to_fluff_total: Counter,

    // Quantum entropy metrics
    /// Entropy quality score
    entropy_quality_ratio: Gauge,
    /// Quantum tests passed/failed
    quantum_tests_total: CounterVec,
    /// Entropy pool reseeds
    entropy_reseeds_total: Counter,
    /// QRNG device status
    qrng_device_status: GaugeVec,

    // Performance metrics
    /// Average latency gauge
    average_latency_seconds: Gauge,
    /// Circuit rotation events
    circuit_rotations_total: Counter,
    /// Onion service status
    onion_service_status: Gauge,

    // Privacy metrics
    /// Anonymity score (0.0 - 1.0)
    anonymity_score: Gauge,
    /// Traffic analysis resistance score
    traffic_analysis_resistance: Gauge,
    /// Circuit diversity (unique paths)
    circuit_diversity_ratio: Gauge,

    /// Last update timestamp
    last_update: Arc<RwLock<Instant>>,
}

/// Configuration for Prometheus metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusConfig {
    /// Enable metrics collection
    pub enabled: bool,
    /// Metrics update interval
    pub update_interval: Duration,
    /// Metrics endpoint address
    pub endpoint: Option<SocketAddr>,
    /// Include sensitive metrics (circuit details, etc.)
    pub include_sensitive: bool,
    /// Metric retention period
    pub retention_period: Duration,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            update_interval: Duration::from_secs(30),
            endpoint: Some("127.0.0.1:9090".parse().unwrap()),
            include_sensitive: false,
            retention_period: Duration::from_secs(3600), // 1 hour
        }
    }
}

impl TorPrometheusMetrics {
    /// Create a new Prometheus metrics collector
    pub fn new(config: PrometheusConfig) -> Result<Self> {
        if !config.enabled {
            info!("Prometheus metrics disabled by configuration");
        }

        let registry = Arc::new(Registry::new());

        // Circuit metrics
        let active_circuits =
            register_gauge!("q_tor_active_circuits", "Number of active Tor circuits")?;

        let circuit_creation_total = register_counter!(
            "q_tor_circuit_creation_total",
            "Total number of circuit creation attempts"
        )?;

        let circuit_failures_total = register_counter_vec!(
            "q_tor_circuit_failures_total",
            "Total number of circuit failures by type",
            &["failure_type"]
        )?;

        let circuit_latency_seconds = register_histogram_vec!(
            "q_tor_circuit_latency_seconds",
            "Circuit latency in seconds",
            &["circuit_type"],
            vec![0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        )?;

        // Connection metrics
        let connections_total = register_counter!(
            "q_tor_connections_total",
            "Total number of connections made"
        )?;

        let connection_failures_total = register_counter_vec!(
            "q_tor_connection_failures_total",
            "Total number of connection failures by type",
            &["failure_type"]
        )?;

        let connection_duration_seconds = register_histogram!(
            "q_tor_connection_duration_seconds",
            "Connection duration in seconds",
            vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]
        )?;

        let bytes_transferred_total = register_counter_vec!(
            "q_tor_bytes_transferred_total",
            "Total bytes transferred",
            &["direction"] // sent, received
        )?;

        // Dandelion++ metrics
        let dandelion_transactions_total = register_counter_vec!(
            "q_tor_dandelion_transactions_total",
            "Total Dandelion++ transactions by type",
            &["transaction_type"] // started, received
        )?;

        let dandelion_stem_forwards_total = register_counter!(
            "q_tor_dandelion_stem_forwards_total",
            "Total Dandelion++ stem phase forwards"
        )?;

        let dandelion_fluff_broadcasts_total = register_counter!(
            "q_tor_dandelion_fluff_broadcasts_total",
            "Total Dandelion++ fluff phase broadcasts"
        )?;

        let dandelion_stem_to_fluff_total = register_counter!(
            "q_tor_dandelion_stem_to_fluff_total",
            "Total Dandelion++ stem to fluff transitions"
        )?;

        // Quantum entropy metrics
        let entropy_quality_ratio = register_gauge!(
            "q_tor_entropy_quality_ratio",
            "Current entropy quality ratio (0.0 - 1.0)"
        )?;

        let quantum_tests_total = register_counter_vec!(
            "q_tor_quantum_tests_total",
            "Total quantum randomness tests by result",
            &["result"] // passed, failed
        )?;

        let entropy_reseeds_total =
            register_counter!("q_tor_entropy_reseeds_total", "Total entropy pool reseeds")?;

        let qrng_device_status = register_gauge_vec!(
            "q_tor_qrng_device_status",
            "QRNG device status (1=active, 0=inactive)",
            &["device"] // primary, backup
        )?;

        // Performance metrics
        let average_latency_seconds = register_gauge!(
            "q_tor_average_latency_seconds",
            "Average connection latency in seconds"
        )?;

        let circuit_rotations_total =
            register_counter!("q_tor_circuit_rotations_total", "Total circuit rotations")?;

        let onion_service_status = register_gauge!(
            "q_tor_onion_service_status",
            "Onion service status (1=active, 0=inactive)"
        )?;

        // Privacy metrics
        let anonymity_score = register_gauge!(
            "q_tor_anonymity_score",
            "Current anonymity score (0.0 - 1.0)"
        )?;

        let traffic_analysis_resistance = register_gauge!(
            "q_tor_traffic_analysis_resistance",
            "Traffic analysis resistance score (0.0 - 1.0)"
        )?;

        let circuit_diversity_ratio = register_gauge!(
            "q_tor_circuit_diversity_ratio",
            "Circuit path diversity ratio (0.0 - 1.0)"
        )?;

        let metrics = Self {
            registry,
            active_circuits,
            circuit_creation_total,
            circuit_failures_total,
            circuit_latency_seconds,
            connections_total,
            connection_failures_total,
            connection_duration_seconds,
            bytes_transferred_total,
            dandelion_transactions_total,
            dandelion_stem_forwards_total,
            dandelion_fluff_broadcasts_total,
            dandelion_stem_to_fluff_total,
            entropy_quality_ratio,
            quantum_tests_total,
            entropy_reseeds_total,
            qrng_device_status,
            average_latency_seconds,
            circuit_rotations_total,
            onion_service_status,
            anonymity_score,
            traffic_analysis_resistance,
            circuit_diversity_ratio,
            last_update: Arc::new(RwLock::new(Instant::now())),
        };

        info!("✅ Tor Prometheus metrics initialized");
        Ok(metrics)
    }

    /// Update circuit metrics
    pub async fn update_circuit_metrics(&self, active_count: usize, creation_attempts: u64) {
        self.active_circuits.set(active_count as f64);

        // Only increment if there were new attempts
        let current_total = self.circuit_creation_total.get();
        if creation_attempts > current_total as u64 {
            let new_attempts = creation_attempts - current_total as u64;
            for _ in 0..new_attempts {
                self.circuit_creation_total.inc();
            }
        }

        debug!(
            "📊 Updated circuit metrics: {} active, {} total attempts",
            active_count, creation_attempts
        );
    }

    /// Record circuit latency
    pub async fn record_circuit_latency(&self, circuit_type: &str, latency: Duration) {
        self.circuit_latency_seconds
            .with_label_values(&[circuit_type])
            .observe(latency.as_secs_f64());

        debug!(
            "📊 Recorded circuit latency: {} = {}ms",
            circuit_type,
            latency.as_millis()
        );
    }

    /// Record circuit failure
    pub async fn record_circuit_failure(&self, failure_type: &str) {
        self.circuit_failures_total
            .with_label_values(&[failure_type])
            .inc();

        warn!("📊 Recorded circuit failure: {}", failure_type);
    }

    /// Update connection metrics
    pub async fn update_connection_metrics(
        &self,
        total_connections: u64,
        bytes_sent: u64,
        bytes_received: u64,
    ) {
        // Update counters (only increment new values)
        let current_connections = self.connections_total.get();
        if total_connections > current_connections as u64 {
            let new_connections = total_connections - current_connections as u64;
            for _ in 0..new_connections {
                self.connections_total.inc();
            }
        }

        // Update bytes transferred
        self.bytes_transferred_total
            .with_label_values(&["sent"])
            .inc_by(bytes_sent as f64);
        self.bytes_transferred_total
            .with_label_values(&["received"])
            .inc_by(bytes_received as f64);

        debug!(
            "📊 Updated connection metrics: {} connections, {}B sent, {}B received",
            total_connections, bytes_sent, bytes_received
        );
    }

    /// Record connection duration
    pub async fn record_connection_duration(&self, duration: Duration) {
        self.connection_duration_seconds
            .observe(duration.as_secs_f64());
    }

    /// Record connection failure
    pub async fn record_connection_failure(&self, failure_type: &str) {
        self.connection_failures_total
            .with_label_values(&[failure_type])
            .inc();
    }

    /// Update Dandelion++ metrics
    pub async fn update_dandelion_metrics(&self, stats: &DandelionStatistics) {
        // Update counters (increment new values only)
        let started_current = self
            .dandelion_transactions_total
            .with_label_values(&["started"])
            .get();
        let received_current = self
            .dandelion_transactions_total
            .with_label_values(&["received"])
            .get();
        let forwards_current = self.dandelion_stem_forwards_total.get();
        let broadcasts_current = self.dandelion_fluff_broadcasts_total.get();
        let transitions_current = self.dandelion_stem_to_fluff_total.get();

        // Increment only new values
        if stats.transactions_started > started_current as u64 {
            let new_started = stats.transactions_started - started_current as u64;
            for _ in 0..new_started {
                self.dandelion_transactions_total
                    .with_label_values(&["started"])
                    .inc();
            }
        }

        if stats.transactions_received > received_current as u64 {
            let new_received = stats.transactions_received - received_current as u64;
            for _ in 0..new_received {
                self.dandelion_transactions_total
                    .with_label_values(&["received"])
                    .inc();
            }
        }

        if stats.stem_forwards > forwards_current as u64 {
            let new_forwards = stats.stem_forwards - forwards_current as u64;
            for _ in 0..new_forwards {
                self.dandelion_stem_forwards_total.inc();
            }
        }

        if stats.fluff_broadcasts > broadcasts_current as u64 {
            let new_broadcasts = stats.fluff_broadcasts - broadcasts_current as u64;
            for _ in 0..new_broadcasts {
                self.dandelion_fluff_broadcasts_total.inc();
            }
        }

        debug!(
            "📊 Updated Dandelion++ metrics: {} started, {} received, {} forwards",
            stats.transactions_started, stats.transactions_received, stats.stem_forwards
        );
    }

    /// Update quantum entropy metrics
    pub async fn update_entropy_metrics(&self, quality: &EntropyQuality) {
        self.entropy_quality_ratio.set(quality.overall_score);

        // Update test counters
        self.quantum_tests_total
            .with_label_values(&["passed"])
            .inc_by(quality.tests_passed as f64);
        self.quantum_tests_total
            .with_label_values(&["failed"])
            .inc_by(quality.tests_failed as f64);

        // Update device status
        if quality.primary_quality > 0.9 {
            self.qrng_device_status
                .with_label_values(&["primary"])
                .set(1.0);
        } else {
            self.qrng_device_status
                .with_label_values(&["primary"])
                .set(0.0);
        }

        if let Some(backup_quality) = quality.backup_quality {
            if backup_quality > 0.9 {
                self.qrng_device_status
                    .with_label_values(&["backup"])
                    .set(1.0);
            } else {
                self.qrng_device_status
                    .with_label_values(&["backup"])
                    .set(0.0);
            }
        }

        debug!(
            "📊 Updated entropy metrics: quality={:.3}, tests_passed={}, tests_failed={}",
            quality.overall_score, quality.tests_passed, quality.tests_failed
        );
    }

    /// Record entropy reseed event
    pub async fn record_entropy_reseed(&self) {
        self.entropy_reseeds_total.inc();
        info!("📊 Recorded entropy reseed event");
    }

    /// Update performance metrics
    pub async fn update_performance_metrics(&self, avg_latency: Duration, rotations: u64) {
        self.average_latency_seconds.set(avg_latency.as_secs_f64());

        // Only increment if there were new rotations
        let current_rotations = self.circuit_rotations_total.get();
        if rotations > current_rotations as u64 {
            let new_rotations = rotations - current_rotations as u64;
            for _ in 0..new_rotations {
                self.circuit_rotations_total.inc();
            }
        }

        debug!(
            "📊 Updated performance metrics: {}ms avg latency, {} rotations",
            avg_latency.as_millis(),
            rotations
        );
    }

    /// Update onion service status
    pub async fn update_onion_service_status(&self, active: bool) {
        self.onion_service_status
            .set(if active { 1.0 } else { 0.0 });
    }

    /// Update privacy metrics
    pub async fn update_privacy_metrics(
        &self,
        anonymity: f64,
        traffic_resistance: f64,
        diversity: f64,
    ) {
        self.anonymity_score.set(anonymity.clamp(0.0, 1.0));
        self.traffic_analysis_resistance
            .set(traffic_resistance.clamp(0.0, 1.0));
        self.circuit_diversity_ratio.set(diversity.clamp(0.0, 1.0));

        debug!("📊 Updated privacy metrics: anonymity={:.3}, traffic_resistance={:.3}, diversity={:.3}", 
               anonymity, traffic_resistance, diversity);
    }

    /// Record randomness test results
    pub async fn record_randomness_test(&self, test: &RandomnessTest) {
        if test.quality_score >= 0.9 {
            self.quantum_tests_total
                .with_label_values(&["passed"])
                .inc();
        } else {
            self.quantum_tests_total
                .with_label_values(&["failed"])
                .inc();
        }

        info!(
            "📊 Recorded randomness test: quality={:.3}, passed_tests={}/{}",
            test.quality_score, test.passed_tests, test.total_tests
        );
    }

    /// Get metrics in Prometheus format
    pub async fn get_metrics(&self) -> Result<String> {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();

        encoder
            .encode_to_string(&metric_families)
            .context("Failed to encode metrics")
    }

    /// Get current metrics summary
    pub async fn get_metrics_summary(&self) -> MetricsSummary {
        MetricsSummary {
            active_circuits: self.active_circuits.get() as u64,
            total_connections: self.connections_total.get() as u64,
            entropy_quality: self.entropy_quality_ratio.get(),
            average_latency_ms: (self.average_latency_seconds.get() * 1000.0) as u64,
            anonymity_score: self.anonymity_score.get(),
            dandelion_transactions: self
                .dandelion_transactions_total
                .with_label_values(&["started"])
                .get() as u64,
            circuit_failures: self
                .circuit_failures_total
                .with_label_values(&["timeout"])
                .get() as u64,
            last_update: SystemTime::now(),
        }
    }

    /// Update last metrics timestamp
    pub async fn mark_updated(&self) {
        let mut last_update = self.last_update.write().await;
        *last_update = Instant::now();
    }

    /// Reset all metrics (for testing)
    #[cfg(test)]
    pub async fn reset_metrics(&self) {
        use prometheus::core::Metric;

        // Reset counters and gauges to 0
        self.active_circuits.set(0.0);
        self.entropy_quality_ratio.set(0.0);
        self.average_latency_seconds.set(0.0);
        self.anonymity_score.set(0.0);

        info!("🧪 Reset all Prometheus metrics for testing");
    }

    /// Start metrics HTTP endpoint (if configured)
    pub async fn start_metrics_server(&self, config: PrometheusConfig) -> Result<()> {
        if !config.enabled || config.endpoint.is_none() {
            return Ok(());
        }

        let endpoint = config.endpoint.unwrap();
        info!("🚀 Starting Tor metrics server on {}", endpoint);

        // This would start an HTTP server in production
        // For now, just log the configuration
        debug!(
            "Metrics configuration: update_interval={:?}, include_sensitive={}",
            config.update_interval, config.include_sensitive
        );

        Ok(())
    }
}

/// Summary of key metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub active_circuits: u64,
    pub total_connections: u64,
    pub entropy_quality: f64,
    pub average_latency_ms: u64,
    pub anonymity_score: f64,
    pub dandelion_transactions: u64,
    pub circuit_failures: u64,
    pub last_update: SystemTime,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_creation() {
        let config = PrometheusConfig::default();
        let metrics = TorPrometheusMetrics::new(config).unwrap();

        // Test basic metric updates
        metrics.update_circuit_metrics(4, 10).await;
        metrics.update_privacy_metrics(0.95, 0.89, 0.72).await;

        let summary = metrics.get_metrics_summary().await;
        assert_eq!(summary.active_circuits, 4);
        assert_eq!(summary.anonymity_score, 0.95);
    }

    #[tokio::test]
    async fn test_prometheus_export() {
        let config = PrometheusConfig::default();
        let metrics = TorPrometheusMetrics::new(config).unwrap();

        // Add some test data
        metrics.update_circuit_metrics(2, 5).await;
        metrics.entropy_quality_ratio.set(0.97);

        let exported = metrics.get_metrics().await.unwrap();
        assert!(exported.contains("q_tor_active_circuits"));
        assert!(exported.contains("q_tor_entropy_quality_ratio"));
    }

    #[test]
    fn test_metrics_config() {
        let config = PrometheusConfig::default();
        assert!(config.enabled);
        assert!(config.endpoint.is_some());
        assert!(!config.include_sensitive);
    }
}
