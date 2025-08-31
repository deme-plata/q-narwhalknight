use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use tracing::debug;

/// Tor performance and usage metrics
#[derive(Debug)]
pub struct TorMetrics {
    /// Connection latency history (last 100 connections)
    latency_history: Arc<RwLock<VecDeque<Duration>>>,
    /// Total connections made
    connection_count: Arc<RwLock<u64>>,
    /// Total bytes sent through Tor
    bytes_sent: Arc<RwLock<u64>>,
    /// Total bytes received through Tor
    bytes_received: Arc<RwLock<u64>>,
    /// Circuit failure count
    circuit_failures: Arc<RwLock<u64>>,
    /// Last metrics update
    last_update: Arc<RwLock<Instant>>,
}

impl TorMetrics {
    /// Create new metrics tracker
    pub fn new() -> Self {
        Self {
            latency_history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            connection_count: Arc::new(RwLock::new(0)),
            bytes_sent: Arc::new(RwLock::new(0)),
            bytes_received: Arc::new(RwLock::new(0)),
            circuit_failures: Arc::new(RwLock::new(0)),
            last_update: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Record connection latency
    pub async fn record_connection_latency(&self, latency: Duration) {
        debug!("📊 Recording Tor connection latency: {}ms", latency.as_millis());
        
        let mut history = self.latency_history.write().await;
        
        // Keep only last 100 measurements
        if history.len() >= 100 {
            history.pop_front();
        }
        
        history.push_back(latency);
        
        // Update connection count
        {
            let mut count = self.connection_count.write().await;
            *count += 1;
        }

        // Update last update time
        {
            let mut last_update = self.last_update.write().await;
            *last_update = Instant::now();
        }
    }

    /// Record bytes sent
    pub async fn record_bytes_sent(&self, bytes: u64) {
        let mut total = self.bytes_sent.write().await;
        *total += bytes;
        
        debug!("📤 Recorded {} bytes sent via Tor (total: {})", bytes, *total);
    }

    /// Record bytes received
    pub async fn record_bytes_received(&self, bytes: u64) {
        let mut total = self.bytes_received.write().await;
        *total += bytes;
        
        debug!("📥 Recorded {} bytes received via Tor (total: {})", bytes, *total);
    }

    /// Record circuit failure
    pub async fn record_circuit_failure(&self) {
        let mut failures = self.circuit_failures.write().await;
        *failures += 1;
        
        debug!("❌ Recorded Tor circuit failure (total: {})", *failures);
    }

    /// Get current metrics snapshot
    pub async fn get_current_metrics(&self) -> TorMetricsSnapshot {
        let latency_history = self.latency_history.read().await;
        let connection_count = *self.connection_count.read().await;
        let bytes_sent = *self.bytes_sent.read().await;
        let bytes_received = *self.bytes_received.read().await;
        let circuit_failures = *self.circuit_failures.read().await;
        let last_update = *self.last_update.read().await;

        // Calculate average latency
        let average_latency = if latency_history.is_empty() {
            Duration::from_millis(0)
        } else {
            let total_ms: u64 = latency_history.iter()
                .map(|d| d.as_millis() as u64)
                .sum();
            Duration::from_millis(total_ms / latency_history.len() as u64)
        };

        // Calculate 95th percentile latency
        let mut latencies: Vec<u64> = latency_history.iter()
            .map(|d| d.as_millis() as u64)
            .collect();
        latencies.sort();
        
        let p95_latency = if latencies.is_empty() {
            Duration::from_millis(0)
        } else {
            let index = (latencies.len() as f64 * 0.95) as usize;
            let index = index.min(latencies.len() - 1);
            Duration::from_millis(latencies[index])
        };

        TorMetricsSnapshot {
            connection_count,
            average_latency,
            p95_latency,
            bytes_sent,
            bytes_received,
            circuit_failures,
            success_rate: if connection_count > 0 {
                1.0 - (circuit_failures as f64 / connection_count as f64)
            } else {
                1.0
            },
            last_update,
        }
    }

    /// Get Prometheus-format metrics
    pub async fn get_prometheus_metrics(&self) -> String {
        let metrics = self.get_current_metrics().await;
        
        format!(
            "# HELP tor_connections_total Total Tor connections made\n\
             # TYPE tor_connections_total counter\n\
             tor_connections_total {}\n\
             \n\
             # HELP tor_latency_ms_avg Average Tor connection latency in milliseconds\n\
             # TYPE tor_latency_ms_avg gauge\n\
             tor_latency_ms_avg {}\n\
             \n\
             # HELP tor_latency_ms_p95 95th percentile Tor connection latency in milliseconds\n\
             # TYPE tor_latency_ms_p95 gauge\n\
             tor_latency_ms_p95 {}\n\
             \n\
             # HELP tor_bytes_sent_total Total bytes sent through Tor\n\
             # TYPE tor_bytes_sent_total counter\n\
             tor_bytes_sent_total {}\n\
             \n\
             # HELP tor_bytes_received_total Total bytes received through Tor\n\
             # TYPE tor_bytes_received_total counter\n\
             tor_bytes_received_total {}\n\
             \n\
             # HELP tor_circuit_failures_total Total Tor circuit failures\n\
             # TYPE tor_circuit_failures_total counter\n\
             tor_circuit_failures_total {}\n\
             \n\
             # HELP tor_success_rate Tor connection success rate (0-1)\n\
             # TYPE tor_success_rate gauge\n\
             tor_success_rate {}\n",
            metrics.connection_count,
            metrics.average_latency.as_millis(),
            metrics.p95_latency.as_millis(),
            metrics.bytes_sent,
            metrics.bytes_received,
            metrics.circuit_failures,
            metrics.success_rate
        )
    }

    /// Reset all metrics
    pub async fn reset_metrics(&self) {
        let mut latency_history = self.latency_history.write().await;
        let mut connection_count = self.connection_count.write().await;
        let mut bytes_sent = self.bytes_sent.write().await;
        let mut bytes_received = self.bytes_received.write().await;
        let mut circuit_failures = self.circuit_failures.write().await;

        latency_history.clear();
        *connection_count = 0;
        *bytes_sent = 0;
        *bytes_received = 0;
        *circuit_failures = 0;

        debug!("🔄 Tor metrics reset");
    }

    /// Check if metrics indicate performance issues
    pub async fn check_performance_health(&self) -> TorHealthStatus {
        let metrics = self.get_current_metrics().await;

        // Check various health indicators
        let latency_ok = metrics.average_latency < Duration::from_millis(500);
        let success_rate_ok = metrics.success_rate > 0.95;
        let recent_activity = metrics.last_update.elapsed() < Duration::from_secs(60);

        if latency_ok && success_rate_ok && recent_activity {
            TorHealthStatus::Healthy
        } else if !latency_ok && metrics.average_latency > Duration::from_millis(1000) {
            TorHealthStatus::HighLatency
        } else if !success_rate_ok {
            TorHealthStatus::CircuitIssues
        } else {
            TorHealthStatus::Degraded
        }
    }
}

/// Snapshot of Tor metrics at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorMetricsSnapshot {
    pub connection_count: u64,
    pub average_latency: Duration,
    pub p95_latency: Duration,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub circuit_failures: u64,
    pub success_rate: f64,
    pub last_update: Instant,
}

/// Onion service information for network discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnionServiceInfo {
    pub onion_name: String,
    pub onion_address: String,
    pub port: u16,
    pub service_type: String,
    pub version: String,
}

/// Tor health status
#[derive(Debug, Clone, PartialEq)]
pub enum TorHealthStatus {
    Healthy,
    Degraded,
    HighLatency,
    CircuitIssues,
    Offline,
}

impl TorHealthStatus {
    /// Convert to human-readable status
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::HighLatency => "high_latency",
            Self::CircuitIssues => "circuit_issues",
            Self::Offline => "offline",
        }
    }

    /// Check if status requires action
    pub fn requires_action(&self) -> bool {
        matches!(self, Self::HighLatency | Self::CircuitIssues | Self::Offline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_tracking() {
        let metrics = TorMetrics::new();
        
        // Record some test data
        metrics.record_connection_latency(Duration::from_millis(100)).await;
        metrics.record_connection_latency(Duration::from_millis(200)).await;
        metrics.record_bytes_sent(1024).await;
        metrics.record_bytes_received(2048).await;

        let snapshot = metrics.get_current_metrics().await;
        
        assert_eq!(snapshot.connection_count, 2);
        assert_eq!(snapshot.average_latency, Duration::from_millis(150));
        assert_eq!(snapshot.bytes_sent, 1024);
        assert_eq!(snapshot.bytes_received, 2048);
        assert_eq!(snapshot.success_rate, 1.0); // No failures recorded
    }

    #[tokio::test]
    async fn test_health_status() {
        let metrics = TorMetrics::new();
        
        // Test healthy status
        metrics.record_connection_latency(Duration::from_millis(100)).await;
        let health = metrics.check_performance_health().await;
        assert_eq!(health, TorHealthStatus::Healthy);
        
        // Test high latency status
        for _ in 0..10 {
            metrics.record_connection_latency(Duration::from_millis(1500)).await;
        }
        let health = metrics.check_performance_health().await;
        assert_eq!(health, TorHealthStatus::HighLatency);
    }

    #[test]
    fn test_health_status_helpers() {
        assert_eq!(TorHealthStatus::Healthy.as_str(), "healthy");
        assert!(!TorHealthStatus::Healthy.requires_action());
        
        assert_eq!(TorHealthStatus::CircuitIssues.as_str(), "circuit_issues");
        assert!(TorHealthStatus::CircuitIssues.requires_action());
    }
}