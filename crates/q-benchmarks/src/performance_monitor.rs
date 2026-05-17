//! Real-time Performance Monitoring Module

use crate::PerformanceMetrics;
use anyhow::Result;
use std::time::{Duration, Instant};
use tokio::time::interval;
use tracing::{debug, info};

pub struct PerformanceMonitor {
    monitoring_interval: Duration,
    last_metrics: Option<PerformanceMetrics>,
}

impl PerformanceMonitor {
    pub fn new(interval_seconds: u64) -> Self {
        Self {
            monitoring_interval: Duration::from_secs(interval_seconds),
            last_metrics: None,
        }
    }

    pub async fn start_monitoring<F>(&mut self, mut metrics_collector: F) -> Result<()>
    where
        F: FnMut() -> Result<PerformanceMetrics> + Send,
    {
        info!("📊 Starting real-time performance monitoring");
        let mut interval = interval(self.monitoring_interval);

        loop {
            interval.tick().await;

            match metrics_collector() {
                Ok(metrics) => {
                    self.analyze_metrics(&metrics).await;
                    self.last_metrics = Some(metrics);
                }
                Err(e) => {
                    debug!("Failed to collect metrics: {}", e);
                }
            }
        }
    }

    async fn analyze_metrics(&self, current: &PerformanceMetrics) {
        if let Some(previous) = &self.last_metrics {
            let tps_change = current.tps_metrics.transactions_per_second
                - previous.tps_metrics.transactions_per_second;
            let memory_change =
                current.memory_metrics.heap_usage_mb - previous.memory_metrics.heap_usage_mb;

            debug!(
                "Performance delta - TPS: {:.0}, Memory: {:.1}MB",
                tps_change, memory_change
            );
        }
    }
}
