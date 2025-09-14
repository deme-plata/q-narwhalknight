//! Performance Regression Detection Module

use crate::PerformanceMetrics;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAlert {
    pub metric_name: String,
    pub current_value: f64,
    pub baseline_value: f64,
    pub regression_percentage: f64,
    pub severity: AlertSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

pub struct RegressionDetector {
    baseline_metrics: Option<PerformanceMetrics>,
    metrics_history: VecDeque<PerformanceMetrics>,
    max_history_size: usize,
}

impl RegressionDetector {
    pub fn new() -> Self {
        Self {
            baseline_metrics: None,
            metrics_history: VecDeque::new(),
            max_history_size: 100,
        }
    }

    pub fn set_baseline(&mut self, metrics: PerformanceMetrics) {
        info!("📊 Setting performance baseline");
        self.baseline_metrics = Some(metrics);
    }

    pub fn add_measurement(&mut self, metrics: PerformanceMetrics) -> Vec<RegressionAlert> {
        self.metrics_history.push_back(metrics.clone());

        if self.metrics_history.len() > self.max_history_size {
            self.metrics_history.pop_front();
        }

        self.detect_regressions(&metrics)
    }

    fn detect_regressions(&self, current: &PerformanceMetrics) -> Vec<RegressionAlert> {
        let mut alerts = Vec::new();

        if let Some(baseline) = &self.baseline_metrics {
            // TPS regression
            if current.tps_metrics.transactions_per_second
                < baseline.tps_metrics.transactions_per_second * 0.9
            {
                alerts.push(RegressionAlert {
                    metric_name: "TPS".to_string(),
                    current_value: current.tps_metrics.transactions_per_second,
                    baseline_value: baseline.tps_metrics.transactions_per_second,
                    regression_percentage: (1.0
                        - current.tps_metrics.transactions_per_second
                            / baseline.tps_metrics.transactions_per_second)
                        * 100.0,
                    severity: AlertSeverity::Critical,
                });
            }

            // Memory regression
            if current.memory_metrics.peak_memory_mb > baseline.memory_metrics.peak_memory_mb * 1.2
            {
                alerts.push(RegressionAlert {
                    metric_name: "Memory Usage".to_string(),
                    current_value: current.memory_metrics.peak_memory_mb,
                    baseline_value: baseline.memory_metrics.peak_memory_mb,
                    regression_percentage: (current.memory_metrics.peak_memory_mb
                        / baseline.memory_metrics.peak_memory_mb
                        - 1.0)
                        * 100.0,
                    severity: AlertSeverity::Warning,
                });
            }
        }

        if !alerts.is_empty() {
            warn!(
                "⚠️ Performance regressions detected: {} alerts",
                alerts.len()
            );
        }

        alerts
    }
}
