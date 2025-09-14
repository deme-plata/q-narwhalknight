//! Performance Monitoring for GPU STARK Operations
//!
//! Comprehensive performance monitoring system to track GPU STARK proving
//! performance and validate Phase 3 targets from Server Beta analysis.

use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Performance monitoring system for GPU STARK operations
pub struct PerformanceMonitor {
    proving_metrics: VecDeque<ProvingMetric>,
    verification_metrics: VecDeque<VerificationMetric>,
    gpu_utilization: VecDeque<GpuUtilization>,
    performance_targets: Phase3Targets,
    max_history_size: usize,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self {
            proving_metrics: VecDeque::new(),
            verification_metrics: VecDeque::new(),
            gpu_utilization: VecDeque::new(),
            performance_targets: Phase3Targets::default(),
            max_history_size: 1000, // Keep last 1000 measurements
        }
    }

    /// Record STARK proving performance
    pub fn record_proving_time(&mut self, trace_size: usize, duration: Duration) {
        let metric = ProvingMetric {
            timestamp: Instant::now(),
            trace_size,
            proving_time: duration,
            constraints_per_second: if duration.as_secs() > 0 {
                trace_size as f64 / duration.as_secs_f64()
            } else {
                0.0
            },
            meets_targets: self.check_proving_targets(trace_size, duration),
        };

        self.proving_metrics.push_back(metric);
        if self.proving_metrics.len() > self.max_history_size {
            self.proving_metrics.pop_front();
        }
    }

    /// Record STARK verification performance
    pub fn record_verification_time(&mut self, proof_size: usize, duration: Duration) {
        let metric = VerificationMetric {
            timestamp: Instant::now(),
            proof_size,
            verification_time: duration,
            meets_targets: duration <= self.performance_targets.max_verification_time,
        };

        self.verification_metrics.push_back(metric);
        if self.verification_metrics.len() > self.max_history_size {
            self.verification_metrics.pop_front();
        }
    }

    /// Record GPU utilization metrics
    pub fn record_gpu_utilization(
        &mut self,
        compute_usage: f32,
        memory_usage: f32,
        temperature: f32,
    ) {
        let utilization = GpuUtilization {
            timestamp: Instant::now(),
            compute_usage_percent: compute_usage,
            memory_usage_percent: memory_usage,
            temperature_celsius: temperature,
            efficiency_score: self.calculate_efficiency_score(compute_usage, memory_usage),
        };

        self.gpu_utilization.push_back(utilization);
        if self.gpu_utilization.len() > self.max_history_size {
            self.gpu_utilization.pop_front();
        }
    }

    /// Get comprehensive performance report
    pub fn generate_performance_report(&self) -> PerformanceReport {
        let proving_stats = self.analyze_proving_performance();
        let verification_stats = self.analyze_verification_performance();
        let gpu_stats = self.analyze_gpu_performance();

        PerformanceReport {
            proving_performance: proving_stats,
            verification_performance: verification_stats,
            gpu_performance: gpu_stats,
            phase3_compliance: self.assess_phase3_compliance(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Get real-time performance dashboard data
    pub fn get_dashboard_data(&self) -> PerformanceDashboard {
        let recent_proving = self.proving_metrics.back().cloned();
        let recent_verification = self.verification_metrics.back().cloned();
        let recent_gpu = self.gpu_utilization.back().cloned();

        PerformanceDashboard {
            current_proving_performance: recent_proving,
            current_verification_performance: recent_verification,
            current_gpu_utilization: recent_gpu,
            targets_met: self.count_targets_met(),
            alert_level: self.determine_alert_level(),
        }
    }

    /// Monitor performance regression over time
    pub fn detect_performance_regression(&self) -> Option<PerformanceRegression> {
        if self.proving_metrics.len() < 10 {
            return None; // Need more data
        }

        let recent: Vec<_> = self.proving_metrics.iter().rev().take(5).collect();
        let baseline: Vec<_> = self.proving_metrics.iter().skip(5).take(5).collect();

        let recent_avg = recent
            .iter()
            .map(|m| m.proving_time.as_millis() as f64)
            .sum::<f64>()
            / recent.len() as f64;

        let baseline_avg = baseline
            .iter()
            .map(|m| m.proving_time.as_millis() as f64)
            .sum::<f64>()
            / baseline.len() as f64;

        let regression_threshold = 1.2; // 20% performance degradation
        if recent_avg > baseline_avg * regression_threshold {
            Some(PerformanceRegression {
                regression_factor: recent_avg / baseline_avg,
                affected_operations: "STARK Proving".to_string(),
                detection_time: Instant::now(),
                recommended_action: "GPU memory optimization or cooling needed".to_string(),
            })
        } else {
            None
        }
    }

    /// Export performance data for analysis
    pub fn export_metrics(&self, format: ExportFormat) -> Result<String> {
        match format {
            ExportFormat::Json => self.export_json(),
            ExportFormat::Csv => self.export_csv(),
            ExportFormat::Prometheus => self.export_prometheus(),
        }
    }

    // Private helper methods

    fn check_proving_targets(&self, trace_size: usize, duration: Duration) -> bool {
        let target_time = if trace_size > 100_000 {
            self.performance_targets.max_large_circuit_proving_time
        } else {
            self.performance_targets.max_standard_proving_time
        };

        duration <= target_time
    }

    fn analyze_proving_performance(&self) -> ProvingPerformanceStats {
        if self.proving_metrics.is_empty() {
            return ProvingPerformanceStats::default();
        }

        let durations: Vec<Duration> = self
            .proving_metrics
            .iter()
            .map(|m| m.proving_time)
            .collect();
        let avg_duration = Duration::from_millis(
            durations.iter().map(|d| d.as_millis()).sum::<u128>() as u64 / durations.len() as u64,
        );

        let min_duration = durations.iter().min().copied().unwrap_or_default();
        let max_duration = durations.iter().max().copied().unwrap_or_default();

        let targets_met = self
            .proving_metrics
            .iter()
            .filter(|m| m.meets_targets)
            .count();
        let success_rate = targets_met as f64 / self.proving_metrics.len() as f64 * 100.0;

        ProvingPerformanceStats {
            total_proofs_generated: self.proving_metrics.len(),
            average_proving_time: avg_duration,
            min_proving_time: min_duration,
            max_proving_time: max_duration,
            success_rate_percent: success_rate,
            average_constraints_per_second: self
                .proving_metrics
                .iter()
                .map(|m| m.constraints_per_second)
                .sum::<f64>()
                / self.proving_metrics.len() as f64,
        }
    }

    fn analyze_verification_performance(&self) -> VerificationPerformanceStats {
        if self.verification_metrics.is_empty() {
            return VerificationPerformanceStats::default();
        }

        let durations: Vec<Duration> = self
            .verification_metrics
            .iter()
            .map(|m| m.verification_time)
            .collect();
        let avg_duration = Duration::from_millis(
            durations.iter().map(|d| d.as_millis()).sum::<u128>() as u64 / durations.len() as u64,
        );

        let targets_met = self
            .verification_metrics
            .iter()
            .filter(|m| m.meets_targets)
            .count();
        let success_rate = targets_met as f64 / self.verification_metrics.len() as f64 * 100.0;

        VerificationPerformanceStats {
            total_verifications: self.verification_metrics.len(),
            average_verification_time: avg_duration,
            success_rate_percent: success_rate,
        }
    }

    fn analyze_gpu_performance(&self) -> GpuPerformanceStats {
        if self.gpu_utilization.is_empty() {
            return GpuPerformanceStats::default();
        }

        let avg_compute = self
            .gpu_utilization
            .iter()
            .map(|u| u.compute_usage_percent)
            .sum::<f32>()
            / self.gpu_utilization.len() as f32;

        let avg_memory = self
            .gpu_utilization
            .iter()
            .map(|u| u.memory_usage_percent)
            .sum::<f32>()
            / self.gpu_utilization.len() as f32;

        let avg_temperature = self
            .gpu_utilization
            .iter()
            .map(|u| u.temperature_celsius)
            .sum::<f32>()
            / self.gpu_utilization.len() as f32;

        let avg_efficiency = self
            .gpu_utilization
            .iter()
            .map(|u| u.efficiency_score)
            .sum::<f32>()
            / self.gpu_utilization.len() as f32;

        GpuPerformanceStats {
            average_compute_utilization: avg_compute,
            average_memory_utilization: avg_memory,
            average_temperature: avg_temperature,
            efficiency_score: avg_efficiency,
            thermal_throttling_detected: avg_temperature > 85.0,
        }
    }

    fn assess_phase3_compliance(&self) -> Phase3Compliance {
        let proving_compliance = self
            .proving_metrics
            .iter()
            .filter(|m| m.meets_targets)
            .count() as f64
            / self.proving_metrics.len().max(1) as f64;

        let verification_compliance = self
            .verification_metrics
            .iter()
            .filter(|m| m.meets_targets)
            .count() as f64
            / self.verification_metrics.len().max(1) as f64;

        Phase3Compliance {
            proving_targets_met: (proving_compliance * 100.0) as u32,
            verification_targets_met: (verification_compliance * 100.0) as u32,
            overall_compliance: ((proving_compliance + verification_compliance) / 2.0 * 100.0)
                as u32,
            ready_for_production: proving_compliance >= 0.95 && verification_compliance >= 0.95,
        }
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check proving performance
        if let Some(latest_proving) = self.proving_metrics.back() {
            if !latest_proving.meets_targets {
                recommendations
                    .push("Consider GPU memory optimization for proving performance".to_string());
            }
        }

        // Check GPU utilization
        if let Some(latest_gpu) = self.gpu_utilization.back() {
            if latest_gpu.temperature_celsius > 80.0 {
                recommendations.push("GPU temperature high - check cooling system".to_string());
            }
            if latest_gpu.memory_usage_percent > 85.0 {
                recommendations.push(
                    "GPU memory usage high - enable aggressive memory optimization".to_string(),
                );
            }
            if latest_gpu.compute_usage_percent < 50.0 {
                recommendations.push(
                    "GPU compute utilization low - consider increasing batch size".to_string(),
                );
            }
        }

        // Performance regression detection
        if let Some(_regression) = self.detect_performance_regression() {
            recommendations
                .push("Performance regression detected - investigate system changes".to_string());
        }

        if recommendations.is_empty() {
            recommendations
                .push("Performance is optimal - maintaining Phase 3 targets".to_string());
        }

        recommendations
    }

    fn calculate_efficiency_score(&self, compute_usage: f32, memory_usage: f32) -> f32 {
        // Efficiency = balanced compute and memory usage
        let balance_factor = 1.0 - (compute_usage - memory_usage).abs() / 100.0;
        let utilization_factor = (compute_usage + memory_usage) / 200.0;
        balance_factor * utilization_factor * 100.0
    }

    fn count_targets_met(&self) -> TargetMetrics {
        let proving_met = self
            .proving_metrics
            .iter()
            .filter(|m| m.meets_targets)
            .count();
        let verification_met = self
            .verification_metrics
            .iter()
            .filter(|m| m.meets_targets)
            .count();

        TargetMetrics {
            proving_targets_met: proving_met,
            proving_total: self.proving_metrics.len(),
            verification_targets_met: verification_met,
            verification_total: self.verification_metrics.len(),
        }
    }

    fn determine_alert_level(&self) -> AlertLevel {
        let compliance = self.assess_phase3_compliance();
        match compliance.overall_compliance {
            90..=100 => AlertLevel::Green,
            75..=89 => AlertLevel::Yellow,
            50..=74 => AlertLevel::Orange,
            _ => AlertLevel::Red,
        }
    }

    // Export methods
    fn export_json(&self) -> Result<String> {
        let report = self.generate_performance_report();
        Ok(format!("{:#?}", report)) // Simplified JSON representation
    }

    fn export_csv(&self) -> Result<String> {
        let mut csv = "timestamp,trace_size,proving_time_ms,meets_targets\n".to_string();
        for metric in &self.proving_metrics {
            csv.push_str(&format!(
                "{:?},{},{},{}\n",
                metric.timestamp,
                metric.trace_size,
                metric.proving_time.as_millis(),
                metric.meets_targets
            ));
        }
        Ok(csv)
    }

    fn export_prometheus(&self) -> Result<String> {
        let stats = self.analyze_proving_performance();
        let prometheus = format!(
            "# HELP stark_proving_time_seconds Time to generate STARK proof\n\
             # TYPE stark_proving_time_seconds gauge\n\
             stark_proving_time_seconds {:.3}\n\
             # HELP stark_success_rate_percent Percentage of proofs meeting targets\n\
             # TYPE stark_success_rate_percent gauge\n\
             stark_success_rate_percent {:.1}\n",
            stats.average_proving_time.as_secs_f64(),
            stats.success_rate_percent
        );
        Ok(prometheus)
    }
}

// Data structures for performance tracking

#[derive(Clone, Debug)]
pub struct ProvingMetric {
    pub timestamp: Instant,
    pub trace_size: usize,
    pub proving_time: Duration,
    pub constraints_per_second: f64,
    pub meets_targets: bool,
}

#[derive(Clone, Debug)]
pub struct VerificationMetric {
    pub timestamp: Instant,
    pub proof_size: usize,
    pub verification_time: Duration,
    pub meets_targets: bool,
}

#[derive(Clone, Debug)]
pub struct GpuUtilization {
    pub timestamp: Instant,
    pub compute_usage_percent: f32,
    pub memory_usage_percent: f32,
    pub temperature_celsius: f32,
    pub efficiency_score: f32,
}

#[derive(Debug)]
pub struct Phase3Targets {
    pub max_standard_proving_time: Duration, // <2s for standard circuits
    pub max_large_circuit_proving_time: Duration, // <5s for large circuits
    pub max_verification_time: Duration,     // <10ms verification
    pub min_constraints_per_second: f64,     // >1000 constraints/sec
}

impl Default for Phase3Targets {
    fn default() -> Self {
        Self {
            max_standard_proving_time: Duration::from_millis(2000),
            max_large_circuit_proving_time: Duration::from_millis(5000),
            max_verification_time: Duration::from_millis(10),
            min_constraints_per_second: 1000.0,
        }
    }
}

#[derive(Debug, Default)]
pub struct ProvingPerformanceStats {
    pub total_proofs_generated: usize,
    pub average_proving_time: Duration,
    pub min_proving_time: Duration,
    pub max_proving_time: Duration,
    pub success_rate_percent: f64,
    pub average_constraints_per_second: f64,
}

#[derive(Debug, Default)]
pub struct VerificationPerformanceStats {
    pub total_verifications: usize,
    pub average_verification_time: Duration,
    pub success_rate_percent: f64,
}

#[derive(Debug, Default)]
pub struct GpuPerformanceStats {
    pub average_compute_utilization: f32,
    pub average_memory_utilization: f32,
    pub average_temperature: f32,
    pub efficiency_score: f32,
    pub thermal_throttling_detected: bool,
}

#[derive(Debug)]
pub struct Phase3Compliance {
    pub proving_targets_met: u32,      // Percentage
    pub verification_targets_met: u32, // Percentage
    pub overall_compliance: u32,       // Percentage
    pub ready_for_production: bool,
}

#[derive(Debug)]
pub struct PerformanceReport {
    pub proving_performance: ProvingPerformanceStats,
    pub verification_performance: VerificationPerformanceStats,
    pub gpu_performance: GpuPerformanceStats,
    pub phase3_compliance: Phase3Compliance,
    pub recommendations: Vec<String>,
}

#[derive(Debug)]
pub struct PerformanceDashboard {
    pub current_proving_performance: Option<ProvingMetric>,
    pub current_verification_performance: Option<VerificationMetric>,
    pub current_gpu_utilization: Option<GpuUtilization>,
    pub targets_met: TargetMetrics,
    pub alert_level: AlertLevel,
}

#[derive(Debug)]
pub struct TargetMetrics {
    pub proving_targets_met: usize,
    pub proving_total: usize,
    pub verification_targets_met: usize,
    pub verification_total: usize,
}

#[derive(Debug)]
pub enum AlertLevel {
    Green,  // >90% targets met
    Yellow, // 75-89% targets met
    Orange, // 50-74% targets met
    Red,    // <50% targets met
}

#[derive(Debug)]
pub struct PerformanceRegression {
    pub regression_factor: f64,
    pub affected_operations: String,
    pub detection_time: Instant,
    pub recommended_action: String,
}

pub enum ExportFormat {
    Json,
    Csv,
    Prometheus,
}
