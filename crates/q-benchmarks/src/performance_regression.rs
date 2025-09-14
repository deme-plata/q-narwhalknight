/// Performance Regression Detection System for Q-NarwhalKnight
///
/// Monitors performance metrics over time and detects significant regressions
/// in ZK proof generation, verification, consensus operations, and GPU acceleration
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::Duration;

/// Performance regression detection threshold
const REGRESSION_THRESHOLD: f64 = 0.15; // 15% performance degradation
const BASELINE_SAMPLES: usize = 10; // Minimum samples for baseline
const MOVING_AVERAGE_WINDOW: usize = 5; // Window for moving average

/// Core metrics tracked for regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: DateTime<Utc>,
    pub git_commit: String,
    pub build_config: BuildConfig,
    pub zk_metrics: ZkPerformanceMetrics,
    pub consensus_metrics: ConsensusPerformanceMetrics,
    pub gpu_metrics: Option<GpuPerformanceMetrics>,
    pub system_metrics: SystemMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    pub profile: String, // debug, release
    pub features: Vec<String>,
    pub rust_version: String,
    pub target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkPerformanceMetrics {
    // SNARK performance
    pub groth16_proving_time_ms: f64,
    pub groth16_verification_time_ms: f64,
    pub groth16_proof_size_bytes: u64,

    // PLONK performance
    pub plonk_setup_time_ms: f64,
    pub plonk_proving_time_ms: f64,
    pub plonk_verification_time_ms: f64,

    // Circuit complexity metrics
    pub constraint_count: u64,
    pub witness_generation_time_ms: f64,
    pub circuit_compilation_time_ms: f64,

    // Memory usage
    pub peak_memory_mb: f64,
    pub average_memory_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusPerformanceMetrics {
    // DAG-Knight performance
    pub vertex_processing_time_ms: f64,
    pub consensus_round_time_ms: f64,
    pub finality_time_ms: f64,

    // Throughput metrics
    pub transactions_per_second: f64,
    pub vertices_per_second: f64,
    pub network_latency_ms: f64,

    // Memory and storage
    pub dag_memory_usage_mb: f64,
    pub state_storage_size_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPerformanceMetrics {
    // GPU acceleration metrics
    pub gpu_fft_time_ms: f64,
    pub gpu_field_ops_time_ms: f64,
    pub gpu_fri_commitment_time_ms: f64,

    // GPU utilization
    pub gpu_utilization_percent: f64,
    pub gpu_memory_usage_mb: f64,
    pub gpu_power_usage_watts: f64,

    // Speedup factors
    pub cpu_vs_gpu_speedup: f64,
    pub parallel_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_cores: u32,
    pub total_memory_gb: f64,
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub disk_usage_percent: f64,
}

/// Performance regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysis {
    pub timestamp: DateTime<Utc>,
    pub commit_range: (String, String), // (baseline_commit, current_commit)
    pub detected_regressions: Vec<RegressionAlert>,
    pub performance_summary: PerformanceSummary,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAlert {
    pub metric_name: String,
    pub metric_category: String,
    pub baseline_value: f64,
    pub current_value: f64,
    pub regression_percent: f64,
    pub severity: RegressionSeverity,
    pub first_detected: DateTime<Utc>,
    pub affected_commits: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Minor,    // 5-15% regression
    Moderate, // 15-30% regression
    Major,    // 30-50% regression
    Critical, // >50% regression
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_metrics_tracked: u32,
    pub regressions_detected: u32,
    pub improvements_detected: u32,
    pub overall_performance_trend: f64, // Positive = improvement, Negative = regression
}

/// Main performance regression detection engine
pub struct PerformanceRegressionDetector {
    metrics_history: VecDeque<PerformanceMetrics>,
    baseline_metrics: Option<PerformanceMetrics>,
    regression_alerts: Vec<RegressionAlert>,
    config: RegressionConfig,
}

#[derive(Debug, Clone)]
pub struct RegressionConfig {
    pub regression_threshold: f64,
    pub improvement_threshold: f64,
    pub baseline_window: usize,
    pub history_limit: usize,
    pub alert_cooldown_hours: u64,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            regression_threshold: REGRESSION_THRESHOLD,
            improvement_threshold: -0.10, // 10% improvement
            baseline_window: BASELINE_SAMPLES,
            history_limit: 1000,
            alert_cooldown_hours: 24,
        }
    }
}

impl PerformanceRegressionDetector {
    pub fn new(config: RegressionConfig) -> Self {
        Self {
            metrics_history: VecDeque::new(),
            baseline_metrics: None,
            regression_alerts: Vec::new(),
            config,
        }
    }

    /// Add new performance measurement
    pub fn record_metrics(&mut self, metrics: PerformanceMetrics) -> Result<()> {
        // Maintain history limit
        if self.metrics_history.len() >= self.config.history_limit {
            self.metrics_history.pop_front();
        }

        self.metrics_history.push_back(metrics.clone());

        // Update baseline if we have enough samples
        if self.metrics_history.len() >= self.config.baseline_window
            && self.baseline_metrics.is_none()
        {
            self.baseline_metrics = Some(self.calculate_baseline_metrics()?);
        }

        // Detect regressions if we have a baseline
        if self.baseline_metrics.is_some() {
            self.detect_regressions(&metrics)?;
        }

        Ok(())
    }

    /// Perform comprehensive regression analysis
    pub fn analyze_regressions(&self) -> Result<RegressionAnalysis> {
        let current_metrics = self
            .metrics_history
            .back()
            .ok_or_else(|| anyhow::anyhow!("No metrics available for analysis"))?;

        let baseline_metrics = self
            .baseline_metrics
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No baseline metrics available"))?;

        let detected_regressions = self.get_active_regressions();

        let performance_summary = PerformanceSummary {
            total_metrics_tracked: self.count_tracked_metrics(),
            regressions_detected: detected_regressions.len() as u32,
            improvements_detected: self.count_improvements(baseline_metrics, current_metrics)?,
            overall_performance_trend: self.calculate_overall_trend()?,
        };

        let recommendations = self.generate_recommendations(&detected_regressions);

        Ok(RegressionAnalysis {
            timestamp: Utc::now(),
            commit_range: (
                baseline_metrics.git_commit.clone(),
                current_metrics.git_commit.clone(),
            ),
            detected_regressions,
            performance_summary,
            recommendations,
        })
    }

    /// Calculate baseline metrics from historical data
    fn calculate_baseline_metrics(&self) -> Result<PerformanceMetrics> {
        if self.metrics_history.len() < self.config.baseline_window {
            return Err(anyhow::anyhow!(
                "Insufficient data for baseline calculation"
            ));
        }

        let baseline_window = self
            .metrics_history
            .iter()
            .take(self.config.baseline_window)
            .collect::<Vec<_>>();

        // Calculate averaged metrics for baseline
        let mut baseline = baseline_window[0].clone();

        // Average ZK metrics
        baseline.zk_metrics.groth16_proving_time_ms =
            self.average_metric(&baseline_window, |m| m.zk_metrics.groth16_proving_time_ms);
        baseline.zk_metrics.groth16_verification_time_ms = self
            .average_metric(&baseline_window, |m| {
                m.zk_metrics.groth16_verification_time_ms
            });
        baseline.zk_metrics.plonk_proving_time_ms =
            self.average_metric(&baseline_window, |m| m.zk_metrics.plonk_proving_time_ms);

        // Average consensus metrics
        baseline.consensus_metrics.vertex_processing_time_ms = self
            .average_metric(&baseline_window, |m| {
                m.consensus_metrics.vertex_processing_time_ms
            });
        baseline.consensus_metrics.transactions_per_second = self
            .average_metric(&baseline_window, |m| {
                m.consensus_metrics.transactions_per_second
            });

        // Average GPU metrics if available
        if let Some(ref mut gpu_metrics) = baseline.gpu_metrics {
            if baseline_window.iter().all(|m| m.gpu_metrics.is_some()) {
                gpu_metrics.gpu_fft_time_ms = self.average_metric(&baseline_window, |m| {
                    m.gpu_metrics.as_ref().unwrap().gpu_fft_time_ms
                });
                gpu_metrics.cpu_vs_gpu_speedup = self.average_metric(&baseline_window, |m| {
                    m.gpu_metrics.as_ref().unwrap().cpu_vs_gpu_speedup
                });
            }
        }

        Ok(baseline)
    }

    /// Helper function to calculate average of a metric
    fn average_metric<F>(&self, metrics: &[&PerformanceMetrics], extractor: F) -> f64
    where
        F: Fn(&PerformanceMetrics) -> f64,
    {
        let sum: f64 = metrics.iter().map(|m| extractor(m)).sum();
        sum / metrics.len() as f64
    }

    /// Detect performance regressions in new metrics
    fn detect_regressions(&mut self, current_metrics: &PerformanceMetrics) -> Result<()> {
        let baseline = self.baseline_metrics.as_ref().unwrap();

        // Check ZK performance regressions
        self.check_metric_regression(
            "groth16_proving_time_ms",
            "zk_performance",
            baseline.zk_metrics.groth16_proving_time_ms,
            current_metrics.zk_metrics.groth16_proving_time_ms,
            &current_metrics.git_commit,
        )?;

        self.check_metric_regression(
            "groth16_verification_time_ms",
            "zk_performance",
            baseline.zk_metrics.groth16_verification_time_ms,
            current_metrics.zk_metrics.groth16_verification_time_ms,
            &current_metrics.git_commit,
        )?;

        self.check_metric_regression(
            "plonk_proving_time_ms",
            "zk_performance",
            baseline.zk_metrics.plonk_proving_time_ms,
            current_metrics.zk_metrics.plonk_proving_time_ms,
            &current_metrics.git_commit,
        )?;

        // Check consensus performance regressions
        self.check_metric_regression(
            "vertex_processing_time_ms",
            "consensus_performance",
            baseline.consensus_metrics.vertex_processing_time_ms,
            current_metrics.consensus_metrics.vertex_processing_time_ms,
            &current_metrics.git_commit,
        )?;

        // Check TPS regression (inverse check - lower TPS is regression)
        self.check_inverse_metric_regression(
            "transactions_per_second",
            "consensus_throughput",
            baseline.consensus_metrics.transactions_per_second,
            current_metrics.consensus_metrics.transactions_per_second,
            &current_metrics.git_commit,
        )?;

        // Check GPU performance regressions if available
        if let (Some(baseline_gpu), Some(current_gpu)) =
            (&baseline.gpu_metrics, &current_metrics.gpu_metrics)
        {
            self.check_metric_regression(
                "gpu_fft_time_ms",
                "gpu_performance",
                baseline_gpu.gpu_fft_time_ms,
                current_gpu.gpu_fft_time_ms,
                &current_metrics.git_commit,
            )?;

            self.check_inverse_metric_regression(
                "cpu_vs_gpu_speedup",
                "gpu_acceleration",
                baseline_gpu.cpu_vs_gpu_speedup,
                current_gpu.cpu_vs_gpu_speedup,
                &current_metrics.git_commit,
            )?;
        }

        Ok(())
    }

    /// Check for regression in a metric (higher = worse)
    fn check_metric_regression(
        &mut self,
        metric_name: &str,
        category: &str,
        baseline_value: f64,
        current_value: f64,
        commit: &str,
    ) -> Result<()> {
        let change_percent = (current_value - baseline_value) / baseline_value;

        if change_percent > self.config.regression_threshold {
            let severity = match change_percent {
                x if x > 0.50 => RegressionSeverity::Critical,
                x if x > 0.30 => RegressionSeverity::Major,
                x if x > 0.15 => RegressionSeverity::Moderate,
                _ => RegressionSeverity::Minor,
            };

            // Check if this regression already exists
            if let Some(existing) = self
                .regression_alerts
                .iter_mut()
                .find(|alert| alert.metric_name == metric_name)
            {
                existing.current_value = current_value;
                existing.regression_percent = change_percent * 100.0;
                existing.affected_commits.push(commit.to_string());
            } else {
                self.regression_alerts.push(RegressionAlert {
                    metric_name: metric_name.to_string(),
                    metric_category: category.to_string(),
                    baseline_value,
                    current_value,
                    regression_percent: change_percent * 100.0,
                    severity,
                    first_detected: Utc::now(),
                    affected_commits: vec![commit.to_string()],
                });
            }
        }

        Ok(())
    }

    /// Check for regression in inverse metrics (lower = worse, like TPS or speedup)
    fn check_inverse_metric_regression(
        &mut self,
        metric_name: &str,
        category: &str,
        baseline_value: f64,
        current_value: f64,
        commit: &str,
    ) -> Result<()> {
        let change_percent = (baseline_value - current_value) / baseline_value;

        if change_percent > self.config.regression_threshold {
            let severity = match change_percent {
                x if x > 0.50 => RegressionSeverity::Critical,
                x if x > 0.30 => RegressionSeverity::Major,
                x if x > 0.15 => RegressionSeverity::Moderate,
                _ => RegressionSeverity::Minor,
            };

            if let Some(existing) = self
                .regression_alerts
                .iter_mut()
                .find(|alert| alert.metric_name == metric_name)
            {
                existing.current_value = current_value;
                existing.regression_percent = change_percent * 100.0;
                existing.affected_commits.push(commit.to_string());
            } else {
                self.regression_alerts.push(RegressionAlert {
                    metric_name: metric_name.to_string(),
                    metric_category: category.to_string(),
                    baseline_value,
                    current_value,
                    regression_percent: change_percent * 100.0,
                    severity,
                    first_detected: Utc::now(),
                    affected_commits: vec![commit.to_string()],
                });
            }
        }

        Ok(())
    }

    /// Get currently active regression alerts
    fn get_active_regressions(&self) -> Vec<RegressionAlert> {
        // Filter out regressions older than cooldown period
        let cutoff_time =
            Utc::now() - chrono::Duration::hours(self.config.alert_cooldown_hours as i64);

        self.regression_alerts
            .iter()
            .filter(|alert| alert.first_detected > cutoff_time)
            .cloned()
            .collect()
    }

    /// Count total metrics being tracked
    fn count_tracked_metrics(&self) -> u32 {
        let base_metrics = 10; // ZK + consensus core metrics
        let gpu_metrics = if self.metrics_history.back().unwrap().gpu_metrics.is_some() {
            6
        } else {
            0
        };
        base_metrics + gpu_metrics
    }

    /// Count performance improvements
    fn count_improvements(
        &self,
        baseline: &PerformanceMetrics,
        current: &PerformanceMetrics,
    ) -> Result<u32> {
        let mut improvements = 0;

        // Check ZK improvements (lower time = better)
        if (baseline.zk_metrics.groth16_proving_time_ms
            - current.zk_metrics.groth16_proving_time_ms)
            / baseline.zk_metrics.groth16_proving_time_ms
            > -self.config.improvement_threshold
        {
            improvements += 1;
        }

        // Check TPS improvements (higher = better)
        if (current.consensus_metrics.transactions_per_second
            - baseline.consensus_metrics.transactions_per_second)
            / baseline.consensus_metrics.transactions_per_second
            > -self.config.improvement_threshold
        {
            improvements += 1;
        }

        Ok(improvements)
    }

    /// Calculate overall performance trend
    fn calculate_overall_trend(&self) -> Result<f64> {
        if self.metrics_history.len() < 5 {
            return Ok(0.0);
        }

        // Use simple linear regression on TPS as primary performance indicator
        let recent_metrics: Vec<_> = self
            .metrics_history
            .iter()
            .rev()
            .take(MOVING_AVERAGE_WINDOW)
            .collect();

        let tps_values: Vec<f64> = recent_metrics
            .iter()
            .map(|m| m.consensus_metrics.transactions_per_second)
            .collect();

        // Simple trend calculation (positive = improving, negative = degrading)
        let first_tps = tps_values.last().unwrap();
        let last_tps = tps_values.first().unwrap();

        Ok((last_tps - first_tps) / first_tps)
    }

    /// Generate actionable recommendations based on detected regressions
    fn generate_recommendations(&self, regressions: &[RegressionAlert]) -> Vec<String> {
        let mut recommendations = Vec::new();

        if regressions.is_empty() {
            recommendations.push(
                "✅ No performance regressions detected. System performance is stable.".to_string(),
            );
            return recommendations;
        }

        // Analyze regression patterns
        let critical_count = regressions
            .iter()
            .filter(|r| matches!(r.severity, RegressionSeverity::Critical))
            .count();
        let major_count = regressions
            .iter()
            .filter(|r| matches!(r.severity, RegressionSeverity::Major))
            .count();

        if critical_count > 0 {
            recommendations.push(format!("🚨 CRITICAL: {} critical performance regressions detected. Immediate investigation required.", critical_count));
        }

        if major_count > 0 {
            recommendations.push(format!("⚠️  MAJOR: {} major performance regressions detected. Review recent commits and consider rollback.", major_count));
        }

        // Category-specific recommendations
        let zk_regressions = regressions
            .iter()
            .filter(|r| r.metric_category == "zk_performance")
            .count();
        let consensus_regressions = regressions
            .iter()
            .filter(|r| r.metric_category == "consensus_performance")
            .count();
        let gpu_regressions = regressions
            .iter()
            .filter(|r| r.metric_category == "gpu_performance")
            .count();

        if zk_regressions > 0 {
            recommendations.push(format!("🔐 Zero-Knowledge Performance: {} ZK-related regressions detected. Check arkworks library versions and circuit optimizations.", zk_regressions));
        }

        if consensus_regressions > 0 {
            recommendations.push(format!("📊 Consensus Performance: {} consensus-related regressions detected. Review DAG-Knight algorithm changes and network optimizations.", consensus_regressions));
        }

        if gpu_regressions > 0 {
            recommendations.push(format!("🚀 GPU Acceleration: {} GPU-related regressions detected. Verify WebGPU drivers and compute shader optimizations.", gpu_regressions));
        }

        // Specific metric recommendations
        for regression in regressions {
            match regression.metric_name.as_str() {
                "groth16_proving_time_ms" => {
                    recommendations.push("🔧 Groth16 Proving: Check constraint system optimizations and parallel proving implementation.".to_string());
                }
                "transactions_per_second" => {
                    recommendations.push("📈 TPS Regression: Review mempool batching, network latency, and vertex processing optimizations.".to_string());
                }
                "cpu_vs_gpu_speedup" => {
                    recommendations.push("⚡ GPU Speedup: Analyze GPU memory allocation patterns and compute pipeline efficiency.".to_string());
                }
                _ => {}
            }
        }

        // General recommendations
        recommendations.push("📋 Next Steps: 1) Run git bisect on affected commits 2) Profile specific workloads 3) Check for dependency updates 4) Validate hardware configuration".to_string());

        recommendations
    }

    /// Export performance data for external analysis
    pub fn export_metrics_csv(&self, filepath: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(filepath)?;

        // Write CSV header
        writeln!(file, "timestamp,commit,groth16_proving_ms,groth16_verification_ms,plonk_proving_ms,tps,vertex_processing_ms,peak_memory_mb,gpu_fft_ms,gpu_speedup")?;

        // Write data
        for metrics in &self.metrics_history {
            let gpu_fft = metrics
                .gpu_metrics
                .as_ref()
                .map(|g| g.gpu_fft_time_ms)
                .unwrap_or(0.0);
            let gpu_speedup = metrics
                .gpu_metrics
                .as_ref()
                .map(|g| g.cpu_vs_gpu_speedup)
                .unwrap_or(1.0);

            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{},{}",
                metrics.timestamp.format("%Y-%m-%d %H:%M:%S"),
                metrics.git_commit,
                metrics.zk_metrics.groth16_proving_time_ms,
                metrics.zk_metrics.groth16_verification_time_ms,
                metrics.zk_metrics.plonk_proving_time_ms,
                metrics.consensus_metrics.transactions_per_second,
                metrics.consensus_metrics.vertex_processing_time_ms,
                metrics.zk_metrics.peak_memory_mb,
                gpu_fft,
                gpu_speedup,
            )?;
        }

        Ok(())
    }
}

/// Helper function to collect system metrics
pub fn collect_system_metrics() -> Result<SystemMetrics> {
    Ok(SystemMetrics {
        cpu_cores: num_cpus::get() as u32,
        total_memory_gb: 16.0,  // TODO: Get actual system memory
        cpu_usage_percent: 0.0, // TODO: Implement system monitoring
        memory_usage_percent: 0.0,
        disk_usage_percent: 0.0,
    })
}

/// Integration with CI/CD pipeline
#[cfg(feature = "ci-integration")]
pub mod ci_integration {
    use super::*;

    /// Generate CI/CD-friendly regression report
    pub fn generate_ci_report(analysis: &RegressionAnalysis) -> String {
        let mut report = String::new();

        report.push_str("## 📊 Performance Regression Analysis\n\n");

        if analysis.detected_regressions.is_empty() {
            report.push_str("✅ **No performance regressions detected**\n\n");
        } else {
            report.push_str(&format!(
                "⚠️  **{} performance regressions detected**\n\n",
                analysis.detected_regressions.len()
            ));

            for regression in &analysis.detected_regressions {
                let severity_emoji = match regression.severity {
                    RegressionSeverity::Critical => "🚨",
                    RegressionSeverity::Major => "⚠️ ",
                    RegressionSeverity::Moderate => "📊",
                    RegressionSeverity::Minor => "ℹ️ ",
                };

                report.push_str(&format!(
                    "{} **{}**: {:.1}% regression ({:.3} → {:.3})\n",
                    severity_emoji,
                    regression.metric_name,
                    regression.regression_percent,
                    regression.baseline_value,
                    regression.current_value
                ));
            }
        }

        report.push_str(&format!(
            "\n📈 **Overall Trend**: {:.1}%\n",
            analysis.performance_summary.overall_performance_trend * 100.0
        ));

        report
    }

    /// Set CI/CD exit codes based on regression severity
    pub fn get_ci_exit_code(analysis: &RegressionAnalysis) -> i32 {
        let critical_regressions = analysis
            .detected_regressions
            .iter()
            .any(|r| matches!(r.severity, RegressionSeverity::Critical));

        if critical_regressions {
            return 1; // Fail CI on critical regressions
        }

        let major_regressions = analysis
            .detected_regressions
            .iter()
            .filter(|r| matches!(r.severity, RegressionSeverity::Major))
            .count();

        if major_regressions >= 3 {
            return 1; // Fail CI on multiple major regressions
        }

        0 // Pass CI
    }
}
