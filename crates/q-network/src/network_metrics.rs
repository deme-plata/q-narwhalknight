/// Advanced Network Metrics and Monitoring System
/// 
/// This module provides comprehensive monitoring and analytics for the unified
/// network manager, tracking performance, reliability, and optimization opportunities
/// across all transport layers.

use anyhow::Result;
use chrono::{DateTime, Utc, Duration as ChronoDuration};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::Duration,
};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::loopix_discovery::{NetworkLayer, MessageClass};

/// Comprehensive metrics for network layer performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMetrics {
    pub layer: NetworkLayer,
    
    // Performance metrics
    pub total_messages_sent: u64,
    pub total_messages_received: u64,
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
    
    // Latency statistics
    pub avg_latency_ms: f64,
    pub min_latency_ms: u64,
    pub max_latency_ms: u64,
    pub latency_p95_ms: u64,
    pub latency_p99_ms: u64,
    
    // Reliability metrics
    pub success_rate: f64,
    pub failure_count: u64,
    pub timeout_count: u64,
    pub retry_count: u64,
    
    // Connection metrics
    pub active_connections: usize,
    pub total_connections_established: u64,
    pub connection_failures: u64,
    pub avg_connection_duration_sec: f64,
    
    // Bandwidth utilization
    pub bandwidth_utilization_percent: f64,
    pub peak_bandwidth_mbps: f64,
    pub avg_bandwidth_mbps: f64,
    
    // Security metrics
    pub anonymity_level: f64, // 0.0 to 1.0
    pub encryption_strength: u32, // Bits
    pub metadata_leakage_score: f64, // 0.0 (none) to 1.0 (high)
    
    // Last updated timestamp
    pub last_updated: DateTime<Utc>,
    pub uptime_seconds: u64,
}

impl Default for LayerMetrics {
    fn default() -> Self {
        Self {
            layer: NetworkLayer::LibP2P,
            total_messages_sent: 0,
            total_messages_received: 0,
            total_bytes_sent: 0,
            total_bytes_received: 0,
            avg_latency_ms: 0.0,
            min_latency_ms: u64::MAX,
            max_latency_ms: 0,
            latency_p95_ms: 0,
            latency_p99_ms: 0,
            success_rate: 1.0,
            failure_count: 0,
            timeout_count: 0,
            retry_count: 0,
            active_connections: 0,
            total_connections_established: 0,
            connection_failures: 0,
            avg_connection_duration_sec: 0.0,
            bandwidth_utilization_percent: 0.0,
            peak_bandwidth_mbps: 0.0,
            avg_bandwidth_mbps: 0.0,
            anonymity_level: 0.0,
            encryption_strength: 0,
            metadata_leakage_score: 0.0,
            last_updated: Utc::now(),
            uptime_seconds: 0,
        }
    }
}

/// Routing decision analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingAnalytics {
    pub strategy: OptimizationCategory,
    pub message_class: MessageClass,
    pub selected_layers: Vec<NetworkLayer>,
    pub decision_latency_ms: u64,
    pub actual_send_latency_ms: Option<u64>,
    pub success: bool,
    pub fallback_used: bool,
    pub redundancy_level: usize,
    pub timestamp: DateTime<Utc>,
    pub reasoning: String,
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub layer: NetworkLayer,
    pub metric_name: String,
    pub time_window_minutes: u32,
    pub trend_direction: TrendDirection,
    pub change_percentage: f64,
    pub current_value: f64,
    pub previous_value: f64,
    pub confidence_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Volatile,
}

/// Network health assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkHealthAssessment {
    pub overall_health_score: f64, // 0.0 to 1.0
    pub layer_health_scores: HashMap<NetworkLayer, f64>,
    pub critical_issues: Vec<CriticalIssue>,
    pub recommendations: Vec<OptimizationRecommendation>,
    pub predicted_failure_probability: f64,
    pub time_to_failure_estimate_hours: Option<f64>,
    pub assessment_timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalIssue {
    pub layer: NetworkLayer,
    pub issue_type: IssueType,
    pub severity: IssueSeverity,
    pub description: String,
    pub impact_assessment: String,
    pub recommended_action: String,
    pub detected_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueType {
    HighLatency,
    LowSuccessRate,
    ConnectionFailures,
    BandwidthSaturation,
    SecurityCompromise,
    ConfigurationError,
    ExternalServiceFailure,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Critical,
    High,
    Medium,
    Low,
    Informational,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub category: OptimizationCategory,
    pub description: String,
    pub expected_improvement: String,
    pub implementation_complexity: ComplexityLevel,
    pub priority: u32, // 1 (highest) to 10 (lowest)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationCategory {
    ConnectionPooling,
    BandwidthManagement,
    SecurityEnhancement,
    LatencyOptimization,
    RedundancyConfiguration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Comprehensive network monitoring system
pub struct NetworkMetricsCollector {
    // Layer-specific metrics
    layer_metrics: Arc<RwLock<HashMap<NetworkLayer, LayerMetrics>>>,
    
    // Historical data for trend analysis
    latency_history: Arc<RwLock<HashMap<NetworkLayer, VecDeque<(DateTime<Utc>, u64)>>>>,
    success_rate_history: Arc<RwLock<HashMap<NetworkLayer, VecDeque<(DateTime<Utc>, f64)>>>>,
    bandwidth_history: Arc<RwLock<HashMap<NetworkLayer, VecDeque<(DateTime<Utc>, f64)>>>>,
    
    // Routing analytics
    routing_decisions: Arc<RwLock<VecDeque<RoutingAnalytics>>>,
    
    // Alerting thresholds
    latency_threshold_ms: u64,
    success_rate_threshold: f64,
    bandwidth_threshold_percent: f64,
    
    // Data retention settings
    history_retention_hours: u32,
    max_routing_decisions: usize,
}

impl NetworkMetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            layer_metrics: Arc::new(RwLock::new(HashMap::new())),
            latency_history: Arc::new(RwLock::new(HashMap::new())),
            success_rate_history: Arc::new(RwLock::new(HashMap::new())),
            bandwidth_history: Arc::new(RwLock::new(HashMap::new())),
            routing_decisions: Arc::new(RwLock::new(VecDeque::new())),
            latency_threshold_ms: 1000, // 1 second
            success_rate_threshold: 0.95, // 95%
            bandwidth_threshold_percent: 80.0, // 80%
            history_retention_hours: 24, // 24 hours
            max_routing_decisions: 10000,
        }
    }
    
    /// Record message send event
    pub async fn record_message_sent(
        &self,
        layer: NetworkLayer,
        _message_id: Uuid,
        bytes_sent: usize,
        latency_ms: u64,
        success: bool,
    ) {
        let mut metrics = self.layer_metrics.write().await;
        let layer_clone = layer.clone();
        let layer_metric = metrics.entry(layer).or_insert_with(|| {
            let mut metric = LayerMetrics::default();
            metric.layer = layer_clone.clone();
            metric
        });
        
        // Update basic counters
        layer_metric.total_messages_sent += 1;
        layer_metric.total_bytes_sent += bytes_sent as u64;
        
        if success {
            // Update latency statistics
            self.update_latency_stats(layer_metric, latency_ms);
            
            // Update success rate
            let total_attempts = layer_metric.total_messages_sent;
            let success_count = (layer_metric.success_rate * (total_attempts - 1) as f64) + 1.0;
            layer_metric.success_rate = success_count / total_attempts as f64;
        } else {
            layer_metric.failure_count += 1;
            
            // Update success rate
            let total_attempts = layer_metric.total_messages_sent;
            let success_count = layer_metric.success_rate * (total_attempts - 1) as f64;
            layer_metric.success_rate = success_count / total_attempts as f64;
        }
        
        layer_metric.last_updated = Utc::now();
        
        // Store historical data
        self.store_historical_data(layer_clone.clone(), latency_ms, layer_metric.success_rate).await;

        debug!("📊 Recorded message send for {:?}: {}ms latency, success: {}",
               layer_clone, latency_ms, success);
    }
    
    /// Record routing decision
    pub async fn record_routing_decision(&self, analytics: RoutingAnalytics) {
        let mut decisions = self.routing_decisions.write().await;
        decisions.push_back(analytics.clone());
        
        // Maintain size limit
        if decisions.len() > self.max_routing_decisions {
            decisions.pop_front();
        }
        
        debug!("🧠 Recorded routing decision: {:?} -> {:?}", 
               analytics.message_class, analytics.selected_layers);
    }
    
    /// Update connection metrics
    pub async fn update_connection_metrics(
        &self,
        layer: NetworkLayer,
        active_connections: usize,
        new_connection: bool,
        connection_failed: bool,
    ) {
        let mut metrics = self.layer_metrics.write().await;
        let layer_clone = layer.clone();
        let layer_metric = metrics.entry(layer).or_insert_with(|| {
            let mut metric = LayerMetrics::default();
            metric.layer = layer_clone.clone();
            metric
        });
        
        layer_metric.active_connections = active_connections;
        
        if new_connection {
            layer_metric.total_connections_established += 1;
        }
        
        if connection_failed {
            layer_metric.connection_failures += 1;
        }
        
        layer_metric.last_updated = Utc::now();
    }
    
    /// Update bandwidth metrics
    pub async fn update_bandwidth_metrics(
        &self,
        layer: NetworkLayer,
        current_mbps: f64,
        utilization_percent: f64,
    ) {
        let mut metrics = self.layer_metrics.write().await;
        let layer_clone = layer.clone();
        let layer_metric = metrics.entry(layer).or_insert_with(|| {
            let mut metric = LayerMetrics::default();
            metric.layer = layer_clone.clone();
            metric
        });
        
        layer_metric.bandwidth_utilization_percent = utilization_percent;
        layer_metric.avg_bandwidth_mbps = 
            (layer_metric.avg_bandwidth_mbps * 0.9) + (current_mbps * 0.1);
        
        if current_mbps > layer_metric.peak_bandwidth_mbps {
            layer_metric.peak_bandwidth_mbps = current_mbps;
        }
        
        layer_metric.last_updated = Utc::now();
        
        // Store bandwidth history
        let mut history = self.bandwidth_history.write().await;
        let layer_history = history.entry(layer_clone.clone()).or_insert_with(VecDeque::new);
        layer_history.push_back((Utc::now(), current_mbps));
        
        // Maintain history size
        self.trim_history(layer_history);
    }
    
    /// Get current metrics for all layers
    pub async fn get_layer_metrics(&self) -> HashMap<NetworkLayer, LayerMetrics> {
        self.layer_metrics.read().await.clone()
    }
    
    /// Generate comprehensive network health assessment
    pub async fn generate_health_assessment(&self) -> NetworkHealthAssessment {
        let metrics = self.layer_metrics.read().await.clone();
        let mut layer_health_scores = HashMap::new();
        let mut critical_issues = Vec::new();
        let mut recommendations = Vec::new();
        
        let mut total_health_score = 0.0;
        let mut layer_count = 0;
        
        for (layer, metric) in &metrics {
            // Calculate health score for this layer (0.0 to 1.0)
            let latency_score = self.calculate_latency_score(metric.avg_latency_ms);
            let success_rate_score = metric.success_rate;
            let connection_score = self.calculate_connection_score(metric);
            let bandwidth_score = self.calculate_bandwidth_score(metric);
            
            let layer_health = (latency_score + success_rate_score + connection_score + bandwidth_score) / 4.0;
            layer_health_scores.insert(layer.clone(), layer_health);
            
            total_health_score += layer_health;
            layer_count += 1;
            
            // Identify critical issues
            self.identify_critical_issues(layer.clone(), metric, &mut critical_issues);
            
            // Generate recommendations
            self.generate_recommendations(layer.clone(), metric, &mut recommendations);
        }
        
        let overall_health_score = if layer_count > 0 {
            total_health_score / layer_count as f64
        } else {
            0.0
        };
        
        // Predict failure probability based on trends
        let predicted_failure_probability = self.calculate_failure_probability(&metrics).await;
        let time_to_failure_estimate = self.estimate_time_to_failure(&metrics).await;
        
        NetworkHealthAssessment {
            overall_health_score,
            layer_health_scores,
            critical_issues,
            recommendations,
            predicted_failure_probability,
            time_to_failure_estimate_hours: time_to_failure_estimate,
            assessment_timestamp: Utc::now(),
        }
    }
    
    /// Analyze performance trends
    pub async fn analyze_performance_trends(&self) -> Vec<PerformanceTrend> {
        let mut trends = Vec::new();
        
        // Analyze latency trends
        let latency_history = self.latency_history.read().await;
        for (layer, history) in latency_history.iter() {
            if let Some(trend) = self.calculate_trend("latency_ms", history, 30) {
                trends.push(PerformanceTrend {
                    layer: layer.clone(),
                    metric_name: "latency_ms".to_string(),
                    time_window_minutes: 30,
                    trend_direction: trend.0,
                    change_percentage: trend.1,
                    current_value: trend.2,
                    previous_value: trend.3,
                    confidence_level: trend.4,
                });
            }
        }
        
        // Analyze success rate trends
        let success_history = self.success_rate_history.read().await;
        for (layer, history) in success_history.iter() {
            if let Some(trend) = self.calculate_success_rate_trend(history, 30) {
                trends.push(PerformanceTrend {
                    layer: layer.clone(),
                    metric_name: "success_rate".to_string(),
                    time_window_minutes: 30,
                    trend_direction: trend.0,
                    change_percentage: trend.1,
                    current_value: trend.2,
                    previous_value: trend.3,
                    confidence_level: trend.4,
                });
            }
        }
        
        trends
    }
    
    /// Get routing decision analytics
    pub async fn get_routing_analytics(&self, time_window_hours: u32) -> Vec<RoutingAnalytics> {
        let decisions = self.routing_decisions.read().await;
        let cutoff = Utc::now() - ChronoDuration::hours(time_window_hours as i64);
        
        decisions.iter()
            .filter(|decision| decision.timestamp > cutoff)
            .cloned()
            .collect()
    }
    
    /// Export metrics to JSON for external analysis
    pub async fn export_metrics_json(&self) -> Result<String> {
        let metrics = self.get_layer_metrics().await;
        let health_assessment = self.generate_health_assessment().await;
        let trends = self.analyze_performance_trends().await;
        let routing_analytics = self.get_routing_analytics(24).await;
        
        let export_data = serde_json::json!({
            "timestamp": Utc::now(),
            "layer_metrics": metrics,
            "health_assessment": health_assessment,
            "performance_trends": trends,
            "routing_analytics": routing_analytics,
        });
        
        Ok(serde_json::to_string_pretty(&export_data)?)
    }
    
    // Private helper methods
    
    fn update_latency_stats(&self, metric: &mut LayerMetrics, latency_ms: u64) {
        if metric.total_messages_sent == 1 {
            metric.avg_latency_ms = latency_ms as f64;
            metric.min_latency_ms = latency_ms;
            metric.max_latency_ms = latency_ms;
        } else {
            let count = metric.total_messages_sent as f64;
            metric.avg_latency_ms = ((metric.avg_latency_ms * (count - 1.0)) + latency_ms as f64) / count;
            metric.min_latency_ms = metric.min_latency_ms.min(latency_ms);
            metric.max_latency_ms = metric.max_latency_ms.max(latency_ms);
        }
    }
    
    async fn store_historical_data(&self, layer: NetworkLayer, latency_ms: u64, success_rate: f64) {
        let now = Utc::now();
        let layer_clone = layer.clone();

        // Store latency history
        {
            let mut history = self.latency_history.write().await;
            let layer_history = history.entry(layer).or_insert_with(VecDeque::new);
            layer_history.push_back((now, latency_ms));
            self.trim_history(layer_history);
        }

        // Store success rate history
        {
            let mut history = self.success_rate_history.write().await;
            let layer_history = history.entry(layer_clone).or_insert_with(VecDeque::new);
            layer_history.push_back((now, success_rate));
            self.trim_history(layer_history);
        }
    }
    
    fn trim_history<T>(&self, history: &mut VecDeque<(DateTime<Utc>, T)>) {
        let cutoff = Utc::now() - ChronoDuration::hours(self.history_retention_hours as i64);
        while let Some((timestamp, _)) = history.front() {
            if *timestamp < cutoff {
                history.pop_front();
            } else {
                break;
            }
        }
    }
    
    fn calculate_latency_score(&self, avg_latency_ms: f64) -> f64 {
        // Score decreases as latency increases
        let max_acceptable_latency = 2000.0; // 2 seconds
        (max_acceptable_latency - avg_latency_ms).max(0.0) / max_acceptable_latency
    }
    
    fn calculate_connection_score(&self, metric: &LayerMetrics) -> f64 {
        if metric.total_connections_established == 0 {
            return 1.0; // No connections attempted yet
        }
        
        let connection_success_rate = 
            (metric.total_connections_established - metric.connection_failures) as f64 
            / metric.total_connections_established as f64;
        
        connection_success_rate
    }
    
    fn calculate_bandwidth_score(&self, metric: &LayerMetrics) -> f64 {
        // Score decreases as bandwidth utilization approaches 100%
        (100.0 - metric.bandwidth_utilization_percent).max(0.0) / 100.0
    }
    
    fn identify_critical_issues(&self, layer: NetworkLayer, metric: &LayerMetrics, issues: &mut Vec<CriticalIssue>) {
        // High latency issue
        if metric.avg_latency_ms > self.latency_threshold_ms as f64 {
            issues.push(CriticalIssue {
                layer: layer.clone(),
                issue_type: IssueType::HighLatency,
                severity: if metric.avg_latency_ms > self.latency_threshold_ms as f64 * 2.0 {
                    IssueSeverity::Critical
                } else {
                    IssueSeverity::High
                },
                description: format!("Average latency ({:.0}ms) exceeds threshold ({}ms)", 
                                   metric.avg_latency_ms, self.latency_threshold_ms),
                impact_assessment: "High latency may cause consensus delays and poor user experience".to_string(),
                recommended_action: "Investigate network conditions and consider routing optimization".to_string(),
                detected_at: Utc::now(),
            });
        }
        
        // Low success rate issue
        if metric.success_rate < self.success_rate_threshold {
            issues.push(CriticalIssue {
                layer: layer.clone(),
                issue_type: IssueType::LowSuccessRate,
                severity: if metric.success_rate < 0.8 {
                    IssueSeverity::Critical
                } else {
                    IssueSeverity::High
                },
                description: format!("Success rate ({:.1}%) below threshold ({:.1}%)", 
                                   metric.success_rate * 100.0, self.success_rate_threshold * 100.0),
                impact_assessment: "Low success rate indicates reliability issues".to_string(),
                recommended_action: "Check layer configuration and external service availability".to_string(),
                detected_at: Utc::now(),
            });
        }
        
        // Bandwidth saturation issue
        if metric.bandwidth_utilization_percent > self.bandwidth_threshold_percent {
            issues.push(CriticalIssue {
                layer: layer.clone(),
                issue_type: IssueType::BandwidthSaturation,
                severity: if metric.bandwidth_utilization_percent > 95.0 {
                    IssueSeverity::Critical
                } else {
                    IssueSeverity::Medium
                },
                description: format!("Bandwidth utilization ({:.1}%) exceeds threshold ({:.1}%)", 
                                   metric.bandwidth_utilization_percent, self.bandwidth_threshold_percent),
                impact_assessment: "High bandwidth usage may cause congestion and delays".to_string(),
                recommended_action: "Consider load balancing or bandwidth optimization".to_string(),
                detected_at: Utc::now(),
            });
        }
    }
    
    fn generate_recommendations(&self, layer: NetworkLayer, metric: &LayerMetrics, recommendations: &mut Vec<OptimizationRecommendation>) {
        // Latency optimization recommendations
        if metric.avg_latency_ms > 500.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::LatencyOptimization,
                description: format!("Optimize {:?} layer routing to reduce latency", layer),
                expected_improvement: "15-30% latency reduction".to_string(),
                implementation_complexity: ComplexityLevel::Medium,
                priority: 3,
            });
        }
        
        // Connection pooling recommendations
        if metric.connection_failures > 10 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::ConnectionPooling,
                description: format!("Implement connection pooling for {:?} layer", layer),
                expected_improvement: "Reduced connection establishment overhead".to_string(),
                implementation_complexity: ComplexityLevel::Low,
                priority: 2,
            });
        }
        
        // Redundancy recommendations
        if metric.success_rate < 0.95 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::RedundancyConfiguration,
                description: format!("Enable redundant routing for {:?} layer", layer),
                expected_improvement: "Improved reliability through fallback mechanisms".to_string(),
                implementation_complexity: ComplexityLevel::High,
                priority: 1,
            });
        }
    }
    
    async fn calculate_failure_probability(&self, _metrics: &HashMap<NetworkLayer, LayerMetrics>) -> f64 {
        // Simplified failure probability calculation
        // Real implementation would use machine learning or statistical analysis
        0.05 // 5% base failure probability
    }
    
    async fn estimate_time_to_failure(&self, _metrics: &HashMap<NetworkLayer, LayerMetrics>) -> Option<f64> {
        // Simplified time-to-failure estimation
        // Real implementation would analyze trends and extrapolate
        Some(168.0) // 1 week estimate
    }
    
    fn calculate_trend<T>(&self, _metric_name: &str, _history: &VecDeque<(DateTime<Utc>, T)>, _window_minutes: u32) -> Option<(TrendDirection, f64, f64, f64, f64)> {
        // Simplified trend calculation
        // Real implementation would perform linear regression or more sophisticated analysis
        Some((TrendDirection::Stable, 0.0, 0.0, 0.0, 0.8))
    }
    
    fn calculate_success_rate_trend(&self, _history: &VecDeque<(DateTime<Utc>, f64)>, _window_minutes: u32) -> Option<(TrendDirection, f64, f64, f64, f64)> {
        // Simplified success rate trend calculation
        Some((TrendDirection::Stable, 0.0, 0.95, 0.95, 0.8))
    }
}

/// Metrics reporting and alerting system
pub struct MetricsReporter {
    collector: Arc<NetworkMetricsCollector>,
    alert_thresholds: AlertThresholds,
}

#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub critical_latency_ms: u64,
    pub critical_success_rate: f64,
    pub critical_bandwidth_percent: f64,
    pub critical_health_score: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            critical_latency_ms: 5000, // 5 seconds
            critical_success_rate: 0.8, // 80%
            critical_bandwidth_percent: 95.0, // 95%
            critical_health_score: 0.5, // 50%
        }
    }
}

impl MetricsReporter {
    pub fn new(collector: Arc<NetworkMetricsCollector>) -> Self {
        Self {
            collector,
            alert_thresholds: AlertThresholds::default(),
        }
    }
    
    /// Generate comprehensive metrics report
    pub async fn generate_report(&self) -> Result<String> {
        let health_assessment = self.collector.generate_health_assessment().await;
        let trends = self.collector.analyze_performance_trends().await;
        let metrics = self.collector.get_layer_metrics().await;
        
        let mut report = String::new();
        report.push_str("# Q-NarwhalKnight Network Metrics Report\n\n");
        report.push_str(&format!("Generated: {}\n\n", Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        
        // Overall health
        report.push_str(&format!("## Overall Network Health: {:.1}%\n\n", 
                                health_assessment.overall_health_score * 100.0));
        
        // Layer performance
        report.push_str("## Layer Performance\n\n");
        for (layer, metric) in &metrics {
            report.push_str(&format!("### {:?}\n", layer));
            report.push_str(&format!("- Messages Sent: {}\n", metric.total_messages_sent));
            report.push_str(&format!("- Average Latency: {:.0}ms\n", metric.avg_latency_ms));
            report.push_str(&format!("- Success Rate: {:.1}%\n", metric.success_rate * 100.0));
            report.push_str(&format!("- Active Connections: {}\n", metric.active_connections));
            report.push_str(&format!("- Bandwidth Utilization: {:.1}%\n\n", metric.bandwidth_utilization_percent));
        }
        
        // Critical issues
        if !health_assessment.critical_issues.is_empty() {
            report.push_str("## Critical Issues\n\n");
            for issue in &health_assessment.critical_issues {
                report.push_str(&format!("- **{:?}** ({:?}): {}\n", 
                                        issue.severity, issue.issue_type, issue.description));
            }
            report.push_str("\n");
        }
        
        // Recommendations
        if !health_assessment.recommendations.is_empty() {
            report.push_str("## Optimization Recommendations\n\n");
            for rec in &health_assessment.recommendations {
                report.push_str(&format!("- **Priority {}**: {} ({})\n", 
                                        rec.priority, rec.description, rec.expected_improvement));
            }
            report.push_str("\n");
        }
        
        Ok(report)
    }
    
    /// Check for alerts and return critical issues
    pub async fn check_alerts(&self) -> Vec<CriticalIssue> {
        let health_assessment = self.collector.generate_health_assessment().await;
        
        health_assessment.critical_issues.into_iter()
            .filter(|issue| {
                matches!(issue.severity, IssueSeverity::Critical | IssueSeverity::High)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_metrics_collection() {
        let collector = NetworkMetricsCollector::new();
        
        // Record some test metrics
        collector.record_message_sent(
            NetworkLayer::LibP2P,
            Uuid::new_v4(),
            1024,
            50,
            true,
        ).await;
        
        let metrics = collector.get_layer_metrics().await;
        assert!(metrics.contains_key(&NetworkLayer::LibP2P));
        
        let libp2p_metrics = &metrics[&NetworkLayer::LibP2P];
        assert_eq!(libp2p_metrics.total_messages_sent, 1);
        assert_eq!(libp2p_metrics.total_bytes_sent, 1024);
        assert_eq!(libp2p_metrics.avg_latency_ms, 50.0);
    }
    
    #[tokio::test]
    async fn test_health_assessment() {
        let collector = NetworkMetricsCollector::new();
        
        // Record metrics for multiple layers
        for layer in [NetworkLayer::LibP2P, NetworkLayer::Tor, NetworkLayer::DNSPhantom] {
            collector.record_message_sent(layer, Uuid::new_v4(), 1024, 100, true).await;
        }
        
        let assessment = collector.generate_health_assessment().await;
        assert!(assessment.overall_health_score > 0.0);
        assert_eq!(assessment.layer_health_scores.len(), 3);
    }
}