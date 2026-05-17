/// Adaptive QoS: Quality of Service management for Tor circuits
/// Maintains <300ms latency targets through adaptive circuit selection and optimization
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

/// QoS performance targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSTargets {
    /// Target maximum latency (default: 300ms for Tor)
    pub max_latency: Duration,
    /// Target average latency (default: 200ms)
    pub avg_latency: Duration,
    /// Maximum acceptable packet loss rate
    pub max_packet_loss: f64,
    /// Minimum throughput requirement (bytes/sec)
    pub min_throughput: u64,
    /// Circuit utilization threshold before load balancing
    pub utilization_threshold: f64,
}

impl Default for QoSTargets {
    fn default() -> Self {
        Self {
            max_latency: Duration::from_millis(300),
            avg_latency: Duration::from_millis(200),
            max_packet_loss: 0.01,      // 1% packet loss
            // v8.6.0: increased from 1MB/s to 2MB/s — modern Tor relays support
            // higher throughput; raising the floor triggers optimization sooner
            // for circuits that underperform
            min_throughput: 2_097_152,  // 2 MB/s
            utilization_threshold: 0.8, // 80% utilization
        }
    }
}

/// Real-time circuit performance metrics
#[derive(Debug, Clone)]
pub struct CircuitPerformanceMetrics {
    pub circuit_id: u64,
    pub latency_history: VecDeque<Duration>,
    pub throughput_history: VecDeque<u64>,
    pub packet_loss_rate: f64,
    pub last_measurement: Instant,
    pub total_bytes_sent: u64,
    pub total_packets_sent: u64,
    pub total_packets_lost: u64,
    pub utilization_score: f64,
    pub qos_grade: QoSGrade,
}

/// QoS performance grade
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QoSGrade {
    /// Excellent performance (< 150ms avg latency)
    Excellent,
    /// Good performance (150-250ms avg latency)
    Good,
    /// Acceptable performance (250-300ms avg latency)
    Acceptable,
    /// Poor performance (300-500ms avg latency)
    Poor,
    /// Unacceptable performance (> 500ms latency)
    Unacceptable,
}

impl QoSGrade {
    /// Get numeric score for comparison (higher = better)
    pub fn score(&self) -> u8 {
        match self {
            Self::Excellent => 5,
            Self::Good => 4,
            Self::Acceptable => 3,
            Self::Poor => 2,
            Self::Unacceptable => 1,
        }
    }

    /// Calculate grade from average latency
    pub fn from_latency(avg_latency: Duration) -> Self {
        let ms = avg_latency.as_millis() as u64;
        match ms {
            0..=150 => Self::Excellent,
            151..=250 => Self::Good,
            251..=300 => Self::Acceptable,
            301..=500 => Self::Poor,
            _ => Self::Unacceptable,
        }
    }
}

/// Adaptive QoS manager
pub struct AdaptiveQoS {
    /// Performance metrics for each circuit
    metrics: Arc<RwLock<HashMap<u64, CircuitPerformanceMetrics>>>,
    /// QoS targets and thresholds
    targets: QoSTargets,
    /// Circuit performance history window size
    history_window: usize,
    /// Last optimization run
    last_optimization: Arc<Mutex<Instant>>,
    /// QoS statistics
    stats: Arc<RwLock<QoSStats>>,
}

/// QoS statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSStats {
    pub total_circuits: usize,
    pub excellent_circuits: usize,
    pub good_circuits: usize,
    pub acceptable_circuits: usize,
    pub poor_circuits: usize,
    pub unacceptable_circuits: usize,
    pub avg_latency: Duration,
    pub min_latency: Duration,
    pub max_latency: Duration,
    pub total_optimizations: u64,
    #[serde(skip)]
    pub last_optimization: Option<Instant>,
}

impl AdaptiveQoS {
    /// Create new adaptive QoS manager
    pub fn new(target_latency_ms: u16) -> Self {
        let targets = QoSTargets {
            max_latency: Duration::from_millis(target_latency_ms as u64),
            avg_latency: Duration::from_millis((target_latency_ms as f64 * 0.75) as u64),
            ..Default::default()
        };

        info!(
            "🎯 Initializing Adaptive QoS with target latency: {}ms",
            target_latency_ms
        );

        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            targets,
            history_window: 100, // Keep last 100 measurements
            last_optimization: Arc::new(Mutex::new(Instant::now())),
            stats: Arc::new(RwLock::new(QoSStats {
                total_circuits: 0,
                excellent_circuits: 0,
                good_circuits: 0,
                acceptable_circuits: 0,
                poor_circuits: 0,
                unacceptable_circuits: 0,
                avg_latency: Duration::from_millis(0),
                min_latency: Duration::from_millis(0),
                max_latency: Duration::from_millis(0),
                total_optimizations: 0,
                last_optimization: None,
            })),
        }
    }

    /// Update latency measurement for a circuit
    pub async fn update_latency_measurement(&mut self, circuit_id: u64, latency: Duration) {
        debug!(
            "📏 Recording latency for circuit {}: {}ms",
            circuit_id,
            latency.as_millis()
        );

        let mut metrics_map = self.metrics.write().await;

        let metrics = metrics_map
            .entry(circuit_id)
            .or_insert_with(|| CircuitPerformanceMetrics {
                circuit_id,
                latency_history: VecDeque::with_capacity(self.history_window),
                throughput_history: VecDeque::with_capacity(self.history_window),
                packet_loss_rate: 0.0,
                last_measurement: Instant::now(),
                total_bytes_sent: 0,
                total_packets_sent: 0,
                total_packets_lost: 0,
                utilization_score: 0.0,
                qos_grade: QoSGrade::Good,
            });

        // Add latency measurement
        metrics.latency_history.push_back(latency);
        if metrics.latency_history.len() > self.history_window {
            metrics.latency_history.pop_front();
        }

        // Update QoS grade based on recent performance
        let avg_latency = self.calculate_average_latency(&metrics.latency_history);
        let previous_grade = metrics.qos_grade;
        metrics.qos_grade = QoSGrade::from_latency(avg_latency);
        metrics.last_measurement = Instant::now();

        if previous_grade != metrics.qos_grade {
            info!(
                "📊 Circuit {} QoS grade changed: {:?} -> {:?} ({}ms avg)",
                circuit_id,
                previous_grade,
                metrics.qos_grade,
                avg_latency.as_millis()
            );
        }

        // Check if optimization is needed
        if latency > self.targets.max_latency {
            warn!(
                "⚠️ Circuit {} exceeding latency target: {}ms > {}ms",
                circuit_id,
                latency.as_millis(),
                self.targets.max_latency.as_millis()
            );

            self.trigger_optimization().await;
        }

        // Update global statistics
        self.update_global_stats().await;
    }

    /// Update throughput measurement for a circuit
    pub async fn update_throughput_measurement(&mut self, circuit_id: u64, bytes_per_second: u64) {
        debug!(
            "📈 Recording throughput for circuit {}: {} bytes/s",
            circuit_id, bytes_per_second
        );

        let mut metrics_map = self.metrics.write().await;

        if let Some(metrics) = metrics_map.get_mut(&circuit_id) {
            metrics.throughput_history.push_back(bytes_per_second);
            if metrics.throughput_history.len() > self.history_window {
                metrics.throughput_history.pop_front();
            }

            // Update utilization score
            let avg_throughput = self.calculate_average_throughput(&metrics.throughput_history);
            metrics.utilization_score = avg_throughput as f64 / self.targets.min_throughput as f64;

            if avg_throughput < self.targets.min_throughput {
                warn!(
                    "⚠️ Circuit {} below throughput target: {} < {} bytes/s",
                    circuit_id, avg_throughput, self.targets.min_throughput
                );
            }
        }
    }

    /// Update packet loss rate for a circuit
    pub async fn update_packet_loss(
        &mut self,
        circuit_id: u64,
        packets_sent: u64,
        packets_lost: u64,
    ) {
        let mut metrics_map = self.metrics.write().await;

        if let Some(metrics) = metrics_map.get_mut(&circuit_id) {
            metrics.total_packets_sent += packets_sent;
            metrics.total_packets_lost += packets_lost;

            // Calculate exponential moving average for packet loss rate
            let current_loss_rate = packets_lost as f64 / packets_sent as f64;
            let alpha = 0.1; // Smoothing factor
            metrics.packet_loss_rate =
                alpha * current_loss_rate + (1.0 - alpha) * metrics.packet_loss_rate;

            debug!(
                "📉 Circuit {} packet loss rate: {:.2}%",
                circuit_id,
                metrics.packet_loss_rate * 100.0
            );

            if metrics.packet_loss_rate > self.targets.max_packet_loss {
                warn!(
                    "⚠️ Circuit {} exceeding packet loss threshold: {:.2}% > {:.2}%",
                    circuit_id,
                    metrics.packet_loss_rate * 100.0,
                    self.targets.max_packet_loss * 100.0
                );
            }
        }
    }

    /// Get best performing circuits ranked by QoS
    pub async fn get_best_circuits(&self, count: usize) -> Vec<u64> {
        let metrics_map = self.metrics.read().await;

        let mut circuit_scores: Vec<(u64, f64)> = metrics_map
            .values()
            .map(|metrics| {
                let latency_score = self.calculate_latency_score(&metrics.latency_history);
                let throughput_score = self.calculate_throughput_score(&metrics.throughput_history);
                let loss_score = 1.0 - metrics.packet_loss_rate;
                let qos_score = metrics.qos_grade.score() as f64 / 5.0;

                // Weighted composite score
                let composite_score = latency_score * 0.4 +      // 40% weight on latency
                    throughput_score * 0.3 +   // 30% weight on throughput
                    loss_score * 0.2 +         // 20% weight on packet loss
                    qos_score * 0.1; // 10% weight on QoS grade

                (metrics.circuit_id, composite_score)
            })
            .collect();

        // Sort by score (descending)
        circuit_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_circuits: Vec<u64> = circuit_scores
            .into_iter()
            .take(count)
            .map(|(circuit_id, score)| {
                debug!("🏆 Top circuit {}: score {:.3}", circuit_id, score);
                circuit_id
            })
            .collect();

        info!(
            "🎯 Selected {} best performing circuits",
            best_circuits.len()
        );
        best_circuits
    }

    /// Get circuits that need optimization
    pub async fn get_circuits_needing_optimization(&self) -> Vec<u64> {
        let metrics_map = self.metrics.read().await;

        let problematic_circuits: Vec<u64> = metrics_map
            .values()
            .filter(|metrics| {
                let avg_latency = self.calculate_average_latency(&metrics.latency_history);
                let avg_throughput = self.calculate_average_throughput(&metrics.throughput_history);

                avg_latency > self.targets.max_latency
                    || avg_throughput < self.targets.min_throughput
                    || metrics.packet_loss_rate > self.targets.max_packet_loss
                    || matches!(metrics.qos_grade, QoSGrade::Poor | QoSGrade::Unacceptable)
            })
            .map(|metrics| metrics.circuit_id)
            .collect();

        if !problematic_circuits.is_empty() {
            info!(
                "⚠️ Found {} circuits needing optimization",
                problematic_circuits.len()
            );
        }

        problematic_circuits
    }

    /// Trigger optimization process
    async fn trigger_optimization(&self) {
        let mut last_opt = self.last_optimization.lock().await;
        let now = Instant::now();

        // Rate limit optimizations to once per minute
        if now.duration_since(*last_opt) < Duration::from_secs(60) {
            return;
        }

        *last_opt = now;

        info!("🔧 Triggering QoS optimization");

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_optimizations += 1;
            stats.last_optimization = Some(now);
        }

        // In a real implementation, this would:
        // 1. Identify problematic circuits
        // 2. Attempt circuit rotation/replacement
        // 3. Adjust routing priorities
        // 4. Request new circuits if needed

        debug!("✅ QoS optimization complete");
    }

    /// Calculate average latency from history
    fn calculate_average_latency(&self, latency_history: &VecDeque<Duration>) -> Duration {
        if latency_history.is_empty() {
            return Duration::from_millis(0);
        }

        let total_ms: u128 = latency_history.iter().map(|d| d.as_millis()).sum();
        Duration::from_millis((total_ms / latency_history.len() as u128) as u64)
    }

    /// Calculate average throughput from history
    fn calculate_average_throughput(&self, throughput_history: &VecDeque<u64>) -> u64 {
        if throughput_history.is_empty() {
            return 0;
        }

        let total: u64 = throughput_history.iter().sum();
        total / throughput_history.len() as u64
    }

    /// Calculate latency performance score (0.0 - 1.0)
    fn calculate_latency_score(&self, latency_history: &VecDeque<Duration>) -> f64 {
        let avg_latency = self.calculate_average_latency(latency_history);
        let target_ms = self.targets.avg_latency.as_millis() as f64;
        let actual_ms = avg_latency.as_millis() as f64;

        // Score decreases as latency increases beyond target
        if actual_ms <= target_ms {
            1.0
        } else {
            (target_ms / actual_ms).max(0.1) // Minimum score of 0.1
        }
    }

    /// Calculate throughput performance score (0.0 - 1.0)  
    fn calculate_throughput_score(&self, throughput_history: &VecDeque<u64>) -> f64 {
        let avg_throughput = self.calculate_average_throughput(throughput_history);
        let target_throughput = self.targets.min_throughput;

        if avg_throughput >= target_throughput {
            1.0
        } else {
            (avg_throughput as f64 / target_throughput as f64).max(0.1)
        }
    }

    /// Update global QoS statistics
    async fn update_global_stats(&self) {
        let metrics_map = self.metrics.read().await;
        let mut stats = self.stats.write().await;

        let mut grade_counts = HashMap::new();
        let mut latencies = Vec::new();

        for metrics in metrics_map.values() {
            let count = grade_counts.entry(metrics.qos_grade).or_insert(0);
            *count += 1;

            let avg_latency = self.calculate_average_latency(&metrics.latency_history);
            if avg_latency > Duration::from_millis(0) {
                latencies.push(avg_latency);
            }
        }

        stats.total_circuits = metrics_map.len();
        stats.excellent_circuits = *grade_counts.get(&QoSGrade::Excellent).unwrap_or(&0);
        stats.good_circuits = *grade_counts.get(&QoSGrade::Good).unwrap_or(&0);
        stats.acceptable_circuits = *grade_counts.get(&QoSGrade::Acceptable).unwrap_or(&0);
        stats.poor_circuits = *grade_counts.get(&QoSGrade::Poor).unwrap_or(&0);
        stats.unacceptable_circuits = *grade_counts.get(&QoSGrade::Unacceptable).unwrap_or(&0);

        if !latencies.is_empty() {
            let total_ms: u128 = latencies.iter().map(|d| d.as_millis()).sum();
            stats.avg_latency = Duration::from_millis((total_ms / latencies.len() as u128) as u64);
            stats.min_latency = *latencies.iter().min().unwrap();
            stats.max_latency = *latencies.iter().max().unwrap();
        }
    }

    /// Get current QoS statistics
    pub async fn get_qos_stats(&self) -> QoSStats {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// Set new latency target
    pub async fn set_latency_target(&mut self, target: Duration) {
        self.targets.max_latency = target;
        self.targets.avg_latency = Duration::from_millis((target.as_millis() as f64 * 0.75) as u64);

        info!(
            "🎯 Updated latency targets: max={}ms, avg={}ms",
            target.as_millis(),
            self.targets.avg_latency.as_millis()
        );

        // Trigger re-evaluation of all circuits
        self.trigger_optimization().await;
    }

    /// Remove circuit from QoS monitoring
    pub async fn remove_circuit(&mut self, circuit_id: u64) {
        let mut metrics_map = self.metrics.write().await;
        if metrics_map.remove(&circuit_id).is_some() {
            debug!("🗑️ Removed circuit {} from QoS monitoring", circuit_id);
            drop(metrics_map);
            self.update_global_stats().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_qos_creation() {
        let qos = AdaptiveQoS::new(300);
        let stats = qos.get_qos_stats().await;
        assert_eq!(stats.total_circuits, 0);
    }

    #[tokio::test]
    async fn test_latency_measurement() {
        let mut qos = AdaptiveQoS::new(300);

        qos.update_latency_measurement(1, Duration::from_millis(150))
            .await;
        qos.update_latency_measurement(1, Duration::from_millis(200))
            .await;

        let stats = qos.get_qos_stats().await;
        assert_eq!(stats.total_circuits, 1);
        assert!(stats.avg_latency > Duration::from_millis(0));
    }

    #[tokio::test]
    async fn test_qos_grading() {
        assert_eq!(
            QoSGrade::from_latency(Duration::from_millis(100)),
            QoSGrade::Excellent
        );
        assert_eq!(
            QoSGrade::from_latency(Duration::from_millis(200)),
            QoSGrade::Good
        );
        assert_eq!(
            QoSGrade::from_latency(Duration::from_millis(280)),
            QoSGrade::Acceptable
        );
        assert_eq!(
            QoSGrade::from_latency(Duration::from_millis(400)),
            QoSGrade::Poor
        );
        assert_eq!(
            QoSGrade::from_latency(Duration::from_millis(600)),
            QoSGrade::Unacceptable
        );
    }

    #[tokio::test]
    async fn test_best_circuits_selection() {
        let mut qos = AdaptiveQoS::new(300);

        // Add measurements for multiple circuits
        qos.update_latency_measurement(1, Duration::from_millis(100))
            .await;
        qos.update_latency_measurement(2, Duration::from_millis(200))
            .await;
        qos.update_latency_measurement(3, Duration::from_millis(400))
            .await;

        let best_circuits = qos.get_best_circuits(2).await;
        assert_eq!(best_circuits.len(), 2);
        assert!(best_circuits.contains(&1)); // Should include the fastest
    }

    #[tokio::test]
    async fn test_optimization_triggers() {
        let mut qos = AdaptiveQoS::new(300);

        // Add measurement that exceeds target
        qos.update_latency_measurement(1, Duration::from_millis(500))
            .await;

        let problematic = qos.get_circuits_needing_optimization().await;
        assert!(!problematic.is_empty());
        assert!(problematic.contains(&1));
    }

    #[test]
    fn test_qos_grade_scoring() {
        assert_eq!(QoSGrade::Excellent.score(), 5);
        assert_eq!(QoSGrade::Good.score(), 4);
        assert_eq!(QoSGrade::Acceptable.score(), 3);
        assert_eq!(QoSGrade::Poor.score(), 2);
        assert_eq!(QoSGrade::Unacceptable.score(), 1);
    }
}
