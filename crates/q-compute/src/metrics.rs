//! Prometheus Metrics Export for q-compute
//!
//! Exposes CPU, GPU, RAM, network, disk, per-layer core allocation,
//! and trainer cheat gauges in Prometheus text exposition format.
//!
//! Gated behind the `metrics` cargo feature so nodes that don't
//! need scraping pay zero overhead.

use crate::ResourceSnapshot;
use prometheus::{
    Encoder, GaugeVec, IntGauge, Opts, Registry, TextEncoder,
};
use std::sync::OnceLock;
use tracing::warn;

/// Global metrics registry (initialized once on first access)
static METRICS: OnceLock<ComputeMetrics> = OnceLock::new();

/// All Prometheus metrics exported by q-compute.
pub struct ComputeMetrics {
    registry: Registry,

    // ---- resource gauges ----
    cpu_usage_percent: prometheus::Gauge,
    gpu_usage_percent: prometheus::Gauge,
    ram_usage_bytes: prometheus::Gauge,
    net_rx_bytes_total: prometheus::Gauge,
    net_tx_bytes_total: prometheus::Gauge,
    disk_read_bytes_total: prometheus::Gauge,
    disk_write_bytes_total: prometheus::Gauge,

    // ---- per-layer core assignment ----
    layer_cores: GaugeVec,

    // ---- trainer ----
    trainer_cheats_active: IntGauge,
}

impl ComputeMetrics {
    /// Create and register all metrics against a fresh `Registry`.
    fn new() -> Self {
        let registry = Registry::new();

        let cpu_usage_percent = prometheus::Gauge::new(
            "qnk_compute_cpu_usage_percent",
            "Overall CPU utilization percentage (0-100)",
        )
        .expect("valid metric");

        let gpu_usage_percent = prometheus::Gauge::new(
            "qnk_compute_gpu_usage_percent",
            "GPU utilization percentage (0-100, 0 if no GPU)",
        )
        .expect("valid metric");

        let ram_usage_bytes = prometheus::Gauge::new(
            "qnk_compute_ram_usage_bytes",
            "RAM currently in use (bytes)",
        )
        .expect("valid metric");

        let net_rx_bytes_total = prometheus::Gauge::new(
            "qnk_compute_net_rx_bytes_total",
            "Network receive rate (bytes/sec)",
        )
        .expect("valid metric");

        let net_tx_bytes_total = prometheus::Gauge::new(
            "qnk_compute_net_tx_bytes_total",
            "Network transmit rate (bytes/sec)",
        )
        .expect("valid metric");

        let disk_read_bytes_total = prometheus::Gauge::new(
            "qnk_compute_disk_read_bytes_total",
            "Disk I/O read+write rate (bytes/sec)",
        )
        .expect("valid metric");

        let disk_write_bytes_total = prometheus::Gauge::new(
            "qnk_compute_disk_write_bytes_total",
            "Disk I/O write rate (bytes/sec, currently combined with read)",
        )
        .expect("valid metric");

        let layer_cores = GaugeVec::new(
            Opts::new(
                "qnk_compute_layer_cores",
                "Number of CPU cores assigned to each compute layer",
            ),
            &["layer"],
        )
        .expect("valid metric");

        let trainer_cheats_active = IntGauge::new(
            "qnk_compute_trainer_cheats_active",
            "Number of trainer cheats currently active",
        )
        .expect("valid metric");

        // Register everything
        registry.register(Box::new(cpu_usage_percent.clone())).expect("register");
        registry.register(Box::new(gpu_usage_percent.clone())).expect("register");
        registry.register(Box::new(ram_usage_bytes.clone())).expect("register");
        registry.register(Box::new(net_rx_bytes_total.clone())).expect("register");
        registry.register(Box::new(net_tx_bytes_total.clone())).expect("register");
        registry.register(Box::new(disk_read_bytes_total.clone())).expect("register");
        registry.register(Box::new(disk_write_bytes_total.clone())).expect("register");
        registry.register(Box::new(layer_cores.clone())).expect("register");
        registry.register(Box::new(trainer_cheats_active.clone())).expect("register");

        Self {
            registry,
            cpu_usage_percent,
            gpu_usage_percent,
            ram_usage_bytes,
            net_rx_bytes_total,
            net_tx_bytes_total,
            disk_read_bytes_total,
            disk_write_bytes_total,
            layer_cores,
            trainer_cheats_active,
        }
    }

    /// Update all resource gauges from a `ResourceSnapshot`.
    pub fn update_from_snapshot(&self, snapshot: &ResourceSnapshot) {
        self.cpu_usage_percent.set(snapshot.cpu_total as f64);
        self.gpu_usage_percent.set(snapshot.gpu_utilization as f64);
        self.ram_usage_bytes.set(snapshot.ram_used as f64);
        self.net_rx_bytes_total.set(snapshot.net_rx_bps as f64);
        self.net_tx_bytes_total.set(snapshot.net_tx_bps as f64);
        // disk_io_bps is combined read+write; split evenly as a reasonable default
        // since /proc/diskstats is already aggregated in resource_monitor.
        self.disk_read_bytes_total.set((snapshot.disk_io_bps / 2) as f64);
        self.disk_write_bytes_total.set((snapshot.disk_io_bps / 2) as f64);
    }

    /// Update per-layer core assignment gauge.
    ///
    /// `layers` is an iterator of `(layer_name, cores_assigned)` pairs,
    /// matching the output of `ComputeStatus::layers`.
    pub fn update_layer_cores<'a, I>(&self, layers: I)
    where
        I: IntoIterator<Item = (&'a str, u32)>,
    {
        for (name, cores) in layers {
            self.layer_cores
                .with_label_values(&[name])
                .set(cores as f64);
        }
    }

    /// Update the trainer cheats active counter.
    pub fn update_trainer_cheats(&self, count: i64) {
        self.trainer_cheats_active.set(count);
    }

    /// Encode all registered metrics into Prometheus text exposition format.
    fn gather(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buf = Vec::new();
        if let Err(e) = encoder.encode(&metric_families, &mut buf) {
            warn!("Failed to encode Prometheus metrics: {}", e);
            return String::new();
        }
        String::from_utf8(buf).unwrap_or_default()
    }
}

/// Get the global `ComputeMetrics` singleton (lazily initialized).
pub fn global_metrics() -> &'static ComputeMetrics {
    METRICS.get_or_init(ComputeMetrics::new)
}

/// Convenience: update metrics from a snapshot + layer info + trainer count,
/// then return the Prometheus text exposition.
///
/// This is the main entry point for `/metrics` HTTP handlers.
pub fn gather_metrics() -> String {
    global_metrics().gather()
}

/// Update all metrics from a full `ComputeStatus`.
///
/// Call this periodically (e.g., once per second from the orchestrator loop)
/// to keep metrics fresh.
pub fn update_all(snapshot: &ResourceSnapshot, layers: &[(String, crate::LayerStats)], trainer_cheats_count: usize) {
    let m = global_metrics();
    m.update_from_snapshot(snapshot);
    m.update_layer_cores(
        layers.iter().map(|(name, stats)| (name.as_str(), stats.cores_assigned)),
    );
    m.update_trainer_cheats(trainer_cheats_count as i64);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot() -> ResourceSnapshot {
        ResourceSnapshot {
            cpu_per_core: vec![45.0, 50.0, 55.0, 60.0],
            cpu_total: 52.5,
            gpu_utilization: 78.3,
            gpu_memory_used: 4_000_000_000,
            gpu_memory_total: 8_000_000_000,
            gpu_temperature: 65.0,
            gpu_name: "NVIDIA GeForce RTX 4090".to_string(),
            ram_used: 16_000_000_000,
            ram_total: 32_000_000_000,
            net_tx_bps: 50_000_000,
            net_rx_bps: 120_000_000,
            net_capacity_bps: 1_250_000_000,
            disk_io_bps: 200_000_000,
            timestamp_ms: 1710000000000,
        }
    }

    #[test]
    fn test_metrics_creation() {
        let m = ComputeMetrics::new();
        // Metrics start at zero
        let output = m.gather();
        assert!(output.contains("qnk_compute_cpu_usage_percent"));
        assert!(output.contains("qnk_compute_gpu_usage_percent"));
        assert!(output.contains("qnk_compute_ram_usage_bytes"));
        assert!(output.contains("qnk_compute_net_rx_bytes_total"));
        assert!(output.contains("qnk_compute_net_tx_bytes_total"));
        assert!(output.contains("qnk_compute_trainer_cheats_active"));
    }

    #[test]
    fn test_update_from_snapshot() {
        let m = ComputeMetrics::new();
        let snap = make_snapshot();
        m.update_from_snapshot(&snap);

        let output = m.gather();
        // CPU should be 52.5
        assert!(output.contains("qnk_compute_cpu_usage_percent 52.5"));
        // GPU should be 78.3
        assert!(output.contains("qnk_compute_gpu_usage_percent 78.3"));
        // RAM should be 16 billion (format varies: 1.6e10 or 16000000000)
        assert!(output.contains("qnk_compute_ram_usage_bytes"));
        // Net RX should be 120_000_000
        assert!(output.contains("qnk_compute_net_rx_bytes_total"));
    }

    #[test]
    fn test_layer_cores_labels() {
        let m = ComputeMetrics::new();
        m.update_layer_cores([
            ("Mining", 6),
            ("AI Inference", 2),
            ("ZK Proofs", 0),
        ]);

        let output = m.gather();
        assert!(output.contains("qnk_compute_layer_cores{layer=\"Mining\"} 6"));
        assert!(output.contains("qnk_compute_layer_cores{layer=\"AI Inference\"} 2"));
        assert!(output.contains("qnk_compute_layer_cores{layer=\"ZK Proofs\"} 0"));
    }

    #[test]
    fn test_trainer_cheats_counter() {
        let m = ComputeMetrics::new();
        m.update_trainer_cheats(5);
        let output = m.gather();
        assert!(output.contains("qnk_compute_trainer_cheats_active 5"));
    }

    #[test]
    fn test_gather_metrics_function() {
        // This exercises the global singleton path
        let output = gather_metrics();
        assert!(output.contains("qnk_compute_cpu_usage_percent"));
    }

    #[test]
    fn test_update_all() {
        let snap = make_snapshot();
        let layers = vec![
            ("Mining".to_string(), crate::LayerStats {
                cores_assigned: 8,
                tasks_completed: crate::AtomicU64Ser(100),
                tasks_pending: 2,
                revenue_micro_qug: 5000,
                active_since_ms: 1000,
            }),
            ("AI Inference".to_string(), crate::LayerStats {
                cores_assigned: 4,
                tasks_completed: crate::AtomicU64Ser(50),
                tasks_pending: 0,
                revenue_micro_qug: 2000,
                active_since_ms: 2000,
            }),
        ];
        update_all(&snap, &layers, 3);

        let output = gather_metrics();
        assert!(output.contains("qnk_compute_cpu_usage_percent"));
        assert!(output.contains("qnk_compute_trainer_cheats_active 3"));
    }
}
