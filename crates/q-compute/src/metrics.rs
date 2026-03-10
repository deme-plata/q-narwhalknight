//! Prometheus Metrics Export for q-compute
//!
//! Exposes CPU, GPU, RAM, network, disk, per-layer core allocation,
//! trainer cheat gauges, job queue depth, tunnel statistics, and latency
//! histograms in Prometheus text exposition format.
//!
//! Gated behind the `metrics` cargo feature so nodes that don't
//! need scraping pay zero overhead.
//!
//! ## Design
//!
//! All metrics are implemented with `AtomicU64` / `AtomicI64` counters and
//! manual histogram buckets — **no** external `prometheus` crate dependency.
//! Values that represent fractional quantities (e.g., seconds, percentages)
//! are stored as fixed-point micro-units (1e-6) in the atomic and converted
//! on render.

use crate::{ComputeLayer, LayerStats, ResourceSnapshot};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

// ---------------------------------------------------------------------------
// MetricValue enum (issue #028 requirement)
// ---------------------------------------------------------------------------

/// Typed metric value for generic metric storage.
pub enum MetricValue {
    /// Monotonically increasing counter.
    Counter(AtomicU64),
    /// Gauge that can go up and down.
    Gauge(AtomicI64),
    /// Distribution of observations into fixed buckets.
    Histogram {
        /// Upper bounds for each finite bucket (in natural units).
        buckets: Vec<f64>,
        /// Cumulative count per bucket + 1 for +Inf (len = buckets.len() + 1).
        counts: Vec<AtomicU64>,
        /// Running sum in micro-units (value * 1_000_000).
        sum: AtomicU64,
    },
}

impl MetricValue {
    /// Create a counter initialized to zero.
    pub fn counter() -> Self {
        MetricValue::Counter(AtomicU64::new(0))
    }

    /// Create a gauge initialized to zero.
    pub fn gauge() -> Self {
        MetricValue::Gauge(AtomicI64::new(0))
    }

    /// Create a histogram with the given bucket upper bounds.
    pub fn histogram(buckets: Vec<f64>) -> Self {
        let n = buckets.len();
        let counts = (0..=n).map(|_| AtomicU64::new(0)).collect();
        MetricValue::Histogram {
            buckets,
            counts,
            sum: AtomicU64::new(0),
        }
    }

    /// Increment a counter by 1. No-op for other types.
    pub fn inc_counter(&self) {
        if let MetricValue::Counter(c) = self {
            c.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get counter value. Returns 0 for non-counter types.
    pub fn counter_value(&self) -> u64 {
        match self {
            MetricValue::Counter(c) => c.load(Ordering::Relaxed),
            _ => 0,
        }
    }

    /// Set a gauge to the given value. No-op for other types.
    pub fn set_gauge(&self, v: i64) {
        if let MetricValue::Gauge(g) = self {
            g.store(v, Ordering::Relaxed);
        }
    }

    /// Get gauge value. Returns 0 for non-gauge types.
    pub fn gauge_value(&self) -> i64 {
        match self {
            MetricValue::Gauge(g) => g.load(Ordering::Relaxed),
            _ => 0,
        }
    }

    /// Record an observation in a histogram. No-op for other types.
    pub fn observe(&self, value: f64) {
        if let MetricValue::Histogram {
            buckets,
            counts,
            sum,
        } = self
        {
            let value_us = (value * 1_000_000.0) as u64;
            sum.fetch_add(value_us, Ordering::Relaxed);
            for (i, &bound) in buckets.iter().enumerate() {
                if value <= bound {
                    counts[i].fetch_add(1, Ordering::Relaxed);
                }
            }
            // +Inf bucket always incremented
            counts[buckets.len()].fetch_add(1, Ordering::Relaxed);
        }
    }
}

// ---------------------------------------------------------------------------
// Histogram helper
// ---------------------------------------------------------------------------

/// A Prometheus-style histogram backed by atomic counters.
///
/// Values are recorded in seconds (f64).  Internally each bucket boundary is
/// stored as micro-seconds (u64) so we can use `AtomicU64` for lock-free
/// updates.  An `+Inf` bucket is always appended automatically.
pub struct AtomicHistogram {
    /// Upper-bound of each finite bucket, in **micro-seconds** (1e-6 s).
    bucket_bounds_us: Vec<u64>,
    /// Cumulative count for each bucket (len = bucket_bounds_us.len() + 1 for +Inf).
    bucket_counts: Vec<AtomicU64>,
    /// Running sum in micro-seconds.
    sum_us: AtomicU64,
    /// Total observation count (== +Inf bucket value).
    count: AtomicU64,
}

impl AtomicHistogram {
    /// Create a histogram with the given bucket upper-bounds **in seconds**.
    fn new(bounds_secs: &[f64]) -> Self {
        let bucket_bounds_us: Vec<u64> = bounds_secs
            .iter()
            .map(|s| (*s * 1_000_000.0) as u64)
            .collect();
        // +1 for the implicit +Inf bucket
        let bucket_counts: Vec<AtomicU64> = (0..bucket_bounds_us.len() + 1)
            .map(|_| AtomicU64::new(0))
            .collect();
        Self {
            bucket_bounds_us,
            bucket_counts,
            sum_us: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Record a single observation (in seconds).
    pub fn observe(&self, value_secs: f64) {
        let value_us = (value_secs * 1_000_000.0) as u64;
        self.sum_us.fetch_add(value_us, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        for (i, bound) in self.bucket_bounds_us.iter().enumerate() {
            if value_us <= *bound {
                self.bucket_counts[i].fetch_add(1, Ordering::Relaxed);
            }
        }
        // +Inf bucket always gets incremented
        self.bucket_counts
            .last()
            .unwrap()
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Render Prometheus exposition lines for this histogram.
    ///
    /// `name` is the metric base name (e.g. `qnk_compute_job_duration_seconds`).
    /// `help` is the HELP string.
    /// `labels` is an optional label set like `{layer="Mining"}` (empty string
    /// if no extra labels).
    fn render(&self, name: &str, help: &str, labels: &str, out: &mut String) {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} histogram");
        self.render_data(name, labels, out);
    }

    /// Render just the data lines (no HELP/TYPE) for per-label histogram series.
    ///
    /// Note: `observe()` already stores cumulative counts in each bucket
    /// (every bucket with `bound >= value` is incremented), so we emit
    /// the raw atomic values directly without re-accumulating.
    fn render_data(&self, name: &str, labels: &str, out: &mut String) {
        for (i, bound_us) in self.bucket_bounds_us.iter().enumerate() {
            let count = self.bucket_counts[i].load(Ordering::Relaxed);
            let le = *bound_us as f64 / 1_000_000.0;
            if labels.is_empty() {
                let _ = writeln!(out, "{name}_bucket{{le=\"{le}\"}} {count}");
            } else {
                let inner = labels.trim_start_matches('{').trim_end_matches('}');
                let _ = writeln!(
                    out,
                    "{name}_bucket{{{inner},le=\"{le}\"}} {count}"
                );
            }
        }
        // +Inf bucket (== total count, since observe always increments it)
        let inf_count = self.bucket_counts.last().unwrap().load(Ordering::Relaxed);
        if labels.is_empty() {
            let _ = writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {inf_count}");
        } else {
            let inner = labels.trim_start_matches('{').trim_end_matches('}');
            let _ = writeln!(
                out,
                "{name}_bucket{{{inner},le=\"+Inf\"}} {inf_count}"
            );
        }

        let sum = self.sum_us.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let count = self.count.load(Ordering::Relaxed);
        if labels.is_empty() {
            let _ = writeln!(out, "{name}_sum {sum}");
            let _ = writeln!(out, "{name}_count {count}");
        } else {
            let _ = writeln!(out, "{name}_sum{labels} {sum}");
            let _ = writeln!(out, "{name}_count{labels} {count}");
        }
    }

    /// Return total observation count.
    pub fn total_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Return sum of observations in seconds.
    pub fn total_sum_secs(&self) -> f64 {
        self.sum_us.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }
}

// ---------------------------------------------------------------------------
// Rendering helpers for simple metric types
// ---------------------------------------------------------------------------

fn render_gauge_f64(name: &str, help: &str, value: f64, labels: &str, out: &mut String) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} gauge");
    if labels.is_empty() {
        let _ = writeln!(out, "{name} {value}");
    } else {
        let _ = writeln!(out, "{name}{labels} {value}");
    }
}

fn render_gauge_i64(name: &str, help: &str, value: i64, labels: &str, out: &mut String) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} gauge");
    if labels.is_empty() {
        let _ = writeln!(out, "{name} {value}");
    } else {
        let _ = writeln!(out, "{name}{labels} {value}");
    }
}

fn render_counter_u64(name: &str, help: &str, value: u64, labels: &str, out: &mut String) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} counter");
    if labels.is_empty() {
        let _ = writeln!(out, "{name} {value}");
    } else {
        let _ = writeln!(out, "{name}{labels} {value}");
    }
}

// ---------------------------------------------------------------------------
// Task-duration histogram buckets (per-layer)
// ---------------------------------------------------------------------------

/// Standard task-duration histogram buckets (in seconds) as specified by issue #028.
const TASK_DURATION_BUCKETS: &[f64] = &[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 30.0, 60.0];

// ---------------------------------------------------------------------------
// Per-layer atomic metric set
// ---------------------------------------------------------------------------

struct PerLayerMetrics {
    tasks_total: AtomicU64,            // counter: total tasks started
    tasks_active: AtomicI64,           // gauge: currently in-flight tasks
    task_duration: AtomicHistogram,    // histogram: task duration in seconds
    cpu_seconds_us: AtomicU64,         // counter, stored as micro-seconds
    gpu_seconds_us: AtomicU64,         // counter, stored as micro-seconds
    revenue_micro_qug: AtomicU64,      // counter
    cores_assigned: AtomicU64,         // gauge
}

impl PerLayerMetrics {
    fn new() -> Self {
        Self {
            tasks_total: AtomicU64::new(0),
            tasks_active: AtomicI64::new(0),
            task_duration: AtomicHistogram::new(TASK_DURATION_BUCKETS),
            cpu_seconds_us: AtomicU64::new(0),
            gpu_seconds_us: AtomicU64::new(0),
            revenue_micro_qug: AtomicU64::new(0),
            cores_assigned: AtomicU64::new(0),
        }
    }
}

// ---------------------------------------------------------------------------
// Tunnel metrics
// ---------------------------------------------------------------------------

struct TunnelMetrics {
    peers_connected: AtomicI64,             // gauge
    bytes_sent_total: AtomicU64,            // counter
    bytes_received_total: AtomicU64,        // counter
    bandwidth_bytes_total: AtomicU64,       // counter (legacy combined)
    rekey_total: AtomicU64,                 // counter
    rtt_histogram: AtomicHistogram,         // histogram (seconds)
}

impl TunnelMetrics {
    fn new() -> Self {
        // RTT buckets in seconds: 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 5s
        let rtt_buckets = vec![
            0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0, 5.0,
        ];
        Self {
            peers_connected: AtomicI64::new(0),
            bytes_sent_total: AtomicU64::new(0),
            bytes_received_total: AtomicU64::new(0),
            bandwidth_bytes_total: AtomicU64::new(0),
            rekey_total: AtomicU64::new(0),
            rtt_histogram: AtomicHistogram::new(&rtt_buckets),
        }
    }
}

// ---------------------------------------------------------------------------
// Job queue metrics
// ---------------------------------------------------------------------------

/// Job statuses tracked for queue depth.
const JOB_STATUSES: &[&str] = &["pending", "running", "completed", "failed"];

struct JobMetrics {
    /// Queue depth gauge per status index (same order as JOB_STATUSES).
    queue_depth: [AtomicI64; 4],
    /// Completion counter per compute-layer index (8 layers).
    completions: [AtomicU64; 8],
    /// Failure counter per compute-layer index (8 layers).
    failures: [AtomicU64; 8],
}

impl JobMetrics {
    fn new() -> Self {
        Self {
            queue_depth: [
                AtomicI64::new(0),
                AtomicI64::new(0),
                AtomicI64::new(0),
                AtomicI64::new(0),
            ],
            completions: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            failures: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// GPU device metrics (up to 8 GPUs)
// ---------------------------------------------------------------------------

const MAX_GPUS: usize = 8;

struct GpuDeviceMetrics {
    /// Utilization 0-100 per device, stored as micro-units.
    utilization_percent: [AtomicU64; MAX_GPUS],
    /// VRAM used in bytes per device.
    vram_used_bytes: [AtomicU64; MAX_GPUS],
    /// Temperature in milli-Celsius (to avoid float atomics).
    temperature_milli_c: [AtomicU64; MAX_GPUS],
    /// Number of devices currently tracked.
    device_count: AtomicU64,
}

impl GpuDeviceMetrics {
    fn new() -> Self {
        Self {
            utilization_percent: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            vram_used_bytes: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            temperature_milli_c: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            device_count: AtomicU64::new(0),
        }
    }
}

// ---------------------------------------------------------------------------
// Reputation tracking (per-peer)
// ---------------------------------------------------------------------------

struct ReputationMetrics {
    /// Peer ID -> reputation score stored as fixed-point (score * 1_000_000).
    scores: RwLock<HashMap<String, u64>>,
}

impl ReputationMetrics {
    fn new() -> Self {
        Self {
            scores: RwLock::new(HashMap::new()),
        }
    }
}

// ---------------------------------------------------------------------------
// ComputeMetrics — the big struct
// ---------------------------------------------------------------------------

/// All Prometheus metrics exported by q-compute.
///
/// Thread-safe: all interior fields are `Atomic*` or behind `RwLock`.
///
/// Metric naming follows the Prometheus convention:
/// `qnk_compute_<subsystem>_<unit>_<suffix>`.
pub struct ComputeMetrics {
    // ---- resource gauges (stored as micro-units for f64 precision) ----
    cpu_usage_percent_u: AtomicU64,     // value * 1_000_000
    gpu_usage_percent_u: AtomicU64,
    gpu_temperature_u: AtomicU64,       // celsius * 1_000_000
    ram_usage_bytes: AtomicU64,
    net_rx_bytes_total: AtomicU64,
    net_tx_bytes_total: AtomicU64,
    disk_read_bytes_total: AtomicU64,
    disk_write_bytes_total: AtomicU64,

    // ---- per-layer ----
    /// Index by `ComputeLayer as usize` (0..8)
    per_layer: [PerLayerMetrics; 8],

    // ---- trainer ----
    trainer_cheats_active: AtomicI64,

    // ---- job duration histogram (global) ----
    /// p50/p95/p99-friendly buckets for job execution time.
    job_duration: AtomicHistogram,

    // ---- tunnel ----
    tunnel: TunnelMetrics,

    // ---- job queue ----
    jobs: JobMetrics,

    // ---- GPU devices ----
    gpu_devices: GpuDeviceMetrics,

    // ---- P2P ----
    peer_count: AtomicI64,

    // ---- Reputation ----
    reputation: ReputationMetrics,
}

impl ComputeMetrics {
    /// Create a fresh set of zeroed metrics.
    fn new() -> Self {
        // Job duration buckets (seconds): 0.001, 0.005, 0.01, 0.025, 0.05,
        // 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0
        let job_buckets = vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
            120.0,
        ];

        Self {
            cpu_usage_percent_u: AtomicU64::new(0),
            gpu_usage_percent_u: AtomicU64::new(0),
            gpu_temperature_u: AtomicU64::new(0),
            ram_usage_bytes: AtomicU64::new(0),
            net_rx_bytes_total: AtomicU64::new(0),
            net_tx_bytes_total: AtomicU64::new(0),
            disk_read_bytes_total: AtomicU64::new(0),
            disk_write_bytes_total: AtomicU64::new(0),
            per_layer: [
                PerLayerMetrics::new(),
                PerLayerMetrics::new(),
                PerLayerMetrics::new(),
                PerLayerMetrics::new(),
                PerLayerMetrics::new(),
                PerLayerMetrics::new(),
                PerLayerMetrics::new(),
                PerLayerMetrics::new(),
            ],
            trainer_cheats_active: AtomicI64::new(0),
            job_duration: AtomicHistogram::new(&job_buckets),
            tunnel: TunnelMetrics::new(),
            jobs: JobMetrics::new(),
            gpu_devices: GpuDeviceMetrics::new(),
            peer_count: AtomicI64::new(0),
            reputation: ReputationMetrics::new(),
        }
    }

    // ======================================================================
    // Issue #028 required methods
    // ======================================================================

    /// Record the start of a task on the given compute layer.
    ///
    /// Increments `tasks_total` (counter) and `tasks_active` (gauge).
    pub fn record_task_start(&self, layer: ComputeLayer) {
        let pl = &self.per_layer[layer as usize];
        pl.tasks_total.fetch_add(1, Ordering::Relaxed);
        pl.tasks_active.fetch_add(1, Ordering::Relaxed);
    }

    /// Record the completion of a task on the given compute layer.
    ///
    /// Decrements `tasks_active` and records the duration in the per-layer
    /// `task_duration_seconds` histogram.
    pub fn record_task_end(&self, layer: ComputeLayer, duration_secs: f64) {
        let pl = &self.per_layer[layer as usize];
        pl.tasks_active.fetch_sub(1, Ordering::Relaxed);
        pl.task_duration.observe(duration_secs);
    }

    /// Bulk-update resource gauges from a `ResourceSnapshot`.
    pub fn update_resources(&self, snapshot: &ResourceSnapshot) {
        self.update_from_snapshot(snapshot);
    }

    /// Update tunnel aggregate metrics.
    ///
    /// `count` sets the tunnel gauge. `bytes_sent` and `bytes_recv` are
    /// added to their respective counters (cumulative).
    pub fn update_tunnels(&self, count: i64, bytes_sent: u64, bytes_recv: u64) {
        self.tunnel
            .peers_connected
            .store(count, Ordering::Relaxed);
        self.tunnel
            .bytes_sent_total
            .fetch_add(bytes_sent, Ordering::Relaxed);
        self.tunnel
            .bytes_received_total
            .fetch_add(bytes_recv, Ordering::Relaxed);
    }

    /// Render all metrics in Prometheus text exposition format.
    ///
    /// Returns a `String` suitable for `Content-Type: text/plain; version=0.0.4`.
    ///
    /// ```text
    /// # HELP qnk_compute_tasks_total Total compute tasks
    /// # TYPE qnk_compute_tasks_total counter
    /// qnk_compute_tasks_total{layer="Mining"} 12345
    /// qnk_compute_tasks_total{layer="AI Inference"} 678
    /// ```
    pub fn render_prometheus(&self) -> String {
        self.gather()
    }

    // ======================================================================
    // Existing public update API (backward-compatible)
    // ======================================================================

    /// Update all resource gauges from a `ResourceSnapshot`.
    pub fn update_from_snapshot(&self, snapshot: &ResourceSnapshot) {
        self.cpu_usage_percent_u
            .store(f64_to_micro(snapshot.cpu_total as f64), Ordering::Relaxed);
        self.gpu_usage_percent_u
            .store(f64_to_micro(snapshot.gpu_utilization as f64), Ordering::Relaxed);
        self.gpu_temperature_u
            .store(f64_to_micro(snapshot.gpu_temperature as f64), Ordering::Relaxed);
        self.ram_usage_bytes
            .store(snapshot.ram_used, Ordering::Relaxed);
        self.net_rx_bytes_total
            .store(snapshot.net_rx_bps, Ordering::Relaxed);
        self.net_tx_bytes_total
            .store(snapshot.net_tx_bps, Ordering::Relaxed);
        self.disk_read_bytes_total
            .store(snapshot.disk_io_bps / 2, Ordering::Relaxed);
        self.disk_write_bytes_total
            .store(snapshot.disk_io_bps / 2, Ordering::Relaxed);

        // Also populate single-GPU device metrics from the snapshot for
        // backwards compatibility (device index 0).
        if snapshot.gpu_utilization > 0.0 || snapshot.gpu_memory_used > 0 {
            self.set_gpu_device_count(1);
            self.set_gpu_utilization(0, snapshot.gpu_utilization as f64);
            self.set_gpu_vram_used(0, snapshot.gpu_memory_used);
            self.set_gpu_temperature(0, snapshot.gpu_temperature as f64);
        }
    }

    /// Update per-layer core assignment gauge.
    pub fn update_layer_cores<'a, I>(&self, layers: I)
    where
        I: IntoIterator<Item = (&'a str, u32)>,
    {
        for (name, cores) in layers {
            if let Some(idx) = layer_name_to_index(name) {
                self.per_layer[idx]
                    .cores_assigned
                    .store(cores as u64, Ordering::Relaxed);
            }
        }
    }

    /// Update the trainer cheats active counter.
    pub fn update_trainer_cheats(&self, count: i64) {
        self.trainer_cheats_active.store(count, Ordering::Relaxed);
    }

    // ---- Per-layer counters ----

    /// Add CPU seconds consumed by a compute layer.
    pub fn add_layer_cpu_seconds(&self, layer: ComputeLayer, secs: f64) {
        let us = (secs * 1_000_000.0) as u64;
        self.per_layer[layer as usize]
            .cpu_seconds_us
            .fetch_add(us, Ordering::Relaxed);
    }

    /// Add GPU seconds consumed by a compute layer.
    pub fn add_layer_gpu_seconds(&self, layer: ComputeLayer, secs: f64) {
        let us = (secs * 1_000_000.0) as u64;
        self.per_layer[layer as usize]
            .gpu_seconds_us
            .fetch_add(us, Ordering::Relaxed);
    }

    /// Add revenue (in micro-QUG) earned by a compute layer.
    pub fn add_layer_revenue(&self, layer: ComputeLayer, micro_qug: u64) {
        self.per_layer[layer as usize]
            .revenue_micro_qug
            .fetch_add(micro_qug, Ordering::Relaxed);
    }

    // ---- Job duration histogram (global) ----

    /// Record a job execution time in seconds.
    pub fn observe_job_duration(&self, secs: f64) {
        self.job_duration.observe(secs);
    }

    // ---- Tunnel metrics ----

    /// Set the current number of connected tunnel peers (absolute value).
    pub fn set_tunnel_peers(&self, count: i64) {
        self.tunnel.peers_connected.store(count, Ordering::Relaxed);
    }

    /// Record a tunnel RTT measurement in seconds.
    pub fn observe_tunnel_rtt(&self, secs: f64) {
        self.tunnel.rtt_histogram.observe(secs);
    }

    /// Add bytes transferred through tunnels (combined).
    pub fn add_tunnel_bandwidth(&self, bytes: u64) {
        self.tunnel
            .bandwidth_bytes_total
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// Increment the tunnel re-key counter.
    pub fn inc_tunnel_rekey(&self) {
        self.tunnel.rekey_total.fetch_add(1, Ordering::Relaxed);
    }

    // ---- Job queue metrics ----

    /// Set the job queue depth for a given status.
    /// `status_index`: 0=pending, 1=running, 2=completed, 3=failed
    pub fn set_job_queue_depth(&self, status_index: usize, depth: i64) {
        if status_index < JOB_STATUSES.len() {
            self.jobs.queue_depth[status_index].store(depth, Ordering::Relaxed);
        }
    }

    /// Increment the job completion counter for a layer.
    pub fn inc_job_completions(&self, layer: ComputeLayer) {
        self.jobs.completions[layer as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the job failure counter for a layer.
    pub fn inc_job_failures(&self, layer: ComputeLayer) {
        self.jobs.failures[layer as usize].fetch_add(1, Ordering::Relaxed);
    }

    // ---- GPU device metrics ----

    /// Set the number of GPU devices being tracked.
    pub fn set_gpu_device_count(&self, count: u64) {
        let capped = count.min(MAX_GPUS as u64);
        self.gpu_devices.device_count.store(capped, Ordering::Relaxed);
    }

    /// Set GPU utilization (0-100) for a specific device.
    pub fn set_gpu_utilization(&self, device: usize, percent: f64) {
        if device < MAX_GPUS {
            self.gpu_devices.utilization_percent[device]
                .store(f64_to_micro(percent), Ordering::Relaxed);
        }
    }

    /// Set VRAM used in bytes for a specific device.
    pub fn set_gpu_vram_used(&self, device: usize, bytes: u64) {
        if device < MAX_GPUS {
            self.gpu_devices.vram_used_bytes[device].store(bytes, Ordering::Relaxed);
        }
    }

    /// Set GPU temperature in Celsius for a specific device.
    pub fn set_gpu_temperature(&self, device: usize, celsius: f64) {
        if device < MAX_GPUS {
            let milli = (celsius * 1000.0) as u64;
            self.gpu_devices.temperature_milli_c[device].store(milli, Ordering::Relaxed);
        }
    }

    // ---- P2P peer count ----

    /// Set the current connected peer count.
    pub fn set_peer_count(&self, count: i64) {
        self.peer_count.store(count, Ordering::Relaxed);
    }

    // ---- Reputation ----

    /// Set the reputation score for a given peer.
    pub fn set_reputation(&self, peer_id: &str, score: f64) {
        if let Ok(mut map) = self.reputation.scores.write() {
            map.insert(peer_id.to_string(), f64_to_micro(score));
        }
    }

    // ======================================================================
    // Rendering
    // ======================================================================

    /// Encode all registered metrics into Prometheus text exposition format.
    fn gather(&self) -> String {
        let mut out = String::with_capacity(8192);

        // ---- Per-layer: tasks_total (counter) ----
        {
            let _ = writeln!(
                out,
                "# HELP qnk_compute_tasks_total Total compute tasks"
            );
            let _ = writeln!(out, "# TYPE qnk_compute_tasks_total counter");
            for layer in ComputeLayer::all() {
                let idx = *layer as usize;
                let val = self.per_layer[idx].tasks_total.load(Ordering::Relaxed);
                let name = layer.name();
                let _ = writeln!(
                    out,
                    "qnk_compute_tasks_total{{layer=\"{name}\"}} {val}"
                );
            }
        }

        // ---- Per-layer: tasks_active (gauge) ----
        {
            let _ = writeln!(
                out,
                "# HELP qnk_compute_tasks_active Currently active compute tasks"
            );
            let _ = writeln!(out, "# TYPE qnk_compute_tasks_active gauge");
            for layer in ComputeLayer::all() {
                let idx = *layer as usize;
                let val = self.per_layer[idx].tasks_active.load(Ordering::Relaxed);
                let name = layer.name();
                let _ = writeln!(
                    out,
                    "qnk_compute_tasks_active{{layer=\"{name}\"}} {val}"
                );
            }
        }

        // ---- Per-layer: task_duration_seconds (histogram) ----
        {
            let _ = writeln!(
                out,
                "# HELP qnk_compute_task_duration_seconds Task execution duration in seconds"
            );
            let _ = writeln!(out, "# TYPE qnk_compute_task_duration_seconds histogram");
            for layer in ComputeLayer::all() {
                let idx = *layer as usize;
                let labels = format!("{{layer=\"{}\"}}", layer.name());
                self.per_layer[idx]
                    .task_duration
                    .render_data("qnk_compute_task_duration_seconds", &labels, &mut out);
            }
        }

        // ---- Resource gauges ----
        render_gauge_f64(
            "qnk_compute_cpu_utilization_percent",
            "Overall CPU utilization percentage (0-100)",
            micro_to_f64(self.cpu_usage_percent_u.load(Ordering::Relaxed)),
            "",
            &mut out,
        );
        render_gauge_f64(
            "qnk_compute_gpu_utilization_percent",
            "GPU utilization percentage (0-100, 0 if no GPU)",
            micro_to_f64(self.gpu_usage_percent_u.load(Ordering::Relaxed)),
            "",
            &mut out,
        );
        render_gauge_f64(
            "qnk_compute_gpu_temperature_celsius",
            "GPU temperature in degrees Celsius",
            micro_to_f64(self.gpu_temperature_u.load(Ordering::Relaxed)),
            "",
            &mut out,
        );
        render_gauge_f64(
            "qnk_compute_memory_used_bytes",
            "Memory currently in use (bytes)",
            self.ram_usage_bytes.load(Ordering::Relaxed) as f64,
            "",
            &mut out,
        );

        // Legacy resource names for backward compatibility
        render_gauge_f64(
            "qnk_compute_cpu_usage_percent",
            "Overall CPU utilization percentage (0-100)",
            micro_to_f64(self.cpu_usage_percent_u.load(Ordering::Relaxed)),
            "",
            &mut out,
        );
        render_gauge_f64(
            "qnk_compute_gpu_usage_percent",
            "GPU utilization percentage (0-100, 0 if no GPU)",
            micro_to_f64(self.gpu_usage_percent_u.load(Ordering::Relaxed)),
            "",
            &mut out,
        );
        render_gauge_f64(
            "qnk_compute_ram_usage_bytes",
            "RAM currently in use (bytes)",
            self.ram_usage_bytes.load(Ordering::Relaxed) as f64,
            "",
            &mut out,
        );
        render_gauge_f64(
            "qnk_compute_net_rx_bytes_total",
            "Network receive rate (bytes/sec)",
            self.net_rx_bytes_total.load(Ordering::Relaxed) as f64,
            "",
            &mut out,
        );
        render_gauge_f64(
            "qnk_compute_net_tx_bytes_total",
            "Network transmit rate (bytes/sec)",
            self.net_tx_bytes_total.load(Ordering::Relaxed) as f64,
            "",
            &mut out,
        );
        render_gauge_f64(
            "qnk_compute_disk_read_bytes_total",
            "Disk I/O read rate (bytes/sec)",
            self.disk_read_bytes_total.load(Ordering::Relaxed) as f64,
            "",
            &mut out,
        );
        render_gauge_f64(
            "qnk_compute_disk_write_bytes_total",
            "Disk I/O write rate (bytes/sec)",
            self.disk_write_bytes_total.load(Ordering::Relaxed) as f64,
            "",
            &mut out,
        );

        // ---- Trainer ----
        render_gauge_i64(
            "qnk_compute_trainer_cheats_active",
            "Number of trainer cheats currently active",
            self.trainer_cheats_active.load(Ordering::Relaxed),
            "",
            &mut out,
        );

        // ---- Per-layer: cores (gauge), CPU seconds (counter), GPU seconds (counter), revenue (counter) ----
        {
            let _ = writeln!(
                out,
                "# HELP qnk_compute_layer_cores Number of CPU cores assigned to each compute layer"
            );
            let _ = writeln!(out, "# TYPE qnk_compute_layer_cores gauge");
            for layer in ComputeLayer::all() {
                let idx = *layer as usize;
                let val = self.per_layer[idx].cores_assigned.load(Ordering::Relaxed);
                let name = layer.name();
                let _ = writeln!(
                    out,
                    "qnk_compute_layer_cores{{layer=\"{name}\"}} {val}"
                );
            }
        }
        {
            let _ = writeln!(
                out,
                "# HELP qnk_compute_layer_cpu_seconds_total Cumulative CPU seconds consumed per layer"
            );
            let _ = writeln!(out, "# TYPE qnk_compute_layer_cpu_seconds_total counter");
            for layer in ComputeLayer::all() {
                let idx = *layer as usize;
                let us = self.per_layer[idx].cpu_seconds_us.load(Ordering::Relaxed);
                let secs = us as f64 / 1_000_000.0;
                let name = layer.name();
                let _ = writeln!(
                    out,
                    "qnk_compute_layer_cpu_seconds_total{{layer=\"{name}\"}} {secs}"
                );
            }
        }
        {
            let _ = writeln!(
                out,
                "# HELP qnk_compute_layer_gpu_seconds_total Cumulative GPU seconds consumed per layer"
            );
            let _ = writeln!(out, "# TYPE qnk_compute_layer_gpu_seconds_total counter");
            for layer in ComputeLayer::all() {
                let idx = *layer as usize;
                let us = self.per_layer[idx].gpu_seconds_us.load(Ordering::Relaxed);
                let secs = us as f64 / 1_000_000.0;
                let name = layer.name();
                let _ = writeln!(
                    out,
                    "qnk_compute_layer_gpu_seconds_total{{layer=\"{name}\"}} {secs}"
                );
            }
        }
        {
            let _ = writeln!(
                out,
                "# HELP qnk_compute_revenue_micro_qug_total Cumulative revenue in micro-QUG per layer"
            );
            let _ = writeln!(
                out,
                "# TYPE qnk_compute_revenue_micro_qug_total counter"
            );
            for layer in ComputeLayer::all() {
                let idx = *layer as usize;
                let val = self.per_layer[idx]
                    .revenue_micro_qug
                    .load(Ordering::Relaxed);
                let name = layer.name();
                let _ = writeln!(
                    out,
                    "qnk_compute_revenue_micro_qug_total{{layer=\"{name}\"}} {val}"
                );
            }
        }

        // ---- Job duration histogram (global) ----
        self.job_duration.render(
            "qnk_compute_job_duration_seconds",
            "Histogram of compute job execution durations in seconds",
            "",
            &mut out,
        );

        // ---- Tunnel metrics ----
        render_gauge_i64(
            "qnk_compute_tunnel_count",
            "Number of currently connected compute tunnels",
            self.tunnel.peers_connected.load(Ordering::Relaxed),
            "",
            &mut out,
        );

        // Legacy name for backward compat
        render_gauge_i64(
            "qnk_compute_tunnel_peers_connected",
            "Number of currently connected compute tunnel peers",
            self.tunnel.peers_connected.load(Ordering::Relaxed),
            "",
            &mut out,
        );

        render_counter_u64(
            "qnk_compute_tunnel_bytes_sent_total",
            "Total bytes sent through compute tunnels",
            self.tunnel.bytes_sent_total.load(Ordering::Relaxed),
            "",
            &mut out,
        );

        render_counter_u64(
            "qnk_compute_tunnel_bytes_received_total",
            "Total bytes received through compute tunnels",
            self.tunnel.bytes_received_total.load(Ordering::Relaxed),
            "",
            &mut out,
        );

        self.tunnel.rtt_histogram.render(
            "qnk_compute_tunnel_rtt_seconds",
            "Histogram of compute tunnel round-trip times in seconds",
            "",
            &mut out,
        );

        render_counter_u64(
            "qnk_compute_tunnel_bandwidth_bytes_total",
            "Total bytes transferred through compute tunnels",
            self.tunnel.bandwidth_bytes_total.load(Ordering::Relaxed),
            "",
            &mut out,
        );

        render_counter_u64(
            "qnk_compute_tunnel_rekey_total",
            "Total number of tunnel session re-key operations",
            self.tunnel.rekey_total.load(Ordering::Relaxed),
            "",
            &mut out,
        );

        // ---- Job queue depth ----
        {
            let _ = writeln!(
                out,
                "# HELP qnk_compute_job_queue_depth Current job queue depth by status"
            );
            let _ = writeln!(out, "# TYPE qnk_compute_job_queue_depth gauge");
            for (i, status) in JOB_STATUSES.iter().enumerate() {
                let val = self.jobs.queue_depth[i].load(Ordering::Relaxed);
                let _ = writeln!(
                    out,
                    "qnk_compute_job_queue_depth{{status=\"{status}\"}} {val}"
                );
            }
        }

        // ---- Job completions ----
        {
            let _ = writeln!(
                out,
                "# HELP qnk_compute_job_completions_total Total completed jobs per layer"
            );
            let _ = writeln!(out, "# TYPE qnk_compute_job_completions_total counter");
            for layer in ComputeLayer::all() {
                let idx = *layer as usize;
                let val = self.jobs.completions[idx].load(Ordering::Relaxed);
                let name = layer.name();
                let _ = writeln!(
                    out,
                    "qnk_compute_job_completions_total{{layer=\"{name}\"}} {val}"
                );
            }
        }

        // ---- Job failures ----
        {
            let _ = writeln!(
                out,
                "# HELP qnk_compute_job_failures_total Total failed jobs per layer"
            );
            let _ = writeln!(out, "# TYPE qnk_compute_job_failures_total counter");
            for layer in ComputeLayer::all() {
                let idx = *layer as usize;
                let val = self.jobs.failures[idx].load(Ordering::Relaxed);
                let name = layer.name();
                let _ = writeln!(
                    out,
                    "qnk_compute_job_failures_total{{layer=\"{name}\"}} {val}"
                );
            }
        }

        // ---- GPU device metrics ----
        let gpu_count = self
            .gpu_devices
            .device_count
            .load(Ordering::Relaxed)
            .min(MAX_GPUS as u64) as usize;
        if gpu_count > 0 {
            {
                let _ = writeln!(
                    out,
                    "# HELP qnk_compute_gpu_utilization_percent GPU utilization per device (0-100)"
                );
                let _ = writeln!(out, "# TYPE qnk_compute_gpu_utilization_percent gauge");
                for i in 0..gpu_count {
                    let val = micro_to_f64(
                        self.gpu_devices.utilization_percent[i].load(Ordering::Relaxed),
                    );
                    let _ = writeln!(
                        out,
                        "qnk_compute_gpu_utilization_percent{{device=\"{i}\"}} {val}"
                    );
                }
            }
            {
                let _ = writeln!(
                    out,
                    "# HELP qnk_compute_gpu_vram_used_bytes VRAM used per device (bytes)"
                );
                let _ = writeln!(out, "# TYPE qnk_compute_gpu_vram_used_bytes gauge");
                for i in 0..gpu_count {
                    let val = self.gpu_devices.vram_used_bytes[i].load(Ordering::Relaxed);
                    let _ = writeln!(
                        out,
                        "qnk_compute_gpu_vram_used_bytes{{device=\"{i}\"}} {val}"
                    );
                }
            }
            {
                let _ = writeln!(
                    out,
                    "# HELP qnk_compute_gpu_temperature_celsius GPU temperature per device (Celsius)"
                );
                let _ = writeln!(out, "# TYPE qnk_compute_gpu_temperature_celsius gauge");
                for i in 0..gpu_count {
                    let milli =
                        self.gpu_devices.temperature_milli_c[i].load(Ordering::Relaxed);
                    let celsius = milli as f64 / 1000.0;
                    let _ = writeln!(
                        out,
                        "qnk_compute_gpu_temperature_celsius{{device=\"{i}\"}} {celsius}"
                    );
                }
            }
        }

        // ---- Peer count ----
        render_gauge_i64(
            "qnk_compute_peer_count",
            "Number of connected P2P peers",
            self.peer_count.load(Ordering::Relaxed),
            "",
            &mut out,
        );

        // ---- Per-peer reputation ----
        if let Ok(scores) = self.reputation.scores.read() {
            if !scores.is_empty() {
                let _ = writeln!(
                    out,
                    "# HELP qnk_compute_reputation_score Reputation score per peer"
                );
                let _ = writeln!(out, "# TYPE qnk_compute_reputation_score gauge");
                // Sort for deterministic output
                let mut sorted: Vec<_> = scores.iter().collect();
                sorted.sort_by(|(a, _), (b, _)| a.cmp(b));
                for (peer_id, &raw) in &sorted {
                    let score = micro_to_f64(raw);
                    let _ = writeln!(
                        out,
                        "qnk_compute_reputation_score{{peer=\"{peer_id}\"}} {score}"
                    );
                }
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Encode an f64 as micro-units for storage in AtomicU64.
fn f64_to_micro(v: f64) -> u64 {
    (v * 1_000_000.0) as u64
}

/// Decode micro-units back to f64.
fn micro_to_f64(v: u64) -> f64 {
    v as f64 / 1_000_000.0
}

/// Map a human-readable layer name back to a `per_layer` array index.
fn layer_name_to_index(name: &str) -> Option<usize> {
    match name {
        "Mining" => Some(0),
        "AI Inference" => Some(1),
        "ZK Proofs" => Some(2),
        "Bridge Verify" => Some(3),
        "IPFS Pinning" => Some(4),
        "VDF Compute" => Some(5),
        "Render Farm" => Some(6),
        "Idle Crypto" => Some(7),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

static METRICS: OnceLock<ComputeMetrics> = OnceLock::new();

/// Get the global `ComputeMetrics` singleton (lazily initialized).
pub fn global_metrics() -> &'static ComputeMetrics {
    METRICS.get_or_init(ComputeMetrics::new)
}

/// Convenience: return all metrics in Prometheus text exposition format.
///
/// This is the main entry point for `/metrics` HTTP handlers.
pub fn gather_metrics() -> String {
    global_metrics().gather()
}

/// Update all metrics from a full `ComputeStatus` snapshot.
///
/// Call this periodically (e.g., once per second from the orchestrator loop)
/// to keep metrics fresh.
pub fn update_all(
    snapshot: &ResourceSnapshot,
    layers: &[(String, LayerStats)],
    trainer_cheats_count: usize,
) {
    let m = global_metrics();
    m.update_from_snapshot(snapshot);
    m.update_layer_cores(
        layers
            .iter()
            .map(|(name, stats)| (name.as_str(), stats.cores_assigned)),
    );
    m.update_trainer_cheats(trainer_cheats_count as i64);
}

// ---------------------------------------------------------------------------
// MetricsExporter — convenience wrapper for HTTP handlers
// ---------------------------------------------------------------------------

/// High-level exporter that wraps the global metrics singleton and provides
/// a `render_prometheus()` method returning `text/plain` content suitable for
/// a Prometheus scrape endpoint.
pub struct MetricsExporter;

impl MetricsExporter {
    /// Create a new exporter (zero cost -- just references the global singleton).
    pub fn new() -> Self {
        Self
    }

    /// Render all metrics in Prometheus text exposition format (`text/plain;
    /// version=0.0.4; charset=utf-8`).
    pub fn render_prometheus(&self) -> String {
        gather_metrics()
    }

    /// Access the underlying global metrics for direct mutation.
    pub fn metrics(&self) -> &'static ComputeMetrics {
        global_metrics()
    }
}

impl Default for MetricsExporter {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

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

    // 1. Counter increment (AtomicU64-based tasks_total)
    #[test]
    fn test_counter_increment() {
        let m = ComputeMetrics::new();
        // Mining layer tasks_total starts at 0
        assert_eq!(
            m.per_layer[ComputeLayer::Mining as usize]
                .tasks_total
                .load(Ordering::Relaxed),
            0
        );
        m.record_task_start(ComputeLayer::Mining);
        assert_eq!(
            m.per_layer[ComputeLayer::Mining as usize]
                .tasks_total
                .load(Ordering::Relaxed),
            1
        );
        m.record_task_start(ComputeLayer::Mining);
        m.record_task_start(ComputeLayer::Mining);
        assert_eq!(
            m.per_layer[ComputeLayer::Mining as usize]
                .tasks_total
                .load(Ordering::Relaxed),
            3
        );
    }

    // 2. Gauge set/get (tasks_active goes up and down)
    #[test]
    fn test_gauge_set_get() {
        let m = ComputeMetrics::new();
        // Start 3 tasks
        m.record_task_start(ComputeLayer::AiInference);
        m.record_task_start(ComputeLayer::AiInference);
        m.record_task_start(ComputeLayer::AiInference);
        assert_eq!(
            m.per_layer[ComputeLayer::AiInference as usize]
                .tasks_active
                .load(Ordering::Relaxed),
            3
        );
        // End 2 tasks
        m.record_task_end(ComputeLayer::AiInference, 0.1);
        m.record_task_end(ComputeLayer::AiInference, 0.2);
        assert_eq!(
            m.per_layer[ComputeLayer::AiInference as usize]
                .tasks_active
                .load(Ordering::Relaxed),
            1
        );

        // Also test peer_count gauge via set/read
        m.set_peer_count(42);
        assert_eq!(m.peer_count.load(Ordering::Relaxed), 42);
        m.set_peer_count(0);
        assert_eq!(m.peer_count.load(Ordering::Relaxed), 0);
    }

    // 3. Histogram: observe into correct bucket
    #[test]
    fn test_histogram_observe_correct_bucket() {
        let h = AtomicHistogram::new(&[0.01, 0.05, 0.1]);
        h.observe(0.005); // <= 0.01
        h.observe(0.008); // <= 0.01
        h.observe(0.030); // <= 0.05
        h.observe(0.200); // > all finite buckets, only +Inf

        assert_eq!(h.total_count(), 4);

        let mut out = String::new();
        h.render("test_hist", "Test histogram", "", &mut out);

        // le=0.01 bucket: 2 observations (cumulative)
        assert!(out.contains("test_hist_bucket{le=\"0.01\"} 2"));
        // le=0.05 bucket: 2 + 1 = 3 (cumulative)
        assert!(out.contains("test_hist_bucket{le=\"0.05\"} 3"));
        // le=0.1 bucket: 3 + 0 = 3 (cumulative)
        assert!(out.contains("test_hist_bucket{le=\"0.1\"} 3"));
        // +Inf: 4
        assert!(out.contains("test_hist_bucket{le=\"+Inf\"} 4"));
        assert!(out.contains("test_hist_count 4"));
    }

    // 4. Prometheus render format (HELP/TYPE/data lines)
    #[test]
    fn test_prometheus_render_format() {
        let m = ComputeMetrics::new();
        m.record_task_start(ComputeLayer::Mining);
        m.record_task_start(ComputeLayer::Mining);
        m.record_task_end(ComputeLayer::Mining, 0.5);

        let output = m.render_prometheus();

        // Has HELP and TYPE lines for tasks_total
        assert!(output.contains("# HELP qnk_compute_tasks_total Total compute tasks"));
        assert!(output.contains("# TYPE qnk_compute_tasks_total counter"));

        // Mining layer tasks_total should be 2
        assert!(output.contains("qnk_compute_tasks_total{layer=\"Mining\"} 2"));

        // tasks_active: started 2, ended 1, so active = 1
        assert!(output.contains("qnk_compute_tasks_active{layer=\"Mining\"} 1"));

        // Per-layer histogram buckets present
        assert!(output.contains(
            "qnk_compute_task_duration_seconds_bucket{layer=\"Mining\",le=\"+Inf\"}"
        ));

        // Resource gauges present
        assert!(output.contains("# TYPE qnk_compute_cpu_utilization_percent gauge"));
        assert!(output.contains("# TYPE qnk_compute_tunnel_count gauge"));
        assert!(output.contains("# TYPE qnk_compute_peer_count gauge"));
        assert!(output.contains("# TYPE qnk_compute_memory_used_bytes gauge"));
        assert!(output.contains("# TYPE qnk_compute_gpu_temperature_celsius gauge"));
    }

    // 5. Per-layer task counting
    #[test]
    fn test_per_layer_task_counting() {
        let m = ComputeMetrics::new();

        m.record_task_start(ComputeLayer::Mining);
        m.record_task_start(ComputeLayer::Mining);
        m.record_task_start(ComputeLayer::Mining);
        m.record_task_start(ComputeLayer::AiInference);
        m.record_task_start(ComputeLayer::ZkProofGen);
        m.record_task_start(ComputeLayer::ZkProofGen);

        let output = m.render_prometheus();

        assert!(output.contains("qnk_compute_tasks_total{layer=\"Mining\"} 3"));
        assert!(output.contains("qnk_compute_tasks_total{layer=\"AI Inference\"} 1"));
        assert!(output.contains("qnk_compute_tasks_total{layer=\"ZK Proofs\"} 2"));
        // Untouched layer stays at 0
        assert!(output.contains("qnk_compute_tasks_total{layer=\"Idle Crypto\"} 0"));
    }

    // 6. Resource snapshot update
    #[test]
    fn test_resource_snapshot_update() {
        let m = ComputeMetrics::new();
        let snap = make_snapshot();
        m.update_resources(&snap);

        let output = m.render_prometheus();
        assert!(output.contains("qnk_compute_cpu_utilization_percent 52.5"));
        assert!(output.contains("qnk_compute_gpu_utilization_percent 78.3"));
        assert!(output.contains("qnk_compute_gpu_temperature_celsius 65"));
        assert!(output.contains("qnk_compute_memory_used_bytes 16000000000"));
        // Legacy names also populated
        assert!(output.contains("qnk_compute_cpu_usage_percent 52.5"));
        assert!(output.contains("qnk_compute_ram_usage_bytes 16000000000"));
        assert!(output.contains("qnk_compute_net_rx_bytes_total 120000000"));
        assert!(output.contains("qnk_compute_net_tx_bytes_total 50000000"));
    }

    // 7. Tunnel metrics update
    #[test]
    fn test_tunnel_metrics_update() {
        let m = ComputeMetrics::new();

        m.update_tunnels(5, 1_000_000, 2_000_000);

        let output = m.render_prometheus();
        assert!(output.contains("qnk_compute_tunnel_count 5"));
        assert!(output.contains("qnk_compute_tunnel_bytes_sent_total 1000000"));
        assert!(output.contains("qnk_compute_tunnel_bytes_received_total 2000000"));

        // Calling again should accumulate bytes (counter behavior)
        m.update_tunnels(3, 500_000, 300_000);
        let output = m.render_prometheus();
        assert!(output.contains("qnk_compute_tunnel_count 3")); // gauge: replaced
        assert!(output.contains("qnk_compute_tunnel_bytes_sent_total 1500000")); // counter: accumulated
        assert!(output.contains("qnk_compute_tunnel_bytes_received_total 2300000")); // counter: accumulated
    }

    // 8. Revenue tracking per layer
    #[test]
    fn test_revenue_tracking() {
        let m = ComputeMetrics::new();
        m.add_layer_revenue(ComputeLayer::AiInference, 5000);
        m.add_layer_revenue(ComputeLayer::AiInference, 3000);
        m.add_layer_revenue(ComputeLayer::Mining, 10000);

        let output = m.render_prometheus();
        assert!(output.contains(
            "qnk_compute_revenue_micro_qug_total{layer=\"AI Inference\"} 8000"
        ));
        assert!(output.contains(
            "qnk_compute_revenue_micro_qug_total{layer=\"Mining\"} 10000"
        ));
        // Untouched layer is 0
        assert!(output.contains(
            "qnk_compute_revenue_micro_qug_total{layer=\"ZK Proofs\"} 0"
        ));
    }

    // ---- Additional tests beyond the required 8 ----

    // 9. Per-layer task duration histogram with standard buckets
    #[test]
    fn test_per_layer_task_duration_histogram() {
        let m = ComputeMetrics::new();
        // Fast task 0.03s on VDF — should land in le=0.05 and above
        m.record_task_start(ComputeLayer::VdfCompute);
        m.record_task_end(ComputeLayer::VdfCompute, 0.03);

        // Slow task 45s on VDF — should only land in le=60.0 and +Inf
        m.record_task_start(ComputeLayer::VdfCompute);
        m.record_task_end(ComputeLayer::VdfCompute, 45.0);

        let output = m.render_prometheus();

        // 0.03 fits in le=0.05 bucket
        assert!(output.contains(
            "qnk_compute_task_duration_seconds_bucket{layer=\"VDF Compute\",le=\"0.05\"} 1"
        ));
        // 0.03 does NOT fit in le=0.01
        assert!(output.contains(
            "qnk_compute_task_duration_seconds_bucket{layer=\"VDF Compute\",le=\"0.01\"} 0"
        ));
        // Both fit in le=60.0
        assert!(output.contains(
            "qnk_compute_task_duration_seconds_bucket{layer=\"VDF Compute\",le=\"60\"} 2"
        ));
        // Count
        assert!(output.contains(
            "qnk_compute_task_duration_seconds_count{layer=\"VDF Compute\"} 2"
        ));
    }

    // 10. Reputation score per peer
    #[test]
    fn test_reputation_score() {
        let m = ComputeMetrics::new();
        m.set_reputation("12D3KooWAlpha", 0.95);
        m.set_reputation("12D3KooWBeta", 0.42);

        let output = m.render_prometheus();
        assert!(output.contains("# TYPE qnk_compute_reputation_score gauge"));
        assert!(output.contains("qnk_compute_reputation_score{peer=\"12D3KooWAlpha\"} 0.95"));
        assert!(output.contains("qnk_compute_reputation_score{peer=\"12D3KooWBeta\"} 0.42"));
    }

    // 11. Peer count gauge
    #[test]
    fn test_peer_count() {
        let m = ComputeMetrics::new();
        m.set_peer_count(17);
        let output = m.render_prometheus();
        assert!(output.contains("qnk_compute_peer_count 17"));
    }

    // 12. MetricValue enum (issue #028 requirement)
    #[test]
    fn test_metric_value_enum() {
        // Counter
        let c = MetricValue::counter();
        c.inc_counter();
        c.inc_counter();
        assert_eq!(c.counter_value(), 2);

        // Gauge
        let g = MetricValue::gauge();
        g.set_gauge(42);
        assert_eq!(g.gauge_value(), 42);
        g.set_gauge(-10);
        assert_eq!(g.gauge_value(), -10);

        // Histogram
        let h = MetricValue::histogram(vec![1.0, 5.0, 10.0]);
        h.observe(3.0);
        h.observe(7.0);
        // Verify via counter_value / gauge_value return 0 for wrong type
        assert_eq!(h.counter_value(), 0);
        assert_eq!(h.gauge_value(), 0);
    }

    // 13. Global singleton and update_all
    #[test]
    fn test_gather_metrics_and_update_all() {
        let snap = make_snapshot();
        let layers = vec![
            (
                "Mining".to_string(),
                crate::LayerStats {
                    cores_assigned: 8,
                    tasks_completed: crate::AtomicU64Ser(100),
                    tasks_pending: 2,
                    revenue_micro_qug: 5000,
                    active_since_ms: 1000,
                },
            ),
            (
                "AI Inference".to_string(),
                crate::LayerStats {
                    cores_assigned: 4,
                    tasks_completed: crate::AtomicU64Ser(50),
                    tasks_pending: 0,
                    revenue_micro_qug: 2000,
                    active_since_ms: 2000,
                },
            ),
        ];
        update_all(&snap, &layers, 3);

        let output = gather_metrics();
        assert!(output.contains("qnk_compute_cpu_usage_percent"));
        assert!(output.contains("qnk_compute_trainer_cheats_active 3"));
    }

    // 14. Job duration histogram (global)
    #[test]
    fn test_job_duration_histogram() {
        let m = ComputeMetrics::new();
        m.observe_job_duration(0.002); // 2ms
        m.observe_job_duration(0.050); // 50ms
        m.observe_job_duration(0.500); // 500ms
        m.observe_job_duration(5.000); // 5s

        let output = m.gather();
        assert!(output.contains("# TYPE qnk_compute_job_duration_seconds histogram"));
        assert!(output.contains("qnk_compute_job_duration_seconds_bucket{le=\"+Inf\"} 4"));
        assert!(output.contains("qnk_compute_job_duration_seconds_count 4"));
        // 2ms <= 0.005 bucket
        assert!(output.contains("qnk_compute_job_duration_seconds_bucket{le=\"0.005\"} 1"));
    }

    // 15. Tunnel RTT, bandwidth, and rekey
    #[test]
    fn test_tunnel_rtt_bandwidth_rekey() {
        let m = ComputeMetrics::new();
        m.set_tunnel_peers(7);
        m.observe_tunnel_rtt(0.025);
        m.observe_tunnel_rtt(0.100);
        m.add_tunnel_bandwidth(1_000_000);
        m.add_tunnel_bandwidth(500_000);
        m.inc_tunnel_rekey();
        m.inc_tunnel_rekey();
        m.inc_tunnel_rekey();

        let output = m.gather();
        assert!(output.contains("qnk_compute_tunnel_peers_connected 7"));
        assert!(output.contains("qnk_compute_tunnel_bandwidth_bytes_total 1500000"));
        assert!(output.contains("qnk_compute_tunnel_rekey_total 3"));
        assert!(output.contains("qnk_compute_tunnel_rtt_seconds_count 2"));
        assert!(output.contains("# TYPE qnk_compute_tunnel_rtt_seconds histogram"));
    }

    // 16. Job queue depth and completion/failure counters
    #[test]
    fn test_job_queue_metrics() {
        let m = ComputeMetrics::new();
        m.set_job_queue_depth(0, 10); // pending=10
        m.set_job_queue_depth(1, 3); // running=3
        m.set_job_queue_depth(2, 100); // completed=100
        m.set_job_queue_depth(3, 2); // failed=2

        m.inc_job_completions(ComputeLayer::Mining);
        m.inc_job_completions(ComputeLayer::Mining);
        m.inc_job_completions(ComputeLayer::AiInference);
        m.inc_job_failures(ComputeLayer::ZkProofGen);

        let output = m.gather();
        assert!(output.contains("qnk_compute_job_queue_depth{status=\"pending\"} 10"));
        assert!(output.contains("qnk_compute_job_queue_depth{status=\"running\"} 3"));
        assert!(output.contains("qnk_compute_job_queue_depth{status=\"completed\"} 100"));
        assert!(output.contains("qnk_compute_job_queue_depth{status=\"failed\"} 2"));
        assert!(output.contains(
            "qnk_compute_job_completions_total{layer=\"Mining\"} 2"
        ));
        assert!(output.contains(
            "qnk_compute_job_completions_total{layer=\"AI Inference\"} 1"
        ));
        assert!(output.contains(
            "qnk_compute_job_failures_total{layer=\"ZK Proofs\"} 1"
        ));
    }

    // 17. GPU device metrics
    #[test]
    fn test_gpu_device_metrics() {
        let m = ComputeMetrics::new();
        m.set_gpu_device_count(2);
        m.set_gpu_utilization(0, 85.5);
        m.set_gpu_utilization(1, 42.0);
        m.set_gpu_vram_used(0, 6_000_000_000);
        m.set_gpu_vram_used(1, 3_000_000_000);
        m.set_gpu_temperature(0, 72.5);
        m.set_gpu_temperature(1, 58.0);

        let output = m.gather();
        assert!(output.contains("qnk_compute_gpu_utilization_percent{device=\"0\"} 85.5"));
        assert!(output.contains("qnk_compute_gpu_utilization_percent{device=\"1\"} 42"));
        assert!(output.contains("qnk_compute_gpu_vram_used_bytes{device=\"0\"} 6000000000"));
        assert!(output.contains("qnk_compute_gpu_vram_used_bytes{device=\"1\"} 3000000000"));
        assert!(output.contains("qnk_compute_gpu_temperature_celsius{device=\"0\"} 72.5"));
        assert!(output.contains("qnk_compute_gpu_temperature_celsius{device=\"1\"} 58"));
    }

    // 18. Per-layer CPU/GPU seconds counters
    #[test]
    fn test_per_layer_cpu_gpu_seconds() {
        let m = ComputeMetrics::new();
        m.add_layer_cpu_seconds(ComputeLayer::Mining, 10.5);
        m.add_layer_cpu_seconds(ComputeLayer::Mining, 5.25);
        m.add_layer_gpu_seconds(ComputeLayer::AiInference, 3.0);

        let output = m.gather();
        assert!(output.contains(
            "qnk_compute_layer_cpu_seconds_total{layer=\"Mining\"} 15.75"
        ));
        assert!(output.contains(
            "qnk_compute_layer_gpu_seconds_total{layer=\"AI Inference\"} 3"
        ));
    }

    // 19. Layer cores labels
    #[test]
    fn test_layer_cores_labels() {
        let m = ComputeMetrics::new();
        m.update_layer_cores([("Mining", 6), ("AI Inference", 2), ("ZK Proofs", 0)]);

        let output = m.gather();
        assert!(output.contains("qnk_compute_layer_cores{layer=\"Mining\"} 6"));
        assert!(output.contains("qnk_compute_layer_cores{layer=\"AI Inference\"} 2"));
        assert!(output.contains("qnk_compute_layer_cores{layer=\"ZK Proofs\"} 0"));
    }

    // 20. MetricsExporter wrapper
    #[test]
    fn test_metrics_exporter() {
        let exporter = MetricsExporter::new();
        let output = exporter.render_prometheus();
        assert!(output.contains("qnk_compute_cpu_usage_percent"));
        assert!(output.contains("qnk_compute_job_duration_seconds"));
        assert!(output.contains("qnk_compute_tunnel_peers_connected"));
    }

    // 21. HELP and TYPE lines present for all major metric families
    #[test]
    fn test_help_and_type_lines() {
        let m = ComputeMetrics::new();
        m.set_gpu_device_count(1);
        m.set_reputation("testpeer", 1.0);
        let output = m.gather();

        let expected_types = [
            ("qnk_compute_tasks_total", "counter"),
            ("qnk_compute_tasks_active", "gauge"),
            ("qnk_compute_task_duration_seconds", "histogram"),
            ("qnk_compute_cpu_utilization_percent", "gauge"),
            ("qnk_compute_gpu_utilization_percent", "gauge"),
            ("qnk_compute_gpu_temperature_celsius", "gauge"),
            ("qnk_compute_memory_used_bytes", "gauge"),
            ("qnk_compute_cpu_usage_percent", "gauge"),
            ("qnk_compute_ram_usage_bytes", "gauge"),
            ("qnk_compute_trainer_cheats_active", "gauge"),
            ("qnk_compute_layer_cores", "gauge"),
            ("qnk_compute_layer_cpu_seconds_total", "counter"),
            ("qnk_compute_layer_gpu_seconds_total", "counter"),
            ("qnk_compute_revenue_micro_qug_total", "counter"),
            ("qnk_compute_job_duration_seconds", "histogram"),
            ("qnk_compute_tunnel_count", "gauge"),
            ("qnk_compute_tunnel_peers_connected", "gauge"),
            ("qnk_compute_tunnel_bytes_sent_total", "counter"),
            ("qnk_compute_tunnel_bytes_received_total", "counter"),
            ("qnk_compute_tunnel_rtt_seconds", "histogram"),
            ("qnk_compute_tunnel_bandwidth_bytes_total", "counter"),
            ("qnk_compute_tunnel_rekey_total", "counter"),
            ("qnk_compute_job_queue_depth", "gauge"),
            ("qnk_compute_job_completions_total", "counter"),
            ("qnk_compute_job_failures_total", "counter"),
            ("qnk_compute_peer_count", "gauge"),
            ("qnk_compute_reputation_score", "gauge"),
        ];

        for (name, typ) in &expected_types {
            let help_line = format!("# HELP {name}");
            let type_line = format!("# TYPE {name} {typ}");
            assert!(
                output.contains(&help_line),
                "Missing HELP for {name}"
            );
            assert!(
                output.contains(&type_line),
                "Missing TYPE for {name}"
            );
        }
    }

    // 22. Boundary: out-of-range GPU device index is ignored
    #[test]
    fn test_gpu_out_of_range_ignored() {
        let m = ComputeMetrics::new();
        m.set_gpu_utilization(99, 50.0);
        m.set_gpu_vram_used(99, 1000);
        m.set_gpu_temperature(99, 80.0);
        let output = m.gather();
        // No GPU count set, so GPU device section should not appear
        assert!(!output.contains("qnk_compute_gpu_utilization_percent{device="));
    }

    // 23. Job queue out-of-range status index
    #[test]
    fn test_job_queue_out_of_range() {
        let m = ComputeMetrics::new();
        m.set_job_queue_depth(99, 42);
        let output = m.gather();
        assert!(output.contains("qnk_compute_job_queue_depth{status=\"pending\"} 0"));
    }
}
