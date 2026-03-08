//! Resource Monitor — 100ms sampling of CPU/GPU/RAM/NET/DISK
//!
//! Feeds the orchestrator with real-time utilization data
//! so it can assign work to idle resources.

use crate::ResourceSnapshot;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;
use sysinfo::{System, CpuRefreshKind, MemoryRefreshKind, RefreshKind};
use tracing::{debug, trace};

/// How often to sample resources (100ms = 10 samples/sec)
const SAMPLE_INTERVAL: Duration = Duration::from_millis(100);

/// Resource monitor that runs in background, sampling every 100ms
pub struct ResourceMonitor {
    /// Latest snapshot (lock-free read via RwLock)
    latest: Arc<RwLock<ResourceSnapshot>>,
    /// Historical snapshots for trend analysis (last 60 seconds = 600 samples)
    history: Arc<RwLock<Vec<ResourceSnapshot>>>,
    /// Stop flag
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl ResourceMonitor {
    pub fn new() -> Self {
        Self {
            latest: Arc::new(RwLock::new(ResourceSnapshot {
                cpu_per_core: Vec::new(),
                cpu_total: 0.0,
                gpu_utilization: 0.0,
                gpu_memory_used: 0,
                gpu_memory_total: 0,
                ram_used: 0,
                ram_total: 0,
                net_tx_bps: 0,
                net_rx_bps: 0,
                net_capacity_bps: 0,
                disk_io_bps: 0,
                timestamp_ms: 0,
            })),
            history: Arc::new(RwLock::new(Vec::with_capacity(600))),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Get the latest resource snapshot (lock-free read)
    pub fn snapshot(&self) -> ResourceSnapshot {
        self.latest.read().clone()
    }

    /// Get historical snapshots for trend analysis
    pub fn history(&self, last_n: usize) -> Vec<ResourceSnapshot> {
        let h = self.history.read();
        let start = h.len().saturating_sub(last_n);
        h[start..].to_vec()
    }

    /// Idle CPU percentage (how much headroom we have)
    pub fn idle_cpu_pct(&self) -> f32 {
        100.0 - self.latest.read().cpu_total
    }

    /// Idle RAM in bytes
    pub fn idle_ram_bytes(&self) -> u64 {
        let snap = self.latest.read();
        snap.ram_total.saturating_sub(snap.ram_used)
    }

    /// Start the background sampling task
    pub fn spawn(&self) -> tokio::task::JoinHandle<()> {
        let latest = self.latest.clone();
        let history = self.history.clone();
        let running = self.running.clone();
        running.store(true, std::sync::atomic::Ordering::SeqCst);

        tokio::spawn(async move {
            let mut sys = System::new_with_specifics(
                RefreshKind::new()
                    .with_cpu(CpuRefreshKind::everything())
                    .with_memory(MemoryRefreshKind::everything()),
            );

            let mut prev_net_rx: u64 = 0;
            let mut prev_net_tx: u64 = 0;
            let mut prev_time = Instant::now();
            let mut sample_count: u64 = 0;

            debug!("📊 [RESOURCE MONITOR] Started — sampling every {}ms", SAMPLE_INTERVAL.as_millis());

            while running.load(std::sync::atomic::Ordering::Relaxed) {
                tokio::time::sleep(SAMPLE_INTERVAL).await;
                sys.refresh_cpu_usage();
                sys.refresh_memory();

                let now = Instant::now();
                let elapsed = now.duration_since(prev_time);
                prev_time = now;

                // CPU per-core
                let cpu_per_core: Vec<f32> = sys.cpus().iter().map(|c| c.cpu_usage()).collect();
                let cpu_total = if cpu_per_core.is_empty() {
                    0.0
                } else {
                    cpu_per_core.iter().sum::<f32>() / cpu_per_core.len() as f32
                };

                // RAM
                let ram_used = sys.used_memory();
                let ram_total = sys.total_memory();

                // Network (read from /proc/net/dev on Linux)
                let (net_rx, net_tx) = get_network_bytes();
                let dt_secs = elapsed.as_secs_f64().max(0.001);
                let net_rx_bps = ((net_rx.saturating_sub(prev_net_rx)) as f64 / dt_secs) as u64;
                let net_tx_bps = ((net_tx.saturating_sub(prev_net_tx)) as f64 / dt_secs) as u64;
                prev_net_rx = net_rx;
                prev_net_tx = net_tx;

                // GPU — placeholder (needs OpenCL/NVML integration)
                let (gpu_util, gpu_mem_used, gpu_mem_total) = get_gpu_stats();

                let timestamp_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;

                let snapshot = ResourceSnapshot {
                    cpu_per_core,
                    cpu_total,
                    gpu_utilization: gpu_util,
                    gpu_memory_used: gpu_mem_used,
                    gpu_memory_total: gpu_mem_total,
                    ram_used,
                    ram_total,
                    net_tx_bps,
                    net_rx_bps,
                    net_capacity_bps: estimate_net_capacity(net_rx_bps + net_tx_bps),
                    disk_io_bps: 0, // TODO: /proc/diskstats
                    timestamp_ms,
                };

                // Update latest
                *latest.write() = snapshot.clone();

                // Append to history (ring buffer, keep last 600 = 60s)
                {
                    let mut h = history.write();
                    h.push(snapshot);
                    if h.len() > 600 {
                        h.drain(0..100); // Drain in batches to avoid per-sample overhead
                    }
                }

                sample_count += 1;
                if sample_count % 100 == 0 {
                    trace!(
                        "📊 [RESOURCE MONITOR] CPU={:.1}% RAM={:.1}% NET={:.1}Mbps GPU={:.0}% (sample #{})",
                        cpu_total,
                        (ram_used as f64 / ram_total.max(1) as f64) * 100.0,
                        (net_rx_bps + net_tx_bps) as f64 / 125_000.0,
                        gpu_util,
                        sample_count,
                    );
                }
            }

            debug!("📊 [RESOURCE MONITOR] Stopped after {} samples", sample_count);
        })
    }

    /// Stop the monitor
    pub fn stop(&self) {
        self.running.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Read total network bytes from /proc/net/dev (Linux) or return 0 (Windows)
fn get_network_bytes() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/net/dev") {
            let mut total_rx: u64 = 0;
            let mut total_tx: u64 = 0;
            for line in content.lines().skip(2) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 10 {
                    let iface = parts[0].trim_end_matches(':');
                    if iface == "lo" { continue; }
                    if let (Ok(rx), Ok(tx)) = (parts[1].parse::<u64>(), parts[9].parse::<u64>()) {
                        total_rx += rx;
                        total_tx += tx;
                    }
                }
            }
            return (total_rx, total_tx);
        }
    }
    (0, 0)
}

/// Get GPU stats — placeholder until OpenCL/NVML integration
fn get_gpu_stats() -> (f32, u64, u64) {
    // Try nvidia-smi as a quick check
    #[cfg(target_os = "linux")]
    {
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let parts: Vec<&str> = stdout.trim().split(", ").collect();
                if parts.len() == 3 {
                    let util = parts[0].parse::<f32>().unwrap_or(0.0);
                    let mem_used = parts[1].parse::<u64>().unwrap_or(0) * 1024 * 1024; // MiB→bytes
                    let mem_total = parts[2].parse::<u64>().unwrap_or(0) * 1024 * 1024;
                    return (util, mem_used, mem_total);
                }
            }
        }
    }
    (0.0, 0, 0)
}

/// Estimate network capacity from observed throughput
fn estimate_net_capacity(current_bps: u64) -> u64 {
    // Heuristic: if we see high throughput, capacity is at least 2x that
    // Default to 1Gbps if we can't tell
    let min_capacity = 125_000_000; // 1 Gbps in bytes/sec
    let estimated = current_bps.saturating_mul(2);
    estimated.max(min_capacity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_monitor_creation() {
        let monitor = ResourceMonitor::new();
        let snap = monitor.snapshot();
        assert_eq!(snap.cpu_total, 0.0);
        assert_eq!(snap.ram_used, 0);
    }

    #[test]
    fn test_idle_calculations() {
        let monitor = ResourceMonitor::new();
        assert_eq!(monitor.idle_cpu_pct(), 100.0);
    }

    #[test]
    fn test_network_bytes_linux() {
        let (rx, tx) = get_network_bytes();
        // On Linux CI, should return something > 0
        #[cfg(target_os = "linux")]
        assert!(rx > 0 || tx > 0, "Expected non-zero network bytes on Linux");
    }
}
