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
    history: Arc<RwLock<std::collections::VecDeque<ResourceSnapshot>>>,
    /// Stop flag
    running: Arc<std::sync::atomic::AtomicBool>,
    /// #030: Cached GPU stats (updated every 2s via spawn_blocking)
    gpu_cache: Arc<RwLock<(f32, u64, u64)>>,
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
            history: Arc::new(RwLock::new(std::collections::VecDeque::with_capacity(600))),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            gpu_cache: Arc::new(RwLock::new((0.0, 0, 0))),
        }
    }

    /// Get the latest resource snapshot (lock-free read)
    pub fn snapshot(&self) -> ResourceSnapshot {
        self.latest.read().clone()
    }

    /// Get historical snapshots for trend analysis
    pub fn history(&self, last_n: usize) -> Vec<ResourceSnapshot> {
        let h = self.history.read();
        let len = h.len();
        let start = len.saturating_sub(last_n);
        h.iter().skip(start).cloned().collect()
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
        let gpu_cache = self.gpu_cache.clone();
        running.store(true, std::sync::atomic::Ordering::SeqCst);

        // #030: Spawn a separate task for GPU sampling (every 2s, via spawn_blocking)
        let gpu_cache_writer = gpu_cache.clone();
        let gpu_running = running.clone();
        tokio::spawn(async move {
            let mut gpu_interval = tokio::time::interval(Duration::from_secs(2));
            while gpu_running.load(std::sync::atomic::Ordering::Relaxed) {
                gpu_interval.tick().await;
                let result = tokio::task::spawn_blocking(get_gpu_stats).await;
                if let Ok(stats) = result {
                    *gpu_cache_writer.write() = stats;
                }
            }
        });

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

                // #030: Read cached GPU stats (non-blocking)
                let (gpu_util, gpu_mem_used, gpu_mem_total) = *gpu_cache.read();

                let timestamp_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;

                // Disk I/O — read from /proc/diskstats on Linux
                let disk_io_bps = get_disk_io_bps(elapsed.as_secs_f64());

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
                    disk_io_bps,
                    timestamp_ms,
                };

                // Update latest
                *latest.write() = snapshot.clone();

                // #031: VecDeque ring buffer — O(1) push_back + pop_front
                {
                    let mut h = history.write();
                    if h.len() >= 600 {
                        h.pop_front();
                    }
                    h.push_back(snapshot);
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

/// Which GPU detection backend we last succeeded with.
/// Cached so we don't retry failing backends on every 2-second tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuBackend {
    /// Haven't probed yet
    Unknown,
    /// nvidia-smi CLI worked
    NvidiaSmi,
    /// rocm-smi CLI worked (AMD)
    RocmSmi,
    /// sysinfo component list detected a GPU (basic: reports presence, not utilization)
    Sysinfo,
    /// No GPU detected at all
    None,
}

/// Get GPU stats with multi-backend detection.
///
/// Probe order:
/// 1. **sysinfo component scan** -- detects GPU thermal sensors on some drivers,
///    but does not provide utilization. If a GPU component is found we note it
///    and still try CLI tools for richer data.
/// 2. **nvidia-smi** -- NVIDIA proprietary driver CLI (most common).
/// 3. **rocm-smi** -- AMD ROCm driver CLI.
///
/// After the first successful backend is found it is cached in a process-global
/// static so subsequent calls skip failing paths (the 2-second GPU poll loop
/// calls this function repeatedly).
fn get_gpu_stats() -> (f32, u64, u64) {
    use std::sync::Mutex;
    static CACHED_BACKEND: std::sync::LazyLock<Mutex<GpuBackend>> =
        std::sync::LazyLock::new(|| Mutex::new(GpuBackend::Unknown));

    let mut backend = CACHED_BACKEND.lock().unwrap();

    // If we already know there is no GPU, short-circuit
    if *backend == GpuBackend::None {
        return (0.0, 0, 0);
    }

    // ── 1. Try nvidia-smi (if backend is Unknown or NvidiaSmi) ──
    if *backend == GpuBackend::Unknown || *backend == GpuBackend::NvidiaSmi {
        if let Some(stats) = try_nvidia_smi() {
            *backend = GpuBackend::NvidiaSmi;
            return stats;
        }
    }

    // ── 2. Try rocm-smi (AMD ROCm) ──
    if *backend == GpuBackend::Unknown || *backend == GpuBackend::RocmSmi {
        if let Some(stats) = try_rocm_smi() {
            *backend = GpuBackend::RocmSmi;
            return stats;
        }
    }

    // ── 3. sysinfo component scan (last resort — detects presence only) ──
    if *backend == GpuBackend::Unknown || *backend == GpuBackend::Sysinfo {
        if let Some(stats) = try_sysinfo_gpu() {
            *backend = GpuBackend::Sysinfo;
            return stats;
        }
    }

    // No GPU found on any backend
    if *backend == GpuBackend::Unknown {
        *backend = GpuBackend::None;
    }
    (0.0, 0, 0)
}

/// Try to query GPU stats via `nvidia-smi` (NVIDIA proprietary driver).
fn try_nvidia_smi() -> Option<(f32, u64, u64)> {
    #[cfg(target_os = "linux")]
    {
        let output = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = stdout.trim().split(", ").collect();
        if parts.len() == 3 {
            let util = parts[0].parse::<f32>().unwrap_or(0.0);
            let mem_used = parts[1].parse::<u64>().unwrap_or(0) * 1024 * 1024; // MiB -> bytes
            let mem_total = parts[2].parse::<u64>().unwrap_or(0) * 1024 * 1024;
            return Some((util, mem_used, mem_total));
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (); // suppress unused warning
    None
}

/// Try to query GPU stats via `rocm-smi` (AMD ROCm driver).
///
/// `rocm-smi` output for `--showuse --showmeminfo vram` is multi-line:
/// ```text
/// GPU[0]          : GPU use (%): 42
/// GPU[0]          : vram Total Memory (B): 17163091968
/// GPU[0]          : vram Total Used Memory (B): 2147483648
/// ```
fn try_rocm_smi() -> Option<(f32, u64, u64)> {
    #[cfg(target_os = "linux")]
    {
        let output = std::process::Command::new("rocm-smi")
            .args(["--showuse", "--showmeminfo", "vram"])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut util: f32 = 0.0;
        let mut mem_total: u64 = 0;
        let mut mem_used: u64 = 0;

        for line in stdout.lines() {
            let line = line.trim();
            if line.contains("GPU use (%)") {
                // "GPU[0]          : GPU use (%): 42"
                if let Some(val) = line.rsplit(':').next() {
                    util = val.trim().parse::<f32>().unwrap_or(0.0);
                }
            } else if line.contains("vram Total Memory (B)") {
                if let Some(val) = line.rsplit(':').next() {
                    mem_total = val.trim().parse::<u64>().unwrap_or(0);
                }
            } else if line.contains("vram Total Used Memory (B)") {
                if let Some(val) = line.rsplit(':').next() {
                    mem_used = val.trim().parse::<u64>().unwrap_or(0);
                }
            }
        }

        // Only return if we got at least the utilization value
        if util > 0.0 || mem_total > 0 {
            return Some((util, mem_used, mem_total));
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (); // suppress unused warning
    None
}

/// Try to detect a GPU via `sysinfo` component list.
///
/// `sysinfo` exposes hardware thermal sensors. On some Linux drivers (notably
/// NVIDIA with the open-source `nouveau` driver, and some AMD AMDGPU setups)
/// there will be a component whose label contains "gpu". This does NOT provide
/// utilization or memory — only presence detection — so we return
/// `(1.0, 0, 0)` as a sentinel meaning "GPU present, utilization unknown".
fn try_sysinfo_gpu() -> Option<(f32, u64, u64)> {
    use sysinfo::Components;

    let components = Components::new_with_refreshed_list();
    for component in &components {
        let label = component.label().to_lowercase();
        if label.contains("gpu") || label.contains("radeon") || label.contains("nvidia") || label.contains("geforce") {
            // GPU detected via thermal sensor — return sentinel utilization
            // The temperature is informational but we don't report it as utilization.
            debug!(
                "GPU detected via sysinfo component: '{}' (temp={:.1}C)",
                component.label(),
                component.temperature(),
            );
            return Some((1.0, 0, 0));
        }
    }
    None
}

/// Read disk I/O bytes/sec from /proc/diskstats (Linux) or return 0 (other OS)
///
/// /proc/diskstats format (fields 6,10 are sectors read/written):
/// major minor name rd_ios rd_merge rd_sectors rd_ticks wr_ios wr_merge wr_sectors ...
/// We track the delta of rd_sectors + wr_sectors between samples.
/// Sector size is 512 bytes on Linux.
fn get_disk_io_bps(dt_secs: f64) -> u64 {
    use std::sync::Mutex;

    static PREV_SECTORS: std::sync::LazyLock<Mutex<u64>> = std::sync::LazyLock::new(|| Mutex::new(0));

    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/diskstats") {
            let mut total_sectors: u64 = 0;
            for line in content.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 14 { continue; }
                let name = parts[2];
                // Only count whole-disk devices (sda, nvme0n1, vda), not partitions
                if name.ends_with(|c: char| c.is_ascii_digit()) && !name.starts_with("nvme") {
                    // Skip partitions like sda1, vda1
                    let base = name.trim_end_matches(|c: char| c.is_ascii_digit());
                    if base != name { continue; }
                }
                // For nvme, skip partition entries (nvme0n1p1, etc.)
                if name.contains("p") && name.starts_with("nvme") {
                    continue;
                }
                // Skip loop, dm, ram devices
                if name.starts_with("loop") || name.starts_with("dm-") || name.starts_with("ram") {
                    continue;
                }

                // Field 6 = sectors read, Field 10 = sectors written (0-indexed from field 0)
                let rd_sectors = parts[5].parse::<u64>().unwrap_or(0);
                let wr_sectors = parts[9].parse::<u64>().unwrap_or(0);
                total_sectors += rd_sectors + wr_sectors;
            }

            let mut prev = PREV_SECTORS.lock().unwrap();
            let delta = total_sectors.saturating_sub(*prev);
            *prev = total_sectors;

            if dt_secs > 0.001 {
                // Each sector = 512 bytes
                return ((delta as f64 * 512.0) / dt_secs) as u64;
            }
        }
    }
    let _ = dt_secs; // suppress unused warning on non-Linux
    0
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
