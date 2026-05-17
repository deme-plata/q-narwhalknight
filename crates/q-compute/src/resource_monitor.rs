//! Resource Monitor — 100ms sampling of CPU/GPU/RAM/NET/DISK
//!
//! Feeds the orchestrator with real-time utilization data
//! so it can assign work to idle resources.
//!
//! ## GPU monitoring (#012)
//!
//! GPU stats are queried asynchronously via `tokio::process::Command` to avoid
//! blocking the tokio runtime. Results are cached for 2 seconds (GPU utilization
//! doesn't meaningfully change at 100ms granularity). Backend detection (nvidia-smi,
//! rocm-smi, or sysinfo fallback) runs once at startup. If the GPU tool is
//! unavailable, a single warning is logged and all subsequent GPU reads return
//! cached/zero values without retrying.

use crate::ResourceSnapshot;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;
use sysinfo::{System, CpuRefreshKind, MemoryRefreshKind, RefreshKind};
use tracing::{debug, trace, warn};

/// How often to sample resources (100ms = 10 samples/sec)
const SAMPLE_INTERVAL: Duration = Duration::from_millis(100);

/// How long to cache GPU results before re-querying
const GPU_CACHE_TTL: Duration = Duration::from_secs(2);

/// Maximum time to wait for a GPU CLI query before giving up.
/// 2 seconds is generous — nvidia-smi typically completes in 50-200ms.
/// If it takes longer, something is wrong (driver hang, GPU reset, etc.)
/// and we fall back to cached data.
const GPU_QUERY_TIMEOUT: Duration = Duration::from_secs(2);

/// Cached GPU query result
#[derive(Debug, Clone)]
struct GpuStats {
    utilization: f32,
    memory_used: u64,
    memory_total: u64,
    temperature: f32,
    gpu_name: String,
}

impl Default for GpuStats {
    fn default() -> Self {
        Self {
            utilization: 0.0,
            memory_used: 0,
            memory_total: 0,
            temperature: 0.0,
            gpu_name: String::new(),
        }
    }
}

impl GpuStats {
    fn as_tuple(&self) -> (f32, u64, u64, f32, String) {
        (self.utilization, self.memory_used, self.memory_total, self.temperature, self.gpu_name.clone())
    }
}

/// GPU result cache — avoids re-querying the CLI tool on every sample tick.
/// The backend is detected once at startup via `GpuBackend::detect()`.
struct GpuCache {
    last_result: Option<GpuStats>,
    last_query: tokio::time::Instant,
    backend: GpuBackend,
}

impl GpuCache {
    fn new(backend: GpuBackend) -> Self {
        Self {
            last_result: None,
            // Start in the past so the first tick triggers a query immediately
            last_query: tokio::time::Instant::now() - GPU_CACHE_TTL,
            backend,
        }
    }
}

/// Resource monitor that runs in background, sampling every 100ms
pub struct ResourceMonitor {
    /// Latest snapshot (lock-free read via RwLock)
    latest: Arc<RwLock<ResourceSnapshot>>,
    /// Historical snapshots for trend analysis (last 60 seconds = 600 samples)
    history: Arc<RwLock<std::collections::VecDeque<ResourceSnapshot>>>,
    /// Stop flag
    running: Arc<AtomicBool>,
    /// GPU cache with backend detection and TTL (updated every 2s, non-blocking)
    gpu_cache: Arc<RwLock<GpuCache>>,
}

impl ResourceMonitor {
    pub fn new() -> Self {
        // Backend detection happens lazily on first GPU poll tick (async),
        // so we initialize with Unknown here.
        Self {
            latest: Arc::new(RwLock::new(ResourceSnapshot {
                cpu_per_core: Vec::new(),
                cpu_total: 0.0,
                gpu_utilization: 0.0,
                gpu_memory_used: 0,
                gpu_memory_total: 0,
                gpu_temperature: 0.0,
                gpu_name: String::new(),
                ram_used: 0,
                ram_total: 0,
                net_tx_bps: 0,
                net_rx_bps: 0,
                net_capacity_bps: 0,
                disk_io_bps: 0,
                timestamp_ms: 0,
            })),
            history: Arc::new(RwLock::new(std::collections::VecDeque::with_capacity(600))),
            running: Arc::new(AtomicBool::new(false)),
            gpu_cache: Arc::new(RwLock::new(GpuCache::new(GpuBackend::Unknown))),
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

        // #012: Spawn a separate task for GPU sampling (every 2s, fully async)
        let gpu_cache_writer = gpu_cache.clone();
        let gpu_running = running.clone();
        tokio::spawn(async move {
            // Detect GPU backend once at startup (async)
            let backend = GpuBackend::detect().await;
            debug!("GPU backend detected at startup: {:?}", backend);
            {
                let mut cache = gpu_cache_writer.write();
                cache.backend = backend;
            }

            // If no GPU was detected, log once and exit the task entirely.
            // This avoids any further wasted work.
            if backend == GpuBackend::None {
                debug!("No GPU detected -- GPU sampling task exiting (will not retry)");
                return;
            }

            let mut gpu_interval = tokio::time::interval(GPU_CACHE_TTL);
            // Track whether we have already logged a timeout warning so we
            // don't spam the log on every 2s tick.
            let mut timeout_warned = false;

            while gpu_running.load(std::sync::atomic::Ordering::Relaxed) {
                gpu_interval.tick().await;

                let current_backend = gpu_cache_writer.read().backend;
                if current_backend == GpuBackend::None {
                    // Backend may have been set to None after a detection failure
                    continue;
                }

                // Query GPU stats asynchronously with a 2-second timeout
                let query_result = tokio::time::timeout(
                    GPU_QUERY_TIMEOUT,
                    query_gpu_async(current_backend),
                ).await;

                match query_result {
                    Ok(Some(stats)) => {
                        timeout_warned = false; // Reset warning flag on success
                        let mut cache = gpu_cache_writer.write();
                        cache.last_result = Some(stats);
                        cache.last_query = tokio::time::Instant::now();
                    }
                    Ok(None) => {
                        // Query ran but returned no data -- keep cached value
                        trace!("GPU query returned no data, keeping cached value");
                    }
                    Err(_elapsed) => {
                        // Timeout -- log warning once, then use cached value
                        if !timeout_warned {
                            warn!(
                                "GPU query exceeded {}s timeout, using cached value \
                                 (this warning will not repeat until next successful query)",
                                GPU_QUERY_TIMEOUT.as_secs()
                            );
                            timeout_warned = true;
                        }
                    }
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

            debug!("Resource monitor started -- sampling every {}ms", SAMPLE_INTERVAL.as_millis());

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

                // #012: Read cached GPU stats (non-blocking read of GpuCache)
                let (gpu_util, gpu_mem_used, gpu_mem_total, gpu_temp, gpu_name) = {
                    let cache = gpu_cache.read();
                    cache.last_result.clone().unwrap_or_default().as_tuple()
                };

                let timestamp_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;

                // Disk I/O -- read from /proc/diskstats on Linux
                let disk_io_bps = get_disk_io_bps(elapsed.as_secs_f64());

                let snapshot = ResourceSnapshot {
                    cpu_per_core,
                    cpu_total,
                    gpu_utilization: gpu_util,
                    gpu_memory_used: gpu_mem_used,
                    gpu_memory_total: gpu_mem_total,
                    gpu_temperature: gpu_temp,
                    gpu_name,
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

                // #031: VecDeque ring buffer -- O(1) push_back + pop_front
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
                        "Resource monitor: CPU={:.1}% RAM={:.1}% NET={:.1}Mbps GPU={:.0}% temp={:.0}C (sample #{})",
                        cpu_total,
                        (ram_used as f64 / ram_total.max(1) as f64) * 100.0,
                        (net_rx_bps + net_tx_bps) as f64 / 125_000.0,
                        gpu_util,
                        gpu_temp,
                        sample_count,
                    );
                }
            }

            debug!("Resource monitor stopped after {} samples", sample_count);
        })
    }

    /// Stop the monitor
    pub fn stop(&self) {
        self.running.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Read total network bytes from /proc/net/dev (Linux) or return 0 (other OS)
fn get_network_bytes() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/net/dev") {
            return parse_proc_net_dev(&content);
        }
    }
    (0, 0)
}

/// Parse /proc/net/dev content into (total_rx_bytes, total_tx_bytes).
/// Skips the loopback interface.
fn parse_proc_net_dev(content: &str) -> (u64, u64) {
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
    (total_rx, total_tx)
}

/// Which GPU detection backend we last succeeded with.
/// Detected once at startup via `GpuBackend::detect()`.
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

impl GpuBackend {
    /// Detect which GPU backend is available. Called once at startup.
    ///
    /// Probe order:
    /// 1. **nvidia-smi** -- NVIDIA proprietary driver CLI (most common).
    /// 2. **rocm-smi** -- AMD ROCm driver CLI.
    /// 3. **sysinfo component scan** -- detects presence via thermal sensors.
    async fn detect() -> Self {
        // 1. Check if nvidia-smi is available
        if try_nvidia_smi_async().await.is_some() {
            debug!("GPU backend: nvidia-smi detected");
            return GpuBackend::NvidiaSmi;
        }

        // 2. Check if rocm-smi is available
        if try_rocm_smi_async().await.is_some() {
            debug!("GPU backend: rocm-smi detected");
            return GpuBackend::RocmSmi;
        }

        // 3. Fallback: sysinfo component scan (sync, but lightweight)
        if try_sysinfo_gpu().is_some() {
            debug!("GPU backend: sysinfo thermal sensor detected");
            return GpuBackend::Sysinfo;
        }

        debug!("GPU backend: none detected -- GPU monitoring disabled");
        GpuBackend::None
    }
}

/// Query GPU stats asynchronously using the pre-detected backend.
async fn query_gpu_async(backend: GpuBackend) -> Option<GpuStats> {
    match backend {
        GpuBackend::NvidiaSmi => try_nvidia_smi_async().await,
        GpuBackend::RocmSmi => try_rocm_smi_async().await,
        GpuBackend::Sysinfo => try_sysinfo_gpu(),
        GpuBackend::None | GpuBackend::Unknown => None,
    }
}

/// Try to query GPU stats via `nvidia-smi` (NVIDIA proprietary driver).
/// Uses `tokio::process::Command` to avoid blocking the tokio worker thread.
///
/// Queries: utilization.gpu, memory.used, memory.total, temperature.gpu, name
/// Example output: `85, 4096, 8192, 72, NVIDIA GeForce RTX 4090`
async fn try_nvidia_smi_async() -> Option<GpuStats> {
    #[cfg(target_os = "linux")]
    {
        let output = tokio::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .await
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        return parse_nvidia_smi_output(&stdout);
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Parse nvidia-smi CSV output into `GpuStats`.
///
/// Expected format (from `--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name`):
/// ```text
/// 85, 4096, 8192, 72, NVIDIA GeForce RTX 4090
/// ```
///
/// Fields:
/// - utilization.gpu: GPU utilization percentage (0-100)
/// - memory.used: GPU memory used in MiB
/// - memory.total: GPU memory total in MiB
/// - temperature.gpu: GPU temperature in degrees Celsius
/// - name: GPU device name (may contain commas in theory, but NVIDIA names don't)
///
/// For multi-GPU systems, nvidia-smi outputs one line per GPU. We take the first
/// GPU's stats (GPU 0) which is typically the primary compute GPU.
fn parse_nvidia_smi_output(stdout: &str) -> Option<GpuStats> {
    // Take first non-empty line (GPU 0 for multi-GPU systems)
    let line = stdout.lines().find(|l| !l.trim().is_empty())?;
    let parts: Vec<&str> = line.splitn(5, ", ").collect();
    if parts.len() >= 4 {
        let utilization = parts[0].trim().parse::<f32>().unwrap_or(0.0);
        let memory_used = parts[1].trim().parse::<u64>().unwrap_or(0) * 1024 * 1024; // MiB -> bytes
        let memory_total = parts[2].trim().parse::<u64>().unwrap_or(0) * 1024 * 1024;
        let temperature = parts[3].trim().parse::<f32>().unwrap_or(0.0);
        let gpu_name = if parts.len() >= 5 {
            parts[4].trim().to_string()
        } else {
            String::new()
        };
        return Some(GpuStats { utilization, memory_used, memory_total, temperature, gpu_name });
    }
    None
}

/// Try to query GPU stats via `rocm-smi` (AMD ROCm driver).
/// Uses `tokio::process::Command` to avoid blocking the tokio worker thread.
///
/// `rocm-smi` output for `--showuse --showmeminfo vram --showtemp` is multi-line:
/// ```text
/// GPU[0]          : GPU use (%): 42
/// GPU[0]          : vram Total Memory (B): 17163091968
/// GPU[0]          : vram Total Used Memory (B): 2147483648
/// GPU[0]          : Temperature (Sensor edge) (C): 65.0
/// ```
async fn try_rocm_smi_async() -> Option<GpuStats> {
    #[cfg(target_os = "linux")]
    {
        let output = tokio::process::Command::new("rocm-smi")
            .args(["--showuse", "--showmeminfo", "vram", "--showtemp"])
            .output()
            .await
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        return parse_rocm_smi_output(&stdout);
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Parse rocm-smi multi-line output into `GpuStats`.
///
/// Expected lines (any order, prefixed with GPU[N]):
/// - `GPU[0]          : GPU use (%): 42`
/// - `GPU[0]          : vram Total Memory (B): 17163091968`
/// - `GPU[0]          : vram Total Used Memory (B): 2147483648`
/// - `GPU[0]          : Temperature (Sensor edge) (C): 65.0`
fn parse_rocm_smi_output(stdout: &str) -> Option<GpuStats> {
    let mut utilization: f32 = 0.0;
    let mut memory_total: u64 = 0;
    let mut memory_used: u64 = 0;
    let mut temperature: f32 = 0.0;

    for line in stdout.lines() {
        let line = line.trim();
        if line.contains("GPU use (%)") {
            if let Some(val) = line.rsplit(':').next() {
                utilization = val.trim().parse::<f32>().unwrap_or(0.0);
            }
        } else if line.contains("vram Total Memory (B)") {
            if let Some(val) = line.rsplit(':').next() {
                memory_total = val.trim().parse::<u64>().unwrap_or(0);
            }
        } else if line.contains("vram Total Used Memory (B)") {
            if let Some(val) = line.rsplit(':').next() {
                memory_used = val.trim().parse::<u64>().unwrap_or(0);
            }
        } else if line.contains("Temperature") && line.contains("(C)") {
            if let Some(val) = line.rsplit(':').next() {
                temperature = val.trim().parse::<f32>().unwrap_or(0.0);
            }
        }
    }

    // Only return if we got at least the utilization value or memory info
    if utilization > 0.0 || memory_total > 0 {
        return Some(GpuStats {
            utilization,
            memory_used,
            memory_total,
            temperature,
            gpu_name: "AMD GPU".to_string(), // rocm-smi doesn't provide name in this query
        });
    }
    None
}

/// Try to detect a GPU via `sysinfo` component list.
///
/// `sysinfo` exposes hardware thermal sensors. On some Linux drivers (notably
/// NVIDIA with the open-source `nouveau` driver, and some AMD AMDGPU setups)
/// there will be a component whose label contains "gpu". This does NOT provide
/// utilization or memory -- only presence detection -- so we return
/// a sentinel `GpuStats { utilization: 1.0, .. }` meaning "GPU present, utilization unknown".
fn try_sysinfo_gpu() -> Option<GpuStats> {
    use sysinfo::Components;

    let components = Components::new_with_refreshed_list();
    for component in &components {
        let label = component.label().to_lowercase();
        if label.contains("gpu") || label.contains("radeon") || label.contains("nvidia") || label.contains("geforce") {
            // GPU detected via thermal sensor -- return sentinel utilization
            debug!(
                "GPU detected via sysinfo component: '{}' (temp={:.1}C)",
                component.label(),
                component.temperature(),
            );
            return Some(GpuStats {
                utilization: 1.0,
                memory_used: 0,
                memory_total: 0,
                temperature: component.temperature(),
                gpu_name: component.label().to_string(),
            });
        }
    }
    None
}

/// Read disk I/O bytes/sec from /proc/diskstats (Linux) or return 0 (other OS).
///
/// `/proc/diskstats` format (selected fields):
/// ```text
/// major minor name rd_ios rd_merge rd_sectors rd_ticks wr_ios wr_merge wr_sectors ...
///   8     0   sda    1234    56    789012     3456     7890    12    345678  ...
/// ```
///
/// Fields 6 and 10 (0-indexed from the name column) are sectors read/written.
/// We track the delta of `rd_sectors + wr_sectors` between samples.
/// Sector size is 512 bytes on Linux.
fn get_disk_io_bps(dt_secs: f64) -> u64 {
    #[cfg(target_os = "linux")]
    {
        use std::sync::Mutex;
        static PREV_SECTORS: std::sync::LazyLock<Mutex<u64>> = std::sync::LazyLock::new(|| Mutex::new(0));

        if let Ok(content) = std::fs::read_to_string("/proc/diskstats") {
            let total_sectors = parse_proc_diskstats(&content);

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

/// Parse /proc/diskstats content and return total sectors (read + written)
/// across all whole-disk devices. Skips partitions, loop, dm, ram devices.
fn parse_proc_diskstats(content: &str) -> u64 {
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
        if name.contains('p') && name.starts_with("nvme") {
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
    total_sectors
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

    // ===================================================================
    // ResourceMonitor basic tests
    // ===================================================================

    #[test]
    fn test_resource_monitor_creation() {
        let monitor = ResourceMonitor::new();
        let snap = monitor.snapshot();
        assert_eq!(snap.cpu_total, 0.0);
        assert_eq!(snap.ram_used, 0);
        assert_eq!(snap.gpu_temperature, 0.0);
        assert!(snap.gpu_name.is_empty());
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
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (rx, tx);
        }
    }

    #[test]
    fn test_gpu_stats_default() {
        let stats = GpuStats::default();
        assert_eq!(stats.utilization, 0.0);
        assert_eq!(stats.memory_used, 0);
        assert_eq!(stats.memory_total, 0);
        assert_eq!(stats.temperature, 0.0);
        assert!(stats.gpu_name.is_empty());
        let (u, mu, mt, t, n) = stats.as_tuple();
        assert_eq!(u, 0.0);
        assert_eq!(mu, 0);
        assert_eq!(mt, 0);
        assert_eq!(t, 0.0);
        assert!(n.is_empty());
    }

    #[test]
    fn test_gpu_cache_starts_expired() {
        // GpuCache should start with last_query in the past so the first tick
        // triggers an immediate query rather than waiting for the TTL.
        let cache = GpuCache::new(GpuBackend::Unknown);
        assert!(cache.last_result.is_none());
        assert!(cache.last_query.elapsed() >= GPU_CACHE_TTL);
    }

    // ===================================================================
    // nvidia-smi parsing tests
    // ===================================================================

    #[test]
    fn test_parse_nvidia_smi_typical_output() {
        // Typical nvidia-smi output for a single RTX 4090
        let output = "85, 4096, 24576, 72, NVIDIA GeForce RTX 4090\n";
        let stats = parse_nvidia_smi_output(output).expect("Should parse successfully");
        assert_eq!(stats.utilization, 85.0);
        assert_eq!(stats.memory_used, 4096 * 1024 * 1024);
        assert_eq!(stats.memory_total, 24576 * 1024 * 1024);
        assert_eq!(stats.temperature, 72.0);
        assert_eq!(stats.gpu_name, "NVIDIA GeForce RTX 4090");
    }

    #[test]
    fn test_parse_nvidia_smi_idle_gpu() {
        // GPU at idle
        let output = "0, 512, 8192, 35, NVIDIA GeForce RTX 3070\n";
        let stats = parse_nvidia_smi_output(output).expect("Should parse successfully");
        assert_eq!(stats.utilization, 0.0);
        assert_eq!(stats.memory_used, 512 * 1024 * 1024);
        assert_eq!(stats.memory_total, 8192 * 1024 * 1024);
        assert_eq!(stats.temperature, 35.0);
        assert_eq!(stats.gpu_name, "NVIDIA GeForce RTX 3070");
    }

    #[test]
    fn test_parse_nvidia_smi_full_load() {
        // GPU under full load
        let output = "100, 23000, 24576, 89, NVIDIA A100-SXM4-80GB\n";
        let stats = parse_nvidia_smi_output(output).expect("Should parse successfully");
        assert_eq!(stats.utilization, 100.0);
        assert_eq!(stats.memory_used, 23000 * 1024 * 1024);
        assert_eq!(stats.memory_total, 24576 * 1024 * 1024);
        assert_eq!(stats.temperature, 89.0);
        assert_eq!(stats.gpu_name, "NVIDIA A100-SXM4-80GB");
    }

    #[test]
    fn test_parse_nvidia_smi_multi_gpu_takes_first() {
        // Multi-GPU system -- should take GPU 0 (first line)
        let output = "\
85, 4096, 24576, 72, NVIDIA GeForce RTX 4090
92, 8192, 24576, 78, NVIDIA GeForce RTX 4090
";
        let stats = parse_nvidia_smi_output(output).expect("Should parse successfully");
        assert_eq!(stats.utilization, 85.0);
        assert_eq!(stats.memory_used, 4096 * 1024 * 1024);
        assert_eq!(stats.temperature, 72.0);
    }

    #[test]
    fn test_parse_nvidia_smi_without_name() {
        // Output with only 4 fields (no name)
        let output = "50, 2048, 8192, 55\n";
        let stats = parse_nvidia_smi_output(output).expect("Should parse successfully");
        assert_eq!(stats.utilization, 50.0);
        assert_eq!(stats.memory_used, 2048 * 1024 * 1024);
        assert_eq!(stats.memory_total, 8192 * 1024 * 1024);
        assert_eq!(stats.temperature, 55.0);
        assert!(stats.gpu_name.is_empty());
    }

    #[test]
    fn test_parse_nvidia_smi_empty_output() {
        assert!(parse_nvidia_smi_output("").is_none());
        assert!(parse_nvidia_smi_output("\n").is_none());
        assert!(parse_nvidia_smi_output("  \n  \n").is_none());
    }

    #[test]
    fn test_parse_nvidia_smi_garbage_output() {
        // Garbage that doesn't match expected format
        let output = "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n";
        assert!(parse_nvidia_smi_output(output).is_none());
    }

    #[test]
    fn test_parse_nvidia_smi_partial_garbage_values() {
        // Some fields parse, some don't -- should still return with defaults for bad fields
        let output = "abc, 4096, xyz, 72, Test GPU\n";
        let stats = parse_nvidia_smi_output(output).expect("Should parse with defaults");
        assert_eq!(stats.utilization, 0.0);  // "abc" -> 0.0
        assert_eq!(stats.memory_used, 4096 * 1024 * 1024);
        assert_eq!(stats.memory_total, 0);   // "xyz" -> 0
        assert_eq!(stats.temperature, 72.0);
        assert_eq!(stats.gpu_name, "Test GPU");
    }

    #[test]
    fn test_parse_nvidia_smi_leading_blank_lines() {
        // Sometimes nvidia-smi prefixes blank lines
        let output = "\n\n85, 4096, 24576, 72, NVIDIA GeForce RTX 4090\n";
        let stats = parse_nvidia_smi_output(output).expect("Should skip blank lines");
        assert_eq!(stats.utilization, 85.0);
    }

    #[test]
    fn test_parse_nvidia_smi_tesla_datacenter_gpu() {
        let output = "47, 15234, 40960, 62, Tesla V100-SXM2-32GB\n";
        let stats = parse_nvidia_smi_output(output).expect("Should parse datacenter GPU");
        assert_eq!(stats.utilization, 47.0);
        assert_eq!(stats.memory_used, 15234 * 1024 * 1024);
        assert_eq!(stats.memory_total, 40960 * 1024 * 1024);
        assert_eq!(stats.temperature, 62.0);
        assert_eq!(stats.gpu_name, "Tesla V100-SXM2-32GB");
    }

    // ===================================================================
    // rocm-smi parsing tests
    // ===================================================================

    #[test]
    fn test_parse_rocm_smi_typical_output() {
        let output = "\
GPU[0]          : GPU use (%): 42
GPU[0]          : vram Total Memory (B): 17163091968
GPU[0]          : vram Total Used Memory (B): 2147483648
GPU[0]          : Temperature (Sensor edge) (C): 65.0
";
        let stats = parse_rocm_smi_output(output).expect("Should parse successfully");
        assert_eq!(stats.utilization, 42.0);
        assert_eq!(stats.memory_total, 17163091968);
        assert_eq!(stats.memory_used, 2147483648);
        assert_eq!(stats.temperature, 65.0);
        assert_eq!(stats.gpu_name, "AMD GPU");
    }

    #[test]
    fn test_parse_rocm_smi_zero_utilization_with_memory() {
        // Idle AMD GPU -- utilization 0 but memory allocated
        let output = "\
GPU[0]          : GPU use (%): 0
GPU[0]          : vram Total Memory (B): 8589934592
GPU[0]          : vram Total Used Memory (B): 0
GPU[0]          : Temperature (Sensor edge) (C): 32.0
";
        let stats = parse_rocm_smi_output(output).expect("Should parse with 0% utilization if memory_total > 0");
        assert_eq!(stats.utilization, 0.0);
        assert_eq!(stats.memory_total, 8589934592);
        assert_eq!(stats.memory_used, 0);
        assert_eq!(stats.temperature, 32.0);
    }

    #[test]
    fn test_parse_rocm_smi_partial_output() {
        // Only utilization available
        let output = "GPU[0]          : GPU use (%): 75\n";
        let stats = parse_rocm_smi_output(output).expect("Should parse with partial data");
        assert_eq!(stats.utilization, 75.0);
        assert_eq!(stats.memory_total, 0);
        assert_eq!(stats.memory_used, 0);
        assert_eq!(stats.temperature, 0.0);
    }

    #[test]
    fn test_parse_rocm_smi_empty_output() {
        assert!(parse_rocm_smi_output("").is_none());
        assert!(parse_rocm_smi_output("\n\n").is_none());
    }

    #[test]
    fn test_parse_rocm_smi_no_matching_lines() {
        // Output from a different command that doesn't match our patterns
        let output = "Device: 0\nSerial Number: N/A\n";
        assert!(parse_rocm_smi_output(output).is_none());
    }

    #[test]
    fn test_parse_rocm_smi_temperature_variants() {
        // Some ROCm versions use different temperature sensor labels
        let output = "\
GPU[0]          : GPU use (%): 55
GPU[0]          : vram Total Memory (B): 16000000000
GPU[0]          : vram Total Used Memory (B): 4000000000
GPU[0]          : Temperature (Sensor junction) (C): 78.0
";
        let stats = parse_rocm_smi_output(output).expect("Should parse junction sensor");
        assert_eq!(stats.temperature, 78.0);
    }

    // ===================================================================
    // /proc/diskstats parsing tests
    // ===================================================================

    #[test]
    fn test_parse_proc_diskstats_typical() {
        // Realistic /proc/diskstats content with sda and sda1
        let content = "\
   8       0 sda 12345 678 901234 5678 9012 345 678901 2345 0 6789 8023
   8       1 sda1 12000 600 890000 5500 9000 340 670000 2300 0 6500 7800
   8      16 sdb 5000 200 300000 1000 4000 100 200000 500 0 3000 1500
   8      17 sdb1 4800 180 290000 900 3800 90 190000 450 0 2800 1350
";
        let total = parse_proc_diskstats(content);
        // sda: rd=901234 + wr=678901 = 1580135
        // sda1: skipped (partition)
        // sdb: rd=300000 + wr=200000 = 500000
        // sdb1: skipped (partition)
        assert_eq!(total, 901234 + 678901 + 300000 + 200000);
    }

    #[test]
    fn test_parse_proc_diskstats_nvme() {
        // NVMe device with partitions
        let content = "\
 259       0 nvme0n1 100000 5000 2000000 50000 80000 3000 1500000 40000 0 60000 90000
 259       1 nvme0n1p1 95000 4800 1900000 48000 78000 2900 1450000 38000 0 58000 86000
 259       2 nvme0n1p2 5000 200 100000 2000 2000 100 50000 2000 0 2000 4000
";
        let total = parse_proc_diskstats(content);
        // nvme0n1: rd=2000000 + wr=1500000 = 3500000
        // nvme0n1p1: skipped (partition with 'p')
        // nvme0n1p2: skipped (partition with 'p')
        assert_eq!(total, 2000000 + 1500000);
    }

    #[test]
    fn test_parse_proc_diskstats_skips_loop_dm_ram() {
        let content = "\
   7       0 loop0 100 0 200 50 0 0 0 0 0 100 50
   7       1 loop1 50 0 100 25 0 0 0 0 0 50 25
 253       0 dm-0 5000 0 400000 2000 3000 0 200000 1000 0 3000 3000
   1       0 ram0 0 0 0 0 0 0 0 0 0 0 0
   8       0 sda 10000 500 800000 4000 7000 300 500000 2000 0 5000 6000
";
        let total = parse_proc_diskstats(content);
        // Only sda counts: rd=800000 + wr=500000 = 1300000
        assert_eq!(total, 800000 + 500000);
    }

    #[test]
    fn test_parse_proc_diskstats_empty() {
        assert_eq!(parse_proc_diskstats(""), 0);
    }

    #[test]
    fn test_parse_proc_diskstats_short_lines() {
        // Lines with fewer than 14 fields should be skipped
        let content = "   8       0 sda 12345 678\n";
        assert_eq!(parse_proc_diskstats(content), 0);
    }

    #[test]
    fn test_parse_proc_diskstats_vda_virtual_disk() {
        // Virtual disk common in cloud VMs
        let content = "\
 254       0 vda 50000 2000 1000000 25000 40000 1500 800000 20000 0 30000 45000
 254       1 vda1 48000 1900 950000 24000 39000 1400 780000 19000 0 29000 43000
";
        let total = parse_proc_diskstats(content);
        // vda: rd=1000000 + wr=800000 = 1800000
        // vda1: skipped (partition)
        assert_eq!(total, 1000000 + 800000);
    }

    // ===================================================================
    // /proc/net/dev parsing tests
    // ===================================================================

    #[test]
    fn test_parse_proc_net_dev_typical() {
        let content = "\
Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 1234567   8901    0    0    0     0          0         0  1234567    8901    0    0    0     0       0          0
  eth0: 9876543  21098    0    0    0     0          0         0  5432109   10987    0    0    0     0       0          0
";
        let (rx, tx) = parse_proc_net_dev(content);
        // lo is skipped, only eth0 counts
        assert_eq!(rx, 9876543);
        assert_eq!(tx, 5432109);
    }

    #[test]
    fn test_parse_proc_net_dev_multiple_interfaces() {
        let content = "\
Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 1000000   500    0    0    0     0          0         0  1000000     500    0    0    0     0       0          0
  eth0: 5000000  2000    0    0    0     0          0         0  3000000    1500    0    0    0     0       0          0
 wlan0: 2000000  1000    0    0    0     0          0         0  1000000     800    0    0    0     0       0          0
";
        let (rx, tx) = parse_proc_net_dev(content);
        // eth0 + wlan0 (lo skipped)
        assert_eq!(rx, 5000000 + 2000000);
        assert_eq!(tx, 3000000 + 1000000);
    }

    #[test]
    fn test_parse_proc_net_dev_empty() {
        let (rx, tx) = parse_proc_net_dev("");
        assert_eq!(rx, 0);
        assert_eq!(tx, 0);
    }

    // ===================================================================
    // Async integration tests
    // ===================================================================

    /// Verify that GPU monitoring does not block the tokio runtime.
    ///
    /// We spawn the resource monitor, then concurrently run a future that
    /// must complete within 500ms. If GPU sampling were blocking the worker
    /// thread, this concurrent future would be starved and the timeout
    /// would fire.
    #[tokio::test]
    async fn test_gpu_monitoring_does_not_block_runtime() {
        let monitor = ResourceMonitor::new();
        let handle = monitor.spawn();

        // Run a concurrent async task that should complete almost instantly.
        // If GPU sampling blocks the runtime, this will time out.
        let concurrent_work = async {
            let mut sum = 0u64;
            for _ in 0..10 {
                tokio::time::sleep(Duration::from_millis(5)).await;
                sum += 1;
            }
            sum
        };

        let result = tokio::time::timeout(Duration::from_millis(500), concurrent_work).await;
        assert!(result.is_ok(), "Concurrent async task was blocked -- GPU monitoring likely blocking the runtime");
        assert_eq!(result.unwrap(), 10);

        // Verify we can still read a snapshot (the monitor task is running)
        let snap = monitor.snapshot();
        // cpu_total could be 0.0 if the monitor hasn't had time to sample yet,
        // but reading should not panic or block.
        let _ = snap.cpu_total;
        // New fields should be initialized to defaults
        assert!(snap.gpu_temperature >= 0.0);

        monitor.stop();
        // Give the tasks time to notice the stop flag
        tokio::time::sleep(Duration::from_millis(150)).await;
        drop(handle);
    }

    #[tokio::test]
    async fn test_gpu_backend_detect_completes() {
        // GpuBackend::detect() should return without blocking, regardless of
        // whether nvidia-smi or rocm-smi are installed.
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            GpuBackend::detect(),
        ).await;
        assert!(result.is_ok(), "GpuBackend::detect() timed out");
        // On a server without GPU, we expect None; with GPU, NvidiaSmi or RocmSmi.
        // Just verify it doesn't panic.
        let _backend = result.unwrap();
    }

    // ===================================================================
    // Utility tests
    // ===================================================================

    #[test]
    fn test_estimate_net_capacity() {
        // Low throughput -- should return minimum 1Gbps
        assert_eq!(estimate_net_capacity(1000), 125_000_000);

        // High throughput -- should return 2x observed
        assert_eq!(estimate_net_capacity(200_000_000), 400_000_000);

        // Zero throughput -- minimum
        assert_eq!(estimate_net_capacity(0), 125_000_000);
    }

    #[test]
    fn test_gpu_query_timeout_is_two_seconds() {
        // Verify the constant matches the acceptance criteria
        assert_eq!(GPU_QUERY_TIMEOUT, Duration::from_secs(2));
    }

    #[test]
    fn test_gpu_cache_ttl_is_two_seconds() {
        assert_eq!(GPU_CACHE_TTL, Duration::from_secs(2));
    }
}
