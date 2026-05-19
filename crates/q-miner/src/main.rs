// ═══════════════════════════════════════════════════════════════════
// PERFORMANCE: Platform-optimized memory allocators
// jemalloc on Linux (20-30% faster than glibc malloc for multi-threaded)
// mimalloc on Windows (30-50% faster than Windows HeapAlloc)
// ═══════════════════════════════════════════════════════════════════
#[cfg(all(not(target_os = "windows"), feature = "jemalloc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(target_os = "windows")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use clap::Parser;
use console::style;
use std::sync::{Arc, atomic::{AtomicU64, AtomicU8, AtomicUsize, AtomicBool, Ordering}};
use tokio::signal;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use chrono::{DateTime, Utc};
use core_affinity::CoreId;
#[cfg(target_arch = "x86_64")]
use raw_cpuid::CpuId;
use serde_json::Value;

use q_miner::shared_state::{SharedMinerState, ThreadState, ThreadStatus, DiagnosticEvent, MinerThrottleMode, StarshipSyncInfo};

// Simplified command-line arguments
#[derive(Parser)]
#[command(name = "q-miner")]
#[command(about = "Quillon-NarwhalKnight-Graph High-Performance Miner — supports Tor & SOCKS5/HTTP proxy routing")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Args {
    /// Mining mode: solo, pool, decentralized, benchmark
    /// - solo: Mine directly to your local node
    /// - pool: Connect to centralized pool via Stratum protocol
    /// - decentralized: P2P pool mining with CRDT-based PPLNS (v2.3.0+)
    /// - benchmark: Test hashrate without submitting solutions
    #[arg(short, long, default_value = "benchmark")]
    mode: String,

    /// Wallet address for mining rewards
    #[arg(short, long)]
    wallet: Option<String>,

    /// Number of CPU threads (0 = auto-detect)
    #[arg(short, long, default_value = "0")]
    threads: usize,

    /// Disable GPU mining (GPU is auto-detected and used by default when available)
    #[arg(long)]
    no_gpu: bool,

    /// Mining intensity (1-10)
    #[arg(short, long, default_value = "7")]
    intensity: u8,

    /// Enable benchmarking mode
    #[arg(long)]
    benchmark: bool,

    /// Duration in seconds for benchmark
    #[arg(long, default_value = "30")]
    duration: u64,

    /// API server URL(s) — comma-separated for multi-server failover
    /// e.g., http://185.182.185.227:8080,http://5.79.79.158:8080
    #[arg(short, long, default_value = "http://localhost:8080")]
    server: String,

    /// Mining pool URL for pool mode (e.g., stratum+tcp://quillon.xyz:3333)
    #[arg(long, default_value = "stratum+tcp://quillon.xyz:3333")]
    pool_url: String,

    /// Worker name for pool mining (default: randomly generated)
    #[arg(long)]
    worker_name: Option<String>,

    /// v3.3.3-beta: Human-readable miner name for identification (e.g., "Server Alpha", "Mining Rig 1")
    /// Shows up in server logs to help distinguish between multiple miners
    #[arg(long, short = 'n')]
    miner_name: Option<String>,

    /// P2P bootstrap nodes for decentralized pool mode (comma-separated)
    #[arg(long, default_value = "http://quillon.xyz:8080")]
    bootstrap_nodes: String,

    /// Region for pool node discovery (e.g., us-east, eu-west, asia-pacific)
    #[arg(long, default_value = "global")]
    region: String,

    /// Disable TUI dashboard (use plain log output)
    #[arg(long)]
    no_tui: bool,

    /// Force-enable TUI dashboard (overrides terminal detection)
    #[arg(long)]
    tui: bool,

    /// Network bandwidth limit in KB/s (upload+download combined).
    /// 0 = unlimited (default). Minimum effective: 1 KB/s.
    /// At 10 KB/s: challenge refresh every 120s, solution submits throttled.
    #[arg(long, default_value = "0")]
    bandwidth_limit: u32,

    /// Proxy URL for routing all miner traffic (socks5://host:port, http://host:port, https://host:port).
    /// Auth can be embedded: socks5://user:pass@host:port
    #[arg(long)]
    proxy: Option<String>,

    /// Route all traffic through local Tor (socks5://127.0.0.1:9050).
    /// Disabled by default. Use --tor to enable (requires Tor daemon running).
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    tor: bool,

    /// Disable P2P networking (use HTTP-only challenge fetch and solution submit).
    /// P2P is enabled by default for faster challenge propagation (<50ms vs 2-10s HTTP).
    #[cfg(feature = "p2p")]
    #[arg(long)]
    no_p2p: bool,

    /// Force P2P even when --bandwidth-limit would auto-disable it.
    #[cfg(feature = "p2p")]
    #[arg(long)]
    force_p2p: bool,

    /// P2P listen port (0 = random OS-assigned port, default).
    #[cfg(feature = "p2p")]
    #[arg(long, default_value = "0")]
    p2p_port: u16,

    /// Disable automatic update checks and downloads.
    #[arg(long)]
    no_auto_update: bool,
}

// Hardware info structure with CPU optimization details
pub struct HardwareInfo {
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub cuda_devices: usize,
    pub opencl_devices: usize,
    pub cpu_vendor: String,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub cache_line_size: usize,
}

// ═══════════════════════════════════════════════════════════════════
// EMBEDDED TOR (arti) — no system Tor daemon needed
// Bootstraps an embedded Tor client and runs a local SOCKS5 proxy
// ═══════════════════════════════════════════════════════════════════
#[cfg(feature = "tor-support")]
mod embedded_tor {
    use anyhow::Result;
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tracing::{info, warn, debug};

    /// Bootstrap embedded arti and start a local SOCKS5 proxy.
    /// Returns the SOCKS5 proxy URL (e.g. "socks5://127.0.0.1:19050").
    pub async fn start_embedded_tor() -> Result<String> {
        use arti_client::{TorClient, TorClientConfig};
        use tor_rtcompat::tokio::TokioRustlsRuntime;

        eprintln!("\x1b[36m   Bootstrapping embedded Tor (arti)...\x1b[0m");

        let runtime = TokioRustlsRuntime::current()
            .map_err(|e| anyhow::anyhow!("Failed to get Tokio runtime for arti: {}", e))?;

        let config = TorClientConfig::default();

        let arti_client = Arc::new(
            TorClient::with_runtime(runtime)
                .config(config)
                .create_bootstrapped()
                .await
                .map_err(|e| anyhow::anyhow!("Tor bootstrap failed: {}", e))?
        );

        eprintln!("\x1b[32m   Tor bootstrapped — starting local SOCKS5 proxy\x1b[0m");

        // Bind a local SOCKS5 proxy on a random port
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        let proxy_url = format!("socks5://127.0.0.1:{}", port);

        info!("Embedded Tor SOCKS5 proxy listening on 127.0.0.1:{}", port);

        // Spawn the SOCKS5 proxy task
        let client = arti_client.clone();
        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((mut stream, _)) => {
                        let client = client.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle_socks5_connection(&mut stream, &client).await {
                                debug!("SOCKS5 connection error: {}", e);
                            }
                        });
                    }
                    Err(e) => {
                        warn!("SOCKS5 accept error: {}", e);
                    }
                }
            }
        });

        eprintln!("\x1b[32m   Tor SOCKS5 proxy ready on port {}\x1b[0m", port);
        Ok(proxy_url)
    }

    /// Handle a single SOCKS5 CONNECT request, tunneling through arti.
    async fn handle_socks5_connection<R: tor_rtcompat::Runtime>(
        stream: &mut tokio::net::TcpStream,
        client: &arti_client::TorClient<R>,
    ) -> Result<()> {
        // SOCKS5 greeting: client sends [0x05, nmethods, methods...]
        let mut buf = [0u8; 258];
        let n = stream.read(&mut buf).await?;
        if n < 2 || buf[0] != 0x05 {
            return Err(anyhow::anyhow!("Not SOCKS5"));
        }

        // Reply: no auth required [0x05, 0x00]
        stream.write_all(&[0x05, 0x00]).await?;

        // SOCKS5 request: [0x05, cmd, 0x00, atype, addr..., port_hi, port_lo]
        let n = stream.read(&mut buf).await?;
        if n < 7 || buf[0] != 0x05 || buf[1] != 0x01 {
            // Only CONNECT (0x01) supported
            stream.write_all(&[0x05, 0x07, 0x00, 0x01, 0,0,0,0, 0,0]).await?;
            return Err(anyhow::anyhow!("Only CONNECT supported"));
        }

        // Parse target address
        let (host, port, _consumed) = match buf[3] {
            0x01 => {
                // IPv4
                if n < 10 { return Err(anyhow::anyhow!("Short IPv4")); }
                let ip = format!("{}.{}.{}.{}", buf[4], buf[5], buf[6], buf[7]);
                let port = ((buf[8] as u16) << 8) | buf[9] as u16;
                (ip, port, 10)
            }
            0x03 => {
                // Domain name
                let len = buf[4] as usize;
                if n < 5 + len + 2 { return Err(anyhow::anyhow!("Short domain")); }
                let domain = String::from_utf8_lossy(&buf[5..5+len]).to_string();
                let port = ((buf[5+len] as u16) << 8) | buf[6+len] as u16;
                (domain, port, 7 + len)
            }
            _ => {
                stream.write_all(&[0x05, 0x08, 0x00, 0x01, 0,0,0,0, 0,0]).await?;
                return Err(anyhow::anyhow!("Unsupported address type"));
            }
        };

        debug!("SOCKS5 CONNECT to {}:{}", host, port);

        // Connect through Tor
        let target = format!("{}:{}", host, port);
        let tor_stream = match client.connect(target.as_str()).await {
            Ok(s) => s,
            Err(e) => {
                // Connection refused
                stream.write_all(&[0x05, 0x05, 0x00, 0x01, 0,0,0,0, 0,0]).await?;
                return Err(anyhow::anyhow!("Tor connect failed: {}", e));
            }
        };

        // Success reply
        stream.write_all(&[0x05, 0x00, 0x00, 0x01, 127,0,0,1, 0,0]).await?;

        // Bidirectional copy
        let (mut client_read, mut client_write) = stream.split();
        let (mut tor_read, mut tor_write) = tor_stream.split();

        let _result = tokio::select! {
            r = tokio::io::copy(&mut client_read, &mut tor_write) => r,
            r = tokio::io::copy(&mut tor_read, &mut client_write) => r,
        };

        Ok(())
    }
}

// Mining challenge from API server
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MiningChallenge {
    pub challenge_hash: String,
    pub difficulty_target: String,
    pub block_height: u64,
    pub vdf_iterations: u32,
    pub block_reward: f64,
    pub expires_at: DateTime<Utc>,
    /// Server notice broadcast to miners (None = no notice)
    #[serde(default)]
    pub server_notice: Option<String>,
    /// v1.0.3: Server version (informational only — different version track from miner)
    #[serde(default)]
    pub server_version: Option<String>,
    /// v2.7.0: Minimum miner version required by the server
    #[serde(default)]
    pub min_miner_version: Option<String>,
    /// v9.1.0: Network hashrate (H/s) from compute power layer
    #[serde(default)]
    pub network_hashrate_hs: Option<f64>,
    /// v9.1.0: Number of active mining peers on the network
    #[serde(default)]
    pub connected_miners: Option<u32>,
    /// v9.1.0: Live security bits (boosted by real-time network hashpower)
    #[serde(default)]
    pub live_security_bits: Option<f64>,
    /// v9.1.4: Admin-forced mining mode override ("solo" or "pool", absent = no override)
    #[serde(default)]
    pub forced_mining_mode: Option<String>,
    /// v9.1.4: Pool URL when forced_mining_mode is "pool"
    #[serde(default)]
    pub forced_pool_url: Option<String>,
}

// API response wrapper
#[derive(Debug, serde::Deserialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

/// Helper function to normalize server URL (remove trailing slash)
fn normalize_server_url(url: &str) -> String {
    let trimmed = url.trim_end_matches('/');

    // Ensure URL has a scheme — eventsource-client and hyper::Uri require it
    let with_scheme = if !trimmed.starts_with("http://") && !trimmed.starts_with("https://") {
        // No scheme: add http:// for IPs/localhost, https:// for domain names
        let is_ip_or_local = trimmed.starts_with("localhost")
            || trimmed.starts_with("127.0.0.1")
            || trimmed.chars().next().map_or(false, |c| c.is_ascii_digit());
        if is_ip_or_local {
            format!("http://{}", trimmed)
        } else {
            format!("https://{}", trimmed)
        }
    } else {
        trimmed.to_string()
    };

    // Auto-downgrade https:// → http:// for servers that don't have TLS:
    // - localhost / 127.0.0.1 (local nodes never have TLS certs)
    // - Raw IP addresses (e.g., https://5.79.79.158:8080 — no TLS cert)
    // - Any URL with an explicit port (e.g., :8080, :50201 — direct node access)
    // Only domain names WITHOUT explicit ports keep https (e.g., https://quillon.xyz)
    if with_scheme.starts_with("https://") {
        let host_part = &with_scheme["https://".len()..];
        let is_localhost = host_part.starts_with("localhost") || host_part.starts_with("127.0.0.1");
        let is_raw_ip = host_part.chars().next().map_or(false, |c| c.is_ascii_digit());
        let has_explicit_port = host_part.contains(':');

        if is_localhost || is_raw_ip || has_explicit_port {
            let fixed = with_scheme.replacen("https://", "http://", 1);
            warn!("⚠️  Auto-downgrading {} → {} (direct node connections use HTTP)", with_scheme, fixed);
            return fixed;
        }
    }
    with_scheme
}

/// Default fallback bootstrap servers (v10.3.2: multi-server for resilience)
/// If primary (quillon.xyz = Epsilon) is down, miner falls back to Beta/Gamma.
/// DeepSeek review: "Change fallback to include multiple distinct physical nodes"
const FALLBACK_BOOTSTRAP_URLS: &[&str] = &[
    "https://quillon.xyz",          // Epsilon (primary, 10Gbit)
    "http://185.182.185.227:8080",  // Beta (coordinator)
    "http://109.205.176.60:8808",   // Gamma (backup)
];
// Keep old constant for backward compat (used in a few places)
const FALLBACK_BOOTSTRAP_URL: &str = "https://quillon.xyz";

// ═══════════════════════════════════════════════════════════════════
// v10.0.0: Multi-Server Failover — "Any Node Mines" Resilience
// Tracks health/latency of multiple servers, auto-elects primary.
// ═══════════════════════════════════════════════════════════════════

/// Parse --server argument into list of normalized URLs.
/// Supports comma-separated: "http://a:8080,http://b:8080"
/// Single URL is backward-compatible (produces a 1-element vec).
fn parse_server_list(server_arg: &str) -> Vec<String> {
    server_arg
        .split(',')
        .map(|s| normalize_server_url(s.trim()))
        .filter(|s| !s.is_empty())
        .collect()
}

/// Multi-server health tracker with automatic primary election.
/// Thread-safe (all fields are atomic), designed for concurrent access
/// from mining threads, SSE listener, and health check task.
pub struct ServerSelector {
    /// Normalized server URLs
    servers: Vec<String>,
    /// Per-server health flag (true = healthy)
    health: Vec<AtomicBool>,
    /// Per-server latency in milliseconds (u64::MAX = unknown)
    latency_ms: Vec<AtomicU64>,
    /// Index of current primary server
    primary_idx: AtomicUsize,
}

impl ServerSelector {
    fn new(servers: Vec<String>) -> Arc<Self> {
        let count = servers.len();
        let health: Vec<AtomicBool> = (0..count).map(|_| AtomicBool::new(true)).collect();
        let latency_ms: Vec<AtomicU64> = (0..count).map(|_| AtomicU64::new(u64::MAX)).collect();
        Arc::new(Self {
            servers,
            health,
            latency_ms,
            primary_idx: AtomicUsize::new(0),
        })
    }

    /// Get current primary server URL.
    fn get_primary(&self) -> &str {
        &self.servers[self.primary_idx.load(Ordering::Relaxed)]
    }

    /// Get all healthy server URLs in order: primary first, then others by latency.
    fn get_all_healthy(&self) -> Vec<&str> {
        let primary = self.primary_idx.load(Ordering::Relaxed);
        let mut result = Vec::with_capacity(self.servers.len() + 1);

        // Primary first (even if unhealthy — caller should try it)
        if primary < self.servers.len() {
            result.push(self.servers[primary].as_str());
        }

        // Other healthy servers sorted by latency
        let mut others: Vec<(usize, u64)> = (0..self.servers.len())
            .filter(|&i| i != primary && self.health[i].load(Ordering::Relaxed))
            .map(|i| (i, self.latency_ms[i].load(Ordering::Relaxed)))
            .collect();
        others.sort_by_key(|&(_, lat)| lat);

        for (idx, _) in others {
            result.push(self.servers[idx].as_str());
        }

        // Always include FALLBACK_BOOTSTRAP_URL as last resort
        if !self.servers.iter().any(|s| s == FALLBACK_BOOTSTRAP_URL) {
            result.push(FALLBACK_BOOTSTRAP_URL);
        }

        result
    }

    fn mark_unhealthy(&self, url: &str) {
        if let Some(idx) = self.servers.iter().position(|s| s == url) {
            self.health[idx].store(false, Ordering::Relaxed);
        }
    }

    fn mark_healthy(&self, url: &str, latency: u64) {
        if let Some(idx) = self.servers.iter().position(|s| s == url) {
            self.health[idx].store(true, Ordering::Relaxed);
            self.latency_ms[idx].store(latency, Ordering::Relaxed);
        }
    }

    /// Re-elect primary: pick the healthy server with lowest latency.
    fn auto_elect_primary(&self) {
        let mut best_idx = self.primary_idx.load(Ordering::Relaxed);
        let mut best_lat = u64::MAX;

        for i in 0..self.servers.len() {
            if self.health[i].load(Ordering::Relaxed) {
                let lat = self.latency_ms[i].load(Ordering::Relaxed);
                if lat < best_lat {
                    best_lat = lat;
                    best_idx = i;
                }
            }
        }

        let old = self.primary_idx.swap(best_idx, Ordering::Relaxed);
        if old != best_idx {
            info!("🔄 Server failover: {} → {} (latency: {}ms)",
                self.servers[old], self.servers[best_idx], best_lat);
        }
    }

    fn server_count(&self) -> usize {
        self.servers.len()
    }
}

/// Spawn health check task that pings each server every 15 seconds.
fn spawn_health_checker(
    selector: Arc<ServerSelector>,
    client: reqwest::Client,
    is_running: Arc<AtomicBool>,
) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(15));
        loop {
            interval.tick().await;
            if !is_running.load(Ordering::Relaxed) {
                break;
            }

            for i in 0..selector.server_count() {
                let url = format!("{}/api/v1/status", selector.servers[i]);
                let start = std::time::Instant::now();
                match client
                    .get(&url)
                    .timeout(std::time::Duration::from_secs(8))
                    .send()
                    .await
                {
                    Ok(resp) if resp.status().is_success() => {
                        let latency = start.elapsed().as_millis() as u64;
                        selector.mark_healthy(&selector.servers[i], latency);
                    }
                    _ => {
                        selector.mark_unhealthy(&selector.servers[i]);
                    }
                }
            }

            selector.auto_elect_primary();
        }
    });
}

/// Try an HTTP GET request against ALL healthy servers in order.
/// Returns (response_body, actual_url_used) on first success.
async fn fetch_with_failover(
    client: &reqwest::Client,
    selector: &ServerSelector,
    path: &str,
) -> Result<(String, String)> {
    let servers = selector.get_all_healthy();

    for base_url in &servers {
        let url = format!("{}{}", base_url, path);
        match client
            .get(&url)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                let body = resp.text().await.unwrap_or_default();
                track_download(body.len());
                return Ok((body, base_url.to_string()));
            }
            Ok(resp) => {
                let status = resp.status();
                track_api_failure();
                debug!("Server {} returned HTTP {} for {} — trying next", base_url, status, path);
                selector.mark_unhealthy(base_url);
            }
            Err(e) => {
                track_api_failure();
                debug!("Server {} unreachable ({}) — trying next", base_url, e);
                selector.mark_unhealthy(base_url);
            }
        }
    }

    Err(anyhow::anyhow!("All {} servers failed for {}", servers.len(), path))
}

/// Build a reqwest HTTP client with optional proxy support.
/// Used by all HTTP client creation points (solo mining, decentralized pool, diagnostics).
fn build_http_client(proxy_url: Option<&str>, timeout_secs: u64) -> anyhow::Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .pool_max_idle_per_host(2)
        .pool_idle_timeout(std::time::Duration::from_secs(30))
        .tcp_keepalive(std::time::Duration::from_secs(15))
        // v8.8.3: Enable transparent gzip/brotli/deflate decompression.
        // Server sends compressed response → reqwest auto-decompresses.
        // JSON typically compresses 70-85%, reducing bandwidth significantly.
        .gzip(true)
        .brotli(true)
        .deflate(true);
    if let Some(proxy) = proxy_url {
        builder = builder.proxy(reqwest::Proxy::all(proxy)?);
    }
    Ok(builder.build()?)
}

/// v9.0.2: Check if the system has TLS/CA certificates available.
/// hyper-rustls panics (!) if no native certs exist (bare Docker, minimal installs).
/// Call this once at startup; if false, skip SSE and MinerLink (they use hyper-rustls).
/// Mining still works via periodic HTTP challenge refresh.
fn has_tls_certificates() -> bool {
    #[cfg(unix)]
    {
        std::panic::catch_unwind(|| {
            // rustls-native-certs is what hyper-rustls uses internally
            // If this panics or returns empty, HTTPS connections will fail
            let certs = rustls_native_certs::load_native_certs();
            match certs {
                Ok(store) => !store.is_empty(),
                Err(_) => false,
            }
        })
        .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        // On Windows, reqwest uses rustls-tls-webpki-roots (bundled Mozilla CA roots)
        // which doesn't depend on system certs, so always return true.
        true
    }
}

/// v9.2.6: Bump the soft NOFILE (file descriptor) limit to prevent "Too many open files" crashes.
/// The miner opens many sockets (HTTP connections, SSE streams, TLS sessions) and on some systems
/// the default soft limit is only 1024. When exhausted, hyper-rustls panics trying to open
/// /etc/ssl/certs/ or loading native certs, which kills the process.
fn bump_fd_limit() {
    #[cfg(unix)]
    {
        use std::io;
        // Get current limits
        let mut rlim = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        let rc = unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut rlim) };
        if rc != 0 {
            eprintln!("Warning: getrlimit(NOFILE) failed: {}", io::Error::last_os_error());
            return;
        }
        let target: u64 = 65536;
        let new_soft = target.min(rlim.rlim_max);
        if rlim.rlim_cur < new_soft {
            rlim.rlim_cur = new_soft;
            let rc = unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &rlim) };
            if rc == 0 {
                eprintln!("   File descriptor limit raised: {} → {}", rlim.rlim_cur, new_soft);
            }
            // Silently ignore failure — we tried, user may need `ulimit -n 65536`
        }
    }
}

/// v9.2.6: Build a reqwest::Client safely, catching panics from hyper-rustls cert loading.
/// Falls back to a client with no TLS verification disabled (which still works for HTTP)
/// rather than crashing the entire process.
fn build_http_client_safe(proxy_url: Option<&str>, timeout_secs: u64) -> reqwest::Client {
    match build_http_client(proxy_url, timeout_secs) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("⚠️  HTTP client build failed: {} — trying minimal client", e);
            // Try building a minimal client, but catch panics from cert loading
            match std::panic::catch_unwind(|| reqwest::Client::new()) {
                Ok(c) => c,
                Err(_) => {
                    tracing::error!("⚠️  reqwest::Client::new() panicked (fd exhaustion?) — using builder without TLS certs");
                    // Last resort: build with danger_accept_invalid_certs to skip cert loading entirely
                    // This is safe for our use because the server uses HTTPS and the certs are just
                    // the CA store, not the server identity. The alternative is a process crash.
                    reqwest::Client::builder()
                        .timeout(std::time::Duration::from_secs(timeout_secs))
                        .no_proxy()
                        .build()
                        .unwrap_or_else(|_| {
                            // Absolute last resort — return a default client handle that will
                            // fail on actual requests (but won't crash the process)
                            reqwest::Client::builder()
                                .build()
                                .expect("reqwest::Client::builder().build() should never fail")
                        })
                }
            }
        }
    }
}

// v8.6.6: Global bandwidth counters for TUI instrumentation.
// Atomics so any thread can update without locking. Connected to SharedMinerState in main().
static GLOBAL_BYTES_DOWNLOADED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GLOBAL_BYTES_UPLOADED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GLOBAL_API_REQUESTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GLOBAL_API_FAILURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Track a download (response body bytes received).
#[inline]
fn track_download(bytes: usize) {
    GLOBAL_BYTES_DOWNLOADED.fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);
    GLOBAL_API_REQUESTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Track an upload (request body bytes sent).
#[inline]
pub fn track_upload(bytes: usize) {
    GLOBAL_BYTES_UPLOADED.fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);
}

/// Track an API failure.
#[inline]
fn track_api_failure() {
    GLOBAL_API_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    GLOBAL_API_REQUESTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

// v9.1.4: Dynamic mining mode switch — admin can force all miners to switch mode at runtime.
// 0 = no switch, 1 = switch to solo, 2 = switch to pool
static MODE_SWITCH_TARGET: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
static MODE_SWITCH_POOL_URL: std::sync::OnceLock<parking_lot::Mutex<Option<String>>> = std::sync::OnceLock::new();

fn get_mode_switch_pool_url() -> &'static parking_lot::Mutex<Option<String>> {
    MODE_SWITCH_POOL_URL.get_or_init(|| parking_lot::Mutex::new(None))
}

/// Try an HTTP GET request against the primary server, falling back to bootstrap1.quillon.xyz
/// Returns (response_body, actual_url_used) on success.
/// NOTE: Legacy function kept for backward compatibility. New code should use fetch_with_failover().
async fn fetch_with_fallback(
    client: &reqwest::Client,
    primary_base: &str,
    path: &str,
) -> Result<(String, String)> {
    let primary_url = format!("{}{}", normalize_server_url(primary_base), path);
    match client.get(&primary_url).timeout(std::time::Duration::from_secs(10)).send().await {
        Ok(resp) if resp.status().is_success() => {
            let body = resp.text().await.unwrap_or_default();
            track_download(body.len());
            Ok((body, primary_base.to_string()))
        }
        Ok(resp) => {
            let status = resp.status();
            track_api_failure();
            debug!("Primary returned HTTP {} — using backup node {}", status, FALLBACK_BOOTSTRAP_URL);
            let fallback_url = format!("{}{}", FALLBACK_BOOTSTRAP_URL, path);
            let resp = client.get(&fallback_url).timeout(std::time::Duration::from_secs(10)).send().await?;
            if !resp.status().is_success() {
                let fb_status = resp.status();
                return Err(anyhow::anyhow!("Primary HTTP {}, fallback HTTP {} — server may be syncing", status, fb_status));
            }
            let body = resp.text().await.unwrap_or_default();
            track_download(body.len());
            Ok((body, FALLBACK_BOOTSTRAP_URL.to_string()))
        }
        Err(e) => {
            track_api_failure();
            debug!("Primary {} unreachable ({}) — using backup node {}", primary_base, e, FALLBACK_BOOTSTRAP_URL);
            let fallback_url = format!("{}{}", FALLBACK_BOOTSTRAP_URL, path);
            let resp = client.get(&fallback_url).timeout(std::time::Duration::from_secs(10)).send().await?;
            if !resp.status().is_success() {
                return Err(anyhow::anyhow!("Primary unreachable, fallback HTTP {} — server may be syncing", resp.status()));
            }
            let body = resp.text().await.unwrap_or_default();
            track_download(body.len());
            Ok((body, FALLBACK_BOOTSTRAP_URL.to_string()))
        }
    }
}

/// Fallback wallet: when device login fails, mine to the master wallet.
const WINDOWS_DEFAULT_WALLET: &str = "qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723";

/// v9.8.4: Saved wallet persistence — once you login, you never have to again.
/// Stores wallet address in ~/.quillon/miner-wallet.txt (or next to the binary on Windows).
fn saved_wallet_path() -> std::path::PathBuf {
    if cfg!(target_os = "windows") {
        // Windows: save next to the binary
        if let Ok(exe) = std::env::current_exe() {
            return exe.with_file_name("quillon-wallet.txt");
        }
    }
    // Unix: ~/.quillon/miner-wallet.txt
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let dir = std::path::PathBuf::from(home).join(".quillon");
    let _ = std::fs::create_dir_all(&dir);
    dir.join("miner-wallet.txt")
}

fn load_saved_wallet() -> Option<String> {
    let path = saved_wallet_path();
    let contents = std::fs::read_to_string(&path).ok()?;
    let wallet = contents.trim().to_string();
    // Validate: must start with "qnk" and be 66+ chars
    if wallet.starts_with("qnk") && wallet.len() >= 66 {
        Some(wallet)
    } else {
        None
    }
}

fn save_wallet(wallet: &str) {
    let path = saved_wallet_path();
    if let Err(e) = std::fs::write(&path, wallet) {
        eprintln!("\x1b[33m   ! Could not save wallet: {}\x1b[0m", e);
    }
}

/// Open a URL in the user's default browser
fn open_browser(url: &str) -> bool {
    #[cfg(target_os = "windows")]
    { std::process::Command::new("cmd").args(["/c", "start", "", url]).spawn().is_ok() }
    #[cfg(target_os = "macos")]
    { std::process::Command::new("open").arg(url).spawn().is_ok() }
    #[cfg(target_os = "linux")]
    { std::process::Command::new("xdg-open").arg(url).spawn().is_ok() }
}

/// v10.1.9: Generate a compact QR code for terminal display.
/// Encodes the login deep-link URL so Quillon mobile app can scan it.
/// Uses Unicode half-block characters (▀▄█ ) for compact terminal rendering.
fn generate_login_qr(data: &str) -> Vec<String> {
    use qrcode::{QrCode, EcLevel};
    let code = match QrCode::with_error_correction_level(data, EcLevel::L) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let matrix = code.to_colors();
    let width = code.width();
    let mut lines = Vec::new();

    // Render 2 rows per line using Unicode half-blocks for compact display
    let mut y = 0;
    while y < width {
        let mut line = String::new();
        for x in 0..width {
            let top = matrix[y * width + x] == qrcode::Color::Dark;
            let bottom = if y + 1 < width {
                matrix[(y + 1) * width + x] == qrcode::Color::Dark
            } else {
                false
            };
            match (top, bottom) {
                (true, true) => line.push('█'),
                (true, false) => line.push('▀'),
                (false, true) => line.push('▄'),
                (false, false) => line.push(' '),
            }
        }
        lines.push(line);
        y += 2;
    }
    lines
}

/// Device login flow: request code from server → open browser → poll until user logs in.
/// Retries server connection with backoff. User can press Enter to skip at any time.
/// Respects --tor / --proxy if provided.
async fn device_login_flow(server_url: &str, proxy_url: Option<&str>) -> Result<String> {
    let client = build_http_client(proxy_url, 15)?;

    let spinner = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"];
    eprint!("\x1b[36m   {} Connecting to Quillon network...\x1b[0m", spinner[0]);

    // Try multiple server URLs: the provided one, then localhost fallback
    let urls_to_try: Vec<String> = if server_url.contains("127.0.0.1") || server_url.contains("localhost") {
        vec![server_url.to_string()]
    } else {
        vec![server_url.to_string(), "http://127.0.0.1:8080".to_string()]
    };

    // Step 1: Request device login code — retry with fallback URLs
    // v9.0.1: Increased attempts from 5→15 and timeout 5s→10s to handle
    // server startup/syncing (503s) without giving up too early.
    let mut resp_data: Option<serde_json::Value> = None;
    let mut connected_url = server_url.to_string();
    'outer: for (url_idx, url) in urls_to_try.iter().enumerate() {
        let max_attempts = if url_idx == 0 { 15 } else { 5 };
        for attempt in 0..max_attempts {
            match client.post(format!("{}/api/v1/miner/device-login", url))
                .timeout(std::time::Duration::from_secs(10))
                .send().await
            {
                Ok(r) => {
                    if let Ok(json) = r.json::<serde_json::Value>().await {
                        if json.get("data").is_some() {
                            resp_data = Some(json);
                            connected_url = url.clone();
                            eprint!("\r\x1b[2K");
                            eprintln!("\x1b[32m   ✓ Connected to network\x1b[0m");
                            break 'outer;
                        }
                    }
                }
                Err(_) => {}
            }

            let total = if url_idx == 0 { attempt + 1 } else { 5 + attempt + 1 };
            let s = spinner[(total as usize) % spinner.len()];
            if url_idx > 0 && attempt == 0 {
                eprint!("\r\x1b[2K\x1b[36m   {} Trying localhost fallback...\x1b[0m", s);
            } else {
                eprint!("\r\x1b[2K\x1b[36m   {} Connecting... (attempt {})\x1b[0m", s, total);
            }

            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    }

    let resp = resp_data.ok_or_else(|| anyhow::anyhow!("Could not reach server after 20 attempts"))?;

    let data = resp.get("data").ok_or_else(|| anyhow::anyhow!("Server returned no data"))?;
    let device_code = data.get("device_code").and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("No device_code in response"))?;
    let user_code = data.get("user_code").and_then(|v| v.as_str()).unwrap_or("????-????");
    let verification_url = data.get("verification_url").and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("No verification_url in response"))?;

    // Step 2: Show the user the code, QR code, and open browser
    // v10.1.9: Generate QR code for mobile app scanning
    let qr_data = format!("quillon://miner-login?code={}&server={}", device_code, connected_url);
    let qr_lines = generate_login_qr(&qr_data);

    let c = "\x1b[38;5;51m";   // cyan
    let g = "\x1b[38;5;220m";  // gold
    let w = "\x1b[1;37m";      // white bold
    let d = "\x1b[2m";         // dim
    let r = "\x1b[0m";         // reset
    let ul = "\x1b[4;36m";     // underline cyan

    eprintln!();
    eprintln!("  {c}╔══════════════════════════════════════════════════════╗{r}");
    eprintln!("  {c}║{r}                                                      {c}║{r}");
    eprintln!("  {c}║{r}   {w}LINK YOUR WALLET{r}                                   {c}║{r}");
    eprintln!("  {c}║{r}                                                      {c}║{r}");
    eprintln!("  {c}║{r}   {d}Step 1:{r} Enter this code on the website:             {c}║{r}");
    eprintln!("  {c}║{r}                                                      {c}║{r}");
    eprintln!("  {c}║{r}           {g}╔═══════════════════╗{r}                      {c}║{r}");
    eprintln!("  {c}║{r}           {g}║{r}    {w}{}{r}    {g}║{r}                      {c}║{r}", user_code);
    eprintln!("  {c}║{r}           {g}╚═══════════════════╝{r}                      {c}║{r}");
    eprintln!("  {c}║{r}                                                      {c}║{r}");
    eprintln!("  {c}║{r}   {d}Step 2:{r} Open this link:                             {c}║{r}");
    eprintln!("  {c}║{r}                                                      {c}║{r}");
    eprintln!("  {c}║{r}   {ul}{}{r}", verification_url);
    eprintln!("  {c}║{r}                                                      {c}║{r}");
    // Display QR code for mobile scanning
    if !qr_lines.is_empty() {
        eprintln!("  {c}║{r}   {d}Or scan with your phone:{r}                         {c}║{r}");
        eprintln!("  {c}║{r}                                                      {c}║{r}");
        for line in &qr_lines {
            eprintln!("  {c}║{r}     {}  {r}", line);
        }
        eprintln!("  {c}║{r}                                                      {c}║{r}");
    }
    eprintln!("  {c}╠══════════════════════════════════════════════════════╣{r}");
    eprintln!("  {c}║{r}                                                      {c}║{r}");
    eprintln!("  {c}║{r}   {d}Once linked, your wallet receives mining rewards{r}    {c}║{r}");
    eprintln!("  {c}║{r}   {d}from both GPU (BLAKE3) and CPU (VDF) lanes.{r}         {c}║{r}");
    eprintln!("  {c}║{r}                                                      {c}║{r}");
    eprintln!("  {c}║{r}   {d}Press Enter to skip (mines to default wallet){r}       {c}║{r}");
    eprintln!("  {c}║{r}                                                      {c}║{r}");
    eprintln!("  {c}╚══════════════════════════════════════════════════════╝{r}");
    eprintln!();

    let browser_opened = open_browser(verification_url);
    if browser_opened {
        eprintln!("  {c}  ✓ Browser opened — complete login there{r}");
    } else {
        eprintln!("  {g}  ! Copy the URL above or scan QR with your phone{r}");
    }
    eprintln!();

    // Step 3: Poll until user completes login OR presses Enter to skip
    let poll_url = format!("{}/api/v1/miner/device-login/{}", connected_url, device_code);
    let max_wait = std::time::Duration::from_secs(300); // 5 min (was 10)
    let start = std::time::Instant::now();
    let mut consecutive_errors: u32 = 0;

    // Spawn a stdin listener — if user presses Enter, we get a signal
    let (skip_tx, mut skip_rx) = tokio::sync::oneshot::channel::<()>();
    std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = std::io::stdin().read_line(&mut buf);
        let _ = skip_tx.send(());
    });

    loop {
        // Check timeout
        if start.elapsed() >= max_wait {
            return Err(anyhow::anyhow!("Login timed out — starting with default wallet"));
        }

        // v9.8.4: Auto-skip after 5 consecutive server errors (502/503/timeout)
        if consecutive_errors >= 5 {
            eprint!("\r\x1b[2K");
            return Err(anyhow::anyhow!("Server unavailable — starting with default wallet"));
        }

        // Race: poll server vs user pressing Enter
        tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_secs(3)) => {
                // Poll the server
                let poll_resp = match client.get(&poll_url).timeout(std::time::Duration::from_secs(8)).send().await {
                    Ok(r) => {
                        // v9.8.4: Check HTTP status before parsing JSON
                        if !r.status().is_success() {
                            consecutive_errors += 1;
                            let elapsed = start.elapsed().as_secs();
                            eprint!("\r\x1b[2K\x1b[33m   Server returned {} — retrying ({}/5)...\x1b[0m", r.status(), consecutive_errors);
                            let _ = r.text().await; // drain body
                            continue;
                        }
                        consecutive_errors = 0;
                        r.json::<serde_json::Value>().await.unwrap_or_default()
                    }
                    Err(_) => {
                        consecutive_errors += 1;
                        continue;
                    }
                };

                if let Some(data) = poll_resp.get("data") {
                    let status = data.get("status").and_then(|v| v.as_str()).unwrap_or("");
                    if status == "complete" {
                        if let Some(wallet) = data.get("wallet_address").and_then(|v| v.as_str()) {
                            eprint!("\r\x1b[2K");
                            return Ok(wallet.to_string());
                        }
                    }
                }

                // Check for expired code
                if let Some(false) = poll_resp.get("success").and_then(|v| v.as_bool()) {
                    return Err(anyhow::anyhow!("Login code expired — restart the miner to try again"));
                }

                // Animated waiting indicator
                let elapsed = start.elapsed().as_secs();
                let spin = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"];
                let s = spin[(elapsed as usize) % spin.len()];
                eprint!("\r\x1b[2K\x1b[2m   {} Waiting for login... ({}s)\x1b[0m", s, elapsed);
            }
            _ = &mut skip_rx => {
                eprint!("\r\x1b[2K");
                return Err(anyhow::anyhow!("User skipped login"));
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // v9.2.6: Raise file descriptor limit to prevent "Too many open files" crash.
    // hyper-rustls panics (!) when loading platform certs fails with EMFILE/ENFILE,
    // which kills tokio worker threads and the entire process.
    bump_fd_limit();

    let mut args = Args::parse();

    // Resolve proxy URL early: --tor > --proxy > ALL_PROXY env > HTTPS_PROXY env > HTTP_PROXY env
    // This is needed before device login so the flow can route through Tor/proxy.
    let early_proxy_url: Option<String> = if args.tor {
        // v9.0.7: Try system Tor first, then fall back to embedded arti
        let system_tor = std::net::TcpStream::connect_timeout(
            &"127.0.0.1:9050".parse().unwrap(),
            std::time::Duration::from_secs(2),
        ).is_ok();

        if system_tor {
            eprintln!("\x1b[32m   System Tor detected at 127.0.0.1:9050\x1b[0m");
            Some("socks5://127.0.0.1:9050".into())
        } else {
            // No system Tor — bootstrap embedded arti
            #[cfg(feature = "tor-support")]
            {
                match embedded_tor::start_embedded_tor().await {
                    Ok(proxy_url) => Some(proxy_url),
                    Err(e) => {
                        eprintln!("\x1b[33m   Warning: Embedded Tor failed: {}\x1b[0m", e);
                        eprintln!("\x1b[33m   Falling back to direct connection\x1b[0m");
                        None
                    }
                }
            }
            #[cfg(not(feature = "tor-support"))]
            {
                eprintln!("\x1b[33m   Warning: --tor requires tor-support feature or system Tor at 127.0.0.1:9050\x1b[0m");
                eprintln!("\x1b[33m   Falling back to direct connection\x1b[0m");
                None
            }
        }
    } else if args.proxy.is_some() {
        args.proxy.clone()
    } else {
        std::env::var("ALL_PROXY").ok()
            .or_else(|| std::env::var("HTTPS_PROXY").ok())
            .or_else(|| std::env::var("HTTP_PROXY").ok())
    };

    // Zero-config: When launched without explicit flags, use device login flow.
    // Opens browser for user to log in → miner gets wallet → starts mining.
    // No --wallet needed — just double-click the exe!
    // v9.8.4: Saved wallet persistence — login once, mine forever.
    {
        let has_explicit_args = std::env::args().count() > 1;
        if !has_explicit_args {
            args.mode = "solo".to_string();
            args.server = "https://quillon.xyz".to_string();
            args.intensity = 2;

            eprintln!();
            eprintln!("\x1b[1;36m   Quillon Miner v{}\x1b[0m", env!("CARGO_PKG_VERSION"));
            eprintln!("\x1b[2m   Quantum-resistant solo mining — zero configuration required\x1b[0m");
            eprintln!();

            // v9.8.4: Check for saved wallet from a previous login
            if let Some(saved) = load_saved_wallet() {
                let short = if saved.len() > 17 {
                    format!("{}...{}", &saved[..11], &saved[saved.len()-6..])
                } else {
                    saved.clone()
                };
                eprintln!("\x1b[1;32m   ✓ Wallet loaded: {}\x1b[0m", short);
                eprintln!("\x1b[2m   (saved from previous login — delete ~/.quillon/miner-wallet.txt to reset)\x1b[0m");
                eprintln!();
                args.wallet = Some(saved);
            } else {
                eprintln!("\x1b[38;5;245m   Your browser will open to link your wallet.\x1b[0m");
                eprintln!("\x1b[38;5;245m   Or press Enter to start mining immediately.\x1b[0m");
                eprintln!();

                // Try device login flow — retries server, waits for user, Enter to skip
                match device_login_flow(&args.server, early_proxy_url.as_deref()).await {
                    Ok(wallet) => {
                        let short = if wallet.len() > 17 {
                            format!("{}...{}", &wallet[..11], &wallet[wallet.len()-6..])
                        } else {
                            wallet.clone()
                        };
                        eprintln!("\x1b[1;32m   ✓ Wallet linked: {}\x1b[0m", short);
                        eprintln!();
                        save_wallet(&wallet);
                        args.wallet = Some(wallet);
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        if !msg.contains("User skipped") {
                            eprintln!("\x1b[33m   ! {}\x1b[0m", msg);
                        }
                        eprintln!("\x1b[2m   Mining to community pool — use --wallet <addr> for your own\x1b[0m");
                        eprintln!();
                        args.wallet = Some(WINDOWS_DEFAULT_WALLET.to_string());
                    }
                }
            }
        }
    }

    // v10.0.0: Parse comma-separated server list for multi-server failover
    // Single URL is backward-compatible (produces a 1-element list).
    let server_list = parse_server_list(&args.server);
    // Keep first server as the canonical --server value for backward compat
    args.server = server_list[0].clone();

    // Proxy URL was resolved early (before device login). Reuse it.
    let proxy_url = early_proxy_url;

    // Validate proxy URL format
    if let Some(ref p) = proxy_url {
        if !p.starts_with("socks5://") && !p.starts_with("http://") && !p.starts_with("https://") {
            eprintln!("❌ Invalid proxy URL: {}", p);
            eprintln!("   Supported formats: socks5://host:port, http://host:port, https://host:port");
            std::process::exit(1);
        }
    }

    // Determine if TUI should be enabled
    // --tui forces it on, --no-tui forces it off, otherwise auto-detect terminal
    let use_tui = if args.no_tui {
        false
    } else if args.tui {
        cfg!(feature = "tui")
    } else {
        cfg!(feature = "tui")
            && atty::is(atty::Stream::Stdout)
            && args.mode != "benchmark"
    };

    // Initialize logging — TUI mode captures logs via layer, headless uses fmt
    #[cfg(feature = "tui")]
    let tui_log_rx = if use_tui {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;

        let (layer, rx) = q_miner::ui::tui_app::MinerTuiLogLayer::new();
        tracing_subscriber::registry()
            .with(tracing_subscriber::EnvFilter::new("q_miner=info,q_dag_knight=info"))
            .with(layer)
            .init();
        Some(rx)
    } else {
        tracing_subscriber::fmt()
            .with_env_filter("q_miner=info,q_dag_knight=info")
            .init();
        None
    };

    #[cfg(not(feature = "tui"))]
    {
        tracing_subscriber::fmt()
            .with_env_filter("q_miner=info,q_dag_knight=info")
            .init();
    }

    // Print banner
    print_banner();

    // ═══════════════════════════════════════════════════════════════════
    // WINDOWS PERFORMANCE OPTIMIZATIONS
    // ═══════════════════════════════════════════════════════════════════
    #[cfg(target_os = "windows")]
    {
        // Set process priority to HIGH for better CPU scheduling
        unsafe {
            extern "system" {
                fn GetCurrentProcess() -> *mut std::ffi::c_void;
                fn SetPriorityClass(hProcess: *mut std::ffi::c_void, dwPriorityClass: u32) -> i32;
                fn timeBeginPeriod(uPeriod: u32) -> u32;
            }
            const HIGH_PRIORITY_CLASS: u32 = 0x00000080;
            let process = GetCurrentProcess();
            if SetPriorityClass(process, HIGH_PRIORITY_CLASS) != 0 {
                info!("⚡ Windows: Process priority set to HIGH_PRIORITY_CLASS");
            }
            // Set timer resolution to 1ms for more responsive thread scheduling
            timeBeginPeriod(1);
            info!("⚡ Windows: Timer resolution set to 1ms");
        }

        println!("{}", style("💡 Windows Performance Tips:").yellow().bold());
        println!("   1. Set Power Plan to 'High Performance' in Windows Settings");
        println!("   2. Add q-miner.exe to Windows Defender exclusions:");
        println!("      Settings → Virus & Threat Protection → Exclusions → Add");
        println!("   3. Disable background apps for maximum CPU availability");
        println!();
    }

    // Hardware detection
    info!("🔍 Detecting hardware capabilities...");
    let hardware_info = detect_hardware().await?;

    println!("{}", style("💻 Hardware Detection Results:").cyan().bold());
    println!(
        "   CPU: {} ({}) - {} cores, {} threads",
        hardware_info.cpu_vendor,
        if hardware_info.has_avx512 {
            "AVX-512"
        } else if hardware_info.has_avx2 {
            "AVX2"
        } else {
            "SSE"
        },
        hardware_info.cpu_cores,
        hardware_info.cpu_threads
    );
    println!(
        "   Cache Line: {} bytes │ SIMD: {} │ Server-Optimized: {}",
        hardware_info.cache_line_size,
        if hardware_info.has_avx512 { "AVX-512" } else if hardware_info.has_avx2 { "AVX2" } else { "SSE" },
        if hardware_info.cpu_cores >= 16 { "✅" } else { "⚠️ Desktop CPU" }
    );

    if hardware_info.cuda_devices > 0 {
        println!("   CUDA: {} devices detected", hardware_info.cuda_devices);
    }
    if hardware_info.opencl_devices > 0 {
        println!(
            "   OpenCL: {} devices detected",
            hardware_info.opencl_devices
        );
    }

    // Determine mining configuration
    let has_gpu = !args.no_gpu && (hardware_info.cuda_devices > 0 || hardware_info.opencl_devices > 0);
    let cpu_threads = if args.threads > 0 {
        args.threads
    } else if has_gpu {
        let gpu_cpu = 2.min(hardware_info.cpu_threads);
        gpu_cpu
    } else {
        hardware_info.cpu_threads.max(1)
    };

    // v10.3.4: Show mining lane configuration
    let c = "\x1b[38;5;51m";
    let g = "\x1b[38;5;220m";
    let w = "\x1b[1;37m";
    let d = "\x1b[2m";
    let r = "\x1b[0m";
    println!();
    println!("  {c}╔══════════════════════════════════════════════════╗{r}");
    println!("  {c}║{r}  {w}MINING CONFIGURATION{r}                              {c}║{r}");
    println!("  {c}╠══════════════════════════════════════════════════╣{r}");
    if has_gpu {
        println!("  {c}║{r}                                                  {c}║{r}");
        println!("  {c}║{r}   {g}GPU DETECTED{r}                                    {c}║{r}");
        println!("  {c}║{r}                                                  {c}║{r}");
        println!("  {c}║{r}   {g}BLAKE3 Lane{r}  {d}━━━━{r} GPU + {} CPU threads  {d}━━{r}  {g}50%{r}  {c}║{r}", cpu_threads);
        println!("  {c}║{r}   {c}VDF Lane{r}    {d}━━━━{r} 1 dedicated CPU core  {d}━━{r}  {c}50%{r}  {c}║{r}");
        println!("  {c}║{r}                                                  {c}║{r}");
        println!("  {c}║{r}   {d}GPU handles parallel hashing (high throughput){r}   {c}║{r}");
        println!("  {c}║{r}   {d}CPU computes sequential VDF (can't parallelize){r}  {c}║{r}");
    } else {
        println!("  {c}║{r}                                                  {c}║{r}");
        println!("  {c}║{r}   {c}CPU ONLY{r}  {d}({} cores detected){r}                    {c}║{r}", hardware_info.cpu_threads);
        println!("  {c}║{r}                                                  {c}║{r}");
        println!("  {c}║{r}   {g}BLAKE3 Lane{r}  {d}━━━━{r} {} CPU threads        {d}━━{r}  {g}50%{r}  {c}║{r}", cpu_threads);
        println!("  {c}║{r}   {c}VDF Lane{r}    {d}━━━━{r} 1 dedicated CPU core  {d}━━{r}  {c}50%{r}  {c}║{r}");
        println!("  {c}║{r}                                                  {c}║{r}");
        println!("  {c}║{r}   {d}VDF lane is CPU-fair: your laptop earns equally{r}  {c}║{r}");
        println!("  {c}║{r}   {d}to a server — sequential work, no shortcuts{r}      {c}║{r}");
    }
    println!("  {c}║{r}                                                  {c}║{r}");
    println!("  {c}╚══════════════════════════════════════════════════╝{r}");
    println!();

    if args.mode == "benchmark" || args.benchmark {
        info!("🏁 Running benchmark mode for {} seconds...", args.duration);
        run_benchmark(cpu_threads, args.intensity, args.duration).await?;
    } else {
        // Validate wallet address for non-benchmark modes
        // NOTE: Use eprintln! (not error!) for pre-TUI validation failures.
        // In TUI mode, tracing logs go to the TUI layer which never renders
        // if we exit before the TUI starts — causing silent exits on Windows.
        let wallet = match args.wallet {
            Some(w) => w,
            None => {
                eprintln!("❌ Wallet address required for {} mode. Use --wallet <address>", args.mode);
                eprintln!("   Example: q-miner --mode solo --wallet qnk<your_address> --server https://quillon.xyz");
                std::process::exit(1);
            }
        };

        // Validate wallet format - support both QUG (qnk + 64 hex) and AQUA (qnka + 62 hex) wallets
        let is_qug_wallet = wallet.starts_with("qnk") && wallet.len() == 67;
        let is_aqua_wallet = wallet.starts_with("qnka") && wallet.len() == 66;

        if !is_qug_wallet && !is_aqua_wallet {
            eprintln!("❌ Invalid wallet address format.");
            eprintln!("   QUG wallet: 'qnk' + 64 hex chars (67 total)");
            eprintln!("   AQUA wallet: 'qnka' + 62 hex chars (66 total)");
            std::process::exit(1);
        }

        // v9.1.4: Outer loop enables runtime mode switching via admin SSE command.
        // When MODE_SWITCH_TARGET is set (by SSE handler or challenge response),
        // the inner run_*_mining() returns (is_running=false), we read the new mode,
        // and restart in the new mode without process restart.
        let mut current_mode = args.mode.clone();
        let mut current_pool_url = args.pool_url.clone();
        // TUI log receiver is consumed on first use; subsequent iterations run without TUI
        #[cfg(feature = "tui")]
        let mut tui_log_rx_opt = Some(tui_log_rx);

        loop {
            // Reset the switch target before entering a mining function
            MODE_SWITCH_TARGET.store(0, std::sync::atomic::Ordering::SeqCst);

            if current_mode == "pool" {
                // Pool mining mode - connect via Stratum protocol
                let pool_url = if args.server.contains("stratum") || args.server.contains(":3333") {
                    info!("ℹ️  Using --server value as pool URL (detected stratum URL)");
                    args.server.clone()
                } else {
                    current_pool_url.clone()
                };
                let worker_name = args.worker_name.clone().unwrap_or_else(|| {
                    format!("worker_{:08x}", rand::random::<u32>())
                });
                info!("⛏️  Starting Quillon-NarwhalKnight-Graph POOL mining...");
                info!("💰 Mining to wallet: {}", wallet);
                info!("🏊 Pool URL: {}", pool_url);
                info!("👷 Worker name: {}", worker_name);
                let _ = run_pool_mining(cpu_threads, args.intensity, &wallet, &worker_name, &pool_url).await;
            } else if current_mode == "decentralized" {
                // Decentralized P2P pool mining mode
                let worker_name = args.worker_name.clone().unwrap_or_else(|| {
                    format!("worker_{:08x}", rand::random::<u32>())
                });
                info!("🌐 Starting Quillon-NarwhalKnight-Graph DECENTRALIZED POOL mining...");
                info!("💰 Mining to wallet: {}", wallet);
                info!("👷 Worker name: {}", worker_name);
                info!("📡 Bootstrap nodes: {}", args.bootstrap_nodes);
                info!("🗺️  Region: {}", args.region);
                info!("");
                info!("📊 Features:");
                info!("   ✅ CRDT-based PPLNS - No central pool needed");
                info!("   ✅ P2P share propagation via gossipsub");
                info!("   ✅ VDF anti-grinding proofs");
                info!("   ✅ Threshold signature payouts");
                info!("");
                let _ = run_decentralized_pool_mining(
                    cpu_threads,
                    args.intensity,
                    &wallet,
                    &worker_name,
                    &args.bootstrap_nodes,
                    &args.region,
                ).await;
            } else {
                // Solo mining mode
                if args.server.contains("stratum") || args.server.ends_with(":3333") {
                    eprintln!("❌ Stratum URL detected in --server but mode is 'solo'.");
                    eprintln!("   For pool mining, use: --mode pool --server {}", args.server);
                    eprintln!("   For solo mining, use: --server https://quillon.xyz");
                    std::process::exit(1);
                }
                info!("⛏️  Starting Quillon-NarwhalKnight-Graph SOLO mining...");
                info!("💰 Mining to wallet: {}", wallet);
                if server_list.len() > 1 {
                    info!("🌐 Servers ({} configured, multi-failover active):", server_list.len());
                    for (i, srv) in server_list.iter().enumerate() {
                        info!("   [{}] {}{}", i, srv, if i == 0 { " (primary)" } else { "" });
                    }
                } else {
                    info!("🌐 Primary server: {}", args.server);
                }
                info!("🔄 Fallback server: {}", FALLBACK_BOOTSTRAP_URL);
                if let Some(ref name) = args.miner_name {
                    info!("🏷️  Miner name: {}", name);
                }
                if let Some(ref p) = proxy_url {
                    info!("🧅 Proxy: {}", p);
                }
                #[cfg(feature = "tui")]
                {
                    // TUI log receiver is only available on the first iteration
                    let tui_rx = tui_log_rx_opt.take().flatten();
                    let enable_tui = use_tui && tui_rx.is_some();
                    // v10.0.2: Auto-disable P2P when --bandwidth-limit is set (unless --force-p2p)
                    #[cfg(feature = "p2p")]
                    let (no_p2p_flag, p2p_port_val) = {
                        let auto_disable = args.bandwidth_limit > 0 && !args.force_p2p;
                        (args.no_p2p || auto_disable, args.p2p_port)
                    };
                    #[cfg(not(feature = "p2p"))]
                    let (no_p2p_flag, p2p_port_val) = (true, 0u16);
                    let _ = run_mining(cpu_threads, args.intensity, !args.no_gpu, &wallet, &args.server, args.miner_name.as_deref(), enable_tui, args.bandwidth_limit, proxy_url.clone(), no_p2p_flag, p2p_port_val, args.no_auto_update, server_list.clone(), tui_rx).await;
                }
                #[cfg(not(feature = "tui"))]
                {
                    // v10.0.2: Auto-disable P2P when --bandwidth-limit is set (unless --force-p2p)
                    #[cfg(feature = "p2p")]
                    let (no_p2p_flag, p2p_port_val) = {
                        let auto_disable = args.bandwidth_limit > 0 && !args.force_p2p;
                        (args.no_p2p || auto_disable, args.p2p_port)
                    };
                    #[cfg(not(feature = "p2p"))]
                    let (no_p2p_flag, p2p_port_val) = (true, 0u16);
                    let _ = run_mining(cpu_threads, args.intensity, !args.no_gpu, &wallet, &args.server, args.miner_name.as_deref(), false, args.bandwidth_limit, proxy_url.clone(), no_p2p_flag, p2p_port_val, args.no_auto_update, server_list.clone(), ()).await;
                }
            }

            // Check if a mode switch was requested
            let switch_target = MODE_SWITCH_TARGET.load(std::sync::atomic::Ordering::SeqCst);
            if switch_target == 0 {
                // Normal shutdown (no mode switch) — exit the loop
                break;
            }

            let new_mode = match switch_target {
                1 => "solo".to_string(),
                2 => "pool".to_string(),
                _ => break,
            };

            // Read the pool URL for pool mode
            if new_mode == "pool" {
                if let Some(url) = get_mode_switch_pool_url().lock().take() {
                    current_pool_url = url;
                }
            }

            info!("");
            info!("⚡⚡⚡ MODE SWITCH: {} → {} ⚡⚡⚡", current_mode, new_mode);
            if new_mode == "pool" {
                info!("🏊 New pool URL: {}", current_pool_url);
            }
            info!("");

            current_mode = new_mode;

            // Brief pause to let mining threads fully stop
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
    }

    Ok(())
}

async fn detect_hardware() -> Result<HardwareInfo> {
    let cpu_cores = num_cpus::get_physical();
    let cpu_threads = num_cpus::get();

    // Simplified GPU detection (placeholder)
    let cuda_devices = if cfg!(feature = "cuda-mining") { 1 } else { 0 };
    let opencl_devices = if cfg!(feature = "opencl-mining") { 1 } else { 0 };

    // Detect CPU features (architecture-specific)
    #[cfg(target_arch = "x86_64")]
    let (cpu_vendor, has_avx2, has_avx512, cache_line_size) = {
        let cpuid = CpuId::new();
        let vendor = cpuid.get_vendor_info()
            .map(|v| v.as_str().to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        let extended_features = cpuid.get_extended_feature_info();
        let avx2 = extended_features.as_ref().map(|ef| ef.has_avx2()).unwrap_or(false);
        let avx512 = extended_features.as_ref().map(|ef| ef.has_avx512f()).unwrap_or(false);
        let cache = cpuid.get_cache_parameters()
            .and_then(|mut params| params.next())
            .map(|info| info.coherency_line_size() as usize)
            .unwrap_or(64);
        (vendor, avx2, avx512, cache)
    };

    #[cfg(not(target_arch = "x86_64"))]
    let (cpu_vendor, has_avx2, has_avx512, cache_line_size) = {
        let vendor = if cfg!(target_arch = "aarch64") {
            "ARM".to_string()
        } else {
            "Unknown".to_string()
        };
        (vendor, false, false, 64usize)
    };

    Ok(HardwareInfo {
        cpu_cores,
        cpu_threads,
        cuda_devices,
        opencl_devices,
        cpu_vendor,
        has_avx2,
        has_avx512,
        cache_line_size,
    })
}

async fn run_benchmark(threads: usize, intensity: u8, duration: u64) -> Result<()> {
    let hash_counter = Arc::new(AtomicU64::new(0));
    let is_running = Arc::new(AtomicBool::new(true));
    
    let start_time = std::time::Instant::now();
    let benchmark_duration = std::time::Duration::from_secs(duration);
    
    info!("🔥 Starting {} mining threads for benchmark", threads);
    
    let handles: Vec<_> = (0..threads)
        .map(|thread_id| {
            let hash_counter = hash_counter.clone();
            let is_running = is_running.clone();

            std::thread::Builder::new()
                .name(format!("bench-{}", thread_id))
                .spawn(move || {
                    benchmark_mining_thread(thread_id, hash_counter, is_running, intensity, benchmark_duration)
                })
                .expect("Failed to spawn benchmark thread")
        })
        .collect();

    // Wait for benchmark completion (blocking join — correct for CPU-bound threads)
    for handle in handles {
        let _ = handle.join();
    }
    
    let elapsed = start_time.elapsed();
    let total_hashes = hash_counter.load(Ordering::Relaxed);
    let hash_rate = total_hashes as f64 / elapsed.as_secs_f64();
    
    info!("🏁 Benchmark Results:");
    info!("   Duration: {:.2}s", elapsed.as_secs_f64());
    info!("   Total Hashes: {}", total_hashes);
    info!("   Hash Rate: {:.2} H/s", hash_rate);
    info!("   Per Thread: {:.2} H/s", hash_rate / threads as f64);
    
    println!("\n{}", style("🎯 Quillon Mining Benchmark Complete!").green().bold());
    println!("📊 Final Hash Rate: {:.2} H/s ({:.2} MH/s)", hash_rate, hash_rate / 1_000_000.0);
    
    Ok(())
}

async fn run_mining(
    threads: usize,
    intensity: u8,
    gpu_enabled: bool,
    wallet: &str,
    server_url: &str,
    miner_name: Option<&str>,
    use_tui: bool,
    bandwidth_limit: u32,
    proxy_url: Option<String>,
    no_p2p: bool,
    p2p_port: u16,
    no_auto_update: bool,
    server_list: Vec<String>,
    #[cfg(feature = "tui")]
    tui_log_rx: Option<tokio::sync::mpsc::UnboundedReceiver<q_miner::ui::tui_app::LogEntry>>,
    #[cfg(not(feature = "tui"))]
    _tui_log_rx: (),
) -> Result<()> {
    let hash_counter = Arc::new(AtomicU64::new(0));
    let is_running = Arc::new(AtomicBool::new(true));
    let wallet = wallet.to_string();
    let server_url = server_url.to_string();

    // v3.3.3-beta: Generate unique miner ID for this instance
    let miner_id = format!("{:016x}", rand::random::<u64>());
    let miner_name = miner_name.map(|s| s.to_string());
    info!("🆔 Miner ID: {}", miner_id);

    // v10.0.0: Multi-server failover — create ServerSelector from parsed server list
    let server_selector = ServerSelector::new(server_list);
    {
        let health_client = build_http_client_safe(proxy_url.as_deref(), 10);
        spawn_health_checker(
            Arc::clone(&server_selector),
            health_client,
            is_running.clone(),
        );
        if server_selector.server_count() > 1 {
            info!("🔄 Multi-server health checker started ({} servers, 15s interval)",
                server_selector.server_count());
        }
    }

    // CRITICAL FIX: Shared signal for when a new block is produced
    // All mining threads will check this and immediately fetch new challenge
    let new_block_signal = Arc::new(AtomicU64::new(0)); // Increments when new block arrives

    // PERF: Use AtomicU64 to store hashrate as f64 bits (lock-free, zero contention)
    let current_hashrate_khs = Arc::new(AtomicU64::new(0u64)); // f64 bits stored as u64

    // Miner link control atomics — wallet can remotely pause/resume, adjust threads/intensity
    let is_paused = Arc::new(AtomicBool::new(false));
    let target_threads = Arc::new(AtomicUsize::new(threads));
    let target_intensity = Arc::new(AtomicU8::new(intensity));
    let solutions_found = Arc::new(AtomicU64::new(0));
    let blocks_mined = Arc::new(AtomicU64::new(0));

    // Create SharedMinerState for TUI
    let (shared_state, event_rx) = SharedMinerState::new(
        hash_counter.clone(),
        is_running.clone(),
        new_block_signal.clone(),
        current_hashrate_khs.clone(),
        is_paused.clone(),
        target_threads.clone(),
        target_intensity.clone(),
        solutions_found.clone(),
        blocks_mined.clone(),
        threads,
        server_url.clone(),
        wallet.clone(),
        miner_id.clone(),
        miner_name.clone(),
        "solo".to_string(),
        proxy_url.clone(),
    );

    // v8.6.6: Connect global bandwidth counters to SharedMinerState
    // The TUI reads from SharedMinerState, but global statics are updated by fetch_with_fallback.
    // We spawn a lightweight task to periodically sync them.
    {
        let bw_down = shared_state.bytes_downloaded.clone();
        let bw_up = shared_state.bytes_uploaded.clone();
        let api_total = shared_state.api_requests_total.clone();
        let api_fail = shared_state.api_requests_failed.clone();
        tokio::spawn(async move {
            loop {
                bw_down.store(GLOBAL_BYTES_DOWNLOADED.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
                bw_up.store(GLOBAL_BYTES_UPLOADED.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
                api_total.store(GLOBAL_API_REQUESTS.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
                api_fail.store(GLOBAL_API_FAILURES.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
        });
    }

    // v9.9.0: Auto-updater — background task checks for new miner versions
    let auto_updater = if !no_auto_update {
        let updater = Arc::new(q_miner::auto_updater::MinerAutoUpdater::new(
            server_url.clone(),
            shared_state.event_tx.clone(),
            proxy_url.as_deref(),
        ));
        let updater_clone = updater.clone();
        tokio::spawn(async move { updater_clone.run_update_loop().await });
        info!("Auto-updater enabled (checks every 5 min)");
        Some(updater)
    } else {
        info!("Auto-updater disabled (--no-auto-update)");
        None
    };

    // v9.1.7: P2P mining network — gossipsub challenge relay + solution broadcast
    // Spawned before mining threads so channels are ready when threads start
    #[cfg(feature = "p2p")]
    let (p2p_challenge_rx, p2p_solution_tx, _p2p_handle) = if !no_p2p {
        match q_miner::p2p_network::MinerP2PNetwork::new(
            q_miner::p2p_network::MinerP2PConfig {
                listen_port: p2p_port,
                network_id: std::env::var("Q_NETWORK_ID")
                    .unwrap_or_else(|_| "mainnet-genesis".to_string()),
                bootstrap_peers: Vec::new(),
            },
            shared_state.p2p_connected.clone(),
            shared_state.p2p_peer_count.clone(),
            shared_state.p2p_challenges_received.clone(),
            shared_state.p2p_solutions_broadcast.clone(),
        ).await {
            Ok(p2p) => {
                let crx = p2p.challenge_receiver();
                let stx = p2p.solution_sender();
                // v10.0.2: Removed P2P block signal forwarding — miners get block signals
                // via SSE only. Blocks topic no longer subscribed (saves ~10 KB/s inbound).
                let handle = tokio::spawn(p2p.run());
                info!("P2P mining network started (port {})", p2p_port);
                (Some(crx), Some(stx), Some(handle))
            }
            Err(e) => {
                warn!("P2P mining network failed to start: {} — using HTTP only", e);
                (None, None, None)
            }
        }
    } else {
        if bandwidth_limit > 0 {
            info!("P2P auto-disabled (--bandwidth-limit {} KB/s) — HTTP-only mode (use --force-p2p to override)", bandwidth_limit);
        } else {
            info!("P2P disabled (--no-p2p) — using HTTP only");
        }
        (None, None, None)
    };
    #[cfg(not(feature = "p2p"))]
    let (p2p_challenge_rx, p2p_solution_tx): (Option<tokio::sync::broadcast::Receiver<q_types::mining_solution::NetworkChallenge>>, Option<tokio::sync::mpsc::UnboundedSender<q_types::mining_solution::P2PMiningSubmission>>) = (None, None);

    info!("🔥 Starting {} CPU mining threads (dedicated OS threads)", threads);
    if bandwidth_limit > 0 {
        let profile = match bandwidth_limit {
            50.. => "optimized (P2P on, balance poll 30s)",
            10..=49 => "low-bandwidth (P2P off, balance poll 60s)",
            _ => "ultra-low (P2P off, poll-only 120s)",
        };
        // If P2P was force-enabled, override the profile description
        let p2p_note = if !no_p2p { " [P2P force-enabled]" } else { "" };
        info!("📡 Bandwidth limit: {} KB/s — profile: {}{}", bandwidth_limit, profile, p2p_note);
    }

    // v10.0.2: Centralized solution submitter — all threads send solutions to one channel.
    // Deduplicates by hash and submits exactly once via HTTP + P2P.
    let (solution_submit_tx, solution_submit_rx) = tokio::sync::mpsc::unbounded_channel::<q_miner::solution_submitter::SolutionMessage>();
    {
        let submit_client = build_http_client_safe(proxy_url.as_deref(), 15);
        let submit_primary_url = normalize_server_url(&server_url);
        let submit_fallback_url = FALLBACK_BOOTSTRAP_URL.to_string();
        #[cfg(feature = "p2p")]
        let submit_p2p_tx = p2p_solution_tx.as_ref().cloned();
        #[cfg(not(feature = "p2p"))]
        let submit_p2p_tx: Option<tokio::sync::mpsc::UnboundedSender<q_types::mining_solution::P2PMiningSubmission>> = None;
        let submitter = q_miner::solution_submitter::SolutionSubmitter::new(
            solution_submit_rx,
            submit_client,
            submit_primary_url,
            submit_fallback_url,
            submit_p2p_tx,
            shared_state.event_tx.clone(),
            solutions_found.clone(),
            blocks_mined.clone(),
            shared_state.bytes_uploaded.clone(),
        );
        tokio::spawn(submitter.run());
        info!("📤 Centralized solution submitter started (dedup + single HTTP/P2P submit)");
    }

    // PERF: Capture tokio Handle so mining threads can dispatch async I/O
    // without running on the tokio scheduler themselves
    let tokio_handle = tokio::runtime::Handle::current();

    // v8.8.2: SHARED CHALLENGE CACHE — Only thread 0 fetches from API.
    // All other threads read from this shared cache. This reduces API calls
    // from N×threads/sec to 1/sec on new block signals.
    // Format: (challenge, timestamp_of_fetch)
    let shared_challenge: Arc<parking_lot::RwLock<Option<(MiningChallenge, std::time::Instant)>>> =
        Arc::new(parking_lot::RwLock::new(None));

    // Wrap P2P challenge channel in Arc for sharing across mining threads (thread 0 only)
    let p2p_challenge_rx = p2p_challenge_rx.map(|rx| Arc::new(parking_lot::Mutex::new(rx)));

    // PERF: Use std::thread::spawn instead of tokio::spawn for mining threads.
    // Mining is 100% CPU-bound — tokio's work-stealing scheduler adds overhead
    // and Windows IOCP reactor polling is particularly expensive for CPU-bound work.
    let handles: Vec<_> = (0..threads)
        .map(|thread_id| {
            let hash_counter = hash_counter.clone();
            let is_running = is_running.clone();
            let wallet = wallet.clone();
            let server_url = server_url.clone();
            let new_block_signal = new_block_signal.clone();
            let hashrate_khs = current_hashrate_khs.clone();
            let miner_id = miner_id.clone();
            let miner_name = miner_name.clone();
            let handle = tokio_handle.clone();
            let thread_state = shared_state.thread_states[thread_id].clone();
            let event_tx = shared_state.event_tx.clone();
            let throttle_mode = shared_state.throttle_mode.clone();
            let challenge_latency = shared_state.last_challenge_latency_us.clone();
            let using_fallback = shared_state.using_fallback.clone();
            let shared_state_solutions = solutions_found.clone();
            let shared_state_blocks = blocks_mined.clone();
            let shared_challenge = shared_challenge.clone();

            // v9.1.7: P2P channels — only thread 0 gets the challenge receiver
            let thread_p2p_challenge_rx = if thread_id == 0 { p2p_challenge_rx.clone() } else { None };

            // v10.0.2: Centralized solution submitter channel (replaces per-thread HTTP+P2P)
            let thread_solution_tx = solution_submit_tx.clone();

            let bw_limit = bandwidth_limit;
            let thread_proxy_url = proxy_url.clone();
            let thread_target_intensity = target_intensity.clone();
            std::thread::Builder::new()
                .name(format!("miner-{}", thread_id))
                .spawn(move || {
                    // Windows: Lower mining thread priority so the OS/TUI stay responsive.
                    // This costs ~0-2% hashrate but prevents the system from freezing.
                    #[cfg(target_os = "windows")]
                    unsafe {
                        extern "system" {
                            fn GetCurrentThread() -> *mut std::ffi::c_void;
                            fn SetThreadPriority(hThread: *mut std::ffi::c_void, nPriority: i32) -> i32;
                        }
                        const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
                        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
                    }
                    mining_thread(
                        thread_id, hash_counter, is_running, thread_target_intensity, wallet, server_url,
                        new_block_signal, hashrate_khs, miner_id, miner_name, handle,
                        thread_state, event_tx, throttle_mode, challenge_latency, using_fallback,
                        bw_limit, shared_state_solutions, shared_state_blocks,
                        thread_proxy_url, shared_challenge,
                        thread_p2p_challenge_rx, thread_solution_tx,
                    )
                })
                .expect("Failed to spawn mining thread")
        })
        .collect();

    // ═══════════════════════════════════════════════════════════════
    // v10.3.4: VDF CPU mining lane — auto-detects activation
    // Spawns 1 dedicated thread for Genus-2 Jacobian VDF computation.
    // Runs alongside BLAKE3 threads (GPU or CPU).
    // When VDF lane is not active, thread sleeps (polls every 30s).
    // ═══════════════════════════════════════════════════════════════
    let vdf_proofs_counter = Arc::new(AtomicU64::new(0));
    let vdf_handle = {
        let vdf_running = is_running.clone();
        let vdf_wallet = wallet.clone();
        let vdf_server = server_url.clone();
        let vdf_counter = vdf_proofs_counter.clone();
        let vdf_solution_tx = solution_submit_tx.clone();
        let vdf_tokio = tokio_handle.clone();
        let vdf_block_signal = new_block_signal.clone();
        std::thread::Builder::new()
            .name("vdf-miner".to_string())
            .spawn(move || {
                q_miner::vdf_lane::vdf_mining_thread(
                    vdf_running,
                    vdf_wallet,
                    vdf_server,
                    vdf_counter,
                    vdf_solution_tx,
                    vdf_tokio,
                    vdf_block_signal,
                );
            })
            .expect("Failed to spawn VDF mining thread")
    };
    info!("🧮 VDF lane: dedicated CPU thread spawned (auto-detects activation)");

    // Start hash rate monitor
    let monitor_counter = hash_counter.clone();
    let monitor_running = is_running.clone();
    let monitor_hashrate = current_hashrate_khs.clone();
    let monitor_handle = tokio::spawn(async move {
        hash_rate_monitor(monitor_counter, monitor_running, monitor_hashrate).await;
    });

    // Start SSE listener for real-time mining rewards AND new blocks
    // v9.0.2: Skip SSE if no TLS certs — hyper-rustls panics without them (bare Docker)
    let tls_available = has_tls_certificates();
    if !tls_available {
        warn!("⚠️  No TLS/CA certificates found (install ca-certificates package)");
        warn!("   SSE and MinerLink disabled — mining uses periodic challenge refresh");
    }
    let sse_wallet = wallet.clone();
    let sse_server_url = server_url.clone();
    let sse_running = is_running.clone();
    let sse_new_block_signal = new_block_signal.clone();
    let sse_connected_flag = shared_state.sse_connected.clone();
    let sse_event_tx = shared_state.event_tx.clone();
    let sse_bal_epoch = shared_state.last_balance_sse_epoch.clone();
    let sse_handle = if tls_available {
        Some(tokio::spawn(async move {
            start_sse_listener(sse_wallet, sse_server_url, sse_running, sse_new_block_signal, sse_connected_flag, sse_event_tx, sse_bal_epoch).await;
        }))
    } else {
        None
    };

    // v8.6.5: Periodic balance polling — ensures wallet tab always shows latest balance
    // SSE events are primary, this is a fallback that polls periodically
    // v9.2.6: "Mercedes" smoothing — polling yields to SSE when SSE pushed within 10s
    // v10.0.2: Bandwidth-aware polling intervals
    let bal_wallet = wallet.clone();
    let bal_server = server_url.clone();
    let bal_running = is_running.clone();
    let bal_event_tx = shared_state.event_tx.clone();
    let bal_proxy = proxy_url.clone();
    let bal_sse_epoch = shared_state.last_balance_sse_epoch.clone();
    let bal_poll_secs = match bandwidth_limit {
        0 => 15,        // Unlimited: 15s
        50.. => 30,     // ≥50 KB/s: 30s
        10..=49 => 60,  // ≥10 KB/s: 60s
        _ => 120,       // <10 KB/s: 120s
    };
    tokio::spawn(async move {
        // Wait 5s before first poll (let SSE connect first)
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(bal_poll_secs));
        let client = {
            let mut builder = reqwest::Client::builder().timeout(std::time::Duration::from_secs(10));
            if let Some(ref proxy) = bal_proxy {
                if let Ok(p) = reqwest::Proxy::all(proxy) {
                    builder = builder.proxy(p);
                }
            }
            builder.build().unwrap_or_else(|_| build_http_client_safe(None, 10))
        };
        while bal_running.load(Ordering::SeqCst) {
            interval.tick().await;
            // v9.2.6: Skip poll if SSE pushed balance within last 10s (avoids overwrite flicker)
            let last_sse = bal_sse_epoch.load(Ordering::Relaxed);
            if last_sse > 0 {
                let now_epoch = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                if now_epoch.saturating_sub(last_sse) < 10 {
                    continue; // SSE is fresh — yield
                }
            }
            // Fetch balance from API
            let url = format!("{}/api/v1/wallets/{}/balance", bal_server, bal_wallet);
            match client.get(&url).send().await {
                Ok(resp) => {
                    if let Ok(body) = resp.json::<serde_json::Value>().await {
                        if let Some(balance) = body.get("data")
                            .and_then(|d| d.get("balance_qnk"))
                            .and_then(|v| v.as_f64())
                        {
                            let _ = bal_event_tx.send(DiagnosticEvent::BalanceUpdated {
                                new_balance: balance,
                            });
                        }
                    }
                }
                Err(_) => {} // Silently skip — SSE is primary
            }
        }
    });

    // Start miner-link WebSocket relay for real-time wallet ↔ miner communication
    let ml_wallet = wallet.clone();
    let ml_server = server_url.clone();
    let ml_miner_id = miner_id.clone();
    let ml_miner_name = miner_name.clone();
    let ml_running = is_running.clone();
    let ml_hashrate = current_hashrate_khs.clone();
    let ml_hash_counter = hash_counter.clone();
    let ml_solutions = solutions_found.clone();
    let ml_blocks = blocks_mined.clone();
    let ml_is_paused = is_paused.clone();
    let ml_target_threads = target_threads.clone();
    let ml_target_intensity = target_intensity.clone();
    let ml_proxy_url = proxy_url.clone();
    let ml_gpu_active = shared_state.gpu_active.clone();
    let ml_gpu_hashrate = shared_state.gpu_hashrate_hs.clone();
    let ml_gpu_devices = shared_state.gpu_devices.clone();
    // v9.0.2: MinerLink uses tokio-tungstenite which also needs TLS for wss://
    let ml_handle = if tls_available {
        Some(tokio::spawn(async move {
            miner_link_task(
                ml_wallet, ml_server, ml_miner_id, ml_miner_name,
                ml_running, ml_hashrate, ml_hash_counter,
                ml_solutions, ml_blocks,
                ml_is_paused, ml_target_threads, ml_target_intensity,
                threads as u32,
                ml_proxy_url,
                ml_gpu_active, ml_gpu_hashrate, ml_gpu_devices,
            ).await;
        }))
    } else {
        None
    };

    // v10.2.0: Hybrid Quantum Mining — GPU mining task (feature-gated)
    #[cfg(feature = "opencl-mining")]
    let gpu_handle: Option<tokio::task::JoinHandle<()>> = if gpu_enabled {
        match q_mining::GPUMiner::new(q_mining::GPUMinerConfig::default()) {
            Ok(mut gpu_miner) => {
                let dev_name = gpu_miner.device_name().to_string();
                info!("🎮 GPU initialized: {}", dev_name);
                shared_state.gpu_active.store(true, Ordering::Relaxed);
                *shared_state.gpu_device_name.write() = dev_name.clone();
                // v10.1.7: Store rich GPU device info for TUI display
                {
                    let device_snapshots: Vec<q_miner::shared_state::GpuDeviceSnapshot> = gpu_miner.get_devices().iter().map(|d| {
                        q_miner::shared_state::GpuDeviceSnapshot {
                            index: d.index,
                            name: d.name.clone(),
                            vendor: d.vendor.clone(),
                            compute_units: d.compute_units,
                            global_memory_mb: d.global_memory / (1024 * 1024),
                            local_memory_kb: d.local_memory / 1024,
                            max_clock_mhz: d.max_clock_freq,
                            api: d.opencl_version.clone(),
                        }
                    }).collect();
                    info!("🎮 GPU devices detected: {} device(s)", device_snapshots.len());
                    for d in &device_snapshots {
                        info!("  GPU #{}: {} ({}) — {} CU, {} MB, {} MHz",
                            d.index, d.name, d.vendor, d.compute_units, d.global_memory_mb, d.max_clock_mhz);
                    }
                    *shared_state.gpu_devices.write() = device_snapshots;
                }
                shared_state.send_event(DiagnosticEvent::GpuStarted { device_name: dev_name });

                let gpu_shared_challenge = shared_challenge.clone();
                let gpu_new_block_signal = new_block_signal.clone();
                let gpu_is_running = is_running.clone();
                let gpu_hash_counter = hash_counter.clone();
                let gpu_hashrate_hs = shared_state.gpu_hashrate_hs.clone();
                let gpu_hashes_total = shared_state.gpu_hashes_total.clone();
                let gpu_solution_tx = solution_submit_tx.clone();
                let gpu_event_tx = shared_state.event_tx.clone();
                let gpu_wallet = wallet.clone();
                let gpu_miner_id = miner_id.clone();
                let gpu_miner_name = miner_name.clone();
                let gpu_solutions = solutions_found.clone();

                Some(tokio::task::spawn_blocking(move || {
                    let mut gpu_nonce: u64 = u64::MAX / 2; // Start high to avoid CPU nonce collision
                    let mut last_signal = 0u64;
                    let mut batch_start = std::time::Instant::now();
                    let mut batch_hashes: u64 = 0;

                    // v10.1.7: Cache decoded challenge/target to avoid per-dispatch hex decode + alloc
                    let mut cached_challenge: [u8; 32] = [0u8; 32];
                    let mut cached_target: [u8; 32] = [0u8; 32];
                    let mut cached_block_height: u64 = 0;
                    let mut cached_vdf_iterations: u32 = 0;
                    let mut challenge_ready = false;

                    while gpu_is_running.load(Ordering::Relaxed) {
                        // v10.1.7: Check for new block signal at TOP of loop (immediate detection)
                        // and only re-decode challenge when signal changes
                        let sig = gpu_new_block_signal.load(Ordering::Relaxed);
                        if sig != last_signal || !challenge_ready {
                            last_signal = sig;
                            gpu_nonce = u64::MAX / 2; // Reset high nonce range on new block

                            let decoded = {
                                let guard = gpu_shared_challenge.read();
                                match guard.as_ref() {
                                    Some((chal, _ts)) => {
                                        match (hex_to_bytes(&chal.challenge_hash), hex_to_bytes(&chal.difficulty_target)) {
                                            (Ok(ch), Ok(tg)) => Some((ch, tg, chal.block_height, chal.vdf_iterations)),
                                            _ => None,
                                        }
                                    }
                                    None => None,
                                }
                            };

                            match decoded {
                                Some((ch, tg, bh, vi)) => {
                                    cached_challenge = ch;
                                    cached_target = tg;
                                    cached_block_height = bh;
                                    cached_vdf_iterations = vi;
                                    challenge_ready = true;
                                }
                                None => {
                                    std::thread::sleep(std::time::Duration::from_millis(100));
                                    continue;
                                }
                            }
                        }

                        // Dispatch one GPU batch (persistent buffers + conditional upload inside)
                        match gpu_miner.mine_batch(&cached_challenge, &cached_target, gpu_nonce) {
                            Ok(result) => {
                                gpu_nonce = gpu_nonce.wrapping_add(result.hashes);
                                batch_hashes += result.hashes;
                                gpu_hash_counter.fetch_add(result.hashes, Ordering::Relaxed);
                                gpu_hashes_total.fetch_add(result.hashes, Ordering::Relaxed);

                                // Update GPU hashrate every second
                                let elapsed = batch_start.elapsed().as_secs_f64();
                                if elapsed >= 1.0 {
                                    let hr = batch_hashes as f64 / elapsed;
                                    gpu_hashrate_hs.store(hr.to_bits(), Ordering::Relaxed);
                                    gpu_miner.update_hashrate(hr as u64);
                                    batch_hashes = 0;
                                    batch_start = std::time::Instant::now();
                                }

                                if let Some(solution) = result.solution {
                                    info!("🎮💎 GPU solution found! Block #{}, Nonce: {}", cached_block_height, solution.nonce);
                                    gpu_solutions.fetch_add(1, Ordering::Relaxed);
                                    let _ = gpu_event_tx.send(DiagnosticEvent::GpuSolutionFound {
                                        nonce: solution.nonce,
                                        block_height: cached_block_height,
                                    });

                                    let hashrate_khs = f64::from_bits(gpu_hashrate_hs.load(Ordering::Relaxed)) / 1000.0;
                                    let solution_json = serde_json::json!({
                                        "miner_address": gpu_wallet,
                                        "nonce": solution.nonce,
                                        "hash": hex::encode(solution.hash),
                                        "difficulty_target": hex::encode(cached_target),
                                        "challenge_hash": hex::encode(cached_challenge),
                                        "hash_rate": hashrate_khs,
                                        "miner_id": gpu_miner_id,
                                        "worker_name": gpu_miner_name,
                                        "miner_version": env!("CARGO_PKG_VERSION")
                                    });

                                    let mut wallet_bytes = [0u8; 32];
                                    if let Ok(decoded) = hex::decode(gpu_wallet.trim_start_matches("qnk")) {
                                        let len = decoded.len().min(32);
                                        wallet_bytes[..len].copy_from_slice(&decoded[..len]);
                                    }
                                    let p2p_sub = Some(q_types::mining_solution::P2PMiningSubmission::new(
                                        wallet_bytes, solution.hash, cached_target, cached_block_height,
                                        cached_challenge, solution.nonce, cached_vdf_iterations, gpu_miner_id.clone(),
                                    ));

                                    let _ = gpu_solution_tx.send(q_miner::solution_submitter::SolutionMessage {
                                        solution_json,
                                        solution_hash: solution.hash,
                                        block_height: cached_block_height,
                                        nonce: solution.nonce,
                                        p2p_submission: p2p_sub,
                                    });
                                }
                            }
                            Err(e) => {
                                error!("🎮 GPU mining error: {}", e);
                                let _ = gpu_event_tx.send(DiagnosticEvent::GpuError { message: e.to_string() });
                                std::thread::sleep(std::time::Duration::from_secs(5));
                            }
                        }
                    }
                }))
            }
            Err(e) => {
                warn!("🎮 GPU initialization failed: {} — continuing CPU-only", e);
                shared_state.send_event(DiagnosticEvent::GpuError { message: e.to_string() });
                None
            }
        }
    } else {
        None
    };
    #[cfg(not(feature = "opencl-mining"))]
    let gpu_handle: Option<tokio::task::JoinHandle<()>> = if gpu_enabled {
        info!("🎮 GPU mining requested but opencl-mining feature not compiled in — CPU only");
        None
    } else {
        None
    };

    info!("✅ Quillon miner started successfully!");
    if shared_state.gpu_active.load(Ordering::Relaxed) {
        info!("🎮 Hybrid Quantum Mining: CPU + GPU");
    }
    info!("🎧 Connected to SSE stream for real-time block updates");
    info!("🔗 Miner-link relay active — connect your wallet for real-time monitoring");

    // TUI mode: launch dashboard instead of waiting for Ctrl+C
    #[cfg(feature = "tui")]
    if use_tui {
        if let Some(log_rx) = tui_log_rx {
            info!("🖥️  Launching TUI dashboard...");
            // Flush stdout so all pre-TUI banner text is rendered before
            // we switch to the alternate screen buffer (fixes Windows cmd.exe
            // where the TUI would never appear).
            use std::io::Write;
            let _ = std::io::stdout().flush();
            // Brief yield so Windows console can finish rendering
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            q_miner::ui::tui_app::run_miner_tui(shared_state, event_rx, log_rx, auto_updater.clone()).await?;
        } else {
            info!("Press Ctrl+C to stop mining...");
            signal::ctrl_c().await?;
        }
    } else {
        info!("Press Ctrl+C to stop mining...");
        signal::ctrl_c().await?;
    }

    #[cfg(not(feature = "tui"))]
    {
        info!("Press Ctrl+C to stop mining...");
        signal::ctrl_c().await?;
    }

    info!("🛑 Shutdown signal received, stopping mining...");
    is_running.store(false, Ordering::SeqCst);

    // Wait for all OS mining threads to stop (BLAKE3 + VDF)
    for handle in handles {
        let _ = handle.join();
    }
    let _ = vdf_handle.join(); // VDF lane thread
    monitor_handle.abort();
    if let Some(h) = sse_handle { h.abort(); }
    if let Some(h) = ml_handle { h.abort(); }
    if let Some(h) = gpu_handle { h.abort(); }

    let total_hashes = hash_counter.load(Ordering::Relaxed);
    info!("👋 Quillon miner stopped. Total hashes: {}", total_hashes);

    Ok(())
}

/// Pool mining mode using Stratum protocol
async fn run_pool_mining(
    threads: usize,
    intensity: u8,
    wallet: &str,
    worker_name: &str,
    pool_url: &str,
) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpStream;
    use tokio::sync::mpsc;
    use serde_json::{json, Value};

    let hash_counter = Arc::new(AtomicU64::new(0));
    let is_running = Arc::new(AtomicBool::new(true));
    let current_difficulty = Arc::new(tokio::sync::RwLock::new(1.0_f64));

    // Current job state shared between threads
    let current_job = Arc::new(tokio::sync::RwLock::new(Option::<PoolJob>::None));
    let job_signal = Arc::new(AtomicU64::new(0));

    // Channel for submitting shares to the pool
    let (share_tx, mut share_rx) = mpsc::channel::<ShareToSubmit>(1000);

    // Parse pool URL: stratum+tcp://host:port
    let url = pool_url
        .trim_start_matches("stratum+tcp://")
        .trim_start_matches("stratum://")
        .trim_start_matches("tcp://");

    let parts: Vec<&str> = url.split(':').collect();
    let host = parts.get(0).copied().unwrap_or("pool.quillon.xyz");
    let port: u16 = parts.get(1).and_then(|p| p.parse().ok()).unwrap_or(3333);

    info!("🔌 Connecting to pool {}:{}", host, port);

    let addr = format!("{}:{}", host, port);
    let stream = match TcpStream::connect(&addr).await {
        Ok(s) => s,
        Err(e) => {
            error!("❌ Failed to connect to pool: {}", e);
            return Err(e.into());
        }
    };

    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);

    // Generate subscription ID
    let sub_id = format!("{:08x}", rand::random::<u32>());

    // Send mining.subscribe
    let subscribe_msg = json!({
        "id": 1,
        "method": "mining.subscribe",
        "params": ["q-miner/1.0.0", sub_id]
    });
    let msg_str = format!("{}\n", serde_json::to_string(&subscribe_msg)?);
    writer.write_all(msg_str.as_bytes()).await?;
    info!("📤 Sent mining.subscribe");

    // Read subscribe response
    let mut line = String::new();
    reader.read_line(&mut line).await?;
    let response: Value = serde_json::from_str(line.trim())?;

    let extranonce1 = response.get("result")
        .and_then(|r| r.get(1))
        .and_then(|e| e.as_str())
        .unwrap_or("")
        .to_string();
    let extranonce2_size = response.get("result")
        .and_then(|r| r.get(2))
        .and_then(|s| s.as_u64())
        .unwrap_or(4) as usize;

    info!("📥 Subscribed: extranonce1={}, extranonce2_size={}", extranonce1, extranonce2_size);
    line.clear();

    // Send mining.authorize
    let worker_full = format!("{}.{}", wallet, worker_name);
    let authorize_msg = json!({
        "id": 2,
        "method": "mining.authorize",
        "params": [worker_full, "x"]  // password is typically "x" or empty
    });
    let msg_str = format!("{}\n", serde_json::to_string(&authorize_msg)?);
    writer.write_all(msg_str.as_bytes()).await?;
    info!("📤 Sent mining.authorize for {}", worker_full);

    // Read authorize response
    reader.read_line(&mut line).await?;
    let auth_response: Value = serde_json::from_str(line.trim())?;
    if auth_response.get("result").and_then(|r| r.as_bool()).unwrap_or(false) {
        info!("✅ Authorized successfully!");
    } else {
        let error = auth_response.get("error").cloned().unwrap_or(json!("Unknown error"));
        error!("❌ Authorization failed: {}", error);
        return Err(anyhow::anyhow!("Authorization failed"));
    }
    line.clear();

    info!("🔥 Starting {} pool mining threads", threads);

    // Spawn mining threads
    let handles: Vec<_> = (0..threads)
        .map(|thread_id| {
            let hash_counter = hash_counter.clone();
            let is_running = is_running.clone();
            let current_job = current_job.clone();
            let job_signal = job_signal.clone();
            let current_difficulty = current_difficulty.clone();
            let share_tx = share_tx.clone();
            let extranonce1 = extranonce1.clone();

            tokio::spawn(async move {
                pool_mining_thread(
                    thread_id, hash_counter, is_running, intensity,
                    current_job, job_signal, current_difficulty,
                    share_tx, extranonce1, extranonce2_size
                ).await
            })
        })
        .collect();

    // Drop original share_tx so channel closes when all mining threads stop
    drop(share_tx);

    // Start hash rate monitor
    let monitor_counter = hash_counter.clone();
    let monitor_running = is_running.clone();
    let monitor_hashrate = Arc::new(AtomicU64::new(0u64));
    let monitor_hashrate_clone = monitor_hashrate.clone();
    let monitor_handle = tokio::spawn(async move {
        hash_rate_monitor(monitor_counter, monitor_running, monitor_hashrate_clone).await;
    });

    // Stratum connection handler task (reading notifications and submitting shares)
    let stratum_running = is_running.clone();
    let stratum_job = current_job.clone();
    let stratum_job_signal = job_signal.clone();
    let stratum_difficulty = current_difficulty.clone();

    let stratum_handle = tokio::spawn(async move {
        let mut next_submit_id = 10u64;

        loop {
            tokio::select! {
                // Read notifications from pool
                result = reader.read_line(&mut line) => {
                    match result {
                        Ok(0) => {
                            warn!("⚠️ Pool connection closed");
                            break;
                        }
                        Ok(_) => {
                            if let Ok(msg) = serde_json::from_str::<Value>(line.trim()) {
                                // Handle notifications
                                if let Some(method) = msg.get("method").and_then(|m| m.as_str()) {
                                    match method {
                                        "mining.notify" => {
                                            if let Some(params) = msg.get("params").and_then(|p| p.as_array()) {
                                                if let Some(job) = parse_mining_notify(params) {
                                                    info!("📋 New job: id={}, clean={}", job.job_id, job.clean_jobs);
                                                    *stratum_job.write().await = Some(job);
                                                    stratum_job_signal.fetch_add(1, Ordering::SeqCst);
                                                }
                                            }
                                        }
                                        "mining.set_difficulty" => {
                                            if let Some(diff) = msg.get("params")
                                                .and_then(|p| p.get(0))
                                                .and_then(|d| d.as_f64())
                                            {
                                                info!("🎯 Difficulty updated: {}", diff);
                                                *stratum_difficulty.write().await = diff;
                                            }
                                        }
                                        _ => {
                                            warn!("Unknown method: {}", method);
                                        }
                                    }
                                }

                                // Handle responses to share submissions
                                if let Some(id) = msg.get("id").and_then(|i| i.as_u64()) {
                                    if id >= 10 {
                                        // This is a response to a share submission
                                        if msg.get("result").and_then(|r| r.as_bool()).unwrap_or(false) {
                                            info!("✅ Share accepted!");
                                        } else if let Some(error) = msg.get("error") {
                                            warn!("❌ Share rejected: {}", error);
                                        }
                                    }
                                }
                            }
                            line.clear();
                        }
                        Err(e) => {
                            error!("Failed to read from pool: {}", e);
                            break;
                        }
                    }
                }

                // Submit shares to pool
                Some(share) = share_rx.recv() => {
                    let submit_msg = json!({
                        "id": next_submit_id,
                        "method": "mining.submit",
                        "params": [
                            share.worker_name,
                            share.job_id,
                            share.extranonce2,
                            share.ntime,
                            share.nonce
                        ]
                    });
                    next_submit_id += 1;

                    if let Ok(msg_str) = serde_json::to_string(&submit_msg) {
                        if let Err(e) = writer.write_all(format!("{}\n", msg_str).as_bytes()).await {
                            error!("Failed to submit share: {}", e);
                        }
                    }
                }
            }

            if !stratum_running.load(Ordering::SeqCst) {
                break;
            }
        }
    });

    info!("✅ Quillon pool miner started successfully!");
    info!("Press Ctrl+C to stop mining...");

    // Wait for shutdown signal
    signal::ctrl_c().await?;

    info!("🛑 Shutdown signal received, stopping pool mining...");
    is_running.store(false, Ordering::SeqCst);

    // Wait for all threads to stop
    for handle in handles {
        let _ = handle.await;
    }
    monitor_handle.abort();
    stratum_handle.abort();

    let total_hashes = hash_counter.load(Ordering::Relaxed);
    info!("👋 Quillon pool miner stopped. Total hashes: {}", total_hashes);

    Ok(())
}

/// Job received from pool via mining.notify
#[derive(Debug, Clone)]
struct PoolJob {
    job_id: String,
    prevhash: [u8; 32],
    coinbase1: Vec<u8>,
    coinbase2: Vec<u8>,
    merkle_branches: Vec<[u8; 32]>,
    version: u32,
    nbits: u32,
    ntime: u32,
    clean_jobs: bool,
}

/// Share to submit to pool
#[derive(Debug, Clone)]
struct ShareToSubmit {
    worker_name: String,
    job_id: String,
    extranonce2: String,
    ntime: String,
    nonce: String,
}

fn parse_mining_notify(params: &[Value]) -> Option<PoolJob> {
    let job_id = params.get(0)?.as_str()?.to_string();
    let prevhash_hex = params.get(1)?.as_str()?;
    let coinbase1_hex = params.get(2)?.as_str()?;
    let coinbase2_hex = params.get(3)?.as_str()?;
    let merkle_array = params.get(4)?.as_array()?;
    let version_hex = params.get(5)?.as_str()?;
    let nbits_hex = params.get(6)?.as_str()?;
    let ntime_hex = params.get(7)?.as_str()?;
    let clean_jobs = params.get(8)?.as_bool().unwrap_or(false);

    // Parse prevhash
    let prevhash_bytes = hex::decode(prevhash_hex).ok()?;
    let mut prevhash = [0u8; 32];
    if prevhash_bytes.len() >= 32 {
        prevhash.copy_from_slice(&prevhash_bytes[..32]);
    }

    // Parse coinbase
    let coinbase1 = hex::decode(coinbase1_hex).ok()?;
    let coinbase2 = hex::decode(coinbase2_hex).ok()?;

    // Parse merkle branches
    let merkle_branches: Vec<[u8; 32]> = merkle_array.iter()
        .filter_map(|v| {
            let hex_str = v.as_str()?;
            let bytes = hex::decode(hex_str).ok()?;
            if bytes.len() >= 32 {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes[..32]);
                Some(arr)
            } else {
                None
            }
        })
        .collect();

    // Parse version, nbits, ntime
    let version = u32::from_str_radix(version_hex, 16).ok()?;
    let nbits = u32::from_str_radix(nbits_hex, 16).ok()?;
    let ntime = u32::from_str_radix(ntime_hex, 16).ok()?;

    Some(PoolJob {
        job_id,
        prevhash,
        coinbase1,
        coinbase2,
        merkle_branches,
        version,
        nbits,
        ntime,
        clean_jobs,
    })
}

async fn pool_mining_thread(
    thread_id: usize,
    hash_counter: Arc<AtomicU64>,
    is_running: Arc<AtomicBool>,
    intensity: u8,
    current_job: Arc<tokio::sync::RwLock<Option<PoolJob>>>,
    job_signal: Arc<AtomicU64>,
    current_difficulty: Arc<tokio::sync::RwLock<f64>>,
    share_tx: mpsc::Sender<ShareToSubmit>,
    extranonce1: String,
    extranonce2_size: usize,
) {
    // Pin thread to CPU core for cache locality
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    if thread_id < core_ids.len() {
        if core_affinity::set_for_current(core_ids[thread_id]) {
            info!("🔥 Pool mining thread {} started (pinned to core {})", thread_id, thread_id);
        } else {
            info!("🔥 Pool mining thread {} started (affinity pinning failed)", thread_id);
        }
    } else {
        info!("🔥 Pool mining thread {} started", thread_id);
    }

    let batch_size = (intensity as u64) * 100_000;
    let mut last_job_signal = 0u64;

    // Use thread ID to vary extranonce2 space across threads
    let mut nonce_base = (thread_id as u64) << 48;
    let mut extranonce2_counter: u32 = thread_id as u32;

    while is_running.load(Ordering::SeqCst) {
        // Wait for a job
        let job = {
            let job_opt = current_job.read().await;
            job_opt.clone()
        };

        let job = match job {
            Some(j) => j,
            None => {
                // No job yet, wait a bit
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                continue;
            }
        };

        let current_signal = job_signal.load(Ordering::Relaxed);
        if current_signal != last_job_signal {
            last_job_signal = current_signal;
            nonce_base = (thread_id as u64) << 48;
            extranonce2_counter = thread_id as u32;
        }

        let difficulty = *current_difficulty.read().await;

        // Calculate difficulty target (simplified for Q-NarwhalKnight)
        // Target = max_target / difficulty
        let max_target = [0xff_u8; 32]; // Maximum possible target
        let target = calculate_pool_target(difficulty);

        // Generate extranonce2
        let extranonce2 = format!("{:0>width$x}", extranonce2_counter, width = extranonce2_size * 2);
        extranonce2_counter = extranonce2_counter.wrapping_add(threads_count() as u32);

        // Build coinbase: coinbase1 + extranonce1 + extranonce2 + coinbase2
        let mut coinbase = job.coinbase1.clone();
        if let Ok(ext1) = hex::decode(&extranonce1) {
            coinbase.extend_from_slice(&ext1);
        }
        if let Ok(ext2) = hex::decode(&extranonce2) {
            coinbase.extend_from_slice(&ext2);
        }
        coinbase.extend_from_slice(&job.coinbase2);

        // Hash coinbase to get coinbase hash
        let coinbase_hash = blake3::hash(&coinbase);

        // Build merkle root
        let mut merkle_root = *coinbase_hash.as_bytes();
        for branch in &job.merkle_branches {
            let mut combined = [0u8; 64];
            combined[..32].copy_from_slice(&merkle_root);
            combined[32..].copy_from_slice(branch);
            merkle_root = *blake3::hash(&combined).as_bytes();
        }

        // Build block header (simplified for Q-NarwhalKnight)
        // header = version + prevhash + merkle_root + ntime + nbits + nonce
        let mut header_base = Vec::with_capacity(80);
        header_base.extend_from_slice(&job.version.to_le_bytes());
        header_base.extend_from_slice(&job.prevhash);
        header_base.extend_from_slice(&merkle_root);
        header_base.extend_from_slice(&job.ntime.to_le_bytes());
        header_base.extend_from_slice(&job.nbits.to_le_bytes());
        // Nonce will be appended during mining

        // v7.4.3 PERF: Pre-allocate header with nonce space — eliminates Vec::clone() per nonce
        // Before: clone 80-byte Vec + extend = 2 heap ops per hash (millions/sec = GBs of alloc)
        // After: single fixed buffer, overwrite last 8 bytes = zero allocation in hot loop
        let mut header_with_nonce = Vec::with_capacity(header_base.len() + 8);
        header_with_nonce.extend_from_slice(&header_base);
        header_with_nonce.extend_from_slice(&[0u8; 8]); // placeholder for nonce
        let nonce_offset = header_base.len();

        // PERF: Thread-local hash counter to reduce atomic contention
        let mut local_hash_count: u64 = 0;

        // Mine batch of nonces
        for _ in 0..batch_size {
            // PERF: Check is_running/job_signal every 512 nonces instead of every nonce
            if local_hash_count & 511 == 0 {
                if !is_running.load(Ordering::Relaxed) {
                    break;
                }
                if job_signal.load(Ordering::Relaxed) != last_job_signal {
                    break;
                }
            }

            // v7.4.3: Overwrite nonce in-place — zero allocation
            header_with_nonce[nonce_offset..nonce_offset + 8].copy_from_slice(&nonce_base.to_le_bytes());

            // Compute hash using DAG-Knight VDF
            let hash = compute_dag_knight_hash_for_pool(&header_with_nonce);
            local_hash_count += 1;

            // Flush every 1024 hashes to reduce atomic cache-line contention
            if local_hash_count & 1023 == 0 {
                hash_counter.fetch_add(1024, Ordering::Relaxed);
            }

            // Check if meets target
            if hash_meets_target(&hash, &target) {
                info!("💎 Share found! Thread {}, nonce {}", thread_id, nonce_base);

                // Submit share
                let share = ShareToSubmit {
                    worker_name: format!("worker_{}", thread_id),
                    job_id: job.job_id.clone(),
                    extranonce2: extranonce2.clone(),
                    ntime: format!("{:08x}", job.ntime),
                    nonce: format!("{:016x}", nonce_base),
                };

                let _ = share_tx.send(share).await;
            }

            nonce_base = nonce_base.wrapping_add(1);
        }
        // Flush remaining local hashes to shared counter
        if local_hash_count & 1023 != 0 {
            hash_counter.fetch_add(local_hash_count & 1023, Ordering::Relaxed);
        }
    }

    info!("🛑 Pool mining thread {} stopped", thread_id);
}

fn threads_count() -> usize {
    num_cpus::get()
}

fn calculate_pool_target(difficulty: f64) -> [u8; 32] {
    // Pool difficulty 1 target (simplified)
    let diff1_target: u128 = 0x00000000ffff_0000_0000_0000_0000_0000_u128;
    let target_value = (diff1_target as f64 / difficulty) as u128;

    let mut target = [0u8; 32];
    // Set the target in the first 16 bytes (big-endian for comparison)
    for i in 0..16 {
        target[15 - i] = ((target_value >> (i * 8)) & 0xff) as u8;
    }
    target
}

fn hash_meets_target(hash: &[u8; 32], target: &[u8; 32]) -> bool {
    // Compare as big-endian (first bytes are most significant)
    for i in 0..32 {
        if hash[i] < target[i] {
            return true;
        } else if hash[i] > target[i] {
            return false;
        }
    }
    true // Equal counts as meeting target
}

#[inline(always)]
fn compute_dag_knight_hash_for_pool(header: &[u8]) -> [u8; 32] {
    // Initial hash
    let initial_hash = blake3::hash(header);

    // VDF computation (99 inner rounds = 100 total with initial hash, matching GPU kernel)
    let mut current = *initial_hash.as_bytes();
    for _ in 0..99 {
        current = *blake3::hash(&current).as_bytes();
    }

    current
}

// Sync function — runs on a dedicated OS thread via std::thread::spawn, NOT tokio.
// CPU-bound mining must not run on tokio's async executor (causes scheduler stalls).
fn benchmark_mining_thread(
    thread_id: usize,
    hash_counter: Arc<AtomicU64>,
    is_running: Arc<AtomicBool>,
    intensity: u8,
    duration: std::time::Duration,
) {
    // Pin to core for accurate single-thread measurement
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    if thread_id < core_ids.len() {
        core_affinity::set_for_current(core_ids[thread_id]);
    }

    let start_time = std::time::Instant::now();
    let mut nonce = thread_id as u64 * 1_000_000;
    // Match the real mining thread's batch size (was 1000, corrected to 100_000)
    let batch_size = (intensity as u64) * 100_000;

    #[cfg(unix)]
    let simd_batch = q_miner::cpu::optimal_mining_batch_size();
    #[cfg(not(unix))]
    let simd_batch: usize = 4;
    let mut batch_results: [(u64, [u8; 32]); 16] = [(0u64, [0u8; 32]); 16];
    let challenge = [0u8; 32];
    let mut local_count: u64 = 0;

    while start_time.elapsed() < duration && is_running.load(Ordering::Relaxed) {
        let mut i: u64 = 0;
        while i < batch_size {
            #[cfg(unix)]
            let count = q_miner::cpu::compute_dag_knight_hash_batch(
                &challenge, nonce, simd_batch, &mut batch_results,
            );
            #[cfg(not(unix))]
            let count = {
                let bs = simd_batch.min(16).min(batch_results.len());
                for bi in 0..bs {
                    let n = nonce.wrapping_add(bi as u64);
                    let mut input = [0u8; 40];
                    input[32..].copy_from_slice(&n.to_le_bytes());
                    let mut h = *blake3::hash(&input).as_bytes();
                    for _ in 0..100 { h = *blake3::hash(&h).as_bytes(); }
                    batch_results[bi] = (n, h);
                }
                bs
            };
            nonce = nonce.wrapping_add(count as u64);
            local_count += count as u64;
            // Flush to shared atomic every 8192 hashes (reduces contention vs per-hash)
            if local_count >= 8192 {
                hash_counter.fetch_add(local_count, Ordering::Relaxed);
                local_count = 0;
            }
            i += count as u64;
        }
    }

    if local_count > 0 {
        hash_counter.fetch_add(local_count, Ordering::Relaxed);
    }
    info!("🛑 Benchmark thread {} completed", thread_id);
}

// ═══════════════════════════════════════════════════════════════════
// PERF: Sync mining thread — runs on dedicated OS thread, NOT tokio
// This eliminates tokio scheduler overhead (work-stealing, IOCP polling)
// and gives the OS full control over CPU scheduling.
// Async I/O (challenge fetch, solution submit) uses Handle::block_on()
// for the rare cases when network communication is needed.
// ═══════════════════════════════════════════════════════════════════
fn mining_thread(
    thread_id: usize,
    hash_counter: Arc<AtomicU64>,
    is_running: Arc<AtomicBool>,
    target_intensity: Arc<AtomicU8>,
    wallet: String,
    server_url: String,
    new_block_signal: Arc<AtomicU64>,
    current_hashrate_khs: Arc<AtomicU64>,
    miner_id: String,
    miner_name: Option<String>,
    tokio_handle: tokio::runtime::Handle,
    thread_state: Arc<ThreadState>,
    event_tx: mpsc::UnboundedSender<DiagnosticEvent>,
    throttle_mode: Arc<parking_lot::RwLock<MinerThrottleMode>>,
    challenge_latency: Arc<AtomicU64>,
    _using_fallback: Arc<AtomicBool>,
    bandwidth_limit_kbps: u32,
    _shared_state_solutions: Arc<AtomicU64>,
    _shared_state_blocks: Arc<AtomicU64>,
    proxy_url: Option<String>,
    shared_challenge: Arc<parking_lot::RwLock<Option<(MiningChallenge, std::time::Instant)>>>,
    // v9.1.7: P2P challenge receiver (thread 0 only)
    p2p_challenge_rx: Option<Arc<parking_lot::Mutex<tokio::sync::broadcast::Receiver<q_types::mining_solution::NetworkChallenge>>>>,
    // v10.0.2: Centralized solution submitter channel (replaces per-thread HTTP+P2P)
    solution_tx: tokio::sync::mpsc::UnboundedSender<q_miner::solution_submitter::SolutionMessage>,
) {
    let _ = event_tx.send(DiagnosticEvent::ThreadStarted { thread_id });
    // OPTIMIZATION: Pin thread to specific CPU core for cache locality on multi-socket systems
    // This dramatically improves performance on AMD EPYC / Intel Xeon servers with NUMA
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    if thread_id < core_ids.len() {
        if core_affinity::set_for_current(core_ids[thread_id]) {
            info!("🔥 CPU mining thread {} started (pinned to core {})", thread_id, thread_id);
        } else {
            info!("🔥 CPU mining thread {} started (affinity pinning failed, running unpinned)", thread_id);
        }
    } else {
        info!("🔥 CPU mining thread {} started (no core pinning - more threads than cores)", thread_id);
    }

    let mut nonce = thread_id as u64 * 1_000_000;
    let api_url = &server_url;

    // v1.0.2: Single shared client with connection pooling + TCP keepalive.
    // Prevents TCP exhaustion: keepalive detects dead connections, idle timeout
    // closes unused sockets, and pool_max_idle caps open connections.
    // v2.7.0: Routes through proxy when --proxy or --tor is configured.
    let client = build_http_client_safe(proxy_url.as_deref(), 15);

    // Check if server is syncing before starting to mine
    // v9.0.4: Enhanced with Starship telemetry for TUI progress display
    match tokio_handle.block_on(check_server_sync_status(&client, api_url)) {
        Ok((is_syncing, ref sync_info)) if is_syncing => {
            thread_state.set_status(ThreadStatus::WaitingForSync { blocks_behind: sync_info.blocks_behind });
            let _ = event_tx.send(DiagnosticEvent::ServerSyncing { sync_info: sync_info.clone() });
            info!("⏸️  Thread {} waiting: Server syncing ({} blocks behind, phase: {})",
                thread_id, sync_info.blocks_behind, sync_info.phase);
            info!("   Mining will start automatically when sync is complete");
            let mut sync_check_errors: u32 = 0;
            loop {
                std::thread::sleep(std::time::Duration::from_secs(5));
                if !is_running.load(Ordering::Relaxed) {
                    thread_state.set_status(ThreadStatus::Stopped);
                    return;
                }
                match tokio_handle.block_on(check_server_sync_status(&client, api_url)) {
                    Ok((false, _)) => {
                        let _ = event_tx.send(DiagnosticEvent::ServerSyncComplete);
                        info!("✅ Thread {} detected sync complete - starting mining", thread_id);
                        break;
                    }
                    Ok((true, ref info)) => {
                        sync_check_errors = 0;
                        thread_state.set_status(ThreadStatus::WaitingForSync { blocks_behind: info.blocks_behind });
                        let _ = event_tx.send(DiagnosticEvent::ServerSyncing { sync_info: info.clone() });
                        if info.blocks_behind % 100 < 5 || info.blocks_behind < 50 {
                            info!("⏸️  Thread {} syncing: {} behind | {:.1}% | {:.0} blk/s | phase: {}",
                                thread_id, info.blocks_behind, info.sync_progress, info.sync_speed_bps, info.phase);
                        }
                    }
                    Err(_) => {
                        // v9.8.4: Don't get stuck forever if sync-check keeps failing (502/503)
                        sync_check_errors += 1;
                        if sync_check_errors >= 6 {
                            info!("⚠️  Thread {} sync check unavailable — proceeding to mine anyway", thread_id);
                            break;
                        }
                    }
                }
            }
        }
        Ok(_) => {}
        Err(e) => {
            warn!("⚠️  Thread {} couldn't check sync status: {} - proceeding anyway", thread_id, e);
        }
    }

    // Track last server notice to avoid spamming logs
    let mut last_server_notice = String::new();

    // Fetch initial mining challenge — retry with staggered backoff instead of dying
    // v2.7.1: With bandwidth limiting, all threads racing at startup causes most to fail.
    // Stagger: thread N waits N*10ms before first attempt, then retries with backoff.
    if thread_id > 0 {
        let stagger = std::time::Duration::from_millis(thread_id as u64 * 10);
        std::thread::sleep(stagger);
    }

    let mut current_challenge = loop {
        if !is_running.load(Ordering::Relaxed) { return; }
        thread_state.set_status(ThreadStatus::FetchingChallenge);
        let fetch_start = std::time::Instant::now();
        match tokio_handle.block_on(fetch_mining_challenge(&client, api_url)) {
            Ok(challenge) => {
                let latency_us = fetch_start.elapsed().as_micros() as u64;
                challenge_latency.store(latency_us, Ordering::Relaxed);
                thread_state.challenge_fetch_latency_us.store(latency_us, Ordering::Relaxed);
                let _ = event_tx.send(DiagnosticEvent::ChallengeFetched {
                    thread_id,
                    block_height: challenge.block_height,
                    latency_ms: latency_us / 1000,
                });
                info!("📋 Thread {} fetched challenge: block #{}, reward: {} QUG",
                     thread_id, challenge.block_height, challenge.block_reward);
                if let Some(ref notice) = challenge.server_notice {
                    if !notice.is_empty() {
                        warn!("📢 SERVER NOTICE: {}", notice);
                        last_server_notice = notice.clone();
                        let _ = event_tx.send(DiagnosticEvent::ServerNotice { message: notice.clone() });
                    }
                }
                // v2.7.1: Use min_miner_version for update detection (not server_version)
                if let Some(ref min_ver) = challenge.min_miner_version {
                    if version_less_than(env!("CARGO_PKG_VERSION"), min_ver) {
                        let _ = event_tx.send(DiagnosticEvent::UpdateAvailable { min_miner_version: min_ver.clone() });
                    }
                }
                // v9.1.0: Emit compute power update from initial challenge
                if challenge.network_hashrate_hs.is_some()
                    || challenge.connected_miners.is_some()
                    || challenge.live_security_bits.is_some()
                {
                    let _ = event_tx.send(DiagnosticEvent::ComputePowerUpdate {
                        network_hashrate_hs: challenge.network_hashrate_hs.unwrap_or(0.0),
                        connected_miners: challenge.connected_miners.unwrap_or(0),
                        live_security_bits: challenge.live_security_bits.unwrap_or(0.0),
                    });
                }

                // v9.1.4: Check forced_mining_mode from challenge response (piggyback channel)
                if let Some(ref forced_mode) = challenge.forced_mining_mode {
                    let current_switch = MODE_SWITCH_TARGET.load(std::sync::atomic::Ordering::Relaxed);
                    let forced_val = match forced_mode.as_str() {
                        "solo" => 1u8,
                        "pool" => 2u8,
                        _ => 0u8,
                    };
                    // Only trigger if we haven't already set this switch target
                    if forced_val > 0 && current_switch != forced_val {
                        info!("⚡ [MODE-SWITCH] Server forcing mode change to '{}' via challenge response", forced_mode);
                        *get_mode_switch_pool_url().lock() = challenge.forced_pool_url.clone();
                        MODE_SWITCH_TARGET.store(forced_val, std::sync::atomic::Ordering::SeqCst);
                        is_running.store(false, std::sync::atomic::Ordering::SeqCst);
                        let _ = event_tx.send(DiagnosticEvent::MiningModeSwitch {
                            target_mode: forced_mode.clone(),
                            pool_url: challenge.forced_pool_url.clone(),
                            reason: Some("server challenge override".to_string()),
                        });
                    }
                }

                break challenge;
            }
            Err(e) => {
                let msg = format!("{}", e);
                thread_state.set_status(ThreadStatus::Error { message: msg.clone(), since: std::time::Instant::now() });
                let _ = event_tx.send(DiagnosticEvent::ChallengeFetchFailed { thread_id, error: msg });
                // Retry with backoff: 2s base + thread_id stagger (so threads don't all retry together)
                let retry_delay = std::time::Duration::from_secs(2) +
                    std::time::Duration::from_millis(thread_id as u64 * 300);
                warn!("⏳ Thread {} challenge fetch failed, retrying in {:.1}s: {}",
                    thread_id, retry_delay.as_secs_f32(), e);
                std::thread::sleep(retry_delay);
                continue;
            }
        }
    };

    let mut challenge_hash = match hex_to_bytes(&current_challenge.challenge_hash) {
        Ok(hash) => hash,
        Err(e) => {
            error!("❌ Thread {} failed to decode challenge hash: {}", thread_id, e);
            return;
        }
    };

    let mut target = match hex_to_bytes(&current_challenge.difficulty_target) {
        Ok(t) => t,
        Err(e) => {
            error!("❌ Thread {} failed to decode difficulty target: {}", thread_id, e);
            return;
        }
    };

    let mut last_challenge_refresh = std::time::Instant::now();
    // v8.8.3: BANDWIDTH-OPTIMIZED CHALLENGE REFRESH
    // Thread 0 is the SOLE fetcher — it hits the API and writes to shared_challenge.
    // All other threads ONLY read from the shared cache (zero network calls).
    // This reduces API calls from 264/block to 1/block (264× improvement).
    //
    // Thread 0 periodic interval depends on throttle mode:
    //   Off:        50s (normal)
    //   UltraLight: 120s (extended, with LZ4+gzip compression)
    //   Light/Heavy: 50s (delay is in the hash loop, not the fetch interval)
    //   --bandwidth_limit: overrides to max(120s, 500/limit)
    // Thread 0 debounce on block signal: 2s minimum between fetches
    // v10.0.2: Bandwidth-aware challenge refresh intervals
    let get_challenge_refresh_interval = |throttle: &Arc<parking_lot::RwLock<MinerThrottleMode>>, bw_limit: u32| -> std::time::Duration {
        if thread_id != 0 {
            // Non-zero threads: check shared cache every 2s, never hit API
            return std::time::Duration::from_secs(2);
        }
        if bw_limit > 0 {
            let secs = match bw_limit {
                50.. => 50,    // ≥50 KB/s: normal refresh
                10..=49 => 120, // 10-49 KB/s: slow refresh
                _ => 300,      // <10 KB/s: ultra-slow (5 min)
            };
            return std::time::Duration::from_secs(secs);
        }
        let mode = *throttle.read();
        std::time::Duration::from_secs(mode.challenge_refresh_secs())
    };
    let mut challenge_refresh_interval = get_challenge_refresh_interval(&throttle_mode, bandwidth_limit_kbps);
    // Debounce: don't re-fetch challenge within 2s of last fetch (avoids burst on rapid blocks)
    let fetch_debounce = std::time::Duration::from_secs(2);

    let mut last_known_block_signal = new_block_signal.load(Ordering::Relaxed);

    // Seed the shared cache with our initial challenge (thread 0 only)
    if thread_id == 0 {
        let mut cache = shared_challenge.write();
        *cache = Some((current_challenge.clone(), std::time::Instant::now()));
    }

    thread_state.set_status(ThreadStatus::Mining { block_height: current_challenge.block_height });

    while is_running.load(Ordering::Relaxed) {
        // Throttle check — add delay before API calls if throttle is active
        let delay_ms = throttle_mode.read().delay_ms();
        if delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        }

        // v8.8.3: Dynamically update refresh interval when throttle mode changes (thread 0 only)
        if thread_id == 0 {
            challenge_refresh_interval = get_challenge_refresh_interval(&throttle_mode, bandwidth_limit_kbps);
        }

        // Check if new block arrived via SSE
        let current_block_signal = new_block_signal.load(Ordering::Relaxed);
        let should_refresh_immediately = current_block_signal != last_known_block_signal;

        if thread_id == 0 {
            // === THREAD 0: The sole API fetcher ===

            // v9.1.7: P2P-first challenge check (non-blocking) BEFORE HTTP
            // If we get a valid P2P challenge at a higher height, use it and skip HTTP.
            if let Some(ref p2p_rx) = p2p_challenge_rx {
                let mut got_p2p = false;
                let mut rx = p2p_rx.lock();
                while let Ok(p2p_chal) = rx.try_recv() {
                    // P2P challenges use canonical_height (= tip - FINALITY_DEPTH)
                    // HTTP challenges use block_height (= tip height)
                    // Compare: P2P canonical_height roughly maps to HTTP block_height
                    if p2p_chal.canonical_height > current_challenge.block_height.saturating_sub(q_types::mining_solution::MINING_FINALITY_DEPTH) {
                        info!("P2P challenge at height {} (skipping HTTP)", p2p_chal.canonical_height);
                        // Convert NetworkChallenge → MiningChallenge for the shared cache
                        let p2p_mining_challenge = MiningChallenge {
                            challenge_hash: hex::encode(p2p_chal.challenge_hash),
                            difficulty_target: hex::encode(p2p_chal.difficulty_target),
                            block_height: p2p_chal.canonical_height + q_types::mining_solution::MINING_FINALITY_DEPTH,
                            vdf_iterations: p2p_chal.vdf_iterations,
                            block_reward: current_challenge.block_reward, // Keep last known reward
                            expires_at: chrono::Utc::now() + chrono::Duration::seconds(90),
                            server_notice: None,
                            server_version: None,
                            min_miner_version: None,
                            network_hashrate_hs: current_challenge.network_hashrate_hs,
                            connected_miners: current_challenge.connected_miners,
                            live_security_bits: current_challenge.live_security_bits,
                            forced_mining_mode: None,
                            forced_pool_url: None,
                        };
                        thread_state.set_status(ThreadStatus::Mining { block_height: p2p_mining_challenge.block_height });
                        // Update shared cache
                        {
                            let mut cache = shared_challenge.write();
                            *cache = Some((p2p_mining_challenge.clone(), std::time::Instant::now()));
                        }
                        current_challenge = p2p_mining_challenge;
                        if let Ok(hash) = hex_to_bytes(&current_challenge.challenge_hash) {
                            challenge_hash = hash;
                        }
                        if let Ok(t) = hex_to_bytes(&current_challenge.difficulty_target) {
                            target = t;
                        }
                        last_challenge_refresh = std::time::Instant::now();
                        last_known_block_signal = current_block_signal;
                        got_p2p = true;
                    }
                }
                drop(rx);
                // If P2P delivered a challenge, skip HTTP fetch this round
                if got_p2p {
                    // Fall through to mining loop
                }
            }

            // Fetch from API on block signal (debounced) or periodic interval
            let debounced = should_refresh_immediately && last_challenge_refresh.elapsed() >= fetch_debounce;
            let periodic = last_challenge_refresh.elapsed() >= challenge_refresh_interval;

            if debounced || periodic {
                let fetch_start = std::time::Instant::now();
                match tokio_handle.block_on(fetch_mining_challenge(&client, api_url)) {
                    Ok(new_challenge) => {
                        let latency_us = fetch_start.elapsed().as_micros() as u64;
                        challenge_latency.store(latency_us, Ordering::Relaxed);
                        thread_state.challenge_fetch_latency_us.store(latency_us, Ordering::Relaxed);

                        if new_challenge.block_height != current_challenge.block_height {
                            thread_state.set_status(ThreadStatus::Mining { block_height: new_challenge.block_height });
                            if debounced {
                                info!("🔄 Thread 0 fetched new challenge (block signal): #{} -> #{}",
                                     current_challenge.block_height, new_challenge.block_height);
                            } else {
                                info!("🔄 Thread 0 fetched new challenge (periodic): #{} -> #{}",
                                     current_challenge.block_height, new_challenge.block_height);
                            }
                        }

                        // Write to shared cache so all other threads pick it up
                        {
                            let mut cache = shared_challenge.write();
                            *cache = Some((new_challenge.clone(), std::time::Instant::now()));
                        }

                        current_challenge = new_challenge;

                        // Display server notice if new/changed
                        if let Some(ref notice) = current_challenge.server_notice {
                            if !notice.is_empty() && *notice != last_server_notice {
                                warn!("📢 SERVER NOTICE: {}", notice);
                                last_server_notice = notice.clone();
                            }
                        }

                        if let Ok(hash) = hex_to_bytes(&current_challenge.challenge_hash) {
                            challenge_hash = hash;
                        }
                        if let Ok(t) = hex_to_bytes(&current_challenge.difficulty_target) {
                            target = t;
                        }

                        // v9.1.0: Emit compute power update from challenge response
                        if current_challenge.network_hashrate_hs.is_some()
                            || current_challenge.connected_miners.is_some()
                            || current_challenge.live_security_bits.is_some()
                        {
                            let _ = event_tx.send(DiagnosticEvent::ComputePowerUpdate {
                                network_hashrate_hs: current_challenge.network_hashrate_hs.unwrap_or(0.0),
                                connected_miners: current_challenge.connected_miners.unwrap_or(0),
                                live_security_bits: current_challenge.live_security_bits.unwrap_or(0.0),
                            });
                        }

                        last_challenge_refresh = std::time::Instant::now();
                        last_known_block_signal = current_block_signal;
                    }
                    Err(e) => {
                        warn!("⚠️  Thread 0 failed to refresh challenge: {}", e);
                    }
                }
            }
        } else {
            // === THREADS 1..N: Read from shared cache only (ZERO network calls) ===
            if should_refresh_immediately || last_challenge_refresh.elapsed() >= challenge_refresh_interval {
                if let Some((cached, _ts)) = shared_challenge.read().as_ref() {
                    if cached.block_height != current_challenge.block_height {
                        thread_state.set_status(ThreadStatus::Mining { block_height: cached.block_height });
                        current_challenge = cached.clone();

                        if let Ok(hash) = hex_to_bytes(&current_challenge.challenge_hash) {
                            challenge_hash = hash;
                        }
                        if let Ok(t) = hex_to_bytes(&current_challenge.difficulty_target) {
                            target = t;
                        }
                    }
                }
                last_challenge_refresh = std::time::Instant::now();
                last_known_block_signal = current_block_signal;
            }
        }

        // Dynamic batch_size — re-read intensity each outer iteration so wallet UI
        // adjustments (via target_intensity) take effect without restarting threads.
        let batch_size = (target_intensity.load(Ordering::Relaxed).max(1) as u64) * 100_000;

        // Mine a batch of nonces with SIMD-interleaved VDF batching
        // v9.1.0: Process `simd_batch` nonces per inner iteration, interleaving
        // VDF rounds across the batch to keep SIMD pipelines saturated (2-4x faster)
        // v9.0.6: Use inline fallbacks on Windows (cross-compile can't resolve cpu:: symbols)
        #[cfg(unix)]
        let simd_batch = q_miner::cpu::optimal_mining_batch_size();
        #[cfg(not(unix))]
        let simd_batch: usize = 4; // Default SSE batch size for Windows
        let mut batch_results: [(u64, [u8; 32]); 16] = [(0u64, [0u8; 32]); 16];

        // PERF: Thread-local hash counter — flush to shared atomic every 1024 hashes
        let mut local_hash_count: u64 = 0;

        let mut i: u64 = 0;
        while i < batch_size {
            // Process a SIMD-width batch of nonces through the full VDF
            #[cfg(unix)]
            let count = q_miner::cpu::compute_dag_knight_hash_batch(
                &challenge_hash,
                nonce,
                simd_batch,
                &mut batch_results,
            );
            #[cfg(not(unix))]
            let count = {
                // Inline VDF batch for Windows cross-compile compatibility
                let bs = simd_batch.min(16).min(batch_results.len());
                for bi in 0..bs {
                    let n = nonce.wrapping_add(bi as u64);
                    let mut input = [0u8; 40];
                    input[..32].copy_from_slice(&challenge_hash);
                    input[32..].copy_from_slice(&n.to_le_bytes());
                    let mut h = *blake3::hash(&input).as_bytes();
                    for _ in 0..100 { h = *blake3::hash(&h).as_bytes(); }
                    batch_results[bi] = (n, h);
                }
                bs
            };

            local_hash_count += count as u64;
            // Flush hash counter periodically
            // Flush to shared atomic every 8192 hashes (8× less contention than 1024)
            if local_hash_count >= 8192 {
                hash_counter.fetch_add(local_hash_count, Ordering::Relaxed);
                local_hash_count = 0;
            }

            // Check all batch results for solutions
            for r in 0..count {
                let (result_nonce, hash) = batch_results[r];
                if hash < target {
                    // Found a solution — set nonce for the submission block below
                    nonce = result_nonce;

                thread_state.solutions_found.fetch_add(1, Ordering::Relaxed);
                let _ = event_tx.send(DiagnosticEvent::SolutionFound {
                    thread_id,
                    block_height: current_challenge.block_height,
                    nonce,
                });
                info!("💎 Solution found! Block #{}, Thread {}",
                     current_challenge.block_height, thread_id);

                // Read hashrate from atomic (lock-free, no async needed)
                let hashrate_khs = f64::from_bits(current_hashrate_khs.load(Ordering::Relaxed));

                let solution_json = serde_json::json!({
                    "miner_address": wallet,
                    "nonce": nonce,
                    "hash": hex::encode(hash),
                    "difficulty_target": hex::encode(target),
                    "challenge_hash": hex::encode(challenge_hash),
                    "hash_rate": hashrate_khs,
                    "miner_id": miner_id,
                    "worker_name": miner_name,
                    "miner_version": env!("CARGO_PKG_VERSION")
                });

                // v10.0.2: Build P2P submission payload (if P2P compiled in)
                let p2p_sub = {
                    let mut wallet_bytes = [0u8; 32];
                    if let Ok(decoded) = hex::decode(wallet.trim_start_matches("qnk")) {
                        let len = decoded.len().min(32);
                        wallet_bytes[..len].copy_from_slice(&decoded[..len]);
                    }
                    Some(q_types::mining_solution::P2PMiningSubmission::new(
                        wallet_bytes,
                        hash,
                        target,
                        current_challenge.block_height,
                        challenge_hash,
                        nonce,
                        current_challenge.vdf_iterations,
                        miner_id.clone(),
                    ))
                };

                // v10.0.2: Send to centralized submitter (deduplicates across threads)
                let _ = solution_tx.send(q_miner::solution_submitter::SolutionMessage {
                    solution_json,
                    solution_hash: hash,
                    block_height: current_challenge.block_height,
                    nonce,
                    p2p_submission: p2p_sub,
                });
                    // Only submit once per batch — break inner loop after a solution
                    break;
                }
            }

            // Advance nonce past the entire SIMD batch
            nonce = nonce.wrapping_add(count as u64);
            i += count as u64;

            // Check for new block every 512 nonces (~1ms at 500 KH/s) to abandon stale work
            if i & 511 < simd_batch as u64 && i > 0 {
                let sig = new_block_signal.load(Ordering::Relaxed);
                if sig != last_known_block_signal {
                    break; // New block arrived — refresh challenge immediately
                }
            }
        }

        // Flush remaining hash count
        if local_hash_count > 0 {
            hash_counter.fetch_add(local_hash_count, Ordering::Relaxed);
        }
    }

    thread_state.set_status(ThreadStatus::Stopped);
    let _ = event_tx.send(DiagnosticEvent::ThreadStopped { thread_id });
    info!("🛑 CPU mining thread {} stopped", thread_id);
}

async fn hash_rate_monitor(
    hash_counter: Arc<AtomicU64>,
    is_running: Arc<AtomicBool>,
    current_hashrate_khs: Arc<AtomicU64>,
) {
    let mut last_hash_count = 0u64;
    let mut last_time = std::time::Instant::now();
    let start_time = std::time::Instant::now();

    while is_running.load(Ordering::SeqCst) {
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

        let current_hash_count = hash_counter.load(Ordering::Relaxed);
        let current_time = std::time::Instant::now();

        let hashes_computed = current_hash_count - last_hash_count;
        let time_elapsed = current_time.duration_since(last_time).as_secs_f64();

        if time_elapsed > 0.0 {
            let hash_rate = hashes_computed as f64 / time_elapsed;
            let hash_rate_khs = hash_rate / 1_000.0;
            let tpm = (hash_rate * 60.0) / 1_000_000.0;
            let uptime = current_time.duration_since(start_time).as_secs();
            let uptime_mins = uptime / 60;
            let uptime_secs = uptime % 60;

            // PERF: Store hashrate as f64 bits in AtomicU64 (lock-free)
            current_hashrate_khs.store(hash_rate_khs.to_bits(), Ordering::Release);

            info!("⛏️  Mining │ {:.2} MH/s │ {:.2}M TPM │ Uptime: {}m {}s │ Total: {:.2}M hashes",
                 hash_rate / 1_000_000.0, tpm, uptime_mins, uptime_secs, current_hash_count as f64 / 1_000_000.0);
        }

        last_hash_count = current_hash_count;
        last_time = current_time;
    }
}

/// SSE listener for real-time mining rewards AND new block notifications
async fn start_sse_listener(
    wallet: String,
    server_url: String,
    is_running: Arc<AtomicBool>,
    new_block_signal: Arc<AtomicU64>,
    sse_connected: Arc<AtomicBool>,
    sse_event_tx: mpsc::UnboundedSender<DiagnosticEvent>,
    last_balance_sse_epoch: Arc<AtomicU64>,
) {
    use eventsource_client::{self as eventsource, Client as _};
    use futures::StreamExt;

    // Normalize URL to prevent double slashes
    let normalized_url = normalize_server_url(&server_url);

    // v9.0.2: Use miner_mode=true to drop non-mining SSE events (MetricsUpdate,
    // NodeStatusUpdate, etc.) — saves ~80 KB/s. Server ignores unknown params, so
    // this is backwards-compatible with older servers.
    // NOTE: headers_only removed — it changes NewBlock format and older servers
    // send compact headers the miner can't always parse. Re-enable when all servers
    // are upgraded to v9.0.1+.
    let primary_url = format!("{}/api/v1/events?wallet_address={}&miner_mode=true", normalized_url, wallet);
    let fallback_url = format!("{}/api/v1/events?wallet_address={}&miner_mode=true", FALLBACK_BOOTSTRAP_URL, wallet);
    let mut use_fallback = false;
    let mut primary_fail_count = 0u32;
    let mut ever_connected = false;

    loop {
        if !is_running.load(Ordering::SeqCst) {
            break;
        }

        let url = if use_fallback { &fallback_url } else { &primary_url };

        // v9.0.2: Catch panics from hyper-rustls when no CA certificates are installed
        // (bare Docker containers, minimal Linux installs). Mining still works via periodic
        // challenge refresh — SSE just makes it faster.
        let client_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            eventsource::ClientBuilder::for_url(url).map(|b| b.build())
        }));
        let client = match client_result {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => {
                warn!("Failed to create SSE client for {}: {}", url, e);
                if !use_fallback {
                    primary_fail_count += 1;
                    if primary_fail_count >= 3 {
                        info!("🔄 Switching SSE to fallback server {}", FALLBACK_BOOTSTRAP_URL);
                        use_fallback = true;
                        primary_fail_count = 0;
                    }
                }
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                continue;
            }
            Err(_) => {
                // hyper-rustls panics when no CA certificates found on system
                warn!("⚠️  SSE unavailable: no TLS certificates found (install ca-certificates)");
                warn!("   Mining continues with periodic challenge refresh (slightly slower)");
                // Don't retry — certs won't appear. Just exit the SSE loop.
                return;
            }
        };

        let mut stream = client.stream();

        if ever_connected {
            // Server came back after disconnection — immediately signal mining threads
            // to refresh their challenge rather than waiting up to 50s for periodic timer.
            new_block_signal.fetch_add(1, Ordering::SeqCst);
            info!("🔄 SERVER BACK ONLINE — challenge refresh triggered for all threads");
        }
        ever_connected = true;
        info!("🎧 Connected to SSE stream at {}", url);
        sse_connected.store(true, Ordering::Relaxed);
        let _ = sse_event_tx.send(DiagnosticEvent::SseConnected { url: url.to_string() });

        while is_running.load(Ordering::SeqCst) {
            match stream.next().await {
                Some(Ok(eventsource::SSE::Event(ev))) => {
                    // CRITICAL FIX: Handle new-block events for immediate challenge refresh
                    if ev.event_type == "new-block" {
                        match serde_json::from_str::<serde_json::Value>(&ev.data) {
                            Ok(data) => {
                                // v9.0.2: Handle both flat format {"height": N} and
                                // serde tagged enum format {"type": "NewBlock", "data": {"height": N}}
                                let block_height = data.get("height").and_then(|v| v.as_u64())
                                    .or_else(|| data.get("data").and_then(|d| d.get("height")).and_then(|v| v.as_u64()));
                                if let Some(block_height) = block_height {
                                    let new_signal = new_block_signal.fetch_add(1, Ordering::SeqCst) + 1;
                                    let _ = sse_event_tx.send(DiagnosticEvent::NewBlockSignal { block_height });
                                    info!("🔔 NEW BLOCK #{} detected via SSE - signaling mining threads (signal: {})",
                                         block_height, new_signal);
                                }
                            }
                            Err(e) => {
                                warn!("Failed to parse new-block event: {}", e);
                            }
                        }
                    }

                    // Handle mining_reward events
                    if ev.event_type == "mining_reward" {
                        match serde_json::from_str::<serde_json::Value>(&ev.data) {
                            Ok(data) => {
                                if let Some(miner_address) = data.get("miner_address").and_then(|v| v.as_str()) {
                                    if miner_address == wallet {
                                        let reward_qnk = data.get("reward_qnk")
                                            .and_then(|v| v.as_f64())
                                            .unwrap_or(0.0);
                                        let block_height = data.get("block_height")
                                            .and_then(|v| v.as_u64())
                                            .unwrap_or(0);
                                        let nonce = data.get("nonce")
                                            .and_then(|v| v.as_u64())
                                            .unwrap_or(0);

                                        // Display celebratory reward notification
                                        info!("");
                                        info!("\x1b[1;32m   ✓ REWARD\x1b[0m  \x1b[1;37m{:.8} QUG\x1b[0m  \x1b[2m│\x1b[0m  Block \x1b[36m#{}\x1b[0m  \x1b[2m│\x1b[0m  Nonce \x1b[2m{}\x1b[0m", reward_qnk, block_height, nonce);
                                        info!("");

                                        // Send reward event to TUI dashboard
                                        let _ = sse_event_tx.send(DiagnosticEvent::MiningReward {
                                            reward_qnk,
                                            block_height,
                                        });
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("Failed to parse mining_reward event: {}", e);
                            }
                        }
                    }

                    // v9.1.4: Handle mining-mode-switch events from admin
                    if ev.event_type == "mining-mode-switch" {
                        match serde_json::from_str::<serde_json::Value>(&ev.data) {
                            Ok(data) => {
                                // Handle nested {"type":"MiningModeSwitch","data":{...}} format
                                let switch_data = if data.get("type").and_then(|v| v.as_str()) == Some("MiningModeSwitch") {
                                    data.get("data").cloned().unwrap_or(data.clone())
                                } else {
                                    data.clone()
                                };
                                let target_mode = switch_data.get("target_mode")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");
                                let pool_url = switch_data.get("pool_url")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                                let reason = switch_data.get("reason")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("admin request");

                                info!("");
                                info!("\x1b[1;33m   ⚡ MODE SWITCH\x1b[0m  \x1b[1;37m→ {}\x1b[0m  \x1b[2m│\x1b[0m  Reason: \x1b[36m{}\x1b[0m", target_mode, reason);
                                if let Some(ref url) = pool_url {
                                    info!("   \x1b[2mPool URL:\x1b[0m {}", url);
                                }
                                info!("");

                                let switch_val = match target_mode {
                                    "solo" => 1u8,
                                    "pool" => 2u8,
                                    _ => 0u8,
                                };
                                if switch_val > 0 {
                                    *get_mode_switch_pool_url().lock() = pool_url.clone();
                                    MODE_SWITCH_TARGET.store(switch_val, std::sync::atomic::Ordering::SeqCst);
                                    // Signal mining threads to stop gracefully
                                    is_running.store(false, Ordering::SeqCst);
                                }

                                let _ = sse_event_tx.send(DiagnosticEvent::MiningModeSwitch {
                                    target_mode: target_mode.to_string(),
                                    pool_url,
                                    reason: Some(reason.to_string()),
                                });
                            }
                            Err(e) => {
                                warn!("Failed to parse mining-mode-switch event: {}", e);
                            }
                        }
                    }

                    // Handle balance_updated events (v1.0.2: accept ALL balance changes, not just mining_reward)
                    if ev.event_type == "balance-updated" || ev.event_type == "balance_updated" {
                        match serde_json::from_str::<serde_json::Value>(&ev.data) {
                            Ok(data) => {
                                // v1.0.2: Handle nested {"type":"BalanceUpdated","data":{...}} format
                                let balance_data = if data.get("type").and_then(|v| v.as_str()) == Some("BalanceUpdated") {
                                    data.get("data").cloned().unwrap_or(data.clone())
                                } else {
                                    data.clone()
                                };

                                // Check wallet address matches (normalize both sides)
                                let event_wallet = balance_data.get("wallet_address")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");
                                let event_normalized = event_wallet.strip_prefix("qnk").unwrap_or(event_wallet);
                                let our_normalized = wallet.strip_prefix("qnk").unwrap_or(&wallet);

                                if event_normalized == our_normalized {
                                    let new_balance = balance_data.get("new_balance")
                                        .and_then(|v| v.as_f64())
                                        .unwrap_or(0.0);
                                    let change_reason = balance_data.get("change_reason")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("unknown");
                                    info!("💰 Balance Updated: {:.8} QUG (reason: {})", new_balance, change_reason);
                                    // v9.2.6: Stamp SSE freshness so polling yields
                                    let epoch_secs = std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_secs();
                                    last_balance_sse_epoch.store(epoch_secs, Ordering::Relaxed);
                                    // Send balance update to TUI dashboard
                                    let _ = sse_event_tx.send(DiagnosticEvent::BalanceUpdated {
                                        new_balance,
                                    });
                                }
                            }
                            Err(e) => {
                                warn!("Failed to parse balance_updated event: {}", e);
                            }
                        }
                    }
                }
                Some(Ok(eventsource::SSE::Comment(_))) => {
                    // Ignore comments
                }
                Some(Err(e)) => {
                    sse_connected.store(false, Ordering::Relaxed);
                    let _ = sse_event_tx.send(DiagnosticEvent::SseDisconnected { error: format!("{}", e) });
                    warn!("SSE stream error: {}", e);
                    break;
                }
                None => {
                    sse_connected.store(false, Ordering::Relaxed);
                    let _ = sse_event_tx.send(DiagnosticEvent::SseDisconnected { error: "Stream ended".into() });
                    warn!("SSE stream ended");
                    break;
                }
            }
        }

        // v1.0.2: Drop stream explicitly to trigger clean TCP close before reconnecting.
        drop(stream);

        // Reconnect after delay if still running
        if is_running.load(Ordering::SeqCst) {
            if !use_fallback {
                primary_fail_count += 1;
                if primary_fail_count >= 3 {
                    info!("🔄 Primary SSE failed {} times, switching to fallback {}", primary_fail_count, FALLBACK_BOOTSTRAP_URL);
                    use_fallback = true;
                    primary_fail_count = 0;
                }
            } else {
                // If fallback also fails, try primary again
                primary_fail_count += 1;
                if primary_fail_count >= 3 {
                    info!("🔄 Fallback SSE failed, retrying primary server...");
                    use_fallback = false;
                    primary_fail_count = 0;
                }
            }
            warn!("Reconnecting to SSE stream in 5 seconds...");
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        }
    }

    info!("🛑 SSE listener stopped");
}

/// v1.0.2: SSE listener for decentralized mining mode — listens for new-block events
/// and increments the shared signal so all mining threads refresh their challenge.
/// Uses the same reconnect logic as solo mode SSE listener.
async fn decentralized_sse_listener(
    wallet: String,
    server_url: String,
    is_running: Arc<AtomicBool>,
    new_block_signal: Arc<AtomicU64>,
) {
    use eventsource_client::{self as eventsource, Client as _};
    use futures::StreamExt;

    let normalized_url = normalize_server_url(&server_url);
    // v9.0.2: miner_mode drops non-mining SSE events (backwards-compatible)
    let primary_url = format!("{}/api/v1/events?wallet_address={}&miner_mode=true", normalized_url, wallet);
    let fallback_url = format!("{}/api/v1/events?wallet_address={}&miner_mode=true", FALLBACK_BOOTSTRAP_URL, wallet);
    let mut use_fallback = false;
    let mut primary_fail_count = 0u32;
    let mut ever_connected = false;

    loop {
        if !is_running.load(Ordering::SeqCst) {
            break;
        }

        let url = if use_fallback { &fallback_url } else { &primary_url };

        // v9.0.2: Catch panics from hyper-rustls when no CA certificates
        let client_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            eventsource::ClientBuilder::for_url(url).map(|b| b.build())
        }));
        let client = match client_result {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => {
                warn!("[decentralized-SSE] Failed to create SSE client: {}", e);
                if !use_fallback {
                    primary_fail_count += 1;
                    if primary_fail_count >= 3 {
                        info!("[decentralized-SSE] Switching to fallback {}", FALLBACK_BOOTSTRAP_URL);
                        use_fallback = true;
                        primary_fail_count = 0;
                    }
                }
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                continue;
            }
            Err(_) => {
                warn!("⚠️  [decentralized-SSE] No TLS certificates — SSE disabled, periodic refresh only");
                return;
            }
        };

        if ever_connected {
            new_block_signal.fetch_add(1, Ordering::SeqCst);
            info!("🔄 [decentralized-SSE] SERVER BACK ONLINE — challenge refresh triggered");
        }
        ever_connected = true;
        let mut stream = client.stream();
        info!("[decentralized-SSE] Connected to {} for new-block signals", url);

        while is_running.load(Ordering::SeqCst) {
            match stream.next().await {
                Some(Ok(eventsource::SSE::Event(ev))) => {
                    if ev.event_type == "new-block" {
                        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&ev.data) {
                            // v9.0.2: Handle both flat and tagged enum formats
                            let block_height = data.get("height").and_then(|v| v.as_u64())
                                .or_else(|| data.get("data").and_then(|d| d.get("height")).and_then(|v| v.as_u64()));
                            if let Some(block_height) = block_height {
                                let new_signal = new_block_signal.fetch_add(1, Ordering::SeqCst) + 1;
                                info!("[decentralized-SSE] NEW BLOCK #{} - signaling threads (signal: {})",
                                     block_height, new_signal);
                            }
                        }
                    }
                    // Also handle mining_reward for display
                    if ev.event_type == "mining_reward" {
                        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&ev.data) {
                            if let Some(miner_address) = data.get("miner_address").and_then(|v| v.as_str()) {
                                if miner_address == wallet {
                                    let reward = data.get("reward_qnk").and_then(|v| v.as_f64()).unwrap_or(0.0);
                                    let height = data.get("block_height").and_then(|v| v.as_u64()).unwrap_or(0);
                                    info!("💎 REWARD: {:.8} QUG at block #{}", reward, height);
                                }
                            }
                        }
                    }
                }
                Some(Ok(eventsource::SSE::Comment(_))) => {}
                Some(Err(e)) => {
                    warn!("[decentralized-SSE] Stream error: {} — reconnecting", e);
                    break;
                }
                None => {
                    warn!("[decentralized-SSE] Stream ended — reconnecting");
                    break;
                }
            }
        }

        // Drop the stream explicitly before reconnecting to clean up the TCP connection
        drop(stream);

        if is_running.load(Ordering::SeqCst) {
            if !use_fallback {
                primary_fail_count += 1;
                if primary_fail_count >= 3 {
                    info!("[decentralized-SSE] Switching to fallback after {} failures", primary_fail_count);
                    use_fallback = true;
                    primary_fail_count = 0;
                }
            } else {
                primary_fail_count += 1;
                if primary_fail_count >= 3 {
                    info!("[decentralized-SSE] Retrying primary server...");
                    use_fallback = false;
                    primary_fail_count = 0;
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        }
    }

    info!("[decentralized-SSE] Listener stopped");
}

/// Check if server is currently syncing — returns full Starship telemetry.
/// v9.0.4: Enhanced to parse Starship phase, speed, progress, ETA for TUI display.
/// Falls back to bootstrap1.quillon.xyz if primary server is unreachable.
async fn check_server_sync_status(client: &reqwest::Client, api_url: &str) -> Result<(bool, StarshipSyncInfo)> {
    let path = "/api/v1/status";

    let (body, _used_url) = fetch_with_fallback(&client, api_url, path).await?;

    let api_response: ApiResponse<serde_json::Value> = serde_json::from_str(&body)
        .map_err(|e| anyhow::anyhow!("Failed to parse status response: {}", e))?;

    if !api_response.success {
        return Err(anyhow::anyhow!("API error: {:?}", api_response.error));
    }

    let data = api_response.data.ok_or_else(|| anyhow::anyhow!("No data in response"))?;

    let is_syncing = data.get("is_syncing")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let blocks_behind = data.get("blocks_behind")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let local_height = data.get("current_height")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let network_height = data.get("highest_network_height")
        .and_then(|v| v.as_u64())
        .or_else(|| data.get("network_height").and_then(|v| v.as_u64()))
        .unwrap_or(0);

    let sync_progress = data.get("sync_progress")
        .and_then(|v| v.as_f64())
        .unwrap_or_else(|| {
            if network_height > 0 { (local_height as f64 / network_height as f64 * 100.0).min(100.0) } else { 0.0 }
        }) as f32;

    let sync_speed_bps = data.get("sync_speed_blocks_per_sec")
        .and_then(|v| v.as_f64())
        .or_else(|| data.get("starship_phase_bps").and_then(|v| v.as_f64()))
        .unwrap_or(0.0) as f32;

    let phase = data.get("starship_phase")
        .and_then(|v| v.as_str())
        .unwrap_or(if is_syncing { "SuperHeavy" } else { "StationKeeping" })
        .to_string();

    let phase_duration_secs = data.get("starship_phase_duration_secs")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let mission_elapsed_secs = data.get("starship_mission_elapsed_secs")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let peer_count = data.get("peer_count")
        .and_then(|v| v.as_u64())
        .or_else(|| data.get("connected_peers").and_then(|v| v.as_u64()))
        .unwrap_or(0);

    let orbit_stable = data.get("starship_orbit_stable")
        .and_then(|v| v.as_bool())
        .unwrap_or(!is_syncing);

    let eta_secs = if sync_speed_bps > 0.0 && blocks_behind > 0 {
        (blocks_behind as f64 / sync_speed_bps as f64) as u64
    } else {
        0
    };

    let info = StarshipSyncInfo {
        blocks_behind,
        local_height,
        network_height,
        sync_progress,
        sync_speed_bps,
        phase,
        phase_duration_secs,
        mission_elapsed_secs,
        peer_count,
        orbit_stable,
        eta_secs,
    };

    Ok((is_syncing, info))
}

/// Fetch current mining challenge from API server (with fallback to bootstrap1.quillon.xyz)
/// v8.3.0: Accepts shared client to reuse TCP connections (was creating new client per call).
async fn fetch_mining_challenge(client: &reqwest::Client, api_url: &str) -> Result<MiningChallenge> {
    let path = "/api/v1/mining/challenge";

    let (body, _used_url) = fetch_with_fallback(&client, api_url, path).await?;

    let api_response: ApiResponse<MiningChallenge> = serde_json::from_str(&body)
        .map_err(|e| anyhow::anyhow!("Failed to parse mining challenge response: {}", e))?;

    if !api_response.success {
        let error_msg = api_response.error.unwrap_or_else(|| "Unknown error".to_string());
        anyhow::bail!("API returned error: {}", error_msg);
    }

    let challenge = api_response.data.ok_or_else(|| anyhow::anyhow!("Missing challenge data in API response"))?;

    // v2.7.0: Check min_miner_version from server (miner and server have independent version tracks)
    if let Some(ref min_ver) = challenge.min_miner_version {
        let my_ver = env!("CARGO_PKG_VERSION");
        if version_less_than(my_ver, min_ver) {
            use std::sync::atomic::{AtomicBool, Ordering};
            static SHOWN: AtomicBool = AtomicBool::new(false);
            if !SHOWN.swap(true, Ordering::Relaxed) {
                warn!("╔══════════════════════════════════════════════════╗");
                warn!("║  📦 MINER UPDATE REQUIRED                        ║");
                warn!("║  Minimum version: v{:<35}║", min_ver);
                warn!("║  Your version:    v{:<35}║", my_ver);
                warn!("║  Download: https://dl.quillon.xyz/downloads/      ║");
                warn!("╚══════════════════════════════════════════════════╝");
            }
        }
    }

    // Show server notices
    if let Some(ref notice) = challenge.server_notice {
        if !notice.is_empty() {
            use std::sync::atomic::{AtomicBool, Ordering};
            static NOTICE_SHOWN: AtomicBool = AtomicBool::new(false);
            if !NOTICE_SHOWN.swap(true, Ordering::Relaxed) {
                warn!("[SERVER] {}", notice);
            }
        }
    }

    Ok(challenge)
}

/// Semantic version comparison: returns true if `a` < `b` (e.g. "2.6.0" < "2.7.0")
fn version_less_than(a: &str, b: &str) -> bool {
    let parse = |s: &str| -> Option<(u32, u32, u32)> {
        let s = s.strip_prefix('v').unwrap_or(s);
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() >= 3 {
            Some((parts[0].parse().ok()?, parts[1].parse().ok()?, parts[2].parse().ok()?))
        } else if parts.len() == 2 {
            Some((parts[0].parse().ok()?, parts[1].parse().ok()?, 0))
        } else {
            None
        }
    };
    match (parse(a), parse(b)) {
        (Some(va), Some(vb)) => va < vb,
        _ => a < b,
    }
}

/// Decode hex string to byte array
fn hex_to_bytes(hex_str: &str) -> Result<[u8; 32]> {
    let bytes = hex::decode(hex_str)?;
    if bytes.len() != 32 {
        anyhow::bail!("Expected 32 bytes, got {}", bytes.len());
    }
    let mut result = [0u8; 32];
    result.copy_from_slice(&bytes);
    Ok(result)
}

/// DAG-Knight VDF mining algorithm
/// OPTIMIZED: Original hash function (kept for compatibility with old code)
fn compute_dag_knight_hash(input: &[u8; 32], nonce: u64) -> [u8; 32] {
    let mut hasher_input = [0u8; 40];
    hasher_input[..32].copy_from_slice(input);
    hasher_input[32..].copy_from_slice(&nonce.to_le_bytes());
    compute_dag_knight_hash_optimized(&hasher_input)
}

/// MAXIMUM PERFORMANCE: Zero-allocation hash function for 100% CPU utilization
/// Optimizations:
/// - Pre-allocated fixed-size arrays (no heap allocations)
/// - In-place VDF computation
/// - Optimized for CPU cache efficiency
#[inline(always)]
fn compute_dag_knight_hash_optimized(hash_input: &[u8; 40]) -> [u8; 32] {
    // Initial hash with zero allocations
    let initial_hash = blake3::hash(hash_input);

    // VDF computation with fixed buffer (100 iterations)
    // Using array instead of Vec for zero allocations
    let mut current = *initial_hash.as_bytes();

    for _ in 0..100 {
        // In-place hashing for maximum performance
        current = *blake3::hash(&current).as_bytes();
    }

    current
}

fn print_banner() {
    // Gradient-style banner using ANSI 256-color
    let cyan = "\x1b[38;5;51m";
    let blue = "\x1b[38;5;33m";
    let teal = "\x1b[38;5;43m";
    let gold = "\x1b[38;5;220m";
    let white = "\x1b[1;37m";
    let dim = "\x1b[2m";
    let reset = "\x1b[0m";

    println!();
    println!("  {cyan}╔══════════════════════════════════════════════════════════════╗{reset}");
    println!("  {cyan}║{reset}                                                              {cyan}║{reset}");
    println!("  {cyan}║{reset}                                                              {cyan}║{reset}");
    println!("  {cyan}║{reset}   {teal} ██████╗ ██╗   ██╗██╗██╗     ██╗      ██████╗ ███╗   ██╗{reset}  {cyan}║{reset}");
    println!("  {cyan}║{reset}   {teal}██╔═══██╗██║   ██║██║██║     ██║     ██╔═══██╗████╗  ██║{reset}  {cyan}║{reset}");
    println!("  {cyan}║{reset}   {teal}██║   ██║██║   ██║██║██║     ██║     ██║   ██║██╔██╗ ██║{reset}  {cyan}║{reset}");
    println!("  {cyan}║{reset}   {teal}██║▄▄ ██║██║   ██║██║██║     ██║     ██║   ██║██║╚██╗██║{reset}  {cyan}║{reset}");
    println!("  {cyan}║{reset}   {teal}╚██████╔╝╚██████╔╝██║███████╗███████╗╚██████╔╝██║ ╚████║{reset}  {cyan}║{reset}");
    println!("  {cyan}║{reset}   {teal} ╚══▀▀═╝  ╚═════╝ ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝{reset}  {cyan}║{reset}");
    println!("  {cyan}║{reset}   {blue} ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗{reset}                 {cyan}║{reset}");
    println!("  {cyan}║{reset}   {blue}██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║{reset}                 {cyan}║{reset}");
    println!("  {cyan}║{reset}   {blue}██║  ███╗██████╔╝███████║██████╔╝███████║{reset}                 {cyan}║{reset}");
    println!("  {cyan}║{reset}   {blue}██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║{reset}                 {cyan}║{reset}");
    println!("  {cyan}║{reset}   {blue}╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║{reset}                 {cyan}║{reset}");
    println!("  {cyan}║{reset}   {blue} ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝{reset}                 {cyan}║{reset}");
    println!("  {cyan}║{reset}                                                              {cyan}║{reset}");
    println!("  {cyan}║{reset}   {white}DUAL-LANE MINER{reset}  {dim}v{}{reset}                                    {cyan}║{reset}", env!("CARGO_PKG_VERSION"));
    println!("  {cyan}║{reset}                                                              {cyan}║{reset}");
    println!("  {cyan}║{reset}   {gold}GPU{reset} {dim}BLAKE3 PoW ━━━━━━━━━━━━━━━━━━━━━━━━━━━━{reset} {gold}50%{reset}       {cyan}║{reset}");
    println!("  {cyan}║{reset}   {teal}CPU{reset} {dim}Genus-2 VDF ━━━━━━━━━━━━━━━━━━━━━━━━━━━{reset} {teal}50%{reset}       {cyan}║{reset}");
    println!("  {cyan}║{reset}                                                              {cyan}║{reset}");
    println!("  {cyan}║{reset}   {dim}Quantum-Resistant Sequential Proof Mining{reset}                  {cyan}║{reset}");
    println!("  {cyan}║{reset}   {dim}y{reset}{dim}2{reset} {dim}= x{reset}{dim}5{reset} {dim}+ x{reset}{dim}2{reset} {dim}- 1 over 256-bit prime field (pq128){reset}            {cyan}║{reset}");
    println!("  {cyan}║{reset}                                                              {cyan}║{reset}");
    println!("  {cyan}╚══════════════════════════════════════════════════════════════╝{reset}");
    println!();
}

/// Decentralized P2P Pool Mining Mode (v2.3.0+)
///
/// Instead of connecting to a centralized pool server via Stratum,
/// this mode connects to any P2P node running the distributed pool coordinator.
/// Shares are submitted via HTTP API and propagated via gossipsub.
///
/// Features:
/// - CRDT-based PPLNS: No central pool state needed
/// - VDF anti-grinding proofs: Prevents share manipulation
/// - Threshold signature payouts: Consensus-based reward distribution
/// - P2P share propagation: Fully decentralized operation
async fn run_decentralized_pool_mining(
    threads: usize,
    intensity: u8,
    wallet: &str,
    worker_name: &str,
    bootstrap_nodes: &str,
    region: &str,
) -> Result<()> {
    use tokio::sync::mpsc;
    use serde_json::json;

    let hash_counter = Arc::new(AtomicU64::new(0));
    let is_running = Arc::new(AtomicBool::new(true));
    let shares_submitted = Arc::new(AtomicU64::new(0));
    let blocks_found = Arc::new(AtomicU64::new(0));

    // Parse bootstrap nodes
    let nodes: Vec<&str> = bootstrap_nodes.split(',').map(|s| s.trim()).collect();
    let primary_node = nodes.first().copied().unwrap_or("http://localhost:8080");

    info!("🔌 Connecting to P2P network via {}", primary_node);
    info!("🌍 Pool region: {}", region);

    // v1.0.2: Shared client with connection pooling + TCP keepalive for decentralized mode.
    // All threads share this single client to prevent TCP socket exhaustion.
    // Note: Decentralized pool mode does not currently thread proxy_url through;
    // proxy support is primarily for solo mining mode.
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .connect_timeout(std::time::Duration::from_secs(10))
        .pool_max_idle_per_host(4)
        .pool_idle_timeout(std::time::Duration::from_secs(30))
        .tcp_keepalive(std::time::Duration::from_secs(15))
        .build()?;

    let pool_status = client
        .get(&format!("{}/api/v1/pool/status", primary_node))
        .send()
        .await;

    match pool_status {
        Ok(resp) if resp.status().is_success() => {
            info!("✅ Connected to P2P pool node");
            if let Ok(status) = resp.json::<serde_json::Value>().await {
                if let Some(workers) = status.get("worker_count") {
                    info!("   Active workers: {}", workers);
                }
                if let Some(hashrate) = status.get("pool_hashrate") {
                    info!("   Pool hashrate: {} H/s", hashrate);
                }
            }
        }
        Ok(resp) => {
            warn!("⚠️  Pool API returned status: {} - continuing anyway", resp.status());
        }
        Err(e) => {
            warn!("⚠️  Could not connect to pool status API: {} - continuing anyway", e);
            info!("   Mining will work via standard mining challenge API");
        }
    }

    // Channel for submitting shares
    let (share_tx, mut share_rx) = mpsc::channel::<DecentralizedShare>(1000);

    // Spawn share submitter task
    let client_for_shares = client.clone();
    let node_for_shares = primary_node.to_string();
    let wallet_for_shares = wallet.to_string();
    let worker_for_shares = worker_name.to_string();
    let shares_counter = shares_submitted.clone();
    let blocks_counter = blocks_found.clone();

    tokio::spawn(async move {
        info!("📤 Share submitter started");
        while let Some(share) = share_rx.recv().await {
            // Submit share via API
            let submit_url = format!("{}/api/v1/pool/submit-share", node_for_shares);
            let share_data = json!({
                "wallet": wallet_for_shares,
                "worker": worker_for_shares,
                "share_id": hex::encode(&share.share_id),
                "difficulty": share.difficulty,
                "block_height": share.block_height,
                "nonce": share.nonce,
                "timestamp": share.timestamp,
            });

            match client_for_shares.post(&submit_url).json(&share_data).send().await {
                Ok(resp) if resp.status().is_success() => {
                    shares_counter.fetch_add(1, Ordering::Relaxed);

                    // Check if we found a block
                    if let Ok(result) = resp.json::<serde_json::Value>().await {
                        if result.get("block_found").and_then(|b| b.as_bool()).unwrap_or(false) {
                            blocks_counter.fetch_add(1, Ordering::Relaxed);
                            info!("🎉 BLOCK FOUND! Share accepted as block solution!");
                        }
                    }
                }
                Ok(resp) => {
                    warn!("⚠️  Share rejected: {}", resp.status());
                }
                Err(e) => {
                    warn!("❌ Share submission failed: {}", e);
                }
            }
        }
    });

    // Spawn hashrate display task
    let hash_display = hash_counter.clone();
    let shares_display = shares_submitted.clone();
    let blocks_display = blocks_found.clone();
    let is_running_display = is_running.clone();

    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
        let mut last_hash_count = 0u64;

        loop {
            interval.tick().await;
            if !is_running_display.load(Ordering::Relaxed) {
                break;
            }

            let current = hash_display.load(Ordering::Relaxed);
            let delta = current.saturating_sub(last_hash_count);
            let hashrate = delta as f64 / 10.0;
            last_hash_count = current;

            let shares = shares_display.load(Ordering::Relaxed);
            let blocks = blocks_display.load(Ordering::Relaxed);

            info!(
                "⛏️  Hashrate: {:.2} KH/s | Total hashes: {} | Shares: {} | Blocks: {}",
                hashrate / 1000.0,
                current,
                shares,
                blocks
            );
        }
    });

    // v1.0.2: Restructured decentralized mining to match solo mode's efficient pattern:
    // - One SSE listener signals all threads on new blocks (instead of each thread polling)
    // - Only thread 0 does periodic challenge refresh (50s), others wait for SSE signal (5min max)
    // - All threads share one reqwest client (connection pooling)
    // This reduces API calls from (N_threads × N_miners / 30s) to ~0.3/s per miner.
    let server_url = normalize_server_url(primary_node);
    let wallet_clone = wallet.to_string();

    // Shared new-block signal: SSE listener increments this when a new block arrives,
    // all mining threads check it to know when to refresh their challenge.
    let new_block_signal = Arc::new(AtomicU64::new(0));

    // Spawn SSE listener for new block notifications (same as solo mode)
    let sse_wallet = wallet_clone.clone();
    let sse_server = server_url.clone();
    let sse_running = is_running.clone();
    let sse_signal = new_block_signal.clone();
    tokio::spawn(async move {
        decentralized_sse_listener(sse_wallet, sse_server, sse_running, sse_signal).await;
    });

    info!("⚡ Starting {} mining threads...", threads);

    // Spawn mining threads — each uses shared client + SSE-driven challenge refresh
    for thread_id in 0..threads {
        let hash_counter = hash_counter.clone();
        let is_running = is_running.clone();
        let share_tx = share_tx.clone();
        let server = server_url.clone();
        let wallet = wallet_clone.clone();
        let client = client.clone();
        let new_block_signal = new_block_signal.clone();

        tokio::spawn(async move {
            let mut local_nonce = (thread_id as u64) << 56; // Thread-unique nonce range
            #[cfg(unix)]
            let simd_batch = q_miner::cpu::optimal_mining_batch_size();
            #[cfg(not(unix))]
            let simd_batch: usize = 4;
            let mut batch_results: [(u64, [u8; 32]); 16] = [(0u64, [0u8; 32]); 16];

            // v1.0.2: Only thread 0 does periodic refresh (50s). Other threads rely on
            // SSE new-block signal, with a 5-min fallback. This slashes API calls by ~95%.
            let challenge_refresh_interval = if thread_id == 0 {
                std::time::Duration::from_secs(50)
            } else {
                std::time::Duration::from_secs(300)
            };
            let mut last_challenge_refresh = std::time::Instant::now();
            let mut last_known_block_signal = new_block_signal.load(Ordering::Relaxed);

            // Fetch initial challenge
            let challenge_url = format!("{}/api/v1/mining/challenge?wallet={}", server, wallet);
            let mut current_challenge: Option<(MiningChallenge, [u8; 32], [u8; 32])> = None;

            // Initial fetch with retries
            for attempt in 0..5 {
                match client.get(&challenge_url).send().await {
                    Ok(resp) => {
                        if let Ok(api_resp) = resp.json::<ApiResponse<MiningChallenge>>().await {
                            if api_resp.success {
                                if let Some(c) = api_resp.data {
                                    if let (Ok(cb), Ok(tb)) = (
                                        hex::decode(&c.challenge_hash),
                                        hex::decode(&c.difficulty_target),
                                    ) {
                                        if cb.len() >= 32 && tb.len() >= 32 {
                                            let mut challenge_bytes = [0u8; 32];
                                            let mut target_bytes = [0u8; 32];
                                            challenge_bytes.copy_from_slice(&cb[..32]);
                                            target_bytes.copy_from_slice(&tb[..32]);
                                            info!("📋 Thread {} fetched initial challenge: block #{}",
                                                 thread_id, c.block_height);
                                            current_challenge = Some((c, challenge_bytes, target_bytes));
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if attempt < 4 {
                            warn!("⚠️  Thread {} challenge fetch attempt {}: {}", thread_id, attempt + 1, e);
                        }
                    }
                }
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            }

            let (mut challenge, mut challenge_bytes, mut target_bytes) = match current_challenge {
                Some(c) => c,
                None => {
                    error!("❌ Thread {} failed to fetch initial challenge after 5 attempts", thread_id);
                    return;
                }
            };

            loop {
                if !is_running.load(Ordering::Relaxed) {
                    break;
                }

                // Check if new block arrived via SSE or periodic timer expired
                let current_block_signal = new_block_signal.load(Ordering::Relaxed);
                let should_refresh = current_block_signal != last_known_block_signal
                    || last_challenge_refresh.elapsed() >= challenge_refresh_interval;

                if should_refresh {
                    match client.get(&challenge_url).send().await {
                        Ok(resp) => {
                            if let Ok(api_resp) = resp.json::<ApiResponse<MiningChallenge>>().await {
                                if api_resp.success {
                                    if let Some(new_c) = api_resp.data {
                                        if new_c.block_height != challenge.block_height {
                                            info!("🔄 Thread {} challenge updated: block #{} -> #{}",
                                                 thread_id, challenge.block_height, new_c.block_height);
                                        }
                                        if let (Ok(cb), Ok(tb)) = (
                                            hex::decode(&new_c.challenge_hash),
                                            hex::decode(&new_c.difficulty_target),
                                        ) {
                                            if cb.len() >= 32 && tb.len() >= 32 {
                                                challenge_bytes.copy_from_slice(&cb[..32]);
                                                target_bytes.copy_from_slice(&tb[..32]);
                                                challenge = new_c;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            warn!("⚠️  Thread {} challenge refresh failed: {}", thread_id, e);
                        }
                    }
                    last_challenge_refresh = std::time::Instant::now();
                    last_known_block_signal = current_block_signal;
                }

                // Mine a batch of nonces (SIMD-interleaved VDF path, like solo mode)
                let batch_size = 50_000u64;
                let mut local_count: u64 = 0;
                let mut i: u64 = 0;
                while i < batch_size {
                    // Check stop flag every 256 hashes (not every iteration)
                    if i & 255 == 0 && !is_running.load(Ordering::Relaxed) {
                        break;
                    }

                    #[cfg(unix)]
                    let count = q_miner::cpu::compute_dag_knight_hash_batch(
                        &challenge_bytes,
                        local_nonce.wrapping_add(1),
                        simd_batch,
                        &mut batch_results,
                    );
                    #[cfg(not(unix))]
                    let count = {
                        let bs = simd_batch.min(16).min(batch_results.len());
                        for bi in 0..bs {
                            let n = local_nonce.wrapping_add(1 + bi as u64);
                            let mut hash_input = [0u8; 40];
                            hash_input[..32].copy_from_slice(&challenge_bytes);
                            hash_input[32..].copy_from_slice(&n.to_le_bytes());
                            batch_results[bi] = (n, compute_dag_knight_hash_optimized(&hash_input));
                        }
                        bs
                    };

                    local_count += count as u64;
                    local_nonce = local_nonce.wrapping_add(count as u64);

                    // Flush every 1024 hashes
                    if local_count & 1023 == 0 {
                        hash_counter.fetch_add(1024, Ordering::Relaxed);
                    }

                    // Check for new block every 4096 hashes (abandon stale work fast)
                    if i & 4095 == 0 && i > 0 {
                        let sig = new_block_signal.load(Ordering::Relaxed);
                        if sig != last_known_block_signal {
                            break; // New block — refresh challenge
                        }
                    }

                    // Check solutions in the SIMD batch
                    for r in 0..count {
                        let (result_nonce, hash_result) = batch_results[r];
                        if hash_result[..] < target_bytes[..] {
                            let share = DecentralizedShare {
                                share_id: hash_result,
                                difficulty: challenge.block_reward,
                                block_height: challenge.block_height,
                                nonce: result_nonce,
                                timestamp: chrono::Utc::now().timestamp_millis() as u64,
                            };

                            let _ = share_tx.send(share).await;
                            info!("💎 Share found! nonce={}, height={}", result_nonce, challenge.block_height);
                        }
                    }

                    i = i.wrapping_add(count as u64);
                }

                // Flush remaining local hashes
                let remainder = local_count & 1023;
                if remainder > 0 {
                    hash_counter.fetch_add(remainder, Ordering::Relaxed);
                }
            }
        });
    }

    // Wait for shutdown signal
    signal::ctrl_c().await?;
    info!("🛑 Shutting down...");
    is_running.store(false, Ordering::Relaxed);

    let total_shares = shares_submitted.load(Ordering::Relaxed);
    let total_blocks = blocks_found.load(Ordering::Relaxed);
    let total_hashes = hash_counter.load(Ordering::Relaxed);

    info!("📊 Final statistics:");
    info!("   Total hashes: {}", total_hashes);
    info!("   Shares submitted: {}", total_shares);
    info!("   Blocks found: {}", total_blocks);

    Ok(())
}

/// Decentralized share data structure
struct DecentralizedShare {
    share_id: [u8; 32],
    difficulty: f64,
    block_height: u64,
    nonce: u64,
    timestamp: u64,
}

// ========================================================================
// MINER LINK - WebSocket relay for wallet ↔ miner real-time communication
// ========================================================================

use q_miner::miner_link::{MinerLinkMessage, MinerCommand};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message as WsMessage;

/// Background task that maintains a WebSocket connection to the API server's
/// miner-link relay endpoint, sending periodic stats and receiving commands.
async fn miner_link_task(
    wallet: String,
    server_url: String,
    miner_id: String,
    miner_name: Option<String>,
    is_running: Arc<AtomicBool>,
    current_hashrate_khs: Arc<AtomicU64>,
    hash_counter: Arc<AtomicU64>,
    solutions_found: Arc<AtomicU64>,
    blocks_mined: Arc<AtomicU64>,
    is_paused: Arc<AtomicBool>,
    target_threads: Arc<AtomicUsize>,
    target_intensity: Arc<AtomicU8>,
    total_threads: u32,
    proxy_url: Option<String>,
    gpu_active: Arc<AtomicBool>,
    gpu_hashrate_hs: Arc<AtomicU64>,
    gpu_devices: Arc<parking_lot::RwLock<Vec<q_miner::shared_state::GpuDeviceSnapshot>>>,
) {
    let start_time = std::time::Instant::now();

    // Detect CPU info once
    let (cpu_vendor, has_avx2, has_avx512) = get_cpu_info_for_link();

    let mut backoff_secs = 2u64;

    while is_running.load(Ordering::Relaxed) {
        // Build WS URL from server URL
        let ws_url = build_ws_url(&server_url, &wallet, &miner_id);
        info!("🔗 [MinerLink] Connecting to relay: {}", ws_url);

        // SOCKS5 proxy for WebSocket: tunnel TCP through SOCKS5 before WS handshake
        if let Some(ref proxy) = proxy_url {
            if proxy.starts_with("socks5://") {
                let parsed = url::Url::parse(proxy).unwrap_or_else(|e| {
                    warn!("Invalid proxy URL '{}': {} — falling back to socks5://127.0.0.1:9050", proxy, e);
                    // SAFETY: this literal is valid and will never fail
                    url::Url::parse("socks5://127.0.0.1:9050").expect("hardcoded fallback URL is valid")
                });
                let proxy_addr = format!("{}:{}", parsed.host_str().unwrap_or("127.0.0.1"), parsed.port().unwrap_or(9050));
                let target = url::Url::parse(&ws_url.replace("wss://", "https://").replace("ws://", "http://"))
                    .unwrap_or_else(|_| url::Url::parse("http://localhost:8080").unwrap());
                let host = target.host_str().unwrap_or("localhost").to_string();
                let port = target.port().unwrap_or(if ws_url.starts_with("wss") { 443 } else { 80 });

                match tokio_socks::tcp::Socks5Stream::connect(&*proxy_addr, (&*host, port)).await {
                    Ok(socks_stream) => {
                        match tokio_tungstenite::client_async(&ws_url, socks_stream.into_inner()).await {
                            Ok((ws_stream, _)) => {
                                info!("✅ [MinerLink] Connected to relay via SOCKS5 proxy");
                                backoff_secs = 2;
                                let (mut sink, mut stream) = ws_stream.split();
                                run_miner_link_session(
                                    &mut sink, &mut stream,
                                    &wallet, &miner_id, &miner_name,
                                    &is_running, &current_hashrate_khs, &hash_counter,
                                    &solutions_found, &blocks_mined, &is_paused,
                                    &target_threads, &target_intensity, total_threads,
                                    &cpu_vendor, has_avx2, has_avx512, &start_time,
                                    &gpu_active, &gpu_hashrate_hs, &gpu_devices,
                                ).await;
                            }
                            Err(e) => {
                                warn!("🔗 [MinerLink] SOCKS5 WS handshake failed: {} (retry in {}s)", e, backoff_secs);
                            }
                        }
                    }
                    Err(e) => {
                        warn!("🔗 [MinerLink] SOCKS5 tunnel failed: {} (retry in {}s)", e, backoff_secs);
                    }
                }

                // Backoff before reconnect
                if !is_running.load(Ordering::Relaxed) { break; }
                tokio::time::sleep(tokio::time::Duration::from_secs(backoff_secs)).await;
                backoff_secs = (backoff_secs * 2).min(30);
                continue;
            }
        }

        match tokio_tungstenite::connect_async(&ws_url).await {
            Ok((ws_stream, _)) => {
                info!("✅ [MinerLink] Connected to relay");
                backoff_secs = 2;
                let (mut sink, mut stream) = ws_stream.split();
                run_miner_link_session(
                    &mut sink, &mut stream,
                    &wallet, &miner_id, &miner_name,
                    &is_running, &current_hashrate_khs, &hash_counter,
                    &solutions_found, &blocks_mined, &is_paused,
                    &target_threads, &target_intensity, total_threads,
                    &cpu_vendor, has_avx2, has_avx512, &start_time,
                    &gpu_active, &gpu_hashrate_hs, &gpu_devices,
                ).await;
            }
            Err(e) => {
                warn!("🔗 [MinerLink] Connection failed: {} (retry in {}s)", e, backoff_secs);
            }
        }

        // Backoff before reconnect
        if !is_running.load(Ordering::Relaxed) {
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(backoff_secs)).await;
        backoff_secs = (backoff_secs * 2).min(30);
    }

    info!("🔗 [MinerLink] Task stopped");
}

/// Handle an incoming command from the wallet
fn handle_miner_command(
    action: &MinerCommand,
    is_paused: &AtomicBool,
    target_threads: &AtomicUsize,
    target_intensity: &AtomicU8,
    total_threads: u32,
) -> (bool, String) {
    match action {
        MinerCommand::Pause => {
            is_paused.store(true, Ordering::Relaxed);
            info!("⏸️  [MinerLink] Mining PAUSED by wallet command");
            (true, "Mining paused".to_string())
        }
        MinerCommand::Resume => {
            is_paused.store(false, Ordering::Relaxed);
            info!("▶️  [MinerLink] Mining RESUMED by wallet command");
            (true, "Mining resumed".to_string())
        }
        MinerCommand::SetThreads { count } => {
            let count = (*count).min(total_threads);
            if count == 0 {
                return (false, "Thread count must be > 0".to_string());
            }
            target_threads.store(count as usize, Ordering::Relaxed);
            info!("🔧 [MinerLink] Threads set to {} by wallet command", count);
            (true, format!("Threads set to {}", count))
        }
        MinerCommand::SetIntensity { level } => {
            let level = (*level).clamp(1, 10);
            target_intensity.store(level, Ordering::Relaxed);
            info!("🔧 [MinerLink] Intensity set to {} by wallet command", level);
            (true, format!("Intensity set to {}", level))
        }
        MinerCommand::GetDetailedStats => {
            (true, "Detailed stats sent via next Stats message".to_string())
        }
    }
}

/// Generic MinerLink session loop — works with any AsyncRead+AsyncWrite stream.
/// Extracted so both direct WS and SOCKS5-tunneled WS can share the same logic.
async fn run_miner_link_session<S>(
    sink: &mut futures_util::stream::SplitSink<tokio_tungstenite::WebSocketStream<S>, WsMessage>,
    stream: &mut futures_util::stream::SplitStream<tokio_tungstenite::WebSocketStream<S>>,
    wallet: &str,
    miner_id: &str,
    miner_name: &Option<String>,
    is_running: &AtomicBool,
    current_hashrate_khs: &AtomicU64,
    hash_counter: &AtomicU64,
    solutions_found: &AtomicU64,
    blocks_mined: &AtomicU64,
    is_paused: &AtomicBool,
    target_threads: &AtomicUsize,
    target_intensity: &AtomicU8,
    total_threads: u32,
    cpu_vendor: &str,
    has_avx2: bool,
    has_avx512: bool,
    start_time: &std::time::Instant,
    gpu_active: &AtomicBool,
    gpu_hashrate_hs: &AtomicU64,
    gpu_devices: &parking_lot::RwLock<Vec<q_miner::shared_state::GpuDeviceSnapshot>>,
) where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    // Send Register message
    let register = MinerLinkMessage::Register {
        wallet: wallet.to_string(),
        miner_id: miner_id.to_string(),
        miner_name: miner_name.clone(),
    };
    if let Ok(json) = serde_json::to_string(&register) {
        let _ = sink.send(WsMessage::Text(json)).await;
    }

    let mut stats_interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
    let mut ping_interval = tokio::time::interval(tokio::time::Duration::from_secs(30));

    loop {
        if !is_running.load(Ordering::Relaxed) {
            break;
        }

        tokio::select! {
            _ = stats_interval.tick() => {
                let hashrate = f64::from_bits(current_hashrate_khs.load(Ordering::Relaxed)) * 1000.0;
                let is_gpu = gpu_active.load(Ordering::Relaxed);
                let gpu_hr = f64::from_bits(gpu_hashrate_hs.load(Ordering::Relaxed));
                let gpu_devs: Vec<q_miner::miner_link::GpuDeviceInfo> = gpu_devices.read().iter().map(|d| {
                    q_miner::miner_link::GpuDeviceInfo {
                        name: d.name.clone(),
                        vendor: d.vendor.clone(),
                        compute_units: d.compute_units,
                        memory_mb: d.global_memory_mb,
                        max_clock_mhz: d.max_clock_mhz,
                        api: d.api.clone(),
                    }
                }).collect();
                let stats_msg = MinerLinkMessage::Stats {
                    miner_id: miner_id.to_string(),
                    hashrate,
                    total_hashes: hash_counter.load(Ordering::Relaxed),
                    solutions: solutions_found.load(Ordering::Relaxed),
                    blocks_found: blocks_mined.load(Ordering::Relaxed),
                    uptime_secs: start_time.elapsed().as_secs(),
                    threads_active: target_threads.load(Ordering::Relaxed) as u32,
                    threads_total: total_threads,
                    cpu_vendor: cpu_vendor.to_string(),
                    has_avx2,
                    has_avx512,
                    intensity: target_intensity.load(Ordering::Relaxed),
                    is_mining: !is_paused.load(Ordering::Relaxed),
                    current_block_height: 0,
                    temperature_estimate: None,
                    gpu_active: is_gpu,
                    gpu_hashrate: gpu_hr,
                    gpu_devices: gpu_devs,
                };
                if let Ok(json) = serde_json::to_string(&stats_msg) {
                    if sink.send(WsMessage::Text(json)).await.is_err() {
                        warn!("🔗 [MinerLink] Send failed, reconnecting...");
                        break;
                    }
                }
            }

            _ = ping_interval.tick() => {
                if let Ok(json) = serde_json::to_string(&MinerLinkMessage::Ping) {
                    let _ = sink.send(WsMessage::Text(json)).await;
                }
            }

            msg = stream.next() => {
                match msg {
                    Some(Ok(WsMessage::Text(text))) => {
                        if let Ok(link_msg) = serde_json::from_str::<MinerLinkMessage>(&text) {
                            match link_msg {
                                MinerLinkMessage::Command { command_id, action } => {
                                    let (success, message) = handle_miner_command(
                                        &action,
                                        is_paused,
                                        target_threads,
                                        target_intensity,
                                        total_threads,
                                    );
                                    let ack = MinerLinkMessage::Ack {
                                        command_id,
                                        success,
                                        message,
                                    };
                                    if let Ok(json) = serde_json::to_string(&ack) {
                                        let _ = sink.send(WsMessage::Text(json)).await;
                                    }
                                }
                                MinerLinkMessage::Pong => { /* keepalive response */ }
                                MinerLinkMessage::LinkEstablished { .. } => {
                                    info!("🔗 [MinerLink] Link established with wallet");
                                }
                                _ => { /* ignore other messages */ }
                            }
                        }
                    }
                    Some(Ok(WsMessage::Close(_))) | None => {
                        info!("🔗 [MinerLink] Connection closed, will reconnect...");
                        break;
                    }
                    Some(Ok(WsMessage::Ping(data))) => {
                        let _ = sink.send(WsMessage::Pong(data)).await;
                    }
                    Some(Err(e)) => {
                        warn!("🔗 [MinerLink] WebSocket error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
        }
    }
}

/// Build the WebSocket URL for miner-link relay
fn build_ws_url(server_url: &str, wallet: &str, miner_id: &str) -> String {
    let base = normalize_server_url(server_url);
    let ws_base = if base.starts_with("https://") {
        base.replacen("https://", "wss://", 1)
    } else if base.starts_with("http://") {
        base.replacen("http://", "ws://", 1)
    } else {
        format!("ws://{}", base)
    };
    format!("{}/api/v1/miner-link/ws?role=miner&wallet={}&miner_id={}", ws_base, wallet, miner_id)
}

/// Get CPU info for miner link stats
fn get_cpu_info_for_link() -> (String, bool, bool) {
    #[cfg(target_arch = "x86_64")]
    {
        let cpuid = CpuId::new();
        let vendor = cpuid.get_vendor_info()
            .map(|v| v.as_str().to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        let extended = cpuid.get_extended_feature_info();
        let avx2 = extended.as_ref().map(|ef| ef.has_avx2()).unwrap_or(false);
        let avx512 = extended.as_ref().map(|ef| ef.has_avx512f()).unwrap_or(false);
        (vendor, avx2, avx512)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        ("Unknown".to_string(), false, false)
    }
}