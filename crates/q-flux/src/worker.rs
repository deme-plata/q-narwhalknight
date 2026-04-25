use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use dashmap::DashMap;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, Semaphore};
use tokio_rustls::TlsAcceptor;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use futures::future::select_all;

use crate::access_control::AccessControl;
use crate::access_log::AccessLogger;
use crate::acceptor::SharedTlsConfig;
use crate::acme::ChallengeStore;
use crate::config::{FluxConfig, StaticConfig, UpstreamConfig};
use crate::h2_proxy;
use crate::health::HealthMap;
use crate::libp2p_aware::{BandwidthLimiter, PeerTracker};
use crate::metrics::{Metrics, RateLimiter};
use crate::proxy::{self, DrainReceiver};
use crate::static_serve::FileCache;
use crate::upstream::UpstreamPool;
use crate::acceptor;

/// Per-IP connection counter shared across all workers.
pub type IpConnTracker = Arc<DashMap<std::net::IpAddr, u64>>;

/// Total active connections (shared).
pub type ActiveConnCount = Arc<AtomicU64>;

/// Max concurrent connection handlers per worker.
/// 48 workers × 1024 = 49,152 total concurrent handlers.
/// Each handler holds a reference to the shared upstream pool + ~8KB buffer.
/// At 49K concurrent × ~10KB = ~480MB. Handles 10K+ SSE connections + burst downloads.
const MAX_HANDLERS_PER_WORKER: usize = 1024;

/// How often to garbage-collect the per-IP connection tracker (seconds).
/// IPs with 0 active connections are removed to prevent unbounded growth.
const IP_TRACKER_GC_INTERVAL_SECS: u64 = 60;

/// How often to clean up stale rate limiter buckets (seconds).
const RATE_LIMITER_GC_INTERVAL_SECS: u64 = 300;

/// Stream wrapper that yields pre-read bytes before delegating to the inner stream.
/// Used when we peek at the first bytes of a TLS stream to detect WebSocket upgrades
/// on port 443, then need to pass those bytes along with the stream to the HTTP proxy.
struct PrefixedStream<S> {
    prefix: Vec<u8>,
    offset: usize,
    inner: S,
}

impl<S> PrefixedStream<S> {
    fn new(prefix: Vec<u8>, inner: S) -> Self {
        Self { prefix, offset: 0, inner }
    }
}

impl<S: tokio::io::AsyncRead + Unpin> tokio::io::AsyncRead for PrefixedStream<S> {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        if self.offset < self.prefix.len() {
            let remaining = &self.prefix[self.offset..];
            let to_copy = remaining.len().min(buf.remaining());
            buf.put_slice(&remaining[..to_copy]);
            self.offset += to_copy;
            return std::task::Poll::Ready(Ok(()));
        }
        std::pin::Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl<S: tokio::io::AsyncWrite + Unpin> tokio::io::AsyncWrite for PrefixedStream<S> {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<Result<usize, std::io::Error>> {
        std::pin::Pin::new(&mut self.inner).poll_write(cx, buf)
    }
    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        std::pin::Pin::new(&mut self.inner).poll_flush(cx)
    }
    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        std::pin::Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

/// Spawn worker threads. Each worker:
/// - Has its own TcpListener (SO_REUSEPORT gives it a fair share of connections)
/// - Has its own upstream connection pool (no cross-thread contention)
/// - Runs on a dedicated tokio single-threaded runtime pinned to a core
/// - Has a semaphore limiting concurrent connection handlers (prevents OOM)
#[allow(clippy::too_many_arguments)]
pub fn spawn_workers(
    config: &FluxConfig,
    shared_tls: SharedTlsConfig,
    metrics: Metrics,
    shutdown_tx: &tokio::sync::broadcast::Sender<()>,
    shutdown_flag: Arc<AtomicBool>,
    health_map: HealthMap,
    access_logger: Option<AccessLogger>,
    rate_limiter: Option<Arc<RateLimiter>>,
    peer_tracker: Arc<PeerTracker>,
    drain_rx: DrainReceiver,
    access_control: Arc<AccessControl>,
    challenge_store: ChallengeStore,
    global_backend_counters: Arc<dashmap::DashMap<String, Arc<crate::upstream::BackendCounters>>>,
) -> Vec<std::thread::JoinHandle<()>> {
    let worker_count = config.worker_count();
    let ip_tracker: IpConnTracker = Arc::new(DashMap::new());
    let active_conns: ActiveConnCount = Arc::new(AtomicU64::new(0));
    let mut handles = Vec::with_capacity(worker_count);

    // Global upstream semaphore: shared across ALL workers to precisely cap
    // total concurrent backend connections. This prevents the death spiral where
    // 48 workers × N permits each overwhelm a single backend.
    let global_upstream_semaphore = if config.upstream.max_upstream_global > 0 {
        let limit = config.upstream.max_upstream_global;
        tracing::info!(
            limit,
            workers = worker_count,
            "Global upstream semaphore: {} max concurrent backend requests (shared across {} workers)",
            limit, worker_count,
        );
        Some(Arc::new(Semaphore::new(limit)))
    } else {
        tracing::info!(
            per_worker = config.upstream.max_inflight_per_worker,
            workers = worker_count,
            total = config.upstream.max_inflight_per_worker * worker_count,
            "Per-worker upstream semaphore: {} × {} = {} total",
            config.upstream.max_inflight_per_worker, worker_count,
            config.upstream.max_inflight_per_worker * worker_count,
        );
        None
    };

    for worker_id in 0..worker_count {
        let config = config.clone();
        let shared_tls = shared_tls.clone();
        let metrics = metrics.clone();
        let ip_tracker = ip_tracker.clone();
        let active_conns = active_conns.clone();
        let shutdown_rx = shutdown_tx.subscribe();
        let shutdown_flag = shutdown_flag.clone();
        let health_map = health_map.clone();
        let access_logger = access_logger.clone();
        let rate_limiter = rate_limiter.clone();
        let peer_tracker = peer_tracker.clone();
        let global_sem = global_upstream_semaphore.clone();
        let drain_rx = drain_rx.clone();
        let access_control = access_control.clone();
        let challenge_store = challenge_store.clone();
        let global_backend_counters = global_backend_counters.clone();

        let handle = std::thread::Builder::new()
            .name(format!("q-flux-w{}", worker_id))
            .spawn(move || {
                pin_to_core(worker_id);

                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to build tokio runtime for worker");

                rt.block_on(async move {
                    worker_loop(
                        worker_id, &config, shared_tls, metrics, ip_tracker,
                        active_conns, shutdown_rx, shutdown_flag, health_map,
                        access_logger, rate_limiter, peer_tracker, global_sem,
                        drain_rx, access_control, challenge_store,
                        global_backend_counters,
                    ).await;
                });
            })
            .expect("Failed to spawn worker thread");

        handles.push(handle);
    }

    // Spawn metrics reporter with shutdown awareness
    let metrics_clone = metrics.clone();
    let metrics_shutdown_rx = shutdown_tx.subscribe();
    std::thread::Builder::new()
        .name("q-flux-metrics".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(metrics_reporter(metrics_clone, metrics_shutdown_rx));
        })
        .ok();

    handles
}

#[allow(clippy::too_many_arguments)]
async fn worker_loop(
    worker_id: usize,
    config: &FluxConfig,
    shared_tls: SharedTlsConfig,
    metrics: Metrics,
    ip_tracker: IpConnTracker,
    active_conns: ActiveConnCount,
    mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
    shutdown_flag: Arc<AtomicBool>,
    health_map: HealthMap,
    access_logger: Option<AccessLogger>,
    rate_limiter: Option<Arc<RateLimiter>>,
    peer_tracker: Arc<PeerTracker>,
    global_upstream_semaphore: Option<Arc<Semaphore>>,
    drain_rx: DrainReceiver,
    access_control: Arc<AccessControl>,
    challenge_store: ChallengeStore,
    global_backend_counters: Arc<dashmap::DashMap<String, Arc<crate::upstream::BackendCounters>>>,
) {
    // Backpressure: limit concurrent connection handlers to prevent OOM.
    // If all permits taken, accept() still runs but spawn waits for a permit.
    let handler_semaphore = Arc::new(Semaphore::new(MAX_HANDLERS_PER_WORKER));

    // Create listeners — each worker binds to the same ports via SO_REUSEPORT
    let mut listeners: Vec<TcpListener> = Vec::new();
    for addr in &config.server.listen {
        match acceptor::create_listener(addr) {
            Ok(socket) => {
                match acceptor::into_tokio_listener(socket) {
                    Ok(listener) => {
                        tracing::info!(worker = worker_id, addr = %addr, "Worker bound");
                        listeners.push(listener);
                    }
                    Err(e) => {
                        tracing::error!(worker = worker_id, addr = %addr, "Failed to create tokio listener: {}", e);
                    }
                }
            }
            Err(e) => {
                tracing::error!(worker = worker_id, addr = %addr, "Failed to bind: {}", e);
            }
        }
    }

    if listeners.is_empty() {
        tracing::error!(worker = worker_id, "No listeners — worker exiting");
        return;
    }

    let max_conns = config.limits.max_connections as u64;
    let max_per_ip = config.limits.max_conns_per_ip as u64;
    let body_limit = config.limits.request_body_limit;
    let streaming_body_threshold = config.limits.streaming_body_threshold;
    let static_config = Arc::new(config.static_files.clone());

    // Build h1-only TLS config for libp2p WebSocket proxy port.
    // Browser js-libp2p needs HTTP/1.1 for WebSocket upgrade — h2 ALPN breaks it.
    let libp2p_ws_tls: Option<Arc<rustls::ServerConfig>> = if config.libp2p_ws.enabled {
        match acceptor::build_tls_config_h1_only(&config.tls) {
            Ok(cfg) => {
                if worker_id == 0 {
                    tracing::info!(
                        port = config.libp2p_ws.port,
                        backend = %config.libp2p_ws.backend,
                        "LibP2P WS proxy enabled: port {} → {}",
                        config.libp2p_ws.port, config.libp2p_ws.backend,
                    );
                }
                Some(cfg)
            }
            Err(e) => {
                tracing::error!("Failed to build libp2p WS TLS config: {}", e);
                None
            }
        }
    } else {
        None
    };

    // In-memory file cache with pre-compressed gzip variants.
    // Shared across all connections on this worker. Each worker gets its own
    // cache instance (no cross-thread contention), populated on first access.
    let file_cache = FileCache::new(&config.static_files);
    if let Some(ref cache) = file_cache {
        tracing::info!(
            worker = worker_id,
            max_file = config.static_files.cache_max_file_size / 1024,
            max_total = config.static_files.cache_max_total / (1024 * 1024),
            gzip = config.static_files.gzip,
            "File cache enabled (max {}KB/file, {}MB total, gzip={})",
            config.static_files.cache_max_file_size / 1024,
            config.static_files.cache_max_total / (1024 * 1024),
            config.static_files.gzip,
        );
        let _ = cache; // suppress unused
    }

    // CRITICAL: Create ONE UpstreamPool per worker, shared across all connections.
    // Previous bug: UpstreamPool::new() was called per-connection, creating a NEW
    // hyper Client each time. This defeated connection pooling — every request
    // opened a fresh TCP connection to upstream (same failure mode as keepalive=off).
    // Now: one hyper Client per worker with pooled keepalive connections to upstream.
    // Super-cluster: cluster peers are passed as failover backends (local-first).
    // Global semaphore: shared across ALL workers to cap total backend load.
    let upstream = Arc::new(UpstreamPool::new_with_counters(
        &config.upstream,
        metrics.clone(),
        health_map.clone(),
        config.cluster.peers.clone(),
        global_upstream_semaphore.clone(),
        global_backend_counters.clone(),
    ));

    // Per-vhost upstream pools: each [[vhosts]] entry with a `backend` gets its own
    // connection pool. Keyed by lowercase domain name. Shares the global semaphore.
    let vhost_upstream_map: Arc<HashMap<String, Arc<UpstreamPool>>> = {
        let mut map = HashMap::new();
        for vhost in &config.vhosts {
            if let Some(ref backend) = vhost.backend {
                let vh_cfg = UpstreamConfig {
                    backends: vec![backend.clone()],
                    ..config.upstream.clone()
                };
                let pool = Arc::new(UpstreamPool::new_with_counters(
                    &vh_cfg,
                    metrics.clone(),
                    health_map.clone(),
                    vec![],
                    global_upstream_semaphore.clone(),
                    global_backend_counters.clone(),
                ));
                for domain in &vhost.domains {
                    map.insert(domain.to_lowercase(), pool.clone());
                }
            }
        }
        Arc::new(map)
    };

    // Per-vhost static configs: each [[vhosts]] entry with a `static_root` gets its
    // own StaticConfig. Keyed by lowercase domain name.
    let vhost_static_map: Arc<HashMap<String, Arc<StaticConfig>>> = {
        let mut map = HashMap::new();
        for vhost in &config.vhosts {
            if let Some(ref root) = vhost.static_root {
                let vh_static = Arc::new(StaticConfig {
                    root: Some(root.clone()),
                    spa_fallback: vhost.spa_fallback,
                    gzip: config.static_files.gzip,
                    cache_max_file_size: config.static_files.cache_max_file_size,
                    cache_max_total: config.static_files.cache_max_total,
                    proxy_compression: config.static_files.proxy_compression,
                    proxy_paths: vhost.proxy_paths.clone(),
                });
                for domain in &vhost.domains {
                    map.insert(domain.to_lowercase(), vh_static.clone());
                }
            }
        }
        Arc::new(map)
    };

    // Spawn adaptive concurrency adjuster (AIMD) — runs every 1s, adjusting
    // the effective permit limit based on upstream response latency.
    // Only worker 0 runs this to avoid redundant adjustments.
    if worker_id == 0 {
        let adaptive = upstream.adaptive.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
            loop {
                interval.tick().await;
                adaptive.adjust();
            }
        });
    }

    // PeerTracker: per-peer connection limits and circuit breakers for libp2p peers.
    // Shared across all workers and the admin server. Pre-seeded with known infrastructure.
    // (Created in main.rs and passed in.)

    // BandwidthLimiter: per-peer token-bucket bandwidth limiting for WebSocket splice.
    // Shared across all connections on this worker.
    let bandwidth_limiter = Arc::new(BandwidthLimiter::new());

    // Periodic PeerTracker maintenance (every 60s):
    //   1. Evict stale peers (idle > 5min with no active connections)
    //   2. Cleanup bandwidth limiter buckets for evicted peers
    //   3. Auto-classify Unknown peers → Miner based on traffic heuristics
    //      (active > 1hr with > 100MB transferred)
    {
        let tracker = peer_tracker.clone();
        let bw_limiter = bandwidth_limiter.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                tracker.cleanup_stale(std::time::Duration::from_secs(300));
                bw_limiter.cleanup(tracker.peer_map());
                // Auto-tier: promote Unknown peers with >1hr activity and >100MB
                tracker.auto_classify(
                    std::time::Duration::from_secs(3600),  // 1 hour
                    100 * 1024 * 1024,                     // 100 MB
                );
            }
        });
    }

    // Spawn periodic IP tracker garbage collection to prevent unbounded DashMap growth.
    // With billions of miners cycling through, stale IPs must be evicted.
    let gc_ip_tracker = ip_tracker.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(IP_TRACKER_GC_INTERVAL_SECS));
        loop {
            interval.tick().await;
            let before = gc_ip_tracker.len();
            gc_ip_tracker.retain(|_, count| *count > 0);
            let removed = before.saturating_sub(gc_ip_tracker.len());
            if removed > 0 {
                tracing::debug!(worker = worker_id, removed, remaining = gc_ip_tracker.len(), "IP tracker GC");
            }
        }
    });

    // Spawn periodic rate limiter cleanup (evict stale per-IP token buckets)
    if let Some(ref rl) = rate_limiter {
        let rl = rl.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(RATE_LIMITER_GC_INTERVAL_SECS));
            loop {
                interval.tick().await;
                rl.cleanup();
            }
        });
    }

    // Spawn TLS drain watcher: logs when a TLS reload occurs so operators
    // can observe that old connections are draining to the previous config.
    // This is informational only -- old connections naturally use their captured
    // Arc<ServerConfig> and are not forcefully terminated.
    {
        let drain_tls = shared_tls.clone();
        let drain_timeout = config.tls.drain_timeout_secs;
        tokio::spawn(async move {
            loop {
                drain_tls.drain_notified().await;
                let count = drain_tls.reload_count();
                tracing::info!(
                    worker = worker_id,
                    reload_count = count,
                    drain_timeout_secs = drain_timeout,
                    "TLS config reloaded (reload #{}), new connections use updated certs. \
                     Old connections drain naturally (timeout hint: {}s).",
                    count,
                    drain_timeout,
                );
            }
        });
    }

    // Accept loop — exits on shutdown signal
    loop {
        // Fast-path shutdown check (no channel overhead)
        if shutdown_flag.load(Ordering::Relaxed) {
            tracing::info!(worker = worker_id, "Shutdown flag set — stopping accept loop");
            break;
        }

        let accept_result = tokio::select! {
            r = accept_any(&listeners) => r,
            _ = shutdown_rx.recv() => {
                tracing::info!(worker = worker_id, "Shutdown signal received — stopping accept loop");
                break;
            }
        };
        let (tcp_stream, client_addr) = match accept_result {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(worker = worker_id, "Accept error: {}", e);
                continue;
            }
        };

        // Global connection limit
        if active_conns.load(Ordering::Relaxed) >= max_conns {
            tracing::warn!(worker = worker_id, "Global connection limit reached");
            drop(tcp_stream);
            continue;
        }

        // Issue #027: IP access control — checked BEFORE TLS handshake (zero resources for blocked IPs)
        let client_ip = client_addr.ip();
        if !access_control.is_allowed(client_ip) {
            tracing::debug!(worker = worker_id, ip = %client_ip, "IP blocked by access control");
            metrics.rate_limited(); // Reuse rate_limited counter for blocked IPs
            drop(tcp_stream);
            continue;
        }

        // Per-IP connection limit (concurrent connection cap)
        {
            let mut count = ip_tracker.entry(client_ip).or_insert(0);
            if *count >= max_per_ip {
                metrics.rate_limited();
                tracing::debug!(worker = worker_id, ip = %client_ip, "Per-IP limit exceeded ({}/{})", *count, max_per_ip);
                drop(count); // Release entry lock before dropping stream
                drop(tcp_stream);
                continue;
                // count was NOT incremented, so no decrement needed
            }
            *count += 1;
        }

        active_conns.fetch_add(1, Ordering::Relaxed);
        metrics.conn_opened();

        // Token-bucket rate limiting (per-IP request rate cap)
        if let Some(ref rl) = rate_limiter {
            if !rl.check(client_ip) {
                metrics.rate_limited();
                tracing::debug!(worker = worker_id, ip = %client_ip, "Token-bucket rate limited");
                cleanup_conn(&ip_tracker, client_ip, &active_conns, &metrics);
                drop(tcp_stream);
                continue;
            }
        }

        // Determine if this is a TLS port (443/9443) or plain HTTP (80)
        let local_port = tcp_stream.local_addr().map(|a| a.port()).unwrap_or(443);
        let is_tls = local_port != 80; // All ports except 80 are TLS
        let is_libp2p_ws = config.libp2p_ws.enabled && local_port == config.libp2p_ws.port;
        let libp2p_ws_backend = if is_libp2p_ws {
            Some(config.libp2p_ws.backend.clone())
        } else {
            None
        };
        // Also proxy WebSocket upgrades on port 443 to libp2p backend?
        let libp2p_ws_on_443 = !is_libp2p_ws
            && is_tls
            && config.libp2p_ws.enabled
            && config.libp2p_ws.proxy_on_main_port;
        let libp2p_443_backend = if libp2p_ws_on_443 {
            Some(config.libp2p_ws.backend.clone())
        } else {
            None
        };

        // Hot-reload: load current TLS config per connection (read lock, ~10ns).
        // If certs were reloaded via admin API, new connections get the new config.
        // LibP2P WS port uses h1-only TLS config (no h2 ALPN).
        let tls_acceptor = if is_libp2p_ws {
            if let Some(ref h1_cfg) = libp2p_ws_tls {
                TlsAcceptor::from(h1_cfg.clone())
            } else {
                TlsAcceptor::from(shared_tls.load())
            }
        } else {
            TlsAcceptor::from(shared_tls.load())
        };
        let metrics = metrics.clone();
        let ip_tracker = ip_tracker.clone();
        let active_conns = active_conns.clone();
        let upstream = upstream.clone();
        let peer_tracker = peer_tracker.clone();
        let bandwidth_limiter = bandwidth_limiter.clone();
        let semaphore = handler_semaphore.clone();
        let static_config = static_config.clone();
        let access_logger = access_logger.clone();
        let drain_rx = drain_rx.clone();
        let file_cache = file_cache.clone();
        let challenge_store = challenge_store.clone();
        let vhost_upstream_map = vhost_upstream_map.clone();
        let vhost_static_map = vhost_static_map.clone();
        // Extract config fields needed inside spawn (config is &'_ not 'static)
        let onion_static_root = config.static_files.root.as_ref()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        tokio::spawn(async move {
            // ── LibP2P WebSocket proxy: separate path ──────────────────────
            // LibP2P connections are long-lived (hours). They MUST NOT:
            //  1. Be subject to MAX_CONN_LIFETIME timeout (would kill after 5min)
            //  2. Hold a handler semaphore permit (wastes capacity)
            // Handle them first, before semaphore acquisition.
            if is_libp2p_ws && is_tls {
                let tls_result = tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    tls_acceptor.accept(tcp_stream),
                ).await;

                match tls_result {
                    Ok(Ok(tls_stream)) => {
                        metrics.tls_handshake_ok();
                        if let Some(ref backend_addr) = libp2p_ws_backend {
                            match tokio::net::TcpStream::connect(backend_addr).await {
                                Ok(backend_stream) => {
                                    tracing::info!(client = %client_addr,
                                        "LibP2P WS proxy established (no timeout)");
                                    // BufReader on read side (reduces syscalls for large gossipsub msgs).
                                    // NO BufWriter on write side — BufWriter would hold the initial HTTP
                                    // WebSocket upgrade request (~500 bytes) in its 64KB buffer without
                                    // flushing, deadlocking the WS handshake (backend never gets request,
                                    // client never gets HTTP 101 response). Use raw split halves instead.
                                    let (client_read, mut client_write) = tokio::io::split(tls_stream);
                                    let (backend_read, mut backend_write) = tokio::io::split(backend_stream);
                                    let mut cr = tokio::io::BufReader::with_capacity(65536, client_read);
                                    let mut br = tokio::io::BufReader::with_capacity(65536, backend_read);
                                    // No timeout — libp2p connections are long-lived
                                    let _ = tokio::select! {
                                        r = tokio::io::copy(&mut cr, &mut backend_write) => r,
                                        r = tokio::io::copy(&mut br, &mut client_write) => r,
                                    };
                                }
                                Err(e) => {
                                    tracing::warn!(client = %client_addr, backend = ?backend_addr,
                                        "LibP2P WS backend connect failed: {}", e);
                                }
                            }
                        }
                    }
                    Ok(Err(e)) => {
                        metrics.tls_handshake_fail();
                        tracing::debug!(client = %client_addr, "LibP2P WS TLS handshake failed: {}", e);
                    }
                    Err(_) => {
                        metrics.tls_handshake_fail();
                        tracing::debug!(client = %client_addr, "LibP2P WS TLS handshake timeout");
                    }
                }
                // Single cleanup for the libp2p path
                cleanup_conn(&ip_tracker, client_ip, &active_conns, &metrics);
                return;
            }

            // ── LibP2P WebSocket via main HTTPS port (443) ──────────────────
            // When proxy_on_main_port is enabled, detect WebSocket upgrade
            // requests on port 443 and route them to the libp2p backend.
            // This allows nodes behind restrictive NAT/firewalls to connect
            // via the only port guaranteed to be open everywhere (443).
            // Like port 9443, this path has NO timeout (libp2p is long-lived).
            if libp2p_ws_on_443 {
                let tls_result = tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    tls_acceptor.accept(tcp_stream),
                ).await;

                match tls_result {
                    Ok(Ok(mut tls_stream)) => {
                        metrics.tls_handshake_ok();
                        // Extract SNI for vhost routing before any borrow issues
                        let sni_443 = tls_stream.get_ref().1
                            .server_name().map(|s| s.to_lowercase());
                        let is_h2 = tls_stream.get_ref().1
                            .alpn_protocol()
                            .map(|p| p == b"h2")
                            .unwrap_or(false);

                        if !is_h2 {
                            // Read first bytes to detect WebSocket upgrade header
                            let mut peek = vec![0u8; 4096];
                            match tokio::time::timeout(
                                std::time::Duration::from_secs(5),
                                tls_stream.read(&mut peek),
                            ).await {
                                Ok(Ok(n)) if n > 0 => {
                                    if crate::simd_parse::is_websocket_upgrade(&peek[..n]) {
                                        // WebSocket on 443 → tunnel to libp2p backend (no timeout)
                                        if let Some(ref backend_addr) = libp2p_443_backend {
                                            tracing::info!(client = %client_addr,
                                                "LibP2P WS proxy via port 443 (no timeout)");
                                            match tokio::net::TcpStream::connect(backend_addr.as_str()).await {
                                                Ok(mut backend_stream) => {
                                                    let _ = backend_stream.write_all(&peek[..n]).await;
                                                    // Same fix as port 9443 path: no BufWriter on write side.
                                                    let (cr, mut cw) = tokio::io::split(tls_stream);
                                                    let (br, mut bw) = tokio::io::split(backend_stream);
                                                    let mut cr = tokio::io::BufReader::with_capacity(65536, cr);
                                                    let mut br = tokio::io::BufReader::with_capacity(65536, br);
                                                    let _ = tokio::select! {
                                                        r = tokio::io::copy(&mut cr, &mut bw) => r,
                                                        r = tokio::io::copy(&mut br, &mut cw) => r,
                                                    };
                                                }
                                                Err(e) => {
                                                    tracing::warn!(client = %client_addr,
                                                        "LibP2P WS (443) backend failed: {}", e);
                                                }
                                            }
                                        }
                                        cleanup_conn(&ip_tracker, client_ip, &active_conns, &metrics);
                                        return;
                                    }

                                    // Not WebSocket — pass stream + peeked bytes to normal proxy
                                    let prefixed = PrefixedStream::new(peek[..n].to_vec(), tls_stream);
                                    let _permit = match semaphore.try_acquire() {
                                        Ok(p) => p,
                                        Err(_) => {
                                            tracing::warn!(client = %client_addr, "Worker at capacity");
                                            cleanup_conn(&ip_tracker, client_ip, &active_conns, &metrics);
                                            return;
                                        }
                                    };
                                    // Vhost routing for the non-WS fallback path
                                    let eff_up443 = sni_443.as_deref()
                                        .and_then(|s| vhost_upstream_map.get(s))
                                        .cloned().unwrap_or_else(|| upstream.clone());
                                    let eff_st443 = sni_443.as_deref()
                                        .and_then(|s| vhost_static_map.get(s))
                                        .cloned().unwrap_or_else(|| static_config.clone());
                                    const MAX_CONN_443: std::time::Duration =
                                        std::time::Duration::from_secs(300);
                                    let handler = async {
                                        let cache_ref = file_cache.as_deref();
                                        match access_logger {
                                            Some(ref logger) => {
                                                proxy::handle_connection_logged(
                                                    prefixed, client_addr, &eff_up443, &metrics,
                                                    body_limit, streaming_body_threshold,
                                                    &eff_st443, logger,
                                                    &peer_tracker, &bandwidth_limiter,
                                                    drain_rx.clone(), cache_ref,
                                                ).await;
                                            }
                                            None => {
                                                proxy::handle_connection(
                                                    prefixed, client_addr, &eff_up443, &metrics,
                                                    body_limit, streaming_body_threshold,
                                                    &eff_st443,
                                                    &peer_tracker, &bandwidth_limiter,
                                                    drain_rx.clone(), cache_ref,
                                                ).await;
                                            }
                                        }
                                    };
                                    let _ = tokio::time::timeout(MAX_CONN_443, handler).await;
                                    cleanup_conn(&ip_tracker, client_ip, &active_conns, &metrics);
                                    return;
                                }
                                _ => {
                                    // Read failed — close connection
                                    cleanup_conn(&ip_tracker, client_ip, &active_conns, &metrics);
                                    return;
                                }
                            }
                        }

                        // H2 on port 443 — handle via H2 proxy
                        let _permit = match semaphore.try_acquire() {
                            Ok(p) => p,
                            Err(_) => {
                                tracing::warn!(client = %client_addr, "Worker at capacity");
                                cleanup_conn(&ip_tracker, client_ip, &active_conns, &metrics);
                                return;
                            }
                        };
                        let eff_up_h2 = sni_443.as_deref()
                            .and_then(|s| vhost_upstream_map.get(s))
                            .cloned().unwrap_or_else(|| upstream.clone());
                        let eff_st_h2 = sni_443.as_deref()
                            .and_then(|s| vhost_static_map.get(s))
                            .cloned().unwrap_or_else(|| static_config.clone());
                        const MAX_CONN_H2: std::time::Duration =
                            std::time::Duration::from_secs(300);
                        let handler = async {
                            h2_proxy::handle_h2_connection(
                                tls_stream, client_addr, eff_up_h2.clone(),
                                metrics.clone(), body_limit, eff_st_h2.clone(),
                                access_logger.clone(),
                            ).await;
                        };
                        let _ = tokio::time::timeout(MAX_CONN_H2, handler).await;
                        cleanup_conn(&ip_tracker, client_ip, &active_conns, &metrics);
                        return;
                    }
                    Ok(Err(e)) => {
                        metrics.tls_handshake_fail();
                        tracing::debug!(client = %client_addr, "TLS handshake failed (443 WS): {}", e);
                    }
                    Err(_) => {
                        metrics.tls_handshake_fail();
                        tracing::debug!(client = %client_addr, "TLS handshake timeout (443 WS)");
                    }
                }
                cleanup_conn(&ip_tracker, client_ip, &active_conns, &metrics);
                return;
            }

            // ── Normal HTTP/H2 path ────────────────────────────────────────
            // Acquire semaphore permit — backpressure if too many concurrent handlers.
            // try_acquire: if no permits, drop connection immediately with 503.
            let _permit = match semaphore.try_acquire() {
                Ok(permit) => permit,
                Err(_) => {
                    tracing::warn!(client = %client_addr, "Worker at capacity, dropping connection");
                    // Still need to clean up IP tracker and active count
                    cleanup_conn(&ip_tracker, client_ip, &active_conns, &metrics);
                    return;
                }
            };

            // Overall connection timeout: 5 minutes for normal requests,
            // SSE/WebSocket connections will be dropped after this limit.
            // This prevents connection leaks from keeping handlers alive forever.
            const MAX_CONN_LIFETIME: std::time::Duration = std::time::Duration::from_secs(300);

            let handler = async {
            if is_tls {
                let tls_result = tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    tls_acceptor.accept(tcp_stream),
                ).await;

                match tls_result {
                    Ok(Ok(tls_stream)) => {
                        metrics.tls_handshake_ok();

                        // SNI-based vhost routing: select per-domain upstream pool
                        // and static root when [[vhosts]] are configured.
                        let sni = tls_stream.get_ref().1
                            .server_name()
                            .map(|s| s.to_lowercase());
                        let eff_upstream = sni.as_deref()
                            .and_then(|s| vhost_upstream_map.get(s))
                            .cloned()
                            .unwrap_or_else(|| upstream.clone());
                        let eff_static = sni.as_deref()
                            .and_then(|s| vhost_static_map.get(s))
                            .cloned()
                            .unwrap_or_else(|| static_config.clone());

                        // ALPN-based protocol routing (Phase 3):
                        // If client negotiated "h2", handle via HTTP/2 multiplexed proxy.
                        // Otherwise, fall through to HTTP/1.1 proxy.
                        let is_h2 = tls_stream.get_ref().1
                            .alpn_protocol()
                            .map(|p| p == b"h2")
                            .unwrap_or(false);

                        if is_h2 {
                            h2_proxy::handle_h2_connection(
                                tls_stream, client_addr, eff_upstream.clone(),
                                metrics.clone(), body_limit, eff_static.clone(),
                                access_logger.clone(),
                            ).await;
                        } else {
                            // HTTP/1.1 path — use logged variant if access logger configured
                            let cache_ref = file_cache.as_deref();
                            match access_logger {
                                Some(ref logger) => {
                                    proxy::handle_connection_logged(
                                        tls_stream, client_addr, &eff_upstream, &metrics,
                                        body_limit, streaming_body_threshold,
                                        &eff_static, logger,
                                        &peer_tracker, &bandwidth_limiter,
                                        drain_rx.clone(), cache_ref,
                                    ).await;
                                }
                                None => {
                                    proxy::handle_connection(
                                        tls_stream, client_addr, &eff_upstream, &metrics,
                                        body_limit, streaming_body_threshold,
                                        &eff_static,
                                        &peer_tracker, &bandwidth_limiter,
                                        drain_rx.clone(), cache_ref,
                                    ).await;
                                }
                            }
                        }
                    }
                    Ok(Err(e)) => {
                        metrics.tls_handshake_fail();
                        tracing::debug!(client = %client_addr, "TLS handshake failed: {}", e);
                    }
                    Err(_) => {
                        metrics.tls_handshake_fail();
                        tracing::debug!(client = %client_addr, "TLS handshake timeout");
                    }
                }
            } else {
                // Plain HTTP — read request, check for ACME challenges, then redirect
                let mut tcp_stream = tcp_stream;
                let mut peek_buf = [0u8; 1024];
                let (host, path) = match tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    tokio::io::AsyncReadExt::read(&mut tcp_stream, &mut peek_buf),
                ).await {
                    Ok(Ok(n)) if n > 0 => {
                        let h = extract_host_header(&peek_buf[..n])
                            .unwrap_or_else(|| client_addr.ip().to_string());
                        let p = extract_request_path(&peek_buf[..n])
                            .unwrap_or_default();
                        (h, p)
                    }
                    _ => (client_addr.ip().to_string(), String::new()),
                };

                // Issue #021: Serve ACME HTTP-01 challenges on port 80
                if let Some(token) = path.strip_prefix("/.well-known/acme-challenge/") {
                    // Clone the proof out before await to avoid holding
                    // parking_lot guard across await (not Send).
                    let proof = challenge_store.read().get(token).cloned();
                    if let Some(proof) = proof {
                        let response = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/octet-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                            proof.len(), proof
                        );
                        let _ = tcp_stream.write_all(response.as_bytes()).await;
                    } else {
                        let _ = tcp_stream.write_all(
                            b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                        ).await;
                    }
                } else if host.contains(".onion") || client_addr.ip().is_loopback() {
                    // v10.3.8: Serve Tor onion requests directly over HTTP (no HTTPS redirect).
                    // Tor hidden services don't have TLS certs — the Tor circuit provides encryption.
                    // Also serves localhost requests directly for local development.
                    let static_root = onion_static_root.clone();
                    if !static_root.is_empty() {
                        // Try static file first
                        let clean = path.split('?').next().unwrap_or(&path);
                        // v10.3.8 SECURITY: Reject path traversal attempts
                        let decoded_path = clean.replace("%2e", ".").replace("%2E", ".").replace("%2f", "/").replace("%2F", "/");
                        if decoded_path.contains("..") || decoded_path.contains("//") {
                            let _ = tcp_stream.write_all(
                                b"HTTP/1.1 403 Forbidden\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                            ).await;
                            return;
                        }
                        let file_path = if clean.is_empty() || clean == "/" {
                            format!("{}/index.html", static_root)
                        } else {
                            format!("{}{}", static_root, clean)
                        };
                        // SECURITY: Verify resolved path stays within static_root
                        if let Ok(canonical) = tokio::fs::canonicalize(&file_path).await {
                            if !canonical.to_string_lossy().starts_with(&static_root) {
                                let _ = tcp_stream.write_all(
                                    b"HTTP/1.1 403 Forbidden\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                                ).await;
                                return;
                            }
                        }

                        if let Ok(contents) = tokio::fs::read(&file_path).await {
                            let content_type = if file_path.ends_with(".html") { "text/html" }
                                else if file_path.ends_with(".js") { "application/javascript" }
                                else if file_path.ends_with(".css") { "text/css" }
                                else if file_path.ends_with(".json") { "application/json" }
                                else if file_path.ends_with(".png") { "image/png" }
                                else if file_path.ends_with(".svg") { "image/svg+xml" }
                                else if file_path.ends_with(".ico") { "image/x-icon" }
                                else if file_path.ends_with(".woff2") { "font/woff2" }
                                else { "application/octet-stream" };
                            let response = format!(
                                "HTTP/1.1 200 OK\r\ncontent-type: {}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                                content_type, contents.len()
                            );
                            let _ = tcp_stream.write_all(response.as_bytes()).await;
                            let _ = tcp_stream.write_all(&contents).await;
                        } else if path.starts_with("/api/") {
                            // API request — forward to upstream
                            let upstream_url = format!("http://127.0.0.1:8080{}", path);
                            match reqwest::get(&upstream_url).await {
                                Ok(resp) => {
                                    let status = resp.status().as_u16();
                                    let body = resp.bytes().await.unwrap_or_default();
                                    let response = format!(
                                        "HTTP/1.1 {} OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                                        status, body.len()
                                    );
                                    let _ = tcp_stream.write_all(response.as_bytes()).await;
                                    let _ = tcp_stream.write_all(&body).await;
                                }
                                Err(_) => {
                                    let _ = tcp_stream.write_all(
                                        b"HTTP/1.1 502 Bad Gateway\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                                    ).await;
                                }
                            }
                        } else {
                            // SPA fallback — serve index.html
                            let index = format!("{}/index.html", static_root);
                            if let Ok(contents) = tokio::fs::read(&index).await {
                                let response = format!(
                                    "HTTP/1.1 200 OK\r\ncontent-type: text/html\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                                    contents.len()
                                );
                                let _ = tcp_stream.write_all(response.as_bytes()).await;
                                let _ = tcp_stream.write_all(&contents).await;
                            } else {
                                let _ = tcp_stream.write_all(
                                    b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                                ).await;
                            }
                        }
                    } else {
                        let _ = tcp_stream.write_all(
                            b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                        ).await;
                    }
                } else {
                    let redirect = format!(
                        "HTTP/1.1 301 Moved Permanently\r\nlocation: https://{}{}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                        host, path,
                    );
                    let _ = tcp_stream.write_all(redirect.as_bytes()).await;
                }
            }

            }; // end handler async block

            // Enforce maximum connection lifetime
            let _ = tokio::time::timeout(MAX_CONN_LIFETIME, handler).await;

            // Cleanup — semaphore permit auto-drops when _permit goes out of scope
            cleanup_conn(&ip_tracker, client_ip, &active_conns, &metrics);
        });
    }

    // Accept loop exited. In-flight request tasks are still running on this
    // worker's runtime. We wait here so that block_on() does not return and
    // drop the runtime (which would cancel all spawned tasks). The main
    // thread's drain timeout controls how long we actually wait — when it
    // fires, process exit kills this thread regardless.
    tracing::info!(
        worker = worker_id,
        "Accept loop stopped, waiting for in-flight requests to drain"
    );

    // Sleep longer than the main thread's drain timeout (default 30s).
    // The main thread will exit the process when its timer fires, which
    // terminates this sleep and all spawned tasks.
    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── extract_host_header ─────────────────────────────────────────

    #[test]
    fn test_extract_host_basic() {
        let data = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n";
        assert_eq!(extract_host_header(data), Some("example.com".to_string()));
    }

    #[test]
    fn test_extract_host_with_port() {
        let data = b"GET / HTTP/1.1\r\nHost: example.com:8080\r\n\r\n";
        assert_eq!(extract_host_header(data), Some("example.com:8080".to_string()));
    }

    #[test]
    fn test_extract_host_case_insensitive() {
        let data = b"GET / HTTP/1.1\r\nhost: quillon.xyz\r\n\r\n";
        assert_eq!(extract_host_header(data), Some("quillon.xyz".to_string()));
    }

    #[test]
    fn test_extract_host_missing() {
        let data = b"GET / HTTP/1.1\r\nAccept: */*\r\n\r\n";
        assert_eq!(extract_host_header(data), None);
    }

    #[test]
    fn test_extract_host_empty_data() {
        let data = b"";
        assert_eq!(extract_host_header(data), None);
    }

    #[test]
    fn test_extract_host_multiple_headers() {
        let data = b"GET /path HTTP/1.1\r\nUser-Agent: curl/7.0\r\nHost: api.quillon.xyz\r\nAccept: */*\r\n\r\n";
        assert_eq!(extract_host_header(data), Some("api.quillon.xyz".to_string()));
    }

    // ── cleanup_conn ────────────────────────────────────────────────

    #[test]
    fn test_cleanup_conn_decrements_ip_count() {
        let ip_tracker: IpConnTracker = Arc::new(DashMap::new());
        let active_conns: ActiveConnCount = Arc::new(AtomicU64::new(1));
        let metrics = Metrics::new();
        let ip: std::net::IpAddr = "192.168.1.1".parse().unwrap();

        ip_tracker.insert(ip, 3);
        cleanup_conn(&ip_tracker, ip, &active_conns, &metrics);
        assert_eq!(*ip_tracker.get(&ip).unwrap(), 2);
        assert_eq!(active_conns.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_cleanup_conn_removes_at_one() {
        let ip_tracker: IpConnTracker = Arc::new(DashMap::new());
        let active_conns: ActiveConnCount = Arc::new(AtomicU64::new(1));
        let metrics = Metrics::new();
        let ip: std::net::IpAddr = "10.0.0.1".parse().unwrap();

        ip_tracker.insert(ip, 1);
        cleanup_conn(&ip_tracker, ip, &active_conns, &metrics);
        assert!(ip_tracker.get(&ip).is_none(), "IP entry should be removed when count reaches 0");
    }

    #[test]
    fn test_cleanup_conn_no_underflow() {
        let ip_tracker: IpConnTracker = Arc::new(DashMap::new());
        let active_conns: ActiveConnCount = Arc::new(AtomicU64::new(0));
        let metrics = Metrics::new();
        let ip: std::net::IpAddr = "10.0.0.2".parse().unwrap();

        // No entry in ip_tracker, active_conns already 0 — should not panic or underflow
        cleanup_conn(&ip_tracker, ip, &active_conns, &metrics);
        assert_eq!(active_conns.load(Ordering::Relaxed), 0, "should not underflow past 0");
    }

    #[test]
    fn test_cleanup_conn_concurrent_safety() {
        let ip_tracker: IpConnTracker = Arc::new(DashMap::new());
        let active_conns: ActiveConnCount = Arc::new(AtomicU64::new(100));
        let metrics = Metrics::new();
        let ip: std::net::IpAddr = "10.0.0.3".parse().unwrap();

        ip_tracker.insert(ip, 100);

        // Simulate 100 concurrent cleanups
        let handles: Vec<_> = (0..100).map(|_| {
            let ip_tracker = ip_tracker.clone();
            let active_conns = active_conns.clone();
            let metrics = metrics.clone();
            std::thread::spawn(move || {
                cleanup_conn(&ip_tracker, ip, &active_conns, &metrics);
            })
        }).collect();

        for h in handles { h.join().unwrap(); }

        assert!(ip_tracker.get(&ip).is_none(), "IP should be removed after all cleanups");
        assert_eq!(active_conns.load(Ordering::Relaxed), 0, "active_conns should reach 0");
    }

    // ── constants ───────────────────────────────────────────────────

    #[test]
    fn test_max_handlers_per_worker_reasonable() {
        assert!(MAX_HANDLERS_PER_WORKER >= 64, "should allow at least 64 concurrent handlers");
        assert!(MAX_HANDLERS_PER_WORKER <= 65536, "should not be unreasonably large");
    }
}

/// Decrement IP counter and global active count on connection close.
///
/// Uses the DashMap entry API to atomically decrement-and-remove.
/// The `entry()` call holds the shard lock for the entire scope,
/// preventing the TOCTOU race where another thread could insert a
/// new entry between `drop(count)` and `remove(&client_ip)`.
fn cleanup_conn(
    ip_tracker: &IpConnTracker,
    client_ip: std::net::IpAddr,
    active_conns: &ActiveConnCount,
    metrics: &Metrics,
) {
    if let dashmap::mapref::entry::Entry::Occupied(mut entry) = ip_tracker.entry(client_ip) {
        let count = entry.get_mut();
        if *count > 1 {
            *count -= 1;
        } else {
            entry.remove();
        }
    }
    // Underflow-safe decrement: CAS loop prevents wrapping to u64::MAX
    loop {
        let current = active_conns.load(Ordering::Relaxed);
        if current == 0 {
            break; // Already zero — don't underflow
        }
        if active_conns.compare_exchange_weak(current, current - 1, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
            break;
        }
    }
    metrics.conn_closed();
}

/// Extract Host header value from raw HTTP request bytes.
fn extract_host_header(data: &[u8]) -> Option<String> {
    let mut headers = [httparse::EMPTY_HEADER; 32];
    let mut req = httparse::Request::new(&mut headers);
    if req.parse(data).is_ok() {
        for h in req.headers.iter() {
            if h.name.eq_ignore_ascii_case("host") {
                return std::str::from_utf8(h.value).ok().map(|s| s.to_string());
            }
        }
    }
    None
}

/// Extract Host header value from raw HTTP request bytes.
/// Exposed for testing via pub(crate).
pub(crate) fn extract_host_header_pub(data: &[u8]) -> Option<String> {
    extract_host_header(data)
}

/// Extract the request path from raw HTTP request bytes.
fn extract_request_path(data: &[u8]) -> Option<String> {
    let mut headers = [httparse::EMPTY_HEADER; 32];
    let mut req = httparse::Request::new(&mut headers);
    if req.parse(data).is_ok() {
        req.path.map(|p| p.to_string())
    } else {
        None
    }
}

/// Accept a connection from any of the provided listeners.
/// Uses `select_all` to race all listeners concurrently, supporting any count.
async fn accept_any(listeners: &[TcpListener]) -> std::io::Result<(TcpStream, SocketAddr)> {
    if listeners.is_empty() {
        return Err(std::io::Error::other("No listeners"));
    }
    // Build a vec of pinned accept futures — one per listener.
    let futs: Vec<_> = listeners.iter().map(|l| Box::pin(l.accept())).collect();
    let (result, _index, _remaining) = select_all(futs).await;
    result
}

/// Pin thread to a specific CPU core.
fn pin_to_core(core_id: usize) {
    #[cfg(target_os = "linux")]
    {
        use std::mem;
        unsafe {
            let mut set: libc::cpu_set_t = mem::zeroed();
            libc::CPU_ZERO(&mut set);
            libc::CPU_SET(core_id % num_cpus::get(), &mut set);
            let result = libc::sched_setaffinity(0, mem::size_of::<libc::cpu_set_t>(), &set);
            if result == 0 {
                tracing::debug!(worker = core_id, "Pinned to core {}", core_id % num_cpus::get());
            } else {
                tracing::warn!(worker = core_id, "Failed to pin to core — continuing unpinned");
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = core_id;
    }
}

/// Cleanup connection tracking — exposed for testing.
pub(crate) fn cleanup_conn_pub(
    ip_tracker: &IpConnTracker,
    client_ip: std::net::IpAddr,
    active_conns: &ActiveConnCount,
    metrics: &Metrics,
) {
    cleanup_conn(ip_tracker, client_ip, active_conns, metrics);
}

/// Periodically log metrics. Stops cleanly when shutdown signal is received.
async fn metrics_reporter(metrics: Metrics, mut shutdown_rx: broadcast::Receiver<()>) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
    loop {
        tokio::select! {
            biased;

            _ = shutdown_rx.recv() => {
                tracing::debug!("Metrics reporter received shutdown signal");
                break;
            }

            _ = interval.tick() => {
                let snap = metrics.snapshot();
                tracing::info!("METRICS: {}", snap);
            }
        }
    }
}
