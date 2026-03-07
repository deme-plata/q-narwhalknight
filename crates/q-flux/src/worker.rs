use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use dashmap::DashMap;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, Semaphore};
use tokio_rustls::TlsAcceptor;
use tokio::io::AsyncWriteExt;

use futures::future::select_all;

use crate::access_log::AccessLogger;
use crate::acceptor::SharedTlsConfig;
use crate::config::FluxConfig;
use crate::h2_proxy;
use crate::health::HealthMap;
use crate::libp2p_aware::PeerTracker;
use crate::metrics::{Metrics, RateLimiter};
use crate::proxy;
use crate::upstream::UpstreamPool;
use crate::acceptor;

/// Per-IP connection counter shared across all workers.
pub type IpConnTracker = Arc<DashMap<std::net::IpAddr, u64>>;

/// Total active connections (shared).
pub type ActiveConnCount = Arc<AtomicU64>;

/// Max concurrent connection handlers per worker.
/// 48 workers × 8192 = 393,216 total concurrent handlers.
/// Each handler holds a reference to the shared upstream pool + ~8KB buffer.
/// At 393K concurrent × ~10KB = ~4GB. Scale further via config.
const MAX_HANDLERS_PER_WORKER: usize = 8192;

/// How often to garbage-collect the per-IP connection tracker (seconds).
/// IPs with 0 active connections are removed to prevent unbounded growth.
const IP_TRACKER_GC_INTERVAL_SECS: u64 = 60;

/// How often to clean up stale rate limiter buckets (seconds).
const RATE_LIMITER_GC_INTERVAL_SECS: u64 = 300;

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
) -> Vec<std::thread::JoinHandle<()>> {
    let worker_count = config.worker_count();
    let ip_tracker: IpConnTracker = Arc::new(DashMap::new());
    let active_conns: ActiveConnCount = Arc::new(AtomicU64::new(0));
    let mut handles = Vec::with_capacity(worker_count);

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
                        access_logger, rate_limiter,
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
    let static_config = Arc::new(config.static_files.clone());

    // CRITICAL: Create ONE UpstreamPool per worker, shared across all connections.
    // Previous bug: UpstreamPool::new() was called per-connection, creating a NEW
    // hyper Client each time. This defeated connection pooling — every request
    // opened a fresh TCP connection to upstream (same failure mode as keepalive=off).
    // Now: one hyper Client per worker with pooled keepalive connections to upstream.
    let upstream = Arc::new(UpstreamPool::new(&config.upstream, metrics.clone(), health_map.clone()));

    // PeerTracker: per-peer connection limits and circuit breakers for libp2p peers.
    // Shared across all connections on this worker. Pre-seeded with known infrastructure.
    let peer_tracker = Arc::new(PeerTracker::new(
        vec![
            "12D3KooWSBxw".to_string(),   // Beta bootstrap
            "12D3KooWFfZK".to_string(),   // Gamma bootstrap
            "12D3KooWPwin".to_string(),   // Alpha bootstrap
            "12D3KooWLJJR".to_string(),   // Delta bootstrap
        ],
        vec![
            "12D3KooWFpbX".to_string(),   // Epsilon 10Gbit supernode
        ],
    ));

    // Periodic PeerTracker stale-peer cleanup (every 60s, evict idle >5min)
    {
        let tracker = peer_tracker.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                tracker.cleanup_stale(std::time::Duration::from_secs(300));
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

        // Per-IP connection limit (concurrent connection cap)
        let client_ip = client_addr.ip();
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

        // Token-bucket rate limiting (per-IP request rate cap)
        if let Some(ref rl) = rate_limiter {
            if !rl.check(client_ip) {
                metrics.rate_limited();
                tracing::debug!(worker = worker_id, ip = %client_ip, "Token-bucket rate limited");
                // Decrement the connection counter we just incremented
                cleanup_conn(&ip_tracker, client_ip, &active_conns, &metrics);
                drop(tcp_stream);
                continue;
            }
        }

        active_conns.fetch_add(1, Ordering::Relaxed);
        metrics.conn_opened();

        // Determine if this is a TLS port (443) or plain HTTP (80)
        let local_port = tcp_stream.local_addr().map(|a| a.port()).unwrap_or(443);
        let is_tls = local_port == 443;

        // Hot-reload: load current TLS config per connection (read lock, ~10ns).
        // If certs were reloaded via admin API, new connections get the new config.
        let tls_acceptor = TlsAcceptor::from(shared_tls.load());
        let metrics = metrics.clone();
        let ip_tracker = ip_tracker.clone();
        let active_conns = active_conns.clone();
        let upstream = upstream.clone();
        let peer_tracker = peer_tracker.clone();
        let semaphore = handler_semaphore.clone();
        let static_config = static_config.clone();
        let access_logger = access_logger.clone();

        tokio::spawn(async move {
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

            if is_tls {
                let tls_result = tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    tls_acceptor.accept(tcp_stream),
                ).await;

                match tls_result {
                    Ok(Ok(tls_stream)) => {
                        metrics.tls_handshake_ok();

                        // ALPN-based protocol routing (Phase 3):
                        // If client negotiated "h2", handle via HTTP/2 multiplexed proxy.
                        // Otherwise, fall through to HTTP/1.1 proxy.
                        let is_h2 = tls_stream.get_ref().1
                            .alpn_protocol()
                            .map(|p| p == b"h2")
                            .unwrap_or(false);

                        if is_h2 {
                            h2_proxy::handle_h2_connection(
                                tls_stream, client_addr, upstream.clone(),
                                metrics.clone(), body_limit, static_config.clone(),
                                access_logger.clone(),
                            ).await;
                        } else {
                            // HTTP/1.1 path — use logged variant if access logger configured
                            match access_logger {
                                Some(ref logger) => {
                                    proxy::handle_connection_logged(
                                        tls_stream, client_addr, &upstream, &metrics,
                                        body_limit, &static_config, logger,
                                        &peer_tracker,
                                    ).await;
                                }
                                None => {
                                    proxy::handle_connection(
                                        tls_stream, client_addr, &upstream, &metrics,
                                        body_limit, &static_config,
                                        &peer_tracker,
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
                // Plain HTTP — read first line to get Host header, then redirect
                let mut tcp_stream = tcp_stream;
                let mut peek_buf = [0u8; 1024];
                let host = match tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    tokio::io::AsyncReadExt::read(&mut tcp_stream, &mut peek_buf),
                ).await {
                    Ok(Ok(n)) if n > 0 => {
                        extract_host_header(&peek_buf[..n])
                            .unwrap_or_else(|| client_addr.ip().to_string())
                    }
                    _ => client_addr.ip().to_string(),
                };

                let redirect = format!(
                    "HTTP/1.1 301 Moved Permanently\r\nlocation: https://{}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                    host
                );
                let _ = tcp_stream.write_all(redirect.as_bytes()).await;
            }

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
    active_conns.fetch_sub(1, Ordering::Relaxed);
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
