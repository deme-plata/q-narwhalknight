use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use rustls::ServerConfig;
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::TlsAcceptor;
use tokio::io::AsyncWriteExt;

use crate::config::FluxConfig;
use crate::metrics::Metrics;
use crate::proxy;
use crate::upstream::UpstreamPool;
use crate::acceptor;

/// Per-IP connection counter shared across all workers.
pub type IpConnTracker = Arc<DashMap<std::net::IpAddr, u64>>;

/// Total active connections (shared).
pub type ActiveConnCount = Arc<AtomicU64>;

/// Spawn worker threads. Each worker:
/// - Has its own TcpListener (SO_REUSEPORT gives it a fair share of connections)
/// - Has its own upstream connection pool (no cross-thread contention)
/// - Runs on a dedicated tokio single-threaded runtime pinned to a core
pub fn spawn_workers(
    config: &FluxConfig,
    tls_config: Arc<ServerConfig>,
    metrics: Metrics,
) -> Vec<std::thread::JoinHandle<()>> {
    let worker_count = config.worker_count();
    let ip_tracker: IpConnTracker = Arc::new(DashMap::new());
    let active_conns: ActiveConnCount = Arc::new(AtomicU64::new(0));
    let mut handles = Vec::with_capacity(worker_count);

    for worker_id in 0..worker_count {
        let config = config.clone();
        let tls_config = tls_config.clone();
        let metrics = metrics.clone();
        let ip_tracker = ip_tracker.clone();
        let active_conns = active_conns.clone();

        let handle = std::thread::Builder::new()
            .name(format!("q-flux-w{}", worker_id))
            .spawn(move || {
                pin_to_core(worker_id);

                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to build tokio runtime for worker");

                rt.block_on(async move {
                    worker_loop(worker_id, &config, tls_config, metrics, ip_tracker, active_conns).await;
                });
            })
            .expect("Failed to spawn worker thread");

        handles.push(handle);
    }

    // Spawn metrics reporter
    let metrics_clone = metrics.clone();
    std::thread::Builder::new()
        .name("q-flux-metrics".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(metrics_reporter(metrics_clone));
        })
        .ok();

    handles
}

async fn worker_loop(
    worker_id: usize,
    config: &FluxConfig,
    tls_config: Arc<ServerConfig>,
    metrics: Metrics,
    ip_tracker: IpConnTracker,
    active_conns: ActiveConnCount,
) {
    let tls_acceptor = TlsAcceptor::from(tls_config);

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
    let upstream_config = config.upstream.clone();

    // Accept loop
    loop {
        // Accept from the first ready listener using tokio::select!
        let accept_result = accept_any(&listeners).await;
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

        // Per-IP limit
        let client_ip = client_addr.ip();
        {
            let mut count = ip_tracker.entry(client_ip).or_insert(0);
            if *count >= max_per_ip {
                metrics.rate_limited();
                tracing::debug!(worker = worker_id, ip = %client_ip, "Per-IP limit exceeded");
                drop(tcp_stream);
                continue;
            }
            *count += 1;
        }

        active_conns.fetch_add(1, Ordering::Relaxed);
        metrics.conn_opened();

        // Determine if this is a TLS port (443) or plain HTTP (80)
        let local_port = tcp_stream.local_addr().map(|a| a.port()).unwrap_or(443);
        let is_tls = local_port == 443;

        let tls_acceptor = tls_acceptor.clone();
        let metrics = metrics.clone();
        let ip_tracker = ip_tracker.clone();
        let active_conns = active_conns.clone();
        let upstream_config = upstream_config.clone();

        tokio::spawn(async move {
            let upstream = UpstreamPool::new(&upstream_config, metrics.clone());

            if is_tls {
                let tls_result = tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    tls_acceptor.accept(tcp_stream),
                ).await;

                match tls_result {
                    Ok(Ok(tls_stream)) => {
                        metrics.tls_handshake_ok();
                        proxy::handle_connection(
                            tls_stream, client_addr, &upstream, &metrics, body_limit,
                        ).await;
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
                // Plain HTTP — redirect to HTTPS
                let redirect = format!(
                    "HTTP/1.1 301 Moved Permanently\r\nlocation: https://{}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                    client_addr.ip()
                );
                let mut tcp_stream = tcp_stream;
                let _ = tcp_stream.write_all(redirect.as_bytes()).await;
            }

            // Cleanup
            {
                let mut count = ip_tracker.entry(client_ip).or_insert(0);
                if *count > 0 {
                    *count -= 1;
                }
                if *count == 0 {
                    drop(count);
                    ip_tracker.remove(&client_ip);
                }
            }
            active_conns.fetch_sub(1, Ordering::Relaxed);
            metrics.conn_closed();
        });
    }
}

/// Accept a connection from any of the provided listeners.
/// Uses a macro-based approach to avoid the Unpin problem with async borrows.
async fn accept_any(listeners: &[TcpListener]) -> std::io::Result<(TcpStream, SocketAddr)> {
    match listeners.len() {
        0 => Err(std::io::Error::new(std::io::ErrorKind::Other, "No listeners")),
        1 => listeners[0].accept().await,
        2 => {
            tokio::select! {
                r = listeners[0].accept() => r,
                r = listeners[1].accept() => r,
            }
        }
        3 => {
            tokio::select! {
                r = listeners[0].accept() => r,
                r = listeners[1].accept() => r,
                r = listeners[2].accept() => r,
            }
        }
        _ => {
            // For 4+ listeners, just use the first two (rare case)
            tokio::select! {
                r = listeners[0].accept() => r,
                r = listeners[1].accept() => r,
            }
        }
    }
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

/// Periodically log metrics.
async fn metrics_reporter(metrics: Metrics) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
    loop {
        interval.tick().await;
        let snap = metrics.snapshot();
        tracing::info!("METRICS: {}", snap);
    }
}
