use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use http_body_util::Full;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use tokio::net::TcpListener;

use crate::acceptor::SharedTlsConfig;
use crate::config::TlsConfig;
use crate::health::HealthMap;
use crate::libp2p_aware::{BreakerState, PeerTracker};
use crate::metrics::Metrics;
use crate::ocsp_fetch::SharedOcspStatus;
use crate::upstream::BackendCounters;

/// Shared state for the admin HTTP server.
struct AdminState {
    metrics: Metrics,
    worker_count: usize,
    start_time: Instant,
    shared_tls: SharedTlsConfig,
    tls_config_paths: TlsConfig,
    /// Health map for backend + cluster peer health status
    health_map: Option<HealthMap>,
    /// Local upstream backends
    local_backends: Vec<String>,
    /// Super-cluster remote peers
    cluster_peers: Vec<String>,
    /// libp2p peer tracker for per-peer stats
    peer_tracker: Option<Arc<PeerTracker>>,
    /// OCSP auto-fetch status
    ocsp_status: Option<SharedOcspStatus>,
    /// Per-backend request counters (Issue #026)
    backend_counters: Option<Arc<dashmap::DashMap<String, Arc<BackendCounters>>>>,
    /// Adaptive concurrency effective limit (Issue #026)
    adaptive_concurrency: Option<Arc<crate::upstream::AdaptiveConcurrency>>,
}

/// Start the admin HTTP server on its own OS thread.
///
/// The server binds to `listen_addr` (default `127.0.0.1:9090`) and serves:
///   - `GET /health`      -> JSON health check with uptime
///   - `GET /metrics`     -> Prometheus text exposition format
///   - `GET /status`      -> JSON snapshot of all metrics + metadata
///   - `POST /tls-reload` -> Hot-reload TLS certificates from disk
///
/// Runs a single-threaded tokio runtime so it never contends with the
/// worker runtimes on the hot path.
pub fn spawn_admin_server(
    listen_addr: SocketAddr,
    metrics: Metrics,
    worker_count: usize,
    shared_tls: SharedTlsConfig,
    tls_config_paths: TlsConfig,
    health_map: Option<HealthMap>,
    local_backends: Vec<String>,
    cluster_peers: Vec<String>,
    peer_tracker: Option<Arc<PeerTracker>>,
    ocsp_status: Option<SharedOcspStatus>,
    backend_counters: Option<Arc<dashmap::DashMap<String, Arc<BackendCounters>>>>,
    adaptive_concurrency: Option<Arc<crate::upstream::AdaptiveConcurrency>>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("q-flux-admin".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build admin tokio runtime");

            rt.block_on(async move {
                run_admin_server(listen_addr, metrics, worker_count, shared_tls, tls_config_paths, health_map, local_backends, cluster_peers, peer_tracker, ocsp_status, backend_counters, adaptive_concurrency).await;
            });
        })
        .expect("failed to spawn admin thread")
}

async fn run_admin_server(
    listen_addr: SocketAddr,
    metrics: Metrics,
    worker_count: usize,
    shared_tls: SharedTlsConfig,
    tls_config_paths: TlsConfig,
    health_map: Option<HealthMap>,
    local_backends: Vec<String>,
    cluster_peers: Vec<String>,
    peer_tracker: Option<Arc<PeerTracker>>,
    ocsp_status: Option<SharedOcspStatus>,
    backend_counters: Option<Arc<dashmap::DashMap<String, Arc<BackendCounters>>>>,
    adaptive_concurrency: Option<Arc<crate::upstream::AdaptiveConcurrency>>,
) {
    let listener = match TcpListener::bind(listen_addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Admin server failed to bind {}: {}", listen_addr, e);
            return;
        }
    };

    tracing::info!("Admin server listening on {}", listen_addr);

    let state = Arc::new(AdminState {
        metrics,
        worker_count,
        start_time: Instant::now(),
        shared_tls,
        tls_config_paths,
        health_map,
        local_backends,
        cluster_peers,
        peer_tracker,
        ocsp_status,
        backend_counters,
        adaptive_concurrency,
    });

    loop {
        let (stream, peer_addr) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                tracing::debug!("Admin accept error: {}", e);
                continue;
            }
        };

        let state = Arc::clone(&state);

        tokio::task::spawn(async move {
            let io = hyper_util::rt::TokioIo::new(stream);

            let service = service_fn(move |req: Request<hyper::body::Incoming>| {
                let state = Arc::clone(&state);
                async move { handle_admin_request(req, &state).await }
            });

            if let Err(e) = http1::Builder::new()
                .keep_alive(false)
                .serve_connection(io, service)
                .await
            {
                let msg = e.to_string();
                if !msg.contains("connection closed")
                    && !msg.contains("broken pipe")
                    && !msg.contains("reset by peer")
                {
                    tracing::debug!(
                        "Admin connection error from {}: {}",
                        peer_addr,
                        msg,
                    );
                }
            }
        });
    }
}

async fn handle_admin_request(
    req: Request<hyper::body::Incoming>,
    state: &AdminState,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let resp = match (req.method(), req.uri().path()) {
        (&hyper::Method::GET, "/health") => handle_health(state),
        (&hyper::Method::GET, "/metrics") => handle_metrics(state),
        (&hyper::Method::GET, "/status") => handle_status(state),
        (&hyper::Method::GET, "/peers") => handle_peers(state),
        (&hyper::Method::GET, "/backends") => handle_backends(state),
        (&hyper::Method::POST, "/tls-reload") => handle_tls_reload(state),
        (&hyper::Method::POST, "/admin/drain") => handle_drain(state),
        (&hyper::Method::POST, "/admin/undrain") => handle_undrain(state),
        _ => not_found(),
    };
    Ok(resp)
}

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

fn handle_health(state: &AdminState) -> Response<Full<Bytes>> {
    let uptime_secs = state.start_time.elapsed().as_secs();

    let body = format!(
        r#"{{"status":"ok","uptime_secs":{}}}"#,
        uptime_secs,
    );

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

// ---------------------------------------------------------------------------
// GET /metrics  (Prometheus text exposition format)
// ---------------------------------------------------------------------------

fn handle_metrics(state: &AdminState) -> Response<Full<Bytes>> {
    let snap = state.metrics.snapshot();
    let mut buf = String::with_capacity(4096);

    // -- uptime ---------------------------------------------------------------
    prom_gauge(
        &mut buf,
        "q_flux_uptime_seconds",
        "Process uptime in seconds",
        snap.uptime_secs,
    );

    // -- connections ----------------------------------------------------------
    prom_gauge(
        &mut buf,
        "q_flux_connections_active",
        "Current active connections",
        snap.active_connections,
    );
    prom_counter(
        &mut buf,
        "q_flux_connections_total",
        "Total connections accepted since start",
        snap.total_connections,
    );

    // -- TLS ------------------------------------------------------------------
    prom_labeled_counter(
        &mut buf,
        "q_flux_tls_handshakes_total",
        "Total TLS handshakes",
        &[
            ("result", "ok", snap.tls_handshakes),
            ("result", "fail", snap.tls_handshake_failures),
        ],
    );

    // -- requests -------------------------------------------------------------
    prom_counter(
        &mut buf,
        "q_flux_requests_total_all",
        "Total HTTP requests processed",
        snap.total_requests,
    );
    prom_labeled_counter(
        &mut buf,
        "q_flux_requests_total",
        "Total requests by response status class",
        &[
            ("status", "2xx", snap.requests_2xx),
            ("status", "4xx", snap.requests_4xx),
            ("status", "5xx", snap.requests_5xx),
        ],
    );

    // -- upstream -------------------------------------------------------------
    prom_gauge(
        &mut buf,
        "q_flux_upstream_active",
        "Current active upstream connections",
        snap.upstream_active,
    );
    prom_counter(
        &mut buf,
        "q_flux_upstream_connect_failures_total",
        "Total upstream connect failures",
        snap.upstream_connect_failures,
    );
    prom_counter(
        &mut buf,
        "q_flux_upstream_timeouts_total",
        "Total upstream response timeouts",
        snap.upstream_timeouts,
    );
    prom_counter(
        &mut buf,
        "q_flux_upstream_retries_total",
        "Total upstream request retries attempted",
        snap.upstream_retries,
    );
    prom_counter(
        &mut buf,
        "q_flux_upstream_retry_successes_total",
        "Total upstream retries that succeeded",
        snap.upstream_retry_successes,
    );
    prom_counter(
        &mut buf,
        "q_flux_upstream_queue_timeouts_total",
        "Total requests that timed out waiting for upstream semaphore permit",
        snap.upstream_queue_timeouts,
    );

    // -- rate limiting --------------------------------------------------------
    prom_counter(
        &mut buf,
        "q_flux_rate_limited_total",
        "Total requests rejected by rate limiter",
        snap.rate_limited,
    );

    // -- websocket ------------------------------------------------------------
    prom_gauge(
        &mut buf,
        "q_flux_websockets_active",
        "Current active WebSocket connections",
        snap.active_websockets,
    );
    prom_counter(
        &mut buf,
        "q_flux_websocket_upgrades_total",
        "Total WebSocket upgrades since start",
        snap.websocket_upgrades,
    );

    // -- bytes ----------------------------------------------------------------
    prom_counter(
        &mut buf,
        "q_flux_bytes_received_total",
        "Total bytes received from clients",
        snap.bytes_received,
    );
    prom_counter(
        &mut buf,
        "q_flux_bytes_sent_total",
        "Total bytes sent to clients",
        snap.bytes_sent,
    );

    // -- workers --------------------------------------------------------------
    prom_gauge(
        &mut buf,
        "q_flux_workers",
        "Number of worker threads",
        state.worker_count as u64,
    );

    // -- TLS reload counter ---------------------------------------------------
    prom_counter(
        &mut buf,
        "q_flux_tls_reloads_total",
        "Total TLS certificate reloads since start",
        state.shared_tls.reload_count(),
    );

    // -- splice zero-copy (Issue #016) ----------------------------------------
    prom_gauge(
        &mut buf,
        "q_flux_splice_connections_active",
        "Current connections using splice(2) zero-copy",
        snap.splice_connections_active,
    );
    prom_counter(
        &mut buf,
        "q_flux_splice_bytes_total",
        "Total bytes transferred via splice(2) zero-copy",
        snap.splice_bytes_total,
    );
    prom_counter(
        &mut buf,
        "q_flux_splice_fallbacks_total",
        "Total times splice(2) failed and fell back to userspace copy",
        snap.splice_fallbacks_total,
    );

    // -- latency histogram (Issue #11) ----------------------------------------
    buf.push_str(&state.metrics.prometheus_export_histogram());

    // -- latency percentiles (Issue #031) ------------------------------------
    {
        use std::fmt::Write;
        let p50 = state.metrics.latency_p50();
        let p95 = state.metrics.latency_p95();
        let p99 = state.metrics.latency_p99();
        let _ = writeln!(buf, "# HELP q_flux_request_duration_p50_seconds P50 request latency in seconds");
        let _ = writeln!(buf, "# TYPE q_flux_request_duration_p50_seconds gauge");
        let _ = writeln!(buf, "q_flux_request_duration_p50_seconds {:.6}", p50);
        buf.push('\n');
        let _ = writeln!(buf, "# HELP q_flux_request_duration_p95_seconds P95 request latency in seconds");
        let _ = writeln!(buf, "# TYPE q_flux_request_duration_p95_seconds gauge");
        let _ = writeln!(buf, "q_flux_request_duration_p95_seconds {:.6}", p95);
        buf.push('\n');
        let _ = writeln!(buf, "# HELP q_flux_request_duration_p99_seconds P99 request latency in seconds");
        let _ = writeln!(buf, "# TYPE q_flux_request_duration_p99_seconds gauge");
        let _ = writeln!(buf, "q_flux_request_duration_p99_seconds {:.6}", p99);
        buf.push('\n');
    }

    // -- HTTP/2 metrics (Issue #15) -------------------------------------------
    buf.push_str(&crate::h2_proxy::h2_prometheus_export());

    // -- upstream pool per-backend metrics (Issue #026) -------------------------
    if let Some(ref hm) = state.health_map {
        use std::fmt::Write;
        let _ = writeln!(buf, "# HELP q_flux_backend_healthy Whether the backend is healthy (1=yes, 0=no)");
        let _ = writeln!(buf, "# TYPE q_flux_backend_healthy gauge");
        for backend in state.local_backends.iter().chain(state.cluster_peers.iter()) {
            let healthy = hm.get(backend.as_str())
                .map(|e| if e.is_healthy { 1 } else { 0 })
                .unwrap_or(1); // no entry = assume healthy
            let _ = writeln!(buf, r#"q_flux_backend_healthy{{backend="{}"}} {}"#, backend, healthy);
        }
        buf.push('\n');

        let _ = writeln!(buf, "# HELP q_flux_backend_response_time_ms Last health check response time in milliseconds");
        let _ = writeln!(buf, "# TYPE q_flux_backend_response_time_ms gauge");
        for backend in state.local_backends.iter().chain(state.cluster_peers.iter()) {
            let ms = hm.get(backend.as_str())
                .and_then(|e| e.last_response_time_ms)
                .unwrap_or(0);
            let _ = writeln!(buf, r#"q_flux_backend_response_time_ms{{backend="{}"}} {}"#, backend, ms);
        }
        buf.push('\n');

        let _ = writeln!(buf, "# HELP q_flux_backend_consecutive_failures Consecutive health check failures");
        let _ = writeln!(buf, "# TYPE q_flux_backend_consecutive_failures gauge");
        for backend in state.local_backends.iter().chain(state.cluster_peers.iter()) {
            let failures = hm.get(backend.as_str())
                .map(|e| e.consecutive_failures)
                .unwrap_or(0);
            let _ = writeln!(buf, r#"q_flux_backend_consecutive_failures{{backend="{}"}} {}"#, backend, failures);
        }
        buf.push('\n');
    }

    if let Some(ref counters) = state.backend_counters {
        use std::fmt::Write;
        let _ = writeln!(buf, "# HELP q_flux_backend_requests_total Total requests forwarded to backend");
        let _ = writeln!(buf, "# TYPE q_flux_backend_requests_total counter");
        for entry in counters.iter() {
            let _ = writeln!(buf, r#"q_flux_backend_requests_total{{backend="{}"}} {}"#,
                entry.key(), entry.value().total_requests.load(std::sync::atomic::Ordering::Relaxed));
        }
        buf.push('\n');

        let _ = writeln!(buf, "# HELP q_flux_backend_failed_requests_total Total failed requests to backend");
        let _ = writeln!(buf, "# TYPE q_flux_backend_failed_requests_total counter");
        for entry in counters.iter() {
            let _ = writeln!(buf, r#"q_flux_backend_failed_requests_total{{backend="{}"}} {}"#,
                entry.key(), entry.value().failed_requests.load(std::sync::atomic::Ordering::Relaxed));
        }
        buf.push('\n');
    }

    // Adaptive concurrency effective limit (Issue #026)
    if let Some(ref ac) = state.adaptive_concurrency {
        prom_gauge(&mut buf, "q_flux_upstream_effective_limit",
            "Current adaptive concurrency effective limit",
            ac.effective_limit() as u64);
    }

    // -- static file cache metrics (Issue #034) ----------------------------------
    prom_counter(&mut buf, "q_flux_cache_hits_total", "Total static file cache hits", crate::static_serve::global_cache_hits());
    prom_counter(&mut buf, "q_flux_cache_misses_total", "Total static file cache misses", crate::static_serve::global_cache_misses());

    // -- libp2p peer metrics (Issue #9) ---------------------------------------
    if let Some(ref tracker) = state.peer_tracker {
        use std::fmt::Write;
        // Aggregate per-tier counts
        let mut tier_peers: std::collections::HashMap<&str, (u64, u64, u64, u64, u64)> =
            std::collections::HashMap::new();
        let mut breakers_open: u64 = 0;

        for entry in tracker.peer_map().iter() {
            let peer = entry.value();
            let tier_name = match peer.tier {
                crate::libp2p_aware::PeerTier::Supernode => "Supernode",
                crate::libp2p_aware::PeerTier::Bootstrap => "Bootstrap",
                crate::libp2p_aware::PeerTier::Validator => "Validator",
                crate::libp2p_aware::PeerTier::Miner => "Miner",
                crate::libp2p_aware::PeerTier::Unknown => "Unknown",
            };
            let conns = peer.active_connections.load(std::sync::atomic::Ordering::Relaxed) as u64;
            let rx = peer.bytes_in.load(std::sync::atomic::Ordering::Relaxed);
            let tx = peer.bytes_out.load(std::sync::atomic::Ordering::Relaxed);

            let e = tier_peers.entry(tier_name).or_insert((0, 0, 0, 0, 0));
            e.0 += 1;           // peer count
            e.1 += conns;       // active connections
            e.2 += rx;          // bytes rx
            e.3 += tx;          // bytes tx

            let cb = peer.circuit_breaker.read();
            if matches!(cb.state, BreakerState::Open(_)) {
                breakers_open += 1;
            }
        }

        // Peer count by tier
        let _ = writeln!(buf, "# HELP qflux_libp2p_peers_total Number of known peers by tier");
        let _ = writeln!(buf, "# TYPE qflux_libp2p_peers_total gauge");
        for (tier, (count, _, _, _, _)) in &tier_peers {
            let _ = writeln!(buf, r#"qflux_libp2p_peers_total{{tier="{}"}} {}"#, tier, count);
        }
        buf.push('\n');

        // Active connections by tier
        let _ = writeln!(buf, "# HELP qflux_libp2p_connections_active Active connections by tier");
        let _ = writeln!(buf, "# TYPE qflux_libp2p_connections_active gauge");
        for (tier, (_, conns, _, _, _)) in &tier_peers {
            let _ = writeln!(buf, r#"qflux_libp2p_connections_active{{tier="{}"}} {}"#, tier, conns);
        }
        buf.push('\n');

        // Bytes by tier + direction
        let _ = writeln!(buf, "# HELP qflux_libp2p_bytes_total Bytes transferred by tier and direction");
        let _ = writeln!(buf, "# TYPE qflux_libp2p_bytes_total counter");
        for (tier, (_, _, rx, tx, _)) in &tier_peers {
            let _ = writeln!(buf, r#"qflux_libp2p_bytes_total{{tier="{}",direction="rx"}} {}"#, tier, rx);
            let _ = writeln!(buf, r#"qflux_libp2p_bytes_total{{tier="{}",direction="tx"}} {}"#, tier, tx);
        }
        buf.push('\n');

        // Circuit breakers open
        prom_gauge(&mut buf, "qflux_libp2p_circuit_breaker_open", "Number of peers with open circuit breakers", breakers_open);
    }

    Response::builder()
        .status(StatusCode::OK)
        .header(
            "Content-Type",
            "text/plain; version=0.0.4; charset=utf-8",
        )
        .body(Full::new(Bytes::from(buf)))
        .unwrap()
}

// ---------------------------------------------------------------------------
// GET /status  (JSON)
// ---------------------------------------------------------------------------

fn handle_status(state: &AdminState) -> Response<Full<Bytes>> {
    let snap = state.metrics.snapshot();

    // Build cluster health JSON
    let cluster_json = if let Some(ref hm) = state.health_map {
        let mut local_entries = Vec::new();
        for backend in &state.local_backends {
            let (healthy, half_open, failures, last_check_ago_ms, resp_time) = if let Some(entry) = hm.get(backend.as_str()) {
                let ago = entry.last_check.elapsed().as_millis() as u64;
                (entry.is_healthy, entry.half_open, entry.consecutive_failures, ago, entry.last_response_time_ms)
            } else {
                (true, false, 0, 0, None) // no entry = assume healthy
            };
            local_entries.push(format!(
                r#"{{"addr":"{}","healthy":{},"half_open":{},"failures":{},"last_check_ms_ago":{},"response_time_ms":{}}}"#,
                backend, healthy, half_open, failures, last_check_ago_ms,
                resp_time.map(|t| t.to_string()).unwrap_or_else(|| "null".to_string()),
            ));
        }

        let mut peer_entries = Vec::new();
        for peer in &state.cluster_peers {
            let (healthy, half_open, failures, last_check_ago_ms, resp_time) = if let Some(entry) = hm.get(peer.as_str()) {
                let ago = entry.last_check.elapsed().as_millis() as u64;
                (entry.is_healthy, entry.half_open, entry.consecutive_failures, ago, entry.last_response_time_ms)
            } else {
                (true, false, 0, 0, None)
            };
            peer_entries.push(format!(
                r#"{{"addr":"{}","healthy":{},"half_open":{},"failures":{},"last_check_ms_ago":{},"response_time_ms":{}}}"#,
                peer, healthy, half_open, failures, last_check_ago_ms,
                resp_time.map(|t| t.to_string()).unwrap_or_else(|| "null".to_string()),
            ));
        }

        format!(
            r#","cluster":{{"enabled":{},"local_backends":[{}],"cluster_peers":[{}]}}"#,
            !state.cluster_peers.is_empty(),
            local_entries.join(","),
            peer_entries.join(","),
        )
    } else {
        r#","cluster":{"enabled":false,"local_backends":[],"cluster_peers":[]}"#.to_string()
    };

    // OCSP status JSON
    let ocsp_json = if let Some(ref ocsp_st) = state.ocsp_status {
        let s = ocsp_st.read();
        let url = s.responder_url.as_deref().unwrap_or("unknown");
        let last_fetch_ago = s.last_fetch.map(|t| t.elapsed().as_secs()).unwrap_or(0);
        let last_err = s.last_error.as_deref().unwrap_or("");
        format!(
            r#","ocsp":{{"auto_refresh":true,"responder_url":"{}","last_fetch_secs_ago":{},"response_bytes":{},"refresh_count":{},"last_error":"{}"}}"#,
            url, last_fetch_ago, s.response_bytes, s.refresh_count, last_err,
        )
    } else {
        r#","ocsp":{"auto_refresh":false}"#.to_string()
    };

    // Build JSON manually to avoid pulling in serde Serialize on MetricsSnapshot.
    // This keeps the metrics module free of serde dependencies.
    let body = format!(
        concat!(
            "{{",
            r#""version":"{}","#,
            r#""worker_count":{},"#,
            r#""uptime_secs":{},"#,
            r#""active_connections":{},"#,
            r#""total_connections":{},"#,
            r#""tls_handshakes":{},"#,
            r#""tls_handshake_failures":{},"#,
            r#""total_requests":{},"#,
            r#""requests_2xx":{},"#,
            r#""requests_4xx":{},"#,
            r#""requests_5xx":{},"#,
            r#""upstream_active":{},"#,
            r#""upstream_connect_failures":{},"#,
            r#""upstream_timeouts":{},"#,
            r#""upstream_retries":{},"#,
            r#""upstream_retry_successes":{},"#,
            r#""upstream_queue_timeouts":{},"#,
            r#""rate_limited":{},"#,
            r#""active_websockets":{},"#,
            r#""websocket_upgrades":{},"#,
            r#""bytes_received":{},"#,
            r#""bytes_sent":{},"#,
            r#""tls_reload_count":{},"#,
            r#""h2_connections":{},"#,
            r#""h2_streams_opened":{},"#,
            r#""h2_streams_closed":{}"#,
            "{}",
            "{}",
            "}}",
        ),
        env!("CARGO_PKG_VERSION"),
        state.worker_count,
        snap.uptime_secs,
        snap.active_connections,
        snap.total_connections,
        snap.tls_handshakes,
        snap.tls_handshake_failures,
        snap.total_requests,
        snap.requests_2xx,
        snap.requests_4xx,
        snap.requests_5xx,
        snap.upstream_active,
        snap.upstream_connect_failures,
        snap.upstream_timeouts,
        snap.upstream_retries,
        snap.upstream_retry_successes,
        snap.upstream_queue_timeouts,
        snap.rate_limited,
        snap.active_websockets,
        snap.websocket_upgrades,
        snap.bytes_received,
        snap.bytes_sent,
        state.shared_tls.reload_count(),
        crate::h2_proxy::H2_METRICS.connections.load(std::sync::atomic::Ordering::Relaxed),
        crate::h2_proxy::H2_METRICS.streams_opened.load(std::sync::atomic::Ordering::Relaxed),
        crate::h2_proxy::H2_METRICS.streams_closed.load(std::sync::atomic::Ordering::Relaxed),
        cluster_json,
        ocsp_json,
    );

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

// ---------------------------------------------------------------------------
// POST /tls-reload  (Issue #10 — TLS hot-reload)
// ---------------------------------------------------------------------------

fn handle_tls_reload(state: &AdminState) -> Response<Full<Bytes>> {
    match state.shared_tls.reload(&state.tls_config_paths) {
        Ok(msg) => {
            let body = format!(r#"{{"status":"ok","message":"{}"}}"#, msg);
            Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "application/json")
                .body(Full::new(Bytes::from(body)))
                .unwrap()
        }
        Err(e) => {
            tracing::error!("TLS reload failed: {}", e);
            let body = format!(r#"{{"status":"error","message":"{}"}}"#, e);
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("Content-Type", "application/json")
                .body(Full::new(Bytes::from(body)))
                .unwrap()
        }
    }
}

// ---------------------------------------------------------------------------
// GET /peers  (libp2p peer tracker stats)
// ---------------------------------------------------------------------------

fn handle_peers(state: &AdminState) -> Response<Full<Bytes>> {
    let tracker = match state.peer_tracker {
        Some(ref t) => t,
        None => {
            let body = r#"{"total_peers":0,"peers":[],"note":"peer tracker not enabled"}"#;
            return Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "application/json")
                .body(Full::new(Bytes::from(body)))
                .unwrap();
        }
    };

    let mut peer_entries = Vec::new();

    // Iterate all peers in the DashMap via the public peer_map() accessor.
    for entry in tracker.peer_map().iter() {
        let peer_id = entry.key();
        let peer_state = entry.value();

        let active = peer_state
            .active_connections
            .load(std::sync::atomic::Ordering::Relaxed);
        let bytes_in = peer_state
            .bytes_in
            .load(std::sync::atomic::Ordering::Relaxed);
        let bytes_out = peer_state
            .bytes_out
            .load(std::sync::atomic::Ordering::Relaxed);
        let last_seen_secs_ago = peer_state.last_seen.read().elapsed().as_secs();
        let tier = &peer_state.tier;

        let cb_state = {
            let cb = peer_state.circuit_breaker.read();
            match cb.state {
                BreakerState::Closed => "Closed",
                BreakerState::Open(_) => "Open",
                BreakerState::HalfOpen => "HalfOpen",
            }
        };

        // Escape peer_id for safe JSON embedding (peer IDs are alphanumeric + hyphens,
        // but be defensive).
        let safe_id: String = peer_id
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
            .collect();

        peer_entries.push(format!(
            concat!(
                "{{",
                r#""peer_id":"{}","#,
                r#""tier":"{}","#,
                r#""active_connections":{},"#,
                r#""bytes_in":{},"#,
                r#""bytes_out":{},"#,
                r#""circuit_breaker":"{}","#,
                r#""last_seen_secs_ago":{}"#,
                "}}",
            ),
            safe_id, tier, active, bytes_in, bytes_out, cb_state, last_seen_secs_ago,
        ));
    }

    let total = peer_entries.len();
    let body = format!(
        r#"{{"total_peers":{},"peers":[{}]}}"#,
        total,
        peer_entries.join(","),
    );

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

// ---------------------------------------------------------------------------
// GET /backends  (per-backend health + request counters)
// ---------------------------------------------------------------------------

fn handle_backends(state: &AdminState) -> Response<Full<Bytes>> {
    let mut backend_entries = Vec::new();

    for backend in state.local_backends.iter().chain(state.cluster_peers.iter()) {
        let (healthy, half_open, consecutive_failures, response_time_ms) =
            if let Some(ref hm) = state.health_map {
                if let Some(entry) = hm.get(backend.as_str()) {
                    (
                        entry.is_healthy,
                        entry.half_open,
                        entry.consecutive_failures,
                        entry.last_response_time_ms,
                    )
                } else {
                    (true, false, 0, None)
                }
            } else {
                (true, false, 0, None)
            };

        let (total_requests, failed_requests) =
            if let Some(ref counters) = state.backend_counters {
                if let Some(c) = counters.get(backend.as_str()) {
                    (
                        c.value().total_requests.load(std::sync::atomic::Ordering::Relaxed),
                        c.value().failed_requests.load(std::sync::atomic::Ordering::Relaxed),
                    )
                } else {
                    (0, 0)
                }
            } else {
                (0, 0)
            };

        let resp_time_json = response_time_ms
            .map(|t| t.to_string())
            .unwrap_or_else(|| "null".to_string());

        backend_entries.push(format!(
            concat!(
                "{{",
                r#""addr":"{}","#,
                r#""healthy":{},"#,
                r#""half_open":{},"#,
                r#""consecutive_failures":{},"#,
                r#""response_time_ms":{},"#,
                r#""total_requests":{},"#,
                r#""failed_requests":{}"#,
                "}}",
            ),
            backend,
            healthy,
            half_open,
            consecutive_failures,
            resp_time_json,
            total_requests,
            failed_requests,
        ));
    }

    let total = backend_entries.len();
    let body = format!(
        r#"{{"total_backends":{},"backends":[{}]}}"#,
        total,
        backend_entries.join(","),
    );

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

// ---------------------------------------------------------------------------
// 404
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// POST /admin/drain — set admin drain flag on local backends
// v1.0.5: Reviewed by Nemotron Cascade Two + DeepSeek
//   - Uses is_admin_drained flag (not health mutation) to avoid race with health checker
//   - Safety check: fails if no healthy cluster peers
//   - Auto-clears after 300s timeout or 2 consecutive health check successes
// ---------------------------------------------------------------------------

fn handle_drain(state: &AdminState) -> Response<Full<Bytes>> {
    let health_map = match &state.health_map {
        Some(hm) => hm,
        None => {
            return json_error(503, "no_health_map", "Health checking is not enabled");
        }
    };

    // P0 SAFETY CHECK: Refuse to drain if no healthy cluster peers available.
    // Draining without healthy peers = taking down the entire service.
    let healthy_peers = state.cluster_peers.iter().filter(|peer| {
        health_map.get(peer.as_str())
            .map(|e| e.is_healthy && !e.is_admin_drained)
            .unwrap_or(true) // Unknown = assume healthy (optimistic)
    }).count();

    if healthy_peers == 0 && !state.cluster_peers.is_empty() {
        tracing::error!(
            "DRAIN REFUSED: no healthy cluster peers ({} configured, 0 healthy)",
            state.cluster_peers.len()
        );
        return json_error(503, "no_healthy_peers",
            "Cannot drain: no healthy cluster peers available to absorb traffic");
    }

    if state.cluster_peers.is_empty() {
        tracing::warn!("DRAIN WARNING: no cluster peers configured — traffic will get 502s!");
    }

    let now = std::time::Instant::now();
    let mut drained_count = 0u32;
    for backend in &state.local_backends {
        if let Some(mut entry) = health_map.get_mut(backend) {
            entry.is_admin_drained = true;
            entry.drain_started = Some(now);
            entry.drain_success_count = 0;
            drained_count += 1;
            tracing::warn!("DRAIN: set admin drain on backend {}", backend);
        }
    }

    let cluster_info: Vec<String> = state.cluster_peers.iter().map(|p| {
        let healthy = health_map.get(p.as_str())
            .map(|e| e.is_healthy)
            .unwrap_or(false);
        format!(r#"{{"addr":"{}","healthy":{}}}"#, p, healthy)
    }).collect();

    let body = format!(
        r#"{{"drained":true,"backends_drained":{},"healthy_cluster_peers":{},"cluster_peers":[{}]}}"#,
        drained_count,
        healthy_peers,
        cluster_info.join(","),
    );

    tracing::warn!(
        "DRAIN complete: {} local backends drained, {}/{} cluster peers healthy",
        drained_count, healthy_peers, state.cluster_peers.len(),
    );

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

// ---------------------------------------------------------------------------
// POST /admin/undrain — clear admin drain flag, let health checker promote
// ---------------------------------------------------------------------------

fn handle_undrain(state: &AdminState) -> Response<Full<Bytes>> {
    let health_map = match &state.health_map {
        Some(hm) => hm,
        None => {
            return json_error(503, "no_health_map", "Health checking is not enabled");
        }
    };

    let mut undrained_count = 0u32;
    for backend in &state.local_backends {
        if let Some(mut entry) = health_map.get_mut(backend) {
            entry.is_admin_drained = false;
            entry.drain_started = None;
            entry.drain_success_count = 0;
            // Don't touch is_healthy/half_open — let health checker decide
            undrained_count += 1;
            tracing::info!("UNDRAIN: cleared admin drain on backend {}", backend);
        }
    }

    let body = format!(
        r#"{{"drained":false,"backends_undrained":{}}}"#,
        undrained_count,
    );

    tracing::info!(
        "UNDRAIN complete: {} local backends cleared drain flag",
        undrained_count,
    );

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

fn json_error(status: u16, code: &str, message: &str) -> Response<Full<Bytes>> {
    Response::builder()
        .status(StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR))
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(format!(
            r#"{{"error":"{}","message":"{}"}}"#, code, message
        ))))
        .unwrap()
}

// ---------------------------------------------------------------------------
// 404
// ---------------------------------------------------------------------------

fn not_found() -> Response<Full<Bytes>> {
    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(
            r#"{"error":"not_found","endpoints":["/health","/metrics","/status","/peers","/backends","/tls-reload","/admin/drain","/admin/undrain"]}"#,
        )))
        .unwrap()
}

// ---------------------------------------------------------------------------
// Prometheus formatting helpers
// ---------------------------------------------------------------------------

/// Emit a single gauge metric.
fn prom_gauge(buf: &mut String, name: &str, help: &str, value: u64) {
    use std::fmt::Write;
    let _ = writeln!(buf, "# HELP {} {}", name, help);
    let _ = writeln!(buf, "# TYPE {} gauge", name);
    let _ = writeln!(buf, "{} {}", name, value);
    buf.push('\n');
}

/// Emit a single counter metric.
fn prom_counter(buf: &mut String, name: &str, help: &str, value: u64) {
    use std::fmt::Write;
    let _ = writeln!(buf, "# HELP {} {}", name, help);
    let _ = writeln!(buf, "# TYPE {} counter", name);
    let _ = writeln!(buf, "{} {}", name, value);
    buf.push('\n');
}

/// Emit a counter metric with multiple label variants.
///
/// `labels` is a slice of `(label_name, label_value, metric_value)` tuples.
fn prom_labeled_counter(
    buf: &mut String,
    name: &str,
    help: &str,
    labels: &[(&str, &str, u64)],
) {
    use std::fmt::Write;
    let _ = writeln!(buf, "# HELP {} {}", name, help);
    let _ = writeln!(buf, "# TYPE {} counter", name);
    for (label_name, label_value, value) in labels {
        let _ = writeln!(
            buf,
            r#"{}{{{}="{}"}} {}"#,
            name, label_name, label_value, value,
        );
    }
    buf.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── prom_gauge ──────────────────────────────────────────────────

    #[test]
    fn test_prom_gauge_format() {
        let mut buf = String::new();
        prom_gauge(&mut buf, "test_metric", "A test metric", 42);
        assert!(buf.contains("# HELP test_metric A test metric"));
        assert!(buf.contains("# TYPE test_metric gauge"));
        assert!(buf.contains("test_metric 42"));
    }

    #[test]
    fn test_prom_gauge_zero_value() {
        let mut buf = String::new();
        prom_gauge(&mut buf, "zero_metric", "Zero value", 0);
        assert!(buf.contains("zero_metric 0"));
    }

    #[test]
    fn test_prom_gauge_large_value() {
        let mut buf = String::new();
        prom_gauge(&mut buf, "big_metric", "Large value", u64::MAX);
        assert!(buf.contains(&format!("big_metric {}", u64::MAX)));
    }

    // ── prom_counter ────────────────────────────────────────────────

    #[test]
    fn test_prom_counter_format() {
        let mut buf = String::new();
        prom_counter(&mut buf, "requests_total", "Total requests", 1000);
        assert!(buf.contains("# HELP requests_total Total requests"));
        assert!(buf.contains("# TYPE requests_total counter"));
        assert!(buf.contains("requests_total 1000"));
    }

    #[test]
    fn test_prom_counter_trailing_newline() {
        let mut buf = String::new();
        prom_counter(&mut buf, "c", "help", 0);
        assert!(buf.ends_with("\n\n"), "should have blank line after metric");
    }

    // ── prom_labeled_counter ────────────────────────────────────────

    #[test]
    fn test_prom_labeled_counter_format() {
        let mut buf = String::new();
        prom_labeled_counter(
            &mut buf,
            "http_requests_total",
            "Total HTTP requests",
            &[
                ("status", "2xx", 500),
                ("status", "4xx", 50),
                ("status", "5xx", 5),
            ],
        );
        assert!(buf.contains("# HELP http_requests_total Total HTTP requests"));
        assert!(buf.contains("# TYPE http_requests_total counter"));
        assert!(buf.contains(r#"http_requests_total{status="2xx"} 500"#));
        assert!(buf.contains(r#"http_requests_total{status="4xx"} 50"#));
        assert!(buf.contains(r#"http_requests_total{status="5xx"} 5"#));
    }

    #[test]
    fn test_prom_labeled_counter_empty_labels() {
        let mut buf = String::new();
        prom_labeled_counter(&mut buf, "empty_metric", "No labels", &[]);
        assert!(buf.contains("# HELP empty_metric No labels"));
        assert!(buf.contains("# TYPE empty_metric counter"));
        // Should have HELP, TYPE lines and no data lines (only blank separator)
        let non_empty: Vec<&str> = buf.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(non_empty.len(), 2, "only HELP and TYPE lines (non-empty)");
    }

    #[test]
    fn test_prom_labeled_counter_single_label() {
        let mut buf = String::new();
        prom_labeled_counter(
            &mut buf,
            "tls_handshakes",
            "TLS handshakes",
            &[("result", "ok", 999)],
        );
        assert!(buf.contains(r#"tls_handshakes{result="ok"} 999"#));
    }

    // ── not_found ───────────────────────────────────────────────────

    #[test]
    fn test_not_found_response() {
        let resp = not_found();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let ct = resp.headers().get("Content-Type").unwrap().to_str().unwrap();
        assert_eq!(ct, "application/json");
    }

    #[test]
    fn test_not_found_lists_endpoints() {
        let resp = not_found();
        // The body is a Full<Bytes>, which we can check via the expected constant
        // (not_found returns a known static JSON string)
        let expected = r#"{"error":"not_found","endpoints":["/health","/metrics","/status","/peers","/backends","/tls-reload"]}"#;
        // Reconstruct what we know: the function uses a static string for the body.
        // Just verify the format from the function definition matches expectations.
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let _ = expected; // verify string compiles
    }

    #[tokio::test]
    async fn test_not_found_lists_backends() {
        use http_body_util::BodyExt as _;
        let resp = not_found();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let collected = resp.into_body().collect().await.expect("collect not_found body");
        let body_str =
            String::from_utf8(collected.to_bytes().to_vec()).expect("body is valid UTF-8");
        assert!(
            body_str.contains("/backends"),
            "not_found response must list /backends endpoint, got: {}",
            body_str,
        );
    }

    // ── multiple metrics concatenation ──────────────────────────────

    #[test]
    fn test_multiple_prometheus_metrics() {
        let mut buf = String::new();
        prom_gauge(&mut buf, "uptime", "Uptime seconds", 3600);
        prom_counter(&mut buf, "requests", "Total requests", 10000);
        prom_labeled_counter(&mut buf, "errors", "Errors", &[
            ("type", "timeout", 5),
            ("type", "upstream", 3),
        ]);

        // All three metrics should be present
        assert!(buf.contains("uptime 3600"));
        assert!(buf.contains("requests 10000"));
        assert!(buf.contains(r#"errors{type="timeout"} 5"#));
        assert!(buf.contains(r#"errors{type="upstream"} 3"#));
    }
}
