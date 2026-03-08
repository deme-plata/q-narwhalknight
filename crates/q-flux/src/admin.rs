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
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("q-flux-admin".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build admin tokio runtime");

            rt.block_on(async move {
                run_admin_server(listen_addr, metrics, worker_count, shared_tls, tls_config_paths, health_map, local_backends, cluster_peers, peer_tracker, ocsp_status).await;
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
        (&hyper::Method::POST, "/tls-reload") => handle_tls_reload(state),
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

    // -- latency histogram (Issue #11) ----------------------------------------
    buf.push_str(&state.metrics.prometheus_export_histogram());

    // -- HTTP/2 metrics (Issue #15) -------------------------------------------
    buf.push_str(&crate::h2_proxy::h2_prometheus_export());

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
            let (healthy, failures, last_check_ago_ms) = if let Some(entry) = hm.get(backend.as_str()) {
                let ago = entry.last_check.elapsed().as_millis() as u64;
                (entry.is_healthy, entry.consecutive_failures, ago)
            } else {
                (true, 0, 0) // no entry = assume healthy
            };
            local_entries.push(format!(
                r#"{{"addr":"{}","healthy":{},"failures":{},"last_check_ms_ago":{}}}"#,
                backend, healthy, failures, last_check_ago_ms,
            ));
        }

        let mut peer_entries = Vec::new();
        for peer in &state.cluster_peers {
            let (healthy, failures, last_check_ago_ms) = if let Some(entry) = hm.get(peer.as_str()) {
                let ago = entry.last_check.elapsed().as_millis() as u64;
                (entry.is_healthy, entry.consecutive_failures, ago)
            } else {
                (true, 0, 0)
            };
            peer_entries.push(format!(
                r#"{{"addr":"{}","healthy":{},"failures":{},"last_check_ms_ago":{}}}"#,
                peer, healthy, failures, last_check_ago_ms,
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
// 404
// ---------------------------------------------------------------------------

fn not_found() -> Response<Full<Bytes>> {
    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(
            r#"{"error":"not_found","endpoints":["/health","/metrics","/status","/peers","/tls-reload"]}"#,
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
