use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing_subscriber::EnvFilter;

mod access_control;
mod access_log;
mod acme;
mod admin;
mod config;
mod acceptor;
mod worker;
mod proxy;
mod upstream;
mod metrics;
mod health;
mod static_serve;
mod tui;

// Issue #017: kTLS kernel TLS offload
#[cfg(target_os = "linux")]
mod ktls;

// Phase 2+ modules (compiled but wired in incrementally)
#[cfg(target_os = "linux")]
#[allow(dead_code)]
mod io_uring_loop;
#[allow(dead_code)]
mod simd_parse;
#[allow(dead_code)]
mod h2_proxy;
#[cfg(feature = "quic")]
#[allow(dead_code)]
mod quic_proxy;
#[allow(dead_code)]
mod libp2p_aware;
mod ocsp_fetch;
mod vhost;

#[derive(Parser)]
#[command(name = "q-flux", about = "High-performance reverse proxy for Q-NarwhalKnight")]
struct Cli {
    /// Path to configuration file
    #[arg(short, long, default_value = "q-flux.toml")]
    config: PathBuf,

    /// Override number of workers (0 = auto)
    #[arg(short, long)]
    workers: Option<usize>,

    /// Override log level
    #[arg(short, long)]
    log_level: Option<String>,

    /// Drain timeout in seconds for graceful shutdown (default: 30)
    #[arg(long, default_value = "30")]
    drain_timeout: u64,

    /// Enable live TUI dashboard (interactive terminal UI)
    #[arg(long)]
    tui: bool,
}

fn main() -> anyhow::Result<()> {
    // Install rustls crypto provider before any TLS operations.
    // The workspace has both ring and aws-lc-rs as transitive deps,
    // so we must explicitly select one.
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls CryptoProvider");

    let cli = Cli::parse();

    // Load config
    let mut config = config::FluxConfig::load(&cli.config)?;

    // CLI overrides
    if let Some(w) = cli.workers {
        config.server.workers = w;
    }
    if let Some(ref level) = cli.log_level {
        config.logging.level = level.clone();
    }

    let drain_timeout = Duration::from_secs(cli.drain_timeout);
    let tui_mode = cli.tui;

    // Initialize logging — suppress in TUI mode (TUI owns the terminal)
    if tui_mode {
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::new("error"))
            .with_target(false)
            .with_writer(std::io::sink)
            .init();
    } else {
        let filter = EnvFilter::try_new(&config.logging.level)
            .unwrap_or_else(|_| EnvFilter::new("info"));
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .with_thread_names(true)
            .init();
    }

    let worker_count = config.worker_count();

    tracing::info!(
        "q-flux v{} starting -- {} workers, {} backends",
        env!("CARGO_PKG_VERSION"),
        worker_count,
        config.upstream.backends.len(),
    );

    // io_uring / splice feature detection
    #[cfg(target_os = "linux")]
    {
        let features = io_uring_loop::probe_io_uring_features();
        tracing::info!(
            io_uring = features.basic,
            multishot_accept = features.multishot_accept,
            provided_buffers = features.provided_buffers,
            splice_enabled = config.io_uring.splice_enabled,
            "io_uring probe: {}",
            if features.basic { "available" } else { "not available (using epoll)" },
        );
    }

    // Build TLS config (shared across all workers, hot-reloadable).
    // With vhosts configured, use SNI-based cert selection so each domain
    // gets its own certificate during the TLS handshake.
    let shared_tls = if config.vhosts.is_empty() {
        let tls_config = acceptor::build_tls_config(&config.tls)?;
        tracing::info!("TLS config loaded from {} / {}", config.tls.cert.display(), config.tls.key.display());
        acceptor::SharedTlsConfig::new(tls_config)
    } else {
        let vhost_router = vhost::VhostRouter::from_config(&config.tls, &config.vhosts)?;
        let tls_config = acceptor::build_tls_config_with_vhosts(vhost_router)?;
        tracing::info!(
            "TLS config with SNI routing: default {} + {} vhost cert(s)",
            config.tls.cert.display(),
            config.vhosts.len(),
        );
        acceptor::SharedTlsConfig::new(tls_config)
    };

    // Issue #017: kTLS kernel offload detection
    if config.tls.enable_ktls {
        let ktls_features = acceptor::probe_ktls();
        if ktls_features.available {
            tracing::info!("kTLS: {}", ktls_features);
        } else {
            tracing::warn!(
                "kTLS enabled in config but not available on this kernel. \
                 Ensure kernel >= 4.13 and 'modprobe tls' has been run. \
                 Falling back to userspace TLS.",
            );
        }
    }

    // Initialize metrics
    let metrics = metrics::Metrics::new();

    // Access logger (Issue #13): optional structured JSON access log
    let access_logger: Option<access_log::AccessLogger> = match &config.logging.access_log {
        Some(path) => {
            let path_str = path.to_string_lossy();
            if path_str == "-" || path_str == "stdout" {
                tracing::info!("Access logging to stdout");
                Some(access_log::AccessLogger::new_stdout(8192))
            } else {
                match access_log::AccessLogger::new_file(&path_str, 8192) {
                    Ok(logger) => {
                        tracing::info!("Access logging to {}", path_str);
                        Some(logger)
                    }
                    Err(e) => {
                        tracing::error!("Failed to open access log {}: {} — continuing without", path_str, e);
                        None
                    }
                }
            }
        }
        None => None,
    };

    // Rate limiter (Issue #12): token-bucket per-IP + global rate limiting
    let rate_limiter = if config.limits.rate_limit_per_ip > 0 {
        let rl = Arc::new(metrics::RateLimiter::new(
            config.limits.rate_limit_per_ip as u64,
            config.limits.rate_limit_burst as u64,
            config.limits.rate_limit_global_rps as u64,
        ));
        tracing::info!(
            "Rate limiter: {} req/s/IP, burst {}, global {} req/s",
            config.limits.rate_limit_per_ip,
            config.limits.rate_limit_burst,
            config.limits.rate_limit_global_rps,
        );
        Some(rl)
    } else {
        tracing::info!("Rate limiter disabled (rate_limit_per_ip = 0)");
        None
    };

    // Issue #027: IP access control (allowlist/blocklist with CIDR)
    let access_control = match access_control::AccessControl::new(
        &config.access_control.mode,
        &config.access_control.allowlist,
        &config.access_control.blocklist,
    ) {
        Ok(ac) => {
            if ac.is_active() {
                tracing::info!(
                    mode = ac.mode_name(),
                    rules = ac.rule_count(),
                    "IP access control: {} mode with {} rules",
                    ac.mode_name(),
                    ac.rule_count(),
                );
            } else {
                tracing::info!("IP access control: disabled");
            }
            Arc::new(ac)
        }
        Err(e) => {
            anyhow::bail!("Invalid access_control config: {}", e);
        }
    };

    // Issue #021: ACME certificate automation
    let acme_challenge_store: acme::ChallengeStore = Arc::new(parking_lot::RwLock::new(
        std::collections::HashMap::new(),
    ));
    if config.acme.enabled {
        let acme_config = config.acme.clone();
        let shared_tls_for_acme = shared_tls.clone();
        let tls_config_for_acme = config.tls.clone();
        let challenge_store = acme_challenge_store.clone();
        std::thread::Builder::new()
            .name("q-flux-acme".into())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to build ACME runtime");
                rt.block_on(async move {
                    acme::acme_renewal_task(
                        acme_config,
                        shared_tls_for_acme,
                        tls_config_for_acme,
                        challenge_store,
                    ).await;
                });
            })
            .expect("Failed to spawn ACME thread");
        tracing::info!(
            domains = ?config.acme.domains,
            "ACME certificate automation started (renewal check every 12h)"
        );
    }

    // Issue #026: Global backend request counters (shared across all workers + admin)
    let global_backend_counters: Arc<dashmap::DashMap<String, Arc<upstream::BackendCounters>>> =
        Arc::new(dashmap::DashMap::new());
    // Pre-populate for all known backends
    for b in &config.upstream.backends {
        global_backend_counters.insert(b.clone(), Arc::new(upstream::BackendCounters::new()));
    }
    for p in &config.cluster.peers {
        global_backend_counters.insert(p.clone(), Arc::new(upstream::BackendCounters::new()));
    }

    // Log listen addresses
    for addr in &config.server.listen {
        tracing::info!("Listening on {}", addr);
    }

    // Create shutdown broadcast channel.
    // Workers subscribe to this channel. When the main thread sends a message,
    // all workers stop accepting new connections and begin draining.
    let (shutdown_tx, _) = tokio::sync::broadcast::channel::<()>(1);

    // Shared flag so workers can cheaply poll shutdown state without channel recv overhead
    // in the hot accept loop.
    let shutdown_flag = Arc::new(AtomicBool::new(false));

    // Issue #018: Drain signal for long-lived connections (SSE, WebSocket).
    // When set to true, all active SSE/WS connections initiate graceful close.
    let (drain_tx, drain_rx) = tokio::sync::watch::channel(false);

    // Create shared health map and spawn the background health checker.
    // The health map is shared between the checker (writer) and all workers (readers).
    // Super-cluster: also register cluster peers in the health map so the upstream
    // pool can check their health status during failover decisions.
    let mut all_backends = config.upstream.backends.clone();
    all_backends.extend(config.cluster.peers.clone());
    let health_map = health::new_health_map(&all_backends);

    let health_config = health::HealthCheckConfig {
        interval: config.upstream.health_check_interval,
        timeout: config.upstream.health_check_timeout,
        path: config.upstream.health_check_path.clone(),
        failure_threshold: config.upstream.failure_threshold,
        healthy_threshold: config.upstream.healthy_threshold,
    };

    // The health checker needs its own multi-thread runtime since workers each
    // run on single-thread current_thread runtimes. We spawn a dedicated thread.
    let hc_backends = config.upstream.backends.clone();
    let hc_health_map = health_map.clone();
    std::thread::Builder::new()
        .name("q-flux-health".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build health-checker runtime");
            rt.block_on(async move {
                health::spawn_health_checker(hc_backends, hc_health_map, health_config).await.ok();
            });
        })
        .expect("Failed to spawn health-checker thread");

    // Super-cluster: spawn a separate health checker for cluster peers (slower interval)
    if !config.cluster.peers.is_empty() {
        let cluster_peers = config.cluster.peers.clone();
        let cluster_health_map = health_map.clone();
        let cluster_health_config = health::HealthCheckConfig {
            interval: config.cluster.health_check_interval,
            timeout: config.cluster.health_check_timeout,
            path: config.cluster.health_check_path
                .clone()
                .unwrap_or_else(|| config.upstream.health_check_path.clone()),
            failure_threshold: 5,
            healthy_threshold: config.upstream.healthy_threshold,
        };
        tracing::info!(
            peers = cluster_peers.len(),
            interval_secs = cluster_health_config.interval.as_secs(),
            timeout_secs = cluster_health_config.timeout.as_secs(),
            "Super-cluster: health-checking {} remote peer(s)",
            cluster_peers.len(),
        );
        std::thread::Builder::new()
            .name("q-flux-cluster-health".into())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to build cluster health-checker runtime");
                rt.block_on(async move {
                    health::spawn_health_checker(cluster_peers, cluster_health_map, cluster_health_config).await.ok();
                });
            })
            .expect("Failed to spawn cluster health-checker thread");
    }

    // Spawn OCSP auto-refresh background task
    let ocsp_status: ocsp_fetch::SharedOcspStatus =
        Arc::new(parking_lot::RwLock::new(ocsp_fetch::OcspStatus::default()));
    {
        let tls_cfg = config.tls.clone();
        let shared_tls_for_ocsp = shared_tls.clone();
        let ocsp_st = ocsp_status.clone();
        std::thread::Builder::new()
            .name("q-flux-ocsp".into())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to build OCSP runtime");
                rt.block_on(async move {
                    ocsp_fetch::ocsp_refresh_task(
                        tls_cfg,
                        shared_tls_for_ocsp,
                        ocsp_st,
                        Duration::from_secs(12 * 3600), // refresh every 12 hours
                    )
                    .await;
                });
            })
            .expect("Failed to spawn OCSP refresh thread");
        tracing::info!("OCSP auto-refresh task started (12h interval)");
    }

    // Clone health_map for admin server (Arc<DashMap> is cheap to clone)
    let admin_health_map = health_map.clone();

    // Shared PeerTracker: tracks per-peer stats for libp2p connections.
    // Created once and shared across all workers + admin server.
    let peer_tracker = Arc::new(libp2p_aware::PeerTracker::new(
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

    // Spawn workers with shutdown receivers
    let handles = worker::spawn_workers(
        &config,
        shared_tls.clone(),
        metrics.clone(),
        &shutdown_tx,
        shutdown_flag.clone(),
        health_map,
        access_logger,
        rate_limiter,
        peer_tracker.clone(),
        drain_rx,
        access_control,
        acme_challenge_store,
        global_backend_counters.clone(),
    );

    tracing::info!("All {} workers started -- q-flux is ready", worker_count);

    // Spawn admin HTTP server on its own thread (health, metrics, status, TLS reload)
    let _admin_handle = admin::spawn_admin_server(
        config.server.admin_listen,
        metrics.clone(),
        worker_count,
        shared_tls,
        config.tls.clone(),
        Some(admin_health_map),
        config.upstream.backends.clone(),
        config.cluster.peers.clone(),
        Some(peer_tracker),
        Some(ocsp_status),
        Some(global_backend_counters),
        None, // adaptive concurrency — populated per-worker; admin reads via metrics
    );

    if tui_mode {
        // TUI mode: run interactive dashboard on main thread.
        // When user presses 'q', TUI exits and we trigger graceful shutdown.
        let tui_app = tui::TuiApp::new(metrics.clone(), worker_count);
        if let Err(e) = tui_app.run() {
            eprintln!("TUI error: {}", e);
        }

        // TUI exited — trigger graceful shutdown
        shutdown_flag.store(true, Ordering::SeqCst);
        let _ = shutdown_tx.send(());
        // Issue #018: signal SSE/WS connections to drain
        let _ = drain_tx.send(true);

        // Brief drain period, then exit
        let drain_start = std::time::Instant::now();
        while drain_start.elapsed() < drain_timeout {
            let snap = metrics.snapshot();
            if snap.active_connections == 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(500));
        }

        drop(handles);
        return Ok(());
    }

    // Daemon mode: Run a small tokio runtime on the main thread for signal handling.
    // Workers each have their own current_thread runtime on dedicated threads,
    // so this runtime is exclusively for catching OS signals.
    let signal_rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to build signal-handling runtime");

    signal_rt.block_on(async {
        wait_for_shutdown_signal().await;

        tracing::info!(
            "Shutdown signal received -- stopping new connections, draining in-flight requests (timeout: {}s)",
            drain_timeout.as_secs(),
        );

        // Set the atomic flag first (workers check this in accept loop for fast path)
        shutdown_flag.store(true, Ordering::SeqCst);

        // Broadcast shutdown to all worker runtimes via the channel
        let _ = shutdown_tx.send(());

        // Issue #018: signal SSE/WS connections to drain gracefully
        let _ = drain_tx.send(true);

        // Wait for the drain timeout to let in-flight requests complete.
        // Workers will stop accepting but continue processing spawned tasks.
        // After the timeout, we log final metrics and exit -- worker thread
        // join handles get dropped, which is fine since the runtimes will be
        // shut down by the process exit.
        let drain_start = std::time::Instant::now();
        let check_interval = Duration::from_secs(1);

        loop {
            let elapsed = drain_start.elapsed();
            if elapsed >= drain_timeout {
                break;
            }

            // Log periodic drain progress
            let snap = metrics.snapshot();
            if snap.active_connections > 0 {
                tracing::info!(
                    "Draining: {} active connections, {} active upstream, {} active websockets, \
                     {} draining ({} completed, {} forced) ({:.0}s / {}s)",
                    snap.active_connections,
                    snap.upstream_active,
                    snap.active_websockets,
                    snap.drain_active,
                    snap.drain_completed_total,
                    snap.drain_forced_total,
                    elapsed.as_secs_f64(),
                    drain_timeout.as_secs(),
                );
            } else {
                tracing::info!("All connections drained after {:.1}s", elapsed.as_secs_f64());
                break;
            }

            tokio::time::sleep(check_interval).await;
        }

        // Final metrics snapshot before exit
        let final_snap = metrics.snapshot();
        tracing::info!(
            "FINAL METRICS SNAPSHOT (shutdown): uptime={}s total_conns={} total_reqs={} \
             2xx={} 4xx={} 5xx={} tls_ok={} tls_fail={} upstream_fail={} upstream_timeout={} \
             rate_limited={} ws_upgrades={} bytes_rx={} bytes_tx={} \
             remaining_active_conns={} remaining_upstream={} remaining_ws={}",
            final_snap.uptime_secs,
            final_snap.total_connections,
            final_snap.total_requests,
            final_snap.requests_2xx,
            final_snap.requests_4xx,
            final_snap.requests_5xx,
            final_snap.tls_handshakes,
            final_snap.tls_handshake_failures,
            final_snap.upstream_connect_failures,
            final_snap.upstream_timeouts,
            final_snap.rate_limited,
            final_snap.websocket_upgrades,
            final_snap.bytes_received,
            final_snap.bytes_sent,
            final_snap.active_connections,
            final_snap.upstream_active,
            final_snap.active_websockets,
        );

        if final_snap.active_connections > 0 {
            tracing::warn!(
                "Force-closing {} remaining connections after {}s drain timeout",
                final_snap.active_connections,
                drain_timeout.as_secs(),
            );
        }

        tracing::info!("q-flux shutdown complete");
    });

    // Drop worker handles -- threads will be terminated by process exit.
    // We intentionally do NOT join here because worker threads may be blocked
    // on in-flight request I/O that exceeds the drain timeout.
    drop(handles);

    Ok(())
}

/// Wait for either SIGTERM or SIGINT (Ctrl-C).
/// On non-Unix platforms, only Ctrl-C is supported.
async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};

        let mut sigterm = signal(SignalKind::terminate())
            .expect("Failed to register SIGTERM handler");
        let mut sigint = signal(SignalKind::interrupt())
            .expect("Failed to register SIGINT handler");

        tokio::select! {
            _ = sigterm.recv() => {
                tracing::info!("Received SIGTERM");
            }
            _ = sigint.recv() => {
                tracing::info!("Received SIGINT");
            }
        }
    }

    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for Ctrl-C");
        tracing::info!("Received Ctrl-C");
    }
}
