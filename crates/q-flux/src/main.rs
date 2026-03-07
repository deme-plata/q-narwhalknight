use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing_subscriber::EnvFilter;

mod access_log;
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

    // Build TLS config (shared across all workers)
    let tls_config = acceptor::build_tls_config(&config.tls)?;
    tracing::info!("TLS config loaded from {} / {}", config.tls.cert.display(), config.tls.key.display());

    // Initialize metrics
    let metrics = metrics::Metrics::new();

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

    // Create shared health map and spawn the background health checker.
    // The health map is shared between the checker (writer) and all workers (readers).
    let health_map = health::new_health_map(&config.upstream.backends);
    let health_config = health::HealthCheckConfig {
        interval: config.upstream.health_check_interval,
        timeout: config.upstream.health_check_timeout,
        path: config.upstream.health_check_path.clone(),
        failure_threshold: 3,
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

    // Spawn workers with shutdown receivers
    let handles = worker::spawn_workers(
        &config,
        tls_config,
        metrics.clone(),
        &shutdown_tx,
        shutdown_flag.clone(),
        health_map,
    );

    tracing::info!("All {} workers started -- q-flux is ready", worker_count);

    // Spawn admin HTTP server on its own thread (health, metrics, status endpoints)
    let _admin_handle = admin::spawn_admin_server(
        config.server.admin_listen,
        metrics.clone(),
        worker_count,
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
                    "Draining: {} active connections, {} active upstream, {} active websockets ({:.0}s / {}s)",
                    snap.active_connections,
                    snap.upstream_active,
                    snap.active_websockets,
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
