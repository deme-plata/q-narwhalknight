use clap::Parser;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

mod config;
mod acceptor;
mod worker;
mod proxy;
mod upstream;
mod metrics;

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

    // Initialize logging
    let filter = EnvFilter::try_new(&config.logging.level)
        .unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_names(true)
        .init();

    let worker_count = config.worker_count();

    tracing::info!(
        "q-flux v{} starting — {} workers, {} backends",
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

    // Spawn workers
    let handles = worker::spawn_workers(&config, tls_config, metrics);

    tracing::info!("All {} workers started — q-flux is ready", worker_count);

    // Wait for shutdown signal
    // We can't use tokio::signal here since we're on the main thread (not in a runtime).
    // Instead, wait for all worker threads to complete (they run forever unless killed).
    for handle in handles {
        handle.join().ok();
    }

    Ok(())
}
