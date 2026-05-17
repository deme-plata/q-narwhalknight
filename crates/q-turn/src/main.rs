mod config;
mod stun;
mod auth;
mod allocation;
mod turn;
mod server;

use clap::Parser;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "q-turn", about = "STUN/TURN relay server — zero IP leak WebRTC for Q-NarwhalKnight")]
struct Cli {
    /// Path to q-turn.toml configuration file
    #[arg(short, long, default_value = "q-turn.toml")]
    config: PathBuf,

    /// Log level override (e.g. debug, info, warn)
    #[arg(short, long)]
    log_level: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cfg = config::Config::load(&cli.config)?;

    let level = cli.log_level.as_deref().unwrap_or(&cfg.logging.level);
    let filter = EnvFilter::try_new(level).unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .enable_all()
        .build()?;

    rt.block_on(server::run(cfg))
}
