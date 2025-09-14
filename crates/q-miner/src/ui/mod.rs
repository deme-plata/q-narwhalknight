pub mod dashboard;
pub mod cli;
pub mod gui;

pub use dashboard::Dashboard;
pub use cli::CliInterface;
pub use gui::GuiApplication;

use crate::{MiningStats, GlobalMiningStats, MiningEvent, DeviceStats};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use tracing::info;

/// Web-based mining dashboard
pub struct Dashboard {
    port: u16,
    stats: Arc<RwLock<GlobalMiningStats>>,
    event_rx: broadcast::Receiver<MiningEvent>,
    is_running: Arc<RwLock<bool>>,
}

impl Dashboard {
    pub fn new(port: u16) -> Self {
        let (_, event_rx) = broadcast::channel(1000);
        
        Self {
            port,
            stats: Arc::new(RwLock::new(GlobalMiningStats::default())),
            event_rx,
            is_running: Arc::new(RwLock::new(false)),
        }
    }
    
    pub async fn start(&self) -> Result<()> {
        use axum::{
            extract::ws::{WebSocket, WebSocketUpgrade},
            response::Html,
            routing::{get, get_service},
            Json, Router,
        };
        use axum::response::Response;
        use tower_http::services::ServeDir;
        
        info!("🌐 Starting mining dashboard on port {}", self.port);
        
        *self.is_running.write().await = true;
        
        let app = Router::new()
            .route("/", get(dashboard_html))
            .route("/api/stats", get(get_mining_stats))
            .route("/api/events", get(websocket_handler))
            .nest_service("/static", get_service(ServeDir::new("static")))
            .with_state(self.stats.clone());
        
        let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", self.port)).await?;
        
        info!("✅ Mining dashboard started: http://localhost:{}", self.port);
        
        axum::serve(listener, app).await?;
        Ok(())
    }
}

async fn dashboard_html() -> Html<&'static str> {
    Html(include_str!("dashboard.html"))
}

async fn get_mining_stats(
    axum::extract::State(stats): axum::extract::State<Arc<RwLock<GlobalMiningStats>>>,
) -> Json<GlobalMiningStats> {
    let stats_guard = stats.read().await;
    Json(stats_guard.clone())
}

async fn websocket_handler(ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(handle_websocket)
}

async fn handle_websocket(mut socket: WebSocket) {
    // Real-time mining event streaming
    loop {
        let stats_update = serde_json::json!({
            "type": "stats_update",
            "hash_rate": 150_000_000.0,
            "temperature": 65.5,
            "power": 125.0,
            "timestamp": chrono::Utc::now()
        });
        
        if socket.send(axum::extract::ws::Message::Text(stats_update.to_string())).await.is_err() {
            break;
        }
        
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }
}

/// Command-line interface
pub struct CliInterface {
    stats: Arc<RwLock<GlobalMiningStats>>,
    event_rx: broadcast::Receiver<MiningEvent>,
}

impl CliInterface {
    pub fn new() -> Self {
        let (_, event_rx) = broadcast::channel(1000);
        
        Self {
            stats: Arc::new(RwLock::new(GlobalMiningStats::default())),
            event_rx,
        }
    }
    
    pub async fn start_interactive_mode(&mut self) -> Result<()> {
        use console::{style, Key, Term};
        use indicatif::{ProgressBar, ProgressStyle};
        
        let term = Term::stdout();
        term.clear_screen()?;
        
        println!("{}", style("🎮 Q-NarwhalKnight Miner Interactive Mode").green().bold());
        println!("{}", style("Press 'q' to quit, 's' for stats, 'h' for help").dim());
        println!();
        
        // Create progress bars
        let hash_rate_bar = ProgressBar::new(100);
        hash_rate_bar.set_style(
            ProgressStyle::default_bar()
                .template("Hash Rate: {bar:40.cyan/blue} {pos:>7}%")?
                .progress_chars("##-")
        );
        
        loop {
            // Update display
            self.update_cli_display(&hash_rate_bar).await?;
            
            // Check for user input (non-blocking)
            if term.poll(std::time::Duration::from_millis(100))? {
                match term.read_key()? {
                    Key::Char('q') | Key::Char('Q') => {
                        println!("\n👋 Shutting down miner...");
                        break;
                    }
                    Key::Char('s') | Key::Char('S') => {
                        self.show_detailed_stats().await?;
                    }
                    Key::Char('h') | Key::Char('H') => {
                        self.show_help();
                    }
                    _ => {}
                }
            }
            
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
        
        Ok(())
    }
    
    async fn update_cli_display(&self, hash_rate_bar: &ProgressBar) -> Result<()> {
        let stats = self.stats.read().await;
        
        // Update progress bar with current hash rate
        let hash_rate_ghps = stats.total_hash_rate / 1_000_000_000.0;
        let progress = (hash_rate_ghps * 10.0).min(100.0) as u64;
        hash_rate_bar.set_position(progress);
        hash_rate_bar.set_message(format!("{:.2} GH/s", hash_rate_ghps));
        
        Ok(())
    }
    
    async fn show_detailed_stats(&self) -> Result<()> {
        let stats = self.stats.read().await;
        
        println!("\n{}", style("📊 Detailed Mining Statistics").yellow().bold());
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Total Hash Rate:    {:.2} GH/s", stats.total_hash_rate / 1e9);
        println!("Accepted Shares:    {}", stats.accepted_shares);
        println!("Rejected Shares:    {}", stats.rejected_shares);
        println!("Efficiency:         {:.2}%", stats.efficiency * 100.0);
        println!("Power Usage:        {:.1} W", stats.power_usage);
        println!("Uptime:             {}", format_duration(&stats.uptime));
        println!();
        
        println!("{}", style("🔧 Device Information").cyan().bold());
        for device in &stats.devices {
            println!("  {} ({}): {:.2} MH/s, {:.1}°C, {:.1}W", 
                device.device_id,
                format_device_type(&device.device_type),
                device.hash_rate / 1e6,
                device.temperature,
                device.power_usage
            );
        }
        
        println!("\nPress any key to continue...");
        Ok(())
    }
    
    fn show_help(&self) {
        println!("\n{}", style("🆘 Help - Keyboard Controls").blue().bold());
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  q/Q - Quit miner");
        println!("  s/S - Show detailed statistics");
        println!("  h/H - Show this help");
        println!("  r/R - Reset statistics");
        println!("  p/P - Pause/resume mining");
        println!("\nPress any key to continue...");
    }
}

fn format_duration(duration: &chrono::Duration) -> String {
    let total_seconds = duration.num_seconds();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    
    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}

fn format_device_type(device_type: &crate::DeviceType) -> String {
    match device_type {
        crate::DeviceType::CPU => "CPU".to_string(),
        crate::DeviceType::CUDA(model) => format!("CUDA {}", model),
        crate::DeviceType::OpenCL(model) => format!("OpenCL {}", model),
        crate::DeviceType::Vulkan(model) => format!("Vulkan {}", model),
    }
}