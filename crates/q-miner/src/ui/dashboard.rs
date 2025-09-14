//! Web dashboard for mining statistics

use crate::{MiningStats, GlobalMiningStats};
use anyhow::Result;
use axum::{
    extract::State,
    response::{Html, Json},
    routing::get,
    Router,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

pub struct Dashboard {
    port: u16,
    stats: Arc<RwLock<GlobalMiningStats>>,
}

impl Dashboard {
    pub fn new(port: u16) -> Self {
        Self {
            port,
            stats: Arc::new(RwLock::new(GlobalMiningStats::default())),
        }
    }
    
    pub async fn start(&self) -> Result<()> {
        info!("🌐 Starting mining dashboard on port {}", self.port);
        
        let app = Router::new()
            .route("/", get(dashboard_html))
            .route("/api/stats", get(get_mining_stats))
            .with_state(self.stats.clone());
        
        let listener = tokio::net::TcpListener::bind(("0.0.0.0", self.port)).await?;
        info!("✅ Mining dashboard started: http://localhost:{}", self.port);
        
        axum::serve(listener, app).await?;
        Ok(())
    }
}

async fn dashboard_html() -> Html<&'static str> {
    Html(include_str!("dashboard.html"))
}

async fn get_mining_stats(
    State(stats): State<Arc<RwLock<GlobalMiningStats>>>,
) -> Json<GlobalMiningStats> {
    let stats_guard = stats.read().await;
    Json(stats_guard.clone())
}