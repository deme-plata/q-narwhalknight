use axum::{http::StatusCode, response::Json, routing::get, Router};
use serde_json::json;
use std::env;
use tokio::net::TcpListener;
use tracing::{info, warn};
use tracing_subscriber::fmt::init;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init();

    let node_id = env::var("Q_NODE_ID").unwrap_or_else(|_| "test-node".to_string());
    let port = env::var("Q_API_PORT").unwrap_or_else(|_| "8080".to_string());

    info!("🚀 Starting Q-NarwhalKnight test node: {}", node_id);

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/status", get(node_status))
        .route("/peers", get(peer_info));

    let addr = format!("0.0.0.0:{}", port);
    info!("🌐 Test node listening on {}", addr);

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check() -> StatusCode {
    StatusCode::OK
}

async fn node_status() -> Json<serde_json::Value> {
    let node_id = env::var("Q_NODE_ID").unwrap_or_else(|_| "test-node".to_string());

    Json(json!({
        "node_id": node_id,
        "status": "healthy",
        "type": "quantum-consensus-validator",
        "version": "0.1.0-test",
        "uptime": 0,
        "consensus": {
            "state": "ready",
            "round": 0,
            "validators": 2,
            "byzantine_threshold": 0
        },
        "networking": {
            "peers_connected": 0,
            "discovery_active": true,
            "tor_enabled": true
        }
    }))
}

async fn peer_info() -> Json<serde_json::Value> {
    Json(json!({
        "discovered_peers": [],
        "connected_peers": [],
        "bootstrap_complete": false
    }))
}
