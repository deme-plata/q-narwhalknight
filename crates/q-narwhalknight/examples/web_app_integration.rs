use axum::{extract::Extension, response::Json, routing::get, Router};
use q_narwhalknight::{web::DNSPhantomExtension, DNSPhantomMesh};
use serde_json::{json, Value};
use std::sync::Arc;

async fn mesh_status(Extension(mesh): DNSPhantomExtension) -> Json<Value> {
    let health = mesh.mesh_health().await;

    Json(json!({
        "status": "operational",
        "discovered_peers": health.discovered_peer_count,
        "connected_peers": health.connected_peer_count,
        "dns_anomalies": health.dns_anomaly_count,
        "discovery_active": health.discovery_active
    }))
}

async fn mesh_peers(Extension(mesh): DNSPhantomExtension) -> Json<Value> {
    let discovered = mesh.discovered_peers().await;
    let connected = mesh.connected_peers().await;

    Json(json!({
        "discovered_peers": discovered.len(),
        "connected_peers": connected,
        "peer_details": discovered
    }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("🌐 Starting web app with DNS-Phantom mesh integration...");

    let mesh = Arc::new(DNSPhantomMesh::new().await?);
    mesh.start_autonomous_discovery().await?;

    let app = Router::new()
        .route("/api/mesh/status", get(mesh_status))
        .route("/api/mesh/peers", get(mesh_peers))
        .layer(Extension(mesh.clone()));

    println!("🚀 Web server starting on http://localhost:3000");
    println!("📊 Try: curl http://localhost:3000/api/mesh/status");

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;

    Ok(())
}
