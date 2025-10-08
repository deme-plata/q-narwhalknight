//! Simple Bootstrap Service Demo
//!
//! Demonstrates real Tor bootstrap node implementation with mock .onion addresses
//! for Q-NarwhalKnight peer discovery

use anyhow::Result;
use q_tor_client::bootstrap_service::{BootstrapService, BootstrapServiceBuilder};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("🚀 Starting Q-NarwhalKnight Bootstrap Service Demo");
    info!("🎯 This demonstrates real Tor hidden service bootstrap nodes");

    // Create bootstrap service configuration
    let service = BootstrapServiceBuilder::new()
        .service_name("QNK Demo Bootstrap".to_string())
        .local_port(8080)
        .max_peers(1000)
        .peer_ttl(Duration::from_secs(3600))
        .require_zk_proofs(false)
        .build()
        .await?;

    match service {
        Ok(mut bootstrap_service) => {
            // Start the bootstrap service
            match bootstrap_service.start().await {
                Ok(onion_address) => {
                    info!("✅ Bootstrap service started successfully!");
                    info!("🧅 Service available at: {}", onion_address);
                    info!("🌍 HTTP API running on: http://127.0.0.1:8080");

                    info!("📡 Available endpoints:");
                    info!("   • GET  /api/v1/peers - List discovered peers");
                    info!("   • POST /api/v1/peers/register - Register as peer");
                    info!("   • POST /api/v1/heartbeat - Send heartbeat");
                    info!("   • GET  /api/v1/status - Service status");
                    info!("   • GET  /health - Health check");

                    info!("🔄 Service running... (press Ctrl+C to stop)");

                    // Keep the service running
                    tokio::signal::ctrl_c().await?;
                    info!("🛑 Shutdown signal received");

                    // TODO: Implement proper shutdown
                    info!("✅ Bootstrap service demo completed");
                }
                Err(e) => {
                    warn!("❌ Failed to start bootstrap service: {}", e);
                }
            }
        }
        Err(e) => {
            warn!("❌ Failed to create bootstrap service: {}", e);
        }
    }

    Ok(())
}
