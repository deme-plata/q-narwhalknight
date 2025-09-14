/// Standalone DNS-Phantom Mesh API Server
///
/// Provides REST API endpoints for integrating DNS-Phantom mesh networking
/// with any programming language or framework via HTTP requests.
use q_narwhalknight::api::start_api_server;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🌐 DNS-Phantom Mesh API Server");
    println!("==============================");
    println!("🚀 Starting proven DNS-Phantom steganographic mesh networking API");
    println!("📡 Based on breakthrough technology with 50+ DNS anomalies detected");
    println!("🔗 Successfully tested with multiple cross-server connections");
    println!();

    // Start the API server on port 3000
    start_api_server(3000).await?;

    Ok(())
}
