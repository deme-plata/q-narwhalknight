// Test libp2p bootstrap connectivity to 185.182.185.227:6881

use anyhow::Result;
use q_bep44_discovery::test_libp2p_bootstrap;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    println!("🧪 Testing libp2p bootstrap connectivity to 185.182.185.227:6881");
    println!("🌐 This will test the real libp2p Kademlia DHT implementation");

    // Run the bootstrap test
    match test_libp2p_bootstrap().await {
        Ok(_) => {
            println!("✅ libp2p bootstrap test completed successfully");
        }
        Err(e) => {
            println!("❌ libp2p bootstrap test failed: {}", e);
            return Err(e);
        }
    }

    Ok(())
}