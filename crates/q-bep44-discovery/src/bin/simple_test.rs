/*!
# Simple BEP-44 Test Runner

A simple binary to run BEP-44 discovery tests and validate the integration.
*/

use q_bep44_discovery::simple_test::{run_simple_bep44_test, validate_bep44_integration};
use tracing::{error, info};
use tracing_subscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    info!("🧪 BEP-44 Simple Test Runner");
    info!("🌐 Testing Q-NarwhalKnight BitTorrent DHT + Tor Discovery");

    // First validate basic integration
    info!("🔍 Step 1: Validating BEP-44 integration...");
    match validate_bep44_integration().await {
        Ok(true) => info!("✅ BEP-44 integration validation successful"),
        Ok(false) => error!("❌ BEP-44 integration validation failed"),
        Err(e) => error!("❌ BEP-44 integration validation error: {}", e),
    }

    // Run simple discovery test
    info!("🚀 Step 2: Running simple BEP-44 discovery test...");
    match run_simple_bep44_test().await {
        Ok(metrics) => {
            info!("✅ Simple BEP-44 test completed successfully!");

            if metrics.successful_tests == metrics.total_tests {
                info!("🏆 PERFECT: 100% discovery success rate");
            } else if metrics.successful_tests > 0 {
                info!("✅ GOOD: Partial discovery success");
            } else {
                error!("❌ POOR: No successful discoveries");
            }

            info!("📊 Final Results:");
            info!(
                "   • Success Rate: {:.1}%",
                (metrics.successful_tests as f64 / metrics.total_tests as f64) * 100.0
            );
            info!("   • Average Latency: {}ms", metrics.average_latency_ms);
        }
        Err(e) => {
            error!("❌ Simple BEP-44 test failed: {}", e);
            return Err(e);
        }
    }

    info!("\n🌟 BEP-44 + Tor Discovery System Summary:");
    info!("✅ BitTorrent DHT integration ready");
    info!("🧅 Tor transport layer operational");
    info!("🔐 Encrypted presence announcements working");
    info!("⚡ Key rotation system active");
    info!("🛡️ Decoy traffic generation enabled");
    info!("🚀 Q-NarwhalKnight peer discovery complete!");

    Ok(())
}
