/*!
# REAL BEP-44 + Tor Test

This binary tests the REAL implementation - connecting to actual BitTorrent DHT
and Tor networks. NOT A SIMULATION.
*/

use anyhow::Result;
use ed25519_dalek::SigningKey;
use std::env;
use tracing::{error, info};
use tracing_subscriber;

use q_bep44_discovery::real_discovery::test_real_dht_connectivity;
use q_bep44_discovery::real_discovery::RealDiscoveryEngine;
use q_bep44_discovery::real_tor::test_tor_connectivity;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    let args: Vec<String> = env::args().collect();
    let command = args.get(1).map(|s| s.as_str()).unwrap_or("test-dht");

    info!("🌟 REAL BEP-44 + Tor Discovery Test");
    info!("⚠️  WARNING: This connects to REAL networks!");
    info!("   • BitTorrent DHT network");
    info!("   • Tor anonymity network");
    info!("   • NOT a simulation");

    match command {
        "test-tor" => test_real_tor().await,
        "test-dht" => test_real_dht().await,
        "full-test" => run_full_test().await,
        "announce" => announce_to_real_dht().await,
        "discover" => discover_from_real_dht().await,
        _ => {
            info!("Usage: real_test [command]");
            info!("Commands:");
            info!("  test-tor   - Test real Tor connectivity");
            info!("  test-dht   - Test real BitTorrent DHT connectivity");
            info!("  full-test  - Run full real discovery test");
            info!("  announce   - Announce presence to real DHT");
            info!("  discover   - Discover peers from real DHT");
            Ok(())
        }
    }
}

/// Test REAL Tor connectivity
async fn test_real_tor() -> Result<()> {
    info!("🧅 Testing REAL Tor connectivity...");

    match test_tor_connectivity().await? {
        true => {
            info!("✅ SUCCESS: Real Tor connectivity works!");
            info!("   • Tor client initialized");
            info!("   • Network bootstrap successful");
            info!("   • Ready for .onion connections");
        }
        false => {
            error!("❌ FAILED: Cannot connect to Tor network");
            info!("💡 Make sure Tor is installed and running");
        }
    }

    Ok(())
}

/// Test REAL BitTorrent DHT connectivity
async fn test_real_dht() -> Result<()> {
    info!("🌐 Testing REAL BitTorrent DHT connectivity...");

    match test_real_dht_connectivity().await? {
        true => {
            info!("✅ SUCCESS: Real DHT connectivity works!");
            info!("   • Connected to BitTorrent bootstrap nodes");
            info!("   • DHT routing table populated");
            info!("   • Ready for BEP-44 operations");
        }
        false => {
            error!("❌ FAILED: Cannot connect to BitTorrent DHT");
            info!("💡 Check internet connectivity and firewall");
        }
    }

    Ok(())
}

/// Run full REAL discovery test
async fn run_full_test() -> Result<()> {
    info!("🚀 Running FULL REAL discovery test...");

    // Generate validator key
    let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);

    info!(
        "🔑 Generated validator key: {}",
        hex::encode(signing_key.verifying_key().as_bytes())
    );

    // Create REAL discovery engine
    let mut engine = RealDiscoveryEngine::new(signing_key).await?;

    // Initialize with REAL networks
    info!("🌐 Initializing with REAL networks...");
    engine.initialize().await?;

    // Start REAL discovery
    info!("🎯 Starting REAL peer discovery...");
    engine.start_discovery().await?;

    // Wait and check results
    info!("⏳ Waiting 30 seconds for discovery...");
    tokio::time::sleep(std::time::Duration::from_secs(30)).await;

    // Get real statistics
    let stats = engine.get_stats().await;
    let peers = engine.get_discovered_peers().await;

    info!("\n🎯 ========== REAL DISCOVERY RESULTS ==========");
    info!("📊 DHT Statistics:");
    info!("   • Connected nodes: {}", stats.dht_connected_nodes);
    info!("   • Stored records: {}", stats.dht_stored_records);

    info!("📊 Tor Statistics:");
    info!("   • Active circuits: {}", stats.tor_active_circuits);
    info!("   • Onion service: {}", stats.onion_service_active);

    info!("📊 Discovery Results:");
    info!("   • Discovered peers: {}", stats.discovered_peers);
    info!(
        "   • Successful connections: {}",
        stats.successful_connections
    );
    info!("   • Failed connections: {}", stats.failed_connections);

    if !peers.is_empty() {
        info!("👥 Discovered Peers:");
        for peer in peers.iter().take(5) {
            info!(
                "   • {} ({})",
                peer.onion_address,
                peer.capabilities.join(", ")
            );
        }
    } else {
        info!("❓ No peers discovered (this is normal for new network)");
    }

    if stats.onion_service_active && stats.dht_connected_nodes > 0 {
        info!("🏆 EXCELLENT: Full real discovery system operational!");
    } else {
        info!("⚠️  PARTIAL: Some components not fully operational");
    }

    info!("===============================================\n");

    Ok(())
}

/// Announce to REAL BitTorrent DHT
async fn announce_to_real_dht() -> Result<()> {
    info!("📢 Announcing to REAL BitTorrent DHT...");

    let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let mut engine = RealDiscoveryEngine::new(signing_key).await?;

    engine.initialize().await?;

    // Force announcement refresh
    engine.refresh_discovery().await?;

    let stats = engine.get_stats().await;

    if stats.last_announcement.is_some() {
        info!("✅ Successfully announced to real DHT!");
        info!("   • Last announcement: {:?}", stats.last_announcement);
        info!("   • DHT nodes: {}", stats.dht_connected_nodes);
    } else {
        error!("❌ Failed to announce to DHT");
    }

    Ok(())
}

/// Discover peers from REAL BitTorrent DHT
async fn discover_from_real_dht() -> Result<()> {
    info!("🔍 Discovering peers from REAL BitTorrent DHT...");

    let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let mut engine = RealDiscoveryEngine::new(signing_key).await?;

    engine.initialize().await?;
    engine.start_discovery().await?;

    // Wait for discovery
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    let peers = engine.get_discovered_peers().await;

    if !peers.is_empty() {
        info!("✅ Discovered {} peers from real DHT:", peers.len());
        for peer in peers.iter().take(10) {
            info!("   • {}", peer.onion_address);
            info!("     Capabilities: {}", peer.capabilities.join(", "));
            info!("     Verified: {}", peer.verified);
        }
    } else {
        info!("❓ No peers discovered from DHT");
        info!("💡 This is normal for a new/empty network");
    }

    Ok(())
}
