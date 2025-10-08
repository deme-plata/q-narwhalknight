/// Simple single-node libp2p test
/// Validates the improved libp2p bootstrap implementation

use anyhow::Result;
use q_bep44_discovery::libp2p_discovery::LibP2PDiscoveryClient;
use q_bep44_discovery::QnkDhtConfig;
use std::net::{SocketAddr, IpAddr, Ipv4Addr};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing with detailed logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,libp2p=debug,q_bep44_discovery=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Q-NarwhalKnight Single-Node libp2p Bootstrap Test         ║");
    println!("║  Testing improved libp2p with Kademlia DHT + Gossipsub    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Parse command line arguments for node configuration
    let args: Vec<String> = std::env::args().collect();

    let node_id = if args.len() > 1 {
        args[1].parse::<u8>().unwrap_or(1)
    } else {
        1
    };

    let listen_port: u16 = if args.len() > 2 {
        args[2].parse().unwrap_or(7000)
    } else {
        7000
    };

    let bootstrap_addr: SocketAddr = if args.len() > 3 {
        args[3].parse().unwrap_or_else(|_| {
            "185.182.185.227:6881".parse().unwrap()
        })
    } else {
        "185.182.185.227:6881".parse().unwrap()
    };

    let test_duration: u64 = if args.len() > 4 {
        args[4].parse().unwrap_or(60)
    } else {
        60
    };

    info!("🎯 Node Configuration:");
    info!("   Node ID: {}", node_id);
    info!("   Listen port: {}", listen_port);
    info!("   Bootstrap: {}", bootstrap_addr);
    info!("   Test duration: {}s", test_duration);

    // Create validator ID
    let mut validator_id = [0u8; 32];
    validator_id[0] = node_id;
    for i in 1..32 {
        validator_id[i] = ((node_id as usize + i) % 256) as u8;
    }

    info!("🔑 Validator ID: {}", hex::encode(&validator_id[0..8]));

    // Create DHT configuration
    let config = QnkDhtConfig {
        bootstrap_nodes: vec![bootstrap_addr],
        listen_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), listen_port),
        storage_path: format!("./test-libp2p-node-{}", node_id),
        tor_proxy: None,
        persist_dht: false,
        announce_interval: Duration::from_secs(300),
    };

    // Create and initialize libp2p client (auto-starts in background)
    info!("🚀 Creating libp2p discovery client...");
    let client = LibP2PDiscoveryClient::new(config, validator_id).await?;

    info!("✅ Libp2p client initialized and started automatically in background");
    info!("📡 Discovery process running for {}s...", test_duration);

    // Monitor discovered peers (client is already running in background)
    info!("⏳ Waiting {} seconds to observe peer discovery...", test_duration);
    tokio::time::sleep(Duration::from_secs(test_duration)).await;
    info!("⏰ Test duration elapsed");

    // Final report
    let final_peers = client.get_discovered_peers().await;

    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║           Single-Node Test Results                   ║");
    println!("╠═══════════════════════════════════════════════════════╣");
    println!("║ Node ID:             {:3}                            ║", node_id);
    println!("║ Test Duration:       {:3}s                           ║", test_duration);
    println!("║ Peers Discovered:    {:3}                            ║", final_peers.len());
    println!("║ Auto-Discovery:      Active                          ║");
    println!("╠═══════════════════════════════════════════════════════╣");

    // Evaluate success
    let success = final_peers.len() > 0;

    if success {
        println!("║ Status: ✅ TEST PASSED                                ║");
        println!("║ Node successfully discovered peers                    ║");
    } else {
        println!("║ Status: ⚠️  TEST ISSUES                               ║");
        println!("║ No peers were discovered                              ║");
    }

    println!("╚═══════════════════════════════════════════════════════╝\n");

    // Print discovered peers
    if !final_peers.is_empty() {
        info!("📋 Discovered Peers:");
        for (idx, peer) in final_peers.iter().enumerate() {
            info!("   {}. Validator: {}", idx + 1, hex::encode(&peer.validator_id[0..8]));
            info!("      Endpoint: {}", peer.p2p_endpoint);
            info!("      Last seen: {}", peer.last_seen);
        }
    }

    if success {
        Ok(())
    } else {
        Err(anyhow::anyhow!("Test failed: no peers discovered"))
    }
}