/// DNS Phantom Auto-Starting Node
///
/// This binary automatically starts a Q-NarwhalKnight node with DNS Phantom steganographic
/// networking enabled. When launched, it immediately begins:
/// - DNS steganographic peer discovery
/// - Transaction propagation through covert DNS channels
/// - Block verification via DNS queries
/// - Consensus message routing through DNS infrastructure
use anyhow::Result;
use clap::{Arg, Command};
use q_dns_phantom::node_integration::{
    DNSPhantomNode, DefaultBlockVerifier, DefaultTransactionVerifier, NodeIntegrationConfig,
};
use q_types::NodeId;
use std::sync::Arc;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let matches = Command::new("phantom-node")
        .version("2.0.0")
        .about("Q-NarwhalKnight DNS Phantom Node - Automatic Steganographic Networking")
        .arg(
            Arg::new("node-name")
                .long("name")
                .value_name("NAME")
                .help("Node identifier name")
                .default_value("phantom-node"),
        )
        .arg(
            Arg::new("stealth")
                .long("stealth")
                .help("Enable maximum stealth mode")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("tx-rate")
                .long("tx-rate")
                .value_name("RATE")
                .help("Maximum transaction propagation rate per second")
                .default_value("100"),
        )
        .get_matches();

    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,q_dns_phantom=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let node_name = matches.get_one::<String>("node-name").unwrap();
    let stealth_mode = matches.get_flag("stealth");
    let tx_rate: usize = matches
        .get_one::<String>("tx-rate")
        .unwrap()
        .parse()
        .unwrap_or(100);

    info!("╔══════════════════════════════════════════════════════╗");
    info!("║     Q-NARWHALKNIGHT DNS PHANTOM NODE v2.0.0         ║");
    info!("║     Automatic Steganographic Networking System      ║");
    info!("╚══════════════════════════════════════════════════════╝");
    info!("");
    info!("🔮 Node Name: {}", node_name);
    info!(
        "🥷 Stealth Mode: {}",
        if stealth_mode { "ENABLED" } else { "Standard" }
    );
    info!("📊 TX Propagation Rate: {} tx/s", tx_rate);
    info!("");

    // Generate or load node ID
    let node_id: NodeId = {
        use rand::RngCore;
        let mut id = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut id);

        // Make it memorable by setting first bytes to node name hash
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(node_name.as_bytes());
        let hash = hasher.finalize();
        id[..8].copy_from_slice(&hash[..8]);

        id
    };

    info!("🆔 Node ID: {}", hex::encode(node_id));
    info!("");
    info!("═══════════════════════════════════════════════════════");
    info!("         INITIALIZING DNS PHANTOM SUBSYSTEMS");
    info!("═══════════════════════════════════════════════════════");

    // Configure DNS Phantom node
    let mut config = NodeIntegrationConfig::default();
    config.stealth_mode = stealth_mode;
    config.max_tx_propagation_rate = tx_rate;
    config.auto_start = true;
    config.propagate_transactions = true;
    config.propagate_blocks = true;
    config.consensus_via_dns = true;

    // Create verifiers
    let tx_verifier = Arc::new(DefaultTransactionVerifier);
    let block_verifier = Arc::new(DefaultBlockVerifier);

    // Create DNS Phantom node
    info!("🔧 Creating DNS Phantom node instance...");
    let phantom_node =
        Arc::new(DNSPhantomNode::new(node_id, tx_verifier, block_verifier, config).await?);

    // Subscribe to events
    let mut event_receiver = phantom_node.subscribe_to_events();

    // Start event logger
    tokio::spawn(async move {
        loop {
            match event_receiver.recv().await {
                Ok(event) => match event {
                    q_dns_phantom::node_integration::NodeEvent::DNSPhantomStarted => {
                        info!("✅ DNS Phantom network started successfully");
                    }
                    q_dns_phantom::node_integration::NodeEvent::PeerDiscovered {
                        peer_id,
                        via_dns,
                    } => {
                        if via_dns {
                            info!(
                                "🔍 Peer discovered via DNS steganography: {}",
                                hex::encode(&peer_id[..8])
                            );
                        }
                    }
                    q_dns_phantom::node_integration::NodeEvent::TransactionVerified {
                        tx_hash,
                        score,
                    } => {
                        info!(
                            "✅ Transaction verified: {} (score: {:.2})",
                            hex::encode(&tx_hash[..8.min(tx_hash.len())]),
                            score
                        );
                    }
                    q_dns_phantom::node_integration::NodeEvent::BlockReceived {
                        block_hash,
                        height,
                    } => {
                        info!(
                            "⛓️ Block received: height={}, hash={}",
                            height,
                            hex::encode(&block_hash[..8.min(block_hash.len())])
                        );
                    }
                    q_dns_phantom::node_integration::NodeEvent::DNSAnomalyDetected {
                        severity,
                        description,
                    } => {
                        if severity > 0.8 {
                            error!(
                                "🚨 CRITICAL DNS anomaly: {} (severity: {:.2})",
                                description, severity
                            );
                        }
                    }
                    _ => {}
                },
                Err(e) => {
                    error!("Event receiver error: {}", e);
                    break;
                }
            }
        }
    });

    // Start the DNS Phantom node
    info!("");
    info!("🚀 Starting DNS Phantom node...");
    phantom_node.clone().start().await?;

    info!("");
    info!("═══════════════════════════════════════════════════════");
    info!("      DNS PHANTOM NODE OPERATIONAL");
    info!("═══════════════════════════════════════════════════════");
    info!("");
    info!("📡 Steganographic channels active:");
    info!("   • Peer Discovery: ✓ Active");
    info!("   • Transaction Propagation: ✓ Active");
    info!("   • Block Verification: ✓ Active");
    info!("   • Consensus Routing: ✓ Active");
    info!("");
    info!("🌐 DNS Providers in use:");
    info!("   • Cloudflare DNS-over-HTTPS");
    info!("   • Google Public DNS");
    info!("   • Quad9 Secure DNS");
    info!("   • OpenDNS");
    info!("");
    info!("💡 The node is now:");
    info!("   1. Discovering peers through DNS steganography");
    info!("   2. Verifying transactions from other nodes");
    info!("   3. Propagating blocks through covert channels");
    info!("   4. Participating in consensus via DNS");
    info!("");
    info!("Press Ctrl+C to stop the node");
    info!("");

    // Simulate some test transactions after a delay
    let phantom_node_clone = phantom_node.clone();
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;

        info!("📤 Submitting test transaction for propagation...");

        // Create a test transaction
        let test_tx = format!(
            "TX:{}:TEST:{}:AMOUNT:1000",
            hex::encode(&node_id[..8]),
            chrono::Utc::now().timestamp()
        );

        if let Err(e) = phantom_node_clone
            .submit_transaction(test_tx.as_bytes().to_vec())
            .await
        {
            error!("Failed to submit test transaction: {}", e);
        } else {
            info!("✅ Test transaction submitted successfully");
        }

        // Periodically show peer count
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;

            match phantom_node_clone.get_discovered_peers().await {
                Ok(peers) => {
                    info!(
                        "📊 Network Status: {} peers discovered via DNS steganography",
                        peers.len()
                    );
                    for (i, peer) in peers.iter().take(5).enumerate() {
                        info!("   {}. Peer: {}", i + 1, hex::encode(&peer[..8]));
                    }
                }
                Err(e) => {
                    error!("Failed to get peer list: {}", e);
                }
            }
        }
    });

    // Keep the main thread alive
    tokio::signal::ctrl_c().await?;

    info!("");
    info!("🛑 Shutting down DNS Phantom node...");
    phantom_node.stop().await?;
    info!("👋 Node stopped successfully");

    Ok(())
}
