/// Example: Bitcoin-Tor Anonymous Peer Discovery
///
/// This example demonstrates how Q-NarwhalKnight nodes can discover each other
/// through the Bitcoin network while maintaining complete anonymity through Tor.
///
/// Usage:
/// ```bash
/// cargo run --example bitcoin_tor_discovery -- --node-id 1 --onion test1.onion
/// cargo run --example bitcoin_tor_discovery -- --node-id 2 --onion test2.onion
/// ```
use anyhow::Result;
use clap::{Arg, Command};
use q_bitcoin_bridge::{
    BitcoinBridgeConfig, BitcoinNetworkType, IntegratedBitcoinBridge, PeerNetworkEvent,
};
use q_tor_client::TorClient;
use std::{sync::Arc, time::Duration};
use tokio::time;
use tracing::{error, info, warn};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Parse command line arguments
    let matches = Command::new("Bitcoin-Tor Discovery Example")
        .about("Demonstrates anonymous peer discovery through Bitcoin network")
        .arg(
            Arg::new("node-id")
                .long("node-id")
                .value_name("ID")
                .help("Node identifier (1-255)")
                .required(true),
        )
        .arg(
            Arg::new("onion")
                .long("onion")
                .value_name("ADDRESS")
                .help("Onion service address")
                .required(true),
        )
        .arg(
            Arg::new("bitcoin-rpc")
                .long("bitcoin-rpc")
                .value_name("URL")
                .help("Bitcoin RPC URL")
                .default_value("http://127.0.0.1:18332"),
        ) // Testnet default
        .arg(
            Arg::new("network")
                .long("network")
                .value_name("NETWORK")
                .help("Bitcoin network (mainnet, testnet, regtest)")
                .default_value("testnet"),
        )
        .get_matches();

    let node_id_num: u8 = matches.get_one::<String>("node-id").unwrap().parse()?;
    let onion_address = matches.get_one::<String>("onion").unwrap().clone();
    let bitcoin_rpc_url = matches.get_one::<String>("bitcoin-rpc").unwrap().clone();
    let network = matches.get_one::<String>("network").unwrap();

    // Create node ID
    let mut node_id = [0u8; 32];
    node_id[0] = node_id_num;
    for i in 1..32 {
        node_id[i] = (node_id_num.wrapping_mul(i as u8)).wrapping_add(42);
    }

    info!("Starting Q-Knight node with ID: {}", hex::encode(node_id));
    info!("Onion address: {}", onion_address);

    // Configure Bitcoin bridge
    let bitcoin_network = match network {
        "mainnet" => BitcoinNetworkType::Mainnet,
        "testnet" => BitcoinNetworkType::Testnet,
        "regtest" => BitcoinNetworkType::Regtest,
        _ => BitcoinNetworkType::Testnet,
    };

    let bridge_config = BitcoinBridgeConfig {
        bitcoin_rpc_url,
        bitcoin_rpc_user: "bitcoin".to_string(),
        bitcoin_rpc_password: "password".to_string(),
        bitcoin_network,
        tor_enabled: true,
        bitcoin_tor_proxy: "127.0.0.1:9050".to_string(),
        discovery_interval: Duration::from_secs(60), // Check every minute
        max_peers_advertised: 20,
        advertisement_ttl: Duration::from_secs(1800), // 30 minutes
        onion_service_port: 8333,
        use_steganography: true,
        cover_traffic_enabled: true,
        min_confirmation_depth: 1,
    };

    // Initialize Tor client
    info!("Initializing Tor client...");
    let tor_client = Arc::new(TorClient::new(Default::default()).await?);

    // Create Bitcoin bridge
    info!("Creating Bitcoin-Tor bridge...");
    let mut bridge =
        IntegratedBitcoinBridge::new(bridge_config, node_id, onion_address.clone(), tor_client)
            .await?;

    // Subscribe to peer events
    let mut event_receiver = bridge.subscribe_to_events();

    // Start the bridge
    info!("Starting Bitcoin-Tor discovery...");
    bridge.start().await?;

    // Spawn event handler
    tokio::spawn(async move {
        while let Ok(event) = event_receiver.recv().await {
            handle_peer_event(event).await;
        }
    });

    info!("🚀 Q-Knight Bitcoin-Tor bridge is running!");
    info!("📡 Advertising node on Bitcoin network through Tor");
    info!("🔍 Scanning Bitcoin network for other Q-Knight nodes");
    info!("🧅 All connections routed through Tor for maximum anonymity");

    // Status reporting loop
    let bridge_ref = Arc::new(bridge);
    let mut status_interval = time::interval(Duration::from_secs(30));

    loop {
        status_interval.tick().await;

        // Get and display statistics
        let stats = bridge_ref.get_connection_stats().await;
        let active_peers = bridge_ref.get_active_peers().await;

        info!("📊 Status Update:");
        info!("   Active connections: {}", stats.active_connections);
        info!("   Pending attempts: {}", stats.pending_attempts);
        info!("   Total discovered: {}", stats.total_discovered_peers);

        if !active_peers.is_empty() {
            info!("   Connected peers:");
            for (peer_id, peer_info) in &active_peers {
                info!(
                    "     - {} ({})",
                    hex::encode(peer_id)[..8].to_string(),
                    peer_info.address
                );
            }
        }

        // Demonstrate manual connection (every 2 minutes)
        if stats.active_connections < 3 && stats.total_discovered_peers > stats.active_connections {
            info!("🔗 Attempting to connect to additional peers...");
            // In a real implementation, you'd select specific peers to connect to
        }
    }
}

/// Handle peer network events
async fn handle_peer_event(event: PeerNetworkEvent) {
    match event {
        PeerNetworkEvent::PeerDiscovered {
            node_id,
            advertisement,
            confidence,
        } => {
            info!(
                "🎯 Discovered peer: {} ({}:{}) confidence: {:.2}",
                hex::encode(node_id)[..8].to_string(),
                advertisement.onion_address,
                advertisement.port,
                confidence
            );
            info!("   Capabilities: {:?}", advertisement.capabilities);
            info!("   Protocol: {}", advertisement.protocol_version);
        }

        PeerNetworkEvent::PeerConnected {
            node_id,
            peer_info,
            connection_id,
        } => {
            info!(
                "✅ Connected to peer: {} ({})",
                hex::encode(node_id)[..8].to_string(),
                peer_info.address
            );
            info!("   Connection ID: {}", connection_id);
            info!("   Agent: {}", peer_info.agent_version);
        }

        PeerNetworkEvent::PeerDisconnected {
            node_id,
            connection_id,
            reason,
        } => {
            warn!(
                "❌ Disconnected from peer: {} (connection: {})",
                hex::encode(node_id)[..8].to_string(),
                connection_id
            );
            warn!("   Reason: {}", reason);
        }

        PeerNetworkEvent::ConnectionFailed {
            node_id,
            error,
            will_retry,
        } => {
            error!(
                "🚫 Connection failed to peer: {} - {}",
                hex::encode(node_id)[..8].to_string(),
                error
            );
            if will_retry {
                info!("   Will retry connection later");
            }
        }

        PeerNetworkEvent::PeerExpired { node_id } => {
            info!(
                "⏰ Peer advertisement expired: {}",
                hex::encode(node_id)[..8].to_string()
            );
        }
    }
}

/// Example of how to create a test network
#[cfg(feature = "test-network")]
async fn create_test_network() -> Result<()> {
    info!("🧪 Creating test network with multiple nodes");

    let mut nodes = Vec::new();

    // Create 3 test nodes
    for i in 1..=3 {
        let mut node_id = [0u8; 32];
        node_id[0] = i;

        let onion_address = format!("test-node-{}.onion", i);

        let config = BitcoinBridgeConfig {
            bitcoin_network: BitcoinNetworkType::Regtest, // Use regtest for testing
            discovery_interval: Duration::from_secs(30),
            max_peers_advertised: 10,
            ..Default::default()
        };

        let tor_client = Arc::new(TorClient::new(Default::default()).await?);
        let bridge =
            IntegratedBitcoinBridge::new(config, node_id, onion_address, tor_client).await?;

        nodes.push(bridge);
    }

    // Start all nodes
    for (i, mut bridge) in nodes.into_iter().enumerate() {
        info!("Starting test node {}", i + 1);
        bridge.start().await?;

        // Give each node time to initialize
        time::sleep(Duration::from_secs(2)).await;
    }

    info!("✅ Test network created successfully");

    // Let nodes discover each other
    time::sleep(Duration::from_secs(60)).await;

    Ok(())
}

/// Demonstrate steganographic features
#[cfg(feature = "steganography-demo")]
async fn demonstrate_steganography() -> Result<()> {
    use chrono::Utc;
    use q_bitcoin_bridge::{steganography, NodeAdvertisement};

    info!("🎭 Demonstrating steganographic encoding");

    // Create test advertisement
    let advertisement = NodeAdvertisement {
        node_id: [42u8; 32],
        onion_address: "stealthnode.onion".to_string(),
        port: 8333,
        protocol_version: "qk/0.1".to_string(),
        capabilities: vec!["DAG".to_string(), "QR".to_string()],
        signature: vec![0u8; 64], // Mock signature
        timestamp: Utc::now(),
        expires_at: Utc::now() + chrono::Duration::hours(1),
    };

    // Encode steganographically
    let encoded = steganography::encode_steganographic(&advertisement).await?;
    info!("📦 Encoded advertisement: {} bytes", encoded.len());

    // Decode
    let decoded = steganography::decode_steganographic(&encoded).await?;
    info!("✅ Successfully decoded advertisement");
    info!(
        "   Node ID: {}",
        hex::encode(decoded.node_id)[..16].to_string()
    );
    info!("   Address: {}", decoded.onion_address);

    // Demonstrate cover traffic generation
    let cover_traffic = steganography::generate_cover_traffic().await?;
    info!("🎭 Generated {} cover transactions", cover_traffic.len());

    Ok(())
}
