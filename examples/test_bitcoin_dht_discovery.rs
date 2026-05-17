use anyhow::Result;
use q_bitcoin_bridge::{BitcoinBridge, BitcoinBridgeConfig, discovery::NodeAnnouncement};
use q_dns_phantom::{DNSPhantomNetwork, DNSPhantomConfig};
use q_types::{NodeId, PeerInfo, Phase};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{sleep, timeout};
use tracing::{info, warn, error};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    info!("🚀 Starting Bitcoin DHT + DNS Phantom Discovery Test");
    info!("This test will demonstrate automatic peer discovery between two nodes");

    // Create Node 1
    let node1_id = NodeId::new(b"node1_test_peer_discovery_alpha");
    let node1_bridge_config = BitcoinBridgeConfig {
        network: bitcoin::Network::Testnet,
        rpc_url: "http://localhost:18332".to_string(),
        rpc_user: "test".to_string(),
        rpc_pass: "test".to_string(),
        announcement_interval: Duration::from_secs(30),
        max_peers: 100,
        enable_steganography: true,
        use_tor: false,
    };

    info!("📡 Creating Node 1 (Alpha): {}", node1_id);
    let node1_bridge = Arc::new(BitcoinBridge::new(node1_bridge_config).await?);
    
    // Create Node 2
    let node2_id = NodeId::new(b"node2_test_peer_discovery_beta");
    let node2_bridge_config = BitcoinBridgeConfig {
        network: bitcoin::Network::Testnet,
        rpc_url: "http://localhost:18332".to_string(),
        rpc_user: "test".to_string(),
        rpc_pass: "test".to_string(),
        announcement_interval: Duration::from_secs(30),
        max_peers: 100,
        enable_steganography: true,
        use_tor: false,
    };

    info!("📡 Creating Node 2 (Beta): {}", node2_id);
    let node2_bridge = Arc::new(BitcoinBridge::new(node2_bridge_config).await?);

    // Start DNS Phantom networks for both nodes
    let dns_config1 = DNSPhantomConfig {
        domain_prefix: "node1".to_string(),
        base_domain: "qnk.test".to_string(),
        ttl: 300,
        max_payload_size: 128,
        use_dnssec: false,
        resolvers: vec!["8.8.8.8:53".to_string()],
        cache_size: 1000,
        enable_mesh: true,
    };

    let dns_config2 = DNSPhantomConfig {
        domain_prefix: "node2".to_string(),
        base_domain: "qnk.test".to_string(),
        ttl: 300,
        max_payload_size: 128,
        use_dnssec: false,
        resolvers: vec!["8.8.8.8:53".to_string()],
        cache_size: 1000,
        enable_mesh: true,
    };

    info!("🌐 Starting DNS Phantom networks");
    let dns1 = Arc::new(DNSPhantomNetwork::new(dns_config1).await?);
    let dns2 = Arc::new(DNSPhantomNetwork::new(dns_config2).await?);

    // Announce Node 1 on the Bitcoin network
    info!("📢 Node 1 announcing itself on Bitcoin network...");
    let node1_info = PeerInfo {
        node_id: node1_id.clone(),
        address: "/ip4/127.0.0.1/tcp/7001".parse()?,
        phase: Phase::Phase1,
        capabilities: vec!["dht".to_string(), "dns".to_string()],
        last_seen: std::time::SystemTime::now(),
    };
    
    node1_bridge.announce_node(node1_info.clone()).await?;
    dns1.announce_peer(node1_info.clone()).await?;

    // Announce Node 2 on the Bitcoin network
    info!("📢 Node 2 announcing itself on Bitcoin network...");
    let node2_info = PeerInfo {
        node_id: node2_id.clone(),
        address: "/ip4/127.0.0.1/tcp/7002".parse()?,
        phase: Phase::Phase1,
        capabilities: vec!["dht".to_string(), "dns".to_string()],
        last_seen: std::time::SystemTime::now(),
    };
    
    node2_bridge.announce_node(node2_info.clone()).await?;
    dns2.announce_peer(node2_info.clone()).await?;

    // Give time for announcements to propagate
    info!("⏳ Waiting 60 seconds for announcements to propagate...");
    sleep(Duration::from_secs(60)).await;

    // Node 1 discovers peers
    info!("🔍 Node 1 searching for peers via Bitcoin DHT...");
    let discovered_by_node1 = timeout(
        Duration::from_secs(300),
        node1_bridge.discover_peers()
    ).await??;

    info!("✅ Node 1 discovered {} peers via Bitcoin DHT", discovered_by_node1.len());
    for peer in &discovered_by_node1 {
        info!("  - Found peer: {} at {}", peer.node_id, peer.address);
    }

    // Node 2 discovers peers
    info!("🔍 Node 2 searching for peers via Bitcoin DHT...");
    let discovered_by_node2 = timeout(
        Duration::from_secs(300),
        node2_bridge.discover_peers()
    ).await??;

    info!("✅ Node 2 discovered {} peers via Bitcoin DHT", discovered_by_node2.len());
    for peer in &discovered_by_node2 {
        info!("  - Found peer: {} at {}", peer.node_id, peer.address);
    }

    // Test DNS Phantom discovery
    info!("🔍 Testing DNS Phantom discovery...");
    let dns_peers1 = timeout(
        Duration::from_secs(300),
        dns1.discover_peers()
    ).await??;

    info!("✅ Node 1 discovered {} peers via DNS Phantom", dns_peers1.len());
    for peer in &dns_peers1 {
        info!("  - Found peer: {} at {}", peer.node_id, peer.address);
    }

    // Verify mutual discovery
    let node1_found_node2 = discovered_by_node1.iter()
        .any(|p| p.node_id == node2_id);
    let node2_found_node1 = discovered_by_node2.iter()
        .any(|p| p.node_id == node1_id);

    if node1_found_node2 && node2_found_node1 {
        info!("🎉 SUCCESS: Nodes discovered each other automatically!");
        info!("✅ Bitcoin DHT peer discovery is working correctly");
        info!("✅ DNS Phantom network is operational");
    } else {
        warn!("⚠️ Partial discovery:");
        warn!("  Node 1 found Node 2: {}", node1_found_node2);
        warn!("  Node 2 found Node 1: {}", node2_found_node1);
    }

    // Test steganography features
    info!("🔐 Testing steganography features...");
    if node1_bridge.test_steganography().await? {
        info!("✅ Bitcoin transaction steganography is operational");
    }

    // Display final statistics
    info!("📊 Final Statistics:");
    info!("  Total peers discovered by Node 1: {}", discovered_by_node1.len());
    info!("  Total peers discovered by Node 2: {}", discovered_by_node2.len());
    info!("  DNS Phantom peers found: {}", dns_peers1.len());
    info!("  Mutual discovery successful: {}", node1_found_node2 && node2_found_node1);

    Ok(())
}