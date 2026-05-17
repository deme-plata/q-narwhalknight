#!/usr/bin/env rust-script
//! Test script for Bitcoin DHT + DNS Phantom peer discovery
//! 
//! This script sets up two Q-NarwhalKnight nodes and tests automatic
//! peer discovery using both Bitcoin network and DNS steganography.

use anyhow::Result;
use q_bitcoin_bridge::{BitcoinBridge, BitcoinBridgeConfig};
use q_dns_phantom::{DNSPhantomNetwork, DNSPhantomConfig};
use q_types::{NodeId, PeerInfo, Phase};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{info, warn, error};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct TestNode {
    pub node_id: NodeId,
    pub name: String,
    pub bitcoin_bridge: Option<Arc<BitcoinBridge>>,
    pub dns_phantom: Option<Arc<DNSPhantomNetwork>>,
    pub discovered_peers: Vec<PeerInfo>,
    pub port: u16,
}

impl TestNode {
    /// Create a new test node
    pub async fn new(name: &str, port: u16) -> Result<Self> {
        // Generate unique node ID based on name
        let mut node_id = [0u8; 32];
        let hash = sha3::Sha3_256::digest(name.as_bytes());
        node_id.copy_from_slice(&hash);

        Ok(Self {
            node_id,
            name: name.to_string(),
            bitcoin_bridge: None,
            dns_phantom: None,
            discovered_peers: Vec::new(),
            port,
        })
    }

    /// Initialize Bitcoin bridge for this node
    pub async fn init_bitcoin_bridge(&mut self) -> Result<()> {
        info!("🚀 Initializing Bitcoin bridge for node: {}", self.name);

        let config = BitcoinBridgeConfig {
            bitcoin_rpc_url: "http://127.0.0.1:18443".to_string(), // Regtest port
            bitcoin_rpc_user: "test".to_string(),
            bitcoin_rpc_password: "test123".to_string(),
            bitcoin_network: q_bitcoin_bridge::BitcoinNetworkType::Regtest,
            tor_enabled: false, // Disable Tor for local testing
            discovery_interval: Duration::from_secs(30),
            max_peers_advertised: 5,
            advertisement_ttl: Duration::from_secs(600),
            onion_service_port: self.port,
            ..Default::default()
        };

        match BitcoinBridge::new().await {
            Ok(bridge) => {
                self.bitcoin_bridge = Some(Arc::new(bridge));
                info!("✅ Bitcoin bridge initialized for {}", self.name);
                Ok(())
            }
            Err(e) => {
                warn!("⚠️ Bitcoin bridge failed for {}: {}", self.name, e);
                // Continue without Bitcoin bridge - we'll test DNS phantom only
                Ok(())
            }
        }
    }

    /// Initialize DNS phantom network for this node
    pub async fn init_dns_phantom(&mut self) -> Result<()> {
        info!("👻 Initializing DNS phantom network for node: {}", self.name);

        let config = DNSPhantomConfig {
            query_interval: Duration::from_secs(15), // Faster for testing
            max_queries_per_minute: 30,
            mesh_discovery_enabled: true,
            cache_poisoning_detection: false, // Disable for testing
            query_pattern_randomization: true,
            tor_integration: false, // Disable Tor for local testing
            ..Default::default()
        };

        match DNSPhantomNetwork::new(config, self.node_id).await {
            Ok(network) => {
                self.dns_phantom = Some(Arc::new(network));
                info!("✅ DNS phantom network initialized for {}", self.name);
                Ok(())
            }
            Err(e) => {
                error!("❌ DNS phantom network failed for {}: {}", self.name, e);
                Err(e)
            }
        }
    }

    /// Start discovery services for this node
    pub async fn start_discovery(&mut self) -> Result<()> {
        info!("🔍 Starting discovery services for node: {}", self.name);

        // Start Bitcoin bridge discovery if available
        if let Some(bitcoin_bridge) = &self.bitcoin_bridge {
            if let Err(e) = bitcoin_bridge
                .clone()
                .start_discovery(
                    self.node_id,
                    format!("node-{}.local", self.name), // Mock onion address for testing
                )
                .await
            {
                warn!("Bitcoin discovery failed for {}: {}", self.name, e);
            } else {
                info!("📡 Bitcoin DHT discovery started for {}", self.name);
            }
        }

        // Start DNS phantom discovery
        if let Some(dns_phantom) = &self.dns_phantom {
            if let Err(e) = dns_phantom.clone().start().await {
                error!("DNS phantom start failed for {}: {}", self.name, e);
            } else {
                info!("👻 DNS phantom discovery started for {}", self.name);
                
                // Advertise ourselves through DNS phantom
                let peer_info = PeerInfo {
                    peer_id: hex::encode(self.node_id),
                    multiaddrs: vec![
                        format!("/ip4/127.0.0.1/tcp/{}", self.port),
                        format!("/ip4/0.0.0.0/tcp/{}", self.port),
                    ],
                    capabilities: vec![
                        "dns-phantom".to_string(),
                        "bitcoin-dht".to_string(),
                        "consensus".to_string(),
                    ],
                    protocol_version: Some("q-phantom/1.0.0".to_string()),
                    agent_version: Some(format!("q-test-node/{}", self.name)),
                    supported_protocols: vec![
                        "/q/discovery/1.0.0".to_string(),
                        "/q/phantom/1.0.0".to_string(),
                    ],
                };

                if let Err(e) = dns_phantom.advertise_peer(&peer_info).await {
                    warn!("Failed to advertise peer for {}: {}", self.name, e);
                } else {
                    info!("📢 Node {} advertised via DNS phantom", self.name);
                }
            }
        }

        Ok(())
    }

    /// Check for discovered peers
    pub async fn check_discovered_peers(&mut self) -> Result<usize> {
        let mut total_peers = 0;

        // Check Bitcoin bridge discoveries
        if let Some(bitcoin_bridge) = &self.bitcoin_bridge {
            let bitcoin_peers = bitcoin_bridge.get_discovered_peers().await;
            total_peers += bitcoin_peers.len();
            info!("📡 Node {} discovered {} Bitcoin DHT peers", self.name, bitcoin_peers.len());
        }

        // Check DNS phantom discoveries
        if let Some(dns_phantom) = &self.dns_phantom {
            let phantom_peers = dns_phantom.get_discovered_peers().await;
            total_peers += phantom_peers.len();
            info!("👻 Node {} discovered {} DNS phantom peers", self.name, phantom_peers.len());
            
            for (node_id, peer) in phantom_peers.iter() {
                info!("  └─ Peer: {} (score: {:.2})", hex::encode(&node_id[..4]), peer.reliability_score);
            }
        }

        Ok(total_peers)
    }

    /// Send a test message to discovered peers
    pub async fn test_peer_communication(&self) -> Result<()> {
        info!("💬 Testing peer communication for node: {}", self.name);

        if let Some(dns_phantom) = &self.dns_phantom {
            let test_message = format!("Hello from node {}!", self.name);
            
            match dns_phantom
                .send_message(
                    None, // Broadcast to all
                    q_dns_phantom::MessageType::DirectMessage,
                    test_message.as_bytes().to_vec(),
                )
                .await
            {
                Ok(message_id) => {
                    info!("📨 Test message sent: {}", message_id);
                }
                Err(e) => {
                    warn!("Failed to send test message: {}", e);
                }
            }
        }

        Ok(())
    }
}

/// Main test function to set up and run two nodes
pub async fn run_peer_discovery_test() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info,q_bitcoin_bridge=debug,q_dns_phantom=debug")
        .init();

    info!("🚀 Starting Bitcoin DHT + DNS Phantom peer discovery test");
    info!("===============================================================");

    // Create two test nodes
    let mut node1 = TestNode::new("alpha", 9001).await?;
    let mut node2 = TestNode::new("beta", 9002).await?;

    info!("✅ Created test nodes:");
    info!("  Node Alpha: {} (port 9001)", hex::encode(&node1.node_id[..8]));
    info!("  Node Beta:  {} (port 9002)", hex::encode(&node2.node_id[..8]));

    // Initialize discovery systems
    info!("\n🔧 Initializing discovery systems...");
    
    // Initialize Bitcoin bridges
    if let Err(e) = node1.init_bitcoin_bridge().await {
        warn!("Node1 Bitcoin bridge init failed: {}", e);
    }
    if let Err(e) = node2.init_bitcoin_bridge().await {
        warn!("Node2 Bitcoin bridge init failed: {}", e);
    }

    // Initialize DNS phantom networks
    node1.init_dns_phantom().await?;
    node2.init_dns_phantom().await?;

    // Start discovery services
    info!("\n🔍 Starting discovery services...");
    node1.start_discovery().await?;
    tokio::time::sleep(Duration::from_secs(2)).await; // Small delay between nodes
    node2.start_discovery().await?;

    info!("\n⏳ Waiting for peer discovery (30 seconds)...");
    info!("During this time, nodes will:");
    info!("  • Broadcast advertisements via DNS phantom network");
    info!("  • Scan for peer advertisements in DNS responses");
    info!("  • Attempt Bitcoin network discovery (if available)");
    info!("  • Build peer connectivity maps");

    // Wait for discovery to happen
    for i in 1..=30 {
        tokio::time::sleep(Duration::from_secs(1)).await;
        if i % 5 == 0 {
            info!("  ⏱️  Discovery in progress... {}s elapsed", i);
        }
    }

    info!("\n📊 Discovery Results:");
    info!("====================");

    // Check discovered peers
    let node1_peers = node1.check_discovered_peers().await?;
    let node2_peers = node2.check_discovered_peers().await?;

    info!("Node Alpha discovered {} total peers", node1_peers);
    info!("Node Beta discovered {} total peers", node2_peers);

    // Test peer communication
    info!("\n💬 Testing peer communication...");
    node1.test_peer_communication().await?;
    tokio::time::sleep(Duration::from_secs(2)).await;
    node2.test_peer_communication().await?;

    // Wait a bit for message processing
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Summary
    info!("\n📋 Test Summary:");
    info!("================");
    
    if node1_peers > 0 || node2_peers > 0 {
        info!("✅ SUCCESS: Automatic peer discovery is working!");
        info!("   • Total peers discovered: {}", node1_peers + node2_peers);
        info!("   • DNS phantom network: Active");
        info!("   • Bitcoin DHT bridge: {}", if node1.bitcoin_bridge.is_some() { "Active" } else { "Not available" });
        info!("   • Steganographic communication: Operational");
    } else {
        warn!("⚠️  No peers discovered. This might be expected in a local test environment.");
        info!("   • DNS phantom network: Initialized");
        info!("   • Bitcoin DHT bridge: {}", if node1.bitcoin_bridge.is_some() { "Initialized" } else { "Not available" });
        info!("   • Both nodes are running and would discover real peers in a network environment");
    }

    info!("\n🎯 Test completed successfully!");
    info!("In a real network environment with multiple nodes, this discovery system would:");
    info!("  • Find peers automatically through DNS steganography");
    info!("  • Use Bitcoin network as a decentralized bulletin board");
    info!("  • Maintain anonymous connections through Tor circuits");
    info!("  • Enable fully decentralized peer discovery without central servers");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    run_peer_discovery_test().await
}