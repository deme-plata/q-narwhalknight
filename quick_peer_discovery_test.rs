#!/usr/bin/env rust-script
//! Quick Q-NarwhalKnight Peer Discovery Test
//! Demonstrates automatic peer discovery and connection capabilities

use std::{
    collections::HashMap,
    time::{Duration, Instant},
    thread,
};
use tokio::{
    net::{TcpListener, TcpStream},
    io::{AsyncReadExt, AsyncWriteExt},
    time::sleep,
};

#[derive(Debug, Clone)]
struct PeerInfo {
    id: u8,
    name: String,
    address: String,
    discovered_at: Instant,
    connection_status: ConnectionStatus,
}

#[derive(Debug, Clone)]
enum ConnectionStatus {
    Discovered,
    Connecting,
    Connected,
    Failed,
}

#[derive(Debug)]
struct AutoDiscoveryNode {
    id: u8,
    name: String,
    port: u16,
    discovered_peers: HashMap<u8, PeerInfo>,
    connections: Vec<u8>,
}

impl AutoDiscoveryNode {
    fn new(id: u8, name: String, port: u16) -> Self {
        Self {
            id,
            name,
            port,
            discovered_peers: HashMap::new(),
            connections: Vec::new(),
        }
    }

    async fn start_discovery_listener(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 {} starting automatic peer discovery on port {}", self.name, self.port);
        
        let listener = TcpListener::bind(format!("127.0.0.1:{}", self.port)).await?;
        let name = self.name.clone();
        let node_id = self.id;
        
        tokio::spawn(async move {
            while let Ok((mut socket, addr)) = listener.accept().await {
                println!("📡 {} discovered incoming connection from {}", name, addr);
                
                let mut buffer = [0; 256];
                if let Ok(n) = socket.read(&mut buffer).await {
                    let message = String::from_utf8_lossy(&buffer[..n]);
                    if message.starts_with("PEER_DISCOVERY:") {
                        let peer_data = &message[15..];
                        println!("🤝 {} received peer discovery: {}", name, peer_data);
                        
                        // Send acknowledgment with our own info
                        let response = format!("PEER_ACK:{}:{}", node_id, name);
                        let _ = socket.write_all(response.as_bytes()).await;
                        println!("✅ {} sent peer acknowledgment", name);
                    }
                }
            }
        });
        
        Ok(())
    }

    async fn discover_peers(&mut self, peer_ports: &[u16]) {
        println!("🔎 {} attempting to discover peers on ports: {:?}", self.name, peer_ports);
        
        for &port in peer_ports {
            if port == self.port {
                continue; // Skip self
            }
            
            match self.attempt_peer_discovery(port).await {
                Ok(peer_info) => {
                    println!("🎉 {} successfully discovered peer: {}", self.name, peer_info.name);
                    self.discovered_peers.insert(peer_info.id, peer_info);
                }
                Err(e) => {
                    println!("⚠️  {} failed to discover peer on port {}: {}", self.name, port, e);
                }
            }
        }
        
        println!("📊 {} discovery complete. Found {} peers", 
                self.name, self.discovered_peers.len());
    }

    async fn attempt_peer_discovery(&self, port: u16) -> Result<PeerInfo, Box<dyn std::error::Error>> {
        let address = format!("127.0.0.1:{}", port);
        
        match TcpStream::connect(&address).await {
            Ok(mut stream) => {
                // Send discovery message
                let discovery_msg = format!("PEER_DISCOVERY:{}:{}", self.id, self.name);
                stream.write_all(discovery_msg.as_bytes()).await?;
                
                // Wait for response
                let mut buffer = [0; 256];
                let n = stream.read(&mut buffer).await?;
                let response = String::from_utf8_lossy(&buffer[..n]);
                
                if response.starts_with("PEER_ACK:") {
                    let parts: Vec<&str> = response[9..].split(':').collect();
                    if parts.len() >= 2 {
                        let peer_id: u8 = parts[0].parse()?;
                        let peer_name = parts[1..].join(":");
                        
                        return Ok(PeerInfo {
                            id: peer_id,
                            name: peer_name,
                            address: address.clone(),
                            discovered_at: Instant::now(),
                            connection_status: ConnectionStatus::Connected,
                        });
                    }
                }
                
                Err("Invalid peer response".into())
            }
            Err(e) => Err(e.into())
        }
    }

    async fn establish_connections(&mut self) {
        println!("🔗 {} attempting to establish connections with discovered peers", self.name);
        
        for (peer_id, peer_info) in &mut self.discovered_peers {
            println!("🤝 {} connecting to peer {} ({})", 
                    self.name, peer_info.name, peer_info.address);
            
            peer_info.connection_status = ConnectionStatus::Connecting;
            
            // Simulate connection establishment
            match TcpStream::connect(&peer_info.address).await {
                Ok(mut stream) => {
                    let connect_msg = format!("CONNECT:{}:{}", self.id, self.name);
                    if stream.write_all(connect_msg.as_bytes()).await.is_ok() {
                        peer_info.connection_status = ConnectionStatus::Connected;
                        self.connections.push(*peer_id);
                        println!("✅ {} successfully connected to peer {}", 
                                self.name, peer_info.name);
                    } else {
                        peer_info.connection_status = ConnectionStatus::Failed;
                        println!("❌ {} failed to send connect message to peer {}", 
                                self.name, peer_info.name);
                    }
                }
                Err(e) => {
                    peer_info.connection_status = ConnectionStatus::Failed;
                    println!("❌ {} failed to connect to peer {}: {}", 
                            self.name, peer_info.name, e);
                }
            }
        }
        
        println!("🌐 {} established {} connections", self.name, self.connections.len());
    }

    fn print_network_status(&self) {
        println!("\n📊 NETWORK STATUS FOR {}", self.name);
        println!("================================");
        println!("Node ID: {}", self.id);
        println!("Listening Port: {}", self.port);
        println!("Discovered Peers: {}", self.discovered_peers.len());
        println!("Active Connections: {}", self.connections.len());
        
        if !self.discovered_peers.is_empty() {
            println!("\n🤝 Discovered Peers:");
            for (id, peer) in &self.discovered_peers {
                println!("  • Peer {} ({}): {:?} @ {}", 
                        id, peer.name, peer.connection_status, peer.address);
            }
        }
        
        if !self.connections.is_empty() {
            println!("\n🔗 Active Connections: {:?}", self.connections);
        }
        println!();
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Q-NARWHALKNIGHT AUTOMATIC PEER DISCOVERY TEST");
    println!("=================================================");
    println!("Testing automatic peer discovery and connection establishment");
    println!();
    
    // Create test nodes
    let mut nodes = vec![
        AutoDiscoveryNode::new(1, "Alpha-Validator".to_string(), 8001),
        AutoDiscoveryNode::new(2, "Beta-Consensus".to_string(), 8002),
        AutoDiscoveryNode::new(3, "Gamma-Storage".to_string(), 8003),
        AutoDiscoveryNode::new(4, "Delta-Network".to_string(), 8004),
        AutoDiscoveryNode::new(5, "Epsilon-Bridge".to_string(), 8005),
    ];
    
    let peer_ports: Vec<u16> = nodes.iter().map(|n| n.port).collect();
    
    // Phase 1: Start discovery listeners
    println!("1️⃣ Starting discovery listeners...");
    for node in &mut nodes {
        node.start_discovery_listener().await?;
        println!("   ✅ {} discovery listener started", node.name);
    }
    
    // Wait for listeners to be ready
    sleep(Duration::from_millis(500)).await;
    println!();
    
    // Phase 2: Peer discovery
    println!("2️⃣ Executing automatic peer discovery...");
    for node in &mut nodes {
        node.discover_peers(&peer_ports).await;
        sleep(Duration::from_millis(200)).await; // Stagger discovery attempts
    }
    println!();
    
    // Phase 3: Connection establishment
    println!("3️⃣ Establishing network connections...");
    for node in &mut nodes {
        node.establish_connections().await;
        sleep(Duration::from_millis(100)).await;
    }
    println!();
    
    // Phase 4: Network analysis
    println!("4️⃣ Analyzing network topology...");
    
    let mut total_discoveries = 0;
    let mut total_connections = 0;
    let mut successful_nodes = 0;
    
    for node in &nodes {
        node.print_network_status();
        
        total_discoveries += node.discovered_peers.len();
        total_connections += node.connections.len();
        
        if node.connections.len() > 0 {
            successful_nodes += 1;
        }
    }
    
    // Final analysis
    println!("🎯 NETWORK FORMATION ANALYSIS");
    println!("==============================");
    println!("Total Nodes: {}", nodes.len());
    println!("Nodes with Connections: {}", successful_nodes);
    println!("Total Peer Discoveries: {}", total_discoveries);
    println!("Total Active Connections: {}", total_connections);
    println!("Average Connections per Node: {:.1}", 
            total_connections as f64 / nodes.len() as f64);
    
    let formation_success = (successful_nodes as f64 / nodes.len() as f64) * 100.0;
    println!("Network Formation Success: {:.1}%", formation_success);
    
    match formation_success {
        f if f >= 80.0 => {
            println!("🟢 EXCELLENT - Nodes automatically discovered and connected to each other!");
            println!("✅ Automatic peer discovery works as designed");
        },
        f if f >= 60.0 => {
            println!("🟡 GOOD - Most nodes successfully formed connections");
            println!("✅ Automatic discovery functional with minor issues");
        },
        f if f >= 40.0 => {
            println!("🟠 FAIR - Partial network formation achieved");
            println!("⚠️  Some connectivity issues detected");
        },
        _ => {
            println!("🔴 POOR - Limited automatic connectivity");
            println!("❌ Peer discovery mechanism needs improvement");
        }
    }
    
    println!("\n🎉 AUTOMATIC PEER DISCOVERY TEST COMPLETE!");
    println!("Result: Nodes {} automatically connect to each other", 
            if formation_success >= 60.0 { "CAN" } else { "CANNOT reliably" });
    
    Ok(())
}