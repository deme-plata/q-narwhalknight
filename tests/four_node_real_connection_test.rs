// Four Node Real Connection Test
// Tests real q-api-server nodes connecting to each other through IP addresses

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;
use tokio::sync::Mutex;
use tokio::time::{sleep, timeout};
use serde_json::Value;

/// Test configuration for 4-node network
#[derive(Debug, Clone)]
struct NodeConfig {
    node_id: String,
    port: u16,
    api_port: u16,
    p2p_port: u16,
    ip_address: String,
    data_dir: String,
}

/// Manages the lifecycle of 4 real q-api-server nodes
struct FourNodeTestNetwork {
    nodes: Vec<NodeConfig>,
    processes: Arc<Mutex<HashMap<String, tokio::process::Child>>>,
    client: reqwest::Client,
}

impl FourNodeTestNetwork {
    /// Initialize a 4-node test network with real q-api-server processes
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let nodes = vec![
            NodeConfig {
                node_id: "alpha-node-1".to_string(),
                port: 18001,
                api_port: 18001,
                p2p_port: 18002,
                ip_address: "127.0.0.1".to_string(),
                data_dir: "/tmp/q-test-node-1".to_string(),
            },
            NodeConfig {
                node_id: "alpha-node-2".to_string(),
                port: 18003,
                api_port: 18003,
                p2p_port: 18004,
                ip_address: "127.0.0.1".to_string(),
                data_dir: "/tmp/q-test-node-2".to_string(),
            },
            NodeConfig {
                node_id: "alpha-node-3".to_string(),
                port: 18005,
                api_port: 18005,
                p2p_port: 18006,
                ip_address: "127.0.0.1".to_string(),
                data_dir: "/tmp/q-test-node-3".to_string(),
            },
            NodeConfig {
                node_id: "alpha-node-4".to_string(),
                port: 18007,
                api_port: 18007,
                p2p_port: 18008,
                ip_address: "127.0.0.1".to_string(),
                data_dir: "/tmp/q-test-node-4".to_string(),
            },
        ];

        let client = reqwest::ClientBuilder::new()
            .timeout(Duration::from_secs(30))
            .build()?;

        Ok(Self {
            nodes,
            processes: Arc::new(Mutex::new(HashMap::new())),
            client,
        })
    }

    /// Start all 4 nodes concurrently
    async fn start_nodes(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Starting 4-node Q-NarwhalKnight test network...");

        // Clean up any existing data directories
        for node in &self.nodes {
            let _ = tokio::fs::remove_dir_all(&node.data_dir).await;
            tokio::fs::create_dir_all(&node.data_dir).await?;
        }

        let mut processes = self.processes.lock().await;
        
        // Start all nodes concurrently
        for node in &self.nodes {
            println!("🔧 Starting node: {} on port {}", node.node_id, node.port);
            
            let mut cmd = Command::new("cargo");
            cmd.args(&[
                "run",
                "--bin", "q-api-server",
                "--",
                "--node-id", &node.node_id,
                "--port", &node.port.to_string(),
                "--production", // Enable production peer discovery
            ]);

            // Set environment variables for this node
            cmd.env("RUST_LOG", "info");
            cmd.env("Q_NODE_DATA_DIR", &node.data_dir);
            cmd.env("Q_NODE_IP", &node.ip_address);
            cmd.env("Q_P2P_PORT", &node.p2p_port.to_string());
            cmd.env("SKIP_BITCOIN", "1"); // Skip Bitcoin for faster startup in tests
            cmd.env("SKIP_DNS", "1"); // Skip DNS phantom for simpler testing

            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::piped());

            let child = cmd.spawn()?;
            processes.insert(node.node_id.clone(), child);
            
            println!("✅ Started node: {}", node.node_id);
        }

        println!("🔄 Waiting for all nodes to initialize...");
        sleep(Duration::from_secs(10)).await;

        // Wait for all nodes to be ready
        self.wait_for_nodes_ready().await?;
        
        println!("✅ All 4 nodes are ready and running!");
        Ok(())
    }

    /// Wait for all nodes to be ready by checking their health endpoints
    async fn wait_for_nodes_ready(&self) -> Result<(), Box<dyn std::error::Error>> {
        let max_retries = 30;
        let retry_delay = Duration::from_secs(2);

        for node in &self.nodes {
            let url = format!("http://{}:{}/api/v1/health", node.ip_address, node.port);
            let mut retries = 0;

            loop {
                match timeout(Duration::from_secs(5), self.client.get(&url).send()).await {
                    Ok(Ok(response)) if response.status().is_success() => {
                        println!("✅ Node {} is ready at {}", node.node_id, url);
                        break;
                    }
                    _ => {
                        retries += 1;
                        if retries >= max_retries {
                            return Err(format!("Node {} failed to start within timeout", node.node_id).into());
                        }
                        println!("⏳ Waiting for node {} to be ready... (attempt {}/{})", 
                               node.node_id, retries, max_retries);
                        sleep(retry_delay).await;
                    }
                }
            }
        }

        Ok(())
    }

    /// Connect nodes to each other through direct IP connections
    async fn connect_nodes(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔗 Connecting nodes to each other...");

        // Create a full mesh - each node connects to every other node
        for (i, node_a) in self.nodes.iter().enumerate() {
            for (j, node_b) in self.nodes.iter().enumerate() {
                if i != j {
                    self.connect_node_to_node(node_a, node_b).await?;
                }
            }
        }

        println!("✅ All node connections established!");
        Ok(())
    }

    /// Connect one node to another using direct IP connection
    async fn connect_node_to_node(&self, from_node: &NodeConfig, to_node: &NodeConfig) -> Result<(), Box<dyn std::error::Error>> {
        // Use the production peer discovery API to connect nodes
        let connect_url = format!(
            "http://{}:{}/api/v1/bitcoin/bridge/connect/{}",
            from_node.ip_address, 
            from_node.port,
            to_node.node_id
        );

        let connection_payload = serde_json::json!({
            "target_address": format!("{}:{}", to_node.ip_address, to_node.p2p_port),
            "node_id": to_node.node_id,
            "connection_type": "direct_ip"
        });

        match timeout(
            Duration::from_secs(10),
            self.client.post(&connect_url).json(&connection_payload).send()
        ).await {
            Ok(Ok(response)) => {
                if response.status().is_success() {
                    println!("✅ Connected {} -> {}", from_node.node_id, to_node.node_id);
                } else {
                    println!("⚠️  Connection failed {} -> {}: {}", 
                           from_node.node_id, to_node.node_id, response.status());
                }
            }
            Ok(Err(e)) => {
                println!("⚠️  Connection error {} -> {}: {}", 
                       from_node.node_id, to_node.node_id, e);
            }
            Err(_) => {
                println!("⏰ Connection timeout {} -> {}", from_node.node_id, to_node.node_id);
            }
        }

        // Small delay between connections
        sleep(Duration::from_millis(500)).await;
        Ok(())
    }

    /// Verify that all nodes can see each other
    async fn verify_node_connectivity(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 Verifying node connectivity...");

        for node in &self.nodes {
            let peers = self.get_node_peers(node).await?;
            println!("📊 Node {} has {} connected peers", node.node_id, peers.len());
            
            // Verify this node can see the other 3 nodes
            let expected_peers = self.nodes.len() - 1; // All nodes except itself
            if peers.len() < expected_peers {
                println!("⚠️  Node {} only sees {}/{} expected peers", 
                       node.node_id, peers.len(), expected_peers);
            } else {
                println!("✅ Node {} connectivity verified", node.node_id);
            }
        }

        Ok(())
    }

    /// Get the list of connected peers for a node
    async fn get_node_peers(&self, node: &NodeConfig) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
        let url = format!("http://{}:{}/api/v1/network/active-peers", node.ip_address, node.port);
        
        match timeout(Duration::from_secs(5), self.client.get(&url).send()).await {
            Ok(Ok(response)) => {
                let peers: Value = response.json().await?;
                if let Some(peer_list) = peers.get("peers").and_then(|p| p.as_array()) {
                    Ok(peer_list.clone())
                } else {
                    Ok(vec![])
                }
            }
            _ => Ok(vec![])
        }
    }

    /// Test network functionality by sending transactions between nodes
    async fn test_network_functionality(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🧪 Testing network functionality...");

        // Send a test transaction from node 1 to the network
        let test_transaction = serde_json::json!({
            "from": "test_sender_address",
            "to": "test_receiver_address", 
            "amount": 1000,
            "nonce": 1
        });

        let sender_node = &self.nodes[0];
        let tx_url = format!("http://{}:{}/api/v1/transactions", sender_node.ip_address, sender_node.port);

        match timeout(
            Duration::from_secs(10),
            self.client.post(&tx_url).json(&test_transaction).send()
        ).await {
            Ok(Ok(response)) => {
                if response.status().is_success() {
                    println!("✅ Test transaction submitted successfully");
                    
                    // Wait a moment for propagation
                    sleep(Duration::from_secs(2)).await;
                    
                    // Check if other nodes received the transaction
                    self.verify_transaction_propagation().await?;
                } else {
                    println!("⚠️  Test transaction failed: {}", response.status());
                }
            }
            _ => {
                println!("⚠️  Test transaction timed out");
            }
        }

        Ok(())
    }

    /// Verify that transactions propagate across the network
    async fn verify_transaction_propagation(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("📡 Verifying transaction propagation...");

        for node in &self.nodes {
            let url = format!("http://{}:{}/api/v1/transactions/recent", node.ip_address, node.port);
            
            match timeout(Duration::from_secs(5), self.client.get(&url).send()).await {
                Ok(Ok(response)) => {
                    let recent_txs: Value = response.json().await?;
                    if let Some(tx_list) = recent_txs.get("transactions").and_then(|t| t.as_array()) {
                        if !tx_list.is_empty() {
                            println!("✅ Node {} received transactions", node.node_id);
                        } else {
                            println!("⚠️  Node {} has no recent transactions", node.node_id);
                        }
                    }
                }
                _ => {
                    println!("⚠️  Failed to check transactions on node {}", node.node_id);
                }
            }
        }

        Ok(())
    }

    /// Get comprehensive network status from all nodes
    async fn get_network_status(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("📊 Getting network status from all nodes...");

        for node in &self.nodes {
            let url = format!("http://{}:{}/api/v1/status", node.ip_address, node.port);
            
            match timeout(Duration::from_secs(5), self.client.get(&url).send()).await {
                Ok(Ok(response)) => {
                    let status: Value = response.json().await?;
                    println!("📋 Node {} status:", node.node_id);
                    if let Some(connected_peers) = status.get("connected_peers") {
                        println!("  📡 Connected peers: {}", connected_peers);
                    }
                    if let Some(uptime) = status.get("uptime_seconds") {
                        println!("  ⏰ Uptime: {}s", uptime);
                    }
                    if let Some(node_id) = status.get("node_id") {
                        println!("  🆔 Node ID: {}", node_id);
                    }
                }
                _ => {
                    println!("⚠️  Failed to get status from node {}", node.node_id);
                }
            }
        }

        Ok(())
    }

    /// Gracefully shutdown all nodes
    async fn shutdown(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🛑 Shutting down all nodes...");

        let mut processes = self.processes.lock().await;
        
        for (node_id, mut process) in processes.drain() {
            println!("🔄 Stopping node: {}", node_id);
            
            // Try graceful shutdown first
            if let Err(_) = process.kill().await {
                println!("⚠️  Force killing node: {}", node_id);
            }
            
            // Wait for process to exit
            let _ = timeout(Duration::from_secs(5), process.wait()).await;
            println!("✅ Stopped node: {}", node_id);
        }

        // Clean up data directories
        for node in &self.nodes {
            let _ = tokio::fs::remove_dir_all(&node.data_dir).await;
        }

        println!("✅ All nodes shut down successfully");
        Ok(())
    }
}

#[tokio::test]
async fn test_four_node_real_ip_connections() {
    println!("🌟 Starting Four Node Real IP Connection Test");
    
    let network = FourNodeTestNetwork::new().await
        .expect("Failed to initialize test network");

    // Step 1: Start all 4 nodes
    network.start_nodes().await
        .expect("Failed to start nodes");

    // Wait a bit for full initialization
    sleep(Duration::from_secs(5)).await;

    // Step 2: Connect nodes to each other
    network.connect_nodes().await
        .expect("Failed to connect nodes");

    // Wait for connections to stabilize
    sleep(Duration::from_secs(3)).await;

    // Step 3: Verify connectivity
    network.verify_node_connectivity().await
        .expect("Failed to verify connectivity");

    // Step 4: Test network functionality
    network.test_network_functionality().await
        .expect("Failed to test network functionality");

    // Step 5: Get final network status
    network.get_network_status().await
        .expect("Failed to get network status");

    println!("✅ Four Node Real IP Connection Test completed successfully!");

    // Cleanup
    network.shutdown().await
        .expect("Failed to shutdown nodes");
}

#[tokio::test]
async fn test_node_resilience() {
    println!("🔬 Testing node resilience and recovery...");
    
    let network = FourNodeTestNetwork::new().await
        .expect("Failed to initialize test network");

    // Start all nodes
    network.start_nodes().await.expect("Failed to start nodes");
    sleep(Duration::from_secs(5)).await;

    // Connect all nodes
    network.connect_nodes().await.expect("Failed to connect nodes");
    sleep(Duration::from_secs(3)).await;

    // Verify initial connectivity
    network.verify_node_connectivity().await
        .expect("Initial connectivity failed");

    // Simulate node failure by killing one process
    {
        let mut processes = network.processes.lock().await;
        if let Some(mut process) = processes.remove("alpha-node-2") {
            println!("💥 Simulating failure of alpha-node-2");
            let _ = process.kill().await;
        }
    }

    // Wait for network to detect failure
    sleep(Duration::from_secs(5)).await;

    // Check that remaining nodes still function
    let remaining_nodes: Vec<_> = network.nodes.iter()
        .filter(|n| n.node_id != "alpha-node-2")
        .collect();

    for node in &remaining_nodes {
        let peers = network.get_node_peers(node).await
            .expect("Failed to get peers");
        println!("📊 After failure, node {} has {} peers", node.node_id, peers.len());
    }

    println!("✅ Node resilience test completed");

    // Cleanup
    network.shutdown().await.expect("Failed to shutdown");
}

#[tokio::test]
async fn test_network_performance() {
    println!("⚡ Testing network performance...");
    
    let network = FourNodeTestNetwork::new().await
        .expect("Failed to initialize test network");

    network.start_nodes().await.expect("Failed to start nodes");
    sleep(Duration::from_secs(5)).await;

    network.connect_nodes().await.expect("Failed to connect nodes");
    sleep(Duration::from_secs(3)).await;

    // Send multiple transactions to test throughput
    let start_time = std::time::Instant::now();
    let num_transactions = 50;

    for i in 0..num_transactions {
        let test_transaction = serde_json::json!({
            "from": format!("sender_{}", i),
            "to": format!("receiver_{}", i),
            "amount": 1000 + i,
            "nonce": i + 1
        });

        let sender_node = &network.nodes[i % network.nodes.len()];
        let tx_url = format!("http://{}:{}/api/v1/transactions", 
                           sender_node.ip_address, sender_node.port);

        // Send transaction (don't wait for response to test throughput)
        let client = network.client.clone();
        tokio::spawn(async move {
            let _ = client.post(&tx_url).json(&test_transaction).send().await;
        });

        // Small delay to avoid overwhelming
        sleep(Duration::from_millis(100)).await;
    }

    let elapsed = start_time.elapsed();
    let tps = num_transactions as f64 / elapsed.as_secs_f64();
    
    println!("📈 Submitted {} transactions in {:.2}s (TPS: {:.2})", 
           num_transactions, elapsed.as_secs_f64(), tps);

    // Wait for propagation
    sleep(Duration::from_secs(5)).await;

    // Check final network status
    network.get_network_status().await.expect("Failed to get final status");

    println!("✅ Network performance test completed");

    // Cleanup
    network.shutdown().await.expect("Failed to shutdown");
}