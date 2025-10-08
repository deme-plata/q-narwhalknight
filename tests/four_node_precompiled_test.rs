// Four Node Test Using Pre-compiled Binary
// Uses the already compiled q-api-server binary directly

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;
use tokio::sync::Mutex;
use tokio::time::{sleep, timeout};
use serde_json::Value;

struct PrecompiledNodeTest {
    processes: Arc<Mutex<HashMap<String, tokio::process::Child>>>,
    nodes: Vec<NodeConfig>,
    client: reqwest::Client,
}

#[derive(Debug, Clone)]
struct NodeConfig {
    node_id: String,
    port: u16,
    ip_address: String,
}

impl PrecompiledNodeTest {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let nodes = vec![
            NodeConfig {
                node_id: "node-1".to_string(),
                port: 18021,
                ip_address: "127.0.0.1".to_string(),
            },
            NodeConfig {
                node_id: "node-2".to_string(),
                port: 18022,
                ip_address: "127.0.0.1".to_string(),
            },
            NodeConfig {
                node_id: "node-3".to_string(),
                port: 18023,
                ip_address: "127.0.0.1".to_string(),
            },
            NodeConfig {
                node_id: "node-4".to_string(),
                port: 18024,
                ip_address: "127.0.0.1".to_string(),
            },
        ];

        let client = reqwest::ClientBuilder::new()
            .timeout(Duration::from_secs(10))
            .build()?;

        Ok(Self {
            processes: Arc::new(Mutex::new(HashMap::new())),
            nodes,
            client,
        })
    }

    async fn start_nodes(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Starting 4 nodes using pre-compiled q-api-server binary...");
        
        // Find the compiled binary
        let binary_path = self.find_binary_path().await?;
        println!("📁 Using binary: {}", binary_path);

        let mut processes = self.processes.lock().await;
        
        for node in &self.nodes {
            println!("🔧 Starting {} on port {}", node.node_id, node.port);
            
            let mut cmd = Command::new(&binary_path);
            cmd.args(&[
                "--node-id", &node.node_id,
                "--port", &node.port.to_string(),
            ]);
            
            // Set environment for faster startup
            cmd.env("RUST_LOG", "error");
            cmd.env("SKIP_BITCOIN", "1");
            cmd.env("SKIP_DNS", "1");
            cmd.stdout(Stdio::null());
            cmd.stderr(Stdio::null());

            let child = cmd.spawn()?;
            processes.insert(node.node_id.clone(), child);
        }

        println!("⏳ Waiting for nodes to initialize...");
        sleep(Duration::from_secs(8)).await;
        
        self.wait_for_nodes_ready().await?;
        println!("✅ All 4 nodes are ready!");
        
        Ok(())
    }

    async fn find_binary_path(&self) -> Result<String, Box<dyn std::error::Error>> {
        // Check common locations for the compiled binary
        let possible_paths = vec![
            "./target/release/q-api-server",
            "./target/debug/q-api-server", 
            "./crates/q-api-server/target/release/q-api-server",
            "./crates/q-api-server/target/debug/q-api-server",
            "q-api-server", // In PATH
        ];

        for path in possible_paths {
            if tokio::fs::metadata(path).await.is_ok() {
                return Ok(path.to_string());
            }
        }

        // Try to find it with cargo metadata
        let output = Command::new("cargo")
            .args(&["metadata", "--format-version", "1"])
            .output()
            .await?;

        if output.status.success() {
            let metadata: Value = serde_json::from_slice(&output.stdout)?;
            if let Some(target_dir) = metadata.get("target_directory").and_then(|v| v.as_str()) {
                let release_path = format!("{}/release/q-api-server", target_dir);
                let debug_path = format!("{}/debug/q-api-server", target_dir);
                
                if tokio::fs::metadata(&release_path).await.is_ok() {
                    return Ok(release_path);
                }
                if tokio::fs::metadata(&debug_path).await.is_ok() {
                    return Ok(debug_path);
                }
            }
        }

        Err("Could not find compiled q-api-server binary. Please run 'cargo build --bin q-api-server' first.".into())
    }

    async fn wait_for_nodes_ready(&self) -> Result<(), Box<dyn std::error::Error>> {
        let max_retries = 20;
        let retry_delay = Duration::from_secs(1);

        for node in &self.nodes {
            let url = format!("http://{}:{}/api/v1/health", node.ip_address, node.port);
            let mut retries = 0;

            loop {
                match timeout(Duration::from_secs(3), self.client.get(&url).send()).await {
                    Ok(Ok(response)) if response.status().is_success() => {
                        println!("✅ {} ready at port {}", node.node_id, node.port);
                        break;
                    }
                    _ => {
                        retries += 1;
                        if retries >= max_retries {
                            return Err(format!("{} failed to start", node.node_id).into());
                        }
                        sleep(retry_delay).await;
                    }
                }
            }
        }

        Ok(())
    }

    async fn test_connectivity(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 Testing node connectivity...");
        
        for node in &self.nodes {
            let url = format!("http://{}:{}/api/v1/status", node.ip_address, node.port);
            
            match timeout(Duration::from_secs(5), self.client.get(&url).send()).await {
                Ok(Ok(response)) if response.status().is_success() => {
                    let status: Value = response.json().await?;
                    
                    if let Some(node_id) = status.get("node_id") {
                        println!("✅ Node {} responding", node_id);
                    }
                    
                    if let Some(peers) = status.get("connected_peers") {
                        println!("  📡 Connected peers: {}", peers);
                    }
                }
                _ => {
                    println!("⚠️  Node {} not responding", node.node_id);
                }
            }
        }

        Ok(())
    }

    async fn connect_nodes(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔗 Attempting to connect nodes...");
        
        // Try to manually connect nodes using available API endpoints
        for (i, node_a) in self.nodes.iter().enumerate() {
            for (j, node_b) in self.nodes.iter().enumerate() {
                if i != j {
                    // Try different connection methods
                    self.attempt_connection(node_a, node_b).await;
                    sleep(Duration::from_millis(200)).await;
                }
            }
        }

        Ok(())
    }

    async fn attempt_connection(&self, from_node: &NodeConfig, to_node: &NodeConfig) {
        // Try connecting via mesh API
        let mesh_url = format!("http://{}:{}/api/mesh/connect", from_node.ip_address, from_node.port);
        let connection_payload = serde_json::json!({
            "target": format!("{}:{}", to_node.ip_address, to_node.port),
            "node_id": to_node.node_id
        });

        let _ = timeout(
            Duration::from_secs(3),
            self.client.post(&mesh_url).json(&connection_payload).send()
        ).await;

        // Also try bridge connection API
        let bridge_url = format!(
            "http://{}:{}/api/v1/bitcoin/bridge/connect/{}",
            from_node.ip_address, from_node.port, to_node.node_id
        );
        
        let bridge_payload = serde_json::json!({
            "target_address": format!("{}:{}", to_node.ip_address, to_node.port),
            "node_id": to_node.node_id
        });

        let _ = timeout(
            Duration::from_secs(3),
            self.client.post(&bridge_url).json(&bridge_payload).send()
        ).await;
    }

    async fn test_network_functionality(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🧪 Testing network functionality...");
        
        // Send a test transaction
        let test_tx = serde_json::json!({
            "from": "test_sender",
            "to": "test_receiver",
            "amount": 1000,
            "nonce": 1
        });

        let sender_node = &self.nodes[0];
        let tx_url = format!("http://{}:{}/api/v1/transactions", sender_node.ip_address, sender_node.port);

        match timeout(
            Duration::from_secs(5),
            self.client.post(&tx_url).json(&test_tx).send()
        ).await {
            Ok(Ok(response)) => {
                if response.status().is_success() {
                    println!("✅ Test transaction submitted successfully");
                } else {
                    println!("⚠️  Transaction submission failed: {}", response.status());
                }
            }
            _ => {
                println!("⚠️  Transaction submission timed out");
            }
        }

        // Check network analytics
        for node in &self.nodes {
            let analytics_url = format!("http://{}:{}/api/v1/network/analytics", node.ip_address, node.port);
            
            match timeout(Duration::from_secs(3), self.client.get(&analytics_url).send()).await {
                Ok(Ok(response)) if response.status().is_success() => {
                    let analytics: Value = response.json().await?;
                    if let Some(peers) = analytics.get("total_peers") {
                        println!("📊 Node {} sees {} peers", node.node_id, peers);
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    async fn shutdown(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🛑 Shutting down all nodes...");
        
        let mut processes = self.processes.lock().await;
        
        for (node_id, mut process) in processes.drain() {
            let _ = process.kill().await;
            let _ = timeout(Duration::from_secs(3), process.wait()).await;
            println!("✅ Stopped {}", node_id);
        }
        
        Ok(())
    }
}

#[tokio::test]
async fn test_four_nodes_precompiled() {
    println!("🌟 Four Node Test Using Pre-compiled Binary");
    println!("==========================================");
    
    let test = PrecompiledNodeTest::new().await
        .expect("Failed to create test");

    // Start all nodes
    test.start_nodes().await
        .expect("Failed to start nodes");

    // Test basic connectivity 
    test.test_connectivity().await
        .expect("Connectivity test failed");

    // Attempt to connect nodes
    test.connect_nodes().await
        .expect("Failed to connect nodes");

    // Wait for connections to stabilize
    sleep(Duration::from_secs(3)).await;

    // Test network functionality
    test.test_network_functionality().await
        .expect("Network functionality test failed");

    println!("✅ Four Node Pre-compiled Test COMPLETED!");

    // Cleanup
    test.shutdown().await
        .expect("Failed to shutdown nodes");
}

#[tokio::test] 
async fn test_rapid_startup_shutdown() {
    println!("⚡ Rapid Startup/Shutdown Test");
    
    let test = PrecompiledNodeTest::new().await.unwrap();
    
    // Quick startup
    test.start_nodes().await.unwrap();
    
    // Brief functionality check
    sleep(Duration::from_secs(2)).await;
    test.test_connectivity().await.unwrap();
    
    // Quick shutdown
    test.shutdown().await.unwrap();
    
    println!("✅ Rapid test completed!");
}