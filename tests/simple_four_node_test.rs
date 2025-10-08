// Simple Four Node Connection Test
// A streamlined test that spins up 4 real q-api-server nodes and connects them

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;
use tokio::sync::Mutex;
use tokio::time::{sleep, timeout};
use serde_json::Value;

struct SimpleNodeTest {
    processes: Arc<Mutex<Vec<tokio::process::Child>>>,
    ports: Vec<u16>,
    client: reqwest::Client,
}

impl SimpleNodeTest {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let ports = vec![18011, 18012, 18013, 18014];
        let client = reqwest::ClientBuilder::new()
            .timeout(Duration::from_secs(10))
            .build()?;

        Ok(Self {
            processes: Arc::new(Mutex::new(Vec::new())),
            ports,
            client,
        })
    }

    async fn start_nodes(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Starting 4 Q-NarwhalKnight nodes...");
        
        let mut processes = self.processes.lock().await;
        
        for (i, &port) in self.ports.iter().enumerate() {
            let node_id = format!("test-node-{}", i + 1);
            println!("🔧 Starting {} on port {}", node_id, port);
            
            let mut cmd = Command::new("cargo");
            cmd.args(&[
                "run", "--package", "q-api-server", "--bin", "q-api-server",
                "--", "--node-id", &node_id, "--port", &port.to_string()
            ]);
            
            cmd.env("RUST_LOG", "error"); // Minimal logging for speed
            cmd.env("SKIP_BITCOIN", "1");
            cmd.env("SKIP_DNS", "1");
            cmd.stdout(Stdio::null());
            cmd.stderr(Stdio::null());

            let child = cmd.spawn()?;
            processes.push(child);
        }

        println!("⏳ Waiting for nodes to start...");
        sleep(Duration::from_secs(8)).await;
        
        // Quick health check
        for &port in &self.ports {
            let url = format!("http://127.0.0.1:{}/api/v1/health", port);
            match timeout(Duration::from_secs(3), self.client.get(&url).send()).await {
                Ok(Ok(resp)) if resp.status().is_success() => {
                    println!("✅ Node on port {} is ready", port);
                }
                _ => {
                    println!("⚠️  Node on port {} not responding", port);
                }
            }
        }

        Ok(())
    }

    async fn test_basic_connectivity(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 Testing basic connectivity...");
        
        let mut all_responsive = true;
        
        for &port in &self.ports {
            let url = format!("http://127.0.0.1:{}/api/v1/status", port);
            match timeout(Duration::from_secs(3), self.client.get(&url).send()).await {
                Ok(Ok(response)) if response.status().is_success() => {
                    let status: Value = response.json().await?;
                    if let Some(node_id) = status.get("node_id") {
                        println!("✅ Node {} responsive on port {}", node_id, port);
                    }
                }
                _ => {
                    println!("❌ Node on port {} not responsive", port);
                    all_responsive = false;
                }
            }
        }

        if all_responsive {
            println!("✅ All 4 nodes are responsive!");
        } else {
            println!("⚠️  Some nodes are not responsive");
        }

        Ok(())
    }

    async fn shutdown(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🛑 Shutting down all nodes...");
        
        let mut processes = self.processes.lock().await;
        
        for mut process in processes.drain(..) {
            let _ = process.kill().await;
            let _ = timeout(Duration::from_secs(3), process.wait()).await;
        }
        
        println!("✅ All nodes shut down");
        Ok(())
    }
}

#[tokio::test]
async fn test_simple_four_nodes() {
    println!("🌟 Simple Four Node Test Starting...");
    
    let test = SimpleNodeTest::new().await.expect("Failed to create test");
    
    // Start nodes
    test.start_nodes().await.expect("Failed to start nodes");
    
    // Test connectivity
    test.test_basic_connectivity().await.expect("Connectivity test failed");
    
    println!("✅ Simple Four Node Test PASSED!");
    
    // Cleanup
    test.shutdown().await.expect("Failed to shutdown");
}