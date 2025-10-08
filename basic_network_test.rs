#!/usr/bin/env cargo script
//! Simple Bitcoin network connectivity test
//! This is a standalone test to check basic network connectivity capabilities

use std::collections::HashMap;
use std::net::{TcpListener, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;

/// Basic node simulation for testing
#[derive(Debug)]
struct MockNode {
    id: u32,
    address: SocketAddr,
    is_running: Arc<AtomicBool>,
    connected_peers: Arc<std::sync::Mutex<Vec<u32>>>,
}

impl MockNode {
    fn new(id: u32, port: u16) -> Result<Self, Box<dyn std::error::Error>> {
        let address = format!("127.0.0.1:{}", port).parse()?;
        
        Ok(Self {
            id,
            address,
            is_running: Arc::new(AtomicBool::new(false)),
            connected_peers: Arc::new(std::sync::Mutex::new(Vec::new())),
        })
    }

    fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Starting node {} on {}", self.id, self.address);
        
        // Try to bind to the address
        let listener = TcpListener::bind(self.address)?;
        self.is_running.store(true, Ordering::SeqCst);
        
        let is_running = self.is_running.clone();
        thread::spawn(move || {
            while is_running.load(Ordering::SeqCst) {
                if let Ok((_stream, addr)) = listener.accept() {
                    println!("📡 Node {} accepted connection from {}", 1, addr);
                }
                thread::sleep(Duration::from_millis(100));
            }
        });
        
        // Simulate some startup time
        thread::sleep(Duration::from_millis(500));
        
        Ok(())
    }

    fn attempt_connection(&self, peer_address: SocketAddr) -> bool {
        match std::net::TcpStream::connect_timeout(&peer_address, Duration::from_secs(2)) {
            Ok(_stream) => {
                println!("✅ Node {} successfully connected to {}", self.id, peer_address);
                true
            },
            Err(e) => {
                println!("❌ Node {} failed to connect to {}: {}", self.id, peer_address, e);
                false
            }
        }
    }

    fn stop(&self) {
        self.is_running.store(false, Ordering::SeqCst);
        println!("🛑 Stopped node {}", self.id);
    }
}

/// Test results structure
#[derive(Debug)]
struct NetworkTestResult {
    nodes_started: usize,
    connections_attempted: usize,
    connections_successful: usize,
    success_rate: f64,
    test_duration: Duration,
    overall_success: bool,
}

impl NetworkTestResult {
    fn new(
        nodes_started: usize,
        connections_attempted: usize,
        connections_successful: usize,
        test_duration: Duration,
    ) -> Self {
        let success_rate = if connections_attempted > 0 {
            (connections_successful as f64 / connections_attempted as f64) * 100.0
        } else {
            0.0
        };

        let overall_success = nodes_started > 0 && success_rate >= 50.0;

        Self {
            nodes_started,
            connections_attempted,
            connections_successful,
            success_rate,
            test_duration,
            overall_success,
        }
    }

    fn print_summary(&self) {
        println!("\n📊 Bitcoin Network Connectivity Test Results");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Nodes Started: {}", self.nodes_started);
        println!("Connections Attempted: {}", self.connections_attempted);
        println!("Connections Successful: {}", self.connections_successful);
        println!("Success Rate: {:.1}%", self.success_rate);
        println!("Test Duration: {:.2}s", self.test_duration.as_secs_f64());
        
        if self.overall_success {
            println!("✅ Overall Result: SUCCESS");
            println!("📡 Nodes can connect through network infrastructure");
        } else {
            println!("❌ Overall Result: FAILED");
            if self.nodes_started == 0 {
                println!("🔥 Issue: No nodes could start - check port availability");
            } else if self.success_rate < 50.0 {
                println!("🔥 Issue: Network connectivity too low - check firewall/routing");
            }
        }
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

fn run_basic_network_test(node_count: usize) -> Result<NetworkTestResult, Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    println!("🌐 Q-NarwhalKnight Basic Network Connectivity Test");
    println!("Nodes: {} | Testing TCP connectivity simulation", node_count);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut nodes = HashMap::new();
    let start_port = 9000;
    
    // Phase 1: Create and start nodes
    println!("\n📡 Phase 1: Starting {} nodes...", node_count);
    for i in 0..node_count {
        let node_id = i as u32 + 1;
        let port = start_port + i as u16;
        
        match MockNode::new(node_id, port) {
            Ok(node) => {
                match node.start() {
                    Ok(_) => {
                        nodes.insert(node_id, node);
                        println!("  ✅ Node {} started on port {}", node_id, port);
                    },
                    Err(e) => {
                        println!("  ❌ Node {} failed to start: {}", node_id, e);
                    }
                }
            },
            Err(e) => {
                println!("  ❌ Failed to create node {}: {}", node_id, e);
            }
        }
    }

    let nodes_started = nodes.len();
    println!("📊 Started {} out of {} nodes", nodes_started, node_count);

    // Phase 2: Test connectivity between nodes
    println!("\n🔗 Phase 2: Testing inter-node connectivity...");
    let mut connections_attempted = 0;
    let mut connections_successful = 0;

    let node_ids: Vec<u32> = nodes.keys().copied().collect();
    
    for &node_id in &node_ids {
        if let Some(node) = nodes.get(&node_id) {
            for &peer_id in &node_ids {
                if node_id != peer_id {
                    if let Some(peer) = nodes.get(&peer_id) {
                        connections_attempted += 1;
                        if node.attempt_connection(peer.address) {
                            connections_successful += 1;
                        }
                        // Small delay between connection attempts
                        thread::sleep(Duration::from_millis(100));
                    }
                }
            }
        }
    }

    // Phase 3: Cleanup
    println!("\n🛑 Phase 3: Shutting down nodes...");
    for (node_id, node) in nodes.iter() {
        node.stop();
    }

    let test_duration = start_time.elapsed();
    
    let result = NetworkTestResult::new(
        nodes_started,
        connections_attempted,
        connections_successful,
        test_duration,
    );

    Ok(result)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let node_count = if args.len() > 1 {
        args[1].parse().unwrap_or(4)
    } else {
        4
    };

    match run_basic_network_test(node_count) {
        Ok(result) => {
            result.print_summary();
            
            // Set exit code based on test result
            if !result.overall_success {
                std::process::exit(1);
            }
        },
        Err(e) => {
            println!("❌ Test failed with error: {}", e);
            std::process::exit(1);
        }
    }
}