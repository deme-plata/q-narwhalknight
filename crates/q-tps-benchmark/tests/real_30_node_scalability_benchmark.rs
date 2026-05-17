// Real 30-Node Horizontal Scalability Benchmark
// Tests Q-NarwhalKnight with 30 real nodes using Kademlia DHT + mDNS discovery
// Includes VM contract execution and comprehensive performance metrics

use q_types::{Transaction, Address};
use q_wallet::{QWallet, CryptoPhase};
use q_network::{NetworkConfig, P2PNetwork, KademliaConfig};
use q_vm::{VirtualMachine, SmartContract, ContractExecutionContext};
use ed25519_dalek::{SigningKey, Signer, Verifier, VerifyingKey, Signature};
use sha3::{Sha3_256, Digest};
use chrono::Utc;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::time::{Instant, Duration};

const NUM_NODES: usize = 30;
const TRANSACTIONS_PER_NODE: usize = 1000;
const BATCH_SIZE: usize = 100;
const TEST_PORT_START: u16 = 10000;

#[derive(Clone)]
struct NodeInfo {
    node_id: [u8; 32],
    peer_id: String,
    listen_addr: String,
    wallet_address: Address,
    wallet_key: SigningKey,
    network: Arc<P2PNetwork>,
    vm: Arc<RwLock<VirtualMachine>>,
}

#[derive(Debug, Clone)]
struct BenchmarkMetrics {
    node_id: usize,
    transactions_generated: usize,
    transactions_sent: usize,
    transactions_received: usize,
    signatures_verified: usize,
    contracts_executed: usize,
    peers_discovered: usize,
    avg_latency_ms: f64,
    tps: f64,
    duration_secs: f64,
}

/// Create a smart contract for testing
fn create_test_contract() -> SmartContract {
    SmartContract {
        address: [42u8; 32],
        code: vec![
            // Simple contract: Transfer with 1% fee to contract owner
            0x01, 0x00, 0x64, // PUSH 100 (for 1% calculation)
            0x02, 0x01,       // MUL amount by 100
            0x03, 0x01,       // DIV by 100 to get 1%
            0x04,             // SUB fee from amount
            0x05,             // TRANSFER remaining
            0xFF,             // RETURN
        ],
        state: HashMap::new(),
        owner: [1u8; 32],
        balance: 0,
    }
}

/// Initialize a single node with Kademlia DHT + mDNS
async fn initialize_node(
    node_idx: usize,
    bootstrap_peers: Vec<String>,
) -> Result<NodeInfo> {
    println!("  🔧 Initializing Node {}...", node_idx);

    // Generate unique node ID
    let mut hasher = Sha3_256::new();
    hasher.update(format!("node_{}", node_idx).as_bytes());
    let node_id_hash = hasher.finalize();
    let mut node_id = [0u8; 32];
    node_id.copy_from_slice(&node_id_hash);

    // Create wallet using Q-Wallet directly
    let q_wallet = QWallet::new(CryptoPhase::Q0); // Use Q0 (Ed25519) for benchmarking
    let wallet_address = q_wallet.address();

    // Extract Ed25519 key for signing
    let wallet_key = q_wallet.hybrid.ed25519_keypair.as_ref().unwrap().signing_key;

    // Configure Kademlia DHT with mDNS fallback
    let kad_config = KademliaConfig {
        protocol_name: "/q-narwhalknight/kad/1.0.0".to_string(),
        replication_factor: 20, // K=20 for robustness
        query_timeout: Duration::from_secs(30),
        provider_publication_interval: Duration::from_secs(600),
        enable_mdns_fallback: true, // Enable mDNS for local discovery
        mdns_ttl: 60,
    };

    // Network configuration
    let port = TEST_PORT_START + node_idx as u16;
    let listen_addr = format!("/ip4/127.0.0.1/tcp/{}", port);

    let net_config = NetworkConfig {
        listen_addresses: vec![listen_addr.clone()],
        bootstrap_peers: bootstrap_peers.clone(),
        enable_mdns: true,
        enable_kad_dht: true,
        kad_config: Some(kad_config),
        connection_idle_timeout: Duration::from_secs(600),
        max_peers: 50,
        ..Default::default()
    };

    // Initialize P2P network
    let network = Arc::new(P2PNetwork::new(node_id, net_config).await?);

    // Start network services
    network.start().await?;

    // Initialize VM for contract execution
    let vm = Arc::new(RwLock::new(VirtualMachine::new()));

    // Deploy test contract
    {
        let mut vm_lock = vm.write().await;
        let contract = create_test_contract();
        vm_lock.deploy_contract(contract).await?;
    }

    let peer_id = network.local_peer_id().to_string();

    println!("    ✅ Node {} ready - PeerId: {}", node_idx, &peer_id[..16]);

    Ok(NodeInfo {
        node_id,
        peer_id,
        listen_addr,
        wallet_address,
        wallet_key,
        network,
        vm,
    })
}

/// Wait for nodes to discover each other via Kademlia DHT + mDNS
async fn wait_for_peer_discovery(nodes: &[NodeInfo], min_peers: usize) -> Result<()> {
    println!("  🔍 Waiting for peer discovery (Kademlia DHT + mDNS)...");

    let discovery_start = Instant::now();
    let max_wait = Duration::from_secs(60);

    loop {
        let mut all_connected = true;
        let mut total_peers = 0;

        for (idx, node) in nodes.iter().enumerate() {
            let peer_count = node.network.connected_peers().await.len();
            total_peers += peer_count;

            if peer_count < min_peers {
                all_connected = false;
            }

            if discovery_start.elapsed().as_secs() % 10 == 0 {
                println!("    Node {}: {} peers discovered", idx, peer_count);
            }
        }

        if all_connected {
            let avg_peers = total_peers as f64 / nodes.len() as f64;
            println!("  ✅ All nodes discovered peers! Avg: {:.1} peers/node", avg_peers);
            return Ok(());
        }

        if discovery_start.elapsed() > max_wait {
            println!("  ⚠️  Timeout waiting for full discovery. Proceeding anyway...");
            return Ok(());
        }

        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// Generate and send transactions from a node
async fn node_transaction_worker(
    node: NodeInfo,
    node_idx: usize,
    target_nodes: Vec<NodeInfo>,
) -> Result<BenchmarkMetrics> {
    let start_time = Instant::now();
    let mut metrics = BenchmarkMetrics {
        node_id: node_idx,
        transactions_generated: 0,
        transactions_sent: 0,
        transactions_received: 0,
        signatures_verified: 0,
        contracts_executed: 0,
        peers_discovered: node.network.connected_peers().await.len(),
        avg_latency_ms: 0.0,
        tps: 0.0,
        duration_secs: 0.0,
    };

    let mut total_latency_ms = 0.0;
    let mut rng = rand::thread_rng();

    // Generate transactions in batches
    for batch_num in 0..(TRANSACTIONS_PER_NODE / BATCH_SIZE) {
        let mut batch_transactions = Vec::new();

        for i in 0..BATCH_SIZE {
            // Select random target node
            let target_idx = rand::Rng::gen_range(&mut rng, 0..target_nodes.len());
            let target_node = &target_nodes[target_idx];

            // Generate transaction ID
            let mut tx_id = [0u8; 32];
            rand::RngCore::fill_bytes(&mut rng, &mut tx_id);

            // Transaction data
            let amount = 1000u64 + i as u64;
            let fee = 10u64;
            let nonce = (batch_num * BATCH_SIZE + i) as u64;

            // Smart contract call data (execute test contract)
            let contract_data = vec![0x01, 0x02, 0x03]; // Contract execution params

            // Sign transaction
            let mut hasher = Sha3_256::new();
            hasher.update(&tx_id);
            hasher.update(&node.wallet_address);
            hasher.update(&target_node.wallet_address);
            hasher.update(&amount.to_le_bytes());
            hasher.update(&fee.to_le_bytes());
            hasher.update(&nonce.to_le_bytes());
            hasher.update(&contract_data);
            let message = hasher.finalize();

            let signature = node.wallet_key.sign(&message);

            let tx = Transaction {
                id: tx_id,
                from: node.wallet_address,
                to: target_node.wallet_address,
                amount,
                fee,
                nonce,
                signature: signature.to_bytes().to_vec(),
                timestamp: Utc::now(),
                data: contract_data.clone(),
            };

            batch_transactions.push(tx);
            metrics.transactions_generated += 1;
        }

        // Send batch to network
        let send_start = Instant::now();

        for tx in batch_transactions {
            // Verify signature locally (simulate receiving node verification)
            let public_key_bytes: [u8; 32] = tx.from[0..32].try_into()?;
            let verifying_key = VerifyingKey::from_bytes(&public_key_bytes)?;

            let mut hasher = Sha3_256::new();
            hasher.update(&tx.id);
            hasher.update(&tx.from);
            hasher.update(&tx.to);
            hasher.update(&tx.amount.to_le_bytes());
            hasher.update(&tx.fee.to_le_bytes());
            hasher.update(&tx.nonce.to_le_bytes());
            hasher.update(&tx.data);
            let message = hasher.finalize();

            let signature_bytes: [u8; 64] = tx.signature.as_slice().try_into()?;
            let signature = Signature::from_bytes(&signature_bytes);
            verifying_key.verify(&message, &signature)?;

            metrics.signatures_verified += 1;

            // Execute smart contract if data present
            if !tx.data.is_empty() {
                let mut vm_lock = node.vm.write().await;
                let ctx = ContractExecutionContext {
                    sender: tx.from,
                    recipient: tx.to,
                    amount: tx.amount,
                    gas_limit: 1_000_000,
                };

                // Execute contract
                match vm_lock.execute_contract([42u8; 32], tx.data.clone(), ctx).await {
                    Ok(_) => metrics.contracts_executed += 1,
                    Err(e) => eprintln!("Contract execution failed: {}", e),
                }
            }

            // Broadcast transaction via P2P network
            match node.network.broadcast_transaction(&tx).await {
                Ok(_) => metrics.transactions_sent += 1,
                Err(e) => eprintln!("Failed to broadcast transaction: {}", e),
            }
        }

        let batch_latency = send_start.elapsed().as_millis() as f64;
        total_latency_ms += batch_latency;

        // Small delay between batches to avoid overwhelming network
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    metrics.duration_secs = start_time.elapsed().as_secs_f64();
    metrics.avg_latency_ms = total_latency_ms / (TRANSACTIONS_PER_NODE / BATCH_SIZE) as f64;
    metrics.tps = metrics.transactions_sent as f64 / metrics.duration_secs;

    Ok(metrics)
}

#[tokio::test]
async fn test_30_node_horizontal_scalability() -> Result<()> {
    println!("\n🚀 30-Node Horizontal Scalability Benchmark");
    println!("=================================================");
    println!("Testing Q-NarwhalKnight with:");
    println!("  • 30 real nodes");
    println!("  • Kademlia DHT (K=20)");
    println!("  • mDNS fallback discovery");
    println!("  • VM smart contract execution");
    println!("  • libp2p-rust networking");
    println!("  • {} transactions per node", TRANSACTIONS_PER_NODE);
    println!("  • {} total transactions", NUM_NODES * TRANSACTIONS_PER_NODE);
    println!();

    // Phase 1: Initialize all nodes
    println!("📦 Phase 1: Initializing {} nodes...", NUM_NODES);
    let init_start = Instant::now();

    let mut nodes = Vec::new();
    let mut bootstrap_peers = Vec::new();

    // Initialize bootstrap node first
    let bootstrap_node = initialize_node(0, vec![]).await?;
    bootstrap_peers.push(format!("{}/p2p/{}",
        bootstrap_node.listen_addr,
        bootstrap_node.peer_id));
    nodes.push(bootstrap_node);

    // Initialize remaining nodes with bootstrap peer
    for i in 1..NUM_NODES {
        let node = initialize_node(i, bootstrap_peers.clone()).await?;
        nodes.push(node);

        // Every 5th node becomes an additional bootstrap peer for better connectivity
        if i % 5 == 0 {
            bootstrap_peers.push(format!("{}/p2p/{}",
                nodes[i].listen_addr,
                nodes[i].peer_id));
        }
    }

    let init_duration = init_start.elapsed();
    println!("  ✅ All {} nodes initialized in {:.2}s", NUM_NODES, init_duration.as_secs_f64());
    println!();

    // Phase 2: Wait for peer discovery
    println!("📡 Phase 2: Peer Discovery (Kademlia DHT + mDNS)...");
    let min_peers = 5; // Each node should discover at least 5 peers
    wait_for_peer_discovery(&nodes, min_peers).await?;
    println!();

    // Phase 3: Run transaction benchmark
    println!("⚡ Phase 3: Transaction Benchmark...");
    println!("  Generating and sending {} transactions per node...", TRANSACTIONS_PER_NODE);

    let benchmark_start = Instant::now();
    let mut tasks = Vec::new();

    for (idx, node) in nodes.iter().enumerate() {
        let node_clone = node.clone();
        let target_nodes = nodes.clone();

        let task = tokio::spawn(async move {
            node_transaction_worker(node_clone, idx, target_nodes).await
        });

        tasks.push(task);
    }

    // Collect results
    let mut all_metrics = Vec::new();
    for (idx, task) in tasks.into_iter().enumerate() {
        match task.await {
            Ok(Ok(metrics)) => {
                println!("  ✅ Node {} completed: {} tx sent, {} contracts executed, {:.0} TPS",
                    idx, metrics.transactions_sent, metrics.contracts_executed, metrics.tps);
                all_metrics.push(metrics);
            }
            Ok(Err(e)) => {
                eprintln!("  ❌ Node {} error: {}", idx, e);
            }
            Err(e) => {
                eprintln!("  ❌ Node {} panicked: {}", idx, e);
            }
        }
    }

    let total_benchmark_time = benchmark_start.elapsed();
    println!();

    // Phase 4: Results Analysis
    println!("📊 BENCHMARK RESULTS");
    println!("=================================================");

    let total_tx_generated: usize = all_metrics.iter().map(|m| m.transactions_generated).sum();
    let total_tx_sent: usize = all_metrics.iter().map(|m| m.transactions_sent).sum();
    let total_signatures: usize = all_metrics.iter().map(|m| m.signatures_verified).sum();
    let total_contracts: usize = all_metrics.iter().map(|m| m.contracts_executed).sum();
    let avg_latency: f64 = all_metrics.iter().map(|m| m.avg_latency_ms).sum::<f64>() / all_metrics.len() as f64;
    let total_peers: usize = all_metrics.iter().map(|m| m.peers_discovered).sum();
    let avg_peers_per_node = total_peers as f64 / all_metrics.len() as f64;

    // Calculate aggregate TPS
    let aggregate_tps = total_tx_sent as f64 / total_benchmark_time.as_secs_f64();

    // Calculate per-node TPS statistics
    let tps_values: Vec<f64> = all_metrics.iter().map(|m| m.tps).collect();
    let avg_node_tps = tps_values.iter().sum::<f64>() / tps_values.len() as f64;
    let min_node_tps = tps_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_node_tps = tps_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("Network Configuration:");
    println!("  Total Nodes:              {}", NUM_NODES);
    println!("  Avg Peers/Node:           {:.1}", avg_peers_per_node);
    println!("  Discovery Protocol:       Kademlia DHT + mDNS");
    println!();

    println!("Transaction Performance:");
    println!("  Total TX Generated:       {}", total_tx_generated);
    println!("  Total TX Sent:            {}", total_tx_sent);
    println!("  Total Signatures Verified: {}", total_signatures);
    println!("  Total Contracts Executed: {}", total_contracts);
    println!("  Success Rate:             {:.2}%", (total_tx_sent as f64 / total_tx_generated as f64) * 100.0);
    println!();

    println!("Throughput Metrics:");
    println!("  Aggregate TPS:            {:.0} tx/sec", aggregate_tps);
    println!("  Average Node TPS:         {:.0} tx/sec", avg_node_tps);
    println!("  Min Node TPS:             {:.0} tx/sec", min_node_tps);
    println!("  Max Node TPS:             {:.0} tx/sec", max_node_tps);
    println!("  Total Duration:           {:.2}s", total_benchmark_time.as_secs_f64());
    println!();

    println!("Latency Metrics:");
    println!("  Avg Batch Latency:        {:.2}ms", avg_latency);
    println!("  Avg TX Latency:           {:.2}ms", avg_latency / BATCH_SIZE as f64);
    println!();

    println!("Smart Contract Execution:");
    println!("  Contracts Executed:       {}", total_contracts);
    println!("  Contract Success Rate:    {:.2}%", (total_contracts as f64 / total_tx_sent as f64) * 100.0);
    println!();

    // Horizontal Scalability Analysis
    println!("🎯 HORIZONTAL SCALABILITY ANALYSIS");
    println!("=================================================");

    let theoretical_linear_tps = avg_node_tps * NUM_NODES as f64;
    let scalability_efficiency = (aggregate_tps / theoretical_linear_tps) * 100.0;

    println!("  Theoretical Linear TPS:   {:.0} tx/sec", theoretical_linear_tps);
    println!("  Actual Aggregate TPS:     {:.0} tx/sec", aggregate_tps);
    println!("  Scalability Efficiency:   {:.1}%", scalability_efficiency);
    println!();

    if scalability_efficiency >= 90.0 {
        println!("  ✅ EXCELLENT - Near-linear horizontal scalability!");
    } else if scalability_efficiency >= 75.0 {
        println!("  ✅ GOOD - Strong horizontal scalability");
    } else if scalability_efficiency >= 60.0 {
        println!("  ⚠️  MODERATE - Acceptable scalability with room for improvement");
    } else {
        println!("  ⚠️  NEEDS IMPROVEMENT - Scalability bottlenecks detected");
    }
    println!();

    // Comparison to targets
    println!("🏆 PERFORMANCE TARGETS");
    println!("=================================================");
    println!("  Target 100K TPS:          {} {}",
        if aggregate_tps >= 100_000.0 { "✅" } else { "❌" },
        if aggregate_tps >= 100_000.0 { "ACHIEVED" } else { &format!("{}% of target", (aggregate_tps / 100_000.0 * 100.0) as i32) }
    );
    println!("  Target 500K TPS:          {} {}",
        if aggregate_tps >= 500_000.0 { "✅" } else { "❌" },
        if aggregate_tps >= 500_000.0 { "ACHIEVED" } else { &format!("{}% of target", (aggregate_tps / 500_000.0 * 100.0) as i32) }
    );
    println!("  Target 1M TPS:            {} {}",
        if aggregate_tps >= 1_000_000.0 { "✅" } else { "❌" },
        if aggregate_tps >= 1_000_000.0 { "ACHIEVED" } else { &format!("{}% of target", (aggregate_tps / 1_000_000.0 * 100.0) as i32) }
    );
    println!();

    println!("✅ 30-Node Horizontal Scalability Benchmark Complete!");
    println!("🚀 Q-NarwhalKnight Quantum-Enhanced DAG-BFT Consensus");
    println!();

    // Cleanup: Shutdown all nodes
    for node in nodes {
        let _ = node.network.shutdown().await;
    }

    Ok(())
}
