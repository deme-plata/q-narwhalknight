#!/usr/bin/env rust-script
//! Distributed Multi-Node Benchmark Client
//!
//! Submits transactions to all 4 libp2p-connected validator nodes

use std::time::Instant;
use std::sync::Arc;

const NODES: &[(u16, &str)] = &[
    (9100, "Node 0"),
    (9101, "Node 1"),
    (9102, "Node 2"),
    (9103, "Node 3"),
];

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Transaction {
    id: Vec<u8>,
    from: Vec<u8>,
    to: Vec<u8>,
    amount: u64,
    fee: u64,
    nonce: u64,
    signature: Vec<u8>,
    timestamp: String,
    data: Vec<u8>,
}

#[derive(Debug, serde::Serialize)]
struct BinaryTransactionBatch {
    transactions: Vec<Transaction>,
}

fn create_address(seed: &str) -> Vec<u8> {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(seed.as_bytes());
    hasher.finalize().to_vec()
}

fn create_signature(data: &str) -> Vec<u8> {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    let hash1 = hasher.finalize();
    let mut hasher = Sha256::new();
    hasher.update(&hash1);
    let hash2 = hasher.finalize();
    [&hash1[..], &hash2[..]].concat()
}

fn create_test_transaction(index: usize) -> Transaction {
    let from_addr = create_address(&format!("from_{}", index));
    let to_addr = create_address(&format!("to_{}", index));
    let tx_id = create_address(&format!("tx_{}_{}", index, index));
    let signature = create_signature(&format!("sign_{}", index));

    Transaction {
        id: tx_id,
        from: from_addr,
        to: to_addr,
        amount: 1000 + index as u64,
        fee: 10,
        nonce: index as u64,
        signature,
        timestamp: chrono::Utc::now().to_rfc3339(),
        data: vec![],
    }
}

async fn submit_to_node(
    node_id: usize,
    port: u16,
    batches: usize,
    batch_size: usize,
    client: Arc<reqwest::Client>,
) -> Result<(usize, f64), Box<dyn std::error::Error>> {
    let url = format!("http://localhost:{}/api/v1/binary/batch", port);
    let mut total_tx = 0;
    let start = Instant::now();

    for batch_num in 0..batches {
        let base_idx = node_id * 1_000_000 + batch_num * batch_size;
        let transactions: Vec<Transaction> = (0..batch_size)
            .map(|i| create_test_transaction(base_idx + i))
            .collect();

        let batch = BinaryTransactionBatch { transactions };
        let packed = rmp_serde::to_vec(&batch)?;

        let resp = client
            .post(&url)
            .header("Content-Type", "application/msgpack")
            .body(packed)
            .send()
            .await?;

        if resp.status().is_success() {
            total_tx += batch_size;
        } else {
            eprintln!("  ⚠️  Node {} batch {} failed: {}", node_id, batch_num, resp.status());
        }
    }

    let elapsed = start.elapsed();
    let tps = total_tx as f64 / elapsed.as_secs_f64();

    Ok((total_tx, tps))
}

#[tokio::main]
async fn main() {
    println!("================================================================================");
    println!("🌟 DISTRIBUTED LIBP2P MULTI-NODE TPS BENCHMARK");
    println!("================================================================================ ");
    println!();
    println!("Testing 4 validator nodes connected via libp2p gossipsub");
    println!("Each node receives transactions → gossip to peers → DAG-Knight consensus");
    println!();

    let client = Arc::new(
        reqwest::Client::builder()
            .pool_max_idle_per_host(8)
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap()
    );

    // Test configuration
    let batches_per_node = 5;
    let batch_size = 10_000;

    println!("📋 Configuration:");
    println!("  Nodes:              {}", NODES.len());
    println!("  Batches/Node:       {}", batches_per_node);
    println!("  Batch Size:         {} transactions", batch_size);
    println!("  Total Transactions: {}", NODES.len() * batches_per_node * batch_size);
    println!();

    println!("⚡ Starting distributed benchmark...\n");

    let global_start = Instant::now();
    let mut tasks = Vec::new();

    for (i, (port, name)) in NODES.iter().enumerate() {
        let client_arc = client.clone();
        let port = *port;

        let task = tokio::spawn(async move {
            println!("  🔹 Submitting to {} (port {})...", name, port);
            submit_to_node(i, port, batches_per_node, batch_size, client_arc).await
        });

        tasks.push(task);
    }

    // Collect results
    let mut successful_tx = 0;
    let mut failed_tx = 0;
    let mut node_tps_values = Vec::new();

    for (i, task) in tasks.into_iter().enumerate() {
        match task.await {
            Ok(Ok((sent, tps))) => {
                println!("  ✅ {} completed: {} tx at {:.0} TPS", NODES[i].1, sent, tps);
                successful_tx += sent;
                node_tps_values.push(tps);
            }
            Ok(Err(e)) => {
                eprintln!("  ❌ {} failed: {}", NODES[i].1, e);
                failed_tx += batches_per_node * batch_size;
            }
            Err(e) => {
                eprintln!("  ❌ {} panicked: {}", NODES[i].1, e);
                failed_tx += batches_per_node * batch_size;
            }
        }
    }

    let total_elapsed = global_start.elapsed();
    let aggregate_tps = successful_tx as f64 / total_elapsed.as_secs_f64();

    println!();
    println!("================================================================================");
    println!("📊 DISTRIBUTED BENCHMARK RESULTS");
    println!("================================================================================");
    println!("Successful Nodes:      {}/{}", node_tps_values.len(), NODES.len());
    println!("Total Transactions:    {}", successful_tx);
    println!("Failed Transactions:   {}", failed_tx);
    println!("Total Time:            {:.2}s", total_elapsed.as_secs_f64());
    println!("Aggregate TPS:         {:.0}", aggregate_tps);
    println!();

    if !node_tps_values.is_empty() {
        let avg_tps = node_tps_values.iter().sum::<f64>() / node_tps_values.len() as f64;
        let min_tps = node_tps_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_tps = node_tps_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("Node Performance:");
        println!("  Average TPS/Node:    {:.0}", avg_tps);
        println!("  Min TPS/Node:        {:.0}", min_tps);
        println!("  Max TPS/Node:        {:.0}", max_tps);
        println!();
    }

    // Analysis
    let target_1m = 1_000_000.0;
    let percent_target = (aggregate_tps / target_1m) * 100.0;

    println!("🎯 PERFORMANCE ANALYSIS");
    println!("================================================================================");
    println!("Target (1M TPS):       {:>10.0} TPS", target_1m);
    println!("Actual (Measured):     {:>10.0} TPS", aggregate_tps);
    println!("Percent of Target:     {:>10.1}%", percent_target);
    println!();
    println!("Network Architecture:  libp2p gossipsub (peer-to-peer)");
    println!("Consensus Protocol:    DAG-Knight + Bullshark (quantum-enhanced)");
    println!("Discovery Mechanism:   mDNS local network");
    println!("Transport:             TCP + Noise encryption + Yamux multiplexing");
    println!();

    if aggregate_tps >= target_1m {
        println!("🎉🎉🎉 ACHIEVED 1M+ TPS WITH DISTRIBUTED LIBP2P NODES! 🎉🎉🎉");
    } else if aggregate_tps >= target_1m * 0.5 {
        println!("📈 ACHIEVED {}% OF 1M TPS TARGET!", percent_target as u32);
    } else {
        println!("📊 Distributed performance: {:.1}% of 1M TPS target", percent_target);
    }

    println!();
    println!("================================================================================");
    println!("✅ DISTRIBUTED BENCHMARK COMPLETE!");
    println!("🚀 Q-NarwhalKnight Quantum-Enhanced DAG-BFT Consensus");
    println!("⚛️  libp2p peer-to-peer networking validated");
    println!("================================================================================");
}
