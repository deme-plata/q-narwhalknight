/// Multi-Client 1M TPS Benchmark
///
/// Tests Q-NarwhalKnight's ability to reach 1M+ TPS with concurrent clients
/// Each client sends 10K transaction batches in parallel

use std::time::Instant;
use std::sync::Arc;
use tokio::sync::Semaphore;
use reqwest::Client;
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};

const SERVER_URL: &str = "http://localhost:8080/api/v1/binary/batch";

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Serialize)]
struct BinaryTransactionBatch {
    transactions: Vec<Transaction>,
}

fn create_address(seed: &str) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(seed.as_bytes());
    hasher.finalize().to_vec()
}

fn create_signature(data: &str) -> Vec<u8> {
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

async fn client_worker(
    client_id: usize,
    num_batches: usize,
    batch_size: usize,
    client: Arc<Client>,
) -> Result<(usize, f64), Box<dyn std::error::Error + Send + Sync>> {
    let mut total_tx = 0;
    let start = Instant::now();

    for batch_num in 0..num_batches {
        // Generate batch
        let base_idx = client_id * 1_000_000 + batch_num * batch_size;
        let transactions: Vec<Transaction> = (0..batch_size)
            .map(|i| create_test_transaction(base_idx + i))
            .collect();

        let batch = BinaryTransactionBatch { transactions };

        // Serialize to MessagePack
        let packed = rmp_serde::to_vec(&batch)?;

        // Submit batch
        let resp = client
            .post(SERVER_URL)
            .header("Content-Type", "application/msgpack")
            .body(packed)
            .send()
            .await?;

        if resp.status().is_success() {
            total_tx += batch_size;
        } else {
            eprintln!("Client {} batch {} failed: {}", client_id, batch_num, resp.status());
        }
    }

    let elapsed = start.elapsed();
    let tps = total_tx as f64 / elapsed.as_secs_f64();

    Ok((total_tx, tps))
}

async fn run_multi_client_test(num_clients: usize, batches_per_client: usize, batch_size: usize) {
    println!("🚀 Multi-Client Benchmark");
    println!("{}", "=".repeat(80));
    println!("Concurrent Clients:    {}", num_clients);
    println!("Batches/Client:        {}", batches_per_client);
    println!("Batch Size:            {} transactions", batch_size);
    println!("Total Transactions:    {}", num_clients * batches_per_client * batch_size);
    println!();

    // Build shared HTTP client with connection pooling
    let client = Arc::new(
        Client::builder()
            .pool_max_idle_per_host(num_clients)
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap()
    );

    let semaphore = Arc::new(Semaphore::new(num_clients));
    let mut tasks = Vec::new();

    println!("⚡ Starting {} concurrent clients...", num_clients);
    let global_start = Instant::now();

    for client_id in 0..num_clients {
        let sem = semaphore.clone();
        let client_arc = client.clone();

        let task = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            println!("  🔹 Client {} starting...", client_id);
            client_worker(client_id, batches_per_client, batch_size, client_arc).await
        });

        tasks.push(task);
    }

    // Collect results
    let mut successful_tx = 0;
    let mut failed_tx = 0;
    let mut client_tps_values = Vec::new();

    for (i, task) in tasks.into_iter().enumerate() {
        match task.await {
            Ok(Ok((sent, tps))) => {
                println!("  ✅ Client {} completed: {} tx at {:.0} TPS", i, sent, tps);
                successful_tx += sent;
                client_tps_values.push(tps);
            }
            Ok(Err(e)) => {
                eprintln!("  ❌ Client {} failed: {}", i, e);
                failed_tx += batches_per_client * batch_size;
            }
            Err(e) => {
                eprintln!("  ❌ Client {} panicked: {}", i, e);
                failed_tx += batches_per_client * batch_size;
            }
        }
    }

    let total_elapsed = global_start.elapsed();
    let aggregate_tps = successful_tx as f64 / total_elapsed.as_secs_f64();

    println!();
    println!("{}", "=".repeat(80));
    println!("📊 MULTI-CLIENT BENCHMARK RESULTS");
    println!("{}", "=".repeat(80));
    println!("Successful Clients:    {}/{}", client_tps_values.len(), num_clients);
    println!("Total Transactions:    {}", successful_tx);
    println!("Failed Transactions:   {}", failed_tx);
    println!("Total Time:            {:.2}s", total_elapsed.as_secs_f64());
    println!("Aggregate TPS:         {:.0}", aggregate_tps);
    println!();

    if !client_tps_values.is_empty() {
        let avg_tps = client_tps_values.iter().sum::<f64>() / client_tps_values.len() as f64;
        let min_tps = client_tps_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_tps = client_tps_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("Client Performance:");
        println!("  Average TPS/Client:  {:.0}", avg_tps);
        println!("  Min TPS/Client:      {:.0}", min_tps);
        println!("  Max TPS/Client:      {:.0}", max_tps);
        println!();
    }

    // Compare to targets
    let target_1m = 1_000_000.0;
    let percent_target = (aggregate_tps / target_1m) * 100.0;

    println!("🎯 PERFORMANCE ANALYSIS");
    println!("{}", "=".repeat(80));
    println!("Target (1M TPS):       {:>10.0} TPS", target_1m);
    println!("Actual (Measured):     {:>10.0} TPS", aggregate_tps);
    println!("Percent of Target:     {:>10.1}%", percent_target);
    println!();

    if aggregate_tps >= target_1m {
        println!("🎉🎉🎉 ACHIEVED 1M+ TPS! MILESTONE REACHED! 🎉🎉🎉");
        println!("Quantum-Enhanced DAG-Knight BFT Consensus at Scale!");
    } else if aggregate_tps >= target_1m * 0.9 {
        println!("🎉 ACHIEVED 90%+ OF 1M TPS TARGET! Almost there!");
    } else if aggregate_tps >= target_1m * 0.75 {
        println!("⚡ ACHIEVED 75%+ OF 1M TPS TARGET! Excellent progress!");
    } else if aggregate_tps >= target_1m * 0.5 {
        println!("📈 ACHIEVED 50%+ OF 1M TPS TARGET! Good progress!");
    } else {
        println!("📊 Current: {:.1}% of 1M TPS target", percent_target);
    }
}

#[tokio::test]
async fn test_multi_client_1m_tps() {
    println!("🌟 Q-NarwhalKnight 1M TPS Multi-Client Benchmark\n");

    // Progressive scaling toward 1M TPS
    let tests = vec![
        // Warmup
        (4, 1, 5000),    // 20K total (warmup)

        // Progressive scaling
        (8, 2, 10000),   // 160K total
        (12, 2, 10000),  // 240K total

        // 1M TPS attempt
        (16, 5, 10000),  // 800K total (aiming for 1M TPS throughput)
    ];

    for (clients, batches, batch_size) in tests {
        run_multi_client_test(clients, batches, batch_size).await;
        println!("\n{}\n", "─".repeat(80));
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    }

    println!("✅ Multi-client benchmark complete!");
    println!("🚀 Q-NarwhalKnight Quantum-Enhanced DAG-BFT Consensus System");
}
