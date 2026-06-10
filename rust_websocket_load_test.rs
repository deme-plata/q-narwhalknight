/// High-Performance Rust WebSocket Binary Load Test
/// This will actually stress the 16 parallel workers

use tokio::runtime::Runtime;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use std::time::Instant;
use std::sync::Arc;
use tokio::sync::Semaphore;

const SERVER_URL: &str = "ws://localhost:9050/api/v1/binary/stream";

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
    let tx_id = create_address(&format!("tx_{}_{}", index, chrono::Utc::now().timestamp_nanos_opt().unwrap()));
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

async fn client_worker(client_id: usize, num_transactions: usize) -> Result<(usize, f64), Box<dyn std::error::Error>> {
    let url = url::Url::parse(SERVER_URL)?;
    let (ws_stream, _) = connect_async(url).await?;
    let (mut write, _read) = ws_stream.split();

    let start = Instant::now();

    for i in 0..num_transactions {
        let tx = create_test_transaction(client_id * 10000 + i);
        let packed = rmp_serde::to_vec(&tx)?;
        write.send(Message::Binary(packed)).await?;
    }

    let elapsed = start.elapsed();
    let tps = num_transactions as f64 / elapsed.as_secs_f64();

    Ok((num_transactions, tps))
}

async fn run_load_test(num_clients: usize, txs_per_client: usize) {
    println!("🚀 Rust WebSocket Binary Load Test");
    println!("{}", "=".repeat(80));
    println!("Concurrent Clients: {}", num_clients);
    println!("Transactions/Client: {}", txs_per_client);
    println!("Total Transactions: {}", num_clients * txs_per_client);
    println!("Server: {}", SERVER_URL);
    println!();

    let semaphore = Arc::new(Semaphore::new(num_clients));
    let mut tasks = Vec::new();

    println!("⚡ Starting {} concurrent Rust WebSocket clients...", num_clients);
    let start = Instant::now();

    for client_id in 0..num_clients {
        let sem = semaphore.clone();

        let task = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            client_worker(client_id, txs_per_client).await
        });

        tasks.push(task);
    }

    // Collect results
    let mut successful = 0;
    let mut failed = 0;
    let mut client_tps_values = Vec::new();

    for task in tasks {
        match task.await {
            Ok(Ok((sent, tps))) => {
                successful += sent;
                client_tps_values.push(tps);
            }
            _ => {
                failed += txs_per_client;
            }
        }
    }

    let total_elapsed = start.elapsed();
    let aggregate_tps = successful as f64 / total_elapsed.as_secs_f64();

    println!();
    println!("✅ Load Test Complete!");
    println!();
    println!("📊 RUST WEBSOCKET BINARY LOAD TEST RESULTS");
    println!("=" .repeat(80));
    println!("Successful Clients:    {}/{}", client_tps_values.len(), num_clients);
    println!("Total Transactions:    {}", successful);
    println!("Failed Transactions:   {}", failed);
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

    // Compare to baselines
    let baseline_single = 21817.0;
    let target_16_workers = 349072.0;

    let improvement = aggregate_tps / baseline_single;
    let percent_target = (aggregate_tps / target_16_workers) * 100.0;

    println!("🎯 PERFORMANCE ANALYSIS");
    println!("=" .repeat(80));
    println!("Baseline (Single Worker):      {:>10.0} TPS", baseline_single);
    println!("Projected (16 Workers):        {:>10.0} TPS", target_16_workers);
    println!("Actual (Measured):             {:>10.0} TPS", aggregate_tps);
    println!();
    println!("Improvement over baseline:     {:>10.1}x", improvement);
    println!("Percent of 16x target:         {:>10.1}%", percent_target);
    println!();

    if aggregate_tps >= target_16_workers {
        println!("🎉 EXCEEDED 16x TARGET! Parallel workers delivering full performance!");
    } else if aggregate_tps >= target_16_workers * 0.8 {
        println!("✅ ACHIEVED 80%+ OF TARGET! Parallel workers working well!");
    } else if aggregate_tps >= target_16_workers * 0.5 {
        println!("⚡ ACHIEVED 50%+ OF TARGET! Parallel workers active!");
    } else if aggregate_tps >= baseline_single * 2.0 {
        println!("📈 ACHIEVED 2x+ IMPROVEMENT! Parallel workers providing speedup!");
    } else {
        println!("⚠️  Performance below expectations");
    }
}

fn main() {
    println!("🌟 Q-NarwhalKnight High-Performance Rust Load Test\n");

    let rt = Runtime::new().unwrap();

    // Progressive load tests
    let tests = vec![
        (4, 5000),   // 20K total
        (8, 5000),   // 40K total
        (16, 5000),  // 80K total
        (32, 5000),  // 160K total
    ];

    for (clients, txs) in tests {
        rt.block_on(run_load_test(clients, txs));
        println!("\n{}\n", "─".repeat(80));
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}
