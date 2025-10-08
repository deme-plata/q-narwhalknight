/// High-Performance Batch HTTP Benchmark
/// Tests 50K transactions per batch to achieve 1M+ TPS

use reqwest::Client;
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use std::time::Instant;

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

#[tokio::test]
async fn test_batch_http_extreme() {
    println!("{}", "=".repeat(80));
    println!("EXTREME BATCH HTTP BENCHMARK - 50K Transactions Per Batch");
    println!("{}", "=".repeat(80));

    let client = Client::builder()
        .pool_max_idle_per_host(100)
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .unwrap();

    // Test progression: 1K, 5K, 10K, 25K, 50K per batch
    let batch_sizes = vec![1000, 5000, 10000, 25000, 50000];

    for batch_size in batch_sizes {
        println!();
        println!("{}", "─".repeat(80));
        println!("Testing batch size: {}", batch_size);
        println!("{}", "─".repeat(80));

        // Generate transactions
        println!("📝 Generating {} transactions...", batch_size);
        let gen_start = Instant::now();
        let transactions: Vec<Transaction> = (0..batch_size)
            .map(|i| create_test_transaction(i))
            .collect();
        println!("✅ Generated in {:?}", gen_start.elapsed());

        let batch = BinaryTransactionBatch { transactions };

        // Serialize to MessagePack
        println!("📦 Serializing to MessagePack...");
        let pack_start = Instant::now();
        let packed = rmp_serde::to_vec(&batch).unwrap();
        let pack_time = pack_start.elapsed();
        println!("✅ Packed {} bytes in {:?}", packed.len(), pack_time);

        // Submit batch
        println!("⚡ Submitting batch to server...");
        let submit_start = Instant::now();

        match client
            .post("http://localhost:9050/api/v1/binary/batch")
            .header("Content-Type", "application/msgpack")
            .body(packed)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                let submit_time = submit_start.elapsed();
                let tps = batch_size as f64 / submit_time.as_secs_f64();

                println!("✅ Batch submitted successfully!");
                println!("   Time: {:?}", submit_time);
                println!("   TPS: {:.0}", tps);
                println!("   Latency/tx: {:.3}ms", submit_time.as_secs_f64() * 1000.0 / batch_size as f64);
            }
            Ok(resp) => {
                println!("❌ Server returned error: {}", resp.status());
                if let Ok(text) = resp.text().await {
                    println!("   Response: {}", text);
                }
            }
            Err(e) => {
                println!("❌ Request failed: {}", e);
            }
        }

        // Brief pause between tests
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }

    println!();
    println!("{}", "=".repeat(80));
    println!("BENCHMARK COMPLETE");
    println!("{}", "=".repeat(80));
}
