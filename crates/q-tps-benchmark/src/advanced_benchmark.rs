/// Advanced TPS Benchmark - Find Bottlenecks Preventing 1M TPS
///
/// This benchmark progressively tests the performance stack to identify
/// where bottlenecks occur in the path to 1M+ TPS.
///
/// Tests:
/// 1. HTTP JSON (baseline - 1K TPS)
/// 2. HTTP Binary MessagePack (10x improvement - 10K TPS)
/// 3. HTTP Binary Batch (100x improvement - 100K TPS)
/// 4. WebSocket Streaming (1000x improvement - 1M+ TPS target)
///
/// Run with: cargo run --release --bin advanced_benchmark

use q_types::{Transaction, Address};
use ed25519_dalek::{SigningKey, Signer};
use sha3::{Sha3_256, Digest};
use chrono::Utc;
use anyhow::Result;
use rmp_serde;
use reqwest::{Client, ClientBuilder};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use futures::{SinkExt, StreamExt};
use rand::RngCore;
use serde::{Serialize, Deserialize};
use std::time::{Duration, Instant};

const API_BASE: &str = "http://localhost:8080";

#[derive(Debug, Serialize, Deserialize)]
struct TransactionBatch {
    pub transactions: Vec<Transaction>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BatchAck {
    pub accepted: usize,
    pub rejected: usize,
    pub processing_time_us: u64,
}

#[derive(Debug)]
struct BenchmarkResult {
    test_name: String,
    total_transactions: usize,
    duration: Duration,
    tps: f64,
    latency_avg_ms: f64,
    latency_p99_ms: f64,
    success_rate: f64,
}

impl BenchmarkResult {
    fn print(&self) {
        println!("\n{}", "=".repeat(80));
        println!("📊 {}", self.test_name);
        println!("{}", "=".repeat(80));
        println!("Transactions:     {}", self.total_transactions);
        println!("Duration:         {:.2}s", self.duration.as_secs_f64());
        println!("TPS:              {:.0}", self.tps);
        println!("Latency (avg):    {:.2}ms", self.latency_avg_ms);
        println!("Latency (P99):    {:.2}ms", self.latency_p99_ms);
        println!("Success Rate:     {:.1}%", self.success_rate * 100.0);
        println!("{}", "=".repeat(80));
    }

    fn gap_to_1m(&self) {
        let gap = 1_000_000.0 / self.tps;
        println!("📈 Gap to 1M TPS: {:.1}x improvement needed", gap);

        if gap > 100.0 {
            println!("🔴 Major bottleneck detected - HTTP overhead likely the issue");
            println!("   → Solution: Enable WebSocket streaming or binary batch protocol");
        } else if gap > 10.0 {
            println!("🟡 Moderate bottleneck - serialization or batching issue");
            println!("   → Solution: Increase batch sizes or optimize deserialization");
        } else if gap > 2.0 {
            println!("🟢 Close to target - likely signature verification bottleneck");
            println!("   → Solution: Enable SIMD batch signature verification");
        } else {
            println!("✅ Already at or near 1M TPS - only minor optimizations needed");
        }
    }
}

/// Generate realistic signed transactions
fn generate_transactions(count: usize, sender_key: &SigningKey, sender_addr: Address, receiver_addr: Address) -> Vec<Transaction> {
    let mut transactions = Vec::with_capacity(count);

    for i in 0..count {
        let mut tx_id = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut tx_id);

        let amount = 1000u64;
        let fee = 1u64;
        let nonce = i as u64;

        // Real signature
        let mut hasher = Sha3_256::new();
        hasher.update(&tx_id);
        hasher.update(&sender_addr);
        hasher.update(&receiver_addr);
        hasher.update(&amount.to_le_bytes());
        hasher.update(&fee.to_le_bytes());
        hasher.update(&nonce.to_le_bytes());
        let message = hasher.finalize();

        let signature = sender_key.sign(&message);

        transactions.push(Transaction {
            id: tx_id,
            from: sender_addr,
            to: receiver_addr,
            amount: amount.into(),
            fee: fee.into(),
            nonce,
            signature: signature.to_bytes().to_vec(),
            timestamp: Utc::now(),
            data: vec![],
            token_type: q_types::TokenType::QUG,
            fee_token_type: q_types::TokenType::QUGUSD,
            tx_type: q_types::TransactionType::default(),
            pqc_signature: None,
            signature_phase: q_types::TxSignaturePhase::default(),
            pqc_public_key: None,
            zk_proof_bundle: None,
            privacy_level: q_types::TransactionPrivacyLevel::default(),
            bulletproof: None,
            nullifier: None,
            memo: None,
        });
    }

    transactions
}

/// Test 1: HTTP JSON (Baseline)
async fn benchmark_http_json(client: &Client, transactions: &[Transaction]) -> Result<BenchmarkResult> {
    println!("\n🧪 Test 1: HTTP JSON Protocol (Baseline)");

    let start = Instant::now();
    let mut successful = 0;
    let mut latencies = Vec::new();

    for tx in transactions {
        let req_start = Instant::now();

        let response = client
            .post(format!("{}/api/v1/transactions/send", API_BASE))
            .json(&serde_json::json!({
                "from": hex::encode(&tx.from),
                "to": hex::encode(&tx.to),
                "amount": tx.amount,
                "timestamp": tx.timestamp.to_rfc3339(),
            }))
            .send()
            .await;

        let latency = req_start.elapsed();
        latencies.push(latency);

        if let Ok(resp) = response {
            if resp.status().is_success() {
                successful += 1;
            }
        }
    }

    let duration = start.elapsed();
    let tps = successful as f64 / duration.as_secs_f64();

    latencies.sort();
    let avg_latency = latencies.iter().map(|d| d.as_millis() as f64).sum::<f64>() / latencies.len() as f64;
    let p99_latency = latencies[(latencies.len() as f64 * 0.99) as usize].as_millis() as f64;

    Ok(BenchmarkResult {
        test_name: "HTTP JSON Protocol (Baseline)".to_string(),
        total_transactions: transactions.len(),
        duration,
        tps,
        latency_avg_ms: avg_latency,
        latency_p99_ms: p99_latency,
        success_rate: successful as f64 / transactions.len() as f64,
    })
}

/// Test 2: HTTP Binary MessagePack (Single)
async fn benchmark_http_binary_single(client: &Client, transactions: &[Transaction]) -> Result<BenchmarkResult> {
    println!("\n🧪 Test 2: HTTP Binary MessagePack (Single Transaction)");

    let start = Instant::now();
    let mut successful = 0;
    let mut latencies = Vec::new();

    for tx in transactions {
        let req_start = Instant::now();

        let packed = rmp_serde::to_vec(&tx)?;

        let response = client
            .post(format!("{}/api/v1/binary/transaction", API_BASE))
            .header("Content-Type", "application/msgpack")
            .body(packed)
            .send()
            .await;

        let latency = req_start.elapsed();
        latencies.push(latency);

        if let Ok(resp) = response {
            if resp.status().is_success() {
                successful += 1;
            }
        }
    }

    let duration = start.elapsed();
    let tps = successful as f64 / duration.as_secs_f64();

    latencies.sort();
    let avg_latency = latencies.iter().map(|d| d.as_millis() as f64).sum::<f64>() / latencies.len() as f64;
    let p99_latency = latencies[(latencies.len() as f64 * 0.99) as usize].as_millis() as f64;

    Ok(BenchmarkResult {
        test_name: "HTTP Binary MessagePack (Single)".to_string(),
        total_transactions: transactions.len(),
        duration,
        tps,
        latency_avg_ms: avg_latency,
        latency_p99_ms: p99_latency,
        success_rate: successful as f64 / transactions.len() as f64,
    })
}

/// Test 3: HTTP Binary Batch (Progressive batch sizes)
async fn benchmark_http_binary_batch(client: &Client, transactions: &[Transaction], batch_size: usize) -> Result<BenchmarkResult> {
    println!("\n🧪 Test 3: HTTP Binary Batch ({} tx per batch)", batch_size);

    let start = Instant::now();
    let mut successful = 0;
    let mut latencies = Vec::new();

    for chunk in transactions.chunks(batch_size) {
        let req_start = Instant::now();

        let batch = TransactionBatch {
            transactions: chunk.to_vec(),
        };

        let packed = rmp_serde::to_vec(&batch)?;

        let response = client
            .post(format!("{}/api/v1/binary/batch", API_BASE))
            .header("Content-Type", "application/msgpack")
            .body(packed)
            .send()
            .await;

        let latency = req_start.elapsed();
        latencies.push(latency);

        if let Ok(resp) = response {
            if resp.status().is_success() {
                successful += chunk.len();
            }
        }
    }

    let duration = start.elapsed();
    let tps = successful as f64 / duration.as_secs_f64();

    latencies.sort();
    let avg_latency = if !latencies.is_empty() {
        latencies.iter().map(|d| d.as_millis() as f64).sum::<f64>() / latencies.len() as f64
    } else {
        0.0
    };
    let p99_latency = if !latencies.is_empty() {
        latencies[(latencies.len() as f64 * 0.99) as usize].as_millis() as f64
    } else {
        0.0
    };

    Ok(BenchmarkResult {
        test_name: format!("HTTP Binary Batch ({}x)", batch_size),
        total_transactions: transactions.len(),
        duration,
        tps,
        latency_avg_ms: avg_latency,
        latency_p99_ms: p99_latency,
        success_rate: successful as f64 / transactions.len() as f64,
    })
}

/// Test 4: WebSocket Streaming (Target: 1M+ TPS)
async fn benchmark_websocket_stream(transactions: &[Transaction], batch_size: usize) -> Result<BenchmarkResult> {
    println!("\n🧪 Test 4: WebSocket Streaming (batch size: {}, target: 1M+ TPS)", batch_size);

    // Connect to WebSocket
    let ws_url = "ws://localhost:8080/api/v1/ws/transactions";
    let (ws_stream, _) = match connect_async(ws_url).await {
        Ok(conn) => conn,
        Err(e) => {
            println!("❌ WebSocket connection failed: {}", e);
            println!("   → WebSocket endpoint may not be implemented yet");
            return Ok(BenchmarkResult {
                test_name: "WebSocket Streaming (Not Available)".to_string(),
                total_transactions: 0,
                duration: Duration::from_secs(0),
                tps: 0.0,
                latency_avg_ms: 0.0,
                latency_p99_ms: 0.0,
                success_rate: 0.0,
            });
        }
    };

    let (mut write, mut read) = ws_stream.split();

    // Spawn task to receive acknowledgments
    let total_batches = (transactions.len() + batch_size - 1) / batch_size;
    let ack_handle = tokio::spawn(async move {
        let mut received_acks = 0;
        let mut total_accepted = 0;

        while let Some(result) = read.next().await {
            match result {
                Ok(Message::Binary(data)) => {
                    if let Ok(ack) = rmp_serde::from_slice::<BatchAck>(&data) {
                        received_acks += 1;
                        total_accepted += ack.accepted;

                        if received_acks >= total_batches {
                            break;
                        }
                    }
                }
                Ok(Message::Close(_)) => break,
                Err(_) => break,
                _ => {}
            }
        }

        (received_acks, total_accepted)
    });

    let start = Instant::now();

    // Stream all batches
    for (batch_id, chunk) in transactions.chunks(batch_size).enumerate() {
        let batch = TransactionBatch {
            transactions: chunk.to_vec(),
        };

        let packed = rmp_serde::to_vec(&batch)?;
        write.send(Message::Binary(packed)).await?;
    }

    write.close().await?;

    let send_duration = start.elapsed();

    // Wait for acknowledgments
    let (acks_received, total_accepted) = tokio::time::timeout(
        Duration::from_secs(30),
        ack_handle
    ).await??;

    let total_duration = start.elapsed();
    let tps = total_accepted as f64 / total_duration.as_secs_f64();

    let latency_per_tx = total_duration.as_millis() as f64 / total_accepted as f64;

    Ok(BenchmarkResult {
        test_name: format!("WebSocket Streaming ({}x batches)", batch_size),
        total_transactions: transactions.len(),
        duration: total_duration,
        tps,
        latency_avg_ms: latency_per_tx,
        latency_p99_ms: latency_per_tx * 1.5, // Estimate
        success_rate: total_accepted as f64 / transactions.len() as f64,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("\n{}", "=".repeat(80));
    println!("🚀 ADVANCED TPS BENCHMARK - Path to 1M+ TPS");
    println!("{}", "=".repeat(80));
    println!("📡 Target API: {}", API_BASE);
    println!("🎯 Goal: Identify bottlenecks preventing 1M+ TPS");
    println!("{}", "=".repeat(80));

    // Generate signing keys
    println!("\n👛 Generating Ed25519 signing keys...");
    let mut sender_seed = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut sender_seed);
    let sender_key = SigningKey::from_bytes(&sender_seed);

    let mut receiver_seed = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut receiver_seed);

    let sender_address: Address = sender_seed;
    let receiver_address: Address = receiver_seed;

    println!("  ✅ Sender: {}", hex::encode(&sender_address[..8]));
    println!("  ✅ Receiver: {}", hex::encode(&receiver_address[..8]));

    // Build HTTP client
    let client = ClientBuilder::new()
        .pool_max_idle_per_host(200)
        .timeout(Duration::from_secs(60))
        .build()?;

    let mut all_results = Vec::new();

    // Progressive testing: start small, scale up to find bottleneck
    let test_configs = vec![
        ("Small batch", 100, vec![100, 500, 1000]),
        ("Medium batch", 1000, vec![100, 1000, 5000]),
        ("Large batch", 10000, vec![1000, 5000, 10000]),
    ];

    for (config_name, tx_count, batch_sizes) in test_configs {
        println!("\n{}", "=".repeat(80));
        println!("📦 Test Configuration: {} ({} transactions)", config_name, tx_count);
        println!("{}", "=".repeat(80));

        // Generate transactions for this test
        let transactions = generate_transactions(tx_count, &sender_key, sender_address, receiver_address);

        // Test HTTP JSON (only for small batch)
        if tx_count <= 100 {
            if let Ok(result) = benchmark_http_json(&client, &transactions).await {
                result.print();
                result.gap_to_1m();
                all_results.push(result);
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        // Test HTTP Binary Single (only for small batch)
        if tx_count <= 100 {
            if let Ok(result) = benchmark_http_binary_single(&client, &transactions).await {
                result.print();
                result.gap_to_1m();
                all_results.push(result);
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        // Test HTTP Binary Batch with progressive batch sizes
        for &batch_size in &batch_sizes {
            if let Ok(result) = benchmark_http_binary_batch(&client, &transactions, batch_size).await {
                result.print();
                result.gap_to_1m();
                all_results.push(result);
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        // Test WebSocket Streaming
        for &batch_size in &batch_sizes {
            if let Ok(result) = benchmark_websocket_stream(&transactions, batch_size).await {
                if result.tps > 0.0 {
                    result.print();
                    result.gap_to_1m();
                    all_results.push(result);
                }
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }

    // Final analysis
    println!("\n{}", "=".repeat(80));
    println!("📊 PERFORMANCE ANALYSIS SUMMARY");
    println!("{}", "=".repeat(80));

    if let Some(best) = all_results.iter().max_by(|a, b| a.tps.partial_cmp(&b.tps).unwrap()) {
        println!("\n🏆 BEST PERFORMANCE:");
        println!("   Test: {}", best.test_name);
        println!("   TPS: {:.0}", best.tps);
        println!("   Success Rate: {:.1}%", best.success_rate * 100.0);

        let gap = 1_000_000.0 / best.tps;
        println!("\n📈 GAP TO 1M TPS: {:.1}x improvement needed", gap);

        println!("\n🔧 RECOMMENDED OPTIMIZATIONS:");
        if gap > 100.0 {
            println!("   1. ✅ Enable WebSocket streaming (1000x improvement)");
            println!("   2. ✅ Increase batch sizes to 10,000+ transactions");
            println!("   3. ✅ Enable SIMD batch signature verification");
            println!("   4. ⏳ Optimize database writes (batch commits)");
        } else if gap > 10.0 {
            println!("   1. ✅ Already using binary protocol - good!");
            println!("   2. ✅ Increase batch sizes to 50,000 transactions");
            println!("   3. ✅ Enable SIMD parallel signature verification");
            println!("   4. ✅ Enable io_uring kernel I/O (Linux only)");
        } else if gap > 2.0 {
            println!("   1. ✅ Already optimized for high throughput!");
            println!("   2. ✅ Enable SIMD if not already active");
            println!("   3. ⏳ Profile CPU usage to find remaining bottleneck");
            println!("   4. ⏳ Consider hardware acceleration (GPU for signatures)");
        } else {
            println!("   ✅ CONGRATULATIONS! You've achieved 1M+ TPS target!");
            println!("   ⏳ Focus on stability testing and mainnet deployment");
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("✅ Benchmark Complete!");
    println!("{}", "=".repeat(80));

    Ok(())
}
