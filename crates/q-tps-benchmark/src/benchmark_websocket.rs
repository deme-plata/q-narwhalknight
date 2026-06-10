/// WebSocket Streaming TPS Benchmark
///
/// This benchmark measures TPS using persistent WebSocket connection:
/// - Zero HTTP overhead (persistent connection)
/// - Binary MessagePack protocol
/// - Continuous streaming of transaction batches
/// - Server processes with 16 parallel workers
///
/// Target: 1M+ TPS
///
/// Run with: cargo run --release --bin benchmark_websocket

use q_types::{Transaction, Address};
use ed25519_dalek::{SigningKey, Signer};
use sha3::{Sha3_256, Digest};
use chrono::Utc;
use anyhow::Result;
use rmp_serde;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use futures::{SinkExt, StreamExt};
use rand::RngCore;
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
struct TransactionBatch {
    pub batch_id: u64,
    pub transactions: Vec<Transaction>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BatchAck {
    pub batch_id: u64,
    pub accepted: usize,
    pub rejected: usize,
    pub processing_time_us: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 WebSocket Streaming TPS Benchmark");
    println!("{}", "=".repeat(80));
    println!();
    println!("🎯 Target: 1M+ TPS with zero HTTP overhead");
    println!("📡 Protocol: Binary MessagePack over persistent WebSocket");
    println!("⚡ Architecture: 16 parallel workers on server");
    println!();

    // Create real signing keys
    println!("👛 Creating Ed25519 signing key...");
    let mut sender_seed = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut sender_seed);
    let sender_key = SigningKey::from_bytes(&sender_seed);

    let mut receiver_seed = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut receiver_seed);
    let receiver_address: Address = receiver_seed;
    let sender_address: Address = sender_seed;

    println!("  Sender: {}", hex::encode(&sender_address[..8]));
    println!("  Receiver: {}", hex::encode(&receiver_address[..8]));
    println!();

    // Connect to WebSocket endpoint
    println!("🔌 Connecting to WebSocket endpoint...");
    let (ws_stream, _) = connect_async("ws://localhost:8200/api/v1/ws/transactions").await?;
    println!("✅ WebSocket connection established");
    println!();

    let (mut write, mut read) = ws_stream.split();

    // Configuration
    let total_transactions = 1_000_000; // 1M transactions
    let batch_size = 10_000; // 10K per batch
    let num_batches = total_transactions / batch_size;

    println!("📝 Test Configuration:");
    println!("  Total transactions: {}", total_transactions);
    println!("  Batch size: {}", batch_size);
    println!("  Number of batches: {}", num_batches);
    println!();

    // Spawn task to receive and count acknowledgments
    let ack_handle = tokio::spawn(async move {
        let mut received_acks = 0;
        let mut total_accepted = 0;
        let mut total_rejected = 0;

        while let Some(result) = read.next().await {
            match result {
                Ok(Message::Binary(data)) => {
                    if let Ok(ack) = rmp_serde::from_slice::<BatchAck>(&data) {
                        received_acks += 1;
                        total_accepted += ack.accepted;
                        total_rejected += ack.rejected;

                        if received_acks % 10 == 0 {
                            println!("📬 Received {} acks ({} tx accepted)", received_acks, total_accepted);
                        }
                    }
                }
                Ok(Message::Close(_)) => break,
                Err(e) => {
                    eprintln!("❌ WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }

        (received_acks, total_accepted, total_rejected)
    });

    println!("🚀 Starting WebSocket streaming benchmark...");
    println!();

    let start = std::time::Instant::now();

    // Stream all batches
    for batch_id in 0..num_batches {
        let mut transactions = Vec::with_capacity(batch_size);

        // Generate real transactions with signatures
        for i in 0..batch_size {
            let mut tx_id = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut tx_id);

            // Real transaction data
            let amount = 1000u64;
            let fee = 1u64;
            let nonce = batch_id as u64 * batch_size as u64 + i as u64;

            // Create message to sign (same as production)
            let mut hasher = Sha3_256::new();
            hasher.update(&tx_id);
            hasher.update(&sender_address);
            hasher.update(&receiver_address);
            hasher.update(&amount.to_le_bytes());
            hasher.update(&fee.to_le_bytes());
            hasher.update(&nonce.to_le_bytes());
            let message = hasher.finalize();

            // Real Ed25519 signature
            let signature = sender_key.sign(&message);

            transactions.push(Transaction {
                id: tx_id,
                from: sender_address,
                to: receiver_address,
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

        let batch = TransactionBatch {
            batch_id: batch_id as u64,
            transactions,
        };

        // Serialize to MessagePack
        let packed = rmp_serde::to_vec(&batch)?;

        // Send over WebSocket
        write.send(Message::Binary(packed)).await?;

        if (batch_id + 1) % 10 == 0 {
            println!("📤 Sent batch {}/{} ({} tx total)",
                     batch_id + 1, num_batches, (batch_id + 1) as u64 * batch_size as u64);
        }
    }

    // Close the write side
    write.close().await?;

    let send_elapsed = start.elapsed();
    println!();
    println!("✅ All batches sent in {:.2}s", send_elapsed.as_secs_f64());
    println!();

    // Wait for acknowledgments
    println!("⏳ Waiting for server acknowledgments...");
    let (acks_received, total_accepted, total_rejected) = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        ack_handle
    ).await??;

    let total_elapsed = start.elapsed();

    println!();
    println!("📊 WEBSOCKET STREAMING BENCHMARK RESULTS");
    println!("{}", "=".repeat(80));
    println!();
    println!("Transactions sent:    {}", total_transactions);
    println!("Transactions accepted: {}", total_accepted);
    println!("Transactions rejected: {}", total_rejected);
    println!("Batches acknowledged:  {}/{}", acks_received, num_batches);
    println!();
    println!("Total time:           {:.2}s", total_elapsed.as_secs_f64());
    println!("Send time:            {:.2}s", send_elapsed.as_secs_f64());
    println!();

    let tps = total_accepted as f64 / total_elapsed.as_secs_f64();
    let latency_ms = total_elapsed.as_millis() as f64 / total_accepted as f64;

    println!("🚀 TPS:                {:.0}", tps);
    println!("⚡ Latency:            {:.4}ms per transaction", latency_ms);
    println!();

    // Comparison with HTTP batch
    let http_batch_tps = 22_664.0;
    let improvement = tps / http_batch_tps;

    println!("📈 PERFORMANCE COMPARISON");
    println!("{}", "=".repeat(80));
    println!();
    println!("HTTP Batch Protocol:   {:.0} TPS (baseline)", http_batch_tps);
    println!("WebSocket Streaming:   {:.0} TPS", tps);
    println!("Improvement Factor:    {:.1}x", improvement);
    println!();

    if tps >= 1_000_000.0 {
        println!("🎉 ✅ ACHIEVED 1M+ TPS TARGET!");
    } else if tps >= 500_000.0 {
        println!("🎯 ✅ ACHIEVED 500K+ TPS - Halfway to 1M target!");
    } else if tps >= 226_000.0 {
        println!("✅ ACHIEVED 226K+ TPS - 10x improvement over HTTP!");
    } else {
        println!("⚠️  Below 226K TPS target - investigating bottleneck");
    }

    println!();
    println!("🚀 NEXT STEPS TO REACH 1M+ TPS:");
    println!("  1. ✅ WebSocket streaming (current)");
    println!("  2. Add zero-copy rkyv deserialization (2-3x improvement)");
    println!("  3. Enable io_uring kernel I/O on Linux (5-10x improvement)");
    println!("  4. SIMD batch signature verification (already enabled)");
    println!();

    Ok(())
}
