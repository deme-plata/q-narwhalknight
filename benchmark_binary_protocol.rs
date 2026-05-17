/// Binary Protocol TPS Benchmark
///
/// This benchmark measures the actual TPS using:
/// - Real transactions with Ed25519 signatures
/// - Binary MessagePack protocol
/// - Real API endpoints
/// - Production code paths
///
/// Run with: cargo run --release --bin benchmark_binary_protocol

use q_types::{Transaction, Address};
use q_wallet::{WalletManager, MemoryWalletStore};
use ed25519_dalek::{SigningKey, Signer};
use sha3::{Sha3_256, Digest};
use chrono::Utc;
use anyhow::Result;
use rmp_serde;
use reqwest;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Binary Protocol TPS Benchmark");
    println!("=" .repeat(80));
    println!();

    // Initialize real wallet manager
    let wallet_store = MemoryWalletStore::new();
    let wallet_manager = WalletManager::new(wallet_store);

    // Create real wallets
    println!("👛 Creating real wallets...");
    let wallet1_id = wallet_manager.create_wallet("sender".to_string(), "password123".to_string()).await?;
    let wallet2_id = wallet_manager.create_wallet("receiver".to_string(), "password456".to_string()).await?;

    println!("  ✅ Wallet 1 (sender): {}", wallet1_id);
    println!("  ✅ Wallet 2 (receiver): {}", wallet2_id);
    println!();

    // Get wallet info
    let wallet1_info = wallet_manager.get_wallet(&wallet1_id).await?;
    let wallet2_info = wallet_manager.get_wallet(&wallet2_id).await?;

    // Create real signing keys from wallet seeds
    let sender_key = SigningKey::from_bytes(&wallet1_info.seed);
    let receiver_address: Address = wallet2_info.address;

    println!("💰 Wallet Setup:");
    println!("  Sender address: {}", hex::encode(&wallet1_info.address));
    println!("  Receiver address: {}", hex::encode(&receiver_address));
    println!();

    // Generate real transactions with actual signatures
    println!("📝 Generating real transactions with Ed25519 signatures...");
    let batch_size = 10000; // Test with 10K transactions for real TPS measurement
    let mut transactions = Vec::with_capacity(batch_size);

    let start_gen = std::time::Instant::now();

    for i in 0..batch_size {
        // Create real transaction ID
        let mut tx_id = [0u8; 32];
        let mut rng = rand::thread_rng();
        rand::RngCore::fill_bytes(&mut rng, &mut tx_id);

        // Real transaction data
        let amount = 1000u64;
        let fee = 1u64;
        let nonce = i as u64;
        let data = vec![];

        // Create message to sign (exact same as production code)
        let mut hasher = Sha3_256::new();
        hasher.update(&tx_id);
        hasher.update(&wallet1_info.address);
        hasher.update(&receiver_address);
        hasher.update(&amount.to_le_bytes());
        hasher.update(&fee.to_le_bytes());
        hasher.update(&nonce.to_le_bytes());
        hasher.update(&data);
        let message = hasher.finalize();

        // Real Ed25519 signature
        let signature = sender_key.sign(&message);

        // Create real transaction
        let tx = Transaction {
            id: tx_id,
            from: wallet1_info.address,
            to: receiver_address,
            amount,
            fee,
            nonce,
            signature: signature.to_bytes().to_vec(),
            timestamp: Utc::now(),
            data,
        };

        transactions.push(tx);
    }

    let gen_time = start_gen.elapsed();
    println!("  ✅ Generated {} real transactions in {:.2}ms", batch_size, gen_time.as_millis());
    println!("  📊 Generation rate: {:.0} tx/sec", batch_size as f64 / gen_time.as_secs_f64());
    println!();

    // Test 1: JSON Protocol (Baseline)
    println!("🔬 Test 1: JSON Protocol (Baseline)");
    println!("-" .repeat(80));

    let client = reqwest::Client::new();
    let mut successful = 0;

    let json_start = std::time::Instant::now();

    for tx in &transactions {
        let request_body = serde_json::json!({
            "transaction": {
                "id": tx.id.to_vec(),
                "from": tx.from.to_vec(),
                "to": tx.to.to_vec(),
                "amount": tx.amount,
                "fee": tx.fee,
                "nonce": tx.nonce,
                "signature": tx.signature.clone(),
                "timestamp": tx.timestamp.to_rfc3339(),
                "data": tx.data.clone()
            }
        });

        let response = client
            .post("http://localhost:8200/api/v1/transactions")
            .json(&request_body)
            .send()
            .await;

        if let Ok(resp) = response {
            if resp.status().is_success() {
                successful += 1;
            }
        }
    }

    let json_elapsed = json_start.elapsed();
    let json_tps = successful as f64 / json_elapsed.as_secs_f64();
    let json_latency = json_elapsed.as_millis() as f64 / successful as f64;

    println!("  Successful: {}/{}", successful, batch_size);
    println!("  Time: {:.2}s", json_elapsed.as_secs_f64());
    println!("  TPS: {:.0}", json_tps);
    println!("  Latency: {:.2}ms per tx", json_latency);
    println!();

    // Test 2: Binary MessagePack Protocol (Single)
    println!("🔬 Test 2: Binary MessagePack Protocol (Single Transaction)");
    println!("-" .repeat(80));

    successful = 0;
    let binary_start = std::time::Instant::now();

    for tx in &transactions {
        // Serialize to MessagePack
        let packed = rmp_serde::to_vec(&tx)?;

        let response = client
            .post("http://localhost:8200/api/v1/binary/transaction")
            .header("Content-Type", "application/msgpack")
            .body(packed)
            .send()
            .await;

        if let Ok(resp) = response {
            if resp.status().is_success() {
                successful += 1;
            }
        }
    }

    let binary_elapsed = binary_start.elapsed();
    let binary_tps = successful as f64 / binary_elapsed.as_secs_f64();
    let binary_latency = binary_elapsed.as_millis() as f64 / successful as f64;

    println!("  Successful: {}/{}", successful, batch_size);
    println!("  Time: {:.2}s", binary_elapsed.as_secs_f64());
    println!("  TPS: {:.0}", binary_tps);
    println!("  Latency: {:.2}ms per tx", binary_latency);
    println!();

    // Test 3: Binary Batch Protocol (LARGE batches for maximum TPS)
    println!("🔬 Test 3: Binary Batch Protocol (1000 tx per batch - HIGH PERFORMANCE)");
    println!("-" .repeat(80));

    successful = 0;
    let batch_start = std::time::Instant::now();

    for chunk in transactions.chunks(1000) {
        #[derive(serde::Serialize)]
        struct BinaryTransactionBatch {
            transactions: Vec<Transaction>,
        }

        let batch = BinaryTransactionBatch {
            transactions: chunk.to_vec(),
        };

        // Serialize to MessagePack
        let packed = rmp_serde::to_vec(&batch)?;

        let response = client
            .post("http://localhost:8200/api/v1/binary/batch")
            .header("Content-Type", "application/msgpack")
            .body(packed)
            .send()
            .await;

        if let Ok(resp) = response {
            if resp.status().is_success() {
                successful += chunk.len();
            }
        }
    }

    let batch_elapsed = batch_start.elapsed();
    let batch_tps = successful as f64 / batch_elapsed.as_secs_f64();
    let batch_latency = batch_elapsed.as_millis() as f64 / successful as f64;

    println!("  Successful: {}/{}", successful, batch_size);
    println!("  Time: {:.2}s", batch_elapsed.as_secs_f64());
    println!("  TPS: {:.0}", batch_tps);
    println!("  Latency: {:.4}ms per tx", batch_latency);
    println!();

    // Summary
    println!("📊 PERFORMANCE SUMMARY");
    println!("=" .repeat(80));
    println!();
    println!("┌─────────────────────┬──────────────┬──────────────┬──────────────┐");
    println!("│ Metric              │ JSON (Base)  │ Binary (1x)  │ Binary Batch │");
    println!("├─────────────────────┼──────────────┼──────────────┼──────────────┤");
    println!("│ TPS                 │ {:>11.0}  │ {:>11.0}  │ {:>11.0}  │", json_tps, binary_tps, batch_tps);
    println!("│ Latency (ms/tx)     │ {:>11.2}  │ {:>11.2}  │ {:>11.4}  │", json_latency, binary_latency, batch_latency);

    let binary_improvement = binary_tps / json_tps;
    let batch_improvement = batch_tps / json_tps;

    println!("├─────────────────────┼──────────────┼──────────────┼──────────────┤");
    println!("│ Improvement Factor  │      1.00x   │ {:>11.1}x  │ {:>11.1}x  │", binary_improvement, batch_improvement);
    println!("└─────────────────────┴──────────────┴──────────────┴──────────────┘");
    println!();

    println!("🎯 KEY FINDINGS:");
    println!("  • JSON Protocol:     {:.0} TPS ({:.2}ms latency)", json_tps, json_latency);
    println!("  • Binary Single:     {:.0} TPS ({:.2}ms latency) - {:.1}x improvement", binary_tps, binary_latency, binary_improvement);
    println!("  • Binary Batch:      {:.0} TPS ({:.4}ms latency) - {:.1}x improvement", batch_tps, batch_latency, batch_improvement);
    println!();

    if batch_improvement >= 100.0 {
        println!("✅ ACHIEVED 100x+ IMPROVEMENT WITH BINARY BATCH PROTOCOL!");
    } else if batch_improvement >= 50.0 {
        println!("✅ ACHIEVED 50x+ IMPROVEMENT - Binary batch protocol is highly effective!");
    } else if batch_improvement >= 10.0 {
        println!("✅ ACHIEVED 10x+ IMPROVEMENT - Significant performance gain!");
    } else {
        println!("⚠️  Improvement below expected target - HTTP overhead still dominates");
    }

    println!();
    println!("🚀 NEXT STEPS TO 1M+ TPS:");
    println!("  1. Enable WebSocket streaming (eliminate HTTP overhead)");
    println!("  2. Optimize background batch processor (16 parallel workers)");
    println!("  3. Enable kernel I/O (io_uring) for 5-10x improvement");
    println!("  4. SIMD batch signature verification");
    println!();

    Ok(())
}
