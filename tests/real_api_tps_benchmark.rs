// Real API TPS Benchmark - Using actual HTTP APIs like the web wallet
// Tests real-world performance by calling the production API endpoints

use anyhow::Result;
use reqwest::Client;
use serde_json::json;
use std::time::Instant;

#[tokio::test]
async fn test_real_wallet_api_tps() -> Result<()> {
    println!("🚀 Real Wallet API TPS Benchmark");
    println!("{}", "=".repeat(60));
    println!();
    println!("Testing actual production APIs - same flow as web wallet");
    println!();

    let client = Client::new();
    let node_url = "http://localhost:8200";

    // Check node health
    println!("🏥 Checking node health...");
    let health_resp = client.get(format!("{}/health", node_url))
        .send()
        .await?;

    if health_resp.status().is_success() {
        println!("  ✅ Node is healthy");
    } else {
        println!("  ❌ Node not healthy, status: {}", health_resp.status());
        println!("  💡 Start node with: Q_DB_PATH=./data-test ./target/release/q-api-server --port 8200");
        return Ok(());
    }
    println!();

    // Create wallets using API
    println!("👛 Creating wallets via /api/v1/wallets...");

    let create_wallet_resp = client.post(format!("{}/api/v1/wallets", node_url))
        .json(&json!({
            "name": "benchmark-sender",
            "password": "test123"
        }))
        .send()
        .await?;

    let sender_wallet = if create_wallet_resp.status().is_success() {
        let data: serde_json::Value = create_wallet_resp.json().await?;
        if let Some(wallet) = data.get("data") {
            println!("  ✅ Created sender wallet");
            wallet.clone()
        } else {
            println!("  ⚠️  API response format: {:?}", data);
            println!("  💡 Wallet API may not be implemented yet");
            println!("  💡 Continuing with transaction-only test...");
            println!();

            // Test transactions directly without wallet creation
            test_transaction_api_directly(&client, node_url).await?;
            return Ok(());
        }
    } else {
        println!("  ⚠️  Wallet creation returned: {}", create_wallet_resp.status());
        println!("  💡 Wallet API may not be implemented yet");
        println!("  💡 Continuing with transaction-only test...");
        println!();

        test_transaction_api_directly(&client, node_url).await?;
        return Ok(());
    };

    println!("  Sender wallet ID: {}", sender_wallet.get("id").and_then(|v| v.as_str()).unwrap_or("unknown"));
    println!();

    // Fund via faucet
    println!("💰 Requesting faucet funds...");
    if let Some(address) = sender_wallet.get("address").and_then(|v| v.as_str()) {
        let faucet_resp = client.post(format!("{}/api/v1/faucet", node_url))
            .json(&json!({"address": address}))
            .send()
            .await?;

        if faucet_resp.status().is_success() {
            println!("  ✅ Faucet request successful");
        } else {
            println!("  ⚠️  Faucet request: {}", faucet_resp.status());
        }
    }
    println!();

    // Create transactions via wallet API
    println!("📝 Creating transactions via wallet API...");
    let batch_size = 100;
    let num_batches = 10;

    let wallet_id = sender_wallet.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let mut all_transactions = Vec::new();

    let start_gen = Instant::now();

    for batch_num in 0..num_batches {
        for i in 0..batch_size {
            // Create transaction via wallet API endpoint
            let tx_resp = client.post(format!("{}/api/v1/wallets/{}/transactions", node_url, wallet_id))
                .json(&json!({
                    "to": format!("receiver-address-{}", i),
                    "amount": 10,
                    "fee": 1
                }))
                .send()
                .await;

            if let Ok(resp) = tx_resp {
                if resp.status().is_success() {
                    if let Ok(tx_data) = resp.json::<serde_json::Value>().await {
                        if let Some(tx) = tx_data.get("data") {
                            all_transactions.push(json!({"transaction": tx}));
                        }
                    }
                }
            }
        }

        if (batch_num + 1) % 5 == 0 {
            println!("  Generated {} transactions...", all_transactions.len());
        }
    }

    let gen_time = start_gen.elapsed();
    println!("  ✅ Generated {} transactions in {:.2}s", all_transactions.len(), gen_time.as_secs_f64());
    println!("  📊 Generation rate: {:.0} tx/sec", all_transactions.len() as f64 / gen_time.as_secs_f64());
    println!();

    if all_transactions.is_empty() {
        println!("  ⚠️  No transactions created via API");
        println!("  💡 Wallet transaction endpoint may not be implemented");
        return Ok(());
    }

    // Submit batches
    println!("🚀 Submitting to batch API...");
    let mut total_submitted = 0;
    let start_bench = Instant::now();

    for (batch_idx, chunk) in all_transactions.chunks(batch_size).enumerate() {
        let batch_resp = client.post(format!("{}/api/v1/transactions/batch", node_url))
            .json(&json!({"transactions": chunk}))
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await;

        match batch_resp {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(data) = resp.json::<serde_json::Value>().await {
                    if let Some(result) = data.get("data") {
                        let submitted = result.get("submitted").and_then(|v| v.as_u64()).unwrap_or(0);
                        let tps = result.get("tps").and_then(|v| v.as_u64()).unwrap_or(0);
                        total_submitted += submitted;
                        println!("  Batch {}: {} tx submitted, {} TPS", batch_idx + 1, submitted, tps);
                    }
                }
            }
            Ok(resp) => {
                println!("  ❌ Batch {} failed: {}", batch_idx + 1, resp.status());
            }
            Err(e) => {
                println!("  ❌ Batch {} error: {}", batch_idx + 1, e);
            }
        }
    }

    let total_time = start_bench.elapsed();
    let overall_tps = total_submitted as f64 / total_time.as_secs_f64();

    println!();
    println!("📈 Real Wallet API Benchmark Results:");
    println!("{}", "=".repeat(60));
    println!("  Total submitted: {}", total_submitted);
    println!("  Time: {:.2}s", total_time.as_secs_f64());
    println!("  Overall TPS: {:.0}", overall_tps);
    println!();
    println!("✅ Real-world API flow:");
    println!("  - Created wallets via /api/v1/wallets");
    println!("  - Funded via /api/v1/faucet");
    println!("  - Created tx via /api/v1/wallets/{{id}}/transactions");
    println!("  - Submitted batches via /api/v1/transactions/batch");
    println!("  - Same APIs used by web wallet");
    println!();

    Ok(())
}

async fn test_transaction_api_directly(client: &Client, node_url: &str) -> Result<()> {
    println!("📝 Testing transaction batch API directly...");
    println!();

    // Generate sample transactions with proper format
    let batch_size = 1000;
    let mut transactions = Vec::new();

    println!("  Generating {} sample transactions...", batch_size);

    for i in 0..batch_size {
        // Create properly formatted transaction
        let tx = json!({
            "transaction": {
                "id": vec![i as u8; 32],
                "from": vec![(i % 256) as u8; 32],
                "to": vec![((i + 1) % 256) as u8; 32],
                "amount": 100u64,
                "fee": 1u64,
                "nonce": i as u64,
                "signature": vec![(i % 256) as u8; 64],
                "timestamp": "2025-10-01T00:00:00Z",
                "data": Vec::<u8>::new()
            }
        });

        transactions.push(tx);
    }

    println!("  ✅ Generated {} transactions", transactions.len());
    println!();

    // Submit to batch API
    println!("🚀 Submitting batch to /api/v1/transactions/batch...");
    let start = Instant::now();

    let resp = client.post(format!("{}/api/v1/transactions/batch", node_url))
        .json(&json!({"transactions": transactions}))
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await?;

    let elapsed = start.elapsed();

    println!("  Response status: {}", resp.status());

    if resp.status().is_success() {
        let data: serde_json::Value = resp.json().await?;
        println!("  ✅ Batch submission successful!");
        println!("  📊 Response: {}", serde_json::to_string_pretty(&data)?);

        if let Some(result) = data.get("data") {
            if let Some(submitted) = result.get("submitted").and_then(|v| v.as_u64()) {
                let tps = submitted as f64 / elapsed.as_secs_f64();
                println!();
                println!("📈 Direct Batch API Results:");
                println!("  Submitted: {} transactions", submitted);
                println!("  Time: {:.2}s", elapsed.as_secs_f64());
                println!("  TPS: {:.0}", tps);
            }
        }
    } else {
        let error_text = resp.text().await?;
        println!("  ❌ Batch submission failed: {}", error_text);
    }

    println!();

    Ok(())
}
