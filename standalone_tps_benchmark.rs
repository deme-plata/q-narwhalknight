/// Standalone Real-World TPS Benchmark for Q-NarwhalKnight
/// Tests actual throughput on live quillon.xyz API with real transactions
/// NO internal crate dependencies - pure API testing

use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::Semaphore;
use serde::{Deserialize, Serialize};
use reqwest::{Client, ClientBuilder};
use sha2::{Sha256, Digest};
use chrono::Utc;

const API_BASE: &str = "https://quillon.xyz";
const MAX_CONCURRENT: usize = 100;

#[derive(Debug, Clone)]
struct Wallet {
    address: String,
    balance: f64,
}

impl Wallet {
    fn new(seed: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(seed.as_bytes());
        let hash = hasher.finalize();
        let address = format!("{:x}", hash)[..40].to_string();

        Self {
            address,
            balance: 0.0,
        }
    }
}

#[derive(Debug, Serialize)]
struct FaucetRequest {
    address: String,
}

#[derive(Debug, Serialize)]
struct TransactionRequest {
    from: String,
    to: String,
    amount: f64,
    timestamp: String,
}

#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

#[derive(Debug)]
struct BenchmarkResults {
    total_transactions: usize,
    successful: usize,
    failed: usize,
    elapsed: Duration,
    tps: f64,
    latencies: Vec<Duration>,
}

impl BenchmarkResults {
    fn print_report(&self) {
        println!("\n{}", "=".repeat(80));
        println!("📊 REAL-WORLD TPS BENCHMARK RESULTS");
        println!("{}", "=".repeat(80));
        println!("📈 Total Transactions: {}", self.total_transactions);
        println!("✅ Successful: {}", self.successful);
        println!("❌ Failed: {}", self.failed);
        println!("⏱️  Total Time: {:.2}s", self.elapsed.as_secs_f64());
        println!("⚡ TPS: {:.2}", self.tps);

        if !self.latencies.is_empty() {
            let mut sorted = self.latencies.clone();
            sorted.sort();

            let avg = sorted.iter().map(|d| d.as_millis()).sum::<u128>() as f64 / sorted.len() as f64;
            let median = sorted[sorted.len() / 2].as_millis();
            let min = sorted[0].as_millis();
            let max = sorted[sorted.len() - 1].as_millis();
            let p95 = sorted[(sorted.len() as f64 * 0.95) as usize].as_millis();
            let p99 = sorted[(sorted.len() as f64 * 0.99) as usize].as_millis();

            println!("\n🕐 Latency Statistics:");
            println!("  • Average: {:.2}ms", avg);
            println!("  • Median: {}ms", median);
            println!("  • Min: {}ms", min);
            println!("  • Max: {}ms", max);
            println!("  • P95: {}ms", p95);
            println!("  • P99: {}ms", p99);
        }

        println!("{}", "=".repeat(80));
    }
}

async fn fund_wallet(client: &Client, address: &str) -> anyhow::Result<()> {
    let response = client
        .post(format!("{}/api/faucet", API_BASE))
        .json(&FaucetRequest {
            address: address.to_string(),
        })
        .send()
        .await?;

    if response.status().is_success() {
        Ok(())
    } else {
        anyhow::bail!("Faucet request failed: {}", response.status())
    }
}

async fn send_transaction(
    client: &Client,
    from: &str,
    to: &str,
    amount: f64,
) -> anyhow::Result<Duration> {
    let start = Instant::now();

    let response = client
        .post(format!("{}/api/transaction", API_BASE))
        .json(&TransactionRequest {
            from: from.to_string(),
            to: to.to_string(),
            amount,
            timestamp: Utc::now().to_rfc3339(),
        })
        .send()
        .await?;

    let latency = start.elapsed();

    if response.status().is_success() {
        Ok(latency)
    } else {
        anyhow::bail!("Transaction failed: {}", response.status())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🚀 Q-NarwhalKnight Real-World TPS Benchmark");
    println!("📡 Testing against: {}", API_BASE);
    println!("🔧 Max concurrent: {}\n", MAX_CONCURRENT);

    // Build HTTP client with connection pooling
    let client = ClientBuilder::new()
        .pool_max_idle_per_host(MAX_CONCURRENT)
        .timeout(Duration::from_secs(30))
        .build()?;

    // Create test wallets
    println!("💰 Creating test wallets...");
    let num_wallets = 50;
    let mut wallets = Vec::new();
    for i in 0..num_wallets {
        wallets.push(Wallet::new(&format!("test_wallet_{}", i)));
    }
    println!("✅ Created {} wallets", wallets.len());

    // Fund wallets
    println!("\n💸 Funding wallets from faucet...");
    for (i, wallet) in wallets.iter().enumerate() {
        if let Err(e) = fund_wallet(&client, &wallet.address).await {
            println!("⚠️  Warning: Failed to fund wallet {}: {}", i, e);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    println!("✅ Wallet funding complete");

    // Warmup
    println!("\n🔥 Warming up API...");
    for _ in 0..10 {
        let from = &wallets[0].address;
        let to = &wallets[1].address;
        let _ = send_transaction(&client, from, to, 0.001).await;
    }
    println!("✅ Warmup complete");

    // Run benchmark
    println!("\n⚡ Starting TPS benchmark...");
    let num_transactions = 1000;
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT));
    let client = Arc::new(client);

    let mut tasks = Vec::new();
    let start = Instant::now();

    for i in 0..num_transactions {
        let sem = semaphore.clone();
        let client = client.clone();
        let wallets = wallets.clone();

        let task = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();

            let from_idx = i % wallets.len();
            let to_idx = (i + 1) % wallets.len();

            send_transaction(
                &client,
                &wallets[from_idx].address,
                &wallets[to_idx].address,
                0.001,
            )
            .await
        });

        tasks.push(task);
    }

    // Wait for all transactions
    let mut successful = 0;
    let mut failed = 0;
    let mut latencies = Vec::new();

    for task in tasks {
        match task.await {
            Ok(Ok(latency)) => {
                successful += 1;
                latencies.push(latency);
            }
            Ok(Err(_)) | Err(_) => {
                failed += 1;
            }
        }
    }

    let elapsed = start.elapsed();
    let tps = successful as f64 / elapsed.as_secs_f64();

    // Print results
    let results = BenchmarkResults {
        total_transactions: num_transactions,
        successful,
        failed,
        elapsed,
        tps,
        latencies,
    };

    results.print_report();

    Ok(())
}
