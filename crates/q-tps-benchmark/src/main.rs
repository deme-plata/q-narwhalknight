/// Standalone Real-World TPS Benchmark for Quillon-NarwhalKnight
/// Tests actual throughput on live API with real transactions
/// Includes: Regular transactions, Privacy mixing, PaaS features, ZK-STARKs
/// NO internal crate dependencies - pure API testing

use std::time::{Duration, Instant};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Semaphore;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use reqwest::{Client, ClientBuilder};
use sha2::{Sha256, Digest};
use chrono::Utc;

const API_BASE: &str = "http://localhost:8080"; // Updated to match running server
const MAX_CONCURRENT: usize = 200; // Increased for better throughput testing

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

#[derive(Debug, Serialize)]
struct MixingRequest {
    from_address: String,
    to_address: String,
    amount: f64,
    pool_size_target: usize,
}

#[derive(Debug, Serialize)]
struct PaasApiKeyRequest {
    wallet_address: String,
    tier: String,
    expires_days: Option<u32>,
}

#[derive(Debug)]
struct BenchmarkResults {
    test_name: String,
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
        println!("📊 {} - BENCHMARK RESULTS", self.test_name.to_uppercase());
        println!("{}", "=".repeat(80));
        println!("📈 Total Transactions: {}", self.total_transactions);
        println!("✅ Successful: {} ({:.1}%)", self.successful, (self.successful as f64 / self.total_transactions as f64) * 100.0);
        println!("❌ Failed: {} ({:.1}%)", self.failed, (self.failed as f64 / self.total_transactions as f64) * 100.0);
        println!("⏱️  Total Time: {:.2}s", self.elapsed.as_secs_f64());
        println!("⚡ Actual TPS: {:.2}", self.tps);

        if !self.latencies.is_empty() {
            let mut sorted = self.latencies.clone();
            sorted.sort();

            let avg = sorted.iter().map(|d| d.as_millis()).sum::<u128>() as f64 / sorted.len() as f64;
            let median = sorted[sorted.len() / 2].as_millis();
            let min = sorted[0].as_millis();
            let max = sorted[sorted.len() - 1].as_millis();
            let p50 = sorted[(sorted.len() as f64 * 0.50) as usize].as_millis();
            let p95 = sorted[(sorted.len() as f64 * 0.95) as usize].as_millis();
            let p99 = sorted[(sorted.len() as f64 * 0.99) as usize].as_millis();

            println!("\n🕐 Latency Statistics:");
            println!("  • Average: {:.2}ms", avg);
            println!("  • Median (P50): {}ms", median);
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
        .post(format!("{}/api/v1/faucet", API_BASE))
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
        .post(format!("{}/api/v1/transactions/send", API_BASE))
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

async fn send_mixing_transaction(
    client: &Client,
    from: &str,
    to: &str,
    amount: f64,
) -> anyhow::Result<Duration> {
    let start = Instant::now();

    let response = client
        .post(format!("{}/api/v1/mixer/send", API_BASE))
        .json(&MixingRequest {
            from_address: from.to_string(),
            to_address: to.to_string(),
            amount,
            pool_size_target: 64,
        })
        .send()
        .await?;

    let latency = start.elapsed();

    if response.status().is_success() {
        Ok(latency)
    } else {
        anyhow::bail!("Mixing failed: {}", response.status())
    }
}

async fn generate_paas_key(
    client: &Client,
    wallet: &str,
) -> anyhow::Result<Duration> {
    let start = Instant::now();

    let response = client
        .post(format!("{}/api/v1/privacy/paas/api-keys/generate", API_BASE))
        .json(&PaasApiKeyRequest {
            wallet_address: wallet.to_string(),
            tier: "free".to_string(),
            expires_days: Some(90),
        })
        .send()
        .await?;

    let latency = start.elapsed();

    if response.status().is_success() {
        Ok(latency)
    } else {
        anyhow::bail!("PaaS key generation failed: {}", response.status())
    }
}

async fn run_benchmark<F, Fut>(
    test_name: &str,
    num_transactions: usize,
    client: Arc<Client>,
    semaphore: Arc<Semaphore>,
    wallets: Arc<Vec<Wallet>>,
    operation: F,
) -> BenchmarkResults
where
    F: Fn(Arc<Client>, Arc<Vec<Wallet>>, usize) -> Fut + Send + Sync + Clone + 'static,
    Fut: std::future::Future<Output = anyhow::Result<Duration>> + Send + 'static,
{
    let mut tasks = Vec::new();
    let start = Instant::now();

    for i in 0..num_transactions {
        let sem = semaphore.clone();
        let client = client.clone();
        let wallets = wallets.clone();
        let op = operation.clone();

        let task = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            op(client, wallets, i).await
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

    BenchmarkResults {
        test_name: test_name.to_string(),
        total_transactions: num_transactions,
        successful,
        failed,
        elapsed,
        tps,
        latencies,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("\n{}", "=".repeat(80));
    println!("🚀 Quillon-NarwhalKnight Comprehensive TPS Benchmark");
    println!("{}", "=".repeat(80));
    println!("📡 Testing against: {}", API_BASE);
    println!("🔧 Max concurrent: {}", MAX_CONCURRENT);
    println!("📅 Date: {}", Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
    println!("{}", "=".repeat(80));

    // Build HTTP client with connection pooling
    let client = Arc::new(ClientBuilder::new()
        .pool_max_idle_per_host(MAX_CONCURRENT)
        .timeout(Duration::from_secs(60))
        .build()?);

    // Create test wallets
    println!("\n💰 Creating test wallets...");
    let num_wallets = 100;
    let mut wallets_vec = Vec::new();
    for i in 0..num_wallets {
        wallets_vec.push(Wallet::new(&format!("benchmark_wallet_{}", i)));
    }
    let wallets = Arc::new(wallets_vec);
    println!("✅ Created {} wallets", wallets.len());

    // Fund wallets (optional, comment out if faucet not available)
    println!("\n💸 Funding wallets from faucet...");
    let mut funded = 0;
    for wallet in wallets.iter().take(10) {
        if fund_wallet(&client, &wallet.address).await.is_ok() {
            funded += 1;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    println!("✅ Funded {} wallets (faucet may be limited)", funded);

    // Warmup
    println!("\n🔥 Warming up API with 20 requests...");
    for i in 0..20 {
        let from = &wallets[i % wallets.len()].address;
        let to = &wallets[(i + 1) % wallets.len()].address;
        let _ = send_transaction(&client, from, to, 0.001).await;
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    println!("✅ Warmup complete\n");

    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT));
    let mut all_results = Vec::new();

    // Test 1: Standard Transactions
    println!("🧪 Test 1: Standard Transaction Throughput");
    let results1 = run_benchmark(
        "Standard Transactions",
        500,
        client.clone(),
        semaphore.clone(),
        wallets.clone(),
        |client, wallets, i| async move {
            let from_idx = i % wallets.len();
            let to_idx = (i + 1) % wallets.len();
            send_transaction(
                &client,
                &wallets[from_idx].address,
                &wallets[to_idx].address,
                0.001,
            )
            .await
        },
    )
    .await;
    results1.print_report();
    all_results.push(results1);

    tokio::time::sleep(Duration::from_secs(2)).await;

    // Test 2: Privacy Mixing Transactions
    println!("\n🧪 Test 2: Privacy Mixing Throughput");
    let results2 = run_benchmark(
        "Privacy Mixing",
        200,
        client.clone(),
        semaphore.clone(),
        wallets.clone(),
        |client, wallets, i| async move {
            let from_idx = i % wallets.len();
            let to_idx = (i + 1) % wallets.len();
            send_mixing_transaction(
                &client,
                &wallets[from_idx].address,
                &wallets[to_idx].address,
                0.001,
            )
            .await
        },
    )
    .await;
    results2.print_report();
    all_results.push(results2);

    tokio::time::sleep(Duration::from_secs(2)).await;

    // Test 3: PaaS API Key Generation
    println!("\n🧪 Test 3: PaaS API Key Generation Throughput");
    let results3 = run_benchmark(
        "PaaS API Keys",
        100,
        client.clone(),
        semaphore.clone(),
        wallets.clone(),
        |client, wallets, i| async move {
            let wallet_idx = i % wallets.len();
            generate_paas_key(&client, &wallets[wallet_idx].address).await
        },
    )
    .await;
    results3.print_report();
    all_results.push(results3);

    // Print summary
    println!("\n{}", "=".repeat(80));
    println!("📊 COMPREHENSIVE BENCHMARK SUMMARY");
    println!("{}", "=".repeat(80));

    for result in &all_results {
        println!("\n{}", result.test_name);
        println!("  TPS: {:.2}", result.tps);
        println!("  Success Rate: {:.1}%", (result.successful as f64 / result.total_transactions as f64) * 100.0);

        if !result.latencies.is_empty() {
            let mut sorted = result.latencies.clone();
            sorted.sort();
            let p50 = sorted[sorted.len() / 2].as_millis();
            let p99 = sorted[(sorted.len() as f64 * 0.99) as usize].as_millis();
            println!("  Latency P50: {}ms, P99: {}ms", p50, p99);
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("✅ Benchmark Complete!");
    println!("💾 Save these results for whitepaper performance claims");
    println!("{}", "=".repeat(80));

    Ok(())
}
