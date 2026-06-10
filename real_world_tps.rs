/// Real-World TPS Benchmark for Q-NarwhalKnight Production System
/// Tests actual throughput on live quillon.xyz API with real transactions
///
/// Dependencies declared in workspace Cargo.toml:
/// - tokio, reqwest, serde, sha2, chrono, futures, anyhow

use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::Semaphore;
use serde::{Deserialize, Serialize};
use reqwest::{Client, ClientBuilder};
use sha2::{Sha256, Digest};
extern crate chrono;
extern crate futures;
extern crate anyhow;

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

struct TpsBenchmark {
    client: Client,
    wallets: Vec<Wallet>,
    semaphore: Arc<Semaphore>,
}

impl TpsBenchmark {
    async fn new(num_wallets: usize) -> anyhow::Result<Self> {
        println!("🔧 Initializing benchmark...");

        let client = ClientBuilder::new()
            .danger_accept_invalid_certs(true)
            .timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(MAX_CONCURRENT)
            .build()?;

        let mut wallets = Vec::new();
        for i in 0..num_wallets {
            let seed = format!("rust_tps_wallet_{}_{}", i, chrono::Utc::now().timestamp_millis());
            wallets.push(Wallet::new(&seed));
        }

        println!("✅ Created {} wallets", num_wallets);

        Ok(Self {
            client,
            wallets,
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT)),
        })
    }

    async fn check_health(&self) -> anyhow::Result<bool> {
        let url = format!("{}/health", API_BASE);
        let resp = self.client.get(&url).send().await?;
        let json: ApiResponse<String> = resp.json().await?;

        println!("✅ API Health: success={}", json.success);
        Ok(json.success)
    }

    async fn request_faucet(&self, wallet: &Wallet) -> anyhow::Result<bool> {
        let url = format!("{}/api/faucet", API_BASE);
        let payload = FaucetRequest {
            address: wallet.address.clone(),
        };

        let resp = self.client
            .post(&url)
            .json(&payload)
            .send()
            .await?;

        Ok(resp.status().is_success())
    }

    async fn fund_wallets(&mut self) -> anyhow::Result<usize> {
        println!("\n💰 Funding {} wallets via faucet...", self.wallets.len());

        let mut funded = 0;
        let batch_size = 10;

        for chunk in self.wallets.chunks(batch_size) {
            let mut tasks = Vec::new();

            for wallet in chunk {
                let wallet_clone = wallet.clone();
                let self_clone = self.client.clone();
                let task = tokio::spawn(async move {
                    let url = format!("{}/api/faucet", API_BASE);
                    let payload = FaucetRequest {
                        address: wallet_clone.address.clone(),
                    };

                    self_clone
                        .post(&url)
                        .json(&payload)
                        .send()
                        .await
                        .map(|r| r.status().is_success())
                        .unwrap_or(false)
                });
                tasks.push(task);
            }

            let results = futures::future::join_all(tasks).await;
            funded += results.iter().filter(|r| r.as_ref().ok() == Some(&true)).count();

            println!("  Funded {}/{} wallets...", funded, self.wallets.len());
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        // Update balances
        for wallet in &mut self.wallets {
            wallet.balance = 100.0; // Assume faucet gives 100 QNK
        }

        println!("✅ Successfully funded {} wallets", funded);
        Ok(funded)
    }

    async fn send_transaction(
        &self,
        from: &Wallet,
        to: &Wallet,
        amount: f64,
    ) -> anyhow::Result<(bool, Duration)> {
        let _permit = self.semaphore.acquire().await?;

        let start = Instant::now();

        let url = format!("{}/api/transactions", API_BASE);
        let payload = TransactionRequest {
            from: from.address.clone(),
            to: to.address.clone(),
            amount,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let result = self.client
            .post(&url)
            .json(&payload)
            .send()
            .await;

        let latency = start.elapsed();

        match result {
            Ok(resp) if resp.status().is_success() => Ok((true, latency)),
            _ => Ok((false, latency)),
        }
    }

    async fn execute_transaction_wave(&mut self, num_transactions: usize) -> anyhow::Result<BenchmarkResults> {
        println!("\n🚀 Executing {} transactions...", num_transactions);

        let mut tasks = Vec::new();
        let start = Instant::now();

        for i in 0..num_transactions {
            let from_idx = i % self.wallets.len();
            let to_idx = (i + 1) % self.wallets.len();

            let from_wallet = self.wallets[from_idx].clone();
            let to_wallet = self.wallets[to_idx].clone();

            let client = self.client.clone();
            let semaphore = self.semaphore.clone();

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();

                let task_start = Instant::now();
                let url = format!("{}/api/transactions", API_BASE);
                let payload = TransactionRequest {
                    from: from_wallet.address,
                    to: to_wallet.address,
                    amount: 0.1,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                };

                let result = client
                    .post(&url)
                    .json(&payload)
                    .send()
                    .await
                    .map(|r| r.status().is_success())
                    .unwrap_or(false);

                (result, task_start.elapsed())
            });

            tasks.push(task);
        }

        let results = futures::future::join_all(tasks).await;
        let elapsed = start.elapsed();

        let mut successful = 0;
        let mut failed = 0;
        let mut latencies = Vec::new();

        for result in results {
            if let Ok((success, latency)) = result {
                if success {
                    successful += 1;
                    latencies.push(latency);
                } else {
                    failed += 1;
                }
            } else {
                failed += 1;
            }
        }

        let tps = successful as f64 / elapsed.as_secs_f64();

        Ok(BenchmarkResults {
            total_transactions: num_transactions,
            successful,
            failed,
            elapsed,
            tps,
            latencies,
        })
    }

    async fn run_benchmark(&mut self, transaction_count: usize) -> anyhow::Result<()> {
        println!("{}", "=".repeat(80));
        println!("🌊 Q-NARWHALKNIGHT REAL-WORLD TPS BENCHMARK (RUST)");
        println!("{}", "=".repeat(80));
        println!("📍 Endpoint: {}", API_BASE);
        println!("🔢 Wallets: {}", self.wallets.len());
        println!("💸 Transactions: {}", transaction_count);
        println!("🔄 Concurrency: {}", MAX_CONCURRENT);
        println!("{}", "=".repeat(80));

        // Health check
        if !self.check_health().await? {
            anyhow::bail!("API health check failed");
        }

        // Fund wallets
        let funded = self.fund_wallets().await?;
        if funded < self.wallets.len() / 2 {
            println!("⚠️  Only {} wallets funded, but continuing...", funded);
        }

        // Execute transactions in waves
        let wave_size = 1000.min(transaction_count);
        let num_waves = (transaction_count + wave_size - 1) / wave_size;

        println!("\n📊 Executing {} waves of up to {} transactions each...", num_waves, wave_size);

        let mut all_results = Vec::new();

        for wave in 0..num_waves {
            let wave_txs = wave_size.min(transaction_count - wave * wave_size);
            println!("\n--- Wave {}/{} ({} transactions) ---", wave + 1, num_waves, wave_txs);

            let result = self.execute_transaction_wave(wave_txs).await?;

            println!("  ✅ Success: {}/{}", result.successful, result.total_transactions);
            println!("  ⚡ TPS: {:.2}", result.tps);
            if !result.latencies.is_empty() {
                let avg_latency = result.latencies.iter().map(|d| d.as_millis()).sum::<u128>() as f64
                    / result.latencies.len() as f64;
                println!("  ⏱️  Avg Latency: {:.2}ms", avg_latency);
            }

            all_results.push(result);

            if wave < num_waves - 1 {
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }

        // Aggregate results
        let total_successful: usize = all_results.iter().map(|r| r.successful).sum();
        let total_failed: usize = all_results.iter().map(|r| r.failed).sum();
        let total_elapsed: Duration = all_results.iter().map(|r| r.elapsed).sum();
        let all_latencies: Vec<Duration> = all_results.iter()
            .flat_map(|r| r.latencies.clone())
            .collect();

        let overall_tps = total_successful as f64 / total_elapsed.as_secs_f64();

        let final_results = BenchmarkResults {
            total_transactions: transaction_count,
            successful: total_successful,
            failed: total_failed,
            elapsed: total_elapsed,
            tps: overall_tps,
            latencies: all_latencies,
        };

        final_results.print_report();

        // Save report
        let report_file = format!("rust_tps_report_{}.json", chrono::Utc::now().timestamp());
        let report_json = serde_json::json!({
            "benchmark_type": "real_world_tps_rust",
            "endpoint": API_BASE,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "configuration": {
                "num_wallets": self.wallets.len(),
                "total_transactions": transaction_count,
                "max_concurrent": MAX_CONCURRENT,
            },
            "results": {
                "total_transactions": transaction_count,
                "successful": total_successful,
                "failed": total_failed,
                "elapsed_seconds": total_elapsed.as_secs_f64(),
                "tps": overall_tps,
            }
        });

        std::fs::write(&report_file, serde_json::to_string_pretty(&report_json)?)?;
        println!("\n📄 Report saved to: {}", report_file);

        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Run progressively larger benchmarks
    let benchmarks = vec![
        (10, 100, "Warmup"),
        (50, 1000, "Small Scale"),
        (100, 5000, "Medium Scale"),
        (200, 10000, "Large Scale"),
        // Uncomment for massive tests:
        // (500, 50000, "Very Large Scale"),
        // (1000, 100000, "Massive Scale"),
    ];

    for (num_wallets, num_transactions, name) in benchmarks {
        println!("\n\n{}", "=".repeat(80));
        println!("🎯 Running {} Benchmark", name);
        println!("{}", "=".repeat(80));

        let mut benchmark = TpsBenchmark::new(num_wallets).await?;
        benchmark.run_benchmark(num_transactions).await?;

        println!("\n⏸️  Pausing 5 seconds before next benchmark...\n");
        tokio::time::sleep(Duration::from_secs(5)).await;
    }

    Ok(())
}
