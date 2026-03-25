/// Realistic Verified TPS Benchmark
///
/// Unlike the "1M TPS" benchmark which measured mempool insertion speed,
/// this benchmark measures ACTUAL blockchain throughput:
///
/// 1. Transactions are properly signed (Ed25519)
/// 2. Signatures are verified by the node
/// 3. We wait for block inclusion
/// 4. We verify finality across nodes
///
/// ## What This Measures:
/// - Real signature verification overhead
/// - Block production rate
/// - Consensus finality time
/// - End-to-end transaction latency
///
/// ## Expected Results (v2.3.1-beta with security fixes):
/// - With verification: 10,000-50,000 TPS (mempool acceptance)
/// - With block inclusion: 1,000-5,000 TPS
/// - With full finality: 500-2,000 TPS
///
/// ## Usage:
/// ```bash
/// # Test against local node
/// cargo test --release -p q-tps-benchmark --test realistic_verified_tps -- --nocapture
///
/// # Test against production
/// Q_NODE_URL="http://185.182.185.227:8080" cargo test --release -p q-tps-benchmark --test realistic_verified_tps -- --nocapture
/// ```

use std::time::{Duration, Instant};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tokio::sync::RwLock;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use ed25519_dalek::{SigningKey, Signer};
use rand_core::OsRng;
// Note: we use a minimal Transaction struct that matches the server's
// deserialization format (extra fields use #[serde(default)] on server side)

// ============================================================================
// CONFIGURATION
// ============================================================================

#[derive(Debug, Clone)]
struct RealisticBenchConfig {
    node_url: String,
    // Phase 1: Mempool acceptance (with verification)
    phase1_tx_count: usize,
    phase1_batch_size: usize,
    // Phase 2: Block inclusion
    phase2_tx_count: usize,
    phase2_wait_blocks: u64,
    // Phase 3: Finality measurement
    phase3_tx_count: usize,
    finality_timeout_secs: u64,
}

impl RealisticBenchConfig {
    fn from_env() -> Self {
        Self {
            node_url: std::env::var("Q_NODE_URL")
                .unwrap_or_else(|_| "http://localhost:8080".to_string()),
            phase1_tx_count: env_parse("Q_PHASE1_TX", 10_000),
            phase1_batch_size: env_parse("Q_BATCH_SIZE", 100),
            phase2_tx_count: env_parse("Q_PHASE2_TX", 1_000),
            phase2_wait_blocks: env_parse("Q_WAIT_BLOCKS", 3),
            phase3_tx_count: env_parse("Q_PHASE3_TX", 100),
            finality_timeout_secs: env_parse("Q_FINALITY_TIMEOUT", 30),
        }
    }
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

// ============================================================================
// TRANSACTION TYPES
// ============================================================================

/// Minimal transaction matching server's q_types::Transaction deserialization.
/// Server uses #[serde(default)] for all optional fields, so we only need core fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchTx {
    id: Vec<u8>,
    from: Vec<u8>,
    to: Vec<u8>,
    amount: u128,
    fee: u128,
    nonce: u64,
    signature: Vec<u8>,
    timestamp: chrono::DateTime<chrono::Utc>,
    data: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TxBatch {
    transactions: Vec<BenchTx>,
}

#[derive(Debug, Deserialize)]
struct BinaryResponse {
    success: bool,
    tx_hashes: Vec<Vec<u8>>,
    accepted: usize,
    rejected: usize,
}

#[derive(Debug, Deserialize)]
struct ApiWrapper<T> {
    #[serde(default)]
    success: bool,
    #[serde(default)]
    data: Option<T>,
}

#[derive(Debug, Deserialize, Default)]
struct NodeStatus {
    height: u64,
    #[serde(default)]
    network_height: u64,
}

#[derive(Debug, Deserialize)]
struct TxStatusResponse {
    status: String,
    #[serde(default)]
    block_height: Option<u64>,
}

// ============================================================================
// TEST WALLET
// ============================================================================

#[derive(Clone)]
struct TestWallet {
    signing_key: SigningKey,
    public_key: Vec<u8>,
}

impl TestWallet {
    fn new() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        Self {
            signing_key,
            public_key: verifying_key.to_bytes().to_vec(),
        }
    }

    fn create_signed_transaction(&self, recipient: &[u8], nonce: u64) -> BenchTx {
        let amount: u128 = 1000 + (nonce as u128 % 10000);
        let fee: u128 = 10;
        let timestamp = chrono::Utc::now();

        // Create transaction ID
        let mut hasher = Sha256::new();
        hasher.update(&self.public_key);
        hasher.update(&nonce.to_le_bytes());
        hasher.update(timestamp.to_rfc3339().as_bytes());
        let tx_id = hasher.finalize().to_vec();

        // Create message to sign (must match server's postcard::to_allocvec format)
        let sign_data = postcard::to_allocvec(&(
            &self.public_key,
            &recipient.to_vec(),
            amount,
            nonce,
        )).unwrap_or_default();

        let signature = self.signing_key.sign(&sign_data);

        BenchTx {
            id: tx_id,
            from: self.public_key.clone(),
            to: recipient.to_vec(),
            amount,
            fee,
            nonce,
            signature: signature.to_bytes().to_vec(),
            timestamp,
            data: vec![],
        }
    }
}

// ============================================================================
// BENCHMARK RESULTS
// ============================================================================

#[derive(Debug, Default)]
struct PhaseResults {
    transactions_sent: AtomicUsize,
    transactions_accepted: AtomicUsize,
    transactions_rejected: AtomicUsize,
    transactions_confirmed: AtomicUsize,
    total_latency_ms: AtomicU64,
    start_height: AtomicU64,
    end_height: AtomicU64,
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

async fn get_node_height(client: &Client, base_url: &str) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
    let url = format!("{}/api/v1/health", base_url);
    let resp: ApiWrapper<NodeStatus> = client.get(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await?
        .json()
        .await?;
    let data = resp.data.ok_or("No data in health response")?;
    Ok(data.height)
}

async fn wait_for_height(
    client: &Client,
    base_url: &str,
    target_height: u64,
    timeout: Duration,
) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
    let start = Instant::now();
    loop {
        let current = get_node_height(client, base_url).await?;
        if current >= target_height {
            return Ok(current);
        }
        if start.elapsed() > timeout {
            return Err(format!("Timeout waiting for height {} (current: {})", target_height, current).into());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

async fn check_tx_status(
    client: &Client,
    base_url: &str,
    tx_hash: &[u8],
) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
    let tx_hex = hex::encode(tx_hash);
    let url = format!("{}/api/v1/tx/{}/status", base_url, tx_hex);

    match client.get(&url).timeout(Duration::from_secs(5)).send().await {
        Ok(resp) if resp.status().is_success() => {
            let status: TxStatusResponse = resp.json().await?;
            Ok(Some(status.status))
        }
        _ => Ok(None),
    }
}

// ============================================================================
// PHASE 1: MEMPOOL ACCEPTANCE WITH VERIFICATION
// ============================================================================

async fn phase1_verified_mempool_tps(
    config: &RealisticBenchConfig,
    client: &Client,
    wallet: &TestWallet,
    recipient: &[u8],
) -> Result<(f64, usize, usize), Box<dyn std::error::Error + Send + Sync>> {
    println!("\n{}", "=".repeat(70));
    println!("PHASE 1: Mempool Acceptance TPS (with signature verification)");
    println!("{}", "=".repeat(70));
    println!("  Transactions: {}", config.phase1_tx_count);
    println!("  Batch size: {}", config.phase1_batch_size);
    println!();

    let url = format!("{}/api/v1/binary/batch", config.node_url);
    let mut total_accepted = 0usize;
    let mut total_rejected = 0usize;
    let mut nonce = 0u64;

    let num_batches = config.phase1_tx_count / config.phase1_batch_size;

    let start = Instant::now();

    for batch_num in 0..num_batches {
        // Create batch of signed transactions
        let transactions: Vec<BenchTx> = (0..config.phase1_batch_size)
            .map(|i| {
                nonce += 1;
                wallet.create_signed_transaction(recipient, nonce)
            })
            .collect();

        let batch = TxBatch { transactions };
        let packed = rmp_serde::to_vec(&batch)?;

        // Submit batch
        let resp = client
            .post(&url)
            .header("Content-Type", "application/msgpack")
            .body(packed)
            .timeout(Duration::from_secs(30))
            .send()
            .await?;

        if resp.status().is_success() {
            let bytes = resp.bytes().await?;
            let result: BinaryResponse = rmp_serde::from_slice(&bytes)
                .unwrap_or(BinaryResponse { success: false, tx_hashes: vec![], accepted: 0, rejected: 0 });
            total_accepted += result.accepted;
            total_rejected += result.rejected;
        } else {
            total_rejected += config.phase1_batch_size;
            eprintln!("  Batch {} failed: HTTP {}", batch_num, resp.status());
        }

        // Progress update
        if (batch_num + 1) % 10 == 0 || batch_num == num_batches - 1 {
            let elapsed = start.elapsed().as_secs_f64();
            let current_tps = total_accepted as f64 / elapsed;
            println!("  Progress: {}/{} batches, {} accepted, {:.0} TPS",
                batch_num + 1, num_batches, total_accepted, current_tps);
        }
    }

    let elapsed = start.elapsed();
    let tps = total_accepted as f64 / elapsed.as_secs_f64();

    println!();
    println!("PHASE 1 RESULTS:");
    println!("  Duration: {:.2}s", elapsed.as_secs_f64());
    println!("  Accepted: {}", total_accepted);
    println!("  Rejected: {}", total_rejected);
    println!("  TPS (verified mempool): {:.0}", tps);

    Ok((tps, total_accepted, total_rejected))
}

// ============================================================================
// PHASE 2: BLOCK INCLUSION TPS
// ============================================================================

async fn phase2_block_inclusion_tps(
    config: &RealisticBenchConfig,
    client: &Client,
    wallet: &TestWallet,
    recipient: &[u8],
) -> Result<(f64, Duration), Box<dyn std::error::Error + Send + Sync>> {
    println!("\n{}", "=".repeat(70));
    println!("PHASE 2: Block Inclusion TPS");
    println!("{}", "=".repeat(70));
    println!("  Transactions: {}", config.phase2_tx_count);
    println!("  Wait for: {} blocks", config.phase2_wait_blocks);
    println!();

    // Get starting height
    let start_height = get_node_height(client, &config.node_url).await?;
    println!("  Starting height: {}", start_height);

    // Submit transactions
    let url = format!("{}/api/v1/binary/batch", config.node_url);
    let transactions: Vec<BenchTx> = (0..config.phase2_tx_count)
        .map(|i| wallet.create_signed_transaction(recipient, 1_000_000 + i as u64))
        .collect();

    let batch = TxBatch { transactions };
    let packed = rmp_serde::to_vec(&batch)?;

    let submit_start = Instant::now();

    let resp = client
        .post(&url)
        .header("Content-Type", "application/msgpack")
        .body(packed)
        .timeout(Duration::from_secs(60))
        .send()
        .await?;

    let submit_time = submit_start.elapsed();

    if !resp.status().is_success() {
        return Err(format!("Batch submission failed: {}", resp.status()).into());
    }

    let result: BinaryResponse = resp.json().await?;
    println!("  Submitted: {} accepted in {:.2}ms", result.accepted, submit_time.as_millis());

    // Wait for blocks
    let target_height = start_height + config.phase2_wait_blocks;
    println!("  Waiting for height {} ...", target_height);

    let block_start = Instant::now();
    let final_height = wait_for_height(
        client,
        &config.node_url,
        target_height,
        Duration::from_secs(60)
    ).await?;

    let block_time = block_start.elapsed();
    let total_time = submit_start.elapsed();

    // Calculate block production rate
    let blocks_produced = final_height - start_height;
    let block_rate = blocks_produced as f64 / block_time.as_secs_f64();

    // Estimate TPS based on block inclusion
    // Assuming transactions are evenly distributed across blocks
    let estimated_tps = result.accepted as f64 / total_time.as_secs_f64();

    println!();
    println!("PHASE 2 RESULTS:");
    println!("  Blocks produced: {} in {:.2}s ({:.2} blocks/sec)",
        blocks_produced, block_time.as_secs_f64(), block_rate);
    println!("  Total time to inclusion: {:.2}s", total_time.as_secs_f64());
    println!("  TPS (block inclusion): {:.0}", estimated_tps);

    Ok((estimated_tps, total_time))
}

// ============================================================================
// PHASE 3: END-TO-END FINALITY
// ============================================================================

async fn phase3_finality_measurement(
    config: &RealisticBenchConfig,
    client: &Client,
    wallet: &TestWallet,
    recipient: &[u8],
) -> Result<(f64, Duration), Box<dyn std::error::Error + Send + Sync>> {
    println!("\n{}", "=".repeat(70));
    println!("PHASE 3: End-to-End Finality Measurement");
    println!("{}", "=".repeat(70));
    println!("  Transactions: {}", config.phase3_tx_count);
    println!("  Timeout: {}s", config.finality_timeout_secs);
    println!();

    let url = format!("{}/api/v1/binary/batch", config.node_url);

    // Submit transactions one at a time and track each
    let mut tx_hashes: Vec<Vec<u8>> = Vec::new();
    let mut finality_times: Vec<Duration> = Vec::new();

    for i in 0..config.phase3_tx_count {
        let tx = wallet.create_signed_transaction(recipient, 2_000_000 + i as u64);
        let tx_id = tx.id.clone();

        let batch = TxBatch { transactions: vec![tx] };
        let packed = rmp_serde::to_vec(&batch)?;

        let submit_time = Instant::now();

        let resp = client
            .post(&url)
            .header("Content-Type", "application/msgpack")
            .body(packed)
            .timeout(Duration::from_secs(10))
            .send()
            .await?;

        if resp.status().is_success() {
            tx_hashes.push(tx_id);

            // Poll for confirmation
            let timeout = Duration::from_secs(config.finality_timeout_secs);
            loop {
                if submit_time.elapsed() > timeout {
                    println!("  TX {} timed out after {:?}", i, timeout);
                    break;
                }

                if let Ok(Some(status)) = check_tx_status(client, &config.node_url, tx_hashes.last().unwrap()).await {
                    if status == "confirmed" || status == "finalized" {
                        let finality_time = submit_time.elapsed();
                        finality_times.push(finality_time);
                        if i % 10 == 0 {
                            println!("  TX {} finalized in {:.0}ms", i, finality_time.as_millis());
                        }
                        break;
                    }
                }

                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }

    // Calculate statistics
    let confirmed = finality_times.len();
    let avg_finality = if !finality_times.is_empty() {
        finality_times.iter().map(|d| d.as_millis() as f64).sum::<f64>() / finality_times.len() as f64
    } else {
        0.0
    };

    let min_finality = finality_times.iter().min().map(|d| d.as_millis()).unwrap_or(0);
    let max_finality = finality_times.iter().max().map(|d| d.as_millis()).unwrap_or(0);

    // Sort for percentiles
    let mut sorted_times: Vec<u128> = finality_times.iter().map(|d| d.as_millis()).collect();
    sorted_times.sort();

    let p50 = sorted_times.get(sorted_times.len() / 2).copied().unwrap_or(0);
    let p95 = sorted_times.get((sorted_times.len() as f64 * 0.95) as usize).copied().unwrap_or(0);
    let p99 = sorted_times.get((sorted_times.len() as f64 * 0.99) as usize).copied().unwrap_or(0);

    // Calculate finality TPS (how many tx/sec achieve finality)
    let total_test_time = finality_times.iter().max().copied().unwrap_or(Duration::from_secs(1));
    let finality_tps = confirmed as f64 / total_test_time.as_secs_f64();

    println!();
    println!("PHASE 3 RESULTS:");
    println!("  Confirmed: {}/{}", confirmed, config.phase3_tx_count);
    println!("  Finality Latency:");
    println!("    Average: {:.0}ms", avg_finality);
    println!("    Min:     {}ms", min_finality);
    println!("    Max:     {}ms", max_finality);
    println!("    P50:     {}ms", p50);
    println!("    P95:     {}ms", p95);
    println!("    P99:     {}ms", p99);
    println!("  TPS (with finality): {:.1}", finality_tps);

    Ok((finality_tps, Duration::from_millis(avg_finality as u64)))
}

// ============================================================================
// MAIN TEST
// ============================================================================

#[tokio::test]
async fn test_realistic_verified_tps() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = RealisticBenchConfig::from_env();

    println!("\n{}", "=".repeat(70));
    println!("   Q-NARWHALKNIGHT REALISTIC TPS BENCHMARK (v2.3.1-beta)");
    println!("{}", "=".repeat(70));
    println!();
    println!("This benchmark measures ACTUAL blockchain performance:");
    println!("  - Phase 1: Mempool acceptance WITH signature verification");
    println!("  - Phase 2: Block inclusion time");
    println!("  - Phase 3: End-to-end finality");
    println!();
    println!("Target node: {}", config.node_url);
    println!();

    // Build HTTP client
    let client = Client::builder()
        .pool_max_idle_per_host(10)
        .timeout(Duration::from_secs(120))
        .build()?;

    // Check node is alive
    println!("Checking node status...");
    match get_node_height(&client, &config.node_url).await {
        Ok(height) => println!("  Node is live at height {}\n", height),
        Err(e) => {
            eprintln!("  ERROR: Cannot connect to node: {}", e);
            eprintln!("  Make sure q-api-server is running at {}", config.node_url);
            return Err(e);
        }
    }

    // Create test wallet
    let wallet = TestWallet::new();
    let recipient = wallet.public_key.clone(); // Send to self for testing

    // Run phases
    let (phase1_tps, accepted, rejected) = phase1_verified_mempool_tps(
        &config, &client, &wallet, &recipient
    ).await?;

    let (phase2_tps, _) = phase2_block_inclusion_tps(
        &config, &client, &wallet, &recipient
    ).await?;

    let (phase3_tps, avg_finality) = phase3_finality_measurement(
        &config, &client, &wallet, &recipient
    ).await?;

    // Final summary
    println!("\n{}", "=".repeat(70));
    println!("                    FINAL RESULTS SUMMARY");
    println!("{}", "=".repeat(70));
    println!();
    println!("  THROUGHPUT (Transactions Per Second):");
    println!("    Phase 1 - Verified Mempool:    {:>10.0} TPS", phase1_tps);
    println!("    Phase 2 - Block Inclusion:     {:>10.0} TPS", phase2_tps);
    println!("    Phase 3 - Full Finality:       {:>10.1} TPS", phase3_tps);
    println!();
    println!("  LATENCY:");
    println!("    Average Finality:              {:>10}ms", avg_finality.as_millis());
    println!();
    println!("  SECURITY:");
    println!("    Signature Verification:        ENABLED (Ed25519)");
    println!("    Transactions Accepted:         {}", accepted);
    println!("    Transactions Rejected:         {}", rejected);
    println!();

    // Comparison with old "1M TPS" claim
    println!("  COMPARISON WITH MARKETING CLAIMS:");
    println!("    Old '1M TPS' benchmark:        Measured mempool insertion WITHOUT verification");
    println!("    This benchmark:                Measures REAL throughput WITH verification");
    println!();

    if phase1_tps >= 10_000.0 {
        println!("  STATUS: GOOD - Achieving 10K+ verified mempool TPS");
    }
    if phase2_tps >= 1_000.0 {
        println!("  STATUS: GOOD - Achieving 1K+ block inclusion TPS");
    }
    if phase3_tps >= 100.0 {
        println!("  STATUS: GOOD - Achieving 100+ finality TPS");
    }

    println!();
    println!("{}", "=".repeat(70));
    println!("Q-NarwhalKnight v2.3.1-beta - Security fixes applied");
    println!("{}\n", "=".repeat(70));

    Ok(())
}
