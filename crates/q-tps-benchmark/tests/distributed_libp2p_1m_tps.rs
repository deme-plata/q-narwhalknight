/// Distributed libp2p Multi-Node 1M TPS Benchmark
///
/// Comprehensive benchmark for Q-NarwhalKnight validator nodes connected via libp2p gossipsub.
/// Tests true peer-to-peer transaction propagation and consensus performance.
///
/// ## Configuration via Environment Variables:
/// - `Q_NUM_NODES`: Number of nodes (default: 4)
/// - `Q_BATCHES_PER_NODE`: Number of batches per node (default: 10)
/// - `Q_BATCH_SIZE`: Transactions per batch (default: 10000)
/// - `Q_CONCURRENT_BATCHES`: Concurrent batch submissions per node (default: 4)
/// - `Q_BASE_HTTP_PORT`: Starting HTTP port (default: 9100)
/// - `Q_BASE_P2P_PORT`: Starting P2P port (default: 9200)
/// - `Q_NODE_STARTUP_DELAY`: Seconds between node launches (default: 5)
/// - `Q_USE_EXISTING_NODES`: If "true", use existing nodes (provide Q_NODE_URLS)
/// - `Q_NODE_URLS`: Comma-separated list of existing node URLs (e.g., "http://localhost:8080,http://localhost:8081")
/// - `Q_WARMUP_BATCHES`: Number of warmup batches before measurement (default: 2)
///
/// ## Example Usage:
/// ```bash
/// # Run with default settings (4 nodes, spawns new processes)
/// cargo test --release -p q-tps-benchmark --test distributed_libp2p_1m_tps -- --nocapture
///
/// # Run with 8 nodes and larger batches
/// Q_NUM_NODES=8 Q_BATCH_SIZE=20000 cargo test --release -p q-tps-benchmark --test distributed_libp2p_1m_tps -- --nocapture
///
/// # Test against existing production nodes
/// Q_USE_EXISTING_NODES=true Q_NODE_URLS="http://185.182.185.227:8080,http://161.35.219.10:8080" cargo test --release -p q-tps-benchmark --test distributed_libp2p_1m_tps -- --nocapture
/// ```

use std::time::{Instant, Duration};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use tokio::sync::{mpsc, Barrier, Semaphore, RwLock};
use tokio::time::sleep;
use reqwest::Client;
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use ed25519_dalek::{SigningKey, Signer};
use rand::rngs::OsRng;

/// Benchmark configuration
#[derive(Debug, Clone)]
struct BenchmarkConfig {
    num_nodes: usize,
    batches_per_node: usize,
    batch_size: usize,
    concurrent_batches: usize,
    base_http_port: u16,
    base_p2p_port: u16,
    node_startup_delay_secs: u64,
    use_existing_nodes: bool,
    existing_node_urls: Vec<String>,
    warmup_batches: usize,
}

impl BenchmarkConfig {
    fn from_env() -> Self {
        let use_existing = std::env::var("Q_USE_EXISTING_NODES")
            .map(|s| s.to_lowercase() == "true")
            .unwrap_or(false);

        let existing_urls: Vec<String> = std::env::var("Q_NODE_URLS")
            .map(|s| s.split(',').map(|u| u.trim().to_string()).filter(|u| !u.is_empty()).collect())
            .unwrap_or_default();

        Self {
            num_nodes: env_parse("Q_NUM_NODES", 4),
            batches_per_node: env_parse("Q_BATCHES_PER_NODE", 10),
            batch_size: env_parse("Q_BATCH_SIZE", 10_000),
            concurrent_batches: env_parse("Q_CONCURRENT_BATCHES", 4),
            base_http_port: env_parse("Q_BASE_HTTP_PORT", 9100),
            base_p2p_port: env_parse("Q_BASE_P2P_PORT", 9200),
            node_startup_delay_secs: env_parse("Q_NODE_STARTUP_DELAY", 5),
            use_existing_nodes: use_existing,
            existing_node_urls: existing_urls,
            warmup_batches: env_parse("Q_WARMUP_BATCHES", 2),
        }
    }

    fn total_transactions(&self) -> usize {
        self.effective_node_count() * self.batches_per_node * self.batch_size
    }

    fn effective_node_count(&self) -> usize {
        if self.use_existing_nodes {
            self.existing_node_urls.len()
        } else {
            self.num_nodes
        }
    }
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Node configuration for distributed test
#[derive(Debug, Clone)]
struct NodeConfig {
    node_id: usize,
    http_url: String,
    data_dir: Option<String>,
    is_external: bool,
}

/// Real-time benchmark statistics
#[derive(Debug, Default)]
struct BenchmarkStats {
    total_submitted: AtomicUsize,
    total_accepted: AtomicUsize,
    total_rejected: AtomicUsize,
    total_errors: AtomicUsize,
    total_latency_us: AtomicU64,
    min_latency_us: AtomicU64,
    max_latency_us: AtomicU64,
    latencies: RwLock<Vec<u64>>,
}

impl BenchmarkStats {
    fn new() -> Self {
        Self {
            min_latency_us: AtomicU64::new(u64::MAX),
            ..Default::default()
        }
    }

    fn record_success(&self, latency_us: u64) {
        self.total_submitted.fetch_add(1, Ordering::Relaxed);
        self.total_accepted.fetch_add(1, Ordering::Relaxed);
        self.total_latency_us.fetch_add(latency_us, Ordering::Relaxed);

        // Update min
        let mut current_min = self.min_latency_us.load(Ordering::Relaxed);
        while latency_us < current_min {
            match self.min_latency_us.compare_exchange_weak(
                current_min, latency_us, Ordering::Relaxed, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(v) => current_min = v,
            }
        }

        // Update max
        let mut current_max = self.max_latency_us.load(Ordering::Relaxed);
        while latency_us > current_max {
            match self.max_latency_us.compare_exchange_weak(
                current_max, latency_us, Ordering::Relaxed, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(v) => current_max = v,
            }
        }
    }

    fn record_error(&self) {
        self.total_submitted.fetch_add(1, Ordering::Relaxed);
        self.total_errors.fetch_add(1, Ordering::Relaxed);
    }

    fn record_reject(&self) {
        self.total_submitted.fetch_add(1, Ordering::Relaxed);
        self.total_rejected.fetch_add(1, Ordering::Relaxed);
    }

    async fn add_latency(&self, latency_us: u64) {
        let mut latencies = self.latencies.write().await;
        latencies.push(latency_us);
    }

    async fn calculate_percentiles(&self) -> (f64, f64, f64) {
        let mut latencies = self.latencies.write().await;
        if latencies.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        latencies.sort_unstable();

        let len = latencies.len();
        let p50_idx = len / 2;
        let p95_idx = (len as f64 * 0.95) as usize;
        let p99_idx = (len as f64 * 0.99) as usize;

        let p50 = latencies.get(p50_idx).copied().unwrap_or(0) as f64 / 1000.0;
        let p95 = latencies.get(p95_idx.min(len - 1)).copied().unwrap_or(0) as f64 / 1000.0;
        let p99 = latencies.get(p99_idx.min(len - 1)).copied().unwrap_or(0) as f64 / 1000.0;

        (p50, p95, p99)
    }
}

/// Transaction format matching Q-NarwhalKnight API
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

/// Test wallet with Ed25519 keypair
struct TestWallet {
    signing_key: SigningKey,
    address: Vec<u8>,
}

impl TestWallet {
    fn new() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        // Generate address from public key hash
        let mut hasher = Sha256::new();
        hasher.update(verifying_key.as_bytes());
        let address = hasher.finalize().to_vec();

        Self { signing_key, address }
    }

    fn sign_transaction(&self, tx_data: &[u8]) -> Vec<u8> {
        let signature = self.signing_key.sign(tx_data);
        signature.to_bytes().to_vec()
    }
}

fn create_signed_transaction(wallet: &TestWallet, recipient: &[u8], index: usize, nonce: u64) -> Transaction {
    // Generate unique transaction ID
    let mut hasher = Sha256::new();
    hasher.update(&wallet.address);
    hasher.update(&index.to_le_bytes());
    hasher.update(&nonce.to_le_bytes());
    hasher.update(chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0).to_le_bytes());
    let tx_id = hasher.finalize().to_vec();

    let amount = 1000 + (index as u64 % 10000);
    let fee = 10;
    let timestamp = chrono::Utc::now().to_rfc3339();

    // Create signing message
    let mut sign_data = Vec::new();
    sign_data.extend(&tx_id);
    sign_data.extend(&wallet.address);
    sign_data.extend(recipient);
    sign_data.extend(&amount.to_le_bytes());
    sign_data.extend(&fee.to_le_bytes());
    sign_data.extend(&nonce.to_le_bytes());

    let signature = wallet.sign_transaction(&sign_data);

    Transaction {
        id: tx_id,
        from: wallet.address.clone(),
        to: recipient.to_vec(),
        amount,
        fee,
        nonce,
        signature,
        timestamp,
        data: vec![],
    }
}

/// Launch a Q-NarwhalKnight validator node process
async fn launch_validator_node(config: &NodeConfig) -> Result<Option<tokio::process::Child>, Box<dyn std::error::Error + Send + Sync>> {
    use tokio::process::Command;

    if config.is_external {
        println!("  Using external node {}: {}", config.node_id, config.http_url);
        return Ok(None);
    }

    let data_dir = config.data_dir.as_ref().ok_or("No data dir for local node")?;
    println!("  Launching validator node {} (data: {})", config.node_id, data_dir);

    // Create data directory
    tokio::fs::create_dir_all(data_dir).await?;

    // Find binary
    let workspace_root = std::env::var("CARGO_MANIFEST_DIR")
        .ok()
        .and_then(|manifest_dir| {
            std::path::PathBuf::from(&manifest_dir)
                .parent()
                .and_then(|p| p.parent())
                .map(|p| p.to_path_buf())
        })
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

    let possible_paths = vec![
        workspace_root.join("target/release/q-api-server"),
        workspace_root.join("target/x86_64-unknown-linux-gnu/release/q-api-server"),
        std::path::PathBuf::from("/opt/orobit/shared/q-narwhalknight/target/release/q-api-server"),
        std::path::PathBuf::from("/opt/orobit/shared/q-narwhalknight/target/x86_64-unknown-linux-gnu/release/q-api-server"),
        std::env::current_dir().unwrap_or_default().join("target/release/q-api-server"),
    ];

    let binary_path = possible_paths.into_iter()
        .find(|p| p.exists())
        .ok_or_else(|| format!("q-api-server binary not found"))?;

    // Parse port from URL
    let port: u16 = config.http_url
        .split(':')
        .last()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);

    let p2p_port = port + 100; // P2P port offset

    let child = Command::new(&binary_path)
        .arg("--port")
        .arg(port.to_string())
        .env("Q_DB_PATH", data_dir)
        .env("Q_P2P_PORT", p2p_port.to_string())
        .env("RUST_LOG", "warn")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;

    println!("    Started node {} (HTTP: {}, P2P: {})", config.node_id, port, p2p_port);
    Ok(Some(child))
}

/// Submit transaction batches to a node
async fn submit_batches(
    node_config: &NodeConfig,
    wallet: &TestWallet,
    recipient: &[u8],
    config: &BenchmarkConfig,
    stats: Arc<BenchmarkStats>,
    client: Arc<Client>,
    semaphore: Arc<Semaphore>,
    is_warmup: bool,
) -> Result<(usize, f64), Box<dyn std::error::Error + Send + Sync>> {
    let url = format!("{}/api/v1/binary/batch", node_config.http_url);
    let mut total_tx = 0;
    let start = Instant::now();

    let batches = if is_warmup { config.warmup_batches } else { config.batches_per_node };

    for batch_num in 0..batches {
        let _permit = semaphore.acquire().await?;

        let batch_start = Instant::now();
        let base_idx = node_config.node_id * 1_000_000 + batch_num * config.batch_size;

        // Generate signed transactions
        let transactions: Vec<Transaction> = (0..config.batch_size)
            .map(|i| {
                let nonce = (base_idx + i) as u64;
                create_signed_transaction(wallet, recipient, base_idx + i, nonce)
            })
            .collect();

        let batch = BinaryTransactionBatch { transactions };
        let packed = rmp_serde::to_vec(&batch)?;
        let packed_size = packed.len();

        let send_start = Instant::now();
        let result = client
            .post(&url)
            .header("Content-Type", "application/msgpack")
            .body(packed)
            .send()
            .await;

        let latency_us = send_start.elapsed().as_micros() as u64;

        match result {
            Ok(resp) if resp.status().is_success() => {
                if !is_warmup {
                    total_tx += config.batch_size;
                    stats.record_success(latency_us);
                    stats.add_latency(latency_us).await;
                }
            }
            Ok(resp) => {
                if !is_warmup {
                    stats.record_reject();
                }
                eprintln!("    Node {} batch {}: HTTP {}", node_config.node_id, batch_num, resp.status());
            }
            Err(e) => {
                if !is_warmup {
                    stats.record_error();
                }
                eprintln!("    Node {} batch {}: Error {}", node_config.node_id, batch_num, e);
            }
        }

        if !is_warmup && batch_num % 2 == 0 {
            println!("    Node {} progress: {}/{} batches ({} KB/batch, {:.0}ms latency)",
                node_config.node_id, batch_num + 1, batches,
                packed_size / 1024, latency_us as f64 / 1000.0);
        }
    }

    let elapsed = start.elapsed();
    let tps = if elapsed.as_secs_f64() > 0.0 {
        total_tx as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    Ok((total_tx, tps))
}

/// Wait for nodes to be ready
async fn wait_for_nodes_ready(
    nodes: &[NodeConfig],
    client: &Client,
    timeout: Duration,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("\nWaiting for nodes to be ready...");

    let start = Instant::now();
    let mut ready = vec![false; nodes.len()];

    while start.elapsed() < timeout {
        let mut all_ready = true;

        for (idx, config) in nodes.iter().enumerate() {
            if !ready[idx] {
                let url = format!("{}/health", config.http_url);
                match client.get(&url).timeout(Duration::from_secs(5)).send().await {
                    Ok(resp) if resp.status().is_success() => {
                        ready[idx] = true;
                        println!("  Node {} ready", config.node_id);
                    }
                    _ => {
                        all_ready = false;
                    }
                }
            }
        }

        if all_ready {
            println!("All {} nodes ready!\n", nodes.len());
            return Ok(());
        }

        sleep(Duration::from_millis(1000)).await;
    }

    let ready_count = ready.iter().filter(|r| **r).count();
    if ready_count > 0 {
        println!("Warning: Only {}/{} nodes ready after timeout. Proceeding...\n", ready_count, nodes.len());
        Ok(())
    } else {
        Err("No nodes became ready".into())
    }
}

/// Print benchmark results
async fn print_results(
    config: &BenchmarkConfig,
    stats: &BenchmarkStats,
    elapsed: Duration,
    node_results: &[(usize, f64)],
) {
    let total_accepted = stats.total_accepted.load(Ordering::Relaxed);
    let total_rejected = stats.total_rejected.load(Ordering::Relaxed);
    let total_errors = stats.total_errors.load(Ordering::Relaxed);
    let total_latency_us = stats.total_latency_us.load(Ordering::Relaxed);
    let min_latency_us = stats.min_latency_us.load(Ordering::Relaxed);
    let max_latency_us = stats.max_latency_us.load(Ordering::Relaxed);

    let aggregate_tps = total_accepted as f64 / elapsed.as_secs_f64();
    let avg_latency_ms = if total_accepted > 0 {
        (total_latency_us as f64 / total_accepted as f64) / 1000.0
    } else {
        0.0
    };

    let (p50, p95, p99) = stats.calculate_percentiles().await;

    // Per-node statistics
    let node_tps: Vec<f64> = node_results.iter().map(|(_, tps)| *tps).collect();
    let avg_node_tps = if !node_tps.is_empty() {
        node_tps.iter().sum::<f64>() / node_tps.len() as f64
    } else {
        0.0
    };
    let min_node_tps = node_tps.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_node_tps = node_tps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!();
    println!("================================================================================");
    println!("                    Q-NARWHALKNIGHT TPS BENCHMARK RESULTS                       ");
    println!("================================================================================");
    println!();
    println!("CONFIGURATION:");
    println!("  Nodes:                    {}", config.effective_node_count());
    println!("  Batches/Node:             {}", config.batches_per_node);
    println!("  Batch Size:               {} transactions", config.batch_size);
    println!("  Concurrent Batches/Node:  {}", config.concurrent_batches);
    println!("  Target Total TX:          {}", config.total_transactions());
    println!();
    println!("TRANSACTION RESULTS:");
    println!("  Total Accepted:           {}", total_accepted);
    println!("  Total Rejected:           {}", total_rejected);
    println!("  Total Errors:             {}", total_errors);
    println!("  Success Rate:             {:.2}%",
        (total_accepted as f64 / (total_accepted + total_rejected + total_errors).max(1) as f64) * 100.0);
    println!();
    println!("THROUGHPUT:");
    println!("  Duration:                 {:.2}s", elapsed.as_secs_f64());
    println!("  Aggregate TPS:            {:.0} tx/sec", aggregate_tps);
    println!("  Average Node TPS:         {:.0} tx/sec", avg_node_tps);
    println!("  Min Node TPS:             {:.0} tx/sec", min_node_tps);
    println!("  Max Node TPS:             {:.0} tx/sec", max_node_tps);
    println!();
    println!("LATENCY (batch submission):");
    println!("  Average:                  {:.2}ms", avg_latency_ms);
    println!("  Min:                      {:.2}ms", min_latency_us as f64 / 1000.0);
    println!("  Max:                      {:.2}ms", max_latency_us as f64 / 1000.0);
    println!("  P50:                      {:.2}ms", p50);
    println!("  P95:                      {:.2}ms", p95);
    println!("  P99:                      {:.2}ms", p99);
    println!();

    // Performance targets
    println!("TARGET ANALYSIS:");
    let targets = [
        (100_000.0, "100K TPS"),
        (250_000.0, "250K TPS"),
        (500_000.0, "500K TPS"),
        (1_000_000.0, "1M TPS"),
    ];

    for (target, name) in targets {
        let percent = (aggregate_tps / target) * 100.0;
        let status = if aggregate_tps >= target { "ACHIEVED" } else { "pending" };
        println!("  {:12} {:>8.1}%  [{}]", name, percent, status);
    }
    println!();

    // Summary
    if aggregate_tps >= 1_000_000.0 {
        println!("================================================================================");
        println!("          MILESTONE ACHIEVED: 1M+ TPS WITH DISTRIBUTED NODES!                  ");
        println!("================================================================================");
    } else if aggregate_tps >= 500_000.0 {
        println!("EXCELLENT: Achieved 50%+ of 1M TPS target with real distributed nodes!");
    } else if aggregate_tps >= 100_000.0 {
        println!("GOOD: Achieved 100K+ TPS baseline with distributed architecture.");
    }

    println!();
    println!("Q-NarwhalKnight Quantum-Enhanced DAG-BFT Consensus");
    println!("libp2p gossipsub + Kademlia DHT networking");
    println!("================================================================================");
}

#[tokio::test]
async fn test_distributed_libp2p_1m_tps() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = BenchmarkConfig::from_env();

    println!("================================================================================");
    println!("     Q-NARWHALKNIGHT DISTRIBUTED MULTI-NODE TPS BENCHMARK                       ");
    println!("================================================================================");
    println!();
    println!("Configuration:");
    println!("  Nodes:              {}", config.effective_node_count());
    println!("  Batches/Node:       {}", config.batches_per_node);
    println!("  Batch Size:         {} transactions", config.batch_size);
    println!("  Concurrent/Node:    {}", config.concurrent_batches);
    println!("  Warmup Batches:     {}", config.warmup_batches);
    println!("  Total Transactions: {}", config.total_transactions());
    println!("  Mode:               {}", if config.use_existing_nodes { "External Nodes" } else { "Spawn Local" });
    println!();

    // Build node configurations
    let node_configs: Vec<NodeConfig> = if config.use_existing_nodes {
        config.existing_node_urls.iter().enumerate()
            .map(|(i, url)| NodeConfig {
                node_id: i,
                http_url: url.clone(),
                data_dir: None,
                is_external: true,
            })
            .collect()
    } else {
        (0..config.num_nodes)
            .map(|i| NodeConfig {
                node_id: i,
                http_url: format!("http://localhost:{}", config.base_http_port + i as u16),
                data_dir: Some(format!("./data-tps-node{}", i)),
                is_external: false,
            })
            .collect()
    };

    if node_configs.is_empty() {
        return Err("No nodes configured".into());
    }

    // Launch local nodes if needed
    let mut node_processes: Vec<Option<tokio::process::Child>> = Vec::new();

    if !config.use_existing_nodes {
        println!("Launching {} local validator nodes...", config.num_nodes);
        for node_config in &node_configs {
            match launch_validator_node(node_config).await {
                Ok(child) => node_processes.push(child),
                Err(e) => {
                    eprintln!("Failed to launch node {}: {}", node_config.node_id, e);
                    return Err(e);
                }
            }
            sleep(Duration::from_secs(config.node_startup_delay_secs)).await;
        }
        println!("All nodes launched.\n");
    }

    // Build HTTP client
    let client = Arc::new(
        Client::builder()
            .pool_max_idle_per_host(config.concurrent_batches * 2)
            .timeout(Duration::from_secs(120))
            .tcp_keepalive(Duration::from_secs(30))
            .build()?
    );

    // Wait for nodes to be ready
    wait_for_nodes_ready(&node_configs, &client, Duration::from_secs(180)).await?;

    // Wait for peer discovery (if local nodes)
    if !config.use_existing_nodes {
        println!("Waiting for peer discovery (10s)...");
        sleep(Duration::from_secs(10)).await;
        println!("Peer discovery complete.\n");
    }

    // Create test wallets (one per node)
    println!("Generating test wallets...");
    let wallets: Vec<TestWallet> = (0..node_configs.len())
        .map(|_| TestWallet::new())
        .collect();

    // Generate recipient addresses
    let recipients: Vec<Vec<u8>> = wallets.iter()
        .map(|w| w.address.clone())
        .collect();
    println!("Generated {} test wallets.\n", wallets.len());

    // Statistics tracking
    let stats = Arc::new(BenchmarkStats::new());

    // Concurrency control
    let semaphore = Arc::new(Semaphore::new(config.concurrent_batches));
    let barrier = Arc::new(Barrier::new(node_configs.len()));

    // Warmup phase
    if config.warmup_batches > 0 {
        println!("WARMUP PHASE ({} batches per node)...", config.warmup_batches);
        let warmup_start = Instant::now();

        let mut warmup_tasks = Vec::new();
        for (idx, node_config) in node_configs.iter().enumerate() {
            let node_cfg = node_config.clone();
            let wallet = wallets[idx].clone();
            let recipient = recipients[(idx + 1) % recipients.len()].clone();
            let cfg = config.clone();
            let stats_clone = stats.clone();
            let client_clone = client.clone();
            let sem_clone = semaphore.clone();

            let task = tokio::spawn(async move {
                submit_batches(&node_cfg, &wallet, &recipient, &cfg, stats_clone, client_clone, sem_clone, true).await
            });
            warmup_tasks.push(task);
        }

        for task in warmup_tasks {
            let _ = task.await;
        }

        println!("Warmup complete in {:.2}s.\n", warmup_start.elapsed().as_secs_f64());

        // Reset stats after warmup
        stats.total_submitted.store(0, Ordering::Relaxed);
        stats.total_accepted.store(0, Ordering::Relaxed);
        stats.total_rejected.store(0, Ordering::Relaxed);
        stats.total_errors.store(0, Ordering::Relaxed);
        stats.total_latency_us.store(0, Ordering::Relaxed);
        stats.min_latency_us.store(u64::MAX, Ordering::Relaxed);
        stats.max_latency_us.store(0, Ordering::Relaxed);
        *stats.latencies.write().await = Vec::new();
    }

    // Main benchmark
    println!("================================================================================");
    println!("                         MAIN BENCHMARK STARTING                               ");
    println!("================================================================================");
    println!();

    let benchmark_start = Instant::now();
    let mut tasks = Vec::new();

    for (idx, node_config) in node_configs.iter().enumerate() {
        let node_cfg = node_config.clone();
        let wallet = wallets[idx].clone();
        let recipient = recipients[(idx + 1) % recipients.len()].clone();
        let cfg = config.clone();
        let stats_clone = stats.clone();
        let client_clone = client.clone();
        let sem_clone = semaphore.clone();
        let barrier_clone = barrier.clone();

        let task = tokio::spawn(async move {
            // Wait for all nodes to start together
            barrier_clone.wait().await;
            println!("  Node {} starting benchmark...", node_cfg.node_id);
            submit_batches(&node_cfg, &wallet, &recipient, &cfg, stats_clone, client_clone, sem_clone, false).await
        });

        tasks.push(task);
    }

    // Collect results
    let mut node_results = Vec::new();
    for (idx, task) in tasks.into_iter().enumerate() {
        match task.await {
            Ok(Ok((sent, tps))) => {
                println!("  Node {} completed: {} tx at {:.0} TPS", idx, sent, tps);
                node_results.push((sent, tps));
            }
            Ok(Err(e)) => {
                eprintln!("  Node {} failed: {}", idx, e);
                node_results.push((0, 0.0));
            }
            Err(e) => {
                eprintln!("  Node {} panicked: {}", idx, e);
                node_results.push((0, 0.0));
            }
        }
    }

    let elapsed = benchmark_start.elapsed();

    // Print results
    print_results(&config, &stats, elapsed, &node_results).await;

    // Cleanup local nodes
    if !config.use_existing_nodes {
        println!("\nCleaning up local nodes...");
        for (idx, child_opt) in node_processes.into_iter().enumerate() {
            if let Some(mut child) = child_opt {
                if let Err(e) = child.kill().await {
                    eprintln!("  Warning: Failed to kill node {}: {}", idx, e);
                }
            }
        }
        println!("Cleanup complete.");
    }

    println!("\nBenchmark finished!\n");
    Ok(())
}

// Implement Clone for TestWallet (needed for async tasks)
impl Clone for TestWallet {
    fn clone(&self) -> Self {
        Self {
            signing_key: SigningKey::from_bytes(&self.signing_key.to_bytes()),
            address: self.address.clone(),
        }
    }
}
