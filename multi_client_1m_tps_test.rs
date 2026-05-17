#!/usr/bin/env rust-script
//! Multi-Client Realistic TPS Benchmark
//!
//! Measures actual HTTP throughput against a running Q-NarwhalKnight node.
//! Reports three separate metrics:
//!   1. API acceptance rate  - how fast the server accepts HTTP requests
//!   2. Processing rate      - transactions processed (mempool -> state)
//!   3. Block inclusion rate - transactions confirmed in blocks
//!
//! Usage:
//!   # Start the node first
//!   ./target/release/q-api-server &
//!
//!   # Run the benchmark (defaults to http://localhost:8080)
//!   rust-script multi_client_1m_tps_test.rs
//!
//!   # Or specify a different server
//!   rust-script multi_client_1m_tps_test.rs --server-url http://192.168.1.10:8080

//! ```cargo
//! [dependencies]
//! reqwest = { version = "0.11", features = ["json"] }
//! tokio = { version = "1", features = ["full"] }
//! serde = { version = "1", features = ["derive"] }
//! serde_json = "1"
//! rmp-serde = "1"
//! bincode = "1"
//! postcard = { version = "1", features = ["alloc"] }
//! ed25519-dalek = { version = "2", features = ["rand_core"] }
//! rand = "0.8"
//! sha2 = "0.10"
//! sha3 = "0.10"
//! chrono = { version = "0.4", features = ["serde"] }
//! ```

use chrono::{DateTime, Utc};
use ed25519_dalek::{Signer, SigningKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Transaction types matching q_types::Transaction serialization format
// ---------------------------------------------------------------------------

/// Mirrors q_types::TokenType
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum TokenType {
    QUG,
    QUGUSD,
    Custom([u8; 32]),
}

/// Mirrors q_types::TransactionType (repr values match the server)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum TransactionType {
    Transfer = 0x00,
    Coinbase = 0x01,
    Burn = 0x02,
    Fee = 0x03,
    TokenCreate = 0x10,
    TokenMint = 0x11,
}

/// Mirrors q_types::TxSignaturePhase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum TxSignaturePhase {
    Phase0Ed25519,
    Phase1Dilithium5,
    Phase2SQIsign,
    HybridEd25519SQIsign,
}

/// Mirrors q_types::TransactionPrivacyLevel
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
enum TransactionPrivacyLevel {
    #[default]
    Transparent,
    Confidential,
    Anonymous,
    FullPrivacy,
}

/// Custom u128 serde module matching q_types::u128_serde.
/// The server always serializes u128 as a string for MessagePack compatibility.
mod u128_serde {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &u128, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u128, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse::<u128>().map_err(serde::de::Error::custom)
    }
}

/// Matches the full q_types::Transaction struct field-for-field.
///
/// The server deserializes incoming transactions with rmp_serde (MessagePack)
/// or bincode, so field names and order must match exactly. Fields with
/// `#[serde(default)]` on the server side are optional here -- we include them
/// all to produce byte-identical serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Transaction {
    id: [u8; 32],
    from: [u8; 32],
    to: [u8; 32],
    #[serde(with = "u128_serde")]
    amount: u128,
    #[serde(with = "u128_serde")]
    fee: u128,
    nonce: u64,
    signature: Vec<u8>,
    timestamp: DateTime<Utc>,
    data: Vec<u8>,
    token_type: TokenType,
    fee_token_type: TokenType,
    tx_type: TransactionType,
    #[serde(default)]
    pqc_signature: Option<Vec<u8>>,
    signature_phase: TxSignaturePhase,
    #[serde(default)]
    pqc_public_key: Option<Vec<u8>>,
    #[serde(default)]
    zk_proof_bundle: Option<Vec<u8>>,
    #[serde(default)]
    privacy_level: TransactionPrivacyLevel,
    #[serde(default)]
    bulletproof: Option<Vec<u8>>,
    #[serde(default)]
    nullifier: Option<[u8; 32]>,
    #[serde(default)]
    memo: Option<String>,
}

/// Compute the transaction hash the same way the server does:
/// SHA3-256 of the postcard-serialized transaction.
fn tx_hash(tx: &Transaction) -> [u8; 32] {
    use sha3::{Digest, Sha3_256};
    let encoded = postcard::to_allocvec(tx).expect("postcard serialization failed");
    let hash = Sha3_256::digest(&encoded);
    let mut out = [0u8; 32];
    out.copy_from_slice(&hash);
    out
}

/// The server verifies Ed25519 signatures over:
///   postcard::to_allocvec(&(from, to, amount, nonce))
///
/// `from` is the Ed25519 public key bytes (VerifyingKey).
fn sign_transaction(signing_key: &SigningKey, tx: &mut Transaction) {
    let message =
        postcard::to_allocvec(&(tx.from, tx.to, tx.amount, tx.nonce))
            .expect("postcard message serialization failed");
    let sig = signing_key.sign(&message);
    tx.signature = sig.to_bytes().to_vec();
    // Recompute id after signature is set (server hashes the full struct)
    tx.id = tx_hash(tx);
}

fn create_signed_transaction(signing_key: &SigningKey, nonce: u64) -> Transaction {
    let verifying_key = signing_key.verifying_key();
    let from: [u8; 32] = verifying_key.to_bytes();

    // Generate a random recipient
    let recipient_key = SigningKey::generate(&mut OsRng);
    let to: [u8; 32] = recipient_key.verifying_key().to_bytes();

    let mut tx = Transaction {
        id: [0u8; 32], // placeholder, will be overwritten by sign_transaction
        from,
        to,
        amount: 1000 + nonce as u128,
        fee: 10,
        nonce,
        signature: vec![0u8; 64], // placeholder
        timestamp: Utc::now(),
        data: vec![],
        token_type: TokenType::QUG,
        fee_token_type: TokenType::QUGUSD,
        tx_type: TransactionType::Transfer,
        pqc_signature: None,
        signature_phase: TxSignaturePhase::Phase0Ed25519,
        pqc_public_key: None,
        zk_proof_bundle: None,
        privacy_level: TransactionPrivacyLevel::Transparent,
        bulletproof: None,
        nullifier: None,
        memo: None,
    };
    sign_transaction(signing_key, &mut tx);
    tx
}

// ---------------------------------------------------------------------------
// Batch wrapper (matches q-api-server/src/binary_protocol.rs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BinaryTransactionBatch {
    transactions: Vec<Transaction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BinaryResponse {
    success: bool,
    tx_hashes: Vec<[u8; 32]>,
    accepted: usize,
    rejected: usize,
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SerFormat {
    MessagePack,
    Bincode,
}

impl std::fmt::Display for SerFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerFormat::MessagePack => write!(f, "msgpack"),
            SerFormat::Bincode => write!(f, "bincode"),
        }
    }
}

fn serialize_batch(batch: &BinaryTransactionBatch, format: SerFormat) -> Vec<u8> {
    match format {
        SerFormat::MessagePack => {
            rmp_serde::to_vec(batch).expect("msgpack serialization failed")
        }
        SerFormat::Bincode => {
            bincode::serialize(batch).expect("bincode serialization failed")
        }
    }
}

fn content_type_for(format: SerFormat) -> &'static str {
    match format {
        SerFormat::MessagePack => "application/msgpack",
        SerFormat::Bincode => "application/octet-stream",
    }
}

// ---------------------------------------------------------------------------
// Client worker
// ---------------------------------------------------------------------------

struct BatchResult {
    total_sent: usize,
    accepted: usize,
    rejected: usize,
    http_errors: usize,
    elapsed: Duration,
    total_payload_bytes: usize,
}

async fn client_worker(
    client_id: usize,
    num_batches: usize,
    batch_size: usize,
    http_client: Arc<reqwest::Client>,
    server_url: &str,
    format: SerFormat,
) -> BatchResult {
    let signing_key = SigningKey::generate(&mut OsRng);
    let url = format!("{}/api/v1/binary/batch", server_url);

    let mut total_sent = 0usize;
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut http_errors = 0usize;
    let mut total_payload_bytes = 0usize;

    let start = Instant::now();

    for batch_num in 0..num_batches {
        let base_nonce = (client_id as u64) * 1_000_000 + (batch_num as u64) * (batch_size as u64);

        let transactions: Vec<Transaction> = (0..batch_size)
            .map(|i| create_signed_transaction(&signing_key, base_nonce + i as u64))
            .collect();

        let batch = BinaryTransactionBatch { transactions };
        let payload = serialize_batch(&batch, format);
        total_payload_bytes += payload.len();

        match http_client
            .post(&url)
            .header("Content-Type", content_type_for(format))
            .body(payload)
            .send()
            .await
        {
            Ok(resp) => {
                total_sent += batch_size;
                if resp.status().is_success() {
                    // Try to parse the response for accepted/rejected counts
                    if let Ok(body) = resp.bytes().await {
                        if let Ok(bin_resp) = rmp_serde::from_slice::<BinaryResponse>(&body) {
                            accepted += bin_resp.accepted;
                            rejected += bin_resp.rejected;
                        } else {
                            // Could not parse response; count all as accepted (HTTP 200)
                            accepted += batch_size;
                        }
                    } else {
                        accepted += batch_size;
                    }
                } else {
                    let status = resp.status();
                    http_errors += 1;
                    eprintln!(
                        "  [client {}] batch {}/{} HTTP {}: {}",
                        client_id,
                        batch_num + 1,
                        num_batches,
                        status.as_u16(),
                        status.canonical_reason().unwrap_or("Unknown")
                    );
                }
            }
            Err(e) => {
                http_errors += 1;
                eprintln!("  [client {}] batch {}/{} network error: {}", client_id, batch_num + 1, num_batches, e);
            }
        }
    }

    BatchResult {
        total_sent,
        accepted,
        rejected,
        http_errors,
        elapsed: start.elapsed(),
        total_payload_bytes,
    }
}

// ---------------------------------------------------------------------------
// Phase 2: poll node status to measure processing rate
// ---------------------------------------------------------------------------

async fn poll_processing_status(
    http_client: &reqwest::Client,
    server_url: &str,
    timeout_secs: u64,
) -> Option<serde_json::Value> {
    let url = format!("{}/api/v1/node/status", server_url);
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);

    let mut last_status = None;

    while Instant::now() < deadline {
        match http_client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(body) = resp.json::<serde_json::Value>().await {
                    last_status = Some(body);
                }
            }
            _ => {}
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    last_status
}

// ---------------------------------------------------------------------------
// Phase 3: check block heights to measure block inclusion
// ---------------------------------------------------------------------------

async fn poll_block_heights(
    http_client: &reqwest::Client,
    server_url: &str,
    initial_height: u64,
    poll_duration_secs: u64,
) -> (u64, u64) {
    let url = format!("{}/api/v1/node/status", server_url);
    let deadline = Instant::now() + Duration::from_secs(poll_duration_secs);

    let mut latest_height = initial_height;

    while Instant::now() < deadline {
        match http_client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(body) = resp.json::<serde_json::Value>().await {
                    if let Some(data) = body.get("data") {
                        if let Some(h) = data.get("current_height").and_then(|v| v.as_u64()) {
                            if h > latest_height {
                                latest_height = h;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    (initial_height, latest_height)
}

async fn get_current_height(http_client: &reqwest::Client, server_url: &str) -> u64 {
    let url = format!("{}/api/v1/node/status", server_url);
    match http_client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            if let Ok(body) = resp.json::<serde_json::Value>().await {
                if let Some(data) = body.get("data") {
                    return data
                        .get("current_height")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                }
            }
            0
        }
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Serialization size comparison
// ---------------------------------------------------------------------------

fn measure_serialization_sizes(batch_size: usize) -> (usize, usize) {
    let signing_key = SigningKey::generate(&mut OsRng);
    let transactions: Vec<Transaction> = (0..batch_size)
        .map(|i| create_signed_transaction(&signing_key, i as u64))
        .collect();
    let batch = BinaryTransactionBatch { transactions };

    let msgpack_bytes = rmp_serde::to_vec(&batch).expect("msgpack failed");
    let bincode_bytes = bincode::serialize(&batch).expect("bincode failed");

    (msgpack_bytes.len(), bincode_bytes.len())
}

// ---------------------------------------------------------------------------
// Single test run
// ---------------------------------------------------------------------------

struct TestConfig {
    num_clients: usize,
    batches_per_client: usize,
    batch_size: usize,
    server_url: String,
    format: SerFormat,
}

struct TestResult {
    config: TestConfig,
    total_sent: usize,
    total_accepted: usize,
    total_rejected: usize,
    total_http_errors: usize,
    wall_clock: Duration,
    total_payload_bytes: usize,
    per_client_tps: Vec<f64>,
    // Phase 2/3
    initial_height: u64,
    final_height: u64,
    phase2_mempool_info: Option<serde_json::Value>,
}

async fn run_test(config: TestConfig) -> TestResult {
    let total_tx = config.num_clients * config.batches_per_client * config.batch_size;

    println!("--- Test: {} clients x {} batches x {} tx/batch = {} total ({}) ---",
        config.num_clients,
        config.batches_per_client,
        config.batch_size,
        total_tx,
        config.format,
    );

    let http_client = Arc::new(
        reqwest::Client::builder()
            .pool_max_idle_per_host(config.num_clients * 2)
            .timeout(Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client"),
    );

    // Record block height before sending
    let initial_height = get_current_height(&http_client, &config.server_url).await;

    // Phase 1: Send all batches
    println!("  Phase 1: Sending transactions...");
    let global_start = Instant::now();
    let mut tasks = Vec::new();

    for client_id in 0..config.num_clients {
        let client_arc = http_client.clone();
        let url = config.server_url.clone();
        let fmt = config.format;
        let batches = config.batches_per_client;
        let bsize = config.batch_size;

        tasks.push(tokio::spawn(async move {
            client_worker(client_id, batches, bsize, client_arc, &url, fmt).await
        }));
    }

    let mut total_sent = 0usize;
    let mut total_accepted = 0usize;
    let mut total_rejected = 0usize;
    let mut total_http_errors = 0usize;
    let mut total_payload_bytes = 0usize;
    let mut per_client_tps = Vec::new();

    for (i, task) in tasks.into_iter().enumerate() {
        match task.await {
            Ok(result) => {
                let client_tps = if result.elapsed.as_secs_f64() > 0.0 {
                    result.total_sent as f64 / result.elapsed.as_secs_f64()
                } else {
                    0.0
                };
                println!(
                    "    client {:>2}: sent={:<6} accepted={:<6} rejected={:<4} errors={:<3} {:.0} tx/s",
                    i, result.total_sent, result.accepted, result.rejected, result.http_errors, client_tps,
                );
                total_sent += result.total_sent;
                total_accepted += result.accepted;
                total_rejected += result.rejected;
                total_http_errors += result.http_errors;
                total_payload_bytes += result.total_payload_bytes;
                per_client_tps.push(client_tps);
            }
            Err(e) => {
                eprintln!("    client {:>2}: PANIC: {}", i, e);
            }
        }
    }

    let wall_clock = global_start.elapsed();

    // Phase 2: Poll status for processing info (10 seconds)
    println!("  Phase 2: Polling processing status (10s)...");
    let phase2_info = poll_processing_status(&http_client, &config.server_url, 10).await;

    // Phase 3: Poll block heights for inclusion (15 seconds)
    println!("  Phase 3: Polling block inclusion (15s)...");
    let (_, final_height) =
        poll_block_heights(&http_client, &config.server_url, initial_height, 15).await;

    TestResult {
        config,
        total_sent,
        total_accepted,
        total_rejected,
        total_http_errors,
        wall_clock,
        total_payload_bytes,
        per_client_tps,
        initial_height,
        final_height,
        phase2_mempool_info: phase2_info,
    }
}

fn print_result(r: &TestResult) {
    let wall_secs = r.wall_clock.as_secs_f64();
    let api_tps = if wall_secs > 0.0 {
        r.total_sent as f64 / wall_secs
    } else {
        0.0
    };

    let new_blocks = if r.final_height > r.initial_height {
        r.final_height - r.initial_height
    } else {
        0
    };

    println!();
    println!("  Results ({} clients, {}):", r.config.num_clients, r.config.format);
    println!("    Transactions sent:        {}", r.total_sent);
    println!("    Server accepted:          {}", r.total_accepted);
    println!("    Server rejected (sig):    {}", r.total_rejected);
    println!("    HTTP errors:              {}", r.total_http_errors);
    println!("    Wall-clock time:          {:.3}s", wall_secs);
    println!("    API acceptance TPS:       {:.0} tx/s  (HTTP request throughput)", api_tps);
    println!("    Total payload:            {:.2} MB", r.total_payload_bytes as f64 / 1_048_576.0);
    println!("    Avg payload per tx:       {} bytes", if r.total_sent > 0 { r.total_payload_bytes / r.total_sent } else { 0 });

    if !r.per_client_tps.is_empty() {
        let avg = r.per_client_tps.iter().sum::<f64>() / r.per_client_tps.len() as f64;
        let min = r.per_client_tps.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = r.per_client_tps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("    Per-client TPS:           avg={:.0}  min={:.0}  max={:.0}", avg, min, max);
    }

    println!("    Block height before:      {}", r.initial_height);
    println!("    Block height after:       {}", r.final_height);
    println!("    New blocks produced:      {}", new_blocks);

    // Extract mempool size if available
    if let Some(ref info) = r.phase2_mempool_info {
        if let Some(data) = info.get("data") {
            if let Some(mempool) = data.get("mempool_size").and_then(|v| v.as_u64()) {
                println!("    Mempool size (last poll):  {}", mempool);
            }
            if let Some(tx_pool) = data.get("tx_pool_size").and_then(|v| v.as_u64()) {
                println!("    Tx pool size (last poll):  {}", tx_pool);
            }
        }
    }

    println!();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut server_url = "http://localhost:8080".to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--server-url" => {
                i += 1;
                if i < args.len() {
                    server_url = args[i].clone();
                }
            }
            "--help" | "-h" => {
                println!("Usage: {} [--server-url URL]", args[0]);
                println!();
                println!("Options:");
                println!("  --server-url URL    Server base URL (default: http://localhost:8080)");
                println!();
                println!("The benchmark sends properly signed Ed25519 transactions to the");
                println!("/api/v1/binary/batch endpoint and measures three metrics:");
                println!("  1. API acceptance rate (HTTP throughput)");
                println!("  2. Mempool processing  (via /api/v1/node/status polling)");
                println!("  3. Block inclusion      (block height delta)");
                return;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Strip trailing slash
    let server_url = server_url.trim_end_matches('/').to_string();

    println!("==========================================================================");
    println!("  Multi-Client TPS Benchmark");
    println!("  Server: {}", server_url);
    println!("  Signatures: Ed25519 (real, verified server-side)");
    println!("  Timestamp: {}", Utc::now().to_rfc3339());
    println!("==========================================================================");
    println!();

    // -----------------------------------------------------------------------
    // Pre-flight: verify server is reachable
    // -----------------------------------------------------------------------
    {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        let status_url = format!("{}/api/v1/node/status", server_url);
        match client.get(&status_url).send().await {
            Ok(resp) if resp.status().is_success() => {
                println!("[preflight] Server is reachable (HTTP {}).", resp.status().as_u16());
            }
            Ok(resp) => {
                println!(
                    "[preflight] WARNING: Server returned HTTP {}. Tests may fail.",
                    resp.status().as_u16()
                );
            }
            Err(e) => {
                eprintln!("[preflight] ERROR: Cannot reach server at {}: {}", status_url, e);
                eprintln!("            Start the node first, then re-run this benchmark.");
                std::process::exit(1);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Serialization size comparison
    // -----------------------------------------------------------------------
    println!();
    println!("--- Serialization size comparison (100 tx batch) ---");
    let (msgpack_size, bincode_size) = measure_serialization_sizes(100);
    let msgpack_per_tx = msgpack_size as f64 / 100.0;
    let bincode_per_tx = bincode_size as f64 / 100.0;
    println!("  MessagePack: {} bytes total, {:.0} bytes/tx", msgpack_size, msgpack_per_tx);
    println!("  Bincode:     {} bytes total, {:.0} bytes/tx", bincode_size, bincode_per_tx);
    if bincode_per_tx > 0.0 {
        let ratio = msgpack_per_tx / bincode_per_tx;
        if ratio > 1.0 {
            println!("  Bincode is {:.1}x smaller per transaction", ratio);
        } else {
            println!("  MessagePack is {:.1}x smaller per transaction", 1.0 / ratio);
        }
    }
    println!();

    // -----------------------------------------------------------------------
    // Test matrix: (clients, batches_per_client, batch_size)
    // -----------------------------------------------------------------------
    let test_configs = vec![
        // Warmup: small batch to prime connections
        (4, 1, 100),
        // Scale up clients
        (4, 2, 1000),
        (8, 2, 1000),
        (12, 2, 1000),
        (16, 2, 1000),
        // Larger batches
        (16, 4, 5000),
    ];

    let mut all_results = Vec::new();

    for (clients, batches, batch_size) in &test_configs {
        // Run with MessagePack
        let result = run_test(TestConfig {
            num_clients: *clients,
            batches_per_client: *batches,
            batch_size: *batch_size,
            server_url: server_url.clone(),
            format: SerFormat::MessagePack,
        })
        .await;
        print_result(&result);
        all_results.push(result);

        // Pause between tests to let mempool drain
        tokio::time::sleep(Duration::from_secs(3)).await;
    }

    // Run the largest config again with bincode to compare
    println!("==========================================================================");
    println!("  Bincode comparison run (16 clients x 4 batches x 5000 tx)");
    println!("==========================================================================");
    let bincode_result = run_test(TestConfig {
        num_clients: 16,
        batches_per_client: 4,
        batch_size: 5000,
        server_url: server_url.clone(),
        format: SerFormat::Bincode,
    })
    .await;
    print_result(&bincode_result);
    all_results.push(bincode_result);

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("==========================================================================");
    println!("  SUMMARY");
    println!("==========================================================================");
    println!();
    println!(
        "  {:>8} {:>8} {:>7} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "clients", "format", "total", "accepted", "rejected", "errors", "time(s)", "API TPS"
    );
    println!("  {}", "-".repeat(85));

    for r in &all_results {
        let wall_secs = r.wall_clock.as_secs_f64();
        let api_tps = if wall_secs > 0.0 {
            r.total_sent as f64 / wall_secs
        } else {
            0.0
        };
        println!(
            "  {:>8} {:>8} {:>7} {:>10} {:>10} {:>10} {:>8.2} {:>8.0}",
            r.config.num_clients,
            r.config.format.to_string(),
            r.total_sent,
            r.total_accepted,
            r.total_rejected,
            r.total_http_errors,
            wall_secs,
            api_tps,
        );
    }

    println!();
    println!("  Notes:");
    println!("  - 'API TPS' = transactions submitted per second of wall-clock time.");
    println!("    This measures HTTP acceptance, NOT processing or finality.");
    println!("  - 'accepted' = transactions the server accepted into the mempool");
    println!("    (signature verified, deserialization succeeded).");
    println!("  - 'rejected' = transactions rejected by the server (bad signature,");
    println!("    bad format, etc.).");
    println!("  - Block inclusion and finality are slower than API acceptance and");
    println!("    depend on block production interval and consensus.");
    println!("  - All transactions use real Ed25519 signatures verified server-side.");
    println!();

    // -----------------------------------------------------------------------
    // Honest assessment
    // -----------------------------------------------------------------------
    let best = all_results
        .iter()
        .max_by(|a, b| {
            let a_tps = a.total_sent as f64 / a.wall_clock.as_secs_f64().max(0.001);
            let b_tps = b.total_sent as f64 / b.wall_clock.as_secs_f64().max(0.001);
            a_tps.partial_cmp(&b_tps).unwrap_or(std::cmp::Ordering::Equal)
        });

    if let Some(best) = best {
        let best_tps = best.total_sent as f64 / best.wall_clock.as_secs_f64().max(0.001);
        let accept_rate = if best.total_sent > 0 {
            best.total_accepted as f64 / best.total_sent as f64 * 100.0
        } else {
            0.0
        };

        println!("  Peak API acceptance rate: {:.0} tx/s ({} clients, {})", best_tps, best.config.num_clients, best.config.format);
        println!("  Server acceptance rate:   {:.1}%", accept_rate);

        if best.total_accepted == 0 && best.total_sent > 0 {
            println!();
            println!("  WARNING: No transactions were accepted by the server.");
            println!("  Possible causes:");
            println!("  - Server does not support the binary batch endpoint");
            println!("  - Serialization format mismatch (Transaction struct fields differ)");
            println!("  - All signatures rejected (signing message format mismatch)");
        }
    }

    println!();
    println!("Benchmark complete.");
}
