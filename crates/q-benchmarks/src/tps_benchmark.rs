//! TPS (Transactions Per Second) Benchmarking Module
//!
//! Measures baseline and target TPS performance for Q-NarwhalKnight
//! with detailed transaction processing analysis using ACTUAL Ed25519
//! signature verification and bincode serialization instead of sleep-based delays.

use crate::{BenchmarkConfig, TpsMetrics};
use anyhow::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task;
use tracing::{debug, info, warn};

use q_types::{
    Address, Amount, Digest, SecretKey, Sha3_256, TokenType, Transaction, TransactionType,
    TxSignaturePhase,
};

/// Shared signing context for benchmarks - avoids re-creating keys per transaction
struct SigningContext {
    signing_key: SecretKey,
    from_address: Address,
}

impl SigningContext {
    fn new(node_id: u32) -> Self {
        let mut seed = [0u8; 32];
        seed[0] = node_id as u8;
        seed[1] = 0xBE;
        seed[2] = 0xEF;
        let signing_key = SecretKey::from_bytes(&seed);
        let from_address: Address = signing_key.verifying_key().to_bytes();
        Self {
            signing_key,
            from_address,
        }
    }

    /// Create and sign a transaction using real Ed25519 cryptography
    fn create_and_sign(&self, nonce: u64) -> Transaction {
        use ed25519_dalek::Signer;

        let to: Address = [0xDDu8; 32];
        let amount: Amount = 1_000_000;
        let fee: Amount = 100;

        // Build canonical message
        let mut msg = Vec::with_capacity(32 + 32 + 16 + 16 + 8);
        msg.extend_from_slice(&self.from_address);
        msg.extend_from_slice(&to);
        msg.extend_from_slice(&amount.to_le_bytes());
        msg.extend_from_slice(&fee.to_le_bytes());
        msg.extend_from_slice(&nonce.to_le_bytes());

        let signature = self.signing_key.sign(&msg);

        Transaction {
            id: {
                let mut hasher = Sha3_256::new();
                hasher.update(&msg);
                let result = hasher.finalize();
                let mut id = [0u8; 32];
                id.copy_from_slice(&result);
                id
            },
            from: self.from_address,
            to,
            amount,
            fee,
            nonce,
            signature: signature.to_bytes().to_vec(),
            timestamp: chrono::Utc::now(),
            data: Vec::new(),
            token_type: TokenType::QUG,
            fee_token_type: TokenType::QUGUSD,
            tx_type: TransactionType::Transfer,
            pqc_signature: None,
            signature_phase: TxSignaturePhase::Phase0Ed25519,
            pqc_public_key: None,
            zk_proof_bundle: None,
            privacy_level: Default::default(),
            bulletproof: None,
            nullifier: None,
            memo: None,
        }
    }

    /// Verify a transaction signature using real Ed25519 verification
    fn verify(&self, tx: &Transaction) -> bool {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};

        let verifying_key = match VerifyingKey::from_bytes(&tx.from) {
            Ok(vk) => vk,
            Err(_) => return false,
        };

        let mut msg = Vec::with_capacity(32 + 32 + 16 + 16 + 8);
        msg.extend_from_slice(&tx.from);
        msg.extend_from_slice(&tx.to);
        msg.extend_from_slice(&tx.amount.to_le_bytes());
        msg.extend_from_slice(&tx.fee.to_le_bytes());
        msg.extend_from_slice(&tx.nonce.to_le_bytes());

        if tx.signature.len() != 64 {
            return false;
        }
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&tx.signature);
        let signature = match Signature::from_bytes(&sig_bytes) {
            sig => sig,
        };

        verifying_key.verify(&msg, &signature).is_ok()
    }
}

/// Measure baseline TPS performance of the system
pub async fn measure_baseline_tps(config: &BenchmarkConfig) -> Result<TpsMetrics> {
    info!(
        "Starting TPS benchmark - Target: {:.0} TPS, Duration: {}s",
        config.target_tps, config.duration_seconds
    );

    let transaction_counter = Arc::new(AtomicU64::new(0));
    let start_time = Instant::now();
    let mut measurement_tasks = Vec::new();

    // Start transaction generation tasks (one per simulated node)
    for i in 0..config.node_count {
        let counter = Arc::clone(&transaction_counter);
        let target_rate = config.target_tps / config.node_count as f64;
        let duration = Duration::from_secs(config.duration_seconds);

        let task = task::spawn(async move {
            simulate_transaction_load(i, target_rate, duration, counter).await
        });
        measurement_tasks.push(task);
    }

    // Measure TPS in intervals
    let measurement_task = {
        let counter = Arc::clone(&transaction_counter);
        let duration = Duration::from_secs(config.duration_seconds);
        let interval = Duration::from_millis(config.measurement_interval_ms);

        task::spawn(async move { measure_tps_over_time(counter, duration, interval).await })
    };

    // Wait for all tasks to complete
    for task in measurement_tasks {
        task.await?;
    }

    let tps_measurements = measurement_task.await?;
    let elapsed = start_time.elapsed();
    let total_transactions = transaction_counter.load(Ordering::Relaxed);

    // Calculate metrics
    let actual_tps = total_transactions as f64 / elapsed.as_secs_f64();
    let peak_tps = tps_measurements.iter().copied().fold(0.0_f64, f64::max);
    let sustained_tps = calculate_sustained_tps(&tps_measurements);
    let efficiency_ratio = actual_tps / config.target_tps;

    info!("TPS Benchmark Results:");
    info!("   Total Transactions: {}", total_transactions);
    info!("   Actual TPS: {:.0}", actual_tps);
    info!("   Peak TPS: {:.0}", peak_tps);
    info!("   Sustained TPS: {:.0}", sustained_tps);
    info!("   Efficiency: {:.1}%", efficiency_ratio * 100.0);

    if actual_tps < config.target_tps * 0.8 {
        warn!("TPS significantly below target - performance investigation needed");
    }

    // Also run the crypto overhead comparison
    let _ = measure_crypto_overhead().await;

    Ok(TpsMetrics {
        transactions_per_second: actual_tps,
        peak_tps,
        sustained_tps,
        target_tps: config.target_tps,
        efficiency_ratio,
    })
}

/// Simulate transaction load from a single node using REAL cryptographic operations
async fn simulate_transaction_load(
    node_id: u32,
    target_tps: f64,
    duration: Duration,
    counter: Arc<AtomicU64>,
) -> Result<()> {
    let start = Instant::now();
    let interval = Duration::from_nanos((1_000_000_000.0 / target_tps) as u64);

    let ctx = SigningContext::new(node_id);

    debug!(
        "Node {} starting transaction simulation at {:.0} TPS",
        node_id, target_tps
    );

    let mut nonce = 0u64;

    while start.elapsed() < duration {
        let tx_start = Instant::now();

        // Process a single transaction with REAL crypto
        process_single_transaction_real(&ctx, nonce)?;
        nonce += 1;
        counter.fetch_add(1, Ordering::Relaxed);

        // Rate limiting to maintain target TPS
        let elapsed = tx_start.elapsed();
        if elapsed < interval {
            tokio::time::sleep(interval - elapsed).await;
        }
    }

    debug!("Node {} completed transaction simulation", node_id);
    Ok(())
}

/// Process a single transaction using ACTUAL cryptographic operations:
/// 1. Ed25519 signing (creates the transaction)
/// 2. bincode serialization (simulates network/storage encoding)
/// 3. bincode deserialization
/// 4. Ed25519 signature verification
/// 5. SHA3-256 hash validation
fn process_single_transaction_real(ctx: &SigningContext, nonce: u64) -> Result<()> {
    // Step 1: Create and sign with real Ed25519
    let tx = ctx.create_and_sign(nonce);

    // Step 2: Serialize with bincode (simulates network encoding)
    let encoded = bincode::serialize(&tx).expect("bincode serialize");

    // Step 3: Deserialize (simulates receiving from network)
    let decoded: Transaction = bincode::deserialize(&encoded).expect("bincode deserialize");

    // Step 4: Verify Ed25519 signature
    let valid = ctx.verify(&decoded);
    assert!(valid, "Signature verification must pass");

    // Step 5: Verify transaction hash
    let mut hasher = Sha3_256::new();
    let mut msg = Vec::with_capacity(32 + 32 + 16 + 16 + 8);
    msg.extend_from_slice(&decoded.from);
    msg.extend_from_slice(&decoded.to);
    msg.extend_from_slice(&decoded.amount.to_le_bytes());
    msg.extend_from_slice(&decoded.fee.to_le_bytes());
    msg.extend_from_slice(&decoded.nonce.to_le_bytes());
    hasher.update(&msg);
    let hash = hasher.finalize();
    let mut expected_id = [0u8; 32];
    expected_id.copy_from_slice(&hash);
    assert_eq!(decoded.id, expected_id, "Transaction hash must match");

    std::hint::black_box(&encoded);
    Ok(())
}

/// Measure the overhead of real cryptographic operations vs no-crypto baseline.
/// This gives visibility into how much time is spent on crypto vs everything else.
async fn measure_crypto_overhead() -> Result<()> {
    let iterations = 1000u64;
    let ctx = SigningContext::new(99);

    // Measure with full crypto pipeline
    let crypto_start = Instant::now();
    for i in 0..iterations {
        process_single_transaction_real(&ctx, i)?;
    }
    let crypto_elapsed = crypto_start.elapsed();

    // Measure without crypto (just data construction + serialization)
    let nocrypto_start = Instant::now();
    for i in 0..iterations {
        let to: Address = [0xDDu8; 32];
        let amount: Amount = 1_000_000u128;
        let fee: Amount = 100u128;

        // Build the message bytes (same as crypto path but no sign/verify)
        let mut msg = Vec::with_capacity(32 + 32 + 16 + 16 + 8);
        msg.extend_from_slice(&ctx.from_address);
        msg.extend_from_slice(&to);
        msg.extend_from_slice(&amount.to_le_bytes());
        msg.extend_from_slice(&fee.to_le_bytes());
        msg.extend_from_slice(&i.to_le_bytes());

        // Hash only (no signing)
        let mut hasher = Sha3_256::new();
        hasher.update(&msg);
        let _hash = hasher.finalize();

        // Serialize the raw message bytes (approximate tx serialization cost)
        let encoded = bincode::serialize(&msg).expect("serialize");
        let _decoded: Vec<u8> = bincode::deserialize(&encoded).expect("deserialize");

        std::hint::black_box(&encoded);
    }
    let nocrypto_elapsed = nocrypto_start.elapsed();

    let crypto_per_tx_us = crypto_elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    let nocrypto_per_tx_us = nocrypto_elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    let overhead_pct = if nocrypto_per_tx_us > 0.0 {
        ((crypto_per_tx_us - nocrypto_per_tx_us) / nocrypto_per_tx_us) * 100.0
    } else {
        0.0
    };

    info!("Crypto overhead analysis ({} iterations):", iterations);
    info!("  With real crypto: {:.1}us/tx", crypto_per_tx_us);
    info!("  Without crypto:   {:.1}us/tx", nocrypto_per_tx_us);
    info!(
        "  Crypto overhead:  {:.1}us/tx ({:.0}%)",
        crypto_per_tx_us - nocrypto_per_tx_us,
        overhead_pct
    );
    info!(
        "  Max theoretical TPS (crypto-bound): {:.0}",
        1_000_000.0 / crypto_per_tx_us
    );

    Ok(())
}

/// Measure TPS over time intervals
async fn measure_tps_over_time(
    counter: Arc<AtomicU64>,
    duration: Duration,
    interval: Duration,
) -> Vec<f64> {
    let mut measurements = Vec::new();
    let start = Instant::now();
    let mut last_count = 0u64;
    let mut last_time = start;

    while start.elapsed() < duration {
        tokio::time::sleep(interval).await;

        let current_count = counter.load(Ordering::Relaxed);
        let current_time = Instant::now();

        let transactions_in_interval = current_count - last_count;
        let time_elapsed = current_time.duration_since(last_time).as_secs_f64();
        let tps = transactions_in_interval as f64 / time_elapsed;

        measurements.push(tps);
        debug!("Interval TPS: {:.0}", tps);

        last_count = current_count;
        last_time = current_time;
    }

    measurements
}

/// Calculate sustained TPS (95th percentile of measurements)
fn calculate_sustained_tps(measurements: &[f64]) -> f64 {
    if measurements.is_empty() {
        return 0.0;
    }

    let mut sorted = measurements.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Use 95th percentile as sustained TPS
    let index = (sorted.len() as f64 * 0.95) as usize;
    sorted.get(index).copied().unwrap_or(0.0)
}

/// Benchmark TPS scaling with different node counts
pub async fn benchmark_tps_scaling(
    base_config: &BenchmarkConfig,
) -> Result<Vec<(u32, TpsMetrics)>> {
    let mut results = Vec::new();

    info!("Starting TPS scaling benchmark");

    for node_count in [1, 2, 4, 8, 16] {
        info!("Testing with {} nodes", node_count);

        let config = BenchmarkConfig {
            node_count,
            duration_seconds: 30, // Shorter runs for scaling test
            ..base_config.clone()
        };

        let metrics = measure_baseline_tps(&config).await?;
        results.push((node_count, metrics));

        // Brief pause between scaling tests
        tokio::time::sleep(Duration::from_secs(2)).await;
    }

    // Analyze scaling characteristics
    analyze_scaling_results(&results);

    Ok(results)
}

/// Analyze TPS scaling results and detect bottlenecks
fn analyze_scaling_results(results: &[(u32, TpsMetrics)]) {
    info!("TPS Scaling Analysis:");

    for (node_count, metrics) in results {
        let efficiency = metrics.efficiency_ratio * 100.0;
        info!(
            "   {} nodes: {:.0} TPS ({:.1}% efficiency)",
            node_count, metrics.transactions_per_second, efficiency
        );
    }

    // Calculate scaling efficiency
    if results.len() >= 2 {
        let baseline_tps = results[0].1.transactions_per_second;
        let final_tps = results.last().unwrap().1.transactions_per_second;
        let final_nodes = results.last().unwrap().0;

        let scaling_efficiency = final_tps / (baseline_tps * final_nodes as f64);

        info!(
            "Scaling Efficiency: {:.1}% (1.0 = perfect linear scaling)",
            scaling_efficiency * 100.0
        );

        if scaling_efficiency < 0.7 {
            warn!("Poor scaling efficiency detected - bottleneck investigation needed");
        }
    }
}
