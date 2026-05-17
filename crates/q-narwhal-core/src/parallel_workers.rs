//! Parallel Worker Pool for Narwhal Mempool
//!
//! Implements 10 concurrent workers for parallel certificate processing
//! Target: 1M+ TPS (10 workers × 100 certs/sec × 10k tx/cert)

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

use crate::NarwhalCore;
use q_types::Transaction;

/// Configuration for parallel worker pool
#[derive(Clone, Debug)]
pub struct WorkerPoolConfig {
    /// Number of parallel workers
    pub worker_count: usize,
    /// Transactions per certificate batch
    pub batch_size: usize,
    /// Maximum pending transactions per worker
    pub queue_size: usize,
    /// Enable SIMD batch verification
    pub enable_simd: bool,
}

impl Default for WorkerPoolConfig {
    fn default() -> Self {
        Self {
            worker_count: 10,
            batch_size: 10_000,
            queue_size: 100_000,
            enable_simd: true,
        }
    }
}

/// Worker task for processing transaction batches
struct Worker {
    id: usize,
    rx: mpsc::Receiver<WorkerMessage>,
    narwhal: Arc<NarwhalCore>,
    simd_engine: Option<Arc<q_crypto_simd::SimdCryptoEngine>>,
    config: WorkerPoolConfig,
}

/// Messages sent to workers
enum WorkerMessage {
    ProcessBatch(Vec<Transaction>),
    Shutdown,
}

impl Worker {
    async fn run(mut self) {
        info!("🔧 Worker {} started", self.id);

        while let Some(msg) = self.rx.recv().await {
            match msg {
                WorkerMessage::ProcessBatch(txs) => {
                    if let Err(e) = self.process_batch(txs).await {
                        warn!("⚠️  Worker {} batch processing failed: {}", self.id, e);
                    }
                }
                WorkerMessage::Shutdown => {
                    info!("🛑 Worker {} shutting down", self.id);
                    break;
                }
            }
        }
    }

    async fn process_batch(&self, txs: Vec<Transaction>) -> Result<()> {
        let start = std::time::Instant::now();
        let tx_count = txs.len();

        // Step 1: Parallel signature verification with SIMD (if enabled)
        if let Some(ref simd) = self.simd_engine {
            // Use SIMD for batch signature verification
            // Convert raw transaction data to typed Ed25519 structures
            use ed25519_dalek::{Signature, VerifyingKey};
            use sha3::{Sha3_256, Digest};

            let mut signatures = Vec::new();
            let mut public_keys = Vec::new();
            let mut messages = Vec::new();

            // Build typed signature/key/message vectors
            for tx in &txs {
                // Reconstruct the message that was signed
                let mut hasher = Sha3_256::new();
                hasher.update(&tx.id);
                hasher.update(&tx.from);
                hasher.update(&tx.to);
                hasher.update(&tx.amount.to_le_bytes());
                hasher.update(&tx.fee.to_le_bytes());
                hasher.update(&tx.nonce.to_le_bytes());
                hasher.update(&tx.data);
                messages.push(hasher.finalize().to_vec());

                // Extract public key from address (first 32 bytes)
                if tx.from.len() < 32 {
                    warn!("⚠️  Worker {} invalid address length, falling back to sequential", self.id);
                    self.verify_sequential(&txs)?;
                    return Ok(());
                }
                let public_key_bytes: [u8; 32] = match tx.from[0..32].try_into() {
                    Ok(bytes) => bytes,
                    Err(_) => {
                        warn!("⚠️  Worker {} failed to extract public key, falling back to sequential", self.id);
                        self.verify_sequential(&txs)?;
                        return Ok(());
                    }
                };

                match VerifyingKey::from_bytes(&public_key_bytes) {
                    Ok(pk) => public_keys.push(pk),
                    Err(_) => {
                        warn!("⚠️  Worker {} invalid public key, falling back to sequential", self.id);
                        self.verify_sequential(&txs)?;
                        return Ok(());
                    }
                }

                // Parse signature (should be 64 bytes for Ed25519)
                if tx.signature.len() != 64 {
                    warn!("⚠️  Worker {} invalid signature length, falling back to sequential", self.id);
                    self.verify_sequential(&txs)?;
                    return Ok(());
                }
                let signature_bytes: [u8; 64] = match tx.signature.as_slice().try_into() {
                    Ok(bytes) => bytes,
                    Err(_) => {
                        warn!("⚠️  Worker {} failed to parse signature, falling back to sequential", self.id);
                        self.verify_sequential(&txs)?;
                        return Ok(());
                    }
                };
                signatures.push(Signature::from_bytes(&signature_bytes));
            }

            // Convert messages to Vec<&[u8]> for SIMD API
            let message_refs: Vec<&[u8]> = messages.iter().map(|m| m.as_slice()).collect();

            // Batch verify with SIMD (8-16 parallel ops)
            match simd.batch_verify_signatures(&signatures, &message_refs, &public_keys).await {
                Ok(result) => {
                    if result.invalid_signatures > 0 {
                        return Err(anyhow::anyhow!(
                            "Worker {}: {}/{} signatures invalid",
                            self.id,
                            result.invalid_signatures,
                            result.total_signatures
                        ));
                    }
                    debug!("✅ Worker {}: Verified {} signatures in {}ms ({}x speedup)",
                        self.id, result.valid_signatures, result.processing_time_ms, result.performance_gain);
                }
                Err(e) => {
                    warn!("⚠️  Worker {} SIMD verification failed: {}, using fallback", self.id, e);
                    // Fallback to sequential verification
                    self.verify_sequential(&txs)?;
                }
            }
        } else {
            // Sequential verification (fallback)
            self.verify_sequential(&txs)?;
        }

        // Step 2: Create certificate with verified transactions
        let certificate = self.narwhal.create_certificate(txs).await?;

        let elapsed = start.elapsed();
        let tps = tx_count as f64 / elapsed.as_secs_f64();

        info!(
            "✅ Worker {} processed {} tx in {:.2}ms ({:.0} TPS)",
            self.id,
            tx_count,
            elapsed.as_millis(),
            tps
        );

        Ok(())
    }

    fn verify_sequential(&self, txs: &[Transaction]) -> Result<()> {
        use ed25519_dalek::{Verifier, VerifyingKey, Signature};
        use sha3::{Sha3_256, Digest};

        // Real production signature verification
        for tx in txs {
            // Reconstruct the message that was signed
            let mut hasher = Sha3_256::new();
            hasher.update(&tx.id);
            hasher.update(&tx.from);
            hasher.update(&tx.to);
            hasher.update(&tx.amount.to_le_bytes());
            hasher.update(&tx.fee.to_le_bytes());
            hasher.update(&tx.nonce.to_le_bytes());
            hasher.update(&tx.data);
            let message = hasher.finalize();

            // Extract public key from address (first 32 bytes)
            if tx.from.len() < 32 {
                return Err(anyhow::anyhow!("Invalid address length: {} bytes", tx.from.len()));
            }
            let public_key_bytes: [u8; 32] = tx.from[0..32].try_into()
                .map_err(|_| anyhow::anyhow!("Failed to extract public key from address"))?;

            // Parse signature (should be 64 bytes for Ed25519)
            if tx.signature.len() != 64 {
                return Err(anyhow::anyhow!("Invalid signature length: {} bytes (expected 64)", tx.signature.len()));
            }
            let signature_bytes: [u8; 64] = tx.signature.as_slice().try_into()
                .map_err(|_| anyhow::anyhow!("Failed to parse signature bytes"))?;

            // Verify the signature
            let verifying_key = VerifyingKey::from_bytes(&public_key_bytes)
                .map_err(|e| anyhow::anyhow!("Invalid public key: {}", e))?;
            let signature = Signature::from_bytes(&signature_bytes);

            verifying_key.verify(&message, &signature)
                .map_err(|e| anyhow::anyhow!("Signature verification failed for tx {}: {}", hex::encode(&tx.id), e))?;
        }
        Ok(())
    }
}

/// Parallel worker pool for high-throughput certificate processing
pub struct ParallelWorkerPool {
    config: WorkerPoolConfig,
    workers: Vec<mpsc::Sender<WorkerMessage>>,
    next_worker: Arc<RwLock<usize>>,
    stats: Arc<RwLock<PoolStats>>,
}

/// Statistics for worker pool
#[derive(Default, Debug, Clone)]
pub struct PoolStats {
    pub total_batches: u64,
    pub total_transactions: u64,
    pub total_certificates: u64,
    pub avg_latency_ms: f64,
}

impl ParallelWorkerPool {
    /// Create new worker pool with specified configuration
    pub async fn new(
        config: WorkerPoolConfig,
        narwhal: Arc<NarwhalCore>,
        simd_engine: Option<Arc<q_crypto_simd::SimdCryptoEngine>>,
    ) -> Result<Self> {
        info!("🚀 Initializing ParallelWorkerPool with {} workers", config.worker_count);
        info!("   📦 Batch size: {} transactions", config.batch_size);
        info!("   🔄 Queue size: {} transactions per worker", config.queue_size);
        info!("   ⚡ SIMD enabled: {}", config.enable_simd && simd_engine.is_some());

        let mut workers = Vec::new();

        // Spawn worker tasks
        for id in 0..config.worker_count {
            let (tx, rx) = mpsc::channel(config.queue_size);

            let worker = Worker {
                id,
                rx,
                narwhal: narwhal.clone(),
                simd_engine: simd_engine.clone(),
                config: config.clone(),
            };

            // Spawn worker task
            tokio::spawn(async move {
                worker.run().await;
            });

            workers.push(tx);
        }

        info!("✅ ParallelWorkerPool initialized with {} workers", config.worker_count);

        Ok(Self {
            config,
            workers,
            next_worker: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(PoolStats::default())),
        })
    }

    /// Submit batch of transactions for processing (round-robin worker selection)
    pub async fn submit_batch(&self, txs: Vec<Transaction>) -> Result<()> {
        if txs.is_empty() {
            return Ok(());
        }

        // Round-robin worker selection
        let mut next = self.next_worker.write().await;
        let worker_id = *next;
        *next = (*next + 1) % self.workers.len();
        drop(next);

        // Send to worker
        self.workers[worker_id]
            .send(WorkerMessage::ProcessBatch(txs))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send batch to worker: {}", e))?;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_batches += 1;

        Ok(())
    }

    /// Submit large batch split across multiple workers
    pub async fn submit_large_batch(&self, txs: Vec<Transaction>) -> Result<()> {
        let total_txs = txs.len();
        if total_txs == 0 {
            return Ok(());
        }

        info!("📦 Splitting {} transactions across {} workers", total_txs, self.workers.len());

        // Split into batches
        let chunks: Vec<_> = txs.chunks(self.config.batch_size).collect();

        // Distribute across workers
        for (i, chunk) in chunks.iter().enumerate() {
            let worker_id = i % self.workers.len();
            self.workers[worker_id]
                .send(WorkerMessage::ProcessBatch(chunk.to_vec()))
                .await
                .map_err(|e| anyhow::anyhow!("Failed to send batch to worker {}: {}", worker_id, e))?;
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_batches += chunks.len() as u64;
        stats.total_transactions += total_txs as u64;

        Ok(())
    }

    /// Get current pool statistics
    pub async fn get_stats(&self) -> PoolStats {
        (*self.stats.read().await).clone()
    }

    /// Get pool configuration
    pub fn config(&self) -> &WorkerPoolConfig {
        &self.config
    }

    /// Shutdown all workers
    pub async fn shutdown(&self) -> Result<()> {
        info!("🛑 Shutting down ParallelWorkerPool...");

        for (i, worker) in self.workers.iter().enumerate() {
            if let Err(e) = worker.send(WorkerMessage::Shutdown).await {
                warn!("⚠️  Failed to send shutdown to worker {}: {}", i, e);
            }
        }

        info!("✅ ParallelWorkerPool shutdown complete");
        Ok(())
    }
}

/// Helper function to calculate theoretical TPS
pub fn calculate_theoretical_tps(config: &WorkerPoolConfig, cert_creation_time_ms: f64) -> f64 {
    let certs_per_second_per_worker = 1000.0 / cert_creation_time_ms;
    let total_certs_per_second = certs_per_second_per_worker * config.worker_count as f64;
    total_certs_per_second * config.batch_size as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theoretical_tps_calculation() {
        let config = WorkerPoolConfig {
            worker_count: 10,
            batch_size: 10_000,
            ..Default::default()
        };

        // With 100ms certificate creation time:
        // 10 workers × 10 certs/sec × 10,000 tx/cert = 1,000,000 TPS
        let tps = calculate_theoretical_tps(&config, 100.0);
        assert_eq!(tps, 1_000_000.0);

        // With 50ms certificate creation time:
        // 10 workers × 20 certs/sec × 10,000 tx/cert = 2,000,000 TPS
        let tps_fast = calculate_theoretical_tps(&config, 50.0);
        assert_eq!(tps_fast, 2_000_000.0);
    }
}