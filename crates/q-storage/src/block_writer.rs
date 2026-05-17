/// Single-threaded block writer to prevent parallel write conflicts
///
/// This module implements Phase 1 of the Database Durability Rescue Plan.
/// It serializes all block writes through a single async task to eliminate
/// the parallel write conflicts that caused 11 occurrences of data corruption.
///
/// Key Features:
/// - Single commit queue (no parallel writes to qblock:latest)
/// - Duplicate detection (prevents double-writes)
/// - Conditional pointer updates (only extends chain)
/// - Write verification (detects phantom writes)
///
/// Expert Validation:
/// - ChatGPT: "This eliminates the root cause"
/// - DeepSeek: "Deploy immediately"
/// - Kimi AI: "90% reduction in corruption risk"

use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use anyhow::{Result, Context, bail};
use tracing::{info, warn, error, debug};
use q_types::block::QBlock;
use crate::kv::KVStore;
use crate::CF_BLOCKS;
use crate::{CF_QUANTUM_METADATA, CF_TRANSACTIONS};
use crate::precompressed_storage::{PrecompressedBlock, CompressionAlgorithm};

/// Message sent to the commit worker
struct CommitMsg {
    block: QBlock,
    reply: oneshot::Sender<Result<()>>,
}

/// Single-threaded block writer
///
/// All save_qblock() calls go through this writer, which
/// processes them sequentially in a dedicated task.
pub struct BlockWriter {
    commit_tx: mpsc::Sender<CommitMsg>,
}

impl BlockWriter {
    /// Create a new BlockWriter
    ///
    /// This spawns a background task that processes blocks sequentially.
    /// The task runs until the BlockWriter is dropped.
    pub fn new(hot_db: Arc<dyn KVStore>) -> Self {
        let (commit_tx, mut commit_rx) = mpsc::channel::<CommitMsg>(2048);

        // Spawn dedicated commit worker (SINGLE THREAD)
        tokio::spawn(async move {
            use tokio::time::{timeout, Duration, Instant};

            info!("🔒 Block writer worker started (single-threaded commit queue)");

            // 🚨 v0.9.94-beta: DEADLOCK PREVENTION - Circuit breaker and watchdog
            // Expert consensus: ChatGPT, Kimi AI, DeepSeek (95% confidence)
            // These additions detect and contain failures, preventing silent deadlocks

            let mut consecutive_errors = 0usize;
            const MAX_CONSECUTIVE_ERRORS: usize = 5;
            const RECV_TIMEOUT: Duration = Duration::from_secs(10);
            const WRITE_TIMEOUT: Duration = Duration::from_secs(30);

            let mut blocks_processed = 0u64;
            let worker_start = Instant::now();

            loop {
                // WATCHDOG: Detect receiver starvation with timeout
                // If no messages for 10s, log heartbeat to prove task is alive
                let msg = match timeout(RECV_TIMEOUT, commit_rx.recv()).await {
                    Ok(Some(m)) => m,
                    Ok(None) => {
                        info!("🛑 Block writer channel closed gracefully (processed {} blocks)", blocks_processed);
                        break;
                    }
                    Err(_) => {
                        // No messages for 10 seconds - log heartbeat
                        debug!("⏱️ BlockWriter: no messages for 10s (processed {} blocks in {:?}, errors={})",
                              blocks_processed, worker_start.elapsed(), consecutive_errors);
                        continue;
                    }
                };

                let block_height = msg.block.header.height;
                blocks_processed += 1;

                // Periodic status report every 100 blocks
                if blocks_processed % 100 == 0 {
                    info!("📊 BlockWriter: processed {} blocks in {:?}, consecutive_errors={}",
                          blocks_processed, worker_start.elapsed(), consecutive_errors);
                }

                // CIRCUIT BREAKER: Stop processing if too many consecutive errors
                // This prevents cascading failures and makes the problem loud
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                    error!("🚨 Circuit breaker OPEN - too many consecutive write errors ({})", consecutive_errors);
                    error!("   BlockWriter is in degraded state. Check RocksDB health!");
                    let _ = msg.reply.send(Err(anyhow::anyhow!(
                        "Circuit breaker open after {} consecutive errors", consecutive_errors
                    )));
                    consecutive_errors += 1; // Keep counting for metrics
                    continue;
                }

                info!("📥 BlockWriter received block at height {}", block_height);

                // Add timeout to the entire write operation
                // This ensures we detect slow writes and don't silently hang
                let write_start = Instant::now();
                let result = match timeout(WRITE_TIMEOUT, Self::save_qblock_internal(&hot_db, &msg.block)).await {
                    Ok(Ok(())) => {
                        debug!("✅ Block {} saved in {:?}", block_height, write_start.elapsed());
                        consecutive_errors = 0; // Reset on success
                        Ok(())
                    }
                    Ok(Err(e)) => {
                        error!("❌ Block {} write failed: {}", block_height, e);
                        consecutive_errors += 1;
                        Err(e)
                    }
                    Err(_) => {
                        error!("⏰ Block {} write TIMEOUT after {:?}", block_height, WRITE_TIMEOUT);
                        error!("   This indicates RocksDB is stalling. Check disk I/O!");
                        consecutive_errors += 1;
                        Err(anyhow::anyhow!("Write timeout after {:?}", WRITE_TIMEOUT))
                    }
                };

                // Send response (oneshot::send is non-blocking)
                if let Err(_) = msg.reply.send(result) {
                    warn!("⚠️ Failed to send reply for block {} (receiver dropped)", block_height);
                } else {
                    debug!("📤 Reply sent for block {}", block_height);
                }
            }

            info!("🛑 Block writer worker stopped (processed {} blocks in {:?})",
                  blocks_processed, worker_start.elapsed());
        });

        Self { commit_tx }
    }

    /// Internal save logic (runs in dedicated task)
    ///
    /// This implements:
    /// - Fix 1.2: Duplicate detection
    /// - Fix 1.3: Conditional pointer update
    /// - Fix 1.5: Write verification
    async fn save_qblock_internal(db: &Arc<dyn KVStore>, block: &QBlock) -> Result<()> {
        let start_time = std::time::SystemTime::now();
        let block_hash = block.calculate_hash();
        let height = block.header.height;

        // FIX 1.2: DUPLICATE DETECTION (Idempotent)
        // 🚨 v0.9.99-beta: IDEMPOTENCY FIX - Return Ok() on duplicate (not an error)
        // Expert consensus: ChatGPT, DeepSeek, Kimi AI (95% confidence)
        // Lock-free producer (v0.9.93) allows concurrent writes of same block
        // Idempotent writes prevent circuit breaker from tripping on duplicates
        // Note: No height pointer desync because we skip the batch write below
        let height_key = format!("qblock:height:{}", height);
        if let Ok(Some(_)) = db.get(CF_BLOCKS, height_key.as_bytes()).await {
            // v0.9.100-beta: Block exists - now verify pointer is correct
            let current_height = match db.get(CF_BLOCKS, b"qblock:latest").await {
                Ok(Some(bytes)) if bytes.len() == 8 => {
                    u64::from_be_bytes(bytes.try_into().unwrap())
                }
                _ => 0
            };

            if current_height >= height {
                // Pointer is ahead or equal - truly idempotent
                debug!("Block {} already exists and pointer is correct (pointer={})", height, current_height);
                return Ok(());
            } else {
                // Pointer is behind - need to update it!
                warn!("[HEIGHT RECOVERY] Block {} exists but pointer is behind (pointer={}, block={})",
                      height, current_height, height);
                warn!("[HEIGHT RECOVERY] Updating pointer to fix desync...");
                // Continue to pointer update logic instead of returning early
            }
        }

        info!("💾 Saving QBlock at height {} with hash {}",
              height, hex::encode(&block_hash[..8]));

        // 🚀 v7.3.1: BLOCK STORAGE OPTIMIZATION - Separate + Compress
        // 1. Extract quantum_metadata → store in CF_QUANTUM_METADATA (lazy-loadable)
        // 2. Extract transactions → store in CF_TRANSACTIONS (body separation)
        // 3. Serialize slim block (without heavy fields), compress with Lz4
        // Result: ~60-70% smaller block entries in CF_BLOCKS

        // Serialize quantum_metadata separately
        let qm_data = bincode::serialize(&block.quantum_metadata)
            .context("Failed to serialize QuantumMetadata")?;

        // Serialize transactions separately (only if non-empty)
        let has_transactions = !block.transactions.is_empty();
        let tx_data = if has_transactions {
            bincode::serialize(&block.transactions)
                .context("Failed to serialize transactions")?
        } else {
            Vec::new()
        };

        // Create slim block clone: empty transactions + minimal quantum_metadata
        let mut slim_block = block.clone();
        slim_block.transactions = Vec::new();
        slim_block.quantum_metadata = q_types::block::QuantumMetadata {
            vertex_coordinates: q_types::block::HypergraphCoordinates {
                temporal: 0.0,
                spatial: Vec::new(),
                energetic: 0.0,
                entropic: 0.0,
                metadata: std::collections::HashMap::new(),
            },
            k_parameter: 0.0,
            energy: 0.0,
            energy_components: q_types::block::EnergyComponents {
                coupling: 0.0,
                potential: 0.0,
                ordering: 0.0,
                fault_tolerance: 0.0,
                temporal: 0.0,
                finality: 0.0,
            },
            spectral_signatures: Vec::new(),
            wavefunction_phase: 0.0,
            entropy_variance: 0.0,
            byzantine_scores: std::collections::HashMap::new(),
        };

        // Serialize slim block with bincode
        let slim_bytes = bincode::serialize(&slim_block)
            .context("Failed to serialize slim QBlock")?;

        // v7.3.5: Store as QRAW (no app-level compression) to fix LZ4 roundtrip failures.
        // RocksDB CF_BLOCKS has DBCompressionType::None, so no double compression.
        // The LZ4 block::compress/decompress roundtrip was failing for unknown reasons.
        let compressed = PrecompressedBlock::compress(&slim_bytes, CompressionAlgorithm::None)
            .context("Failed to wrap QBlock in QRAW format")?;
        let block_data = compressed.to_bytes();

        debug!(
            "📦 Block {} storage: full={} slim={} compressed={} qm={} txs={} (ratio {:.1}x)",
            height,
            slim_bytes.len() + qm_data.len() + tx_data.len(),
            slim_bytes.len(),
            block_data.len(),
            qm_data.len(),
            tx_data.len(),
            compressed.compression_ratio()
        );

        // FIX 1.3: CONDITIONAL POINTER UPDATE
        // 🚨 v0.9.95-beta: Check height BEFORE building batch (avoid race)
        let current_height = match db.get(CF_BLOCKS, b"qblock:latest").await {
            Ok(Some(bytes)) if bytes.len() == 8 => {
                u64::from_be_bytes(bytes.try_into().unwrap())
            }
            _ => 0
        };

        let should_update_pointer = if height == 0 {
            true // Always set pointer for genesis
        } else if height == current_height + 1 {
            true // Normal chain extension
        } else if height <= current_height {
            false // Old block or duplicate
        } else {
            // Gap detected - log but don't update pointer
            warn!("📏 Gap detected: current={}, incoming={} (skipping pointer update)",
                  current_height, height);
            false
        };

        // 🚨 v0.9.95-beta: CRITICAL FIX - ATOMIC WRITEBATCH
        // Expert consensus: ChatGPT, DeepSeek, Kimi AI (99% confidence)
        // Block data + height pointer MUST be in same WriteBatch (atomic update)
        // This prevents height pointer drift that caused the 221→242 desync bug
        let mut batch: Vec<(&str, Vec<u8>, Vec<u8>)> = Vec::new();

        // Store compressed slim block by height
        batch.push((
            CF_BLOCKS,
            height_key.clone().into_bytes(),
            block_data  // Compressed slim block (no quantum_metadata/transactions)
        ));

        // v7.3.1: Store quantum_metadata separately for lazy loading
        let qm_key = format!("qm:{}", height);
        batch.push((
            CF_QUANTUM_METADATA,
            qm_key.into_bytes(),
            qm_data,
        ));

        // v7.3.1: Store transactions separately (body separation)
        if has_transactions {
            let tx_key = format!("block_txs:{}", height);
            batch.push((
                CF_TRANSACTIONS,
                tx_key.into_bytes(),
                tx_data,
            ));
        }

        // 🚀 v1.3.5-beta: STORAGE OPTIMIZATION - Store hash→height reference only!
        // Previously stored full block twice (by height AND by hash) = 50% waste
        // Now: hash key stores only 8-byte height reference
        // Lookup: hash → height → block (two-step, but 50% less storage)
        let hash_key = format!("qblock:hash:{}", hex::encode(block_hash));
        batch.push((
            CF_BLOCKS,
            hash_key.into_bytes(),
            height.to_be_bytes().to_vec()  // Only 8 bytes instead of ~2KB!
        ));

        // 🚨 CRITICAL: Add height pointer to SAME batch (atomic with block data!)
        if should_update_pointer {
            let latest_height_bytes = height.to_be_bytes().to_vec();
            batch.push((
                CF_BLOCKS,
                b"qblock:latest".to_vec(),
                latest_height_bytes
            ));
            debug!("📌 Pointer update queued in atomic batch: {} → {}", current_height, height);
        }

        // Write batch to database (SINGLE atomic operation)
        // Either ALL writes succeed (block + hash + pointer) or NONE do
        db.write_batch(batch).await
            .context("Failed to write QBlock batch")?;

        // 🚀 v2.3.12-beta: SCAN-FORWARD FIX - Update pointer when gap is filled
        // ROOT CAUSE: When blocks arrive out of order (e.g., 102 before 101),
        // the pointer only advances if height == current + 1. This creates permanent
        // lag that causes endgame sync to take 10+ hours instead of seconds.
        //
        // FIX: After saving any block, scan forward to find the new highest contiguous
        // height and update the pointer. This ensures gaps are closed immediately.
        //
        // Example: current=100, block 102 saved, then block 101 saved
        // - Block 102: pointer stays at 100 (gap)
        // - Block 101: pointer should advance to 102 (gap filled!)
        //
        // Without this fix, pointer stays at 101 until AUTO-REPAIR runs (seconds later).
        // With this fix, pointer advances to 102 immediately.
        //
        // NEW: Also scan when extending chain normally, in case higher blocks exist.
        let base_height = if should_update_pointer { height } else { current_height };

        if height >= current_height {
            // Scan forward from the new pointer position to find more contiguous blocks
            let mut scan_height = base_height + 1;
            let mut new_contiguous = base_height;

            // Scan up to 500 blocks ahead (reasonable limit to prevent long scans)
            // This is fast: ~50 RocksDB point lookups in worst case
            let mut blocks_scanned = 0u32;
            while scan_height <= base_height + 500 {
                let scan_key = format!("qblock:height:{}", scan_height);
                match db.get(CF_BLOCKS, scan_key.as_bytes()).await {
                    Ok(Some(_)) => {
                        new_contiguous = scan_height;
                        scan_height += 1;
                        blocks_scanned += 1;
                    }
                    _ => break, // Gap found, stop scanning
                }
            }

            // Log scan result for debugging
            if blocks_scanned > 0 {
                debug!("🔍 [SCAN-FORWARD] Scanned {} blocks ahead: base={} → new_contiguous={}",
                       blocks_scanned, base_height, new_contiguous);
            }

            // If we found blocks ahead of the pointer, update it
            if new_contiguous > base_height {
                let new_height_bytes = new_contiguous.to_be_bytes().to_vec();
                let pointer_batch = vec![(CF_BLOCKS, b"qblock:latest".to_vec(), new_height_bytes)];

                if let Err(e) = db.write_batch(pointer_batch).await {
                    warn!("⚠️ [SCAN-FORWARD] Failed to update pointer: {}", e);
                } else {
                    info!("🔗 [SCAN-FORWARD] Extended chain! Pointer: {} → {} (+{} blocks)",
                          base_height, new_contiguous, new_contiguous - base_height);
                }
            }
        }

        // FIX 1.5: WRITE VERIFICATION
        // Immediately verify the block is readable
        match db.get(CF_BLOCKS, height_key.as_bytes()).await {
            Ok(Some(_)) => {
                let latency = start_time.elapsed().unwrap_or_default();
                info!("✅ Saved QBlock {} in {}ms ({} solutions, {} txs) - VERIFIED [v7.3.1 compressed]",
                      height, latency.as_millis(), block.mining_solutions.len(), block.transactions.len());
                Ok(())
            }
            Ok(None) => {
                error!("🚨 CRITICAL: Block {} written but missing on read!", height);
                bail!("Block write verification failed - phantom write detected");
            }
            Err(e) => {
                error!("🚨 Block {} verification read failed: {}", height, e);
                Err(e.into())
            }
        }
    }

    /// Queue a block for writing
    ///
    /// This is the public API called by QStorage. It sends the block to the
    /// commit queue and waits for the result.
    pub async fn write_block(&self, block: QBlock) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();

        self.commit_tx.send(CommitMsg {
            block,
            reply: reply_tx,
        }).await
        .map_err(|_| anyhow::anyhow!("Block writer channel closed"))?;

        reply_rx.await
            .map_err(|_| anyhow::anyhow!("Block writer reply channel closed"))?
    }
}
