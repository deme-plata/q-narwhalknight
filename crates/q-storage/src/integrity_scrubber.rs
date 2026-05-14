//! # Integrity Scrubber
//!
//! Background random-walk integrity verifier for the Q-NarwhalKnight chain.
//!
//! Once a node reaches the network tip it has spare IO/CPU. We use that budget
//! to detect *silent* on-disk corruption — bit-flips, cosmic-ray events, RocksDB
//! inconsistencies between the `qblock:height:<H>` block bytes and the
//! `qblock:hash:<hex(hash)>` → height inverse index.
//!
//! ## Algorithm
//!
//! For each iteration the scrubber:
//!
//! 1. Reads `current_height` from the storage height cache.
//! 2. Picks a random height `H ∈ [1, current_height]`.
//! 3. Loads the block at height `H` via `get_qblock_by_height` (returns the
//!    fully-reconstructed `QBlock`, including the slim/compressed body, quantum
//!    metadata, and transactions).
//! 4. Recomputes the BLAKE3 block hash via `block.calculate_hash()`.
//! 5. Reads the inverse index entry `qblock:hash:<hex(hash)>` from the BLOCKS
//!    CF. If it is missing or stores a height that disagrees with `H`, that is
//!    a mismatch — the on-disk block bytes no longer hash to what the index
//!    says they should.
//!
//! On mismatch the scrubber logs at `ERROR`, increments the mismatch counter,
//! and records a *repair attempt*. The actual P2P re-fetch is intentionally
//! out of scope for this module (it requires a network handle this crate does
//! not own). The TODO below marks the integration point.
//!
//! ## Concurrency
//!
//! The scrubber goes through the standard `StorageEngine` async APIs. It never
//! takes the `global_write_lock` and never blocks the writer hot path.
//!
//! ## Schema reference
//!
//! - `qblock:height:<H>` → serialized (slim, possibly LZ4) block bytes.
//! - `qblock:hash:<hex(block_hash)>` → 8-byte big-endian height.
//!
//! See `crates/q-storage/src/lib.rs::save_qblocks_batch_turbo` for the writer
//! that maintains both entries atomically.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rand::Rng;
use tokio::sync::watch;
use tokio::time::sleep;
use tracing::{debug, error, info, trace, warn};

use crate::{CF_BLOCKS, StorageEngine};

/// Default scrub rate when no override is provided.
const DEFAULT_RATE_BPS: u64 = 10;

/// How long to wait when the scrubber is disabled or the chain is empty
/// before re-checking. Kept short enough to feel responsive to operators
/// toggling the flag, long enough to avoid burning CPU when idle.
const IDLE_POLL: Duration = Duration::from_secs(5);
const EMPTY_CHAIN_POLL: Duration = Duration::from_secs(60);

/// Statistics emitted by the scrubber.
///
/// All counters are monotonically increasing across the lifetime of one
/// `IntegrityScrubber` instance. `last_scrub_time` is a Unix epoch (seconds)
/// captured at the *end* of each successful scrub iteration (clean or dirty).
#[derive(Default, Debug)]
pub struct ScrubberStats {
    pub blocks_scrubbed: AtomicU64,
    pub mismatches_found: AtomicU64,
    pub repairs_attempted: AtomicU64,
    pub repairs_succeeded: AtomicU64,
    pub last_scrub_time: AtomicU64,
}

/// Background integrity scrubber.
///
/// Spawn via `tokio::spawn(scrubber.run(shutdown_rx))` once the node is at
/// network tip. Toggle on/off with `set_enabled`, adjust pace with `set_rate`.
pub struct IntegrityScrubber {
    storage: Arc<StorageEngine>,
    /// Blocks-per-second budget. Treated as `max(1, rate)` internally; a value
    /// of zero is clamped to one to avoid divide-by-zero in the sleep math.
    rate: AtomicU64,
    /// Runtime enable flag. When `false`, `run` polls every `IDLE_POLL` and
    /// does no IO. Useful during initial sync where the chain head is still
    /// moving and `qblock:hash` rows may legitimately be racing block writes.
    enabled: AtomicBool,
    stats: Arc<ScrubberStats>,
}

impl IntegrityScrubber {
    pub fn new(storage: Arc<StorageEngine>) -> Self {
        Self {
            storage,
            rate: AtomicU64::new(DEFAULT_RATE_BPS),
            enabled: AtomicBool::new(true),
            stats: Arc::new(ScrubberStats::default()),
        }
    }

    /// Adjust the scrub rate (blocks per second). `0` is treated as `1`.
    pub fn set_rate(&self, blocks_per_sec: u64) {
        let r = blocks_per_sec.max(1);
        self.rate.store(r, Ordering::Relaxed);
    }

    /// Enable or disable the scrubber at runtime.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    pub fn stats(&self) -> &ScrubberStats {
        &self.stats
    }

    /// Compute the sleep duration between scrubs from the current rate.
    fn period(&self) -> Duration {
        let r = self.rate.load(Ordering::Relaxed).max(1);
        Duration::from_secs_f64(1.0 / (r as f64))
    }

    /// Run forever. Returns cleanly when the shutdown receiver sees a `true`
    /// value or when the sender side of the watch channel is dropped.
    pub async fn run(self, mut shutdown: watch::Receiver<bool>) {
        info!("🧹 [SCRUBBER] Background integrity scrubber starting (rate={} bps, enabled={})",
              self.rate.load(Ordering::Relaxed),
              self.enabled.load(Ordering::Relaxed));

        // `thread_rng` is auto-seeded from OS entropy and re-seeded
        // periodically. We only use it for non-cryptographic block-height
        // sampling, so this is the right knob — no extra feature flags
        // needed on the workspace `rand` dep.
        loop {
            // Shutdown check first — if a watch::Sender::send(true) raced our
            // last sleep we want to bail out immediately.
            if *shutdown.borrow() {
                info!("🧹 [SCRUBBER] Shutdown signal received — exiting");
                return;
            }

            if !self.enabled.load(Ordering::Relaxed) {
                // Disabled — do nothing, just wait for shutdown or re-enable.
                if Self::wait_or_shutdown(IDLE_POLL, &mut shutdown).await {
                    return;
                }
                continue;
            }

            // Pick a height. `get_highest_contiguous_block` returns the
            // verified contiguous tip — anything above it may legitimately
            // not have a hash index yet (still being filled by turbo sync).
            let tip = match self.storage.get_highest_contiguous_block().await {
                Ok(h) => h,
                Err(e) => {
                    warn!("🧹 [SCRUBBER] Failed to read tip: {} — backing off", e);
                    if Self::wait_or_shutdown(IDLE_POLL, &mut shutdown).await {
                        return;
                    }
                    continue;
                }
            };

            if tip == 0 {
                debug!("🧹 [SCRUBBER] Chain empty (tip=0) — sleeping {}s", EMPTY_CHAIN_POLL.as_secs());
                if Self::wait_or_shutdown(EMPTY_CHAIN_POLL, &mut shutdown).await {
                    return;
                }
                continue;
            }

            // Random sample in [1, tip]. We acquire `thread_rng` per
            // iteration; it's a cheap thread-local lookup.
            let height = rand::thread_rng().gen_range(1..=tip);

            self.scrub_one(height).await;

            // Rate limit.
            let period = self.period();
            if Self::wait_or_shutdown(period, &mut shutdown).await {
                return;
            }
        }
    }

    /// Scrub exactly one block at `height`. Public(`pub(crate)`) so tests in
    /// this module can drive it deterministically without spinning the full
    /// `run` loop.
    pub(crate) async fn scrub_one(&self, height: u64) {
        // Load the block. If it's missing (e.g. inside a known-corrupt gap
        // that turbo sync will refill), skip silently — that is a separate
        // failure mode handled by the gap-fill path.
        let block_opt = match self.storage.get_qblock_by_height(height).await {
            Ok(b) => b,
            Err(e) => {
                debug!("🧹 [SCRUBBER] Read error at height {} — skipping: {}", height, e);
                return;
            }
        };

        let block = match block_opt {
            Some(b) => b,
            None => {
                trace!("🧹 [SCRUBBER] No block at height {} (gap) — skipping", height);
                return;
            }
        };

        // Recompute the block hash. `calculate_hash` does BLAKE3 over the
        // serialized header. If the on-disk header bytes have been mutated
        // (bit flip, RocksDB SST corruption survived checksum), the value we
        // get back here will differ from what was indexed at write time.
        let computed_hash = block.calculate_hash();
        let hash_hex = hex::encode(computed_hash);
        let hash_key = format!("qblock:hash:{}", hash_hex);

        // Read the inverse index. The value is an 8-byte big-endian height.
        let indexed_height_opt = match self.storage.hot_db.get(CF_BLOCKS, hash_key.as_bytes()).await {
            Ok(v) => v,
            Err(e) => {
                debug!("🧹 [SCRUBBER] Index read error for height {}: {} — skipping", height, e);
                return;
            }
        };

        let mismatch = match indexed_height_opt {
            None => {
                // The hash index has no entry for this block hash. Two
                // possibilities: (a) the block bytes have been corrupted so
                // their hash no longer matches what was indexed, or (b) the
                // hash row was deleted/never written. Either way, this is a
                // consistency violation worth surfacing.
                error!(
                    "🚨 [SCRUBBER] CORRUPTION: height={} computed_hash={} has no inverse index entry",
                    height, hash_hex
                );
                true
            }
            Some(bytes) if bytes.len() != 8 => {
                error!(
                    "🚨 [SCRUBBER] CORRUPTION: height={} computed_hash={} index value has wrong length ({} bytes)",
                    height, hash_hex, bytes.len()
                );
                true
            }
            Some(bytes) => {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&bytes);
                let indexed_height = u64::from_be_bytes(arr);
                if indexed_height != height {
                    error!(
                        "🚨 [SCRUBBER] CORRUPTION: height={} indexed_height={} computed_hash={} — block bytes do not match the height pointer",
                        height, indexed_height, hash_hex
                    );
                    true
                } else {
                    false
                }
            }
        };

        // Update stats unconditionally — even a mismatch counts as a scrub.
        self.stats.blocks_scrubbed.fetch_add(1, Ordering::Relaxed);
        if let Ok(now) = SystemTime::now().duration_since(UNIX_EPOCH) {
            self.stats.last_scrub_time.store(now.as_secs(), Ordering::Relaxed);
        }

        if mismatch {
            self.stats.mismatches_found.fetch_add(1, Ordering::Relaxed);
            // TODO(integrity-scrubber): wire P2P re-fetch here. The scrubber
            // does not own a network handle in this crate, so for now we log
            // and increment the repair-attempt counter so operators can see
            // the scrubber tried. A follow-up PR should plumb in a callback
            // (e.g. `Arc<dyn BlockRefetcher>`) that turbo_sync provides.
            self.stats.repairs_attempted.fetch_add(1, Ordering::Relaxed);
            warn!(
                "🧹 [SCRUBBER] repair-attempted height={} (P2P re-fetch not yet wired; see TODO)",
                height
            );
        }
    }

    /// Wait for `dur` or until the shutdown channel fires. Returns `true` if
    /// shutdown fired (caller should exit), `false` if the timer elapsed.
    async fn wait_or_shutdown(dur: Duration, shutdown: &mut watch::Receiver<bool>) -> bool {
        tokio::select! {
            _ = sleep(dur) => false,
            res = shutdown.changed() => {
                // Sender dropped — treat as shutdown.
                if res.is_err() {
                    return true;
                }
                *shutdown.borrow()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QStorage;
    use q_types::block::{BlockHeader, QBlock, QuantumMetadata, VDFProof};
    use tempfile::TempDir;

    fn make_block(height: u64) -> QBlock {
        QBlock {
            header: BlockHeader {
                height,
                phase: 11,
                // Use a network_id that bypasses the per-network genesis
                // timestamp filter — we set timestamp=0 below which already
                // satisfies the filter, but a recognizable id helps debugging.
                network_id: "scrubber-test".to_string(),
                prev_block_hash: [0u8; 32],
                solutions_root: [0u8; 32],
                tx_root: [0u8; 32],
                state_root: [0u8; 32],
                // timestamp=0 → bypasses the `timestamp > 0 && timestamp < genesis_ts`
                // pre-genesis filter in `save_qblocks_batch_turbo`.
                timestamp: 0,
                dag_round: height,
                vdf_proof: VDFProof::default(),
                anchor_validator: None,
                proposer: [0u8; 32],
                producer_id: 0,
                total_difficulty: 0,
                producer_public_key: None,
                producer_signature: None,
                coinbase_merkle_root: None,
                total_coinbase_reward: None,
                coinbase_count: None,
            },
            mining_solutions: Vec::new(),
            dag_parents: Vec::new(),
            quantum_metadata: QuantumMetadata::default(),
            transactions: Vec::new(),
            balance_updates: Vec::new(),
            size_bytes: 0,
        }
    }

    async fn open_temp_storage() -> (TempDir, Arc<QStorage>) {
        let dir = TempDir::new().expect("temp dir");
        let node_id = [7u8; 32];
        let storage = QStorage::open(dir.path(), node_id).await.expect("open storage");
        (dir, Arc::new(storage))
    }

    async fn write_blocks(storage: &QStorage, start: u64, count: u64) -> Vec<QBlock> {
        let blocks: Vec<QBlock> = (start..start + count).map(make_block).collect();
        storage
            .save_qblocks_batch_turbo(&blocks)
            .await
            .expect("turbo batch save");
        blocks
    }

    #[tokio::test]
    async fn test_scrubber_detects_corruption() {
        let (_dir, storage) = open_temp_storage().await;
        let blocks = write_blocks(&storage, 1, 1).await;
        let block = &blocks[0];
        let height = block.header.height;

        // Verify the hash index was written as expected.
        let real_hash = block.calculate_hash();
        let real_key = format!("qblock:hash:{}", hex::encode(real_hash));
        assert!(
            storage
                .hot_db
                .get(CF_BLOCKS, real_key.as_bytes())
                .await
                .expect("get hash key")
                .is_some(),
            "real hash index entry must exist before corruption"
        );

        // Inject corruption: delete the real hash row so the recomputed hash
        // no longer maps to anything. This simulates the "block bytes intact
        // but inverse index got nuked" scenario, which the scrubber must
        // surface as a mismatch.
        storage
            .hot_db
            .delete(CF_BLOCKS, real_key.as_bytes())
            .await
            .expect("delete hash key");

        let scrubber = IntegrityScrubber::new(storage.clone());
        scrubber.scrub_one(height).await;

        assert_eq!(
            scrubber.stats().mismatches_found.load(Ordering::Relaxed),
            1,
            "scrubber must detect missing inverse index entry"
        );
        assert_eq!(
            scrubber.stats().repairs_attempted.load(Ordering::Relaxed),
            1,
            "scrubber must attempt repair on mismatch"
        );
        assert_eq!(
            scrubber.stats().blocks_scrubbed.load(Ordering::Relaxed),
            1,
            "exactly one block scrubbed"
        );
    }

    #[tokio::test]
    async fn test_scrubber_passes_clean_blocks() {
        let (_dir, storage) = open_temp_storage().await;
        let _blocks = write_blocks(&storage, 1, 10).await;

        let scrubber = IntegrityScrubber::new(storage.clone());

        // Run 5 deterministic scrubs across heights 1..=10. We deliberately
        // sample several heights rather than relying on the RNG in `run`.
        for h in [1u64, 3, 5, 7, 9] {
            scrubber.scrub_one(h).await;
        }

        let scrubbed = scrubber.stats().blocks_scrubbed.load(Ordering::Relaxed);
        let mismatches = scrubber.stats().mismatches_found.load(Ordering::Relaxed);

        assert!(
            scrubbed >= 1,
            "at least one block should be scrubbed (got {})",
            scrubbed
        );
        assert_eq!(
            mismatches, 0,
            "clean blocks must not produce mismatches (got {})",
            mismatches
        );
    }

    #[tokio::test]
    async fn test_scrubber_respects_disabled() {
        let (_dir, storage) = open_temp_storage().await;
        let _blocks = write_blocks(&storage, 1, 5).await;

        let scrubber = Arc::new(IntegrityScrubber::new(storage.clone()));
        scrubber.set_enabled(false);
        // Crank rate so that *if* the scrubber were active we'd see many
        // iterations in 100ms — guarantees the disabled gate is what's
        // suppressing work.
        scrubber.set_rate(1000);

        let stats_handle = scrubber.stats.clone();
        let (tx, rx) = watch::channel(false);

        // Take the scrubber by value into the spawned task (run consumes self).
        let scrubber_owned = Arc::try_unwrap(scrubber)
            .map_err(|_| "scrubber Arc had other refs")
            .expect("unique scrubber");
        let handle = tokio::spawn(async move { scrubber_owned.run(rx).await });

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Shut it down cleanly.
        tx.send(true).expect("send shutdown");
        handle.await.expect("scrubber task joined");

        assert_eq!(
            stats_handle.blocks_scrubbed.load(Ordering::Relaxed),
            0,
            "disabled scrubber must not scrub anything"
        );
    }

    #[tokio::test]
    async fn test_scrubber_rate_limit() {
        let (_dir, storage) = open_temp_storage().await;
        // Write enough blocks that the random walk has plenty of targets.
        let _blocks = write_blocks(&storage, 1, 20).await;

        let scrubber = IntegrityScrubber::new(storage.clone());
        scrubber.set_rate(2); // 2 blocks/sec → ~3 scrubs in 1.5s window

        let stats_handle = scrubber.stats.clone();
        let (tx, rx) = watch::channel(false);

        let handle = tokio::spawn(async move { scrubber.run(rx).await });

        tokio::time::sleep(Duration::from_millis(1500)).await;

        tx.send(true).expect("send shutdown");
        handle.await.expect("scrubber task joined");

        let scrubbed = stats_handle.blocks_scrubbed.load(Ordering::Relaxed);
        assert!(
            (2..=4).contains(&scrubbed),
            "expected 2..=4 scrubs at 2 bps over 1.5s, got {}",
            scrubbed
        );
    }
}
