/// sled-based KV store implementation for Windows cross-compilation
/// Provides same KVStore trait using pure-Rust embedded database
#[cfg(target_os = "windows")]
use anyhow::{Context, Result};
#[cfg(target_os = "windows")]
use async_trait::async_trait;
#[cfg(target_os = "windows")]
use q_quantum_rng::{QRNGConfig, QuantumRNG};
#[cfg(target_os = "windows")]
use q_types::Phase;
#[cfg(target_os = "windows")]
use sled::Db;
#[cfg(target_os = "windows")]
use std::{path::Path, sync::Arc};
#[cfg(target_os = "windows")]
use tracing::{debug, info, warn};

#[cfg(target_os = "windows")]
use super::kv::KVStore;

/// sled-based KV store for Windows
#[cfg(target_os = "windows")]
pub struct RocksDBKV {
    db: Arc<Db>,
    db_path: String,
    qrng: Option<Arc<QuantumRNG>>,
    phase: Phase,
}

#[cfg(target_os = "windows")]
impl RocksDBKV {
    pub async fn open_hot_db<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_hot_db_with_phase(path, Phase::Phase0).await
    }

    pub async fn open_hot_db_with_phase<P: AsRef<Path>>(path: P, phase: Phase) -> Result<Self> {
        let path = path.as_ref();
        info!("💾 Opening sled database (Windows) at {:?} for {:?}", path, phase);

        // Limit Sled's memory usage to prevent OOM on Windows:
        // - cache_capacity: page cache limit (configurable via SLED_CACHE_MB)
        // - mode(LowSpace): prioritize disk usage over memory
        // NOTE: segment_size CANNOT be changed on existing databases (Sled rejects it)
        // v9.1.7: Reduced default from 256→128 MB. At height 5M+ the old default caused
        // OOM on 8-16GB Windows machines (sled overshoots cache_capacity under burst writes).
        // Users with >=32GB RAM can set SLED_CACHE_MB=256 for better read performance.
        let cache_mb: u64 = std::env::var("SLED_CACHE_MB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(128);
        info!("💾 Sled page cache limit: {} MB (mode: LowSpace)", cache_mb);
        let db = sled::Config::new()
            .path(path)
            .cache_capacity(cache_mb * 1024 * 1024)
            .mode(sled::Mode::LowSpace)
            .flush_every_ms(Some(1000))
            .open()
            .context("Failed to open sled database")?;

        let qrng = if matches!(phase, Phase::Phase2 | Phase::Phase3 | Phase::Phase4) {
            info!("🌌 Initializing quantum RNG for storage encryption");
            let config = QRNGConfig {
                min_entropy_quality: 0.99,
                pool_size: 8192,
                polling_interval_ms: 100,
                ..Default::default()
            };

            match QuantumRNG::new(phase, config).await {
                Ok(qrng) => {
                    info!("✅ Quantum RNG initialized for storage encryption");
                    Some(Arc::new(qrng))
                }
                Err(e) => {
                    warn!("⚠️ Failed to initialize storage QRNG: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            db: Arc::new(db),
            db_path: path.to_string_lossy().to_string(),
            qrng,
            phase,
        })
    }

    pub async fn open_cold_db<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        info!("🧊 Opening sled cold database (Windows) at {:?}", path);

        // Cold DB uses smaller cache (64MB) since it's accessed less frequently
        let db = sled::Config::new()
            .path(path)
            .cache_capacity(64u64 * 1024 * 1024)
            .mode(sled::Mode::LowSpace)
            .flush_every_ms(Some(5000))
            .open()
            .context("Failed to open sled cold database")?;

        Ok(Self {
            db: Arc::new(db),
            db_path: path.to_string_lossy().to_string(),
            qrng: None,
            phase: Phase::Phase0,
        })
    }

    fn get_tree(&self, cf: &str) -> Result<sled::Tree> {
        self.db
            .open_tree(cf)
            .context(format!("Failed to open tree '{}'", cf))
    }

    /// Get a raw database handle (returns Arc<()> on Windows since there's no RocksDB)
    pub fn get_raw_db(&self) -> Arc<()> {
        Arc::new(())
    }

    /// v6.1.1: Stub for RocksDB memory usage reporting (not applicable to Sled)
    pub fn get_memory_usage_mb(&self) -> (f64, f64, f64) {
        (0.0, 0.0, 0.0)
    }

    pub async fn get_stats(&self) -> Result<SledStats> {
        let mut total_size = 0u64;
        for name in self.db.tree_names() {
            if let Ok(tree) = self.db.open_tree(&name) {
                total_size += tree.len() as u64;
            }
        }

        Ok(SledStats {
            total_keys: total_size,
            total_size: self.db.size_on_disk().unwrap_or(0),
        })
    }

    pub async fn create_checkpoint<P: AsRef<Path>>(&self, checkpoint_path: P) -> Result<()> {
        // sled doesn't have built-in checkpoint, flush and copy
        self.db.flush_async().await.context("Failed to flush sled")?;
        info!("💾 sled checkpoint (flush) completed");
        Ok(())
    }
}

#[cfg(target_os = "windows")]
#[async_trait]
impl KVStore for RocksDBKV {
    async fn put(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()> {
        let tree = self.get_tree(cf)?;
        tree.insert(key, value).context("sled put failed")?;
        Ok(())
    }

    async fn put_sync(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()> {
        let tree = self.get_tree(cf)?;
        tree.insert(key, value).context("sled put failed")?;
        // Sled flushes to disk automatically, so we just ensure it's written
        tree.flush_async().await.context("sled flush failed")?;
        Ok(())
    }

    async fn get(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let tree = self.get_tree(cf)?;
        let result = tree.get(key).context("sled get failed")?;
        Ok(result.map(|ivec| ivec.to_vec()))
    }

    async fn delete(&self, cf: &str, key: &[u8]) -> Result<()> {
        let tree = self.get_tree(cf)?;
        tree.remove(key).context("sled delete failed")?;
        Ok(())
    }

    async fn write_batch(&self, batch: Vec<(&str, Vec<u8>, Vec<u8>)>) -> Result<()> {
        // Group operations by tree
        let mut trees: std::collections::HashMap<String, Vec<(Vec<u8>, Vec<u8>)>> =
            std::collections::HashMap::new();

        for (cf_name, key, value) in batch {
            trees
                .entry(cf_name.to_string())
                .or_insert_with(Vec::new)
                .push((key, value));
        }

        // Execute batches per tree
        for (cf_name, ops) in trees {
            let tree = self.get_tree(&cf_name)?;
            let mut batch = sled::Batch::default();

            for (key, value) in ops {
                batch.insert(key, value);
            }

            tree.apply_batch(batch)
                .context("sled batch write failed")?;
        }

        Ok(())
    }

    async fn write_batch_bulk(&self, batch: Vec<(&str, Vec<u8>, Vec<u8>)>) -> Result<()> {
        // Sled is already async and optimized, so bulk mode is same as regular
        // (no explicit fsync control in sled)
        self.write_batch(batch).await
    }

    async fn scan_prefix(&self, cf: &str, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let tree = self.get_tree(cf)?;
        // Cap at 100K entries to prevent OOM from unbounded scans
        const MAX_SCAN_RESULTS: usize = 100_000;
        let mut results = Vec::with_capacity(1024.min(MAX_SCAN_RESULTS));

        for item in tree.scan_prefix(prefix) {
            let (key, value) = item.context("Iterator error")?;
            results.push((key.to_vec(), value.to_vec()));
            if results.len() >= MAX_SCAN_RESULTS {
                warn!("⚠️ scan_prefix('{}') hit {} entry limit - truncating", cf, MAX_SCAN_RESULTS);
                break;
            }
        }

        Ok(results)
    }

    async fn scan_all(&self, cf: &str) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let tree = self.get_tree(cf)?;
        // Cap at 100K entries to prevent OOM from unbounded scans
        const MAX_SCAN_RESULTS: usize = 100_000;
        let mut results = Vec::with_capacity(1024.min(MAX_SCAN_RESULTS));

        for item in tree.iter() {
            let (key, value) = item.context("Iterator error")?;
            results.push((key.to_vec(), value.to_vec()));
            if results.len() >= MAX_SCAN_RESULTS {
                warn!("⚠️ scan_all('{}') hit {} entry limit - truncating", cf, MAX_SCAN_RESULTS);
                break;
            }
        }

        Ok(results)
    }

    async fn flush(&self) -> Result<()> {
        self.db.flush_async().await.context("sled flush failed")?;
        Ok(())
    }

    async fn compact(&self) -> Result<()> {
        // sled auto-compacts
        debug!("🗜️ sled compaction is automatic (no-op)");
        Ok(())
    }

    async fn get_db_size(&self) -> Result<u64> {
        Ok(self.db.size_on_disk().unwrap_or(0))
    }

    async fn write_batch_turbo(&self, batch: Vec<(&str, Vec<u8>, Vec<u8>)>) -> Result<()> {
        // Sled has no separate turbo mode - delegate to regular write_batch
        self.write_batch(batch).await
    }

    async fn create_checkpoint(&self, _checkpoint_dir: &str) -> Result<()> {
        // Sled doesn't support native checkpoints - flush to disk instead
        self.db.flush_async().await.context("sled flush failed during checkpoint")?;
        info!("💾 sled checkpoint (flush) completed");
        Ok(())
    }

    async fn sync_wal(&self) -> Result<()> {
        // Sled manages its own WAL - flush to ensure persistence
        self.db.flush_async().await.context("sled flush failed during WAL sync")?;
        Ok(())
    }

    async fn shutdown_gracefully(&self) -> Result<()> {
        // Flush all pending writes before shutdown
        self.db.flush_async().await.context("sled flush failed during shutdown")?;
        info!("💾 sled graceful shutdown completed");
        Ok(())
    }

    async fn verify_checkpoint(&self, _checkpoint_dir: &str) -> Result<bool> {
        // Sled doesn't have checkpoint verification - return true as no-op
        Ok(true)
    }

    async fn multi_get(&self, cf: &str, keys: &[Vec<u8>]) -> Result<Vec<Option<Vec<u8>>>> {
        let tree = self.get_tree(cf)?;
        let mut results = Vec::with_capacity(keys.len());
        for key in keys {
            let result = tree.get(key).context("sled multi_get failed")?;
            results.push(result.map(|ivec| ivec.to_vec()));
        }
        Ok(results)
    }
}

#[cfg(target_os = "windows")]
pub struct SledStats {
    pub total_keys: u64,
    pub total_size: u64,
}

#[cfg(target_os = "windows")]
impl SledStats {
    pub fn to_prometheus(&self) -> String {
        format!(
            "sled_total_keys {}\nsled_total_size_bytes {}\n",
            self.total_keys, self.total_size
        )
    }
}
