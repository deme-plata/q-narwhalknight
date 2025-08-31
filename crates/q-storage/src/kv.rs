/// High-performance KV store abstraction with RocksDB implementation
/// Optimized for DagKnight/Narwhal/Bullshark access patterns

use anyhow::{Context, Result};
use async_trait::async_trait;
use q_quantum_rng::{QuantumRNG, QuantumRandomness, QRNGConfig};
use q_types::Phase;
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, Options, WriteBatch, DB};
use std::{
    collections::HashMap,
    path::Path,
    sync::Arc,
};
use tokio::task;
use tracing::{debug, info, warn};

use crate::{CF_BLOCKS, CF_BULLSHARK_CERT, CF_DAG_VERTICES, CF_MANIFEST, CF_NARWHAL_PAYLOADS};

/// Async KV store trait for storage abstraction
#[async_trait]
pub trait KVStore: Send + Sync {
    /// Put key-value pair in column family
    async fn put(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()>;
    
    /// Get value by key from column family
    async fn get(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>>;
    
    /// Delete key from column family
    async fn delete(&self, cf: &str, key: &[u8]) -> Result<()>;
    
    /// Write atomic batch across column families
    async fn write_batch(&self, batch: Vec<(&str, Vec<u8>, Vec<u8>)>) -> Result<()>;
    
    /// Scan keys with prefix in column family
    async fn scan_prefix(&self, cf: &str, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;
    
    /// Scan all keys in column family (use with caution)
    async fn scan_all(&self, cf: &str) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;
    
    /// Flush writes to disk
    async fn flush(&self) -> Result<()>;
    
    /// Compact database
    async fn compact(&self) -> Result<()>;
    
    /// Get database size in bytes
    async fn get_db_size(&self) -> Result<u64>;
}

/// RocksDB implementation optimized for DagKnight workloads
pub struct RocksDBKV {
    db: Arc<DB>,
    cf_handles: HashMap<String, Arc<ColumnFamily>>,
    /// Quantum RNG for encryption keys (Phase 2+)
    qrng: Option<Arc<QuantumRNG>>,
    /// Current cryptographic phase
    phase: Phase,
}

impl RocksDBKV {
    /// Open hot database with optimized settings for frequent access
    pub async fn open_hot_db<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_hot_db_with_phase(path, Phase::Phase0).await
    }

    /// Open hot database with specific phase support
    pub async fn open_hot_db_with_phase<P: AsRef<Path>>(path: P, phase: Phase) -> Result<Self> {
        let path = path.as_ref();
        info!("🔥 Opening hot RocksDB at {:?} for {:?}", path, phase);

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        
        // Hot DB optimizations
        opts.set_max_background_jobs(8);
        opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB
        opts.set_max_write_buffer_number(4);
        opts.set_target_file_size_base(64 * 1024 * 1024); // 64MB
        opts.set_level_zero_file_num_compaction_trigger(4);
        opts.set_level_zero_slowdown_writes_trigger(8);
        opts.set_level_zero_stop_writes_trigger(16);

        // Initialize quantum encryption for Phase 2+
        let qrng = if matches!(phase, Phase::Phase2 | Phase::Phase3 | Phase::Phase4) {
            info!("🌌 Initializing quantum RNG for storage encryption");
            let config = QRNGConfig {
                min_entropy_quality: 0.99, // Highest quality for encryption
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

        let cfs = vec![
            Self::create_blocks_cf(),
            Self::create_dag_vertices_cf(),
            Self::create_bullshark_cert_cf(),
            Self::create_manifest_cf(),
        ];

        let mut kv = Self::open_with_cfs(path, opts, cfs).await?;
        kv.qrng = qrng;
        kv.phase = phase;

        Ok(kv)
    }

    /// Open cold database with optimized settings for large payloads
    pub async fn open_cold_db<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        info!("🧊 Opening cold RocksDB at {:?}", path);

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        
        // Cold DB optimizations (favor compression over speed)
        opts.set_max_background_jobs(4);
        opts.set_write_buffer_size(128 * 1024 * 1024); // 128MB
        opts.set_max_write_buffer_number(2);
        opts.set_target_file_size_base(256 * 1024 * 1024); // 256MB
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        let cfs = vec![
            Self::create_narwhal_payloads_cf(),
        ];

        Self::open_with_cfs(path, opts, cfs).await
    }

    /// Open RocksDB with column families
    async fn open_with_cfs<P: AsRef<Path>>(
        path: P, 
        opts: Options, 
        cfs: Vec<ColumnFamilyDescriptor>
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        
        let (db, cf_handles) = task::spawn_blocking(move || -> Result<_> {
            let db = DB::open_cf_descriptors(&opts, path, cfs)
                .context("Failed to open RocksDB")?;
            
            let mut cf_handles = HashMap::new();
            for cf_name in db.cf_names() {
                if let Some(cf) = db.cf_handle(cf_name) {
                    cf_handles.insert(cf_name.to_string(), Arc::new(cf));
                }
            }
            
            Ok((Arc::new(db), cf_handles))
        }).await??;

        Ok(Self {
            db,
            cf_handles,
            qrng: None, // Will be set by caller
            phase: Phase::Phase0, // Will be set by caller
        })
    }

    /// Create blocks column family (height || hash -> block)
    fn create_blocks_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_block_based_table_factory(&rocksdb::BlockBasedOptions::default());
        
        ColumnFamilyDescriptor::new(CF_BLOCKS, opts)
    }

    /// Create DAG vertices column family (round || author || seq -> vertex)
    fn create_dag_vertices_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        
        // Enable prefix seek for round-based queries
        let mut table_opts = rocksdb::BlockBasedOptions::default();
        table_opts.set_index_type(rocksdb::BlockBasedIndexType::HashSearch);
        opts.set_block_based_table_factory(&table_opts);
        
        ColumnFamilyDescriptor::new(CF_DAG_VERTICES, opts)
    }

    /// Create Bullshark certificates column family (round -> certificate)
    fn create_bullshark_cert_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Snappy);
        
        ColumnFamilyDescriptor::new(CF_BULLSHARK_CERT, opts)
    }

    /// Create manifest column family (metadata -> value)
    fn create_manifest_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::None); // Small data
        
        ColumnFamilyDescriptor::new(CF_MANIFEST, opts)
    }

    /// Create Narwhal payloads column family (digest -> payload)
    fn create_narwhal_payloads_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        
        // Large values, optimize for sequential writes
        opts.set_write_buffer_size(256 * 1024 * 1024); // 256MB
        opts.set_max_write_buffer_number(2);
        opts.set_target_file_size_base(512 * 1024 * 1024); // 512MB
        
        ColumnFamilyDescriptor::new(CF_NARWHAL_PAYLOADS, opts)
    }

    /// Get column family handle
    fn get_cf(&self, cf_name: &str) -> Result<&ColumnFamily> {
        self.cf_handles.get(cf_name)
            .map(|cf| cf.as_ref())
            .context(format!("Column family {} not found", cf_name))
    }
}

#[async_trait]
impl KVStore for RocksDBKV {
    async fn put(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()> {
        let db = self.db.clone();
        let cf_handle = self.get_cf(cf)?.clone();
        let key = key.to_vec();
        let value = value.to_vec();

        task::spawn_blocking(move || {
            db.put_cf(&cf_handle, key, value)
                .context("RocksDB put failed")
        }).await??;

        Ok(())
    }

    async fn get(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let db = self.db.clone();
        let cf_handle = self.get_cf(cf)?.clone();
        let key = key.to_vec();

        let result = task::spawn_blocking(move || {
            db.get_cf(&cf_handle, key)
                .context("RocksDB get failed")
        }).await??;

        Ok(result)
    }

    async fn delete(&self, cf: &str, key: &[u8]) -> Result<()> {
        let db = self.db.clone();
        let cf_handle = self.get_cf(cf)?.clone();
        let key = key.to_vec();

        task::spawn_blocking(move || {
            db.delete_cf(&cf_handle, key)
                .context("RocksDB delete failed")
        }).await??;

        Ok(())
    }

    async fn write_batch(&self, batch: Vec<(&str, Vec<u8>, Vec<u8>)>) -> Result<()> {
        let db = self.db.clone();
        let cf_handles = self.cf_handles.clone();

        task::spawn_blocking(move || -> Result<()> {
            let mut write_batch = WriteBatch::default();
            
            for (cf_name, key, value) in batch {
                let cf_handle = cf_handles.get(cf_name)
                    .context(format!("Column family {} not found", cf_name))?;
                write_batch.put_cf(cf_handle.as_ref(), key, value);
            }
            
            db.write(write_batch)
                .context("RocksDB batch write failed")?;
                
            Ok(())
        }).await??;

        Ok(())
    }

    async fn scan_prefix(&self, cf: &str, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let db = self.db.clone();
        let cf_handle = self.get_cf(cf)?.clone();
        let prefix = prefix.to_vec();

        let results = task::spawn_blocking(move || -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
            let mut results = Vec::new();
            let iter = db.prefix_iterator_cf(&cf_handle, &prefix);
            
            for item in iter {
                let (key, value) = item.context("Iterator error")?;
                
                // Check if key still has the prefix
                if !key.starts_with(&prefix) {
                    break;
                }
                
                results.push((key.to_vec(), value.to_vec()));
            }
            
            Ok(results)
        }).await??;

        Ok(results)
    }

    async fn scan_all(&self, cf: &str) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let db = self.db.clone();
        let cf_handle = self.get_cf(cf)?.clone();

        let results = task::spawn_blocking(move || -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
            let mut results = Vec::new();
            let iter = db.iterator_cf(&cf_handle, rocksdb::IteratorMode::Start);
            
            for item in iter {
                let (key, value) = item.context("Iterator error")?;
                results.push((key.to_vec(), value.to_vec()));
            }
            
            Ok(results)
        }).await??;

        Ok(results)
    }

    async fn flush(&self) -> Result<()> {
        let db = self.db.clone();

        task::spawn_blocking(move || {
            db.flush().context("RocksDB flush failed")
        }).await??;

        Ok(())
    }

    async fn compact(&self) -> Result<()> {
        let db = self.db.clone();
        let cf_handles = self.cf_handles.clone();

        task::spawn_blocking(move || -> Result<()> {
            for (cf_name, cf_handle) in cf_handles {
                debug!("🗜️ Compacting column family: {}", cf_name);
                db.compact_range_cf(cf_handle.as_ref(), None::<&[u8]>, None::<&[u8]>);
            }
            Ok(())
        }).await??;

        Ok(())
    }

    async fn get_db_size(&self) -> Result<u64> {
        let db = self.db.clone();

        let size = task::spawn_blocking(move || -> Result<u64> {
            let mut total_size = 0u64;
            
            // Get size of each column family
            for cf_name in db.cf_names() {
                if let Some(cf) = db.cf_handle(cf_name) {
                    if let Ok(Some(size_str)) = db.property_value_cf(cf, "rocksdb.total-sst-files-size") {
                        if let Ok(size) = size_str.parse::<u64>() {
                            total_size += size;
                        }
                    }
                }
            }
            
            Ok(total_size)
        }).await??;

        Ok(size)
    }
}

/// RocksDB write options optimized for Narwhal workloads
impl RocksDBKV {
    /// Get optimized write options
    fn write_options() -> rocksdb::WriteOptions {
        let mut opts = rocksdb::WriteOptions::default();
        opts.set_sync(false); // Use WAL for durability, don't sync every write
        opts.disable_wal(false); // Keep WAL for crash recovery
        opts
    }

    /// Get optimized read options
    fn read_options() -> rocksdb::ReadOptions {
        let mut opts = rocksdb::ReadOptions::default();
        opts.set_verify_checksums(false); // Trade off for speed in hot path
        opts
    }

    /// Get database statistics
    pub async fn get_stats(&self) -> Result<RocksDBStats> {
        let db = self.db.clone();

        let stats = task::spawn_blocking(move || -> Result<RocksDBStats> {
            let mut cf_stats = HashMap::new();

            for cf_name in db.cf_names() {
                if let Some(cf) = db.cf_handle(cf_name) {
                    let stats = RocksDBCFStats {
                        keys: Self::get_cf_property(&db, cf, "rocksdb.estimate-num-keys")?,
                        size: Self::get_cf_property(&db, cf, "rocksdb.total-sst-files-size")?,
                        files: Self::get_cf_property(&db, cf, "rocksdb.num-files-at-level0")?,
                        compactions: Self::get_cf_property(&db, cf, "rocksdb.num-running-compactions")?,
                    };
                    cf_stats.insert(cf_name.to_string(), stats);
                }
            }

            Ok(RocksDBStats {
                column_families: cf_stats,
                total_size: Self::get_total_size(&db)?,
                cache_usage: Self::get_cache_usage(&db)?,
            })
        }).await??;

        Ok(stats)
    }

    /// Get property value from column family
    fn get_cf_property(db: &DB, cf: &ColumnFamily, property: &str) -> Result<u64> {
        db.property_value_cf(cf, property)
            .context("Failed to get property")?
            .context("Property value missing")?
            .parse()
            .context("Failed to parse property value")
    }

    /// Get total database size
    fn get_total_size(db: &DB) -> Result<u64> {
        let mut total_size = 0u64;
        
        for cf_name in db.cf_names() {
            if let Some(cf) = db.cf_handle(cf_name) {
                if let Ok(size) = Self::get_cf_property(db, cf, "rocksdb.total-sst-files-size") {
                    total_size += size;
                }
            }
        }
        
        Ok(total_size)
    }

    /// Get cache usage
    fn get_cache_usage(_db: &DB) -> Result<u64> {
        // TODO: Implement cache usage tracking
        Ok(0)
    }

    /// Create checkpoint for snapshot
    pub async fn create_checkpoint<P: AsRef<Path>>(&self, checkpoint_path: P) -> Result<()> {
        let db = self.db.clone();
        let checkpoint_path = checkpoint_path.as_ref().to_path_buf();

        task::spawn_blocking(move || -> Result<()> {
            let checkpoint = rocksdb::checkpoint::Checkpoint::new(&db)
                .context("Failed to create checkpoint handle")?;
            
            checkpoint.create_checkpoint(&checkpoint_path)
                .context("Failed to create checkpoint")?;
            
            Ok(())
        }).await??;

        Ok(())
    }
}

/// RocksDB statistics for monitoring
#[derive(Debug, Clone)]
pub struct RocksDBStats {
    pub column_families: HashMap<String, RocksDBCFStats>,
    pub total_size: u64,
    pub cache_usage: u64,
}

/// Column family statistics
#[derive(Debug, Clone)]
pub struct RocksDBCFStats {
    pub keys: u64,
    pub size: u64,
    pub files: u64,
    pub compactions: u64,
}

impl RocksDBStats {
    /// Get Prometheus-format metrics
    pub fn to_prometheus(&self) -> String {
        let mut metrics = String::new();
        
        for (cf_name, stats) in &self.column_families {
            metrics.push_str(&format!(
                "rocksdb_keys{{cf=\"{}\"}} {}\n\
                 rocksdb_size_bytes{{cf=\"{}\"}} {}\n\
                 rocksdb_files{{cf=\"{}\"}} {}\n\
                 rocksdb_compactions{{cf=\"{}\"}} {}\n",
                cf_name, stats.keys,
                cf_name, stats.size,
                cf_name, stats.files,
                cf_name, stats.compactions
            ));
        }
        
        metrics.push_str(&format!(
            "rocksdb_total_size_bytes {}\n\
             rocksdb_cache_usage_bytes {}\n",
            self.total_size,
            self.cache_usage
        ));
        
        metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_hot_db_creation() {
        let temp_dir = TempDir::new().unwrap();
        let result = RocksDBKV::open_hot_db(temp_dir.path()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_cold_db_creation() {
        let temp_dir = TempDir::new().unwrap();
        let result = RocksDBKV::open_cold_db(temp_dir.path()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_basic_operations() {
        let temp_dir = TempDir::new().unwrap();
        let kv = RocksDBKV::open_hot_db(temp_dir.path()).await.unwrap();

        // Test put/get
        let key = b"test_key";
        let value = b"test_value";
        
        kv.put(CF_MANIFEST, key, value).await.unwrap();
        let retrieved = kv.get(CF_MANIFEST, key).await.unwrap();
        
        assert_eq!(retrieved, Some(value.to_vec()));

        // Test delete
        kv.delete(CF_MANIFEST, key).await.unwrap();
        let after_delete = kv.get(CF_MANIFEST, key).await.unwrap();
        
        assert_eq!(after_delete, None);
    }

    #[tokio::test]
    async fn test_batch_write() {
        let temp_dir = TempDir::new().unwrap();
        let kv = RocksDBKV::open_hot_db(temp_dir.path()).await.unwrap();

        let batch = vec![
            (CF_MANIFEST, b"key1".to_vec(), b"value1".to_vec()),
            (CF_MANIFEST, b"key2".to_vec(), b"value2".to_vec()),
            (CF_MANIFEST, b"key3".to_vec(), b"value3".to_vec()),
        ];

        kv.write_batch(batch).await.unwrap();

        // Verify all keys were written
        assert_eq!(kv.get(CF_MANIFEST, b"key1").await.unwrap(), Some(b"value1".to_vec()));
        assert_eq!(kv.get(CF_MANIFEST, b"key2").await.unwrap(), Some(b"value2".to_vec()));
        assert_eq!(kv.get(CF_MANIFEST, b"key3").await.unwrap(), Some(b"value3".to_vec()));
    }

    #[tokio::test]
    async fn test_prefix_scan() {
        let temp_dir = TempDir::new().unwrap();
        let kv = RocksDBKV::open_hot_db(temp_dir.path()).await.unwrap();

        // Insert test data with common prefix
        let prefix = b"test_prefix_";
        for i in 0..5 {
            let key = format!("{}key{}", std::str::from_utf8(prefix).unwrap(), i);
            let value = format!("value{}", i);
            kv.put(CF_MANIFEST, key.as_bytes(), value.as_bytes()).await.unwrap();
        }

        // Scan with prefix
        let results = kv.scan_prefix(CF_MANIFEST, prefix).await.unwrap();
        assert_eq!(results.len(), 5);

        // Verify results are sorted
        for (i, (key, _)) in results.iter().enumerate() {
            let expected_key = format!("{}key{}", std::str::from_utf8(prefix).unwrap(), i);
            assert_eq!(key, expected_key.as_bytes());
        }
    }
}