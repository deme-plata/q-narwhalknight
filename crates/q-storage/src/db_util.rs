/// Database Utilities - Spawn_blocking helpers for RocksDB operations
///
/// This module provides utilities to properly execute synchronous RocksDB operations
/// on dedicated blocking threads to prevent Tokio executor thread starvation.
///
/// Key principles:
/// - ALL RocksDB operations must use spawn_blocking
/// - Use WAL fsync instead of per-operation flush for better performance
/// - Batch operations when possible to reduce I/O overhead

use std::sync::Arc;
#[cfg(not(target_os = "windows"))]
use rocksdb::{DB, WriteBatch, WriteOptions};
use anyhow::Result;

/// Write a batch of operations to RocksDB with WAL fsync
///
/// This function ensures data durability by syncing the Write-Ahead Log (WAL)
/// but avoids expensive per-block flush() operations for better performance.
///
/// # Arguments
/// * `db` - Arc reference to RocksDB instance
/// * `batch` - WriteBatch containing all operations to write
///
/// # Returns
/// * `Ok(())` if write succeeded
/// * `Err` if write failed
pub fn write_batch_sync(db: Arc<DB>, batch: WriteBatch) -> Result<()> {
    let mut wo = WriteOptions::default();
    wo.set_sync(true); // WAL fsync for durability
    db.write_opt(batch, &wo)?;
    Ok(())
}

/// Write a batch of operations to RocksDB without WAL fsync (for performance)
///
/// Use this only for non-critical operations where eventual consistency is acceptable.
/// For critical blockchain data, use `write_batch_sync` instead.
///
/// # Arguments
/// * `db` - Arc reference to RocksDB instance
/// * `batch` - WriteBatch containing all operations to write
///
/// # Returns
/// * `Ok(())` if write succeeded
/// * `Err` if write failed
pub fn write_batch_async(db: Arc<DB>, batch: WriteBatch) -> Result<()> {
    let wo = WriteOptions::default();
    db.write_opt(batch, &wo)?;
    Ok(())
}

/// Perform a manual flush of all data to disk
///
/// This should only be called periodically (e.g., every 60-120 seconds)
/// NOT after every block. Per-block flush() is a major performance bottleneck.
///
/// # Arguments
/// * `db` - Arc reference to RocksDB instance
///
/// # Returns
/// * `Ok(())` if flush succeeded
/// * `Err` if flush failed
pub fn flush_database(db: Arc<DB>) -> Result<()> {
    db.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocksdb::{DB, Options};
    use tempfile::tempdir;

    #[test]
    fn test_write_batch_sync() {
        let dir = tempdir().unwrap();
        let path = dir.path();

        let mut opts = Options::default();
        opts.create_if_missing(true);

        let db = Arc::new(DB::open(&opts, path).unwrap());

        let mut batch = WriteBatch::default();
        batch.put(b"key1", b"value1");
        batch.put(b"key2", b"value2");

        let result = write_batch_sync(db.clone(), batch);
        assert!(result.is_ok());

        // Verify data was written
        assert_eq!(db.get(b"key1").unwrap().unwrap(), b"value1");
        assert_eq!(db.get(b"key2").unwrap().unwrap(), b"value2");
    }

    #[test]
    fn test_write_batch_async() {
        let dir = tempdir().unwrap();
        let path = dir.path();

        let mut opts = Options::default();
        opts.create_if_missing(true);

        let db = Arc::new(DB::open(&opts, path).unwrap());

        let mut batch = WriteBatch::default();
        batch.put(b"key3", b"value3");

        let result = write_batch_async(db.clone(), batch);
        assert!(result.is_ok());

        // Verify data was written
        assert_eq!(db.get(b"key3").unwrap().unwrap(), b"value3");
    }

    #[test]
    fn test_flush_database() {
        let dir = tempdir().unwrap();
        let path = dir.path();

        let mut opts = Options::default();
        opts.create_if_missing(true);

        let db = Arc::new(DB::open(&opts, path).unwrap());

        db.put(b"flush_test", b"data").unwrap();

        let result = flush_database(db.clone());
        assert!(result.is_ok());
    }
}
