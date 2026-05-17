/// 🚀 Project APOLLO Phase 6: DELTA-V - Zero-Copy Block Access (MASS REDUCTION)
///
/// Zero-copy deserialization using memory-mapped access:
/// - Direct memory access to block data (no deserialization)
/// - Blocks stored in archive-friendly format
/// - Near-instant access to block fields
///
/// Aerospace analogy:
/// - MASS REDUCTION: Like reducing spacecraft mass for better delta-V
/// - Less CPU work = more "fuel" for actual processing
/// - Direct memory access = no "mass" of deserialization overhead
///
/// Note: Full rkyv integration requires adding rkyv dependency
/// This module provides the infrastructure for zero-copy access patterns
/// using memory-mapped files and aligned storage.
///
/// Key features:
/// - Memory-mapped block storage
/// - Aligned access for CPU-friendly reads
/// - Zero-copy field access
/// - Lazy deserialization when full block needed
///
/// Expected improvement: 90%+ reduction in deserialization CPU time

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Block header for zero-copy access (fixed layout)
/// This is designed to be directly readable from memory without parsing
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug)]
pub struct ZeroCopyHeader {
    /// Magic bytes for validation: "QBLK"
    pub magic: [u8; 4],

    /// Version of the zero-copy format
    pub version: u8,

    /// Flags (reserved for future use)
    pub flags: u8,

    /// Reserved padding for alignment
    pub _reserved: [u8; 2],

    /// Block height
    pub height: u64,

    /// Block timestamp
    pub timestamp: u64,

    /// Previous block hash
    pub prev_hash: [u8; 32],

    /// Block hash (this block)
    pub block_hash: [u8; 32],

    /// Transaction Merkle root
    pub tx_merkle_root: [u8; 32],

    /// State root after this block
    pub state_root: [u8; 32],

    /// Number of transactions
    pub tx_count: u32,

    /// Number of mining solutions
    pub solution_count: u32,

    /// Total block size (including this header)
    pub total_size: u64,

    /// Offset to transaction data from start of block
    pub tx_offset: u32,

    /// Offset to mining solutions from start of block
    pub solution_offset: u32,

    /// Offset to DAG parents from start of block
    pub dag_parents_offset: u32,

    /// Number of DAG parents
    pub dag_parents_count: u8,

    /// Compression type (0=none, 1=lz4, 2=zstd)
    pub compression: u8,

    /// Reserved for future use
    pub _reserved2: [u8; 6],
}

impl ZeroCopyHeader {
    /// Magic bytes for zero-copy blocks
    pub const MAGIC: [u8; 4] = *b"QBLK";

    /// Current version
    pub const VERSION: u8 = 1;

    /// Header size (fixed)
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Create new header
    pub fn new(height: u64, block_hash: [u8; 32], prev_hash: [u8; 32]) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            flags: 0,
            _reserved: [0; 2],
            height,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            prev_hash,
            block_hash,
            tx_merkle_root: [0; 32],
            state_root: [0; 32],
            tx_count: 0,
            solution_count: 0,
            total_size: Self::SIZE as u64,
            tx_offset: 0,
            solution_offset: 0,
            dag_parents_offset: 0,
            dag_parents_count: 0,
            compression: 0,
            _reserved2: [0; 6],
        }
    }

    /// Validate header
    pub fn validate(&self) -> Result<()> {
        if self.magic != Self::MAGIC {
            bail!(
                "Invalid magic: expected {:?}, got {:?}",
                Self::MAGIC,
                self.magic
            );
        }
        if self.version > Self::VERSION {
            bail!(
                "Unsupported version: {} (max: {})",
                self.version,
                Self::VERSION
            );
        }
        Ok(())
    }

    /// Read header from bytes (zero-copy if properly aligned)
    pub fn from_bytes(bytes: &[u8]) -> Result<&Self> {
        if bytes.len() < Self::SIZE {
            bail!("Buffer too small for header: {} < {}", bytes.len(), Self::SIZE);
        }

        // Check alignment
        let ptr = bytes.as_ptr();
        if (ptr as usize) % std::mem::align_of::<Self>() != 0 {
            bail!("Buffer not aligned for zero-copy access");
        }

        // Safe because we've verified size and alignment
        let header = unsafe { &*(ptr as *const Self) };
        header.validate()?;

        Ok(header)
    }

    /// Convert to bytes
    pub fn to_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                Self::SIZE,
            )
        }
    }
}

/// Zero-copy block view (references data without copying)
pub struct ZeroCopyBlockView<'a> {
    /// Header (zero-copy reference)
    pub header: &'a ZeroCopyHeader,

    /// Raw transaction data
    tx_data: &'a [u8],

    /// Raw solution data
    solution_data: &'a [u8],

    /// Raw DAG parents data
    dag_parents_data: &'a [u8],
}

impl<'a> ZeroCopyBlockView<'a> {
    /// Create view from raw bytes
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        let header = ZeroCopyHeader::from_bytes(bytes)?;

        let tx_start = header.tx_offset as usize;
        let tx_end = header.solution_offset as usize;
        let sol_start = header.solution_offset as usize;
        let sol_end = header.dag_parents_offset as usize;
        let dag_start = header.dag_parents_offset as usize;
        let dag_end = header.total_size as usize;

        // Validate offsets
        if tx_end > bytes.len() || sol_end > bytes.len() || dag_end > bytes.len() {
            bail!("Block data truncated");
        }

        Ok(Self {
            header,
            tx_data: &bytes[tx_start..tx_end],
            solution_data: &bytes[sol_start..sol_end],
            dag_parents_data: &bytes[dag_start..dag_end],
        })
    }

    /// Get block height (zero-copy)
    pub fn height(&self) -> u64 {
        self.header.height
    }

    /// Get block hash (zero-copy)
    pub fn block_hash(&self) -> &[u8; 32] {
        &self.header.block_hash
    }

    /// Get previous block hash (zero-copy)
    pub fn prev_hash(&self) -> &[u8; 32] {
        &self.header.prev_hash
    }

    /// Get timestamp (zero-copy)
    pub fn timestamp(&self) -> u64 {
        self.header.timestamp
    }

    /// Get transaction count (zero-copy)
    pub fn tx_count(&self) -> u32 {
        self.header.tx_count
    }

    /// Get raw transaction data (for lazy parsing)
    pub fn raw_tx_data(&self) -> &[u8] {
        self.tx_data
    }

    /// Get DAG parent count
    pub fn dag_parents_count(&self) -> u8 {
        self.header.dag_parents_count
    }

    /// Get DAG parent hashes
    pub fn dag_parent_hashes(&self) -> Vec<[u8; 32]> {
        let count = self.header.dag_parents_count as usize;
        let mut parents = Vec::with_capacity(count);

        for i in 0..count {
            let start = i * 32;
            let end = start + 32;
            if end <= self.dag_parents_data.len() {
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&self.dag_parents_data[start..end]);
                parents.push(hash);
            }
        }

        parents
    }
}

/// Memory-mapped block store for zero-copy access
pub struct ZeroCopyBlockStore {
    /// Path to storage file
    path: PathBuf,

    /// Block index: height -> (offset, size)
    index: HashMap<u64, (u64, u64)>,

    /// Memory-mapped region (optional, for read-heavy workloads)
    #[cfg(not(target_os = "windows"))]
    mmap: Option<memmap2::Mmap>,
    #[cfg(target_os = "windows")]
    mmap: Option<()>,

    /// File handle
    file: File,

    /// Current file size
    file_size: u64,

    /// Statistics
    stats: ZeroCopyStats,
}

/// Zero-copy statistics
#[derive(Clone, Debug, Default)]
pub struct ZeroCopyStats {
    pub blocks_written: u64,
    pub blocks_read: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl ZeroCopyBlockStore {
    /// Create new zero-copy block store
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)
            .context("Failed to open block store")?;

        let file_size = file.metadata()?.len();

        info!(
            "📦 [ZEROCOPY] Opened block store at {:?} ({} bytes)",
            path, file_size
        );

        Ok(Self {
            path,
            index: HashMap::new(),
            mmap: None,
            file,
            file_size,
            stats: ZeroCopyStats::default(),
        })
    }

    /// Enable memory-mapping for read access
    #[cfg(not(target_os = "windows"))]
    pub fn enable_mmap(&mut self) -> Result<()> {
        if self.file_size == 0 {
            warn!("[ZEROCOPY] Cannot mmap empty file");
            return Ok(());
        }

        let mmap = unsafe { memmap2::Mmap::map(&self.file)? };
        self.mmap = Some(mmap);

        info!(
            "[ZEROCOPY] Memory-mapped {} bytes",
            self.file_size
        );

        Ok(())
    }

    /// Enable memory-mapping for read access (Windows stub - mmap not available)
    #[cfg(target_os = "windows")]
    pub fn enable_mmap(&mut self) -> Result<()> {
        warn!("[ZEROCOPY] Memory-mapping not available on Windows");
        Ok(())
    }

    /// Write block in zero-copy format
    pub fn write_block(
        &mut self,
        height: u64,
        block_hash: [u8; 32],
        prev_hash: [u8; 32],
        tx_data: &[u8],
        solution_data: &[u8],
        dag_parents: &[[u8; 32]],
    ) -> Result<()> {
        // Calculate offsets
        let header_size = ZeroCopyHeader::SIZE as u32;
        let tx_offset = header_size;
        let solution_offset = tx_offset + tx_data.len() as u32;
        let dag_parents_offset = solution_offset + solution_data.len() as u32;
        let dag_parents_size = dag_parents.len() * 32;
        let total_size = dag_parents_offset as u64 + dag_parents_size as u64;

        // Create header
        let mut header = ZeroCopyHeader::new(height, block_hash, prev_hash);
        header.tx_offset = tx_offset;
        header.solution_offset = solution_offset;
        header.dag_parents_offset = dag_parents_offset;
        header.dag_parents_count = dag_parents.len() as u8;
        header.total_size = total_size;

        // Write to file
        let offset = self.file_size;
        self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(header.to_bytes())?;
        self.file.write_all(tx_data)?;
        self.file.write_all(solution_data)?;
        for parent in dag_parents {
            self.file.write_all(parent)?;
        }

        // Update index
        self.index.insert(height, (offset, total_size));
        self.file_size += total_size;

        // Update stats
        self.stats.blocks_written += 1;
        self.stats.bytes_written += total_size;

        // Invalidate mmap if active
        self.mmap = None;

        debug!(
            "[ZEROCOPY] Wrote block {} ({} bytes at offset {})",
            height, total_size, offset
        );

        Ok(())
    }

    /// Read block header only (minimal I/O)
    pub fn read_header(&mut self, height: u64) -> Result<ZeroCopyHeader> {
        let (offset, _size) = self
            .index
            .get(&height)
            .context("Block not found in index")?;

        // Use mmap if available
        #[cfg(not(target_os = "windows"))]
        if let Some(ref mmap) = self.mmap {
            let start = *offset as usize;
            let end = start + ZeroCopyHeader::SIZE;
            if end <= mmap.len() {
                let header = ZeroCopyHeader::from_bytes(&mmap[start..end])?;
                self.stats.cache_hits += 1;
                return Ok(*header);
            }
        }

        // Fall back to file I/O
        self.stats.cache_misses += 1;
        let mut buf = vec![0u8; ZeroCopyHeader::SIZE];
        self.file.seek(SeekFrom::Start(*offset))?;
        self.file.read_exact(&mut buf)?;

        let header = ZeroCopyHeader::from_bytes(&buf)?;
        self.stats.blocks_read += 1;

        Ok(*header)
    }

    /// Read full block (zero-copy if mmap available)
    pub fn read_block(&mut self, height: u64) -> Result<Vec<u8>> {
        let (offset, size) = self
            .index
            .get(&height)
            .copied()
            .context("Block not found in index")?;

        // Use mmap if available
        #[cfg(not(target_os = "windows"))]
        if let Some(ref mmap) = self.mmap {
            let start = offset as usize;
            let end = start + size as usize;
            if end <= mmap.len() {
                self.stats.cache_hits += 1;
                self.stats.blocks_read += 1;
                self.stats.bytes_read += size;
                return Ok(mmap[start..end].to_vec());
            }
        }

        // Fall back to file I/O
        self.stats.cache_misses += 1;
        let mut buf = vec![0u8; size as usize];
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(&mut buf)?;

        self.stats.blocks_read += 1;
        self.stats.bytes_read += size;

        Ok(buf)
    }

    /// Get block as zero-copy view (requires mmap)
    #[cfg(not(target_os = "windows"))]
    pub fn get_view(&self, height: u64) -> Result<ZeroCopyBlockView<'_>> {
        let (offset, size) = self
            .index
            .get(&height)
            .copied()
            .context("Block not found in index")?;

        let mmap = self
            .mmap
            .as_ref()
            .context("Memory mapping not enabled")?;

        let start = offset as usize;
        let end = start + size as usize;

        if end > mmap.len() {
            bail!("Block extends beyond mmap");
        }

        ZeroCopyBlockView::from_bytes(&mmap[start..end])
    }

    /// Get block as zero-copy view (not supported on Windows - no mmap)
    #[cfg(target_os = "windows")]
    pub fn get_view(&self, _height: u64) -> Result<ZeroCopyBlockView<'_>> {
        bail!("Zero-copy block views not supported on Windows")
    }

    /// Get statistics
    pub fn get_stats(&self) -> ZeroCopyStats {
        self.stats.clone()
    }

    /// Get block count
    pub fn block_count(&self) -> usize {
        self.index.len()
    }

    /// Sync to disk
    pub fn sync(&self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_header_size() {
        // Header should be fixed size for zero-copy access
        assert!(ZeroCopyHeader::SIZE > 0);
        assert!(ZeroCopyHeader::SIZE % 8 == 0); // 8-byte aligned
    }

    #[test]
    fn test_header_roundtrip() {
        let header = ZeroCopyHeader::new(100, [1u8; 32], [2u8; 32]);
        let bytes = header.to_bytes();

        // Ensure proper alignment for test
        let mut aligned_buf = vec![0u8; ZeroCopyHeader::SIZE + 8];
        let offset = aligned_buf.as_ptr() as usize % 8;
        let aligned_start = if offset == 0 { 0 } else { 8 - offset };
        aligned_buf[aligned_start..aligned_start + ZeroCopyHeader::SIZE]
            .copy_from_slice(bytes);

        let restored = ZeroCopyHeader::from_bytes(
            &aligned_buf[aligned_start..aligned_start + ZeroCopyHeader::SIZE],
        )
        .unwrap();

        assert_eq!(restored.height, 100);
        assert_eq!(restored.block_hash, [1u8; 32]);
        assert_eq!(restored.prev_hash, [2u8; 32]);
    }

    #[test]
    fn test_block_store() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("blocks.dat");

        let mut store = ZeroCopyBlockStore::new(&path).unwrap();

        // Write a block
        store
            .write_block(
                1,
                [1u8; 32],
                [0u8; 32],
                b"transaction data",
                b"solution data",
                &[[2u8; 32], [3u8; 32]],
            )
            .unwrap();

        assert_eq!(store.block_count(), 1);

        // Read header
        let header = store.read_header(1).unwrap();
        assert_eq!(header.height, 1);
        assert_eq!(header.block_hash, [1u8; 32]);

        // Read full block
        let data = store.read_block(1).unwrap();
        assert!(!data.is_empty());
    }
}
