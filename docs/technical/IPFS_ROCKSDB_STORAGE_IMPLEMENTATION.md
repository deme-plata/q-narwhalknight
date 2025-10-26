# IPFS-RocksDB Decentralized Storage System

**Date**: 2025-10-13
**Status**: Implementation in Progress
**Crate**: `q-ipfs-storage`

## 🎯 Overview

This document outlines the complete implementation of a decentralized RocksDB storage system using IPFS, integrated with the Q-NarwhalKnight network infrastructure.

## 📐 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Application Layer                         │
│  q-api-server / q-miner / any service using RocksDB         │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│                  q-ipfs-storage Crate                         │
│                                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Snapshot   │  │   Chunker   │  │ Compressor  │          │
│  │   Manager   │→ │   (256KB)   │→ │   (Zstd)    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│         │                │                 │                  │
│         ▼                ▼                 ▼                  │
│  ┌──────────────────────────────────────────────┐           │
│  │           IPFS Client (libp2p)                │           │
│  │  - Content Addressing (CID)                   │           │
│  │  - DHT (Kademlia)                             │           │
│  │  - Bitswap (Block Exchange)                   │           │
│  └──────────────────────────────────────────────┘           │
│         │                                                     │
└─────────┼─────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│                      q-network Layer                          │
│  - libp2p Gossipsub (Manifest Distribution)                   │
│  - Peer Discovery (mDNS, Kad-DHT)                             │
│  - Network Topology                                           │
└──────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│               Distributed IPFS Network                        │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐               │
│  │  Node 1   │  │  Node 2   │  │  Node 3   │               │
│  │  (Pins)   │  │  (Pins)   │  │  (Pins)   │               │
│  └───────────┘  └───────────┘  └───────────┘               │
└──────────────────────────────────────────────────────────────┘
```

## 🧩 Component Breakdown

### 1. Snapshot Manager (`snapshot.rs`)
**Status**: ✅ Implemented

**Responsibilities**:
- Create RocksDB checkpoints (read-only snapshots)
- Support full and incremental snapshots
- Track snapshot metadata (size, timestamp, parent)
- Cleanup old snapshots (configurable retention)

**Key Functions**:
```rust
// Create a snapshot
let metadata = snapshot_manager.create_snapshot(
    "/path/to/rocksdb",
    SnapshotType::Full
).await?;

// List snapshot files
let files = snapshot_manager.list_snapshot_files(&metadata)?;

// Cleanup old snapshots (keep last 5)
snapshot_manager.cleanup_snapshots(5).await?;
```

### 2. Chunker (`chunker.rs`)
**Status**: ✅ Implemented

**Responsibilities**:
- Split large files into 256 KB chunks (IPFS optimal size)
- Calculate Blake3 hash for each chunk (integrity verification)
- Reassemble chunks back into files
- Verify chunk integrity during reassembly

**Key Functions**:
```rust
// Chunk a file
let chunks = chunker.chunk_file(Path::new("database.sst")).await?;

// Reassemble chunks
chunker.reassemble_chunks(chunks, Path::new("restored.sst")).await?;

// Verify chunk
let valid = chunker.verify_chunk(&metadata, &data)?;
```

### 3. Compressor (`compression.rs`)
**Status**: ✅ Implemented

**Responsibilities**:
- Compress chunks using Zstd (default) or LZ4
- Decompress chunks during restoration
- Configurable compression levels
- No compression option for already compressed data

**Compression Ratios** (typical):
- SST files: 60-80% reduction (Zstd level 3)
- WAL files: 40-60% reduction
- MANIFEST: 70-90% reduction

**Key Functions**:
```rust
// Compress data
let compressed = compressor.compress(&data)?;

// Decompress data
let original = compressor.decompress(&compressed)?;
```

### 4. Manifest (`manifest.rs`)
**To Implement**: Complete storage manifest format

**Responsibilities**:
- Track all chunks for a snapshot
- Store CIDs for each chunk
- Include chunk metadata (hashes, sizes, offsets)
- Version the manifest format
- Support incremental snapshot references

**Manifest Format**:
```json
{
  "version": "1.0.0",
  "snapshot_id": "uuid-here",
  "snapshot_type": "Full",
  "timestamp": 1697208000,
  "total_size_bytes": 1073741824,
  "compressed_size_bytes": 322122547,
  "compression_type": "Zstd",
  "chunks": [
    {
      "file_path": "000123.sst",
      "chunk_index": 0,
      "cid": "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
      "hash": "blake3:abc123...",
      "size": 262144,
      "compressed_size": 78643,
      "offset": 0
    },
    // ... more chunks
  ],
  "parent_snapshot": null
}
```

### 5. IPFS Client (`ipfs_client.rs`)
**To Implement**: IPFS operations via libp2p

**Responsibilities**:
- Upload chunks to IPFS (get CIDs)
- Download chunks from IPFS by CID
- Pin chunks locally
- Request remote pinning
- Use DHT to find content
- Bitswap protocol for block exchange

**Key Functions**:
```rust
// Upload chunk to IPFS
let cid = ipfs_client.put_chunk(&chunk_data).await?;

// Download chunk from IPFS
let data = ipfs_client.get_chunk(&cid).await?;

// Pin chunk locally
ipfs_client.pin_local(&cid).await?;

// Request peers to pin
ipfs_client.request_remote_pin(&cid, replication_factor).await?;
```

### 6. Pinning Manager (`pinning.rs`)
**To Implement**: Distributed pinning strategy

**Responsibilities**:
- Coordinate pinning across network nodes
- Implement replication strategies (3x, 5x, etc.)
- Monitor pin health (are chunks still available?)
- Re-pin chunks if nodes go offline
- Support pinning priorities (critical data = higher replication)

**Pinning Strategies**:
```rust
pub enum PinningStrategy {
    /// Pin only on local node
    Local,
    /// Pin on N random peers
    Replicated(usize),
    /// Pin on geographically distributed peers
    GeoDist ributed { replicas: usize },
    /// Pin on validator nodes only
    ValidatorNodes,
}
```

### 7. Storage Manager (`storage.rs`)
**To Implement**: Main orchestration logic

**Responsibilities**:
- Coordinate all components
- Provide high-level backup/restore API
- Handle incremental snapshots
- Manage concurrent operations
- Emit progress events
- Handle errors and retries

**High-Level API**:
```rust
let mut storage = IpfsRocksStorage::new(config).await?;

// Backup database
let manifest_cid = storage.backup_database(
    "/data/rocksdb",
    BackupOptions {
        snapshot_type: SnapshotType::Full,
        compression: CompressionType::Zstd,
        replication: 3,
    }
).await?;

// Restore database
storage.restore_database(
    &manifest_cid,
    "/data/restored",
    RestoreOptions {
        verify_chunks: true,
        parallel_downloads: 10,
    }
).await?;
```

## 🔄 Backup Workflow

```
1. CREATE SNAPSHOT
   ├─ RocksDB Checkpoint → temp directory
   ├─ Calculate snapshot metadata
   └─ Get list of all SST/WAL/MANIFEST files

2. CHUNK & COMPRESS
   ├─ For each file:
   │  ├─ Split into 256 KB chunks
   │  ├─ Calculate Blake3 hash per chunk
   │  └─ Compress chunk with Zstd
   └─ Track chunk metadata

3. UPLOAD TO IPFS
   ├─ For each compressed chunk:
   │  ├─ Put chunk to IPFS (get CID)
   │  ├─ Pin chunk locally
   │  └─ Store CID in manifest
   └─ Generate final manifest

4. DISTRIBUTE MANIFEST
   ├─ Upload manifest to IPFS (get manifest CID)
   ├─ Pin manifest locally
   ├─ Gossip manifest CID to network
   └─ Request N peers to pin chunks

5. CLEANUP
   ├─ Delete temporary checkpoint
   ├─ Keep manifest CID for restore
   └─ Log backup completion
```

## 🔙 Restore Workflow

```
1. FETCH MANIFEST
   ├─ Download manifest from IPFS by CID
   ├─ Parse manifest JSON
   ├─ Validate manifest structure
   └─ Check all chunk CIDs present

2. DOWNLOAD CHUNKS
   ├─ Create thread pool (parallel downloads)
   ├─ For each chunk:
   │  ├─ Download from IPFS (Bitswap)
   │  ├─ Decompress chunk
   │  ├─ Verify Blake3 hash
   │  └─ Store in temp buffer
   └─ Wait for all chunks

3. REASSEMBLE FILES
   ├─ Group chunks by original file
   ├─ Sort chunks by index
   ├─ Concatenate chunks
   └─ Write to output path

4. VERIFY DATABASE
   ├─ Open restored RocksDB (read-only)
   ├─ Run consistency checks
   ├─ Verify key count matches
   └─ Log restore completion

5. ACTIVATE
   ├─ Close read-only database
   ├─ Move to final location
   └─ Ready for use
```

## 🌐 Network Integration

### Gossip Layer (via q-network)

The manifest CID is gossiped to all network nodes:

```rust
// Pseudo-code for gossip integration
pub async fn gossip_manifest_cid(
    network: &mut QNetwork,
    manifest_cid: &str,
) -> Result<()> {
    let message = GossipMessage::DatabaseBackup {
        cid: manifest_cid.to_string(),
        timestamp: Utc::now(),
        node_id: network.local_peer_id(),
    };

    network.gossipsub_broadcast(
        "q-ipfs-backup",
        postcard::to_stdvec(&message)?
    ).await?;

    Ok(())
}
```

### Peer Discovery

Nodes discover IPFS peers through:
1. **mDNS** (local network)
2. **Kad-DHT** (wide area network)
3. **Bootstrap nodes** (configured endpoints)

### Pinning Coordination

When a node receives a backup manifest via gossip:

```rust
async fn handle_backup_manifest(cid: &str) -> Result<()> {
    // Download manifest
    let manifest = ipfs_client.get_manifest(cid).await?;

    // Check if we should pin (based on strategy)
    if should_pin_backup(&manifest) {
        // Pin manifest
        ipfs_client.pin_local(cid).await?;

        // Pin chunks (prioritize important files)
        for chunk_info in manifest.chunks.iter().take(100) {
            ipfs_client.pin_local(&chunk_info.cid).await?;
        }
    }

    Ok(())
}
```

## 🛠️ Implementation Tasks

### Phase 1: Core Infrastructure ✅
- [x] Create `q-ipfs-storage` crate
- [x] Implement `SnapshotManager`
- [x] Implement `ChunkManager`
- [x] Implement `Compressor`

### Phase 2: IPFS Integration 🔄
- [ ] Implement `manifest.rs` (storage manifest format)
- [ ] Implement `ipfs_client.rs` (libp2p IPFS operations)
- [ ] Implement `pinning.rs` (distributed pinning)
- [ ] Implement `storage.rs` (main orchestration)

### Phase 3: Network Integration 🔄
- [ ] Add gossip support for manifest CIDs
- [ ] Implement peer pinning coordination
- [ ] Add replication monitoring
- [ ] Create health check system

### Phase 4: CLI Tools 📋
- [ ] Create `q-backup` CLI tool
- [ ] Create `q-restore` CLI tool
- [ ] Add progress bars and status reporting
- [ ] Support batch operations

### Phase 5: API Integration 📋
- [ ] Add backup endpoints to q-api-server
- [ ] Add restore endpoints
- [ ] Add status/monitoring endpoints
- [ ] Support scheduled backups

## 📊 Configuration

```toml
[ipfs_storage]
# Snapshot settings
checkpoint_dir = "./checkpoints"
snapshot_retention = 10  # Keep last 10 snapshots

# Chunking settings
chunk_size = 262144  # 256 KB

# Compression settings
compression_type = "Zstd"
compression_level = 3

# IPFS settings
ipfs_api_url = "/ip4/127.0.0.1/tcp/5001"
enable_local_pinning = true
enable_remote_pinning = true
replication_factor = 3

# Network settings
gossip_topic = "q-ipfs-backup"
bootstrap_peers = [
    "/ip4/bootstrap1.quillon.xyz/tcp/4001/p2p/...",
    "/ip4/bootstrap2.quillon.xyz/tcp/4001/p2p/...",
]

# Performance settings
parallel_uploads = 10
parallel_downloads = 20
max_concurrent_pins = 50
```

## 🚀 Usage Examples

### Automatic Backup Every Hour

```rust
use q_ipfs_storage::*;
use tokio::time::{interval, Duration};

#[tokio::main]
async fn main() -> Result<()> {
    let config = StorageConfig::from_file("config.toml")?;
    let mut storage = IpfsRocksStorage::new(config).await?;

    let mut ticker = interval(Duration::from_secs(3600));

    loop {
        ticker.tick().await;

        match storage.backup_database("/data/rocksdb", BackupOptions::default()).await {
            Ok(cid) => println!("✅ Backup complete: {}", cid),
            Err(e) => eprintln!("❌ Backup failed: {}", e),
        }
    }
}
```

### Restore from Latest Backup

```rust
use q_ipfs_storage::*;

#[tokio::main]
async fn main() -> Result<()> {
    let config = StorageConfig::default();
    let mut storage = IpfsRocksStorage::new(config).await?;

    // Get latest manifest CID from gossip or local storage
    let manifest_cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi";

    storage.restore_database(
        manifest_cid,
        "/data/restored",
        RestoreOptions {
            verify_chunks: true,
            parallel_downloads: 20,
        }
    ).await?;

    println!("✅ Database restored successfully");
    Ok(())
}
```

## 🔬 Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock IPFS operations for fast testing
- Verify chunk integrity
- Test compression ratios

### Integration Tests
- Test full backup/restore cycle
- Verify data integrity after restore
- Test network failures and retries
- Test concurrent operations

### Performance Tests
- Measure backup speed (GB/s)
- Measure restore speed (GB/s)
- Test with large databases (>100 GB)
- Test with many small files vs few large files

## 📈 Performance Metrics

**Target Performance**:
- Backup: 100+ MB/s (compressed)
- Restore: 200+ MB/s (parallel downloads)
- Chunk verification: <1ms per chunk
- Compression: 60-80% size reduction
- Network overhead: <5% of data size

**Scalability**:
- Support databases up to 1 TB
- Support 10,000+ chunks per snapshot
- Support 100+ concurrent backups across network
- Support 1,000+ pinning nodes

## 🔐 Security Considerations

1. **Content Addressing**: IPFS CIDs are cryptographic hashes - content is verifiable
2. **Chunk Verification**: Blake3 hashes ensure data integrity
3. **Encryption**: Future support for encrypting chunks before upload
4. **Access Control**: Future support for private IPFS networks
5. **Byzantine Tolerance**: Verify chunks from multiple sources

## 📚 Dependencies

```toml
[dependencies]
rocksdb = "0.22"        # RocksDB bindings
libp2p = "0.53"         # IPFS/networking
tokio = "1.35"          # Async runtime
zstd = "0.13"           # Compression
blake3 = "1.5"          # Hashing
cid = "0.11"            # Content addressing
multihash = "0.19"      # Multi-hash support
serde_json = "1.0"      # Manifest serialization
```

## 🎯 Next Steps

1. **Complete Phase 2**: Implement IPFS client and manifest system
2. **Test with Real Data**: Use actual q-api-server database
3. **Network Testing**: Deploy to multiple nodes
4. **Performance Tuning**: Optimize chunk size and compression
5. **Documentation**: Write user guide and API docs
6. **Production Deploy**: Roll out to mainnet validators

---

**Status**: Foundation complete, IPFS integration in progress
**ETA for Beta**: 1-2 weeks
**ETA for Production**: 3-4 weeks
