# Adaptive Pruning Integration Roadmap

**Version**: v0.3.9-beta → v0.4.0+
**Status**: ✅ Core Implementation Complete - Integration Pending
**Date**: October 30, 2025

---

## Current Implementation Status

### ✅ Completed (v0.3.9-beta)

1. **Core Pruning Engine** (`crates/q-storage/src/pruning.rs`)
   - 4 pruning modes: Full, Archive, Light, Adaptive
   - 4-tier retention system (Critical, Recent, Checkpoints, Historical)
   - Dynamic disk space monitoring
   - Aggressive pruning mode (triggers at 10% free space)
   - Storage efficiency calculation
   - Comprehensive test coverage

2. **Storage Integration**
   - Module exported from `q-storage`
   - Public API available for `q-api-server`
   - Simulated pruning operations (no actual deletion yet)

3. **Documentation**
   - V0.3.9 implementation guide
   - Quantum physics whitepaper updated
   - Storage efficiency analysis (68.5-89% reduction)

### Storage Efficiency Results

For 110,000 block blockchain:

| Mode | Retained Blocks | Storage | Efficiency |
|------|----------------|---------|------------|
| Full | 110,000 | 4,004 MB | 0% (baseline) |
| Archive | 110,000 | ~2,800 MB | 30% |
| **Adaptive** | 34,615 | 1,260 MB | **68.5%** |
| Light | 11,000 | 439 MB | 89.0% |

---

## Phase 1: RocksDB Integration (v0.4.0)

### Objective
Enable actual block deletion from RocksDB storage layer.

### Implementation Tasks

#### Task 1.1: RocksDB Column Family Deletion

**File**: `crates/q-storage/src/pruning.rs`

Add method to delete blocks from RocksDB:

```rust
impl AdaptivePruningEngine {
    /// Delete blocks from RocksDB storage
    pub async fn prune_blocks_from_db(
        &self,
        kv_store: &Arc<RocksDBKV>,
        heights_to_prune: &[u64]
    ) -> Result<u64> {
        let mut deleted_count = 0u64;
        let mut total_space_freed = 0u64;

        for height in heights_to_prune {
            // Delete from blocks column family
            match kv_store.delete_block(*height).await {
                Ok(size) => {
                    deleted_count += 1;
                    total_space_freed += size;
                    debug!("🗑️  Pruned block {} ({} bytes)", height, size);
                }
                Err(e) => {
                    warn!("Failed to prune block {}: {}", height, e);
                }
            }
        }

        info!("✅ Pruned {} blocks, freed {} MB",
              deleted_count, total_space_freed / 1_000_000);

        Ok(total_space_freed)
    }
}
```

#### Task 1.2: Add Block Deletion to RocksDBKV

**File**: `crates/q-storage/src/kv.rs`

```rust
impl RocksDBKV {
    /// Delete block and return size freed
    pub async fn delete_block(&self, height: u64) -> Result<u64> {
        let key = format!("block_{}", height);

        // Get block size before deletion (for stats)
        let block_size = self.get_block_size(height).await?;

        // Delete from blocks CF
        let cf_blocks = self.db.cf_handle(CF_BLOCKS)
            .context("Failed to get blocks CF")?;

        self.db.delete_cf(cf_blocks, key.as_bytes())
            .context("Failed to delete block")?;

        // Also delete DAG vertices if they exist
        self.delete_dag_vertices_for_block(height).await?;

        Ok(block_size)
    }

    async fn get_block_size(&self, height: u64) -> Result<u64> {
        let key = format!("block_{}", height);
        let cf_blocks = self.db.cf_handle(CF_BLOCKS)
            .context("Failed to get blocks CF")?;

        match self.db.get_cf(cf_blocks, key.as_bytes())? {
            Some(data) => Ok(data.len() as u64),
            None => Ok(0),
        }
    }

    async fn delete_dag_vertices_for_block(&self, height: u64) -> Result<()> {
        let cf_dag = self.db.cf_handle(CF_DAG_VERTICES)
            .context("Failed to get DAG vertices CF")?;

        // Delete all DAG vertices associated with this block
        let prefix = format!("vertex_block_{}_", height);

        // Iterator over keys with prefix
        let iter = self.db.prefix_iterator_cf(cf_dag, prefix.as_bytes());

        for item in iter {
            let (key, _) = item?;
            self.db.delete_cf(cf_dag, &key)?;
        }

        Ok(())
    }
}
```

#### Task 1.3: Integrate Pruning into Blockchain Sync

**File**: `crates/q-api-server/src/main.rs`

```rust
// After blockchain initialization, start pruning service
let pruning_config = PruningConfig::default();
let pruning_engine = Arc::new(RwLock::new(
    AdaptivePruningEngine::new(&db_path, pruning_config)
));

// Spawn pruning task
let pruning_engine_clone = Arc::clone(&pruning_engine);
let blockchain_clone = Arc::clone(&blockchain);
tokio::spawn(async move {
    loop {
        // Wait for auto-prune interval (default: 1 hour)
        tokio::time::sleep(Duration::from_secs(3600)).await;

        let engine = pruning_engine_clone.read().await;
        let blockchain = blockchain_clone.read().await;
        let current_height = blockchain.get_height();

        info!("🔄 Starting scheduled pruning (height: {})", current_height);

        match engine.execute_pruning_with_deletion(
            current_height,
            &blockchain.kv_store
        ).await {
            Ok(stats) => {
                info!("✅ Pruning complete: {} blocks pruned, {} MB saved",
                      stats.pruned_blocks, stats.space_saved / 1_000_000);
            }
            Err(e) => {
                error!("❌ Pruning failed: {}", e);
            }
        }
    }
});
```

---

## Phase 2: Background Scheduler (v0.4.1)

### Objective
Implement configurable background pruning with intelligent scheduling.

### Features

#### Feature 2.1: Adaptive Scheduling

Prune more frequently when disk space is low:

```rust
pub struct PruningScheduler {
    config: PruningConfig,
    last_prune: SystemTime,
    consecutive_failures: u32,
}

impl PruningScheduler {
    pub fn next_prune_interval(&self, disk_space_ratio: f64) -> Duration {
        let base_interval = self.config.auto_prune_interval;

        // Increase frequency when disk space is low
        if disk_space_ratio < 0.1 {
            Duration::from_secs(base_interval / 4)  // Every 15 minutes
        } else if disk_space_ratio < 0.2 {
            Duration::from_secs(base_interval / 2)  // Every 30 minutes
        } else {
            Duration::from_secs(base_interval)      // Every 1 hour
        }
    }

    pub async fn run_pruning_loop(&mut self, engine: Arc<RwLock<AdaptivePruningEngine>>) {
        loop {
            let disk_space_ratio = self.get_disk_space_ratio();
            let next_interval = self.next_prune_interval(disk_space_ratio);

            tokio::time::sleep(next_interval).await;

            // Execute pruning
            match self.execute_pruning_cycle(&engine).await {
                Ok(_) => {
                    self.consecutive_failures = 0;
                }
                Err(e) => {
                    self.consecutive_failures += 1;
                    warn!("Pruning failed (attempt {}): {}", self.consecutive_failures, e);

                    // Back off if failures accumulate
                    if self.consecutive_failures > 3 {
                        warn!("⚠️  Multiple pruning failures, increasing interval");
                        tokio::time::sleep(Duration::from_secs(3600)).await;
                    }
                }
            }
        }
    }
}
```

#### Feature 2.2: Pruning Windows

Avoid pruning during peak usage:

```rust
pub struct PruningWindow {
    pub start_hour: u8,  // 0-23
    pub end_hour: u8,    // 0-23
}

impl PruningScheduler {
    pub fn is_in_pruning_window(&self) -> bool {
        use chrono::Timelike;
        let now = chrono::Local::now();
        let hour = now.hour() as u8;

        // Example: Only prune between 2 AM - 6 AM local time
        hour >= self.config.pruning_window.start_hour &&
        hour < self.config.pruning_window.end_hour
    }
}
```

---

## Phase 3: P2P Pruning Coordination (v0.4.2)

### Objective
Ensure ≥67% of network nodes retain full data for network health.

### Implementation

#### Feature 3.1: Advertise Pruning Mode via Gossipsub

**File**: `crates/q-network/src/unified_network_manager.rs`

```rust
// Publish pruning mode announcement
pub async fn announce_pruning_mode(&self, mode: PruningMode) {
    let announcement = PruningAnnouncement {
        peer_id: self.local_peer_id,
        pruning_mode: mode,
        retained_height_range: self.get_retained_height_range(),
        timestamp: SystemTime::now(),
    };

    self.gossipsub.publish(
        Topic::new("/qnk/testnet/pruning_announcements"),
        serde_json::to_vec(&announcement).unwrap()
    );
}
```

#### Feature 3.2: Network Health Monitor

```rust
pub struct NetworkHealthMonitor {
    peer_pruning_modes: HashMap<PeerId, PruningMode>,
}

impl NetworkHealthMonitor {
    pub fn calculate_full_node_ratio(&self) -> f64 {
        let total_peers = self.peer_pruning_modes.len();
        if total_peers == 0 {
            return 0.0;
        }

        let full_nodes = self.peer_pruning_modes.values()
            .filter(|mode| matches!(mode, PruningMode::Full))
            .count();

        full_nodes as f64 / total_peers as f64
    }

    pub fn should_remain_full_node(&self) -> bool {
        // If network has <67% full nodes, stay in Full mode
        self.calculate_full_node_ratio() < 0.67
    }
}
```

---

## Phase 4: Checkpoint Fast Sync (v0.4.3)

### Objective
Allow light clients to sync instantly from checkpoints.

### Implementation

#### Feature 4.1: Checkpoint Download Protocol

```rust
pub struct CheckpointSyncManager {
    pub async fn download_checkpoints(&self) -> Result<Vec<Block>> {
        let mut checkpoints = Vec::new();

        // Download critical checkpoints: 0, 55K, 110K
        for height in [0, 55_000, 110_000] {
            match self.download_checkpoint_block(height).await {
                Ok(block) => checkpoints.push(block),
                Err(e) => {
                    warn!("Failed to download checkpoint {}: {}", height, e);
                }
            }
        }

        Ok(checkpoints)
    }

    async fn download_checkpoint_block(&self, height: u64) -> Result<Block> {
        // Request via gossipsub RPC
        let request = CheckpointRequest { height };

        self.network_manager.request_checkpoint(request).await
    }
}
```

#### Feature 4.2: State Reconstruction

```rust
pub struct StateReconstructor {
    pub async fn reconstruct_from_checkpoints(
        &self,
        checkpoints: &[Block]
    ) -> Result<BlockchainState> {
        let mut state = BlockchainState::new();

        // Apply checkpoints in order
        for checkpoint in checkpoints {
            state.apply_block(checkpoint)?;
        }

        // Validate state hash
        self.validate_state_hash(&state, checkpoints.last().unwrap())?;

        Ok(state)
    }
}
```

---

## Phase 5: Compression & Archives (v0.5.0)

### Objective
Add zstd compression for archived checkpoint blocks.

### Implementation

```rust
use zstd::stream::{encode_all, decode_all};

impl AdaptivePruningEngine {
    pub async fn compress_checkpoint(&self, block: &Block) -> Result<Vec<u8>> {
        let block_bytes = bincode::serialize(block)?;

        // Compress with zstd (level 19 for maximum compression)
        let compressed = encode_all(&block_bytes[..], 19)?;

        info!("Compressed checkpoint {} from {} to {} bytes ({:.1}% reduction)",
              block.height,
              block_bytes.len(),
              compressed.len(),
              100.0 * (1.0 - compressed.len() as f64 / block_bytes.len() as f64));

        Ok(compressed)
    }

    pub async fn decompress_checkpoint(&self, compressed: &[u8]) -> Result<Block> {
        let decompressed = decode_all(compressed)?;
        Ok(bincode::deserialize(&decompressed)?)
    }
}
```

---

## Configuration API

### Environment Variables

```bash
# Pruning mode: full, archive, light, adaptive
Q_PRUNING_MODE=adaptive

# Retention period (days)
Q_PRUNE_RETAIN_DAYS=30

# Minimum free disk space (GB)
Q_PRUNE_MIN_FREE_SPACE=10

# Checkpoint interval (blocks)
Q_PRUNE_CHECKPOINT_INTERVAL=55000

# Auto-prune interval (seconds)
Q_PRUNE_INTERVAL=3600

# Pruning window (hours, 24-hour format)
Q_PRUNE_WINDOW_START=2
Q_PRUNE_WINDOW_END=6

# Aggressive pruning threshold (0.0-1.0)
Q_PRUNE_AGGRESSIVE_THRESHOLD=0.1
```

### Runtime API Endpoints

```bash
# Get current pruning status
GET /api/v1/pruning/status

# Update pruning configuration
POST /api/v1/pruning/config
{
  "mode": "adaptive",
  "retain_days": 30,
  "auto_prune_interval": 3600
}

# Trigger manual pruning
POST /api/v1/pruning/execute

# Get pruning statistics
GET /api/v1/pruning/stats
```

---

## Testing Strategy

### Unit Tests

```bash
# Test pruning engine
cargo test --package q-storage pruning

# Test RocksDB deletion
cargo test --package q-storage test_delete_block

# Test tier classification
cargo test --package q-storage test_retention_tiers
```

### Integration Tests

```bash
# Test full pruning workflow
cargo test --package q-api-server --test pruning_integration

# Test disk space monitoring
cargo test --package q-storage test_disk_space_adaptive

# Test network health preservation
cargo test --package q-network test_pruning_network_health
```

### Performance Tests

```bash
# Benchmark pruning operations
cargo bench --package q-storage pruning_performance

# Test 1M block blockchain pruning
cargo test --package q-storage --release test_large_scale_pruning -- --ignored
```

---

## Deployment Roadmap

### v0.4.0 (1-2 weeks)
- ✅ RocksDB integration
- ✅ Background scheduler
- ✅ Runtime configuration API

### v0.4.1 (2-3 weeks)
- ✅ Adaptive scheduling
- ✅ Pruning windows
- ✅ Enhanced monitoring

### v0.4.2 (3-4 weeks)
- ✅ P2P pruning coordination
- ✅ Network health monitoring
- ✅ Gossipsub announcements

### v0.4.3 (4-6 weeks)
- ✅ Checkpoint fast sync
- ✅ State reconstruction
- ✅ Light client optimization

### v0.5.0 (6-8 weeks)
- ✅ Compression integration
- ✅ Archive mode optimization
- ✅ Production hardening

---

## Success Metrics

### Performance
- Pruning operation completes in <5 minutes for 100K blocks
- Disk I/O reduction: 30-40%
- Database size reduction: 68.5%+
- Sync time improvement: 50% faster for light clients

### Reliability
- Zero data loss during pruning
- Graceful degradation on pruning failures
- Network maintains ≥67% full nodes
- Re-org protection maintained (24 blocks)

### User Experience
- Zero-configuration operation by default
- Clear monitoring and diagnostics
- Reversible mode changes (Full ↔ Adaptive)
- Transparent network health status

---

**Status**: v0.3.9-beta implementation complete, ready for Phase 1 integration
**Next Milestone**: v0.4.0 with RocksDB deletion (ETA: 2 weeks)
