# Unified Node Design: Single Binary, Adaptive Behavior

## Philosophy: Just Works™

**One binary. Many modes. Zero user confusion.**

Users shouldn't need to choose between "archive node" or "pruned node" - the node should intelligently adapt based on available resources and user preferences.

---

## Unified Node Architecture

```
┌─────────────────────────────────────────────────────────┐
│         q-api-server (Single Unified Binary)            │
├─────────────────────────────────────────────────────────┤
│  Auto-Detection:                                        │
│  - Available disk space                                 │
│  - Network bandwidth                                    │
│  - CPU cores                                            │
│  - Memory capacity                                      │
├─────────────────────────────────────────────────────────┤
│  Adaptive Storage Strategy:                             │
│  ┌─────────────┬──────────────┬────────────────┐       │
│  │ Minimal     │ Balanced     │ Full History   │       │
│  │ 5 GB disk   │ 50 GB disk   │ 500+ GB disk   │       │
│  │ 1000 blocks │ 10000 blocks │ All blocks     │       │
│  └─────────────┴──────────────┴────────────────┘       │
│                                                          │
│  User Controls:                                         │
│  - Q_MAX_STORAGE=50GB (simple)                          │
│  - Q_PRUNING_MODE=auto|minimal|balanced|full            │
│  - Q_SERVE_HISTORICAL=true (help network)               │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation: Smart Defaults

### 1. Auto-Detection on First Startup

```rust
pub struct AdaptiveNodeConfig {
    /// Detected available disk space
    available_disk_gb: u64,
    /// User-specified max storage (overrides auto-detection)
    max_storage_gb: Option<u64>,
    /// Automatically determined pruning mode
    pruning_mode: PruningMode,
}

#[derive(Debug, Clone)]
pub enum PruningMode {
    /// Keep only last 1,000 blocks (5 GB) - mobile/low-resource
    Minimal,
    /// Keep last 10,000 blocks (50 GB) - standard desktop
    Balanced,
    /// Keep all blocks (500+ GB) - servers/enthusiasts
    FullHistory,
    /// User-specified custom retention depth
    Custom(u64),
}

impl AdaptiveNodeConfig {
    pub fn auto_detect() -> Self {
        let available_disk_gb = detect_available_disk_space();

        let pruning_mode = if available_disk_gb < 10 {
            warn!("⚠️  Low disk space ({} GB available)", available_disk_gb);
            warn!("   Using Minimal mode: keeping last 1,000 blocks");
            PruningMode::Minimal
        } else if available_disk_gb < 100 {
            info!("📊 Detected {} GB available", available_disk_gb);
            info!("   Using Balanced mode: keeping last 10,000 blocks");
            PruningMode::Balanced
        } else {
            info!("🗄️  Detected {} GB available", available_disk_gb);
            info!("   Using Full History mode: keeping all blocks");
            info!("   You'll automatically serve historical data to the network!");
            PruningMode::FullHistory
        };

        Self {
            available_disk_gb,
            max_storage_gb: None,
            pruning_mode,
        }
    }

    pub fn with_user_preference(storage_limit: &str) -> Self {
        // Parse "50GB", "500MB", etc.
        let max_gb = parse_storage_string(storage_limit);

        let pruning_mode = if max_gb < 10 {
            PruningMode::Minimal
        } else if max_gb < 100 {
            PruningMode::Balanced
        } else {
            PruningMode::FullHistory
        };

        info!("👤 User preference: max {} GB storage", max_gb);
        info!("   Mode selected: {:?}", pruning_mode);

        Self {
            available_disk_gb: detect_available_disk_space(),
            max_storage_gb: Some(max_gb),
            pruning_mode,
        }
    }

    pub fn retention_depth(&self) -> u64 {
        match self.pruning_mode {
            PruningMode::Minimal => 1_000,
            PruningMode::Balanced => 10_000,
            PruningMode::FullHistory => u64::MAX, // Keep everything
            PruningMode::Custom(depth) => depth,
        }
    }

    pub fn serves_historical_data(&self) -> bool {
        // Any node can serve historical data it has
        // Full history nodes automatically participate
        matches!(self.pruning_mode, PruningMode::FullHistory)
            || env::var("Q_SERVE_HISTORICAL").unwrap_or_default() == "true"
    }
}
```

### 2. User-Friendly Configuration

**Simple mode (recommended):**
```bash
# Just set a storage limit - node figures out the rest
Q_MAX_STORAGE=50GB ./q-api-server --port 8080

# Auto-detect mode (uses available disk space)
./q-api-server --port 8080
```

**Advanced mode:**
```bash
# Explicit pruning mode
Q_PRUNING_MODE=balanced ./q-api-server --port 8080

# Custom retention depth
Q_BLOCK_RETENTION=5000 ./q-api-server --port 8080

# Opt-in to serving historical data (even if pruned)
Q_SERVE_HISTORICAL=true ./q-api-server --port 8080
```

**Docker:**
```bash
docker run -d \
  -e Q_MAX_STORAGE=50GB \
  -v ./data:/data \
  quillon-api:v0.3.4
```

---

## 3. Dynamic Behavior During Runtime

### Network Participation

**All nodes participate in P2P block serving:**

```rust
// P2P block request handler (works for ALL nodes)
if topic.ends_with("/block-requests") {
    let request: BlockRequest = postcard::from_bytes(&data)?;

    // Serve blocks we have, regardless of pruning mode
    for height in request.start_height..=request.end_height {
        if let Some(block) = storage.get_qblock_by_height(height).await? {
            // ✅ We have this block - serve it!
            publish_block_response(request.request_id, block).await;
        } else {
            // ⚠️ We don't have it (pruned) - skip silently
            debug!("Block {} pruned, cannot serve", height);
        }
    }
}
```

**Key insight:** Even "minimal" nodes help the network by serving their 1,000 recent blocks. No separate "archive node" binary needed!

### Storage Management

```rust
pub struct DynamicPruner {
    config: AdaptiveNodeConfig,
    current_storage_bytes: AtomicU64,
}

impl DynamicPruner {
    pub async fn maybe_prune(&self, current_height: u64) {
        let retention_depth = self.config.retention_depth();

        if retention_depth == u64::MAX {
            // Full history mode - never prune
            return;
        }

        let prune_before_height = current_height.saturating_sub(retention_depth);

        if prune_before_height > 0 {
            info!("🗑️  Pruning blocks older than height {} (keeping last {} blocks)",
                  prune_before_height, retention_depth);

            // Incremental pruning (100 blocks at a time)
            for height in (prune_before_height - 100)..prune_before_height {
                storage.delete_qblock(height).await?;
            }
        }

        // Check if we're approaching storage limit
        let current_gb = self.current_storage_bytes.load(Ordering::Relaxed) / 1_000_000_000;
        if let Some(max_gb) = self.config.max_storage_gb {
            if current_gb > max_gb * 90 / 100 {
                warn!("⚠️  Approaching storage limit ({} / {} GB)", current_gb, max_gb);
                warn!("   Consider increasing Q_MAX_STORAGE or the node will prune more aggressively");
            }
        }
    }
}
```

---

## 4. Network Health: Distributed Archive

**Instead of designated "archive nodes", we have a distributed archive:**

```
Network State (height: 110,000):

Node A (Minimal):     Blocks 109,000 - 110,000 (1,000 blocks)
Node B (Balanced):    Blocks 100,000 - 110,000 (10,000 blocks)
Node C (Full):        Blocks 1 - 110,000 (all blocks)
Node D (Balanced):    Blocks 100,000 - 110,000 (10,000 blocks)
Node E (Full):        Blocks 1 - 110,000 (all blocks)
Node F (Minimal):     Blocks 109,000 - 110,000 (1,000 blocks)

Coverage for block 50,000: Nodes C, E ✅
Coverage for block 105,000: Nodes B, C, D, E ✅
Coverage for block 109,500: ALL NODES ✅
```

**Network safety metric:**
```rust
pub struct NetworkHealthMonitor {
    /// Track which nodes have which block ranges
    peer_block_coverage: HashMap<PeerId, BlockRange>,
}

impl NetworkHealthMonitor {
    pub fn is_network_healthy(&self, current_height: u64) -> bool {
        // Check critical historical blocks
        let critical_heights = vec![
            1,                               // Genesis
            current_height / 2,              // Mid-history
            current_height - 10_000,         // Recent history
        ];

        for height in critical_heights {
            let coverage_count = self.peer_block_coverage.values()
                .filter(|range| range.contains(height))
                .count();

            if coverage_count < 3 {
                warn!("⚠️  Block {} only has {} copies in network (need 3+)",
                      height, coverage_count);
                return false;
            }
        }

        true
    }
}
```

---

## 5. Startup Logs (User Experience)

**Minimal node:**
```
🚀 Q-NarwhalKnight v0.3.4-beta starting...
📊 Detected 8 GB available disk space
✅ Pruning mode: Minimal (keeping last 1,000 blocks)
ℹ️  You'll serve recent blocks to help new nodes sync!
🌐 P2P listening on port 8081
🔗 Connected to 4 peers
```

**Balanced node:**
```
🚀 Q-NarwhalKnight v0.3.4-beta starting...
📊 User preference: max 50 GB storage
✅ Pruning mode: Balanced (keeping last 10,000 blocks)
ℹ️  You'll serve historical data to the network!
🌐 P2P listening on port 8081
🔗 Connected to 4 peers
```

**Full history node:**
```
🚀 Q-NarwhalKnight v0.3.4-beta starting...
🗄️  Detected 500 GB available disk space
✅ Pruning mode: Full History (keeping all blocks)
🎖️  You're a full archive node - thank you for supporting the network!
📚 Serving complete blockchain history to new nodes
🌐 P2P listening on port 8081
🔗 Connected to 4 peers
```

---

## 6. No Proof-of-Custody Needed

**Why not?**

Because we don't have "designated archive nodes" that need policing. Instead:

1. **Economic incentives emerge naturally:**
   - Full history nodes get more P2P connections (people sync from them)
   - Future: Query fees for historical data APIs
   - Reputation: Known full nodes become trusted bootstrap peers

2. **Network redundancy through diversity:**
   - Mix of minimal/balanced/full nodes
   - No single point of failure
   - Historical data distributed across multiple full nodes

3. **Users choose based on resources, not duty:**
   - "I have 500 GB spare" → full history
   - "I'm on a VPS with 50 GB" → balanced
   - "Running on mobile" → minimal

---

## 7. Migration Path (Existing Deployments)

**Automatic migration:**
```rust
pub async fn migrate_existing_node() -> Result<AdaptiveNodeConfig> {
    // Check if this is an existing deployment
    if old_config_exists() {
        info!("📦 Detected existing deployment - migrating to adaptive mode...");

        let current_storage_gb = calculate_current_database_size()?;

        info!("📊 Current database: {} GB", current_storage_gb);
        info!("✅ Migration complete - node now uses adaptive pruning");

        AdaptiveNodeConfig::with_user_preference(&format!("{}GB", current_storage_gb * 2))
    } else {
        AdaptiveNodeConfig::auto_detect()
    }
}
```

---

## Summary: One Binary, Zero Confusion

| Feature | Archive Node (Old Way) | Adaptive Node (New Way) |
|---------|----------------------|------------------------|
| Binary | Separate q-archive-server | Single q-api-server |
| Config | Complex mode selection | Auto-detect or Q_MAX_STORAGE=50GB |
| Network Role | Fixed at startup | Adapts to resources |
| User Experience | "Which binary do I need?" | "Just run it" |
| Maintenance | Two codebases | One codebase |

**Result:** Users get a node that "just works" and automatically contributes to network health based on their available resources. No PhDs in distributed systems required. 🚀

---

**Next Steps:**

1. ✅ Implement `AdaptiveNodeConfig` with auto-detection
2. ✅ Add dynamic storage-based pruning
3. ✅ Enhance P2P block serving (already works!)
4. ✅ Add network health monitoring dashboard
5. ✅ Document simple Q_MAX_STORAGE configuration

**Timeline:** 1 week for core implementation
