# Adaptive Node Implementation Plan - Week-by-Week

## Vision: The Most User-Friendly Blockchain Node Ever Built

**Goal:** Single binary that adapts to any device (mobile → server) with zero manual configuration.

---

## Week 1: Core Adaptive Engine

### Day 1-2: System Resource Detection

**File:** `crates/q-node-config/src/resource_profile.rs`

```rust
use sysinfo::{System, SystemExt, DiskExt};

#[derive(Debug, Clone, PartialEq)]
pub enum DeviceType {
    Mobile,    // < 5 GB disk, < 2 GB RAM
    Desktop,   // 5-100 GB disk, 2-8 GB RAM
    Server,    // > 100 GB disk, > 8 GB RAM
    IoT,       // Embedded devices
}

#[derive(Debug, Clone)]
pub struct SystemResourceProfile {
    pub disk_space_gb: u64,
    pub memory_gb: u64,
    pub cpu_cores: u64,
    pub device_type: DeviceType,
    pub detected_at: chrono::DateTime<chrono::Utc>,
}

impl SystemResourceProfile {
    pub fn detect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        // Detect available disk space (use mount point of current directory)
        let disk_space_gb = Self::detect_disk_space();

        // Detect available memory
        let memory_gb = (sys.available_memory() / 1_000_000_000) as u64;

        // CPU cores
        let cpu_cores = sys.cpus().len() as u64;

        // Classify device
        let device_type = Self::classify_device(disk_space_gb, memory_gb, cpu_cores);

        Self {
            disk_space_gb,
            memory_gb,
            cpu_cores,
            device_type,
            detected_at: chrono::Utc::now(),
        }
    }

    fn detect_disk_space() -> u64 {
        let mut sys = System::new_all();
        sys.refresh_disks_list();

        // Get disk where data directory is located
        for disk in sys.disks() {
            if std::env::current_dir()
                .unwrap()
                .starts_with(disk.mount_point())
            {
                return disk.available_space() / 1_000_000_000;
            }
        }

        // Fallback: use largest disk
        sys.disks()
            .iter()
            .map(|d| d.available_space())
            .max()
            .unwrap_or(0) / 1_000_000_000
    }

    fn classify_device(disk: u64, memory: u64, cores: u64) -> DeviceType {
        match (disk, memory, cores) {
            (d, m, _) if d < 5 || m < 2 => DeviceType::Mobile,
            (d, m, c) if d < 100 && m < 8 => DeviceType::Desktop,
            (d, m, c) if d >= 100 && m >= 8 => DeviceType::Server,
            _ => DeviceType::Desktop,
        }
    }

    pub fn display_detection_summary(&self) {
        info!("📊 System Resources Detected:");
        info!("   💾 Available Disk: {} GB", self.disk_space_gb);
        info!("   🧠 Available RAM: {} GB", self.memory_gb);
        info!("   ⚙️  CPU Cores: {}", self.cpu_cores);
        info!("   🖥️  Device Type: {:?}", self.device_type);
    }
}
```

**Dependencies to add to `Cargo.toml`:**
```toml
[dependencies]
sysinfo = "0.30"
```

### Day 3-4: Adaptive Configuration Engine

**File:** `crates/q-node-config/src/adaptive_config.rs`

```rust
use crate::resource_profile::{SystemResourceProfile, DeviceType};

#[derive(Debug, Clone, PartialEq)]
pub enum PruningMode {
    Minimal,        // 1,000 blocks (~5 GB)
    Balanced,       // 10,000 blocks (~50 GB)
    FullHistory,    // All blocks (~500+ GB)
    Custom(u64),    // User-specified depth
}

impl PruningMode {
    pub fn retention_depth(&self) -> u64 {
        match self {
            Self::Minimal => 1_000,
            Self::Balanced => 10_000,
            Self::FullHistory => u64::MAX,
            Self::Custom(depth) => *depth,
        }
    }

    pub fn estimated_storage_gb(&self, current_height: u64) -> u64 {
        // Average block size: ~36 KB
        let blocks = self.retention_depth().min(current_height);
        (blocks * 36_000) / 1_000_000_000
    }

    pub fn display_name(&self) -> &str {
        match self {
            Self::Minimal => "Minimal (1K blocks, ~5 GB)",
            Self::Balanced => "Balanced (10K blocks, ~50 GB)",
            Self::FullHistory => "Full History (all blocks)",
            Self::Custom(d) => "Custom",
        }
    }

    pub fn emoji(&self) -> &str {
        match self {
            Self::Minimal => "🔵",
            Self::Balanced => "🟡",
            Self::FullHistory => "🔴",
            Self::Custom(_) => "⚙️",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveNodeConfig {
    pub profile: SystemResourceProfile,
    pub pruning_mode: PruningMode,
    pub serve_historical: bool,
    pub max_concurrent_syncs: u32,
    pub user_override: bool,
}

impl AdaptiveNodeConfig {
    /// Auto-detect optimal configuration
    pub fn auto_detect() -> Self {
        let profile = SystemResourceProfile::detect();
        profile.display_detection_summary();

        let pruning_mode = Self::recommend_pruning_mode(&profile);
        let serve_historical = Self::should_serve_historical(&profile, &pruning_mode);
        let max_concurrent_syncs = Self::recommend_sync_limit(&profile);

        info!("");
        info!("✅ Adaptive Configuration:");
        info!("   {} Mode: {}", pruning_mode.emoji(), pruning_mode.display_name());
        info!("   📤 Serve Historical Data: {}", if serve_historical { "Yes" } else { "No" });
        info!("   🔄 Max Concurrent Syncs: {}", max_concurrent_syncs);

        if pruning_mode == PruningMode::FullHistory {
            info!("");
            info!("🎖️  You're a full archive node - thank you for supporting the network!");
        }

        Self {
            profile,
            pruning_mode,
            serve_historical,
            max_concurrent_syncs,
            user_override: false,
        }
    }

    /// User-specified storage limit
    pub fn with_storage_limit(limit_str: &str) -> Self {
        let profile = SystemResourceProfile::detect();
        profile.display_detection_summary();

        let limit_gb = Self::parse_storage_string(limit_str);

        let pruning_mode = if limit_gb < 10 {
            PruningMode::Minimal
        } else if limit_gb < 100 {
            PruningMode::Balanced
        } else {
            PruningMode::FullHistory
        };

        info!("");
        info!("👤 User Preference: {} GB max storage", limit_gb);
        info!("   {} Mode: {}", pruning_mode.emoji(), pruning_mode.display_name());

        let serve_historical = Self::should_serve_historical(&profile, &pruning_mode);
        let max_concurrent_syncs = Self::recommend_sync_limit(&profile);

        Self {
            profile,
            pruning_mode,
            serve_historical,
            max_concurrent_syncs,
            user_override: true,
        }
    }

    fn recommend_pruning_mode(profile: &SystemResourceProfile) -> PruningMode {
        match profile.device_type {
            DeviceType::Mobile | DeviceType::IoT => PruningMode::Minimal,
            DeviceType::Desktop => {
                if profile.disk_space_gb < 50 {
                    PruningMode::Balanced
                } else {
                    PruningMode::FullHistory
                }
            }
            DeviceType::Server => PruningMode::FullHistory,
        }
    }

    fn should_serve_historical(profile: &SystemResourceProfile, mode: &PruningMode) -> bool {
        // Mobile/IoT: Don't serve historical (bandwidth/battery constraints)
        if matches!(profile.device_type, DeviceType::Mobile | DeviceType::IoT) {
            return false;
        }

        // Minimal nodes: Only serve if explicitly opted in
        if mode == &PruningMode::Minimal {
            return std::env::var("Q_SERVE_HISTORICAL").unwrap_or_default() == "true";
        }

        // Balanced and Full: Always serve
        true
    }

    fn recommend_sync_limit(profile: &SystemResourceProfile) -> u32 {
        match profile.device_type {
            DeviceType::Mobile | DeviceType::IoT => 2,
            DeviceType::Desktop => 5,
            DeviceType::Server => 20,
        }
    }

    fn parse_storage_string(s: &str) -> u64 {
        let s = s.to_uppercase();
        if let Some(num) = s.strip_suffix("GB") {
            num.trim().parse().unwrap_or(50)
        } else if let Some(num) = s.strip_suffix("MB") {
            num.trim().parse::<u64>().unwrap_or(50000) / 1000
        } else {
            s.trim().parse().unwrap_or(50)
        }
    }
}
```

### Day 5: Integration into main.rs

**File:** `crates/q-api-server/src/main.rs` (add near startup)

```rust
use q_node_config::adaptive_config::AdaptiveNodeConfig;

// After argument parsing, before storage initialization
info!("🚀 Q-NarwhalKnight v0.3.4-beta starting...");
info!("");

// Adaptive configuration
let adaptive_config = if let Ok(storage_limit) = std::env::var("Q_MAX_STORAGE") {
    AdaptiveNodeConfig::with_storage_limit(&storage_limit)
} else if let Ok(tier) = std::env::var("Q_NODE_TIER") {
    match tier.to_lowercase().as_str() {
        "minimal" | "mobile" => AdaptiveNodeConfig::with_storage_limit("5GB"),
        "balanced" | "desktop" => AdaptiveNodeConfig::with_storage_limit("50GB"),
        "full" | "archive" | "server" => AdaptiveNodeConfig::with_storage_limit("500GB"),
        _ => AdaptiveNodeConfig::auto_detect(),
    }
} else {
    AdaptiveNodeConfig::auto_detect()
};

info!("");
```

### Day 5: Testing

```bash
# Test auto-detection
./q-api-server --port 8090

# Test with storage limit
Q_MAX_STORAGE=50GB ./q-api-server --port 8090

# Test with tier
Q_NODE_TIER=balanced ./q-api-server --port 8090
```

**Expected logs:**
```
🚀 Q-NarwhalKnight v0.3.4-beta starting...

📊 System Resources Detected:
   💾 Available Disk: 500 GB
   🧠 Available RAM: 16 GB
   ⚙️  CPU Cores: 8
   🖥️  Device Type: Server

✅ Adaptive Configuration:
   🔴 Mode: Full History (all blocks)
   📤 Serve Historical Data: Yes
   🔄 Max Concurrent Syncs: 20

🎖️  You're a full archive node - thank you for supporting the network!
```

---

## Week 2: Dynamic Pruning & Resource Monitoring

### Day 6-7: Dynamic Pruner

**File:** `crates/q-storage/src/dynamic_pruner.rs`

```rust
use q_node_config::adaptive_config::{AdaptiveNodeConfig, PruningMode};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub struct DynamicPruner {
    config: AdaptiveNodeConfig,
    current_storage_bytes: Arc<AtomicU64>,
    last_prune_height: Arc<AtomicU64>,
}

impl DynamicPruner {
    pub fn new(config: AdaptiveNodeConfig) -> Self {
        Self {
            config,
            current_storage_bytes: Arc::new(AtomicU64::new(0)),
            last_prune_height: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Check if pruning is needed (called every new block)
    pub async fn maybe_prune(&self, storage: &QStorageEngine, current_height: u64) -> Result<()> {
        let retention_depth = self.config.pruning_mode.retention_depth();

        // Full history mode: never prune
        if retention_depth == u64::MAX {
            return Ok(());
        }

        // Calculate prune threshold
        let prune_before_height = current_height.saturating_sub(retention_depth);

        if prune_before_height == 0 {
            return Ok(()); // Not enough blocks yet
        }

        let last_prune = self.last_prune_height.load(Ordering::Relaxed);

        // Prune incrementally (every 100 blocks)
        if prune_before_height > last_prune + 100 {
            info!("🗑️  Incremental pruning: removing blocks {} to {}",
                  last_prune, prune_before_height);

            // Prune in small batches to avoid I/O spikes
            for height in last_prune..prune_before_height {
                storage.delete_qblock(height).await?;
            }

            self.last_prune_height.store(prune_before_height, Ordering::Relaxed);

            info!("✅ Pruned {} blocks (keeping last {} blocks)",
                  prune_before_height - last_prune, retention_depth);
        }

        Ok(())
    }

    /// Update current storage usage
    pub async fn update_storage_metrics(&self, storage: &QStorageEngine) {
        if let Ok(size_bytes) = storage.calculate_database_size().await {
            self.current_storage_bytes.store(size_bytes, Ordering::Relaxed);

            let size_gb = size_bytes / 1_000_000_000;
            debug!("📊 Current database size: {} GB", size_gb);
        }
    }
}
```

### Day 8-9: Resource Monitor

**File:** `crates/q-node-config/src/resource_monitor.rs`

```rust
use tokio::time::{interval, Duration};
use std::sync::Arc;

pub struct ResourceMonitor {
    config: Arc<AdaptiveNodeConfig>,
    storage: Arc<QStorageEngine>,
}

impl ResourceMonitor {
    pub fn new(config: AdaptiveNodeConfig, storage: Arc<QStorageEngine>) -> Self {
        Self {
            config: Arc::new(config),
            storage,
        }
    }

    pub async fn start_monitoring(&self) {
        let config = self.config.clone();
        let storage = self.storage.clone();

        tokio::spawn(async move {
            let mut check_interval = interval(Duration::from_secs(300)); // Every 5 minutes

            loop {
                check_interval.tick().await;

                // Check disk usage
                if let Ok(size_gb) = storage.calculate_database_size().await {
                    let size_gb = size_gb / 1_000_000_000;
                    let available_gb = config.profile.disk_space_gb;

                    let usage_percent = (size_gb * 100) / available_gb;

                    if usage_percent > 90 {
                        warn!("🚨 Disk usage critical: {} / {} GB ({}%)",
                              size_gb, available_gb, usage_percent);
                        warn!("   Consider increasing Q_MAX_STORAGE or freeing disk space");
                    } else if usage_percent > 75 {
                        warn!("⚠️  Disk usage high: {} / {} GB ({}%)",
                              size_gb, available_gb, usage_percent);
                    }
                }

                // Check memory usage
                let mut sys = System::new();
                sys.refresh_memory();
                let memory_used_gb = (sys.used_memory() / 1_000_000_000) as u64;
                let memory_total_gb = (sys.total_memory() / 1_000_000_000) as u64;
                let memory_percent = (memory_used_gb * 100) / memory_total_gb;

                if memory_percent > 80 {
                    warn!("⚠️  High memory usage: {} / {} GB ({}%)",
                          memory_used_gb, memory_total_gb, memory_percent);
                }
            }
        });

        info!("✅ Resource monitoring started (checking every 5 minutes)");
    }
}
```

### Day 10: Integration

Add to main.rs after adaptive config:

```rust
// Start resource monitoring
let resource_monitor = ResourceMonitor::new(adaptive_config.clone(), storage.clone());
resource_monitor.start_monitoring().await;

// Start dynamic pruner (runs with each new block)
let dynamic_pruner = DynamicPruner::new(adaptive_config.clone());
```

---

## Week 3: Enhanced P2P & Network Health

### Day 11-12: Bandwidth-Aware P2P Serving

**File:** `crates/q-network/src/adaptive_p2p.rs`

```rust
pub struct AdaptiveP2PServer {
    config: AdaptiveNodeConfig,
    active_sync_sessions: Arc<AtomicU32>,
}

impl AdaptiveP2PServer {
    pub fn should_serve_block_request(&self, request: &BlockRequest) -> bool {
        // Don't serve if disabled
        if !self.config.serve_historical {
            return false;
        }

        // Check concurrent session limit
        let active = self.active_sync_sessions.load(Ordering::Relaxed);
        if active >= self.config.max_concurrent_syncs {
            debug!("⚠️  Max concurrent syncs reached ({}/{}), rejecting request",
                   active, self.config.max_concurrent_syncs);
            return false;
        }

        true
    }

    pub async fn serve_block_request(&self, request: BlockRequest, storage: &QStorageEngine) {
        if !self.should_serve_block_request(&request) {
            return;
        }

        // Increment active sessions
        self.active_sync_sessions.fetch_add(1, Ordering::Relaxed);

        // Serve blocks
        for height in request.start_height..=request.end_height {
            if let Some(block) = storage.get_qblock_by_height(height).await {
                // Publish block response
                publish_block_response(request.request_id, block).await;
            }
        }

        // Decrement when done
        self.active_sync_sessions.fetch_sub(1, Ordering::Relaxed);
    }
}
```

### Day 13-14: Network Health Dashboard

**File:** `crates/q-network/src/network_health.rs`

```rust
pub struct NetworkHealthDashboard {
    peer_modes: HashMap<PeerId, PruningMode>,
}

impl NetworkHealthDashboard {
    pub fn record_peer_mode(&mut self, peer: PeerId, mode: PruningMode) {
        self.peer_modes.insert(peer, mode);
    }

    pub fn generate_health_report(&self) -> String {
        let total = self.peer_modes.len();
        let minimal = self.peer_modes.values().filter(|m| **m == PruningMode::Minimal).count();
        let balanced = self.peer_modes.values().filter(|m| **m == PruningMode::Balanced).count();
        let full = self.peer_modes.values().filter(|m| **m == PruningMode::FullHistory).count();

        format!(
            "📊 Network Health:\n\
             Total Peers: {}\n\
             🔵 Minimal: {} ({:.1}%)\n\
             🟡 Balanced: {} ({:.1}%)\n\
             🔴 Full History: {} ({:.1}%)\n\
             \n\
             Historical Data Safety: {}",
            total,
            minimal, (minimal as f64 / total as f64) * 100.0,
            balanced, (balanced as f64 / total as f64) * 100.0,
            full, (full as f64 / total as f64) * 100.0,
            if full >= 3 { "✅ Secure" } else { "⚠️  At Risk" }
        )
    }
}
```

---

## Configuration Examples

### Docker Compose

```yaml
version: '3.8'
services:
  q-node-minimal:
    image: quillon/node:v0.3.4
    environment:
      - Q_NODE_TIER=minimal
    volumes:
      - ./data-minimal:/data

  q-node-balanced:
    image: quillon/node:v0.3.4
    environment:
      - Q_MAX_STORAGE=50GB
    volumes:
      - ./data-balanced:/data

  q-node-archive:
    image: quillon/node:v0.3.4
    environment:
      - Q_NODE_TIER=full
      - Q_SERVE_HISTORICAL=true
    volumes:
      - ./data-archive:/data
```

### Systemd Service

```ini
[Service]
Environment="Q_NODE_TIER=balanced"
Environment="Q_SERVE_HISTORICAL=true"
ExecStart=/usr/local/bin/q-api-server --port 8080
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Setup Time | < 2 minutes | Time from download to synced |
| User Confusion | 0 support tickets | No "which binary?" questions |
| Network Health | 5+ full nodes | Via health dashboard |
| Resource Efficiency | 90%+ optimal | Auto-config accuracy |

---

**Timeline:** 3 weeks to production-ready adaptive node system 🚀
