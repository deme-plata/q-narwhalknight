use crate::resource_profile::{SystemResourceProfile, DeviceClass};
use serde::{Deserialize, Serialize};
use std::env;
use tracing::{info, warn};

/// Pruning mode determines how many historical blocks to retain
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PruningMode {
    /// Keep only last 1,000 blocks (~5 GB) - mobile/low-resource
    Minimal,
    /// Keep last 10,000 blocks (~50 GB) - standard desktop
    Balanced,
    /// Keep all blocks (500+ GB) - servers/enthusiasts
    FullHistory,
    /// User-specified custom retention depth
    Custom(u64),
}

impl PruningMode {
    /// Get retention depth in blocks
    pub fn retention_depth(&self) -> u64 {
        match self {
            PruningMode::Minimal => 1_000,
            PruningMode::Balanced => 10_000,
            PruningMode::FullHistory => u64::MAX, // Keep everything
            PruningMode::Custom(depth) => *depth,
        }
    }

    /// Get human-readable description
    pub fn description(&self) -> String {
        match self {
            PruningMode::Minimal => "Minimal (1,000 blocks, ~5 GB)".to_string(),
            PruningMode::Balanced => "Balanced (10,000 blocks, ~50 GB)".to_string(),
            PruningMode::FullHistory => "Full History (all blocks, 500+ GB)".to_string(),
            PruningMode::Custom(depth) => format!("Custom ({} blocks)", depth),
        }
    }

    /// Estimate storage requirements in GB
    pub fn estimated_storage_gb(&self) -> u64 {
        const AVG_BLOCK_SIZE_BYTES: u64 = 36_400; // 36.4 KB per block
        match self {
            PruningMode::Minimal => 5,
            PruningMode::Balanced => 50,
            PruningMode::FullHistory => 500,
            PruningMode::Custom(depth) => {
                (depth * AVG_BLOCK_SIZE_BYTES) / 1_000_000_000
            }
        }
    }
}

/// Adaptive node configuration that auto-detects and adjusts based on resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveNodeConfig {
    /// Detected system resource profile
    pub profile: SystemResourceProfile,
    /// Selected pruning mode
    pub pruning_mode: PruningMode,
    /// User-specified max storage in GB (overrides auto-detection)
    pub max_storage_gb: Option<u64>,
    /// Whether to serve historical data to peers
    pub serve_historical: bool,
    /// Max concurrent P2P sync connections
    pub max_concurrent_syncs: u32,
    /// Whether this config was auto-detected or user-specified
    pub is_auto_detected: bool,
}

impl AdaptiveNodeConfig {
    /// Auto-detect optimal configuration based on system resources
    pub fn auto_detect() -> anyhow::Result<Self> {
        info!("🚀 Q-NarwhalKnight: Auto-detecting optimal node configuration...");

        let profile = SystemResourceProfile::auto_detect()?;

        if !profile.meets_minimum_requirements() {
            anyhow::bail!("❌ System does not meet minimum requirements for running a node");
        }

        // Determine pruning mode based on available disk space
        let pruning_mode = Self::select_pruning_mode(&profile);

        // Calculate max concurrent syncs based on device class
        let max_concurrent_syncs = profile.device_class.max_concurrent_syncs();

        // Full history nodes automatically serve historical data
        let serve_historical = matches!(pruning_mode, PruningMode::FullHistory)
            || env::var("Q_SERVE_HISTORICAL").unwrap_or_default() == "true";

        info!("✅ Configuration Selected:");
        info!("   🔧 Pruning Mode: {}", pruning_mode.description());
        info!("   📦 Estimated Storage: {} GB", pruning_mode.estimated_storage_gb());
        info!("   🌐 Serve Historical: {}", if serve_historical { "Yes" } else { "No" });
        info!("   🔗 Max Concurrent Syncs: {}", max_concurrent_syncs);

        Ok(Self {
            profile,
            pruning_mode,
            max_storage_gb: None,
            serve_historical,
            max_concurrent_syncs,
            is_auto_detected: true,
        })
    }

    /// Create configuration with user-specified storage limit
    pub fn with_storage_limit(storage_limit_gb: u64) -> anyhow::Result<Self> {
        info!("👤 User-specified storage limit: {} GB", storage_limit_gb);

        let profile = SystemResourceProfile::auto_detect()?;

        if !profile.meets_minimum_requirements() {
            anyhow::bail!("❌ System does not meet minimum requirements for running a node");
        }

        // Select pruning mode based on user's storage limit
        let pruning_mode = if storage_limit_gb < 10 {
            PruningMode::Minimal
        } else if storage_limit_gb < 100 {
            PruningMode::Balanced
        } else {
            PruningMode::FullHistory
        };

        let max_concurrent_syncs = profile.device_class.max_concurrent_syncs();
        let serve_historical = matches!(pruning_mode, PruningMode::FullHistory);

        info!("✅ Configuration Selected:");
        info!("   🔧 Pruning Mode: {}", pruning_mode.description());
        info!("   📦 Max Storage: {} GB", storage_limit_gb);
        info!("   🌐 Serve Historical: {}", if serve_historical { "Yes" } else { "No" });

        Ok(Self {
            profile,
            pruning_mode,
            max_storage_gb: Some(storage_limit_gb),
            serve_historical,
            max_concurrent_syncs,
            is_auto_detected: false,
        })
    }

    /// Create configuration with explicit pruning mode
    pub fn with_pruning_mode(mode: PruningMode) -> anyhow::Result<Self> {
        info!("👤 User-specified pruning mode: {}", mode.description());

        let profile = SystemResourceProfile::auto_detect()?;

        if !profile.meets_minimum_requirements() {
            anyhow::bail!("❌ System does not meet minimum requirements for running a node");
        }

        let max_concurrent_syncs = profile.device_class.max_concurrent_syncs();
        let serve_historical = matches!(mode, PruningMode::FullHistory);

        Ok(Self {
            profile,
            pruning_mode: mode,
            max_storage_gb: None,
            serve_historical,
            max_concurrent_syncs,
            is_auto_detected: false,
        })
    }

    /// Load configuration from environment variables
    pub fn from_env() -> anyhow::Result<Self> {
        // Check for Q_MAX_STORAGE environment variable
        if let Ok(storage_str) = env::var("Q_MAX_STORAGE") {
            let storage_gb = Self::parse_storage_string(&storage_str)?;
            return Self::with_storage_limit(storage_gb);
        }

        // Check for Q_PRUNING_MODE environment variable
        if let Ok(mode_str) = env::var("Q_PRUNING_MODE") {
            let mode = match mode_str.to_lowercase().as_str() {
                "minimal" => PruningMode::Minimal,
                "balanced" => PruningMode::Balanced,
                "full" | "fullhistory" | "full_history" => PruningMode::FullHistory,
                _ => {
                    warn!("⚠️  Unknown pruning mode '{}', using auto-detection", mode_str);
                    return Self::auto_detect();
                }
            };
            return Self::with_pruning_mode(mode);
        }

        // Check for Q_BLOCK_RETENTION (custom depth)
        if let Ok(depth_str) = env::var("Q_BLOCK_RETENTION") {
            if let Ok(depth) = depth_str.parse::<u64>() {
                return Self::with_pruning_mode(PruningMode::Custom(depth));
            }
        }

        // Default to auto-detection
        Self::auto_detect()
    }

    /// Select pruning mode based on system resources
    fn select_pruning_mode(profile: &SystemResourceProfile) -> PruningMode {
        let available_gb = profile.available_disk_gb;
        let recommended_percentage = profile.device_class.recommended_storage_percentage();
        let usable_gb = (available_gb * recommended_percentage as u64) / 100;

        if usable_gb < 10 {
            warn!("⚠️  Low disk space ({} GB usable)", usable_gb);
            info!("   Using Minimal mode: keeping last 1,000 blocks");
            PruningMode::Minimal
        } else if usable_gb < 100 {
            info!("📊 Detected {} GB usable disk space", usable_gb);
            info!("   Using Balanced mode: keeping last 10,000 blocks");
            PruningMode::Balanced
        } else {
            info!("🗄️  Detected {} GB usable disk space", usable_gb);
            info!("   Using Full History mode: keeping all blocks");
            info!("   🎖️  You'll automatically serve historical data to the network!");
            PruningMode::FullHistory
        }
    }

    /// Parse storage string (e.g., "50GB", "500MB", "1TB")
    fn parse_storage_string(s: &str) -> anyhow::Result<u64> {
        let s = s.trim().to_uppercase();

        if let Some(gb_str) = s.strip_suffix("GB") {
            return Ok(gb_str.trim().parse()?);
        }
        if let Some(mb_str) = s.strip_suffix("MB") {
            let mb: u64 = mb_str.trim().parse()?;
            return Ok(mb / 1000); // Convert MB to GB
        }
        if let Some(tb_str) = s.strip_suffix("TB") {
            let tb: u64 = tb_str.trim().parse()?;
            return Ok(tb * 1000); // Convert TB to GB
        }

        // Try parsing as plain number (assume GB)
        Ok(s.parse()?)
    }

    /// Check if node should prune at current height
    pub fn should_prune(&self, current_height: u64) -> Option<u64> {
        let retention_depth = self.pruning_mode.retention_depth();

        if retention_depth == u64::MAX {
            // Full history mode - never prune
            return None;
        }

        if current_height > retention_depth {
            Some(current_height - retention_depth)
        } else {
            None
        }
    }

    /// Get display summary for startup logs
    pub fn startup_summary(&self) -> Vec<String> {
        let mut lines = vec![
            "🚀 Q-NarwhalKnight Node Configuration".to_string(),
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".to_string(),
        ];

        lines.push(format!("📊 Device: {}", self.profile.device_class.description()));
        lines.push(format!("💾 Disk: {} GB available", self.profile.available_disk_gb));
        lines.push(format!("🧠 RAM: {} GB available", self.profile.available_ram_gb));
        lines.push(format!("⚙️  CPU: {} cores", self.profile.cpu_cores));
        lines.push("".to_string());
        lines.push(format!("🔧 Pruning: {}", self.pruning_mode.description()));
        lines.push(format!("📦 Storage: ~{} GB", self.pruning_mode.estimated_storage_gb()));
        lines.push(format!("🌐 Serving: {}", if self.serve_historical { "Historical data" } else { "Recent blocks only" }));
        lines.push(format!("🔗 Max Syncs: {}", self.max_concurrent_syncs));

        if self.is_auto_detected {
            lines.push("".to_string());
            lines.push("ℹ️  Configuration auto-detected. Override with:".to_string());
            lines.push("   Q_MAX_STORAGE=50GB".to_string());
            lines.push("   Q_PRUNING_MODE=balanced".to_string());
        }

        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_storage_string() {
        assert_eq!(AdaptiveNodeConfig::parse_storage_string("50GB").unwrap(), 50);
        assert_eq!(AdaptiveNodeConfig::parse_storage_string("500MB").unwrap(), 0);
        assert_eq!(AdaptiveNodeConfig::parse_storage_string("1TB").unwrap(), 1000);
        assert_eq!(AdaptiveNodeConfig::parse_storage_string("100").unwrap(), 100);
    }

    #[test]
    fn test_pruning_mode_retention() {
        assert_eq!(PruningMode::Minimal.retention_depth(), 1_000);
        assert_eq!(PruningMode::Balanced.retention_depth(), 10_000);
        assert_eq!(PruningMode::FullHistory.retention_depth(), u64::MAX);
        assert_eq!(PruningMode::Custom(5_000).retention_depth(), 5_000);
    }

    #[test]
    fn test_should_prune() {
        let config = AdaptiveNodeConfig::with_pruning_mode(PruningMode::Minimal).unwrap();

        assert_eq!(config.should_prune(500), None); // Not enough blocks yet
        assert_eq!(config.should_prune(1500), Some(500)); // Should prune blocks 0-500
        assert_eq!(config.should_prune(5000), Some(4000)); // Should prune blocks 0-4000
    }
}
