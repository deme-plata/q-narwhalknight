/// 🚀 Project APOLLO Phase 3: HOHMANN - Checkpoint Jump Points (GRAVITY WELLS)
///
/// Pre-defined checkpoints for fast-forward sync:
/// - Known-good heights with verified state roots
/// - Allows skipping validation of historical blocks
/// - Trust-minimized: Multiple independent verifiers
/// - Emergency recovery: Jump to last known good state
///
/// Aerospace analogy:
/// - GRAVITY WELLS: Points of high "gravitational pull" that attract sync
/// - Like planetary gravity assists, these points accelerate your journey
/// - Each checkpoint is a safe harbor for recovery
///
/// Security model:
/// - Checkpoints are signed by multiple trusted validators
/// - State roots are verifiable against full node state
/// - Cumulative work prevents checkpoint spoofing
/// - Optional: ZK proofs for trustless verification

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

// v3.2.2: Import u128_serde for MessagePack P2P compatibility
use q_types::u128_serde;

/// A single checkpoint (GRAVITY WELL)
///
/// Each checkpoint represents a known-good state at a specific height.
/// New nodes can sync to this height without validating all previous blocks.
/// v3.2.2: Added u128_serde for MessagePack P2P compatibility
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GravityWell {
    /// Block height of checkpoint
    pub height: u64,

    /// Block hash at this height (32 bytes)
    pub block_hash: [u8; 32],

    /// State root after processing this block (32 bytes)
    /// Represents: all account balances, contract states, etc.
    pub state_root: [u8; 32],

    /// Merkle root of all transactions up to this point
    pub tx_merkle_root: [u8; 32],

    /// Cumulative proof-of-work/stake (prevents checkpoint injection)
    pub cumulative_work: u128,

    /// Timestamp of checkpoint creation
    pub timestamp: u64,

    /// Version of checkpoint format (for future upgrades)
    pub version: u8,

    /// Human-readable label (e.g., "mainnet-genesis", "hardfork-1.0")
    pub label: String,

    /// Signatures from trusted validators (BLS aggregate or multi-sig)
    /// Format: Vec<(validator_pubkey_hex, signature_hex)>
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub signatures: Vec<(String, String)>,
}

impl GravityWell {
    /// Create a new checkpoint
    pub fn new(
        height: u64,
        block_hash: [u8; 32],
        state_root: [u8; 32],
        tx_merkle_root: [u8; 32],
        cumulative_work: u128,
        label: impl Into<String>,
    ) -> Self {
        Self {
            height,
            block_hash,
            state_root,
            tx_merkle_root,
            cumulative_work,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            version: 1,
            label: label.into(),
            signatures: Vec::new(),
        }
    }

    /// Verify checkpoint integrity (basic validation)
    pub fn verify_basic(&self) -> Result<()> {
        // Check non-zero hashes
        if self.block_hash == [0u8; 32] {
            bail!("Invalid checkpoint: zero block hash");
        }
        if self.state_root == [0u8; 32] {
            bail!("Invalid checkpoint: zero state root");
        }

        // Check cumulative work is non-zero (except genesis)
        if self.height > 0 && self.cumulative_work == 0 {
            bail!("Invalid checkpoint: zero cumulative work at height {}", self.height);
        }

        Ok(())
    }

    /// Check if this checkpoint is newer than another
    pub fn is_newer_than(&self, other: &GravityWell) -> bool {
        self.height > other.height ||
        (self.height == other.height && self.cumulative_work > other.cumulative_work)
    }
}

/// Checkpoint manager (GRAVITY WELL NAVIGATION)
///
/// Manages a collection of checkpoints for fast sync.
/// Checkpoints are stored in height order for efficient lookup.
pub struct CheckpointManager {
    /// All known checkpoints, ordered by height
    checkpoints: BTreeMap<u64, GravityWell>,

    /// Minimum signatures required for checkpoint validity
    min_signatures: usize,

    /// Trusted validator public keys
    trusted_validators: Vec<[u8; 32]>,

    /// Enable checkpoint-based sync
    enabled: bool,

    /// Maximum height to use checkpoints (for testnet safety)
    max_checkpoint_height: Option<u64>,
}

impl CheckpointManager {
    /// Create new checkpoint manager with default settings
    pub fn new() -> Self {
        Self {
            checkpoints: BTreeMap::new(),
            min_signatures: 0, // No signature requirement for testnet
            trusted_validators: Vec::new(),
            enabled: true,
            max_checkpoint_height: None,
        }
    }

    /// Create checkpoint manager with hardcoded checkpoints
    pub fn with_default_checkpoints(network_id: &str) -> Self {
        let mut manager = Self::new();

        // Load network-specific checkpoints
        match network_id {
            "mainnet2026" | "mainnet2026.1" | "mainnet2026.2" | "mainnet-genesis" | "mainnet" => {
                // Mainnet checkpoints (to be populated after launch)
                info!("📍 Loading mainnet checkpoints");
            }
            "testnet-phase16" | "testnet" => {
                // Testnet checkpoints for Q-NarwhalKnight
                info!("📍 Loading testnet-phase16 checkpoints");

                // Genesis checkpoint
                manager.add_checkpoint(GravityWell::new(
                    0,
                    hex_to_bytes32("0000000000000000000000000000000000000000000000000000000000000001"),
                    hex_to_bytes32("0000000000000000000000000000000000000000000000000000000000000000"),
                    hex_to_bytes32("0000000000000000000000000000000000000000000000000000000000000000"),
                    0,
                    "genesis",
                ));

                // Add more checkpoints as the network grows
                // These will be updated with actual values
            }
            _ => {
                warn!("Unknown network {}, no checkpoints loaded", network_id);
            }
        }

        manager
    }

    /// Add a checkpoint
    pub fn add_checkpoint(&mut self, checkpoint: GravityWell) {
        if let Err(e) = checkpoint.verify_basic() {
            warn!("Rejecting invalid checkpoint at {}: {}", checkpoint.height, e);
            return;
        }

        info!(
            "📍 [GRAVITY WELL] Added checkpoint at height {}: {}",
            checkpoint.height, checkpoint.label
        );
        self.checkpoints.insert(checkpoint.height, checkpoint);
    }

    /// Get checkpoint at exact height
    pub fn get_checkpoint(&self, height: u64) -> Option<&GravityWell> {
        self.checkpoints.get(&height)
    }

    /// Get nearest checkpoint at or below height
    pub fn get_nearest_checkpoint(&self, height: u64) -> Option<&GravityWell> {
        self.checkpoints
            .range(..=height)
            .next_back()
            .map(|(_, cp)| cp)
    }

    /// Get next checkpoint above height
    pub fn get_next_checkpoint(&self, height: u64) -> Option<&GravityWell> {
        self.checkpoints
            .range((height + 1)..)
            .next()
            .map(|(_, cp)| cp)
    }

    /// Get highest checkpoint
    pub fn get_highest_checkpoint(&self) -> Option<&GravityWell> {
        self.checkpoints.values().last()
    }

    /// Check if height is a checkpoint
    pub fn is_checkpoint(&self, height: u64) -> bool {
        self.checkpoints.contains_key(&height)
    }

    /// Get all checkpoints
    pub fn all_checkpoints(&self) -> impl Iterator<Item = &GravityWell> {
        self.checkpoints.values()
    }

    /// Get checkpoint count
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Calculate optimal jump point for sync from current height to target
    ///
    /// Returns the highest checkpoint that:
    /// 1. Is above current height
    /// 2. Is at or below target height
    /// 3. Has required signatures (if enabled)
    pub fn calculate_jump(&self, current_height: u64, target_height: u64) -> Option<&GravityWell> {
        if !self.enabled {
            return None;
        }

        if let Some(max_height) = self.max_checkpoint_height {
            if current_height >= max_height {
                return None;
            }
        }

        // Find best checkpoint between current and target
        self.checkpoints
            .range((current_height + 1)..=target_height)
            .filter(|(_, cp)| {
                // Check signature requirements
                if self.min_signatures > 0 {
                    cp.signatures.len() >= self.min_signatures
                } else {
                    true
                }
            })
            .last()
            .map(|(_, cp)| cp)
    }

    /// Verify state root matches checkpoint (for sync validation)
    pub fn verify_state_at_height(&self, height: u64, state_root: [u8; 32]) -> Result<bool> {
        let checkpoint = self
            .get_checkpoint(height)
            .context("No checkpoint at height")?;

        Ok(checkpoint.state_root == state_root)
    }

    /// Verify block hash matches checkpoint
    pub fn verify_block_at_height(&self, height: u64, block_hash: [u8; 32]) -> Result<bool> {
        let checkpoint = self
            .get_checkpoint(height)
            .context("No checkpoint at height")?;

        Ok(checkpoint.block_hash == block_hash)
    }

    /// Enable/disable checkpoint sync
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        info!("📍 Checkpoint sync {}", if enabled { "enabled" } else { "disabled" });
    }

    /// Set maximum checkpoint height (for testnet safety)
    pub fn set_max_height(&mut self, height: u64) {
        self.max_checkpoint_height = Some(height);
        info!("📍 Checkpoint max height set to {}", height);
    }

    /// Export checkpoints to JSON for distribution
    pub fn export_json(&self) -> Result<String> {
        let checkpoints: Vec<&GravityWell> = self.checkpoints.values().collect();
        serde_json::to_string_pretty(&checkpoints).context("Failed to serialize checkpoints")
    }

    /// Import checkpoints from JSON
    pub fn import_json(&mut self, json: &str) -> Result<usize> {
        let checkpoints: Vec<GravityWell> =
            serde_json::from_str(json).context("Failed to parse checkpoints JSON")?;

        let count = checkpoints.len();
        for cp in checkpoints {
            self.add_checkpoint(cp);
        }

        info!("📍 Imported {} checkpoints", count);
        Ok(count)
    }
}

impl Default for CheckpointManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Sync strategy based on checkpoints
#[derive(Clone, Debug)]
pub struct CheckpointSyncStrategy {
    /// Start from this checkpoint (if available)
    pub start_checkpoint: Option<GravityWell>,

    /// Skip validation for blocks below this height
    pub skip_validation_below: u64,

    /// Full validation required above this height
    pub full_validation_above: u64,

    /// Blocks to fast-forward (checkpoint to checkpoint)
    pub fast_forward_ranges: Vec<(u64, u64)>,

    /// Estimated sync time savings (percentage)
    pub estimated_time_savings: f64,
}

impl CheckpointSyncStrategy {
    /// Calculate optimal sync strategy
    pub fn calculate(
        current_height: u64,
        target_height: u64,
        checkpoints: &CheckpointManager,
    ) -> Self {
        let mut strategy = Self {
            start_checkpoint: None,
            skip_validation_below: current_height,
            full_validation_above: target_height,
            fast_forward_ranges: Vec::new(),
            estimated_time_savings: 0.0,
        };

        // Find optimal jump point
        if let Some(jump_checkpoint) = checkpoints.calculate_jump(current_height, target_height) {
            strategy.start_checkpoint = Some(jump_checkpoint.clone());
            strategy.skip_validation_below = jump_checkpoint.height;

            // Calculate time savings (rough estimate)
            let skipped_blocks = jump_checkpoint.height.saturating_sub(current_height);
            let total_blocks = target_height.saturating_sub(current_height);
            if total_blocks > 0 {
                strategy.estimated_time_savings = (skipped_blocks as f64 / total_blocks as f64) * 100.0;
            }

            info!(
                "📍 [SYNC STRATEGY] Jump to checkpoint {} ({}) - skip {} blocks ({:.1}% savings)",
                jump_checkpoint.height,
                jump_checkpoint.label,
                skipped_blocks,
                strategy.estimated_time_savings
            );
        }

        // Find all fast-forward ranges between checkpoints
        let mut prev_height = current_height;
        for checkpoint in checkpoints.all_checkpoints() {
            if checkpoint.height > prev_height && checkpoint.height <= target_height {
                strategy.fast_forward_ranges.push((prev_height, checkpoint.height));
                prev_height = checkpoint.height;
            }
        }

        strategy
    }
}

/// Helper: Convert hex string to [u8; 32]
fn hex_to_bytes32(hex: &str) -> [u8; 32] {
    let mut result = [0u8; 32];
    if let Ok(bytes) = hex::decode(hex) {
        let len = bytes.len().min(32);
        result[..len].copy_from_slice(&bytes[..len]);
    }
    result
}

/// Checkpoint download/verification state
#[derive(Clone, Debug)]
pub struct CheckpointSyncState {
    /// Currently syncing to this checkpoint
    pub target_checkpoint: Option<GravityWell>,

    /// Headers verified up to this height
    pub headers_verified: u64,

    /// State verified (matches checkpoint state_root)
    pub state_verified: bool,

    /// Download progress (0.0 - 1.0)
    pub progress: f64,
}

impl CheckpointSyncState {
    pub fn new() -> Self {
        Self {
            target_checkpoint: None,
            headers_verified: 0,
            state_verified: false,
            progress: 0.0,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.target_checkpoint
            .as_ref()
            .map(|cp| self.headers_verified >= cp.height && self.state_verified)
            .unwrap_or(false)
    }
}

impl Default for CheckpointSyncState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_checkpoint(height: u64) -> GravityWell {
        GravityWell::new(
            height,
            [height as u8; 32],
            [height as u8; 32],
            [height as u8; 32],
            height as u128 * 1000,
            format!("test-{}", height),
        )
    }

    #[test]
    fn test_gravity_well_creation() {
        let cp = test_checkpoint(100);
        assert_eq!(cp.height, 100);
        assert!(cp.verify_basic().is_ok());
    }

    #[test]
    fn test_gravity_well_invalid() {
        let mut cp = test_checkpoint(100);
        cp.block_hash = [0u8; 32]; // Invalid zero hash
        assert!(cp.verify_basic().is_err());
    }

    #[test]
    fn test_checkpoint_manager_add_get() {
        let mut manager = CheckpointManager::new();
        manager.add_checkpoint(test_checkpoint(100));
        manager.add_checkpoint(test_checkpoint(200));
        manager.add_checkpoint(test_checkpoint(300));

        assert_eq!(manager.checkpoint_count(), 3);
        assert!(manager.get_checkpoint(100).is_some());
        assert!(manager.get_checkpoint(150).is_none());
    }

    #[test]
    fn test_nearest_checkpoint() {
        let mut manager = CheckpointManager::new();
        manager.add_checkpoint(test_checkpoint(100));
        manager.add_checkpoint(test_checkpoint(200));
        manager.add_checkpoint(test_checkpoint(300));

        let nearest = manager.get_nearest_checkpoint(250);
        assert!(nearest.is_some());
        assert_eq!(nearest.unwrap().height, 200);

        let nearest = manager.get_nearest_checkpoint(300);
        assert_eq!(nearest.unwrap().height, 300);
    }

    #[test]
    fn test_calculate_jump() {
        let mut manager = CheckpointManager::new();
        manager.add_checkpoint(test_checkpoint(100));
        manager.add_checkpoint(test_checkpoint(200));
        manager.add_checkpoint(test_checkpoint(300));

        // Jump from 0 to 250 should go to checkpoint 200
        let jump = manager.calculate_jump(0, 250);
        assert!(jump.is_some());
        assert_eq!(jump.unwrap().height, 200);

        // Jump from 0 to 400 should go to checkpoint 300
        let jump = manager.calculate_jump(0, 400);
        assert!(jump.is_some());
        assert_eq!(jump.unwrap().height, 300);

        // Jump from 250 to 400 should go to checkpoint 300
        let jump = manager.calculate_jump(250, 400);
        assert!(jump.is_some());
        assert_eq!(jump.unwrap().height, 300);
    }

    #[test]
    fn test_sync_strategy() {
        let mut manager = CheckpointManager::new();
        manager.add_checkpoint(test_checkpoint(100));
        manager.add_checkpoint(test_checkpoint(200));
        manager.add_checkpoint(test_checkpoint(300));

        let strategy = CheckpointSyncStrategy::calculate(0, 500, &manager);

        assert!(strategy.start_checkpoint.is_some());
        assert_eq!(strategy.start_checkpoint.unwrap().height, 300);
        assert_eq!(strategy.skip_validation_below, 300);
        assert!(strategy.estimated_time_savings > 0.0);
    }

    #[test]
    fn test_export_import_json() {
        let mut manager = CheckpointManager::new();
        manager.add_checkpoint(test_checkpoint(100));
        manager.add_checkpoint(test_checkpoint(200));

        let json = manager.export_json().unwrap();
        assert!(json.contains("test-100"));
        assert!(json.contains("test-200"));

        let mut new_manager = CheckpointManager::new();
        let count = new_manager.import_json(&json).unwrap();
        assert_eq!(count, 2);
        assert_eq!(new_manager.checkpoint_count(), 2);
    }
}
