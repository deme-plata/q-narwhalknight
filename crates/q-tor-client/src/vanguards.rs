/// Vanguards Implementation for Q-NarwhalKnight
///
/// Vanguards protect against guard discovery attacks by adding additional
/// layers of pinned middle relays. This implements "Vanguards-lite" as
/// specified in Tor Proposal 292 (full vanguards) simplified for our use case.
///
/// # Attack Model
/// Without vanguards, an adversary controlling some fraction of the network
/// can eventually discover a client's guard node through repeated circuit
/// building. Once the guard is known, the adversary can:
/// - Perform traffic confirmation attacks
/// - Narrow down client location
/// - Correlate multiple circuits to the same client
///
/// # Protection Mechanism
/// Vanguards add pinned "second layer" and "third layer" guards that rotate
/// slowly, making guard discovery attacks take months/years instead of hours.

use anyhow::{anyhow, Result};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Vanguard layer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VanguardLayer {
    /// Layer 2 (L2) - First vanguard after guard
    /// Rotates every 1-12 days (randomized per relay)
    Layer2,
    /// Layer 3 (L3) - Second vanguard
    /// Rotates every 18-48 hours (randomized per relay)
    Layer3,
}

impl VanguardLayer {
    /// Get the minimum rotation period for this layer
    pub fn min_rotation(&self) -> Duration {
        match self {
            VanguardLayer::Layer2 => Duration::from_secs(24 * 60 * 60), // 1 day
            VanguardLayer::Layer3 => Duration::from_secs(18 * 60 * 60), // 18 hours
        }
    }

    /// Get the maximum rotation period for this layer
    pub fn max_rotation(&self) -> Duration {
        match self {
            VanguardLayer::Layer2 => Duration::from_secs(12 * 24 * 60 * 60), // 12 days
            VanguardLayer::Layer3 => Duration::from_secs(48 * 60 * 60),      // 48 hours
        }
    }

    /// Generate a random rotation period within bounds
    pub fn random_rotation(&self) -> Duration {
        let mut rng = rand::rng();
        let min_secs = self.min_rotation().as_secs();
        let max_secs = self.max_rotation().as_secs();
        Duration::from_secs(rng.random_range(min_secs..=max_secs))
    }

    pub fn name(&self) -> &'static str {
        match self {
            VanguardLayer::Layer2 => "L2",
            VanguardLayer::Layer3 => "L3",
        }
    }
}

/// Helper function to provide default Instant (now) for serde deserialization
fn instant_now() -> Instant {
    Instant::now()
}

/// A pinned vanguard relay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VanguardRelay {
    /// Relay fingerprint (40 hex chars)
    pub fingerprint: String,
    /// Relay nickname
    pub nickname: String,
    /// IP address (for display/debugging only)
    pub address: String,
    /// Layer this relay is assigned to
    pub layer: VanguardLayer,
    /// When this relay was selected (not serialized - Instant doesn't impl serde)
    #[serde(skip, default = "instant_now")]
    pub selected_at: Instant,
    /// When this relay should be rotated out (not serialized - Instant doesn't impl serde)
    #[serde(skip, default = "instant_now")]
    pub rotate_at: Instant,
    /// Number of circuits using this relay
    pub circuit_count: u64,
    /// Whether this relay is currently healthy
    pub is_healthy: bool,
}

impl VanguardRelay {
    /// Create a new vanguard relay with randomized rotation time
    pub fn new(fingerprint: String, nickname: String, address: String, layer: VanguardLayer) -> Self {
        let now = Instant::now();
        let rotation_period = layer.random_rotation();

        Self {
            fingerprint,
            nickname,
            address,
            layer,
            selected_at: now,
            rotate_at: now + rotation_period,
            circuit_count: 0,
            is_healthy: true,
        }
    }

    /// Check if this relay should be rotated
    pub fn should_rotate(&self) -> bool {
        Instant::now() >= self.rotate_at
    }

    /// Get time until rotation
    pub fn time_until_rotation(&self) -> Duration {
        let now = Instant::now();
        if now >= self.rotate_at {
            Duration::ZERO
        } else {
            self.rotate_at - now
        }
    }

    /// Get age of this vanguard assignment
    pub fn age(&self) -> Duration {
        self.selected_at.elapsed()
    }

    /// Mark relay as unhealthy (triggers faster rotation)
    pub fn mark_unhealthy(&mut self) {
        self.is_healthy = false;
        // Unhealthy relays rotate within 1 hour
        self.rotate_at = Instant::now() + Duration::from_secs(3600);
    }

    /// Increment circuit count
    pub fn increment_circuits(&mut self) {
        self.circuit_count += 1;
    }
}

/// Configuration for vanguards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VanguardsConfig {
    /// Enable vanguards protection
    pub enabled: bool,
    /// Number of L2 vanguards to maintain
    pub l2_count: usize,
    /// Number of L3 vanguards to maintain
    pub l3_count: usize,
    /// Minimum relay bandwidth (bytes/s) to be considered
    pub min_bandwidth: u64,
    /// Whether to exclude relays from same /16 as guard
    pub exclude_guard_family: bool,
    /// Path to persist vanguard state
    pub state_path: Option<String>,
    /// Enable paranoid mode (more vanguards, faster rotation)
    pub paranoid_mode: bool,
}

impl Default for VanguardsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            l2_count: 4,      // 4 L2 vanguards
            l3_count: 8,      // 8 L3 vanguards
            min_bandwidth: 1_000_000, // 1 MB/s minimum
            exclude_guard_family: true,
            state_path: Some("/var/lib/qnk/tor/vanguards.json".to_string()),
            paranoid_mode: false,
        }
    }
}

impl VanguardsConfig {
    /// Create high-security configuration
    pub fn high_security() -> Self {
        Self {
            enabled: true,
            l2_count: 6,
            l3_count: 12,
            min_bandwidth: 5_000_000, // 5 MB/s
            exclude_guard_family: true,
            state_path: Some("/var/lib/qnk/tor/vanguards.json".to_string()),
            paranoid_mode: true,
        }
    }

    /// Create paranoid configuration (maximum protection)
    pub fn paranoid() -> Self {
        Self {
            enabled: true,
            l2_count: 8,
            l3_count: 16,
            min_bandwidth: 10_000_000, // 10 MB/s
            exclude_guard_family: true,
            state_path: Some("/var/lib/qnk/tor/vanguards.json".to_string()),
            paranoid_mode: true,
        }
    }
}

/// Relay information from consensus
#[derive(Debug, Clone)]
pub struct RelayInfo {
    pub fingerprint: String,
    pub nickname: String,
    pub address: String,
    pub bandwidth: u64,
    pub flags: HashSet<String>,
    pub family: Vec<String>,
}

impl RelayInfo {
    /// Check if relay is suitable for vanguard duty
    pub fn is_suitable_vanguard(&self, min_bandwidth: u64) -> bool {
        // Must have Stable and Fast flags
        let has_stable = self.flags.contains("Stable");
        let has_fast = self.flags.contains("Fast");
        let has_valid = self.flags.contains("Valid");
        let has_running = self.flags.contains("Running");
        let sufficient_bandwidth = self.bandwidth >= min_bandwidth;

        // Must not be an exit (to avoid exit-position attacks)
        let not_exit = !self.flags.contains("Exit");

        has_stable && has_fast && has_valid && has_running && sufficient_bandwidth && not_exit
    }

    /// Get /16 network prefix for family exclusion
    pub fn network_prefix(&self) -> String {
        // Extract first two octets of IP
        let parts: Vec<&str> = self.address.split('.').collect();
        if parts.len() >= 2 {
            format!("{}.{}", parts[0], parts[1])
        } else {
            self.address.clone()
        }
    }
}

/// Vanguards manager
pub struct VanguardsManager {
    /// Configuration
    config: VanguardsConfig,
    /// Current L2 vanguards
    l2_vanguards: RwLock<Vec<VanguardRelay>>,
    /// Current L3 vanguards
    l3_vanguards: RwLock<Vec<VanguardRelay>>,
    /// Guard node fingerprints (to exclude their families)
    guard_fingerprints: RwLock<HashSet<String>>,
    /// Guard network prefixes (to exclude same /16)
    guard_prefixes: RwLock<HashSet<String>>,
    /// Relay consensus cache
    relay_cache: RwLock<Vec<RelayInfo>>,
    /// Last consensus update
    last_consensus_update: RwLock<Option<Instant>>,
    /// Statistics
    stats: RwLock<VanguardsStats>,
}

/// Statistics for vanguards operation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VanguardsStats {
    pub l2_rotations: u64,
    pub l3_rotations: u64,
    pub unhealthy_relays_replaced: u64,
    pub circuits_built: u64,
    pub guard_discovery_attempts_blocked: u64,
    /// Last rotation time (not serialized - Instant doesn't impl serde)
    #[serde(skip)]
    pub last_rotation: Option<Instant>,
}

impl VanguardsManager {
    /// Create a new vanguards manager
    pub fn new(config: VanguardsConfig) -> Self {
        info!(
            "🛡️ Initializing Vanguards manager: L2={}, L3={}, paranoid={}",
            config.l2_count, config.l3_count, config.paranoid_mode
        );

        Self {
            config,
            l2_vanguards: RwLock::new(Vec::new()),
            l3_vanguards: RwLock::new(Vec::new()),
            guard_fingerprints: RwLock::new(HashSet::new()),
            guard_prefixes: RwLock::new(HashSet::new()),
            relay_cache: RwLock::new(Vec::new()),
            last_consensus_update: RwLock::new(None),
            stats: RwLock::new(VanguardsStats::default()),
        }
    }

    /// Initialize vanguards from relay consensus
    pub async fn initialize(&self, relays: Vec<RelayInfo>) -> Result<()> {
        info!("🛡️ Initializing vanguards from {} relays", relays.len());

        // Update relay cache
        {
            let mut cache = self.relay_cache.write().await;
            *cache = relays;
        }

        // Select initial L2 vanguards
        self.fill_vanguards(VanguardLayer::Layer2).await?;

        // Select initial L3 vanguards
        self.fill_vanguards(VanguardLayer::Layer3).await?;

        // Update timestamp
        {
            let mut last_update = self.last_consensus_update.write().await;
            *last_update = Some(Instant::now());
        }

        let l2_count = self.l2_vanguards.read().await.len();
        let l3_count = self.l3_vanguards.read().await.len();

        info!(
            "✅ Vanguards initialized: {} L2, {} L3",
            l2_count, l3_count
        );

        Ok(())
    }

    /// Register guard nodes (to exclude their families)
    pub async fn register_guards(&self, guards: Vec<RelayInfo>) {
        let mut fingerprints = self.guard_fingerprints.write().await;
        let mut prefixes = self.guard_prefixes.write().await;

        for guard in guards {
            fingerprints.insert(guard.fingerprint.clone());
            prefixes.insert(guard.network_prefix());

            // Also exclude family members
            for family_member in &guard.family {
                fingerprints.insert(family_member.clone());
            }
        }

        info!(
            "🛡️ Registered {} guard fingerprints, {} network prefixes",
            fingerprints.len(),
            prefixes.len()
        );
    }

    /// Fill vanguards for a specific layer up to configured count
    async fn fill_vanguards(&self, layer: VanguardLayer) -> Result<()> {
        let target_count = match layer {
            VanguardLayer::Layer2 => self.config.l2_count,
            VanguardLayer::Layer3 => self.config.l3_count,
        };

        let current_count = match layer {
            VanguardLayer::Layer2 => self.l2_vanguards.read().await.len(),
            VanguardLayer::Layer3 => self.l3_vanguards.read().await.len(),
        };

        if current_count >= target_count {
            return Ok(());
        }

        let needed = target_count - current_count;
        debug!("🛡️ Need {} more {} vanguards", needed, layer.name());

        let candidates = self.get_vanguard_candidates(layer).await?;

        if candidates.is_empty() {
            warn!("⚠️ No suitable candidates for {} vanguards", layer.name());
            return Ok(());
        }

        // Randomly select from candidates using Fisher-Yates sampling
        let mut rng = rand::rng();
        let take_count = needed.min(candidates.len());
        let mut indices: Vec<usize> = (0..candidates.len()).collect();

        // Fisher-Yates partial shuffle for random selection
        for i in 0..take_count {
            let j = rng.random_range(i..candidates.len());
            indices.swap(i, j);
        }

        let selected: Vec<_> = indices[..take_count]
            .iter()
            .map(|&i| candidates[i].clone())
            .collect();

        // Add selected relays as vanguards
        let mut vanguards = match layer {
            VanguardLayer::Layer2 => self.l2_vanguards.write().await,
            VanguardLayer::Layer3 => self.l3_vanguards.write().await,
        };

        for relay in selected {
            let vanguard = VanguardRelay::new(
                relay.fingerprint,
                relay.nickname,
                relay.address,
                layer,
            );
            info!(
                "🛡️ Added {} vanguard: {} ({})",
                layer.name(),
                vanguard.nickname,
                &vanguard.fingerprint[..8]
            );
            vanguards.push(vanguard);
        }

        Ok(())
    }

    /// Get candidate relays for vanguard selection
    async fn get_vanguard_candidates(&self, layer: VanguardLayer) -> Result<Vec<RelayInfo>> {
        let relays = self.relay_cache.read().await;
        let guard_fingerprints = self.guard_fingerprints.read().await;
        let guard_prefixes = self.guard_prefixes.read().await;

        // Get current vanguard fingerprints to exclude
        let current_l2: HashSet<_> = self
            .l2_vanguards
            .read()
            .await
            .iter()
            .map(|v| v.fingerprint.clone())
            .collect();
        let current_l3: HashSet<_> = self
            .l3_vanguards
            .read()
            .await
            .iter()
            .map(|v| v.fingerprint.clone())
            .collect();

        let candidates: Vec<_> = relays
            .iter()
            .filter(|r| {
                // Must be suitable
                if !r.is_suitable_vanguard(self.config.min_bandwidth) {
                    return false;
                }

                // Must not be a guard
                if guard_fingerprints.contains(&r.fingerprint) {
                    return false;
                }

                // Must not be in same /16 as guard (if configured)
                if self.config.exclude_guard_family && guard_prefixes.contains(&r.network_prefix()) {
                    return false;
                }

                // Must not already be a vanguard at either layer
                if current_l2.contains(&r.fingerprint) || current_l3.contains(&r.fingerprint) {
                    return false;
                }

                // L2 candidates should not be L3 and vice versa
                match layer {
                    VanguardLayer::Layer2 => !current_l3.contains(&r.fingerprint),
                    VanguardLayer::Layer3 => !current_l2.contains(&r.fingerprint),
                }
            })
            .cloned()
            .collect();

        Ok(candidates)
    }

    /// Get a vanguard relay for circuit building
    pub async fn get_vanguard(&self, layer: VanguardLayer) -> Option<VanguardRelay> {
        let vanguards = match layer {
            VanguardLayer::Layer2 => self.l2_vanguards.read().await,
            VanguardLayer::Layer3 => self.l3_vanguards.read().await,
        };

        // Select a random healthy vanguard
        let healthy: Vec<_> = vanguards.iter().filter(|v| v.is_healthy).collect();

        if healthy.is_empty() {
            warn!("⚠️ No healthy {} vanguards available", layer.name());
            return None;
        }

        // Randomly select one healthy vanguard
        if healthy.is_empty() {
            return None;
        }
        let mut rng = rand::rng();
        let idx = rng.random_range(0..healthy.len());
        Some(healthy[idx].clone())
    }

    /// Get vanguard path for circuit building (returns L2, L3 fingerprints)
    pub async fn get_vanguard_path(&self) -> Option<(String, String)> {
        let l2 = self.get_vanguard(VanguardLayer::Layer2).await?;
        let l3 = self.get_vanguard(VanguardLayer::Layer3).await?;

        // Increment circuit counts
        {
            let mut l2_vanguards = self.l2_vanguards.write().await;
            if let Some(v) = l2_vanguards.iter_mut().find(|v| v.fingerprint == l2.fingerprint) {
                v.increment_circuits();
            }
        }
        {
            let mut l3_vanguards = self.l3_vanguards.write().await;
            if let Some(v) = l3_vanguards.iter_mut().find(|v| v.fingerprint == l3.fingerprint) {
                v.increment_circuits();
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.circuits_built += 1;
        }

        Some((l2.fingerprint, l3.fingerprint))
    }

    /// Perform maintenance (rotation, health checks)
    pub async fn maintain(&self) -> Result<MaintenanceReport> {
        let mut report = MaintenanceReport::default();

        // Rotate expired L2 vanguards
        report.l2_rotated = self.rotate_expired(VanguardLayer::Layer2).await?;

        // Rotate expired L3 vanguards
        report.l3_rotated = self.rotate_expired(VanguardLayer::Layer3).await?;

        // Fill any gaps
        self.fill_vanguards(VanguardLayer::Layer2).await?;
        self.fill_vanguards(VanguardLayer::Layer3).await?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.l2_rotations += report.l2_rotated as u64;
            stats.l3_rotations += report.l3_rotated as u64;
            stats.last_rotation = Some(Instant::now());
        }

        if report.l2_rotated > 0 || report.l3_rotated > 0 {
            info!(
                "🛡️ Vanguard maintenance: rotated {} L2, {} L3",
                report.l2_rotated, report.l3_rotated
            );
        }

        Ok(report)
    }

    /// Rotate expired vanguards for a layer
    async fn rotate_expired(&self, layer: VanguardLayer) -> Result<usize> {
        let mut vanguards = match layer {
            VanguardLayer::Layer2 => self.l2_vanguards.write().await,
            VanguardLayer::Layer3 => self.l3_vanguards.write().await,
        };

        let before_count = vanguards.len();

        // Remove expired or unhealthy vanguards
        vanguards.retain(|v| !v.should_rotate() && v.is_healthy);

        let rotated = before_count - vanguards.len();

        for _ in 0..rotated {
            debug!("🔄 Rotated {} vanguard", layer.name());
        }

        Ok(rotated)
    }

    /// Mark a relay as unhealthy (e.g., circuit failures)
    pub async fn mark_unhealthy(&self, fingerprint: &str) {
        // Check L2
        {
            let mut vanguards = self.l2_vanguards.write().await;
            if let Some(v) = vanguards.iter_mut().find(|v| v.fingerprint == fingerprint) {
                v.mark_unhealthy();
                warn!("⚠️ Marked L2 vanguard {} as unhealthy", &fingerprint[..8]);

                let mut stats = self.stats.write().await;
                stats.unhealthy_relays_replaced += 1;
                return;
            }
        }

        // Check L3
        {
            let mut vanguards = self.l3_vanguards.write().await;
            if let Some(v) = vanguards.iter_mut().find(|v| v.fingerprint == fingerprint) {
                v.mark_unhealthy();
                warn!("⚠️ Marked L3 vanguard {} as unhealthy", &fingerprint[..8]);

                let mut stats = self.stats.write().await;
                stats.unhealthy_relays_replaced += 1;
            }
        }
    }

    /// Get current vanguard status
    pub async fn get_status(&self) -> VanguardsStatus {
        let l2 = self.l2_vanguards.read().await;
        let l3 = self.l3_vanguards.read().await;
        let stats = self.stats.read().await;

        let l2_healthy = l2.iter().filter(|v| v.is_healthy).count();
        let l3_healthy = l3.iter().filter(|v| v.is_healthy).count();

        let next_l2_rotation = l2
            .iter()
            .map(|v| v.time_until_rotation())
            .min()
            .unwrap_or(Duration::MAX);

        let next_l3_rotation = l3
            .iter()
            .map(|v| v.time_until_rotation())
            .min()
            .unwrap_or(Duration::MAX);

        VanguardsStatus {
            enabled: self.config.enabled,
            l2_count: l2.len(),
            l2_healthy,
            l3_count: l3.len(),
            l3_healthy,
            next_l2_rotation,
            next_l3_rotation,
            total_circuits: stats.circuits_built,
            total_rotations: stats.l2_rotations + stats.l3_rotations,
        }
    }

    /// Get statistics
    pub async fn get_stats(&self) -> VanguardsStats {
        self.stats.read().await.clone()
    }

    /// Save state to disk
    pub async fn save_state(&self) -> Result<()> {
        let path = match &self.config.state_path {
            Some(p) => p,
            None => return Ok(()),
        };

        let state = VanguardsState {
            l2_fingerprints: self
                .l2_vanguards
                .read()
                .await
                .iter()
                .map(|v| v.fingerprint.clone())
                .collect(),
            l3_fingerprints: self
                .l3_vanguards
                .read()
                .await
                .iter()
                .map(|v| v.fingerprint.clone())
                .collect(),
            stats: self.stats.read().await.clone(),
        };

        let json = serde_json::to_string_pretty(&state)?;
        tokio::fs::write(path, json).await?;

        debug!("💾 Saved vanguard state to {}", path);
        Ok(())
    }
}

/// Maintenance report
#[derive(Debug, Clone, Default)]
pub struct MaintenanceReport {
    pub l2_rotated: usize,
    pub l3_rotated: usize,
}

/// Current vanguards status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VanguardsStatus {
    pub enabled: bool,
    pub l2_count: usize,
    pub l2_healthy: usize,
    pub l3_count: usize,
    pub l3_healthy: usize,
    pub next_l2_rotation: Duration,
    pub next_l3_rotation: Duration,
    pub total_circuits: u64,
    pub total_rotations: u64,
}

/// Persisted vanguard state
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VanguardsState {
    l2_fingerprints: Vec<String>,
    l3_fingerprints: Vec<String>,
    stats: VanguardsStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vanguard_layer_rotation_periods() {
        // L2 should rotate slower than L3
        assert!(VanguardLayer::Layer2.min_rotation() > VanguardLayer::Layer3.min_rotation());
        assert!(VanguardLayer::Layer2.max_rotation() > VanguardLayer::Layer3.max_rotation());
    }

    #[test]
    fn test_vanguard_relay_rotation() {
        let relay = VanguardRelay::new(
            "AAAA".to_string(),
            "test".to_string(),
            "1.2.3.4".to_string(),
            VanguardLayer::Layer2,
        );

        // Should not rotate immediately
        assert!(!relay.should_rotate());
        assert!(relay.time_until_rotation() > Duration::ZERO);
    }

    #[test]
    fn test_relay_suitability() {
        let mut relay = RelayInfo {
            fingerprint: "test".to_string(),
            nickname: "test".to_string(),
            address: "1.2.3.4".to_string(),
            bandwidth: 5_000_000,
            flags: HashSet::new(),
            family: vec![],
        };

        // Without flags, not suitable
        assert!(!relay.is_suitable_vanguard(1_000_000));

        // Add required flags
        relay.flags.insert("Stable".to_string());
        relay.flags.insert("Fast".to_string());
        relay.flags.insert("Valid".to_string());
        relay.flags.insert("Running".to_string());

        // Now suitable
        assert!(relay.is_suitable_vanguard(1_000_000));

        // Exit relays not suitable
        relay.flags.insert("Exit".to_string());
        assert!(!relay.is_suitable_vanguard(1_000_000));
    }

    #[test]
    fn test_network_prefix() {
        let relay = RelayInfo {
            fingerprint: "test".to_string(),
            nickname: "test".to_string(),
            address: "192.168.1.1".to_string(),
            bandwidth: 5_000_000,
            flags: HashSet::new(),
            family: vec![],
        };

        assert_eq!(relay.network_prefix(), "192.168");
    }
}
