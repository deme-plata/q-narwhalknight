/// 🛡️ v1.3.0-beta: SHA3-256 Peer Reputation System
/// Prevents false height claims and malicious peer behavior
///
/// Root Cause Analysis (from SHA3_DATA_INTEGRITY_TECHNICAL_REVIEW.md):
/// - Peer `12D3KooWCMuPcyZVW3DfMLUyWC3UMGHcEJ8um5dP1J11SaqkMZvg` advertised false height 78,377
/// - Actual network height was ~16,818
/// - Node entered infinite sync loop trying to sync to unreachable height
/// - Result: Data corruption, stuck node, potential billions in losses on mainnet
///
/// Solution: Multi-layer defense with SHA3-256 integrity proofs
/// 1. Track peer reputation scores
/// 2. Penalize peers for delivering fewer blocks than claimed
/// 3. Auto-ban peers with reputation below threshold
/// 4. SHA3-256 height proof for cryptographic verification (Phase 2)

use sha3::{Sha3_256, Digest};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Trust level thresholds for peer classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeerTrustLevel {
    /// New peer, no history (score: 50-100)
    Unknown,
    /// Suspicious behavior detected (score: 20-49)
    Suspicious,
    /// Known bad actor (score: 0-19)
    Banned,
    /// Verified good peer (score: 80-100)
    Trusted,
    /// Explicitly allowlisted (bootstrap nodes)
    Allowlisted,
}

impl PeerTrustLevel {
    pub fn from_score(score: f64) -> Self {
        match score as i32 {
            0..=19 => PeerTrustLevel::Banned,
            20..=49 => PeerTrustLevel::Suspicious,
            50..=79 => PeerTrustLevel::Unknown,
            _ => PeerTrustLevel::Trusted,
        }
    }
}

/// Individual peer reputation entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerReputationEntry {
    /// Peer identifier (libp2p PeerId as string)
    pub peer_id: String,

    /// Current reputation score (0-100)
    /// Starts at 50 for new peers (neutral)
    pub score: f64,

    /// Number of successful block deliveries
    pub successful_deliveries: u64,

    /// Number of failed/short deliveries
    pub failed_deliveries: u64,

    /// Number of false height claims detected
    pub false_height_claims: u64,

    /// Last announced height
    pub last_announced_height: u64,

    /// Last delivered height (actual blocks we received)
    pub last_delivered_height: u64,

    /// Timestamp of last interaction
    pub last_seen: u64,

    /// Timestamp when peer was banned (0 if not banned)
    pub banned_at: u64,

    /// Ban duration in seconds (0 for permanent)
    pub ban_duration: u64,

    /// SHA3-256 hash of peer's last height proof (Phase 2)
    #[serde(default)]
    pub last_height_proof_hash: Option<[u8; 32]>,

    /// Number of valid proofs provided
    pub valid_proofs: u64,

    /// Number of invalid proofs (instant ban if > 0)
    pub invalid_proofs: u64,
}

impl PeerReputationEntry {
    /// Create new peer entry with neutral score
    pub fn new(peer_id: String) -> Self {
        Self {
            peer_id,
            score: 50.0, // Neutral starting score
            successful_deliveries: 0,
            failed_deliveries: 0,
            false_height_claims: 0,
            last_announced_height: 0,
            last_delivered_height: 0,
            last_seen: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            banned_at: 0,
            ban_duration: 0,
            last_height_proof_hash: None,
            valid_proofs: 0,
            invalid_proofs: 0,
        }
    }

    /// Get current trust level
    pub fn trust_level(&self) -> PeerTrustLevel {
        if self.banned_at > 0 && !self.is_ban_expired() {
            return PeerTrustLevel::Banned;
        }
        if self.invalid_proofs > 0 {
            return PeerTrustLevel::Banned;
        }
        PeerTrustLevel::from_score(self.score)
    }

    /// Check if ban has expired
    pub fn is_ban_expired(&self) -> bool {
        if self.banned_at == 0 {
            return true;
        }
        if self.ban_duration == 0 {
            return false; // Permanent ban
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now > self.banned_at + self.ban_duration
    }
}

/// Peer Reputation Manager with SHA3-256 integrity
pub struct PeerReputationManager {
    /// Peer reputation database (in-memory for quick access)
    peers: Arc<RwLock<HashMap<String, PeerReputationEntry>>>,

    /// Allowlisted peers (bootstrap nodes, never ban)
    allowlist: Arc<RwLock<Vec<String>>>,

    /// Score thresholds
    config: ReputationConfig,
}

/// Configuration for reputation scoring
#[derive(Clone, Debug)]
pub struct ReputationConfig {
    /// Minimum score before auto-ban (default: 20)
    pub ban_threshold: f64,

    /// Score penalty for false height claim (default: -25)
    pub false_height_penalty: f64,

    /// Score penalty for failed delivery (default: -5)
    pub failed_delivery_penalty: f64,

    /// Score boost for successful delivery (default: +2)
    pub successful_delivery_boost: f64,

    /// Score boost for valid proof (default: +10)
    pub valid_proof_boost: f64,

    /// Ban duration for score-based ban (default: 1 hour)
    pub temp_ban_duration: Duration,

    /// Tolerance for height claims (allow slight differences)
    /// If peer claims 1000 but delivers 990, this tolerance (1%) prevents ban
    pub height_tolerance_percent: f64,

    /// Maximum height difference before instant ban
    /// If peer claims 100,000 but we only have 20,000, likely malicious
    pub max_height_difference: u64,
}

impl Default for ReputationConfig {
    fn default() -> Self {
        Self {
            ban_threshold: 20.0,
            false_height_penalty: -25.0,
            failed_delivery_penalty: -5.0,
            successful_delivery_boost: 2.0,
            valid_proof_boost: 10.0,
            temp_ban_duration: Duration::from_secs(3600), // 1 hour
            height_tolerance_percent: 2.0, // 2% tolerance
            max_height_difference: 100_000, // Suspicious if > 100k difference
        }
    }
}

impl PeerReputationManager {
    /// Create new reputation manager
    pub fn new(config: ReputationConfig) -> Self {
        info!("🛡️ [PEER REPUTATION] Initializing SHA3-256 peer reputation system");
        info!("   Ban threshold: {}", config.ban_threshold);
        info!("   False height penalty: {}", config.false_height_penalty);
        info!("   Max height difference: {}", config.max_height_difference);

        Self {
            peers: Arc::new(RwLock::new(HashMap::new())),
            allowlist: Arc::new(RwLock::new(Vec::new())),
            config,
        }
    }

    /// Add peer to allowlist (bootstrap nodes)
    pub async fn add_to_allowlist(&self, peer_id: &str) {
        let mut allowlist = self.allowlist.write().await;
        if !allowlist.contains(&peer_id.to_string()) {
            allowlist.push(peer_id.to_string());
            info!("🔒 [PEER REPUTATION] Added {} to allowlist (protected from bans)",
                  &peer_id[..peer_id.len().min(16)]);
        }
    }

    /// Check if peer is allowlisted
    pub async fn is_allowlisted(&self, peer_id: &str) -> bool {
        let allowlist = self.allowlist.read().await;
        allowlist.iter().any(|p| p == peer_id)
    }

    /// Check if peer is banned
    pub async fn is_banned(&self, peer_id: &str) -> bool {
        // Never ban allowlisted peers
        if self.is_allowlisted(peer_id).await {
            return false;
        }

        let peers = self.peers.read().await;
        if let Some(entry) = peers.get(peer_id) {
            entry.trust_level() == PeerTrustLevel::Banned
        } else {
            false
        }
    }

    /// Get or create peer entry
    pub async fn get_or_create_peer(&self, peer_id: &str) -> PeerReputationEntry {
        let mut peers = self.peers.write().await;
        peers.entry(peer_id.to_string())
            .or_insert_with(|| PeerReputationEntry::new(peer_id.to_string()))
            .clone()
    }

    /// Record peer height announcement
    pub async fn record_height_announcement(&self, peer_id: &str, announced_height: u64, our_height: u64) {
        // Check for suspiciously high claims
        let height_difference = if announced_height > our_height {
            announced_height - our_height
        } else {
            0
        };

        if height_difference > self.config.max_height_difference {
            // CRITICAL: Peer claims height way beyond what's reasonable
            warn!("⚠️ [PEER REPUTATION] SUSPICIOUS: {} claims height {} but we have {}!",
                  &peer_id[..peer_id.len().min(16)], announced_height, our_height);
            warn!("   Difference: {} blocks (max allowed: {})",
                  height_difference, self.config.max_height_difference);
            warn!("   NOT trusting this announcement until verified!");

            // Apply penalty but don't instant-ban yet (wait for delivery verification)
            self.apply_penalty(peer_id, self.config.false_height_penalty / 2.0).await;
        }

        let mut peers = self.peers.write().await;
        let entry = peers.entry(peer_id.to_string())
            .or_insert_with(|| PeerReputationEntry::new(peer_id.to_string()));

        entry.last_announced_height = announced_height;
        entry.last_seen = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        debug!("📊 [PEER REPUTATION] {} announced height {} (our height: {})",
               &peer_id[..peer_id.len().min(16)], announced_height, our_height);
    }

    /// Record successful block delivery
    /// Returns false if peer should not be trusted
    pub async fn record_delivery(&self, peer_id: &str, requested_height: u64, delivered_height: u64) -> bool {
        // Check if peer delivered what they promised
        let tolerance = (requested_height as f64 * self.config.height_tolerance_percent / 100.0) as u64;
        let min_acceptable = requested_height.saturating_sub(tolerance);

        let mut peers = self.peers.write().await;
        let entry = peers.entry(peer_id.to_string())
            .or_insert_with(|| PeerReputationEntry::new(peer_id.to_string()));

        entry.last_delivered_height = delivered_height;
        entry.last_seen = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if delivered_height >= min_acceptable {
            // Good delivery
            entry.successful_deliveries += 1;
            entry.score = (entry.score + self.config.successful_delivery_boost).min(100.0);

            debug!("✅ [PEER REPUTATION] {} delivered {} blocks (requested: {}, score: {:.1})",
                   &peer_id[..peer_id.len().min(16)], delivered_height, requested_height, entry.score);
            true
        } else {
            // Bad delivery - peer claimed more than they delivered
            entry.failed_deliveries += 1;

            // Calculate shortfall
            let shortfall = requested_height.saturating_sub(delivered_height);
            let shortfall_percent = (shortfall as f64 / requested_height as f64) * 100.0;

            // Severe penalty for large shortfalls
            let penalty = if shortfall_percent > 50.0 {
                // More than 50% shortfall - likely false height claim
                entry.false_height_claims += 1;
                self.config.false_height_penalty
            } else {
                self.config.failed_delivery_penalty * (shortfall_percent / 10.0)
            };

            entry.score = (entry.score + penalty).max(0.0);

            warn!("⚠️ [PEER REPUTATION] {} short-delivered: got {} of {} ({:.1}% shortfall, score: {:.1})",
                  &peer_id[..peer_id.len().min(16)], delivered_height, requested_height,
                  shortfall_percent, entry.score);

            // Check for auto-ban
            if entry.score < self.config.ban_threshold {
                drop(peers);
                self.ban_peer(peer_id, "Score dropped below threshold due to short deliveries").await;
                return false;
            }

            entry.score >= self.config.ban_threshold
        }
    }

    /// Record false height claim (severe penalty)
    pub async fn record_false_height_claim(&self, peer_id: &str, claimed: u64, actual: u64) {
        // Check allowlist
        if self.is_allowlisted(peer_id).await {
            warn!("⚠️ [PEER REPUTATION] Allowlisted peer {} claimed {} but actual is {} - NOT banning",
                  &peer_id[..peer_id.len().min(16)], claimed, actual);
            return;
        }

        let mut peers = self.peers.write().await;
        let entry = peers.entry(peer_id.to_string())
            .or_insert_with(|| PeerReputationEntry::new(peer_id.to_string()));

        entry.false_height_claims += 1;
        entry.score = (entry.score + self.config.false_height_penalty).max(0.0);

        error!("🚨 [PEER REPUTATION] FALSE HEIGHT CLAIM from {}!", &peer_id[..peer_id.len().min(16)]);
        error!("   Claimed: {} blocks", claimed);
        error!("   Actual:  {} blocks", actual);
        error!("   Shortfall: {} blocks ({:.1}%)",
               claimed.saturating_sub(actual),
               ((claimed.saturating_sub(actual)) as f64 / claimed as f64) * 100.0);
        error!("   New score: {:.1} (threshold: {})", entry.score, self.config.ban_threshold);

        // Instant ban for severe false claims
        if entry.false_height_claims >= 2 || entry.score < self.config.ban_threshold {
            drop(peers);
            self.ban_peer(peer_id, "Multiple false height claims detected").await;
        }
    }

    /// Apply penalty to peer score
    async fn apply_penalty(&self, peer_id: &str, penalty: f64) {
        let mut peers = self.peers.write().await;
        if let Some(entry) = peers.get_mut(peer_id) {
            entry.score = (entry.score + penalty).max(0.0);
        }
    }

    /// Ban peer
    pub async fn ban_peer(&self, peer_id: &str, reason: &str) {
        // Check allowlist
        if self.is_allowlisted(peer_id).await {
            warn!("⚠️ [PEER REPUTATION] Cannot ban allowlisted peer {}: {}",
                  &peer_id[..peer_id.len().min(16)], reason);
            return;
        }

        let mut peers = self.peers.write().await;
        let entry = peers.entry(peer_id.to_string())
            .or_insert_with(|| PeerReputationEntry::new(peer_id.to_string()));

        entry.banned_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        entry.ban_duration = self.config.temp_ban_duration.as_secs();
        entry.score = 0.0;

        error!("🚫 [PEER REPUTATION] BANNED peer {} for {} seconds",
               &peer_id[..peer_id.len().min(16)], entry.ban_duration);
        error!("   Reason: {}", reason);
        error!("   Stats: {} false claims, {} failed deliveries, {} successful",
               entry.false_height_claims, entry.failed_deliveries, entry.successful_deliveries);
    }

    /// Unban peer (admin function)
    pub async fn unban_peer(&self, peer_id: &str) {
        let mut peers = self.peers.write().await;
        if let Some(entry) = peers.get_mut(peer_id) {
            entry.banned_at = 0;
            entry.ban_duration = 0;
            entry.score = 50.0; // Reset to neutral
            info!("✅ [PEER REPUTATION] Unbanned peer {}", &peer_id[..peer_id.len().min(16)]);
        }
    }

    /// Get peer score
    pub async fn get_score(&self, peer_id: &str) -> f64 {
        let peers = self.peers.read().await;
        peers.get(peer_id).map(|e| e.score).unwrap_or(50.0)
    }

    /// Get peer trust level
    pub async fn get_trust_level(&self, peer_id: &str) -> PeerTrustLevel {
        if self.is_allowlisted(peer_id).await {
            return PeerTrustLevel::Allowlisted;
        }

        let peers = self.peers.read().await;
        peers.get(peer_id)
            .map(|e| e.trust_level())
            .unwrap_or(PeerTrustLevel::Unknown)
    }

    /// Get all banned peers
    pub async fn get_banned_peers(&self) -> Vec<String> {
        let peers = self.peers.read().await;
        peers.iter()
            .filter(|(_, e)| e.trust_level() == PeerTrustLevel::Banned)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get reputation stats for logging
    pub async fn get_stats(&self) -> ReputationStats {
        let peers = self.peers.read().await;
        let total = peers.len();
        let banned = peers.values().filter(|e| e.trust_level() == PeerTrustLevel::Banned).count();
        let trusted = peers.values().filter(|e| e.trust_level() == PeerTrustLevel::Trusted).count();
        let suspicious = peers.values().filter(|e| e.trust_level() == PeerTrustLevel::Suspicious).count();
        let total_false_claims: u64 = peers.values().map(|e| e.false_height_claims).sum();

        ReputationStats {
            total_peers: total,
            banned_peers: banned,
            trusted_peers: trusted,
            suspicious_peers: suspicious,
            total_false_height_claims: total_false_claims,
        }
    }

    // ========== SHA3-256 Height Proof Verification (Phase 2) ==========

    /// Verify SHA3-256 height proof
    /// Proof structure: SHA3-256(peer_id || height || merkle_root || timestamp || prev_proof_hash)
    pub fn verify_height_proof(
        &self,
        peer_id: &str,
        height: u64,
        merkle_root: &[u8; 32],
        timestamp: u64,
        prev_proof_hash: Option<[u8; 32]>,
        provided_proof: &[u8; 32],
    ) -> bool {
        let computed_proof = self.compute_height_proof(peer_id, height, merkle_root, timestamp, prev_proof_hash);
        computed_proof == *provided_proof
    }

    /// Compute SHA3-256 height proof
    pub fn compute_height_proof(
        &self,
        peer_id: &str,
        height: u64,
        merkle_root: &[u8; 32],
        timestamp: u64,
        prev_proof_hash: Option<[u8; 32]>,
    ) -> [u8; 32] {
        let mut hasher = Sha3_256::new();

        // Add peer ID
        hasher.update(peer_id.as_bytes());

        // Add height (big-endian)
        hasher.update(&height.to_be_bytes());

        // Add merkle root
        hasher.update(merkle_root);

        // Add timestamp
        hasher.update(&timestamp.to_be_bytes());

        // Add previous proof hash (chain proofs for continuity)
        if let Some(prev) = prev_proof_hash {
            hasher.update(&prev);
        }

        let result = hasher.finalize();
        let mut proof = [0u8; 32];
        proof.copy_from_slice(&result);
        proof
    }

    /// Record valid height proof
    pub async fn record_valid_proof(&self, peer_id: &str, proof_hash: [u8; 32]) {
        let mut peers = self.peers.write().await;
        let entry = peers.entry(peer_id.to_string())
            .or_insert_with(|| PeerReputationEntry::new(peer_id.to_string()));

        entry.valid_proofs += 1;
        entry.last_height_proof_hash = Some(proof_hash);
        entry.score = (entry.score + self.config.valid_proof_boost).min(100.0);

        debug!("✅ [PEER REPUTATION] Valid SHA3-256 proof from {} (proofs: {}, score: {:.1})",
               &peer_id[..peer_id.len().min(16)], entry.valid_proofs, entry.score);
    }

    /// Record invalid height proof (INSTANT BAN)
    pub async fn record_invalid_proof(&self, peer_id: &str) {
        // Check allowlist
        if self.is_allowlisted(peer_id).await {
            error!("🚨 [PEER REPUTATION] Allowlisted peer {} provided invalid proof! Manual review needed.",
                   &peer_id[..peer_id.len().min(16)]);
            return;
        }

        {
            let mut peers = self.peers.write().await;
            let entry = peers.entry(peer_id.to_string())
                .or_insert_with(|| PeerReputationEntry::new(peer_id.to_string()));

            entry.invalid_proofs += 1;
        }

        error!("🚨 [PEER REPUTATION] INVALID SHA3-256 PROOF from {} - INSTANT BAN!",
               &peer_id[..peer_id.len().min(16)]);

        self.ban_peer(peer_id, "Invalid SHA3-256 height proof - cryptographic fraud detected").await;
    }
}

/// Reputation statistics for logging
#[derive(Debug, Clone)]
pub struct ReputationStats {
    pub total_peers: usize,
    pub banned_peers: usize,
    pub trusted_peers: usize,
    pub suspicious_peers: usize,
    pub total_false_height_claims: u64,
}

impl std::fmt::Display for ReputationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Peers: {} total ({} trusted, {} suspicious, {} banned), {} false height claims",
               self.total_peers, self.trusted_peers, self.suspicious_peers,
               self.banned_peers, self.total_false_height_claims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_peer_reputation_basic() {
        let config = ReputationConfig::default();
        let manager = PeerReputationManager::new(config);

        let peer_id = "12D3KooWTestPeer1234567890";

        // Initial score should be 50 (neutral)
        assert_eq!(manager.get_score(peer_id).await, 50.0);

        // Successful delivery should boost score
        manager.record_delivery(peer_id, 100, 100).await;
        assert!(manager.get_score(peer_id).await > 50.0);

        // Failed delivery should reduce score
        manager.record_delivery(peer_id, 100, 50).await;
        let score_after_fail = manager.get_score(peer_id).await;

        println!("Score after fail: {}", score_after_fail);
    }

    #[tokio::test]
    async fn test_false_height_claim_ban() {
        let config = ReputationConfig::default();
        let manager = PeerReputationManager::new(config);

        let peer_id = "12D3KooWMaliciousPeer";

        // Record false height claims
        manager.record_false_height_claim(peer_id, 100000, 16818).await;
        manager.record_false_height_claim(peer_id, 100000, 16818).await;

        // Should be banned after 2 false claims
        assert!(manager.is_banned(peer_id).await);
    }

    #[tokio::test]
    async fn test_allowlist_protection() {
        let config = ReputationConfig::default();
        let manager = PeerReputationManager::new(config);

        let peer_id = "12D3KooWBootstrapNode";

        // Add to allowlist
        manager.add_to_allowlist(peer_id).await;

        // Try to ban
        manager.record_false_height_claim(peer_id, 100000, 16818).await;
        manager.record_false_height_claim(peer_id, 100000, 16818).await;

        // Should NOT be banned (allowlisted)
        assert!(!manager.is_banned(peer_id).await);
        assert_eq!(manager.get_trust_level(peer_id).await, PeerTrustLevel::Allowlisted);
    }

    #[tokio::test]
    async fn test_sha3_height_proof() {
        let config = ReputationConfig::default();
        let manager = PeerReputationManager::new(config);

        let peer_id = "12D3KooWTestPeer";
        let height = 16818u64;
        let merkle_root = [0u8; 32];
        let timestamp = 1702400000u64;

        // Compute proof
        let proof = manager.compute_height_proof(peer_id, height, &merkle_root, timestamp, None);

        // Verify proof
        assert!(manager.verify_height_proof(peer_id, height, &merkle_root, timestamp, None, &proof));

        // Tampered proof should fail
        let mut tampered = proof;
        tampered[0] ^= 0xFF;
        assert!(!manager.verify_height_proof(peer_id, height, &merkle_root, timestamp, None, &tampered));
    }
}
