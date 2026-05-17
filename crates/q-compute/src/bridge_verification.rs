//! Bridge Verification — Multi-peer attestation and quorum for cross-chain bridge safety.
//!
//! **Issue #016: Bridge Safety — Compute-Verified Cross-Chain Proofs**
//!
//! This module implements the compute-layer verification pipeline for bridge
//! deposits. Instead of each node independently trusting its own RPC query,
//! the orchestrator dispatches verification tasks to N peers. Each peer
//! independently queries the external chain and signs an attestation. The
//! `BridgeVerificationManager` collects attestations and applies a configurable
//! quorum rule (default: 2-of-3) before crediting a deposit.
//!
//! ## Architecture
//!
//! ```text
//! Bridge Deposit Detected
//!   -> Layer 3 (BridgeVerify) assigns verification task
//!   -> 3 nodes independently query external chain RPC
//!   -> Each node produces an Attestation (result hash + timestamp)
//!   -> Attestations collected by AttestationCollector
//!   -> BridgeVerificationManager checks 2-of-3 quorum
//!   -> VerificationResult: Verified | Disputed | InsufficientAttestations
//! ```
//!
//! ## Quorum Rule
//!
//! The default quorum requires `quorum_threshold` (2) out of `quorum_size` (3)
//! attestors to produce identical result hashes. Both values are configurable
//! via `BridgeVerificationConfig`.

#![allow(dead_code)]

use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, warn};

// ═══════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════

/// Default number of attestors required per verification task.
const DEFAULT_QUORUM_SIZE: usize = 3;

/// Default number of attestors that must agree for quorum.
const DEFAULT_QUORUM_THRESHOLD: usize = 2;

/// Default attestation staleness threshold in seconds.
/// Attestations older than this are considered expired and ignored.
const DEFAULT_ATTESTATION_TIMEOUT_SECS: u64 = 300; // 5 minutes

// ═══════════════════════════════════════════════════════════════════
// Attestation — a single peer's signed verification result
// ═══════════════════════════════════════════════════════════════════

/// A single peer's attestation for a bridge verification task.
///
/// The `result_hash` is a SHA3-256 digest of the verification result data
/// (e.g. the serialized confirmation status from the external chain RPC).
/// Peers that independently arrive at the same result will produce identical
/// hashes, enabling byte-level quorum comparison.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Attestation {
    /// The libp2p peer ID of the attestor.
    pub peer_id: String,
    /// The task ID this attestation is for.
    pub task_id: String,
    /// SHA3-256 hash of the verification result.
    pub result_hash: [u8; 32],
    /// Unix timestamp (seconds) when the attestation was created.
    pub timestamp: u64,
    /// Placeholder for Ed25519 signature (64 bytes). Will be populated when
    /// P2P signing is wired in. Stored as Vec<u8> for serde compatibility.
    pub signature: Vec<u8>,
}

impl Attestation {
    /// Create a new attestation from raw result data.
    ///
    /// Hashes `result_data` with SHA3-256 to produce `result_hash`.
    pub fn new(peer_id: String, task_id: String, result_data: &[u8]) -> Self {
        let mut hasher = Sha3_256::new();
        hasher.update(result_data);
        let hash: [u8; 32] = hasher.finalize().into();

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            peer_id,
            task_id,
            result_hash: hash,
            timestamp,
            signature: vec![0u8; 64], // placeholder
        }
    }

    /// Create an attestation with a pre-computed hash and explicit timestamp.
    /// Useful for testing and replaying stored attestations.
    pub fn with_hash(
        peer_id: String,
        task_id: String,
        result_hash: [u8; 32],
        timestamp: u64,
    ) -> Self {
        Self {
            peer_id,
            task_id,
            result_hash,
            timestamp,
            signature: vec![0u8; 64],
        }
    }

    /// Returns the result hash as a hex string (for logging / display).
    pub fn result_hash_hex(&self) -> String {
        hex_encode(&self.result_hash)
    }

    /// Check whether this attestation has expired relative to `now_secs`.
    pub fn is_expired(&self, now_secs: u64, timeout_secs: u64) -> bool {
        if now_secs >= self.timestamp {
            now_secs - self.timestamp > timeout_secs
        } else {
            false // future timestamp — not expired
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// VerificationResult — the outcome of quorum evaluation
// ═══════════════════════════════════════════════════════════════════

/// The outcome of a bridge verification quorum check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationResult {
    /// Quorum reached: at least `threshold` attestors produced this hash.
    Verified {
        /// The agreed-upon result hash.
        result_hash: [u8; 32],
        /// Number of attestors that agreed.
        agreeing_count: usize,
        /// Total attestors that participated.
        total_count: usize,
    },
    /// No quorum: attestors disagree. Contains the distinct hashes and their
    /// counts for diagnostic purposes.
    Disputed {
        /// Each entry is (result_hash, count, peer_ids).
        mismatches: Vec<(String, usize, Vec<String>)>,
    },
    /// Not enough attestations received yet to evaluate quorum.
    InsufficientAttestations {
        /// How many we have so far.
        received: usize,
        /// How many we need to evaluate.
        required: usize,
    },
}

// ═══════════════════════════════════════════════════════════════════
// AttestationCollector — collects results from multiple peers
// ═══════════════════════════════════════════════════════════════════

/// Collects attestations for a single verification task.
///
/// One `AttestationCollector` exists per task_id. It enforces:
/// - No duplicate attestations from the same peer.
/// - Expiry filtering for stale attestations.
/// - Quorum evaluation once enough attestations arrive.
#[derive(Debug)]
pub struct AttestationCollector {
    /// The task this collector is for.
    pub task_id: String,
    /// Collected attestations keyed by peer_id to prevent duplicates.
    attestations: HashMap<String, Attestation>,
    /// When this collector was created (unix seconds).
    created_at: u64,
}

impl AttestationCollector {
    /// Create a new collector for the given task.
    pub fn new(task_id: String) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            task_id,
            attestations: HashMap::new(),
            created_at,
        }
    }

    /// Create a collector with an explicit creation timestamp (for testing).
    pub fn with_timestamp(task_id: String, created_at: u64) -> Self {
        Self {
            task_id,
            attestations: HashMap::new(),
            created_at,
        }
    }

    /// Submit an attestation. Returns `Ok(())` if accepted, or an error
    /// string if rejected (duplicate peer, wrong task_id, expired).
    pub fn submit(
        &mut self,
        attestation: Attestation,
        timeout_secs: u64,
    ) -> Result<(), String> {
        // Validate task_id match
        if attestation.task_id != self.task_id {
            return Err(format!(
                "Task ID mismatch: expected '{}', got '{}'",
                self.task_id, attestation.task_id
            ));
        }

        // Reject duplicate from same peer
        if self.attestations.contains_key(&attestation.peer_id) {
            return Err(format!(
                "Duplicate attestation from peer '{}'",
                attestation.peer_id
            ));
        }

        // Reject expired attestation
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if attestation.is_expired(now, timeout_secs) {
            return Err(format!(
                "Attestation from peer '{}' is expired (age={}s, timeout={}s)",
                attestation.peer_id,
                now.saturating_sub(attestation.timestamp),
                timeout_secs
            ));
        }

        debug!(
            "Bridge attestation accepted: task={} peer={} hash={}",
            self.task_id,
            attestation.peer_id,
            attestation.result_hash_hex()
        );

        self.attestations
            .insert(attestation.peer_id.clone(), attestation);
        Ok(())
    }

    /// Number of attestations collected so far.
    pub fn count(&self) -> usize {
        self.attestations.len()
    }

    /// Evaluate quorum for the collected attestations.
    ///
    /// - If fewer than `quorum_size` attestations are present, returns
    ///   `InsufficientAttestations`.
    /// - If at least `threshold` attestations share the same result_hash,
    ///   returns `Verified`.
    /// - Otherwise returns `Disputed`.
    pub fn evaluate(&self, quorum_size: usize, threshold: usize) -> VerificationResult {
        if self.attestations.len() < quorum_size {
            return VerificationResult::InsufficientAttestations {
                received: self.attestations.len(),
                required: quorum_size,
            };
        }

        // Group attestations by result_hash
        let mut groups: HashMap<[u8; 32], Vec<String>> = HashMap::new();
        for (peer_id, att) in &self.attestations {
            groups
                .entry(att.result_hash)
                .or_default()
                .push(peer_id.clone());
        }

        // Find the largest group
        let mut best_hash: Option<[u8; 32]> = None;
        let mut best_count = 0usize;
        for (hash, peers) in &groups {
            if peers.len() > best_count {
                best_count = peers.len();
                best_hash = Some(*hash);
            }
        }

        if best_count >= threshold {
            let hash = best_hash.unwrap();
            info!(
                "Bridge verification PASSED: task={} quorum={}/{} hash={}",
                self.task_id,
                best_count,
                self.attestations.len(),
                hex_encode(&hash)
            );
            VerificationResult::Verified {
                result_hash: hash,
                agreeing_count: best_count,
                total_count: self.attestations.len(),
            }
        } else {
            let mismatches: Vec<(String, usize, Vec<String>)> = groups
                .into_iter()
                .map(|(hash, peers)| (hex_encode(&hash), peers.len(), peers))
                .collect();

            warn!(
                "Bridge verification DISPUTED: task={} groups={:?}",
                self.task_id,
                mismatches
                    .iter()
                    .map(|(h, c, _)| format!("{}:{}", &h[..8], c))
                    .collect::<Vec<_>>()
            );
            VerificationResult::Disputed { mismatches }
        }
    }

    /// Returns all peer IDs that have submitted attestations.
    pub fn attesting_peers(&self) -> Vec<String> {
        self.attestations.keys().cloned().collect()
    }

    /// Returns the creation timestamp (unix seconds).
    pub fn created_at(&self) -> u64 {
        self.created_at
    }
}

// ═══════════════════════════════════════════════════════════════════
// BridgeVerificationConfig
// ═══════════════════════════════════════════════════════════════════

/// Configuration for the bridge verification quorum.
#[derive(Debug, Clone)]
pub struct BridgeVerificationConfig {
    /// Number of attestors required per task (default: 3).
    pub quorum_size: usize,
    /// Number of agreeing attestors required for quorum (default: 2).
    pub quorum_threshold: usize,
    /// Attestation timeout in seconds (default: 300 = 5 minutes).
    pub attestation_timeout_secs: u64,
}

impl Default for BridgeVerificationConfig {
    fn default() -> Self {
        Self {
            quorum_size: DEFAULT_QUORUM_SIZE,
            quorum_threshold: DEFAULT_QUORUM_THRESHOLD,
            attestation_timeout_secs: DEFAULT_ATTESTATION_TIMEOUT_SECS,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// BridgeVerificationStats — tracking metrics
// ═══════════════════════════════════════════════════════════════════

/// Statistics tracked by the verification manager.
#[derive(Debug)]
pub struct BridgeVerificationStats {
    pub tasks_created: AtomicU64,
    pub tasks_verified: AtomicU64,
    pub tasks_disputed: AtomicU64,
    pub tasks_timed_out: AtomicU64,
    pub attestations_received: AtomicU64,
    pub attestations_rejected: AtomicU64,
}

impl Default for BridgeVerificationStats {
    fn default() -> Self {
        Self {
            tasks_created: AtomicU64::new(0),
            tasks_verified: AtomicU64::new(0),
            tasks_disputed: AtomicU64::new(0),
            tasks_timed_out: AtomicU64::new(0),
            attestations_received: AtomicU64::new(0),
            attestations_rejected: AtomicU64::new(0),
        }
    }
}

impl BridgeVerificationStats {
    /// Snapshot the stats as a simple struct (for serialization / display).
    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            tasks_created: self.tasks_created.load(Ordering::Relaxed),
            tasks_verified: self.tasks_verified.load(Ordering::Relaxed),
            tasks_disputed: self.tasks_disputed.load(Ordering::Relaxed),
            tasks_timed_out: self.tasks_timed_out.load(Ordering::Relaxed),
            attestations_received: self.attestations_received.load(Ordering::Relaxed),
            attestations_rejected: self.attestations_rejected.load(Ordering::Relaxed),
        }
    }
}

/// Non-atomic snapshot of verification statistics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StatsSnapshot {
    pub tasks_created: u64,
    pub tasks_verified: u64,
    pub tasks_disputed: u64,
    pub tasks_timed_out: u64,
    pub attestations_received: u64,
    pub attestations_rejected: u64,
}

// ═══════════════════════════════════════════════════════════════════
// BridgeVerificationManager — the top-level coordinator
// ═══════════════════════════════════════════════════════════════════

/// Manages the full lifecycle of bridge verification tasks:
/// creating collectors, accepting attestations, evaluating quorum,
/// and cleaning up stale tasks.
pub struct BridgeVerificationManager {
    /// Active verification tasks keyed by task_id.
    tasks: HashMap<String, AttestationCollector>,
    /// Configuration (quorum size, threshold, timeouts).
    config: BridgeVerificationConfig,
    /// Accumulated statistics.
    stats: BridgeVerificationStats,
}

impl BridgeVerificationManager {
    /// Create a new manager with default configuration.
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
            config: BridgeVerificationConfig::default(),
            stats: BridgeVerificationStats::default(),
        }
    }

    /// Create a new manager with custom configuration.
    pub fn with_config(config: BridgeVerificationConfig) -> Self {
        Self {
            tasks: HashMap::new(),
            config,
            stats: BridgeVerificationStats::default(),
        }
    }

    /// Register a new verification task. Returns `Err` if the task_id
    /// already exists.
    pub fn create_task(&mut self, task_id: String) -> Result<(), String> {
        if self.tasks.contains_key(&task_id) {
            return Err(format!("Task '{}' already exists", task_id));
        }

        info!("Bridge verification task created: {}", task_id);
        self.tasks
            .insert(task_id.clone(), AttestationCollector::new(task_id));
        self.stats.tasks_created.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Submit an attestation for a task. The task must have been created
    /// via `create_task` first.
    pub fn submit_attestation(&mut self, attestation: Attestation) -> Result<(), String> {
        let task_id = attestation.task_id.clone();

        let collector = self.tasks.get_mut(&task_id).ok_or_else(|| {
            format!("No verification task found for '{}'", task_id)
        })?;

        match collector.submit(attestation, self.config.attestation_timeout_secs) {
            Ok(()) => {
                self.stats
                    .attestations_received
                    .fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                self.stats
                    .attestations_rejected
                    .fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    /// Check the verification status of a task.
    ///
    /// Returns `None` if the task_id is unknown.
    pub fn check_task(&self, task_id: &str) -> Option<VerificationResult> {
        self.tasks.get(task_id).map(|collector| {
            collector.evaluate(self.config.quorum_size, self.config.quorum_threshold)
        })
    }

    /// Evaluate and finalize a task: check quorum, update stats, and remove
    /// the task from the active set if it has been resolved (Verified or
    /// Disputed with enough attestations). Returns the result.
    pub fn finalize_task(&mut self, task_id: &str) -> Option<VerificationResult> {
        let result = self.check_task(task_id)?;

        match &result {
            VerificationResult::Verified { .. } => {
                self.stats.tasks_verified.fetch_add(1, Ordering::Relaxed);
                self.tasks.remove(task_id);
            }
            VerificationResult::Disputed { .. } => {
                self.stats.tasks_disputed.fetch_add(1, Ordering::Relaxed);
                self.tasks.remove(task_id);
            }
            VerificationResult::InsufficientAttestations { .. } => {
                // Keep the task open — more attestations may arrive.
            }
        }

        Some(result)
    }

    /// Remove stale tasks that have been open longer than the attestation
    /// timeout without reaching quorum. Returns the number of tasks pruned.
    pub fn prune_stale_tasks(&mut self) -> usize {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let timeout = self.config.attestation_timeout_secs;
        let stale_ids: Vec<String> = self
            .tasks
            .iter()
            .filter(|(_, collector)| {
                now.saturating_sub(collector.created_at()) > timeout
            })
            .map(|(id, _)| id.clone())
            .collect();

        let count = stale_ids.len();
        for id in &stale_ids {
            warn!("Bridge verification task timed out: {}", id);
            self.tasks.remove(id);
            self.stats.tasks_timed_out.fetch_add(1, Ordering::Relaxed);
        }

        if count > 0 {
            info!("Pruned {} stale bridge verification tasks", count);
        }

        count
    }

    /// Number of active (in-progress) verification tasks.
    pub fn active_task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Returns a snapshot of the verification statistics.
    pub fn stats(&self) -> StatsSnapshot {
        self.stats.snapshot()
    }

    /// Returns a reference to the current configuration.
    pub fn config(&self) -> &BridgeVerificationConfig {
        &self.config
    }

    /// Check if a specific task exists.
    pub fn has_task(&self, task_id: &str) -> bool {
        self.tasks.contains_key(task_id)
    }
}

impl Default for BridgeVerificationManager {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

/// Simple hex encoder (avoids pulling in the `hex` crate).
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Compute SHA3-256 hash of arbitrary data (convenience wrapper).
pub fn sha3_hash(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(data);
    hasher.finalize().into()
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create an attestation with a known hash (deterministic).
    fn make_attestation(peer_id: &str, task_id: &str, data: &[u8]) -> Attestation {
        Attestation::new(peer_id.to_string(), task_id.to_string(), data)
    }

    /// Helper: current unix timestamp in seconds.
    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    // ───────────────────────────────────────────────────────────────
    // Test 1: Unanimous agreement (3/3 match)
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_unanimous_agreement() {
        let mut mgr = BridgeVerificationManager::new();
        mgr.create_task("task-1".into()).unwrap();

        let data = b"btc-tx-confirmed-at-block-800000";

        mgr.submit_attestation(make_attestation("peer-A", "task-1", data))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-B", "task-1", data))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-C", "task-1", data))
            .unwrap();

        let result = mgr.check_task("task-1").unwrap();
        match result {
            VerificationResult::Verified {
                agreeing_count,
                total_count,
                ..
            } => {
                assert_eq!(agreeing_count, 3);
                assert_eq!(total_count, 3);
            }
            other => panic!("Expected Verified, got {:?}", other),
        }
    }

    // ───────────────────────────────────────────────────────────────
    // Test 2: 2-of-3 quorum (2 match, 1 differs)
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_two_of_three_quorum() {
        let mut mgr = BridgeVerificationManager::new();
        mgr.create_task("task-2".into()).unwrap();

        let correct_data = b"btc-deposit-0.05-confirmed";
        let wrong_data = b"btc-deposit-NOT-confirmed";

        mgr.submit_attestation(make_attestation("peer-A", "task-2", correct_data))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-B", "task-2", correct_data))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-C", "task-2", wrong_data))
            .unwrap();

        let result = mgr.check_task("task-2").unwrap();
        match result {
            VerificationResult::Verified {
                agreeing_count,
                total_count,
                ..
            } => {
                assert_eq!(agreeing_count, 2);
                assert_eq!(total_count, 3);
            }
            other => panic!("Expected Verified (2-of-3), got {:?}", other),
        }
    }

    // ───────────────────────────────────────────────────────────────
    // Test 3: All different (no quorum -> Disputed)
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_all_different_disputed() {
        let mut mgr = BridgeVerificationManager::new();
        mgr.create_task("task-3".into()).unwrap();

        mgr.submit_attestation(make_attestation("peer-A", "task-3", b"result-A"))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-B", "task-3", b"result-B"))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-C", "task-3", b"result-C"))
            .unwrap();

        let result = mgr.check_task("task-3").unwrap();
        match result {
            VerificationResult::Disputed { mismatches } => {
                assert_eq!(mismatches.len(), 3, "Expected 3 distinct hash groups");
                for (_, count, _) in &mismatches {
                    assert_eq!(*count, 1, "Each group should have exactly 1 peer");
                }
            }
            other => panic!("Expected Disputed, got {:?}", other),
        }
    }

    // ───────────────────────────────────────────────────────────────
    // Test 4: Single attestation (insufficient)
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_single_attestation_insufficient() {
        let mut mgr = BridgeVerificationManager::new();
        mgr.create_task("task-4".into()).unwrap();

        mgr.submit_attestation(make_attestation("peer-A", "task-4", b"some-data"))
            .unwrap();

        let result = mgr.check_task("task-4").unwrap();
        match result {
            VerificationResult::InsufficientAttestations {
                received,
                required,
            } => {
                assert_eq!(received, 1);
                assert_eq!(required, 3);
            }
            other => panic!("Expected InsufficientAttestations, got {:?}", other),
        }
    }

    // ───────────────────────────────────────────────────────────────
    // Test 5: Duplicate attestation from same peer (rejected)
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_duplicate_attestation_rejected() {
        let mut mgr = BridgeVerificationManager::new();
        mgr.create_task("task-5".into()).unwrap();

        mgr.submit_attestation(make_attestation("peer-A", "task-5", b"data-1"))
            .unwrap();

        // Same peer, different data — should be rejected
        let err = mgr
            .submit_attestation(make_attestation("peer-A", "task-5", b"data-2"))
            .unwrap_err();
        assert!(
            err.contains("Duplicate"),
            "Error should mention 'Duplicate': {}",
            err
        );

        // Only 1 attestation should be recorded
        let result = mgr.check_task("task-5").unwrap();
        match result {
            VerificationResult::InsufficientAttestations { received, .. } => {
                assert_eq!(received, 1);
            }
            other => panic!("Expected InsufficientAttestations, got {:?}", other),
        }

        // Verify rejection stats
        let stats = mgr.stats();
        assert_eq!(stats.attestations_received, 1);
        assert_eq!(stats.attestations_rejected, 1);
    }

    // ───────────────────────────────────────────────────────────────
    // Test 6: Timeout handling (stale attestations)
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_stale_attestation_rejected() {
        let mut mgr = BridgeVerificationManager::new();
        mgr.create_task("task-6".into()).unwrap();

        // Create an attestation that is 600 seconds old (timeout is 300s)
        let old_timestamp = now_secs().saturating_sub(600);
        let stale = Attestation::with_hash(
            "peer-stale".into(),
            "task-6".into(),
            sha3_hash(b"old-data"),
            old_timestamp,
        );

        let err = mgr.submit_attestation(stale).unwrap_err();
        assert!(
            err.contains("expired"),
            "Error should mention 'expired': {}",
            err
        );
    }

    // ───────────────────────────────────────────────────────────────
    // Test 7: Stats tracking
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_tracking() {
        let mut mgr = BridgeVerificationManager::new();

        // Create 2 tasks
        mgr.create_task("task-7a".into()).unwrap();
        mgr.create_task("task-7b".into()).unwrap();

        // Submit attestations for task-7a (unanimous)
        let data = b"confirmed";
        mgr.submit_attestation(make_attestation("peer-A", "task-7a", data))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-B", "task-7a", data))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-C", "task-7a", data))
            .unwrap();

        // Submit attestations for task-7b (all different)
        mgr.submit_attestation(make_attestation("peer-A", "task-7b", b"x"))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-B", "task-7b", b"y"))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-C", "task-7b", b"z"))
            .unwrap();

        // Finalize both
        let r7a = mgr.finalize_task("task-7a").unwrap();
        assert!(matches!(r7a, VerificationResult::Verified { .. }));

        let r7b = mgr.finalize_task("task-7b").unwrap();
        assert!(matches!(r7b, VerificationResult::Disputed { .. }));

        let stats = mgr.stats();
        assert_eq!(stats.tasks_created, 2);
        assert_eq!(stats.tasks_verified, 1);
        assert_eq!(stats.tasks_disputed, 1);
        assert_eq!(stats.attestations_received, 6);
        assert_eq!(stats.attestations_rejected, 0);

        // Both tasks should be removed after finalization
        assert_eq!(mgr.active_task_count(), 0);
    }

    // ───────────────────────────────────────────────────────────────
    // Test 8: Quorum with custom thresholds
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_custom_quorum_threshold() {
        // 3-of-5 quorum
        let config = BridgeVerificationConfig {
            quorum_size: 5,
            quorum_threshold: 3,
            attestation_timeout_secs: 300,
        };
        let mut mgr = BridgeVerificationManager::with_config(config);
        mgr.create_task("task-8".into()).unwrap();

        let correct = b"correct-result";
        let wrong = b"wrong-result";

        // Submit 3 correct + 2 wrong = quorum at 3-of-5
        mgr.submit_attestation(make_attestation("peer-1", "task-8", correct))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-2", "task-8", correct))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-3", "task-8", correct))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-4", "task-8", wrong))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-5", "task-8", wrong))
            .unwrap();

        let result = mgr.check_task("task-8").unwrap();
        match result {
            VerificationResult::Verified {
                agreeing_count,
                total_count,
                result_hash,
            } => {
                assert_eq!(agreeing_count, 3);
                assert_eq!(total_count, 5);
                assert_eq!(result_hash, sha3_hash(correct));
            }
            other => panic!("Expected Verified (3-of-5), got {:?}", other),
        }

        // Now test that 2-of-5 does NOT meet the 3-of-5 threshold
        mgr.create_task("task-8b".into()).unwrap();
        mgr.submit_attestation(make_attestation("peer-1", "task-8b", b"a"))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-2", "task-8b", b"a"))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-3", "task-8b", b"b"))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-4", "task-8b", b"c"))
            .unwrap();
        mgr.submit_attestation(make_attestation("peer-5", "task-8b", b"d"))
            .unwrap();

        let result = mgr.check_task("task-8b").unwrap();
        assert!(
            matches!(result, VerificationResult::Disputed { .. }),
            "2-of-5 should not meet 3-of-5 threshold"
        );
    }

    // ───────────────────────────────────────────────────────────────
    // Test 9: Empty / unknown task verification
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_empty_and_unknown_task() {
        let mgr = BridgeVerificationManager::new();

        // Checking an unknown task returns None
        assert!(mgr.check_task("nonexistent").is_none());

        // Submitting to unknown task returns error
        let mut mgr = mgr;
        let att = make_attestation("peer-X", "nonexistent", b"data");
        let err = mgr.submit_attestation(att).unwrap_err();
        assert!(
            err.contains("No verification task"),
            "Error should mention missing task: {}",
            err
        );

        // A newly created task with no attestations reports insufficient
        mgr.create_task("task-9".into()).unwrap();
        let result = mgr.check_task("task-9").unwrap();
        match result {
            VerificationResult::InsufficientAttestations {
                received,
                required,
            } => {
                assert_eq!(received, 0);
                assert_eq!(required, 3);
            }
            other => panic!(
                "Expected InsufficientAttestations for empty task, got {:?}",
                other
            ),
        }
    }

    // ───────────────────────────────────────────────────────────────
    // Test 10: Multiple concurrent tasks
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_multiple_concurrent_tasks() {
        let mut mgr = BridgeVerificationManager::new();

        // Create 3 tasks simultaneously
        mgr.create_task("btc-deposit-001".into()).unwrap();
        mgr.create_task("eth-deposit-002".into()).unwrap();
        mgr.create_task("zec-deposit-003".into()).unwrap();

        assert_eq!(mgr.active_task_count(), 3);

        // Submit attestations for task 1: verified (3/3)
        let btc_data = b"btc-block-800123-confirmed";
        mgr.submit_attestation(make_attestation("node-1", "btc-deposit-001", btc_data))
            .unwrap();
        mgr.submit_attestation(make_attestation("node-2", "btc-deposit-001", btc_data))
            .unwrap();
        mgr.submit_attestation(make_attestation("node-3", "btc-deposit-001", btc_data))
            .unwrap();

        // Submit attestations for task 2: disputed (all different)
        mgr.submit_attestation(make_attestation("node-1", "eth-deposit-002", b"eth-A"))
            .unwrap();
        mgr.submit_attestation(make_attestation("node-2", "eth-deposit-002", b"eth-B"))
            .unwrap();
        mgr.submit_attestation(make_attestation("node-3", "eth-deposit-002", b"eth-C"))
            .unwrap();

        // Task 3: only 1 attestation so far
        mgr.submit_attestation(make_attestation("node-1", "zec-deposit-003", b"zec-ok"))
            .unwrap();

        // Verify results independently
        let r1 = mgr.check_task("btc-deposit-001").unwrap();
        assert!(matches!(r1, VerificationResult::Verified { .. }));

        let r2 = mgr.check_task("eth-deposit-002").unwrap();
        assert!(matches!(r2, VerificationResult::Disputed { .. }));

        let r3 = mgr.check_task("zec-deposit-003").unwrap();
        assert!(matches!(
            r3,
            VerificationResult::InsufficientAttestations { .. }
        ));

        // Finalize task 1 and 2 — they should be removed
        mgr.finalize_task("btc-deposit-001");
        mgr.finalize_task("eth-deposit-002");
        assert_eq!(mgr.active_task_count(), 1); // only zec remains

        // Complete task 3 with quorum
        mgr.submit_attestation(make_attestation("node-2", "zec-deposit-003", b"zec-ok"))
            .unwrap();
        mgr.submit_attestation(make_attestation("node-3", "zec-deposit-003", b"zec-ok"))
            .unwrap();

        let r3_final = mgr.finalize_task("zec-deposit-003").unwrap();
        assert!(matches!(r3_final, VerificationResult::Verified { .. }));

        assert_eq!(mgr.active_task_count(), 0);

        let stats = mgr.stats();
        assert_eq!(stats.tasks_created, 3);
        assert_eq!(stats.tasks_verified, 2); // btc + zec
        assert_eq!(stats.tasks_disputed, 1); // eth
    }

    // ───────────────────────────────────────────────────────────────
    // Test 11 (bonus): Stale task pruning
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_stale_task_pruning() {
        let config = BridgeVerificationConfig {
            quorum_size: 3,
            quorum_threshold: 2,
            attestation_timeout_secs: 10, // very short for testing
        };
        let mut mgr = BridgeVerificationManager::with_config(config);

        // Manually insert a collector with an old timestamp
        let old_collector =
            AttestationCollector::with_timestamp("old-task".into(), now_secs().saturating_sub(20));
        mgr.tasks.insert("old-task".into(), old_collector);

        // Insert a fresh task
        mgr.create_task("fresh-task".into()).unwrap();

        assert_eq!(mgr.active_task_count(), 2);

        let pruned = mgr.prune_stale_tasks();
        assert_eq!(pruned, 1, "Should prune exactly 1 stale task");
        assert_eq!(mgr.active_task_count(), 1);
        assert!(mgr.has_task("fresh-task"));
        assert!(!mgr.has_task("old-task"));

        let stats = mgr.stats();
        assert_eq!(stats.tasks_timed_out, 1);
    }

    // ───────────────────────────────────────────────────────────────
    // Test 12 (bonus): Attestation hash determinism
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_hash_determinism() {
        let data = b"btc-tx:abc123:confirmed:block-800500";
        let a1 = make_attestation("peer-1", "task-x", data);
        let a2 = make_attestation("peer-2", "task-x", data);

        assert_eq!(
            a1.result_hash, a2.result_hash,
            "Same data from different peers must produce identical hashes"
        );

        // Different data must produce different hashes
        let a3 = make_attestation("peer-3", "task-x", b"different-data");
        assert_ne!(a1.result_hash, a3.result_hash);
    }

    // ───────────────────────────────────────────────────────────────
    // Test 13 (bonus): Duplicate task creation rejected
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_duplicate_task_creation_rejected() {
        let mut mgr = BridgeVerificationManager::new();
        mgr.create_task("task-dup".into()).unwrap();

        let err = mgr.create_task("task-dup".into()).unwrap_err();
        assert!(
            err.contains("already exists"),
            "Error should mention 'already exists': {}",
            err
        );
    }
}
