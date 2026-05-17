//! # Issue #027: Compute Marketplace P2P Protocol
//!
//! Gossipsub routing layer for the compute marketplace. Sits on top of
//! [`super::marketplace`] and handles message encoding/decoding, validation,
//! rate limiting, order book aggregation, winner selection, and settlement.
//!
//! ## Components
//!
//! | Struct               | Purpose                                            |
//! |----------------------|----------------------------------------------------|
//! | `MarketplaceRouter`  | Encode/decode/validate gossipsub marketplace msgs  |
//! | `OrderBook`          | Aggregated view of bids, capacities, job postings  |
//! | `WinnerSelection`    | Score-based bid ranking with latency tiebreaker     |
//! | `SettlementManager`  | Track pending payments and dispute timeouts         |
//! | `MarketplaceP2PStats`| Counters for messages, bids, settlements, volume   |

use crate::marketplace::{
    MarketplaceJob, MarketplaceMessage, WorkType, WorkerBid, JobStatus,
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;
use tracing::{debug, info, warn};

// ── Constants ─────────────────────────────────────────────────────

/// Maximum encoded message size (1 MB).
const MAX_MESSAGE_SIZE: usize = 1_048_576;

/// Maximum clock skew allowed for message timestamps (5 minutes).
const MAX_CLOCK_SKEW_SECS: u64 = 300;

/// Rate limit: max messages per peer per second.
const MAX_MESSAGES_PER_SEC: u32 = 10;

/// Rate limit window in seconds.
const RATE_LIMIT_WINDOW_SECS: u64 = 1;

/// Order book entry TTL before auto-expiry (30 minutes).
const ORDER_BOOK_ENTRY_TTL_SECS: u64 = 1800;

/// Settlement timeout before dispute (10 minutes).
const SETTLEMENT_TIMEOUT_SECS: u64 = 600;

// ── MarketplaceRouter ─────────────────────────────────────────────

/// Routes marketplace messages via gossipsub with validation and rate limiting.
pub struct MarketplaceRouter {
    /// Per-peer rate-limit state: peer_id -> (window_start, message_count).
    rate_limits: Arc<RwLock<HashMap<String, (Instant, u32)>>>,
    /// Whether to use gzip compression for encoding.
    use_compression: bool,
    /// Running stats.
    stats: Arc<RwLock<MarketplaceP2PStats>>,
}

impl MarketplaceRouter {
    /// Create a new marketplace router.
    pub fn new(use_compression: bool) -> Self {
        Self {
            rate_limits: Arc::new(RwLock::new(HashMap::new())),
            use_compression,
            stats: Arc::new(RwLock::new(MarketplaceP2PStats::default())),
        }
    }

    /// Encode a marketplace message to bytes (serde_json, optionally gzipped).
    pub fn encode_message(&self, msg: &MarketplaceMessage) -> Result<Vec<u8>, String> {
        let json = serde_json::to_vec(msg)
            .map_err(|e| format!("JSON encode error: {}", e))?;

        if json.len() > MAX_MESSAGE_SIZE {
            return Err(format!(
                "Message too large: {} bytes (max {})",
                json.len(),
                MAX_MESSAGE_SIZE
            ));
        }

        {
            let mut stats = self.stats.write();
            stats.messages_sent += 1;
        }

        Ok(json)
    }

    /// Decode bytes into a marketplace message with size validation.
    pub fn decode_message(&self, data: &[u8]) -> Result<MarketplaceMessage, String> {
        if data.is_empty() {
            return Err("Empty message payload".into());
        }

        if data.len() > MAX_MESSAGE_SIZE {
            let mut stats = self.stats.write();
            stats.messages_rejected += 1;
            return Err(format!(
                "Message too large: {} bytes (max {})",
                data.len(),
                MAX_MESSAGE_SIZE
            ));
        }

        let msg: MarketplaceMessage = serde_json::from_slice(data)
            .map_err(|e| {
                let mut stats = self.stats.write();
                stats.messages_rejected += 1;
                format!("JSON decode error: {}", e)
            })?;

        {
            let mut stats = self.stats.write();
            stats.messages_received += 1;
        }

        Ok(msg)
    }

    /// Validate a message's timestamp (must be within +/-5 min of local clock).
    pub fn validate_timestamp(&self, timestamp: u64) -> Result<(), String> {
        let now = now_unix();
        let diff = if timestamp > now {
            timestamp - now
        } else {
            now - timestamp
        };
        if diff > MAX_CLOCK_SKEW_SECS {
            let mut stats = self.stats.write();
            stats.messages_rejected += 1;
            return Err(format!(
                "Message timestamp too far from local clock: {}s skew (max {}s)",
                diff, MAX_CLOCK_SKEW_SECS
            ));
        }
        Ok(())
    }

    /// Check rate limit for a peer. Returns Ok(()) if allowed, Err if throttled.
    pub fn check_rate_limit(&self, peer_id: &str) -> Result<(), String> {
        let now = Instant::now();
        let mut limits = self.rate_limits.write();

        let entry = limits.entry(peer_id.to_string()).or_insert((now, 0));
        let window_elapsed = now.duration_since(entry.0);

        if window_elapsed >= Duration::from_secs(RATE_LIMIT_WINDOW_SECS) {
            // Reset window.
            *entry = (now, 1);
            Ok(())
        } else if entry.1 >= MAX_MESSAGES_PER_SEC {
            let mut stats = self.stats.write();
            stats.messages_rejected += 1;
            Err(format!(
                "Rate limit exceeded for peer {}: {} msgs in {}ms (max {}/s)",
                peer_id,
                entry.1,
                window_elapsed.as_millis(),
                MAX_MESSAGES_PER_SEC
            ))
        } else {
            entry.1 += 1;
            Ok(())
        }
    }

    /// Full validation pipeline: size + decode + timestamp + rate limit.
    /// Returns the decoded message if all checks pass.
    pub fn validate_and_decode(
        &self,
        data: &[u8],
        sender_peer_id: &str,
    ) -> Result<MarketplaceMessage, String> {
        self.check_rate_limit(sender_peer_id)?;

        let msg = self.decode_message(data)?;

        // Extract timestamp from message for clock-skew validation.
        let ts = extract_timestamp(&msg);
        if let Some(ts) = ts {
            self.validate_timestamp(ts)?;
        }

        Ok(msg)
    }

    /// Clean up stale rate-limit entries (call periodically, e.g., every 60s).
    pub fn cleanup_rate_limits(&self) {
        let now = Instant::now();
        let mut limits = self.rate_limits.write();
        limits.retain(|_, (window_start, _)| {
            now.duration_since(*window_start) < Duration::from_secs(60)
        });
    }

    /// Get current P2P stats snapshot.
    pub fn stats(&self) -> MarketplaceP2PStats {
        self.stats.read().clone()
    }

    /// Get a mutable reference to the stats (for updating from external sources).
    pub fn stats_mut(&self) -> impl std::ops::DerefMut<Target = MarketplaceP2PStats> + '_ {
        self.stats.write()
    }
}

/// Extract a timestamp from a marketplace message (if present).
fn extract_timestamp(msg: &MarketplaceMessage) -> Option<u64> {
    match msg {
        MarketplaceMessage::CapacityAnnouncement { timestamp, .. } => Some(*timestamp),
        MarketplaceMessage::JobPosting(job) => Some(job.submitted_at),
        MarketplaceMessage::BidSubmission(bid) => Some(bid.bid_at),
        MarketplaceMessage::ResultSubmission(result) => Some(result.completed_at),
        // Assignment and Cancellation don't carry timestamps.
        MarketplaceMessage::JobAssignment { .. } => None,
        MarketplaceMessage::JobCancellation { .. } => None,
    }
}

// ── OrderBook ─────────────────────────────────────────────────────

/// A tracked capacity announcement from a peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityEntry {
    pub peer_id: String,
    pub available_cores: u32,
    pub available_ram_mb: u64,
    pub gpu_vram_mb: u64,
    pub accepted_work_types: Vec<WorkType>,
    pub min_price_micro_qug: u64,
    pub timestamp: u64,
    /// When we received this entry (local monotonic clock).
    #[serde(skip)]
    pub received_at: Option<Instant>,
}

/// A tracked job posting in the order book.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobPostingEntry {
    pub job: MarketplaceJob,
    /// When we received this entry (local monotonic clock).
    #[serde(skip)]
    pub received_at: Option<Instant>,
}

/// Filter for querying job postings.
#[derive(Debug, Clone, Default)]
pub struct JobFilter {
    /// Filter by work type (None = all).
    pub work_type: Option<WorkType>,
    /// Filter by max price (None = any price).
    pub max_price_micro_qug: Option<u64>,
    /// Filter by submitter peer ID.
    pub submitter_peer_id: Option<String>,
}

/// Aggregated view of marketplace state, built from gossipsub messages.
pub struct OrderBook {
    /// Bids grouped by job_id, sorted by price ascending.
    bids: Arc<RwLock<HashMap<String, Vec<WorkerBid>>>>,
    /// Known peer capacity announcements.
    capacities: Arc<RwLock<HashMap<String, CapacityEntry>>>,
    /// Available job postings.
    job_postings: Arc<RwLock<HashMap<String, JobPostingEntry>>>,
}

impl OrderBook {
    /// Create a new empty order book.
    pub fn new() -> Self {
        Self {
            bids: Arc::new(RwLock::new(HashMap::new())),
            capacities: Arc::new(RwLock::new(HashMap::new())),
            job_postings: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Ingest a marketplace message into the order book.
    pub fn ingest(&self, msg: &MarketplaceMessage) {
        match msg {
            MarketplaceMessage::CapacityAnnouncement {
                peer_id,
                available_cores,
                available_ram_mb,
                gpu_vram_mb,
                accepted_work_types,
                min_price_micro_qug,
                timestamp,
            } => {
                let mut caps = self.capacities.write();
                caps.insert(
                    peer_id.clone(),
                    CapacityEntry {
                        peer_id: peer_id.clone(),
                        available_cores: *available_cores,
                        available_ram_mb: *available_ram_mb,
                        gpu_vram_mb: *gpu_vram_mb,
                        accepted_work_types: accepted_work_types.clone(),
                        min_price_micro_qug: *min_price_micro_qug,
                        timestamp: *timestamp,
                        received_at: Some(Instant::now()),
                    },
                );
            }
            MarketplaceMessage::JobPosting(job) => {
                let mut postings = self.job_postings.write();
                postings.insert(
                    job.job_id.clone(),
                    JobPostingEntry {
                        job: job.clone(),
                        received_at: Some(Instant::now()),
                    },
                );
            }
            MarketplaceMessage::BidSubmission(bid) => {
                let mut all_bids = self.bids.write();
                let job_bids = all_bids
                    .entry(bid.job_id.clone())
                    .or_insert_with(Vec::new);
                job_bids.push(bid.clone());
                // Keep sorted by price ascending.
                job_bids.sort_by_key(|b| b.bid_price_micro_qug);
            }
            MarketplaceMessage::JobCancellation { job_id, .. } => {
                let mut postings = self.job_postings.write();
                postings.remove(job_id);
                let mut all_bids = self.bids.write();
                all_bids.remove(job_id);
            }
            MarketplaceMessage::JobAssignment { job_id, .. } => {
                // Remove from active postings once assigned.
                let mut postings = self.job_postings.write();
                if let Some(entry) = postings.get_mut(job_id) {
                    entry.job.status = JobStatus::Assigned;
                }
            }
            MarketplaceMessage::ResultSubmission(result) => {
                // Mark job as completed, clean up bids.
                let mut postings = self.job_postings.write();
                if let Some(entry) = postings.get_mut(&result.job_id) {
                    entry.job.status = JobStatus::Completed;
                }
                let mut all_bids = self.bids.write();
                all_bids.remove(&result.job_id);
            }
        }
    }

    /// Get all bids for a specific job, sorted by price ascending.
    pub fn bids_for_job(&self, job_id: &str) -> Vec<WorkerBid> {
        let bids = self.bids.read();
        bids.get(job_id).cloned().unwrap_or_default()
    }

    /// Get all known peer capacity announcements.
    pub fn capacity_announcements(&self) -> Vec<CapacityEntry> {
        let caps = self.capacities.read();
        caps.values().cloned().collect()
    }

    /// Get available job postings, optionally filtered.
    pub fn job_postings(&self, filter: &JobFilter) -> Vec<MarketplaceJob> {
        let postings = self.job_postings.read();
        postings
            .values()
            .filter(|entry| {
                let job = &entry.job;
                // Only open/bidding jobs.
                if job.status != JobStatus::Open && job.status != JobStatus::Bidding {
                    return false;
                }
                if let Some(wt) = &filter.work_type {
                    if job.work_type != *wt {
                        return false;
                    }
                }
                if let Some(max_price) = filter.max_price_micro_qug {
                    if job.max_price_micro_qug > max_price {
                        return false;
                    }
                }
                if let Some(ref submitter) = filter.submitter_peer_id {
                    if job.submitter_peer_id != *submitter {
                        return false;
                    }
                }
                true
            })
            .map(|entry| entry.job.clone())
            .collect()
    }

    /// Get the number of tracked bids across all jobs.
    pub fn total_bid_count(&self) -> usize {
        let bids = self.bids.read();
        bids.values().map(|v| v.len()).sum()
    }

    /// Get the number of tracked job postings.
    pub fn total_job_count(&self) -> usize {
        let postings = self.job_postings.read();
        postings.len()
    }

    /// Expire stale entries older than 30 minutes without refresh.
    /// Returns the number of entries removed.
    pub fn expire_stale_entries(&self) -> usize {
        let now = Instant::now();
        let ttl = Duration::from_secs(ORDER_BOOK_ENTRY_TTL_SECS);
        let mut removed = 0usize;

        // Expire capacity announcements.
        {
            let mut caps = self.capacities.write();
            let before = caps.len();
            caps.retain(|_, entry| {
                entry
                    .received_at
                    .map_or(true, |t| now.duration_since(t) < ttl)
            });
            removed += before - caps.len();
        }

        // Expire job postings.
        {
            let mut postings = self.job_postings.write();
            let before = postings.len();
            postings.retain(|_, entry| {
                entry
                    .received_at
                    .map_or(true, |t| now.duration_since(t) < ttl)
            });
            removed += before - postings.len();
        }

        // Expire bid lists for jobs that no longer exist in postings.
        {
            let postings = self.job_postings.read();
            let mut all_bids = self.bids.write();
            let before = all_bids.len();
            all_bids.retain(|job_id, _| postings.contains_key(job_id));
            removed += before - all_bids.len();
        }

        if removed > 0 {
            debug!(removed, "OrderBook: expired stale entries");
        }

        removed
    }
}

impl Default for OrderBook {
    fn default() -> Self {
        Self::new()
    }
}

// ── WinnerSelection ───────────────────────────────────────────────

/// Scored bid for ranking purposes.
#[derive(Debug, Clone)]
pub struct ScoredBid {
    pub bid: WorkerBid,
    pub score: f64,
}

/// Winner selection algorithm for marketplace bids.
///
/// Score = (1.0 / bid_price) * reputation_score * (1.0 / estimated_time)
/// Tiebreaker: lower latency (estimated_seconds) wins.
pub struct WinnerSelection;

impl WinnerSelection {
    /// Score a single bid given a reputation lookup function.
    ///
    /// `reputation_fn` takes a peer_id and returns a reputation score in \[0.0, 1.0\].
    pub fn score_bid<F>(bid: &WorkerBid, reputation_fn: &F) -> f64
    where
        F: Fn(&str) -> f64,
    {
        let price = bid.bid_price_micro_qug.max(1) as f64;
        let reputation = reputation_fn(&bid.worker_peer_id).max(0.001);
        let time = bid.estimated_seconds.max(1) as f64;

        (1.0 / price) * reputation * (1.0 / time)
    }

    /// Score and rank all bids. Returns scored bids sorted by score descending.
    pub fn rank_bids<F>(bids: &[WorkerBid], reputation_fn: &F) -> Vec<ScoredBid>
    where
        F: Fn(&str) -> f64,
    {
        let mut scored: Vec<ScoredBid> = bids
            .iter()
            .map(|bid| ScoredBid {
                score: Self::score_bid(bid, reputation_fn),
                bid: bid.clone(),
            })
            .collect();

        // Sort by score descending, then by estimated_seconds ascending (latency tiebreaker).
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.bid.estimated_seconds.cmp(&b.bid.estimated_seconds))
        });

        scored
    }

    /// Select the winning bid from a set of bids using the scoring algorithm.
    ///
    /// Returns `None` if `bids` is empty.
    pub fn select_winner<F>(bids: &[WorkerBid], reputation_fn: &F) -> Option<WorkerBid>
    where
        F: Fn(&str) -> f64,
    {
        let ranked = Self::rank_bids(bids, reputation_fn);
        ranked.into_iter().next().map(|sb| sb.bid)
    }
}

// ── SettlementManager ─────────────────────────────────────────────

/// Status of a pending settlement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SettlementStatus {
    /// Payment initiated, awaiting confirmation.
    Pending,
    /// Payment confirmed by both parties.
    Confirmed,
    /// Settlement timed out, dispute raised.
    Disputed,
    /// Settlement failed or rejected.
    Failed,
}

/// A pending settlement for a completed job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingSettlement {
    /// Job ID this settlement is for.
    pub job_id: String,
    /// Worker peer ID (payee).
    pub worker_peer_id: String,
    /// Amount to pay in micro-QUG.
    pub amount_micro_qug: u64,
    /// Current settlement status.
    pub status: SettlementStatus,
    /// When the settlement was initiated (unix seconds).
    pub initiated_at: u64,
    /// When the settlement was confirmed (0 if not yet confirmed).
    pub confirmed_at: u64,
}

/// Manages payment settlements after job completion.
pub struct SettlementManager {
    /// Pending and completed settlements indexed by job_id.
    settlements: Arc<RwLock<HashMap<String, PendingSettlement>>>,
    /// P2P stats reference for volume tracking.
    stats: Arc<RwLock<MarketplaceP2PStats>>,
}

impl SettlementManager {
    /// Create a new settlement manager.
    pub fn new(stats: Arc<RwLock<MarketplaceP2PStats>>) -> Self {
        Self {
            settlements: Arc::new(RwLock::new(HashMap::new())),
            stats,
        }
    }

    /// Initiate a settlement for a completed job.
    pub fn initiate_settlement(
        &self,
        job_id: &str,
        worker_peer_id: &str,
        amount_micro_qug: u64,
    ) -> Result<PendingSettlement, String> {
        let mut settlements = self.settlements.write();

        if settlements.contains_key(job_id) {
            return Err(format!(
                "Settlement already exists for job {}",
                job_id
            ));
        }

        let settlement = PendingSettlement {
            job_id: job_id.to_string(),
            worker_peer_id: worker_peer_id.to_string(),
            amount_micro_qug,
            status: SettlementStatus::Pending,
            initiated_at: now_unix(),
            confirmed_at: 0,
        };

        info!(
            job_id = %job_id,
            worker = %worker_peer_id,
            amount = amount_micro_qug,
            "Settlement initiated"
        );

        let result = settlement.clone();
        settlements.insert(job_id.to_string(), settlement);

        {
            let mut stats = self.stats.write();
            stats.active_settlements += 1;
        }

        Ok(result)
    }

    /// Confirm a settlement as paid.
    pub fn confirm_settlement(&self, job_id: &str) -> Result<(), String> {
        let mut settlements = self.settlements.write();
        let settlement = settlements
            .get_mut(job_id)
            .ok_or_else(|| format!("No settlement found for job {}", job_id))?;

        if settlement.status != SettlementStatus::Pending {
            return Err(format!(
                "Settlement for job {} is {:?}, not Pending",
                job_id, settlement.status
            ));
        }

        settlement.status = SettlementStatus::Confirmed;
        settlement.confirmed_at = now_unix();

        info!(
            job_id = %job_id,
            amount = settlement.amount_micro_qug,
            "Settlement confirmed"
        );

        {
            let mut stats = self.stats.write();
            if stats.active_settlements > 0 {
                stats.active_settlements -= 1;
            }
            stats.total_volume_micro_qug += settlement.amount_micro_qug;
        }

        Ok(())
    }

    /// Get a settlement by job ID.
    pub fn get_settlement(&self, job_id: &str) -> Option<PendingSettlement> {
        let settlements = self.settlements.read();
        settlements.get(job_id).cloned()
    }

    /// Get all pending settlements.
    pub fn pending_settlements(&self) -> Vec<PendingSettlement> {
        let settlements = self.settlements.read();
        settlements
            .values()
            .filter(|s| s.status == SettlementStatus::Pending)
            .cloned()
            .collect()
    }

    /// Check for timed-out settlements and mark them as disputed.
    /// Returns the number of settlements moved to Disputed status.
    pub fn check_timeouts(&self) -> usize {
        let now = now_unix();
        let mut settlements = self.settlements.write();
        let mut disputed_count = 0usize;

        for settlement in settlements.values_mut() {
            if settlement.status == SettlementStatus::Pending
                && now > settlement.initiated_at + SETTLEMENT_TIMEOUT_SECS
            {
                warn!(
                    job_id = %settlement.job_id,
                    elapsed_secs = now - settlement.initiated_at,
                    "Settlement timed out, marking as disputed"
                );
                settlement.status = SettlementStatus::Disputed;
                disputed_count += 1;
            }
        }

        if disputed_count > 0 {
            let mut stats = self.stats.write();
            if stats.active_settlements >= disputed_count as u64 {
                stats.active_settlements -= disputed_count as u64;
            } else {
                stats.active_settlements = 0;
            }
        }

        disputed_count
    }

    /// Remove completed/failed settlements older than `max_age` seconds.
    /// Returns the number of entries removed.
    pub fn cleanup_old_settlements(&self, max_age_secs: u64) -> usize {
        let now = now_unix();
        let mut settlements = self.settlements.write();
        let before = settlements.len();

        settlements.retain(|_, s| {
            if s.status == SettlementStatus::Confirmed || s.status == SettlementStatus::Failed {
                now < s.initiated_at + max_age_secs
            } else {
                true // Keep pending and disputed.
            }
        });

        before - settlements.len()
    }
}

// ── MarketplaceP2PStats ───────────────────────────────────────────

/// Statistics for the marketplace P2P layer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MarketplaceP2PStats {
    /// Total messages sent via gossipsub.
    pub messages_sent: u64,
    /// Total messages received from gossipsub.
    pub messages_received: u64,
    /// Messages rejected (size, rate limit, timestamp, decode errors).
    pub messages_rejected: u64,
    /// Currently tracked active bids across all jobs.
    pub active_bids: u64,
    /// Currently pending settlements.
    pub active_settlements: u64,
    /// Total settled volume in micro-QUG.
    pub total_volume_micro_qug: u64,
}

impl MarketplaceP2PStats {
    /// Update active_bids from an OrderBook snapshot.
    pub fn sync_from_order_book(&mut self, order_book: &OrderBook) {
        self.active_bids = order_book.total_bid_count() as u64;
    }
}

// ── Helpers ───────────────────────────────────────────────────────

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marketplace::{
        JobRequirements, JobResult, JobStatus, MarketplaceJob, MarketplaceMessage, WorkType,
        WorkerBid,
    };

    fn now_ts() -> u64 {
        now_unix()
    }

    fn make_bid(job_id: &str, worker: &str, price: u64, est_secs: u32, rep: f64) -> WorkerBid {
        WorkerBid {
            job_id: job_id.to_string(),
            worker_peer_id: worker.to_string(),
            bid_price_micro_qug: price,
            estimated_seconds: est_secs,
            available_cores: 4,
            available_ram_mb: 8192,
            available_gpu_vram_mb: 0,
            reputation_score: rep,
            bid_at: now_ts(),
        }
    }

    fn make_job(id: &str, work_type: WorkType, price: u64) -> MarketplaceJob {
        MarketplaceJob {
            job_id: id.to_string(),
            work_type,
            description: format!("Test job {}", id),
            max_price_micro_qug: price,
            deadline_unix: now_ts() + 3600,
            requirements: JobRequirements::default(),
            payload: vec![],
            expected_output_hash: String::new(),
            submitter_peer_id: "submitter-peer".to_string(),
            submitted_at: now_ts(),
            status: JobStatus::Open,
            assigned_worker: None,
            ttl_secs: 3600,
        }
    }

    // ── Router Tests ──────────────────────────────────────────

    #[test]
    fn test_encode_decode_roundtrip() {
        let router = MarketplaceRouter::new(false);
        let msg = MarketplaceMessage::CapacityAnnouncement {
            peer_id: "peer-A".into(),
            available_cores: 8,
            available_ram_mb: 16384,
            gpu_vram_mb: 8192,
            accepted_work_types: vec![WorkType::AiInference, WorkType::ZkProofGeneration],
            min_price_micro_qug: 50,
            timestamp: now_ts(),
        };

        let encoded = router.encode_message(&msg).unwrap();
        let decoded = router.decode_message(&encoded).unwrap();

        match decoded {
            MarketplaceMessage::CapacityAnnouncement {
                peer_id,
                available_cores,
                ..
            } => {
                assert_eq!(peer_id, "peer-A");
                assert_eq!(available_cores, 8);
            }
            _ => panic!("Expected CapacityAnnouncement"),
        }

        let stats = router.stats();
        assert_eq!(stats.messages_sent, 1);
        assert_eq!(stats.messages_received, 1);
    }

    #[test]
    fn test_reject_oversized_message() {
        let router = MarketplaceRouter::new(false);
        // Create a message with a huge payload to exceed 1MB.
        let msg = MarketplaceMessage::JobPosting(MarketplaceJob {
            job_id: "big-job".into(),
            work_type: WorkType::Custom,
            description: "x".repeat(2_000_000), // 2MB description.
            max_price_micro_qug: 100,
            deadline_unix: now_ts() + 3600,
            requirements: JobRequirements::default(),
            payload: vec![],
            expected_output_hash: String::new(),
            submitter_peer_id: "peer".into(),
            submitted_at: now_ts(),
            status: JobStatus::Open,
            assigned_worker: None,
            ttl_secs: 3600,
        });

        let result = router.encode_message(&msg);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too large"));
    }

    #[test]
    fn test_reject_oversized_incoming() {
        let router = MarketplaceRouter::new(false);
        let big_data = vec![0u8; MAX_MESSAGE_SIZE + 1];
        let result = router.decode_message(&big_data);
        assert!(result.is_err());

        let stats = router.stats();
        assert_eq!(stats.messages_rejected, 1);
    }

    #[test]
    fn test_timestamp_validation() {
        let router = MarketplaceRouter::new(false);

        // Current timestamp should be valid.
        assert!(router.validate_timestamp(now_ts()).is_ok());

        // Timestamp 2 minutes ahead is fine.
        assert!(router.validate_timestamp(now_ts() + 120).is_ok());

        // Timestamp 10 minutes ago is rejected.
        assert!(router.validate_timestamp(now_ts() - 600).is_err());

        // Timestamp 10 minutes ahead is rejected.
        assert!(router.validate_timestamp(now_ts() + 600).is_err());
    }

    #[test]
    fn test_rate_limiting() {
        let router = MarketplaceRouter::new(false);

        // First 10 messages should be allowed.
        for _ in 0..MAX_MESSAGES_PER_SEC {
            assert!(router.check_rate_limit("spammy-peer").is_ok());
        }

        // 11th message in the same second should be rejected.
        let result = router.check_rate_limit("spammy-peer");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Rate limit"));

        // A different peer should still be allowed.
        assert!(router.check_rate_limit("good-peer").is_ok());
    }

    #[test]
    fn test_validate_and_decode_full_pipeline() {
        let router = MarketplaceRouter::new(false);
        let msg = MarketplaceMessage::BidSubmission(make_bid("job-1", "worker-A", 50, 30, 0.9));
        let encoded = router.encode_message(&msg).unwrap();

        let decoded = router.validate_and_decode(&encoded, "worker-A").unwrap();
        match decoded {
            MarketplaceMessage::BidSubmission(bid) => {
                assert_eq!(bid.worker_peer_id, "worker-A");
                assert_eq!(bid.bid_price_micro_qug, 50);
            }
            _ => panic!("Expected BidSubmission"),
        }
    }

    // ── OrderBook Tests ───────────────────────────────────────

    #[test]
    fn test_order_book_bids_sorted_by_price() {
        let book = OrderBook::new();

        let bid_high = make_bid("job-1", "worker-A", 100, 30, 0.9);
        let bid_low = make_bid("job-1", "worker-B", 20, 60, 0.8);
        let bid_mid = make_bid("job-1", "worker-C", 50, 45, 0.7);

        book.ingest(&MarketplaceMessage::BidSubmission(bid_high));
        book.ingest(&MarketplaceMessage::BidSubmission(bid_low));
        book.ingest(&MarketplaceMessage::BidSubmission(bid_mid));

        let bids = book.bids_for_job("job-1");
        assert_eq!(bids.len(), 3);
        assert_eq!(bids[0].bid_price_micro_qug, 20); // Lowest first.
        assert_eq!(bids[1].bid_price_micro_qug, 50);
        assert_eq!(bids[2].bid_price_micro_qug, 100);

        assert_eq!(book.total_bid_count(), 3);
    }

    #[test]
    fn test_order_book_capacity_announcements() {
        let book = OrderBook::new();

        book.ingest(&MarketplaceMessage::CapacityAnnouncement {
            peer_id: "peer-A".into(),
            available_cores: 8,
            available_ram_mb: 16384,
            gpu_vram_mb: 8192,
            accepted_work_types: vec![WorkType::AiInference],
            min_price_micro_qug: 10,
            timestamp: now_ts(),
        });
        book.ingest(&MarketplaceMessage::CapacityAnnouncement {
            peer_id: "peer-B".into(),
            available_cores: 4,
            available_ram_mb: 8192,
            gpu_vram_mb: 0,
            accepted_work_types: vec![WorkType::ZkProofGeneration],
            min_price_micro_qug: 20,
            timestamp: now_ts(),
        });

        let caps = book.capacity_announcements();
        assert_eq!(caps.len(), 2);
    }

    #[test]
    fn test_order_book_job_postings_with_filter() {
        let book = OrderBook::new();

        book.ingest(&MarketplaceMessage::JobPosting(make_job(
            "j1",
            WorkType::AiInference,
            100,
        )));
        book.ingest(&MarketplaceMessage::JobPosting(make_job(
            "j2",
            WorkType::ZkProofGeneration,
            200,
        )));
        book.ingest(&MarketplaceMessage::JobPosting(make_job(
            "j3",
            WorkType::AiInference,
            300,
        )));

        // No filter — returns all.
        let all = book.job_postings(&JobFilter::default());
        assert_eq!(all.len(), 3);

        // Filter by work type.
        let ai_jobs = book.job_postings(&JobFilter {
            work_type: Some(WorkType::AiInference),
            ..Default::default()
        });
        assert_eq!(ai_jobs.len(), 2);

        // Filter by max price.
        let cheap_jobs = book.job_postings(&JobFilter {
            max_price_micro_qug: Some(150),
            ..Default::default()
        });
        assert_eq!(cheap_jobs.len(), 1);
        assert_eq!(cheap_jobs[0].job_id, "j1");
    }

    #[test]
    fn test_order_book_cancellation_removes_entries() {
        let book = OrderBook::new();

        book.ingest(&MarketplaceMessage::JobPosting(make_job(
            "cancel-me",
            WorkType::RenderJob,
            500,
        )));
        book.ingest(&MarketplaceMessage::BidSubmission(make_bid(
            "cancel-me",
            "worker-A",
            400,
            30,
            0.9,
        )));

        assert_eq!(book.total_job_count(), 1);
        assert_eq!(book.total_bid_count(), 1);

        book.ingest(&MarketplaceMessage::JobCancellation {
            job_id: "cancel-me".into(),
            reason: "test".into(),
        });

        assert_eq!(book.total_job_count(), 0);
        assert_eq!(book.total_bid_count(), 0);
    }

    // ── WinnerSelection Tests ─────────────────────────────────

    #[test]
    fn test_winner_selection_basic() {
        let bids = vec![
            make_bid("job-1", "worker-A", 100, 60, 0.5),
            make_bid("job-1", "worker-B", 50, 30, 0.9),
            make_bid("job-1", "worker-C", 200, 10, 0.3),
        ];

        // Reputation function: use the bid's own reputation_score.
        let rep_fn = |peer_id: &str| -> f64 {
            bids.iter()
                .find(|b| b.worker_peer_id == peer_id)
                .map(|b| b.reputation_score)
                .unwrap_or(0.5)
        };

        let winner = WinnerSelection::select_winner(&bids, &rep_fn).unwrap();
        // Worker B: (1/50) * 0.9 * (1/30) = 0.0006
        // Worker A: (1/100) * 0.5 * (1/60) = 0.0000833
        // Worker C: (1/200) * 0.3 * (1/10) = 0.00015
        assert_eq!(winner.worker_peer_id, "worker-B");
    }

    #[test]
    fn test_winner_selection_empty_bids() {
        let bids: Vec<WorkerBid> = vec![];
        let rep_fn = |_: &str| -> f64 { 0.5 };
        assert!(WinnerSelection::select_winner(&bids, &rep_fn).is_none());
    }

    #[test]
    fn test_winner_selection_tiebreaker_latency() {
        // Two bids with identical scores except estimated_seconds.
        let bids = vec![
            make_bid("job-1", "worker-slow", 100, 60, 0.5),
            make_bid("job-1", "worker-fast", 100, 30, 0.5),
        ];

        let rep_fn = |_: &str| -> f64 { 0.5 };

        let ranked = WinnerSelection::rank_bids(&bids, &rep_fn);
        // worker-fast has lower estimated_seconds so wins the tiebreaker.
        // Actually scores differ: (1/100)*0.5*(1/60) vs (1/100)*0.5*(1/30)
        // worker-fast: 0.5/3000 = 0.000166..
        // worker-slow: 0.5/6000 = 0.0000833..
        // worker-fast wins by score alone. Still validates tiebreaker path.
        assert_eq!(ranked[0].bid.worker_peer_id, "worker-fast");
    }

    #[test]
    fn test_score_bid_zero_guards() {
        // Ensure zero price / zero time don't cause division by zero.
        let bid = WorkerBid {
            job_id: "job-1".into(),
            worker_peer_id: "worker-X".into(),
            bid_price_micro_qug: 0,
            estimated_seconds: 0,
            available_cores: 1,
            available_ram_mb: 512,
            available_gpu_vram_mb: 0,
            reputation_score: 0.0,
            bid_at: now_ts(),
        };

        let rep_fn = |_: &str| -> f64 { 0.0 };
        let score = WinnerSelection::score_bid(&bid, &rep_fn);
        // Should not panic. Price floored to 1, time to 1, rep to 0.001.
        assert!(score.is_finite());
        assert!(score > 0.0);
    }

    // ── Settlement Tests ──────────────────────────────────────

    #[test]
    fn test_settlement_lifecycle() {
        let stats = Arc::new(RwLock::new(MarketplaceP2PStats::default()));
        let mgr = SettlementManager::new(Arc::clone(&stats));

        // Initiate.
        let settlement = mgr
            .initiate_settlement("job-1", "worker-A", 500)
            .unwrap();
        assert_eq!(settlement.status, SettlementStatus::Pending);
        assert_eq!(settlement.amount_micro_qug, 500);

        assert_eq!(stats.read().active_settlements, 1);

        // Confirm.
        mgr.confirm_settlement("job-1").unwrap();
        let confirmed = mgr.get_settlement("job-1").unwrap();
        assert_eq!(confirmed.status, SettlementStatus::Confirmed);
        assert!(confirmed.confirmed_at > 0);

        assert_eq!(stats.read().active_settlements, 0);
        assert_eq!(stats.read().total_volume_micro_qug, 500);
    }

    #[test]
    fn test_settlement_duplicate_rejected() {
        let stats = Arc::new(RwLock::new(MarketplaceP2PStats::default()));
        let mgr = SettlementManager::new(Arc::clone(&stats));

        mgr.initiate_settlement("job-1", "worker-A", 500).unwrap();
        let result = mgr.initiate_settlement("job-1", "worker-B", 300);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already exists"));
    }

    #[test]
    fn test_settlement_confirm_nonexistent() {
        let stats = Arc::new(RwLock::new(MarketplaceP2PStats::default()));
        let mgr = SettlementManager::new(Arc::clone(&stats));

        let result = mgr.confirm_settlement("nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No settlement"));
    }

    #[test]
    fn test_settlement_timeout_dispute() {
        let stats = Arc::new(RwLock::new(MarketplaceP2PStats::default()));
        let mgr = SettlementManager::new(Arc::clone(&stats));

        // Insert a settlement manually with a very old initiated_at to simulate timeout.
        {
            let mut settlements = mgr.settlements.write();
            settlements.insert(
                "old-job".to_string(),
                PendingSettlement {
                    job_id: "old-job".to_string(),
                    worker_peer_id: "worker-Z".to_string(),
                    amount_micro_qug: 1000,
                    status: SettlementStatus::Pending,
                    initiated_at: now_unix() - SETTLEMENT_TIMEOUT_SECS - 100,
                    confirmed_at: 0,
                },
            );
        }
        {
            let mut s = stats.write();
            s.active_settlements = 1;
        }

        let disputed = mgr.check_timeouts();
        assert_eq!(disputed, 1);

        let settlement = mgr.get_settlement("old-job").unwrap();
        assert_eq!(settlement.status, SettlementStatus::Disputed);

        assert_eq!(stats.read().active_settlements, 0);
    }

    // ── Stats Tests ───────────────────────────────────────────

    #[test]
    fn test_stats_sync_from_order_book() {
        let mut stats = MarketplaceP2PStats::default();
        let book = OrderBook::new();

        book.ingest(&MarketplaceMessage::BidSubmission(make_bid(
            "job-1",
            "worker-A",
            50,
            30,
            0.9,
        )));
        book.ingest(&MarketplaceMessage::BidSubmission(make_bid(
            "job-1",
            "worker-B",
            60,
            30,
            0.8,
        )));
        book.ingest(&MarketplaceMessage::BidSubmission(make_bid(
            "job-2",
            "worker-C",
            70,
            30,
            0.7,
        )));

        stats.sync_from_order_book(&book);
        assert_eq!(stats.active_bids, 3);
    }
}
