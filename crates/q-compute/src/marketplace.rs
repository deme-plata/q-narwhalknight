//! # Proof-of-Useful-Work Marketplace (Issue #017)
//!
//! Replaces idle crypto (Layer 7) with revenue-generating useful work.
//! Nodes announce available capacity, clients submit jobs, and the
//! marketplace matches jobs to workers by price, capability, and reputation.
//!
//! ## Work Categories
//!
//! | Type          | Revenue Source         | Layer |
//! |---------------|------------------------|-------|
//! | AI Inference  | User-paid per token    | 1     |
//! | ZK Proofs     | dApp-paid per proof    | 2     |
//! | IPFS Pinning  | Storage-paid per GB/mo | 4     |
//! | VDF Compute   | Protocol-paid per epoch| 5     |
//! | Render Jobs   | Client-paid per frame  | 6     |
//! | Custom        | Submitter-defined      | 7     |

use crate::ComputeLayer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Maximum jobs a single node can hold in its local marketplace
const MAX_LOCAL_JOBS: usize = 10_000;
/// Default job TTL before auto-expiry
const DEFAULT_JOB_TTL_SECS: u64 = 3600; // 1 hour
/// Bid collection window
const BID_WINDOW_SECS: u64 = 5;
/// Maximum concurrent jobs per worker
const MAX_CONCURRENT_PER_WORKER: u32 = 8;

// ── Job Types ──────────────────────────────────────────────────────

/// Category of useful work
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkType {
    /// AI model inference (chat completion, embedding, etc.)
    AiInference,
    /// ZK-STARK/SNARK proof generation
    ZkProofGeneration,
    /// IPFS content pinning and serving
    IpfsPinning,
    /// Verifiable Delay Function computation
    VdfComputation,
    /// 3D rendering / video encoding
    RenderJob,
    /// Model fine-tuning / training
    ModelTraining,
    /// Custom work type (extensible)
    Custom,
}

impl WorkType {
    /// Map work type to the compute layer that should execute it
    pub fn to_layer(&self) -> ComputeLayer {
        match self {
            WorkType::AiInference => ComputeLayer::AiInference,
            WorkType::ZkProofGeneration => ComputeLayer::ZkProofGen,
            WorkType::IpfsPinning => ComputeLayer::IpfsPin,
            WorkType::VdfComputation => ComputeLayer::VdfCompute,
            WorkType::RenderJob => ComputeLayer::RenderFarm,
            WorkType::ModelTraining | WorkType::Custom => ComputeLayer::IdleCrypto,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            WorkType::AiInference => "ai_inference",
            WorkType::ZkProofGeneration => "zk_proof",
            WorkType::IpfsPinning => "ipfs_pin",
            WorkType::VdfComputation => "vdf",
            WorkType::RenderJob => "render",
            WorkType::ModelTraining => "model_training",
            WorkType::Custom => "custom",
        }
    }
}

/// Requirements for a job to be eligible on a worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRequirements {
    /// Minimum CPU cores needed
    pub min_cores: u32,
    /// Minimum RAM in MB
    pub min_ram_mb: u64,
    /// Minimum GPU VRAM in MB (0 = no GPU needed)
    pub min_gpu_vram_mb: u64,
    /// Minimum bandwidth in Mbps
    pub min_bandwidth_mbps: f64,
    /// Maximum acceptable latency to submitter in ms
    pub max_latency_ms: u32,
}

impl Default for JobRequirements {
    fn default() -> Self {
        Self {
            min_cores: 1,
            min_ram_mb: 512,
            min_gpu_vram_mb: 0,
            min_bandwidth_mbps: 1.0,
            max_latency_ms: 5000,
        }
    }
}

/// A job submitted to the marketplace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceJob {
    /// Unique job identifier
    pub job_id: String,
    /// Type of useful work
    pub work_type: WorkType,
    /// Human-readable description
    pub description: String,
    /// Maximum price the submitter will pay (in micro-QUG)
    pub max_price_micro_qug: u64,
    /// Deadline: job must complete before this unix timestamp
    pub deadline_unix: u64,
    /// Hardware/network requirements
    pub requirements: JobRequirements,
    /// Opaque payload (serialized task parameters, model name, proof inputs, etc.)
    pub payload: Vec<u8>,
    /// SHA-256 hash of the expected output (for verification, empty if unknown)
    pub expected_output_hash: String,
    /// Submitter peer ID
    pub submitter_peer_id: String,
    /// When the job was submitted (unix seconds)
    pub submitted_at: u64,
    /// Current status
    pub status: JobStatus,
    /// Assigned worker (if any)
    pub assigned_worker: Option<String>,
    /// Time-to-live in seconds (auto-expire if not picked up)
    pub ttl_secs: u64,
}

/// Status of a marketplace job
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    /// Waiting for bids
    Open,
    /// Bids collected, winner being selected
    Bidding,
    /// Assigned to a worker
    Assigned,
    /// Worker is executing
    InProgress,
    /// Completed successfully
    Completed,
    /// Failed (worker error, timeout, etc.)
    Failed,
    /// Expired (TTL exceeded)
    Expired,
    /// Cancelled by submitter
    Cancelled,
}

/// A bid from a worker for a job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerBid {
    /// The job being bid on
    pub job_id: String,
    /// Worker peer ID
    pub worker_peer_id: String,
    /// Bid price (micro-QUG) — must be <= job's max_price
    pub bid_price_micro_qug: u64,
    /// Estimated completion time in seconds
    pub estimated_seconds: u32,
    /// Worker's self-reported available resources
    pub available_cores: u32,
    pub available_ram_mb: u64,
    pub available_gpu_vram_mb: u64,
    /// Worker's reputation score (from compute_reputation module)
    pub reputation_score: f64,
    /// Unix timestamp of bid submission
    pub bid_at: u64,
}

/// Result of a completed job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    pub job_id: String,
    pub worker_peer_id: String,
    /// Output hash (SHA-256)
    pub output_hash: String,
    /// Output data (may be empty if delivered out-of-band)
    pub output_data: Vec<u8>,
    /// Actual compute time in milliseconds
    pub compute_time_ms: u64,
    /// Actual price charged (micro-QUG)
    pub price_charged_micro_qug: u64,
    /// Proof of completion (work-type-specific)
    pub proof: Vec<u8>,
    /// Unix timestamp of completion
    pub completed_at: u64,
}

// ── P2P Messages ───────────────────────────────────────────────────

/// Messages exchanged on the `/qnk/{network}/marketplace` gossipsub topic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketplaceMessage {
    /// Node announces its available capacity
    CapacityAnnouncement {
        peer_id: String,
        available_cores: u32,
        available_ram_mb: u64,
        gpu_vram_mb: u64,
        accepted_work_types: Vec<WorkType>,
        min_price_micro_qug: u64,
        timestamp: u64,
    },
    /// Client posts a job to the marketplace
    JobPosting(MarketplaceJob),
    /// Worker submits a bid for a job
    BidSubmission(WorkerBid),
    /// Job assigned to a specific worker
    JobAssignment {
        job_id: String,
        worker_peer_id: String,
        agreed_price_micro_qug: u64,
    },
    /// Worker submits completed result
    ResultSubmission(JobResult),
    /// Job cancellation
    JobCancellation {
        job_id: String,
        reason: String,
    },
}

// ── Worker Configuration ───────────────────────────────────────────

/// Node operator's configuration for what work types to accept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// Which work types this node accepts
    pub accepted_types: Vec<WorkType>,
    /// Minimum price (micro-QUG) to accept a job
    pub min_price_micro_qug: u64,
    /// Maximum concurrent jobs
    pub max_concurrent: u32,
    /// Whether to auto-bid on matching jobs
    pub auto_bid: bool,
    /// Maximum percentage of resources to dedicate to marketplace work
    pub max_resource_pct: f32,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            accepted_types: vec![
                WorkType::AiInference,
                WorkType::ZkProofGeneration,
                WorkType::VdfComputation,
            ],
            min_price_micro_qug: 10, // 10 micro-QUG minimum
            max_concurrent: MAX_CONCURRENT_PER_WORKER,
            auto_bid: true,
            max_resource_pct: 50.0, // Use up to 50% of idle resources
        }
    }
}

// ── Revenue Tracking ───────────────────────────────────────────────

/// Revenue breakdown by work type
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MarketplaceRevenue {
    /// Total revenue earned (micro-QUG)
    pub total_micro_qug: u64,
    /// Revenue per work type
    pub by_type: HashMap<String, u64>,
    /// Jobs completed count
    pub jobs_completed: u64,
    /// Jobs failed count
    pub jobs_failed: u64,
    /// Average job price (micro-QUG)
    pub avg_price_micro_qug: u64,
}

// ── Marketplace Manager ────────────────────────────────────────────

/// Internal state for a job with bidding metadata
struct JobEntry {
    job: MarketplaceJob,
    bids: Vec<WorkerBid>,
    bid_deadline: Option<Instant>,
    created: Instant,
}

/// The Marketplace Manager — matches jobs to workers
pub struct MarketplaceManager {
    /// All known jobs (local + received via P2P)
    jobs: Arc<RwLock<HashMap<String, JobEntry>>>,
    /// Worker configuration for this node
    config: Arc<RwLock<WorkerConfig>>,
    /// Revenue tracking
    revenue: Arc<RwLock<MarketplaceRevenue>>,
    /// This node's peer ID
    local_peer_id: String,
    /// Jobs currently being executed by this node
    active_jobs: Arc<RwLock<HashMap<String, MarketplaceJob>>>,
    /// Outbound message queue (to be sent via gossipsub)
    outbound: Arc<RwLock<Vec<MarketplaceMessage>>>,
}

impl MarketplaceManager {
    /// Create a new marketplace manager
    pub fn new(local_peer_id: String, config: Option<WorkerConfig>) -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            config: Arc::new(RwLock::new(config.unwrap_or_default())),
            revenue: Arc::new(RwLock::new(MarketplaceRevenue::default())),
            local_peer_id,
            active_jobs: Arc::new(RwLock::new(HashMap::new())),
            outbound: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Submit a new job to the marketplace
    pub async fn submit_job(&self, mut job: MarketplaceJob) -> Result<String, String> {
        let jobs = self.jobs.read().await;
        if jobs.len() >= MAX_LOCAL_JOBS {
            return Err("Marketplace full — too many pending jobs".into());
        }
        drop(jobs);

        // Set defaults
        if job.submitted_at == 0 {
            job.submitted_at = now_unix();
        }
        if job.ttl_secs == 0 {
            job.ttl_secs = DEFAULT_JOB_TTL_SECS;
        }
        job.status = JobStatus::Open;

        let job_id = job.job_id.clone();
        info!(job_id = %job_id, work_type = ?job.work_type, price = job.max_price_micro_qug,
              "📋 Marketplace: new job submitted");

        // Queue P2P announcement
        {
            let mut out = self.outbound.write().await;
            out.push(MarketplaceMessage::JobPosting(job.clone()));
        }

        let mut jobs = self.jobs.write().await;
        jobs.insert(job_id.clone(), JobEntry {
            job,
            bids: Vec::new(),
            bid_deadline: Some(Instant::now() + Duration::from_secs(BID_WINDOW_SECS)),
            created: Instant::now(),
        });

        Ok(job_id)
    }

    /// List available jobs (optionally filtered by work type)
    pub async fn list_available_jobs(&self, work_type: Option<WorkType>) -> Vec<MarketplaceJob> {
        let jobs = self.jobs.read().await;
        jobs.values()
            .filter(|e| e.job.status == JobStatus::Open || e.job.status == JobStatus::Bidding)
            .filter(|e| work_type.map_or(true, |wt| e.job.work_type == wt))
            .map(|e| e.job.clone())
            .collect()
    }

    /// Get a specific job by ID
    pub async fn get_job(&self, job_id: &str) -> Option<MarketplaceJob> {
        let jobs = self.jobs.read().await;
        jobs.get(job_id).map(|e| e.job.clone())
    }

    /// Cancel a job (only submitter can cancel)
    pub async fn cancel_job(&self, job_id: &str, peer_id: &str) -> Result<(), String> {
        let mut jobs = self.jobs.write().await;
        let entry = jobs.get_mut(job_id).ok_or("Job not found")?;
        if entry.job.submitter_peer_id != peer_id {
            return Err("Only the submitter can cancel a job".into());
        }
        if entry.job.status == JobStatus::Completed {
            return Err("Cannot cancel a completed job".into());
        }
        entry.job.status = JobStatus::Cancelled;
        info!(job_id = %job_id, "❌ Marketplace: job cancelled");

        // Queue P2P cancellation
        let mut out = self.outbound.write().await;
        out.push(MarketplaceMessage::JobCancellation {
            job_id: job_id.to_string(),
            reason: "Cancelled by submitter".into(),
        });
        Ok(())
    }

    /// Submit a bid for a job (called when this node wants to work on it)
    pub async fn submit_bid(&self, job_id: &str, bid_price: u64, estimated_secs: u32) -> Result<(), String> {
        let jobs = self.jobs.read().await;
        let entry = jobs.get(job_id).ok_or("Job not found")?;
        if entry.job.status != JobStatus::Open && entry.job.status != JobStatus::Bidding {
            return Err(format!("Job is {:?}, not accepting bids", entry.job.status));
        }
        if bid_price > entry.job.max_price_micro_qug {
            return Err("Bid exceeds max price".into());
        }
        drop(jobs);

        let bid = WorkerBid {
            job_id: job_id.to_string(),
            worker_peer_id: self.local_peer_id.clone(),
            bid_price_micro_qug: bid_price,
            estimated_seconds: estimated_secs,
            available_cores: num_cpus::get() as u32,
            available_ram_mb: 0, // filled by caller
            available_gpu_vram_mb: 0,
            reputation_score: 0.0,
            bid_at: now_unix(),
        };

        debug!(job_id = %job_id, price = bid_price, "🏷️ Marketplace: submitting bid");

        let mut out = self.outbound.write().await;
        out.push(MarketplaceMessage::BidSubmission(bid.clone()));
        drop(out);

        self.receive_bid(bid).await;
        Ok(())
    }

    /// Receive a bid from a peer (or self)
    pub async fn receive_bid(&self, bid: WorkerBid) {
        let mut jobs = self.jobs.write().await;
        if let Some(entry) = jobs.get_mut(&bid.job_id) {
            if entry.job.status == JobStatus::Open {
                entry.job.status = JobStatus::Bidding;
                entry.bid_deadline = Some(Instant::now() + Duration::from_secs(BID_WINDOW_SECS));
            }
            entry.bids.push(bid);
        }
    }

    /// Process bid windows and select winners. Call this periodically (~1s).
    pub async fn process_bids(&self) -> Vec<MarketplaceMessage> {
        let mut assignments = Vec::new();
        let mut jobs = self.jobs.write().await;
        let now = Instant::now();

        let job_ids: Vec<String> = jobs.keys().cloned().collect();
        for job_id in job_ids {
            let entry = match jobs.get_mut(&job_id) {
                Some(e) => e,
                None => continue,
            };

            // Check if bid window has closed
            if entry.job.status == JobStatus::Bidding {
                if let Some(deadline) = entry.bid_deadline {
                    if now >= deadline && !entry.bids.is_empty() {
                        // Select winner: lowest price, then highest reputation
                        entry.bids.sort_by(|a, b| {
                            a.bid_price_micro_qug.cmp(&b.bid_price_micro_qug)
                                .then(b.reputation_score.partial_cmp(&a.reputation_score)
                                    .unwrap_or(std::cmp::Ordering::Equal))
                        });

                        let winner = &entry.bids[0];
                        entry.job.status = JobStatus::Assigned;
                        entry.job.assigned_worker = Some(winner.worker_peer_id.clone());

                        info!(job_id = %job_id, worker = %winner.worker_peer_id,
                              price = winner.bid_price_micro_qug,
                              "🏆 Marketplace: job assigned to winner");

                        assignments.push(MarketplaceMessage::JobAssignment {
                            job_id: job_id.clone(),
                            worker_peer_id: winner.worker_peer_id.clone(),
                            agreed_price_micro_qug: winner.bid_price_micro_qug,
                        });
                    }
                }
            }
        }

        // Queue outbound messages
        if !assignments.is_empty() {
            let mut out = self.outbound.write().await;
            out.extend(assignments.clone());
        }

        assignments
    }

    /// Handle incoming P2P marketplace message
    pub async fn handle_message(&self, msg: MarketplaceMessage) {
        match msg {
            MarketplaceMessage::JobPosting(job) => {
                let should_bid = {
                    let config = self.config.read().await;
                    config.auto_bid
                        && config.accepted_types.contains(&job.work_type)
                        && job.max_price_micro_qug >= config.min_price_micro_qug
                };

                let job_id = job.job_id.clone();
                let max_price = job.max_price_micro_qug;

                // Store the job
                let mut jobs = self.jobs.write().await;
                if jobs.len() < MAX_LOCAL_JOBS {
                    jobs.insert(job_id.clone(), JobEntry {
                        job,
                        bids: Vec::new(),
                        bid_deadline: None,
                        created: Instant::now(),
                    });
                }
                drop(jobs);

                // Auto-bid if configured
                if should_bid {
                    let active = self.active_jobs.read().await;
                    let config = self.config.read().await;
                    if (active.len() as u32) < config.max_concurrent {
                        drop(config);
                        drop(active);
                        let _ = self.submit_bid(&job_id, max_price, 60).await;
                    }
                }
            }
            MarketplaceMessage::BidSubmission(bid) => {
                self.receive_bid(bid).await;
            }
            MarketplaceMessage::JobAssignment { job_id, worker_peer_id, agreed_price_micro_qug } => {
                let mut jobs = self.jobs.write().await;
                if let Some(entry) = jobs.get_mut(&job_id) {
                    entry.job.status = JobStatus::Assigned;
                    entry.job.assigned_worker = Some(worker_peer_id.clone());
                }
                drop(jobs);

                // If we're the assigned worker, start working
                if worker_peer_id == self.local_peer_id {
                    info!(job_id = %job_id, price = agreed_price_micro_qug,
                          "🎯 Marketplace: we won the bid! Starting work...");
                    if let Some(job) = self.get_job(&job_id).await {
                        let mut active = self.active_jobs.write().await;
                        active.insert(job_id, job);
                    }
                }
            }
            MarketplaceMessage::ResultSubmission(result) => {
                let mut jobs = self.jobs.write().await;
                if let Some(entry) = jobs.get_mut(&result.job_id) {
                    entry.job.status = JobStatus::Completed;
                    info!(job_id = %result.job_id, worker = %result.worker_peer_id,
                          price = result.price_charged_micro_qug,
                          "✅ Marketplace: job completed");
                }
                drop(jobs);

                // Track revenue if we were the worker
                if result.worker_peer_id == self.local_peer_id {
                    let mut rev = self.revenue.write().await;
                    rev.total_micro_qug += result.price_charged_micro_qug;
                    rev.jobs_completed += 1;

                    // Remove from active
                    let mut active = self.active_jobs.write().await;
                    active.remove(&result.job_id);
                }
            }
            MarketplaceMessage::JobCancellation { job_id, reason } => {
                let mut jobs = self.jobs.write().await;
                if let Some(entry) = jobs.get_mut(&job_id) {
                    entry.job.status = JobStatus::Cancelled;
                    debug!(job_id = %job_id, reason = %reason, "Marketplace: job cancelled");
                }
                // Remove from active if we were working on it
                let mut active = self.active_jobs.write().await;
                active.remove(&job_id);
            }
            MarketplaceMessage::CapacityAnnouncement { .. } => {
                // Capacity announcements are informational — used for bid routing
            }
        }
    }

    /// Expire old jobs. Call periodically (~30s).
    pub async fn expire_stale_jobs(&self) -> usize {
        let now_ts = now_unix();
        let mut jobs = self.jobs.write().await;
        let before = jobs.len();

        jobs.retain(|_, entry| {
            if entry.job.status == JobStatus::Completed
                || entry.job.status == JobStatus::Cancelled
                || entry.job.status == JobStatus::Failed
            {
                // Keep completed/cancelled/failed for 1 hour for audit
                return entry.created.elapsed() < Duration::from_secs(3600);
            }
            // Expire open/bidding jobs past TTL
            if entry.job.status == JobStatus::Open || entry.job.status == JobStatus::Bidding {
                if now_ts > entry.job.submitted_at + entry.job.ttl_secs {
                    entry.job.status = JobStatus::Expired;
                    return false;
                }
            }
            true
        });

        let expired = before - jobs.len();
        if expired > 0 {
            debug!(expired, "🧹 Marketplace: expired stale jobs");
        }
        expired
    }

    /// Drain outbound messages (for gossipsub publishing)
    pub async fn drain_outbound(&self) -> Vec<MarketplaceMessage> {
        let mut out = self.outbound.write().await;
        std::mem::take(&mut *out)
    }

    /// Get revenue summary
    pub async fn get_revenue(&self) -> MarketplaceRevenue {
        self.revenue.read().await.clone()
    }

    /// Get marketplace stats
    pub async fn get_stats(&self) -> MarketplaceStats {
        let jobs = self.jobs.read().await;
        let active = self.active_jobs.read().await;
        let revenue = self.revenue.read().await;

        let mut open = 0u32;
        let mut bidding = 0u32;
        let mut assigned = 0u32;
        let mut completed = 0u32;
        let mut failed = 0u32;

        for entry in jobs.values() {
            match entry.job.status {
                JobStatus::Open => open += 1,
                JobStatus::Bidding => bidding += 1,
                JobStatus::Assigned | JobStatus::InProgress => assigned += 1,
                JobStatus::Completed => completed += 1,
                JobStatus::Failed => failed += 1,
                _ => {}
            }
        }

        MarketplaceStats {
            total_jobs: jobs.len() as u32,
            open_jobs: open,
            bidding_jobs: bidding,
            assigned_jobs: assigned,
            completed_jobs: completed,
            failed_jobs: failed,
            active_local_jobs: active.len() as u32,
            total_revenue_micro_qug: revenue.total_micro_qug,
        }
    }

    /// Update worker config at runtime
    pub async fn set_config(&self, config: WorkerConfig) {
        *self.config.write().await = config;
    }

    /// Get current worker config
    pub async fn get_config(&self) -> WorkerConfig {
        self.config.read().await.clone()
    }
}

/// Summary statistics for the marketplace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceStats {
    pub total_jobs: u32,
    pub open_jobs: u32,
    pub bidding_jobs: u32,
    pub assigned_jobs: u32,
    pub completed_jobs: u32,
    pub failed_jobs: u32,
    pub active_local_jobs: u32,
    pub total_revenue_micro_qug: u64,
}

/// Gossipsub topic for marketplace messages
pub fn marketplace_topic(network_id: &str) -> String {
    format!("/qnk/{}/marketplace", network_id)
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_job(id: &str, work_type: WorkType, price: u64) -> MarketplaceJob {
        MarketplaceJob {
            job_id: id.to_string(),
            work_type,
            description: format!("Test job {}", id),
            max_price_micro_qug: price,
            deadline_unix: now_unix() + 3600,
            requirements: JobRequirements::default(),
            payload: vec![],
            expected_output_hash: String::new(),
            submitter_peer_id: "submitter-peer".to_string(),
            submitted_at: now_unix(),
            status: JobStatus::Open,
            assigned_worker: None,
            ttl_secs: DEFAULT_JOB_TTL_SECS,
        }
    }

    #[tokio::test]
    async fn test_submit_and_list_jobs() {
        let mgr = MarketplaceManager::new("local-peer".into(), None);
        let job = make_job("job-1", WorkType::AiInference, 100);
        mgr.submit_job(job).await.unwrap();

        let available = mgr.list_available_jobs(None).await;
        assert_eq!(available.len(), 1);
        assert_eq!(available[0].job_id, "job-1");
    }

    #[tokio::test]
    async fn test_filter_by_work_type() {
        let mgr = MarketplaceManager::new("local-peer".into(), None);
        mgr.submit_job(make_job("j1", WorkType::AiInference, 100)).await.unwrap();
        mgr.submit_job(make_job("j2", WorkType::ZkProofGeneration, 200)).await.unwrap();
        mgr.submit_job(make_job("j3", WorkType::AiInference, 150)).await.unwrap();

        let ai = mgr.list_available_jobs(Some(WorkType::AiInference)).await;
        assert_eq!(ai.len(), 2);

        let zk = mgr.list_available_jobs(Some(WorkType::ZkProofGeneration)).await;
        assert_eq!(zk.len(), 1);
    }

    #[tokio::test]
    async fn test_submit_bid_and_process() {
        let mgr = MarketplaceManager::new("local-peer".into(), None);
        let job = make_job("bid-job", WorkType::AiInference, 100);
        mgr.submit_job(job).await.unwrap();

        // Submit two bids
        let bid1 = WorkerBid {
            job_id: "bid-job".into(),
            worker_peer_id: "worker-A".into(),
            bid_price_micro_qug: 80,
            estimated_seconds: 30,
            available_cores: 4,
            available_ram_mb: 8192,
            available_gpu_vram_mb: 0,
            reputation_score: 0.9,
            bid_at: now_unix(),
        };
        let bid2 = WorkerBid {
            job_id: "bid-job".into(),
            worker_peer_id: "worker-B".into(),
            bid_price_micro_qug: 50, // Lower price = winner
            estimated_seconds: 60,
            available_cores: 2,
            available_ram_mb: 4096,
            available_gpu_vram_mb: 0,
            reputation_score: 0.8,
            bid_at: now_unix(),
        };

        mgr.receive_bid(bid1).await;
        mgr.receive_bid(bid2).await;

        // Force bid deadline to past
        {
            let mut jobs = mgr.jobs.write().await;
            let entry = jobs.get_mut("bid-job").unwrap();
            entry.bid_deadline = Some(Instant::now() - Duration::from_secs(1));
        }

        let assignments = mgr.process_bids().await;
        assert_eq!(assignments.len(), 1);
        match &assignments[0] {
            MarketplaceMessage::JobAssignment { worker_peer_id, agreed_price_micro_qug, .. } => {
                assert_eq!(worker_peer_id, "worker-B"); // Lowest price
                assert_eq!(*agreed_price_micro_qug, 50);
            }
            _ => panic!("Expected JobAssignment"),
        }
    }

    #[tokio::test]
    async fn test_cancel_job() {
        let mgr = MarketplaceManager::new("local-peer".into(), None);
        let job = make_job("cancel-me", WorkType::RenderJob, 500);
        mgr.submit_job(job).await.unwrap();

        // Wrong submitter can't cancel
        let err = mgr.cancel_job("cancel-me", "wrong-peer").await;
        assert!(err.is_err());

        // Correct submitter can cancel
        mgr.cancel_job("cancel-me", "submitter-peer").await.unwrap();
        let j = mgr.get_job("cancel-me").await.unwrap();
        assert_eq!(j.status, JobStatus::Cancelled);
    }

    #[tokio::test]
    async fn test_bid_price_validation() {
        let mgr = MarketplaceManager::new("local-peer".into(), None);
        let job = make_job("cheap-job", WorkType::VdfComputation, 10);
        mgr.submit_job(job).await.unwrap();

        // Bid above max price should fail
        let err = mgr.submit_bid("cheap-job", 20, 30).await;
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("max price"));
    }

    #[tokio::test]
    async fn test_handle_result_tracks_revenue() {
        let mgr = MarketplaceManager::new("local-peer".into(), None);
        let job = make_job("rev-job", WorkType::AiInference, 100);
        mgr.submit_job(job).await.unwrap();

        // Simulate assignment to us
        mgr.handle_message(MarketplaceMessage::JobAssignment {
            job_id: "rev-job".into(),
            worker_peer_id: "local-peer".into(),
            agreed_price_micro_qug: 75,
        }).await;

        // Simulate completion
        mgr.handle_message(MarketplaceMessage::ResultSubmission(JobResult {
            job_id: "rev-job".into(),
            worker_peer_id: "local-peer".into(),
            output_hash: "abc123".into(),
            output_data: vec![],
            compute_time_ms: 500,
            price_charged_micro_qug: 75,
            proof: vec![],
            completed_at: now_unix(),
        })).await;

        let rev = mgr.get_revenue().await;
        assert_eq!(rev.total_micro_qug, 75);
        assert_eq!(rev.jobs_completed, 1);
    }

    #[tokio::test]
    async fn test_stats() {
        let mgr = MarketplaceManager::new("local-peer".into(), None);
        mgr.submit_job(make_job("s1", WorkType::AiInference, 100)).await.unwrap();
        mgr.submit_job(make_job("s2", WorkType::ZkProofGeneration, 200)).await.unwrap();

        let stats = mgr.get_stats().await;
        assert_eq!(stats.total_jobs, 2);
        assert_eq!(stats.open_jobs, 2);
    }

    #[tokio::test]
    async fn test_work_type_to_layer() {
        assert_eq!(WorkType::AiInference.to_layer(), ComputeLayer::AiInference);
        assert_eq!(WorkType::ZkProofGeneration.to_layer(), ComputeLayer::ZkProofGen);
        assert_eq!(WorkType::IpfsPinning.to_layer(), ComputeLayer::IpfsPin);
        assert_eq!(WorkType::VdfComputation.to_layer(), ComputeLayer::VdfCompute);
        assert_eq!(WorkType::RenderJob.to_layer(), ComputeLayer::RenderFarm);
        assert_eq!(WorkType::Custom.to_layer(), ComputeLayer::IdleCrypto);
    }

    #[tokio::test]
    async fn test_drain_outbound() {
        let mgr = MarketplaceManager::new("local-peer".into(), None);
        mgr.submit_job(make_job("out-1", WorkType::AiInference, 100)).await.unwrap();

        let msgs = mgr.drain_outbound().await;
        assert_eq!(msgs.len(), 1);
        match &msgs[0] {
            MarketplaceMessage::JobPosting(job) => assert_eq!(job.job_id, "out-1"),
            _ => panic!("Expected JobPosting"),
        }

        // Second drain should be empty
        let msgs2 = mgr.drain_outbound().await;
        assert!(msgs2.is_empty());
    }

    #[tokio::test]
    async fn test_marketplace_capacity() {
        let mgr = MarketplaceManager::new("local-peer".into(), None);
        // Submit up to limit (test with a small number)
        for i in 0..100 {
            let _ = mgr.submit_job(make_job(&format!("cap-{}", i), WorkType::Custom, 10)).await;
        }
        let stats = mgr.get_stats().await;
        assert_eq!(stats.total_jobs, 100);
    }

    #[tokio::test]
    async fn test_config_update() {
        let mgr = MarketplaceManager::new("local-peer".into(), None);
        let mut config = mgr.get_config().await;
        assert!(config.auto_bid);

        config.auto_bid = false;
        config.min_price_micro_qug = 1000;
        mgr.set_config(config).await;

        let updated = mgr.get_config().await;
        assert!(!updated.auto_bid);
        assert_eq!(updated.min_price_micro_qug, 1000);
    }
}
