//! # Compute Job Queue Persistence (WAL)
//!
//! Write-ahead log for compute jobs so they survive node restarts.
//!
//! ## Design
//!
//! - Append-only file, line-delimited JSON (one WAL entry per line)
//! - On startup, the entire WAL is replayed to rebuild in-memory state
//! - Jobs stuck `InProgress` for >5min are reset to `Queued` (max 3 retries)
//! - Compaction removes `Settled`/`Failed` jobs older than 24h
//!
//! ## State Machine
//!
//! ```text
//! Queued -> InProgress -> Completed -> Settled
//!                     \-> Failed    -> Settled
//!
//! InProgress (>5min timeout) -> Queued (up to 3 retries, then Failed)
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default WAL file path relative to the node's data directory.
pub const DEFAULT_WAL_PATH: &str = "data/compute-jobs.wal";

/// Jobs stuck InProgress longer than this are timed out (seconds).
const TIMEOUT_SECS: u64 = 300; // 5 minutes

/// Maximum number of times a timed-out job is re-queued before failing.
const MAX_RETRIES: u32 = 3;

/// Compaction removes Settled/Failed jobs older than this (seconds).
const COMPACTION_AGE_SECS: u64 = 86_400; // 24 hours

// ---------------------------------------------------------------------------
// Job types & status
// ---------------------------------------------------------------------------

/// The kind of compute work a job represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JobType {
    Mining,
    Inference,
    ZkProof,
    BridgeVerify,
}

impl std::fmt::Display for JobType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobType::Mining => write!(f, "Mining"),
            JobType::Inference => write!(f, "Inference"),
            JobType::ZkProof => write!(f, "ZkProof"),
            JobType::BridgeVerify => write!(f, "BridgeVerify"),
        }
    }
}

/// Lifecycle status of a compute job.
///
/// ```text
/// Queued -> InProgress -> Completed -> Settled
///                     \-> Failed    -> Settled
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JobStatus {
    /// Waiting to be picked up by a worker.
    Queued,
    /// Currently being executed.
    InProgress,
    /// Finished successfully.
    Completed,
    /// Failed after exhausting retries or encountering a fatal error.
    Failed,
    /// Terminal state -- acknowledged and eligible for compaction.
    Settled,
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobStatus::Queued => write!(f, "Queued"),
            JobStatus::InProgress => write!(f, "InProgress"),
            JobStatus::Completed => write!(f, "Completed"),
            JobStatus::Failed => write!(f, "Failed"),
            JobStatus::Settled => write!(f, "Settled"),
        }
    }
}

impl JobStatus {
    /// Returns `true` if transitioning from `self` to `next` is valid.
    pub fn can_transition_to(self, next: JobStatus) -> bool {
        matches!(
            (self, next),
            (JobStatus::Queued, JobStatus::InProgress)
                | (JobStatus::InProgress, JobStatus::Completed)
                | (JobStatus::InProgress, JobStatus::Failed)
                | (JobStatus::InProgress, JobStatus::Queued) // timeout retry
                | (JobStatus::Completed, JobStatus::Settled)
                | (JobStatus::Failed, JobStatus::Settled)
        )
    }
}

// ---------------------------------------------------------------------------
// Job record
// ---------------------------------------------------------------------------

/// Unique job identifier. Wrapper around a UUID v4 string for minimal deps.
/// Format: "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx"
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(pub String);

impl JobId {
    /// Generate a new random job ID (UUID v4 format).
    pub fn new() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut bytes = [0u8; 16];
        rng.fill(&mut bytes);
        // Set version (4) and variant (RFC 4122).
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        let s = format!(
            "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
            u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            u16::from_be_bytes([bytes[4], bytes[5]]),
            u16::from_be_bytes([bytes[6], bytes[7]]),
            u16::from_be_bytes([bytes[8], bytes[9]]),
            u64::from_be_bytes([0, 0, bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]]),
        );
        JobId(s)
    }

    /// Create a JobId from an existing string (e.g. recovered from WAL).
    pub fn from_string(s: impl Into<String>) -> Self {
        JobId(s.into())
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for JobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A single compute job record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    /// Unique identifier for this job.
    pub job_id: JobId,
    /// What kind of compute work this job does.
    pub job_type: JobType,
    /// Unix timestamp (seconds) when the job was submitted.
    pub submitted_at: u64,
    /// Unix timestamp (seconds) when execution started (0 if not started).
    pub started_at: u64,
    /// Unix timestamp (seconds) when execution finished (0 if not finished).
    pub completed_at: u64,
    /// Current lifecycle status.
    pub status: JobStatus,
    /// SHA-256 hash of the job payload (hex string). Used for dedup/verification.
    pub payload_hash: String,
    /// Peer ID of the node/miner assigned to execute this job (empty if unassigned).
    pub assigned_peer: String,
    /// Revenue earned from this job in micro-QUG (1 QUG = 1_000_000 micro-QUG).
    pub revenue_micro_qug: u64,
    /// Number of times this job has been retried after timeout.
    pub retry_count: u32,
}

impl JobRecord {
    /// Create a new job in `Queued` status.
    pub fn new(job_type: JobType, payload_hash: String) -> Self {
        Self {
            job_id: JobId::new(),
            job_type,
            submitted_at: now_secs(),
            started_at: 0,
            completed_at: 0,
            status: JobStatus::Queued,
            payload_hash,
            assigned_peer: String::new(),
            revenue_micro_qug: 0,
            retry_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// WAL entry (on-disk format)
// ---------------------------------------------------------------------------

/// Operation type written to the WAL.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op")]
enum WalEntry {
    /// A new job was appended.
    #[serde(rename = "append")]
    Append {
        record: JobRecord,
        timestamp: u64,
    },
    /// An existing job's status (and possibly other fields) was updated.
    #[serde(rename = "update")]
    Update {
        record: JobRecord,
        timestamp: u64,
    },
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Aggregate statistics over all tracked jobs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JobWalStats {
    pub total_jobs: u64,
    pub queued: u64,
    pub in_progress: u64,
    pub completed: u64,
    pub failed: u64,
    pub settled: u64,
}

// ---------------------------------------------------------------------------
// JobWal
// ---------------------------------------------------------------------------

/// Write-ahead log backed by an append-only file with in-memory index.
///
/// Thread-safe via `tokio::sync::RwLock` -- all public methods are `async`.
pub struct JobWal {
    /// Path to the WAL file on disk.
    wal_path: PathBuf,
    /// In-memory state rebuilt from WAL replay.
    state: RwLock<WalState>,
}

struct WalState {
    jobs: HashMap<String, JobRecord>,
}

impl WalState {
    fn new() -> Self {
        Self {
            jobs: HashMap::new(),
        }
    }

    fn compute_stats(&self) -> JobWalStats {
        let mut stats = JobWalStats::default();
        for job in self.jobs.values() {
            stats.total_jobs += 1;
            match job.status {
                JobStatus::Queued => stats.queued += 1,
                JobStatus::InProgress => stats.in_progress += 1,
                JobStatus::Completed => stats.completed += 1,
                JobStatus::Failed => stats.failed += 1,
                JobStatus::Settled => stats.settled += 1,
            }
        }
        stats
    }
}

impl JobWal {
    /// Open (or create) a WAL at the given path.
    ///
    /// If the file already exists the WAL is replayed to reconstruct in-memory
    /// state. Jobs that were `InProgress` at the time of the crash are reset to
    /// `Queued` (respecting retry limits).
    pub async fn open(path: impl AsRef<Path>) -> Result<Self, WalError> {
        let wal_path = path.as_ref().to_path_buf();

        // Ensure parent directory exists.
        if let Some(parent) = wal_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| WalError::Io(e.to_string()))?;
        }

        let mut state = WalState::new();

        // Replay existing WAL file if present.
        if wal_path.exists() {
            let contents =
                std::fs::read_to_string(&wal_path).map_err(|e| WalError::Io(e.to_string()))?;
            let mut replayed = 0u64;
            let mut errors = 0u64;
            for line in contents.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<WalEntry>(line) {
                    Ok(entry) => {
                        match entry {
                            WalEntry::Append { record, .. } | WalEntry::Update { record, .. } => {
                                state.jobs.insert(record.job_id.0.clone(), record);
                            }
                        }
                        replayed += 1;
                    }
                    Err(e) => {
                        errors += 1;
                        warn!("WAL replay: skipping corrupt line: {e}");
                    }
                }
            }
            info!(
                "WAL replayed {replayed} entries ({errors} corrupt) from {}",
                wal_path.display()
            );
        }

        // Recovery: reset InProgress jobs back to Queued (or Failed if retries exhausted).
        let now = now_secs();
        let job_ids: Vec<String> = state.jobs.keys().cloned().collect();
        for id in job_ids {
            if let Some(job) = state.jobs.get_mut(&id) {
                if job.status == JobStatus::InProgress {
                    if job.retry_count >= MAX_RETRIES {
                        info!(
                            "WAL recovery: job {} exceeded max retries ({}), marking Failed",
                            job.job_id, job.retry_count
                        );
                        job.status = JobStatus::Failed;
                        job.completed_at = now;
                    } else {
                        info!(
                            "WAL recovery: re-queuing InProgress job {} (retry {}/{})",
                            job.job_id,
                            job.retry_count + 1,
                            MAX_RETRIES
                        );
                        job.status = JobStatus::Queued;
                        job.retry_count += 1;
                        job.started_at = 0;
                        job.assigned_peer = String::new();
                    }
                }
            }
        }

        let wal = Self {
            wal_path,
            state: RwLock::new(state),
        };

        // Persist recovery changes.
        wal.rewrite_wal().await?;

        info!("JobWal opened at {}", wal.wal_path.display());
        Ok(wal)
    }

    /// Open a WAL at the default path.
    pub async fn open_default() -> Result<Self, WalError> {
        Self::open(DEFAULT_WAL_PATH).await
    }

    // -----------------------------------------------------------------------
    // Write operations
    // -----------------------------------------------------------------------

    /// Append a new job to the WAL. Returns the assigned `JobId`.
    pub async fn append_job(&self, job: JobRecord) -> Result<JobId, WalError> {
        let id = job.job_id.clone();
        let entry = WalEntry::Append {
            timestamp: now_secs(),
            record: job.clone(),
        };
        self.append_entry(&entry)?;

        let mut state = self.state.write().await;
        state.jobs.insert(id.0.clone(), job);
        debug!("WAL: appended job {id}");
        Ok(id)
    }

    /// Transition a job to a new status.
    ///
    /// Validates the state machine transition. Updates relevant timestamps and
    /// fields on the record.
    pub async fn update_status(
        &self,
        job_id: &JobId,
        new_status: JobStatus,
        assigned_peer: Option<String>,
        revenue_micro_qug: Option<u64>,
    ) -> Result<(), WalError> {
        let mut state = self.state.write().await;
        let job = state
            .jobs
            .get_mut(&job_id.0)
            .ok_or_else(|| WalError::NotFound(job_id.0.clone()))?;

        if !job.status.can_transition_to(new_status) {
            return Err(WalError::InvalidTransition {
                job_id: job_id.0.clone(),
                from: job.status,
                to: new_status,
            });
        }

        let now = now_secs();
        job.status = new_status;
        match new_status {
            JobStatus::InProgress => {
                job.started_at = now;
                if let Some(peer) = assigned_peer {
                    job.assigned_peer = peer;
                }
            }
            JobStatus::Completed => {
                job.completed_at = now;
                if let Some(rev) = revenue_micro_qug {
                    job.revenue_micro_qug = rev;
                }
            }
            JobStatus::Failed => {
                job.completed_at = now;
            }
            JobStatus::Queued => {
                // timeout-retry
                job.started_at = 0;
                job.assigned_peer = String::new();
            }
            JobStatus::Settled => {
                // no extra fields
            }
        }

        let entry = WalEntry::Update {
            timestamp: now,
            record: job.clone(),
        };
        self.append_entry(&entry)?;

        debug!("WAL: job {} -> {new_status}", job_id);
        Ok(())
    }

    /// Check for timed-out `InProgress` jobs and reset them.
    ///
    /// Returns the number of jobs that were reset.
    pub async fn check_timeouts(&self) -> Result<u32, WalError> {
        let now = now_secs();
        let mut state = self.state.write().await;

        let timed_out: Vec<String> = state
            .jobs
            .values()
            .filter(|j| {
                j.status == JobStatus::InProgress
                    && j.started_at > 0
                    && now.saturating_sub(j.started_at) > TIMEOUT_SECS
            })
            .map(|j| j.job_id.0.clone())
            .collect();

        let mut reset_count = 0u32;
        for id in &timed_out {
            if let Some(job) = state.jobs.get_mut(id) {
                if job.retry_count >= MAX_RETRIES {
                    warn!(
                        "WAL: job {} timed out, retries exhausted ({}/{}), marking Failed",
                        id, job.retry_count, MAX_RETRIES
                    );
                    job.status = JobStatus::Failed;
                    job.completed_at = now;
                } else {
                    warn!(
                        "WAL: job {} timed out after {}s, re-queuing (retry {}/{})",
                        id,
                        now.saturating_sub(job.started_at),
                        job.retry_count + 1,
                        MAX_RETRIES
                    );
                    job.status = JobStatus::Queued;
                    job.retry_count += 1;
                    job.started_at = 0;
                    job.assigned_peer = String::new();
                }

                let entry = WalEntry::Update {
                    timestamp: now,
                    record: job.clone(),
                };
                self.append_entry(&entry)?;
                reset_count += 1;
            }
        }

        if reset_count > 0 {
            info!("WAL: timeout check reset {reset_count} jobs");
        }
        Ok(reset_count)
    }

    /// Compact the WAL by removing `Settled` and `Failed` jobs older than 24h.
    ///
    /// This rewrites the WAL file with only the remaining jobs, reducing disk
    /// usage over time.
    pub async fn compact(&self) -> Result<u64, WalError> {
        let now = now_secs();
        let mut state = self.state.write().await;

        let before = state.jobs.len() as u64;

        state.jobs.retain(|_id, job| {
            let is_terminal = matches!(job.status, JobStatus::Settled | JobStatus::Failed);
            let old_enough = job.completed_at > 0
                && now.saturating_sub(job.completed_at) > COMPACTION_AGE_SECS;
            !(is_terminal && old_enough)
        });

        let removed = before.saturating_sub(state.jobs.len() as u64);

        // Rewrite WAL with remaining jobs.
        drop(state);
        self.rewrite_wal().await?;

        if removed > 0 {
            info!("WAL compacted: removed {removed} old jobs");
        }
        Ok(removed)
    }

    // -----------------------------------------------------------------------
    // Query operations
    // -----------------------------------------------------------------------

    /// Get a single job by ID.
    pub async fn get_job(&self, job_id: &JobId) -> Option<JobRecord> {
        let state = self.state.read().await;
        state.jobs.get(&job_id.0).cloned()
    }

    /// List all jobs, optionally filtered by status and/or type.
    pub async fn list_jobs(
        &self,
        status_filter: Option<JobStatus>,
        type_filter: Option<JobType>,
    ) -> Vec<JobRecord> {
        let state = self.state.read().await;
        state
            .jobs
            .values()
            .filter(|j| {
                status_filter.map_or(true, |s| j.status == s)
                    && type_filter.map_or(true, |t| j.job_type == t)
            })
            .cloned()
            .collect()
    }

    /// Get aggregate stats.
    pub async fn get_stats(&self) -> JobWalStats {
        let state = self.state.read().await;
        state.compute_stats()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Append a single JSON line to the WAL file (synchronous I/O on the file,
    /// but called from async context -- file appends are fast enough that
    /// blocking briefly is acceptable for durability).
    fn append_entry(&self, entry: &WalEntry) -> Result<(), WalError> {
        use std::io::Write;
        let line = serde_json::to_string(entry).map_err(|e| WalError::Serialize(e.to_string()))?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.wal_path)
            .map_err(|e| WalError::Io(e.to_string()))?;
        writeln!(file, "{}", line).map_err(|e| WalError::Io(e.to_string()))?;
        file.flush().map_err(|e| WalError::Io(e.to_string()))?;
        Ok(())
    }

    /// Rewrite the WAL file from the current in-memory state.
    ///
    /// Used after recovery and compaction. Writes to a tmp file then renames
    /// for atomicity.
    async fn rewrite_wal(&self) -> Result<(), WalError> {
        let state = self.state.read().await;
        let tmp_path = self.wal_path.with_extension("wal.tmp");

        {
            use std::io::Write;
            let mut file =
                std::fs::File::create(&tmp_path).map_err(|e| WalError::Io(e.to_string()))?;
            let now = now_secs();
            for job in state.jobs.values() {
                let entry = WalEntry::Append {
                    timestamp: now,
                    record: job.clone(),
                };
                let line = serde_json::to_string(&entry)
                    .map_err(|e| WalError::Serialize(e.to_string()))?;
                writeln!(file, "{}", line).map_err(|e| WalError::Io(e.to_string()))?;
            }
            file.flush().map_err(|e| WalError::Io(e.to_string()))?;
            file.sync_all().map_err(|e| WalError::Io(e.to_string()))?;
        }

        std::fs::rename(&tmp_path, &self.wal_path)
            .map_err(|e| WalError::Io(e.to_string()))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during WAL operations.
#[derive(Debug, Clone)]
pub enum WalError {
    /// File I/O error.
    Io(String),
    /// Serialization/deserialization error.
    Serialize(String),
    /// Job not found.
    NotFound(String),
    /// Invalid state transition.
    InvalidTransition {
        job_id: String,
        from: JobStatus,
        to: JobStatus,
    },
}

impl std::fmt::Display for WalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WalError::Io(e) => write!(f, "WAL I/O error: {e}"),
            WalError::Serialize(e) => write!(f, "WAL serialization error: {e}"),
            WalError::NotFound(id) => write!(f, "Job not found: {id}"),
            WalError::InvalidTransition { job_id, from, to } => {
                write!(f, "Invalid transition for job {job_id}: {from} -> {to}")
            }
        }
    }
}

impl std::error::Error for WalError {}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Helper: create a temporary WAL in a unique directory.
    async fn temp_wal(name: &str) -> (JobWal, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "q-compute-wal-test-{}-{}",
            name,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.wal");
        let wal = JobWal::open(&path).await.unwrap();
        (wal, dir)
    }

    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    // -----------------------------------------------------------------------
    // Test 1: Append a job and retrieve it
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_append_and_get() {
        let (wal, dir) = temp_wal("append_get").await;
        let job = JobRecord::new(JobType::Mining, "abc123".into());
        let id = wal.append_job(job).await.unwrap();

        let retrieved = wal.get_job(&id).await.unwrap();
        assert_eq!(retrieved.job_type, JobType::Mining);
        assert_eq!(retrieved.status, JobStatus::Queued);
        assert_eq!(retrieved.payload_hash, "abc123");

        cleanup(&dir);
    }

    // -----------------------------------------------------------------------
    // Test 2: Valid status transitions (full lifecycle)
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_status_transitions_valid() {
        let (wal, dir) = temp_wal("transitions_valid").await;
        let job = JobRecord::new(JobType::Inference, "hash1".into());
        let id = wal.append_job(job).await.unwrap();

        // Queued -> InProgress
        wal.update_status(&id, JobStatus::InProgress, Some("peer-A".into()), None)
            .await
            .unwrap();
        let j = wal.get_job(&id).await.unwrap();
        assert_eq!(j.status, JobStatus::InProgress);
        assert_eq!(j.assigned_peer, "peer-A");
        assert!(j.started_at > 0);

        // InProgress -> Completed
        wal.update_status(&id, JobStatus::Completed, None, Some(5000))
            .await
            .unwrap();
        let j = wal.get_job(&id).await.unwrap();
        assert_eq!(j.status, JobStatus::Completed);
        assert_eq!(j.revenue_micro_qug, 5000);
        assert!(j.completed_at > 0);

        // Completed -> Settled
        wal.update_status(&id, JobStatus::Settled, None, None)
            .await
            .unwrap();
        let j = wal.get_job(&id).await.unwrap();
        assert_eq!(j.status, JobStatus::Settled);

        cleanup(&dir);
    }

    // -----------------------------------------------------------------------
    // Test 3: Invalid status transition is rejected
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_status_transition_invalid() {
        let (wal, dir) = temp_wal("transitions_invalid").await;
        let job = JobRecord::new(JobType::ZkProof, "hash2".into());
        let id = wal.append_job(job).await.unwrap();

        // Queued -> Completed should fail (must go through InProgress first)
        let result = wal
            .update_status(&id, JobStatus::Completed, None, None)
            .await;
        assert!(result.is_err());
        match result.unwrap_err() {
            WalError::InvalidTransition { from, to, .. } => {
                assert_eq!(from, JobStatus::Queued);
                assert_eq!(to, JobStatus::Completed);
            }
            other => panic!("Expected InvalidTransition, got: {other}"),
        }

        cleanup(&dir);
    }

    // -----------------------------------------------------------------------
    // Test 4: Recovery re-queues InProgress jobs
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_recovery_requeues_in_progress() {
        let dir = std::env::temp_dir().join(format!(
            "q-compute-wal-test-recovery-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.wal");

        // Phase 1: create a WAL with an InProgress job.
        {
            let wal = JobWal::open(&path).await.unwrap();
            let job = JobRecord::new(JobType::BridgeVerify, "hash3".into());
            let id = wal.append_job(job).await.unwrap();
            wal.update_status(&id, JobStatus::InProgress, Some("peer-B".into()), None)
                .await
                .unwrap();
            // Drop wal -- simulates crash.
        }

        // Phase 2: re-open -- InProgress should become Queued.
        {
            let wal = JobWal::open(&path).await.unwrap();
            let jobs = wal.list_jobs(Some(JobStatus::Queued), None).await;
            assert_eq!(
                jobs.len(),
                1,
                "InProgress job should be re-queued on recovery"
            );
            assert_eq!(jobs[0].retry_count, 1);
            assert_eq!(jobs[0].assigned_peer, "");
        }

        cleanup(&dir);
    }

    // -----------------------------------------------------------------------
    // Test 5: Recovery marks jobs as Failed after max retries
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_recovery_max_retries_fails() {
        let dir = std::env::temp_dir().join(format!(
            "q-compute-wal-test-maxretry-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.wal");

        // Create a job with retry_count already at MAX_RETRIES, then set InProgress.
        {
            let wal = JobWal::open(&path).await.unwrap();
            let mut job = JobRecord::new(JobType::Mining, "hash4".into());
            job.retry_count = MAX_RETRIES; // already at limit
            let id = wal.append_job(job).await.unwrap();
            wal.update_status(&id, JobStatus::InProgress, Some("peer-C".into()), None)
                .await
                .unwrap();
        }

        // Re-open -- should be Failed, not re-queued.
        {
            let wal = JobWal::open(&path).await.unwrap();
            let jobs = wal.list_jobs(Some(JobStatus::Failed), None).await;
            assert_eq!(jobs.len(), 1);
            let jobs_queued = wal.list_jobs(Some(JobStatus::Queued), None).await;
            assert_eq!(jobs_queued.len(), 0);
        }

        cleanup(&dir);
    }

    // -----------------------------------------------------------------------
    // Test 6: Timeout detection resets stuck jobs
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_timeout_resets_stuck_jobs() {
        let (wal, dir) = temp_wal("timeout").await;
        let job = JobRecord::new(JobType::Inference, "hash5".into());
        let id = wal.append_job(job).await.unwrap();
        wal.update_status(&id, JobStatus::InProgress, Some("peer-D".into()), None)
            .await
            .unwrap();

        // Manually set started_at to the past to simulate timeout.
        {
            let mut state = wal.state.write().await;
            let job = state.jobs.get_mut(&id.0).unwrap();
            job.started_at = now_secs().saturating_sub(TIMEOUT_SECS + 60);
        }

        let reset = wal.check_timeouts().await.unwrap();
        assert_eq!(reset, 1);

        let j = wal.get_job(&id).await.unwrap();
        assert_eq!(j.status, JobStatus::Queued);
        assert_eq!(j.retry_count, 1);

        cleanup(&dir);
    }

    // -----------------------------------------------------------------------
    // Test 7: Timeout with exhausted retries -> Failed
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_timeout_exhausted_retries() {
        let (wal, dir) = temp_wal("timeout_exhaust").await;
        let mut job = JobRecord::new(JobType::ZkProof, "hash6".into());
        job.retry_count = MAX_RETRIES; // already at limit
        let id = wal.append_job(job).await.unwrap();
        wal.update_status(&id, JobStatus::InProgress, Some("peer-E".into()), None)
            .await
            .unwrap();

        // Simulate timeout.
        {
            let mut state = wal.state.write().await;
            let job = state.jobs.get_mut(&id.0).unwrap();
            job.started_at = now_secs().saturating_sub(TIMEOUT_SECS + 60);
        }

        let reset = wal.check_timeouts().await.unwrap();
        assert_eq!(reset, 1);

        let j = wal.get_job(&id).await.unwrap();
        assert_eq!(j.status, JobStatus::Failed);

        cleanup(&dir);
    }

    // -----------------------------------------------------------------------
    // Test 8: Compaction removes old Settled/Failed jobs
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_compaction() {
        let (wal, dir) = temp_wal("compact").await;

        // Job A: old Settled
        let job_a = JobRecord::new(JobType::Mining, "compA".into());
        let id_a = wal.append_job(job_a).await.unwrap();
        wal.update_status(&id_a, JobStatus::InProgress, None, None)
            .await
            .unwrap();
        wal.update_status(&id_a, JobStatus::Completed, None, Some(100))
            .await
            .unwrap();
        wal.update_status(&id_a, JobStatus::Settled, None, None)
            .await
            .unwrap();
        // Backdate completed_at to trigger compaction.
        {
            let mut state = wal.state.write().await;
            let j = state.jobs.get_mut(&id_a.0).unwrap();
            j.completed_at = now_secs().saturating_sub(COMPACTION_AGE_SECS + 3600);
        }

        // Job B: recent Queued (should survive compaction)
        let job_b = JobRecord::new(JobType::Inference, "compB".into());
        let _id_b = wal.append_job(job_b).await.unwrap();

        // Job C: old Failed
        let job_c = JobRecord::new(JobType::ZkProof, "compC".into());
        let id_c = wal.append_job(job_c).await.unwrap();
        wal.update_status(&id_c, JobStatus::InProgress, None, None)
            .await
            .unwrap();
        wal.update_status(&id_c, JobStatus::Failed, None, None)
            .await
            .unwrap();
        {
            let mut state = wal.state.write().await;
            let j = state.jobs.get_mut(&id_c.0).unwrap();
            j.completed_at = now_secs().saturating_sub(COMPACTION_AGE_SECS + 7200);
        }

        let removed = wal.compact().await.unwrap();
        assert_eq!(removed, 2, "Should remove old Settled + old Failed");

        let stats = wal.get_stats().await;
        assert_eq!(stats.total_jobs, 1, "Only the Queued job should remain");

        cleanup(&dir);
    }

    // -----------------------------------------------------------------------
    // Test 9: list_jobs filtering by type and status
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_list_jobs_filters() {
        let (wal, dir) = temp_wal("list_filter").await;

        let j1 = JobRecord::new(JobType::Mining, "f1".into());
        let j2 = JobRecord::new(JobType::Mining, "f2".into());
        let j3 = JobRecord::new(JobType::Inference, "f3".into());
        let id1 = wal.append_job(j1).await.unwrap();
        let _id2 = wal.append_job(j2).await.unwrap();
        let _id3 = wal.append_job(j3).await.unwrap();

        // Move j1 to InProgress
        wal.update_status(&id1, JobStatus::InProgress, None, None)
            .await
            .unwrap();

        // All jobs
        assert_eq!(wal.list_jobs(None, None).await.len(), 3);

        // Only Queued
        assert_eq!(wal.list_jobs(Some(JobStatus::Queued), None).await.len(), 2);

        // Only Mining
        assert_eq!(wal.list_jobs(None, Some(JobType::Mining)).await.len(), 2);

        // Queued + Mining
        assert_eq!(
            wal.list_jobs(Some(JobStatus::Queued), Some(JobType::Mining))
                .await
                .len(),
            1
        );

        // Queued + Inference
        assert_eq!(
            wal.list_jobs(Some(JobStatus::Queued), Some(JobType::Inference))
                .await
                .len(),
            1
        );

        cleanup(&dir);
    }

    // -----------------------------------------------------------------------
    // Test 10: get_stats returns correct counts
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_get_stats() {
        let (wal, dir) = temp_wal("stats").await;

        let j1 = JobRecord::new(JobType::Mining, "s1".into());
        let j2 = JobRecord::new(JobType::Inference, "s2".into());
        let j3 = JobRecord::new(JobType::ZkProof, "s3".into());
        let j4 = JobRecord::new(JobType::BridgeVerify, "s4".into());

        let id1 = wal.append_job(j1).await.unwrap();
        let id2 = wal.append_job(j2).await.unwrap();
        let id3 = wal.append_job(j3).await.unwrap();
        let _id4 = wal.append_job(j4).await.unwrap();

        // j1: Queued -> InProgress -> Completed
        wal.update_status(&id1, JobStatus::InProgress, None, None)
            .await
            .unwrap();
        wal.update_status(&id1, JobStatus::Completed, None, Some(1000))
            .await
            .unwrap();

        // j2: Queued -> InProgress -> Failed
        wal.update_status(&id2, JobStatus::InProgress, None, None)
            .await
            .unwrap();
        wal.update_status(&id2, JobStatus::Failed, None, None)
            .await
            .unwrap();

        // j3: Queued -> InProgress
        wal.update_status(&id3, JobStatus::InProgress, None, None)
            .await
            .unwrap();

        // j4: stays Queued

        let stats = wal.get_stats().await;
        assert_eq!(stats.total_jobs, 4);
        assert_eq!(stats.queued, 1);
        assert_eq!(stats.in_progress, 1);
        assert_eq!(stats.completed, 1);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.settled, 0);

        cleanup(&dir);
    }

    // -----------------------------------------------------------------------
    // Test 11: WAL file persistence (write, close, re-open, verify)
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_persistence_across_restarts() {
        let dir = std::env::temp_dir().join(format!(
            "q-compute-wal-test-persist-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.wal");

        let id;
        // Phase 1: write jobs
        {
            let wal = JobWal::open(&path).await.unwrap();
            let job = JobRecord::new(JobType::Mining, "persist1".into());
            id = wal.append_job(job).await.unwrap();
            wal.update_status(&id, JobStatus::InProgress, Some("peer-F".into()), None)
                .await
                .unwrap();
            wal.update_status(&id, JobStatus::Completed, None, Some(9999))
                .await
                .unwrap();
        }

        // Phase 2: re-open and verify
        {
            let wal = JobWal::open(&path).await.unwrap();
            let j = wal.get_job(&id).await.unwrap();
            assert_eq!(j.status, JobStatus::Completed);
            assert_eq!(j.revenue_micro_qug, 9999);
            assert_eq!(j.assigned_peer, "peer-F");
            assert_eq!(j.payload_hash, "persist1");
        }

        cleanup(&dir);
    }

    // -----------------------------------------------------------------------
    // Test 12: Not-found error for unknown job ID
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_update_nonexistent_job() {
        let (wal, dir) = temp_wal("notfound").await;
        let fake_id = JobId::from_string("nonexistent-id");
        let result = wal
            .update_status(&fake_id, JobStatus::InProgress, None, None)
            .await;
        assert!(result.is_err());
        match result.unwrap_err() {
            WalError::NotFound(id) => assert_eq!(id, "nonexistent-id"),
            other => panic!("Expected NotFound, got: {other}"),
        }
        cleanup(&dir);
    }

    // -----------------------------------------------------------------------
    // Test 13: Failed -> Settled transition
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_failed_to_settled() {
        let (wal, dir) = temp_wal("failed_settled").await;
        let job = JobRecord::new(JobType::BridgeVerify, "fs1".into());
        let id = wal.append_job(job).await.unwrap();

        wal.update_status(&id, JobStatus::InProgress, None, None)
            .await
            .unwrap();
        wal.update_status(&id, JobStatus::Failed, None, None)
            .await
            .unwrap();
        wal.update_status(&id, JobStatus::Settled, None, None)
            .await
            .unwrap();

        let j = wal.get_job(&id).await.unwrap();
        assert_eq!(j.status, JobStatus::Settled);

        cleanup(&dir);
    }
}
