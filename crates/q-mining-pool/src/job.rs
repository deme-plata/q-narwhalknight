//! Mining job management for pool

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::{JobError, PoolResult};

/// Mining job for pool workers
#[derive(Debug, Clone)]
pub struct MiningJob {
    /// Unique job ID
    pub job_id: String,

    /// Previous block hash
    pub prev_hash: [u8; 32],

    /// Block version
    pub version: u32,

    /// Merkle branch for coinbase reconstruction
    pub merkle_branch: Vec<[u8; 32]>,

    /// Coinbase part 1 (before extranonce)
    pub coinbase1: Vec<u8>,

    /// Coinbase part 2 (after extranonce)
    pub coinbase2: Vec<u8>,

    /// Difficulty bits
    pub bits: u32,

    /// Target difficulty
    pub target: [u8; 32],

    /// Block height
    pub height: u64,

    /// Job creation time
    pub created_at: DateTime<Utc>,

    /// Whether to clean/invalidate previous jobs
    pub clean_jobs: bool,

    /// Job expiry time
    pub expires_at: Instant,
}

impl MiningJob {
    /// Check if job has expired
    pub fn is_stale(&self) -> bool {
        Instant::now() > self.expires_at
    }

    /// Get time remaining until expiry
    pub fn time_remaining(&self) -> Duration {
        self.expires_at.saturating_duration_since(Instant::now())
    }

    /// Convert to Stratum job notification params
    pub fn to_stratum_params(&self, extranonce1: &str) -> Vec<serde_json::Value> {
        vec![
            serde_json::json!(self.job_id),
            serde_json::json!(hex::encode(&self.prev_hash)),
            serde_json::json!(hex::encode(&self.coinbase1)),
            serde_json::json!(hex::encode(&self.coinbase2)),
            serde_json::json!(self.merkle_branch.iter()
                .map(|h| hex::encode(h))
                .collect::<Vec<_>>()),
            serde_json::json!(format!("{:08x}", self.version)),
            serde_json::json!(format!("{:08x}", self.bits)),
            serde_json::json!(format!("{:08x}", self.created_at.timestamp() as u32)),
            serde_json::json!(self.clean_jobs),
        ]
    }
}

/// Block template from node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockTemplate {
    /// Previous block hash
    pub prev_hash: [u8; 32],

    /// Block height
    pub height: u64,

    /// Difficulty target
    pub target: [u8; 32],

    /// Difficulty bits
    pub bits: u32,

    /// Block version
    pub version: u32,

    /// Pending transactions
    pub transactions: Vec<PoolTransaction>,

    /// Coinbase value (block reward + fees)
    pub coinbase_value: u64,

    /// Template creation time
    pub curtime: u64,
}

/// Simplified transaction for pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolTransaction {
    /// Transaction hash
    pub hash: [u8; 32],

    /// Transaction data
    pub data: Vec<u8>,

    /// Transaction fee
    pub fee: u64,
}

/// Job manager for mining pool
pub struct JobManager {
    /// Active jobs by ID
    jobs: DashMap<String, Arc<MiningJob>>,

    /// Current job
    current_job: RwLock<Option<Arc<MiningJob>>>,

    /// Job ID counter
    job_counter: AtomicU64,

    /// Pool wallet address for coinbase
    pool_wallet: String,

    /// Job expiry duration
    job_expiry: Duration,

    /// Maximum number of jobs to keep
    max_jobs: usize,
}

impl JobManager {
    /// Create new job manager
    pub fn new(pool_wallet: String) -> Self {
        Self {
            jobs: DashMap::new(),
            current_job: RwLock::new(None),
            job_counter: AtomicU64::new(1),
            pool_wallet,
            job_expiry: Duration::from_secs(120), // Jobs expire after 2 minutes
            max_jobs: 10,
        }
    }

    /// Generate unique job ID
    fn generate_job_id(&self) -> String {
        let counter = self.job_counter.fetch_add(1, Ordering::Relaxed);
        format!("{:016x}", counter)
    }

    /// Create new job from block template
    pub fn create_job(&self, template: BlockTemplate, clean_jobs: bool) -> PoolResult<Arc<MiningJob>> {
        let job_id = self.generate_job_id();

        // Build coinbase transaction
        let (coinbase1, coinbase2) = self.build_coinbase(&template)?;

        // Build merkle branch
        let merkle_branch = self.build_merkle_branch(&template.transactions)?;

        let job = Arc::new(MiningJob {
            job_id: job_id.clone(),
            prev_hash: template.prev_hash,
            version: template.version,
            merkle_branch,
            coinbase1,
            coinbase2,
            bits: template.bits,
            target: template.target,
            height: template.height,
            created_at: Utc::now(),
            clean_jobs,
            expires_at: Instant::now() + self.job_expiry,
        });

        // Store job
        self.jobs.insert(job_id, Arc::clone(&job));

        // Update current job
        *self.current_job.write() = Some(Arc::clone(&job));

        // Cleanup old jobs
        self.cleanup_old_jobs();

        tracing::info!(
            job_id = %job.job_id,
            height = job.height,
            clean = clean_jobs,
            "Created new mining job"
        );

        Ok(job)
    }

    /// Build coinbase transaction
    fn build_coinbase(&self, template: &BlockTemplate) -> PoolResult<(Vec<u8>, Vec<u8>)> {
        // Simplified coinbase structure for Q-NarwhalKnight
        // Real implementation would follow the actual block format

        let mut coinbase1 = Vec::new();

        // Version
        coinbase1.extend_from_slice(&1u32.to_le_bytes());

        // Input count
        coinbase1.push(1);

        // Coinbase input (null prevout)
        coinbase1.extend_from_slice(&[0u8; 32]); // prev tx hash
        coinbase1.extend_from_slice(&0xffffffff_u32.to_le_bytes()); // prev output index

        // Coinbase script length (placeholder for extranonce)
        coinbase1.push(8); // 4 bytes extranonce1 + 4 bytes extranonce2

        // Extranonce1 will be inserted by pool, extranonce2 by miner

        let mut coinbase2 = Vec::new();

        // Sequence
        coinbase2.extend_from_slice(&0xffffffff_u32.to_le_bytes());

        // Output count
        coinbase2.push(1);

        // Output value
        coinbase2.extend_from_slice(&template.coinbase_value.to_le_bytes());

        // Output script (pay to pool wallet)
        let script = self.build_payout_script(&self.pool_wallet);
        coinbase2.push(script.len() as u8);
        coinbase2.extend_from_slice(&script);

        // Locktime
        coinbase2.extend_from_slice(&0u32.to_le_bytes());

        Ok((coinbase1, coinbase2))
    }

    /// Build payout script for address
    fn build_payout_script(&self, address: &str) -> Vec<u8> {
        // Simplified: just hash the address
        // Real implementation would create proper P2PKH/P2SH script
        let mut hasher = Sha3_256::new();
        hasher.update(address.as_bytes());
        hasher.finalize().to_vec()
    }

    /// Build merkle branch from transactions
    fn build_merkle_branch(&self, transactions: &[PoolTransaction]) -> PoolResult<Vec<[u8; 32]>> {
        if transactions.is_empty() {
            return Ok(vec![]);
        }

        // Get transaction hashes
        let mut hashes: Vec<[u8; 32]> = transactions.iter()
            .map(|tx| tx.hash)
            .collect();

        let mut branch = Vec::new();

        while hashes.len() > 1 {
            // Take first hash for branch
            if let Some(first) = hashes.first() {
                branch.push(*first);
            }

            // Pair up and hash
            let mut new_hashes = Vec::new();
            for chunk in hashes.chunks(2) {
                let mut hasher = Sha3_256::new();
                hasher.update(&chunk[0]);
                if chunk.len() > 1 {
                    hasher.update(&chunk[1]);
                } else {
                    hasher.update(&chunk[0]); // Duplicate if odd
                }
                let result = hasher.finalize();
                new_hashes.push(<[u8; 32]>::try_from(result.as_slice()).unwrap());
            }
            hashes = new_hashes;
        }

        Ok(branch)
    }

    /// Get current job
    pub fn get_current_job(&self) -> Option<Arc<MiningJob>> {
        self.current_job.read().clone()
    }

    /// Get job by ID
    pub fn get_job(&self, job_id: &str) -> Option<Arc<MiningJob>> {
        self.jobs.get(job_id).map(|j| Arc::clone(&j))
    }

    /// Cleanup old/expired jobs
    fn cleanup_old_jobs(&self) {
        // Remove expired jobs
        self.jobs.retain(|_, job| !job.is_stale());

        // Keep only max_jobs
        while self.jobs.len() > self.max_jobs {
            // Remove oldest job
            if let Some(oldest) = self.jobs.iter()
                .min_by_key(|j| j.created_at)
                .map(|j| j.job_id.clone())
            {
                self.jobs.remove(&oldest);
            } else {
                break;
            }
        }
    }

    /// Get job count
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Invalidate all jobs (new block found)
    pub fn invalidate_all(&self) {
        self.jobs.clear();
        *self.current_job.write() = None;
        tracing::info!("All jobs invalidated");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_creation() {
        let manager = JobManager::new("qnk123".to_string());

        let template = BlockTemplate {
            prev_hash: [0; 32],
            height: 100,
            target: [0xff; 32],
            bits: 0x1d00ffff,
            version: 1,
            transactions: vec![],
            coinbase_value: 2_000_000_000,
            curtime: 1234567890,
        };

        let job = manager.create_job(template, false).unwrap();

        assert_eq!(job.height, 100);
        assert!(!job.clean_jobs);
        assert!(!job.is_stale());
    }

    #[test]
    fn test_job_id_uniqueness() {
        let manager = JobManager::new("qnk123".to_string());

        let id1 = manager.generate_job_id();
        let id2 = manager.generate_job_id();
        let id3 = manager.generate_job_id();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }
}
