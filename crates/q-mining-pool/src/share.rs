//! Share validation for mining pool

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::{HashSet, VecDeque};
use parking_lot::RwLock;

use crate::error::{PoolResult, ShareError};
use crate::job::MiningJob;
use crate::worker::WorkerId;

/// LRU-evicting dedup cache — prevents both total-flush bypass and unbounded growth.
struct DedupCache {
    set:   HashSet<String>,
    queue: VecDeque<String>,
    cap:   usize,
}

impl DedupCache {
    fn new(cap: usize) -> Self {
        Self { set: HashSet::new(), queue: VecDeque::new(), cap }
    }

    fn contains(&self, id: &str) -> bool {
        self.set.contains(id)
    }

    /// Insert id, evicting the oldest entry if at capacity.
    fn insert(&mut self, id: String) {
        if self.set.len() >= self.cap {
            if let Some(old) = self.queue.pop_front() {
                self.set.remove(&old);
                tracing::trace!("[POOL] dedup_evict oldest_prefix={}", &old[..old.len().min(12)]);
            }
        }
        self.queue.push_back(id.clone());
        self.set.insert(id);
    }

    fn clear(&mut self) {
        self.set.clear();
        self.queue.clear();
    }
}

/// Mining share
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Share {
    /// Worker who submitted the share
    pub worker_id: WorkerId,

    /// Job ID this share is for
    pub job_id: String,

    /// Share difficulty
    pub difficulty: f64,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// The hash that was submitted
    pub hash: [u8; 32],

    /// Nonce used
    pub nonce: u64,

    /// Whether this share found a block
    pub is_block: bool,

    /// Block height (if this is a block)
    pub block_height: Option<u64>,
}

impl Share {
    /// Create new share
    pub fn new(
        worker_id: WorkerId,
        job_id: String,
        difficulty: f64,
        hash: [u8; 32],
        nonce: u64,
        is_block: bool,
    ) -> Self {
        Self {
            worker_id,
            job_id,
            difficulty,
            timestamp: Utc::now(),
            hash,
            nonce,
            is_block,
            block_height: None,
        }
    }

    /// Get share unique identifier (for duplicate detection)
    pub fn unique_id(&self) -> String {
        format!("{}:{}:{}", self.job_id, self.nonce, hex::encode(&self.hash[..8]))
    }
}

/// Share submission from miner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareSubmission {
    /// Job ID
    pub job_id: String,

    /// Extranonce2 (miner-controlled portion)
    pub extranonce2: String,

    /// Nonce
    pub nonce: u64,

    /// Ntime (timestamp override)
    pub ntime: u64,

    /// Worker name
    pub worker_name: String,
}

/// Result of share validation
#[derive(Debug, Clone)]
pub enum ShareValidationResult {
    /// Valid share
    ValidShare {
        /// Actual share difficulty
        difficulty: f64,
    },

    /// Valid share that found a block
    BlockFound {
        /// Block hash
        hash: [u8; 32],

        /// Block header bytes
        header: Vec<u8>,

        /// Block height
        height: u64,
    },

    /// Share is stale (job expired)
    Stale,

    /// Share difficulty too low
    LowDifficulty {
        got: f64,
        need: f64,
    },

    /// Duplicate share
    Duplicate,

    /// Invalid hash
    InvalidHash,

    /// Invalid nonce
    InvalidNonce,
}

/// Share validator
pub struct ShareValidator {
    /// Network target (for block detection)
    network_target: [u8; 32],

    /// Duplicate share cache — LRU-evicting, never clears all entries at once (POOL-001/003)
    duplicate_cache: RwLock<DedupCache>,
}

impl ShareValidator {
    /// Create new share validator
    pub fn new(network_difficulty: f64) -> Self {
        Self {
            network_target: Self::difficulty_to_target(network_difficulty),
            duplicate_cache: RwLock::new(DedupCache::new(100_000)),
        }
    }

    /// Update network difficulty
    pub fn update_network_difficulty(&mut self, difficulty: f64) {
        self.network_target = Self::difficulty_to_target(difficulty);
    }

    /// Convert difficulty to target bytes
    fn difficulty_to_target(difficulty: f64) -> [u8; 32] {
        // Target = 2^256 / difficulty
        // For simplicity, we use a linear approximation
        // Higher difficulty = lower target = harder to find

        let mut target = [0xff_u8; 32];

        if difficulty >= 1.0 {
            // Calculate how many leading zero bytes we need
            let leading_zeros = (difficulty.log2() / 8.0).floor() as usize;
            let remaining = difficulty / (256.0_f64.powi(leading_zeros as i32));

            for i in 0..leading_zeros.min(32) {
                target[i] = 0;
            }

            if leading_zeros < 32 {
                target[leading_zeros] = (255.0 / remaining) as u8;
            }
        }

        target
    }

    /// Calculate share difficulty from hash
    fn calculate_difficulty(hash: &[u8; 32]) -> f64 {
        // Find the position of the first non-zero byte
        let mut leading_zeros = 0;
        for &byte in hash.iter() {
            if byte == 0 {
                leading_zeros += 1;
            } else {
                break;
            }
        }

        // Calculate difficulty based on leading zeros
        let base_difficulty = 256.0_f64.powi(leading_zeros as i32);

        // Adjust based on the first non-zero byte
        if leading_zeros < 32 {
            let first_nonzero = hash[leading_zeros] as f64;
            if first_nonzero > 0.0 {
                base_difficulty * (255.0 / first_nonzero)
            } else {
                base_difficulty
            }
        } else {
            f64::MAX // All zeros - impossibly high difficulty
        }
    }

    /// Check if hash meets target
    fn meets_target(hash: &[u8; 32], target: &[u8; 32]) -> bool {
        // Hash must be less than or equal to target
        for i in 0..32 {
            if hash[i] < target[i] {
                return true;
            } else if hash[i] > target[i] {
                return false;
            }
        }
        true // Equal is valid
    }

    /// Validate a share submission
    pub fn validate(
        &self,
        job: &MiningJob,
        submission: &ShareSubmission,
        worker_difficulty: f64,
    ) -> PoolResult<ShareValidationResult> {
        // 1. Check if job is stale
        if job.is_stale() {
            return Ok(ShareValidationResult::Stale);
        }

        // 2. Check for duplicate — key excludes extranonce2 (miner-controlled, must not vary to bypass)
        let nonce_prefix = submission.nonce.to_string();
        let unique_id = format!("{}:{}", submission.job_id, nonce_prefix);
        tracing::trace!("[POOL] share validate job={} nonce_prefix={}", submission.job_id, &nonce_prefix[..nonce_prefix.len().min(8)]);
        {
            let cache = self.duplicate_cache.read();
            if cache.contains(&unique_id) {
                tracing::debug!("[POOL] share DUPLICATE job={} nonce_prefix={}", submission.job_id, &nonce_prefix[..nonce_prefix.len().min(8)]);
                return Ok(ShareValidationResult::Duplicate);
            }
        }

        // 3. Reconstruct block header and calculate hash
        let header = self.reconstruct_header(job, submission)?;
        let hash = self.hash_header(&header);

        // 4. Calculate share difficulty
        let share_difficulty = Self::calculate_difficulty(&hash);

        // 5. Check share difficulty
        let share_target = Self::difficulty_to_target(worker_difficulty);
        if !Self::meets_target(&hash, &share_target) {
            return Ok(ShareValidationResult::LowDifficulty {
                got: share_difficulty,
                need: worker_difficulty,
            });
        }

        // 6. Record share to prevent duplicates — LRU eviction, never total-flush
        {
            let mut cache = self.duplicate_cache.write();
            tracing::trace!("[POOL] share dedup_insert cache_len={}", cache.set.len());
            cache.insert(unique_id);
        }

        // 7. Check if this is a block
        if Self::meets_target(&hash, &self.network_target) {
            return Ok(ShareValidationResult::BlockFound {
                hash,
                header,
                height: job.height,
            });
        }

        // Valid share
        Ok(ShareValidationResult::ValidShare {
            difficulty: share_difficulty,
        })
    }

    /// Reconstruct block header from job and submission
    fn reconstruct_header(&self, job: &MiningJob, submission: &ShareSubmission) -> PoolResult<Vec<u8>> {
        let mut header = Vec::with_capacity(140);

        // Version (4 bytes)
        header.extend_from_slice(&job.version.to_le_bytes());

        // Previous block hash (32 bytes)
        header.extend_from_slice(&job.prev_hash);

        // Merkle root (32 bytes)
        // Reconstruct from coinbase with extranonce
        let merkle_root = self.calculate_merkle_root(job, submission)?;
        header.extend_from_slice(&merkle_root);

        // Timestamp (4 bytes)
        header.extend_from_slice(&(submission.ntime as u32).to_le_bytes());

        // Bits/difficulty (4 bytes)
        header.extend_from_slice(&job.bits.to_le_bytes());

        // Nonce (8 bytes for Q-NarwhalKnight, 4 for Bitcoin-style)
        header.extend_from_slice(&submission.nonce.to_le_bytes());

        Ok(header)
    }

    /// Calculate merkle root with coinbase
    fn calculate_merkle_root(&self, job: &MiningJob, submission: &ShareSubmission) -> PoolResult<[u8; 32]> {
        // Build coinbase transaction with extranonce
        let mut coinbase = Vec::new();
        coinbase.extend_from_slice(&job.coinbase1);
        coinbase.extend_from_slice(hex::decode(&submission.extranonce2).unwrap_or_default().as_slice());
        coinbase.extend_from_slice(&job.coinbase2);

        // Hash coinbase
        let coinbase_hash = {
            let mut hasher = Sha3_256::new();
            hasher.update(&coinbase);
            hasher.finalize()
        };

        // Calculate merkle root from coinbase hash and merkle branch
        let mut current = <[u8; 32]>::try_from(coinbase_hash.as_slice()).unwrap();

        for branch_hash in &job.merkle_branch {
            let mut hasher = Sha3_256::new();
            hasher.update(&current);
            hasher.update(branch_hash);
            let result = hasher.finalize();
            current = <[u8; 32]>::try_from(result.as_slice()).unwrap();
        }

        Ok(current)
    }

    /// Hash block header using SHA3-256
    fn hash_header(&self, header: &[u8]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(header);
        let result = hasher.finalize();
        <[u8; 32]>::try_from(result.as_slice()).unwrap()
    }

    /// Clear duplicate cache
    pub fn clear_cache(&self) {
        self.duplicate_cache.write().clear();
    }

    /// Current dedup cache size (for monitoring)
    pub fn cache_size(&self) -> usize {
        self.duplicate_cache.read().set.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_difficulty_to_target() {
        let target = ShareValidator::difficulty_to_target(1.0);
        assert_eq!(target[0], 0xff); // Low difficulty = high target

        let target = ShareValidator::difficulty_to_target(256.0);
        assert_eq!(target[0], 0); // Higher difficulty = lower target
    }

    #[test]
    fn test_meets_target() {
        let hash = [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];

        let easy_target = [0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                          0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                          0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                          0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];

        assert!(ShareValidator::meets_target(&hash, &easy_target));

        let hard_target = [0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
                          0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                          0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                          0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];

        assert!(!ShareValidator::meets_target(&hash, &hard_target));
    }

    #[test]
    fn test_calculate_difficulty() {
        // All zeros = max difficulty
        let hash = [0x00; 32];
        let diff = ShareValidator::calculate_difficulty(&hash);
        assert!(diff > 1e30);

        // First byte non-zero = low difficulty
        let hash = [0xff; 32];
        let diff = ShareValidator::calculate_difficulty(&hash);
        assert!(diff < 2.0);
    }
}
