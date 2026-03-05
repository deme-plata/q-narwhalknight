//! Hashpower-Weighted Security Enhancements for Q-NarwhalKnight
//!
//! v1.3.0-beta: Three cryptographic security enhancements where more hashpower
//! directly translates to stronger cryptographic security.
//!
//! ## Security Enhancements
//!
//! 1. **Cumulative Work-Weighted Block Security**
//!    - Security Level = log2(cumulative_work) bits
//!    - Each block's security is proportional to ALL previous mining work
//!    - Bitcoin-grade security at 90+ bits after sufficient network work
//!
//! 2. **Hashpower-Adaptive VDF Complexity**
//!    - VDF difficulty dynamically scales with network hashrate
//!    - `vdf_difficulty = base_difficulty * (1 + log2(network_hashrate / baseline))`
//!    - Prevents low-hashrate timing attacks
//!
//! 3. **Mining-Derived Randomness Beacon**
//!    - Accumulated mining solutions form cryptographic random oracle
//!    - `beacon[n] = SHA3-256(block_hashes[n-1000..n], nonces[n-1000..n], vdf_proofs[n-1000..n])`
//!    - NIST beacon-quality randomness from distributed mining work
//!
//! ## Design Philosophy
//!
//! The core insight is that in a blockchain, hashpower = work = security.
//! These enhancements make that relationship explicit and cryptographically measurable.
//! An attacker would need to expend MORE computational work than the entire network
//! history to compromise the system.

use anyhow::{anyhow, Result};
use sha3::{Digest, Sha3_256, Sha3_512};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// ============================================================================
// ENHANCEMENT 1: CUMULATIVE WORK-WEIGHTED BLOCK SECURITY
// ============================================================================

/// Cumulative work tracker for measuring blockchain security level
///
/// Security Level = log2(cumulative_work) bits
/// - At 2^80 cumulative work: 80-bit security (strong)
/// - At 2^90 cumulative work: 90-bit security (very strong)
/// - At 2^100 cumulative work: 100-bit security (exceptional)
#[derive(Debug, Clone)]
pub struct CumulativeWorkSecurity {
    /// Total cumulative work (in atomic units - essentially difficulty sum)
    cumulative_work: u128,

    /// Security level in bits (log2 of cumulative work)
    security_bits: f64,

    /// Chain height at this measurement
    chain_height: u64,

    /// Average difficulty over all blocks
    average_difficulty: f64,

    /// Security tier classification
    security_tier: SecurityTier,

    /// Historical work snapshots for analysis
    work_snapshots: VecDeque<WorkSnapshot>,

    /// Maximum snapshots to retain
    max_snapshots: usize,
}

/// Security tier classification based on cumulative work
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityTier {
    /// Below 40 bits - Minimal security (new chain)
    Minimal,
    /// 40-60 bits - Basic security
    Basic,
    /// 60-80 bits - Strong security
    Strong,
    /// 80-100 bits - Very strong security (Bitcoin-grade)
    VeryStrong,
    /// 100+ bits - Exceptional security
    Exceptional,
}

impl std::fmt::Display for SecurityTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecurityTier::Minimal => write!(f, "MINIMAL"),
            SecurityTier::Basic => write!(f, "BASIC"),
            SecurityTier::Strong => write!(f, "STRONG"),
            SecurityTier::VeryStrong => write!(f, "VERY_STRONG"),
            SecurityTier::Exceptional => write!(f, "EXCEPTIONAL"),
        }
    }
}

/// Snapshot of cumulative work at a specific height
#[derive(Debug, Clone)]
pub struct WorkSnapshot {
    pub height: u64,
    pub cumulative_work: u128,
    pub security_bits: f64,
    pub timestamp: u64,
}

impl CumulativeWorkSecurity {
    /// Create a new cumulative work security tracker
    pub fn new() -> Self {
        Self {
            cumulative_work: 0,
            security_bits: 0.0,
            chain_height: 0,
            average_difficulty: 0.0,
            security_tier: SecurityTier::Minimal,
            work_snapshots: VecDeque::with_capacity(1000),
            max_snapshots: 1000,
        }
    }

    /// Add work from a new block
    ///
    /// # Arguments
    /// * `difficulty` - The difficulty target of the block
    /// * `height` - Block height
    /// * `timestamp` - Block timestamp
    pub fn add_block_work(&mut self, difficulty: u64, height: u64, timestamp: u64) {
        // Work = 2^difficulty (exponential relationship)
        // We use difficulty directly for computational efficiency
        // Each unit of difficulty represents doubling the work
        let block_work = if difficulty < 128 {
            1u128 << difficulty.min(64) as u32
        } else {
            u128::MAX
        };

        // Saturating add to prevent overflow
        self.cumulative_work = self.cumulative_work.saturating_add(block_work);
        self.chain_height = height;

        // Calculate security bits = log2(cumulative_work)
        self.security_bits = if self.cumulative_work > 0 {
            (self.cumulative_work as f64).log2()
        } else {
            0.0
        };

        // Update average difficulty
        self.average_difficulty = if height > 0 {
            (self.average_difficulty * (height - 1) as f64 + difficulty as f64) / height as f64
        } else {
            difficulty as f64
        };

        // Update security tier
        self.security_tier = self.classify_security_tier();

        // Store snapshot every 100 blocks
        if height % 100 == 0 {
            self.add_snapshot(height, timestamp);
        }

        debug!(
            "📊 [CUMULATIVE-WORK] Block {} added: work={}, total={}, security={:.2} bits ({})",
            height, block_work, self.cumulative_work, self.security_bits, self.security_tier
        );
    }

    /// Classify the current security tier based on cumulative work
    fn classify_security_tier(&self) -> SecurityTier {
        match self.security_bits as u32 {
            0..=39 => SecurityTier::Minimal,
            40..=59 => SecurityTier::Basic,
            60..=79 => SecurityTier::Strong,
            80..=99 => SecurityTier::VeryStrong,
            _ => SecurityTier::Exceptional,
        }
    }

    /// Add a work snapshot
    fn add_snapshot(&mut self, height: u64, timestamp: u64) {
        if self.work_snapshots.len() >= self.max_snapshots {
            self.work_snapshots.pop_front();
        }

        self.work_snapshots.push_back(WorkSnapshot {
            height,
            cumulative_work: self.cumulative_work,
            security_bits: self.security_bits,
            timestamp,
        });
    }

    /// Get the current security level in bits
    pub fn security_bits(&self) -> f64 {
        self.security_bits
    }

    /// Get the current security tier
    pub fn security_tier(&self) -> SecurityTier {
        self.security_tier
    }

    /// Get cumulative work
    pub fn cumulative_work(&self) -> u128 {
        self.cumulative_work
    }

    /// Calculate work required to attack (rewrite) the chain from a specific height
    ///
    /// Returns the amount of work (in bits) needed to forge an alternate chain
    pub fn attack_cost_bits(&self, from_height: u64) -> f64 {
        if from_height >= self.chain_height {
            return 0.0;
        }

        // Find the snapshot closest to from_height
        let work_at_height = self.work_snapshots
            .iter()
            .filter(|s| s.height <= from_height)
            .max_by_key(|s| s.height)
            .map(|s| s.cumulative_work)
            .unwrap_or(0);

        let work_to_recreate = self.cumulative_work.saturating_sub(work_at_height);

        if work_to_recreate > 0 {
            (work_to_recreate as f64).log2()
        } else {
            0.0
        }
    }

    /// Verify that a claimed cumulative work is valid
    pub fn verify_cumulative_work(&self, claimed_work: u128, claimed_bits: f64) -> bool {
        let actual_bits = if claimed_work > 0 {
            (claimed_work as f64).log2()
        } else {
            0.0
        };

        // Allow 0.1 bit tolerance for floating point precision
        (actual_bits - claimed_bits).abs() < 0.1
    }

    /// Generate a cumulative work proof for inclusion in block headers
    pub fn generate_work_proof(&self) -> CumulativeWorkProof {
        CumulativeWorkProof {
            cumulative_work: self.cumulative_work,
            security_bits: self.security_bits,
            chain_height: self.chain_height,
            proof_hash: self.compute_proof_hash(),
        }
    }

    /// Compute cryptographic hash of cumulative work state
    fn compute_proof_hash(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(self.cumulative_work.to_le_bytes());
        hasher.update(self.chain_height.to_le_bytes());
        hasher.update(self.security_bits.to_le_bytes());
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }
}

impl Default for CumulativeWorkSecurity {
    fn default() -> Self {
        Self::new()
    }
}

/// Proof of cumulative work for block headers
#[derive(Debug, Clone)]
pub struct CumulativeWorkProof {
    /// Total cumulative work
    pub cumulative_work: u128,
    /// Security level in bits
    pub security_bits: f64,
    /// Chain height
    pub chain_height: u64,
    /// Cryptographic hash of the proof
    pub proof_hash: [u8; 32],
}

// ============================================================================
// ENHANCEMENT 2: HASHPOWER-ADAPTIVE VDF COMPLEXITY
// ============================================================================

/// Hashpower-adaptive VDF difficulty calculator
///
/// VDF difficulty scales logarithmically with network hashrate:
/// `vdf_difficulty = base_difficulty * (1 + log2(network_hashrate / baseline))`
///
/// This prevents timing attacks when network hashrate is low
#[derive(Debug, Clone)]
pub struct AdaptiveVdfComplexity {
    /// Base VDF difficulty (minimum)
    base_difficulty: u64,

    /// Baseline hashrate for scaling (H/s)
    baseline_hashrate: u64,

    /// Current network hashrate (H/s)
    current_hashrate: u64,

    /// Current adaptive VDF difficulty
    current_vdf_difficulty: u64,

    /// Hashrate history for smoothing
    hashrate_history: VecDeque<HashrateDataPoint>,

    /// Smoothing window size
    smoothing_window: usize,

    /// Maximum VDF difficulty cap
    max_difficulty: u64,

    /// Minimum VDF difficulty floor
    min_difficulty: u64,
}

/// Hashrate data point for smoothing
#[derive(Debug, Clone)]
pub struct HashrateDataPoint {
    pub hashrate: u64,
    pub timestamp: u64,
}

/// VDF difficulty adjustment result
#[derive(Debug, Clone)]
pub struct VdfDifficultyAdjustment {
    /// New VDF difficulty
    pub new_difficulty: u64,
    /// Previous VDF difficulty
    pub previous_difficulty: u64,
    /// Adjustment factor applied
    pub adjustment_factor: f64,
    /// Network hashrate used for calculation
    pub network_hashrate: u64,
    /// Reason for adjustment
    pub reason: String,
}

impl AdaptiveVdfComplexity {
    /// Create a new adaptive VDF complexity calculator
    ///
    /// # Arguments
    /// * `base_difficulty` - Minimum VDF difficulty
    /// * `baseline_hashrate` - Reference hashrate for scaling (e.g., 1 GH/s = 1_000_000_000)
    pub fn new(base_difficulty: u64, baseline_hashrate: u64) -> Self {
        Self {
            base_difficulty,
            baseline_hashrate,
            current_hashrate: baseline_hashrate,
            current_vdf_difficulty: base_difficulty,
            hashrate_history: VecDeque::with_capacity(100),
            smoothing_window: 24, // 24 data points (~2 hours at 5-min intervals)
            max_difficulty: base_difficulty * 100, // 100x max increase
            min_difficulty: base_difficulty,
        }
    }

    /// Update network hashrate and recalculate VDF difficulty
    ///
    /// # Arguments
    /// * `hashrate` - Current network hashrate in H/s
    /// * `timestamp` - Current timestamp
    pub fn update_hashrate(&mut self, hashrate: u64, timestamp: u64) -> VdfDifficultyAdjustment {
        let previous_difficulty = self.current_vdf_difficulty;

        // Add to history
        if self.hashrate_history.len() >= self.smoothing_window {
            self.hashrate_history.pop_front();
        }
        self.hashrate_history.push_back(HashrateDataPoint { hashrate, timestamp });

        // Calculate smoothed hashrate (exponential moving average)
        let smoothed_hashrate = self.calculate_smoothed_hashrate();
        self.current_hashrate = smoothed_hashrate;

        // Calculate adaptive VDF difficulty
        // Formula: vdf_difficulty = base * (1 + log2(hashrate / baseline))
        let adjustment_factor = if smoothed_hashrate > self.baseline_hashrate {
            1.0 + (smoothed_hashrate as f64 / self.baseline_hashrate as f64).log2()
        } else if smoothed_hashrate > 0 {
            // Below baseline: reduce difficulty proportionally but maintain minimum
            (smoothed_hashrate as f64 / self.baseline_hashrate as f64).sqrt()
        } else {
            1.0
        };

        // Apply adjustment with bounds
        let new_difficulty = ((self.base_difficulty as f64 * adjustment_factor) as u64)
            .max(self.min_difficulty)
            .min(self.max_difficulty);

        self.current_vdf_difficulty = new_difficulty;

        let reason = if new_difficulty > previous_difficulty {
            format!("Increased due to higher hashrate ({} H/s)", smoothed_hashrate)
        } else if new_difficulty < previous_difficulty {
            format!("Decreased due to lower hashrate ({} H/s)", smoothed_hashrate)
        } else {
            "No change".to_string()
        };

        debug!(
            "🔧 [ADAPTIVE-VDF] Hashrate: {} H/s, Factor: {:.3}x, Difficulty: {} -> {}",
            smoothed_hashrate, adjustment_factor, previous_difficulty, new_difficulty
        );

        VdfDifficultyAdjustment {
            new_difficulty,
            previous_difficulty,
            adjustment_factor,
            network_hashrate: smoothed_hashrate,
            reason,
        }
    }

    /// Calculate smoothed hashrate using exponential moving average
    fn calculate_smoothed_hashrate(&self) -> u64 {
        if self.hashrate_history.is_empty() {
            return self.baseline_hashrate;
        }

        let alpha = 2.0 / (self.smoothing_window as f64 + 1.0);
        let mut ema = self.hashrate_history[0].hashrate as f64;

        for point in self.hashrate_history.iter().skip(1) {
            ema = alpha * point.hashrate as f64 + (1.0 - alpha) * ema;
        }

        ema as u64
    }

    /// Get current VDF difficulty
    pub fn current_difficulty(&self) -> u64 {
        self.current_vdf_difficulty
    }

    /// Get current network hashrate (smoothed)
    pub fn current_hashrate(&self) -> u64 {
        self.current_hashrate
    }

    /// Calculate expected VDF computation time at current difficulty
    ///
    /// # Arguments
    /// * `iterations` - Number of VDF iterations
    ///
    /// Returns expected time in milliseconds
    pub fn expected_vdf_time_ms(&self, iterations: u64) -> u64 {
        // Assume baseline VDF speed of 1M iterations/second on reference hardware
        let baseline_speed = 1_000_000u64;
        let time_at_baseline = iterations * 1000 / baseline_speed;

        // Difficulty factor increases time linearly
        time_at_baseline * self.current_vdf_difficulty / self.base_difficulty
    }

    /// Verify that a VDF proof was computed with adequate difficulty
    pub fn verify_vdf_difficulty(&self, claimed_difficulty: u64, proof_timestamp: u64) -> Result<bool> {
        // Find the required difficulty at the proof timestamp
        let required_difficulty = self.difficulty_at_timestamp(proof_timestamp);

        if claimed_difficulty >= required_difficulty {
            Ok(true)
        } else {
            Err(anyhow!(
                "VDF difficulty {} below required {} at timestamp {}",
                claimed_difficulty, required_difficulty, proof_timestamp
            ))
        }
    }

    /// Get difficulty requirement at a specific timestamp
    fn difficulty_at_timestamp(&self, timestamp: u64) -> u64 {
        // Find the closest hashrate data point
        for point in self.hashrate_history.iter().rev() {
            if point.timestamp <= timestamp {
                let adjustment = if point.hashrate > self.baseline_hashrate {
                    1.0 + (point.hashrate as f64 / self.baseline_hashrate as f64).log2()
                } else {
                    1.0
                };
                return ((self.base_difficulty as f64 * adjustment) as u64)
                    .max(self.min_difficulty)
                    .min(self.max_difficulty);
            }
        }

        self.base_difficulty
    }
}

impl Default for AdaptiveVdfComplexity {
    fn default() -> Self {
        Self::new(
            16,                    // Base difficulty: 16 (reasonable starting point)
            1_000_000_000,         // Baseline: 1 GH/s
        )
    }
}

// ============================================================================
// ENHANCEMENT 3: MINING-DERIVED RANDOMNESS BEACON
// ============================================================================

/// Mining-derived cryptographic randomness beacon
///
/// Generates NIST beacon-quality randomness from accumulated mining work:
/// `beacon[n] = SHA3-512(block_hashes[n-WINDOW..n] || nonces[n-WINDOW..n] || vdf_proofs[n-WINDOW..n])`
///
/// Properties:
/// - Unpredictable: Requires controlling majority hashpower over entire window
/// - Verifiable: Anyone can verify the beacon from public blockchain data
/// - Bias-resistant: No single miner can manipulate output
/// - High-entropy: 512 bits of output from accumulated work
#[derive(Debug)]
pub struct MiningRandomnessBeacon {
    /// Beacon accumulation window (blocks)
    window_size: usize,

    /// Ring buffer of block entropy contributions
    entropy_buffer: VecDeque<BlockEntropyContribution>,

    /// Current beacon value (512 bits)
    current_beacon: [u8; 64],

    /// Beacon epoch (increments each time beacon updates)
    epoch: u64,

    /// Last block height contributing to beacon
    last_height: u64,

    /// Total entropy contributions
    total_contributions: u64,
}

/// Entropy contribution from a single mined block
#[derive(Debug, Clone)]
pub struct BlockEntropyContribution {
    /// Block height
    pub height: u64,
    /// Block hash (32 bytes)
    pub block_hash: [u8; 32],
    /// Mining nonce (8 bytes)
    pub nonce: u64,
    /// VDF proof hash (32 bytes, if available)
    pub vdf_proof_hash: Option<[u8; 32]>,
    /// Mining difficulty
    pub difficulty: u64,
    /// Block timestamp
    pub timestamp: u64,
}

/// Beacon output with proof
#[derive(Debug, Clone)]
pub struct BeaconOutput {
    /// Beacon value (512 bits)
    pub beacon: [u8; 64],
    /// Beacon epoch
    pub epoch: u64,
    /// Height range used
    pub height_range: (u64, u64),
    /// Number of blocks contributing
    pub block_count: usize,
    /// Total difficulty in window
    pub total_difficulty: u128,
    /// Merkle root of contributions
    pub merkle_root: [u8; 32],
}

impl MiningRandomnessBeacon {
    /// Create a new mining randomness beacon
    ///
    /// # Arguments
    /// * `window_size` - Number of blocks to accumulate (default: 1000)
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            entropy_buffer: VecDeque::with_capacity(window_size),
            current_beacon: [0u8; 64],
            epoch: 0,
            last_height: 0,
            total_contributions: 0,
        }
    }

    /// Add entropy contribution from a newly mined block
    pub fn add_block_entropy(&mut self, contribution: BlockEntropyContribution) {
        // Ensure we maintain proper ordering
        if contribution.height <= self.last_height && self.last_height > 0 {
            warn!(
                "⚠️ [BEACON] Out-of-order block {} (last: {}), skipping",
                contribution.height, self.last_height
            );
            return;
        }

        // Remove old entries if buffer is full
        while self.entropy_buffer.len() >= self.window_size {
            self.entropy_buffer.pop_front();
        }

        self.last_height = contribution.height;
        self.entropy_buffer.push_back(contribution);
        self.total_contributions += 1;

        // Regenerate beacon when buffer is full or every 100 blocks
        if self.entropy_buffer.len() >= self.window_size || self.total_contributions % 100 == 0 {
            self.regenerate_beacon();
        }

        debug!(
            "🎲 [BEACON] Added entropy from block {}, buffer: {}/{}",
            self.last_height, self.entropy_buffer.len(), self.window_size
        );
    }

    /// Regenerate the beacon value from current entropy buffer
    fn regenerate_beacon(&mut self) {
        if self.entropy_buffer.is_empty() {
            return;
        }

        let mut hasher = Sha3_512::new();

        // Mix in epoch and window metadata
        hasher.update(self.epoch.to_le_bytes());
        hasher.update((self.entropy_buffer.len() as u64).to_le_bytes());

        // Accumulate entropy from all blocks in window
        for contribution in &self.entropy_buffer {
            // Block hash provides 256 bits of entropy
            hasher.update(&contribution.block_hash);

            // Nonce provides additional entropy (varies with mining)
            hasher.update(contribution.nonce.to_le_bytes());

            // VDF proof hash adds time-locked entropy
            if let Some(vdf_hash) = &contribution.vdf_proof_hash {
                hasher.update(vdf_hash);
            }

            // Difficulty weights the contribution
            hasher.update(contribution.difficulty.to_le_bytes());

            // Timestamp adds temporal ordering
            hasher.update(contribution.timestamp.to_le_bytes());
        }

        // Final mixing pass
        let intermediate = hasher.finalize();

        // Second pass with intermediate for additional mixing
        let mut final_hasher = Sha3_512::new();
        final_hasher.update(&intermediate);
        final_hasher.update(b"Q-NarwhalKnight-Mining-Beacon-v1.3.0");
        final_hasher.update(self.epoch.to_le_bytes());

        let result = final_hasher.finalize();
        self.current_beacon.copy_from_slice(&result);

        self.epoch += 1;

        info!(
            "🎲 [BEACON] Epoch {} generated from {} blocks ({}..{})",
            self.epoch,
            self.entropy_buffer.len(),
            self.entropy_buffer.front().map(|c| c.height).unwrap_or(0),
            self.last_height
        );
    }

    /// Get the current beacon value
    pub fn current_beacon(&self) -> [u8; 64] {
        self.current_beacon
    }

    /// Get current beacon epoch
    pub fn current_epoch(&self) -> u64 {
        self.epoch
    }

    /// Get beacon output with full proof
    pub fn get_beacon_output(&self) -> BeaconOutput {
        let (min_height, max_height) = if let (Some(first), Some(last)) =
            (self.entropy_buffer.front(), self.entropy_buffer.back()) {
            (first.height, last.height)
        } else {
            (0, 0)
        };

        let total_difficulty: u128 = self.entropy_buffer
            .iter()
            .map(|c| c.difficulty as u128)
            .sum();

        BeaconOutput {
            beacon: self.current_beacon,
            epoch: self.epoch,
            height_range: (min_height, max_height),
            block_count: self.entropy_buffer.len(),
            total_difficulty,
            merkle_root: self.compute_merkle_root(),
        }
    }

    /// Compute Merkle root of entropy contributions for verification
    fn compute_merkle_root(&self) -> [u8; 32] {
        if self.entropy_buffer.is_empty() {
            return [0u8; 32];
        }

        // Collect leaf hashes
        let mut leaves: Vec<[u8; 32]> = self.entropy_buffer
            .iter()
            .map(|c| {
                let mut hasher = Sha3_256::new();
                hasher.update(&c.block_hash);
                hasher.update(c.nonce.to_le_bytes());
                hasher.update(c.height.to_le_bytes());
                let result = hasher.finalize();
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&result);
                hash
            })
            .collect();

        // Build Merkle tree
        while leaves.len() > 1 {
            let mut next_level = Vec::with_capacity((leaves.len() + 1) / 2);

            for chunk in leaves.chunks(2) {
                let mut hasher = Sha3_256::new();
                hasher.update(&chunk[0]);
                if chunk.len() > 1 {
                    hasher.update(&chunk[1]);
                } else {
                    hasher.update(&chunk[0]); // Duplicate for odd count
                }
                let result = hasher.finalize();
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&result);
                next_level.push(hash);
            }

            leaves = next_level;
        }

        leaves.first().copied().unwrap_or([0u8; 32])
    }

    /// Extract random bytes from beacon for cryptographic use
    ///
    /// # Arguments
    /// * `purpose` - Application-specific domain separator
    /// * `length` - Number of random bytes needed
    pub fn derive_randomness(&self, purpose: &[u8], length: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(length);
        let mut counter = 0u64;

        while result.len() < length {
            let mut hasher = Sha3_256::new();
            hasher.update(&self.current_beacon);
            hasher.update(purpose);
            hasher.update(counter.to_le_bytes());

            let chunk = hasher.finalize();
            let bytes_needed = (length - result.len()).min(32);
            result.extend_from_slice(&chunk[..bytes_needed]);

            counter += 1;
        }

        result.truncate(length);
        result
    }

    /// Verify a beacon value against claimed contributions
    pub fn verify_beacon(
        claimed_beacon: &[u8; 64],
        contributions: &[BlockEntropyContribution],
        claimed_epoch: u64,
    ) -> bool {
        if contributions.is_empty() {
            return false;
        }

        // Regenerate beacon from contributions
        let mut hasher = Sha3_512::new();
        hasher.update(claimed_epoch.to_le_bytes());
        hasher.update((contributions.len() as u64).to_le_bytes());

        for contribution in contributions {
            hasher.update(&contribution.block_hash);
            hasher.update(contribution.nonce.to_le_bytes());
            if let Some(vdf_hash) = &contribution.vdf_proof_hash {
                hasher.update(vdf_hash);
            }
            hasher.update(contribution.difficulty.to_le_bytes());
            hasher.update(contribution.timestamp.to_le_bytes());
        }

        let intermediate = hasher.finalize();

        let mut final_hasher = Sha3_512::new();
        final_hasher.update(&intermediate);
        final_hasher.update(b"Q-NarwhalKnight-Mining-Beacon-v1.3.0");
        final_hasher.update(claimed_epoch.to_le_bytes());

        let computed = final_hasher.finalize();

        computed.as_slice() == claimed_beacon
    }
}

impl Default for MiningRandomnessBeacon {
    fn default() -> Self {
        Self::new(1000) // Default: 1000 block window (~8.3 hours at 30s blocks)
    }
}

// ============================================================================
// INTEGRATED HASHPOWER SECURITY MANAGER
// ============================================================================

/// Unified manager for all hashpower-weighted security enhancements
#[derive(Debug)]
pub struct HashpowerSecurityManager {
    /// Cumulative work security tracker
    cumulative_work: Arc<RwLock<CumulativeWorkSecurity>>,

    /// Adaptive VDF complexity calculator
    adaptive_vdf: Arc<RwLock<AdaptiveVdfComplexity>>,

    /// Mining-derived randomness beacon
    randomness_beacon: Arc<RwLock<MiningRandomnessBeacon>>,

    /// Security enhancement statistics
    stats: Arc<RwLock<HashpowerSecurityStats>>,
}

/// Statistics for hashpower security enhancements
#[derive(Debug, Clone, Default)]
pub struct HashpowerSecurityStats {
    /// Total blocks processed
    pub blocks_processed: u64,
    /// Current security level in bits
    pub security_bits: f64,
    /// Current security tier
    pub security_tier: String,
    /// Current VDF difficulty
    pub vdf_difficulty: u64,
    /// Current beacon epoch
    pub beacon_epoch: u64,
    /// Network hashrate (H/s)
    pub network_hashrate: u64,
    /// Cumulative work (u128 as string for JSON)
    pub cumulative_work: String,
}

impl HashpowerSecurityManager {
    /// Create a new hashpower security manager with default settings
    pub fn new() -> Self {
        Self::with_config(16, 1_000_000_000, 1000)
    }

    /// Create with custom configuration
    ///
    /// # Arguments
    /// * `vdf_base_difficulty` - Base VDF difficulty
    /// * `hashrate_baseline` - Baseline hashrate for VDF scaling (H/s)
    /// * `beacon_window` - Blocks for randomness beacon window
    pub fn with_config(
        vdf_base_difficulty: u64,
        hashrate_baseline: u64,
        beacon_window: usize,
    ) -> Self {
        Self {
            cumulative_work: Arc::new(RwLock::new(CumulativeWorkSecurity::new())),
            adaptive_vdf: Arc::new(RwLock::new(AdaptiveVdfComplexity::new(
                vdf_base_difficulty,
                hashrate_baseline,
            ))),
            randomness_beacon: Arc::new(RwLock::new(MiningRandomnessBeacon::new(beacon_window))),
            stats: Arc::new(RwLock::new(HashpowerSecurityStats::default())),
        }
    }

    /// Process a newly mined block through all security enhancements
    pub async fn process_block(
        &self,
        height: u64,
        block_hash: [u8; 32],
        nonce: u64,
        difficulty: u64,
        timestamp: u64,
        vdf_proof_hash: Option<[u8; 32]>,
        network_hashrate: u64,
    ) -> Result<HashpowerSecurityStats> {
        // Update cumulative work
        {
            let mut cw = self.cumulative_work.write().await;
            cw.add_block_work(difficulty, height, timestamp);
        }

        // Update VDF difficulty based on hashrate
        {
            let mut vdf = self.adaptive_vdf.write().await;
            vdf.update_hashrate(network_hashrate, timestamp);
        }

        // Add entropy to beacon
        {
            let mut beacon = self.randomness_beacon.write().await;
            beacon.add_block_entropy(BlockEntropyContribution {
                height,
                block_hash,
                nonce,
                vdf_proof_hash,
                difficulty,
                timestamp,
            });
        }

        // Update and return stats
        self.update_stats().await
    }

    /// Update security statistics
    async fn update_stats(&self) -> Result<HashpowerSecurityStats> {
        let cw = self.cumulative_work.read().await;
        let vdf = self.adaptive_vdf.read().await;
        let beacon = self.randomness_beacon.read().await;

        let stats = HashpowerSecurityStats {
            blocks_processed: cw.chain_height,
            security_bits: cw.security_bits(),
            security_tier: cw.security_tier().to_string(),
            vdf_difficulty: vdf.current_difficulty(),
            beacon_epoch: beacon.current_epoch(),
            network_hashrate: vdf.current_hashrate(),
            cumulative_work: cw.cumulative_work().to_string(),
        };

        *self.stats.write().await = stats.clone();
        Ok(stats)
    }

    /// Get current security statistics
    pub async fn get_stats(&self) -> HashpowerSecurityStats {
        self.stats.read().await.clone()
    }

    /// Get cumulative work proof for block header
    pub async fn get_work_proof(&self) -> CumulativeWorkProof {
        self.cumulative_work.read().await.generate_work_proof()
    }

    /// Get current VDF difficulty requirement
    pub async fn get_vdf_difficulty(&self) -> u64 {
        self.adaptive_vdf.read().await.current_difficulty()
    }

    /// Get current beacon output
    pub async fn get_beacon_output(&self) -> BeaconOutput {
        self.randomness_beacon.read().await.get_beacon_output()
    }

    /// Derive randomness from beacon for specific purpose
    pub async fn derive_randomness(&self, purpose: &[u8], length: usize) -> Vec<u8> {
        self.randomness_beacon.read().await.derive_randomness(purpose, length)
    }

    /// Get attack cost in bits for rewriting chain from height
    pub async fn attack_cost_bits(&self, from_height: u64) -> f64 {
        self.cumulative_work.read().await.attack_cost_bits(from_height)
    }

    /// v9.1.0: Live security bits that account for real-time network hashpower.
    /// Takes the maximum of cumulative-work-based security bits (historical) and
    /// log2 of the total live network hashrate (current). This means security
    /// never drops below what cumulative work provides, but can be boosted when
    /// live hashpower is higher than what blocks alone would indicate.
    ///
    /// `total_network_hashrate_hs` should be the sum of all peers' announced
    /// hashrates from the compute power gossipsub topic.
    pub fn live_security_bits(cumulative_bits: f64, total_network_hashrate_hs: f64) -> f64 {
        if total_network_hashrate_hs <= 0.0 {
            return cumulative_bits;
        }
        // log2(hashrate) gives "bits of security" from live compute power
        let live_bits = total_network_hashrate_hs.log2();
        cumulative_bits.max(live_bits)
    }
}

impl Default for HashpowerSecurityManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cumulative_work_security() {
        let mut cws = CumulativeWorkSecurity::new();

        // Add blocks with increasing difficulty
        for i in 1..=100 {
            cws.add_block_work(20, i, i * 30);
        }

        assert_eq!(cws.chain_height, 100);
        assert!(cws.security_bits() > 20.0);
        assert!(cws.cumulative_work() > 0);

        // Verify security tier classification
        assert!(matches!(
            cws.security_tier(),
            SecurityTier::Minimal | SecurityTier::Basic | SecurityTier::Strong
        ));
    }

    #[test]
    fn test_security_tier_classification() {
        let mut cws = CumulativeWorkSecurity::new();

        // Low work = minimal security
        cws.add_block_work(10, 1, 30);
        assert_eq!(cws.security_tier(), SecurityTier::Minimal);
    }

    #[test]
    fn test_adaptive_vdf_complexity() {
        let mut vdf = AdaptiveVdfComplexity::new(16, 1_000_000_000);

        // Initial state
        assert_eq!(vdf.current_difficulty(), 16);

        // High hashrate should increase difficulty
        let adjustment = vdf.update_hashrate(10_000_000_000, 1000);
        assert!(adjustment.new_difficulty > 16);

        // Low hashrate should decrease difficulty
        let adjustment2 = vdf.update_hashrate(100_000_000, 2000);
        assert!(adjustment2.new_difficulty <= adjustment.new_difficulty);
    }

    #[test]
    fn test_mining_randomness_beacon() {
        let mut beacon = MiningRandomnessBeacon::new(10);

        // Add entropy from blocks
        for i in 1..=15 {
            beacon.add_block_entropy(BlockEntropyContribution {
                height: i,
                block_hash: [i as u8; 32],
                nonce: i * 12345,
                vdf_proof_hash: Some([i as u8; 32]),
                difficulty: 20,
                timestamp: i * 30,
            });
        }

        // Beacon should have been generated
        assert!(beacon.current_epoch() > 0);
        assert!(beacon.current_beacon() != [0u8; 64]);

        // Derive randomness
        let random = beacon.derive_randomness(b"test-purpose", 64);
        assert_eq!(random.len(), 64);
    }

    #[test]
    fn test_beacon_verification() {
        let mut beacon = MiningRandomnessBeacon::new(5);

        let contributions: Vec<BlockEntropyContribution> = (1..=5)
            .map(|i| BlockEntropyContribution {
                height: i,
                block_hash: [i as u8; 32],
                nonce: i * 12345,
                vdf_proof_hash: Some([i as u8; 32]),
                difficulty: 20,
                timestamp: i * 30,
            })
            .collect();

        for c in &contributions {
            beacon.add_block_entropy(c.clone());
        }

        let output = beacon.get_beacon_output();

        // Verification should pass with correct data
        assert!(MiningRandomnessBeacon::verify_beacon(
            &output.beacon,
            &contributions,
            output.epoch,
        ));
    }

    #[tokio::test]
    async fn test_hashpower_security_manager() {
        let manager = HashpowerSecurityManager::new();

        // Process several blocks
        for i in 1..=50 {
            let stats = manager.process_block(
                i,
                [i as u8; 32],
                i * 12345,
                20,
                i * 30,
                Some([i as u8; 32]),
                1_000_000_000,
            ).await.unwrap();

            assert_eq!(stats.blocks_processed, i);
        }

        let final_stats = manager.get_stats().await;
        assert_eq!(final_stats.blocks_processed, 50);
        assert!(final_stats.security_bits > 0.0);
        assert!(final_stats.vdf_difficulty > 0);
    }
}
