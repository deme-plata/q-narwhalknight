//! Set Reconciliation Sync for Q-NarwhalKnight
//! v4.3.0-beta: Exchange compact block inventory sketches to minimize sync bandwidth.
//!
//! Instead of requesting full block ranges, peers exchange XOR-based sketches
//! of their block height sets and only transfer the differences.
//! Biggest win for steady-state sync when peers are near the same height.
//!
//! Uses binary sub-range fingerprinting for efficient diff narrowing:
//! 1. Both peers create a sketch of their block height range
//! 2. Compare top-level XOR fingerprints — if equal, ranges are identical
//! 3. If different, compare sub-range fingerprints to narrow down missing blocks
//! 4. Only fetch the specific missing block heights

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashSet;
use std::sync::RwLock;
use tracing::{debug, info};

/// Number of sub-ranges to split the height range into for binary search reconciliation
const NUM_SUB_RANGES: usize = 8;

/// Maximum number of differences before recommending full sync
const MAX_DIFF_THRESHOLD: u64 = 50;

/// Compact representation of a block height inventory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockInventorySketch {
    /// XOR of all block height hashes in this range
    pub xor_fingerprint: [u8; 32],
    /// Start of the height range
    pub range_start: u64,
    /// End of the height range (inclusive)
    pub range_end: u64,
    /// Number of blocks in this range
    pub num_elements: u64,
    /// Sketch capacity (max differences detectable)
    pub capacity: u32,
    /// Sub-range fingerprints for binary search reconciliation
    /// Each entry: (sub_range_start, sub_range_end, xor_fingerprint)
    pub sub_ranges: Vec<(u64, u64, [u8; 32])>,
}

/// Result of set reconciliation between two sketches
#[derive(Debug)]
pub struct ReconciliationResult {
    /// Height ranges where local node is missing blocks
    pub missing_ranges: Vec<(u64, u64)>,
    /// Total estimated missing blocks
    pub estimated_missing: u64,
    /// Whether full sync is recommended (too many differences)
    pub recommend_full_sync: bool,
}

/// Set reconciliation manager for efficient block sync
pub struct SetReconciliationManager {
    /// Our known block heights
    local_heights: RwLock<HashSet<u64>>,
}

impl SetReconciliationManager {
    pub fn new() -> Self {
        Self {
            local_heights: RwLock::new(HashSet::new()),
        }
    }

    /// Add a known block height
    pub fn add_height(&self, height: u64) {
        self.local_heights.write().unwrap().insert(height);
    }

    /// Bulk add block heights
    pub fn add_heights(&self, heights: impl Iterator<Item = u64>) {
        let mut set = self.local_heights.write().unwrap();
        set.extend(heights);
    }

    /// Get the count of known heights
    pub fn height_count(&self) -> usize {
        self.local_heights.read().unwrap().len()
    }

    /// Compute XOR fingerprint for a set of heights
    fn compute_fingerprint(heights: &HashSet<u64>, range_start: u64, range_end: u64) -> [u8; 32] {
        let mut xor_result = [0u8; 32];
        for height in heights.iter() {
            if *height >= range_start && *height <= range_end {
                let hash = Self::hash_height(*height);
                for (i, byte) in xor_result.iter_mut().enumerate() {
                    *byte ^= hash[i];
                }
            }
        }
        xor_result
    }

    /// Hash a single block height to 32 bytes
    fn hash_height(height: u64) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(height.to_le_bytes());
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Count elements in a range
    fn count_in_range(heights: &HashSet<u64>, start: u64, end: u64) -> u64 {
        heights.iter().filter(|h| **h >= start && **h <= end).count() as u64
    }

    /// Create a sketch of the local block inventory for a given range
    pub fn create_sketch(&self, range_start: u64, range_end: u64) -> BlockInventorySketch {
        let heights = self.local_heights.read().unwrap();

        let xor_fingerprint = Self::compute_fingerprint(&heights, range_start, range_end);
        let num_elements = Self::count_in_range(&heights, range_start, range_end);

        // Split into sub-ranges for binary search reconciliation
        let total_range = range_end.saturating_sub(range_start) + 1;
        let sub_range_size = total_range / NUM_SUB_RANGES as u64;
        let mut sub_ranges = Vec::with_capacity(NUM_SUB_RANGES);

        for i in 0..NUM_SUB_RANGES {
            let sub_start = range_start + (i as u64) * sub_range_size;
            let sub_end = if i == NUM_SUB_RANGES - 1 {
                range_end
            } else {
                sub_start + sub_range_size - 1
            };
            let sub_fp = Self::compute_fingerprint(&heights, sub_start, sub_end);
            sub_ranges.push((sub_start, sub_end, sub_fp));
        }

        BlockInventorySketch {
            xor_fingerprint,
            range_start,
            range_end,
            num_elements,
            capacity: NUM_SUB_RANGES as u32,
            sub_ranges,
        }
    }

    /// Reconcile local and remote sketches to find missing block ranges
    pub fn reconcile(
        local_sketch: &BlockInventorySketch,
        remote_sketch: &BlockInventorySketch,
    ) -> ReconciliationResult {
        // Quick check: if fingerprints match, ranges are identical
        if local_sketch.xor_fingerprint == remote_sketch.xor_fingerprint {
            return ReconciliationResult {
                missing_ranges: Vec::new(),
                estimated_missing: 0,
                recommend_full_sync: false,
            };
        }

        // Estimate total difference from element counts
        let local_count = local_sketch.num_elements;
        let remote_count = remote_sketch.num_elements;
        let count_diff = if remote_count > local_count {
            remote_count - local_count
        } else {
            0
        };

        // If too many differences, recommend full sync
        if count_diff > MAX_DIFF_THRESHOLD {
            return ReconciliationResult {
                missing_ranges: vec![(local_sketch.range_start, remote_sketch.range_end)],
                estimated_missing: count_diff,
                recommend_full_sync: true,
            };
        }

        // Compare sub-ranges to narrow down missing blocks
        let mut missing_ranges = Vec::new();
        let mut estimated_missing = 0u64;

        let min_len = local_sketch.sub_ranges.len().min(remote_sketch.sub_ranges.len());
        for i in 0..min_len {
            let (l_start, l_end, l_fp) = &local_sketch.sub_ranges[i];
            let (r_start, r_end, r_fp) = &remote_sketch.sub_ranges[i];

            // If sub-range fingerprints differ, this range has missing blocks
            if l_fp != r_fp {
                let range_start = (*l_start).min(*r_start);
                let range_end = (*l_end).max(*r_end);
                missing_ranges.push((range_start, range_end));
                // Rough estimate: proportion of total diff in this sub-range
                estimated_missing += (count_diff / min_len as u64).max(1);
            }
        }

        ReconciliationResult {
            missing_ranges,
            estimated_missing,
            recommend_full_sync: false,
        }
    }

    /// Find specific missing heights by comparing local heights against a remote sketch
    pub fn find_missing_heights(
        &self,
        remote_sketch: &BlockInventorySketch,
    ) -> Vec<(u64, u64)> {
        let local_sketch = self.create_sketch(remote_sketch.range_start, remote_sketch.range_end);
        let result = Self::reconcile(&local_sketch, remote_sketch);
        result.missing_ranges
    }
}

impl Default for SetReconciliationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_sets_match() {
        let mgr1 = SetReconciliationManager::new();
        let mgr2 = SetReconciliationManager::new();

        for h in 1..=100 {
            mgr1.add_height(h);
            mgr2.add_height(h);
        }

        let sketch1 = mgr1.create_sketch(1, 100);
        let sketch2 = mgr2.create_sketch(1, 100);

        assert_eq!(sketch1.xor_fingerprint, sketch2.xor_fingerprint,
            "Identical sets should have identical fingerprints");

        let result = SetReconciliationManager::reconcile(&sketch1, &sketch2);
        assert!(result.missing_ranges.is_empty(), "No missing ranges for identical sets");
        assert_eq!(result.estimated_missing, 0);
        assert!(!result.recommend_full_sync);
    }

    #[test]
    fn test_detect_missing_blocks() {
        let local = SetReconciliationManager::new();
        let remote = SetReconciliationManager::new();

        // Local has 1-90, remote has 1-100
        for h in 1..=90 {
            local.add_height(h);
        }
        for h in 1..=100 {
            remote.add_height(h);
        }

        let local_sketch = local.create_sketch(1, 100);
        let remote_sketch = remote.create_sketch(1, 100);

        assert_ne!(local_sketch.xor_fingerprint, remote_sketch.xor_fingerprint,
            "Different sets should have different fingerprints");

        let result = SetReconciliationManager::reconcile(&local_sketch, &remote_sketch);
        assert!(!result.missing_ranges.is_empty(), "Should detect missing blocks");
        assert!(result.estimated_missing > 0);
    }

    #[test]
    fn test_sub_range_narrowing() {
        let local = SetReconciliationManager::new();
        let remote = SetReconciliationManager::new();

        // Both have 1-800, but remote also has 801-808
        for h in 1..=800 {
            local.add_height(h);
            remote.add_height(h);
        }
        for h in 801..=808 {
            remote.add_height(h);
        }

        let local_sketch = local.create_sketch(1, 808);
        let remote_sketch = remote.create_sketch(1, 808);

        let result = SetReconciliationManager::reconcile(&local_sketch, &remote_sketch);

        // Should narrow to the last sub-range only, not the entire range
        assert!(!result.recommend_full_sync);
        // The missing range should be in the high end
        for (start, _end) in &result.missing_ranges {
            assert!(*start >= 700, "Missing blocks should be detected in the tail, got range starting at {}", start);
        }
    }

    #[test]
    fn test_large_diff_recommends_full_sync() {
        let local = SetReconciliationManager::new();
        let remote = SetReconciliationManager::new();

        // Local has 1-10, remote has 1-1000
        for h in 1..=10 {
            local.add_height(h);
        }
        for h in 1..=1000 {
            remote.add_height(h);
        }

        let local_sketch = local.create_sketch(1, 1000);
        let remote_sketch = remote.create_sketch(1, 1000);

        let result = SetReconciliationManager::reconcile(&local_sketch, &remote_sketch);
        assert!(result.recommend_full_sync, "Large diff should recommend full sync");
    }

    #[test]
    fn test_fingerprint_deterministic() {
        let mgr = SetReconciliationManager::new();
        for h in 1..=50 {
            mgr.add_height(h);
        }

        let sketch1 = mgr.create_sketch(1, 50);
        let sketch2 = mgr.create_sketch(1, 50);

        assert_eq!(sketch1.xor_fingerprint, sketch2.xor_fingerprint);
        assert_eq!(sketch1.num_elements, 50);
        assert_eq!(sketch1.sub_ranges.len(), NUM_SUB_RANGES);
    }

    #[test]
    fn test_empty_sketch() {
        let mgr = SetReconciliationManager::new();
        let sketch = mgr.create_sketch(1, 100);

        assert_eq!(sketch.num_elements, 0);
        assert_eq!(sketch.xor_fingerprint, [0u8; 32], "Empty sketch should have zero fingerprint");
    }

    #[test]
    fn test_find_missing_heights() {
        let local = SetReconciliationManager::new();
        let remote = SetReconciliationManager::new();

        for h in 1..=50 {
            local.add_height(h);
        }
        for h in 1..=60 {
            remote.add_height(h);
        }

        let remote_sketch = remote.create_sketch(1, 60);
        let missing = local.find_missing_heights(&remote_sketch);

        assert!(!missing.is_empty(), "Should find missing heights 51-60");
    }
}
