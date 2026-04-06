//! Corrupt Block Cleanup Tests (v10.2.8)
//!
//! Tests the logic for detecting and cleaning up corrupt blocks near the tip,
//! specifically for the kill -9 corruption scenario where partially-written blocks
//! exist in RocksDB but fail deserialization.
//!
//! Run with: cargo test --package q-storage --test corrupt_block_cleanup_tests

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// MOCK STRUCTURES
// ============================================================================

/// Simulates a block store where blocks can be valid, corrupt, or missing
struct MockBlockStore {
    blocks: HashMap<u64, BlockState>,
    latest_pointer: AtomicU64,
    safe_floor: AtomicU64,
    tip_height: AtomicU64,
}

#[derive(Clone, Debug, PartialEq)]
enum BlockState {
    Valid(Vec<u8>),   // Valid block data
    Corrupt(Vec<u8>), // Data exists but fails deserialization
    // Missing blocks are simply absent from the HashMap
}

impl MockBlockStore {
    fn new() -> Self {
        Self {
            blocks: HashMap::new(),
            latest_pointer: AtomicU64::new(0),
            safe_floor: AtomicU64::new(0),
            tip_height: AtomicU64::new(0),
        }
    }

    /// Insert a valid block at the given height
    fn insert_valid(&mut self, height: u64) {
        // Simulate a valid block: 10KB+ of structured data
        let data = vec![0xABu8; 10_000];
        self.blocks.insert(height, BlockState::Valid(data));
    }

    /// Insert a corrupt block at the given height (simulates kill -9 partial write)
    fn insert_corrupt(&mut self, height: u64) {
        // Simulate corrupt data: small, garbage bytes (like a partial bincode write)
        let data = vec![0x78u8; 500]; // 0x78 = tag 120, matching real error "tag for enum is not valid, found 120"
        self.blocks.insert(height, BlockState::Corrupt(data));
    }

    /// Check if block data exists (raw key check, like hot_db.get())
    fn has_data(&self, height: u64) -> bool {
        self.blocks.contains_key(&height)
    }

    /// Check if block is valid (simulates full deserialization)
    fn is_valid(&self, height: u64) -> bool {
        match self.blocks.get(&height) {
            Some(BlockState::Valid(_)) => true,
            Some(BlockState::Corrupt(_)) => false,
            None => false,
        }
    }

    /// Delete a block
    fn delete(&mut self, height: u64) {
        self.blocks.remove(&height);
    }

    /// Simulate cleanup_corrupt_blocks_near_tip() logic
    /// Returns Some(new_height) if pointers were reset, None if no corruption found
    fn cleanup_corrupt_blocks_near_tip(&mut self, recovered_height: u64) -> Option<u64> {
        const SCAN_BELOW: u64 = 200;

        let scan_start = recovered_height.saturating_sub(SCAN_BELOW);
        if scan_start == 0 || recovered_height < 100 {
            return None;
        }

        let mut deleted_count = 0u64;
        let mut lowest_corrupt: Option<u64> = None;

        for height in scan_start..=recovered_height {
            // Raw key check — does data exist?
            if !self.has_data(height) {
                continue; // Missing block, not our concern
            }

            // Attempt deserialization
            if !self.is_valid(height) {
                // Corrupt block found — delete it
                self.delete(height);
                deleted_count += 1;
                if lowest_corrupt.is_none() || height < lowest_corrupt.unwrap() {
                    lowest_corrupt = Some(height);
                }
            }
        }

        if deleted_count == 0 {
            return None;
        }

        // Reset pointers to just below the first corrupt block
        let new_height = lowest_corrupt.unwrap().saturating_sub(1);
        self.latest_pointer.store(new_height, Ordering::SeqCst);
        self.safe_floor.store(new_height, Ordering::SeqCst);
        self.tip_height.store(new_height, Ordering::SeqCst);

        Some(new_height)
    }

    /// Simulate cleanup_corrupt_blocks_above() — the existing function
    /// Shows how it MISSES corruption below recovered_height
    fn cleanup_corrupt_blocks_above(&mut self, height: u64) -> u64 {
        let scan_limit = height + 100; // Simplified from 10000
        let mut deleted = 0;
        for check_height in (height + 1)..=scan_limit {
            if self.has_data(check_height) && !self.is_valid(check_height) {
                self.delete(check_height);
                deleted += 1;
            } else if !self.has_data(check_height) {
                break; // Existing behavior: stops at first missing block
            }
        }
        deleted
    }

    /// Simulate gap detection (get_first_missing_height)
    /// Uses raw key existence — corrupt blocks appear as "present"
    fn first_missing_height_raw(&self, from: u64, to: u64) -> Option<u64> {
        for h in from..=to {
            if !self.has_data(h) {
                return Some(h);
            }
        }
        None
    }
}

// ============================================================================
// TESTS: CORRUPT BLOCK DETECTION
// ============================================================================

/// Test: The exact Epsilon scenario — kill -9 corrupts blocks 13489423-13489438
#[test]
fn test_epsilon_kill9_scenario() {
    let mut store = MockBlockStore::new();

    // Blocks 13489400-13489422 are valid
    for h in 13489400..=13489422 {
        store.insert_valid(h);
    }
    // Blocks 13489423-13489438 are CORRUPT (kill -9 partial writes)
    for h in 13489423..=13489438 {
        store.insert_corrupt(h);
    }
    // Blocks 13489439-13489443 are valid (written before the kill)
    for h in 13489439..=13489443 {
        store.insert_valid(h);
    }

    // Set pointers as fast recovery would
    store.latest_pointer.store(13489443, Ordering::SeqCst);
    store.safe_floor.store(13489443, Ordering::SeqCst);
    store.tip_height.store(13489443, Ordering::SeqCst);

    // Step 1: cleanup_corrupt_blocks_above(13489443) — existing function
    let deleted_above = store.cleanup_corrupt_blocks_above(13489443);
    assert_eq!(deleted_above, 0, "cleanup_above should find nothing above tip");

    // Step 2: Verify corrupt blocks STILL EXIST (the bug!)
    assert!(store.has_data(13489423), "Corrupt blocks should still exist after cleanup_above");
    assert!(store.has_data(13489438), "Corrupt blocks should still exist after cleanup_above");

    // Step 3: Gap detection sees corrupt blocks as "present" (the bug!)
    let gap = store.first_missing_height_raw(13489400, 13489443);
    assert_eq!(gap, None, "Raw gap detection should NOT find gaps — corrupt blocks have data");

    // Step 4: NEW FIX — cleanup_corrupt_blocks_near_tip()
    let result = store.cleanup_corrupt_blocks_near_tip(13489443);
    assert!(result.is_some(), "Should detect corruption below tip");

    let new_height = result.unwrap();
    assert_eq!(new_height, 13489422, "Should reset to last valid block before corruption");

    // Step 5: Verify corrupt blocks are deleted
    for h in 13489423..=13489438 {
        assert!(!store.has_data(h), "Corrupt block at {} should be deleted", h);
    }

    // Step 6: Verify valid blocks are preserved
    for h in 13489400..=13489422 {
        assert!(store.has_data(h), "Valid block at {} should be preserved", h);
    }
    for h in 13489439..=13489443 {
        assert!(store.has_data(h), "Valid block at {} above corruption should be preserved", h);
    }

    // Step 7: Verify pointers were reset
    assert_eq!(store.latest_pointer.load(Ordering::SeqCst), 13489422);
    assert_eq!(store.safe_floor.load(Ordering::SeqCst), 13489422);
    assert_eq!(store.tip_height.load(Ordering::SeqCst), 13489422);

    // Step 8: Now gap detection works (corrupt blocks are now truly missing)
    let gap_after = store.first_missing_height_raw(13489400, 13489443);
    assert_eq!(gap_after, Some(13489423), "Gap detection should now find the missing blocks for turbo sync");
}

/// Test: No corruption — function is a no-op
#[test]
fn test_no_corruption_is_noop() {
    let mut store = MockBlockStore::new();

    for h in 1000..=1200 {
        store.insert_valid(h);
    }
    store.latest_pointer.store(1200, Ordering::SeqCst);
    store.safe_floor.store(1200, Ordering::SeqCst);

    let result = store.cleanup_corrupt_blocks_near_tip(1200);
    assert_eq!(result, None, "No corruption should return None");

    // Pointers unchanged
    assert_eq!(store.latest_pointer.load(Ordering::SeqCst), 1200);
    assert_eq!(store.safe_floor.load(Ordering::SeqCst), 1200);

    // All blocks still present
    for h in 1000..=1200 {
        assert!(store.has_data(h));
    }
}

/// Test: Corruption at the very tip (last block is corrupt)
#[test]
fn test_corruption_at_tip() {
    let mut store = MockBlockStore::new();

    for h in 5000..=5099 {
        store.insert_valid(h);
    }
    store.insert_corrupt(5100); // Tip block is corrupt
    store.latest_pointer.store(5100, Ordering::SeqCst);

    let result = store.cleanup_corrupt_blocks_near_tip(5100);
    assert_eq!(result, Some(5099), "Should reset to block before corrupt tip");
    assert!(!store.has_data(5100), "Corrupt tip should be deleted");
    assert!(store.has_data(5099), "Valid block before tip preserved");
}

/// Test: Single corrupt block in the middle
#[test]
fn test_single_corrupt_block() {
    let mut store = MockBlockStore::new();

    for h in 10000..=10100 {
        store.insert_valid(h);
    }
    // Replace one block with corrupt data
    store.insert_corrupt(10050);
    store.latest_pointer.store(10100, Ordering::SeqCst);

    let result = store.cleanup_corrupt_blocks_near_tip(10100);
    assert_eq!(result, Some(10049), "Should reset to block before single corruption");
    assert!(!store.has_data(10050), "Single corrupt block deleted");
    assert!(store.has_data(10049), "Block before corruption preserved");
    assert!(store.has_data(10051), "Block after corruption preserved");
}

/// Test: Missing blocks in range are ignored (not treated as corrupt)
#[test]
fn test_missing_blocks_ignored() {
    let mut store = MockBlockStore::new();

    // Blocks with gaps (normal during turbo sync)
    for h in 8000..=8050 {
        store.insert_valid(h);
    }
    // Gap at 8051-8060 (missing, not corrupt)
    for h in 8061..=8100 {
        store.insert_valid(h);
    }
    store.latest_pointer.store(8100, Ordering::SeqCst);

    let result = store.cleanup_corrupt_blocks_near_tip(8100);
    assert_eq!(result, None, "Missing blocks should NOT trigger cleanup");
    assert_eq!(store.latest_pointer.load(Ordering::SeqCst), 8100, "Pointer unchanged");
}

/// Test: Very low height — function skips (guard clause)
#[test]
fn test_low_height_skipped() {
    let mut store = MockBlockStore::new();
    store.insert_corrupt(50);

    let result = store.cleanup_corrupt_blocks_near_tip(50);
    assert_eq!(result, None, "Heights < 100 should be skipped");
}

/// Test: Multiple corruption regions — lowest is used for pointer reset
#[test]
fn test_multiple_corruption_regions() {
    let mut store = MockBlockStore::new();

    for h in 20000..=20200 {
        store.insert_valid(h);
    }
    // Two separate corruption regions
    store.insert_corrupt(20050);
    store.insert_corrupt(20051);
    store.insert_corrupt(20150);
    store.latest_pointer.store(20200, Ordering::SeqCst);

    let result = store.cleanup_corrupt_blocks_near_tip(20200);
    assert_eq!(result, Some(20049), "Should reset to below LOWEST corruption");

    // All corrupt blocks deleted
    assert!(!store.has_data(20050));
    assert!(!store.has_data(20051));
    assert!(!store.has_data(20150));
}

/// Test: Pointer reset values are correct
#[test]
fn test_pointer_reset_consistency() {
    let mut store = MockBlockStore::new();

    for h in 100000..=100100 {
        store.insert_valid(h);
    }
    store.insert_corrupt(100030);
    store.latest_pointer.store(100100, Ordering::SeqCst);
    store.safe_floor.store(100100, Ordering::SeqCst);
    store.tip_height.store(100100, Ordering::SeqCst);

    let result = store.cleanup_corrupt_blocks_near_tip(100100);
    assert_eq!(result, Some(100029));

    // All three pointers must be consistent
    let latest = store.latest_pointer.load(Ordering::SeqCst);
    let floor = store.safe_floor.load(Ordering::SeqCst);
    let tip = store.tip_height.load(Ordering::SeqCst);
    assert_eq!(latest, floor, "latest and safe_floor must match");
    assert_eq!(latest, tip, "latest and tip_height must match");
    assert_eq!(latest, 100029, "All pointers at 100029");
}
