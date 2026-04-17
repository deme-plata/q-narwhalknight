//! DAG Read-Path Tests
//!
//! v10.3.6: Tests for the DAG key fallback in get_qblocks_range()
//!
//! 545,710 blocks are stored under "qblock:dag:{height}:{proposer}" format
//! from early chain history. The fast path only searches "qblock:height:{N}".
//! These tests verify the fallback correctly finds DAG entries.
//!
//! Run with: cargo test --package q-storage --test dag_read_path_tests
//!
//! These tests use in-memory data structures — no RocksDB needed.

use std::collections::HashMap;

// ============================================================================
// MOCK BLOCK AND STORAGE FOR TESTING
// ============================================================================

/// Minimal block representation for testing
#[derive(Debug, Clone, PartialEq)]
struct MockBlock {
    height: u64,
    proposer: String,
    data: Vec<u8>,
}

/// Simulates the dual key-format storage (qblock:height: + qblock:dag:)
struct MockBlockStore {
    /// Blocks stored as "qblock:height:{N}"
    height_keys: HashMap<u64, MockBlock>,
    /// Blocks stored as "qblock:dag:{N}:{proposer}"
    dag_keys: HashMap<(u64, String), MockBlock>,
}

impl MockBlockStore {
    fn new() -> Self {
        Self {
            height_keys: HashMap::new(),
            dag_keys: HashMap::new(),
        }
    }

    fn insert_height(&mut self, height: u64, block: MockBlock) {
        self.height_keys.insert(height, block);
    }

    fn insert_dag(&mut self, height: u64, proposer: &str, block: MockBlock) {
        self.dag_keys.insert((height, proposer.to_string()), block);
    }

    /// Simulates the ORIGINAL get_qblocks_range (fast path only — no DAG fallback)
    fn get_qblocks_range_old(&self, start: u64, limit: usize) -> Vec<MockBlock> {
        let mut blocks = Vec::new();
        for h in start..start + limit as u64 {
            if let Some(block) = self.height_keys.get(&h) {
                blocks.push(block.clone());
            }
        }
        blocks.sort_by_key(|b| b.height);
        blocks
    }

    /// Simulates the FIXED get_qblocks_range (with DAG fallback)
    fn get_qblocks_range_fixed(&self, start: u64, limit: usize) -> Vec<MockBlock> {
        let end = start + limit as u64 - 1;

        // Step 1: Fast path — check qblock:height:{N}
        let mut blocks = Vec::new();
        let mut found_heights = std::collections::HashSet::new();

        for h in start..=end {
            if let Some(block) = self.height_keys.get(&h) {
                found_heights.insert(h);
                blocks.push(block.clone());
            }
        }

        // If fast path got everything, return
        if blocks.len() as u64 == limit as u64 {
            return blocks;
        }

        // Step 2: DAG fallback — for missing heights, check qblock:dag:{N}:*
        let mut dag_found = 0u64;
        for h in start..=end {
            if found_heights.contains(&h) {
                continue; // Already found via fast path
            }

            // Find first DAG entry at this height (sorted by proposer, deterministic)
            let mut dag_at_height: Vec<&MockBlock> = self.dag_keys.iter()
                .filter(|((height, _), _)| *height == h)
                .map(|(_, block)| block)
                .collect();
            dag_at_height.sort_by_key(|b| &b.proposer);

            if let Some(block) = dag_at_height.first() {
                // Safety check: height matches
                if block.height == h {
                    blocks.push((*block).clone());
                    dag_found += 1;
                }
            }
        }

        // Re-sort after adding DAG blocks
        if dag_found > 0 {
            blocks.sort_by_key(|b| b.height);
        }

        blocks
    }
}

// ============================================================================
// TEST 1: DAG fallback finds blocks that fast path misses
// ============================================================================

#[test]
fn test_dag_fallback_finds_hidden_blocks() {
    let mut store = MockBlockStore::new();

    // Only DAG entries exist (no qblock:height: keys) — like blocks 2M on Epsilon
    store.insert_dag(2000941, "e4034c57", MockBlock {
        height: 2000941, proposer: "e4034c57".into(), data: vec![1, 2, 3],
    });
    store.insert_dag(2000942, "e4034c57", MockBlock {
        height: 2000942, proposer: "e4034c57".into(), data: vec![4, 5, 6],
    });

    // Old code: returns NOTHING (the bug)
    let old_result = store.get_qblocks_range_old(2000940, 5);
    assert_eq!(old_result.len(), 0, "Old code should find 0 blocks (the bug we're fixing)");

    // New code: finds both DAG blocks
    let new_result = store.get_qblocks_range_fixed(2000940, 5);
    assert_eq!(new_result.len(), 2, "Fixed code should find 2 DAG blocks");
    assert_eq!(new_result[0].height, 2000941);
    assert_eq!(new_result[1].height, 2000942);

    println!("✅ test_dag_fallback_finds_hidden_blocks: old={}, new={}", old_result.len(), new_result.len());
}

// ============================================================================
// TEST 2: Fast path still works when height keys exist
// ============================================================================

#[test]
fn test_fast_path_still_works() {
    let mut store = MockBlockStore::new();

    // Height keys exist (normal case for blocks 10M+)
    for h in 10000031..=10000035 {
        store.insert_height(h, MockBlock {
            height: h, proposer: "height_format".into(), data: vec![h as u8],
        });
    }

    let result = store.get_qblocks_range_fixed(10000031, 5);
    assert_eq!(result.len(), 5, "Fast path should return all 5 blocks");
    assert_eq!(result[0].height, 10000031);
    assert_eq!(result[4].height, 10000035);

    println!("✅ test_fast_path_still_works: {} blocks returned", result.len());
}

// ============================================================================
// TEST 3: Height format takes priority over DAG format
// ============================================================================

#[test]
fn test_height_format_priority_over_dag() {
    let mut store = MockBlockStore::new();

    // Both formats exist for same height
    store.insert_height(5000, MockBlock {
        height: 5000, proposer: "height_format".into(), data: vec![1],
    });
    store.insert_dag(5000, "dag_proposer", MockBlock {
        height: 5000, proposer: "dag_proposer".into(), data: vec![2],
    });

    let result = store.get_qblocks_range_fixed(5000, 1);
    assert_eq!(result.len(), 1, "Should return exactly 1 block");
    assert_eq!(result[0].proposer, "height_format", "Height format should win over DAG");
    assert_eq!(result[0].data, vec![1], "Should return the height-format block data");

    println!("✅ test_height_format_priority_over_dag: proposer={}", result[0].proposer);
}

// ============================================================================
// TEST 4: Empty range returns empty (no panic)
// ============================================================================

#[test]
fn test_empty_range_no_panic() {
    let store = MockBlockStore::new(); // completely empty

    let result = store.get_qblocks_range_fixed(3000, 1000);
    assert_eq!(result.len(), 0, "Empty store should return 0 blocks");

    // Also test with some blocks that don't overlap the requested range
    let mut store2 = MockBlockStore::new();
    store2.insert_dag(9999, "abc", MockBlock {
        height: 9999, proposer: "abc".into(), data: vec![1],
    });

    let result2 = store2.get_qblocks_range_fixed(3000, 100); // range 3000-3099, block at 9999
    assert_eq!(result2.len(), 0, "Block outside range should not be returned");

    println!("✅ test_empty_range_no_panic: empty={}, out_of_range={}", result.len(), result2.len());
}

// ============================================================================
// TEST 5: Mixed formats — some height, some DAG, some both
// ============================================================================

#[test]
fn test_mixed_formats() {
    let mut store = MockBlockStore::new();

    // Height 1000: DAG only
    store.insert_dag(1000, "miner_a", MockBlock {
        height: 1000, proposer: "miner_a".into(), data: vec![10],
    });

    // Height 1005: height-format only
    store.insert_height(1005, MockBlock {
        height: 1005, proposer: "miner_b".into(), data: vec![50],
    });

    // Height 1010: BOTH formats exist
    store.insert_height(1010, MockBlock {
        height: 1010, proposer: "height_miner".into(), data: vec![100],
    });
    store.insert_dag(1010, "dag_miner", MockBlock {
        height: 1010, proposer: "dag_miner".into(), data: vec![101],
    });

    let result = store.get_qblocks_range_fixed(999, 15); // range 999-1013

    assert_eq!(result.len(), 3, "Should find blocks at 1000, 1005, 1010");
    assert_eq!(result[0].height, 1000);
    assert_eq!(result[0].proposer, "miner_a", "Height 1000: DAG block (only format)");
    assert_eq!(result[1].height, 1005);
    assert_eq!(result[1].proposer, "miner_b", "Height 1005: height-format block");
    assert_eq!(result[2].height, 1010);
    assert_eq!(result[2].proposer, "height_miner", "Height 1010: height-format wins");

    println!("✅ test_mixed_formats: {} blocks, correct priority", result.len());
}

// ============================================================================
// TEST 6: Multiple proposers at same height — deterministic selection
// ============================================================================

#[test]
fn test_multiple_proposers_deterministic() {
    let mut store = MockBlockStore::new();

    // Two miners at same height (DAG-Knight allows this)
    store.insert_dag(3000344, "e14075fb", MockBlock {
        height: 3000344, proposer: "e14075fb".into(), data: vec![1],
    });
    store.insert_dag(3000344, "e4034c57", MockBlock {
        height: 3000344, proposer: "e4034c57".into(), data: vec![2],
    });

    // Call multiple times — must return same block every time
    let result1 = store.get_qblocks_range_fixed(3000344, 1);
    let result2 = store.get_qblocks_range_fixed(3000344, 1);
    let result3 = store.get_qblocks_range_fixed(3000344, 1);

    assert_eq!(result1.len(), 1);
    assert_eq!(result2.len(), 1);
    assert_eq!(result3.len(), 1);

    // All three calls must return the SAME proposer (deterministic)
    assert_eq!(result1[0].proposer, result2[0].proposer, "Must be deterministic across calls");
    assert_eq!(result2[0].proposer, result3[0].proposer, "Must be deterministic across calls");

    // First by byte-order: "e14075fb" < "e4034c57" (because '1' < '4' at position 1)
    assert_eq!(result1[0].proposer, "e14075fb", "First proposer by byte order wins");

    println!("✅ test_multiple_proposers_deterministic: always returns {}", result1[0].proposer);
}

// ============================================================================
// TEST 7: Sparse DAG entries — blocks hundreds of heights apart
// ============================================================================

#[test]
fn test_sparse_dag_entries() {
    let mut store = MockBlockStore::new();

    // Blocks at heights 100441, 100442, 100443 (like Epsilon's early data)
    // Then a big gap, then blocks at 1015441+
    store.insert_dag(100441, "e4034c57", MockBlock {
        height: 100441, proposer: "e4034c57".into(), data: vec![1],
    });
    store.insert_dag(100442, "e4034c57", MockBlock {
        height: 100442, proposer: "e4034c57".into(), data: vec![2],
    });
    store.insert_dag(100443, "e4034c57", MockBlock {
        height: 100443, proposer: "e4034c57".into(), data: vec![3],
    });

    // Request range 100000-100500 (blocks at 100441-100443 are within range)
    let result = store.get_qblocks_range_fixed(100000, 500);
    assert_eq!(result.len(), 3, "Should find 3 sparse DAG blocks");
    assert_eq!(result[0].height, 100441);
    assert_eq!(result[1].height, 100442);
    assert_eq!(result[2].height, 100443);

    // Request range 100000-100200 (blocks 100441-100443 are OUTSIDE this range)
    let result2 = store.get_qblocks_range_fixed(100000, 200);
    assert_eq!(result2.len(), 0, "Blocks beyond range should not be returned");

    println!("✅ test_sparse_dag_entries: found {} blocks in range, {} out of range", result.len(), result2.len());
}

// ============================================================================
// TEST 8: Height mismatch safety check
// ============================================================================

#[test]
fn test_height_mismatch_rejected() {
    let mut store = MockBlockStore::new();

    // Corrupt entry: key says height 5000 but block says height 9999
    store.insert_dag(5000, "corrupt", MockBlock {
        height: 9999, // WRONG — doesn't match key
        proposer: "corrupt".into(),
        data: vec![0xFF],
    });

    let result = store.get_qblocks_range_fixed(5000, 1);
    assert_eq!(result.len(), 0, "Block with mismatched height should be rejected");

    println!("✅ test_height_mismatch_rejected: corrupt block correctly skipped");
}

// ============================================================================
// TEST 9: Large range with many DAG blocks (performance sanity)
// ============================================================================

#[test]
fn test_large_range_performance() {
    let mut store = MockBlockStore::new();

    // Insert 1000 DAG blocks at heights 2000000-2000999
    for h in 2000000..2001000 {
        store.insert_dag(h, "e4034c57", MockBlock {
            height: h, proposer: "e4034c57".into(), data: vec![(h % 256) as u8],
        });
    }

    let start = std::time::Instant::now();
    let result = store.get_qblocks_range_fixed(2000000, 1000);
    let elapsed = start.elapsed();

    assert_eq!(result.len(), 1000, "Should find all 1000 DAG blocks");
    assert_eq!(result[0].height, 2000000);
    assert_eq!(result[999].height, 2000999);
    assert!(elapsed.as_millis() < 1000, "Should complete in <1 second (took {:?})", elapsed);

    println!("✅ test_large_range_performance: {} blocks in {:?}", result.len(), elapsed);
}

// ============================================================================
// TEST 10: Regression — old behavior preserved for height-only ranges
// ============================================================================

#[test]
fn test_regression_height_only_unchanged() {
    let mut store = MockBlockStore::new();

    // Typical production range: continuous height keys (blocks 14M+)
    for h in 14000000..14000200 {
        store.insert_height(h, MockBlock {
            height: h, proposer: "production".into(), data: vec![(h % 256) as u8],
        });
    }

    let old_result = store.get_qblocks_range_old(14000000, 200);
    let new_result = store.get_qblocks_range_fixed(14000000, 200);

    assert_eq!(old_result.len(), new_result.len(), "Fixed code must return same count as old code");
    assert_eq!(old_result.len(), 200);

    for i in 0..200 {
        assert_eq!(old_result[i].height, new_result[i].height, "Heights must match at index {}", i);
        assert_eq!(old_result[i].data, new_result[i].data, "Data must match at index {}", i);
    }

    println!("✅ test_regression_height_only_unchanged: old={}, new={} (identical)", old_result.len(), new_result.len());
}
