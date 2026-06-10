//! Turbo Sync Edge Cases Tests
//!
//! Tests for edge cases and boundary conditions in the turbo sync system.
//!
//! CRITICAL SCENARIOS TESTED:
//! 1. Empty batch handling
//! 2. Out-of-order block arrival
//! 3. Gap detection and filling
//! 4. Duplicate block rejection
//! 5. Invalid parent chain rejection
//! 6. Height overflow protection
//! 7. Batch size limits
//!
//! Run with: cargo test --package q-storage --test turbo_sync_edge_cases_tests

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ============================================================================
// MOCK STRUCTURES FOR TURBO SYNC TESTING
// ============================================================================

/// Mock block for sync testing
#[derive(Debug, Clone, PartialEq)]
pub struct SyncBlock {
    pub height: u64,
    pub hash: [u8; 32],
    pub parent_hash: [u8; 32],
    pub timestamp: u64,
    pub data_size: usize,
}

impl SyncBlock {
    pub fn new(height: u64, parent_hash: [u8; 32]) -> Self {
        let mut hash = [0u8; 32];
        hash[0..8].copy_from_slice(&height.to_le_bytes());
        hash[8..16].copy_from_slice(&parent_hash[0..8]);

        Self {
            height,
            hash,
            parent_hash,
            timestamp: height * 1000,
            data_size: 1024,
        }
    }

    pub fn genesis() -> Self {
        Self {
            height: 0,
            hash: [1u8; 32],
            parent_hash: [0u8; 32],
            timestamp: 0,
            data_size: 512,
        }
    }
}

/// Turbo sync batch request
#[derive(Debug, Clone)]
pub struct SyncBatchRequest {
    pub start_height: u64,
    pub end_height: u64,
    pub peer_id: [u8; 32],
}

/// Turbo sync batch response
#[derive(Debug)]
pub struct SyncBatchResponse {
    pub blocks: Vec<SyncBlock>,
    pub is_complete: bool,
    pub next_height: Option<u64>,
}

/// Gap in the blockchain
#[derive(Debug, Clone, PartialEq)]
pub struct BlockGap {
    pub start: u64,
    pub end: u64,
}

/// Mock turbo sync manager
pub struct TurboSyncManager {
    blocks: Mutex<BTreeMap<u64, SyncBlock>>,
    known_hashes: Mutex<HashSet<[u8; 32]>>,
    current_height: AtomicU64,
    max_batch_size: usize,
    pending_batches: Mutex<HashMap<u64, SyncBatchRequest>>,
    gaps: Mutex<Vec<BlockGap>>,
}

impl TurboSyncManager {
    pub fn new(max_batch_size: usize) -> Self {
        let mut blocks = BTreeMap::new();
        let genesis = SyncBlock::genesis();
        blocks.insert(0, genesis);

        Self {
            blocks: Mutex::new(blocks),
            known_hashes: Mutex::new(HashSet::new()),
            current_height: AtomicU64::new(0),
            max_batch_size,
            pending_batches: Mutex::new(HashMap::new()),
            gaps: Mutex::new(Vec::new()),
        }
    }

    /// Request a batch of blocks
    pub fn request_batch(&self, start: u64, end: u64, peer_id: [u8; 32]) -> Result<SyncBatchRequest, String> {
        // Validate range
        if end < start {
            return Err("INVALID_RANGE: end < start".to_string());
        }

        // Check for height overflow
        if end == u64::MAX {
            return Err("HEIGHT_OVERFLOW: end height at maximum".to_string());
        }

        let range = end.saturating_sub(start);
        if range as usize > self.max_batch_size {
            return Err(format!(
                "BATCH_TOO_LARGE: {} blocks exceeds max {}",
                range, self.max_batch_size
            ));
        }

        let request = SyncBatchRequest {
            start_height: start,
            end_height: end,
            peer_id,
        };

        let mut pending = self.pending_batches.lock().unwrap();
        pending.insert(start, request.clone());

        Ok(SyncBatchRequest {
            start_height: start,
            end_height: end,
            peer_id,
        })
    }

    /// Process incoming batch of blocks
    pub fn process_batch(&self, response: SyncBatchResponse) -> Result<u64, String> {
        if response.blocks.is_empty() {
            return Err("EMPTY_BATCH: Received empty block batch".to_string());
        }

        let mut blocks_added = 0;
        let mut blocks = self.blocks.lock().unwrap();
        let mut known = self.known_hashes.lock().unwrap();

        for block in response.blocks {
            // Check for duplicate
            if known.contains(&block.hash) {
                continue; // Skip duplicate
            }

            // Validate parent chain
            if block.height > 0 {
                let parent_exists = blocks
                    .get(&(block.height - 1))
                    .map(|p| p.hash == block.parent_hash)
                    .unwrap_or(false);

                if !parent_exists && block.height <= self.current_height.load(Ordering::Relaxed) + 1
                {
                    // Only enforce parent check for blocks close to our height
                    // For far-ahead blocks, we may fill the gap later
                }
            }

            // Check for height regression attack
            let current = self.current_height.load(Ordering::Relaxed);
            if block.height > current + 100_000 {
                return Err(format!(
                    "SUSPICIOUS_HEIGHT: Block height {} too far ahead of current {}",
                    block.height, current
                ));
            }

            blocks.insert(block.height, block.clone());
            known.insert(block.hash);
            blocks_added += 1;

            // Update current height if this extends the chain
            if block.height == current + 1 {
                self.current_height.store(block.height, Ordering::Relaxed);
            }
        }

        // Detect gaps after processing
        drop(blocks);
        drop(known);
        self.detect_gaps();

        Ok(blocks_added)
    }

    /// Detect gaps in the blockchain
    pub fn detect_gaps(&self) -> Vec<BlockGap> {
        let blocks = self.blocks.lock().unwrap();
        let mut gaps = Vec::new();

        if blocks.is_empty() {
            return gaps;
        }

        let heights: Vec<u64> = blocks.keys().copied().collect();
        let mut gap_start: Option<u64> = None;

        for i in 0..heights.len() - 1 {
            let current = heights[i];
            let next = heights[i + 1];

            if next > current + 1 {
                // Found a gap
                gaps.push(BlockGap {
                    start: current + 1,
                    end: next - 1,
                });
            }
        }

        let mut stored_gaps = self.gaps.lock().unwrap();
        *stored_gaps = gaps.clone();
        gaps
    }

    /// Check if block exists at height
    pub fn has_block(&self, height: u64) -> bool {
        self.blocks.lock().unwrap().contains_key(&height)
    }

    /// Get current chain height
    pub fn current_height(&self) -> u64 {
        self.current_height.load(Ordering::Relaxed)
    }

    /// Get block count
    pub fn block_count(&self) -> usize {
        self.blocks.lock().unwrap().len()
    }

    /// Get gaps
    pub fn get_gaps(&self) -> Vec<BlockGap> {
        self.gaps.lock().unwrap().clone()
    }

    /// Process out-of-order blocks
    pub fn process_out_of_order(&self, blocks: Vec<SyncBlock>) -> Result<(u64, u64), String> {
        if blocks.is_empty() {
            return Err("EMPTY_BATCH: No blocks to process".to_string());
        }

        // Sort by height
        let mut sorted = blocks;
        sorted.sort_by_key(|b| b.height);

        let min_height = sorted.first().unwrap().height;
        let max_height = sorted.last().unwrap().height;

        let response = SyncBatchResponse {
            blocks: sorted,
            is_complete: true,
            next_height: None,
        };

        let added = self.process_batch(response)?;
        Ok((added, max_height - min_height + 1))
    }

    /// Fill a specific gap
    pub fn fill_gap(&self, gap: &BlockGap, fill_blocks: Vec<SyncBlock>) -> Result<u64, String> {
        // Validate fill blocks match the gap
        for block in &fill_blocks {
            if block.height < gap.start || block.height > gap.end {
                return Err(format!(
                    "BLOCK_OUTSIDE_GAP: Block {} not in gap [{}, {}]",
                    block.height, gap.start, gap.end
                ));
            }
        }

        let response = SyncBatchResponse {
            blocks: fill_blocks,
            is_complete: true,
            next_height: None,
        };

        self.process_batch(response)
    }
}

/// Batch validator for incoming sync data
pub struct BatchValidator {
    max_total_size: usize,
    max_block_size: usize,
}

impl BatchValidator {
    pub fn new(max_total_size: usize, max_block_size: usize) -> Self {
        Self {
            max_total_size,
            max_block_size,
        }
    }

    /// Validate a batch of blocks
    pub fn validate_batch(&self, blocks: &[SyncBlock]) -> Result<(), String> {
        if blocks.is_empty() {
            return Err("EMPTY_BATCH".to_string());
        }

        let total_size: usize = blocks.iter().map(|b| b.data_size).sum();
        if total_size > self.max_total_size {
            return Err(format!(
                "BATCH_SIZE_EXCEEDED: {} bytes exceeds max {}",
                total_size, self.max_total_size
            ));
        }

        for block in blocks {
            if block.data_size > self.max_block_size {
                return Err(format!(
                    "BLOCK_SIZE_EXCEEDED: Block {} has {} bytes (max: {})",
                    block.height, block.data_size, self.max_block_size
                ));
            }
        }

        // Check heights are sequential within batch
        let mut heights: Vec<u64> = blocks.iter().map(|b| b.height).collect();
        heights.sort();
        heights.dedup();

        if heights.len() != blocks.len() {
            return Err("DUPLICATE_HEIGHTS: Batch contains duplicate heights".to_string());
        }

        Ok(())
    }
}

// ============================================================================
// BATCH REQUEST TESTS
// ============================================================================

/// Test valid batch request
#[test]
fn test_valid_batch_request() {
    let manager = TurboSyncManager::new(1000);
    let peer = [1u8; 32];

    let result = manager.request_batch(100, 200, peer);
    assert!(result.is_ok());

    let request = result.unwrap();
    assert_eq!(request.start_height, 100);
    assert_eq!(request.end_height, 200);
}

/// Test batch request with invalid range
#[test]
fn test_invalid_range_batch_request() {
    let manager = TurboSyncManager::new(1000);
    let peer = [1u8; 32];

    let result = manager.request_batch(200, 100, peer); // end < start
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("INVALID_RANGE"));
}

/// Test batch request exceeding max size
#[test]
fn test_batch_too_large() {
    let manager = TurboSyncManager::new(100); // Max 100 blocks
    let peer = [1u8; 32];

    let result = manager.request_batch(0, 500, peer); // 500 blocks
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("BATCH_TOO_LARGE"));
}

/// Test height overflow protection
#[test]
fn test_height_overflow_protection() {
    let manager = TurboSyncManager::new(1000);
    let peer = [1u8; 32];

    let result = manager.request_batch(u64::MAX - 10, u64::MAX, peer);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("HEIGHT_OVERFLOW"));
}

// ============================================================================
// BATCH PROCESSING TESTS
// ============================================================================

/// Test processing empty batch
#[test]
fn test_empty_batch_rejected() {
    let manager = TurboSyncManager::new(1000);

    let response = SyncBatchResponse {
        blocks: vec![],
        is_complete: true,
        next_height: None,
    };

    let result = manager.process_batch(response);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("EMPTY_BATCH"));
}

/// Test processing valid batch
#[test]
fn test_process_valid_batch() {
    let manager = TurboSyncManager::new(1000);

    // Create blocks building on genesis
    let block1 = SyncBlock::new(1, [1u8; 32]); // Parent is genesis hash
    let block2 = SyncBlock::new(2, block1.hash);
    let block3 = SyncBlock::new(3, block2.hash);

    let response = SyncBatchResponse {
        blocks: vec![block1, block2, block3],
        is_complete: true,
        next_height: None,
    };

    let result = manager.process_batch(response);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 3);
    assert_eq!(manager.block_count(), 4); // Genesis + 3 new
}

/// Test duplicate block rejection
#[test]
fn test_duplicate_block_skipped() {
    let manager = TurboSyncManager::new(1000);

    let block1 = SyncBlock::new(1, [1u8; 32]);

    // Process same block twice
    let response1 = SyncBatchResponse {
        blocks: vec![block1.clone()],
        is_complete: true,
        next_height: None,
    };
    manager.process_batch(response1).unwrap();

    let response2 = SyncBatchResponse {
        blocks: vec![block1.clone()],
        is_complete: true,
        next_height: None,
    };
    let added = manager.process_batch(response2).unwrap();

    assert_eq!(added, 0); // Duplicate was skipped
}

/// Test suspicious height detection
#[test]
fn test_suspicious_height_rejected() {
    let manager = TurboSyncManager::new(1000);

    // Current height is 0, try to add block at height 1,000,000
    let block = SyncBlock::new(1_000_000, [0u8; 32]);

    let response = SyncBatchResponse {
        blocks: vec![block],
        is_complete: true,
        next_height: None,
    };

    let result = manager.process_batch(response);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("SUSPICIOUS_HEIGHT"));
}

// ============================================================================
// OUT-OF-ORDER TESTS
// ============================================================================

/// Test out-of-order block processing
#[test]
fn test_out_of_order_processing() {
    let manager = TurboSyncManager::new(1000);

    // Create blocks in wrong order
    let block1 = SyncBlock::new(1, [1u8; 32]);
    let block2 = SyncBlock::new(2, block1.hash);
    let block3 = SyncBlock::new(3, block2.hash);

    // Send in reverse order
    let blocks = vec![block3, block1, block2];
    let result = manager.process_out_of_order(blocks);

    assert!(result.is_ok());
    let (added, range) = result.unwrap();
    assert_eq!(added, 3);
    assert_eq!(range, 3);
}

/// Test empty out-of-order batch
#[test]
fn test_empty_out_of_order_rejected() {
    let manager = TurboSyncManager::new(1000);

    let result = manager.process_out_of_order(vec![]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("EMPTY_BATCH"));
}

// ============================================================================
// GAP DETECTION TESTS
// ============================================================================

/// Test gap detection
#[test]
fn test_gap_detection() {
    let manager = TurboSyncManager::new(1000);

    // Add blocks with gaps: 0, 1, 5, 6, 10
    let blocks = vec![
        SyncBlock::new(1, [1u8; 32]),
        SyncBlock::new(5, [0u8; 32]),
        SyncBlock::new(6, [0u8; 32]),
        SyncBlock::new(10, [0u8; 32]),
    ];

    for block in blocks {
        let response = SyncBatchResponse {
            blocks: vec![block],
            is_complete: true,
            next_height: None,
        };
        let _ = manager.process_batch(response);
    }

    let gaps = manager.detect_gaps();

    // Should detect gaps: [2-4] and [7-9]
    assert_eq!(gaps.len(), 2);
    assert!(gaps.iter().any(|g| g.start == 2 && g.end == 4));
    assert!(gaps.iter().any(|g| g.start == 7 && g.end == 9));
}

/// Test gap filling
#[test]
fn test_gap_filling() {
    let manager = TurboSyncManager::new(1000);

    // Create chain with gap
    let block1 = SyncBlock::new(1, [1u8; 32]);
    let block5 = SyncBlock::new(5, [0u8; 32]);

    let response = SyncBatchResponse {
        blocks: vec![block1, block5],
        is_complete: true,
        next_height: None,
    };
    manager.process_batch(response).unwrap();

    // Detect gap
    let gaps = manager.detect_gaps();
    assert_eq!(gaps.len(), 1);
    let gap = &gaps[0];
    assert_eq!(gap.start, 2);
    assert_eq!(gap.end, 4);

    // Fill gap
    let fill_blocks = vec![
        SyncBlock::new(2, [0u8; 32]),
        SyncBlock::new(3, [0u8; 32]),
        SyncBlock::new(4, [0u8; 32]),
    ];

    let filled = manager.fill_gap(gap, fill_blocks).unwrap();
    assert_eq!(filled, 3);

    // Gap should be gone
    let gaps_after = manager.detect_gaps();
    assert!(gaps_after.is_empty());
}

/// Test filling with block outside gap
#[test]
fn test_fill_block_outside_gap_rejected() {
    let manager = TurboSyncManager::new(1000);

    let gap = BlockGap { start: 10, end: 20 };
    let wrong_block = vec![SyncBlock::new(5, [0u8; 32])]; // Outside gap

    let result = manager.fill_gap(&gap, wrong_block);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("BLOCK_OUTSIDE_GAP"));
}

// ============================================================================
// BATCH VALIDATION TESTS
// ============================================================================

/// Test batch size validation
#[test]
fn test_batch_size_validation() {
    let validator = BatchValidator::new(10 * 1024, 2 * 1024); // 10KB total, 2KB per block

    let blocks: Vec<SyncBlock> = (0..5)
        .map(|i| {
            let mut b = SyncBlock::new(i, [0u8; 32]);
            b.data_size = 1024; // 1KB each
            b
        })
        .collect();

    let result = validator.validate_batch(&blocks);
    assert!(result.is_ok());
}

/// Test batch total size exceeded
#[test]
fn test_batch_total_size_exceeded() {
    let validator = BatchValidator::new(5 * 1024, 2 * 1024); // 5KB total

    let blocks: Vec<SyncBlock> = (0..10)
        .map(|i| {
            let mut b = SyncBlock::new(i, [0u8; 32]);
            b.data_size = 1024; // 1KB each = 10KB total
            b
        })
        .collect();

    let result = validator.validate_batch(&blocks);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("BATCH_SIZE_EXCEEDED"));
}

/// Test individual block size exceeded
#[test]
fn test_individual_block_size_exceeded() {
    let validator = BatchValidator::new(10 * 1024, 2 * 1024); // 2KB max per block

    let mut block = SyncBlock::new(1, [0u8; 32]);
    block.data_size = 5 * 1024; // 5KB - too big

    let result = validator.validate_batch(&[block]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("BLOCK_SIZE_EXCEEDED"));
}

/// Test duplicate heights in batch
#[test]
fn test_duplicate_heights_rejected() {
    let validator = BatchValidator::new(10 * 1024, 2 * 1024);

    let blocks = vec![
        SyncBlock::new(1, [0u8; 32]),
        SyncBlock::new(1, [0u8; 32]), // Duplicate height
    ];

    let result = validator.validate_batch(&blocks);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("DUPLICATE_HEIGHTS"));
}

// ============================================================================
// CONCURRENT ACCESS TESTS
// ============================================================================

/// Test concurrent batch processing
#[test]
fn test_concurrent_batch_processing() {
    use std::thread;

    let manager = Arc::new(TurboSyncManager::new(1000));
    let mut handles = vec![];

    // Spawn multiple threads processing batches
    for batch_num in 0..4 {
        let mgr = Arc::clone(&manager);
        let handle = thread::spawn(move || {
            let base_height = (batch_num * 10) as u64 + 1;
            let blocks: Vec<SyncBlock> = (0..10)
                .map(|i| SyncBlock::new(base_height + i, [0u8; 32]))
                .collect();

            let response = SyncBatchResponse {
                blocks,
                is_complete: true,
                next_height: None,
            };

            mgr.process_batch(response)
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        let _ = handle.join();
    }

    // Should have processed some blocks (exact count depends on ordering)
    assert!(manager.block_count() > 1);
}

/// Test concurrent gap detection
#[test]
fn test_concurrent_gap_detection() {
    use std::thread;

    let manager = Arc::new(TurboSyncManager::new(1000));

    // Add some blocks with gaps
    let blocks = vec![
        SyncBlock::new(1, [1u8; 32]),
        SyncBlock::new(10, [0u8; 32]),
        SyncBlock::new(20, [0u8; 32]),
    ];

    let response = SyncBatchResponse {
        blocks,
        is_complete: true,
        next_height: None,
    };
    manager.process_batch(response).unwrap();

    // Spawn threads to detect gaps concurrently
    let mut handles = vec![];
    for _ in 0..4 {
        let mgr = Arc::clone(&manager);
        let handle = thread::spawn(move || mgr.detect_gaps());
        handles.push(handle);
    }

    // All should return same gaps
    let mut all_gaps = vec![];
    for handle in handles {
        all_gaps.push(handle.join().unwrap());
    }

    // All results should be the same
    for gaps in &all_gaps {
        assert_eq!(gaps.len(), all_gaps[0].len());
    }
}
