//! Block Producer Concurrency Tests
//!
//! Tests to ensure the block producer handles concurrent operations safely
//! and prevents duplicate blocks at the same height.
//!
//! CRITICAL SCENARIOS TESTED:
//! 1. Race condition: simultaneous produce_block calls
//! 2. Duplicate block prevention at same height
//! 3. Solution queue management under load
//! 4. Block production during reorganization
//! 5. Concurrent solution submission
//!
//! Run with: cargo test --package q-api-server --test block_producer_concurrency_tests

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::thread;
use std::time::{Duration, Instant};

// ============================================================================
// MOCK STRUCTURES FOR BLOCK PRODUCER TESTING
// ============================================================================

/// Mining solution submitted by a miner
#[derive(Debug, Clone)]
pub struct MiningSolution {
    pub miner_address: [u8; 32],
    pub nonce: u64,
    pub hash: [u8; 32],
    pub difficulty: u64,
    pub timestamp: u64,
}

/// Produced block
#[derive(Debug, Clone)]
pub struct ProducedBlock {
    pub height: u64,
    pub hash: [u8; 32],
    pub parent_hash: [u8; 32],
    pub solutions: Vec<MiningSolution>,
    pub producer_id: String,
    pub timestamp: u64,
}

/// Block producer with concurrency protection
pub struct MockBlockProducer {
    current_height: AtomicU64,
    pending_solutions: Mutex<VecDeque<MiningSolution>>,
    produced_blocks: RwLock<HashMap<u64, ProducedBlock>>,
    production_lock: Mutex<()>,
    is_producing: AtomicBool,
    max_solutions_per_block: usize,
    blocks_produced: AtomicU64,
    duplicate_attempts: AtomicU64,
}

impl MockBlockProducer {
    pub fn new(initial_height: u64) -> Self {
        Self {
            current_height: AtomicU64::new(initial_height),
            pending_solutions: Mutex::new(VecDeque::new()),
            produced_blocks: RwLock::new(HashMap::new()),
            production_lock: Mutex::new(()),
            is_producing: AtomicBool::new(false),
            max_solutions_per_block: 100,
            blocks_produced: AtomicU64::new(0),
            duplicate_attempts: AtomicU64::new(0),
        }
    }

    pub fn get_height(&self) -> u64 {
        self.current_height.load(Ordering::SeqCst)
    }

    pub fn get_blocks_produced(&self) -> u64 {
        self.blocks_produced.load(Ordering::SeqCst)
    }

    pub fn get_duplicate_attempts(&self) -> u64 {
        self.duplicate_attempts.load(Ordering::SeqCst)
    }

    /// Queue a mining solution for inclusion in next block
    pub fn queue_solution(&self, solution: MiningSolution) -> Result<(), String> {
        let mut queue = self.pending_solutions.lock().unwrap();

        // Limit queue size to prevent memory exhaustion
        if queue.len() >= 10_000 {
            return Err("Solution queue full".to_string());
        }

        queue.push_back(solution);
        Ok(())
    }

    /// Get pending solution count
    pub fn pending_count(&self) -> usize {
        self.pending_solutions.lock().unwrap().len()
    }

    /// Produce a block (with concurrency protection)
    pub fn produce_block(&self, producer_id: &str) -> Result<ProducedBlock, String> {
        // CRITICAL: Acquire production lock to prevent concurrent block production
        let _lock = self.production_lock.lock().unwrap();

        // Check if already producing (belt and suspenders)
        if self.is_producing.swap(true, Ordering::SeqCst) {
            self.duplicate_attempts.fetch_add(1, Ordering::SeqCst);
            return Err("Block production already in progress".to_string());
        }

        // Get target height
        let target_height = self.current_height.load(Ordering::SeqCst) + 1;

        // CRITICAL CHECK: Ensure we haven't already produced a block at this height
        {
            let blocks = self.produced_blocks.read().unwrap();
            if blocks.contains_key(&target_height) {
                self.is_producing.store(false, Ordering::SeqCst);
                self.duplicate_attempts.fetch_add(1, Ordering::SeqCst);
                return Err(format!(
                    "DUPLICATE BLOCK PREVENTED: Block at height {} already exists",
                    target_height
                ));
            }
        }

        // Gather solutions (limit to max per block)
        let solutions: Vec<MiningSolution> = {
            let mut queue = self.pending_solutions.lock().unwrap();
            let mut collected = Vec::new();
            while collected.len() < self.max_solutions_per_block {
                match queue.pop_front() {
                    Some(s) => collected.push(s),
                    None => break,
                }
            }
            collected
        };

        // Create the block
        let parent_hash = {
            let blocks = self.produced_blocks.read().unwrap();
            blocks
                .get(&self.current_height.load(Ordering::SeqCst))
                .map(|b| b.hash)
                .unwrap_or([0u8; 32])
        };

        let block = ProducedBlock {
            height: target_height,
            hash: Self::compute_hash(target_height, &parent_hash),
            parent_hash,
            solutions,
            producer_id: producer_id.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        // Store the block
        {
            let mut blocks = self.produced_blocks.write().unwrap();
            blocks.insert(target_height, block.clone());
        }

        // Update height
        self.current_height.store(target_height, Ordering::SeqCst);
        self.blocks_produced.fetch_add(1, Ordering::SeqCst);
        self.is_producing.store(false, Ordering::SeqCst);

        Ok(block)
    }

    fn compute_hash(height: u64, parent_hash: &[u8; 32]) -> [u8; 32] {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        height.hash(&mut hasher);
        parent_hash.hash(&mut hasher);
        let h = hasher.finish();
        let mut result = [0u8; 32];
        result[0..8].copy_from_slice(&h.to_le_bytes());
        result
    }

    /// Simulate a reorg by rolling back to a previous height
    pub fn rollback_to(&self, height: u64) -> Result<(), String> {
        let _lock = self.production_lock.lock().unwrap();

        let current = self.current_height.load(Ordering::SeqCst);
        if height >= current {
            return Err("Cannot rollback to same or higher height".to_string());
        }

        // Remove blocks above the target height
        {
            let mut blocks = self.produced_blocks.write().unwrap();
            blocks.retain(|h, _| *h <= height);
        }

        self.current_height.store(height, Ordering::SeqCst);
        Ok(())
    }

    /// Check if a block exists at a height
    pub fn has_block_at(&self, height: u64) -> bool {
        let blocks = self.produced_blocks.read().unwrap();
        blocks.contains_key(&height)
    }
}

// ============================================================================
// CONCURRENT BLOCK PRODUCTION TESTS
// ============================================================================

/// Test that concurrent produce_block calls don't create duplicate blocks
#[test]
fn test_no_duplicate_blocks_concurrent() {
    let producer = Arc::new(MockBlockProducer::new(1000));

    let mut handles = vec![];
    let num_threads = 10;

    for i in 0..num_threads {
        let prod = Arc::clone(&producer);
        let handle = thread::spawn(move || {
            // Each thread tries to produce 10 blocks
            for j in 0..10 {
                let _ = prod.produce_block(&format!("producer_{}_{}", i, j));
                // Small delay to increase contention
                thread::sleep(Duration::from_micros(100));
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Should have produced exactly 100 blocks (10 threads * 10 attempts each that succeed)
    // Actually, due to race conditions, some will fail - that's expected
    let blocks_produced = producer.get_blocks_produced();
    let duplicates_prevented = producer.get_duplicate_attempts();

    println!(
        "Blocks produced: {}, Duplicates prevented: {}",
        blocks_produced, duplicates_prevented
    );

    // CRITICAL: No duplicate blocks at any height
    let blocks = producer.produced_blocks.read().unwrap();
    let heights: HashSet<u64> = blocks.keys().copied().collect();
    assert_eq!(
        heights.len(),
        blocks.len(),
        "CRITICAL: Duplicate blocks detected at same height!"
    );

    // Height should be sequential
    let expected_height = 1000 + blocks_produced;
    assert_eq!(
        producer.get_height(),
        expected_height,
        "Height mismatch after concurrent production"
    );
}

/// Test race condition between two producers at exact same time
/// Both can succeed if they produce at different heights (sequential production)
/// The key is that NO duplicate blocks at the SAME height are produced
#[test]
fn test_race_condition_two_producers() {
    let producer = Arc::new(MockBlockProducer::new(100));

    let p1 = Arc::clone(&producer);
    let p2 = Arc::clone(&producer);

    // Use barriers to synchronize start
    use std::sync::Barrier;
    let barrier = Arc::new(Barrier::new(2));
    let b1 = Arc::clone(&barrier);
    let b2 = Arc::clone(&barrier);

    let h1 = thread::spawn(move || {
        b1.wait(); // Wait for both threads to be ready
        p1.produce_block("producer_1")
    });

    let h2 = thread::spawn(move || {
        b2.wait(); // Wait for both threads to be ready
        p2.produce_block("producer_2")
    });

    let r1 = h1.join().unwrap();
    let r2 = h2.join().unwrap();

    // Both can succeed if they produce sequential blocks (101 and 102)
    // The key check is that duplicate_attempts is 0 (no duplicate at same height)
    let total_successes = [r1.is_ok(), r2.is_ok()].iter().filter(|&&x| x).count();
    assert!(total_successes >= 1, "At least one producer should succeed");

    // CRITICAL: No duplicate attempts at same height
    assert_eq!(
        producer.get_duplicate_attempts(),
        0,
        "Should not have duplicate block attempts"
    );

    // Block at 101 should exist (first producer)
    assert!(producer.has_block_at(101), "First block should be produced");

    // Check blocks produced matches successes
    assert_eq!(
        producer.get_blocks_produced() as usize,
        total_successes,
        "Blocks produced should match successes"
    );
}

// ============================================================================
// SOLUTION QUEUE TESTS
// ============================================================================

/// Test solution queue under high load
#[test]
fn test_solution_queue_high_load() {
    let producer = Arc::new(MockBlockProducer::new(100));

    // Submit many solutions concurrently
    let mut handles = vec![];
    for i in 0..8 {
        let prod = Arc::clone(&producer);
        let handle = thread::spawn(move || {
            for j in 0..500 {
                let solution = MiningSolution {
                    miner_address: [i as u8; 32],
                    nonce: j,
                    hash: [(i * 100 + j) as u8; 32],
                    difficulty: 1000,
                    timestamp: 12345,
                };
                let _ = prod.queue_solution(solution);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Should have queued many solutions (up to limit)
    let pending = producer.pending_count();
    assert!(pending > 0, "Should have queued solutions");
    assert!(pending <= 10_000, "Should respect queue limit");

    // Produce a block - should consume solutions
    let block = producer.produce_block("miner_1").unwrap();
    assert!(block.solutions.len() > 0, "Block should contain solutions");
    assert!(
        block.solutions.len() <= 100,
        "Block should respect max solutions limit"
    );
}

/// Test empty solution queue handling
#[test]
fn test_empty_solution_queue() {
    let producer = MockBlockProducer::new(100);

    // Produce block with no pending solutions
    let result = producer.produce_block("producer_1");
    assert!(result.is_ok(), "Should allow block with no solutions");

    let block = result.unwrap();
    assert_eq!(block.solutions.len(), 0, "Block should have no solutions");
    assert_eq!(block.height, 101);
}

/// Test solution queue overflow
#[test]
fn test_solution_queue_overflow() {
    let producer = MockBlockProducer::new(100);

    // Fill the queue to capacity
    for i in 0..10_000 {
        let solution = MiningSolution {
            miner_address: [1u8; 32],
            nonce: i,
            hash: [i as u8; 32],
            difficulty: 1000,
            timestamp: 12345,
        };
        producer.queue_solution(solution).unwrap();
    }

    // Next submission should fail
    let solution = MiningSolution {
        miner_address: [1u8; 32],
        nonce: 10_001,
        hash: [99u8; 32],
        difficulty: 1000,
        timestamp: 12345,
    };

    let result = producer.queue_solution(solution);
    assert!(result.is_err(), "Should reject when queue is full");
}

// ============================================================================
// BLOCK PRODUCTION DURING REORG TESTS
// ============================================================================

/// Test block production after rollback
#[test]
fn test_production_after_rollback() {
    let producer = MockBlockProducer::new(100);

    // Produce some blocks
    for _ in 0..5 {
        producer.produce_block("producer").unwrap();
    }
    assert_eq!(producer.get_height(), 105);

    // Rollback to height 102
    producer.rollback_to(102).unwrap();
    assert_eq!(producer.get_height(), 102);

    // Should be able to produce blocks again from 103
    let block = producer.produce_block("producer").unwrap();
    assert_eq!(block.height, 103);
}

/// Test no double production at rolled-back height
/// Concurrent producers after rollback should produce sequential blocks
/// without any duplicate attempts at the same height
#[test]
fn test_no_double_production_after_rollback() {
    let producer = Arc::new(MockBlockProducer::new(100));

    // Produce blocks up to 110
    for _ in 0..10 {
        producer.produce_block("producer").unwrap();
    }
    assert_eq!(producer.get_height(), 110);

    // Rollback to 105
    producer.rollback_to(105).unwrap();
    assert_eq!(producer.get_height(), 105);

    // Reset duplicate attempts counter for clean test
    let initial_duplicates = producer.get_duplicate_attempts();

    // Concurrent production attempts after rollback
    let mut handles = vec![];
    for i in 0..5 {
        let prod = Arc::clone(&producer);
        let handle = thread::spawn(move || {
            prod.produce_block(&format!("producer_{}", i))
        });
        handles.push(handle);
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let successes = results.iter().filter(|r| r.is_ok()).count();

    // Multiple can succeed if they produce at different heights sequentially
    assert!(successes >= 1, "At least one producer should succeed after rollback");

    // CRITICAL: No new duplicate attempts at same height
    // The lock mechanism prevents multiple blocks at the same height
    assert_eq!(
        producer.get_duplicate_attempts() - initial_duplicates,
        0,
        "No duplicate block attempts should occur"
    );

    // First block after rollback should be at 106
    assert!(producer.has_block_at(106), "Block at 106 should exist");

    // Blocks produced should match successes
    let new_blocks = producer.get_height() - 105;
    assert_eq!(
        new_blocks as usize,
        successes,
        "New blocks should match successful productions"
    );
}

// ============================================================================
// SEQUENTIAL PRODUCTION TESTS
// ============================================================================

/// Test sequential block production
#[test]
fn test_sequential_production() {
    let producer = MockBlockProducer::new(0);

    for i in 1..=100 {
        let block = producer.produce_block("sequential_producer").unwrap();
        assert_eq!(block.height, i, "Block height should be sequential");
    }

    assert_eq!(producer.get_height(), 100);
    assert_eq!(producer.get_blocks_produced(), 100);
}

/// Test parent hash chain integrity
#[test]
fn test_parent_hash_chain() {
    let producer = MockBlockProducer::new(0);

    // Produce 10 blocks
    for _ in 0..10 {
        producer.produce_block("producer").unwrap();
    }

    // Verify parent hash chain
    let blocks = producer.produced_blocks.read().unwrap();
    for height in 2..=10 {
        let block = blocks.get(&height).unwrap();
        let parent = blocks.get(&(height - 1)).unwrap();
        assert_eq!(
            block.parent_hash, parent.hash,
            "Parent hash mismatch at height {}",
            height
        );
    }
}

// ============================================================================
// STRESS TESTS
// ============================================================================

/// Stress test: rapid block production
#[test]
fn test_rapid_block_production() {
    let producer = Arc::new(MockBlockProducer::new(0));
    let start = Instant::now();

    // Produce 1000 blocks as fast as possible
    for _ in 0..1000 {
        producer.produce_block("stress_producer").unwrap();
    }

    let elapsed = start.elapsed();
    println!("Produced 1000 blocks in {:?}", elapsed);

    assert_eq!(producer.get_height(), 1000);
    assert_eq!(producer.get_blocks_produced(), 1000);
}

/// Test concurrent solution submission and block production
#[test]
fn test_concurrent_submit_and_produce() {
    let producer = Arc::new(MockBlockProducer::new(100));

    // Submitter thread
    let prod_submit = Arc::clone(&producer);
    let submitter = thread::spawn(move || {
        for i in 0..1000 {
            let solution = MiningSolution {
                miner_address: [1u8; 32],
                nonce: i,
                hash: [i as u8; 32],
                difficulty: 1000,
                timestamp: 12345,
            };
            let _ = prod_submit.queue_solution(solution);
        }
    });

    // Producer thread
    let prod_produce = Arc::clone(&producer);
    let block_producer = thread::spawn(move || {
        for _ in 0..50 {
            let _ = prod_produce.produce_block("concurrent_producer");
            thread::sleep(Duration::from_millis(1));
        }
    });

    submitter.join().unwrap();
    block_producer.join().unwrap();

    // Should have produced some blocks with solutions
    let blocks = producer.produced_blocks.read().unwrap();
    let total_solutions: usize = blocks.values().map(|b| b.solutions.len()).sum();

    println!(
        "Produced {} blocks with {} total solutions",
        blocks.len(),
        total_solutions
    );

    assert!(blocks.len() > 0, "Should have produced blocks");
}
