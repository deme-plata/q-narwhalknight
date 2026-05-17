/// Ordered block buffer for height-ordered consensus writes
///
/// Ensures blocks are written to storage in sequential height order,
/// preventing out-of-order writes that would break consensus.
///
/// Expert validation: Kimi AI, ChatGPT, DeepSeek (100% consensus)
/// "Height ordering is CRITICAL - out-of-order writes cause consensus failures"

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use q_types::block::QBlock;

/// Wrapper for QBlock with height-based ordering
#[derive(Debug, Clone)]
pub struct OrderedBlock {
    pub height: u64,
    pub block: QBlock,
}

impl PartialEq for OrderedBlock {
    fn eq(&self, other: &Self) -> bool {
        self.height == other.height
    }
}

impl Eq for OrderedBlock {}

impl PartialOrd for OrderedBlock {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedBlock {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap (lowest height first)
        other.height.cmp(&self.height)
    }
}

/// Buffer that enforces height-ordered block delivery
///
/// Blocks arrive via gossipsub in arbitrary order. This buffer
/// reorders them and only releases blocks when they're sequential.
///
/// Example:
/// - Receive blocks: 103, 101, 102
/// - Output order: 101, 102, 103
#[derive(Debug)]
pub struct OrderedBlockBuffer {
    /// Min-heap: lowest height at top
    queue: BinaryHeap<OrderedBlock>,

    /// Next expected height to deliver
    next_expected: u64,

    /// Maximum gap allowed before backpressure
    max_gap: u64,
}

impl OrderedBlockBuffer {
    /// Create new buffer starting at given height
    pub fn new(start_height: u64, max_gap: u64) -> Self {
        Self {
            queue: BinaryHeap::new(),
            next_expected: start_height,
            max_gap,
        }
    }

    /// Insert block into buffer (may be out of order)
    ///
    /// Returns error if block height exceeds max_gap from expected height
    /// (backpressure mechanism to prevent unbounded memory growth)
    pub fn insert(&mut self, block: QBlock) -> Result<(), String> {
        let height = block.header.height;

        // Backpressure: reject if gap too large
        if height > self.next_expected + self.max_gap {
            return Err(format!(
                "Block height {} exceeds max gap from expected {} (max_gap={})",
                height, self.next_expected, self.max_gap
            ));
        }

        // Skip if already processed
        if height < self.next_expected {
            return Ok(()); // Duplicate, silently ignore
        }

        self.queue.push(OrderedBlock { height, block });
        Ok(())
    }

    /// Take next sequential block if available
    ///
    /// Returns Some(block) if next expected height is ready
    /// Returns None if there's a gap in the sequence
    pub fn take_next_ready(&mut self) -> Option<QBlock> {
        if let Some(ordered) = self.queue.peek() {
            if ordered.height == self.next_expected {
                self.next_expected += 1;
                return Some(self.queue.pop().unwrap().block);
            }
        }
        None
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Next expected height
    pub fn next_expected_height(&self) -> u64 {
        self.next_expected
    }

    /// Number of blocks buffered
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Gap to next available block (0 if ready, >0 if waiting)
    pub fn gap_size(&self) -> u64 {
        self.queue.peek()
            .map(|b| b.height.saturating_sub(self.next_expected))
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use q_types::block::{BlockHeader, VDFProof, QuantumMetadata};

    fn create_test_block(height: u64) -> QBlock {
        QBlock {
            header: BlockHeader {
                height,
                phase: 11,
                network_id: "testnet-phase11".to_string(),
                prev_block_hash: [0; 32],
                solutions_root: [0; 32],
                tx_root: [0; 32],
                state_root: [0; 32],
                timestamp: 0,
                dag_round: height,
                vdf_proof: VDFProof {
                    output: vec![],
                    proof: vec![],
                    iterations: 0,
                },
                anchor_validator: None,
                proposer: "test".to_string(),
                producer_id: 0,
                difficulty: 1,
            },
            mining_solutions: vec![],
            dag_parents: vec![],
            quantum_metadata: QuantumMetadata {
                vdf_iterations: 0,
                entropy_source: vec![],
                quantum_signature: vec![],
            },
            transactions: vec![],
            balance_updates: vec![],
            size_bytes: 0,
        }
    }

    #[test]
    fn test_ordered_delivery() {
        let mut buffer = OrderedBlockBuffer::new(100, 1000);

        // Insert out of order
        buffer.insert(create_test_block(103)).unwrap();
        buffer.insert(create_test_block(101)).unwrap();
        buffer.insert(create_test_block(102)).unwrap();
        buffer.insert(create_test_block(100)).unwrap();

        // Should deliver in order
        assert_eq!(buffer.take_next_ready().unwrap().header.height, 100);
        assert_eq!(buffer.take_next_ready().unwrap().header.height, 101);
        assert_eq!(buffer.take_next_ready().unwrap().header.height, 102);
        assert_eq!(buffer.take_next_ready().unwrap().header.height, 103);
        assert!(buffer.take_next_ready().is_none());
    }

    #[test]
    fn test_gap_blocking() {
        let mut buffer = OrderedBlockBuffer::new(100, 1000);

        // Insert 100, 102 (missing 101)
        buffer.insert(create_test_block(100)).unwrap();
        buffer.insert(create_test_block(102)).unwrap();

        // Should deliver 100
        assert_eq!(buffer.take_next_ready().unwrap().header.height, 100);

        // Should NOT deliver 102 (waiting for 101)
        assert!(buffer.take_next_ready().is_none());

        // Insert missing 101
        buffer.insert(create_test_block(101)).unwrap();

        // Should now deliver 101, 102
        assert_eq!(buffer.take_next_ready().unwrap().header.height, 101);
        assert_eq!(buffer.take_next_ready().unwrap().header.height, 102);
    }

    #[test]
    fn test_backpressure() {
        let mut buffer = OrderedBlockBuffer::new(100, 10);

        // Should accept within gap
        assert!(buffer.insert(create_test_block(110)).is_ok());

        // Should reject beyond gap
        assert!(buffer.insert(create_test_block(111)).is_err());
    }

    #[test]
    fn test_duplicate_skip() {
        let mut buffer = OrderedBlockBuffer::new(100, 1000);

        buffer.insert(create_test_block(100)).unwrap();
        assert_eq!(buffer.take_next_ready().unwrap().header.height, 100);

        // Should silently ignore duplicate
        buffer.insert(create_test_block(100)).unwrap();
        assert!(buffer.take_next_ready().is_none());
    }
}
