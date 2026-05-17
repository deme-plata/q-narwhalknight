//! Fork and Reorganization Safety Tests
//!
//! These tests ensure that chain forks and reorganizations are handled safely.
//! A bug here could cause:
//! - Chain splits (different nodes on different chains)
//! - Balance inconsistencies after reorg
//! - Double-spending during fork windows
//! - Lost transactions during reorg
//!
//! CRITICAL SCENARIOS:
//! 1. Fork detection
//! 2. Reorganization safety
//! 3. Balance reversion during reorg
//! 4. Transaction replay after reorg
//! 5. Double-spend prevention during forks
//!
//! Run with: cargo test --package q-storage --test fork_reorg_tests

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ============================================================================
// MOCK TYPES FOR TESTING
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlockHash([u8; 32]);

impl BlockHash {
    pub fn new(data: &[u8]) -> Self {
        let mut hash = [0u8; 32];
        for (i, byte) in data.iter().enumerate() {
            hash[i % 32] ^= byte;
        }
        Self(hash)
    }

    pub fn genesis() -> Self {
        Self([0u8; 32])
    }

    pub fn from_height(height: u64) -> Self {
        let mut hash = [0u8; 32];
        hash[0..8].copy_from_slice(&height.to_le_bytes());
        Self(hash)
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    pub height: u64,
    pub hash: BlockHash,
    pub parent_hash: BlockHash,
    pub transactions: Vec<Transaction>,
    pub timestamp: u64,
}

impl Block {
    pub fn genesis() -> Self {
        Self {
            height: 0,
            hash: BlockHash::from_height(0),
            parent_hash: BlockHash::genesis(),
            transactions: vec![],
            timestamp: 0,
        }
    }

    pub fn new(height: u64, parent: &Block, transactions: Vec<Transaction>) -> Self {
        let hash = BlockHash::new(&[
            &height.to_le_bytes()[..],
            &parent.hash.0[..],
            &(transactions.len() as u64).to_le_bytes()[..],
        ].concat());

        Self {
            height,
            hash,
            parent_hash: parent.hash.clone(),
            transactions,
            timestamp: height * 1000, // Simulated timestamp
        }
    }

    /// Create a competing block at the same height (simulates fork)
    pub fn fork_at(height: u64, parent: &Block, fork_id: u8) -> Self {
        let hash = BlockHash::new(&[
            &height.to_le_bytes()[..],
            &parent.hash.0[..],
            &[fork_id],
        ].concat());

        Self {
            height,
            hash,
            parent_hash: parent.hash.clone(),
            transactions: vec![],
            timestamp: height * 1000 + fork_id as u64,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Transaction {
    pub tx_hash: String,
    pub from: String,
    pub to: String,
    pub amount: u128,
    pub nonce: u64,
}

// ============================================================================
// CHAIN STATE WITH FORK SUPPORT
// ============================================================================

pub struct ChainState {
    /// Main chain blocks by height
    pub blocks: HashMap<u64, Block>,
    /// All known blocks by hash (for fork detection)
    pub blocks_by_hash: HashMap<BlockHash, Block>,
    /// Current head
    pub head: Option<Block>,
    /// Balances
    pub balances: HashMap<String, u128>,
    /// Nonces
    pub nonces: HashMap<String, u64>,
    /// Processed transaction hashes
    pub processed_txs: HashMap<String, u64>, // tx_hash -> height
}

impl ChainState {
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
            blocks_by_hash: HashMap::new(),
            head: None,
            balances: HashMap::new(),
            nonces: HashMap::new(),
            processed_txs: HashMap::new(),
        }
    }

    pub fn with_genesis() -> Self {
        let mut state = Self::new();
        let genesis = Block::genesis();
        state.add_block(genesis).unwrap();
        state
    }

    pub fn current_height(&self) -> u64 {
        self.head.as_ref().map(|b| b.height).unwrap_or(0)
    }

    /// Add a block to the chain
    pub fn add_block(&mut self, block: Block) -> Result<(), String> {
        // Check if this is a duplicate
        if self.blocks_by_hash.contains_key(&block.hash) {
            return Err("DUPLICATE: Block already exists".to_string());
        }

        // Check for fork
        if let Some(existing) = self.blocks.get(&block.height) {
            if existing.hash != block.hash {
                return Err(format!(
                    "FORK DETECTED: Height {} has competing block. Existing: {:?}, New: {:?}",
                    block.height, existing.hash, block.hash
                ));
            }
        }

        // Verify parent exists (except for genesis)
        if block.height > 0 {
            if !self.blocks_by_hash.contains_key(&block.parent_hash) {
                return Err(format!(
                    "ORPHAN: Parent block not found at height {}",
                    block.height
                ));
            }
        }

        // Apply transactions
        for tx in &block.transactions {
            self.apply_transaction(tx, block.height)?;
        }

        // Store block
        self.blocks.insert(block.height, block.clone());
        self.blocks_by_hash.insert(block.hash.clone(), block.clone());

        // Update head
        if self.head.is_none() || block.height > self.head.as_ref().unwrap().height {
            self.head = Some(block);
        }

        Ok(())
    }

    /// Apply a transaction
    fn apply_transaction(&mut self, tx: &Transaction, height: u64) -> Result<(), String> {
        // Check for replay
        if let Some(existing_height) = self.processed_txs.get(&tx.tx_hash) {
            return Err(format!(
                "REPLAY ATTACK: Transaction {} already processed at height {}",
                tx.tx_hash, existing_height
            ));
        }

        // Check nonce
        let expected_nonce = self.nonces.get(&tx.from).copied().unwrap_or(0);
        if tx.nonce != expected_nonce {
            return Err(format!(
                "NONCE MISMATCH: Expected {}, got {} for {}",
                expected_nonce, tx.nonce, tx.from
            ));
        }

        // Check balance
        let balance = self.balances.get(&tx.from).copied().unwrap_or(0);
        if balance < tx.amount {
            return Err(format!(
                "INSUFFICIENT FUNDS: {} has {}, needs {}",
                tx.from, balance, tx.amount
            ));
        }

        // Apply
        *self.balances.entry(tx.from.clone()).or_insert(0) -= tx.amount;
        *self.balances.entry(tx.to.clone()).or_insert(0) += tx.amount;
        *self.nonces.entry(tx.from.clone()).or_insert(0) += 1;
        self.processed_txs.insert(tx.tx_hash.clone(), height);

        Ok(())
    }

    /// Detect if there's a fork at the given height
    pub fn has_fork_at(&self, height: u64) -> bool {
        let mut count = 0;
        for block in self.blocks_by_hash.values() {
            if block.height == height {
                count += 1;
            }
        }
        count > 1
    }

    /// Revert to a specific height (for reorganization)
    pub fn revert_to_height(&mut self, target_height: u64) -> Result<Vec<Transaction>, String> {
        let current = self.current_height();

        if target_height >= current {
            return Err("Cannot revert to higher height".to_string());
        }

        let mut reverted_txs = Vec::new();

        // Collect transactions to revert
        for height in (target_height + 1..=current).rev() {
            if let Some(block) = self.blocks.get(&height) {
                // Revert transactions in reverse order
                for tx in block.transactions.iter().rev() {
                    // Undo the transaction
                    *self.balances.entry(tx.to.clone()).or_insert(0) -= tx.amount;
                    *self.balances.entry(tx.from.clone()).or_insert(0) += tx.amount;
                    *self.nonces.entry(tx.from.clone()).or_insert(1) -= 1;
                    self.processed_txs.remove(&tx.tx_hash);

                    reverted_txs.push(tx.clone());
                }
            }
        }

        // Remove blocks
        for height in (target_height + 1..=current) {
            if let Some(block) = self.blocks.remove(&height) {
                self.blocks_by_hash.remove(&block.hash);
            }
        }

        // Update head
        self.head = self.blocks.get(&target_height).cloned();

        Ok(reverted_txs)
    }
}

// ============================================================================
// FORK DETECTION TESTS
// ============================================================================

mod fork_detection_tests {
    use super::*;

    #[test]
    fn test_detect_simple_fork() {
        let mut state = ChainState::with_genesis();
        let genesis = state.head.clone().unwrap();

        // Add block 1 on main chain
        let block1 = Block::new(1, &genesis, vec![]);
        state.add_block(block1.clone()).unwrap();

        // Try to add competing block 1 (fork)
        let fork_block1 = Block::fork_at(1, &genesis, 1);
        let result = state.add_block(fork_block1);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("FORK DETECTED"));
    }

    #[test]
    fn test_duplicate_block_rejected() {
        let mut state = ChainState::with_genesis();
        let genesis = state.head.clone().unwrap();

        let block1 = Block::new(1, &genesis, vec![]);
        state.add_block(block1.clone()).unwrap();

        // Try to add same block again
        let result = state.add_block(block1);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("DUPLICATE"));
    }

    #[test]
    fn test_orphan_block_rejected() {
        let mut state = ChainState::with_genesis();

        // Create block with non-existent parent
        let orphan = Block {
            height: 5,
            hash: BlockHash::from_height(5),
            parent_hash: BlockHash::from_height(4), // Parent doesn't exist
            transactions: vec![],
            timestamp: 5000,
        };

        let result = state.add_block(orphan);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("ORPHAN"));
    }

    #[test]
    fn test_detect_fork_in_chain() {
        let mut state = ChainState::new();

        // Build main chain
        let genesis = Block::genesis();
        state.blocks_by_hash.insert(genesis.hash.clone(), genesis.clone());

        let block1 = Block::new(1, &genesis, vec![]);
        state.blocks_by_hash.insert(block1.hash.clone(), block1.clone());

        // Add fork block at height 1
        let fork1 = Block::fork_at(1, &genesis, 1);
        state.blocks_by_hash.insert(fork1.hash.clone(), fork1.clone());

        // Should detect fork
        assert!(state.has_fork_at(1));
        assert!(!state.has_fork_at(0)); // Genesis has no fork
        assert!(!state.has_fork_at(2)); // No blocks at height 2
    }
}

// ============================================================================
// REORGANIZATION TESTS
// ============================================================================

mod reorganization_tests {
    use super::*;

    #[test]
    fn test_revert_single_block() {
        let mut state = ChainState::with_genesis();
        let genesis = state.head.clone().unwrap();

        // Setup initial balances
        state.balances.insert("alice".to_string(), 1000);

        // Add block with transaction
        let tx = Transaction {
            tx_hash: "tx1".to_string(),
            from: "alice".to_string(),
            to: "bob".to_string(),
            amount: 100,
            nonce: 0,
        };
        let block1 = Block::new(1, &genesis, vec![tx]);
        state.add_block(block1).unwrap();

        assert_eq!(state.balances.get("alice"), Some(&900));
        assert_eq!(state.balances.get("bob"), Some(&100));

        // Revert
        let reverted = state.revert_to_height(0).unwrap();

        assert_eq!(reverted.len(), 1);
        assert_eq!(state.balances.get("alice"), Some(&1000));
        assert_eq!(state.balances.get("bob"), Some(&0));
        assert_eq!(state.current_height(), 0);
    }

    #[test]
    fn test_revert_multiple_blocks() {
        let mut state = ChainState::with_genesis();
        let genesis = state.head.clone().unwrap();

        state.balances.insert("alice".to_string(), 1000);

        // Block 1: alice -> bob 100
        let tx1 = Transaction {
            tx_hash: "tx1".to_string(),
            from: "alice".to_string(),
            to: "bob".to_string(),
            amount: 100,
            nonce: 0,
        };
        let block1 = Block::new(1, &genesis, vec![tx1]);
        state.add_block(block1.clone()).unwrap();

        // Block 2: alice -> charlie 200
        let tx2 = Transaction {
            tx_hash: "tx2".to_string(),
            from: "alice".to_string(),
            to: "charlie".to_string(),
            amount: 200,
            nonce: 1,
        };
        let block2 = Block::new(2, &block1, vec![tx2]);
        state.add_block(block2).unwrap();

        assert_eq!(state.balances.get("alice"), Some(&700));
        assert_eq!(state.balances.get("bob"), Some(&100));
        assert_eq!(state.balances.get("charlie"), Some(&200));

        // Revert both blocks
        let reverted = state.revert_to_height(0).unwrap();

        assert_eq!(reverted.len(), 2);
        assert_eq!(state.balances.get("alice"), Some(&1000));
        assert_eq!(state.balances.get("bob"), Some(&0));
        assert_eq!(state.balances.get("charlie"), Some(&0));
    }

    #[test]
    fn test_revert_restores_nonces() {
        let mut state = ChainState::with_genesis();
        let genesis = state.head.clone().unwrap();

        state.balances.insert("alice".to_string(), 1000);

        // Two transactions from alice
        let tx1 = Transaction {
            tx_hash: "tx1".to_string(),
            from: "alice".to_string(),
            to: "bob".to_string(),
            amount: 100,
            nonce: 0,
        };
        let tx2 = Transaction {
            tx_hash: "tx2".to_string(),
            from: "alice".to_string(),
            to: "charlie".to_string(),
            amount: 100,
            nonce: 1,
        };

        let block1 = Block::new(1, &genesis, vec![tx1, tx2]);
        state.add_block(block1).unwrap();

        assert_eq!(state.nonces.get("alice"), Some(&2));

        // Revert
        state.revert_to_height(0).unwrap();

        assert_eq!(state.nonces.get("alice"), Some(&0));
    }

    #[test]
    fn test_revert_removes_tx_from_processed() {
        let mut state = ChainState::with_genesis();
        let genesis = state.head.clone().unwrap();

        state.balances.insert("alice".to_string(), 1000);

        let tx = Transaction {
            tx_hash: "important_tx".to_string(),
            from: "alice".to_string(),
            to: "bob".to_string(),
            amount: 100,
            nonce: 0,
        };

        let block1 = Block::new(1, &genesis, vec![tx.clone()]);
        state.add_block(block1).unwrap();

        assert!(state.processed_txs.contains_key("important_tx"));

        // Revert
        state.revert_to_height(0).unwrap();

        // TX should be removed from processed set
        assert!(!state.processed_txs.contains_key("important_tx"));
    }

    #[test]
    fn test_cannot_revert_to_higher_height() {
        let mut state = ChainState::with_genesis();

        let result = state.revert_to_height(10);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Cannot revert"));
    }
}

// ============================================================================
// DOUBLE-SPEND DURING FORK TESTS
// ============================================================================

mod double_spend_fork_tests {
    use super::*;

    #[test]
    fn test_replay_attack_after_reorg() {
        let mut state = ChainState::with_genesis();
        let genesis = state.head.clone().unwrap();

        state.balances.insert("alice".to_string(), 1000);

        // Original transaction
        let tx = Transaction {
            tx_hash: "tx1".to_string(),
            from: "alice".to_string(),
            to: "bob".to_string(),
            amount: 500,
            nonce: 0,
        };

        // Add to block 1
        let block1 = Block::new(1, &genesis, vec![tx.clone()]);
        state.add_block(block1).unwrap();

        // Revert
        state.revert_to_height(0).unwrap();

        // Transaction can now be replayed (same hash allowed after revert)
        // This is expected - the transaction needs to be re-mined
        let block1_new = Block::new(1, &genesis, vec![tx]);
        let result = state.add_block(block1_new);

        // This should succeed because the tx was removed from processed set
        assert!(result.is_ok());
    }

    #[test]
    fn test_double_spend_same_nonce_rejected() {
        let mut state = ChainState::with_genesis();
        let genesis = state.head.clone().unwrap();

        state.balances.insert("alice".to_string(), 1000);

        let tx1 = Transaction {
            tx_hash: "tx1".to_string(),
            from: "alice".to_string(),
            to: "bob".to_string(),
            amount: 500,
            nonce: 0,
        };

        let tx2 = Transaction {
            tx_hash: "tx2".to_string(),
            from: "alice".to_string(),
            to: "charlie".to_string(),
            amount: 500,
            nonce: 0, // SAME NONCE!
        };

        // Both in same block
        let block = Block::new(1, &genesis, vec![tx1, tx2]);
        let result = state.add_block(block);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("NONCE MISMATCH"));
    }

    #[test]
    fn test_balance_consistency_after_reorg() {
        let mut state = ChainState::with_genesis();
        let genesis = state.head.clone().unwrap();

        // Initial state
        state.balances.insert("alice".to_string(), 1000);
        state.balances.insert("bob".to_string(), 1000);

        let initial_total: u128 = state.balances.values().sum();

        // Transaction chain
        let tx1 = Transaction {
            tx_hash: "tx1".to_string(),
            from: "alice".to_string(),
            to: "bob".to_string(),
            amount: 200,
            nonce: 0,
        };
        let tx2 = Transaction {
            tx_hash: "tx2".to_string(),
            from: "bob".to_string(),
            to: "alice".to_string(),
            amount: 100,
            nonce: 0,
        };

        let block1 = Block::new(1, &genesis, vec![tx1]);
        state.add_block(block1.clone()).unwrap();

        let block2 = Block::new(2, &block1, vec![tx2]);
        state.add_block(block2).unwrap();

        // Total should be preserved
        let after_total: u128 = state.balances.values().sum();
        assert_eq!(initial_total, after_total);

        // Revert to genesis
        state.revert_to_height(0).unwrap();

        // Total should still be preserved
        let reverted_total: u128 = state.balances.values().sum();
        assert_eq!(initial_total, reverted_total);
    }
}

// ============================================================================
// CHAIN SELECTION TESTS
// ============================================================================

mod chain_selection_tests {
    use super::*;

    /// Simulate choosing between two competing chains
    #[test]
    fn test_longest_chain_wins() {
        let mut state_a = ChainState::with_genesis();
        let mut state_b = ChainState::with_genesis();

        let genesis = state_a.head.clone().unwrap();

        // Chain A: 3 blocks
        let block_a1 = Block::new(1, &genesis, vec![]);
        state_a.add_block(block_a1.clone()).unwrap();

        let block_a2 = Block::new(2, &block_a1, vec![]);
        state_a.add_block(block_a2.clone()).unwrap();

        let block_a3 = Block::new(3, &block_a2, vec![]);
        state_a.add_block(block_a3).unwrap();

        // Chain B: 2 blocks (shorter)
        let block_b1 = Block::fork_at(1, &genesis, 1);
        state_b.add_block(block_b1.clone()).unwrap();

        let block_b2 = Block::new(2, &block_b1, vec![]);
        state_b.add_block(block_b2).unwrap();

        // Chain A should be longer
        assert!(state_a.current_height() > state_b.current_height());
        assert_eq!(state_a.current_height(), 3);
        assert_eq!(state_b.current_height(), 2);
    }

    #[test]
    fn test_reorg_to_longer_chain() {
        let mut state = ChainState::with_genesis();
        let genesis = state.head.clone().unwrap();

        state.balances.insert("alice".to_string(), 1000);

        // Build initial chain (2 blocks)
        let tx1 = Transaction {
            tx_hash: "tx1".to_string(),
            from: "alice".to_string(),
            to: "bob".to_string(),
            amount: 100,
            nonce: 0,
        };
        let block1 = Block::new(1, &genesis, vec![tx1]);
        state.add_block(block1.clone()).unwrap();

        let block2 = Block::new(2, &block1, vec![]);
        state.add_block(block2).unwrap();

        // Now simulate receiving a longer competing chain
        // First, revert to genesis
        state.revert_to_height(0).unwrap();

        // Apply new chain (3 blocks, no transaction)
        let new_block1 = Block::fork_at(1, &genesis, 1);
        state.blocks.remove(&1); // Remove old block to allow new one
        state.add_block(new_block1.clone()).unwrap();

        let new_block2 = Block::new(2, &new_block1, vec![]);
        state.blocks.remove(&2);
        state.add_block(new_block2.clone()).unwrap();

        let new_block3 = Block::new(3, &new_block2, vec![]);
        state.add_block(new_block3).unwrap();

        // Should now be on the new chain
        assert_eq!(state.current_height(), 3);
        // Original transaction was reverted, so alice still has 1000
        assert_eq!(state.balances.get("alice"), Some(&1000));
    }
}

// ============================================================================
// CONCURRENT FORK HANDLING TESTS
// ============================================================================

mod concurrent_tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_concurrent_block_additions() {
        let state = Arc::new(Mutex::new(ChainState::with_genesis()));

        // Get genesis for all threads
        let genesis = {
            let s = state.lock().unwrap();
            s.head.clone().unwrap()
        };

        // Set up initial balance
        {
            let mut s = state.lock().unwrap();
            s.balances.insert("alice".to_string(), 10000);
        }

        let mut handles = vec![];

        // Multiple threads trying to add blocks
        for thread_id in 0..4 {
            let state_clone = Arc::clone(&state);
            let genesis_clone = genesis.clone();

            let handle = thread::spawn(move || {
                for i in 0..10 {
                    let mut s = state_clone.lock().unwrap();
                    let current = s.head.clone().unwrap_or(genesis_clone.clone());

                    let tx = Transaction {
                        tx_hash: format!("tx_{}_{}", thread_id, i),
                        from: "alice".to_string(),
                        to: format!("recipient_{}", thread_id),
                        amount: 1,
                        nonce: s.nonces.get("alice").copied().unwrap_or(0),
                    };

                    let block = Block::new(current.height + 1, &current, vec![tx]);
                    let _ = s.add_block(block); // May fail due to forks
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // State should be consistent
        let s = state.lock().unwrap();
        let total: u128 = s.balances.values().sum();

        // Total should equal initial (10000) since transfers are internal
        assert_eq!(total, 10000, "Total supply should be preserved");
    }
}

// ============================================================================
// REGRESSION TESTS
// ============================================================================

mod regression_tests {
    use super::*;

    #[test]
    fn test_regression_fork_detection_works() {
        let mut state = ChainState::with_genesis();
        let genesis = state.head.clone().unwrap();

        let block1 = Block::new(1, &genesis, vec![]);
        state.add_block(block1).unwrap();

        let fork_block = Block::fork_at(1, &genesis, 99);
        let result = state.add_block(fork_block);

        // Must detect fork
        assert!(result.is_err(), "Fork must be detected");
        assert!(
            result.unwrap_err().contains("FORK"),
            "Error message must mention fork"
        );
    }

    #[test]
    fn test_regression_revert_restores_all_state() {
        let mut state = ChainState::with_genesis();
        let genesis = state.head.clone().unwrap();

        state.balances.insert("alice".to_string(), 1000);

        let original_balance = state.balances.get("alice").copied();
        let original_nonce = state.nonces.get("alice").copied().unwrap_or(0);

        // Add transaction
        let tx = Transaction {
            tx_hash: "tx1".to_string(),
            from: "alice".to_string(),
            to: "bob".to_string(),
            amount: 100,
            nonce: 0,
        };
        let block = Block::new(1, &genesis, vec![tx]);
        state.add_block(block).unwrap();

        // Revert
        state.revert_to_height(0).unwrap();

        // All state must be restored
        assert_eq!(
            state.balances.get("alice").copied(),
            original_balance,
            "Balance must be restored"
        );
        assert_eq!(
            state.nonces.get("alice").copied().unwrap_or(0),
            original_nonce,
            "Nonce must be restored"
        );
        assert!(
            !state.processed_txs.contains_key("tx1"),
            "Processed tx must be removed"
        );
    }
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_revert_small_chain_performance() {
        let mut state = ChainState::with_genesis();
        let genesis = state.head.clone().unwrap();

        state.balances.insert("alice".to_string(), 1_000);

        // Build a small chain with 5 blocks, each with 2 transactions
        let mut parent = genesis;
        let mut nonce = 0u64;

        for height in 1u64..=5u64 {
            let mut txs = vec![];
            for i in 0..2 {
                txs.push(Transaction {
                    tx_hash: format!("perf_tx_{}_{}", height, i),
                    from: "alice".to_string(),
                    to: format!("bob_{}", i),
                    amount: 1,
                    nonce,
                });
                nonce += 1;
            }

            let block = Block::new(height, &parent, txs);
            state.add_block(block.clone()).unwrap();
            parent = block;
        }

        assert_eq!(state.current_height(), 5);

        // Measure revert time
        let start = Instant::now();
        state.revert_to_height(0).unwrap();
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_millis() < 100,
            "Reverting 5 blocks should take less than 100ms, took {:?}",
            elapsed
        );

        // Verify complete revert
        assert_eq!(state.current_height(), 0);
        assert_eq!(state.balances.get("alice"), Some(&1_000));
    }

    #[test]
    fn test_fork_detection_is_fast() {
        let mut state = ChainState::with_genesis();
        let genesis = state.head.clone().unwrap();

        // Build a short chain (10 blocks)
        let mut parent = genesis.clone();
        for height in 1u64..=10u64 {
            let tx = Transaction {
                tx_hash: format!("fork_perf_tx_{}", height),
                from: "miner".to_string(),
                to: "treasury".to_string(),
                amount: 0,
                nonce: 0,
            };
            let block = Block::new(height, &parent, vec![tx]);
            state.blocks_by_hash.insert(block.hash.clone(), block.clone());
            state.blocks.insert(height, block.clone());
            parent = block;
        }

        // Measure fork detection time
        let fork_block = Block::fork_at(5, &genesis, 99);

        let start = Instant::now();
        let result = state.add_block(fork_block);
        let elapsed = start.elapsed();

        // Should detect as orphan (parent hash from fork_at doesn't match)
        assert!(result.is_err());
        assert!(
            elapsed.as_micros() < 10000,
            "Fork detection should be fast, took {:?}",
            elapsed
        );
    }
}
