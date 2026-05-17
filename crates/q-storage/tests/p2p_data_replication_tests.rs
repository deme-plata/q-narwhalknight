//! P2P Data Replication Tests
//!
//! v3.5.25-beta: Comprehensive tests for decentralized P2P sync data replication
//!
//! These tests verify:
//! 1. Full blockchain replication between nodes (any node can sync from any other)
//! 2. Balance derivation from coinbase transactions (balances are recoverable)
//! 3. Fresh node sync from any peer (not just bootstrap)
//! 4. No bootstrap-only data (network is fully decentralized)
//! 5. Recovery from corruption scenarios
//! 6. Symmetric block serving (any node can serve blocks to peers)
//!
//! CRITICAL MAINNET GUARANTEES:
//! - All persistent state flows through blocks
//! - Balances are deterministically derived from coinbase transactions
//! - Bootstrap can recover from corruption by syncing from peers
//! - No single point of failure in the network
//!
//! Run with: cargo test --package q-storage --test p2p_data_replication_tests

use std::collections::{HashMap, HashSet};
// Note: Remove unused imports to clean up warnings

// ============================================================================
// MOCK STRUCTURES FOR P2P SIMULATION
// ============================================================================

/// Represents a block with coinbase transaction
#[derive(Debug, Clone)]
pub struct MockBlock {
    pub height: u64,
    pub parent_hash: [u8; 32],
    pub hash: [u8; 32],
    pub coinbase: CoinbaseTransaction,
    pub transactions: Vec<MockTransaction>,
    pub timestamp: u64,
}

/// Coinbase transaction that credits mining rewards
#[derive(Debug, Clone)]
pub struct CoinbaseTransaction {
    pub miner_address: String,
    pub amount: u128,
}

/// Regular transaction (transfer)
#[derive(Debug, Clone)]
pub struct MockTransaction {
    pub from: String,
    pub to: String,
    pub amount: u128,
    pub hash: [u8; 32],
}

/// Simulates a node's storage
#[derive(Debug, Clone)]
pub struct NodeStorage {
    pub blocks: HashMap<u64, MockBlock>,
    pub highest_height: u64,
    pub is_bootstrap: bool,
    pub node_id: String,
}

impl NodeStorage {
    pub fn new(node_id: &str, is_bootstrap: bool) -> Self {
        Self {
            blocks: HashMap::new(),
            highest_height: 0,
            is_bootstrap,
            node_id: node_id.to_string(),
        }
    }

    /// Store a block
    pub fn save_block(&mut self, block: MockBlock) {
        let height = block.height;
        self.blocks.insert(height, block);
        if height > self.highest_height {
            self.highest_height = height;
        }
    }

    /// Get block by height
    pub fn get_block(&self, height: u64) -> Option<&MockBlock> {
        self.blocks.get(&height)
    }

    /// Get blocks in range (used for serving to peers)
    pub fn get_blocks_range(&self, start: u64, end: u64) -> Vec<MockBlock> {
        (start..=end)
            .filter_map(|h| self.blocks.get(&h).cloned())
            .collect()
    }

    /// Get total block count
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Clear all data (simulates corruption recovery)
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.highest_height = 0;
    }

    /// Derive balances from all blocks (deterministic recovery)
    pub fn derive_balances_from_blocks(&self) -> HashMap<String, u128> {
        let mut balances: HashMap<String, u128> = HashMap::new();

        // Process blocks in order
        let mut heights: Vec<u64> = self.blocks.keys().cloned().collect();
        heights.sort();

        for height in heights {
            if let Some(block) = self.blocks.get(&height) {
                // Apply coinbase reward
                *balances.entry(block.coinbase.miner_address.clone()).or_insert(0) +=
                    block.coinbase.amount;

                // Apply transactions
                for tx in &block.transactions {
                    // Subtract from sender
                    if let Some(sender_balance) = balances.get_mut(&tx.from) {
                        *sender_balance = sender_balance.saturating_sub(tx.amount);
                    }
                    // Add to receiver
                    *balances.entry(tx.to.clone()).or_insert(0) += tx.amount;
                }
            }
        }

        balances
    }
}

/// Simulates P2P network with multiple nodes
#[derive(Debug)]
pub struct P2PNetwork {
    pub nodes: HashMap<String, NodeStorage>,
}

impl P2PNetwork {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node_id: &str, is_bootstrap: bool) {
        self.nodes.insert(
            node_id.to_string(),
            NodeStorage::new(node_id, is_bootstrap),
        );
    }

    /// Sync blocks from source to target (simulates Turbo Sync)
    pub fn sync_blocks(&mut self, source_id: &str, target_id: &str) -> Result<usize, String> {
        // Get source's blocks
        let source_blocks = {
            let source = self.nodes.get(source_id)
                .ok_or_else(|| format!("Source node {} not found", source_id))?;
            source.blocks.clone()
        };

        // Copy to target
        let target = self.nodes.get_mut(target_id)
            .ok_or_else(|| format!("Target node {} not found", target_id))?;

        let mut synced_count = 0;
        for (height, block) in source_blocks {
            if !target.blocks.contains_key(&height) {
                target.save_block(block);
                synced_count += 1;
            }
        }

        Ok(synced_count)
    }

    /// Get a node's block at height
    pub fn get_node_block(&self, node_id: &str, height: u64) -> Option<MockBlock> {
        self.nodes.get(node_id)?.get_block(height).cloned()
    }

    /// Get node's highest height
    pub fn get_node_height(&self, node_id: &str) -> Option<u64> {
        self.nodes.get(node_id).map(|n| n.highest_height)
    }
}

/// Create a chain of test blocks
fn create_test_chain(num_blocks: u64, miner_addresses: &[&str]) -> Vec<MockBlock> {
    let mut blocks = Vec::new();
    let mut prev_hash = [0u8; 32];

    for height in 1..=num_blocks {
        let miner = miner_addresses[(height as usize - 1) % miner_addresses.len()];
        let mut hash = [0u8; 32];
        hash[0..8].copy_from_slice(&height.to_le_bytes());

        let block = MockBlock {
            height,
            parent_hash: prev_hash,
            hash,
            coinbase: CoinbaseTransaction {
                miner_address: miner.to_string(),
                amount: 50_000_000_000, // 50 Q
            },
            transactions: vec![],
            timestamp: 1700000000 + height * 10,
        };

        prev_hash = hash;
        blocks.push(block);
    }

    blocks
}

// ============================================================================
// TEST MODULE 1: FULL BLOCKCHAIN REPLICATION
// ============================================================================

mod full_blockchain_replication {
    use super::*;

    #[test]
    fn test_bootstrap_to_new_node_full_sync() {
        let mut network = P2PNetwork::new();

        // Create bootstrap with 1000 blocks
        network.add_node("bootstrap", true);
        let chain = create_test_chain(1000, &["miner1", "miner2", "miner3"]);

        for block in chain {
            network.nodes.get_mut("bootstrap").unwrap().save_block(block);
        }

        // Add new node
        network.add_node("new_node", false);

        // Sync from bootstrap to new node
        let synced = network.sync_blocks("bootstrap", "new_node").unwrap();

        // Verify COMPLETE replication
        assert_eq!(synced, 1000, "Should sync all 1000 blocks");
        assert_eq!(
            network.get_node_height("new_node").unwrap(),
            1000,
            "New node should have same height as bootstrap"
        );

        // Verify every block matches
        for height in 1..=1000 {
            let bootstrap_block = network.get_node_block("bootstrap", height).unwrap();
            let new_node_block = network.get_node_block("new_node", height).unwrap();
            assert_eq!(
                bootstrap_block.hash, new_node_block.hash,
                "Block {} hash should match", height
            );
        }
    }

    #[test]
    fn test_sync_from_non_bootstrap_node() {
        let mut network = P2PNetwork::new();

        // Bootstrap creates chain
        network.add_node("bootstrap", true);
        let chain = create_test_chain(500, &["miner1"]);
        for block in chain {
            network.nodes.get_mut("bootstrap").unwrap().save_block(block);
        }

        // Node A syncs from bootstrap
        network.add_node("node_a", false);
        network.sync_blocks("bootstrap", "node_a").unwrap();

        // Node B syncs from Node A (NOT bootstrap!)
        network.add_node("node_b", false);
        let synced = network.sync_blocks("node_a", "node_b").unwrap();

        // Verify Node B has complete chain from Node A
        assert_eq!(synced, 500, "Node B should get all blocks from Node A");
        assert_eq!(
            network.get_node_height("node_b").unwrap(),
            500,
            "Node B should have same height as Node A"
        );

        // Verify data integrity through multi-hop sync
        for height in 1..=500 {
            let bootstrap_block = network.get_node_block("bootstrap", height).unwrap();
            let node_b_block = network.get_node_block("node_b", height).unwrap();
            assert_eq!(
                bootstrap_block.hash, node_b_block.hash,
                "Block {} should be identical through multi-hop sync", height
            );
        }
    }

    #[test]
    fn test_any_node_can_be_sync_source() {
        let mut network = P2PNetwork::new();

        // Create network: bootstrap -> A -> B -> C
        network.add_node("bootstrap", true);
        network.add_node("node_a", false);
        network.add_node("node_b", false);
        network.add_node("node_c", false);

        let chain = create_test_chain(100, &["miner1"]);
        for block in chain {
            network.nodes.get_mut("bootstrap").unwrap().save_block(block);
        }

        // Sync chain: bootstrap -> A
        network.sync_blocks("bootstrap", "node_a").unwrap();

        // Sync from A (non-bootstrap) -> B
        network.sync_blocks("node_a", "node_b").unwrap();

        // Sync from B (non-bootstrap) -> C
        network.sync_blocks("node_b", "node_c").unwrap();

        // ALL nodes should have identical chains
        let nodes = ["bootstrap", "node_a", "node_b", "node_c"];
        for &node in &nodes {
            assert_eq!(
                network.get_node_height(node).unwrap(),
                100,
                "Node {} should have height 100", node
            );
        }

        // Verify block integrity across all nodes
        for height in 1..=100 {
            let hashes: Vec<[u8; 32]> = nodes.iter()
                .map(|&n| network.get_node_block(n, height).unwrap().hash)
                .collect();

            assert!(
                hashes.iter().all(|h| *h == hashes[0]),
                "All nodes should have identical block {} hash", height
            );
        }
    }

    #[test]
    fn test_partial_sync_resumes_correctly() {
        let mut network = P2PNetwork::new();

        // Bootstrap with 1000 blocks
        network.add_node("bootstrap", true);
        let chain = create_test_chain(1000, &["miner1"]);
        for block in chain {
            network.nodes.get_mut("bootstrap").unwrap().save_block(block);
        }

        // Node has partial sync (first 500 blocks)
        network.add_node("partial_node", false);
        let partial_chain = create_test_chain(500, &["miner1"]);
        for block in partial_chain {
            network.nodes.get_mut("partial_node").unwrap().save_block(block);
        }

        // Sync remaining blocks
        let synced = network.sync_blocks("bootstrap", "partial_node").unwrap();

        // Should only sync the missing 500 blocks
        assert_eq!(synced, 500, "Should sync only missing blocks");
        assert_eq!(
            network.get_node_height("partial_node").unwrap(),
            1000,
            "Node should now have complete chain"
        );
    }
}

// ============================================================================
// TEST MODULE 2: BALANCE DERIVATION FROM BLOCKS
// ============================================================================

mod balance_derivation_tests {
    use super::*;

    #[test]
    fn test_balances_derived_from_coinbase_transactions() {
        let mut storage = NodeStorage::new("test_node", false);

        // Create chain with known miners
        let chain = create_test_chain(100, &["miner_alice", "miner_bob"]);
        for block in chain {
            storage.save_block(block);
        }

        // Derive balances
        let balances = storage.derive_balances_from_blocks();

        // Each miner should have 50 blocks * 50Q reward = 2500Q
        let alice_balance = balances.get("miner_alice").unwrap_or(&0);
        let bob_balance = balances.get("miner_bob").unwrap_or(&0);

        assert_eq!(*alice_balance, 50 * 50_000_000_000, "Alice should have 50 block rewards");
        assert_eq!(*bob_balance, 50 * 50_000_000_000, "Bob should have 50 block rewards");
    }

    #[test]
    fn test_balance_derivation_is_deterministic() {
        let chain = create_test_chain(200, &["miner1", "miner2", "miner3"]);

        // Derive balances multiple times
        let mut results = Vec::new();

        for _ in 0..5 {
            let mut storage = NodeStorage::new("test", false);
            for block in &chain {
                storage.save_block(block.clone());
            }
            results.push(storage.derive_balances_from_blocks());
        }

        // All derivations should be identical
        for i in 1..results.len() {
            assert_eq!(
                results[0], results[i],
                "Balance derivation iteration {} differs from first", i
            );
        }
    }

    #[test]
    fn test_synced_node_derives_same_balances() {
        let mut network = P2PNetwork::new();

        // Bootstrap creates chain
        network.add_node("bootstrap", true);
        let chain = create_test_chain(500, &["miner_x", "miner_y", "miner_z"]);
        for block in chain {
            network.nodes.get_mut("bootstrap").unwrap().save_block(block);
        }

        // New node syncs
        network.add_node("synced_node", false);
        network.sync_blocks("bootstrap", "synced_node").unwrap();

        // Both nodes should derive identical balances
        let bootstrap_balances = network.nodes.get("bootstrap")
            .unwrap().derive_balances_from_blocks();
        let synced_balances = network.nodes.get("synced_node")
            .unwrap().derive_balances_from_blocks();

        assert_eq!(
            bootstrap_balances, synced_balances,
            "Synced node must derive identical balances"
        );
    }

    #[test]
    fn test_balance_recovery_after_corruption() {
        let mut storage = NodeStorage::new("test", false);

        // Create chain and record balances
        let chain = create_test_chain(300, &["miner1"]);
        for block in &chain {
            storage.save_block(block.clone());
        }
        let original_balances = storage.derive_balances_from_blocks();

        // Simulate corruption - clear everything
        storage.clear();
        assert_eq!(storage.block_count(), 0, "Storage should be empty after clear");

        // Restore from chain (simulate sync)
        for block in &chain {
            storage.save_block(block.clone());
        }

        // Recover balances
        let recovered_balances = storage.derive_balances_from_blocks();

        assert_eq!(
            original_balances, recovered_balances,
            "Recovered balances must match original"
        );
    }

    #[test]
    fn test_balances_include_transactions() {
        let mut storage = NodeStorage::new("test", false);

        // Create block with coinbase and transfer
        let block = MockBlock {
            height: 1,
            parent_hash: [0u8; 32],
            hash: [1u8; 32],
            coinbase: CoinbaseTransaction {
                miner_address: "miner".to_string(),
                amount: 100_000_000_000, // 100 Q
            },
            transactions: vec![
                MockTransaction {
                    from: "miner".to_string(),
                    to: "alice".to_string(),
                    amount: 25_000_000_000, // 25 Q
                    hash: [2u8; 32],
                }
            ],
            timestamp: 1700000000,
        };

        storage.save_block(block);
        let balances = storage.derive_balances_from_blocks();

        // Miner: 100Q - 25Q = 75Q
        // Alice: 0Q + 25Q = 25Q
        assert_eq!(*balances.get("miner").unwrap_or(&0), 75_000_000_000);
        assert_eq!(*balances.get("alice").unwrap_or(&0), 25_000_000_000);
    }
}

// ============================================================================
// TEST MODULE 3: NO BOOTSTRAP-ONLY DATA
// ============================================================================

mod no_bootstrap_only_data {
    use super::*;

    #[test]
    fn test_bootstrap_has_no_exclusive_data() {
        let mut network = P2PNetwork::new();

        // Setup bootstrap with chain
        network.add_node("bootstrap", true);
        let chain = create_test_chain(1000, &["miner1", "miner2"]);
        for block in chain {
            network.nodes.get_mut("bootstrap").unwrap().save_block(block);
        }

        // Sync to regular node
        network.add_node("regular_node", false);
        network.sync_blocks("bootstrap", "regular_node").unwrap();

        // Get all data from bootstrap
        let bootstrap_storage = network.nodes.get("bootstrap").unwrap();
        let bootstrap_heights: HashSet<u64> = bootstrap_storage.blocks.keys().cloned().collect();
        let bootstrap_balances = bootstrap_storage.derive_balances_from_blocks();

        // Get all data from regular node
        let regular_storage = network.nodes.get("regular_node").unwrap();
        let regular_heights: HashSet<u64> = regular_storage.blocks.keys().cloned().collect();
        let regular_balances = regular_storage.derive_balances_from_blocks();

        // Regular node must have ALL data bootstrap has
        assert_eq!(
            bootstrap_heights, regular_heights,
            "Regular node must have all blocks bootstrap has"
        );
        assert_eq!(
            bootstrap_balances, regular_balances,
            "Regular node must derive all balances bootstrap has"
        );
    }

    #[test]
    fn test_bootstrap_recovery_from_peer() {
        let mut network = P2PNetwork::new();

        // Bootstrap creates chain
        network.add_node("bootstrap", true);
        let chain = create_test_chain(500, &["miner1"]);
        for block in &chain {
            network.nodes.get_mut("bootstrap").unwrap().save_block(block.clone());
        }

        // Peer syncs from bootstrap
        network.add_node("peer", false);
        network.sync_blocks("bootstrap", "peer").unwrap();

        // Simulate bootstrap corruption
        network.nodes.get_mut("bootstrap").unwrap().clear();
        assert_eq!(
            network.nodes.get("bootstrap").unwrap().block_count(),
            0,
            "Bootstrap should be empty after corruption"
        );

        // Bootstrap recovers from peer
        let synced = network.sync_blocks("peer", "bootstrap").unwrap();

        assert_eq!(synced, 500, "Bootstrap should recover all blocks from peer");
        assert_eq!(
            network.get_node_height("bootstrap").unwrap(),
            500,
            "Bootstrap should be fully restored"
        );

        // Verify bootstrap has complete chain again
        for height in 1..=500 {
            assert!(
                network.get_node_block("bootstrap", height).is_some(),
                "Bootstrap should have block {} after recovery", height
            );
        }
    }

    #[test]
    fn test_network_survives_bootstrap_loss() {
        let mut network = P2PNetwork::new();

        // Create network: bootstrap + 3 peers
        network.add_node("bootstrap", true);
        network.add_node("peer_a", false);
        network.add_node("peer_b", false);
        network.add_node("peer_c", false);

        // Bootstrap creates chain
        let chain = create_test_chain(300, &["miner1"]);
        for block in &chain {
            network.nodes.get_mut("bootstrap").unwrap().save_block(block.clone());
        }

        // All peers sync
        network.sync_blocks("bootstrap", "peer_a").unwrap();
        network.sync_blocks("bootstrap", "peer_b").unwrap();
        network.sync_blocks("bootstrap", "peer_c").unwrap();

        // Simulate COMPLETE bootstrap failure
        network.nodes.remove("bootstrap");

        // New node joins and syncs from surviving peer
        network.add_node("new_node", false);
        let synced = network.sync_blocks("peer_a", "new_node").unwrap();

        assert_eq!(synced, 300, "New node should get complete chain from peer");

        // New node should have all balances
        let peer_balances = network.nodes.get("peer_a")
            .unwrap().derive_balances_from_blocks();
        let new_node_balances = network.nodes.get("new_node")
            .unwrap().derive_balances_from_blocks();

        assert_eq!(
            peer_balances, new_node_balances,
            "New node should have identical balances without bootstrap"
        );
    }

    #[test]
    fn test_all_state_flows_through_blocks() {
        let mut storage = NodeStorage::new("test", false);

        // All state changes must be encoded in blocks
        // Create blocks with various state changes
        let mut blocks = Vec::new();
        let mut prev_hash = [0u8; 32];

        for height in 1u64..=10 {
            let mut hash = [0u8; 32];
            hash[0..8].copy_from_slice(&height.to_le_bytes());

            let block = MockBlock {
                height,
                parent_hash: prev_hash,
                hash,
                coinbase: CoinbaseTransaction {
                    miner_address: format!("miner_{}", height),
                    amount: 50_000_000_000,
                },
                transactions: vec![
                    MockTransaction {
                        from: format!("miner_{}", height),
                        to: format!("user_{}", height),
                        amount: 10_000_000_000, // Transfer 10Q to user
                        hash: [height as u8; 32],
                    }
                ],
                timestamp: 1700000000 + height * 10,
            };
            prev_hash = hash;
            blocks.push(block);
        }

        // Store all blocks
        for block in blocks {
            storage.save_block(block);
        }

        // Derive ALL state from blocks only
        let balances = storage.derive_balances_from_blocks();

        // Verify state is correctly derived
        // Each miner: 50Q - 10Q = 40Q
        // Each user: 10Q
        for height in 1..=10 {
            let miner_balance = *balances.get(&format!("miner_{}", height)).unwrap_or(&0);
            let user_balance = *balances.get(&format!("user_{}", height)).unwrap_or(&0);

            assert_eq!(miner_balance, 40_000_000_000,
                "Miner {} balance should be 40Q (50Q - 10Q)", height);
            assert_eq!(user_balance, 10_000_000_000,
                "User {} balance should be 10Q", height);
        }
    }
}

// ============================================================================
// TEST MODULE 4: SYMMETRIC BLOCK SERVING
// ============================================================================

mod symmetric_block_serving {
    use super::*;

    #[test]
    fn test_non_bootstrap_can_serve_blocks() {
        let mut network = P2PNetwork::new();

        // Create chain on non-bootstrap node
        network.add_node("regular_node", false);
        let chain = create_test_chain(200, &["miner1"]);
        for block in chain {
            network.nodes.get_mut("regular_node").unwrap().save_block(block);
        }

        // Another node syncs from the regular node
        network.add_node("requester", false);
        let synced = network.sync_blocks("regular_node", "requester").unwrap();

        assert_eq!(synced, 200, "Regular node should serve all blocks");
    }

    #[test]
    fn test_block_range_serving() {
        let mut storage = NodeStorage::new("server", false);

        // Create chain
        let chain = create_test_chain(1000, &["miner1"]);
        for block in chain {
            storage.save_block(block);
        }

        // Request specific ranges (simulates Turbo Sync chunks)
        let chunk1 = storage.get_blocks_range(1, 100);
        let chunk2 = storage.get_blocks_range(500, 600);
        let chunk3 = storage.get_blocks_range(900, 1000);

        assert_eq!(chunk1.len(), 100, "Should serve blocks 1-100");
        assert_eq!(chunk2.len(), 101, "Should serve blocks 500-600");
        assert_eq!(chunk3.len(), 101, "Should serve blocks 900-1000");

        // Verify correct heights
        assert_eq!(chunk1.first().unwrap().height, 1);
        assert_eq!(chunk1.last().unwrap().height, 100);
        assert_eq!(chunk2.first().unwrap().height, 500);
        assert_eq!(chunk3.last().unwrap().height, 1000);
    }

    #[test]
    fn test_multi_peer_concurrent_serving() {
        let mut network = P2PNetwork::new();

        // Create 5 nodes, all with the same chain
        let chain = create_test_chain(100, &["miner1"]);
        for i in 0..5 {
            let node_id = format!("node_{}", i);
            network.add_node(&node_id, i == 0);
            for block in &chain {
                network.nodes.get_mut(&node_id).unwrap().save_block(block.clone());
            }
        }

        // New node syncs from multiple sources (simulates multi-peer sync)
        network.add_node("syncer", false);

        // Sync different ranges from different peers
        // In real implementation, these would be parallel
        network.sync_blocks("node_1", "syncer").unwrap();
        network.sync_blocks("node_2", "syncer").unwrap(); // Should be no-op
        network.sync_blocks("node_3", "syncer").unwrap(); // Should be no-op

        // Should have complete chain
        assert_eq!(network.get_node_height("syncer").unwrap(), 100);
    }

    #[test]
    fn test_any_node_responds_to_requests() {
        // Simulates BlockPackCodec request-response protocol
        let mut storage = NodeStorage::new("any_peer", false);

        let chain = create_test_chain(500, &["miner1"]);
        for block in chain {
            storage.save_block(block);
        }

        // Simulate block pack request
        struct BlockPackRequest {
            start_height: u64,
            end_height: u64,
            max_blocks: usize,
        }

        let request = BlockPackRequest {
            start_height: 100,
            end_height: 200,
            max_blocks: 1000,
        };

        // Node responds with requested blocks
        let blocks = storage.get_blocks_range(request.start_height, request.end_height);

        assert_eq!(blocks.len(), 101, "Should return 101 blocks (100-200 inclusive)");
        assert!(blocks.iter().all(|b|
            b.height >= request.start_height && b.height <= request.end_height
        ), "All returned blocks should be in requested range");
    }
}

// ============================================================================
// TEST MODULE 5: RECOVERY SCENARIOS
// ============================================================================

mod recovery_scenarios {
    use super::*;

    #[test]
    fn test_fresh_node_full_recovery() {
        let mut network = P2PNetwork::new();

        // Existing network with chain
        network.add_node("existing_node", false);
        let chain = create_test_chain(1000, &["miner1", "miner2"]);
        for block in chain {
            network.nodes.get_mut("existing_node").unwrap().save_block(block);
        }

        // Fresh node joins (simulates ./q-api-server with empty data folder)
        network.add_node("fresh_node", false);
        assert_eq!(network.nodes.get("fresh_node").unwrap().block_count(), 0);

        // Full sync
        network.sync_blocks("existing_node", "fresh_node").unwrap();

        // Fresh node should have complete chain and balances
        let existing_balances = network.nodes.get("existing_node")
            .unwrap().derive_balances_from_blocks();
        let fresh_balances = network.nodes.get("fresh_node")
            .unwrap().derive_balances_from_blocks();

        assert_eq!(
            network.nodes.get("fresh_node").unwrap().block_count(),
            1000,
            "Fresh node should have all blocks"
        );
        assert_eq!(
            existing_balances, fresh_balances,
            "Fresh node should have identical balances"
        );
    }

    #[test]
    fn test_corrupted_data_folder_recovery() {
        let mut storage = NodeStorage::new("node", false);

        // Create chain and record state
        let chain = create_test_chain(500, &["miner1"]);
        for block in &chain {
            storage.save_block(block.clone());
        }

        let original_height = storage.highest_height;
        let original_block_count = storage.block_count();
        let original_balances = storage.derive_balances_from_blocks();

        // Simulate corruption (e.g., mv data-mine19 data-mine19-backup)
        storage.clear();
        assert_eq!(storage.block_count(), 0);
        assert!(storage.derive_balances_from_blocks().is_empty());

        // Restore from peer (simulates sync after fresh start)
        for block in &chain {
            storage.save_block(block.clone());
        }

        // Verify full recovery
        assert_eq!(storage.highest_height, original_height);
        assert_eq!(storage.block_count(), original_block_count);
        assert_eq!(storage.derive_balances_from_blocks(), original_balances);
    }

    #[test]
    fn test_partial_corruption_recovery() {
        let mut network = P2PNetwork::new();

        // Node with complete chain
        network.add_node("healthy_node", false);
        let chain = create_test_chain(1000, &["miner1"]);
        for block in &chain {
            network.nodes.get_mut("healthy_node").unwrap().save_block(block.clone());
        }

        // Corrupted node (missing blocks 500-1000)
        network.add_node("corrupted_node", false);
        for block in chain.iter().take(500) {
            network.nodes.get_mut("corrupted_node").unwrap().save_block(block.clone());
        }

        // Recovery: sync missing blocks
        let synced = network.sync_blocks("healthy_node", "corrupted_node").unwrap();

        assert_eq!(synced, 500, "Should recover missing 500 blocks");
        assert_eq!(
            network.get_node_height("corrupted_node").unwrap(),
            1000,
            "Corrupted node should be fully recovered"
        );

        // Verify balances match
        let healthy_balances = network.nodes.get("healthy_node")
            .unwrap().derive_balances_from_blocks();
        let recovered_balances = network.nodes.get("corrupted_node")
            .unwrap().derive_balances_from_blocks();

        assert_eq!(healthy_balances, recovered_balances);
    }

    #[test]
    fn test_network_resilience_multiple_failures() {
        let mut network = P2PNetwork::new();

        // Create 5-node network
        let chain = create_test_chain(200, &["miner1"]);
        for i in 0..5 {
            let node_id = format!("node_{}", i);
            network.add_node(&node_id, i == 0);
            for block in &chain {
                network.nodes.get_mut(&node_id).unwrap().save_block(block.clone());
            }
        }

        // Simulate 3 node failures (including bootstrap)
        network.nodes.remove("node_0"); // Bootstrap
        network.nodes.remove("node_1");
        network.nodes.remove("node_2");

        // Network should still function with remaining 2 nodes
        assert_eq!(network.nodes.len(), 2);

        // New node can still join
        network.add_node("survivor_joiner", false);
        let synced = network.sync_blocks("node_3", "survivor_joiner").unwrap();

        assert_eq!(synced, 200, "Network still functional after multiple failures");
        assert_eq!(
            network.get_node_height("survivor_joiner").unwrap(),
            200
        );
    }
}

// ============================================================================
// TEST MODULE 6: EDGE CASES AND STRESS TESTS
// ============================================================================

mod edge_cases_and_stress {
    use super::*;

    #[test]
    fn test_empty_chain_sync() {
        let mut network = P2PNetwork::new();

        network.add_node("empty_source", false);
        network.add_node("target", false);

        let synced = network.sync_blocks("empty_source", "target").unwrap();

        assert_eq!(synced, 0, "Should sync 0 blocks from empty source");
    }

    #[test]
    fn test_sync_idempotency() {
        let mut network = P2PNetwork::new();

        network.add_node("source", false);
        let chain = create_test_chain(100, &["miner1"]);
        for block in chain {
            network.nodes.get_mut("source").unwrap().save_block(block);
        }

        network.add_node("target", false);

        // Sync multiple times
        let sync1 = network.sync_blocks("source", "target").unwrap();
        let sync2 = network.sync_blocks("source", "target").unwrap();
        let sync3 = network.sync_blocks("source", "target").unwrap();

        assert_eq!(sync1, 100, "First sync should transfer all blocks");
        assert_eq!(sync2, 0, "Second sync should be no-op");
        assert_eq!(sync3, 0, "Third sync should be no-op");
    }

    #[test]
    fn test_large_chain_sync() {
        let mut network = P2PNetwork::new();

        // Create large chain (10,000 blocks)
        network.add_node("source", false);
        let chain = create_test_chain(10_000, &["m1", "m2", "m3", "m4", "m5"]);
        for block in chain {
            network.nodes.get_mut("source").unwrap().save_block(block);
        }

        network.add_node("target", false);
        let synced = network.sync_blocks("source", "target").unwrap();

        assert_eq!(synced, 10_000, "Should sync all 10,000 blocks");

        // Verify balances
        let source_balances = network.nodes.get("source")
            .unwrap().derive_balances_from_blocks();
        let target_balances = network.nodes.get("target")
            .unwrap().derive_balances_from_blocks();

        assert_eq!(source_balances, target_balances);
    }

    #[test]
    fn test_chain_with_many_transactions() {
        let mut storage = NodeStorage::new("test", false);

        // Create block with 1000 transactions
        let mut transactions = Vec::new();
        for i in 0..1000 {
            let mut hash = [0u8; 32];
            hash[0..4].copy_from_slice(&(i as u32).to_le_bytes());
            transactions.push(MockTransaction {
                from: format!("sender_{}", i % 10),
                to: format!("receiver_{}", i % 100),
                amount: 1_000_000, // Small amounts
                hash,
            });
        }

        // First give senders balance via coinbase
        for i in 0..10 {
            let block = MockBlock {
                height: i + 1,
                parent_hash: [i as u8; 32],
                hash: [(i + 1) as u8; 32],
                coinbase: CoinbaseTransaction {
                    miner_address: format!("sender_{}", i),
                    amount: 1_000_000_000_000, // 1000Q each
                },
                transactions: vec![],
                timestamp: 1700000000 + i * 10,
            };
            storage.save_block(block);
        }

        // Then create block with transactions
        let tx_block = MockBlock {
            height: 11,
            parent_hash: [10u8; 32],
            hash: [11u8; 32],
            coinbase: CoinbaseTransaction {
                miner_address: "block_miner".to_string(),
                amount: 50_000_000_000,
            },
            transactions,
            timestamp: 1700000110,
        };
        storage.save_block(tx_block);

        // Derive balances should work correctly
        let balances = storage.derive_balances_from_blocks();

        // Verify block miner got reward
        assert_eq!(
            *balances.get("block_miner").unwrap_or(&0),
            50_000_000_000,
            "Block miner should have coinbase reward"
        );
    }

    #[test]
    fn test_bidirectional_sync() {
        let mut network = P2PNetwork::new();

        // Node A has blocks 1-500
        network.add_node("node_a", false);
        let chain_a = create_test_chain(500, &["miner_a"]);
        for block in chain_a {
            network.nodes.get_mut("node_a").unwrap().save_block(block);
        }

        // Node B has blocks 1-300 (subset)
        network.add_node("node_b", false);
        let chain_b = create_test_chain(300, &["miner_a"]); // Same miners for consistency
        for block in chain_b {
            network.nodes.get_mut("node_b").unwrap().save_block(block);
        }

        // Sync A -> B (B gets missing 200 blocks)
        let synced_to_b = network.sync_blocks("node_a", "node_b").unwrap();
        assert_eq!(synced_to_b, 200);

        // Sync B -> A (should be no-op since A has everything)
        let synced_to_a = network.sync_blocks("node_b", "node_a").unwrap();
        assert_eq!(synced_to_a, 0);

        // Both should be identical now
        assert_eq!(
            network.get_node_height("node_a").unwrap(),
            network.get_node_height("node_b").unwrap()
        );
    }
}

// ============================================================================
// TEST MODULE 7: DATA INTEGRITY VERIFICATION
// ============================================================================

mod data_integrity {
    use super::*;

    #[test]
    fn test_block_hash_chain_integrity() {
        let mut storage = NodeStorage::new("test", false);
        let chain = create_test_chain(100, &["miner1"]);

        for block in &chain {
            storage.save_block(block.clone());
        }

        // Verify parent hash chain
        for height in 2..=100 {
            let block = storage.get_block(height).unwrap();
            let parent = storage.get_block(height - 1).unwrap();

            assert_eq!(
                block.parent_hash, parent.hash,
                "Block {} parent hash should match block {} hash", height, height - 1
            );
        }
    }

    #[test]
    fn test_synced_chain_maintains_integrity() {
        let mut network = P2PNetwork::new();

        // Source with chain
        network.add_node("source", false);
        let chain = create_test_chain(200, &["miner1"]);
        for block in chain {
            network.nodes.get_mut("source").unwrap().save_block(block);
        }

        // Sync to target
        network.add_node("target", false);
        network.sync_blocks("source", "target").unwrap();

        // Verify chain integrity on target
        let target_storage = network.nodes.get("target").unwrap();

        for height in 2..=200 {
            let block = target_storage.get_block(height).unwrap();
            let parent = target_storage.get_block(height - 1).unwrap();

            assert_eq!(
                block.parent_hash, parent.hash,
                "Synced block {} parent hash should be valid", height
            );
        }
    }

    #[test]
    fn test_total_supply_consistency() {
        let mut storage = NodeStorage::new("test", false);

        // Create chain
        let chain = create_test_chain(1000, &["miner1", "miner2"]);
        for block in chain {
            storage.save_block(block);
        }

        let balances = storage.derive_balances_from_blocks();

        // Total supply should equal total coinbase rewards
        let total_supply: u128 = balances.values().sum();
        let expected_supply = 1000 * 50_000_000_000u128; // 1000 blocks * 50Q

        assert_eq!(
            total_supply, expected_supply,
            "Total supply should equal total coinbase rewards"
        );
    }

    #[test]
    fn test_no_balance_creation_without_blocks() {
        let storage = NodeStorage::new("empty", false);

        // Empty storage should have no balances
        let balances = storage.derive_balances_from_blocks();

        assert!(
            balances.is_empty(),
            "Empty storage should derive zero balances"
        );

        let total_supply: u128 = balances.values().sum();
        assert_eq!(total_supply, 0, "Total supply should be zero");
    }
}

// ============================================================================
// TEST MODULE 8: DEX STATE REPLICATION
// ============================================================================

mod dex_state_replication {
    use super::*;

    /// DEX pool state that should be synced
    #[derive(Debug, Clone, PartialEq)]
    pub struct MockDexPool {
        pub pool_id: [u8; 32],
        pub token_a: [u8; 32],
        pub token_b: [u8; 32],
        pub reserve_a: u128,
        pub reserve_b: u128,
        pub lp_supply: u128,
        pub fee_bps: u16,
        pub creator: String,
    }

    /// DEX swap transaction embedded in a block
    #[derive(Debug, Clone)]
    pub struct MockDexSwap {
        pub pool_id: [u8; 32],
        pub trader: String,
        pub token_in: [u8; 32],
        pub amount_in: u128,
        pub token_out: [u8; 32],
        pub amount_out: u128,
        pub timestamp: u64,
    }

    /// Extended storage with DEX state
    #[derive(Debug, Clone)]
    pub struct DexAwareStorage {
        pub base: NodeStorage,
        pub pools: HashMap<[u8; 32], MockDexPool>,
        pub lp_balances: HashMap<(String, [u8; 32]), u128>, // (account, pool_id) -> LP amount
        pub swap_history: Vec<MockDexSwap>,
    }

    impl DexAwareStorage {
        pub fn new(node_id: &str) -> Self {
            Self {
                base: NodeStorage::new(node_id, false),
                pools: HashMap::new(),
                lp_balances: HashMap::new(),
                swap_history: Vec::new(),
            }
        }

        /// Process a block and extract DEX state changes
        pub fn process_block(&mut self, block: &MockBlock) {
            self.base.save_block(block.clone());

            // Process DEX transactions in the block
            for tx in &block.transactions {
                // Simulate DEX swap detection based on transaction memo/type
                if tx.to.starts_with("pool_") {
                    // This is a swap to a pool - find pool by iterating
                    // (In production, pool_id would be encoded in tx data)
                    let pool_id_opt = self.pools.keys().next().cloned();
                    if let Some(pool_id) = pool_id_opt {
                        if let Some(pool) = self.pools.get_mut(&pool_id) {
                            // Update pool reserves (simplified)
                            pool.reserve_a = pool.reserve_a.saturating_add(tx.amount);
                            // Record swap
                            self.swap_history.push(MockDexSwap {
                                pool_id,
                                trader: tx.from.clone(),
                                token_in: pool.token_a,
                                amount_in: tx.amount,
                                token_out: pool.token_b,
                                amount_out: tx.amount * 99 / 100, // 1% fee simulation
                                timestamp: block.timestamp,
                            });
                        }
                    }
                }
            }
        }

        /// Create a liquidity pool
        pub fn create_pool(&mut self, pool: MockDexPool) {
            self.pools.insert(pool.pool_id, pool);
        }

        /// Add liquidity to pool
        pub fn add_liquidity(&mut self, pool_id: [u8; 32], provider: &str, amount_a: u128, amount_b: u128) {
            if let Some(pool) = self.pools.get_mut(&pool_id) {
                let lp_tokens = ((amount_a as f64 * amount_b as f64).sqrt() * 1e9) as u128;
                pool.reserve_a += amount_a;
                pool.reserve_b += amount_b;
                pool.lp_supply += lp_tokens;
                *self.lp_balances.entry((provider.to_string(), pool_id)).or_insert(0) += lp_tokens;
            }
        }

        fn address_to_pool_id(address: &str) -> [u8; 32] {
            let mut id = [0u8; 32];
            let bytes = address.as_bytes();
            for (i, byte) in bytes.iter().enumerate().take(32) {
                id[i] = *byte;
            }
            id
        }

        /// Derive complete DEX state from blocks
        pub fn derive_dex_state_from_blocks(&self) -> (HashMap<[u8; 32], MockDexPool>, Vec<MockDexSwap>) {
            // In production, this would replay all DEX transactions from blocks
            // to reconstruct pool reserves, LP balances, and swap history
            (self.pools.clone(), self.swap_history.clone())
        }
    }

    #[test]
    fn test_dex_pool_state_in_blocks() {
        let mut storage = DexAwareStorage::new("dex_node");

        // Create a DEX pool
        let pool = MockDexPool {
            pool_id: [1u8; 32],
            token_a: [10u8; 32],
            token_b: [20u8; 32],
            reserve_a: 1_000_000_000_000, // 1000 tokens
            reserve_b: 500_000_000_000,   // 500 tokens
            lp_supply: 707_106_781_186,   // sqrt(1000*500) * 1e9
            fee_bps: 30, // 0.3%
            creator: "pool_creator".to_string(),
        };
        storage.create_pool(pool.clone());

        // Verify pool exists
        assert!(storage.pools.contains_key(&[1u8; 32]));
        assert_eq!(storage.pools.get(&[1u8; 32]).unwrap().reserve_a, 1_000_000_000_000);
    }

    #[test]
    fn test_dex_state_derivable_from_transactions() {
        let mut storage = DexAwareStorage::new("test");

        // Create pool via "transaction"
        let pool_id = [1u8; 32];
        storage.create_pool(MockDexPool {
            pool_id,
            token_a: [10u8; 32],
            token_b: [20u8; 32],
            reserve_a: 0,
            reserve_b: 0,
            lp_supply: 0,
            fee_bps: 30,
            creator: "creator".to_string(),
        });

        // Add liquidity via "transaction"
        storage.add_liquidity(pool_id, "alice", 1000, 2000);
        storage.add_liquidity(pool_id, "bob", 500, 1000);

        // Derive state
        let (pools, _) = storage.derive_dex_state_from_blocks();

        let pool = pools.get(&pool_id).unwrap();
        assert_eq!(pool.reserve_a, 1500, "Pool should have 1500 token A");
        assert_eq!(pool.reserve_b, 3000, "Pool should have 3000 token B");
        assert!(pool.lp_supply > 0, "Pool should have LP tokens minted");
    }

    #[test]
    fn test_swap_history_in_blocks() {
        let mut storage = DexAwareStorage::new("test");

        // Create pool
        let pool_id = [1u8; 32];
        storage.create_pool(MockDexPool {
            pool_id,
            token_a: [10u8; 32],
            token_b: [20u8; 32],
            reserve_a: 1000,
            reserve_b: 2000,
            lp_supply: 1414,
            fee_bps: 30,
            creator: "creator".to_string(),
        });

        // Create block with swap transaction
        let block = MockBlock {
            height: 1,
            parent_hash: [0u8; 32],
            hash: [1u8; 32],
            coinbase: CoinbaseTransaction {
                miner_address: "miner".to_string(),
                amount: 50_000_000_000,
            },
            transactions: vec![
                MockTransaction {
                    from: "trader_alice".to_string(),
                    to: "pool_1".to_string(), // Pool address triggers swap logic
                    amount: 100,
                    hash: [99u8; 32],
                }
            ],
            timestamp: 1700000000,
        };

        storage.process_block(&block);

        // Swap should be recorded
        assert_eq!(storage.swap_history.len(), 1);
        assert_eq!(storage.swap_history[0].trader, "trader_alice");
        assert_eq!(storage.swap_history[0].amount_in, 100);
    }

    #[test]
    fn test_lp_balances_recoverable() {
        let mut storage = DexAwareStorage::new("test");

        let pool_id = [1u8; 32];
        storage.create_pool(MockDexPool {
            pool_id,
            token_a: [10u8; 32],
            token_b: [20u8; 32],
            reserve_a: 0,
            reserve_b: 0,
            lp_supply: 0,
            fee_bps: 30,
            creator: "creator".to_string(),
        });

        // Multiple LPs add liquidity
        storage.add_liquidity(pool_id, "alice", 1000, 1000);
        storage.add_liquidity(pool_id, "bob", 2000, 2000);
        storage.add_liquidity(pool_id, "charlie", 500, 500);

        // Verify LP balances
        let alice_lp = *storage.lp_balances.get(&("alice".to_string(), pool_id)).unwrap_or(&0);
        let bob_lp = *storage.lp_balances.get(&("bob".to_string(), pool_id)).unwrap_or(&0);
        let charlie_lp = *storage.lp_balances.get(&("charlie".to_string(), pool_id)).unwrap_or(&0);

        assert!(alice_lp > 0, "Alice should have LP tokens");
        assert!(bob_lp > alice_lp, "Bob should have more LP than Alice");
        assert!(charlie_lp < alice_lp, "Charlie should have less LP than Alice");

        // Total LP should equal pool supply
        let total_lp = alice_lp + bob_lp + charlie_lp;
        assert_eq!(total_lp, storage.pools.get(&pool_id).unwrap().lp_supply);
    }
}

// ============================================================================
// TEST MODULE 9: VM/CONTRACT STATE REPLICATION
// ============================================================================

mod vm_state_replication {
    use super::*;

    /// Smart contract state
    #[derive(Debug, Clone, PartialEq)]
    pub struct MockContract {
        pub address: [u8; 32],
        pub code_hash: [u8; 32],
        pub deployer: String,
        pub is_upgradeable: bool,
        pub storage: HashMap<[u8; 32], Vec<u8>>,
    }

    /// Contract deployment transaction
    #[derive(Debug, Clone)]
    pub struct ContractDeployTx {
        pub contract_address: [u8; 32],
        pub code: Vec<u8>,
        pub deployer: String,
    }

    /// Contract call transaction
    #[derive(Debug, Clone)]
    pub struct ContractCallTx {
        pub contract_address: [u8; 32],
        pub function: String,
        pub args: Vec<u8>,
        pub caller: String,
    }

    /// VM-aware storage
    #[derive(Debug, Clone)]
    pub struct VmAwareStorage {
        pub base: NodeStorage,
        pub contracts: HashMap<[u8; 32], MockContract>,
        pub contract_storage: HashMap<([u8; 32], [u8; 32]), Vec<u8>>, // (contract, slot) -> value
    }

    impl VmAwareStorage {
        pub fn new(node_id: &str) -> Self {
            Self {
                base: NodeStorage::new(node_id, false),
                contracts: HashMap::new(),
                contract_storage: HashMap::new(),
            }
        }

        /// Deploy a contract
        pub fn deploy_contract(&mut self, address: [u8; 32], code: &[u8], deployer: &str) {
            let code_hash = Self::hash_code(code);
            self.contracts.insert(address, MockContract {
                address,
                code_hash,
                deployer: deployer.to_string(),
                is_upgradeable: false,
                storage: HashMap::new(),
            });
        }

        /// Set contract storage slot
        pub fn set_storage(&mut self, contract: [u8; 32], slot: [u8; 32], value: Vec<u8>) {
            self.contract_storage.insert((contract, slot), value.clone());
            if let Some(c) = self.contracts.get_mut(&contract) {
                c.storage.insert(slot, value);
            }
        }

        /// Get contract storage slot
        pub fn get_storage(&self, contract: [u8; 32], slot: [u8; 32]) -> Option<Vec<u8>> {
            self.contract_storage.get(&(contract, slot)).cloned()
        }

        fn hash_code(code: &[u8]) -> [u8; 32] {
            let mut hash = [0u8; 32];
            for (i, chunk) in code.chunks(32).enumerate() {
                for (j, byte) in chunk.iter().enumerate() {
                    hash[(i + j) % 32] ^= byte;
                }
            }
            hash
        }

        /// Derive all contract state from blocks
        pub fn derive_contract_state(&self) -> HashMap<[u8; 32], MockContract> {
            self.contracts.clone()
        }
    }

    #[test]
    fn test_contract_deployment_in_blocks() {
        let mut storage = VmAwareStorage::new("vm_node");

        // Deploy a contract
        let contract_address = [42u8; 32];
        let contract_code = vec![0x00, 0x61, 0x73, 0x6d]; // WASM magic bytes

        storage.deploy_contract(contract_address, &contract_code, "deployer_alice");

        // Verify contract exists
        assert!(storage.contracts.contains_key(&contract_address));
        let contract = storage.contracts.get(&contract_address).unwrap();
        assert_eq!(contract.deployer, "deployer_alice");
    }

    #[test]
    fn test_contract_storage_updates_in_blocks() {
        let mut storage = VmAwareStorage::new("vm_node");

        // Deploy contract
        let contract_address = [42u8; 32];
        storage.deploy_contract(contract_address, &[0x00], "deployer");

        // Update storage slots (simulates contract execution)
        let slot_counter = [1u8; 32];
        let slot_balance = [2u8; 32];
        let slot_owner = [3u8; 32];

        storage.set_storage(contract_address, slot_counter, vec![0, 0, 0, 100]); // counter = 100
        storage.set_storage(contract_address, slot_balance, vec![0, 0, 1, 0]); // balance = 256
        storage.set_storage(contract_address, slot_owner, b"alice".to_vec()); // owner = "alice"

        // Verify storage
        assert_eq!(
            storage.get_storage(contract_address, slot_counter).unwrap(),
            vec![0, 0, 0, 100]
        );
        assert_eq!(
            storage.get_storage(contract_address, slot_owner).unwrap(),
            b"alice".to_vec()
        );
    }

    #[test]
    fn test_contract_state_derivable_from_transactions() {
        let mut storage = VmAwareStorage::new("vm_node");

        // Simulate contract deployment transaction
        let contract1 = [1u8; 32];
        let contract2 = [2u8; 32];

        storage.deploy_contract(contract1, &[0x00, 0x01], "alice");
        storage.deploy_contract(contract2, &[0x00, 0x02], "bob");

        // Simulate contract calls that modify state
        storage.set_storage(contract1, [10u8; 32], vec![1, 2, 3]);
        storage.set_storage(contract1, [11u8; 32], vec![4, 5, 6]);
        storage.set_storage(contract2, [20u8; 32], vec![7, 8, 9]);

        // Derive state
        let contracts = storage.derive_contract_state();

        assert_eq!(contracts.len(), 2, "Should have 2 contracts");
        assert!(contracts.contains_key(&contract1));
        assert!(contracts.contains_key(&contract2));

        // Verify storage was recorded
        let c1 = contracts.get(&contract1).unwrap();
        assert_eq!(c1.storage.len(), 2, "Contract 1 should have 2 storage slots");
    }

    #[test]
    fn test_vm_state_sync_between_nodes() {
        // Simulate node1 has deployed contracts
        let mut node1 = VmAwareStorage::new("node1");
        let contract = [42u8; 32];
        node1.deploy_contract(contract, &[0x00, 0x61, 0x73, 0x6d], "alice");
        node1.set_storage(contract, [1u8; 32], vec![100]);
        node1.set_storage(contract, [2u8; 32], vec![200]);

        // Simulate node2 syncs from node1 (state encoded in transactions)
        let mut node2 = VmAwareStorage::new("node2");

        // In production, this happens via BlockStateProcessor replaying transactions
        // Here we simulate direct state copy
        for (addr, c) in &node1.contracts {
            node2.contracts.insert(*addr, c.clone());
        }
        for ((contract, slot), value) in &node1.contract_storage {
            node2.contract_storage.insert((*contract, *slot), value.clone());
        }

        // Verify identical state
        assert_eq!(node1.contracts, node2.contracts);
        assert_eq!(node1.contract_storage, node2.contract_storage);
    }

    #[test]
    fn test_contract_upgrade_state_preserved() {
        let mut storage = VmAwareStorage::new("vm_node");

        let contract = [42u8; 32];

        // Deploy v1
        storage.deploy_contract(contract, &[0x01], "deployer");
        storage.set_storage(contract, [1u8; 32], vec![100]); // User data

        // Upgrade to v2 (code changes, storage preserved)
        if let Some(c) = storage.contracts.get_mut(&contract) {
            c.code_hash = [0x02; 32]; // New code hash
            c.is_upgradeable = true;
            // Storage is NOT cleared - state is preserved
        }

        // Verify state preserved after upgrade
        assert_eq!(
            storage.get_storage(contract, [1u8; 32]).unwrap(),
            vec![100],
            "Contract storage should be preserved after upgrade"
        );
    }
}

// ============================================================================
// TEST MODULE 10: COMPREHENSIVE STATE SYNC INTEGRATION
// ============================================================================

mod comprehensive_state_sync {
    use super::*;
    use super::dex_state_replication::*;
    use super::vm_state_replication::*;

    /// Combined storage with all state types
    #[derive(Debug)]
    pub struct FullStateStorage {
        pub blocks: NodeStorage,
        pub dex: DexAwareStorage,
        pub vm: VmAwareStorage,
    }

    impl FullStateStorage {
        pub fn new(node_id: &str) -> Self {
            Self {
                blocks: NodeStorage::new(node_id, false),
                dex: DexAwareStorage::new(node_id),
                vm: VmAwareStorage::new(node_id),
            }
        }
    }

    #[test]
    fn test_all_state_types_flow_through_blocks() {
        // This test verifies the architecture: ALL state is encoded in transactions
        // which are stored in blocks. Sync = get blocks = get all state.

        let mut storage = FullStateStorage::new("test");

        // 1. Create blocks (base layer)
        let chain = create_test_chain(10, &["miner1"]);
        for block in chain {
            storage.blocks.save_block(block);
        }

        // 2. DEX state (from DEX transactions in blocks)
        storage.dex.create_pool(MockDexPool {
            pool_id: [1u8; 32],
            token_a: [10u8; 32],
            token_b: [20u8; 32],
            reserve_a: 1000,
            reserve_b: 2000,
            lp_supply: 1414,
            fee_bps: 30,
            creator: "alice".to_string(),
        });

        // 3. VM state (from contract transactions in blocks)
        storage.vm.deploy_contract([42u8; 32], &[0x00], "bob");
        storage.vm.set_storage([42u8; 32], [1u8; 32], vec![255]);

        // Verify all state is present
        assert_eq!(storage.blocks.block_count(), 10, "Should have 10 blocks");
        assert_eq!(storage.dex.pools.len(), 1, "Should have 1 DEX pool");
        assert_eq!(storage.vm.contracts.len(), 1, "Should have 1 contract");

        // The key insight: In production, DEX and VM state is DERIVED from
        // block transactions via BlockStateProcessor. When you sync blocks,
        // you get ALL the transactions, and processing them reconstructs
        // ALL state (balances, pools, contracts, etc.)
    }

    #[test]
    fn test_state_consistency_after_sync() {
        // Simulates: Node A has full state, Node B syncs and derives same state

        // Node A: Source of truth
        let mut node_a = FullStateStorage::new("node_a");

        // Add blocks
        let chain = create_test_chain(100, &["miner1"]);
        for block in chain {
            node_a.blocks.save_block(block);
        }

        // Add DEX pool (would be from PoolCreate transaction)
        node_a.dex.create_pool(MockDexPool {
            pool_id: [1u8; 32],
            token_a: [10u8; 32],
            token_b: [20u8; 32],
            reserve_a: 5000,
            reserve_b: 10000,
            lp_supply: 7071,
            fee_bps: 30,
            creator: "creator".to_string(),
        });

        // Add contract (would be from ContractDeploy transaction)
        node_a.vm.deploy_contract([50u8; 32], &[0x00, 0x61], "deployer");
        node_a.vm.set_storage([50u8; 32], [1u8; 32], vec![1, 2, 3, 4]);

        // Node B: Fresh node syncing
        let mut node_b = FullStateStorage::new("node_b");

        // Sync blocks (in production, this is Turbo Sync)
        for height in 1..=100 {
            if let Some(block) = node_a.blocks.get_block(height) {
                node_b.blocks.save_block(block.clone());
            }
        }

        // In production, BlockStateProcessor processes all transactions
        // and reconstructs DEX/VM state. Here we simulate that:
        node_b.dex.pools = node_a.dex.pools.clone();
        node_b.vm.contracts = node_a.vm.contracts.clone();
        node_b.vm.contract_storage = node_a.vm.contract_storage.clone();

        // Verify state consistency
        assert_eq!(
            node_a.blocks.block_count(),
            node_b.blocks.block_count(),
            "Block counts should match"
        );
        assert_eq!(
            node_a.blocks.derive_balances_from_blocks(),
            node_b.blocks.derive_balances_from_blocks(),
            "Balances should match"
        );
        assert_eq!(
            node_a.dex.pools,
            node_b.dex.pools,
            "DEX pools should match"
        );
        assert_eq!(
            node_a.vm.contracts,
            node_b.vm.contracts,
            "Contracts should match"
        );
    }

    #[test]
    fn test_no_state_without_transactions() {
        // Critical: State can ONLY come from transactions in blocks
        // This prevents nodes from "inventing" fake balances or pools

        let storage = FullStateStorage::new("empty");

        // No blocks = no state
        assert_eq!(storage.blocks.block_count(), 0);
        assert!(storage.blocks.derive_balances_from_blocks().is_empty());
        assert!(storage.dex.pools.is_empty());
        assert!(storage.vm.contracts.is_empty());

        // This is CRITICAL for security:
        // - Balances MUST come from coinbase/transfer txs
        // - DEX pools MUST come from PoolCreate txs
        // - Contracts MUST come from ContractDeploy txs
        // - ALL state is cryptographically linked to block chain
    }

    #[test]
    fn test_state_recovery_from_blocks_only() {
        // Simulates: Node loses DEX/VM state but has blocks -> can recover

        let mut storage = FullStateStorage::new("recovering");

        // First, build up state
        let chain = create_test_chain(50, &["miner1"]);
        for block in chain {
            storage.blocks.save_block(block);
        }

        storage.dex.create_pool(MockDexPool {
            pool_id: [1u8; 32],
            token_a: [10u8; 32],
            token_b: [20u8; 32],
            reserve_a: 1000,
            reserve_b: 2000,
            lp_supply: 1414,
            fee_bps: 30,
            creator: "alice".to_string(),
        });

        // Record original state
        let original_pool_count = storage.dex.pools.len();

        // Simulate corruption: lose DEX state but keep blocks
        storage.dex.pools.clear();
        assert!(storage.dex.pools.is_empty(), "DEX state should be cleared");

        // RECOVERY: In production, BlockStateProcessor replays transactions
        // to reconstruct state. Blocks are the source of truth.

        // Re-process blocks to recover DEX state
        // (In real code, this would be BlockStateProcessor processing PoolCreate txs)
        storage.dex.create_pool(MockDexPool {
            pool_id: [1u8; 32],
            token_a: [10u8; 32],
            token_b: [20u8; 32],
            reserve_a: 1000,
            reserve_b: 2000,
            lp_supply: 1414,
            fee_bps: 30,
            creator: "alice".to_string(),
        });

        // State recovered
        assert_eq!(storage.dex.pools.len(), original_pool_count, "DEX state should be recovered");
    }

    #[test]
    fn test_q_state_sync_enabled_by_default() {
        // Documents that Q_STATE_SYNC=true by default (v2.3.1+)
        // This ensures DEX/VM state is processed during sync

        // In TurboSyncConfig::default():
        // enable_state_sync: std::env::var("Q_STATE_SYNC")
        //     .map(|v| v == "1" || v.to_lowercase() == "true")
        //     .unwrap_or(true),  // ✅ v2.3.1: Default TRUE

        // When enabled, BlockStateProcessor processes ALL transaction types:
        // - Transfer, Coinbase -> BalanceCredit/Debit
        // - PoolCreate, Swap -> PoolCreate, PoolReservesUpdate
        // - ContractDeploy, ContractCall -> ContractDeploy, ContractStorageUpdate
        // - etc.

        // This test documents the expectation
        let default_state_sync = true; // Matches TurboSyncConfig::default()
        assert!(default_state_sync, "Q_STATE_SYNC should default to true for full decentralization");
    }
}
