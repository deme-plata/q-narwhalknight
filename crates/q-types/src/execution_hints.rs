//! CHIRON-Style Execution Hints for Parallel Block Processing
//!
//! v1.5.0-beta: Implementation based on CHIRON paper (https://arxiv.org/abs/2401.14278)
//!
//! This module enables ~30% faster node synchronization by:
//! 1. Pre-computing transaction dependency graphs during block production
//! 2. Enabling parallel state application during sync
//! 3. Using Block-STM validation for security
//!
//! ## Key Insight
//!
//! Q-NarwhalKnight's `balance_updates` already contains pre/post state.
//! We just need to compute the dependency graph to enable parallelism.
//!
//! ## Usage
//!
//! ```rust,ignore
//! // Block producer computes hints:
//! let hints = BlockExecutionHints::compute_from_transactions(&block.transactions);
//!
//! // Syncing node applies with hints:
//! for batch in &hints.parallel_batches {
//!     batch.par_iter().for_each(|tx_idx| execute_tx(tx_idx));
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Transaction ID (index within block)
/// u16 supports up to 65535 transactions per block (more than sufficient)
pub type TxIndex = u16;

/// Address type (re-export for convenience)
pub type Address = crate::Address;

/// Execution hints for a block (CHIRON-style)
///
/// These hints enable parallel state application during sync by:
/// 1. Recording which transactions depend on which others
/// 2. Grouping independent transactions into parallel batches
/// 3. Providing access sets for Block-STM validation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BlockExecutionHints {
    /// Version for forward compatibility
    pub version: u8,

    /// Transaction dependency graph: tx_index -> [depends_on_tx_indices]
    /// If tx 5 depends on tx 2 and tx 3: dependencies[5] = [2, 3]
    pub dependencies: HashMap<TxIndex, Vec<TxIndex>>,

    /// Independent transaction groups (can execute in parallel)
    /// Example: [[0, 3, 7], [1, 4], [2, 5, 6]] = 3 parallel batches
    /// Each batch can execute fully in parallel; batches execute sequentially
    pub parallel_batches: Vec<Vec<TxIndex>>,

    /// Address access sets per transaction (for Block-STM validation)
    /// Used to detect conflicts if hints are wrong
    pub access_sets: Vec<TxAccessSet>,

    /// Blake3 hash of hints for integrity verification
    /// Allows detecting tampered hints
    pub hints_hash: [u8; 32],

    /// Number of potentially conflicting transactions (for metrics)
    pub conflict_count: u16,
}

/// Access set for a single transaction
///
/// Records which addresses are read and written by a transaction.
/// Used for:
/// 1. Computing dependency graph
/// 2. Block-STM validation (re-verify reads after parallel execution)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TxAccessSet {
    /// Addresses read by this transaction (source balances)
    pub reads: Vec<Address>,
    /// Addresses written by this transaction (destination balances)
    pub writes: Vec<Address>,
}

impl BlockExecutionHints {
    /// Create empty hints (backwards compatible - no parallelism)
    pub fn empty() -> Self {
        Self {
            version: 1,
            dependencies: HashMap::new(),
            parallel_batches: Vec::new(),
            access_sets: Vec::new(),
            hints_hash: [0u8; 32],
            conflict_count: 0,
        }
    }

    /// Check if hints are available and useful
    pub fn has_hints(&self) -> bool {
        !self.parallel_batches.is_empty()
    }

    /// Get number of parallel batches (execution steps)
    pub fn parallelism(&self) -> usize {
        self.parallel_batches.len()
    }

    /// Get maximum batch size (peak parallelism)
    pub fn max_batch_size(&self) -> usize {
        self.parallel_batches.iter().map(|b| b.len()).max().unwrap_or(0)
    }

    /// Get speedup factor (sequential steps / parallel steps)
    pub fn speedup_factor(&self) -> f32 {
        let total_txs: usize = self.parallel_batches.iter().map(|b| b.len()).sum();
        if total_txs == 0 {
            return 1.0;
        }
        total_txs as f32 / self.parallel_batches.len() as f32
    }

    /// Compute hints from transactions (done by block producer)
    ///
    /// This analyzes read/write sets to build a dependency graph,
    /// then uses topological sort to create parallel execution batches.
    ///
    /// # Arguments
    /// * `transactions` - Block transactions to analyze
    ///
    /// # Returns
    /// Computed execution hints with dependency graph and parallel batches
    pub fn compute_from_transactions(transactions: &[crate::Transaction]) -> Self {
        if transactions.is_empty() {
            return Self::empty();
        }

        let start = std::time::Instant::now();

        // Phase 1: Build access sets (read/write per transaction)
        let mut access_sets = Vec::with_capacity(transactions.len());
        let mut write_map: HashMap<Address, Vec<TxIndex>> = HashMap::new();

        for (idx, tx) in transactions.iter().enumerate() {
            let tx_idx = idx as TxIndex;

            // Read set: source address (we read the balance)
            let reads = vec![tx.from.clone()];

            // Write set: both from (deduction) and to (credit)
            let writes = if tx.from == tx.to {
                vec![tx.from.clone()]
            } else {
                vec![tx.from.clone(), tx.to.clone()]
            };

            // Track which transactions write to each address
            for addr in &writes {
                write_map.entry(addr.clone()).or_default().push(tx_idx);
            }

            access_sets.push(TxAccessSet { reads, writes });
        }

        // Phase 2: Build dependency graph
        // Dependencies arise from:
        // - Read-After-Write (RAW): reading a balance written by earlier tx
        // - Write-After-Write (WAW): writing to same address as earlier tx
        let mut dependencies: HashMap<TxIndex, Vec<TxIndex>> = HashMap::new();
        let mut conflict_count = 0u16;

        for (idx, access) in access_sets.iter().enumerate() {
            let tx_idx = idx as TxIndex;
            let mut deps = Vec::new();

            // Check RAW dependencies (reads from previously written addresses)
            for read_addr in &access.reads {
                if let Some(writers) = write_map.get(read_addr) {
                    for &writer_idx in writers {
                        if writer_idx < tx_idx && !deps.contains(&writer_idx) {
                            deps.push(writer_idx);
                        }
                    }
                }
            }

            // Check WAW dependencies (writes to previously written addresses)
            for write_addr in &access.writes {
                if let Some(writers) = write_map.get(write_addr) {
                    for &writer_idx in writers {
                        if writer_idx < tx_idx && !deps.contains(&writer_idx) {
                            deps.push(writer_idx);
                        }
                    }
                }
            }

            if !deps.is_empty() {
                conflict_count += 1;
                deps.sort_unstable();
                deps.dedup();
                dependencies.insert(tx_idx, deps);
            }
        }

        // Phase 3: Compute parallel batches via topological sort
        let parallel_batches = Self::compute_parallel_batches(
            transactions.len(),
            &dependencies,
        );

        // Phase 4: Compute integrity hash
        let hints_data = bincode::serialize(&(&dependencies, &parallel_batches))
            .unwrap_or_default();
        let hints_hash: [u8; 32] = blake3::hash(&hints_data).into();

        let elapsed = start.elapsed();
        tracing::debug!(
            "🔧 [CHIRON] Computed hints for {} txs in {:?}: {} batches, {:.1}x speedup, {} conflicts",
            transactions.len(),
            elapsed,
            parallel_batches.len(),
            parallel_batches.iter().map(|b| b.len()).sum::<usize>() as f32 / parallel_batches.len().max(1) as f32,
            conflict_count
        );

        Self {
            version: 1,
            dependencies,
            parallel_batches,
            access_sets,
            hints_hash,
            conflict_count,
        }
    }

    /// Compute parallel execution batches via topological sort
    ///
    /// Transactions with no unsatisfied dependencies can run in parallel.
    /// This uses a level-based approach: each "level" is a parallel batch.
    fn compute_parallel_batches(
        tx_count: usize,
        dependencies: &HashMap<TxIndex, Vec<TxIndex>>,
    ) -> Vec<Vec<TxIndex>> {
        if tx_count == 0 {
            return vec![];
        }

        let mut batches = Vec::new();
        let mut completed: HashSet<TxIndex> = HashSet::with_capacity(tx_count);
        let mut remaining: HashSet<TxIndex> = (0..tx_count as TxIndex).collect();

        while !remaining.is_empty() {
            // Find all transactions whose dependencies are satisfied
            let ready: Vec<TxIndex> = remaining
                .iter()
                .filter(|&&tx_idx| {
                    dependencies
                        .get(&tx_idx)
                        .map(|deps| deps.iter().all(|d| completed.contains(d)))
                        .unwrap_or(true) // No dependencies = ready
                })
                .copied()
                .collect();

            if ready.is_empty() {
                // Circular dependency detected (shouldn't happen with valid txs)
                // Fall back to sequential for remaining transactions
                tracing::warn!(
                    "⚠️ [CHIRON] Circular dependency detected, {} txs forced sequential",
                    remaining.len()
                );
                let mut sequential: Vec<TxIndex> = remaining.iter().copied().collect();
                sequential.sort_unstable();
                for tx in sequential {
                    batches.push(vec![tx]);
                }
                break;
            }

            // Add ready transactions as a parallel batch
            for &tx in &ready {
                remaining.remove(&tx);
                completed.insert(tx);
            }

            // Sort for deterministic ordering
            let mut sorted_ready = ready;
            sorted_ready.sort_unstable();
            batches.push(sorted_ready);
        }

        batches
    }

    /// Verify hints integrity using the stored hash
    pub fn verify_integrity(&self) -> bool {
        let hints_data = bincode::serialize(&(&self.dependencies, &self.parallel_batches))
            .unwrap_or_default();
        let computed_hash: [u8; 32] = blake3::hash(&hints_data).into();
        computed_hash == self.hints_hash
    }

    /// Get transactions in a specific batch
    pub fn get_batch(&self, batch_idx: usize) -> Option<&Vec<TxIndex>> {
        self.parallel_batches.get(batch_idx)
    }

    /// Get dependencies for a specific transaction
    pub fn get_dependencies(&self, tx_idx: TxIndex) -> Option<&Vec<TxIndex>> {
        self.dependencies.get(&tx_idx)
    }

    /// Get access set for a specific transaction
    pub fn get_access_set(&self, tx_idx: TxIndex) -> Option<&TxAccessSet> {
        self.access_sets.get(tx_idx as usize)
    }

    /// Serialize hints to bytes (for network transmission)
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize hints from bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        bincode::deserialize(data).ok()
    }

    /// Get estimated memory usage in bytes
    pub fn estimated_memory(&self) -> usize {
        // Rough estimate:
        // - dependencies: tx_count * avg_deps * 2 bytes
        // - parallel_batches: tx_count * 2 bytes
        // - access_sets: tx_count * 2 * avg_addr_size
        let tx_count = self.access_sets.len();
        let avg_addr_size = 64; // Conservative estimate
        tx_count * (4 + 2 + 2 * avg_addr_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Transaction;
    use chrono::Utc;
    use sha3::{Digest, Sha3_256};

    /// Deterministically derive a 32-byte test address from a string name.
    /// Stable across runs so dependency-graph asserts are reproducible.
    fn addr(name: &str) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(name.as_bytes());
        hasher.finalize().into()
    }

    fn make_tx(from: &str, to: &str) -> Transaction {
        Transaction {
            id: [0u8; 32],
            from: addr(from),
            to: addr(to),
            amount: 100,
            fee: 0,
            nonce: 0,
            signature: vec![],
            timestamp: Utc::now(),
            data: vec![],
            token_type: crate::TokenType::QUG,
            fee_token_type: crate::TokenType::QUGUSD,
            tx_type: crate::TransactionType::Transfer,
            pqc_signature: None,
            signature_phase: crate::TxSignaturePhase::Phase0Ed25519,
            pqc_public_key: None,
            zk_proof_bundle: None,
            privacy_level: crate::TransactionPrivacyLevel::Transparent,
            bulletproof: None,
            nullifier: None,
            memo: None,
        }
    }

    #[test]
    fn test_empty_hints() {
        let hints = BlockExecutionHints::empty();
        assert!(!hints.has_hints());
        assert_eq!(hints.parallelism(), 0);
        assert_eq!(hints.speedup_factor(), 1.0);
    }

    #[test]
    fn test_independent_transactions() {
        // All transactions are independent (different addresses)
        let txs = vec![
            make_tx("Alice", "Bob"),
            make_tx("Carol", "Dave"),
            make_tx("Eve", "Frank"),
            make_tx("Grace", "Henry"),
        ];

        let hints = BlockExecutionHints::compute_from_transactions(&txs);

        assert!(hints.has_hints());
        // All 4 transactions should be in a single parallel batch
        assert_eq!(hints.parallelism(), 1);
        assert_eq!(hints.max_batch_size(), 4);
        assert_eq!(hints.speedup_factor(), 4.0);
        assert_eq!(hints.conflict_count, 0);
    }

    #[test]
    fn test_dependent_transactions() {
        // TX1 depends on TX0 (Bob is receiver then sender)
        let txs = vec![
            make_tx("Alice", "Bob"),    // TX0
            make_tx("Bob", "Carol"),    // TX1 - depends on TX0
        ];

        let hints = BlockExecutionHints::compute_from_transactions(&txs);

        assert!(hints.has_hints());
        // Should be 2 sequential batches
        assert_eq!(hints.parallelism(), 2);
        assert_eq!(hints.conflict_count, 1);

        // TX1 should depend on TX0
        assert!(hints.get_dependencies(1).is_some());
        assert!(hints.get_dependencies(1).unwrap().contains(&0));
    }

    #[test]
    fn test_mixed_transactions() {
        // Mix of independent and dependent transactions
        let txs = vec![
            make_tx("Alice", "Bob"),     // TX0
            make_tx("Carol", "Dave"),    // TX1 - independent
            make_tx("Bob", "Eve"),       // TX2 - depends on TX0
            make_tx("Frank", "Grace"),   // TX3 - independent
            make_tx("Eve", "Henry"),     // TX4 - depends on TX2
            make_tx("Dave", "Ivan"),     // TX5 - depends on TX1
        ];

        let hints = BlockExecutionHints::compute_from_transactions(&txs);

        assert!(hints.has_hints());
        // Expected batches:
        // Batch 1: [0, 1, 3] - all independent
        // Batch 2: [2, 5] - depend on batch 1
        // Batch 3: [4] - depends on batch 2
        assert_eq!(hints.parallelism(), 3);

        // Speedup: 6 txs / 3 batches = 2x
        assert!(hints.speedup_factor() > 1.5);
    }

    #[test]
    fn test_integrity_verification() {
        let txs = vec![
            make_tx("Alice", "Bob"),
            make_tx("Carol", "Dave"),
        ];

        let hints = BlockExecutionHints::compute_from_transactions(&txs);
        assert!(hints.verify_integrity());

        // Tamper with hints
        let mut tampered = hints.clone();
        tampered.conflict_count = 999;
        // Hash should still match (it's computed from dependencies + batches only)
        assert!(tampered.verify_integrity());

        // Actually tamper with the hashed data
        let mut tampered2 = hints.clone();
        tampered2.parallel_batches.push(vec![99]);
        assert!(!tampered2.verify_integrity());
    }

    #[test]
    fn test_serialization() {
        let txs = vec![
            make_tx("Alice", "Bob"),
            make_tx("Carol", "Dave"),
        ];

        let hints = BlockExecutionHints::compute_from_transactions(&txs);
        let bytes = hints.to_bytes();
        let restored = BlockExecutionHints::from_bytes(&bytes).unwrap();

        assert_eq!(hints.parallelism(), restored.parallelism());
        assert_eq!(hints.hints_hash, restored.hints_hash);
    }

    #[test]
    fn test_self_transfer() {
        // Self-transfer (from == to)
        let txs = vec![
            make_tx("Alice", "Alice"),
        ];

        let hints = BlockExecutionHints::compute_from_transactions(&txs);
        assert!(hints.has_hints());
        assert_eq!(hints.access_sets[0].reads.len(), 1);
        assert_eq!(hints.access_sets[0].writes.len(), 1);
    }

    #[test]
    fn test_large_block() {
        // Simulate a block with many transactions
        let txs: Vec<Transaction> = (0..100)
            .map(|i| make_tx(&format!("User{}", i), &format!("User{}", i + 100)))
            .collect();

        let start = std::time::Instant::now();
        let hints = BlockExecutionHints::compute_from_transactions(&txs);
        let elapsed = start.elapsed();

        assert!(hints.has_hints());
        // All 100 are independent
        assert_eq!(hints.parallelism(), 1);
        assert_eq!(hints.max_batch_size(), 100);

        // Should compute quickly (< 1ms for 100 txs)
        assert!(elapsed.as_millis() < 10);
    }
}
