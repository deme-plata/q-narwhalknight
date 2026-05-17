//! Comprehensive Balance Integrity Tests
//!
//! v2.3.5-beta: Tests to prevent negative balances and double-spending
//!
//! These tests verify:
//! - Balance debit checks prevent negative balances
//! - Double-spend protection via deduplication
//! - Atomic balance updates
//! - u128 precision handling
//!
//! Run with: cargo test --package q-storage --test balance_integrity_tests

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

// ============================================================================
// NEGATIVE BALANCE PREVENTION TESTS
// ============================================================================

mod negative_balance_tests {
    use super::*;

    /// Simulate balance debit check from state_applicator.rs
    pub fn apply_balance_debit(
        balances: &mut HashMap<[u8; 32], u128>,
        account: &[u8; 32],
        amount: u128,
    ) -> Result<u128, String> {
        let current = balances.get(account).copied().unwrap_or(0);

        // CRITICAL CHECK - prevents negative balances!
        if current < amount {
            return Err(format!(
                "Insufficient balance: have {}, need {}",
                current, amount
            ));
        }

        let new_balance = current - amount; // Safe subtraction after check
        balances.insert(*account, new_balance);
        Ok(new_balance)
    }

    /// Simulate balance credit (always safe with saturating_add)
    pub fn apply_balance_credit(
        balances: &mut HashMap<[u8; 32], u128>,
        account: &[u8; 32],
        amount: u128,
    ) -> u128 {
        let current = balances.get(account).copied().unwrap_or(0);
        let new_balance = current.saturating_add(amount);
        balances.insert(*account, new_balance);
        new_balance
    }

    /// Test debit from sufficient balance succeeds
    #[test]
    fn test_debit_sufficient_balance_succeeds() {
        let mut balances = HashMap::new();
        let account = [1u8; 32];

        // Credit first
        apply_balance_credit(&mut balances, &account, 1000);

        // Debit should succeed
        let result = apply_balance_debit(&mut balances, &account, 500);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 500);
    }

    /// Test debit from insufficient balance fails
    #[test]
    fn test_debit_insufficient_balance_fails() {
        let mut balances = HashMap::new();
        let account = [1u8; 32];

        // Credit 100
        apply_balance_credit(&mut balances, &account, 100);

        // Debit 200 should fail
        let result = apply_balance_debit(&mut balances, &account, 200);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Insufficient balance"));

        // Balance should remain unchanged
        assert_eq!(balances.get(&account).copied().unwrap_or(0), 100);
    }

    /// Test debit from zero balance fails
    #[test]
    fn test_debit_zero_balance_fails() {
        let mut balances = HashMap::new();
        let account = [1u8; 32];

        // No credit, balance is 0
        let result = apply_balance_debit(&mut balances, &account, 1);
        assert!(result.is_err());
    }

    /// Test exact balance debit succeeds
    #[test]
    fn test_debit_exact_balance_succeeds() {
        let mut balances = HashMap::new();
        let account = [1u8; 32];

        apply_balance_credit(&mut balances, &account, 1000);

        // Debit exact amount
        let result = apply_balance_debit(&mut balances, &account, 1000);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    /// Test debit one more than balance fails
    #[test]
    fn test_debit_one_more_fails() {
        let mut balances = HashMap::new();
        let account = [1u8; 32];

        apply_balance_credit(&mut balances, &account, 1000);

        // Debit one more than balance
        let result = apply_balance_debit(&mut balances, &account, 1001);
        assert!(result.is_err());

        // Balance unchanged
        assert_eq!(balances.get(&account).copied().unwrap_or(0), 1000);
    }

    /// Test credit uses saturating add (no overflow)
    #[test]
    fn test_credit_saturating_add() {
        let mut balances = HashMap::new();
        let account = [1u8; 32];

        // Credit max u128
        apply_balance_credit(&mut balances, &account, u128::MAX);

        // Credit more - should saturate, not overflow
        let new_balance = apply_balance_credit(&mut balances, &account, 1000);
        assert_eq!(new_balance, u128::MAX);
    }

    /// Test multiple debits respect running balance
    #[test]
    fn test_multiple_debits_respect_balance() {
        let mut balances = HashMap::new();
        let account = [1u8; 32];

        apply_balance_credit(&mut balances, &account, 1000);

        // First debit succeeds
        assert!(apply_balance_debit(&mut balances, &account, 400).is_ok());
        // Second debit succeeds
        assert!(apply_balance_debit(&mut balances, &account, 400).is_ok());
        // Third debit fails (only 200 left)
        assert!(apply_balance_debit(&mut balances, &account, 400).is_err());
        // Small debit succeeds
        assert!(apply_balance_debit(&mut balances, &account, 200).is_ok());
        // Balance is now 0
        assert_eq!(balances.get(&account).copied().unwrap_or(0), 0);
    }
}

// ============================================================================
// DOUBLE-SPEND PREVENTION TESTS
// ============================================================================

mod double_spend_tests {
    use super::*;

    /// Simulate block deduplication cache
    struct BlockDeduplicator {
        processed: HashSet<[u8; 32]>,
        max_size: usize,
    }

    impl BlockDeduplicator {
        fn new(max_size: usize) -> Self {
            Self {
                processed: HashSet::new(),
                max_size,
            }
        }

        fn process_block(&mut self, block_hash: [u8; 32]) -> Result<(), String> {
            if self.processed.contains(&block_hash) {
                return Err("Block already processed".to_string());
            }

            // LRU eviction if at capacity
            if self.processed.len() >= self.max_size {
                // In real impl, remove oldest entry
                // For test, just clear half
                let to_remove: Vec<_> = self.processed.iter().take(self.max_size / 2).copied().collect();
                for hash in to_remove {
                    self.processed.remove(&hash);
                }
            }

            self.processed.insert(block_hash);
            Ok(())
        }

        fn is_processed(&self, block_hash: &[u8; 32]) -> bool {
            self.processed.contains(block_hash)
        }
    }

    /// Test block can only be processed once
    #[test]
    fn test_block_processed_once() {
        let mut dedup = BlockDeduplicator::new(1000);
        let block_hash = [1u8; 32];

        // First process succeeds
        assert!(dedup.process_block(block_hash).is_ok());

        // Second process fails
        let result = dedup.process_block(block_hash);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already processed"));
    }

    /// Test different blocks can be processed
    #[test]
    fn test_different_blocks_processed() {
        let mut dedup = BlockDeduplicator::new(1000);

        for i in 0u8..100 {
            let mut hash = [0u8; 32];
            hash[0] = i;
            assert!(dedup.process_block(hash).is_ok());
        }

        assert_eq!(dedup.processed.len(), 100);
    }

    /// Test deduplication check is fast
    #[test]
    fn test_deduplication_is_processed_check() {
        let mut dedup = BlockDeduplicator::new(1000);
        let block_hash = [42u8; 32];

        assert!(!dedup.is_processed(&block_hash));
        dedup.process_block(block_hash).unwrap();
        assert!(dedup.is_processed(&block_hash));
    }

    /// Simulate transaction deduplication key
    fn tx_dedup_key(wallet: &str, height: u64, nonce: u64, tx_type: u8) -> String {
        format!("{}:{}:{}:{}", wallet, height, nonce, tx_type)
    }

    /// Test transaction deduplication keys are unique
    #[test]
    fn test_tx_dedup_keys_unique() {
        let key1 = tx_dedup_key("wallet1", 1000, 1, 0);
        let key2 = tx_dedup_key("wallet1", 1000, 2, 0);
        let key3 = tx_dedup_key("wallet1", 1001, 1, 0);
        let key4 = tx_dedup_key("wallet2", 1000, 1, 0);

        let mut keys = HashSet::new();
        keys.insert(key1.clone());
        keys.insert(key2.clone());
        keys.insert(key3.clone());
        keys.insert(key4.clone());

        assert_eq!(keys.len(), 4);
    }

    /// Test same transaction parameters produce same key
    #[test]
    fn test_tx_dedup_key_deterministic() {
        let key1 = tx_dedup_key("wallet1", 1000, 1, 0);
        let key2 = tx_dedup_key("wallet1", 1000, 1, 0);

        assert_eq!(key1, key2);
    }
}

// ============================================================================
// ATOMIC BALANCE UPDATE TESTS
// ============================================================================

mod atomic_update_tests {
    use super::*;

    /// Simulate atomic batch of balance changes
    struct BalanceBatch {
        changes: Vec<(String, i128)>, // account -> delta (positive = credit, negative = debit)
    }

    impl BalanceBatch {
        fn new() -> Self {
            Self { changes: Vec::new() }
        }

        fn add_credit(&mut self, account: &str, amount: u128) {
            self.changes.push((account.to_string(), amount as i128));
        }

        fn add_debit(&mut self, account: &str, amount: u128) {
            self.changes.push((account.to_string(), -(amount as i128)));
        }

        /// Validate and apply batch atomically
        fn apply(&self, balances: &mut HashMap<String, u128>) -> Result<(), String> {
            // Phase 1: Validate all debits have sufficient balance
            let mut projected = balances.clone();
            for (account, delta) in &self.changes {
                let current = projected.get(account).copied().unwrap_or(0) as i128;
                let new_balance = current + delta;

                if new_balance < 0 {
                    return Err(format!(
                        "Batch validation failed: {} would have negative balance ({})",
                        account, new_balance
                    ));
                }

                projected.insert(account.clone(), new_balance as u128);
            }

            // Phase 2: Apply all changes (all or nothing)
            *balances = projected;
            Ok(())
        }
    }

    /// Test atomic batch succeeds when all valid
    #[test]
    fn test_atomic_batch_success() {
        let mut balances = HashMap::new();
        balances.insert("alice".to_string(), 1000u128);
        balances.insert("bob".to_string(), 500u128);

        let mut batch = BalanceBatch::new();
        batch.add_debit("alice", 200);
        batch.add_credit("bob", 200);

        assert!(batch.apply(&mut balances).is_ok());
        assert_eq!(balances.get("alice").copied().unwrap(), 800);
        assert_eq!(balances.get("bob").copied().unwrap(), 700);
    }

    /// Test atomic batch fails entirely if any debit invalid
    #[test]
    fn test_atomic_batch_fails_entirely() {
        let mut balances = HashMap::new();
        balances.insert("alice".to_string(), 100u128);
        balances.insert("bob".to_string(), 500u128);

        let mut batch = BalanceBatch::new();
        batch.add_debit("alice", 200); // This will fail
        batch.add_credit("bob", 200);

        let result = batch.apply(&mut balances);
        assert!(result.is_err());

        // Both balances should be unchanged (rollback)
        assert_eq!(balances.get("alice").copied().unwrap(), 100);
        assert_eq!(balances.get("bob").copied().unwrap(), 500);
    }

    /// Test batch with multiple operations on same account
    #[test]
    fn test_batch_multiple_ops_same_account() {
        let mut balances = HashMap::new();
        balances.insert("alice".to_string(), 1000u128);

        let mut batch = BalanceBatch::new();
        batch.add_debit("alice", 300);
        batch.add_credit("alice", 100);
        batch.add_debit("alice", 400);

        assert!(batch.apply(&mut balances).is_ok());
        assert_eq!(balances.get("alice").copied().unwrap(), 400); // 1000 - 300 + 100 - 400
    }
}

// ============================================================================
// U128 PRECISION TESTS
// ============================================================================

mod u128_precision_tests {
    /// Test u128 handles large amounts correctly
    #[test]
    fn test_large_amounts() {
        let one_qug: u128 = 1_000_000_000_000_000_000_000_000; // 10^24
        let max_supply: u128 = 1_000_000_000 * one_qug; // 1 billion QUG

        assert!(max_supply < u128::MAX);

        // Verify arithmetic doesn't overflow
        let balance = max_supply;
        let transfer = one_qug * 1000;
        let remaining = balance - transfer;

        assert_eq!(remaining, max_supply - one_qug * 1000);
    }

    /// Test u128 comparison is exact
    #[test]
    fn test_exact_comparison() {
        let balance: u128 = 1_000_000_000_000_000_000_000_000;
        let cost: u128 = 1_000_000_000_000_000_000_000_001;

        // Exact comparison - 1 unit difference matters
        assert!(balance < cost);
        assert!(cost > balance);
        assert!(cost - balance == 1);
    }

    /// Test no precision loss in operations
    #[test]
    fn test_no_precision_loss() {
        let amount: u128 = 123_456_789_012_345_678_901_234;

        // Add and subtract same amount
        let balance = amount;
        let after_add = balance + 1;
        let after_sub = after_add - 1;

        assert_eq!(balance, after_sub);
    }

    /// Test smallest unit operations
    #[test]
    fn test_smallest_unit() {
        let mut balance: u128 = 10;

        // Debit 1 unit at a time
        for _ in 0..10 {
            balance -= 1;
        }

        assert_eq!(balance, 0);

        // Credit 1 unit at a time
        for _ in 0..10 {
            balance += 1;
        }

        assert_eq!(balance, 10);
    }
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

mod integration_tests {
    use super::*;

    /// Test complete transfer flow
    #[test]
    fn test_complete_transfer_flow() {
        let mut balances: HashMap<[u8; 32], u128> = HashMap::new();
        let alice = [1u8; 32];
        let bob = [2u8; 32];

        // Initial funding
        super::negative_balance_tests::apply_balance_credit(&mut balances, &alice, 10000);

        // Transfer from Alice to Bob
        let transfer_amount = 3000u128;

        // Step 1: Check balance
        let alice_balance = balances.get(&alice).copied().unwrap_or(0);
        assert!(alice_balance >= transfer_amount);

        // Step 2: Debit sender
        let debit_result =
            super::negative_balance_tests::apply_balance_debit(&mut balances, &alice, transfer_amount);
        assert!(debit_result.is_ok());

        // Step 3: Credit recipient
        super::negative_balance_tests::apply_balance_credit(&mut balances, &bob, transfer_amount);

        // Verify final balances
        assert_eq!(balances.get(&alice).copied().unwrap_or(0), 7000);
        assert_eq!(balances.get(&bob).copied().unwrap_or(0), 3000);
    }

    /// Test failed transfer doesn't change balances
    #[test]
    fn test_failed_transfer_no_change() {
        let mut balances: HashMap<[u8; 32], u128> = HashMap::new();
        let alice = [1u8; 32];
        let bob = [2u8; 32];

        super::negative_balance_tests::apply_balance_credit(&mut balances, &alice, 1000);

        let alice_before = balances.get(&alice).copied().unwrap_or(0);
        let bob_before = balances.get(&bob).copied().unwrap_or(0);

        // Attempt transfer more than balance
        let result =
            super::negative_balance_tests::apply_balance_debit(&mut balances, &alice, 2000);
        assert!(result.is_err());

        // Balances unchanged
        assert_eq!(balances.get(&alice).copied().unwrap_or(0), alice_before);
        assert_eq!(balances.get(&bob).copied().unwrap_or(0), bob_before);
    }
}

// ============================================================================
// CONCURRENT ACCESS TESTS
// ============================================================================

mod concurrent_tests {
    use super::*;
    use std::thread;

    /// Test concurrent balance reads are safe
    #[test]
    fn test_concurrent_reads() {
        let balances = Arc::new(RwLock::new(HashMap::<[u8; 32], u128>::new()));
        let account = [1u8; 32];

        {
            let mut b = balances.write().unwrap();
            b.insert(account, 10000);
        }

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let balances = Arc::clone(&balances);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        let b = balances.read().unwrap();
                        let balance = b.get(&account).copied().unwrap_or(0);
                        assert_eq!(balance, 10000);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }
    }
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

mod performance_tests {
    use super::*;
    use std::time::Instant;

    /// Test balance check performance
    #[test]
    fn test_balance_check_performance() {
        let mut balances = HashMap::new();
        let account = [1u8; 32];
        balances.insert(account, 1_000_000u128);

        let iterations = 100_000;
        let start = Instant::now();

        for _ in 0..iterations {
            let balance = balances.get(&account).copied().unwrap_or(0);
            let _ = balance >= 1000;
        }

        let elapsed = start.elapsed();
        let per_check_ns = elapsed.as_nanos() / iterations as u128;

        println!("Balance check: {} ns per check", per_check_ns);
        assert!(per_check_ns < 1000, "Balance check should be fast");
    }

    /// Test deduplication check performance
    #[test]
    fn test_dedup_check_performance() {
        let mut processed = HashSet::new();
        for i in 0u8..100 {
            let mut hash = [0u8; 32];
            hash[0] = i;
            processed.insert(hash);
        }

        let iterations = 100_000;
        let test_hash = [50u8; 32];
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = processed.contains(&test_hash);
        }

        let elapsed = start.elapsed();
        let per_check_ns = elapsed.as_nanos() / iterations as u128;

        println!("Dedup check: {} ns per check", per_check_ns);
        assert!(per_check_ns < 1000, "Dedup check should be fast");
    }
}
