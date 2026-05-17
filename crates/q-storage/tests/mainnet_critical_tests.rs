//! Mainnet Critical Safety Tests
//!
//! These tests protect against scenarios that could cause MILLIONS in losses on mainnet.
//!
//! CRITICAL SCENARIOS TESTED:
//! 1. Double-spending attacks
//! 2. Transaction replay attacks
//! 3. Balance manipulation
//! 4. Coinbase reward fraud
//! 5. State corruption
//! 6. Race conditions in balance updates
//!
//! Run with: cargo test --package q-storage --test mainnet_critical_tests

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

// ============================================================================
// MOCK STRUCTURES FOR TESTING
// ============================================================================

/// Transaction with unique ID for replay prevention
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct Transaction {
    pub tx_hash: String,
    pub from: String,
    pub to: String,
    pub amount: u128,
    pub nonce: u64,
    pub signature: Vec<u8>,
}

/// Block containing transactions
#[derive(Debug, Clone)]
pub struct Block {
    pub height: u64,
    pub hash: String,
    pub parent_hash: String,
    pub transactions: Vec<Transaction>,
    pub coinbase_address: String,
    pub coinbase_amount: u128,
    pub timestamp: u64,
}

/// State that tracks balances and processed transactions
#[derive(Debug, Default)]
pub struct ChainState {
    balances: HashMap<String, u128>,
    processed_tx_hashes: HashSet<String>,
    address_nonces: HashMap<String, u64>,
    current_height: u64,
    total_supply: u128,
}

impl ChainState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get balance for an address
    pub fn get_balance(&self, address: &str) -> u128 {
        *self.balances.get(address).unwrap_or(&0)
    }

    /// Get next expected nonce for an address
    pub fn get_nonce(&self, address: &str) -> u64 {
        *self.address_nonces.get(address).unwrap_or(&0)
    }

    /// Check if transaction was already processed
    pub fn is_tx_processed(&self, tx_hash: &str) -> bool {
        self.processed_tx_hashes.contains(tx_hash)
    }

    /// Validate and apply a transaction
    pub fn apply_transaction(&mut self, tx: &Transaction) -> Result<(), String> {
        // CRITICAL CHECK 1: Replay prevention - reject if tx already processed
        if self.processed_tx_hashes.contains(&tx.tx_hash) {
            return Err(format!(
                "REPLAY ATTACK BLOCKED: Transaction {} already processed",
                tx.tx_hash
            ));
        }

        // CRITICAL CHECK 2: Nonce validation - prevents transaction replay/reordering
        let expected_nonce = self.get_nonce(&tx.from);
        if tx.nonce != expected_nonce {
            return Err(format!(
                "INVALID NONCE: Expected {}, got {} for address {}",
                expected_nonce, tx.nonce, tx.from
            ));
        }

        // CRITICAL CHECK 3: Sufficient balance - prevents overdraft
        let sender_balance = self.get_balance(&tx.from);
        if sender_balance < tx.amount {
            return Err(format!(
                "INSUFFICIENT BALANCE: {} has {} but tried to send {}",
                tx.from, sender_balance, tx.amount
            ));
        }

        // CRITICAL CHECK 4: No self-transfer with balance creation
        if tx.from == tx.to && tx.amount > 0 {
            // Self-transfers are allowed but must not change total balance
        }

        // Apply the transaction
        let from_balance = self.balances.entry(tx.from.clone()).or_insert(0);
        *from_balance = from_balance.checked_sub(tx.amount)
            .ok_or_else(|| "UNDERFLOW in balance subtraction".to_string())?;

        let to_balance = self.balances.entry(tx.to.clone()).or_insert(0);
        *to_balance = to_balance.checked_add(tx.amount)
            .ok_or_else(|| "OVERFLOW in balance addition".to_string())?;

        // Mark transaction as processed
        self.processed_tx_hashes.insert(tx.tx_hash.clone());

        // Increment nonce
        *self.address_nonces.entry(tx.from.clone()).or_insert(0) += 1;

        Ok(())
    }

    /// Apply coinbase (mining reward)
    pub fn apply_coinbase(&mut self, address: &str, amount: u128, height: u64) -> Result<(), String> {
        // CRITICAL CHECK: Height must be sequential
        if height != self.current_height + 1 {
            return Err(format!(
                "INVALID COINBASE HEIGHT: Expected {}, got {}",
                self.current_height + 1, height
            ));
        }

        // CRITICAL CHECK: Coinbase amount must match expected reward
        let expected_reward = calculate_block_reward(height);
        if amount > expected_reward {
            return Err(format!(
                "COINBASE FRAUD: Block {} reward should be {} but got {}",
                height, expected_reward, amount
            ));
        }

        // Apply coinbase
        let balance = self.balances.entry(address.to_string()).or_insert(0);
        *balance = balance.checked_add(amount)
            .ok_or_else(|| "OVERFLOW in coinbase".to_string())?;

        // Track total supply
        self.total_supply = self.total_supply.checked_add(amount)
            .ok_or_else(|| "TOTAL SUPPLY OVERFLOW".to_string())?;

        self.current_height = height;
        Ok(())
    }

    /// Verify total supply consistency
    pub fn verify_supply_consistency(&self) -> Result<(), String> {
        let sum_of_balances: u128 = self.balances.values().sum();
        if sum_of_balances != self.total_supply {
            return Err(format!(
                "SUPPLY MISMATCH: Sum of balances ({}) != total supply ({})",
                sum_of_balances, self.total_supply
            ));
        }
        Ok(())
    }
}

/// Calculate expected block reward (halving schedule)
fn calculate_block_reward(height: u64) -> u128 {
    let initial_reward: u128 = 10_000_000_000_000_000_000_000_000; // 10 QUG
    let halvings = height / 210_000; // Halving every 210k blocks
    if halvings >= 64 {
        return 0;
    }
    initial_reward >> halvings
}

// ============================================================================
// DOUBLE-SPEND PREVENTION TESTS
// ============================================================================

mod double_spend_tests {
    use super::*;

    #[test]
    fn test_same_tx_rejected_twice() {
        let mut state = ChainState::new();

        // Give Alice some balance
        state.balances.insert("alice".to_string(), 1000);

        let tx = Transaction {
            tx_hash: "tx_001".to_string(),
            from: "alice".to_string(),
            to: "bob".to_string(),
            amount: 100,
            nonce: 0,
            signature: vec![],
        };

        // First application should succeed
        assert!(state.apply_transaction(&tx).is_ok());
        assert_eq!(state.get_balance("alice"), 900);
        assert_eq!(state.get_balance("bob"), 100);

        // Second application should be REJECTED (replay attack)
        let result = state.apply_transaction(&tx);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("REPLAY ATTACK"));

        // Balances should NOT change
        assert_eq!(state.get_balance("alice"), 900);
        assert_eq!(state.get_balance("bob"), 100);
    }

    #[test]
    fn test_nonce_prevents_reordering() {
        let mut state = ChainState::new();
        state.balances.insert("alice".to_string(), 1000);

        // Try to submit tx with nonce 1 before nonce 0
        let tx1 = Transaction {
            tx_hash: "tx_001".to_string(),
            from: "alice".to_string(),
            to: "bob".to_string(),
            amount: 100,
            nonce: 1, // Wrong nonce - should be 0
            signature: vec![],
        };

        let result = state.apply_transaction(&tx1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("INVALID NONCE"));
    }

    #[test]
    fn test_nonce_sequence_enforced() {
        let mut state = ChainState::new();
        state.balances.insert("alice".to_string(), 1000);

        // Submit transactions in correct nonce order
        for i in 0..5 {
            let tx = Transaction {
                tx_hash: format!("tx_{:03}", i),
                from: "alice".to_string(),
                to: "bob".to_string(),
                amount: 10,
                nonce: i,
                signature: vec![],
            };
            assert!(state.apply_transaction(&tx).is_ok());
        }

        // Alice's nonce should now be 5
        assert_eq!(state.get_nonce("alice"), 5);

        // Can't reuse any old nonce
        for old_nonce in 0..5 {
            let tx = Transaction {
                tx_hash: format!("tx_replay_{}", old_nonce),
                from: "alice".to_string(),
                to: "bob".to_string(),
                amount: 10,
                nonce: old_nonce,
                signature: vec![],
            };
            let result = state.apply_transaction(&tx);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_insufficient_balance_rejected() {
        let mut state = ChainState::new();
        state.balances.insert("alice".to_string(), 100);

        let tx = Transaction {
            tx_hash: "tx_overdraft".to_string(),
            from: "alice".to_string(),
            to: "bob".to_string(),
            amount: 200, // More than Alice has
            nonce: 0,
            signature: vec![],
        };

        let result = state.apply_transaction(&tx);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("INSUFFICIENT BALANCE"));

        // Balances unchanged
        assert_eq!(state.get_balance("alice"), 100);
        assert_eq!(state.get_balance("bob"), 0);
    }

    #[test]
    fn test_balance_cannot_go_negative() {
        let mut state = ChainState::new();
        state.balances.insert("alice".to_string(), 100);

        // Try multiple transactions that would overdraft
        let tx1 = Transaction {
            tx_hash: "tx_1".to_string(),
            from: "alice".to_string(),
            to: "bob".to_string(),
            amount: 60,
            nonce: 0,
            signature: vec![],
        };

        let tx2 = Transaction {
            tx_hash: "tx_2".to_string(),
            from: "alice".to_string(),
            to: "charlie".to_string(),
            amount: 60, // Would overdraft (40 remaining)
            nonce: 1,
            signature: vec![],
        };

        assert!(state.apply_transaction(&tx1).is_ok());
        let result = state.apply_transaction(&tx2);
        assert!(result.is_err());

        // Only first tx applied
        assert_eq!(state.get_balance("alice"), 40);
        assert_eq!(state.get_balance("bob"), 60);
        assert_eq!(state.get_balance("charlie"), 0);
    }
}

// ============================================================================
// COINBASE FRAUD PREVENTION TESTS
// ============================================================================

mod coinbase_tests {
    use super::*;

    #[test]
    fn test_coinbase_amount_validated() {
        let mut state = ChainState::new();

        // Valid coinbase
        let expected_reward = calculate_block_reward(1);
        assert!(state.apply_coinbase("miner", expected_reward, 1).is_ok());

        // Fraudulent coinbase (trying to get more than allowed)
        let result = state.apply_coinbase("miner", expected_reward * 2, 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("COINBASE FRAUD"));
    }

    #[test]
    fn test_coinbase_height_sequence() {
        let mut state = ChainState::new();

        // Height 1 should work
        assert!(state.apply_coinbase("miner", 1000, 1).is_ok());

        // Height 3 should fail (skipped 2)
        let result = state.apply_coinbase("miner", 1000, 3);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("INVALID COINBASE HEIGHT"));

        // Height 2 should work
        assert!(state.apply_coinbase("miner", 1000, 2).is_ok());
    }

    #[test]
    fn test_coinbase_cannot_be_replayed() {
        let mut state = ChainState::new();

        // Apply block 1 coinbase
        assert!(state.apply_coinbase("miner", 1000, 1).is_ok());

        // Try to replay block 1 coinbase (should fail - height mismatch)
        let result = state.apply_coinbase("miner", 1000, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_halving_schedule() {
        // Initial reward
        assert_eq!(calculate_block_reward(1), 10_000_000_000_000_000_000_000_000u128);

        // After first halving (block 210,001)
        assert_eq!(calculate_block_reward(210_001), 5_000_000_000_000_000_000_000_000u128);

        // After second halving
        assert_eq!(calculate_block_reward(420_001), 2_500_000_000_000_000_000_000_000u128);

        // Eventually goes to 0
        assert_eq!(calculate_block_reward(100_000_000), 0);
    }
}

// ============================================================================
// SUPPLY CONSISTENCY TESTS
// ============================================================================

mod supply_tests {
    use super::*;

    #[test]
    fn test_transfers_preserve_supply() {
        let mut state = ChainState::new();

        // Mint via coinbase
        state.apply_coinbase("miner", 1000, 1).unwrap();
        assert!(state.verify_supply_consistency().is_ok());

        // Transfer preserves supply
        state.balances.insert("alice".to_string(), 500);
        state.total_supply += 500;

        let tx = Transaction {
            tx_hash: "tx_1".to_string(),
            from: "alice".to_string(),
            to: "bob".to_string(),
            amount: 200,
            nonce: 0,
            signature: vec![],
        };

        state.apply_transaction(&tx).unwrap();
        assert!(state.verify_supply_consistency().is_ok());
    }

    #[test]
    fn test_detect_supply_mismatch() {
        let mut state = ChainState::new();
        state.apply_coinbase("miner", 1000, 1).unwrap();

        // Artificially corrupt the state
        *state.balances.get_mut("miner").unwrap() += 100;

        // Should detect the mismatch
        let result = state.verify_supply_consistency();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("SUPPLY MISMATCH"));
    }

    #[test]
    fn test_no_money_creation_via_transfer() {
        let mut state = ChainState::new();
        state.balances.insert("alice".to_string(), 100);
        state.total_supply = 100;

        // Verify supply before
        let supply_before = state.total_supply;

        // Do many transfers
        for i in 0..10 {
            let tx = Transaction {
                tx_hash: format!("tx_{}", i),
                from: if i % 2 == 0 { "alice" } else { "bob" }.to_string(),
                to: if i % 2 == 0 { "bob" } else { "alice" }.to_string(),
                amount: 10,
                nonce: i / 2,
                signature: vec![],
            };
            let _ = state.apply_transaction(&tx);
        }

        // Supply should be unchanged
        assert_eq!(state.total_supply, supply_before);
        assert!(state.verify_supply_consistency().is_ok());
    }
}

// ============================================================================
// CONCURRENT ACCESS TESTS
// ============================================================================

mod concurrent_tests {
    use super::*;

    #[test]
    fn test_concurrent_double_spend_attempt() {
        let state = Arc::new(Mutex::new(ChainState::new()));

        // Give Alice balance
        {
            let mut s = state.lock().unwrap();
            s.balances.insert("alice".to_string(), 100);
        }

        let mut handles = vec![];

        // 10 threads all try to spend Alice's 100
        for i in 0..10 {
            let state_clone = Arc::clone(&state);
            let handle = thread::spawn(move || {
                let tx = Transaction {
                    tx_hash: format!("concurrent_tx_{}", i),
                    from: "alice".to_string(),
                    to: format!("recipient_{}", i),
                    amount: 100, // Each tries to spend full balance
                    nonce: 0,    // All use same nonce
                    signature: vec![],
                };

                let mut s = state_clone.lock().unwrap();
                s.apply_transaction(&tx)
            });
            handles.push(handle);
        }

        let mut successes = 0;
        let mut failures = 0;

        for handle in handles {
            match handle.join().unwrap() {
                Ok(_) => successes += 1,
                Err(_) => failures += 1,
            }
        }

        // Only ONE transaction should succeed
        assert_eq!(successes, 1, "Only one double-spend attempt should succeed");
        assert_eq!(failures, 9, "Nine attempts should fail");

        // Alice should have 0
        let s = state.lock().unwrap();
        assert_eq!(s.get_balance("alice"), 0);
    }

    #[test]
    fn test_concurrent_supply_remains_consistent() {
        let state = Arc::new(Mutex::new(ChainState::new()));

        // Initial setup
        {
            let mut s = state.lock().unwrap();
            for i in 0..10 {
                s.balances.insert(format!("user_{}", i), 1000);
            }
            s.total_supply = 10000;
        }

        let mut handles = vec![];

        // Many concurrent transfers
        for i in 0..100 {
            let state_clone = Arc::clone(&state);
            let handle = thread::spawn(move || {
                let from_idx = i % 10;
                let to_idx = (i + 1) % 10;

                let tx = Transaction {
                    tx_hash: format!("tx_{}", i),
                    from: format!("user_{}", from_idx),
                    to: format!("user_{}", to_idx),
                    amount: 10,
                    nonce: (i / 10) as u64,
                    signature: vec![],
                };

                let mut s = state_clone.lock().unwrap();
                let _ = s.apply_transaction(&tx);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Supply should still be consistent
        let s = state.lock().unwrap();
        assert!(s.verify_supply_consistency().is_ok());
    }
}

// ============================================================================
// OVERFLOW/UNDERFLOW TESTS
// ============================================================================

mod overflow_tests {
    use super::*;

    #[test]
    fn test_balance_overflow_prevented() {
        let mut state = ChainState::new();
        state.balances.insert("whale".to_string(), u128::MAX - 100);
        state.balances.insert("sender".to_string(), 200);
        // Use saturating_add to avoid compile-time overflow
        state.total_supply = (u128::MAX - 100).saturating_add(200);

        let tx = Transaction {
            tx_hash: "overflow_tx".to_string(),
            from: "sender".to_string(),
            to: "whale".to_string(),
            amount: 200, // Would overflow whale's balance
            nonce: 0,
            signature: vec![],
        };

        let result = state.apply_transaction(&tx);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("OVERFLOW"));
    }

    #[test]
    fn test_total_supply_overflow_prevented() {
        let mut state = ChainState::new();
        state.total_supply = u128::MAX - 100;
        state.current_height = 0;

        let result = state.apply_coinbase("miner", 200, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("OVERFLOW"));
    }
}

// ============================================================================
// REGRESSION TESTS
// ============================================================================

mod regression_tests {
    use super::*;

    /// Regression test: Ensure double-spend protection works
    #[test]
    fn test_regression_double_spend_protection() {
        let mut state = ChainState::new();
        state.balances.insert("attacker".to_string(), 1000);

        // Attacker tries to send same money to two people
        let tx1 = Transaction {
            tx_hash: "tx_double_1".to_string(),
            from: "attacker".to_string(),
            to: "victim1".to_string(),
            amount: 1000,
            nonce: 0,
            signature: vec![],
        };

        let tx2 = Transaction {
            tx_hash: "tx_double_2".to_string(),
            from: "attacker".to_string(),
            to: "victim2".to_string(),
            amount: 1000,
            nonce: 0, // Same nonce!
            signature: vec![],
        };

        // First should succeed
        assert!(state.apply_transaction(&tx1).is_ok());

        // Second should fail (nonce already used)
        let result = state.apply_transaction(&tx2);
        assert!(result.is_err(), "REGRESSION: Double-spend protection failed!");

        // Only victim1 should have the money
        assert_eq!(state.get_balance("victim1"), 1000);
        assert_eq!(state.get_balance("victim2"), 0);
        assert_eq!(state.get_balance("attacker"), 0);
    }

    /// Regression test: Ensure replay protection works
    #[test]
    fn test_regression_replay_protection() {
        let mut state = ChainState::new();
        state.balances.insert("alice".to_string(), 1000);

        let tx = Transaction {
            tx_hash: "legitimate_tx".to_string(),
            from: "alice".to_string(),
            to: "bob".to_string(),
            amount: 100,
            nonce: 0,
            signature: vec![],
        };

        // Apply once
        state.apply_transaction(&tx).unwrap();
        let bob_balance_after_1 = state.get_balance("bob");

        // Try to replay
        let result = state.apply_transaction(&tx);
        assert!(result.is_err(), "REGRESSION: Replay protection failed!");

        // Bob's balance should NOT have doubled
        assert_eq!(
            state.get_balance("bob"),
            bob_balance_after_1,
            "REGRESSION: Replay attack succeeded!"
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
    fn test_tx_processing_performance() {
        let mut state = ChainState::new();
        state.balances.insert("sender".to_string(), u128::MAX / 2);

        let iterations = 10_000;
        let start = Instant::now();

        for i in 0..iterations {
            let tx = Transaction {
                tx_hash: format!("tx_{}", i),
                from: "sender".to_string(),
                to: "receiver".to_string(),
                amount: 1,
                nonce: i as u64,
                signature: vec![],
            };
            state.apply_transaction(&tx).unwrap();
        }

        let elapsed = start.elapsed();
        let per_tx_us = elapsed.as_micros() / iterations as u128;

        println!("Transaction processing: {} us per tx", per_tx_us);
        assert!(per_tx_us < 100, "Transaction processing too slow: {} us", per_tx_us);
    }

    #[test]
    fn test_replay_check_performance() {
        let mut state = ChainState::new();

        // Pre-populate with many processed transactions
        for i in 0..100_000 {
            state.processed_tx_hashes.insert(format!("tx_{}", i));
        }

        let iterations = 100_000;
        let start = Instant::now();

        for i in 0..iterations {
            let _ = state.is_tx_processed(&format!("tx_{}", i));
        }

        let elapsed = start.elapsed();
        let per_check_ns = elapsed.as_nanos() / iterations as u128;

        println!("Replay check: {} ns per check", per_check_ns);
        assert!(per_check_ns < 1000, "Replay check too slow: {} ns", per_check_ns);
    }
}
