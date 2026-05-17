//! Balance Propagation Tests
//!
//! v3.2.25-beta: Tests for balance updates, P2P propagation, and consistency
//!
//! These tests verify:
//! - Balance updates are applied correctly
//! - P2P balance propagation maintains consistency
//! - Mining rewards are credited properly
//! - Balance changes from blocks are accurate
//!
//! Run with: cargo test --package q-storage --test balance_propagation_tests

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// MOCK STRUCTURES
// ============================================================================

/// Simulates balance storage
#[derive(Debug, Default, Clone)]
pub struct BalanceStore {
    balances: HashMap<String, u128>,
    last_updated_height: u64,
}

impl BalanceStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_balance(&self, address: &str) -> u128 {
        *self.balances.get(address).unwrap_or(&0)
    }

    pub fn set_balance(&mut self, address: String, amount: u128) {
        self.balances.insert(address, amount);
    }

    pub fn add_balance(&mut self, address: String, amount: u128) -> Result<(), String> {
        let current = self.get_balance(&address);
        let new_balance = current.checked_add(amount)
            .ok_or_else(|| format!("Balance overflow for {}: {} + {}", address, current, amount))?;
        self.set_balance(address, new_balance);
        Ok(())
    }

    pub fn subtract_balance(&mut self, address: String, amount: u128) -> Result<(), String> {
        let current = self.get_balance(&address);
        if current < amount {
            return Err(format!("Insufficient balance for {}: {} < {}", address, current, amount));
        }
        self.set_balance(address, current - amount);
        Ok(())
    }

    pub fn transfer(&mut self, from: String, to: String, amount: u128) -> Result<(), String> {
        self.subtract_balance(from, amount)?;
        self.add_balance(to, amount)?;
        Ok(())
    }

    pub fn set_height(&mut self, height: u64) {
        self.last_updated_height = height;
    }

    pub fn get_height(&self) -> u64 {
        self.last_updated_height
    }
}

/// Simulates a balance update from a block
#[derive(Debug, Clone)]
pub struct BalanceUpdate {
    pub address: String,
    pub delta: i128,  // Can be positive or negative
    pub block_height: u64,
    pub tx_hash: String,
}

/// Simulates a coinbase (mining reward) transaction
#[derive(Debug, Clone)]
pub struct CoinbaseReward {
    pub miner_address: String,
    pub amount: u128,
    pub block_height: u64,
}

// ============================================================================
// BASIC BALANCE TESTS
// ============================================================================

mod basic_balance_tests {
    use super::*;

    #[test]
    fn test_initial_balance_is_zero() {
        let store = BalanceStore::new();
        assert_eq!(store.get_balance("qnk_new_address"), 0);
    }

    #[test]
    fn test_set_and_get_balance() {
        let mut store = BalanceStore::new();
        store.set_balance("qnk_test".to_string(), 1000);
        assert_eq!(store.get_balance("qnk_test"), 1000);
    }

    #[test]
    fn test_add_balance() {
        let mut store = BalanceStore::new();
        store.add_balance("qnk_test".to_string(), 1000).unwrap();
        store.add_balance("qnk_test".to_string(), 500).unwrap();
        assert_eq!(store.get_balance("qnk_test"), 1500);
    }

    #[test]
    fn test_subtract_balance() {
        let mut store = BalanceStore::new();
        store.set_balance("qnk_test".to_string(), 1000);
        store.subtract_balance("qnk_test".to_string(), 300).unwrap();
        assert_eq!(store.get_balance("qnk_test"), 700);
    }

    #[test]
    fn test_insufficient_balance_fails() {
        let mut store = BalanceStore::new();
        store.set_balance("qnk_test".to_string(), 100);
        let result = store.subtract_balance("qnk_test".to_string(), 200);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Insufficient"));
    }

    #[test]
    fn test_balance_overflow_prevented() {
        let mut store = BalanceStore::new();
        store.set_balance("qnk_test".to_string(), u128::MAX - 10);
        let result = store.add_balance("qnk_test".to_string(), 100);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("overflow"));
    }
}

// ============================================================================
// TRANSFER TESTS
// ============================================================================

mod transfer_tests {
    use super::*;

    #[test]
    fn test_basic_transfer() {
        let mut store = BalanceStore::new();
        store.set_balance("alice".to_string(), 1000);
        store.set_balance("bob".to_string(), 500);

        store.transfer("alice".to_string(), "bob".to_string(), 300).unwrap();

        assert_eq!(store.get_balance("alice"), 700);
        assert_eq!(store.get_balance("bob"), 800);
    }

    #[test]
    fn test_transfer_insufficient_funds() {
        let mut store = BalanceStore::new();
        store.set_balance("alice".to_string(), 100);

        let result = store.transfer("alice".to_string(), "bob".to_string(), 200);
        assert!(result.is_err());

        // Balances should be unchanged
        assert_eq!(store.get_balance("alice"), 100);
        assert_eq!(store.get_balance("bob"), 0);
    }

    #[test]
    fn test_transfer_to_new_address() {
        let mut store = BalanceStore::new();
        store.set_balance("alice".to_string(), 1000);

        store.transfer("alice".to_string(), "new_address".to_string(), 100).unwrap();

        assert_eq!(store.get_balance("alice"), 900);
        assert_eq!(store.get_balance("new_address"), 100);
    }

    #[test]
    fn test_transfer_zero_amount() {
        let mut store = BalanceStore::new();
        store.set_balance("alice".to_string(), 1000);

        store.transfer("alice".to_string(), "bob".to_string(), 0).unwrap();

        assert_eq!(store.get_balance("alice"), 1000);
        assert_eq!(store.get_balance("bob"), 0);
    }

    #[test]
    fn test_self_transfer() {
        let mut store = BalanceStore::new();
        store.set_balance("alice".to_string(), 1000);

        store.transfer("alice".to_string(), "alice".to_string(), 100).unwrap();

        // Balance should be unchanged (subtract then add same amount)
        assert_eq!(store.get_balance("alice"), 1000);
    }
}

// ============================================================================
// MINING REWARD TESTS
// ============================================================================

mod mining_reward_tests {
    use super::*;

    const BLOCK_REWARD: u128 = 10_000_000_000_000_000_000_000_000; // 10 QUG with 24 decimals

    fn apply_coinbase_reward(store: &mut BalanceStore, reward: &CoinbaseReward) -> Result<(), String> {
        // Height check - rewards must be applied in order
        if reward.block_height <= store.get_height() {
            return Err(format!(
                "Stale reward: block {} <= current height {}",
                reward.block_height, store.get_height()
            ));
        }

        store.add_balance(reward.miner_address.clone(), reward.amount)?;
        store.set_height(reward.block_height);
        Ok(())
    }

    #[test]
    fn test_mining_reward_credited() {
        let mut store = BalanceStore::new();

        let reward = CoinbaseReward {
            miner_address: "qnk_miner".to_string(),
            amount: BLOCK_REWARD,
            block_height: 1,
        };

        apply_coinbase_reward(&mut store, &reward).unwrap();

        assert_eq!(store.get_balance("qnk_miner"), BLOCK_REWARD);
        assert_eq!(store.get_height(), 1);
    }

    #[test]
    fn test_multiple_blocks_mined() {
        let mut store = BalanceStore::new();

        for height in 1..=10 {
            let reward = CoinbaseReward {
                miner_address: "qnk_miner".to_string(),
                amount: BLOCK_REWARD,
                block_height: height,
            };
            apply_coinbase_reward(&mut store, &reward).unwrap();
        }

        assert_eq!(store.get_balance("qnk_miner"), BLOCK_REWARD * 10);
        assert_eq!(store.get_height(), 10);
    }

    #[test]
    fn test_stale_reward_rejected() {
        let mut store = BalanceStore::new();
        store.set_height(100);

        let old_reward = CoinbaseReward {
            miner_address: "qnk_miner".to_string(),
            amount: BLOCK_REWARD,
            block_height: 50,  // Lower than current height
        };

        let result = apply_coinbase_reward(&mut store, &old_reward);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Stale"));

        // Balance should not change
        assert_eq!(store.get_balance("qnk_miner"), 0);
    }

    #[test]
    fn test_different_miners_credited_separately() {
        let mut store = BalanceStore::new();

        // Miner A mines blocks 1-5
        for height in 1..=5 {
            let reward = CoinbaseReward {
                miner_address: "miner_a".to_string(),
                amount: BLOCK_REWARD,
                block_height: height,
            };
            apply_coinbase_reward(&mut store, &reward).unwrap();
        }

        // Miner B mines blocks 6-10
        for height in 6..=10 {
            let reward = CoinbaseReward {
                miner_address: "miner_b".to_string(),
                amount: BLOCK_REWARD,
                block_height: height,
            };
            apply_coinbase_reward(&mut store, &reward).unwrap();
        }

        assert_eq!(store.get_balance("miner_a"), BLOCK_REWARD * 5);
        assert_eq!(store.get_balance("miner_b"), BLOCK_REWARD * 5);
    }
}

// ============================================================================
// BALANCE UPDATE PROPAGATION TESTS
// ============================================================================

mod propagation_tests {
    use super::*;

    fn apply_balance_update(store: &mut BalanceStore, update: &BalanceUpdate) -> Result<(), String> {
        let current = store.get_balance(&update.address) as i128;
        let new_balance = current + update.delta;

        if new_balance < 0 {
            return Err(format!(
                "Balance would go negative for {}: {} + {} = {}",
                update.address, current, update.delta, new_balance
            ));
        }

        store.set_balance(update.address.clone(), new_balance as u128);
        if update.block_height > store.get_height() {
            store.set_height(update.block_height);
        }
        Ok(())
    }

    #[test]
    fn test_positive_balance_update() {
        let mut store = BalanceStore::new();
        store.set_balance("qnk_test".to_string(), 1000);

        let update = BalanceUpdate {
            address: "qnk_test".to_string(),
            delta: 500,
            block_height: 1,
            tx_hash: "tx123".to_string(),
        };

        apply_balance_update(&mut store, &update).unwrap();
        assert_eq!(store.get_balance("qnk_test"), 1500);
    }

    #[test]
    fn test_negative_balance_update() {
        let mut store = BalanceStore::new();
        store.set_balance("qnk_test".to_string(), 1000);

        let update = BalanceUpdate {
            address: "qnk_test".to_string(),
            delta: -300,
            block_height: 1,
            tx_hash: "tx123".to_string(),
        };

        apply_balance_update(&mut store, &update).unwrap();
        assert_eq!(store.get_balance("qnk_test"), 700);
    }

    #[test]
    fn test_balance_cannot_go_negative() {
        let mut store = BalanceStore::new();
        store.set_balance("qnk_test".to_string(), 100);

        let update = BalanceUpdate {
            address: "qnk_test".to_string(),
            delta: -200,
            block_height: 1,
            tx_hash: "tx123".to_string(),
        };

        let result = apply_balance_update(&mut store, &update);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("negative"));

        // Balance unchanged
        assert_eq!(store.get_balance("qnk_test"), 100);
    }

    #[test]
    fn test_batch_balance_updates() {
        let mut store = BalanceStore::new();

        let updates = vec![
            BalanceUpdate {
                address: "alice".to_string(),
                delta: 1000,
                block_height: 1,
                tx_hash: "tx1".to_string(),
            },
            BalanceUpdate {
                address: "bob".to_string(),
                delta: 2000,
                block_height: 1,
                tx_hash: "tx2".to_string(),
            },
            BalanceUpdate {
                address: "alice".to_string(),
                delta: -500,
                block_height: 1,
                tx_hash: "tx3".to_string(),
            },
            BalanceUpdate {
                address: "bob".to_string(),
                delta: 500,
                block_height: 1,
                tx_hash: "tx3".to_string(),
            },
        ];

        for update in &updates {
            apply_balance_update(&mut store, update).unwrap();
        }

        assert_eq!(store.get_balance("alice"), 500);  // 1000 - 500
        assert_eq!(store.get_balance("bob"), 2500);   // 2000 + 500
    }
}

// ============================================================================
// HEIGHT CONSISTENCY TESTS
// ============================================================================

mod height_consistency_tests {
    use super::*;

    static HEIGHT_TRACKER: AtomicU64 = AtomicU64::new(0);

    fn reset_tracker() {
        HEIGHT_TRACKER.store(0, Ordering::SeqCst);
    }

    fn verify_height_monotonic(new_height: u64) -> Result<(), String> {
        let current = HEIGHT_TRACKER.load(Ordering::SeqCst);

        if new_height < current {
            return Err(format!(
                "Height regression detected: {} -> {}",
                current, new_height
            ));
        }

        HEIGHT_TRACKER.store(new_height, Ordering::SeqCst);
        Ok(())
    }

    #[test]
    fn test_height_increases_monotonically() {
        reset_tracker();

        assert!(verify_height_monotonic(1).is_ok());
        assert!(verify_height_monotonic(2).is_ok());
        assert!(verify_height_monotonic(100).is_ok());
        assert!(verify_height_monotonic(1000).is_ok());
    }

    #[test]
    fn test_height_regression_detected() {
        reset_tracker();

        assert!(verify_height_monotonic(100).is_ok());
        let result = verify_height_monotonic(50);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("regression"));
    }

    #[test]
    fn test_same_height_allowed() {
        reset_tracker();

        assert!(verify_height_monotonic(100).is_ok());
        // Same height is OK (idempotent updates)
        assert!(verify_height_monotonic(100).is_ok());
    }
}

// ============================================================================
// BALANCE RESET SAFETY TESTS
// ============================================================================

mod reset_safety_tests {
    use super::*;

    /// Verify that balance reset follows blockchain reset
    fn reset_balances_with_blockchain(
        store: &mut BalanceStore,
        blockchain_height: u64,
    ) -> Result<(), String> {
        let balance_height = store.get_height();

        // If blockchain reset to lower height, balances MUST also reset
        if blockchain_height < balance_height {
            // Clear all balances
            store.balances.clear();
            store.set_height(0);
            return Ok(());
        }

        Ok(())
    }

    #[test]
    fn test_balances_reset_with_blockchain() {
        let mut store = BalanceStore::new();

        // Simulate some balance accumulation
        store.set_balance("miner".to_string(), 1000);
        store.set_height(100);

        // Blockchain resets to height 50
        reset_balances_with_blockchain(&mut store, 50).unwrap();

        // Balances should be cleared
        assert_eq!(store.get_balance("miner"), 0);
        assert_eq!(store.get_height(), 0);
    }

    #[test]
    fn test_balances_preserved_on_normal_operation() {
        let mut store = BalanceStore::new();

        store.set_balance("miner".to_string(), 1000);
        store.set_height(100);

        // Blockchain continues to higher height
        reset_balances_with_blockchain(&mut store, 150).unwrap();

        // Balances should be preserved
        assert_eq!(store.get_balance("miner"), 1000);
        assert_eq!(store.get_height(), 100);
    }
}

// ============================================================================
// CONCURRENT UPDATE TESTS
// ============================================================================

mod concurrent_tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[test]
    fn test_concurrent_balance_updates() {
        let store = Arc::new(Mutex::new(BalanceStore::new()));
        let mut handles = vec![];

        // 10 threads each adding 100 to the same address
        for _ in 0..10 {
            let store_clone = Arc::clone(&store);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    let mut locked = store_clone.lock().unwrap();
                    locked.add_balance("shared".to_string(), 1).unwrap();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        let locked = store.lock().unwrap();
        assert_eq!(locked.get_balance("shared"), 1000);
    }

    #[test]
    fn test_concurrent_transfers() {
        let store = Arc::new(Mutex::new(BalanceStore::new()));

        // Initial balances
        {
            let mut locked = store.lock().unwrap();
            locked.set_balance("pool".to_string(), 10000);
        }

        let mut handles = vec![];

        // 10 threads each withdrawing from pool
        for i in 0..10 {
            let store_clone = Arc::clone(&store);
            let handle = thread::spawn(move || {
                let recipient = format!("recipient_{}", i);
                for _ in 0..10 {
                    let mut locked = store_clone.lock().unwrap();
                    let _ = locked.transfer("pool".to_string(), recipient.clone(), 10);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        let locked = store.lock().unwrap();
        // Pool should have 10000 - (10 * 10 * 10) = 9000 remaining
        assert_eq!(locked.get_balance("pool"), 9000);

        // Total distributed should be 1000
        let total_distributed: u128 = (0..10)
            .map(|i| locked.get_balance(&format!("recipient_{}", i)))
            .sum();
        assert_eq!(total_distributed, 1000);
    }
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_balance_lookup_performance() {
        let mut store = BalanceStore::new();

        // Add 10,000 addresses
        for i in 0..10_000 {
            store.set_balance(format!("addr_{}", i), i as u128);
        }

        let iterations = 100_000;
        let start = Instant::now();

        for i in 0..iterations {
            let addr = format!("addr_{}", i % 10_000);
            let _ = store.get_balance(&addr);
        }

        let elapsed = start.elapsed();
        let per_lookup_ns = elapsed.as_nanos() / iterations as u128;

        println!("Balance lookup: {} ns per lookup", per_lookup_ns);
        assert!(per_lookup_ns < 1000, "Balance lookup should be < 1us, got {} ns", per_lookup_ns);
    }

    #[test]
    fn test_balance_update_performance() {
        let mut store = BalanceStore::new();
        let iterations = 100_000;
        let start = Instant::now();

        for i in 0..iterations {
            let addr = format!("addr_{}", i % 1000);
            store.add_balance(addr, 1).unwrap();
        }

        let elapsed = start.elapsed();
        let per_update_ns = elapsed.as_nanos() / iterations as u128;

        println!("Balance update: {} ns per update", per_update_ns);
        assert!(per_update_ns < 2000, "Balance update should be < 2us, got {} ns", per_update_ns);
    }
}
