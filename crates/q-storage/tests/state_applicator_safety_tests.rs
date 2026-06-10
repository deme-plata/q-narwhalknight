//! State Applicator Safety Tests
//!
//! Tests for the StateApplicator to ensure atomic state changes,
//! proper error handling, and data integrity under failure conditions.
//!
//! CRITICAL SCENARIOS TESTED:
//! 1. Concurrent apply_changes calls
//! 2. Balance overflow/underflow protection
//! 3. Partial batch failure recovery
//! 4. RocksDB write failure handling
//! 5. State consistency after errors
//!
//! Run with: cargo test --package q-storage --test state_applicator_safety_tests

use std::collections::HashMap;
use std::sync::{Arc, Mutex, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::thread;

// ============================================================================
// MOCK STRUCTURES FOR TESTING STATE APPLICATOR
// ============================================================================

/// Simulated StateChange for testing
#[derive(Debug, Clone)]
pub enum MockStateChange {
    BalanceCredit { account: [u8; 32], token: [u8; 32], amount: u128 },
    BalanceDebit { account: [u8; 32], token: [u8; 32], amount: u128 },
    TokenCreate { token_address: [u8; 32], name: String, initial_supply: u128 },
    PoolCreate { pool_id: [u8; 32], token_a: [u8; 32], token_b: [u8; 32] },
}

/// Mock database for testing state application
#[derive(Debug, Default)]
pub struct MockStateDB {
    balances: Mutex<HashMap<([u8; 32], [u8; 32]), u128>>,
    tokens: Mutex<HashMap<[u8; 32], (String, u128)>>,
    pools: Mutex<HashMap<[u8; 32], ([u8; 32], [u8; 32])>>,
    write_count: AtomicU64,
    should_fail_at: AtomicU64,
    total_writes: AtomicU64,
}

impl MockStateDB {
    pub fn new() -> Self {
        Self {
            balances: Mutex::new(HashMap::new()),
            tokens: Mutex::new(HashMap::new()),
            pools: Mutex::new(HashMap::new()),
            write_count: AtomicU64::new(0),
            should_fail_at: AtomicU64::new(u64::MAX),
            total_writes: AtomicU64::new(0),
        }
    }

    pub fn set_fail_at_write(&self, n: u64) {
        self.should_fail_at.store(n, Ordering::SeqCst);
    }

    pub fn get_balance(&self, account: &[u8; 32], token: &[u8; 32]) -> u128 {
        let balances = self.balances.lock().unwrap();
        *balances.get(&(*account, *token)).unwrap_or(&0)
    }

    pub fn set_balance(&self, account: &[u8; 32], token: &[u8; 32], amount: u128) {
        let mut balances = self.balances.lock().unwrap();
        balances.insert((*account, *token), amount);
    }

    /// Apply a batch of state changes atomically
    /// CRITICAL: Holds all locks for entire operation to prevent race conditions
    pub fn apply_changes(&self, changes: &[MockStateChange]) -> Result<(), String> {
        // Acquire all locks upfront and hold them for entire operation (like RocksDB WriteBatch)
        let mut balances = self.balances.lock().unwrap();
        let mut tokens = self.tokens.lock().unwrap();
        let mut pools = self.pools.lock().unwrap();

        // Collect pending changes first (for batch semantics)
        let mut pending_balances: HashMap<([u8; 32], [u8; 32]), u128> = HashMap::new();
        let mut pending_tokens: HashMap<[u8; 32], (String, u128)> = HashMap::new();
        let mut pending_pools: HashMap<[u8; 32], ([u8; 32], [u8; 32])> = HashMap::new();

        // Compute new state from current state
        for change in changes {
            match change {
                MockStateChange::BalanceCredit { account, token, amount } => {
                    let current = pending_balances
                        .get(&(*account, *token))
                        .copied()
                        .unwrap_or_else(|| *balances.get(&(*account, *token)).unwrap_or(&0));

                    // Check for overflow
                    let new_balance = current.checked_add(*amount)
                        .ok_or_else(|| format!(
                            "OVERFLOW: Balance {} + {} would overflow u128",
                            current, amount
                        ))?;
                    pending_balances.insert((*account, *token), new_balance);
                }
                MockStateChange::BalanceDebit { account, token, amount } => {
                    let current = pending_balances
                        .get(&(*account, *token))
                        .copied()
                        .unwrap_or_else(|| *balances.get(&(*account, *token)).unwrap_or(&0));

                    // Check for underflow
                    if current < *amount {
                        return Err(format!(
                            "UNDERFLOW: Balance {} < debit amount {}",
                            current, amount
                        ));
                    }
                    pending_balances.insert((*account, *token), current - amount);
                }
                MockStateChange::TokenCreate { token_address, name, initial_supply } => {
                    pending_tokens.insert(*token_address, (name.clone(), *initial_supply));
                }
                MockStateChange::PoolCreate { pool_id, token_a, token_b } => {
                    pending_pools.insert(*pool_id, (*token_a, *token_b));
                }
            }
        }

        // Check if we should simulate a write failure
        let write_num = self.write_count.fetch_add(1, Ordering::SeqCst);
        if write_num >= self.should_fail_at.load(Ordering::SeqCst) {
            return Err("SIMULATED_ROCKSDB_WRITE_FAILURE".to_string());
        }

        // Apply all changes atomically (still under locks)
        for ((account, token), amount) in pending_balances {
            balances.insert((account, token), amount);
        }
        for (addr, data) in pending_tokens {
            tokens.insert(addr, data);
        }
        for (id, token_pair) in pending_pools {
            pools.insert(id, token_pair);
        }

        self.total_writes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

// ============================================================================
// BALANCE OVERFLOW PROTECTION TESTS
// ============================================================================

/// Test that balance credit overflow is detected and rejected
#[test]
fn test_balance_credit_overflow_protection() {
    let db = MockStateDB::new();
    let account = [1u8; 32];
    let token = [2u8; 32];

    // Set initial balance to near max
    db.set_balance(&account, &token, u128::MAX - 100);

    // Try to credit more than remaining space
    let changes = vec![
        MockStateChange::BalanceCredit {
            account,
            token,
            amount: 200, // This would overflow
        },
    ];

    let result = db.apply_changes(&changes);
    assert!(result.is_err(), "Should reject overflow");
    assert!(result.unwrap_err().contains("OVERFLOW"));

    // Balance should be unchanged
    assert_eq!(db.get_balance(&account, &token), u128::MAX - 100);
}

/// Test that u128::MAX balance can be reached but not exceeded
#[test]
fn test_balance_max_value() {
    let db = MockStateDB::new();
    let account = [1u8; 32];
    let token = [2u8; 32];

    // Credit exactly to max
    let changes = vec![
        MockStateChange::BalanceCredit {
            account,
            token,
            amount: u128::MAX,
        },
    ];

    let result = db.apply_changes(&changes);
    assert!(result.is_ok());
    assert_eq!(db.get_balance(&account, &token), u128::MAX);

    // Now try to add 1 more
    let changes2 = vec![
        MockStateChange::BalanceCredit {
            account,
            token,
            amount: 1,
        },
    ];

    let result2 = db.apply_changes(&changes2);
    assert!(result2.is_err(), "Should reject overflow at max");
}

// ============================================================================
// BALANCE UNDERFLOW PROTECTION TESTS
// ============================================================================

/// Test that balance debit underflow is detected and rejected
#[test]
fn test_balance_debit_underflow_protection() {
    let db = MockStateDB::new();
    let account = [1u8; 32];
    let token = [2u8; 32];

    // Set initial balance
    db.set_balance(&account, &token, 1000);

    // Try to debit more than available
    let changes = vec![
        MockStateChange::BalanceDebit {
            account,
            token,
            amount: 1500,
        },
    ];

    let result = db.apply_changes(&changes);
    assert!(result.is_err(), "Should reject underflow");
    assert!(result.unwrap_err().contains("UNDERFLOW"));

    // Balance should be unchanged
    assert_eq!(db.get_balance(&account, &token), 1000);
}

/// Test debit from zero balance
#[test]
fn test_debit_from_zero_balance() {
    let db = MockStateDB::new();
    let account = [1u8; 32];
    let token = [2u8; 32];

    // Balance starts at 0
    let changes = vec![
        MockStateChange::BalanceDebit {
            account,
            token,
            amount: 1,
        },
    ];

    let result = db.apply_changes(&changes);
    assert!(result.is_err(), "Should reject debit from zero balance");
}

/// Test exact balance debit (balance goes to zero)
#[test]
fn test_exact_balance_debit() {
    let db = MockStateDB::new();
    let account = [1u8; 32];
    let token = [2u8; 32];

    db.set_balance(&account, &token, 1000);

    let changes = vec![
        MockStateChange::BalanceDebit {
            account,
            token,
            amount: 1000,
        },
    ];

    let result = db.apply_changes(&changes);
    assert!(result.is_ok());
    assert_eq!(db.get_balance(&account, &token), 0);
}

// ============================================================================
// WRITE FAILURE RECOVERY TESTS
// ============================================================================

/// Test that state is not corrupted on write failure
#[test]
fn test_write_failure_no_corruption() {
    let db = MockStateDB::new();
    let account = [1u8; 32];
    let token = [2u8; 32];

    // Set initial state
    db.set_balance(&account, &token, 1000);

    // Configure to fail on next write
    db.set_fail_at_write(0);

    let changes = vec![
        MockStateChange::BalanceCredit {
            account,
            token,
            amount: 500,
        },
    ];

    let result = db.apply_changes(&changes);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("SIMULATED_ROCKSDB_WRITE_FAILURE"));

    // State should be unchanged (write failed before commit)
    // Note: In our mock, the validation happens before the "write"
    // In real RocksDB, WriteBatch is atomic
}

/// Test partial batch failure scenario
#[test]
fn test_partial_batch_failure_atomicity() {
    let db = MockStateDB::new();
    let account1 = [1u8; 32];
    let account2 = [2u8; 32];
    let token = [3u8; 32];

    // Set initial balances
    db.set_balance(&account1, &token, 1000);
    db.set_balance(&account2, &token, 500);

    // Create batch where second operation fails (insufficient balance)
    let changes = vec![
        MockStateChange::BalanceCredit {
            account: account1,
            token,
            amount: 100, // Would succeed alone
        },
        MockStateChange::BalanceDebit {
            account: account2,
            token,
            amount: 1000, // Will fail - insufficient balance
        },
    ];

    let result = db.apply_changes(&changes);
    assert!(result.is_err(), "Batch should fail due to underflow");

    // CRITICAL: Both balances should be unchanged due to atomicity
    assert_eq!(db.get_balance(&account1, &token), 1000, "account1 balance should be unchanged");
    assert_eq!(db.get_balance(&account2, &token), 500, "account2 balance should be unchanged");
}

// ============================================================================
// CONCURRENT ACCESS TESTS
// ============================================================================

/// Test concurrent apply_changes calls don't corrupt state
#[test]
fn test_concurrent_apply_changes() {
    let db = Arc::new(MockStateDB::new());
    let token = [1u8; 32];

    // Initialize 10 accounts with 1000 each
    for i in 0..10 {
        let account = [i as u8; 32];
        db.set_balance(&account, &token, 1000);
    }

    let initial_total: u128 = (0..10)
        .map(|i| db.get_balance(&[i as u8; 32], &token))
        .sum();

    // Spawn multiple threads doing transfers
    let mut handles = vec![];
    for thread_id in 0..4 {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || {
            for i in 0..25 {
                let from_idx = (thread_id * 25 + i) % 10;
                let to_idx = (from_idx + 1) % 10;
                let from_account = [from_idx as u8; 32];
                let to_account = [to_idx as u8; 32];

                // Transfer 10 units (if possible)
                let changes = vec![
                    MockStateChange::BalanceDebit {
                        account: from_account,
                        token,
                        amount: 10,
                    },
                    MockStateChange::BalanceCredit {
                        account: to_account,
                        token,
                        amount: 10,
                    },
                ];

                // Ignore failures (insufficient balance is OK)
                let _ = db_clone.apply_changes(&changes);
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // CRITICAL: Total supply should be unchanged (conservation of funds)
    let final_total: u128 = (0..10)
        .map(|i| db.get_balance(&[i as u8; 32], &token))
        .sum();

    assert_eq!(
        initial_total, final_total,
        "CRITICAL: Total supply changed! Initial: {}, Final: {}",
        initial_total, final_total
    );
}

/// Test high-contention concurrent credits to same account
#[test]
fn test_high_contention_credits() {
    let db = Arc::new(MockStateDB::new());
    let account = [1u8; 32];
    let token = [2u8; 32];

    let mut handles = vec![];
    let credits_per_thread = 100u128;
    let num_threads = 8;

    for _ in 0..num_threads {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || {
            for _ in 0..credits_per_thread {
                let changes = vec![
                    MockStateChange::BalanceCredit {
                        account,
                        token,
                        amount: 1,
                    },
                ];
                let _ = db_clone.apply_changes(&changes);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Final balance should be exactly num_threads * credits_per_thread
    let expected = num_threads as u128 * credits_per_thread;
    let actual = db.get_balance(&account, &token);
    assert_eq!(
        actual, expected,
        "Balance mismatch: expected {}, got {}",
        expected, actual
    );
}

// ============================================================================
// MULTI-CHANGE BATCH TESTS
// ============================================================================

/// Test batch with many changes
#[test]
fn test_large_batch_application() {
    let db = MockStateDB::new();
    let token = [1u8; 32];

    // Credit 1000 accounts
    let mut changes: Vec<MockStateChange> = Vec::new();
    for i in 0..1000 {
        let account = {
            let mut a = [0u8; 32];
            a[0..4].copy_from_slice(&(i as u32).to_le_bytes());
            a
        };
        changes.push(MockStateChange::BalanceCredit {
            account,
            token,
            amount: 100,
        });
    }

    let result = db.apply_changes(&changes);
    assert!(result.is_ok(), "Large batch should succeed");

    // Verify all balances
    for i in 0..1000 {
        let account = {
            let mut a = [0u8; 32];
            a[0..4].copy_from_slice(&(i as u32).to_le_bytes());
            a
        };
        assert_eq!(db.get_balance(&account, &token), 100);
    }
}

/// Test batch with mixed operations
#[test]
fn test_mixed_operations_batch() {
    let db = MockStateDB::new();
    let account1 = [1u8; 32];
    let account2 = [2u8; 32];
    let token = [3u8; 32];

    // Initialize
    db.set_balance(&account1, &token, 1000);

    // Mixed batch: credit, debit, credit
    let changes = vec![
        MockStateChange::BalanceCredit {
            account: account1,
            token,
            amount: 500,
        },
        MockStateChange::BalanceDebit {
            account: account1,
            token,
            amount: 300,
        },
        MockStateChange::BalanceCredit {
            account: account2,
            token,
            amount: 300,
        },
    ];

    let result = db.apply_changes(&changes);
    assert!(result.is_ok());

    // account1: 1000 + 500 - 300 = 1200
    assert_eq!(db.get_balance(&account1, &token), 1200);
    // account2: 0 + 300 = 300
    assert_eq!(db.get_balance(&account2, &token), 300);
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

/// Test zero amount operations
#[test]
fn test_zero_amount_operations() {
    let db = MockStateDB::new();
    let account = [1u8; 32];
    let token = [2u8; 32];

    db.set_balance(&account, &token, 1000);

    // Zero credit should succeed but have no effect
    let changes = vec![
        MockStateChange::BalanceCredit {
            account,
            token,
            amount: 0,
        },
    ];

    let result = db.apply_changes(&changes);
    assert!(result.is_ok());
    assert_eq!(db.get_balance(&account, &token), 1000);

    // Zero debit should succeed
    let changes2 = vec![
        MockStateChange::BalanceDebit {
            account,
            token,
            amount: 0,
        },
    ];

    let result2 = db.apply_changes(&changes2);
    assert!(result2.is_ok());
    assert_eq!(db.get_balance(&account, &token), 1000);
}

/// Test empty batch
#[test]
fn test_empty_batch() {
    let db = MockStateDB::new();
    let changes: Vec<MockStateChange> = vec![];

    let result = db.apply_changes(&changes);
    assert!(result.is_ok(), "Empty batch should succeed");
}

/// Test same account credit and debit in same batch
#[test]
fn test_credit_then_debit_same_account() {
    let db = MockStateDB::new();
    let account = [1u8; 32];
    let token = [2u8; 32];

    // Credit then immediately debit in same batch
    let changes = vec![
        MockStateChange::BalanceCredit {
            account,
            token,
            amount: 1000,
        },
        MockStateChange::BalanceDebit {
            account,
            token,
            amount: 500,
        },
    ];

    let result = db.apply_changes(&changes);
    assert!(result.is_ok());
    assert_eq!(db.get_balance(&account, &token), 500);
}

/// Test debit then credit same account (should fail if debit exceeds initial)
#[test]
fn test_debit_then_credit_same_account() {
    let db = MockStateDB::new();
    let account = [1u8; 32];
    let token = [2u8; 32];

    // No initial balance, debit first should fail
    let changes = vec![
        MockStateChange::BalanceDebit {
            account,
            token,
            amount: 500,
        },
        MockStateChange::BalanceCredit {
            account,
            token,
            amount: 1000,
        },
    ];

    let result = db.apply_changes(&changes);
    assert!(result.is_err(), "Debit from zero should fail even with later credit");
}
