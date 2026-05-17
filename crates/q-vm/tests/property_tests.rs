//! Property-based tests for Q-NarwhalKnight Virtual Machine
//!
//! These tests use proptest to generate random inputs and verify
//! that VM properties hold across a wide range of scenarios.

use proptest::prelude::*;
use q_vm::{
    state::{StateDB, VmState},
    vm::StateAccess,
};
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Property test: State root determinism
/// The same state should always produce the same state root
#[test]
fn prop_state_root_deterministic() {
    let rt = Runtime::new().unwrap();

    proptest!(|(
        balances: Vec<(u64, u64)>,
        nonces: Vec<(u64, u64)>
    )| {
        rt.block_on(async {
            // Create two identical states
            let mut state1 = VmState::default();
            let mut state2 = VmState::default();

            // Apply same operations to both states
            for (account, balance) in &balances {
                state1.balances.insert(*account, *balance);
                state2.balances.insert(*account, *balance);
            }

            for (account, nonce) in &nonces {
                state1.nonces.insert(*account, *nonce);
                state2.nonces.insert(*account, *nonce);
            }

            // Update state roots
            state1.update_state_root();
            state2.update_state_root();

            // State roots should be identical
            prop_assert_eq!(state1.state_root, state2.state_root);
        });
    });
}

/// Property test: Balance operations preserve total supply
/// Total balance across all accounts should be preserved during transfers
#[test]
fn prop_balance_conservation() {
    let rt = Runtime::new().unwrap();

    proptest!(|(
        initial_balances: Vec<(u64, u64)>,
        transfers: Vec<(u64, u64, u64)> // (from, to, amount)
    )| {
        rt.block_on(async {
            let state_db = Arc::new(StateDB::new_in_memory());

            // Set initial balances
            let mut total_initial = 0u64;
            for (account, balance) in &initial_balances {
                state_db.set_balance(*account, *balance).await.unwrap();
                total_initial = total_initial.saturating_add(*balance);
            }

            // Perform transfers (only valid ones)
            for (from, to, amount) in &transfers {
                let from_balance = state_db.get_balance(*from).await.unwrap();
                let to_balance = state_db.get_balance(*to).await.unwrap();

                // Only transfer if sufficient balance
                if from_balance >= *amount && from != to {
                    state_db.set_balance(*from, from_balance - amount).await.unwrap();
                    state_db.set_balance(*to, to_balance + amount).await.unwrap();
                }
            }

            // Calculate total final balance
            let mut total_final = 0u64;
            let all_accounts: std::collections::HashSet<u64> = initial_balances.iter()
                .map(|(acc, _)| *acc)
                .chain(transfers.iter().flat_map(|(from, to, _)| [*from, *to]))
                .collect();

            for account in all_accounts {
                let balance = state_db.get_balance(account).await.unwrap();
                total_final = total_final.saturating_add(balance);
            }

            // Total balance should be conserved
            prop_assert_eq!(total_initial, total_final);
        });
    });
}

/// Property test: Storage operations are idempotent
/// Setting the same key-value pair multiple times should have the same result
#[test]
fn prop_storage_idempotent() {
    let rt = Runtime::new().unwrap();

    proptest!(|(
        contract_address: u64,
        key: Vec<u8>,
        value: Vec<u8>,
        repeat_count in 1..10u8
    )| {
        rt.block_on(async {
            let state_db = Arc::new(StateDB::new_in_memory());

            // Set storage value multiple times
            for _ in 0..repeat_count {
                state_db.set_storage(contract_address, key.clone(), value.clone()).await.unwrap();
            }

            // Retrieve value
            let retrieved_value = state_db.get_storage(contract_address, &key).await.unwrap();

            // Should match the set value
            prop_assert_eq!(retrieved_value, Some(value));
        });
    });
}

/// Property test: Nonce operations are monotonic
/// Nonces should never decrease (in real implementation)
#[test]
fn prop_nonce_monotonic() {
    let rt = Runtime::new().unwrap();

    proptest!(|(
        account: u64,
        nonce_updates: Vec<u64>
    )| {
        rt.block_on(async {
            let state_db = Arc::new(StateDB::new_in_memory());

            let mut previous_nonce = 0u64;

            for &new_nonce in &nonce_updates {
                // Only update if new nonce is greater (monotonic)
                if new_nonce > previous_nonce {
                    // In a real implementation, we'd have a method to update nonce
                    // For now, we simulate by updating state directly
                    let mut state = state_db.state.write().await;
                    state.nonces.insert(account, new_nonce);
                    state.update_state_root();
                    drop(state);

                    previous_nonce = new_nonce;
                }

                let current_nonce = state_db.get_nonce(account).await.unwrap();

                // Current nonce should be >= previous nonce
                prop_assert!(current_nonce >= previous_nonce);
            }
        });
    });
}

/// Property test: State root changes on state modification
/// Any state change should result in a different state root
#[test]
fn prop_state_root_sensitivity() {
    let rt = Runtime::new().unwrap();

    proptest!(|(
        account1: u64,
        balance1: u64,
        account2: u64,
        balance2: u64
    )| {
        // Skip if accounts are the same
        prop_assume!(account1 != account2);
        prop_assume!(balance1 != balance2);

        rt.block_on(async {
            let state_db = Arc::new(StateDB::new_in_memory());

            // Set initial state and get root
            state_db.set_balance(account1, balance1).await.unwrap();
            let initial_root = {
                let state = state_db.state.read().await;
                state.state_root
            };

            // Modify state
            state_db.set_balance(account2, balance2).await.unwrap();
            let final_root = {
                let state = state_db.state.read().await;
                state.state_root
            };

            // State root should change
            prop_assert_ne!(initial_root, final_root);
        });
    });
}

/// Property test: Storage keys are unique per contract
/// Different contracts should have isolated storage
#[test]
fn prop_storage_isolation() {
    let rt = Runtime::new().unwrap();

    proptest!(|(
        contract1: u64,
        contract2: u64,
        key: Vec<u8>,
        value1: Vec<u8>,
        value2: Vec<u8>
    )| {
        // Skip if contracts are the same
        prop_assume!(contract1 != contract2);
        prop_assume!(value1 != value2);

        rt.block_on(async {
            let state_db = Arc::new(StateDB::new_in_memory());

            // Set same key with different values for different contracts
            state_db.set_storage(contract1, key.clone(), value1.clone()).await.unwrap();
            state_db.set_storage(contract2, key.clone(), value2.clone()).await.unwrap();

            // Retrieve values
            let retrieved1 = state_db.get_storage(contract1, &key).await.unwrap();
            let retrieved2 = state_db.get_storage(contract2, &key).await.unwrap();

            // Each contract should have its own value
            prop_assert_eq!(retrieved1, Some(value1));
            prop_assert_eq!(retrieved2, Some(value2));
        });
    });
}

/// Property test: Large data handling
/// VM should handle arbitrarily large data within reasonable limits
#[test]
fn prop_large_data_handling() {
    let rt = Runtime::new().unwrap();

    proptest!(|(
        data_size in 1..1024*1024usize, // Up to 1MB
        contract_address: u64
    )| {
        rt.block_on(async {
            let state_db = Arc::new(StateDB::new_in_memory());

            // Create large data
            let key = b"large_data_key".to_vec();
            let large_value = vec![0xAB; data_size];

            // Store large data
            let store_result = state_db.set_storage(contract_address, key.clone(), large_value.clone()).await;

            // Should succeed for reasonable sizes
            if data_size <= 10 * 1024 * 1024 { // 10MB limit
                prop_assert!(store_result.is_ok());

                // Retrieve and verify
                let retrieved = state_db.get_storage(contract_address, &key).await.unwrap();
                prop_assert_eq!(retrieved, Some(large_value));
            }
        });
    });
}
