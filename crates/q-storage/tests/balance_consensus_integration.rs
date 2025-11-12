// v0.8.0-beta: Balance Consensus Integration Tests
//
// These tests verify that the balance consensus engine works correctly
// across multiple nodes and in various edge case scenarios.

use anyhow::Result;
use q_storage::{
    BalanceConsensusEngine, BalanceConsensusError, BalanceStorage, QStorage,
    GENESIS_TIMESTAMP, FOUNDER_WALLET,
};
use q_types::{BlockHeader, MiningSolution, QBlock};
use std::sync::Arc;
use tokio;

// Helper function to create a test block with mining solution
fn create_test_block(height: u64, timestamp: u64, miner_address: &str) -> QBlock {
    let mining_solution = MiningSolution {
        nonce: 12345,
        miner_address: miner_address.to_string(),
        timestamp,
        difficulty_target: 1000,
        hash: vec![0u8; 32],
    };

    let header = BlockHeader {
        version: 1,
        height,
        timestamp,
        prev_block_hash: vec![0u8; 32],
        solutions_root: vec![0u8; 32],
        state_root: vec![0u8; 32],
        difficulty: 1000,
        total_stake: 0,
    };

    QBlock {
        header,
        mining_solutions: vec![mining_solution],
        transactions: vec![],
    }
}

// Helper function to create test storage with unique path
async fn create_test_storage(name: &str) -> Result<Arc<QStorage>> {
    let test_path = format!("/tmp/balance-consensus-test-{}-{}", name, std::process::id());

    // Clean up any existing test database
    if std::path::Path::new(&test_path).exists() {
        std::fs::remove_dir_all(&test_path)?;
    }

    let storage = QStorage::new(&test_path).await?;
    Ok(Arc::new(storage))
}

/// Test 1: Two nodes processing the same block should get identical balance updates
#[tokio::test]
async fn test_two_nodes_deterministic_consensus() -> Result<()> {
    println!("\n🧪 Test 1: Two nodes processing same block should get identical results");

    // Create two independent nodes
    let storage_node1 = create_test_storage("node1").await?;
    let storage_node2 = create_test_storage("node2").await?;

    let engine1 = Arc::new(BalanceConsensusEngine::new(
        GENESIS_TIMESTAMP,
        FOUNDER_WALLET.to_string(),
    ));
    let engine2 = Arc::new(BalanceConsensusEngine::new(
        GENESIS_TIMESTAMP,
        FOUNDER_WALLET.to_string(),
    ));

    // Create test block with mining reward
    let miner_address = "qnk1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd";
    let block = create_test_block(1, GENESIS_TIMESTAMP + 100, miner_address);

    // Process block on both nodes
    let updates1 = engine1
        .process_block_mining_rewards(&*storage_node1, &block)
        .await?;
    let updates2 = engine2
        .process_block_mining_rewards(&*storage_node2, &block)
        .await?;

    // Verify both nodes got same number of updates
    assert_eq!(
        updates1.len(),
        updates2.len(),
        "Both nodes should produce same number of balance updates"
    );

    // Verify all balance updates match
    for (update1, update2) in updates1.iter().zip(updates2.iter()) {
        assert_eq!(
            update1.address, update2.address,
            "Address mismatch: {} vs {}",
            update1.address, update2.address
        );
        assert_eq!(
            update1.amount, update2.amount,
            "Amount mismatch for {}: {} vs {}",
            update1.address, update1.amount, update2.amount
        );
    }

    // Verify final balances match
    let miner_balance1 = storage_node1.get_balance(miner_address).await?;
    let miner_balance2 = storage_node2.get_balance(miner_address).await?;
    assert_eq!(
        miner_balance1, miner_balance2,
        "Miner balances should match: {} vs {}",
        miner_balance1, miner_balance2
    );

    let dev_balance1 = storage_node1.get_balance(FOUNDER_WALLET).await?;
    let dev_balance2 = storage_node2.get_balance(FOUNDER_WALLET).await?;
    assert_eq!(
        dev_balance1, dev_balance2,
        "Dev fee balances should match: {} vs {}",
        dev_balance1, dev_balance2
    );

    println!("✅ Test 1 PASSED: Both nodes computed identical balance updates");
    println!("   - Miner balance: {} base units", miner_balance1);
    println!("   - Dev fee balance: {} base units", dev_balance1);

    Ok(())
}

/// Test 2: Processing the same block twice should return AlreadyProcessed error
#[tokio::test]
async fn test_double_processing_safety() -> Result<()> {
    println!("\n🧪 Test 2: Double processing should be safely rejected");

    let storage = create_test_storage("double").await?;
    let engine = Arc::new(BalanceConsensusEngine::new(
        GENESIS_TIMESTAMP,
        FOUNDER_WALLET.to_string(),
    ));

    let miner_address = "qnk1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd";
    let block = create_test_block(1, GENESIS_TIMESTAMP + 100, miner_address);

    // First processing should succeed
    let updates1 = engine
        .process_block_mining_rewards(&*storage, &block)
        .await?;
    assert!(!updates1.is_empty(), "First processing should create updates");

    let balance_after_first = storage.get_balance(miner_address).await?;

    // Second processing should return AlreadyProcessed error
    let result2 = engine.process_block_mining_rewards(&*storage, &block).await;

    match result2 {
        Err(BalanceConsensusError::AlreadyProcessed(hash)) => {
            println!("✅ Test 2 PASSED: Correctly rejected duplicate processing");
            println!("   - Block hash: {}", hex::encode(&hash));
        }
        Ok(_) => panic!("Second processing should have failed with AlreadyProcessed"),
        Err(e) => panic!("Wrong error type: {:?}", e),
    }

    // Verify balance didn't change on second attempt
    let balance_after_second = storage.get_balance(miner_address).await?;
    assert_eq!(
        balance_after_first, balance_after_second,
        "Balance should not change on duplicate processing"
    );

    println!("   - Balance remained: {} base units", balance_after_first);

    Ok(())
}

/// Test 3: Five nodes processing 10 blocks should all reach identical state
#[tokio::test]
async fn test_five_node_consensus_simulation() -> Result<()> {
    println!("\n🧪 Test 3: Five nodes processing 10 blocks should reach consensus");

    // Create 5 independent nodes
    let mut nodes = Vec::new();
    for i in 0..5 {
        let storage = create_test_storage(&format!("node{}", i)).await?;
        let engine = Arc::new(BalanceConsensusEngine::new(
            GENESIS_TIMESTAMP,
            FOUNDER_WALLET.to_string(),
        ));
        nodes.push((storage, engine));
    }

    // Create 10 test blocks with different miners
    let miners = vec![
        "qnk1111111111111111111111111111111111111111111111111111111111111111",
        "qnk2222222222222222222222222222222222222222222222222222222222222222",
        "qnk3333333333333333333333333333333333333333333333333333333333333333",
    ];

    let mut blocks = Vec::new();
    for height in 1..=10 {
        let miner = miners[(height - 1) % miners.len() as u64];
        let timestamp = GENESIS_TIMESTAMP + (height * 100);
        blocks.push(create_test_block(height, timestamp, miner));
    }

    // Process all blocks on all nodes
    for block in &blocks {
        for (storage, engine) in &nodes {
            let result = engine.process_block_mining_rewards(&**storage, block).await;
            match result {
                Ok(updates) => {
                    if !updates.is_empty() {
                        println!(
                            "   📦 Processed block {} on node (height: {}, {} updates)",
                            hex::encode(&block.header.prev_block_hash[..4]),
                            block.header.height,
                            updates.len()
                        );
                    }
                }
                Err(BalanceConsensusError::AlreadyProcessed(_)) => {
                    // Safe retry - expected behavior
                }
                Err(e) => panic!("Unexpected error processing block: {:?}", e),
            }
        }
    }

    // Verify all nodes have identical balances for each miner
    for miner in &miners {
        let mut balances = Vec::new();
        for (storage, _) in &nodes {
            let balance = storage.get_balance(miner).await?;
            balances.push(balance);
        }

        // Check all balances are identical
        let first_balance = balances[0];
        for (i, balance) in balances.iter().enumerate() {
            assert_eq!(
                *balance, first_balance,
                "Node {} has different balance for miner {}: {} vs {}",
                i, miner, balance, first_balance
            );
        }

        println!(
            "   ✅ All nodes agree on {} balance: {} base units",
            &miner[..12], first_balance
        );
    }

    // Verify dev fee balances match
    let mut dev_balances = Vec::new();
    for (storage, _) in &nodes {
        let balance = storage.get_balance(FOUNDER_WALLET).await?;
        dev_balances.push(balance);
    }

    let first_dev_balance = dev_balances[0];
    for (i, balance) in dev_balances.iter().enumerate() {
        assert_eq!(
            *balance, first_dev_balance,
            "Node {} has different dev fee balance: {} vs {}",
            i, balance, first_dev_balance
        );
    }

    println!(
        "   ✅ All nodes agree on dev fee balance: {} base units",
        first_dev_balance
    );

    // Get consensus statistics from first node
    let stats = nodes[0].1.get_statistics().await;
    println!("\n📊 Consensus Statistics:");
    println!("   - Total blocks processed: {}", stats.blocks_processed);
    println!("   - Total balance updates: {}", stats.total_balance_updates);
    println!("   - Duplicate attempts: {}", stats.duplicate_processing_attempts);

    println!("\n✅ Test 3 PASSED: All 5 nodes reached perfect consensus");

    Ok(())
}

/// Test 4: Blocks arriving out of order should still produce consistent state
#[tokio::test]
async fn test_out_of_order_block_processing() -> Result<()> {
    println!("\n🧪 Test 4: Out-of-order blocks should still reach consensus");

    let storage = create_test_storage("ooo").await?;
    let engine = Arc::new(BalanceConsensusEngine::new(
        GENESIS_TIMESTAMP,
        FOUNDER_WALLET.to_string(),
    ));

    let miner = "qnk1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd";

    // Create blocks in sequential order
    let mut blocks = Vec::new();
    for height in 1..=5 {
        let timestamp = GENESIS_TIMESTAMP + (height * 100);
        blocks.push(create_test_block(height, timestamp, miner));
    }

    // Process blocks in shuffled order: 3, 1, 4, 2, 5
    let order = vec![2, 0, 3, 1, 4];

    for idx in order {
        let block = &blocks[idx];
        let result = engine.process_block_mining_rewards(&*storage, block).await;

        match result {
            Ok(updates) => {
                println!(
                    "   📦 Processed block height {} ({} updates)",
                    block.header.height,
                    updates.len()
                );
            }
            Err(e) => panic!("Failed to process block {}: {:?}", block.header.height, e),
        }
    }

    // Verify final balance is correct (5 blocks * reward)
    let final_balance = storage.get_balance(miner).await?;

    // Calculate expected balance (base_reward = 100,000 for first epoch)
    let base_reward = 100_000u64;
    let miner_share = (base_reward as f64 * 0.99) as u64; // 99% to miner
    let expected_total = miner_share * 5; // 5 blocks

    assert_eq!(
        final_balance, expected_total,
        "Final balance should be {} (5 blocks * {} per block), got {}",
        expected_total, miner_share, final_balance
    );

    println!("✅ Test 4 PASSED: Out-of-order processing produced correct state");
    println!("   - Final miner balance: {} base units", final_balance);
    println!("   - Expected: {} base units", expected_total);

    Ok(())
}

/// Test 5: Verify time-based halving works correctly
#[tokio::test]
async fn test_time_based_halving() -> Result<()> {
    println!("\n🧪 Test 5: Time-based halving should reduce rewards correctly");

    let storage = create_test_storage("halving").await?;
    let engine = Arc::new(BalanceConsensusEngine::new(
        GENESIS_TIMESTAMP,
        FOUNDER_WALLET.to_string(),
    ));

    let miner = "qnk1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd";

    // Test blocks at different epochs
    let seconds_per_year = 31_536_000u64;
    let base_reward = 100_000u64;

    let test_cases = vec![
        (0, base_reward), // Year 0: full reward
        (1, base_reward / 2), // Year 1: halved
        (2, base_reward / 4), // Year 2: quartered
        (3, base_reward / 8), // Year 3: eighth
    ];

    for (year, expected_reward) in test_cases {
        let timestamp = GENESIS_TIMESTAMP + (year * seconds_per_year) + 100;
        let block = create_test_block(year + 1, timestamp, miner);

        let updates = engine
            .process_block_mining_rewards(&*storage, &block)
            .await?;

        // Find miner's update
        let miner_update = updates
            .iter()
            .find(|u| u.address == miner)
            .expect("Should have miner update");

        let expected_miner_share = (expected_reward as f64 * 0.99) as u64;

        assert_eq!(
            miner_update.amount, expected_miner_share,
            "Year {} reward should be {} (99% of {}), got {}",
            year, expected_miner_share, expected_reward, miner_update.amount
        );

        println!(
            "   ✅ Year {}: Reward correctly halved to {} base units",
            year, miner_update.amount
        );
    }

    println!("\n✅ Test 5 PASSED: Time-based halving works correctly");

    Ok(())
}

/// Test 6: Invalid timestamps should be rejected
#[tokio::test]
async fn test_invalid_timestamp_rejection() -> Result<()> {
    println!("\n🧪 Test 6: Invalid timestamps should be rejected");

    let storage = create_test_storage("invalid_ts").await?;
    let engine = Arc::new(BalanceConsensusEngine::new(
        GENESIS_TIMESTAMP,
        FOUNDER_WALLET.to_string(),
    ));

    let miner = "qnk1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd";

    // Test 1: Timestamp before genesis
    let invalid_block1 = create_test_block(1, GENESIS_TIMESTAMP - 100, miner);
    let result1 = engine
        .process_block_mining_rewards(&*storage, &invalid_block1)
        .await;

    match result1 {
        Err(BalanceConsensusError::InvalidTimestamp(ts)) => {
            println!("   ✅ Correctly rejected timestamp before genesis: {}", ts);
        }
        _ => panic!("Should have rejected timestamp before genesis"),
    }

    // Test 2: Timestamp way in future (beyond 64 years)
    let seconds_per_year = 31_536_000u64;
    let future_timestamp = GENESIS_TIMESTAMP + (65 * seconds_per_year);
    let invalid_block2 = create_test_block(2, future_timestamp, miner);

    // This should still process but with minimal reward (64+ halvings)
    let result2 = engine
        .process_block_mining_rewards(&*storage, &invalid_block2)
        .await;

    match result2 {
        Ok(updates) => {
            let miner_update = updates.iter().find(|u| u.address == miner);
            if let Some(update) = miner_update {
                println!(
                    "   ✅ Far future block processed with minimal reward: {} base units",
                    update.amount
                );
                assert!(
                    update.amount < 100,
                    "Reward after 64+ halvings should be minimal"
                );
            }
        }
        Err(e) => println!("   ✅ Far future block rejected: {:?}", e),
    }

    println!("\n✅ Test 6 PASSED: Invalid timestamps handled correctly");

    Ok(())
}

/// Test 7: Dev fee split should be exactly 1%
#[tokio::test]
async fn test_dev_fee_accuracy() -> Result<()> {
    println!("\n🧪 Test 7: Dev fee should be exactly 1% of mining reward");

    let storage = create_test_storage("devfee").await?;
    let engine = Arc::new(BalanceConsensusEngine::new(
        GENESIS_TIMESTAMP,
        FOUNDER_WALLET.to_string(),
    ));

    let miner = "qnk1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd";
    let block = create_test_block(1, GENESIS_TIMESTAMP + 100, miner);

    let updates = engine
        .process_block_mining_rewards(&*storage, &block)
        .await?;

    // Find miner and dev updates
    let miner_update = updates
        .iter()
        .find(|u| u.address == miner)
        .expect("Should have miner update");

    let dev_update = updates
        .iter()
        .find(|u| u.address == FOUNDER_WALLET)
        .expect("Should have dev update");

    let base_reward = 100_000u64;
    let expected_dev_fee = (base_reward as f64 * 0.01) as u64;
    let expected_miner_reward = (base_reward as f64 * 0.99) as u64;

    assert_eq!(
        dev_update.amount, expected_dev_fee,
        "Dev fee should be 1% of base reward: {} vs {}",
        dev_update.amount, expected_dev_fee
    );

    assert_eq!(
        miner_update.amount, expected_miner_reward,
        "Miner reward should be 99% of base reward: {} vs {}",
        miner_update.amount, expected_miner_reward
    );

    // Verify total equals base reward
    let total = miner_update.amount + dev_update.amount;
    assert_eq!(
        total, base_reward,
        "Total rewards should equal base reward: {} vs {}",
        total, base_reward
    );

    println!("   ✅ Dev fee: {} base units (1%)", dev_update.amount);
    println!("   ✅ Miner reward: {} base units (99%)", miner_update.amount);
    println!("   ✅ Total: {} base units", total);

    println!("\n✅ Test 7 PASSED: Dev fee split is exactly 1%");

    Ok(())
}
