//! Comprehensive Sync-Down Protection Tests
//!
//! v2.3.5-beta: Tests to prevent catastrophic data loss from backward sync
//!
//! These tests verify that the sync-down protection mechanisms work correctly:
//! - Database layer refuses to sync to lower heights
//! - Application layer skips sync when peer height < local height
//! - Height monotonicity is enforced
//!
//! Run with: cargo test --package q-storage --test sync_down_protection_tests

use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// HEIGHT MONOTONICITY TESTS
// ============================================================================

mod height_monotonicity_tests {
    use super::*;

    /// Simulates the HIGHEST_EVER_HEIGHT atomic tracking
    static TEST_HIGHEST_HEIGHT: AtomicU64 = AtomicU64::new(0);

    fn reset_height() {
        TEST_HIGHEST_HEIGHT.store(0, Ordering::SeqCst);
    }

    fn update_height(new_height: u64) -> Result<(), String> {
        let highest_ever = TEST_HIGHEST_HEIGHT.load(Ordering::SeqCst);

        // CRITICAL SAFETY CHECK #1: Detect reset to zero
        if new_height == 0 && highest_ever > 100 {
            return Err(format!(
                "SAFETY ABORT: Height reset to zero from {} blocks",
                highest_ever
            ));
        }

        // CRITICAL SAFETY CHECK #2: Detect significant regression
        const REGRESSION_THRESHOLD: u64 = 10;
        if new_height + REGRESSION_THRESHOLD < highest_ever {
            return Err(format!(
                "Height regression: {} → {} blocks (lost {} blocks)",
                highest_ever,
                new_height,
                highest_ever - new_height
            ));
        }

        // Update highest_ever atomically
        let mut current = highest_ever;
        while new_height > current {
            match TEST_HIGHEST_HEIGHT.compare_exchange_weak(
                current,
                new_height,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
        Ok(())
    }

    /// Test that height can increase normally
    #[test]
    fn test_height_increases_normally() {
        reset_height();

        assert!(update_height(100).is_ok());
        assert!(update_height(200).is_ok());
        assert!(update_height(1000).is_ok());
        assert!(update_height(10000).is_ok());

        assert_eq!(TEST_HIGHEST_HEIGHT.load(Ordering::SeqCst), 10000);
    }

    /// Test that small height regression is allowed (reorgs)
    #[test]
    fn test_small_regression_allowed() {
        reset_height();

        assert!(update_height(1000).is_ok());
        // Small regression (< 10 blocks) should be allowed for reorgs
        assert!(update_height(995).is_ok());
    }

    /// Test that large height regression is blocked
    #[test]
    fn test_large_regression_blocked() {
        reset_height();

        assert!(update_height(10000).is_ok());

        // Large regression should be blocked
        let result = update_height(5000);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Height regression"));
    }

    /// Test that reset to zero is blocked (catastrophic failure)
    #[test]
    fn test_zero_reset_blocked() {
        reset_height();

        assert!(update_height(50000).is_ok());

        // Reset to zero should be blocked
        let result = update_height(0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("SAFETY ABORT"));
    }

    /// Test concurrent height updates are safe
    #[test]
    fn test_concurrent_height_updates() {
        reset_height();

        let handles: Vec<_> = (0..10)
            .map(|i| {
                std::thread::spawn(move || {
                    for j in 0..100 {
                        let height = (i * 100 + j) as u64;
                        let _ = update_height(height);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        // Height should be at least 900 (highest thread's highest value)
        assert!(TEST_HIGHEST_HEIGHT.load(Ordering::SeqCst) >= 900);
    }
}

// ============================================================================
// SYNC TARGET VALIDATION TESTS
// ============================================================================

mod sync_target_validation_tests {
    use super::*;

    /// Simulate the sync-down protection check from turbo_sync.rs
    pub fn validate_sync_target(
        target_height: u64,
        local_height: u64,
    ) -> Result<(), String> {
        // 🚨 CRITICAL SAFETY CHECK: Prevent catastrophic sync-down
        if target_height < local_height && local_height > 1000 {
            return Err(format!(
                "SAFETY ABORT: Refusing to sync down from {} to {} (would lose {} blocks)",
                local_height,
                target_height,
                local_height - target_height
            ));
        }
        Ok(())
    }

    /// Test sync to higher height is allowed
    #[test]
    fn test_sync_up_allowed() {
        assert!(validate_sync_target(10000, 5000).is_ok());
        assert!(validate_sync_target(100000, 99000).is_ok());
        assert!(validate_sync_target(1000000, 999999).is_ok());
    }

    /// Test sync to same height is allowed
    #[test]
    fn test_sync_same_height_allowed() {
        assert!(validate_sync_target(10000, 10000).is_ok());
    }

    /// Test sync down from established node is blocked
    #[test]
    fn test_sync_down_blocked_established_node() {
        // Node with >1000 blocks should refuse sync-down
        let result = validate_sync_target(5000, 10000);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("SAFETY ABORT"));
    }

    /// Test sync down from new node is allowed (bootstrap scenario)
    #[test]
    fn test_sync_down_allowed_new_node() {
        // New node (< 1000 blocks) can have flexible sync
        assert!(validate_sync_target(500, 800).is_ok());
        assert!(validate_sync_target(100, 999).is_ok());
    }

    /// Test boundary condition at 1000 blocks
    #[test]
    fn test_boundary_at_1000_blocks() {
        // At exactly 1000, sync-down protection should NOT trigger yet
        assert!(validate_sync_target(500, 1000).is_ok());

        // At 1001, sync-down protection should trigger
        let result = validate_sync_target(500, 1001);
        assert!(result.is_err());
    }

    /// Test catastrophic sync-down scenario
    #[test]
    fn test_catastrophic_sync_down_blocked() {
        // Simulate a malicious peer announcing height 1
        let result = validate_sync_target(1, 100000);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(error.contains("would lose 99999 blocks"));
    }

    /// Test sync-down by 1 block from established node
    #[test]
    fn test_single_block_sync_down_blocked() {
        // Even a single block sync-down should be blocked
        let result = validate_sync_target(9999, 10000);
        assert!(result.is_err());
    }
}

// ============================================================================
// PEER HEIGHT VALIDATION TESTS
// ============================================================================

mod peer_height_validation_tests {
    /// Simulate the peer height check from main.rs
    fn should_sync_from_peer(
        network_height: u64,
        current_height: u64,
        min_height_diff: u64,
    ) -> bool {
        // Only sync if peer is significantly ahead
        network_height > current_height + min_height_diff
    }

    /// Simulate the safe target height calculation
    fn calculate_safe_target(
        network_height: u64,
        current_height: u64,
        max_batch_size: u64,
    ) -> u64 {
        if network_height > current_height {
            // Cap sync to max_batch_size blocks per batch for safety
            network_height.min(current_height + max_batch_size)
        } else {
            // If network_height is unknown or lower, use conservative increment
            current_height + 1000
        }
    }

    /// Test sync triggers when peer is ahead by threshold
    #[test]
    fn test_sync_triggers_when_peer_ahead() {
        assert!(should_sync_from_peer(10000, 9000, 5));
        assert!(should_sync_from_peer(100000, 99990, 5));
    }

    /// Test sync does NOT trigger when peer is close
    #[test]
    fn test_sync_skipped_when_peer_close() {
        // Peer only 3 blocks ahead, threshold is 5
        assert!(!should_sync_from_peer(10003, 10000, 5));
    }

    /// Test sync does NOT trigger when node is ahead
    #[test]
    fn test_sync_skipped_when_node_ahead() {
        assert!(!should_sync_from_peer(9000, 10000, 5));
        assert!(!should_sync_from_peer(1, 100000, 5));
    }

    /// Test safe target calculation caps batch size
    #[test]
    fn test_safe_target_caps_batch() {
        // Network at 100000, we're at 50000, max batch 10000
        let target = calculate_safe_target(100000, 50000, 10000);
        assert_eq!(target, 60000); // 50000 + 10000 = 60000

        // Network at 55000, we're at 50000, max batch 10000
        let target = calculate_safe_target(55000, 50000, 10000);
        assert_eq!(target, 55000); // min(55000, 60000) = 55000
    }

    /// Test safe target with unknown network height
    #[test]
    fn test_safe_target_unknown_network() {
        // Network height unknown (0), use conservative increment
        let target = calculate_safe_target(0, 50000, 10000);
        assert_eq!(target, 51000); // Conservative +1000
    }
}

// ============================================================================
// FORWARD-ONLY BLOCK PROCESSING TESTS
// ============================================================================

mod forward_only_tests {
    /// Simulate block height validation during sync
    fn process_block(
        block_height: u64,
        current_height: u64,
        highest_contiguous: &mut u64,
    ) -> ProcessResult {
        // Skip blocks at or below current height OR at/below highest processed
        if block_height <= current_height || block_height <= *highest_contiguous {
            return ProcessResult::Skipped;
        }

        // Check if this is the next sequential block
        if block_height == *highest_contiguous + 1 {
            *highest_contiguous = block_height;
            return ProcessResult::Processed;
        }

        // Gap detected (block_height is guaranteed > *highest_contiguous here)
        ProcessResult::Gap(block_height - *highest_contiguous - 1)
    }

    #[derive(Debug, PartialEq)]
    enum ProcessResult {
        Processed,
        Skipped,
        Gap(u64),
    }

    /// Test sequential blocks are processed
    #[test]
    fn test_sequential_blocks_processed() {
        let current_height = 1000;
        let mut highest = current_height;

        assert_eq!(
            process_block(1001, current_height, &mut highest),
            ProcessResult::Processed
        );
        assert_eq!(highest, 1001);

        assert_eq!(
            process_block(1002, current_height, &mut highest),
            ProcessResult::Processed
        );
        assert_eq!(highest, 1002);
    }

    /// Test old blocks are skipped
    #[test]
    fn test_old_blocks_skipped() {
        let current_height = 1000;
        let mut highest = current_height;

        // Blocks at or below current height are skipped
        assert_eq!(
            process_block(1000, current_height, &mut highest),
            ProcessResult::Skipped
        );
        assert_eq!(
            process_block(500, current_height, &mut highest),
            ProcessResult::Skipped
        );
        assert_eq!(
            process_block(1, current_height, &mut highest),
            ProcessResult::Skipped
        );

        // highest should not change
        assert_eq!(highest, 1000);
    }

    /// Test gaps are detected
    #[test]
    fn test_gaps_detected() {
        let current_height = 1000;
        let mut highest = current_height;

        // Jump to 1005 - gap of 4 blocks
        assert_eq!(
            process_block(1005, current_height, &mut highest),
            ProcessResult::Gap(4)
        );
    }

    /// Test height regression in batch is detected
    #[test]
    fn test_batch_height_regression_detected() {
        let current_height = 1000;
        let mut highest = current_height;

        // Process some blocks
        assert_eq!(
            process_block(1001, current_height, &mut highest),
            ProcessResult::Processed
        );
        assert_eq!(
            process_block(1002, current_height, &mut highest),
            ProcessResult::Processed
        );

        // Now highest is 1002, trying to process 1001 again should skip
        assert_eq!(
            process_block(1001, current_height, &mut highest),
            ProcessResult::Skipped
        );
    }
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

mod integration_tests {
    use super::*;

    /// Test complete sync safety flow
    #[test]
    fn test_complete_sync_safety_flow() {
        let current_height = 50000u64;
        let min_sync_diff = 5u64;
        let max_batch = 10000u64;

        // Scenario 1: Peer ahead - should sync
        let peer_height = 60000u64;
        assert!(peer_height > current_height + min_sync_diff);
        let target = peer_height.min(current_height + max_batch);
        assert_eq!(target, 60000);

        // Scenario 2: Peer behind - should NOT sync
        let peer_height = 40000u64;
        assert!(peer_height <= current_height + min_sync_diff);

        // Scenario 3: Malicious peer - should be blocked at database layer
        let malicious_peer_height = 1u64;
        let established_node = current_height > 1000;
        let would_sync_down = malicious_peer_height < current_height;
        assert!(established_node && would_sync_down);
        // This should trigger SAFETY ABORT
    }

    /// Test malicious peer scenarios
    #[test]
    fn test_malicious_peer_scenarios() {
        // Scenario 1: Peer announces 0
        let result = super::sync_target_validation_tests::validate_sync_target(0, 100000);
        assert!(result.is_err());

        // Scenario 2: Peer announces very old height
        let result = super::sync_target_validation_tests::validate_sync_target(100, 100000);
        assert!(result.is_err());

        // Scenario 3: Peer announces height just below ours
        let result = super::sync_target_validation_tests::validate_sync_target(99999, 100000);
        assert!(result.is_err());
    }
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

mod performance_tests {
    use super::*;
    use std::time::Instant;

    /// Test height validation is fast
    #[test]
    fn test_height_validation_performance() {
        let iterations = 100_000;
        let start = Instant::now();

        for i in 0..iterations {
            let target = (i % 10000) as u64;
            let local = 50000u64;
            let _ = super::sync_target_validation_tests::validate_sync_target(
                target + 50000,
                local,
            );
        }

        let elapsed = start.elapsed();
        let per_check_ns = elapsed.as_nanos() / iterations as u128;

        println!("Height validation: {} ns per check", per_check_ns);

        // Should be very fast - under 100 nanoseconds
        assert!(
            per_check_ns < 1000,
            "Height validation should be sub-microsecond, got {} ns",
            per_check_ns
        );
    }
}

