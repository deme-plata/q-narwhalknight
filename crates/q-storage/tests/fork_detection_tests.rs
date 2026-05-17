//! Comprehensive Fork Detection Tests
//!
//! v3.3.9-beta: Tests for fork detection and resolution
//!
//! These tests verify:
//! - Backward reorg detection
//! - Minority fork detection
//! - Normal sync detection
//! - Peer height tracking
//!
//! Run with: cargo test --package q-storage --test fork_detection_tests

use q_storage::fork_detector::{ForkDetector, ForkEvent, PeerHeightInfo};

// ============================================================================
// FORK DETECTOR CREATION TESTS
// ============================================================================

mod creation_tests {
    use super::*;

    /// Test fork detector creation with defaults
    #[tokio::test]
    async fn test_fork_detector_creation() {
        let detector = ForkDetector::new();

        // Should start with no peer heights
        let heights = detector.get_peer_heights().await;
        assert!(heights.is_empty(), "Should start with no peers");

        // Should return None for consensus height with no peers
        let consensus = detector.get_consensus_height().await;
        assert!(consensus.is_none(), "No consensus without peers");
    }

    /// Test default parameters
    #[tokio::test]
    async fn test_default_parameters() {
        let detector = ForkDetector::new();

        // Safe auto reorg should work for reasonable depths
        assert!(detector.is_safe_auto_reorg(100), "100 blocks should be safe");
        assert!(detector.is_safe_auto_reorg(500), "500 blocks should be safe");
        assert!(detector.is_safe_auto_reorg(1000), "1000 blocks should be safe");
        assert!(!detector.is_safe_auto_reorg(1001), "1001 blocks should NOT be safe");
        assert!(!detector.is_safe_auto_reorg(5000), "5000 blocks should NOT be safe");
    }
}

// ============================================================================
// PEER HEIGHT TRACKING TESTS
// ============================================================================

mod peer_tracking_tests {
    use super::*;

    /// Test updating peer heights
    #[tokio::test]
    async fn test_update_peer_height() {
        let detector = ForkDetector::new();

        detector.update_peer_height("peer1".to_string(), 1000).await;
        detector.update_peer_height("peer2".to_string(), 1001).await;
        detector.update_peer_height("peer3".to_string(), 1002).await;

        let heights = detector.get_peer_heights().await;
        assert_eq!(heights.len(), 3, "Should have 3 peers");
    }

    /// Test peer height updates replace old values
    #[tokio::test]
    async fn test_peer_height_update_replaces() {
        let detector = ForkDetector::new();

        detector.update_peer_height("peer1".to_string(), 1000).await;
        detector.update_peer_height("peer1".to_string(), 2000).await;

        let heights = detector.get_peer_heights().await;
        assert_eq!(heights.len(), 1, "Should still have 1 peer");

        let (_, height) = heights.first().unwrap();
        assert_eq!(*height, 2000, "Height should be updated");
    }

    /// Test multiple peers with same height
    #[tokio::test]
    async fn test_multiple_peers_same_height() {
        let detector = ForkDetector::new();

        for i in 0..5 {
            detector.update_peer_height(format!("peer{}", i), 5000).await;
        }

        let heights = detector.get_peer_heights().await;
        assert_eq!(heights.len(), 5, "Should have 5 peers");

        let consensus = detector.get_consensus_height().await;
        assert_eq!(consensus, Some(5000), "Consensus should be 5000");
    }
}

// ============================================================================
// BACKWARD REORG DETECTION TESTS
// ============================================================================

mod backward_reorg_tests {
    use super::*;

    /// Test backward reorg is detected when network is behind
    #[tokio::test]
    async fn test_backward_reorg_detected() {
        let detector = ForkDetector::new();

        // Simulate 5 peers all at height 2000 (network went backward)
        for i in 0..5 {
            detector.update_peer_height(format!("peer{}", i), 2000).await;
        }

        // Our node is at 5000 (ahead of network)
        let event = detector.detect_fork(5000).await.unwrap();

        match event {
            ForkEvent::BackwardReorg {
                our_height,
                network_height,
                reorg_depth,
            } => {
                assert_eq!(our_height, 5000);
                assert_eq!(network_height, 2000);
                assert_eq!(reorg_depth, 3000);
            }
            other => panic!("Expected BackwardReorg, got {:?}", other),
        }
    }

    /// Test backward reorg with exact boundary
    #[tokio::test]
    async fn test_backward_reorg_boundary() {
        let detector = ForkDetector::new();

        // 5 peers at height 99
        for i in 0..5 {
            detector.update_peer_height(format!("peer{}", i), 99).await;
        }

        // Our node at height 100
        let event = detector.detect_fork(100).await.unwrap();

        match event {
            ForkEvent::BackwardReorg { reorg_depth, .. } => {
                assert_eq!(reorg_depth, 1, "Should detect 1 block reorg");
            }
            other => panic!("Expected BackwardReorg, got {:?}", other),
        }
    }

    /// Test large backward reorg
    #[tokio::test]
    async fn test_large_backward_reorg() {
        let detector = ForkDetector::new();

        // Peers at genesis-like height
        for i in 0..5 {
            detector.update_peer_height(format!("peer{}", i), 100).await;
        }

        // Our node way ahead
        let event = detector.detect_fork(100000).await.unwrap();

        match event {
            ForkEvent::BackwardReorg { reorg_depth, .. } => {
                assert_eq!(reorg_depth, 99900);
                assert!(!detector.is_safe_auto_reorg(reorg_depth), "Large reorg should NOT be safe");
            }
            other => panic!("Expected BackwardReorg, got {:?}", other),
        }
    }
}

// ============================================================================
// FORWARD SYNC DETECTION TESTS
// ============================================================================

mod forward_sync_tests {
    use super::*;

    /// Test normal forward sync is detected
    #[tokio::test]
    async fn test_forward_sync_detected() {
        let detector = ForkDetector::new();

        // Peers ahead of us
        for i in 0..5 {
            detector.update_peer_height(format!("peer{}", i), 10000).await;
        }

        // Our node behind
        let event = detector.detect_fork(9000).await.unwrap();

        // When we're behind the network, this can be ForwardSync or MinorityFork
        // depending on implementation. Both indicate we need to sync forward.
        match event {
            ForkEvent::ForwardSync {
                our_height,
                network_height,
            } => {
                assert_eq!(our_height, 9000);
                assert_eq!(network_height, 10000);
            }
            ForkEvent::MinorityFork {
                our_height,
                majority_height,
                ..
            } => {
                // Also valid - we're behind with 0 peers at our height
                assert_eq!(our_height, 9000);
                assert_eq!(majority_height, 10000);
            }
            other => panic!("Expected ForwardSync or MinorityFork, got {:?}", other),
        }
    }

    /// Test sync when exactly at network height
    #[tokio::test]
    async fn test_at_network_height() {
        let detector = ForkDetector::new();

        // All peers at same height as us
        for i in 0..5 {
            detector.update_peer_height(format!("peer{}", i), 5000).await;
        }

        let event = detector.detect_fork(5000).await.unwrap();

        match event {
            ForkEvent::ForwardSync { our_height, network_height } => {
                assert_eq!(our_height, 5000);
                assert_eq!(network_height, 5000);
            }
            other => panic!("Expected ForwardSync, got {:?}", other),
        }
    }
}

// ============================================================================
// MINORITY FORK DETECTION TESTS
// ============================================================================

mod minority_fork_tests {
    use super::*;

    /// Test minority fork is detected
    #[tokio::test]
    async fn test_minority_fork_detected() {
        let detector = ForkDetector::new();

        // 4 peers at height 3000 (majority)
        for i in 0..4 {
            detector.update_peer_height(format!("majority{}", i), 3000).await;
        }

        // 1 peer on our fork
        detector.update_peer_height("our_fork_peer".to_string(), 2500).await;

        // We're at 2500 (minority)
        let event = detector.detect_fork(2500).await.unwrap();

        match event {
            ForkEvent::MinorityFork {
                our_height,
                majority_height,
                peer_count_majority,
                peer_count_our_fork,
            } => {
                assert_eq!(our_height, 2500);
                assert_eq!(majority_height, 3000);
                assert_eq!(peer_count_majority, 4);
                assert_eq!(peer_count_our_fork, 1);
            }
            other => panic!("Expected MinorityFork, got {:?}", other),
        }
    }

    /// Test not a minority fork when consensus is low
    #[tokio::test]
    async fn test_no_minority_fork_low_consensus() {
        let detector = ForkDetector::new();

        // Peers split evenly - no clear consensus
        detector.update_peer_height("peer1".to_string(), 1000).await;
        detector.update_peer_height("peer2".to_string(), 1000).await;
        detector.update_peer_height("peer3".to_string(), 2000).await;
        detector.update_peer_height("peer4".to_string(), 2000).await;
        detector.update_peer_height("peer5".to_string(), 3000).await;

        // No clear majority, shouldn't detect minority fork
        let event = detector.detect_fork(1500).await.unwrap();

        // With split peers, various outcomes are valid depending on implementation
        match event {
            ForkEvent::ForwardSync { .. } => {
                // Expected - no strong consensus means forward sync
            }
            ForkEvent::MinorityFork { peer_count_majority, .. } => {
                // Only ok if majority is weak
                assert!(peer_count_majority < 4, "Shouldn't detect minority with weak consensus");
            }
            ForkEvent::AheadOfNetwork { our_height, network_height } => {
                // Also valid if network height calculated from lower peers
                assert_eq!(our_height, 1500);
                assert!(network_height <= 2000, "Network height should be from peer consensus");
            }
            other => panic!("Unexpected event: {:?}", other),
        }
    }
}

// ============================================================================
// AHEAD OF NETWORK DETECTION TESTS
// ============================================================================

mod ahead_of_network_tests {
    use super::*;

    /// Test ahead of network is detected
    #[tokio::test]
    async fn test_ahead_of_network() {
        let detector = ForkDetector::new();

        // Peers at various heights, none very high
        detector.update_peer_height("peer1".to_string(), 1000).await;
        detector.update_peer_height("peer2".to_string(), 1001).await;
        detector.update_peer_height("peer3".to_string(), 1002).await;

        // We're way ahead (more than 10 blocks)
        let event = detector.detect_fork(1050).await.unwrap();

        match event {
            ForkEvent::AheadOfNetwork {
                our_height,
                network_height,
            } => {
                assert_eq!(our_height, 1050);
                // Network height should be around 1000-1002
                assert!(network_height < 1050);
            }
            other => panic!("Expected AheadOfNetwork, got {:?}", other),
        }
    }
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

mod edge_case_tests {
    use super::*;

    /// Test with insufficient peers
    #[tokio::test]
    async fn test_insufficient_peers() {
        let detector = ForkDetector::new();

        // Only 2 peers (need 3 minimum)
        detector.update_peer_height("peer1".to_string(), 1000).await;
        detector.update_peer_height("peer2".to_string(), 1000).await;

        let event = detector.detect_fork(2000).await.unwrap();

        // Should return forward sync with same height (can't detect fork)
        match event {
            ForkEvent::ForwardSync { our_height, network_height } => {
                assert_eq!(our_height, 2000);
                assert_eq!(network_height, 2000, "Should return our height when insufficient peers");
            }
            other => panic!("Expected ForwardSync with insufficient peers, got {:?}", other),
        }
    }

    /// Test with no peers
    #[tokio::test]
    async fn test_no_peers() {
        let detector = ForkDetector::new();

        let event = detector.detect_fork(5000).await.unwrap();

        match event {
            ForkEvent::ForwardSync { our_height, network_height } => {
                assert_eq!(our_height, 5000);
                assert_eq!(network_height, 5000);
            }
            other => panic!("Expected ForwardSync with no peers, got {:?}", other),
        }
    }

    /// Test height 0
    #[tokio::test]
    async fn test_height_zero() {
        let detector = ForkDetector::new();

        for i in 0..5 {
            detector.update_peer_height(format!("peer{}", i), 0).await;
        }

        let event = detector.detect_fork(0).await.unwrap();

        match event {
            ForkEvent::ForwardSync { our_height, network_height } => {
                assert_eq!(our_height, 0);
                assert_eq!(network_height, 0);
            }
            other => panic!("Expected ForwardSync at genesis, got {:?}", other),
        }
    }

    /// Test very large heights
    #[tokio::test]
    async fn test_large_heights() {
        let detector = ForkDetector::new();

        let large_height = u64::MAX / 2;

        for i in 0..5 {
            detector.update_peer_height(format!("peer{}", i), large_height).await;
        }

        let event = detector.detect_fork(large_height).await.unwrap();

        match event {
            ForkEvent::ForwardSync { our_height, network_height } => {
                assert_eq!(our_height, large_height);
                assert_eq!(network_height, large_height);
            }
            other => panic!("Expected ForwardSync at large height, got {:?}", other),
        }
    }
}

// ============================================================================
// SAFE REORG TESTS
// ============================================================================

mod safe_reorg_tests {
    use super::*;

    /// Test safe reorg threshold
    #[tokio::test]
    async fn test_safe_reorg_threshold() {
        let detector = ForkDetector::new();

        // Under limit
        assert!(detector.is_safe_auto_reorg(1));
        assert!(detector.is_safe_auto_reorg(100));
        assert!(detector.is_safe_auto_reorg(999));
        assert!(detector.is_safe_auto_reorg(1000));

        // Over limit
        assert!(!detector.is_safe_auto_reorg(1001));
        assert!(!detector.is_safe_auto_reorg(2000));
        assert!(!detector.is_safe_auto_reorg(10000));
        assert!(!detector.is_safe_auto_reorg(u64::MAX));
    }

    /// Test reorg depth calculation
    #[tokio::test]
    async fn test_reorg_depth_accuracy() {
        let detector = ForkDetector::new();

        for i in 0..5 {
            detector.update_peer_height(format!("peer{}", i), 1000).await;
        }

        let event = detector.detect_fork(2500).await.unwrap();

        match event {
            ForkEvent::BackwardReorg { reorg_depth, .. } => {
                assert_eq!(reorg_depth, 1500, "Reorg depth should be exact");
            }
            other => panic!("Expected BackwardReorg, got {:?}", other),
        }
    }
}

// ============================================================================
// CONSENSUS HEIGHT TESTS
// ============================================================================

mod consensus_height_tests {
    use super::*;

    /// Test consensus height calculation
    #[tokio::test]
    async fn test_consensus_height() {
        let detector = ForkDetector::new();

        // Clear majority at height 5000
        detector.update_peer_height("peer1".to_string(), 5000).await;
        detector.update_peer_height("peer2".to_string(), 5000).await;
        detector.update_peer_height("peer3".to_string(), 5000).await;
        detector.update_peer_height("peer4".to_string(), 4999).await;

        let consensus = detector.get_consensus_height().await;
        assert_eq!(consensus, Some(5000), "Consensus should be 5000");
    }

    /// Test consensus with tie
    #[tokio::test]
    async fn test_consensus_tie() {
        let detector = ForkDetector::new();

        detector.update_peer_height("peer1".to_string(), 1000).await;
        detector.update_peer_height("peer2".to_string(), 1000).await;
        detector.update_peer_height("peer3".to_string(), 2000).await;
        detector.update_peer_height("peer4".to_string(), 2000).await;

        let consensus = detector.get_consensus_height().await;
        // Should return one of the tied heights
        assert!(consensus == Some(1000) || consensus == Some(2000));
    }
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

mod performance_tests {
    use super::*;
    use std::time::Instant;

    /// Test fork detection performance with many peers
    #[tokio::test]
    async fn test_detection_performance() {
        let detector = ForkDetector::new();

        // Add 100 peers
        for i in 0..100 {
            detector.update_peer_height(format!("peer{}", i), 50000 + (i % 10) as u64).await;
        }

        let iterations = 1000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = detector.detect_fork(50005).await;
        }

        let elapsed = start.elapsed();
        let per_detection_us = elapsed.as_micros() / iterations as u128;

        println!("Fork detection: {} us per detection", per_detection_us);

        // Should be reasonably fast
        assert!(
            per_detection_us < 1000,
            "Fork detection should be under 1ms, got {} us",
            per_detection_us
        );
    }

    /// Test peer update performance
    #[tokio::test]
    async fn test_peer_update_performance() {
        let detector = ForkDetector::new();

        let iterations = 10_000;
        let start = Instant::now();

        for i in 0..iterations {
            detector.update_peer_height(format!("peer{}", i % 100), i as u64).await;
        }

        let elapsed = start.elapsed();
        let per_update_us = elapsed.as_micros() / iterations as u128;

        println!("Peer update: {} us per update", per_update_us);

        assert!(
            per_update_us < 100,
            "Peer update should be under 100us, got {} us",
            per_update_us
        );
    }
}
