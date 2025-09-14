//! # Phase 2 Integration Tests
//!
//! Comprehensive tests validating the complete quantum mixing system integration:
//! - End-to-end mixing transaction flow
//! - Multi-component integration validation
//! - Performance and reliability testing

use q_quantum_mixing::*;
use std::sync::Arc;
use tokio;
use uuid::Uuid;

/// Test complete mixing flow from pool to network broadcast
#[tokio::test]
async fn test_complete_mixing_flow() {
    // Initialize quantum mixing system
    let config = QuantumMixingConfig {
        min_participants: 2,
        max_participants: 5,
        mixing_fee: 10_000,
        compliance_enabled: true,
        decoy_enabled: false, // Disable for simpler test
        quantum_enhanced: true,
    };

    let mixing_service = QuantumMixingService::new(config.clone()).await.unwrap();

    // Create test participants
    let mut participants = Vec::new();
    for i in 0..3 {
        let input = MixingInput {
            amount: (i + 1) * 1_000_000_000, // 1, 2, 3 ORB
            sender_key: [i as u8; 32],
            recipient_address: [(i + 10) as u8; 32],
            commitment: [(i + 20) as u8; 32],
        };

        let participant_id = mixing_service.add_participant(input).await.unwrap();
        participants.push(participant_id);
    }

    // Wait for pool to be ready
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Execute complete mixing round
    let mixing_result = mixing_service.execute_mixing().await.unwrap();
    
    // Verify results
    assert_eq!(mixing_result.participant_count, 3);
    assert_eq!(mixing_result.outputs.len(), 3);
    assert!(!mixing_result.round_id.is_nil());
    assert!(mixing_result.mixing_duration.as_millis() > 0);

    println!("✅ Complete mixing flow test passed!");
}

/// Test compliance integration with mixing pool
#[tokio::test]
async fn test_compliance_integration() {
    let mut config = QuantumMixingConfig::default();
    config.compliance_enabled = true;
    config.min_participants = 1;
    config.max_participants = 2;

    let mixing_service = QuantumMixingService::new(config).await.unwrap();

    // Test approved transaction
    let approved_input = MixingInput {
        amount: 500_000_000, // 0.5 ORB - should be approved
        sender_key: [1u8; 32],
        recipient_address: [2u8; 32],
        commitment: [3u8; 32],
    };

    let participant_id = mixing_service.add_participant(approved_input).await.unwrap();
    assert!(!participant_id.is_nil());

    println!("✅ Compliance integration test passed!");
}

/// Test network manager functionality
#[tokio::test]
async fn test_network_integration() {
    let network_config = NetworkConfig::default();
    let network_manager = MixingNetworkManager::new(network_config).await.unwrap();

    // Test peer connections
    let peer_address = "127.0.0.1:8080".parse().unwrap();
    let peer_key = [42u8; 32];
    
    let peer_id = network_manager.connect_to_peer(peer_address, peer_key).await.unwrap();
    assert!(!peer_id.is_nil());

    // Test mixing round proposal
    let participants = vec![Uuid::new_v4(), Uuid::new_v4()];
    let round_id = network_manager.propose_mixing_round(participants).await.unwrap();
    assert!(!round_id.is_nil());

    // Test consensus voting
    network_manager.vote_on_mixing_round(round_id, ConsensusVote::Approve).await.unwrap();

    println!("✅ Network integration test passed!");
}

/// Test system under load with multiple concurrent operations
#[tokio::test]
async fn test_concurrent_mixing_operations() {
    let config = QuantumMixingConfig {
        min_participants: 2,
        max_participants: 10,
        mixing_fee: 5_000,
        compliance_enabled: false, // Disable for performance test
        decoy_enabled: false,
        quantum_enhanced: true,
    };

    let mixing_service = Arc::new(QuantumMixingService::new(config).await.unwrap());

    // Spawn multiple concurrent participant additions
    let mut handles = Vec::new();
    
    for i in 0..8 {
        let service = Arc::clone(&mixing_service);
        let handle = tokio::spawn(async move {
            let input = MixingInput {
                amount: (i + 1) * 100_000_000, // Variable amounts
                sender_key: [i as u8; 32],
                recipient_address: [(i + 50) as u8; 32],
                commitment: [(i + 100) as u8; 32],
            };

            service.add_participant(input).await
        });
        handles.push(handle);
    }

    // Wait for all participants to be added
    let mut participant_ids = Vec::new();
    for handle in handles {
        let participant_id = handle.await.unwrap().unwrap();
        participant_ids.push(participant_id);
    }

    assert_eq!(participant_ids.len(), 8);
    println!("✅ Concurrent operations test passed with {} participants!", participant_ids.len());
}

/// Test error handling and recovery
#[tokio::test]
async fn test_error_handling() {
    let config = QuantumMixingConfig {
        min_participants: 5, // High requirement for test
        max_participants: 10,
        mixing_fee: 1_000,
        compliance_enabled: true,
        decoy_enabled: false,
        quantum_enhanced: true,
    };

    let mixing_service = QuantumMixingService::new(config).await.unwrap();

    // Try to execute mixing with insufficient participants
    let result = mixing_service.execute_mixing().await;
    assert!(result.is_err()); // Should fail due to insufficient participants

    println!("✅ Error handling test passed!");
}

/// Test quantum entropy integration
#[tokio::test]
async fn test_quantum_entropy_integration() {
    // Test that quantum entropy pool is properly integrated
    let entropy_pool = QuantumEntropyPool::new().await.unwrap();
    
    // Generate random bytes
    let mut buffer1 = [0u8; 32];
    let mut buffer2 = [0u8; 32];
    
    entropy_pool.fill_bytes(&mut buffer1).await.unwrap();
    entropy_pool.fill_bytes(&mut buffer2).await.unwrap();
    
    // Verify randomness (very unlikely to be identical)
    assert_ne!(buffer1, buffer2);
    
    println!("✅ Quantum entropy integration test passed!");
}

/// Performance benchmark test
#[tokio::test]
async fn test_mixing_performance_benchmark() {
    let config = QuantumMixingConfig {
        min_participants: 3,
        max_participants: 5,
        mixing_fee: 1_000,
        compliance_enabled: false, // Disable for pure performance test
        decoy_enabled: false,
        quantum_enhanced: true,
    };

    let mixing_service = QuantumMixingService::new(config).await.unwrap();
    
    let start_time = std::time::Instant::now();

    // Add participants
    for i in 0..3 {
        let input = MixingInput {
            amount: 1_000_000_000, // 1 ORB each
            sender_key: [i as u8; 32],
            recipient_address: [(i + 10) as u8; 32],
            commitment: [(i + 20) as u8; 32],
        };
        mixing_service.add_participant(input).await.unwrap();
    }

    // Execute mixing
    let mixing_result = mixing_service.execute_mixing().await.unwrap();
    
    let total_time = start_time.elapsed();
    
    // Performance assertions (adjust based on target performance)
    assert!(total_time < std::time::Duration::from_secs(5)); // Should complete within 5 seconds
    assert!(mixing_result.mixing_duration < std::time::Duration::from_secs(3)); // Engine should be fast
    
    println!("✅ Performance benchmark passed!");
    println!("   Total time: {:?}", total_time);
    println!("   Mixing engine time: {:?}", mixing_result.mixing_duration);
    println!("   Participants: {}", mixing_result.participant_count);
}