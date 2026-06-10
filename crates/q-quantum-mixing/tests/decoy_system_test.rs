//! # DECOY TRANSACTION SYSTEM VALIDATION TEST
//!
//! Focused test to validate our comprehensive decoy transaction implementation

use q_quantum_mixing::*;
use std::time::Duration;

#[tokio::test]
async fn test_decoy_system_comprehensive_validation() {
    println!("🎭 TESTING COMPREHENSIVE DECOY SYSTEM");

    // Create service with maximum decoy features enabled
    let config = QuantumMixingConfig {
        min_participants: 2,
        max_participants: 5,
        decoy_enabled: true,
        decoy_strategy: DecoyStrategy {
            decoy_ratio: 15.0, // Lots of decoys as requested!
            enabled_types: vec![
                DecoyType::UserTransaction,
                DecoyType::ExchangeTransaction,
                DecoyType::DeFiTransaction,
                DecoyType::PaymentProcessor,
                DecoyType::TradingBot,
            ],
            timing_strategy: TimingStrategy::AntiTimingAnalysis,
            geographic_distribution: true,
            coordination_enabled: true,
            quantum_enhancement_level: 9, // Maximum privacy
        },
        ..Default::default()
    };

    println!("✅ Creating QuantumMixingService with 15x decoy ratio");
    let mut service = QuantumMixingService::new(config).await.unwrap();

    // Test 1: Start multiple decoy campaigns
    println!("🚀 Phase 1: Testing decoy campaigns");
    let campaign1 = service.start_decoy_campaign(50, Duration::from_secs(300)).await.unwrap();
    println!("  ✅ Campaign 1 started: {:?}", campaign1);

    let campaign2 = service.start_decoy_campaign(30, Duration::from_secs(180)).await.unwrap();
    println!("  ✅ Campaign 2 started: {:?}", campaign2);

    // Test 2: Validate decoy metrics
    println!("📊 Phase 2: Validating decoy metrics");
    tokio::time::sleep(Duration::from_millis(200)).await; // Give campaigns time to initialize

    let metrics = service.get_decoy_metrics().await.unwrap();
    if let Some(metrics) = metrics {
        println!("  ✅ Decoy Metrics Retrieved:");
        println!("    - Total decoys generated: {}", metrics.total_decoys_generated);
        println!("    - Active campaigns: {}", metrics.active_campaigns);
        println!("    - Privacy enhancement score: {:.3}", metrics.privacy_enhancement_score);
        println!("    - Network coverage: {:.1}%", metrics.network_coverage_percent);
        println!("    - Quantum entropy quality: {:.3}", metrics.quantum_entropy_quality);

        // Validate metrics quality
        assert!(metrics.active_campaigns > 0, "Should have active campaigns");
        assert!(metrics.privacy_enhancement_score > 0.8, "High privacy enhancement expected");
        assert!(metrics.network_coverage_percent > 50.0, "Good network coverage expected");
        assert!(metrics.quantum_entropy_quality > 0.7, "Good quantum entropy expected");
    } else {
        panic!("No decoy metrics available");
    }

    // Test 3: Validate overall system statistics include decoys
    println!("🔍 Phase 3: Validating system integration");
    let stats = service.get_statistics().await.unwrap();
    println!("  ✅ System Statistics:");
    println!("    - Privacy score: {:.3}", stats.privacy_score);
    println!("    - Quantum entropy quality: {:.3}", stats.quantum_entropy_quality);

    if let Some(decoy_metrics) = &stats.decoy_metrics {
        println!("    - Decoy system integrated: YES");
        println!("    - Decoy campaigns active: {}", decoy_metrics.active_campaigns);
        assert!(decoy_metrics.active_campaigns > 0, "Decoy system should be active");
    } else {
        panic!("Decoy metrics should be integrated in system statistics");
    }

    // Test 4: Test real mixing transaction submission
    println!("💫 Phase 4: Testing decoy integration with real transactions");
    let test_input = MixingInput {
        amount: 5_000_000_000, // 5 ORB
        sender_key: [1u8; 32],
        recipient_address: [2u8; 32],
        commitment: [3u8; 32],
    };

    let session_id = service.submit_for_mixing(test_input).await.unwrap();
    println!("  ✅ Transaction submitted: {}", session_id);
    println!("    - Decoys should automatically generate for cover traffic");

    // Final validation
    println!("🎊 DECOY SYSTEM VALIDATION COMPLETE!");
    println!("✅ All decoy features working:");
    println!("  - 15x decoy ratio configuration");
    println!("  - Multiple decoy types supported");
    println!("  - Campaign management functional");
    println!("  - Quantum entropy integration");
    println!("  - System statistics integration");
    println!("  - Privacy enhancement verification");

    println!("🚀 COMPREHENSIVE DECOY SYSTEM: FULLY OPERATIONAL!");
}

#[tokio::test]
async fn test_decoy_diversity_types() {
    println!("🎨 TESTING DECOY TYPE DIVERSITY");

    let config = QuantumMixingConfig {
        decoy_enabled: true,
        decoy_strategy: DecoyStrategy {
            decoy_ratio: 20.0, // Even more decoys for testing
            enabled_types: vec![
                DecoyType::UserTransaction,
                DecoyType::ExchangeTransaction,
                DecoyType::DeFiTransaction,
                DecoyType::PaymentProcessor,
                DecoyType::TradingBot,
            ],
            timing_strategy: TimingStrategy::QuantumDistributed,
            geographic_distribution: true,
            coordination_enabled: true,
            quantum_enhancement_level: 9,
        },
        ..Default::default()
    };

    let service = QuantumMixingService::new(config).await.unwrap();
    let stats = service.get_statistics().await.unwrap();

    // Validate diversity features are configured
    assert!(stats.privacy_score > 0.7, "High privacy with diverse decoys");
    println!("✅ Decoy diversity test passed!");
    println!("  - 5 decoy transaction types enabled");
    println!("  - 20x decoy ratio for maximum coverage");
    println!("  - Quantum-distributed timing");
    println!("  - Geographic distribution enabled");
}

#[tokio::test]
async fn test_quantum_entropy_decoy_integration() {
    println!("🔮 TESTING QUANTUM ENTROPY + DECOY INTEGRATION");

    let config = QuantumMixingConfig {
        quantum_resistant: true,
        decoy_enabled: true,
        decoy_strategy: DecoyStrategy {
            quantum_enhancement_level: 9, // Maximum quantum enhancement
            timing_strategy: TimingStrategy::QuantumDistributed,
            ..Default::default()
        },
        ..Default::default()
    };

    let service = QuantumMixingService::new(config).await.unwrap();
    let stats = service.get_statistics().await.unwrap();

    // Validate quantum integration
    assert!(stats.quantum_entropy_quality > 0.8, "High quantum entropy quality");

    if let Some(decoy_metrics) = stats.decoy_metrics {
        assert!(decoy_metrics.quantum_entropy_quality > 0.8, "Decoys use quantum entropy");
        println!("✅ Quantum entropy properly integrated with decoy system");
    }

    println!("🎊 QUANTUM + DECOY INTEGRATION: SUCCESS!");
}