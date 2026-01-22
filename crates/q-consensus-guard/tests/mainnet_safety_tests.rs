//! Comprehensive Mainnet Safety Tests
//!
//! v3.3.9-beta: Tests for all mainnet safety features
//!
//! These tests verify that the mainnet safety mechanisms work correctly:
//! - Upgrade Gate: Height-gated validation rules
//! - Consensus Guard: Startup verification
//! - Golden Blocks: Known block validation
//!
//! Run with: cargo test --package q-consensus-guard --test mainnet_safety_tests

use q_consensus_guard::{
    ConsensusGuard, GuardConfig,
    Upgrade, UpgradeGate, UpgradeConfig, MAINNET_UPGRADES,
    GoldenBlock, GoldenBlockRegistry, BlockFingerprint,
};

// ============================================================================
// UPGRADE GATE TESTS
// ============================================================================

mod upgrade_gate_tests {
    use super::*;

    /// Test upgrade gate creation for testnet
    #[test]
    fn test_upgrade_gate_creation_testnet() {
        let gate = UpgradeGate::new(false); // false = testnet

        // Should have activation height for PQ signatures
        let pq_height = gate.activation_height(Upgrade::PostQuantumSignatures);
        assert!(pq_height.is_some(), "Testnet should have PQ signatures defined");
        println!("PQ signatures activation height (testnet): {:?}", pq_height);
    }

    /// Test upgrade gate creation for mainnet
    #[test]
    fn test_upgrade_gate_creation_mainnet() {
        let gate = UpgradeGate::new(true); // true = mainnet

        // Should have activation height for genesis
        let genesis_height = gate.activation_height(Upgrade::Genesis);
        assert_eq!(genesis_height, Some(0), "Genesis should activate at height 0");
    }

    /// Test that upgrades are NOT active before their activation height
    #[test]
    fn test_upgrade_not_active_before_height() {
        let gate = UpgradeGate::new(false); // testnet

        // Get PQ activation height
        let pq_height = gate.activation_height(Upgrade::PostQuantumSignatures).unwrap_or(100000);

        // Should not be active before activation
        assert!(
            !gate.is_active(Upgrade::PostQuantumSignatures, 0),
            "PQ signatures should NOT be active at height 0"
        );

        if pq_height > 1 {
            assert!(
                !gate.is_active(Upgrade::PostQuantumSignatures, pq_height - 1),
                "PQ signatures should NOT be active one block before activation"
            );
        }
    }

    /// Test that upgrades ARE active at and after activation height
    #[test]
    fn test_upgrade_active_at_and_after_height() {
        let gate = UpgradeGate::new(false); // testnet

        // Get PQ activation height
        let pq_height = gate.activation_height(Upgrade::PostQuantumSignatures).unwrap_or(100000);

        // Should be active at activation height
        assert!(
            gate.is_active(Upgrade::PostQuantumSignatures, pq_height),
            "PQ signatures SHOULD be active at activation height {}", pq_height
        );

        // Should be active after activation height
        assert!(
            gate.is_active(Upgrade::PostQuantumSignatures, pq_height + 1),
            "PQ signatures SHOULD be active after activation height"
        );

        assert!(
            gate.is_active(Upgrade::PostQuantumSignatures, pq_height + 1000000),
            "PQ signatures SHOULD be active well after activation"
        );
    }

    /// Test Genesis upgrade is always active
    #[test]
    fn test_genesis_always_active() {
        let gate = UpgradeGate::new(false);

        assert!(gate.is_active(Upgrade::Genesis, 0), "Genesis should be active at height 0");
        assert!(gate.is_active(Upgrade::Genesis, 1000), "Genesis should be active at any height");
        assert!(gate.is_active(Upgrade::Genesis, u64::MAX), "Genesis should be active at max height");
    }

    /// Test that historical blocks validate with OLD rules
    #[test]
    fn test_historical_blocks_use_old_rules() {
        let gate = UpgradeGate::new(false);
        let block_height = 50; // Very early block

        let pq_required = gate.is_active(Upgrade::PostQuantumSignatures, block_height);

        assert!(
            !pq_required,
            "Historical blocks at height {} should NOT require PQ signatures", block_height
        );

        // This is how validation code should work:
        // if pq_required {
        //     verify_dilithium(block)?;  // New rule
        // } else {
        //     verify_ed25519(block)?;    // Old rule - for historical blocks
        // }
    }

    /// Test that future blocks validate with NEW rules
    #[test]
    fn test_future_blocks_use_new_rules() {
        let gate = UpgradeGate::new(false);
        let pq_height = gate.activation_height(Upgrade::PostQuantumSignatures).unwrap_or(100000);
        let block_height = pq_height + 50000; // Well after activation

        let pq_required = gate.is_active(Upgrade::PostQuantumSignatures, block_height);

        assert!(
            pq_required,
            "Future blocks at height {} SHOULD require PQ signatures after activation at {}",
            block_height, pq_height
        );
    }

    /// Test pending upgrades list
    #[test]
    fn test_pending_upgrades() {
        let gate = UpgradeGate::new(false);

        // At height 0, there should be pending upgrades
        let pending = gate.pending_upgrades(0);
        println!("Pending upgrades at height 0: {:?}", pending);

        // At very high height, there should be no pending upgrades (or fewer)
        let pending_high = gate.pending_upgrades(u64::MAX - 1);
        assert!(
            pending_high.len() <= pending.len(),
            "Fewer upgrades should be pending at higher heights"
        );
    }

    /// Test that upgrade checks are fast (no database lookups)
    #[test]
    fn test_upgrade_check_performance() {
        use std::time::Instant;

        let gate = UpgradeGate::new(false);
        let iterations = 100_000;
        let start = Instant::now();

        for height in 0..iterations {
            let _ = gate.is_active(Upgrade::PostQuantumSignatures, height);
        }

        let elapsed = start.elapsed();
        let per_check_ns = elapsed.as_nanos() / iterations as u128;

        println!("Upgrade check performance: {} ns per check", per_check_ns);

        // Should be very fast - under 1 microsecond per check
        assert!(
            per_check_ns < 1000,
            "Upgrade checks should be sub-microsecond, got {} ns",
            per_check_ns
        );
    }
}

// ============================================================================
// CONSENSUS GUARD TESTS
// ============================================================================

mod consensus_guard_tests {
    use super::*;

    /// Test consensus guard creation with default config
    #[test]
    fn test_guard_creation_default() {
        let config = GuardConfig::default();
        let guard = ConsensusGuard::new(config);

        assert!(guard.is_ok(), "Guard should create with default config");
    }

    /// Test consensus guard creation with testnet config
    #[test]
    fn test_guard_creation_testnet() {
        let config = GuardConfig::testnet();
        let guard = ConsensusGuard::new(config);

        assert!(guard.is_ok(), "Guard should create with testnet config");
    }

    /// Test consensus guard creation with mainnet config
    #[test]
    fn test_guard_creation_mainnet() {
        let config = GuardConfig::mainnet();
        let guard = ConsensusGuard::new(config);

        assert!(guard.is_ok(), "Guard should create with mainnet config");
    }

    /// Test that guard is functional after creation
    #[test]
    fn test_guard_functional() {
        let config = GuardConfig::testnet();
        let guard = ConsensusGuard::new(config).unwrap();

        // Guard should be created and functional
        println!("Consensus guard created successfully");
    }
}

// ============================================================================
// GOLDEN BLOCK TESTS
// ============================================================================

mod golden_block_tests {
    use super::*;

    /// Test golden block registry creation for testnet
    #[test]
    fn test_registry_creation_testnet() {
        let registry = GoldenBlockRegistry::new(false);

        // Registry should exist
        println!("Created testnet golden block registry");
    }

    /// Test golden block registry creation for mainnet
    #[test]
    fn test_registry_creation_mainnet() {
        let registry = GoldenBlockRegistry::new(true);

        // Registry should exist
        println!("Created mainnet golden block registry");
    }

    /// Test block fingerprint creation
    #[test]
    fn test_block_fingerprint_creation() {
        let block_bytes = b"test block data";
        let validation_rules = "genesis_rules_v1";
        let post_state_root = [1u8; 32];
        let active_upgrades = vec![0u32]; // Genesis only

        let fingerprint = BlockFingerprint::new(
            block_bytes,
            validation_rules,
            post_state_root,
            active_upgrades.clone(),
        );

        assert_ne!(fingerprint.block_hash, [0u8; 32], "Block hash should not be zero");
        assert_ne!(fingerprint.validation_rules_hash, [0u8; 32], "Rules hash should not be zero");
        assert_eq!(fingerprint.post_state_root, post_state_root);
        assert_eq!(fingerprint.active_upgrades, active_upgrades);
    }

    /// Test fingerprint digest is deterministic
    #[test]
    fn test_fingerprint_digest_deterministic() {
        let fingerprint1 = BlockFingerprint::new(
            b"test data",
            "rules",
            [0u8; 32],
            vec![0],
        );

        let fingerprint2 = BlockFingerprint::new(
            b"test data",
            "rules",
            [0u8; 32],
            vec![0],
        );

        assert_eq!(fingerprint1.digest(), fingerprint2.digest(), "Same inputs should produce same digest");
    }

    /// Test different inputs produce different digests
    #[test]
    fn test_fingerprint_digest_unique() {
        let fp1 = BlockFingerprint::new(b"data1", "rules", [0u8; 32], vec![0]);
        let fp2 = BlockFingerprint::new(b"data2", "rules", [0u8; 32], vec![0]);

        assert_ne!(fp1.digest(), fp2.digest(), "Different inputs should produce different digests");
    }
}

// ============================================================================
// MAINNET UPGRADES CONFIGURATION TESTS
// ============================================================================

mod upgrade_config_tests {
    use super::*;

    /// Test mainnet upgrades are properly defined
    #[test]
    fn test_mainnet_upgrades_defined() {
        assert!(MAINNET_UPGRADES.contains_key(&Upgrade::Genesis), "Genesis must be in mainnet upgrades");
    }

    /// Test genesis upgrade is at height 0
    #[test]
    fn test_genesis_at_height_zero() {
        let genesis_config = MAINNET_UPGRADES.get(&Upgrade::Genesis);
        assert!(genesis_config.is_some(), "Genesis config must exist");

        let config = genesis_config.unwrap();
        assert_eq!(config.activation_height, 0, "Genesis must activate at height 0");
        assert!(config.mandatory, "Genesis must be mandatory");
    }

    /// Test upgrade config has required fields
    #[test]
    fn test_upgrade_config_fields() {
        for (upgrade, config) in MAINNET_UPGRADES.iter() {
            println!("{:?}: height={}, mandatory={}, min_version={}",
                     upgrade, config.activation_height, config.mandatory, config.min_version);

            // All upgrades should have a description
            assert!(!config.description.is_empty(), "{:?} should have a description", upgrade);

            // Min version should be a valid semver-like string
            assert!(!config.min_version.is_empty(), "{:?} should have a min_version", upgrade);
        }
    }
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

mod integration_tests {
    use super::*;

    /// Test complete mainnet safety flow
    #[test]
    fn test_complete_safety_flow() {
        // 1. Create consensus guard
        let config = GuardConfig::testnet();
        let guard = ConsensusGuard::new(config);
        assert!(guard.is_ok(), "Guard creation should succeed");

        // 2. Create upgrade gate
        let gate = UpgradeGate::new(false); // testnet

        // 3. Check upgrade gate for different heights
        let historical_height = 100;
        let pq_height = gate.activation_height(Upgrade::PostQuantumSignatures).unwrap_or(100000);
        let future_height = pq_height + 100000;

        // Historical blocks don't require new rules
        assert!(
            !gate.is_active(Upgrade::PostQuantumSignatures, historical_height),
            "Historical height should use old rules"
        );

        // Future blocks require new rules
        assert!(
            gate.is_active(Upgrade::PostQuantumSignatures, future_height),
            "Future height should use new rules"
        );

        // 4. Golden block registry should be functional
        let registry = GoldenBlockRegistry::new(false);
        println!("Golden block registry created");
    }

    /// Test upgrade gate doesn't break on edge cases
    #[test]
    fn test_upgrade_gate_edge_cases() {
        let gate = UpgradeGate::new(false);

        // Height 0
        assert!(gate.is_active(Upgrade::Genesis, 0), "Genesis should be active at 0");

        // Max u64
        assert!(gate.is_active(Upgrade::Genesis, u64::MAX), "Genesis should be active at max");

        // Various upgrades at height 0
        let _ = gate.is_active(Upgrade::PostQuantumSignatures, 0);
        let _ = gate.is_active(Upgrade::EnhancedBlockValidation, 0);
        let _ = gate.is_active(Upgrade::TransactionV2, 0);
    }

    /// Test that safety features work together without conflicts
    #[test]
    fn test_no_safety_feature_conflicts() {
        // Create all components
        let guard_config = GuardConfig::testnet();
        let guard = ConsensusGuard::new(guard_config);
        let gate = UpgradeGate::new(false);
        let registry = GoldenBlockRegistry::new(false);

        // All should coexist
        assert!(guard.is_ok());
        assert!(gate.activation_height(Upgrade::Genesis).is_some());

        // Operations shouldn't affect each other
        let _ = gate.is_active(Upgrade::PostQuantumSignatures, 1000);
        let _ = gate.pending_upgrades(0);
    }
}

// ============================================================================
// STRESS TESTS
// ============================================================================

mod stress_tests {
    use super::*;
    use std::thread;

    /// Test concurrent upgrade checks are safe
    #[test]
    fn test_concurrent_upgrade_checks() {
        let handles: Vec<_> = (0..10)
            .map(|_| {
                thread::spawn(|| {
                    let gate = UpgradeGate::new(false);
                    for height in 0..10000 {
                        let _ = gate.is_active(Upgrade::PostQuantumSignatures, height);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }
    }

    /// Test guard creation under load
    #[test]
    fn test_guard_creation_under_load() {
        for _ in 0..100 {
            let config = GuardConfig::testnet();
            let guard = ConsensusGuard::new(config);
            assert!(guard.is_ok());
        }
    }

    /// Test upgrade gate creation under load
    #[test]
    fn test_gate_creation_under_load() {
        for i in 0..100 {
            let gate = UpgradeGate::new(i % 2 == 0); // Alternate mainnet/testnet
            let _ = gate.is_active(Upgrade::Genesis, 0);
        }
    }
}
