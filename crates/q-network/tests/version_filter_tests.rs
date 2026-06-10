//! Comprehensive Version Filtering Tests
//!
//! v3.3.9-beta: Tests for peer version filtering and capability announcement
//!
//! These tests verify:
//! - Peer version filtering works correctly
//! - Network ID mismatches are rejected
//! - Legacy peers are handled gracefully
//! - Capability matching works
//!
//! Run with: cargo test --package q-network --test version_filter_tests

use q_network::{
    PeerHeightWithProof,
    SOFTWARE_VERSION, PROTOCOL_VERSION, MIN_PROTOCOL_VERSION,
    get_upgrade_capabilities, create_peer_height_announcement,
    VersionFilterResult, filter_peer_version, should_sync_from_peer,
};

// ============================================================================
// PEER ANNOUNCEMENT CREATION TESTS
// ============================================================================

mod announcement_creation_tests {
    use super::*;

    /// Test creating a peer height announcement includes version info
    #[test]
    fn test_announcement_includes_version() {
        let announcement = create_peer_height_announcement(
            "peer123",
            50000,
            "testnet-phase19",
        );

        assert_eq!(announcement.peer_id, "peer123");
        assert_eq!(announcement.highest_block, 50000);
        assert!(announcement.software_version.is_some(), "Should include software version");
        assert!(announcement.protocol_version.is_some(), "Should include protocol version");
        assert!(!announcement.upgrade_capabilities.is_empty(), "Should include capabilities");
        assert!(announcement.network_id.is_some(), "Should include network ID");
    }

    /// Test announcement has correct version values
    #[test]
    fn test_announcement_version_values() {
        let announcement = create_peer_height_announcement(
            "peer123",
            50000,
            "testnet-phase19",
        );

        assert_eq!(
            announcement.software_version.as_deref(),
            Some(SOFTWARE_VERSION),
            "Software version should match crate version"
        );
        assert_eq!(
            announcement.protocol_version,
            Some(PROTOCOL_VERSION),
            "Protocol version should match constant"
        );
    }

    /// Test announcement includes all required capabilities
    #[test]
    fn test_announcement_capabilities() {
        let announcement = create_peer_height_announcement(
            "peer123",
            50000,
            "testnet-phase19",
        );

        let caps = &announcement.upgrade_capabilities;

        assert!(caps.contains(&"upgrade-gate-v1".to_string()), "Should have upgrade-gate-v1");
        assert!(caps.contains(&"consensus-guard-v1".to_string()), "Should have consensus-guard-v1");
        assert!(caps.contains(&"pq-signatures-ready".to_string()), "Should have pq-signatures-ready");
        assert!(caps.contains(&"sync-down-protection".to_string()), "Should have sync-down-protection");
        assert!(caps.contains(&"version-filter-v1".to_string()), "Should have version-filter-v1");
    }

    /// Test get_upgrade_capabilities returns expected list
    #[test]
    fn test_get_upgrade_capabilities() {
        let caps = get_upgrade_capabilities();

        assert!(!caps.is_empty(), "Should have capabilities");
        assert!(caps.len() >= 5, "Should have at least 5 capabilities");
    }
}

// ============================================================================
// VERSION FILTER TESTS
// ============================================================================

mod version_filter_tests {
    use super::*;

    /// Test compatible peer passes filter
    #[test]
    fn test_compatible_peer_passes() {
        let announcement = create_peer_height_announcement(
            "peer123",
            50000,
            "testnet-phase19",
        );

        let result = filter_peer_version(&announcement, "testnet-phase19");

        match result {
            VersionFilterResult::Compatible { peer_version, peer_protocol, common_capabilities } => {
                assert_eq!(peer_version, SOFTWARE_VERSION);
                assert_eq!(peer_protocol, PROTOCOL_VERSION);
                assert!(!common_capabilities.is_empty());
            }
            other => panic!("Expected Compatible, got {:?}", other),
        }
    }

    /// Test wrong network is rejected
    #[test]
    fn test_wrong_network_rejected() {
        let announcement = create_peer_height_announcement(
            "peer123",
            50000,
            "mainnet", // Different network!
        );

        let result = filter_peer_version(&announcement, "testnet-phase19");

        match result {
            VersionFilterResult::WrongNetwork { expected, actual } => {
                assert_eq!(expected, "testnet-phase19");
                assert_eq!(actual, "mainnet");
            }
            other => panic!("Expected WrongNetwork, got {:?}", other),
        }
    }

    /// Test legacy peer (no version info) is detected
    #[test]
    fn test_legacy_peer_detected() {
        let announcement = PeerHeightWithProof {
            peer_id: "legacy-peer".to_string(),
            highest_block: 50000,
            height_proof: None,
            blockchain_merkle_root: None,
            timestamp: 1234567890,
            software_version: None,  // No version!
            protocol_version: None,  // No protocol!
            upgrade_capabilities: vec![],
            network_id: None,
        };

        let result = filter_peer_version(&announcement, "testnet-phase19");

        match result {
            VersionFilterResult::LegacyPeer { reason } => {
                assert!(reason.contains("No version"), "Should mention missing version");
            }
            other => panic!("Expected LegacyPeer, got {:?}", other),
        }
    }

    /// Test outdated protocol version is rejected
    #[test]
    fn test_outdated_protocol_rejected() {
        let announcement = PeerHeightWithProof {
            peer_id: "old-peer".to_string(),
            highest_block: 50000,
            height_proof: None,
            blockchain_merkle_root: None,
            timestamp: 1234567890,
            software_version: Some("1.0.0".to_string()),
            protocol_version: Some(0), // Below MIN_PROTOCOL_VERSION!
            upgrade_capabilities: vec![],
            network_id: Some("testnet-phase19".to_string()),
        };

        let result = filter_peer_version(&announcement, "testnet-phase19");

        match result {
            VersionFilterResult::Incompatible { reason } => {
                assert!(reason.contains("Protocol version"), "Should mention protocol version");
            }
            other => panic!("Expected Incompatible, got {:?}", other),
        }
    }

    /// Test peer with matching network but no capabilities still passes
    #[test]
    fn test_peer_with_no_capabilities_passes() {
        let announcement = PeerHeightWithProof {
            peer_id: "minimal-peer".to_string(),
            highest_block: 50000,
            height_proof: None,
            blockchain_merkle_root: None,
            timestamp: 1234567890,
            software_version: Some("2.0.0".to_string()),
            protocol_version: Some(MIN_PROTOCOL_VERSION),
            upgrade_capabilities: vec![], // No capabilities
            network_id: Some("testnet-phase19".to_string()),
        };

        let result = filter_peer_version(&announcement, "testnet-phase19");

        match result {
            VersionFilterResult::Compatible { common_capabilities, .. } => {
                assert!(common_capabilities.is_empty(), "Should have no common capabilities");
            }
            other => panic!("Expected Compatible, got {:?}", other),
        }
    }
}

// ============================================================================
// SHOULD_SYNC_FROM_PEER TESTS
// ============================================================================

mod should_sync_tests {
    use super::*;

    /// Test compatible peer allows sync
    #[test]
    fn test_sync_allowed_for_compatible_peer() {
        let announcement = create_peer_height_announcement(
            "peer123",
            50000,
            "testnet-phase19",
        );

        assert!(
            should_sync_from_peer(&announcement, "testnet-phase19", false),
            "Should sync from compatible peer"
        );
        assert!(
            should_sync_from_peer(&announcement, "testnet-phase19", true),
            "Should sync from compatible peer even in strict mode"
        );
    }

    /// Test wrong network blocks sync
    #[test]
    fn test_sync_blocked_for_wrong_network() {
        let announcement = create_peer_height_announcement(
            "peer123",
            50000,
            "mainnet", // Different network!
        );

        assert!(
            !should_sync_from_peer(&announcement, "testnet-phase19", false),
            "Should NOT sync from wrong network"
        );
        assert!(
            !should_sync_from_peer(&announcement, "testnet-phase19", true),
            "Should NOT sync from wrong network even in non-strict mode"
        );
    }

    /// Test legacy peer allowed in non-strict mode
    #[test]
    fn test_legacy_peer_allowed_non_strict() {
        let announcement = PeerHeightWithProof {
            peer_id: "legacy-peer".to_string(),
            highest_block: 50000,
            height_proof: None,
            blockchain_merkle_root: None,
            timestamp: 1234567890,
            software_version: None,
            protocol_version: None,
            upgrade_capabilities: vec![],
            network_id: None,
        };

        assert!(
            should_sync_from_peer(&announcement, "testnet-phase19", false),
            "Legacy peer should be allowed in non-strict mode"
        );
    }

    /// Test legacy peer blocked in strict mode
    #[test]
    fn test_legacy_peer_blocked_strict() {
        let announcement = PeerHeightWithProof {
            peer_id: "legacy-peer".to_string(),
            highest_block: 50000,
            height_proof: None,
            blockchain_merkle_root: None,
            timestamp: 1234567890,
            software_version: None,
            protocol_version: None,
            upgrade_capabilities: vec![],
            network_id: None,
        };

        assert!(
            !should_sync_from_peer(&announcement, "testnet-phase19", true),
            "Legacy peer should be BLOCKED in strict mode"
        );
    }

    /// Test incompatible protocol blocks sync
    #[test]
    fn test_incompatible_protocol_blocks_sync() {
        let announcement = PeerHeightWithProof {
            peer_id: "old-peer".to_string(),
            highest_block: 50000,
            height_proof: None,
            blockchain_merkle_root: None,
            timestamp: 1234567890,
            software_version: Some("0.1.0".to_string()),
            protocol_version: Some(0), // Below minimum!
            upgrade_capabilities: vec![],
            network_id: Some("testnet-phase19".to_string()),
        };

        assert!(
            !should_sync_from_peer(&announcement, "testnet-phase19", false),
            "Incompatible protocol should block sync"
        );
        assert!(
            !should_sync_from_peer(&announcement, "testnet-phase19", true),
            "Incompatible protocol should block sync in strict mode"
        );
    }
}

// ============================================================================
// CAPABILITY MATCHING TESTS
// ============================================================================

mod capability_tests {
    use super::*;

    /// Test common capabilities are found
    #[test]
    fn test_common_capabilities_found() {
        let our_caps = get_upgrade_capabilities();

        let announcement = PeerHeightWithProof {
            peer_id: "peer".to_string(),
            highest_block: 50000,
            height_proof: None,
            blockchain_merkle_root: None,
            timestamp: 1234567890,
            software_version: Some("3.3.9".to_string()),
            protocol_version: Some(2),
            upgrade_capabilities: vec![
                "upgrade-gate-v1".to_string(),
                "pq-signatures-ready".to_string(),
                "unknown-capability".to_string(), // This won't match
            ],
            network_id: Some("testnet-phase19".to_string()),
        };

        let result = filter_peer_version(&announcement, "testnet-phase19");

        match result {
            VersionFilterResult::Compatible { common_capabilities, .. } => {
                assert!(common_capabilities.contains(&"upgrade-gate-v1".to_string()));
                assert!(common_capabilities.contains(&"pq-signatures-ready".to_string()));
                assert!(!common_capabilities.contains(&"unknown-capability".to_string()));
            }
            other => panic!("Expected Compatible, got {:?}", other),
        }
    }

    /// Test peer with all our capabilities
    #[test]
    fn test_full_capability_match() {
        let our_caps = get_upgrade_capabilities();

        let announcement = PeerHeightWithProof {
            peer_id: "full-featured-peer".to_string(),
            highest_block: 50000,
            height_proof: None,
            blockchain_merkle_root: None,
            timestamp: 1234567890,
            software_version: Some("3.3.9".to_string()),
            protocol_version: Some(2),
            upgrade_capabilities: our_caps.clone(),
            network_id: Some("testnet-phase19".to_string()),
        };

        let result = filter_peer_version(&announcement, "testnet-phase19");

        match result {
            VersionFilterResult::Compatible { common_capabilities, .. } => {
                assert_eq!(
                    common_capabilities.len(),
                    our_caps.len(),
                    "Should match all capabilities"
                );
            }
            other => panic!("Expected Compatible, got {:?}", other),
        }
    }
}

// ============================================================================
// SERIALIZATION TESTS
// ============================================================================

mod serialization_tests {
    use super::*;

    /// Test PeerHeightWithProof serializes correctly with new fields
    #[test]
    fn test_announcement_serialization() {
        let announcement = create_peer_height_announcement(
            "peer123",
            50000,
            "testnet-phase19",
        );

        // Serialize to JSON
        let json = serde_json::to_string(&announcement).expect("Should serialize");

        // Should contain version fields
        assert!(json.contains("software_version"));
        assert!(json.contains("protocol_version"));
        assert!(json.contains("upgrade_capabilities"));
        assert!(json.contains("network_id"));

        // Deserialize back
        let deserialized: PeerHeightWithProof = serde_json::from_str(&json).expect("Should deserialize");

        assert_eq!(deserialized.peer_id, announcement.peer_id);
        assert_eq!(deserialized.highest_block, announcement.highest_block);
        assert_eq!(deserialized.software_version, announcement.software_version);
        assert_eq!(deserialized.protocol_version, announcement.protocol_version);
    }

    /// Test backward compatibility - old format without new fields
    #[test]
    fn test_backward_compatible_deserialization() {
        // Old format without version fields
        let old_json = r#"{
            "peer_id": "old-peer",
            "highest_block": 50000,
            "height_proof": null,
            "blockchain_merkle_root": null,
            "timestamp": 1234567890
        }"#;

        // Should deserialize with default values for new fields
        let announcement: PeerHeightWithProof = serde_json::from_str(old_json)
            .expect("Should deserialize old format");

        assert_eq!(announcement.peer_id, "old-peer");
        assert_eq!(announcement.highest_block, 50000);
        assert!(announcement.software_version.is_none(), "Should default to None");
        assert!(announcement.protocol_version.is_none(), "Should default to None");
        assert!(announcement.upgrade_capabilities.is_empty(), "Should default to empty");
        assert!(announcement.network_id.is_none(), "Should default to None");
    }
}

// ============================================================================
// STRESS / PERFORMANCE TESTS
// ============================================================================

mod performance_tests {
    use super::*;
    use std::time::Instant;

    /// Test version filtering performance
    #[test]
    fn test_filter_performance() {
        let announcement = create_peer_height_announcement(
            "peer123",
            50000,
            "testnet-phase19",
        );

        let iterations = 10_000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = filter_peer_version(&announcement, "testnet-phase19");
        }

        let elapsed = start.elapsed();
        let per_filter_us = elapsed.as_micros() / iterations as u128;

        println!("Version filter performance: {} us per filter", per_filter_us);

        // Should be fast - under 100 microseconds per filter
        assert!(
            per_filter_us < 100,
            "Version filter should be fast, got {} us",
            per_filter_us
        );
    }

    /// Test announcement creation performance
    #[test]
    fn test_announcement_creation_performance() {
        let iterations = 10_000;
        let start = Instant::now();

        for i in 0..iterations {
            let _ = create_peer_height_announcement(
                &format!("peer{}", i),
                i as u64,
                "testnet-phase19",
            );
        }

        let elapsed = start.elapsed();
        let per_create_us = elapsed.as_micros() / iterations as u128;

        println!("Announcement creation: {} us per create", per_create_us);

        // Should be fast - under 50 microseconds per creation
        assert!(
            per_create_us < 50,
            "Announcement creation should be fast, got {} us",
            per_create_us
        );
    }
}
