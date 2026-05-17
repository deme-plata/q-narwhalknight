// crates/q-network/src/protocol_handshake.rs
//
// Protocol version negotiation and compatibility checking

use anyhow::{Context, Result};
use q_types::Phase;
use serde::{Deserialize, Serialize};

/// Cryptographic phase capability for post-quantum readiness
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CryptoPhase {
    /// Phase 0: Classical cryptography (Ed25519 + QUIC)
    Phase0,
    /// Phase 1: Post-quantum cryptography (Dilithium5 + Kyber1024)
    Phase1,
    /// Phase 2: Quantum Key Distribution (QKD)
    Phase2,
    /// Phase 3: Quantum VDF / Lattice VRF
    Phase3,
}

impl From<Phase> for CryptoPhase {
    fn from(phase: Phase) -> Self {
        match phase {
            Phase::Phase0 => CryptoPhase::Phase0,
            Phase::Phase1 => CryptoPhase::Phase1,
            Phase::Phase2 => CryptoPhase::Phase2,
            Phase::Phase3 | Phase::Phase4 => CryptoPhase::Phase3,
        }
    }
}

/// Protocol handshake exchanged when peers connect
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProtocolHandshake {
    /// Binary version (e.g., "0.9.57-beta")
    pub binary_version: String,

    /// Turbo sync protocol version
    /// - 0 = OLD format: [start_height, end_height, request_id]
    /// - 1 = NEW format: [protocol_version, start_height, end_height, request_id]
    pub turbo_sync_version: u32,

    /// Supported turbo sync versions for backwards compatibility
    pub supported_turbo_sync_versions: Vec<u32>,

    /// Compilation timestamp (Unix epoch seconds)
    /// Used to detect stale binaries in Docker containers
    pub build_timestamp: u64,

    /// Human-readable build date
    pub build_date: String,

    /// Network ID (must match for communication)
    /// e.g., "mainnet", "mainnet"
    pub network_id: String,

    /// Feature flags for capability negotiation
    pub features: Vec<String>,

    /// ✨ v1.0.15-beta: Post-quantum cryptography capability
    /// Supported cryptographic phases (ordered from strongest to weakest)
    /// Peers will negotiate the strongest common phase
    pub supported_crypto_phases: Vec<CryptoPhase>,

    /// Current active crypto phase
    pub active_crypto_phase: CryptoPhase,

    /// v8.4.0: Self-reported bandwidth tier (Mbps) for sync peer selection
    /// 0 = unknown, 100 = 100Mbps, 1000 = 1Gbps, etc.
    /// Used by gravity-assist to prefer high-bandwidth peers for sync
    #[serde(default)]
    pub bandwidth_tier_mbps: u32,
}

impl ProtocolHandshake {
    /// Create handshake for current binary
    pub fn current() -> Self {
        // Use option_env! for build-time values that may not be available in all crates
        let build_timestamp = option_env!("BUILD_TIMESTAMP")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let build_date = option_env!("BUILD_DATE")
            .unwrap_or("unknown")
            .to_string();

        // ✨ v1.0.15-beta: Support both Phase 0 (Ed25519) and Phase 1 (Dilithium5)
        // Phase 1 is preferred, but we fall back to Phase 0 for compatibility
        let supported_crypto_phases = vec![
            CryptoPhase::Phase1, // Dilithium5 + Kyber1024 (preferred)
            CryptoPhase::Phase0, // Ed25519 + QUIC (fallback)
        ];

        // Start with Phase 0 for backward compatibility
        // Will upgrade to Phase 1 after handshake negotiation
        let active_crypto_phase = CryptoPhase::Phase0;

        Self {
            binary_version: env!("CARGO_PKG_VERSION").to_string(),
            turbo_sync_version: 1, // NEW format with protocol_version field
            supported_turbo_sync_versions: vec![0, 1], // Support both OLD and NEW
            build_timestamp,
            build_date,
            network_id: std::env::var("Q_NETWORK_ID")
                .unwrap_or_else(|_| "mainnet-genesis".to_string()),
            features: {
                let mut f = vec![
                    "turbo-sync".to_string(),
                    "balance-consensus".to_string(),
                    "distributed-ai".to_string(),
                    "aegis-ql".to_string(),
                    "pqc-dilithium5".to_string(), // ✨ NEW: Post-quantum signatures
                    "pqc-kyber1024".to_string(),  // ✨ NEW: Post-quantum key exchange
                ];
                // v8.6.2: Announce supernode capability when bandwidth >= 5 Gbps
                let bw = std::env::var("Q_BANDWIDTH_MBPS")
                    .ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
                if bw >= 5000 {
                    f.push("supernode".to_string());
                }
                // v10.1.5: Advertise QKD protocol support
                if crate::qkd_transport::is_qkd_enabled() {
                    f.push("qkd-bb84".to_string());
                    f.push("qkd-sarg04".to_string());
                    f.push("qkd-npab".to_string());
                }
                f
            },
            supported_crypto_phases,
            active_crypto_phase,
            // v8.4.0: Self-reported bandwidth for sync peer selection
            bandwidth_tier_mbps: std::env::var("Q_BANDWIDTH_MBPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0), // 0 = unknown (backward-compatible with old nodes)
        }
    }

    /// Check if this peer is compatible with our node
    pub fn is_compatible(&self, peer: &ProtocolHandshake) -> Result<()> {
        // Network ID must match exactly
        if self.network_id != peer.network_id {
            anyhow::bail!(
                "Network ID mismatch: our='{}', peer='{}'",
                self.network_id,
                peer.network_id
            );
        }

        // Find common turbo sync version
        let common_versions: Vec<_> = self
            .supported_turbo_sync_versions
            .iter()
            .filter(|v| peer.supported_turbo_sync_versions.contains(v))
            .collect();

        if common_versions.is_empty() {
            anyhow::bail!(
                "No compatible turbo sync protocol versions (ours: {:?}, peer: {:?})",
                self.supported_turbo_sync_versions,
                peer.supported_turbo_sync_versions
            );
        }

        Ok(())
    }

    /// Negotiate the highest common turbo sync version with a peer
    pub fn negotiate_turbo_sync_version(&self, peer: &ProtocolHandshake) -> u32 {
        // Find intersection of supported versions
        let common_versions: Vec<_> = self
            .supported_turbo_sync_versions
            .iter()
            .filter(|v| peer.supported_turbo_sync_versions.contains(v))
            .copied()
            .collect();

        // Return highest common version, or 0 if no overlap
        *common_versions.iter().max().unwrap_or(&0)
    }

    /// ✨ v1.0.15-beta: Negotiate the strongest common cryptographic phase
    ///
    /// This enables gradual network-wide migration to post-quantum cryptography:
    /// - Phase 0 (Ed25519) nodes can only use Phase 0
    /// - Phase 1 (Dilithium5) nodes prefer Phase 1 but fall back to Phase 0
    /// - No hard fork required - organic upgrade path
    ///
    /// # Returns
    /// The strongest crypto phase supported by both peers, or None if no overlap
    pub fn negotiate_crypto_phase(&self, peer: &ProtocolHandshake) -> Option<CryptoPhase> {
        // Find intersection of supported phases
        let mut common_phases: Vec<CryptoPhase> = self
            .supported_crypto_phases
            .iter()
            .filter(|phase| peer.supported_crypto_phases.contains(phase))
            .copied()
            .collect();

        if common_phases.is_empty() {
            return None;
        }

        // Sort by strength (Phase1 > Phase0) and return strongest
        common_phases.sort_by(|a, b| b.cmp(a)); // Descending order
        Some(common_phases[0])
    }

    /// Check if peer supports post-quantum cryptography
    pub fn supports_pqc(&self) -> bool {
        self.supported_crypto_phases.contains(&CryptoPhase::Phase1)
            || self.supported_crypto_phases.contains(&CryptoPhase::Phase2)
            || self.supported_crypto_phases.contains(&CryptoPhase::Phase3)
    }

    /// Check if we can establish a PQC connection with this peer
    pub fn can_use_pqc_with(&self, peer: &ProtocolHandshake) -> bool {
        match self.negotiate_crypto_phase(peer) {
            Some(CryptoPhase::Phase1) | Some(CryptoPhase::Phase2) | Some(CryptoPhase::Phase3) => {
                true
            }
            _ => false,
        }
    }

    /// Serialize handshake to bytes for network transmission
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        postcard::to_allocvec(self).context("Failed to serialize ProtocolHandshake")
    }

    /// Deserialize handshake from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        postcard::from_bytes(data).context("Failed to deserialize ProtocolHandshake")
    }

    /// Get age of binary in days
    pub fn binary_age_days(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if self.build_timestamp == 0 {
            return 0;
        }

        (now.saturating_sub(self.build_timestamp)) / 86400
    }

    /// Check if binary is stale (> 30 days old)
    pub fn is_stale(&self) -> bool {
        self.binary_age_days() > 30
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handshake_serialization() {
        let handshake = ProtocolHandshake::current();
        let bytes = handshake.to_bytes().unwrap();
        let decoded = ProtocolHandshake::from_bytes(&bytes).unwrap();

        assert_eq!(handshake.binary_version, decoded.binary_version);
        assert_eq!(handshake.turbo_sync_version, decoded.turbo_sync_version);
        assert_eq!(handshake.network_id, decoded.network_id);
    }

    #[test]
    fn test_version_negotiation() {
        let us = ProtocolHandshake {
            binary_version: "0.9.57-beta".to_string(),
            turbo_sync_version: 1,
            supported_turbo_sync_versions: vec![0, 1],
            build_timestamp: 1234567890,
            build_date: "2024-01-01 00:00:00 UTC".to_string(),
            network_id: "mainnet-genesis".to_string(),
            features: vec!["turbo-sync".to_string()],
            supported_crypto_phases: vec![CryptoPhase::Phase0],
            active_crypto_phase: CryptoPhase::Phase0,
        };

        let old_peer = ProtocolHandshake {
            binary_version: "0.9.52-beta".to_string(),
            turbo_sync_version: 0,
            supported_turbo_sync_versions: vec![0], // Only supports OLD format
            build_timestamp: 1234567000,
            build_date: "2024-01-01 00:00:00 UTC".to_string(),
            network_id: "mainnet-genesis".to_string(),
            features: vec!["turbo-sync".to_string()],
            supported_crypto_phases: vec![CryptoPhase::Phase0],
            active_crypto_phase: CryptoPhase::Phase0,
        };

        let new_peer = ProtocolHandshake {
            binary_version: "0.9.57-beta".to_string(),
            turbo_sync_version: 1,
            supported_turbo_sync_versions: vec![0, 1], // Supports both
            build_timestamp: 1234567890,
            build_date: "2024-01-01 00:00:00 UTC".to_string(),
            network_id: "mainnet-genesis".to_string(),
            features: vec!["turbo-sync".to_string()],
            supported_crypto_phases: vec![CryptoPhase::Phase0],
            active_crypto_phase: CryptoPhase::Phase0,
        };

        // Should negotiate to version 0 with old peer (highest common)
        assert_eq!(us.negotiate_turbo_sync_version(&old_peer), 0);

        // Should negotiate to version 1 with new peer
        assert_eq!(us.negotiate_turbo_sync_version(&new_peer), 1);

        // Both should be compatible
        assert!(us.is_compatible(&old_peer).is_ok());
        assert!(us.is_compatible(&new_peer).is_ok());
    }

    #[test]
    fn test_incompatible_network() {
        let us = ProtocolHandshake::current();

        let wrong_network = ProtocolHandshake {
            binary_version: "0.9.57-beta".to_string(),
            turbo_sync_version: 1,
            supported_turbo_sync_versions: vec![0, 1],
            build_timestamp: 1234567890,
            build_date: "2024-01-01 00:00:00 UTC".to_string(),
            network_id: "mainnet2026.1".to_string(), // Different network!
            features: vec!["turbo-sync".to_string()],
            supported_crypto_phases: vec![CryptoPhase::Phase0],
            active_crypto_phase: CryptoPhase::Phase0,
        };

        // Should be incompatible due to network ID mismatch
        assert!(us.is_compatible(&wrong_network).is_err());
    }

    #[test]
    fn test_no_common_versions() {
        let us = ProtocolHandshake {
            binary_version: "0.9.57-beta".to_string(),
            turbo_sync_version: 1,
            supported_turbo_sync_versions: vec![1], // Only NEW format
            build_timestamp: 1234567890,
            build_date: "2024-01-01 00:00:00 UTC".to_string(),
            network_id: "mainnet-genesis".to_string(),
            features: vec!["turbo-sync".to_string()],
            supported_crypto_phases: vec![CryptoPhase::Phase0],
            active_crypto_phase: CryptoPhase::Phase0,
        };

        let old_only = ProtocolHandshake {
            binary_version: "0.9.52-beta".to_string(),
            turbo_sync_version: 0,
            supported_turbo_sync_versions: vec![0], // Only OLD format
            build_timestamp: 1234567000,
            build_date: "2024-01-01 00:00:00 UTC".to_string(),
            network_id: "mainnet-genesis".to_string(),
            features: vec!["turbo-sync".to_string()],
            supported_crypto_phases: vec![CryptoPhase::Phase0],
            active_crypto_phase: CryptoPhase::Phase0,
        };

        // Should be incompatible - no common versions
        assert!(us.is_compatible(&old_only).is_err());
    }

    #[test]
    fn test_pqc_phase_negotiation() {
        // Phase 1 node (supports both Phase0 and Phase1)
        let phase1_node = ProtocolHandshake {
            binary_version: "1.0.15-beta".to_string(),
            turbo_sync_version: 1,
            supported_turbo_sync_versions: vec![0, 1],
            build_timestamp: 1700000000,
            build_date: "2025-11-15 00:00:00 UTC".to_string(),
            network_id: "testnet-phase11".to_string(),
            features: vec!["pqc-dilithium5".to_string(), "pqc-kyber1024".to_string()],
            supported_crypto_phases: vec![CryptoPhase::Phase1, CryptoPhase::Phase0],
            active_crypto_phase: CryptoPhase::Phase0,
        };

        // Phase 0 only node (legacy)
        let phase0_node = ProtocolHandshake {
            binary_version: "0.9.80-beta".to_string(),
            turbo_sync_version: 1,
            supported_turbo_sync_versions: vec![0, 1],
            build_timestamp: 1690000000,
            build_date: "2025-10-01 00:00:00 UTC".to_string(),
            network_id: "testnet-phase11".to_string(),
            features: vec!["turbo-sync".to_string()],
            supported_crypto_phases: vec![CryptoPhase::Phase0],
            active_crypto_phase: CryptoPhase::Phase0,
        };

        // Test 1: Phase1 + Phase1 = Phase1 (strongest)
        let negotiated = phase1_node.negotiate_crypto_phase(&phase1_node);
        assert_eq!(negotiated, Some(CryptoPhase::Phase1));

        // Test 2: Phase1 + Phase0 = Phase0 (fallback)
        let negotiated = phase1_node.negotiate_crypto_phase(&phase0_node);
        assert_eq!(negotiated, Some(CryptoPhase::Phase0));

        // Test 3: Phase0 + Phase1 = Phase0 (fallback)
        let negotiated = phase0_node.negotiate_crypto_phase(&phase1_node);
        assert_eq!(negotiated, Some(CryptoPhase::Phase0));

        // Test 4: PQC capability checks
        assert!(phase1_node.supports_pqc());
        assert!(!phase0_node.supports_pqc());

        // Test 5: Can use PQC together?
        assert!(phase1_node.can_use_pqc_with(&phase1_node)); // Both Phase1
        assert!(!phase1_node.can_use_pqc_with(&phase0_node)); // One is Phase0
        assert!(!phase0_node.can_use_pqc_with(&phase1_node)); // One is Phase0
    }

    #[test]
    fn test_pqc_gradual_migration() {
        // Simulate network upgrade scenario
        let old_network = vec![
            // 70% Phase 0 nodes
            ProtocolHandshake {
                binary_version: "0.9.80-beta".to_string(),
                turbo_sync_version: 1,
                supported_turbo_sync_versions: vec![0, 1],
                build_timestamp: 1690000000,
                build_date: "2025-10-01 00:00:00 UTC".to_string(),
                network_id: "testnet-phase11".to_string(),
                features: vec![],
                supported_crypto_phases: vec![CryptoPhase::Phase0],
                active_crypto_phase: CryptoPhase::Phase0,
            },
            // 30% Phase 1 nodes
            ProtocolHandshake {
                binary_version: "1.0.15-beta".to_string(),
                turbo_sync_version: 1,
                supported_turbo_sync_versions: vec![0, 1],
                build_timestamp: 1700000000,
                build_date: "2025-11-15 00:00:00 UTC".to_string(),
                network_id: "testnet-phase11".to_string(),
                features: vec!["pqc-dilithium5".to_string()],
                supported_crypto_phases: vec![CryptoPhase::Phase1, CryptoPhase::Phase0],
                active_crypto_phase: CryptoPhase::Phase0,
            },
        ];

        // New Phase 1 node joins network
        let new_node = ProtocolHandshake::current();

        // Should be able to connect to ALL nodes (via Phase0 fallback)
        for peer in &old_network {
            let negotiated = new_node.negotiate_crypto_phase(peer);
            assert!(negotiated.is_some(), "Should negotiate with all peers");

            // Phase1 nodes upgrade to Phase1 together
            // Phase0 nodes stay at Phase0
            if peer.supports_pqc() {
                assert_eq!(negotiated, Some(CryptoPhase::Phase1));
            } else {
                assert_eq!(negotiated, Some(CryptoPhase::Phase0));
            }
        }

        // ✅ No hard fork required - gradual organic upgrade!
    }

    #[test]
    fn test_crypto_phase_ordering() {
        // Verify Phase1 is stronger than Phase0
        assert!(CryptoPhase::Phase1 > CryptoPhase::Phase0);
        assert!(CryptoPhase::Phase2 > CryptoPhase::Phase1);
        assert!(CryptoPhase::Phase3 > CryptoPhase::Phase2);
    }
}
