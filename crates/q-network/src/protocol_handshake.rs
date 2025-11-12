// crates/q-network/src/protocol_handshake.rs
//
// Protocol version negotiation and compatibility checking

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

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
    /// e.g., "testnet-phase5", "mainnet"
    pub network_id: String,

    /// Feature flags for capability negotiation
    pub features: Vec<String>,
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

        Self {
            binary_version: env!("CARGO_PKG_VERSION").to_string(),
            turbo_sync_version: 1, // NEW format with protocol_version field
            supported_turbo_sync_versions: vec![0, 1], // Support both OLD and NEW
            build_timestamp,
            build_date,
            network_id: std::env::var("Q_NETWORK_ID")
                .unwrap_or_else(|_| "testnet-phase5".to_string()),
            features: vec![
                "turbo-sync".to_string(),
                "balance-consensus".to_string(),
                "distributed-ai".to_string(),
                "aegis-ql".to_string(),
            ],
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
            network_id: "testnet-phase5".to_string(),
            features: vec!["turbo-sync".to_string()],
        };

        let old_peer = ProtocolHandshake {
            binary_version: "0.9.52-beta".to_string(),
            turbo_sync_version: 0,
            supported_turbo_sync_versions: vec![0], // Only supports OLD format
            build_timestamp: 1234567000,
            build_date: "2024-01-01 00:00:00 UTC".to_string(),
            network_id: "testnet-phase5".to_string(),
            features: vec!["turbo-sync".to_string()],
        };

        let new_peer = ProtocolHandshake {
            binary_version: "0.9.57-beta".to_string(),
            turbo_sync_version: 1,
            supported_turbo_sync_versions: vec![0, 1], // Supports both
            build_timestamp: 1234567890,
            build_date: "2024-01-01 00:00:00 UTC".to_string(),
            network_id: "testnet-phase5".to_string(),
            features: vec!["turbo-sync".to_string()],
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
            network_id: "mainnet".to_string(), // Different network!
            features: vec!["turbo-sync".to_string()],
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
            network_id: "testnet-phase5".to_string(),
            features: vec!["turbo-sync".to_string()],
        };

        let old_only = ProtocolHandshake {
            binary_version: "0.9.52-beta".to_string(),
            turbo_sync_version: 0,
            supported_turbo_sync_versions: vec![0], // Only OLD format
            build_timestamp: 1234567000,
            build_date: "2024-01-01 00:00:00 UTC".to_string(),
            network_id: "testnet-phase5".to_string(),
            features: vec!["turbo-sync".to_string()],
        };

        // Should be incompatible - no common versions
        assert!(us.is_compatible(&old_only).is_err());
    }
}
