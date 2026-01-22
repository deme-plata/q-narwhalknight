/// Token Announcement P2P Broadcasting Module
/// v2.3.7-beta: DEX Decentralization - Cross-Node Token Discovery
///
/// This module provides data structures and cryptographic functions for broadcasting
/// custom token deployments across the P2P network using gossipsub.
///
/// Key features:
/// - Ed25519 signature verification for token announcements
/// - Domain-tagged signing for security (prevents signature replay attacks)
/// - Rate limiting with automatic cleanup
/// - Compatible with DAG-Knight consensus patterns

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};

/// Domain tag for token announcement signatures (prevents cross-context replay attacks)
const TOKEN_ANNOUNCEMENT_DOMAIN: &[u8] = b"Q-NARWHALKNIGHT-TOKEN-ANNOUNCEMENT-V1";

/// Domain tag for token sync request signatures
const TOKEN_SYNC_REQUEST_DOMAIN: &[u8] = b"Q-NARWHALKNIGHT-TOKEN-SYNC-REQUEST-V1";

/// Token announcement message for P2P broadcasting
/// This is the primary data structure broadcast on the `/qnk/contract-deployments` gossipsub topic
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenAnnouncement {
    /// Contract address (32-byte hash derived from deployment)
    pub contract_address: [u8; 32],

    /// Token symbol (e.g., "CHAD", "MYTOKEN")
    pub symbol: String,

    /// Token name (e.g., "Chad Token", "My Custom Token")
    pub name: String,

    /// Number of decimal places (typically 8 for QUG ecosystem)
    pub decimals: u8,

    /// Total supply at deployment (in base units)
    pub total_supply: u64,

    /// Deployer's wallet address (32-byte Ed25519 public key)
    pub deployer: [u8; 32],

    /// Contract type ("SecureToken", "AdvancedToken", "RwaToken", etc.)
    pub contract_type: String,

    /// Unix timestamp when token was deployed
    pub timestamp: u64,

    /// Ed25519 signature of the announcement (64 bytes)
    /// Signs: domain_tag || contract_address || symbol || name || decimals || total_supply || deployer || contract_type || timestamp
    pub signature: Vec<u8>,

    /// Announcement version (for future protocol upgrades)
    pub version: u8,
}

impl TokenAnnouncement {
    /// Create a new token announcement (unsigned)
    pub fn new(
        contract_address: [u8; 32],
        symbol: String,
        name: String,
        decimals: u8,
        total_supply: u64,
        deployer: [u8; 32],
        contract_type: String,
        timestamp: u64,
    ) -> Self {
        Self {
            contract_address,
            symbol,
            name,
            decimals,
            total_supply,
            deployer,
            contract_type,
            timestamp,
            signature: Vec::new(),
            version: 1,
        }
    }

    /// Get the canonical signing message for this announcement
    fn signing_message(&self) -> Vec<u8> {
        let mut message = Vec::new();
        message.extend_from_slice(TOKEN_ANNOUNCEMENT_DOMAIN);
        message.extend_from_slice(&self.contract_address);
        message.extend_from_slice(self.symbol.as_bytes());
        message.extend_from_slice(self.name.as_bytes());
        message.push(self.decimals);
        message.extend_from_slice(&self.total_supply.to_le_bytes());
        message.extend_from_slice(&self.deployer);
        message.extend_from_slice(self.contract_type.as_bytes());
        message.extend_from_slice(&self.timestamp.to_le_bytes());
        message
    }

    /// Sign this announcement with the deployer's Ed25519 private key
    #[cfg(feature = "signing")]
    pub fn sign(&mut self, signing_key: &ed25519_dalek::SigningKey) -> Result<()> {
        let message = self.signing_message();
        let signature = crate::signature_verification::sign_ed25519(&message, signing_key);
        self.signature = signature;
        Ok(())
    }

    /// Verify the Ed25519 signature on this announcement
    pub fn verify_signature(&self) -> Result<()> {
        if self.signature.len() != 64 {
            return Err(anyhow!(
                "Invalid signature length: expected 64 bytes, got {}",
                self.signature.len()
            ));
        }

        let message = self.signing_message();

        // Use Ed25519 verification
        crate::signature_verification::verify_block_signature(
            &self.signature,
            &Sha3_256::digest(&message).into(),
            &self.deployer,
            crate::block::SignaturePhase::Phase0Ed25519,
        )
    }

    /// Verify structure validity (symbol, name, etc.)
    pub fn verify_structure(&self) -> Result<()> {
        // Symbol must be 1-12 alphanumeric characters
        if self.symbol.is_empty() || self.symbol.len() > 12 {
            return Err(anyhow!(
                "Invalid symbol length: must be 1-12 characters, got {}",
                self.symbol.len()
            ));
        }
        if !self.symbol.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err(anyhow!("Symbol must be alphanumeric"));
        }

        // Name must be 1-64 characters
        if self.name.is_empty() || self.name.len() > 64 {
            return Err(anyhow!(
                "Invalid name length: must be 1-64 characters, got {}",
                self.name.len()
            ));
        }

        // Decimals must be reasonable (0-18)
        if self.decimals > 18 {
            return Err(anyhow!(
                "Invalid decimals: must be 0-18, got {}",
                self.decimals
            ));
        }

        // Contract address must not be all zeros
        if self.contract_address == [0u8; 32] {
            return Err(anyhow!("Contract address cannot be zero"));
        }

        // Deployer must not be all zeros
        if self.deployer == [0u8; 32] {
            return Err(anyhow!("Deployer address cannot be zero"));
        }

        // Timestamp must be reasonable (not in the future, not too old)
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        if self.timestamp > now + 300 {
            return Err(anyhow!("Timestamp is too far in the future"));
        }
        // Allow tokens from up to 1 year ago for sync
        if self.timestamp < now.saturating_sub(365 * 24 * 60 * 60) {
            return Err(anyhow!("Timestamp is too old"));
        }

        Ok(())
    }

    /// Full verification (signature + structure)
    pub fn verify(&self) -> Result<()> {
        self.verify_structure()?;
        self.verify_signature()?;
        Ok(())
    }
}

/// Token sync request message
/// Used when a new node joins and wants to discover all tokens in the network
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenSyncRequest {
    /// Requesting node's ID
    pub requester: [u8; 32],

    /// Unix timestamp of request
    pub timestamp: u64,

    /// Only request tokens deployed after this timestamp (0 = all tokens)
    pub since_timestamp: Option<u64>,

    /// Ed25519 signature of the request
    pub signature: Vec<u8>,
}

impl TokenSyncRequest {
    /// Create a new token sync request (unsigned)
    pub fn new(requester: [u8; 32], since_timestamp: Option<u64>) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            requester,
            timestamp,
            since_timestamp,
            signature: Vec::new(),
        }
    }

    /// Get the canonical signing message
    fn signing_message(&self) -> Vec<u8> {
        let mut message = Vec::new();
        message.extend_from_slice(TOKEN_SYNC_REQUEST_DOMAIN);
        message.extend_from_slice(&self.requester);
        message.extend_from_slice(&self.timestamp.to_le_bytes());
        message.extend_from_slice(&self.since_timestamp.unwrap_or(0).to_le_bytes());
        message
    }

    /// Sign this request
    #[cfg(feature = "signing")]
    pub fn sign(&mut self, signing_key: &ed25519_dalek::SigningKey) -> Result<()> {
        let message = self.signing_message();
        let signature = crate::signature_verification::sign_ed25519(&message, signing_key);
        self.signature = signature;
        Ok(())
    }

    /// Verify the signature
    pub fn verify_signature(&self) -> Result<()> {
        if self.signature.len() != 64 {
            return Err(anyhow!("Invalid signature length"));
        }

        let message = self.signing_message();
        crate::signature_verification::verify_block_signature(
            &self.signature,
            &Sha3_256::digest(&message).into(),
            &self.requester,
            crate::block::SignaturePhase::Phase0Ed25519,
        )
    }
}

/// Token sync response message
/// Contains all tokens matching the request criteria
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenSyncResponse {
    /// List of token announcements
    pub tokens: Vec<TokenAnnouncement>,

    /// Unix timestamp of response
    pub timestamp: u64,

    /// Responding node's ID
    pub responder: [u8; 32],
}

impl TokenSyncResponse {
    /// Create a new token sync response
    pub fn new(tokens: Vec<TokenAnnouncement>, responder: [u8; 32]) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            tokens,
            timestamp,
            responder,
        }
    }
}

/// Rate limiter for token announcements (prevents spam)
/// Uses the same pattern as PoolAnnouncementRateLimiter
pub struct TokenAnnouncementRateLimiter {
    /// Map of node ID -> (announcement count, window start timestamp)
    limits: std::collections::HashMap<[u8; 32], (u32, u64)>,

    /// Maximum announcements per node per window
    max_per_window: u32,

    /// Window duration in seconds
    window_seconds: u64,
}

impl Default for TokenAnnouncementRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenAnnouncementRateLimiter {
    /// Create a new rate limiter with default settings (10 announcements per minute)
    pub fn new() -> Self {
        Self {
            limits: std::collections::HashMap::new(),
            max_per_window: 10,
            window_seconds: 60,
        }
    }

    /// Create a rate limiter with custom settings
    pub fn with_limits(max_per_window: u32, window_seconds: u64) -> Self {
        Self {
            limits: std::collections::HashMap::new(),
            max_per_window,
            window_seconds,
        }
    }

    /// Check if a node is allowed to send an announcement
    /// Returns Ok(()) if allowed, Err if rate limited
    pub fn check(&mut self, node_id: &[u8; 32]) -> Result<()> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        match self.limits.get_mut(node_id) {
            Some((count, window_start)) => {
                // Check if window has expired
                if now >= *window_start + self.window_seconds {
                    // Reset window
                    *count = 1;
                    *window_start = now;
                    Ok(())
                } else if *count >= self.max_per_window {
                    Err(anyhow!(
                        "Rate limited: {} announcements in {} seconds",
                        self.max_per_window,
                        self.window_seconds
                    ))
                } else {
                    *count += 1;
                    Ok(())
                }
            }
            None => {
                self.limits.insert(*node_id, (1, now));
                Ok(())
            }
        }
    }

    /// Clean up expired entries (should be called periodically)
    pub fn cleanup(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        self.limits.retain(|_, (_, window_start)| {
            now < *window_start + self.window_seconds * 2
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_announcement_creation() {
        let announcement = TokenAnnouncement::new(
            [1u8; 32],
            "TEST".to_string(),
            "Test Token".to_string(),
            8,
            1_000_000_000_000, // 10,000 tokens
            [2u8; 32],
            "SecureToken".to_string(),
            1704067200, // 2024-01-01
        );

        assert_eq!(announcement.symbol, "TEST");
        assert_eq!(announcement.decimals, 8);
        assert_eq!(announcement.version, 1);
    }

    #[test]
    fn test_token_announcement_structure_validation() {
        let mut announcement = TokenAnnouncement::new(
            [1u8; 32],
            "TEST".to_string(),
            "Test Token".to_string(),
            8,
            1_000_000_000_000,
            [2u8; 32],
            "SecureToken".to_string(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        );

        // Valid structure should pass
        assert!(announcement.verify_structure().is_ok());

        // Invalid symbol (too long)
        announcement.symbol = "TOOLONGSYMBOL123".to_string();
        assert!(announcement.verify_structure().is_err());
        announcement.symbol = "TEST".to_string();

        // Invalid decimals
        announcement.decimals = 20;
        assert!(announcement.verify_structure().is_err());
        announcement.decimals = 8;

        // Zero contract address
        announcement.contract_address = [0u8; 32];
        assert!(announcement.verify_structure().is_err());
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = TokenAnnouncementRateLimiter::with_limits(3, 60);
        let node_id = [1u8; 32];

        // First 3 should pass
        assert!(limiter.check(&node_id).is_ok());
        assert!(limiter.check(&node_id).is_ok());
        assert!(limiter.check(&node_id).is_ok());

        // 4th should be rate limited
        assert!(limiter.check(&node_id).is_err());

        // Different node should pass
        let other_node = [2u8; 32];
        assert!(limiter.check(&other_node).is_ok());
    }
}
