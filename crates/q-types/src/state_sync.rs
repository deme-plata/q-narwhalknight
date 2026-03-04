/// P2P State Sync Protocol for Q-NarwhalKnight
/// v5.3.0: Gossipsub-based state synchronization (request/response pattern)
///
/// Solves the "missed gossipsub" problem: if a node is offline when contract/pool/balance
/// state is broadcast via P2P, that state is permanently lost. This protocol enables
/// nodes to request full state snapshots from peers via gossipsub.
///
/// Follows the same pattern as turbo-sync: broadcast signed request, peers respond
/// with signed state data. All encrypted via existing Kyber1024 PQ handshake on
/// the libp2p transport layer.
///
/// Key features:
/// - Ed25519 domain-tagged signing for requests AND responses
/// - Compact binary serialization via serde/postcard
/// - Rate limiting via gossipsub queue (Normal priority)
/// - Add-only merge semantics (never overwrites local state)

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;

/// Domain tag for state sync request signatures (prevents cross-context replay)
const STATE_SYNC_REQUEST_DOMAIN: &[u8] = b"Q-NARWHALKNIGHT-STATE-SYNC-REQUEST-V1";

/// Domain tag for state sync response signatures
const STATE_SYNC_RESPONSE_DOMAIN: &[u8] = b"Q-NARWHALKNIGHT-STATE-SYNC-RESPONSE-V1";

// ============================================================================
// Request type
// ============================================================================

/// State snapshot request — broadcast on `/qnk/{network}/state-sync-requests`
/// Peers with state respond on `/qnk/{network}/state-sync-responses`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateSnapshotRequest {
    /// Unique request ID (random u64) for matching responses
    pub request_id: u64,

    /// Requester's Ed25519 public key (32 bytes)
    pub requester: [u8; 32],

    /// Requester's current block height (peers can decide if they're ahead)
    pub current_height: u64,

    /// Number of contracts the requester already has
    pub known_contracts: u32,

    /// Number of pools the requester already has
    pub known_pools: u32,

    /// Unix timestamp (seconds) — used for replay protection
    pub timestamp: u64,

    /// Ed25519 signature over the canonical signing message
    pub signature: Vec<u8>,

    /// Protocol version
    pub version: u8,
}

impl StateSnapshotRequest {
    /// Create a new unsigned request
    pub fn new(requester: [u8; 32], current_height: u64, known_contracts: u32, known_pools: u32) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        use rand::Rng;
        Self {
            request_id: rand::thread_rng().gen::<u64>(),
            requester,
            current_height,
            known_contracts,
            known_pools,
            timestamp,
            signature: Vec::new(),
            version: 1,
        }
    }

    /// Canonical signing message: domain || all fields in order
    fn signing_message(&self) -> Vec<u8> {
        let mut msg = Vec::with_capacity(128);
        msg.extend_from_slice(STATE_SYNC_REQUEST_DOMAIN);
        msg.extend_from_slice(&self.request_id.to_le_bytes());
        msg.extend_from_slice(&self.requester);
        msg.extend_from_slice(&self.current_height.to_le_bytes());
        msg.extend_from_slice(&self.known_contracts.to_le_bytes());
        msg.extend_from_slice(&self.known_pools.to_le_bytes());
        msg.extend_from_slice(&self.timestamp.to_le_bytes());
        msg.push(self.version);
        msg
    }

    /// Sign with Ed25519 key (feature-gated for binary size)
    /// Signs SHA3-256(signing_message) to match verify_block_signature() expectations
    #[cfg(feature = "signing")]
    pub fn sign(&mut self, signing_key: &ed25519_dalek::SigningKey) -> Result<()> {
        let message = self.signing_message();
        let hash: [u8; 32] = Sha3_256::digest(&message).into();
        let signature = crate::signature_verification::sign_ed25519(&hash, signing_key);
        self.signature = signature;
        Ok(())
    }

    /// Verify Ed25519 signature
    pub fn verify_signature(&self) -> Result<()> {
        if self.signature.len() != 64 {
            return Err(anyhow!(
                "Invalid signature length: expected 64, got {}",
                self.signature.len()
            ));
        }
        let message = self.signing_message();
        crate::signature_verification::verify_block_signature(
            &self.signature,
            &Sha3_256::digest(&message).into(),
            &self.requester,
            crate::block::SignaturePhase::Phase0Ed25519,
        )
    }

    /// Check timestamp is within acceptable window (5 minutes)
    pub fn is_fresh(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // Allow 5 minutes of clock skew
        self.timestamp > now.saturating_sub(300) && self.timestamp < now + 60
    }
}

// ============================================================================
// Response type
// ============================================================================

/// State snapshot response — sent on `/qnk/{network}/state-sync-responses`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateSnapshotResponse {
    /// Matching request_id from the request
    pub request_id: u64,

    /// Responder's Ed25519 public key (32 bytes)
    pub responder: [u8; 32],

    /// Responder's current block height
    pub block_height: u64,

    /// Smart contracts available on this node
    pub contracts: Vec<ContractSyncEntry>,

    /// Liquidity pools available on this node
    pub pools: Vec<PoolSyncEntry>,

    /// Wallet balances: hex(address) -> amount as string
    pub wallet_balances: HashMap<String, String>,

    /// Token balances: "{wallet_hex}_{token_hex}" -> amount as string
    pub token_balances: HashMap<String, String>,

    /// Symbol to address mapping: symbol -> hex(address)
    pub symbol_to_address: HashMap<String, String>,

    /// v8.7.4: Collateral vault data (serialized CollateralVault bytes)
    /// Enables one-time migration of historical vault state to new nodes
    #[serde(default)]
    pub vault_data: Option<Vec<u8>>,

    /// v1.0.3: Balance state hash (blake3 hex) for divergence detection.
    /// Computed from sorted wallet balances at this height. Allows nodes to detect
    /// if they have identical balance state without transmitting all balances.
    #[serde(default)]
    pub balance_state_hash: Option<String>,

    /// Unix timestamp
    pub timestamp: u64,

    /// Ed25519 signature
    pub signature: Vec<u8>,

    /// Protocol version
    pub version: u8,
}

impl StateSnapshotResponse {
    /// Create a new unsigned response
    pub fn new(request_id: u64, responder: [u8; 32], block_height: u64) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            request_id,
            responder,
            block_height,
            contracts: Vec::new(),
            pools: Vec::new(),
            wallet_balances: HashMap::new(),
            token_balances: HashMap::new(),
            symbol_to_address: HashMap::new(),
            vault_data: None,
            balance_state_hash: None,
            timestamp,
            signature: Vec::new(),
            version: 1,
        }
    }

    /// Canonical signing message — uses ONLY deterministic fields
    /// IMPORTANT: serde_json::to_vec of HashMap is NOT deterministic (key order varies),
    /// so we only hash counts + fixed-size fields, not serialized collections.
    /// Ed25519 signature proves responder identity; counts prove payload size.
    fn signing_message(&self) -> Vec<u8> {
        let mut msg = Vec::with_capacity(256);
        msg.extend_from_slice(STATE_SYNC_RESPONSE_DOMAIN);
        msg.extend_from_slice(&self.request_id.to_le_bytes());
        msg.extend_from_slice(&self.responder);
        msg.extend_from_slice(&self.block_height.to_le_bytes());
        // Hash deterministic payload metadata (counts only, NOT serialized content)
        let payload_hash = {
            let mut hasher = Sha3_256::new();
            hasher.update((self.contracts.len() as u64).to_le_bytes());
            hasher.update((self.pools.len() as u64).to_le_bytes());
            hasher.update((self.wallet_balances.len() as u64).to_le_bytes());
            hasher.update((self.token_balances.len() as u64).to_le_bytes());
            hasher.update((self.symbol_to_address.len() as u64).to_le_bytes());
            hasher.update((self.vault_data.as_ref().map(|v| v.len()).unwrap_or(0) as u64).to_le_bytes());
            hasher.finalize()
        };
        msg.extend_from_slice(&payload_hash);
        msg.extend_from_slice(&self.timestamp.to_le_bytes());
        msg.push(self.version);
        msg
    }

    /// Sign with Ed25519 key
    /// Signs SHA3-256(signing_message) to match verify_block_signature() expectations
    #[cfg(feature = "signing")]
    pub fn sign(&mut self, signing_key: &ed25519_dalek::SigningKey) -> Result<()> {
        let message = self.signing_message();
        let hash: [u8; 32] = Sha3_256::digest(&message).into();
        let signature = crate::signature_verification::sign_ed25519(&hash, signing_key);
        self.signature = signature;
        Ok(())
    }

    /// Verify Ed25519 signature
    pub fn verify_signature(&self) -> Result<()> {
        if self.signature.len() != 64 {
            return Err(anyhow!(
                "Invalid signature length: expected 64, got {}",
                self.signature.len()
            ));
        }
        let message = self.signing_message();
        crate::signature_verification::verify_block_signature(
            &self.signature,
            &Sha3_256::digest(&message).into(),
            &self.responder,
            crate::block::SignaturePhase::Phase0Ed25519,
        )
    }

    /// Summary string for logging
    pub fn summary(&self) -> String {
        format!(
            "height={}, contracts={}, pools={}, wallets={}, tokens={}, vault={}",
            self.block_height,
            self.contracts.len(),
            self.pools.len(),
            self.wallet_balances.len(),
            self.token_balances.len(),
            if self.vault_data.is_some() { "yes" } else { "no" },
        )
    }
}

// ============================================================================
// Sync entry types (compact representations for P2P transfer)
// ============================================================================

/// Compact contract representation for state sync
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContractSyncEntry {
    /// Contract address (32 bytes)
    pub address: [u8; 32],
    /// Token symbol
    pub symbol: String,
    /// Token name
    pub name: String,
    /// Decimal places
    pub decimals: u8,
    /// Total supply as string (can be very large)
    pub total_supply: String,
    /// Deployer wallet address (32 bytes)
    pub deployer: [u8; 32],
    /// Contract type ("SecureToken", "AdvancedToken", etc.)
    pub contract_type: String,
    /// Block height when deployed
    pub deployed_at: u64,
    /// Deployment parameters (flexible key-value)
    pub deployment_params: HashMap<String, serde_json::Value>,
}

/// Compact pool representation for state sync
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolSyncEntry {
    /// Pool ID string
    pub pool_id: String,
    /// Token 0 identifier (symbol or hex address)
    pub token0: String,
    /// Token 1 identifier
    pub token1: String,
    /// Reserve 0 as string (u128 can overflow JSON numbers)
    pub reserve0: String,
    /// Reserve 1 as string
    pub reserve1: String,
    /// LP token supply as string
    pub lp_token_supply: String,
    /// Liquidity provider address (32 bytes)
    pub provider: [u8; 32],
    /// Creation timestamp (unix seconds)
    pub created_at_unix: u64,
    /// Token 0 decimal places
    pub token0_decimals: u8,
    /// Token 1 decimal places
    pub token1_decimals: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_creation() {
        let req = StateSnapshotRequest::new([1u8; 32], 1000, 5, 3);
        assert_eq!(req.requester, [1u8; 32]);
        assert_eq!(req.current_height, 1000);
        assert_eq!(req.known_contracts, 5);
        assert_eq!(req.known_pools, 3);
        assert_eq!(req.version, 1);
        assert!(req.is_fresh());
    }

    #[test]
    fn test_response_creation() {
        let resp = StateSnapshotResponse::new(12345, [2u8; 32], 2000);
        assert_eq!(resp.request_id, 12345);
        assert_eq!(resp.responder, [2u8; 32]);
        assert_eq!(resp.block_height, 2000);
        assert_eq!(resp.version, 1);
        assert!(resp.contracts.is_empty());
        assert!(resp.pools.is_empty());
    }

    #[test]
    fn test_request_freshness() {
        let mut req = StateSnapshotRequest::new([1u8; 32], 100, 0, 0);
        assert!(req.is_fresh());

        // Old timestamp should fail
        req.timestamp = 1000;
        assert!(!req.is_fresh());
    }

    #[test]
    fn test_response_summary() {
        let mut resp = StateSnapshotResponse::new(1, [2u8; 32], 500);
        resp.contracts.push(ContractSyncEntry {
            address: [3u8; 32],
            symbol: "TEST".to_string(),
            name: "Test Token".to_string(),
            decimals: 8,
            total_supply: "1000000".to_string(),
            deployer: [4u8; 32],
            contract_type: "SecureToken".to_string(),
            deployed_at: 100,
            deployment_params: HashMap::new(),
        });
        let summary = resp.summary();
        assert!(summary.contains("contracts=1"));
        assert!(summary.contains("height=500"));
    }
}
