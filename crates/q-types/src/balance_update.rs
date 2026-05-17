//! ✅ v3.2.7-beta: P2P Balance Update Messages - SECURITY HARDENED + U128 FIX
//!
//! Enables decentralized mining by broadcasting balance updates across the network.
//! When a node accepts a mining solution and credits a balance, it broadcasts
//! the update to all peers so they can apply the same balance change.
//!
//! ## v3.2.7-beta CRITICAL FIX: U128 Serialization for CBOR
//!
//! CBOR (like MessagePack) does NOT natively support u128 - it truncates to u64!
//! This caused mining rewards to be corrupted during P2P broadcast.
//! Fixed by using u128_serde to serialize as strings.
//!
//! ## Security Model (v1.1.9-beta HARDENED)
//!
//! Balance updates are validated by:
//! 1. Verifying the mining solution nonce produces valid hash
//! 2. Checking the block height is consistent with local chain
//! 3. **MANDATORY** signature verification from originating node (Ed25519/Dilithium5)
//! 4. Deduplication using LRU cache with (wallet, block_height, nonce) key
//! 5. Origin node must be in validator allowlist
//! 6. Rate limiting per origin node (max 100 updates/minute)
//!
//! ## Flow
//!
//! ```text
//! Miner submits solution to LocalNode
//!   → LocalNode validates solution
//!   → LocalNode credits balance locally
//!   → LocalNode SIGNS the update (mandatory)
//!   → LocalNode broadcasts P2PBalanceUpdate via gossipsub
//!   → All peers receive update
//!   → Peers verify signature (MANDATORY)
//!   → Peers check dedup cache
//!   → Peers verify origin is allowed validator
//!   → Peers apply balance update if all checks pass
//! ```

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};

// Import u128_serde from parent module for CBOR compatibility
use crate::u128_serde;

/// P2P Balance Update Message
/// Broadcast when a mining reward is credited to a wallet
///
/// v1.1.9-beta: Signature is now MANDATORY for security
/// v2.5.0: Amount fields upgraded to u128 (version 3)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2PBalanceUpdate {
    /// Version for protocol evolution
    /// - v2: v1.1.9 with mandatory signatures (u64 amounts)
    /// - v3: v2.5.0 with u128 amounts (24 decimals)
    pub version: u8,

    /// Wallet address receiving the balance update (qnk format)
    pub wallet_address: String,

    /// v3.2.7-beta: CRITICAL FIX - Use u128_serde for CBOR compatibility
    /// CBOR truncates u128 to u64, corrupting mining rewards during P2P broadcast!
    #[serde(with = "u128_serde")]
    pub amount: u128,

    /// v3.2.7-beta: CRITICAL FIX - Use u128_serde for CBOR compatibility
    #[serde(with = "u128_serde")]
    pub new_balance: u128,

    /// Block height at which this reward was earned
    pub block_height: u64,

    /// Mining nonce that earned this reward (for verification)
    pub nonce: u64,

    /// Update type (mining_reward, transaction, etc.)
    pub update_type: BalanceUpdateType,

    /// Timestamp in milliseconds since Unix epoch
    pub timestamp_ms: u64,

    /// Node ID of the originating node (PeerId as base58)
    pub origin_node_id: String,

    /// SHA3-256 hash of the mining solution (for verification)
    pub solution_hash: [u8; 32],

    /// v1.1.9-beta: MANDATORY signature from originating node
    /// Ed25519 signature over the signing payload (wallet + amount + height + nonce + timestamp)
    /// This prevents forged balance updates from malicious nodes
    pub signature: Vec<u8>,

    /// v1.1.9-beta: Public key of the signing node (for verification without lookup)
    /// 32 bytes for Ed25519, 2592 bytes for Dilithium5
    pub signer_public_key: Vec<u8>,

    /// v5.1.0: Block hash that contains the transaction proving this balance update
    /// Required after BlockEvidenceRequired upgrade activation
    #[serde(default)]
    pub block_hash: Option<[u8; 32]>,

    /// v5.1.0: Transaction index within the block (for verification)
    #[serde(default)]
    pub tx_index: Option<u32>,
}

/// Type of balance update
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BalanceUpdateType {
    /// Mining reward from valid solution
    MiningReward,
    /// Transaction received (not used for broadcast, just for completeness)
    TransactionReceived,
    /// Faucet claim
    FaucetClaim,
    /// DEX swap output
    DexSwap,
    /// Liquidity provision reward
    LiquidityReward,
    /// Contract execution output
    ContractOutput,
}

impl P2PBalanceUpdate {
    /// Current protocol version (3 = u128 amounts)
    pub const CURRENT_VERSION: u8 = 3;

    /// Create a new mining reward balance update (unsigned - must call sign() before broadcast)
    /// v2.5.0: Now uses u128 for amount and new_balance
    pub fn new_mining_reward(
        wallet_address: String,
        amount: u128,
        new_balance: u128,
        block_height: u64,
        nonce: u64,
        origin_node_id: String,
    ) -> Self {
        // Calculate solution hash
        let mut hasher = Sha3_256::new();
        hasher.update(wallet_address.as_bytes());
        hasher.update(&nonce.to_le_bytes());
        hasher.update(&block_height.to_le_bytes());
        let solution_hash: [u8; 32] = hasher.finalize().into();

        Self {
            version: Self::CURRENT_VERSION, // v2.5.0: Version 3 = u128 amounts
            wallet_address,
            amount,
            new_balance,
            block_height,
            nonce,
            update_type: BalanceUpdateType::MiningReward,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            origin_node_id,
            solution_hash,
            signature: Vec::new(), // Must be filled by sign()
            signer_public_key: Vec::new(), // Must be filled by sign()
            block_hash: None,   // v5.1.0: Set when block evidence is available
            tx_index: None,     // v5.1.0: Set when block evidence is available
        }
    }

    /// Create from legacy v2 message (u64 amounts) for backward compatibility
    pub fn from_legacy_v2(
        wallet_address: String,
        amount: u64,
        new_balance: u64,
        block_height: u64,
        nonce: u64,
        origin_node_id: String,
    ) -> Self {
        Self::new_mining_reward(
            wallet_address,
            super::legacy_to_u128(amount),
            super::legacy_to_u128(new_balance),
            block_height,
            nonce,
            origin_node_id,
        )
    }

    /// Get the payload that should be signed
    /// This is SHA3-256(wallet_address || amount || block_height || nonce || timestamp_ms || origin_node_id)
    /// v2.5.0: Now uses u128 (16 bytes) for amount
    pub fn signing_payload(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(self.wallet_address.as_bytes());
        hasher.update(&self.amount.to_le_bytes()); // 16 bytes for u128
        hasher.update(&self.block_height.to_le_bytes());
        hasher.update(&self.nonce.to_le_bytes());
        hasher.update(&self.timestamp_ms.to_le_bytes());
        hasher.update(self.origin_node_id.as_bytes());
        hasher.finalize().into()
    }

    /// Sign the balance update with an Ed25519 secret key
    /// Must be called before broadcasting
    #[cfg(feature = "signing")]
    pub fn sign(&mut self, secret_key: &ed25519_dalek::SigningKey) {
        use ed25519_dalek::Signer;
        let payload = self.signing_payload();
        let signature = secret_key.sign(&payload);
        self.signature = signature.to_bytes().to_vec();
        self.signer_public_key = secret_key.verifying_key().to_bytes().to_vec();
    }

    /// Sign with raw bytes (for nodes that manage their own keys)
    pub fn sign_with_bytes(&mut self, signature: Vec<u8>, public_key: Vec<u8>) {
        self.signature = signature;
        self.signer_public_key = public_key;
    }

    /// Verify the Ed25519 signature is valid
    /// Returns true if signature is valid, false otherwise
    pub fn verify_signature(&self) -> bool {
        // v1.1.9-beta: Signature is mandatory
        if self.signature.is_empty() || self.signer_public_key.is_empty() {
            return false;
        }

        // For Ed25519 (32-byte public key)
        if self.signer_public_key.len() == 32 {
            use ed25519_dalek::{Signature, VerifyingKey, Verifier};

            let public_key_bytes: [u8; 32] = match self.signer_public_key.clone().try_into() {
                Ok(bytes) => bytes,
                Err(_) => return false,
            };

            let verifying_key = match VerifyingKey::from_bytes(&public_key_bytes) {
                Ok(key) => key,
                Err(_) => return false,
            };

            let signature_bytes: [u8; 64] = match self.signature.clone().try_into() {
                Ok(bytes) => bytes,
                Err(_) => return false,
            };

            let signature = Signature::from_bytes(&signature_bytes);
            let payload = self.signing_payload();

            verifying_key.verify(&payload, &signature).is_ok()
        } else {
            // TODO: Add Dilithium5 verification for post-quantum signatures
            // For now, reject non-Ed25519 signatures until PQC is fully integrated
            false
        }
    }

    /// Check if the update has a valid signature attached
    pub fn is_signed(&self) -> bool {
        !self.signature.is_empty() && !self.signer_public_key.is_empty()
    }

    /// Get unique deduplication key for this update
    /// Used to prevent applying the same update twice
    pub fn dedup_key(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.wallet_address, self.block_height, self.nonce, self.update_type as u8
        )
    }

    /// Verify the solution hash matches
    pub fn verify_solution_hash(&self) -> bool {
        let mut hasher = Sha3_256::new();
        hasher.update(self.wallet_address.as_bytes());
        hasher.update(&self.nonce.to_le_bytes());
        hasher.update(&self.block_height.to_le_bytes());
        let computed: [u8; 32] = hasher.finalize().into();

        computed == self.solution_hash
    }

    /// Full validation: solution hash + signature (v1.1.9-beta)
    /// Both must pass for the update to be accepted
    /// v2.5.0: Accepts version 2 (u64) or version 3 (u128)
    pub fn validate_full(&self) -> Result<(), BalanceUpdateError> {
        // Check version (accept v2 or v3)
        if self.version < 2 {
            return Err(BalanceUpdateError::LegacyVersion(self.version));
        }
        // Note: v3 uses u128, v2 uses u64 but both serialize correctly

        // Check solution hash
        if !self.verify_solution_hash() {
            return Err(BalanceUpdateError::InvalidSolutionHash);
        }

        // Check signature (MANDATORY in v1.1.9+)
        if !self.is_signed() {
            return Err(BalanceUpdateError::MissingSignature);
        }

        if !self.verify_signature() {
            return Err(BalanceUpdateError::InvalidSignature);
        }

        // Check timestamp is not too old (24 hours)
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let age_ms = now_ms.saturating_sub(self.timestamp_ms);
        if age_ms > 24 * 60 * 60 * 1000 {
            return Err(BalanceUpdateError::ExpiredUpdate);
        }

        Ok(())
    }

    /// Serialize to CBOR bytes for gossipsub transmission
    pub fn to_cbor(&self) -> Result<Vec<u8>, serde_cbor::Error> {
        serde_cbor::to_vec(self)
    }

    /// Deserialize from CBOR bytes
    pub fn from_cbor(data: &[u8]) -> Result<Self, serde_cbor::Error> {
        serde_cbor::from_slice(data)
    }
}

/// Errors that can occur during balance update validation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BalanceUpdateError {
    /// Update uses legacy version without mandatory signatures
    LegacyVersion(u8),
    /// Solution hash does not match
    InvalidSolutionHash,
    /// Signature is missing (mandatory in v1.1.9+)
    MissingSignature,
    /// Signature verification failed
    InvalidSignature,
    /// Update is older than 24 hours
    ExpiredUpdate,
    /// Origin node is not in validator allowlist
    UnauthorizedNode,
    /// Rate limit exceeded for this node
    RateLimitExceeded,
    /// Duplicate update (already processed)
    DuplicateUpdate,
}

impl std::fmt::Display for BalanceUpdateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LegacyVersion(v) => write!(f, "Legacy version {} rejected (require v2+)", v),
            Self::InvalidSolutionHash => write!(f, "Invalid solution hash"),
            Self::MissingSignature => write!(f, "Missing mandatory signature"),
            Self::InvalidSignature => write!(f, "Signature verification failed"),
            Self::ExpiredUpdate => write!(f, "Update older than 24 hours"),
            Self::UnauthorizedNode => write!(f, "Origin node not in validator allowlist"),
            Self::RateLimitExceeded => write!(f, "Rate limit exceeded for origin node"),
            Self::DuplicateUpdate => write!(f, "Duplicate update already processed"),
        }
    }
}

impl std::error::Error for BalanceUpdateError {}

/// P2P Miner Statistics Message
/// Broadcast periodically to aggregate network hashrate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2PMinerStats {
    /// Version for protocol evolution
    pub version: u8,

    /// Node ID of the originating node
    pub node_id: String,

    /// Miner wallet address
    pub miner_address: String,

    /// Current hash rate in KH/s
    pub hashrate_khash: f64,

    /// Total solutions submitted in this period
    pub solutions_count: u64,

    /// Time period these stats cover (in seconds)
    pub period_seconds: u64,

    /// Timestamp in milliseconds since Unix epoch
    pub timestamp_ms: u64,

    /// Worker name (optional, for multi-worker setups)
    pub worker_name: Option<String>,
}

impl P2PMinerStats {
    /// Create new miner stats message
    pub fn new(
        node_id: String,
        miner_address: String,
        hashrate_khash: f64,
        solutions_count: u64,
        period_seconds: u64,
    ) -> Self {
        Self {
            version: 1,
            node_id,
            miner_address,
            hashrate_khash,
            solutions_count,
            period_seconds,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            worker_name: None,
        }
    }

    /// Serialize to CBOR bytes for gossipsub transmission
    pub fn to_cbor(&self) -> Result<Vec<u8>, serde_cbor::Error> {
        serde_cbor::to_vec(self)
    }

    /// Deserialize from CBOR bytes
    pub fn from_cbor(data: &[u8]) -> Result<Self, serde_cbor::Error> {
        serde_cbor::from_slice(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balance_update_serialization() {
        let mut update = P2PBalanceUpdate::new_mining_reward(
            "qnk768a02d2d47622b012adcf0dcced5949610a2a1fa39e10372309f25bd152c842".to_string(),
            8584, // 0.00008584 QNK
            1000000000, // 10 QNK
            12345,
            987654321,
            "12D3KooWTestNode123".to_string(),
        );

        // Add fake signature for serialization test
        update.signature = vec![0u8; 64];
        update.signer_public_key = vec![0u8; 32];

        let cbor = update.to_cbor().unwrap();
        let decoded = P2PBalanceUpdate::from_cbor(&cbor).unwrap();

        assert_eq!(decoded.wallet_address, update.wallet_address);
        assert_eq!(decoded.amount, update.amount);
        assert_eq!(decoded.block_height, update.block_height);
        assert_eq!(decoded.nonce, update.nonce);
        assert_eq!(decoded.version, 2); // v1.1.9-beta version
    }

    #[test]
    fn test_solution_hash_verification() {
        let update = P2PBalanceUpdate::new_mining_reward(
            "qnktest123".to_string(),
            1000,
            2000,
            100,
            12345,
            "node1".to_string(),
        );

        assert!(update.verify_solution_hash());
    }

    #[test]
    fn test_dedup_key() {
        let update = P2PBalanceUpdate::new_mining_reward(
            "qnktest".to_string(),
            1000,
            2000,
            100,
            12345,
            "node1".to_string(),
        );

        let key = update.dedup_key();
        assert!(key.contains("qnktest"));
        assert!(key.contains("100"));
        assert!(key.contains("12345"));
    }

    #[test]
    fn test_unsigned_update_fails_validation() {
        let update = P2PBalanceUpdate::new_mining_reward(
            "qnktest".to_string(),
            1000,
            2000,
            100,
            12345,
            "node1".to_string(),
        );

        // Unsigned update should fail validation
        assert!(!update.is_signed());
        let result = update.validate_full();
        assert!(matches!(result, Err(BalanceUpdateError::MissingSignature)));
    }

    #[test]
    fn test_signing_payload_deterministic() {
        let update1 = P2PBalanceUpdate::new_mining_reward(
            "qnktest".to_string(),
            1000,
            2000,
            100,
            12345,
            "node1".to_string(),
        );

        // Create another with same params (except timestamp)
        let mut update2 = P2PBalanceUpdate::new_mining_reward(
            "qnktest".to_string(),
            1000,
            2000,
            100,
            12345,
            "node1".to_string(),
        );
        update2.timestamp_ms = update1.timestamp_ms; // Force same timestamp

        // Payloads should match
        assert_eq!(update1.signing_payload(), update2.signing_payload());
    }

    #[test]
    fn test_miner_stats_serialization() {
        let stats = P2PMinerStats::new(
            "node1".to_string(),
            "qnktest".to_string(),
            1500.5,
            100,
            60,
        );

        let cbor = stats.to_cbor().unwrap();
        let decoded = P2PMinerStats::from_cbor(&cbor).unwrap();

        assert_eq!(decoded.miner_address, stats.miner_address);
        assert_eq!(decoded.hashrate_khash, stats.hashrate_khash);
        assert_eq!(decoded.solutions_count, stats.solutions_count);
    }
}
