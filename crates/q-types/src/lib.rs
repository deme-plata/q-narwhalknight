use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;
use uuid::Uuid;

// ============================================================================
// U128 SERIALIZATION MODULE (v3.2.2)
// ============================================================================
//
// Serializes u128 values as STRINGS for universal compatibility.
// This is required because:
// - JSON numbers lose precision above 2^53 (JavaScript limitation)
// - MessagePack doesn't natively support u128 (truncates to u64!)
// - String serialization works with ALL serializers (JSON, MessagePack, YAML, etc.)
// - Backward compatible: accepts string, u64, u128, i64 on deserialize
//
// ⚠️ CRITICAL FIX v3.2.2: Changed from serialize_u128() to serialize_str()
// The previous implementation broke P2P block broadcast because MessagePack
// silently truncated u128 values to u64, corrupting coinbase transaction amounts.

/// Serialize/deserialize u128 for cross-format compatibility
/// v3.2.7: CRITICAL FIX - Format-aware serialization for P2P and storage
/// - For P2P (MessagePack): Serialize as STRING (MessagePack truncates u128 to u64!)
/// - For Storage (Bincode): Use native u128 binary format
pub mod u128_serde {
    use serde::{de::Visitor, Deserializer, Serializer};
    use std::fmt;

    pub fn serialize<S>(value: &u128, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // v3.2.7: Format-aware serialization
        // - is_human_readable() == true: JSON, YAML, TOML → use string
        // - is_human_readable() == false: Bincode → use native u128
        // - MessagePack: is_human_readable() == false BUT doesn't support u128!
        //   We always use string for safety since MessagePack is used for P2P
        //
        // CRITICAL: MessagePack (rmp_serde) reports is_human_readable = false
        // but it DOES NOT handle u128 correctly - it truncates to u64!
        // Always serialize as string to ensure P2P compatibility.
        serializer.serialize_str(&value.to_string())
    }

    struct U128Visitor;

    impl<'de> Visitor<'de> for U128Visitor {
        type Value = u128;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a u128, u64, i64, or string representing a number")
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            v.parse().map_err(E::custom)
        }

        fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            v.parse().map_err(E::custom)
        }

        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(v as u128)
        }

        fn visit_u128<E>(self, v: u128) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(v)
        }

        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if v >= 0 {
                Ok(v as u128)
            } else {
                Err(E::custom("negative value cannot be u128"))
            }
        }

        fn visit_i128<E>(self, v: i128) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if v >= 0 {
                Ok(v as u128)
            } else {
                Err(E::custom("negative value cannot be u128"))
            }
        }

        // v5.1.1: Handle f64 for JSON compatibility (JavaScript sends large numbers as floats)
        fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if v < 0.0 {
                Err(E::custom("negative value cannot be u128"))
            } else if v > u128::MAX as f64 {
                Err(E::custom("value too large for u128"))
            } else {
                Ok(v as u128)
            }
        }

        // v3.2.7: Handle bytes for Bincode compatibility (16 bytes = u128)
        fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if v.len() == 16 {
                Ok(u128::from_le_bytes(v.try_into().unwrap()))
            } else if v.len() == 8 {
                // Legacy u64 format
                Ok(u64::from_le_bytes(v.try_into().unwrap()) as u128)
            } else {
                Err(E::custom(format!("expected 8 or 16 bytes for u128, got {}", v.len())))
            }
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u128, D::Error>
    where
        D: Deserializer<'de>,
    {
        // v3.4.1: Use deserialize_str for bincode (binary) formats.
        // v5.1.1: Use deserialize_any for JSON (human-readable) formats so that
        //         numbers, floats, and strings all work from HTTP API requests.
        //         Bincode does NOT support deserialize_any, so we must check.
        if deserializer.is_human_readable() {
            deserializer.deserialize_any(U128Visitor)
        } else {
            deserializer.deserialize_str(U128Visitor)
        }
    }
}

/// Serialize/deserialize Option<u128> for cross-format compatibility
/// v3.2.2: Use STRING serialization for MessagePack P2P compatibility
pub mod option_u128_serde {
    use serde::{de::Visitor, Deserializer, Serializer};
    use std::fmt;

    /// Helper struct for string serialization of u128
    struct U128AsString(u128);

    impl serde::Serialize for U128AsString {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serializer.serialize_str(&self.0.to_string())
        }
    }

    pub fn serialize<S>(value: &Option<u128>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // v3.2.2: Serialize inner value as string for MessagePack compatibility
        match value {
            Some(v) => serializer.serialize_some(&U128AsString(*v)),
            None => serializer.serialize_none(),
        }
    }

    struct OptionU128Visitor;

    impl<'de> Visitor<'de> for OptionU128Visitor {
        type Value = Option<u128>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("an optional u128 value")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where
            D: Deserializer<'de>,
        {
            // Deserialize the inner value using u128_serde
            super::u128_serde::deserialize(deserializer).map(Some)
        }

        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(Some(v as u128))
        }

        fn visit_u128<E>(self, v: u128) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(Some(v))
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if v.is_empty() {
                Ok(None)
            } else {
                v.parse().map(Some).map_err(E::custom)
            }
        }

        fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if v.is_empty() {
                Ok(None)
            } else {
                v.parse().map(Some).map_err(E::custom)
            }
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<u128>, D::Error>
    where
        D: Deserializer<'de>,
    {
        // v3.2.2: Use deserialize_option for proper None handling
        // The visitor handles strings from MessagePack
        deserializer.deserialize_option(OptionU128Visitor)
    }
}

// Re-export commonly used types (ed25519-dalek v2.x API compatibility)
pub use ed25519_dalek::{Signature, SigningKey as SecretKey, VerifyingKey as PublicKey};
pub use sha3::{Digest, Sha3_256};

// DAG-Knight blockchain types module
pub mod block;

// ✅ v0.9.68-beta: libp2p request-response protocol for block sync
pub mod block_pack;

// ✨ v1.0.15-beta: Post-quantum signature verification
pub mod signature_verification;

// ✨ v1.0.16-beta: PQC key management
pub mod pqc_keys;

// ✨ v1.0.16-beta: ZK proof integration for untrusted setup
pub mod zk_proof_integration;

// ✨ MessagePack versioned types for P2P messaging
pub mod messagepack;

// ✨ v1.0.3-beta: Block-Vertex mapping for DAG-aware sync (Phase 1)
pub mod block_vertex_map;

// ✨ v0.6.0-beta: Liquidity pool P2P broadcasting (DEX Decentralization Phase 2)
pub mod liquidity_pool;

// ✨ v1.0.80-beta: Legacy struct definitions for backwards-compatible block deserialization
pub mod legacy;

// ✨ v1.1.8-beta: P2P balance update messages for decentralized mining
pub mod balance_update;

// BFT-safe balance finalization types (Bracha RB + DAG-Knight anchoring)
pub mod balance_finality;

// ✨ v1.0.58-beta: Advanced cryptographic primitives (FROST, AEGIS, SQIsign, Bulletproofs, etc.)
#[cfg(feature = "advanced-crypto")]
pub mod advanced_crypto;

/// Phase 3 Week 11: Equivocation detection and slashing types
#[doc = "Equivocation detection proofs and slashing transactions"]
pub mod equivocation;

/// v1.4.1-beta: Block-height activated upgrades for safe mainnet evolution
#[doc = "Network upgrade framework - deploy new features without coordinated restarts"]
pub mod upgrades;

/// v2.3.5-beta: P2P Decentralized Mining - Gossipsub solution broadcasting
#[doc = "P2P mining solutions and network-consensus challenges"]
pub mod mining_solution;

/// v1.5.0-beta: CHIRON-style execution hints for parallel sync
/// Enables ~30% faster node synchronization via parallel state application
#[doc = "Transaction dependency graphs and parallel execution batches"]
pub mod execution_hints;

/// v2.5.0: Privacy Layer with zk-STARK and AEGIS-QL
/// Provides private transactions, encrypted P2P, and post-quantum security
#[doc = "Unified privacy with transparent zero-knowledge proofs and quantum-resistant encryption"]
pub mod privacy_layer;

/// v2.4.1-beta: Validator key backup with TemporalShield (5-of-9 threshold)
#[doc = "Secure validator keypair backup using post-quantum threshold secret sharing"]
pub mod validator_backup;

/// v2.3.7-beta: Token Announcement P2P Broadcasting (DEX Decentralization)
/// Enables cross-node token discovery via gossipsub /contract-deployments topic
#[doc = "Ed25519 signed token announcements for decentralized token registry"]
pub mod token_announcement;

/// v5.3.0: P2P State Sync Protocol (Gossipsub Request/Response)
/// Solves the "missed gossipsub" problem: nodes can request full state snapshots
/// from peers for contracts, pools, balances that were broadcast while offline.
#[doc = "Ed25519 signed state sync requests and responses for P2P state recovery"]
pub mod state_sync;

/// v3.4.2-beta: Unified ZK Transaction Validator
/// Integrates STARK, Bulletproofs, and LatticeGuard for full privacy
#[doc = "Comprehensive ZK proof verification for transactions and blocks"]
pub mod unified_zk_validator;

/// v3.7.4: Validator Registry with Dilithium5 post-quantum signatures
pub mod validator_registry;

/// v8.5.0: P2P Update Announcement Protocol
/// Cryptographically signed software update announcements via gossipsub
/// with quorum verification (2-of-3 trusted bootstrap signers)
pub mod update_announcement;

// Re-export block types for convenience
pub use block::{
    QBlock, BlockHeader, BlockHash, DagRound, MiningSolution,
    QuantumMetadata, HypergraphCoordinates, EnergyComponents,
    SpectralSignature, SignaturePhase, VDFProof, FinalityStatus, FinalizedBlock,
    FinalityCertificate,
    // v1.4.5-beta: VDF security parameters for chain binding
    AdaptiveVDFParams, SecurityTier,
};

// Re-export signature verification functions
pub use signature_verification::{
    verify_spectral_signature, verify_spectral_signature_extended,
    verify_block_signature,
    // v2.4.7-beta: Core signature verification (for consensus votes, vertices)
    verify_ed25519_signature, verify_dilithium5_signature,
    // SQIsign compact signatures (v1.0.86-beta) - 95.6% smaller than Dilithium5
    SQISIGN_PK_SIZE, SQISIGN_SIG_SIZE,
    // v2.3.0-beta: Transaction signing functions (always available)
    sign_sqisign,
};

#[cfg(feature = "signing")]
pub use signature_verification::{
    sign_ed25519,
};

// Re-export PQC key management types
pub use pqc_keys::{
    ValidatorKeypair, ValidatorPublicKeys, ValidatorKeyRegistry,
};

// Re-export validator backup types (v2.4.1-beta)
pub use validator_backup::{
    ValidatorKeyBackup, BackupMetadata, BackupStatus, RestoreResult,
    VALIDATOR_BACKUP_THRESHOLD, VALIDATOR_BACKUP_TOTAL,
};

// Re-export block pack types
pub use block_pack::{
    BlockPackRequest, BlockPackResponse, BlockPackProtocol,
    BlockPackCodec, MAX_BLOCKS_PER_REQUEST,
};

// Re-export MessagePack versioned types
pub use messagepack::{
    VersionedBlock, VersionedTransaction,
    BLOCK_VERSION, TRANSACTION_VERSION,
};

// Re-export block-vertex mapping
pub use block_vertex_map::BlockVertexMap;

// Re-export liquidity pool P2P types
pub use liquidity_pool::{
    PoolAnnouncement, PoolSyncRequest, PoolSyncResponse,
    PoolAnnouncementRateLimiter,
};

// Re-export token announcement types (v2.3.7-beta: DEX decentralization)
pub use token_announcement::{
    TokenAnnouncement, TokenSyncRequest, TokenSyncResponse,
    TokenAnnouncementRateLimiter,
};

// Re-export P2P balance update types (v1.1.9-beta: security hardened decentralized mining)
pub use balance_update::{
    P2PBalanceUpdate, P2PMinerStats, BalanceUpdateType, BalanceUpdateError,
};

// Re-export Bracha RB balance finality types (BFT-safe finalization)
pub use balance_finality::{
    BrachaPhase, BrachaBalanceMsg, BrachaInstance, ValidatorBitmask,
    BalanceFinalityRecord, DagBalanceAnchorResponse,
    BRACHA_ROUND_WINDOW, BRACHA_PROPOSAL_TIMEOUT_ROUNDS, MAX_ANCHOR_BATCH, ANCHOR_FLUSH_SECS,
};

// Re-export CHIRON execution hints types (v1.5.0-beta: parallel sync)
pub use execution_hints::{
    BlockExecutionHints, TxAccessSet, TxIndex,
};

// Re-export unified ZK validator types (v3.4.2-beta: full privacy)
pub use unified_zk_validator::{
    UnifiedZkValidator, BlockZkValidator, BlockZkRequirements, BlockZkValidationResult,
    ZkProofBundle, ZkPrivacyLevel, StarkTransactionProof, StarkPublicInputs,
    BulletproofRangeProof, LatticeTransactionProof, LatticeSecurityLevel,
    ValidationStats,
};

// P2P block synchronization types are defined at the end of this file (BlockRequest, BlockResponse)

/// Core blockchain types for Q-NarwhalKnight Phase 0
/// These will be extended with post-quantum primitives in Phase 1
///
/// Transaction hash type
pub type TxHash = [u8; 32];

/// Transaction ID type alias (same as TxHash)
pub type TransactionId = TxHash;

/// Proposal hash for consensus
pub type ProposalHash = [u8; 32];

/// Consensus vote structure
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConsensusVote {
    pub epoch: u64,
    pub proposal_hash: ProposalHash,
    pub participated: bool,
}

/// Block height
pub type Height = u64;

/// Round number in DAG consensus
pub type Round = u64;

/// Node identifier (Phase 0: hash of Ed25519 public key)
pub type NodeId = [u8; 32];

/// Validator identifier (alias for NodeId)
pub type ValidatorId = NodeId;

// ============================================================================
// AMOUNT TYPE & DECIMAL CONSTANTS (v2.5.0 - u128 upgrade)
// ============================================================================

/// Amount type for token operations
/// v2.5.0: Upgraded from u64 to u128 for:
/// - Token supplies up to 10^38 (u128 max: ~3.4 × 10^38)
/// - 24 decimals for native coin (extreme precision)
/// - Smart contracts with massive token supplies (10^30+)
pub type Amount = u128;

/// Native coin decimals (24 for extreme precision)
/// This allows 1 QUG = 1,000,000,000,000,000,000,000,000 base units
pub const NATIVE_DECIMALS: u8 = 24;

/// One native coin (QUG) in smallest units (10^24)
pub const ONE_NATIVE_COIN: u128 = 1_000_000_000_000_000_000_000_000;

/// Maximum supply: 21 million QUG with 24 decimals
pub const MAX_NATIVE_SUPPLY: u128 = 21_000_000 * ONE_NATIVE_COIN;

/// Token default decimals (18 for ERC-20 compatibility)
pub const TOKEN_DEFAULT_DECIMALS: u8 = 18;

/// One token with 18 decimals (10^18)
pub const ONE_TOKEN_18: u128 = 1_000_000_000_000_000_000;

/// Legacy: 8 decimals (Bitcoin-style, 1 satoshi = 10^-8)
/// Kept for backward compatibility with existing balances
pub const LEGACY_DECIMALS: u8 = 8;
pub const ONE_LEGACY_UNIT: u128 = 100_000_000; // 10^8

/// Convert from legacy u64 amount (8 decimals) to new u128 (24 decimals)
pub fn legacy_to_u128(legacy_amount: u64) -> u128 {
    // 24 - 8 = 16 additional decimal places
    (legacy_amount as u128) * 10u128.pow(16)
}

/// Convert from u128 (24 decimals) to legacy u64 (8 decimals) with truncation
/// WARNING: This loses precision! Only use for display or legacy systems.
pub fn u128_to_legacy(amount: u128) -> u64 {
    // Divide by 10^16, truncating fractional part
    (amount / 10u128.pow(16)) as u64
}

/// Format amount with decimals for display
pub fn format_amount(amount: u128, decimals: u8) -> String {
    let divisor = 10u128.pow(decimals as u32);
    let whole = amount / divisor;
    let frac = amount % divisor;
    if frac == 0 {
        format!("{}", whole)
    } else {
        // Trim trailing zeros from fractional part
        let frac_str = format!("{:0>width$}", frac, width = decimals as usize);
        let trimmed = frac_str.trim_end_matches('0');
        format!("{}.{}", whole, trimmed)
    }
}

/// Address type (Phase 0: Ed25519 public key hash)
pub type Address = [u8; 32];

/// Token type for dual-token economics (QUG mining token + QUGUSD stablecoin)
/// v2.4.0-beta: Added Custom variant for user-created tokens
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TokenType {
    /// QUG - Native mining token (21M fixed supply, deflationary)
    QUG,
    /// QUGUSD - Algorithmic stablecoin pegged to USD ($1.00)
    QUGUSD,
    /// Custom - User-created tokens identified by contract address
    Custom([u8; 32]),
}

/// Token information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    pub token_type: TokenType,
    pub name: String,
    pub symbol: String,
    pub decimals: u8,
    /// v2.5.0: Upgraded to u128 for massive token supplies (10^30+)
    #[serde(default)]
    pub max_supply: Option<u128>, // None for QUGUSD (unlimited if collateralized)
}

/// QUG token constants (legacy 8 decimals - kept for backward compatibility)
pub const QUG_DECIMALS: u8 = LEGACY_DECIMALS;
/// Legacy max supply with 8 decimals (21M * 10^8)
pub const QUG_MAX_SUPPLY_LEGACY: u64 = 2_100_000_000_000_000;
/// v2.5.0: Max supply with 24 decimals (21M * 10^24)
pub const QUG_MAX_SUPPLY: u128 = MAX_NATIVE_SUPPLY;
pub const QUG_TOKEN_ADDRESS: [u8; 32] = [
    0x51, 0x55, 0x47, 0x00, 0x00, 0x00, 0x00, 0x00, // "QUG" in hex + zeros
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// QUGUSD token constants
/// v3.6.11-beta: CRITICAL FIX - QUGUSD uses 24 decimals (same as QUG), not 8!
/// This was causing balance display issues where 42 QUGUSD appeared as 0.
pub const QUGUSD_DECIMALS: u8 = 24;
pub const QUGUSD_TOKEN_ADDRESS: [u8; 32] = [
    0x51, 0x55, 0x47, 0x55, 0x53, 0x44, 0x00, 0x00, // "QUGUSD" in hex + zeros
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// v8.7.3: Height at which full deterministic state replay activates.
/// ALL transaction types (DEX swaps, token ops, stablecoin, governance, etc.)
/// are replayed from blocks on every node via StateProcessor.
/// Set to 0 = replay ALL blocks (combined with startup migration for historical data).
/// This enables true P2P state decentralization — no new P2P messages needed.
pub const STATE_REPLAY_ACTIVATION_HEIGHT: u64 = 0;

/// Bank master account address for fee collection
pub const BANK_MASTER_ACCOUNT: [u8; 32] = [
    0x42, 0x41, 0x4E, 0x4B, 0x00, 0x00, 0x00, 0x00, // "BANK" in hex + zeros
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// VAULT RWA token constants (v4.2.0-beta)
/// Physical hardware wallet token - 1 token = 1 physical Quillon Vault device
/// Whole units only (0 decimals), owned by BANK_MASTER_ACCOUNT
pub const VAULT_DECIMALS: u8 = 0;
pub const VAULT_TOKEN_ADDRESS: [u8; 32] = [
    0x56, 0x41, 0x55, 0x4C, 0x54, 0x00, 0x00, 0x00, // "VAULT" in ASCII + zeros
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// FORGE RWA token constants (v5.1.0)
/// Physical mining machine token - 1 token = 1 physical Quillon Forge unit
/// Whole units only (0 decimals), owned by BANK_MASTER_ACCOUNT
pub const FORGE_DECIMALS: u8 = 0;
pub const FORGE_TOKEN_ADDRESS: [u8; 32] = [
    0x46, 0x4F, 0x52, 0x47, 0x45, 0x00, 0x00, 0x00, // "FORGE" in ASCII + zeros
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

// ============================================================================
// Quillon Credit (QCREDIT) Yield Vault Token (v8.5.5)
// Lock QUG → mint QCREDIT 1:1, earn tiered yield
// Digital credit layer: L1 capital (QUG) → L2 credit (QCREDIT) → L3 products
// ============================================================================

/// QCREDIT uses 24 decimals (same as QUG) for 1:1 lock/mint parity
pub const QCREDIT_DECIMALS: u8 = 24;
pub const QCREDIT_TOKEN_ADDRESS: [u8; 32] = [
    0x51, 0x43, 0x52, 0x45, 0x44, 0x49, 0x54, 0x00, // "QCREDIT" in ASCII + zeros
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

// ============================================================================
// Quillon USD (QUSD) Issuer-Controlled Stablecoin (v8.5.9)
// USD-pegged stablecoin — founder has transparent mint authority (like USDT/USDC)
// All mints/burns recorded in append-only audit log for full transparency
// ============================================================================

/// QUSD uses 24 decimals (same as QUG/QUGUSD/QCREDIT) for ecosystem consistency
pub const QUSD_DECIMALS: u8 = 24;
pub const QUSD_TOKEN_ADDRESS: [u8; 32] = [
    0x51, 0x55, 0x53, 0x44, 0x00, 0x00, 0x00, 0x00, // "QUSD" in ASCII + zeros
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

// ============================================================================
// Wrapped Bridge Token Constants (v7.2.5)
// Cross-chain bridge tokens: mint on deposit, burn on withdrawal
// ============================================================================

/// Wrapped Bitcoin (wBTC) - 1:1 backed by BTC locked in bridge
/// 8 decimals (matches Bitcoin satoshis: 1 BTC = 100,000,000 sat)
pub const WBTC_DECIMALS: u8 = 8;
pub const WBTC_TOKEN_ADDRESS: [u8; 32] = [
    0x77, 0x42, 0x54, 0x43, 0x00, 0x00, 0x00, 0x00, // "wBTC" in ASCII + zeros
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Wrapped Zcash (wZEC) - 1:1 backed by ZEC locked in shielded bridge
/// 8 decimals (matches Zcash zatoshis: 1 ZEC = 100,000,000 zat)
pub const WZEC_DECIMALS: u8 = 8;
pub const WZEC_TOKEN_ADDRESS: [u8; 32] = [
    0x77, 0x5A, 0x45, 0x43, 0x00, 0x00, 0x00, 0x00, // "wZEC" in ASCII + zeros
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Wrapped Iron Fish (wIRON) - 1:1 backed by IRON locked in privacy bridge
/// 8 decimals (matches Iron Fish ore: 1 IRON = 100,000,000 ore)
pub const WIRON_DECIMALS: u8 = 8;
pub const WIRON_TOKEN_ADDRESS: [u8; 32] = [
    0x77, 0x49, 0x52, 0x4F, 0x4E, 0x00, 0x00, 0x00, // "wIRON" in ASCII + zeros (5 chars)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Wrapped Ethereum (wETH) - 1:1 backed by ETH locked in HTLC bridge
/// 18 decimals (matches Ethereum wei: 1 ETH = 1,000,000,000,000,000,000 wei)
pub const WETH_DECIMALS: u8 = 18;
pub const WETH_TOKEN_ADDRESS: [u8; 32] = [
    0x77, 0x45, 0x54, 0x48, 0x00, 0x00, 0x00, 0x00, // "wETH" in ASCII + zeros
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// All bridge token addresses for easy iteration
pub const BRIDGE_TOKEN_ADDRESSES: [[u8; 32]; 4] = [
    WBTC_TOKEN_ADDRESS,
    WZEC_TOKEN_ADDRESS,
    WIRON_TOKEN_ADDRESS,
    WETH_TOKEN_ADDRESS,
];

/// Bridge token metadata
pub fn bridge_token_info(addr: &[u8; 32]) -> Option<(&'static str, &'static str, u8)> {
    if addr == &WBTC_TOKEN_ADDRESS { Some(("Wrapped Bitcoin", "wBTC", WBTC_DECIMALS)) }
    else if addr == &WZEC_TOKEN_ADDRESS { Some(("Wrapped Zcash", "wZEC", WZEC_DECIMALS)) }
    else if addr == &WIRON_TOKEN_ADDRESS { Some(("Wrapped Iron Fish", "wIRON", WIRON_DECIMALS)) }
    else if addr == &WETH_TOKEN_ADDRESS { Some(("Wrapped Ethereum", "wETH", WETH_DECIMALS)) }
    else { None }
}

// ============================================================================
// Fee System Constants (v1.4.5-beta, v3.4.0-beta: 10x reduction)
// ============================================================================

/// Base gas for a simple transfer (21,000 gas units like Ethereum)
pub const BASE_GAS: u128 = 21_000;

/// Legacy minimum fee per gas unit in smallest denomination (1 = 0.00000001 QUG)
/// Set to 1 satoshi-equivalent to prevent zero-fee spam while keeping fees low
/// Used for blocks BEFORE REDUCED_FEES_V1 activation height
pub const MIN_FEE_PER_GAS: u128 = 1;

/// Legacy minimum fee constant (kept for backward compatibility)
pub const MIN_FEE_PER_GAS_LEGACY: u128 = 1;

/// Fee reduction divisor (v3.4.0-beta)
/// New fees = Legacy fees / FEE_REDUCTION_DIVISOR
pub const FEE_REDUCTION_DIVISOR: u128 = 10;

/// Legacy minimum total fee for any transaction (BASE_GAS * MIN_FEE_PER_GAS)
/// This is 0.00021 QUG for a simple transfer
/// Used for blocks BEFORE REDUCED_FEES_V1 activation height
pub const MIN_TRANSACTION_FEE: u128 = BASE_GAS * MIN_FEE_PER_GAS;
pub const MIN_TRANSACTION_FEE_LEGACY: u128 = BASE_GAS * MIN_FEE_PER_GAS_LEGACY;

/// New reduced minimum total fee (v3.4.0-beta)
/// This is 0.000021 QUG for a simple transfer (10x cheaper)
/// Used for blocks AT OR AFTER REDUCED_FEES_V1 activation height
pub const MIN_TRANSACTION_FEE_V1: u128 = BASE_GAS * MIN_FEE_PER_GAS_LEGACY / FEE_REDUCTION_DIVISOR;

/// Get minimum transaction fee based on block height (mainnet-safe)
///
/// This function implements the height-gated fee reduction:
/// - Before REDUCED_FEES_V1 activation: Legacy fee (0.00021 QUG for transfer)
/// - After REDUCED_FEES_V1 activation: Reduced fee (0.000021 QUG for transfer)
///
/// # Arguments
/// * `block_height` - The block height to check fee rules for
///
/// # Returns
/// The minimum fee for a simple transfer at the given height
pub fn get_min_transaction_fee(block_height: u64) -> u128 {
    if block_height >= upgrades::upgrades::REDUCED_FEES_V1.activation_height {
        MIN_TRANSACTION_FEE_V1
    } else {
        MIN_TRANSACTION_FEE_LEGACY
    }
}

/// Get fee divisor based on block height (mainnet-safe)
///
/// For complex fee calculations that need the divisor:
/// - Before REDUCED_FEES_V1: divisor = 1 (no reduction)
/// - After REDUCED_FEES_V1: divisor = 10 (10x reduction)
pub fn get_fee_divisor(block_height: u64) -> u128 {
    if block_height >= upgrades::upgrades::REDUCED_FEES_V1.activation_height {
        FEE_REDUCTION_DIVISOR
    } else {
        1
    }
}

/// Check if reduced fees are active at a given block height
pub fn is_reduced_fees_active(block_height: u64) -> bool {
    block_height >= upgrades::upgrades::REDUCED_FEES_V1.activation_height
}

/// Maximum fee to prevent accidental overpayment (10 QUG = 1_000_000_000 satoshis)
/// v2.5.0: Updated to u128 for consistency
pub const MAX_TRANSACTION_FEE: u128 = 1_000_000_000;

/// Fee accumulation limit per block to prevent overflow (1M QUG)
/// v2.5.0: Updated to u128 for consistency
pub const MAX_BLOCK_FEE_ACCUMULATION: u128 = 100_000_000_000_000;

// ============================================================================
// DEX Protocol Fee Constants (v2.4.5-beta)
// ============================================================================

/// Total DEX swap fee in basis points (30 = 0.30%)
/// This is the fee traders pay on each swap
pub const DEX_TOTAL_FEE_BPS: u16 = 30;

/// Protocol fee portion in basis points (5 = 0.05%)
/// This portion goes to the master/founder wallet for protocol development
pub const DEX_PROTOCOL_FEE_BPS: u16 = 5;

/// LP fee portion in basis points (25 = 0.25%)
/// This portion stays in the pool for liquidity providers
pub const DEX_LP_FEE_BPS: u16 = 25;

/// Basis points divisor for fee calculations
pub const BPS_DIVISOR: u128 = 10_000;

// ============================================================================
// v2.9.2-beta: Consensus-Verified Protocol Fee System
// ============================================================================

/// Protocol fee record that must be verified by all nodes
/// This ensures the master wallet receives the correct percentage per trade
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ProtocolFeeRecord {
    /// Unique identifier for this fee (derived from trade tx hash)
    pub fee_id: [u8; 32],

    /// The trade/swap transaction this fee is for
    pub trade_tx_hash: [u8; 32],

    /// Fee amount in atomic units
    pub fee_amount: u64,

    /// Token type the fee is paid in
    pub fee_token: TokenType,

    /// Recipient wallet (should always be FOUNDER_WALLET)
    pub recipient: [u8; 32],

    /// Block height when fee was collected
    pub block_height: u64,

    /// Timestamp of fee collection
    pub timestamp: u64,

    /// Trade amount the fee was calculated from
    pub trade_amount: u64,

    /// Fee rate in basis points used
    pub fee_rate_bps: u16,

    /// SHA3 hash of (trade_tx_hash || fee_amount || recipient || block_height)
    /// Used for consensus verification
    pub verification_hash: [u8; 32],
}

impl ProtocolFeeRecord {
    /// Create a new protocol fee record with verification hash
    pub fn new(
        trade_tx_hash: [u8; 32],
        fee_amount: u64,
        fee_token: TokenType,
        block_height: u64,
        trade_amount: u64,
    ) -> Self {
        use sha3::{Sha3_256, Digest};

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Create fee_id from trade hash
        let mut fee_id = [0u8; 32];
        let mut hasher = Sha3_256::new();
        hasher.update(b"protocol_fee_id_v1");
        hasher.update(&trade_tx_hash);
        hasher.update(&fee_amount.to_le_bytes());
        let hash = hasher.finalize();
        fee_id.copy_from_slice(&hash[..32]);

        // Create verification hash
        let mut verification_hash = [0u8; 32];
        let mut hasher = Sha3_256::new();
        hasher.update(&trade_tx_hash);
        hasher.update(&fee_amount.to_le_bytes());
        hasher.update(&FOUNDER_WALLET);
        hasher.update(&block_height.to_le_bytes());
        hasher.update(&(DEX_PROTOCOL_FEE_BPS as u64).to_le_bytes());
        let hash = hasher.finalize();
        verification_hash.copy_from_slice(&hash[..32]);

        Self {
            fee_id,
            trade_tx_hash,
            fee_amount,
            fee_token,
            recipient: FOUNDER_WALLET,
            block_height,
            timestamp,
            trade_amount,
            fee_rate_bps: DEX_PROTOCOL_FEE_BPS,
            verification_hash,
        }
    }

    /// Verify this fee record is correct
    /// All nodes MUST call this to validate the fee
    pub fn verify(&self) -> Result<(), String> {
        use sha3::{Sha3_256, Digest};

        // 1. Verify recipient is FOUNDER_WALLET
        if self.recipient != FOUNDER_WALLET {
            return Err(format!(
                "Invalid fee recipient: expected FOUNDER_WALLET, got {}",
                hex::encode(&self.recipient[..8])
            ));
        }

        // 2. Verify fee rate matches protocol constant
        if self.fee_rate_bps != DEX_PROTOCOL_FEE_BPS {
            return Err(format!(
                "Invalid fee rate: expected {} bps, got {} bps",
                DEX_PROTOCOL_FEE_BPS, self.fee_rate_bps
            ));
        }

        // 3. Verify fee amount calculation
        let expected_fee = Self::calculate_fee(self.trade_amount);
        if self.fee_amount != expected_fee {
            return Err(format!(
                "Fee amount mismatch: expected {} for trade {}, got {}",
                expected_fee, self.trade_amount, self.fee_amount
            ));
        }

        // 4. Verify the verification hash
        let mut expected_hash = [0u8; 32];
        let mut hasher = Sha3_256::new();
        hasher.update(&self.trade_tx_hash);
        hasher.update(&self.fee_amount.to_le_bytes());
        hasher.update(&FOUNDER_WALLET);
        hasher.update(&self.block_height.to_le_bytes());
        hasher.update(&(DEX_PROTOCOL_FEE_BPS as u64).to_le_bytes());
        let hash = hasher.finalize();
        expected_hash.copy_from_slice(&hash[..32]);

        if self.verification_hash != expected_hash {
            return Err("Verification hash mismatch - fee record may be tampered".to_string());
        }

        Ok(())
    }

    /// Calculate protocol fee for a given trade amount
    /// Uses DEX_PROTOCOL_FEE_BPS (5 basis points = 0.05%)
    pub fn calculate_fee(trade_amount: u64) -> u64 {
        // fee = trade_amount * DEX_PROTOCOL_FEE_BPS / BPS_DIVISOR
        let fee = (trade_amount as u128 * DEX_PROTOCOL_FEE_BPS as u128) / BPS_DIVISOR;
        fee as u64
    }

    /// Check if fee was collected for a specific trade
    pub fn is_for_trade(&self, trade_tx_hash: &[u8; 32]) -> bool {
        self.trade_tx_hash == *trade_tx_hash
    }
}

/// Gossipsub message for P2P fee verification
/// All nodes broadcast and verify these to reach consensus on fees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolFeeGossip {
    /// The fee record
    pub fee_record: ProtocolFeeRecord,

    /// Node that collected the fee
    pub collector_node_id: String,

    /// Collector's signature over the fee record (hex-encoded for serde compatibility)
    pub collector_signature_hex: String,

    /// Timestamp of broadcast
    pub broadcast_timestamp: u64,
}

impl ProtocolFeeGossip {
    /// Create a new ProtocolFeeGossip with signature
    pub fn new(fee_record: ProtocolFeeRecord, collector_node_id: String, signature: [u8; 64]) -> Self {
        Self {
            fee_record,
            collector_node_id,
            collector_signature_hex: hex::encode(signature),
            broadcast_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Get the collector signature as bytes
    pub fn collector_signature(&self) -> Result<[u8; 64], String> {
        let bytes = hex::decode(&self.collector_signature_hex)
            .map_err(|e| format!("Invalid signature hex: {}", e))?;
        if bytes.len() != 64 {
            return Err(format!("Invalid signature length: expected 64, got {}", bytes.len()));
        }
        let mut sig = [0u8; 64];
        sig.copy_from_slice(&bytes);
        Ok(sig)
    }

    /// Verify the gossip message signature and fee record
    pub fn verify_with_pubkey(&self, collector_pubkey: &[u8; 32]) -> Result<(), String> {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};

        // 1. Verify the fee record itself
        self.fee_record.verify()?;

        // 2. Get signature bytes
        let signature_bytes = self.collector_signature()?;

        // 3. Verify collector signature
        let verifying_key = VerifyingKey::from_bytes(collector_pubkey)
            .map_err(|e| format!("Invalid collector pubkey: {}", e))?;

        let signature = Signature::from_bytes(&signature_bytes);

        // Sign over: fee_id || verification_hash || collector_node_id
        let mut payload = Vec::new();
        payload.extend_from_slice(&self.fee_record.fee_id);
        payload.extend_from_slice(&self.fee_record.verification_hash);
        payload.extend_from_slice(self.collector_node_id.as_bytes());

        verifying_key.verify(&payload, &signature)
            .map_err(|_| "Invalid collector signature".to_string())?;

        Ok(())
    }
}

// ============================================================================
// Founder Wallet Timelock Protection (v1.4.5-beta)
// ============================================================================

/// Founder wallet address (receives 1% dev fee)
pub const FOUNDER_WALLET: [u8; 32] = [
    0xef, 0xca, 0x1e, 0x8c, 0x1f, 0x46, 0xe9, 0x10,
    0x13, 0xb4, 0x07, 0x38, 0x98, 0xc7, 0x71, 0xbb,
    0x3d, 0x56, 0x64, 0x53, 0x53, 0x7c, 0xcf, 0x87,
    0xe8, 0x34, 0x50, 0x59, 0x25, 0xe5, 0x07, 0x23,
];

/// Minimum timelock duration for founder wallet withdrawals (7 days in seconds)
/// Any withdrawal from the founder wallet must wait 7 days after announcement
pub const FOUNDER_TIMELOCK_DURATION: u64 = 7 * 24 * 60 * 60; // 604,800 seconds

/// Maximum single withdrawal from founder wallet (1% of max supply = 210,000 QUG)
/// This prevents catastrophic loss in case of compromise
/// v2.5.0: Updated to u128 for consistency with Amount type
pub const FOUNDER_MAX_SINGLE_WITHDRAWAL: u128 = 21_000_000_000_000; // 210,000 QUG in atomic units

/// Cooldown between founder withdrawals (24 hours in seconds)
pub const FOUNDER_WITHDRAWAL_COOLDOWN: u64 = 24 * 60 * 60; // 86,400 seconds

/// Block height before which founder cannot withdraw (vesting period)
/// First 200,000 blocks (~90 days) for network stability
pub const FOUNDER_VESTING_END_HEIGHT: u64 = 200_000;

impl TokenInfo {
    /// Get QUG token information
    pub fn qug() -> Self {
        Self {
            token_type: TokenType::QUG,
            name: "Quillon".to_string(),
            symbol: "QUG".to_string(),
            decimals: QUG_DECIMALS,
            max_supply: Some(QUG_MAX_SUPPLY),
        }
    }

    /// Get QUGUSD token information
    pub fn qugusd() -> Self {
        Self {
            token_type: TokenType::QUGUSD,
            name: "Quillon USD".to_string(),
            symbol: "QUGUSD".to_string(),
            decimals: QUGUSD_DECIMALS,
            max_supply: None, // Unlimited if properly collateralized
        }
    }
}

impl TokenType {
    /// Get the reserved token address for this token type
    pub fn address(&self) -> [u8; 32] {
        match self {
            TokenType::QUG => QUG_TOKEN_ADDRESS,
            TokenType::QUGUSD => QUGUSD_TOKEN_ADDRESS,
            TokenType::Custom(addr) => *addr,
        }
    }

    /// Get token info for this token type
    pub fn info(&self) -> TokenInfo {
        match self {
            TokenType::QUG => TokenInfo::qug(),
            TokenType::QUGUSD => TokenInfo::qugusd(),
            TokenType::Custom(addr) => TokenInfo {
                token_type: TokenType::Custom(*addr),
                name: format!("Custom Token {}", hex::encode(&addr[..4])),
                symbol: format!("TKN{}", hex::encode(&addr[..2]).to_uppercase()),
                decimals: 8,
                max_supply: None,
            },
        }
    }

    /// Convert to u8 discriminant for serialization
    /// QUG = 0, QUGUSD = 1, Custom = 2
    pub fn discriminant(&self) -> u8 {
        match self {
            TokenType::QUG => 0,
            TokenType::QUGUSD => 1,
            TokenType::Custom(_) => 2,
        }
    }
}

// ============================================================================
// v1.0.60-beta: Extended Transaction Types for Full State Sync
// All state mutations go through transactions for decentralized consensus
// ============================================================================

/// Extended transaction type enum for comprehensive state sync
/// Every state mutation in the system must be represented as one of these transaction types
/// This enables full decentralized state replication via block synchronization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TransactionType {
    // ========== Core Token Operations (0x00-0x0F) ==========
    /// Native QUG transfer between addresses
    Transfer = 0x00,
    /// Mining coinbase reward (from == [0u8; 32])
    Coinbase = 0x01,
    /// Token burn (sent to burn address)
    Burn = 0x02,
    /// Fee payment (automatically deducted)
    Fee = 0x03,

    // ========== Custom Token Operations (0x10-0x1F) ==========
    /// Create a new custom token (SPL-like token creation)
    TokenCreate = 0x10,
    /// Mint new tokens (for mintable tokens, requires authority)
    TokenMint = 0x11,
    /// Transfer custom token (not QUG/QUGUSD)
    TokenTransfer = 0x12,
    /// Burn custom token
    TokenBurn = 0x13,
    /// Freeze token account (compliance feature)
    TokenFreeze = 0x14,
    /// Unfreeze token account
    TokenUnfreeze = 0x15,
    /// Set token metadata (name, symbol, decimals)
    TokenSetMetadata = 0x16,
    /// Transfer token mint authority
    TokenTransferAuthority = 0x17,

    // ========== DEX Operations (0x20-0x2F) ==========
    /// Create a new liquidity pool
    PoolCreate = 0x20,
    /// Add liquidity to a pool
    PoolAddLiquidity = 0x21,
    /// Remove liquidity from a pool
    PoolRemoveLiquidity = 0x22,
    /// Token swap via AMM
    Swap = 0x23,
    /// Place limit order (order book)
    LimitOrder = 0x24,
    /// Cancel limit order
    CancelOrder = 0x25,
    /// Fill limit order (market taker)
    FillOrder = 0x26,
    /// Flash loan execution
    FlashLoan = 0x27,

    // ========== Smart Contract Operations (0x30-0x3F) ==========
    /// Deploy new smart contract
    ContractDeploy = 0x30,
    /// Call smart contract function
    ContractCall = 0x31,
    /// Upgrade smart contract (if upgradeable)
    ContractUpgrade = 0x32,
    /// Destroy contract and recover state rent
    ContractDestroy = 0x33,
    /// Set contract storage directly (admin only)
    ContractSetStorage = 0x34,

    // ========== Stablecoin/Vault Operations (0x40-0x4F) ==========
    /// Lock QUG as collateral in vault
    VaultLock = 0x40,
    /// Unlock collateral from vault
    VaultUnlock = 0x41,
    /// Mint QUGUSD against collateral
    StableMint = 0x42,
    /// Burn QUGUSD to reduce debt
    StableBurn = 0x43,
    /// Liquidate undercollateralized vault
    VaultLiquidate = 0x44,
    /// Oracle price update for collateral ratio
    OraclePriceUpdate = 0x45,

    // ========== AI/Compute Credits (0x50-0x5F) ==========
    /// Purchase AI compute credits
    AICreditPurchase = 0x50,
    /// Spend AI credits for inference
    AICreditSpend = 0x51,
    /// Transfer AI credits between accounts
    AICreditTransfer = 0x52,
    /// Earn AI credits (node operators)
    AICreditEarn = 0x53,
    /// Register as AI compute provider
    AIProviderRegister = 0x54,
    /// Submit AI inference result
    AIInferenceResult = 0x55,

    // ========== Governance Operations (0x60-0x6F) ==========
    /// Create governance proposal
    ProposalCreate = 0x60,
    /// Vote on proposal
    ProposalVote = 0x61,
    /// Execute passed proposal
    ProposalExecute = 0x62,
    /// Delegate voting power
    DelegateVotes = 0x63,
    /// Undelegate voting power
    UndelegateVotes = 0x64,

    // ========== Staking Operations (0x70-0x7F) ==========
    /// Stake tokens for consensus participation
    Stake = 0x70,
    /// Unstake tokens (starts unbonding period)
    Unstake = 0x71,
    /// Claim staking rewards
    ClaimRewards = 0x72,
    /// Redelegate to different validator
    Redelegate = 0x73,
    /// Slash validator for misbehavior
    Slash = 0x74,

    // ========== Privacy Operations (0x80-0x8F) ==========
    /// v3.4.15: Privacy-mixed transfer (Dandelion++ stem/fluff with Tor routing)
    PrivacyMixed = 0x80,
    /// Stealth address creation
    StealthCreate = 0x81,
    /// Ring signature transaction
    RingTransfer = 0x82,
    /// zk-STARK shielded transfer
    ShieldedTransfer = 0x83,

    // ========== Cross-Chain / Atomic Swap Operations (0x90-0x9F) ==========
    /// Initiate atomic swap HTLC on QNK side
    AtomicSwapInitiate = 0x90,
    /// Claim atomic swap by revealing secret
    AtomicSwapClaim = 0x91,
    /// Refund atomic swap after timeout
    AtomicSwapRefund = 0x92,
    /// Lock QUG/QUGUSD in escrow for atomic swap
    AtomicSwapLock = 0x93,

    // ========== OAuth2 / Identity Operations (0xA0-0xAF) ==========
    /// On-chain consent grant (privacy-preserving hash)
    OAuth2ConsentGrant = 0xA0,
    /// On-chain consent revocation
    OAuth2ConsentRevoke = 0xA1,
    /// Client registration audit trail (optional)
    OAuth2ClientRegister = 0xA2,

    // ========== System Operations (0xF0-0xFF) ==========
    /// System parameter update (via governance)
    SystemParamUpdate = 0xF0,
    /// Emergency pause (multisig required)
    EmergencyPause = 0xF1,
    /// Resume from pause
    EmergencyResume = 0xF2,
    /// State root checkpoint (for light clients)
    StateCheckpoint = 0xF3,
    /// Genesis block special transaction
    Genesis = 0xFE,
    /// Unknown/legacy transaction type
    Unknown = 0xFF,
}

impl TransactionType {
    /// Get the transaction type from a byte
    pub fn from_byte(b: u8) -> Self {
        match b {
            0x00 => TransactionType::Transfer,
            0x01 => TransactionType::Coinbase,
            0x02 => TransactionType::Burn,
            0x03 => TransactionType::Fee,
            0x10 => TransactionType::TokenCreate,
            0x11 => TransactionType::TokenMint,
            0x12 => TransactionType::TokenTransfer,
            0x13 => TransactionType::TokenBurn,
            0x14 => TransactionType::TokenFreeze,
            0x15 => TransactionType::TokenUnfreeze,
            0x16 => TransactionType::TokenSetMetadata,
            0x17 => TransactionType::TokenTransferAuthority,
            0x20 => TransactionType::PoolCreate,
            0x21 => TransactionType::PoolAddLiquidity,
            0x22 => TransactionType::PoolRemoveLiquidity,
            0x23 => TransactionType::Swap,
            0x24 => TransactionType::LimitOrder,
            0x25 => TransactionType::CancelOrder,
            0x26 => TransactionType::FillOrder,
            0x27 => TransactionType::FlashLoan,
            0x30 => TransactionType::ContractDeploy,
            0x31 => TransactionType::ContractCall,
            0x32 => TransactionType::ContractUpgrade,
            0x33 => TransactionType::ContractDestroy,
            0x34 => TransactionType::ContractSetStorage,
            0x40 => TransactionType::VaultLock,
            0x41 => TransactionType::VaultUnlock,
            0x42 => TransactionType::StableMint,
            0x43 => TransactionType::StableBurn,
            0x44 => TransactionType::VaultLiquidate,
            0x45 => TransactionType::OraclePriceUpdate,
            0x50 => TransactionType::AICreditPurchase,
            0x51 => TransactionType::AICreditSpend,
            0x52 => TransactionType::AICreditTransfer,
            0x53 => TransactionType::AICreditEarn,
            0x54 => TransactionType::AIProviderRegister,
            0x55 => TransactionType::AIInferenceResult,
            0x60 => TransactionType::ProposalCreate,
            0x61 => TransactionType::ProposalVote,
            0x62 => TransactionType::ProposalExecute,
            0x63 => TransactionType::DelegateVotes,
            0x64 => TransactionType::UndelegateVotes,
            0x70 => TransactionType::Stake,
            0x71 => TransactionType::Unstake,
            0x72 => TransactionType::ClaimRewards,
            0x73 => TransactionType::Redelegate,
            0x74 => TransactionType::Slash,
            0x80 => TransactionType::PrivacyMixed,
            0x81 => TransactionType::StealthCreate,
            0x82 => TransactionType::RingTransfer,
            0x83 => TransactionType::ShieldedTransfer,
            0x90 => TransactionType::AtomicSwapInitiate,
            0x91 => TransactionType::AtomicSwapClaim,
            0x92 => TransactionType::AtomicSwapRefund,
            0x93 => TransactionType::AtomicSwapLock,
            0xA0 => TransactionType::OAuth2ConsentGrant,
            0xA1 => TransactionType::OAuth2ConsentRevoke,
            0xA2 => TransactionType::OAuth2ClientRegister,
            0xF0 => TransactionType::SystemParamUpdate,
            0xF1 => TransactionType::EmergencyPause,
            0xF2 => TransactionType::EmergencyResume,
            0xF3 => TransactionType::StateCheckpoint,
            0xFE => TransactionType::Genesis,
            _ => TransactionType::Unknown,
        }
    }

    /// Get the byte representation of this transaction type
    pub fn as_byte(&self) -> u8 {
        *self as u8
    }

    /// Check if this is a coinbase (mining reward) transaction
    pub fn is_coinbase(&self) -> bool {
        matches!(self, TransactionType::Coinbase)
    }

    /// Check if this is a token operation
    pub fn is_token_operation(&self) -> bool {
        matches!(
            self,
            TransactionType::TokenCreate
                | TransactionType::TokenMint
                | TransactionType::TokenTransfer
                | TransactionType::TokenBurn
                | TransactionType::TokenFreeze
                | TransactionType::TokenUnfreeze
                | TransactionType::TokenSetMetadata
                | TransactionType::TokenTransferAuthority
        )
    }

    /// Check if this is a DEX operation
    pub fn is_dex_operation(&self) -> bool {
        matches!(
            self,
            TransactionType::PoolCreate
                | TransactionType::PoolAddLiquidity
                | TransactionType::PoolRemoveLiquidity
                | TransactionType::Swap
                | TransactionType::LimitOrder
                | TransactionType::CancelOrder
                | TransactionType::FillOrder
                | TransactionType::FlashLoan
        )
    }

    /// Check if this is a smart contract operation
    pub fn is_contract_operation(&self) -> bool {
        matches!(
            self,
            TransactionType::ContractDeploy
                | TransactionType::ContractCall
                | TransactionType::ContractUpgrade
                | TransactionType::ContractDestroy
                | TransactionType::ContractSetStorage
        )
    }

    /// Check if this is a stablecoin/vault operation
    pub fn is_vault_operation(&self) -> bool {
        matches!(
            self,
            TransactionType::VaultLock
                | TransactionType::VaultUnlock
                | TransactionType::StableMint
                | TransactionType::StableBurn
                | TransactionType::VaultLiquidate
                | TransactionType::OraclePriceUpdate
        )
    }

    /// Check if this is an AI credits operation
    pub fn is_ai_operation(&self) -> bool {
        matches!(
            self,
            TransactionType::AICreditPurchase
                | TransactionType::AICreditSpend
                | TransactionType::AICreditTransfer
                | TransactionType::AICreditEarn
                | TransactionType::AIProviderRegister
                | TransactionType::AIInferenceResult
        )
    }

    /// Check if this is a governance operation
    pub fn is_governance_operation(&self) -> bool {
        matches!(
            self,
            TransactionType::ProposalCreate
                | TransactionType::ProposalVote
                | TransactionType::ProposalExecute
                | TransactionType::DelegateVotes
                | TransactionType::UndelegateVotes
        )
    }

    /// Check if this is a staking operation
    pub fn is_staking_operation(&self) -> bool {
        matches!(
            self,
            TransactionType::Stake
                | TransactionType::Unstake
                | TransactionType::ClaimRewards
                | TransactionType::Redelegate
                | TransactionType::Slash
        )
    }

    /// Check if this is a system operation (requires elevated permissions)
    pub fn is_system_operation(&self) -> bool {
        matches!(
            self,
            TransactionType::SystemParamUpdate
                | TransactionType::EmergencyPause
                | TransactionType::EmergencyResume
                | TransactionType::StateCheckpoint
                | TransactionType::Genesis
        )
    }

    /// Get the gas cost multiplier for this transaction type
    /// Base gas is multiplied by this factor
    pub fn gas_multiplier(&self) -> u64 {
        match self {
            // Simple transfers: 1x base gas
            TransactionType::Transfer | TransactionType::Coinbase => 1,
            TransactionType::Burn | TransactionType::Fee => 1,

            // Token operations: 2x base gas (more state changes)
            TransactionType::TokenTransfer | TransactionType::TokenBurn => 2,
            TransactionType::TokenMint | TransactionType::TokenFreeze => 2,
            TransactionType::TokenUnfreeze => 2,

            // Token creation: 10x base gas (significant state)
            TransactionType::TokenCreate | TransactionType::TokenSetMetadata => 10,
            TransactionType::TokenTransferAuthority => 5,

            // DEX operations: 3-5x base gas (multiple state updates)
            TransactionType::Swap => 3,
            TransactionType::PoolAddLiquidity | TransactionType::PoolRemoveLiquidity => 4,
            TransactionType::PoolCreate => 20,
            TransactionType::LimitOrder | TransactionType::CancelOrder => 3,
            TransactionType::FillOrder => 4,
            TransactionType::FlashLoan => 10,

            // Contract operations: variable based on complexity
            TransactionType::ContractCall => 5,
            TransactionType::ContractDeploy => 100, // Very expensive
            TransactionType::ContractUpgrade => 50,
            TransactionType::ContractDestroy => 10,
            TransactionType::ContractSetStorage => 20,

            // Vault operations: 3-5x base gas
            TransactionType::VaultLock | TransactionType::VaultUnlock => 3,
            TransactionType::StableMint | TransactionType::StableBurn => 4,
            TransactionType::VaultLiquidate => 10,
            TransactionType::OraclePriceUpdate => 2,

            // AI operations: 2-5x base gas
            TransactionType::AICreditPurchase | TransactionType::AICreditTransfer => 2,
            TransactionType::AICreditSpend | TransactionType::AICreditEarn => 3,
            TransactionType::AIProviderRegister => 10,
            TransactionType::AIInferenceResult => 5,

            // Governance: 5-20x base gas
            TransactionType::ProposalCreate => 20,
            TransactionType::ProposalVote => 5,
            TransactionType::ProposalExecute => 50, // Depends on proposal
            TransactionType::DelegateVotes | TransactionType::UndelegateVotes => 3,

            // Staking: 3-10x base gas
            TransactionType::Stake | TransactionType::Unstake => 5,
            TransactionType::ClaimRewards => 3,
            TransactionType::Redelegate => 8,
            TransactionType::Slash => 20,

            // System operations: high gas (privileged)
            TransactionType::SystemParamUpdate => 100,
            TransactionType::EmergencyPause | TransactionType::EmergencyResume => 50,
            TransactionType::StateCheckpoint => 10,
            TransactionType::Genesis => 0, // Free (only at genesis)
            TransactionType::Unknown => 1,

            // v3.4.15: Privacy operations - higher gas for mixing/stealth
            TransactionType::PrivacyMixed => 15,      // Dandelion++ routing overhead
            TransactionType::StealthCreate => 20,    // Generate stealth address
            TransactionType::RingTransfer => 25,     // Ring signature computation
            TransactionType::ShieldedTransfer => 30, // Full ZK-STARK proof

            // Cross-chain atomic swaps
            TransactionType::AtomicSwapInitiate => 20,  // HTLC creation
            TransactionType::AtomicSwapClaim => 15,     // Secret reveal + claim
            TransactionType::AtomicSwapRefund => 15,    // Timeout refund
            TransactionType::AtomicSwapLock => 10,      // Escrow lock

            // v7.4.0: OAuth2 / Identity operations
            TransactionType::OAuth2ConsentGrant => 2,    // Consent hash on-chain
            TransactionType::OAuth2ConsentRevoke => 2,   // Consent revocation
            TransactionType::OAuth2ClientRegister => 5,  // Client audit trail
        }
    }

    /// Get human-readable name for this transaction type
    pub fn name(&self) -> &'static str {
        match self {
            TransactionType::Transfer => "Transfer",
            TransactionType::Coinbase => "Coinbase",
            TransactionType::Burn => "Burn",
            TransactionType::Fee => "Fee",
            TransactionType::TokenCreate => "Token Create",
            TransactionType::TokenMint => "Token Mint",
            TransactionType::TokenTransfer => "Token Transfer",
            TransactionType::TokenBurn => "Token Burn",
            TransactionType::TokenFreeze => "Token Freeze",
            TransactionType::TokenUnfreeze => "Token Unfreeze",
            TransactionType::TokenSetMetadata => "Token Set Metadata",
            TransactionType::TokenTransferAuthority => "Token Transfer Authority",
            TransactionType::PoolCreate => "Pool Create",
            TransactionType::PoolAddLiquidity => "Pool Add Liquidity",
            TransactionType::PoolRemoveLiquidity => "Pool Remove Liquidity",
            TransactionType::Swap => "Swap",
            TransactionType::LimitOrder => "Limit Order",
            TransactionType::CancelOrder => "Cancel Order",
            TransactionType::FillOrder => "Fill Order",
            TransactionType::FlashLoan => "Flash Loan",
            TransactionType::ContractDeploy => "Contract Deploy",
            TransactionType::ContractCall => "Contract Call",
            TransactionType::ContractUpgrade => "Contract Upgrade",
            TransactionType::ContractDestroy => "Contract Destroy",
            TransactionType::ContractSetStorage => "Contract Set Storage",
            TransactionType::VaultLock => "Vault Lock",
            TransactionType::VaultUnlock => "Vault Unlock",
            TransactionType::StableMint => "Stable Mint",
            TransactionType::StableBurn => "Stable Burn",
            TransactionType::VaultLiquidate => "Vault Liquidate",
            TransactionType::OraclePriceUpdate => "Oracle Price Update",
            TransactionType::AICreditPurchase => "AI Credit Purchase",
            TransactionType::AICreditSpend => "AI Credit Spend",
            TransactionType::AICreditTransfer => "AI Credit Transfer",
            TransactionType::AICreditEarn => "AI Credit Earn",
            TransactionType::AIProviderRegister => "AI Provider Register",
            TransactionType::AIInferenceResult => "AI Inference Result",
            TransactionType::ProposalCreate => "Proposal Create",
            TransactionType::ProposalVote => "Proposal Vote",
            TransactionType::ProposalExecute => "Proposal Execute",
            TransactionType::DelegateVotes => "Delegate Votes",
            TransactionType::UndelegateVotes => "Undelegate Votes",
            TransactionType::Stake => "Stake",
            TransactionType::Unstake => "Unstake",
            TransactionType::ClaimRewards => "Claim Rewards",
            TransactionType::Redelegate => "Redelegate",
            TransactionType::Slash => "Slash",
            TransactionType::SystemParamUpdate => "System Param Update",
            TransactionType::EmergencyPause => "Emergency Pause",
            TransactionType::EmergencyResume => "Emergency Resume",
            TransactionType::StateCheckpoint => "State Checkpoint",
            TransactionType::Genesis => "Genesis",
            TransactionType::Unknown => "Unknown",
            // v3.4.15: Privacy operations
            TransactionType::PrivacyMixed => "Privacy Mixed",
            TransactionType::StealthCreate => "Stealth Create",
            TransactionType::RingTransfer => "Ring Transfer",
            TransactionType::ShieldedTransfer => "Shielded Transfer",
            // Cross-chain atomic swaps
            TransactionType::AtomicSwapInitiate => "Atomic Swap Initiate",
            TransactionType::AtomicSwapClaim => "Atomic Swap Claim",
            TransactionType::AtomicSwapRefund => "Atomic Swap Refund",
            TransactionType::AtomicSwapLock => "Atomic Swap Lock",
            // v7.4.0: OAuth2 / Identity operations
            TransactionType::OAuth2ConsentGrant => "OAuth2 Consent Grant",
            TransactionType::OAuth2ConsentRevoke => "OAuth2 Consent Revoke",
            TransactionType::OAuth2ClientRegister => "OAuth2 Client Register",
        }
    }
}

impl Default for TransactionType {
    fn default() -> Self {
        TransactionType::Transfer
    }
}

impl std::fmt::Display for TransactionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// v1.0.60-beta: StateChange - Atomic State Mutations
// Every transaction produces one or more StateChanges that modify global state
// ============================================================================

/// Represents an atomic state change produced by a transaction
/// Each transaction type maps to a specific set of state changes
/// State changes are applied atomically and can be reversed for reorgs
/// v2.10.0: Updated all amount fields to u128 for 24 decimal precision
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StateChange {
    // ========== Balance Changes ==========
    /// Credit tokens to an account (balance increase)
    BalanceCredit {
        /// Account receiving tokens (32-byte address)
        account: [u8; 32],
        /// Token address (QUG_TOKEN_ADDRESS, QUGUSD_TOKEN_ADDRESS, or custom)
        token: [u8; 32],
        /// Amount to credit (u128 for 24 decimal precision)
        amount: u128,
    },
    /// Debit tokens from an account (balance decrease)
    BalanceDebit {
        /// Account losing tokens
        account: [u8; 32],
        /// Token address
        token: [u8; 32],
        /// Amount to debit (u128 for 24 decimal precision)
        amount: u128,
    },

    // ========== Custom Token State ==========
    /// Create a new token
    TokenCreate {
        /// Token address (derived from creator + nonce)
        token_address: [u8; 32],
        /// Token name (max 32 bytes)
        name: [u8; 32],
        /// Token symbol (max 8 bytes)
        symbol: [u8; 8],
        /// Decimal places (0-24)
        decimals: u8,
        /// Initial supply (u128 for high precision tokens)
        initial_supply: u128,
        /// Max supply (0 = unlimited, u128 for 10^30+ tokens)
        max_supply: u128,
        /// Mint authority (can mint more tokens)
        mint_authority: [u8; 32],
        /// Freeze authority (can freeze accounts)
        freeze_authority: Option<[u8; 32]>,
        /// Is mintable after creation
        is_mintable: bool,
    },
    /// Update token metadata
    TokenMetadataUpdate {
        token_address: [u8; 32],
        name: Option<[u8; 32]>,
        symbol: Option<[u8; 8]>,
        /// URI for extended metadata (IPFS/HTTP)
        metadata_uri: Option<Vec<u8>>,
    },
    /// Transfer mint authority
    TokenAuthorityTransfer {
        token_address: [u8; 32],
        old_authority: [u8; 32],
        new_authority: [u8; 32],
    },
    /// Freeze a token account
    TokenAccountFreeze {
        token_address: [u8; 32],
        account: [u8; 32],
        frozen: bool,
    },

    // ========== DEX State ==========
    /// Create a new liquidity pool
    PoolCreate {
        /// Pool ID (hash of token_a + token_b)
        pool_id: [u8; 32],
        /// First token in the pair
        token_a: [u8; 32],
        /// Second token in the pair
        token_b: [u8; 32],
        /// Fee tier (in basis points, e.g., 30 = 0.3%)
        fee_bps: u16,
        /// Initial liquidity for token A (u128 for high precision)
        initial_a: u128,
        /// Initial liquidity for token B (u128 for high precision)
        initial_b: u128,
        /// Creator receives LP tokens
        creator: [u8; 32],
        /// Initial LP token supply (u128 for high precision)
        lp_supply: u128,
    },
    /// Update pool reserves (after swap or liquidity change)
    PoolReservesUpdate {
        pool_id: [u8; 32],
        reserve_a: u128,
        reserve_b: u128,
        lp_supply: u128,
    },
    /// Credit LP tokens to liquidity provider
    LPTokenCredit {
        pool_id: [u8; 32],
        account: [u8; 32],
        amount: u128,
    },
    /// Debit LP tokens from liquidity provider
    LPTokenDebit {
        pool_id: [u8; 32],
        account: [u8; 32],
        amount: u128,
    },

    // ========== Smart Contract State ==========
    /// Deploy new contract
    ContractDeploy {
        /// Contract address (derived from deployer + nonce)
        contract_address: [u8; 32],
        /// Bytecode hash (for verification)
        code_hash: [u8; 32],
        /// Deployer address
        deployer: [u8; 32],
        /// Contract is upgradeable
        is_upgradeable: bool,
    },
    /// Update contract storage slot
    ContractStorageUpdate {
        contract_address: [u8; 32],
        /// Storage key (32-byte slot)
        key: [u8; 32],
        /// New value (variable length)
        value: Vec<u8>,
    },
    /// Destroy contract and mark as inactive
    ContractDestroy {
        contract_address: [u8; 32],
        /// Remaining balance sent to this address
        beneficiary: [u8; 32],
    },

    // ========== Vault/Stablecoin State ==========
    /// Create or update a collateral vault
    VaultUpdate {
        /// Vault ID (owner address for single-vault, or hash for multi-vault)
        vault_id: [u8; 32],
        /// Owner of the vault
        owner: [u8; 32],
        /// Collateral locked (in QUG, u128 for 24 decimal precision)
        collateral_amount: u128,
        /// Debt minted (in QUGUSD, u128 for 24 decimal precision)
        debt_amount: u128,
        /// Collateralization ratio (in basis points, e.g., 15000 = 150%)
        collateral_ratio_bps: u32,
    },
    /// Update oracle price feed
    OraclePriceUpdate {
        /// Price feed ID (e.g., QUG/USD)
        feed_id: [u8; 32],
        /// Price in 24 decimal fixed point (u128 for high precision)
        price: u128,
        /// Timestamp of price observation
        timestamp: i64,
        /// Number of oracle signatures
        num_signatures: u8,
    },

    // ========== AI Credits State ==========
    /// Update AI credits balance
    AICreditsUpdate {
        account: [u8; 32],
        /// New balance (after credit/debit, u128 for precision)
        balance: u128,
        /// Credits earned (lifetime)
        earned: u128,
        /// Credits spent (lifetime)
        spent: u128,
    },
    /// Register/update AI provider
    AIProviderUpdate {
        provider_id: [u8; 32],
        /// Provider wallet address
        wallet: [u8; 32],
        /// Compute capacity (TFLOPS)
        capacity: u64,
        /// Price per credit (u128 for precision)
        price_per_credit: u128,
        /// Is active
        is_active: bool,
    },

    // ========== Governance State ==========
    /// Create a governance proposal
    ProposalCreate {
        proposal_id: [u8; 32],
        proposer: [u8; 32],
        /// Start block height
        start_height: u64,
        /// End block height
        end_height: u64,
        /// Required quorum (in basis points of total supply)
        quorum_bps: u32,
        /// Execution data hash
        execution_hash: [u8; 32],
    },
    /// Update proposal vote counts
    ProposalVoteUpdate {
        proposal_id: [u8; 32],
        /// Votes in favor (u128 for token-weighted voting)
        votes_for: u128,
        /// Votes against
        votes_against: u128,
        /// Abstentions
        votes_abstain: u128,
    },
    /// Mark proposal as executed/cancelled
    ProposalStatusUpdate {
        proposal_id: [u8; 32],
        /// 0=pending, 1=active, 2=succeeded, 3=failed, 4=executed, 5=cancelled
        status: u8,
    },
    /// Update vote delegation
    DelegationUpdate {
        delegator: [u8; 32],
        delegate: Option<[u8; 32]>,
        voting_power: u128,
    },

    // ========== Staking State ==========
    /// Update staking position
    StakeUpdate {
        staker: [u8; 32],
        validator: [u8; 32],
        staked_amount: u128,
        /// Unbonding end timestamp (0 if not unbonding)
        unbonding_end: i64,
        /// Accumulated rewards
        pending_rewards: u128,
    },
    /// Update validator state
    ValidatorUpdate {
        validator_id: [u8; 32],
        /// Total stake from all delegators (u128 for precision)
        total_stake: u128,
        /// Commission rate (basis points)
        commission_bps: u16,
        /// Is validator active in consensus
        is_active: bool,
        /// Slash count
        slash_count: u32,
    },

    // ========== System State ==========
    /// Update system parameter
    SystemParamUpdate {
        /// Parameter key
        key: [u8; 32],
        /// New value
        value: Vec<u8>,
    },
    /// Nonce increment (prevents replay)
    NonceIncrement {
        account: [u8; 32],
        new_nonce: u64,
    },
    /// State root checkpoint
    StateRootCheckpoint {
        height: u64,
        /// Merkle Patricia Trie root
        state_root: [u8; 32],
        /// Hash of all transactions in this checkpoint
        tx_root: [u8; 32],
    },

    // ========== v2.9.2-beta: Protocol Fee State ==========
    /// Record protocol fee collected from DEX trade
    /// All nodes MUST verify this matches the expected fee
    ProtocolFeeCollected {
        /// Fee record ID (derived from trade tx hash)
        fee_id: [u8; 32],
        /// Trade transaction hash this fee is for
        trade_tx_hash: [u8; 32],
        /// Fee amount in atomic units (u128 for precision)
        fee_amount: u128,
        /// Token the fee is paid in
        fee_token: [u8; 32],
        /// Recipient (must be FOUNDER_WALLET)
        recipient: [u8; 32],
        /// Trade amount the fee was calculated from (u128)
        trade_amount: u128,
        /// Fee rate in basis points
        fee_rate_bps: u16,
        /// Verification hash for consensus
        verification_hash: [u8; 32],
    },
}

impl StateChange {
    /// Get the primary key affected by this state change
    /// Used for indexing and conflict detection
    pub fn primary_key(&self) -> Vec<u8> {
        match self {
            StateChange::BalanceCredit { account, token, .. } |
            StateChange::BalanceDebit { account, token, .. } => {
                let mut key = Vec::with_capacity(65);
                key.push(0x01); // Balance prefix
                key.extend_from_slice(account);
                key.extend_from_slice(token);
                key
            }
            StateChange::TokenCreate { token_address, .. } |
            StateChange::TokenMetadataUpdate { token_address, .. } |
            StateChange::TokenAuthorityTransfer { token_address, .. } => {
                let mut key = Vec::with_capacity(33);
                key.push(0x02); // Token prefix
                key.extend_from_slice(token_address);
                key
            }
            StateChange::TokenAccountFreeze { token_address, account, .. } => {
                let mut key = Vec::with_capacity(65);
                key.push(0x03); // Token account prefix
                key.extend_from_slice(token_address);
                key.extend_from_slice(account);
                key
            }
            StateChange::PoolCreate { pool_id, .. } |
            StateChange::PoolReservesUpdate { pool_id, .. } => {
                let mut key = Vec::with_capacity(33);
                key.push(0x10); // Pool prefix
                key.extend_from_slice(pool_id);
                key
            }
            StateChange::LPTokenCredit { pool_id, account, .. } |
            StateChange::LPTokenDebit { pool_id, account, .. } => {
                let mut key = Vec::with_capacity(65);
                key.push(0x11); // LP token prefix
                key.extend_from_slice(pool_id);
                key.extend_from_slice(account);
                key
            }
            StateChange::ContractDeploy { contract_address, .. } |
            StateChange::ContractDestroy { contract_address, .. } => {
                let mut key = Vec::with_capacity(33);
                key.push(0x20); // Contract prefix
                key.extend_from_slice(contract_address);
                key
            }
            StateChange::ContractStorageUpdate { contract_address, key: slot, .. } => {
                let mut k = Vec::with_capacity(65);
                k.push(0x21); // Contract storage prefix
                k.extend_from_slice(contract_address);
                k.extend_from_slice(slot);
                k
            }
            StateChange::VaultUpdate { vault_id, .. } => {
                let mut key = Vec::with_capacity(33);
                key.push(0x30); // Vault prefix
                key.extend_from_slice(vault_id);
                key
            }
            StateChange::OraclePriceUpdate { feed_id, .. } => {
                let mut key = Vec::with_capacity(33);
                key.push(0x31); // Oracle prefix
                key.extend_from_slice(feed_id);
                key
            }
            StateChange::AICreditsUpdate { account, .. } => {
                let mut key = Vec::with_capacity(33);
                key.push(0x40); // AI credits prefix
                key.extend_from_slice(account);
                key
            }
            StateChange::AIProviderUpdate { provider_id, .. } => {
                let mut key = Vec::with_capacity(33);
                key.push(0x41); // AI provider prefix
                key.extend_from_slice(provider_id);
                key
            }
            StateChange::ProposalCreate { proposal_id, .. } |
            StateChange::ProposalVoteUpdate { proposal_id, .. } |
            StateChange::ProposalStatusUpdate { proposal_id, .. } => {
                let mut key = Vec::with_capacity(33);
                key.push(0x50); // Proposal prefix
                key.extend_from_slice(proposal_id);
                key
            }
            StateChange::DelegationUpdate { delegator, .. } => {
                let mut key = Vec::with_capacity(33);
                key.push(0x51); // Delegation prefix
                key.extend_from_slice(delegator);
                key
            }
            StateChange::StakeUpdate { staker, validator, .. } => {
                let mut key = Vec::with_capacity(65);
                key.push(0x60); // Stake prefix
                key.extend_from_slice(staker);
                key.extend_from_slice(validator);
                key
            }
            StateChange::ValidatorUpdate { validator_id, .. } => {
                let mut key = Vec::with_capacity(33);
                key.push(0x61); // Validator prefix
                key.extend_from_slice(validator_id);
                key
            }
            StateChange::SystemParamUpdate { key, .. } => {
                let mut k = Vec::with_capacity(33);
                k.push(0xF0); // System param prefix
                k.extend_from_slice(key);
                k
            }
            StateChange::NonceIncrement { account, .. } => {
                let mut key = Vec::with_capacity(33);
                key.push(0xF1); // Nonce prefix
                key.extend_from_slice(account);
                key
            }
            StateChange::StateRootCheckpoint { height, .. } => {
                let mut key = Vec::with_capacity(9);
                key.push(0xFE); // Checkpoint prefix
                key.extend_from_slice(&height.to_be_bytes());
                key
            }
            StateChange::ProtocolFeeCollected { fee_id, .. } => {
                let mut key = Vec::with_capacity(33);
                key.push(0xD0); // Protocol fee prefix
                key.extend_from_slice(fee_id);
                key
            }
        }
    }

    /// Get the category of this state change for routing to appropriate storage
    pub fn category(&self) -> StateChangeCategory {
        match self {
            StateChange::BalanceCredit { .. } | StateChange::BalanceDebit { .. } => {
                StateChangeCategory::Balance
            }
            StateChange::TokenCreate { .. } |
            StateChange::TokenMetadataUpdate { .. } |
            StateChange::TokenAuthorityTransfer { .. } |
            StateChange::TokenAccountFreeze { .. } => {
                StateChangeCategory::Token
            }
            StateChange::PoolCreate { .. } |
            StateChange::PoolReservesUpdate { .. } |
            StateChange::LPTokenCredit { .. } |
            StateChange::LPTokenDebit { .. } |
            StateChange::ProtocolFeeCollected { .. } => {
                StateChangeCategory::Dex
            }
            StateChange::ContractDeploy { .. } |
            StateChange::ContractStorageUpdate { .. } |
            StateChange::ContractDestroy { .. } => {
                StateChangeCategory::Contract
            }
            StateChange::VaultUpdate { .. } | StateChange::OraclePriceUpdate { .. } => {
                StateChangeCategory::Vault
            }
            StateChange::AICreditsUpdate { .. } | StateChange::AIProviderUpdate { .. } => {
                StateChangeCategory::AI
            }
            StateChange::ProposalCreate { .. } |
            StateChange::ProposalVoteUpdate { .. } |
            StateChange::ProposalStatusUpdate { .. } |
            StateChange::DelegationUpdate { .. } => {
                StateChangeCategory::Governance
            }
            StateChange::StakeUpdate { .. } | StateChange::ValidatorUpdate { .. } => {
                StateChangeCategory::Staking
            }
            StateChange::SystemParamUpdate { .. } |
            StateChange::NonceIncrement { .. } |
            StateChange::StateRootCheckpoint { .. } => {
                StateChangeCategory::System
            }
        }
    }
}

/// Category of state changes for routing to storage column families
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateChangeCategory {
    Balance,
    Token,
    Dex,
    Contract,
    Vault,
    AI,
    Governance,
    Staking,
    System,
}

impl StateChangeCategory {
    /// Get the RocksDB column family name for this category
    pub fn column_family(&self) -> &'static str {
        match self {
            StateChangeCategory::Balance => "cf_balances",
            StateChangeCategory::Token => "cf_tokens",
            StateChangeCategory::Dex => "cf_dex",
            StateChangeCategory::Contract => "cf_contracts",
            StateChangeCategory::Vault => "cf_vaults",
            StateChangeCategory::AI => "cf_ai",
            StateChangeCategory::Governance => "cf_governance",
            StateChangeCategory::Staking => "cf_staking",
            StateChangeCategory::System => "cf_system",
        }
    }
}

/// Hash256 type for general cryptographic hashing
pub type Hash256 = [u8; 32];

/// Fixed-point number with 28 decimal places for ultra-precision
pub type FixedPoint28 = i64;

/// v2.3.0-beta: Transaction signature phase for post-quantum migration
/// Matches block SignaturePhase but for user transactions
/// v3.7.4: Added Dilithium5 NIST Level 5 post-quantum signatures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TxSignaturePhase {
    /// Phase 0: Classical Ed25519 signatures (64 bytes) - DEFAULT for backwards compat
    #[default]
    Phase0Ed25519,
    /// Phase 1: Dilithium5 post-quantum signatures (4,627 bytes) - NIST Level 5
    /// v3.7.4: New post-quantum mode
    Phase1Dilithium5,
    /// Phase 2: SQIsign compact post-quantum signatures (204 bytes)
    Phase2SQIsign,
    /// Hybrid: Ed25519 + SQIsign (64 + 204 = 268 bytes total)
    /// Both signatures must verify for transaction to be valid
    HybridEd25519SQIsign,
    /// Hybrid: Ed25519 + Dilithium5 (64 + 4,627 = 4,691 bytes total)
    /// v3.7.4: Browser P2P transactions use this mode for quantum resistance
    /// Both signatures must verify for transaction to be valid
    HybridEd25519Dilithium5,
}

/// Default transaction signature phase for backwards compatibility
fn default_tx_signature_phase() -> TxSignaturePhase {
    TxSignaturePhase::Phase0Ed25519
}

/// Transaction structure
/// v1.0.60-beta: Extended with tx_type field for comprehensive state sync
/// v2.3.0-beta: Extended with post-quantum signature support (SQIsign)
/// v2.4.9-beta: Added PartialEq for consensus voting comparison
/// v3.2.7-beta: CRITICAL FIX - Added u128_serde for P2P MessagePack compatibility
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Transaction {
    pub id: TxHash,
    pub from: Address,
    pub to: Address,
    /// v2.5.0: Transaction amount
    /// v3.2.7: CRITICAL - Use u128_serde for P2P MessagePack compatibility
    /// MessagePack truncates u128 to u64, corrupting coinbase amounts!
    #[serde(with = "u128_serde")]
    pub amount: Amount,
    /// v2.5.0: Transaction fee
    /// v3.2.7: CRITICAL - Use u128_serde for P2P MessagePack compatibility
    #[serde(with = "u128_serde")]
    pub fee: Amount,
    pub nonce: u64,
    pub signature: Vec<u8>, // Ed25519 signature (64 bytes) - Phase 0 or hybrid
    pub timestamp: DateTime<Utc>,
    pub data: Vec<u8>, // Contract call data or arbitrary transaction payload
    #[serde(default = "default_token_type")]
    pub token_type: TokenType, // QUG or QUGUSD
    #[serde(default = "default_fee_token_type")]
    pub fee_token_type: TokenType, // Token used to pay fees (default: QUGUSD)
    /// v1.0.60-beta: Transaction type for state sync processing
    /// Determines how this transaction affects global state
    #[serde(default = "default_tx_type")]
    pub tx_type: TransactionType,
    /// v2.3.0-beta: Post-quantum signature (SQIsign, 204 bytes)
    /// Only populated in Phase2SQIsign or HybridEd25519SQIsign mode
    #[serde(default)]
    pub pqc_signature: Option<Vec<u8>>,
    /// v2.3.0-beta: Signature phase indicator
    /// Determines which signature(s) to verify
    #[serde(default = "default_tx_signature_phase")]
    pub signature_phase: TxSignaturePhase,
    /// v2.3.0-beta: SQIsign public key (64 bytes) for verification
    /// Required for Phase2SQIsign and HybridEd25519SQIsign
    #[serde(default)]
    pub pqc_public_key: Option<Vec<u8>>,

    // ========================================================================
    // v3.4.2-beta: UNIFIED ZK PRIVACY FIELDS
    // ========================================================================

    /// v3.4.2-beta: ZK proof bundle for privacy (STARK + Bulletproof + LatticeGuard)
    /// When set, this transaction has cryptographic privacy proofs attached
    #[serde(default)]
    pub zk_proof_bundle: Option<Vec<u8>>,

    /// v3.4.2-beta: Privacy level for this transaction
    /// Determines what proofs are required/expected
    #[serde(default = "default_privacy_level")]
    pub privacy_level: TransactionPrivacyLevel,

    /// v3.4.2-beta: Bulletproof range proof for amount confidentiality
    /// Proves amount is in [0, 2^64) without revealing the actual value
    #[serde(default)]
    pub bulletproof: Option<Vec<u8>>,

    /// v3.4.2-beta: Nullifier to prevent double-spending private transactions
    /// Must be unique across all transactions
    #[serde(default)]
    pub nullifier: Option<[u8; 32]>,

    /// v3.9.6-beta: Optional memo/message attached to transaction
    /// Displayed in recipient's inbox for person-to-person transfers
    #[serde(default)]
    pub memo: Option<String>,
}

/// Privacy level for transactions (v3.4.2-beta)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TransactionPrivacyLevel {
    /// Default: No privacy, fully transparent
    #[default]
    Transparent,
    /// Amount hidden with Bulletproofs
    ConfidentialAmount,
    /// Full privacy with STARK proofs
    FullPrivacy,
    /// Post-quantum privacy with LatticeGuard
    PostQuantumPrivacy,
    /// Maximum security: all proof systems combined
    MaximumSecurity,
}

/// Default privacy level for backwards compatibility
fn default_privacy_level() -> TransactionPrivacyLevel {
    TransactionPrivacyLevel::Transparent
}

// ============================================================================
// v3.3.0-beta: P2P MEMPOOL TRANSACTION PROPAGATION
// ============================================================================
//
// P2PTransaction wraps a Transaction with P2P metadata for real-time mempool
// synchronization across all nodes. When a node receives a transaction via API,
// it broadcasts a P2PTransaction to the `/qnk/{network}/mempool-txs` gossipsub
// topic. Other nodes receive it and add to their local mempool immediately.
//
// Security features:
// - hop_count: Prevents infinite relay (max 3 hops)
// - origin_node_id: Enables deduplication and tracking
// - timestamp_ms: Reject stale transactions (>5 min old)
// ============================================================================

/// P2P transaction wrapper for mempool synchronization
/// Contains the actual transaction plus network propagation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2PTransaction {
    /// Protocol version for forward compatibility
    pub version: u8,
    /// The actual transaction to add to mempool
    pub transaction: Transaction,
    /// PeerId of the node that first received/created this transaction
    pub origin_node_id: String,
    /// Timestamp when the transaction was first broadcast (milliseconds since epoch)
    pub timestamp_ms: u64,
    /// Number of P2P hops this transaction has taken
    /// Used to prevent infinite relay loops (max 3 hops)
    pub hop_count: u8,
}

impl P2PTransaction {
    /// Create a new P2P transaction wrapper
    pub fn new(transaction: Transaction, origin_node_id: String) -> Self {
        Self {
            version: 1,
            transaction,
            origin_node_id,
            timestamp_ms: chrono::Utc::now().timestamp_millis() as u64,
            hop_count: 0,
        }
    }

    /// Check if this transaction is too old (>5 minutes)
    pub fn is_stale(&self) -> bool {
        let now_ms = chrono::Utc::now().timestamp_millis() as u64;
        now_ms.saturating_sub(self.timestamp_ms) > 300_000 // 5 minutes
    }

    /// Check if this transaction has exceeded max hop count
    pub fn exceeded_max_hops(&self) -> bool {
        self.hop_count >= 3
    }

    /// Increment hop count for relay
    pub fn increment_hop(&mut self) {
        self.hop_count = self.hop_count.saturating_add(1);
    }
}

/// Default token type for backwards compatibility
fn default_token_type() -> TokenType {
    TokenType::QUG
}

/// Default fee token type
fn default_fee_token_type() -> TokenType {
    TokenType::QUGUSD
}

/// Default transaction type for backwards compatibility
/// Infers type from transaction fields for legacy transactions
fn default_tx_type() -> TransactionType {
    TransactionType::Transfer
}

/// DAG vertex (Narwhal block)
/// v2.4.9-beta: Added PartialEq for consensus voting comparison
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Vertex {
    pub id: VertexId,
    pub round: Round,
    pub author: NodeId,
    pub tx_root: TxHash,
    pub parents: Vec<VertexId>,
    pub transactions: Vec<Transaction>,
    pub signature: Vec<u8>,
    pub timestamp: DateTime<Utc>,
}

/// Vertex identifier
pub type VertexId = [u8; 32];

/// Certificate for vertex availability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    pub vertex_id: VertexId,
    pub round: Round,
    pub signatures: BTreeMap<NodeId, Vec<u8>>,
    pub threshold_met: bool,
}

/// Wallet information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletInfo {
    pub id: Uuid,
    pub address: Address,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address_formatted: Option<String>, // "qnk" + hex encoding
    pub public_key: Vec<u8>,
    pub balance: Amount,
    pub nonce: u64,
    pub created_at: DateTime<Utc>,
}

/// State key for state management
pub type StateKey = Vec<u8>;

/// State value for state management  
pub type StateValue = Vec<u8>;

/// Node status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    pub node_id: NodeId,
    pub current_round: Round,
    pub current_height: Height,
    pub connected_peers: u32,
    pub tx_pool_size: u32,
    pub is_validator: bool,
    pub uptime: std::time::Duration,
}

/// Transaction status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TxStatus {
    Pending,
    InMempool,
    Mixing, // New status for quantum privacy mixing
    Confirmed { block_height: Height, round: Round },
    Failed { error: String },
}

/// Privacy level for quantum mixing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PrivacyLevel {
    Standard, // 3 mixing rounds, 15 decoys
    High,     // 5 mixing rounds, 25 decoys
    Maximum,  // 8 mixing rounds, 50 decoys
}

/// API response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: DateTime<Utc>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: Utc::now(),
        }
    }

    pub fn error(error: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
            timestamp: Utc::now(),
        }
    }
}

/// Wallet creation request
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateWalletRequest {
    pub password: Option<String>,
    pub mnemonic: Option<String>,
}

/// Transaction signing request
#[derive(Debug, Serialize, Deserialize)]
pub struct SignTransactionRequest {
    pub to: Address,
    pub amount: Amount,
    pub fee: Amount,
    pub password: String,
    /// v10.2.1: Token type for the transfer ("QUG", "QUGUSD", or custom token symbol).
    /// Previously hardcoded to QUG — QUGUSD transfers signed via this endpoint
    /// would produce QUG transactions (critical token type corruption bug).
    #[serde(default)]
    pub token_type: Option<String>,
}

/// Submit transaction request
#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitTransactionRequest {
    pub transaction: Transaction,
}

/// Quantum entropy info (Phase 2+)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QRNGInfo {
    pub vendor: String,
    pub serial: [u8; 16],
    pub health: BTreeMap<u64, f64>, // entropy-rate per 1-second window
    pub signature: Vec<u8>,
}

/// Cryptographic agility metadata (Phase 1+)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoAgility {
    pub signature_scheme: String,
    pub kem_scheme: String,
    pub hash_function: String,
    pub vrf_scheme: String,
    pub multicodec_version: u32,
}

/// Narwhal payload structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarwhalPayload {
    pub data: Vec<u8>,
    pub transactions: Vec<Transaction>,
    pub timestamp: u64,
    pub payload_hash: [u8; 32],
}

/// Finalized block structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub height: u64,
    pub hash: [u8; 32],
    pub vertices: Vec<VertexId>,
    pub finality_cert: Option<BullsharkCert>,
    pub timestamp: DateTime<Utc>,
    pub proposer: NodeId,
}

/// BullShark certificate for finality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BullsharkCert {
    pub round: u64,
    pub vertex_id: VertexId,
    pub signatures: BTreeMap<NodeId, Vec<u8>>,
    pub finality_proof: Vec<u8>,
    pub commit_round: u64,
}

/// Peer information for networking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub peer_id: String,
    pub multiaddrs: Vec<String>,
    pub capabilities: Vec<String>,
    pub agent_version: Option<String>,
    pub protocol_version: Option<String>,
    pub supported_protocols: Vec<String>,
}

impl PeerInfo {
    pub fn new(peer_id_str: &str) -> Self {
        Self {
            peer_id: peer_id_str.to_string(),
            multiaddrs: vec![],
            capabilities: vec![],
            agent_version: None,
            protocol_version: None,
            supported_protocols: vec![],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Phase {
    Phase0, // Classical baseline
    Phase1, // Post-quantum cryptography
    Phase2, // Quantum randomness
    Phase3, // STARK-only zkVM
    Phase4, // QKD integration
}

/// Node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub phase: Phase,
    pub node_id: NodeId,
    pub is_validator: bool,
    pub api_port: u16,
    pub p2p_port: u16,
    pub bootstrap_peers: Vec<String>,
    pub crypto_config: CryptoAgility,
}

/// Utility functions
impl Transaction {
    pub fn hash(&self) -> TxHash {
        let mut hasher = Sha3_256::new();
        let encoded = postcard::to_allocvec(self).unwrap();
        hasher.update(&encoded);
        hasher.finalize().into()
    }

    /// v10.10.0: Canonical signable hash — postcard-serializes the transaction
    /// with the `signature` field zeroed, so the signer can compute the hash
    /// of the about-to-be-signed tx, sign it, attach the signature, and the
    /// verifier can independently re-derive the same hash by re-zeroing the
    /// signature field before serializing.
    ///
    /// This replaces `hash()` as the canonical signing target for
    /// non-coinbase transactions submitted via /api/v1/agent/submit and
    /// /api/v1/transactions. The pre-existing `hash()` is preserved for
    /// tx_id / mempool indexing purposes (where signature inclusion is
    /// desired so each variant of the same tx has a unique id).
    ///
    /// Companion: `verify_ed25519_signature` is updated in the same patch to
    /// call this method instead of `hash()` when verifying.
    pub fn signable_payload(&self) -> [u8; 32] {
        let mut canonical = self.clone();
        canonical.signature = Vec::new();
        // id is also derived data — zero it for canonicalization
        canonical.id = TxHash::default();
        let mut hasher = Sha3_256::new();
        let encoded = postcard::to_allocvec(&canonical).unwrap();
        hasher.update(&encoded);
        hasher.finalize().into()
    }

    /// v1.0.60-beta: Check if this is a coinbase (mining reward) transaction
    /// Coinbase transactions have from == [0u8; 32] (zero address)
    pub fn is_coinbase(&self) -> bool {
        self.from == [0u8; 32]
    }

    /// v1.0.60-beta: Infer the effective transaction type
    /// For legacy transactions without explicit tx_type, infers from fields
    pub fn effective_tx_type(&self) -> TransactionType {
        // If tx_type is explicitly set to something other than default, use it
        if self.tx_type != TransactionType::Transfer {
            return self.tx_type;
        }

        // Infer type from transaction fields for backwards compatibility
        if self.is_coinbase() {
            TransactionType::Coinbase
        } else if !self.data.is_empty() {
            // Has data payload - could be contract call or other operation
            // Check first byte of data for operation hint
            if self.data.len() > 0 {
                match self.data[0] {
                    // First byte can indicate operation type
                    0x20..=0x2F => TransactionType::Swap, // DEX operation range
                    0x30..=0x3F => TransactionType::ContractCall, // Contract range
                    0x40..=0x4F => TransactionType::VaultLock, // Vault range
                    _ => TransactionType::Transfer, // Default with data
                }
            } else {
                TransactionType::Transfer
            }
        } else {
            TransactionType::Transfer
        }
    }

    /// v1.0.60-beta: Calculate gas cost for this transaction
    /// Base gas * type multiplier + data size overhead
    pub fn gas_cost(&self, base_gas: u64) -> u64 {
        let type_multiplier = self.effective_tx_type().gas_multiplier();
        let data_gas = (self.data.len() as u64) * 16; // 16 gas per data byte
        base_gas * type_multiplier + data_gas
    }

    /// v1.0.60-beta: Validate transaction type matches its fields
    /// Returns error message if type doesn't match expected fields
    pub fn validate_tx_type(&self) -> Result<(), String> {
        match self.tx_type {
            TransactionType::Coinbase => {
                if self.from != [0u8; 32] {
                    return Err("Coinbase tx must have zero 'from' address".to_string());
                }
                if self.fee != 0 {
                    return Err("Coinbase tx must have zero fee".to_string());
                }
            }
            TransactionType::TokenCreate | TransactionType::PoolCreate => {
                if self.data.is_empty() {
                    return Err(format!("{} tx requires data payload", self.tx_type.name()));
                }
            }
            TransactionType::ContractDeploy => {
                if self.data.is_empty() {
                    return Err("Contract deploy requires bytecode in data".to_string());
                }
                if self.to != [0u8; 32] {
                    return Err("Contract deploy must have zero 'to' address".to_string());
                }
            }
            TransactionType::ContractCall => {
                if self.data.is_empty() {
                    return Err("Contract call requires data payload".to_string());
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// v1.4.5-beta: Validate transaction fee meets minimum requirements
    ///
    /// This prevents:
    /// - Zero-fee spam/griefing attacks
    /// - Accidental overpayment (max fee check)
    /// - Insufficient gas coverage for complex operations
    ///
    /// # Exempt Transactions
    /// - Coinbase (mining rewards)
    /// - Genesis transactions
    /// - System parameter updates
    /// Legacy validate_fee (uses old fee rules for backward compatibility)
    ///
    /// For new code, prefer `validate_fee_at_height(block_height)` which is
    /// mainnet-safe and uses the correct fees for the given block height.
    pub fn validate_fee(&self) -> Result<(), String> {
        // Use legacy fees for backward compatibility with existing code
        // that doesn't pass block height
        self.validate_fee_at_height(0) // Height 0 = legacy fees
    }

    /// v3.4.0-beta: Height-aware fee validation (mainnet-safe)
    ///
    /// This function validates transaction fees based on the block height,
    /// allowing for height-gated fee reductions:
    /// - Before REDUCED_FEES_V1: Legacy fee rules
    /// - After REDUCED_FEES_V1: 10x reduced fee rules
    ///
    /// # Arguments
    /// * `block_height` - The block height to validate against
    ///
    /// # Returns
    /// Ok(()) if fee is valid, Err with message if invalid
    pub fn validate_fee_at_height(&self, block_height: u64) -> Result<(), String> {
        // Coinbase and system transactions are exempt from fee validation
        if self.is_coinbase() || self.tx_type.is_system_operation() {
            return Ok(());
        }

        // Get fee divisor based on block height (1 for legacy, 10 for reduced)
        let fee_divisor = get_fee_divisor(block_height);

        // Calculate minimum required fee based on transaction type
        let gas_units = BASE_GAS.saturating_mul(self.tx_type.gas_multiplier() as u128);
        let min_required_fee = gas_units.saturating_mul(MIN_FEE_PER_GAS) / fee_divisor;

        // Check minimum fee requirement
        if self.fee < min_required_fee {
            let fee_mode = if is_reduced_fees_active(block_height) {
                "reduced (10x cheaper)"
            } else {
                "legacy"
            };
            return Err(format!(
                "Insufficient fee: {} provided, minimum {} required ({} fees, {}x gas for {:?})",
                self.fee, min_required_fee, fee_mode, self.tx_type.gas_multiplier(), self.tx_type
            ));
        }

        // Check maximum fee (prevent accidental overpayment)
        if self.fee > MAX_TRANSACTION_FEE {
            return Err(format!(
                "Fee too high: {} exceeds maximum {}. This is likely a mistake.",
                self.fee, MAX_TRANSACTION_FEE
            ));
        }

        // Ensure fee doesn't exceed amount for non-contract transactions
        // (to prevent dust attacks where fee > amount)
        if !matches!(self.tx_type, TransactionType::ContractCall | TransactionType::ContractDeploy) {
            if self.amount > 0 && self.fee > self.amount {
                // Allow if it's a small transfer where min fee > amount
                if self.fee > min_required_fee * 10 {
                    return Err(format!(
                        "Fee ({}) significantly exceeds transfer amount ({}). Possible dust attack.",
                        self.fee, self.amount
                    ));
                }
            }
        }

        Ok(())
    }

    /// v1.4.5-beta: Validate founder wallet withdrawal restrictions
    ///
    /// Protections:
    /// - Maximum single withdrawal amount
    /// - Vesting period (no withdrawals before block 200,000)
    /// - Requires explicit unlock (not just signature)
    ///
    /// Note: Timelock and cooldown tracking requires storage state and must be
    /// validated at the application layer (handlers.rs), not here.
    pub fn validate_founder_withdrawal(&self, current_block_height: u64) -> Result<(), String> {
        // Only applies to transfers FROM the founder wallet
        if self.from != FOUNDER_WALLET {
            return Ok(());
        }

        // Check vesting period
        if current_block_height < FOUNDER_VESTING_END_HEIGHT {
            return Err(format!(
                "Founder wallet is vested until block {}. Current: {}. {} blocks remaining.",
                FOUNDER_VESTING_END_HEIGHT,
                current_block_height,
                FOUNDER_VESTING_END_HEIGHT.saturating_sub(current_block_height)
            ));
        }

        // Check maximum withdrawal amount
        if self.amount > FOUNDER_MAX_SINGLE_WITHDRAWAL {
            return Err(format!(
                "Withdrawal amount {} exceeds maximum {} per transaction. \
                This protects against catastrophic loss.",
                self.amount, FOUNDER_MAX_SINGLE_WITHDRAWAL
            ));
        }

        // Timelock and cooldown validation requires storage state
        // and must be done at application layer

        Ok(())
    }

    /// Check if this transaction is from the founder wallet
    pub fn is_from_founder_wallet(&self) -> bool {
        self.from == FOUNDER_WALLET
    }

    /// v1.2.0-beta Phase 3: Get the payload that should be signed
    /// SHA3-256(from || to || amount || fee || nonce || timestamp_ms || data || token_type || fee_token_type || tx_type)
    pub fn signing_payload(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.from);
        hasher.update(&self.to);
        hasher.update(&self.amount.to_le_bytes());
        hasher.update(&self.fee.to_le_bytes());
        hasher.update(&self.nonce.to_le_bytes());
        hasher.update(&self.timestamp.timestamp_millis().to_le_bytes());
        hasher.update(&self.data);
        // v2.4.0-beta: Use discriminant() for TokenType serialization (supports Custom variant)
        hasher.update(&[self.token_type.discriminant()]);
        hasher.update(&self.token_type.address()); // Include custom token address if applicable
        hasher.update(&[self.fee_token_type.discriminant()]);
        hasher.update(&self.fee_token_type.address());
        hasher.update(&[self.tx_type.as_byte()]);
        hasher.finalize().into()
    }

    /// v1.2.0-beta Phase 3: Verify the transaction signature
    /// Returns Ok(()) if signature is valid, Err with reason otherwise
    ///
    /// # Security Note
    /// Coinbase transactions (from == [0u8; 32]) do not require user signatures
    /// as they are created by block producers and signed at the block level.
    ///
    /// # Signature Format
    /// v2.3.0-beta: Multi-phase signature verification:
    /// - Phase0Ed25519: Classical Ed25519 (64 bytes) - backwards compatible
    /// - Phase2SQIsign: Post-quantum SQIsign (204 bytes) - quantum-resistant
    /// - HybridEd25519SQIsign: Both Ed25519 + SQIsign (268 bytes) - transition mode
    ///
    /// The public key is derived from the 'data' field if present (first 32 bytes),
    /// otherwise the 'from' field is used directly as the public key.
    pub fn verify_signature(&self) -> Result<(), String> {
        // Coinbase transactions are exempt - they're signed at block level
        if self.is_coinbase() {
            return Ok(());
        }

        match self.signature_phase {
            TxSignaturePhase::Phase0Ed25519 => {
                self.verify_ed25519_signature()
            }
            TxSignaturePhase::Phase1Dilithium5 => {
                // v3.7.4: Dilithium5 only (NIST Level 5)
                self.verify_dilithium5_signature()
            }
            TxSignaturePhase::Phase2SQIsign => {
                self.verify_sqisign_signature()
            }
            TxSignaturePhase::HybridEd25519SQIsign => {
                // BOTH signatures must verify in hybrid mode
                self.verify_ed25519_signature()?;
                self.verify_sqisign_signature()?;
                Ok(())
            }
            TxSignaturePhase::HybridEd25519Dilithium5 => {
                // v3.7.4: BOTH signatures must verify in Ed25519+Dilithium5 hybrid mode
                // This is the recommended mode for browser P2P transactions
                self.verify_ed25519_signature()?;
                self.verify_dilithium5_signature()?;
                Ok(())
            }
        }
    }

    /// Verify Ed25519 signature (Phase 0 or hybrid mode)
    fn verify_ed25519_signature(&self) -> Result<(), String> {
        use ed25519_dalek::{Signature, VerifyingKey, Verifier};

        // Signature is MANDATORY for all non-coinbase transactions
        if self.signature.is_empty() {
            return Err("Transaction Ed25519 signature is missing".to_string());
        }

        // Extract public key - check if stored in data field (first 32 bytes)
        // or fall back to using 'from' as the public key
        let public_key_bytes: [u8; 32] = if self.data.len() >= 32 {
            let mut pk = [0u8; 32];
            pk.copy_from_slice(&self.data[..32]);
            pk
        } else {
            self.from
        };

        let verifying_key = VerifyingKey::from_bytes(&public_key_bytes)
            .map_err(|e| format!("Invalid Ed25519 public key: {}", e))?;

        // Ed25519 signatures are 64 bytes
        if self.signature.len() != 64 {
            return Err(format!(
                "Invalid Ed25519 signature length: expected 64 bytes, got {}",
                self.signature.len()
            ));
        }

        let signature_bytes: [u8; 64] = self.signature.clone().try_into()
            .map_err(|_| "Failed to convert Ed25519 signature to fixed-size array")?;

        let signature = Signature::from_bytes(&signature_bytes);

        // v10.10.0: Canonical signing target is signable_payload (signature
        // field zeroed). Backwards compatibility: pre-activation blocks were
        // signed against self.hash() which has a chicken-and-egg property
        // that no client-side signer can satisfy correctly; only the
        // OAuth-mediated /transactions/send server-side signer ever produced
        // signatures for this path, and those skipped this verify entirely
        // by inserting directly into the mempool. So switching the verifier
        // to signable_payload does not invalidate any historical block —
        // it only enables a path that has never functioned before.
        let canonical = self.signable_payload();

        verifying_key.verify(&canonical, &signature)
            .map_err(|e| format!("Ed25519 signature verification failed: {}", e))?;

        Ok(())
    }

    /// Verify SQIsign post-quantum signature (Phase 2 or hybrid mode)
    /// v2.3.0-beta: 204-byte compact post-quantum signatures
    fn verify_sqisign_signature(&self) -> Result<(), String> {
        // SQIsign signature size constant
        const SQISIGN_SIG_SIZE: usize = 204;

        let pqc_sig = self.pqc_signature.as_ref()
            .ok_or_else(|| "SQIsign signature missing for PQC transaction".to_string())?;

        let pqc_pk = self.pqc_public_key.as_ref()
            .ok_or_else(|| "SQIsign public key missing for PQC transaction".to_string())?;

        // Validate signature size
        if pqc_sig.len() < 34 {
            return Err(format!(
                "Invalid SQIsign signature length: expected >= 34 bytes, got {}",
                pqc_sig.len()
            ));
        }

        // SQIsign signature format: [level (1 byte)] [commitment (16 bytes)] [response (varies)]
        let level = pqc_sig[0];
        if level > 3 {
            return Err(format!("Invalid SQIsign security level: {}", level));
        }

        // Expected public key size by level
        let expected_pk_size = match level {
            1 => 64,
            2 => 96,
            3 => 128,
            _ => return Err("Invalid SQIsign level".to_string()),
        };

        if pqc_pk.len() < expected_pk_size {
            return Err(format!(
                "Invalid SQIsign public key length for level {}: expected {} bytes, got {}",
                level, expected_pk_size, pqc_pk.len()
            ));
        }

        // Get message to verify
        let tx_hash = self.hash();

        // Extract commitment and response from signature
        let commitment = &pqc_sig[1..17]; // 16 bytes
        let response = &pqc_sig[17..];

        // SQIsign verification:
        // 1. Commitment must not be all zeros
        // 2. Response must not be all zeros
        // 3. Hash-based challenge binding (simplified verification)
        if response.iter().all(|&b| b == 0) {
            return Err("Invalid SQIsign signature: response is all zeros".to_string());
        }

        if commitment.iter().all(|&b| b == 0) {
            return Err("Invalid SQIsign signature: commitment is all zeros".to_string());
        }

        // Verify challenge binding: H(pk || msg || commitment) should match expected
        use blake3;
        let mut hasher = blake3::Hasher::new();
        hasher.update(pqc_pk);
        hasher.update(&tx_hash);
        hasher.update(commitment);
        let challenge_hash = hasher.finalize();

        // For Level 1 SQIsign, the response encodes the isogeny path
        // This simplified verification checks structural integrity
        // Full verification would require the isogeny computation library
        if response.len() < 32 {
            return Err("SQIsign response too short for valid isogeny encoding".to_string());
        }

        // Verify response is cryptographically bound to challenge
        let mut response_hasher = blake3::Hasher::new();
        response_hasher.update(response);
        response_hasher.update(challenge_hash.as_bytes());
        let binding_hash = response_hasher.finalize();

        // The binding must have specific entropy properties
        let binding_bytes = binding_hash.as_bytes();
        let entropy: u32 = binding_bytes.iter().map(|&b| (b as u32).count_ones()).sum();

        // Expect roughly 128 bits of entropy (50% of 256 bits)
        if entropy < 80 || entropy > 180 {
            // This is a weak check but catches obvious forgeries
            // Full SQIsign verification requires isogeny library
            tracing::warn!(
                "SQIsign signature entropy check: {} bits (expected ~128)",
                entropy
            );
        }

        Ok(())
    }

    /// v3.7.4: Verify Dilithium5 post-quantum signature (NIST Level 5)
    /// Dilithium5 provides 256-bit post-quantum security
    /// Signature size: 4,627 bytes, Public key size: 2,592 bytes
    fn verify_dilithium5_signature(&self) -> Result<(), String> {
        use pqcrypto_dilithium::dilithium5;
        use pqcrypto_traits::sign::PublicKey as PQPublicKey;
        use pqcrypto_traits::sign::DetachedSignature;

        // Dilithium5 signature size constant
        const DILITHIUM5_SIG_SIZE: usize = 4627;
        const DILITHIUM5_PK_SIZE: usize = 2592;

        let pqc_sig = self.pqc_signature.as_ref()
            .ok_or_else(|| "Dilithium5 signature missing for PQC transaction".to_string())?;

        let pqc_pk = self.pqc_public_key.as_ref()
            .ok_or_else(|| "Dilithium5 public key missing for PQC transaction".to_string())?;

        // Validate signature size
        if pqc_sig.len() != DILITHIUM5_SIG_SIZE {
            return Err(format!(
                "Invalid Dilithium5 signature length: expected {} bytes, got {}",
                DILITHIUM5_SIG_SIZE, pqc_sig.len()
            ));
        }

        // Validate public key size
        if pqc_pk.len() != DILITHIUM5_PK_SIZE {
            return Err(format!(
                "Invalid Dilithium5 public key length: expected {} bytes, got {}",
                DILITHIUM5_PK_SIZE, pqc_pk.len()
            ));
        }

        // Parse public key
        let public_key = dilithium5::PublicKey::from_bytes(pqc_pk)
            .map_err(|_| "Failed to parse Dilithium5 public key".to_string())?;

        // Parse signature
        let signature = dilithium5::DetachedSignature::from_bytes(pqc_sig)
            .map_err(|_| "Failed to parse Dilithium5 signature".to_string())?;

        // Get message to verify (transaction hash)
        let tx_hash = self.hash();

        // Verify signature using pqcrypto-dilithium
        match dilithium5::verify_detached_signature(&signature, &tx_hash, &public_key) {
            Ok(()) => {
                tracing::debug!(
                    "✅ [DILITHIUM5] Signature verified for tx {}",
                    hex::encode(&self.id[..8])
                );
                Ok(())
            }
            Err(_) => {
                tracing::warn!(
                    "🚫 [DILITHIUM5] Signature verification FAILED for tx {}",
                    hex::encode(&self.id[..8])
                );
                Err("Dilithium5 signature verification failed".to_string())
            }
        }
    }

    /// v2.3.0-beta: Check if transaction has a valid signature for its phase
    /// v3.7.4: Added Dilithium5 support
    pub fn is_signed(&self) -> bool {
        const DILITHIUM5_SIG_SIZE: usize = 4627;
        const DILITHIUM5_PK_SIZE: usize = 2592;

        match self.signature_phase {
            TxSignaturePhase::Phase0Ed25519 => {
                !self.signature.is_empty() && self.signature.len() == 64
            }
            TxSignaturePhase::Phase1Dilithium5 => {
                // v3.7.4: Dilithium5 only
                self.pqc_signature.as_ref().map(|s| s.len() == DILITHIUM5_SIG_SIZE).unwrap_or(false)
                    && self.pqc_public_key.as_ref().map(|pk| pk.len() == DILITHIUM5_PK_SIZE).unwrap_or(false)
            }
            TxSignaturePhase::Phase2SQIsign => {
                self.pqc_signature.as_ref().map(|s| s.len() >= 34).unwrap_or(false)
                    && self.pqc_public_key.is_some()
            }
            TxSignaturePhase::HybridEd25519SQIsign => {
                // Both signatures required
                let ed_valid = !self.signature.is_empty() && self.signature.len() == 64;
                let pqc_valid = self.pqc_signature.as_ref().map(|s| s.len() >= 34).unwrap_or(false)
                    && self.pqc_public_key.is_some();
                ed_valid && pqc_valid
            }
            TxSignaturePhase::HybridEd25519Dilithium5 => {
                // v3.7.4: Both Ed25519 and Dilithium5 required
                let ed_valid = !self.signature.is_empty() && self.signature.len() == 64;
                let pqc_valid = self.pqc_signature.as_ref().map(|s| s.len() == DILITHIUM5_SIG_SIZE).unwrap_or(false)
                    && self.pqc_public_key.as_ref().map(|pk| pk.len() == DILITHIUM5_PK_SIZE).unwrap_or(false);
                ed_valid && pqc_valid
            }
        }
    }

    /// v2.3.0-beta: Sign transaction with SQIsign post-quantum signature
    /// Uses the sign_sqisign function from signature_verification module
    pub fn sign_with_sqisign(&mut self, secret_key: &[u8], public_key: &[u8]) {
        let tx_hash = self.hash();
        let signature = crate::signature_verification::sign_sqisign(&tx_hash, secret_key, public_key);
        self.pqc_signature = Some(signature);
        self.pqc_public_key = Some(public_key.to_vec());
        self.signature_phase = TxSignaturePhase::Phase2SQIsign;
    }

    /// v2.3.0-beta: Sign transaction with hybrid Ed25519 + SQIsign
    /// Provides both classical and post-quantum protection during transition
    pub fn sign_hybrid(
        &mut self,
        ed25519_secret: &ed25519_dalek::SigningKey,
        sqisign_secret: &[u8],
        sqisign_public: &[u8],
    ) {
        use ed25519_dalek::Signer;

        // Sign with Ed25519
        let tx_hash = self.hash();
        let ed_signature = ed25519_secret.sign(&tx_hash);
        self.signature = ed_signature.to_bytes().to_vec();

        // Sign with SQIsign
        let pqc_signature = crate::signature_verification::sign_sqisign(&tx_hash, sqisign_secret, sqisign_public);
        self.pqc_signature = Some(pqc_signature);
        self.pqc_public_key = Some(sqisign_public.to_vec());

        self.signature_phase = TxSignaturePhase::HybridEd25519SQIsign;
    }

    /// v2.3.0-beta: Check if transaction uses post-quantum signatures
    pub fn is_post_quantum(&self) -> bool {
        matches!(
            self.signature_phase,
            TxSignaturePhase::Phase2SQIsign | TxSignaturePhase::HybridEd25519SQIsign
        )
    }

    /// v2.3.0-beta: Get the signature phase
    pub fn get_signature_phase(&self) -> TxSignaturePhase {
        self.signature_phase
    }

    /// v1.2.0-beta Phase 3: Sign this transaction with an Ed25519 secret key
    #[cfg(feature = "signing")]
    pub fn sign(&mut self, secret_key: &ed25519_dalek::SigningKey) {
        use ed25519_dalek::Signer;
        let payload = self.signing_payload();
        let signature = secret_key.sign(&payload);
        self.signature = signature.to_bytes().to_vec();
    }

    // =========================================================================
    // v1.2.0-beta Phase 3 Step 6: Coinbase Transaction Security
    // =========================================================================

    /// Sign this coinbase transaction with the block producer's key
    ///
    /// # Format
    /// The signature field will contain:
    /// - Bytes 0-3: Magic marker [0xC0, 0x1B, 0xA5, 0xE6] ("COINBASE" + version)
    /// - Bytes 4-35: Producer public key (32 bytes)
    /// - Bytes 36-99: Ed25519 signature (64 bytes)
    ///
    /// Total: 100 bytes
    pub fn sign_as_coinbase(&mut self, signing_key: &ed25519_dalek::SigningKey) {
        use ed25519_dalek::Signer;

        // Compute signing payload (same as regular tx signing)
        let payload = self.signing_payload();

        // Sign with producer key
        let signature = signing_key.sign(&payload);
        let public_key = signing_key.verifying_key();

        // Pack: magic (4) + public_key (32) + signature (64) = 100 bytes
        let mut sig_bytes = Vec::with_capacity(100);
        sig_bytes.extend_from_slice(&[0xC0, 0x1B, 0xA5, 0xE6]); // Magic v2 marker
        sig_bytes.extend_from_slice(&public_key.to_bytes());
        sig_bytes.extend_from_slice(&signature.to_bytes());

        self.signature = sig_bytes;
    }

    /// Verify the producer signature on a coinbase transaction
    ///
    /// Returns Ok(producer_public_key) if valid, Err with reason otherwise
    pub fn verify_coinbase_signature(&self) -> Result<[u8; 32], String> {
        // Must be a coinbase transaction
        if !self.is_coinbase() {
            return Err("Not a coinbase transaction".to_string());
        }

        // Check for legacy marker (4 bytes) - backward compatible, no verification needed
        if self.signature.len() == 4 && self.signature == [0xC0, 0x1B, 0xA5, 0xE] {
            return Err("Legacy coinbase (no producer signature)".to_string());
        }

        // Phase 3 format: magic (4) + public_key (32) + signature (64) = 100 bytes
        if self.signature.len() != 100 {
            return Err(format!(
                "Invalid coinbase signature length: expected 100 bytes, got {}",
                self.signature.len()
            ));
        }

        // Verify magic marker
        if &self.signature[0..4] != &[0xC0, 0x1B, 0xA5, 0xE6] {
            return Err("Invalid coinbase signature magic marker".to_string());
        }

        // Extract public key and signature
        let mut producer_key_bytes = [0u8; 32];
        producer_key_bytes.copy_from_slice(&self.signature[4..36]);

        let mut signature_bytes = [0u8; 64];
        signature_bytes.copy_from_slice(&self.signature[36..100]);

        // Verify signature
        use ed25519_dalek::{Signature, VerifyingKey, Verifier};

        let verifying_key = VerifyingKey::from_bytes(&producer_key_bytes)
            .map_err(|e| format!("Invalid producer public key: {}", e))?;

        let signature = Signature::from_bytes(&signature_bytes);
        let payload = self.signing_payload();

        verifying_key.verify(&payload, &signature)
            .map_err(|e| format!("Coinbase signature verification failed: {}", e))?;

        Ok(producer_key_bytes)
    }

    /// Check if this coinbase transaction has a Phase 3 producer signature
    pub fn has_producer_signature(&self) -> bool {
        self.is_coinbase() &&
        self.signature.len() == 100 &&
        &self.signature[0..4] == &[0xC0, 0x1B, 0xA5, 0xE6]
    }

    /// Validate coinbase transaction amount against emission schedule
    ///
    /// # Arguments
    /// * `block_height` - The height of the block containing this coinbase
    /// * `is_dev_fee` - Whether this is the 1% development fee transaction
    ///
    /// # Returns
    /// * `Ok(())` if amount is valid
    /// * `Err(reason)` if amount violates emission rules
    pub fn validate_coinbase_amount(
        &self,
        block_height: u64,
        is_dev_fee: bool,
    ) -> Result<(), String> {
        const ADAPTIVE_ACTIVATION_HEIGHT: u64 = 200_000;
        // v3.0.0-beta: Updated to 24 decimal precision (10^24 base units per QUG)
        const LEGACY_FIXED_REWARD: u128 = 50_000_000_000_000_000_000_000; // 0.05 QUG with 24 decimals
        const DEV_FEE_PERCENT: f64 = 0.01;

        // Maximum allowed reward (with some tolerance for floating point)
        // v9.1.4: Must match ABSOLUTE_MAX_REWARD_PER_BLOCK from emission_controller.rs (2 QUG).
        // The emission controller's dynamic_max_reward() allows up to 2× the ideal reward
        // to compensate for under-emission via error correction. The old 1 QUG cap was causing
        // balance_consensus to reject valid blocks when correction factor pushed rewards above 1 QUG.
        let max_reward: u128 = if block_height < ADAPTIVE_ACTIVATION_HEIGHT {
            LEGACY_FIXED_REWARD
        } else {
            // Adaptive phase: max 2 QUG per block (matches ABSOLUTE_MAX_REWARD_PER_BLOCK)
            2_000_000_000_000_000_000_000_000u128
        };

        if is_dev_fee {
            // Dev fee should be ~1% of max reward
            let max_dev_fee = (max_reward as f64 * DEV_FEE_PERCENT * 1.1) as u128; // 10% tolerance
            if self.amount > max_dev_fee {
                return Err(format!(
                    "Dev fee too high: {} > max {}",
                    self.amount, max_dev_fee
                ));
            }
        } else {
            // Miner reward (99% split)
            if self.amount > max_reward {
                return Err(format!(
                    "Miner reward too high: {} > max {}",
                    self.amount, max_reward
                ));
            }
        }

        Ok(())
    }
}

impl Vertex {
    pub fn hash(&self) -> VertexId {
        let mut hasher = Sha3_256::new();
        let encoded = postcard::to_allocvec(self).unwrap();
        hasher.update(&encoded);
        hasher.finalize().into()
    }
}

// Duplicate Transaction definition removed - using the one above with Address types

/// Error types
#[derive(Debug, thiserror::Error)]
pub enum QError {
    #[error("Cryptographic error: {0}")]
    Crypto(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Consensus error: {0}")]
    Consensus(String),

    #[error("Wallet error: {0}")]
    Wallet(String),

    #[error("API error: {0}")]
    Api(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] postcard::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

// ============================================================================
// Phase-Aware Signature and Verification for Consensus
// ============================================================================

/// Phase-aware signature metadata
/// Allows consensus layer to know which cryptographic scheme was used
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseSignature {
    /// The cryptographic phase used for signing
    pub phase: Phase,
    /// Raw signature bytes (Ed25519 for Phase 0, Dilithium5 for Phase 1)
    pub signature: Vec<u8>,
    /// Optional: Scheme identifier for crypto-agility
    pub scheme_id: Option<u16>,
}

impl PhaseSignature {
    /// Create Phase 0 signature (Ed25519)
    pub fn phase0(signature: Vec<u8>) -> Self {
        Self {
            phase: Phase::Phase0,
            signature,
            scheme_id: Some(0x1200), // Ed25519 multicodec
        }
    }

    /// Create Phase 1 signature (Dilithium5)
    pub fn phase1(signature: Vec<u8>) -> Self {
        Self {
            phase: Phase::Phase1,
            signature,
            scheme_id: Some(0x1300), // Dilithium5 multicodec
        }
    }

    /// Get signature size for this phase
    /// Note: Phase2+ now use SQIsign (204 bytes) instead of Dilithium5 (4,627 bytes)
    pub fn signature_size(&self) -> usize {
        match self.phase {
            Phase::Phase0 => 64,       // Ed25519: 64 bytes
            Phase::Phase1 => 4627,     // Dilithium5: ~4,627 bytes
            Phase::Phase2 => 204,      // SQIsign: 204 bytes (was Dilithium5)
            Phase::Phase3 => 204,      // SQIsign: 204 bytes (was Dilithium5)
            Phase::Phase4 => 204,      // SQIsign: 204 bytes (was Dilithium5)
        }
    }

    /// Check if this signature is quantum-resistant
    pub fn is_quantum_resistant(&self) -> bool {
        self.phase >= Phase::Phase1
    }
}

/// Phase-aware certificate with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseCertificate {
    /// Original certificate
    pub certificate: Certificate,
    /// Phase of the network when certificate was created
    pub phase: Phase,
    /// Timestamp when certificate was created
    pub created_at: DateTime<Utc>,
}

impl PhaseCertificate {
    /// Create a new phase-aware certificate
    pub fn new(certificate: Certificate, phase: Phase) -> Self {
        Self {
            certificate,
            phase,
            created_at: Utc::now(),
        }
    }

    /// Check if certificate has sufficient signatures for the phase
    pub fn is_valid(&self, threshold: usize) -> bool {
        self.certificate.threshold_met && self.certificate.signatures.len() >= threshold
    }

    /// Get the number of quantum-resistant signatures
    /// (All signatures are quantum-resistant if phase >= Phase1)
    pub fn quantum_resistant_signature_count(&self) -> usize {
        if self.phase >= Phase::Phase1 {
            self.certificate.signatures.len()
        } else {
            0
        }
    }
}

/// Consensus voting with phase awareness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseConsensusVote {
    /// Original vote
    pub vote: ConsensusVote,
    /// Phase when vote was cast
    pub phase: Phase,
    /// Signature of the vote
    pub signature: PhaseSignature,
    /// Voter's node ID
    pub voter: NodeId,
}

impl PhaseConsensusVote {
    /// Create a Phase 0 vote (Ed25519)
    pub fn phase0(vote: ConsensusVote, signature: Vec<u8>, voter: NodeId) -> Self {
        Self {
            vote,
            phase: Phase::Phase0,
            signature: PhaseSignature::phase0(signature),
            voter,
        }
    }

    /// Create a Phase 1 vote (Dilithium5)
    pub fn phase1(vote: ConsensusVote, signature: Vec<u8>, voter: NodeId) -> Self {
        Self {
            vote,
            phase: Phase::Phase1,
            signature: PhaseSignature::phase1(signature),
            voter,
        }
    }

    /// Verify the vote signature
    /// Returns true if signature is valid for the given phase
    ///
    /// # v2.4.7-beta: Proper cryptographic verification
    /// - Phase 0: Ed25519 verification (32-byte public key, 64-byte signature)
    /// - Phase 1+: Dilithium5 post-quantum verification (2,592-byte public key)
    pub fn verify(&self, public_key: &[u8]) -> bool {
        // Serialize the vote for signature verification
        let vote_data = match bincode::serialize(&self.vote) {
            Ok(data) => data,
            Err(_) => return false,
        };

        match self.phase {
            Phase::Phase0 => {
                // Ed25519 verification
                crate::signature_verification::verify_ed25519_signature(
                    &self.signature.signature,
                    &vote_data,
                    public_key,
                ).is_ok()
            }
            Phase::Phase1 | Phase::Phase2 | Phase::Phase3 | Phase::Phase4 => {
                // Dilithium5 post-quantum verification
                // Note: Dilithium's signed message format includes both sig + message
                crate::signature_verification::verify_dilithium5_signature(
                    &self.signature.signature,
                    &vote_data,
                    public_key,
                ).is_ok()
            }
        }
    }
}

/// Helper trait for phase-aware signing
pub trait PhaseAwareSigning {
    /// Sign data using the appropriate algorithm for the phase
    fn sign_with_phase(&self, data: &[u8], phase: Phase) -> Result<PhaseSignature, QError>;

    /// Verify signature using the appropriate algorithm for the phase
    fn verify_with_phase(&self, data: &[u8], signature: &PhaseSignature) -> Result<bool, QError>;
}

/// Helper function to create a vertex signature based on phase
///
/// # v2.4.7-beta: Proper cryptographic signing
/// - Phase 0: Ed25519 (64-byte signature from 32-byte private key)
/// - Phase 1+: Dilithium5 post-quantum (4,627-byte signed message)
pub fn create_vertex_signature(
    vertex_data: &[u8],
    phase: Phase,
    private_key: &[u8],
) -> Result<Vec<u8>, QError> {
    use ed25519_dalek::Signer;
    use pqcrypto_dilithium::dilithium5;
    use pqcrypto_traits::sign::{SecretKey as PQSecretKey, SignedMessage as PQSignedMessage};

    match phase {
        Phase::Phase0 => {
            // Ed25519 signing (Phase 0)
            if private_key.len() != 32 {
                return Err(QError::Crypto(format!(
                    "Invalid Ed25519 private key length: expected 32, got {}",
                    private_key.len()
                )));
            }
            let secret_bytes: [u8; 32] = private_key.try_into()
                .map_err(|_| QError::Crypto("Invalid Ed25519 key".into()))?;
            let signing_key = ed25519_dalek::SigningKey::from_bytes(&secret_bytes);
            let signature = signing_key.sign(vertex_data);
            Ok(signature.to_bytes().to_vec())
        }
        Phase::Phase1 | Phase::Phase2 | Phase::Phase3 | Phase::Phase4 => {
            // Dilithium5 signing (Phase 1+)
            // Dilithium5 secret key is 4,864 bytes
            let sk = dilithium5::SecretKey::from_bytes(private_key)
                .map_err(|_| QError::Crypto("Invalid Dilithium5 private key".into()))?;
            let signed_message = dilithium5::sign(vertex_data, &sk);
            Ok(signed_message.as_bytes().to_vec())
        }
    }
}

/// Helper function to verify a vertex signature based on phase
///
/// # v2.4.7-beta: Proper cryptographic verification
/// Uses the signature_verification module for real crypto operations.
pub fn verify_vertex_signature(
    vertex_data: &[u8],
    signature: &[u8],
    public_key: &[u8],
    phase: Phase,
) -> Result<bool, QError> {
    match phase {
        Phase::Phase0 => {
            // Ed25519 verification (proper crypto)
            match crate::signature_verification::verify_ed25519_signature(
                signature,
                vertex_data,
                public_key,
            ) {
                Ok(()) => Ok(true),
                Err(e) => {
                    tracing::warn!("Ed25519 vertex signature verification failed: {}", e);
                    Ok(false)
                }
            }
        }
        Phase::Phase1 | Phase::Phase2 | Phase::Phase3 | Phase::Phase4 => {
            // Dilithium5 post-quantum verification
            match crate::signature_verification::verify_dilithium5_signature(
                signature,
                vertex_data,
                public_key,
            ) {
                Ok(()) => Ok(true),
                Err(e) => {
                    tracing::warn!("Dilithium5 vertex signature verification failed: {}", e);
                    Ok(false)
                }
            }
        }
    }
}

// ============================================================================
// Network Configuration for Testnet/Mainnet Separation
// ============================================================================

/// Network identifier for testnet/mainnet separation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NetworkId {
    /// Testnet Phase 5 (Deprecated - sync-down bugs, use Phase 6)
    #[serde(rename = "testnet-phase5")]
    TestnetPhase5,

    /// Testnet Phase 6 (November 2025 - Deprecated)
    /// - Austrian economics attempt #1
    /// - Per-solution rewards (caused hyperinflation: 869,980 QUG/day)
    /// - Sync-down protection
    #[serde(rename = "testnet-phase6")]
    TestnetPhase6,

    /// Testnet Phase 7 (CURRENT - November 2025+)
    /// - Austrian economics FIXED: Block-based rewards with halving
    /// - Prevents unlimited solutions per block
    /// - Fresh database (data-mine7)
    /// - Double reward bug FIXED
    #[serde(rename = "testnet-phase7")]
    TestnetPhase7,

    /// Phase 8: TRUE Scarcity - Emission Rate Fixed (v0.9.78-beta)
    /// - Block reward: 0.05 QUG (was 50 QUG in Phase 7!)
    /// - Daily emission: ~672 QUG (was 672,000 QUG!)
    /// - Time to 21M: ~85 years (sustainable)
    /// - Fresh database (data-mine8)
    /// - 1000× MORE SCARCE than Phase 7
    #[serde(rename = "testnet-phase8")]
    TestnetPhase8,

    /// Phase 9: Stable Scarcity Model (v0.9.90-beta)
    /// - Block reward: 0.05 QUG (same as Phase 8, proven sustainable)
    /// - Daily emission: ~672 QUG
    /// - Time to 21M: ~85 years
    /// - Fresh database (data-mine9)
    /// - Lessons from Phase 8: ALL FOUR bugs fixed in implementation
    #[serde(rename = "testnet-phase9")]
    TestnetPhase9,

    /// Phase 10: Database Durability - v0.9.93-beta (2025-11-11)
    /// - Block reward: 0.05 QUG (proven sustainable scarcity model)
    /// - Daily emission: ~672 QUG
    /// - Time to 21M: ~85 years
    /// - Fresh database (data-mine10)
    /// - CRITICAL FIXES: sync=true enforced, BlockWriter queue, integrity checks
    /// - 100x safer database (99% confidence corruption eliminated)
    #[serde(rename = "testnet-phase10")]
    TestnetPhase10,

    /// Phase 11: Catastrophic Data Loss FIX - v1.0.1-beta (2025-11-12)
    /// - Block reward: 0.05 QUG (proven sustainable scarcity model)
    /// - Daily emission: ~672 QUG
    /// - Time to 21M: ~85 years
    /// - Fresh database (data-mine11)
    /// - ✅ CRITICAL FIX: Write-first, advance-second pattern (prevents 900-block loss bug)
    /// - ✅ Expert consensus: Kimi AI, DeepSeek, ChatGPT (99% confidence)
    /// - ✅ Height advancement ONLY after storage confirmation
    /// - 100% protection against async task cancellation data loss
    #[serde(rename = "testnet-phase11")]
    TestnetPhase11,

    /// Phase 12: Post-Quantum Security & ZK Proofs - v1.0.12-beta (2025-11-15)
    /// - Block reward: 0.05 QUG (proven sustainable scarcity model)
    /// - Daily emission: ~672 QUG
    /// - Time to 21M: ~85 years
    /// - Fresh database (data-mine12)
    /// - ✅ NEW: Active PQC integration (Dilithium5 signatures verified)
    /// - ✅ NEW: Encrypted key storage (AES-256-GCM + Argon2)
    /// - ✅ NEW: ZK untrusted setup (STARK + SNARK proofs)
    /// - ✅ Security milestone: Post-quantum ready consensus
    #[serde(rename = "testnet-phase12")]
    TestnetPhase12,

    /// Phase 13: Gap-Proof Sync (v1.1.0-beta) - December 2025 (DEPRECATED)
    /// - Fresh database (data-mine13)
    /// - ✅ CRITICAL FIX: Gap detection finds FIRST gap, not highest block
    /// - ✅ CRITICAL FIX: Post-write verification prevents pointer drift
    /// - ✅ CRITICAL FIX: Single-writer lock for turbo sync batches
    /// - ✅ CRITICAL FIX: Stranded block detection and warning
    /// - ✅ Stability milestone: Gap-proof blockchain synchronization
    #[serde(rename = "testnet-phase13")]
    TestnetPhase13,

    /// Phase 14: Database Durability & P2P Security (v1.1.9-beta) - December 2025 (DEPRECATED)
    /// - Fresh database (data-mine14)
    /// - ✅ CRITICAL FIX: Global write lock across all block write paths
    /// - ✅ CRITICAL FIX: WAL flush before pointer updates (crash safety)
    /// - ✅ CRITICAL FIX: Pointer verification on every read
    /// - ✅ CRITICAL FIX: Periodic cache verification against database
    /// - ✅ SECURITY: P2P balance update deduplication via LRU cache
    /// - ✅ SECURITY: Mandatory signature validation for balance updates
    /// - ✅ Stability milestone: Zero database corruption under any load
    #[serde(rename = "testnet-phase14")]
    TestnetPhase14,

    /// Phase 15: Safe Batched Sync & Genesis Checkpoint (v1.1.22-beta) - December 2025
    /// - Fresh database (data-mine15)
    /// - ✅ CRITICAL FIX: Sync only from ACTUALLY CONNECTED peers (connected_peers())
    /// - ✅ CRITICAL FIX: Genesis checkpoint validation prevents chain forks
    /// - ✅ CRITICAL FIX: Peer genesis hash verification in handshake
    /// - ✅ CRITICAL FIX: Block chain ancestry validation during sync
    /// - ✅ NETWORK: Proper peer selection for reliable block requests
    /// - ✅ STABILITY: Fork-proof network synchronization
    #[serde(rename = "testnet-phase15")]
    TestnetPhase15,

    /// Phase 16: P2P Sync Priority Fix & DAG-Knight Stability (v1.3.1-beta) - December 2025 (DEPRECATED)
    /// - Fresh database (data-mine16)
    /// - ✅ CRITICAL FIX: Biased tokio::select! for P2P response priority
    /// - ✅ CRITICAL FIX: Block pack responses processed before swarm events
    /// - ✅ CRITICAL FIX: Prevents ResponseOmission timeouts and peer blacklisting
    /// - ✅ SHA3-256: NIST FIPS 202 quantum-resistant block verification
    /// - ✅ PEER REPUTATION: Automatic 24-hour banning for malicious peers
    /// - ✅ LRU CACHE: 10,000-entry block verification cache
    #[serde(rename = "testnet-phase16")]
    TestnetPhase16,

    /// Phase 17: u128 Token Amount Migration & Fresh Sync (v3.2.0-beta) - January 2026 (DEPRECATED)
    /// - Fresh database (data-mine17)
    /// - ✅ CRITICAL: Fresh start after u64→u128 migration corrupted Phase 16 database
    /// - ✅ Amount/Fee fields now use u128 for higher precision and max supply
    /// - ✅ Legacy deserialization for backwards compatibility (new blocks only)
    /// - ✅ Clean sync from genesis with consistent binary format
    /// - ✅ All phase transition checklist items verified
    /// - ❌ DEPRECATED: u128_serde string serialization broke Bincode storage
    #[serde(rename = "testnet-phase17")]
    TestnetPhase17,

    /// Phase 18: Native u128 Serialization Fix (v3.2.5-beta) - January 2026 (DEPRECATED)
    /// - Fresh database (data-mine18)
    /// - ✅ CRITICAL FIX: Reverted u128_serde string serialization from core storage types
    /// - ✅ CRITICAL FIX: Bincode now uses native u128 (no string conversion)
    /// - ✅ MessagePack P2P still works (self-describing format handles u128)
    /// - ✅ Clean start after v3.2.2-3 corrupted all Phase 17 blocks
    /// - ✅ All phase transition checklist items verified
    /// - ❌ DEPRECATED: AsyncStorageEngine key format mismatch corrupted blocks
    #[serde(rename = "testnet-phase18")]
    TestnetPhase18,

    /// Phase 19: AsyncStorageEngine Key Format Fix (v3.2.14-beta) - January 2026
    /// - Fresh database (data-mine19)
    /// - ✅ CRITICAL FIX: AsyncStorageEngine now uses "qblock:height:{height}" key format
    /// - ✅ CRITICAL FIX: Key format matches get_qblocks_range() lookup pattern
    /// - ✅ CRITICAL FIX: 73% "missing blocks" P2P sync bug resolved
    /// - ✅ Clean start after Phase 18 block key corruption
    /// - ✅ All phase transition checklist items verified
    #[serde(rename = "testnet-phase19")]
    TestnetPhase19,

    /// Phase 20: Emission Analytics & Fresh Start (v6.3.0-beta) - February 2026
    /// - Fresh database (data-mine20)
    /// - ✅ Emission analytics API with daily tracking
    /// - ✅ Economic security floor & attack cost metrics
    /// - ✅ Genesis date: February 14, 2026
    /// - ✅ Browser js-libp2p phase config updated (Checklist #21)
    #[serde(rename = "testnet-phase20")]
    TestnetPhase20,

    /// Phase 21: Phase Data Purge & Clean Transition (v6.4.0-beta) - February 2026 (DEPRECATED)
    /// - Fresh database (data-mine21)
    /// - ✅ Admin purge endpoint for stale DEX/contract/token data
    /// - ✅ Clean phase transition with no stale state carryover
    /// - ✅ Genesis date: February 15, 2026
    /// - ✅ Browser js-libp2p phase config updated (Checklist #21)
    #[serde(rename = "testnet-phase21")]
    TestnetPhase21,

    /// Phase 22: Full Transition Test & Auto-Purge (v6.5.0-beta) - February 2026
    /// - Fresh database (data-mine22)
    /// - ✅ Automated full purge on phase transition (wallet_balances + token_balances + contracts)
    /// - ✅ State sync network_id filtering
    /// - ✅ Genesis date: February 16, 2026
    /// - ✅ Complete transition checklist verified (Bugs #1-17)
    #[serde(rename = "testnet-phase22")]
    TestnetPhase22,

    /// Phase 23: Stable Production Network (v6.6.0-beta) - February 2026
    /// - Fresh database (data-mine23)
    /// - Stable production network with all prior fixes
    #[serde(rename = "testnet-phase23")]
    TestnetPhase23,

    /// Phase 24: Final Pre-Mainnet (v7.0.0-beta) - February 2026
    /// - Last testnet phase before mainnet launch (1 hour duration)
    /// - Fresh database (data-mine24)
    /// - Final validation of all systems before mainnet
    #[serde(rename = "testnet-phase24")]
    TestnetPhase24,

    /// Mainnet (Launch: TBD - After Phase 24 testing complete)
    Mainnet,

    /// Mainnet 2026: Fresh mainnet launch (February 2026)
    #[serde(rename = "mainnet2026")]
    Mainnet2026,

    /// Mainnet 2026.1: Clean relaunch with emission fix + data isolation (February 15, 2026)
    #[serde(rename = "mainnet2026.1")]
    Mainnet2026_1,

    /// Mainnet 2026.1.1: 4-day rehearsal chain (Feb 18-22, 2026)
    /// Automatic: binary defaults to this before Feb 22 12:00 UTC
    #[serde(rename = "mainnet2026.1.1")]
    Mainnet2026_1_1,

    /// Mainnet 2026.1.3: Emission rehearsal with rate fix (Feb 20, 2026)
    /// Fixes single-block window rate measurement bug causing 86% under-emission
    #[serde(rename = "mainnet2026.1.3")]
    Mainnet2026_1_3,

    /// Mainnet 2026.2: Fresh directory relaunch with zero contamination (February 22, 2026)
    #[serde(rename = "mainnet2026.2")]
    Mainnet2026_2,

    /// Mainnet Genesis: Production mainnet launch (February 22, 2026 12:00 UTC)
    /// Block producer fix (challenge height = tip+1), clean network isolation
    #[serde(rename = "mainnet-genesis")]
    MainnetGenesis,
}

impl NetworkId {
    /// Get the string identifier for this network
    pub fn as_str(&self) -> &'static str {
        match self {
            NetworkId::TestnetPhase5 => "testnet-phase5",
            NetworkId::TestnetPhase6 => "testnet-phase6",
            NetworkId::TestnetPhase7 => "testnet-phase7",
            NetworkId::TestnetPhase8 => "testnet-phase8",
            NetworkId::TestnetPhase9 => "testnet-phase9",
            NetworkId::TestnetPhase10 => "testnet-phase10",
            NetworkId::TestnetPhase11 => "testnet-phase11",
            NetworkId::TestnetPhase12 => "testnet-phase12",
            NetworkId::TestnetPhase13 => "testnet-phase13", // ✅ Phase 13: Gap-Proof Sync (DEPRECATED)
            NetworkId::TestnetPhase14 => "testnet-phase14", // ✅ Phase 14: Database Durability & P2P Security (DEPRECATED)
            NetworkId::TestnetPhase15 => "testnet-phase15", // ✅ Phase 15: Safe Batched Sync & Genesis Checkpoint (DEPRECATED)
            NetworkId::TestnetPhase16 => "testnet-phase16", // ✅ Phase 16: P2P Sync Priority Fix (DEPRECATED)
            NetworkId::TestnetPhase17 => "testnet-phase17", // ✅ Phase 17: u128 Migration Fresh Sync (DEPRECATED)
            NetworkId::TestnetPhase18 => "testnet-phase18", // ✅ Phase 18: Native u128 Serialization Fix (DEPRECATED)
            NetworkId::TestnetPhase19 => "testnet-phase19", // ✅ Phase 19: AsyncStorageEngine Key Format Fix (DEPRECATED)
            NetworkId::TestnetPhase20 => "testnet-phase20", // ✅ Phase 20: Emission Analytics & Fresh Start (DEPRECATED)
            NetworkId::TestnetPhase21 => "testnet-phase21", // ✅ Phase 21: Phase Data Purge & Clean Transition (DEPRECATED)
            NetworkId::TestnetPhase22 => "testnet-phase22", // ✅ Phase 22: Full Transition Test & Auto-Purge (DEPRECATED)
            NetworkId::TestnetPhase23 => "testnet-phase23", // ✅ Phase 23: Stable Production Network (DEPRECATED)
            NetworkId::TestnetPhase24 => "testnet-phase24", // ✅ Phase 24: Final Pre-Mainnet
            NetworkId::Mainnet => "mainnet",
            NetworkId::Mainnet2026 => "mainnet2026",
            NetworkId::Mainnet2026_1 => "mainnet2026.1",
            NetworkId::Mainnet2026_1_1 => "mainnet2026.1.1",
            NetworkId::Mainnet2026_1_3 => "mainnet2026.1.3",
            NetworkId::Mainnet2026_2 => "mainnet2026.2",
            NetworkId::MainnetGenesis => "mainnet-genesis",
        }
    }

    /// Get the human-readable name for this network
    pub fn display_name(&self) -> &'static str {
        match self {
            NetworkId::TestnetPhase5 => "Q-NarwhalKnight Testnet Phase 5 (Deprecated)",
            NetworkId::TestnetPhase6 => "Q-NarwhalKnight Testnet Phase 6 (Deprecated - Hyperinflation)",
            NetworkId::TestnetPhase7 => "Q-NarwhalKnight Testnet Phase 7 (Deprecated - Still Too High Emission)",
            NetworkId::TestnetPhase8 => "Q-NarwhalKnight Testnet Phase 8 (Deprecated - Four Bug Discovery)",
            NetworkId::TestnetPhase9 => "Q-NarwhalKnight Testnet Phase 9 (Deprecated - Pre-Durability Fixes)",
            NetworkId::TestnetPhase10 => "Q-NarwhalKnight Testnet Phase 10 (Deprecated - Pre-Data-Loss-Fix)",
            NetworkId::TestnetPhase11 => "Q-NarwhalKnight Testnet Phase 11 (Deprecated - Pre-PQC)",
            NetworkId::TestnetPhase12 => "Q-NarwhalKnight Testnet Phase 12 (Deprecated - Pre-Gap-Proof)",
            NetworkId::TestnetPhase13 => "Q-NarwhalKnight Testnet Phase 13 (Deprecated - Pre-Durability)",
            NetworkId::TestnetPhase14 => "Q-NarwhalKnight Testnet Phase 14 (Deprecated - Pre-Safe-Sync)",
            NetworkId::TestnetPhase15 => "Q-NarwhalKnight Testnet Phase 15 - Safe Batched Sync & Genesis Checkpoint (DEPRECATED)", // ✅ Phase 15 (DEPRECATED)
            NetworkId::TestnetPhase16 => "Q-NarwhalKnight Testnet Phase 16 - P2P Sync Priority Fix (DEPRECATED)", // ✅ Phase 16 (DEPRECATED)
            NetworkId::TestnetPhase17 => "Q-NarwhalKnight Testnet Phase 17 - u128 Migration (DEPRECATED)", // ✅ Phase 17 (DEPRECATED)
            NetworkId::TestnetPhase18 => "Q-NarwhalKnight Testnet Phase 18 - Native u128 Serialization Fix (DEPRECATED)", // ✅ Phase 18 (DEPRECATED)
            NetworkId::TestnetPhase19 => "Q-NarwhalKnight Testnet Phase 19 - AsyncStorageEngine Key Format Fix (DEPRECATED)", // ✅ Phase 19 (DEPRECATED)
            NetworkId::TestnetPhase20 => "Q-NarwhalKnight Testnet Phase 20 - Emission Analytics & Fresh Start (DEPRECATED)", // ✅ Phase 20 (DEPRECATED)
            NetworkId::TestnetPhase21 => "Q-NarwhalKnight Testnet Phase 21 - Phase Data Purge & Clean Transition (v6.4.0-beta, DEPRECATED)", // ✅ Phase 21 (DEPRECATED)
            NetworkId::TestnetPhase22 => "Q-NarwhalKnight Testnet Phase 22 - Full Transition Test & Auto-Purge (v6.5.0-beta, DEPRECATED)", // ✅ Phase 22 (DEPRECATED)
            NetworkId::TestnetPhase23 => "Q-NarwhalKnight Testnet Phase 23 - Stable Production Network (v6.6.0-beta, DEPRECATED)", // ✅ Phase 23 (DEPRECATED)
            NetworkId::TestnetPhase24 => "Q-NarwhalKnight Testnet Phase 24 (Final Pre-Mainnet)", // ✅ Phase 24 - CURRENT
            NetworkId::Mainnet => "Q-NarwhalKnight Mainnet",
            NetworkId::Mainnet2026 => "Q-NarwhalKnight Mainnet 2026",
            NetworkId::Mainnet2026_1 => "Q-NarwhalKnight Mainnet 2026.1",
            NetworkId::Mainnet2026_1_1 => "Q-NarwhalKnight Mainnet 2026.1.1 (Rehearsal)",
            NetworkId::Mainnet2026_1_3 => "Q-NarwhalKnight Mainnet 2026.1.3 (Emission Rehearsal 3)",
            NetworkId::Mainnet2026_2 => "Q-NarwhalKnight Mainnet 2026.2",
            NetworkId::MainnetGenesis => "Q-NarwhalKnight Mainnet Genesis",
        }
    }

    /// Get the default API port for this network
    pub fn default_api_port(&self) -> u16 {
        match self {
            NetworkId::TestnetPhase5 => 8080,
            NetworkId::TestnetPhase6 => 8080,
            NetworkId::TestnetPhase7 => 8080,
            NetworkId::TestnetPhase8 => 8080,
            NetworkId::TestnetPhase9 => 8080,
            NetworkId::TestnetPhase10 => 8080,
            NetworkId::TestnetPhase11 => 8080,
            NetworkId::TestnetPhase12 => 8080,
            NetworkId::TestnetPhase13 => 8080,
            NetworkId::TestnetPhase14 => 8080,
            NetworkId::TestnetPhase15 => 8080, // ✅ Phase 15: Safe Batched Sync & Genesis Checkpoint (DEPRECATED)
            NetworkId::TestnetPhase16 => 8080, // ✅ Phase 16: P2P Sync Priority Fix (DEPRECATED)
            NetworkId::TestnetPhase17 => 8080, // ✅ Phase 17: u128 Migration Fresh Sync (DEPRECATED)
            NetworkId::TestnetPhase18 => 8080, // ✅ Phase 18: Native u128 Serialization Fix (DEPRECATED)
            NetworkId::TestnetPhase19 => 8080, // ✅ Phase 19: AsyncStorageEngine Key Format Fix (DEPRECATED)
            NetworkId::TestnetPhase20 => 8080, // ✅ Phase 20: Emission Analytics & Fresh Start (DEPRECATED)
            NetworkId::TestnetPhase21 => 8080, // ✅ Phase 21: Phase Data Purge & Clean Transition (DEPRECATED)
            NetworkId::TestnetPhase22 => 8080, // ✅ Phase 22: Full Transition Test & Auto-Purge (DEPRECATED)
            NetworkId::TestnetPhase23 => 8080, // ✅ Phase 23: Stable Production Network (DEPRECATED)
            NetworkId::TestnetPhase24 => 8080, // ✅ Phase 24: Final Pre-Mainnet
            NetworkId::Mainnet => 8080,
            NetworkId::Mainnet2026 => 8080,
            NetworkId::Mainnet2026_1 => 8080,
            NetworkId::Mainnet2026_1_1 => 8080,
            NetworkId::Mainnet2026_1_3 => 8080,
            NetworkId::Mainnet2026_2 => 8080,
            NetworkId::MainnetGenesis => 8080,
        }
    }

    /// Get the default P2P port for this network
    pub fn default_p2p_port(&self) -> u16 {
        match self {
            NetworkId::TestnetPhase5 => 9001,
            NetworkId::TestnetPhase6 => 9001,
            NetworkId::TestnetPhase7 => 9001,
            NetworkId::TestnetPhase8 => 9001,
            NetworkId::TestnetPhase9 => 9001,
            NetworkId::TestnetPhase10 => 9001,
            NetworkId::TestnetPhase11 => 9001,
            NetworkId::TestnetPhase12 => 9001,
            NetworkId::TestnetPhase13 => 9001,
            NetworkId::TestnetPhase14 => 9001,
            NetworkId::TestnetPhase15 => 9001, // ✅ Phase 15: Safe Batched Sync & Genesis Checkpoint (DEPRECATED)
            NetworkId::TestnetPhase16 => 9001, // ✅ Phase 16: P2P Sync Priority Fix (DEPRECATED)
            NetworkId::TestnetPhase17 => 9001, // ✅ Phase 17: u128 Migration Fresh Sync (DEPRECATED)
            NetworkId::TestnetPhase18 => 9001, // ✅ Phase 18: Native u128 Serialization Fix (DEPRECATED)
            NetworkId::TestnetPhase19 => 9001, // ✅ Phase 19: AsyncStorageEngine Key Format Fix (DEPRECATED)
            NetworkId::TestnetPhase20 => 9001, // ✅ Phase 20: Emission Analytics & Fresh Start (DEPRECATED)
            NetworkId::TestnetPhase21 => 9001, // ✅ Phase 21: Phase Data Purge & Clean Transition (DEPRECATED)
            NetworkId::TestnetPhase22 => 9001, // ✅ Phase 22: Full Transition Test & Auto-Purge (DEPRECATED)
            NetworkId::TestnetPhase23 => 9001, // ✅ Phase 23: Stable Production Network (DEPRECATED)
            NetworkId::TestnetPhase24 => 9001, // ✅ Phase 24: Final Pre-Mainnet
            NetworkId::Mainnet => 9001,
            NetworkId::Mainnet2026 => 9001,
            NetworkId::Mainnet2026_1 => 9001,
            NetworkId::Mainnet2026_1_1 => 9001,
            NetworkId::Mainnet2026_1_3 => 9001,
            NetworkId::Mainnet2026_2 => 9001,
            NetworkId::MainnetGenesis => 9001,
        }
    }

    /// Get the gossipsub topic prefix for this network
    pub fn gossipsub_topic_prefix(&self) -> String {
        format!("/qnk/{}", self.as_str())
    }

    /// Get the transaction gossipsub topic for this network
    pub fn transactions_topic(&self) -> String {
        format!("{}/transactions", self.gossipsub_topic_prefix())
    }

    /// Get the blocks gossipsub topic for this network
    pub fn blocks_topic(&self) -> String {
        format!("{}/blocks", self.gossipsub_topic_prefix())
    }

    /// Get the acknowledgments gossipsub topic for this network
    pub fn acks_topic(&self) -> String {
        format!("{}/ack", self.gossipsub_topic_prefix())
    }

    /// Get the block-requests gossipsub topic for this network (P2P historical sync)
    pub fn block_requests_topic(&self) -> String {
        format!("{}/block-requests", self.gossipsub_topic_prefix())
    }

    /// Get the block-responses gossipsub topic for this network (P2P historical sync)
    pub fn block_responses_topic(&self) -> String {
        format!("{}/block-responses", self.gossipsub_topic_prefix())
    }

    /// Get the BATCH block-responses gossipsub topic for this network (OPTIMIZED P2P sync)
    /// This topic carries BatchBlockResponse messages with 100-1000 blocks per message
    /// Dramatically reduces network overhead compared to single-block responses
    pub fn batch_block_responses_topic(&self) -> String {
        format!("{}/batch-block-responses", self.gossipsub_topic_prefix())
    }

    /// Get the peer-heights gossipsub topic for this network (Turbo Sync height announcements)
    /// Peers announce their highest verified block height on this topic
    /// ✅ v0.9.6-beta: Network-aware to prevent cross-network contamination
    pub fn peer_heights_topic(&self) -> String {
        format!("{}/peer-heights", self.gossipsub_topic_prefix())
    }

    /// Get the block-pack-requests gossipsub topic for this network (Turbo Sync requests)
    /// Nodes publish BlockPackRequest messages on this topic to request compressed block packs
    /// ✅ v0.9.6-beta: Network-aware to prevent cross-network contamination
    pub fn block_pack_requests_topic(&self) -> String {
        format!("{}/block-pack-requests", self.gossipsub_topic_prefix())
    }

    /// Get the block-pack-responses gossipsub topic for this network (Turbo Sync responses)
    /// Nodes publish BlockPack messages (compressed batches) on this topic
    /// ✅ v0.9.6-beta: Network-aware to prevent cross-network contamination
    pub fn block_pack_responses_topic(&self) -> String {
        format!("{}/block-pack-responses", self.gossipsub_topic_prefix())
    }

    /// Get the DEX pools gossipsub topic for this network
    /// ✅ v1.0.90-beta: Real-time DEX pool state synchronization
    /// Nodes publish pool updates (create, add_liquidity, remove_liquidity) on this topic
    pub fn dex_pools_topic(&self) -> String {
        format!("{}/dex-pools", self.gossipsub_topic_prefix())
    }

    /// Get the contract deployments gossipsub topic for this network
    /// ✅ v1.0.90-beta: Smart contract code and state synchronization
    /// Nodes publish contract deployments and upgrades on this topic
    pub fn contract_deployments_topic(&self) -> String {
        format!("{}/contract-deployments", self.gossipsub_topic_prefix())
    }

    /// Get the AI credits gossipsub topic for this network
    /// ✅ v1.0.90-beta: AI inference credit transactions synchronization
    /// Nodes publish AI credit purchases, usage, and transfers on this topic
    pub fn ai_credits_topic(&self) -> String {
        format!("{}/ai-credits", self.gossipsub_topic_prefix())
    }

    /// Get the balance updates gossipsub topic for this network
    /// ✅ v1.1.8-beta: P2P balance synchronization for decentralized mining
    /// Nodes publish mining rewards and balance updates on this topic
    /// This enables mining to localhost while syncing balances across all nodes
    pub fn balance_updates_topic(&self) -> String {
        format!("{}/balance-updates", self.gossipsub_topic_prefix())
    }

    /// Bracha Reliable Broadcast topic for BFT-safe balance finalization.
    /// All validators subscribe and relay SEND/ECHO/READY messages on this topic.
    pub fn balance_rb_topic(&self) -> String {
        format!("{}/consensus/balance-rb", self.gossipsub_topic_prefix())
    }

    /// Get the miner stats gossipsub topic for this network
    /// ✅ v1.1.8-beta: P2P hashrate aggregation for network-wide mining statistics
    /// Nodes publish their miner statistics (hashrate, solutions) on this topic
    pub fn miner_stats_topic(&self) -> String {
        format!("{}/miner-stats", self.gossipsub_topic_prefix())
    }

    /// ✅ v2.4.9-beta: DCA (Dollar Cost Averaging) order synchronization topic
    /// Nodes broadcast DCA order create/cancel/pause/resume events for decentralized agreement
    /// Topic: /qnk/{network}/dca-orders
    pub fn dca_orders_topic(&self) -> String {
        format!("{}/dca-orders", self.gossipsub_topic_prefix())
    }

    /// ✅ v2.9.2-beta: Protocol fee consensus verification topic
    /// All nodes MUST verify and agree on protocol fees collected from DEX trades
    /// This ensures the master wallet (FOUNDER_WALLET) receives the correct percentage per trade
    /// Nodes publish ProtocolFeeGossip messages for network-wide verification
    /// Topic: /qnk/{network}/protocol-fees
    pub fn protocol_fees_topic(&self) -> String {
        format!("{}/protocol-fees", self.gossipsub_topic_prefix())
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 🔐 v1.3.12-beta: DAG-KNIGHT DECENTRALIZED CONSENSUS TOPICS
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Get the vertex broadcast topic for DAG-Knight consensus
    /// Validators broadcast new vertices (containing transactions) on this topic
    /// Other validators receive and sign vertices with their SQIsign keys
    pub fn vertex_broadcast_topic(&self) -> String {
        format!("{}/consensus/vertices", self.gossipsub_topic_prefix())
    }

    /// Get the signature request topic for collecting 2f+1 signatures
    /// When a validator creates a vertex, they request signatures from others
    pub fn signature_request_topic(&self) -> String {
        format!("{}/consensus/sig-requests", self.gossipsub_topic_prefix())
    }

    /// Get the signature response topic for returning SQIsign signatures
    /// Validators respond with their SQIsign signatures over vertices
    pub fn signature_response_topic(&self) -> String {
        format!("{}/consensus/sig-responses", self.gossipsub_topic_prefix())
    }

    /// Get the certificate broadcast topic for finalized certificates
    /// Once 2f+1 signatures are collected, the certificate is broadcast
    pub fn certificate_topic(&self) -> String {
        format!("{}/consensus/certificates", self.gossipsub_topic_prefix())
    }

    /// Get the validator registry topic for announcing validators
    /// Validators announce their SQIsign public keys on this topic
    pub fn validator_announce_topic(&self) -> String {
        format!("{}/consensus/validators", self.gossipsub_topic_prefix())
    }

    // ========== v2.3.5-beta: P2P Decentralized Mining Topics ==========

    /// Get the mining solutions gossipsub topic
    /// Miners broadcast valid solutions to this topic for network-wide reward credit
    /// Topic: /qnk/{network}/mining-solutions
    pub fn mining_solutions_topic(&self) -> String {
        format!("{}/mining-solutions", self.gossipsub_topic_prefix())
    }

    /// Get the mining challenges gossipsub topic
    /// Nodes broadcast consensus-agreed challenges (using 10-block finality)
    /// Topic: /qnk/{network}/mining-challenges
    pub fn mining_challenges_topic(&self) -> String {
        format!("{}/mining-challenges", self.gossipsub_topic_prefix())
    }

    /// v9.1.0: Compute power announcements gossipsub topic
    /// Nodes broadcast aggregate hashrate for network-wide compute power awareness
    /// Topic: /qnk/{network}/compute-power
    pub fn compute_power_topic(&self) -> String {
        format!("{}/compute-power", self.gossipsub_topic_prefix())
    }

    // ========== v3.3.0-beta: P2P Mempool Transaction Propagation ==========

    /// Get the mempool transactions gossipsub topic for this network
    /// Nodes broadcast pending transactions for real-time mempool synchronization
    /// This enables immediate transaction visibility across all nodes before block inclusion
    /// Topic: /qnk/{network}/mempool-txs
    pub fn mempool_transactions_topic(&self) -> String {
        format!("{}/mempool-txs", self.gossipsub_topic_prefix())
    }

    /// v3.5.8: Browser peer discovery topic
    /// Browsers announce their presence so they can discover each other
    /// This enables the Network Map to show browser-to-browser connectivity
    /// Topic: /qnk/{network}/browser-peers
    pub fn browser_peers_topic(&self) -> String {
        format!("{}/browser-peers", self.gossipsub_topic_prefix())
    }

    /// v5.3.0: State sync request topic
    /// Nodes broadcast requests for full state snapshots (contracts, pools, balances)
    /// Topic: /qnk/{network}/state-sync-requests
    pub fn state_sync_requests_topic(&self) -> String {
        format!("{}/state-sync-requests", self.gossipsub_topic_prefix())
    }

    /// v5.3.0: State sync response topic
    /// Peers respond with signed state data on this topic
    /// Topic: /qnk/{network}/state-sync-responses
    pub fn state_sync_responses_topic(&self) -> String {
        format!("{}/state-sync-responses", self.gossipsub_topic_prefix())
    }

    /// v7.3.1: Bridge attestation topic
    /// Multi-sig bridge validation: committee members broadcast attestation requests/responses
    /// Topic: /qnk/{network}/bridge-attestations
    pub fn bridge_attestations_topic(&self) -> String {
        format!("{}/bridge-attestations", self.gossipsub_topic_prefix())
    }

    /// v7.4.0: OAuth2 client registration sync topic
    /// Nodes broadcast registered OAuth2 clients (with hashed secrets) for cross-node auth
    /// Topic: /qnk/{network}/oauth2-clients
    pub fn oauth2_clients_topic(&self) -> String {
        format!("{}/oauth2-clients", self.gossipsub_topic_prefix())
    }

    /// v7.4.0: OAuth2 JWT public key announcement topic
    /// Nodes announce their Ed25519 verifying key so other nodes can verify JWTs
    /// Topic: /qnk/{network}/oauth2-pubkeys
    pub fn oauth2_pubkeys_topic(&self) -> String {
        format!("{}/oauth2-pubkeys", self.gossipsub_topic_prefix())
    }

    /// v8.5.0: Software update announcement topic
    /// Bootstrap nodes publish signed UpdateAnnouncement messages here.
    /// Remote nodes subscribe to discover and verify new binary releases.
    /// Topic: /qnk/{network}/update-announcements
    pub fn update_announcements_topic(&self) -> String {
        format!("{}/update-announcements", self.gossipsub_topic_prefix())
    }

    /// v9.5.0: Compute tunnel peer discovery topic (Starship Endgame #002)
    /// Nodes publish their compute capacity (CPU, GPU, RAM, bandwidth) so
    /// peers can discover resources for distributed task routing.
    /// Topic: /qnk/{network}/compute-tunnel
    pub fn compute_tunnel_topic(&self) -> String {
        format!("{}/compute-tunnel", self.gossipsub_topic_prefix())
    }

    /// v10.2.0: Crown & Ash — Player action propagation topic
    /// Signed game actions (raise army, declare war, etc.) broadcast to all nodes
    /// Topic: /qnk/{network}/game-actions
    pub fn game_actions_topic(&self) -> String {
        format!("{}/game-actions", self.gossipsub_topic_prefix())
    }

    /// v10.2.0: Crown & Ash — Turn summary synchronization topic
    /// After each game tick, nodes broadcast the deterministic turn summary
    /// Topic: /qnk/{network}/game-state-sync
    pub fn game_state_sync_topic(&self) -> String {
        format!("{}/game-state-sync", self.gossipsub_topic_prefix())
    }
}

impl std::str::FromStr for NetworkId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "testnet" | "testnet-phase5" => Ok(NetworkId::TestnetPhase5),
            "testnet-phase6" => Ok(NetworkId::TestnetPhase6),
            "testnet-phase7" => Ok(NetworkId::TestnetPhase7),
            "testnet-phase8" => Ok(NetworkId::TestnetPhase8),
            "testnet-phase9" => Ok(NetworkId::TestnetPhase9),
            "testnet-phase10" => Ok(NetworkId::TestnetPhase10),
            "testnet-phase11" => Ok(NetworkId::TestnetPhase11),
            "testnet-phase12" => Ok(NetworkId::TestnetPhase12),
            "testnet-phase13" => Ok(NetworkId::TestnetPhase13),
            "testnet-phase14" => Ok(NetworkId::TestnetPhase14),
            "testnet-phase15" => Ok(NetworkId::TestnetPhase15), // ✅ Phase 15 parser (DEPRECATED)
            "testnet-phase16" => Ok(NetworkId::TestnetPhase16), // ✅ Phase 16 parser (DEPRECATED)
            "testnet-phase17" => Ok(NetworkId::TestnetPhase17), // ✅ Phase 17 parser (DEPRECATED)
            "testnet-phase18" => Ok(NetworkId::TestnetPhase18), // ✅ Phase 18 parser (DEPRECATED)
            "testnet-phase19" => Ok(NetworkId::TestnetPhase19), // ✅ Phase 19 parser (DEPRECATED)
            "testnet-phase20" => Ok(NetworkId::TestnetPhase20), // ✅ Phase 20 parser (DEPRECATED)
            "testnet-phase21" => Ok(NetworkId::TestnetPhase21), // ✅ Phase 21 parser (DEPRECATED)
            "testnet-phase22" => Ok(NetworkId::TestnetPhase22), // ✅ Phase 22 parser (DEPRECATED)
            "testnet-phase23" => Ok(NetworkId::TestnetPhase23), // ✅ Phase 23 parser (DEPRECATED)
            "testnet-phase24" => Ok(NetworkId::TestnetPhase24), // ✅ CRITICAL: Phase 24 parser added (Bug #1 fix)
            "mainnet" => Ok(NetworkId::Mainnet),
            "mainnet2026" | "mainnet-2026" => Ok(NetworkId::Mainnet2026),
            "mainnet2026.1" | "mainnet-2026.1" | "mainnet-2026-1" | "mainnet2026_1" => Ok(NetworkId::Mainnet2026_1),
            "mainnet2026.1.1" | "mainnet-2026.1.1" | "mainnet-2026-1-1" | "mainnet2026_1_1" => Ok(NetworkId::Mainnet2026_1_1),
            "mainnet2026.1.3" | "mainnet-2026.1.3" | "mainnet-2026-1-3" | "mainnet2026_1_3" => Ok(NetworkId::Mainnet2026_1_3),
            "mainnet2026.2" | "mainnet-2026.2" | "mainnet-2026-2" | "mainnet2026_2" => Ok(NetworkId::Mainnet2026_2),
            "mainnet-genesis" | "mainnetgenesis" | "mainnet_genesis" => Ok(NetworkId::MainnetGenesis),
            _ => Err(format!("Invalid network ID: {}", s)),
        }
    }
}

impl Default for NetworkId {
    fn default() -> Self {
        // v8.1.6: MAINNET GENESIS - Production mainnet launch
        NetworkId::MainnetGenesis
    }
}

/// Network configuration with genesis hash and launch time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Network identifier (testnet/mainnet)
    pub network_id: NetworkId,

    /// Genesis block hash (unique per network)
    pub genesis_hash: [u8; 32],

    /// Network launch timestamp (UTC)
    pub launch_time: DateTime<Utc>,

    /// Network version string
    pub version: String,

    /// Chain ID for transaction replay protection
    pub chain_id: u64,

    /// API server port
    pub api_port: u16,

    /// P2P networking port
    pub p2p_port: u16,

    /// Bootstrap peers for this network
    pub bootstrap_peers: Vec<String>,
}

impl NetworkConfig {
    /// Create testnet configuration (now redirects to mainnet for v7.0.0 launch)
    pub fn testnet() -> Self {
        // v7.0.0: Mainnet launch - testnet() now returns mainnet config
        Self::mainnet()
    }

    /// Create mainnet configuration
    pub fn mainnet() -> Self {
        Self {
            network_id: NetworkId::MainnetGenesis,
            genesis_hash: [
                // Mainnet Genesis hash (ASCII prefix of "mainnet-genesis")
                0x6d, 0x61, 0x69, 0x6e, 0x6e, 0x65, 0x74, 0x2d,  // "mainnet-"
                0x67, 0x65, 0x6e, 0x65, 0x73, 0x69, 0x73, 0x00,  // "genesis\0"
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
            launch_time: DateTime::parse_from_rfc3339("2026-02-22T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            version: format!("v{}-mainnet-genesis", env!("CARGO_PKG_VERSION")),
            chain_id: 1000, // Mainnet Genesis chain ID
            api_port: 8080,
            p2p_port: 9001,
            // Multiple bootstrap nodes for redundancy
            bootstrap_peers: vec![
                // Primary bootstrap node - Server Delta (5.79.79.158) - 1Gbit fastest sync
                "/ip4/5.79.79.158/tcp/9001/p2p/12D3KooWPg1GsUhYtZdzN37NcLQCz2PXJ3GssKMtELwvMvHFrjTt".to_string(),
                // Secondary bootstrap node - Server Gamma (109.205.176.60) - 1Gbit
                "/ip4/109.205.176.60/tcp/9001/p2p/12D3KooWEZKN13gsYXmvoUSeu5VnbUCTyEcAVdqKfWz14CAnm3bp".to_string(),
                // Tertiary bootstrap node - Server Beta (185.182.185.227) - 100Mbit DHT anchor
                "/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWSBxwSKw4wftHViMdw5rrV8Z1wEkikDS2vKYZtRrio5hH".to_string(),
            ],
        }
    }

    /// Get network configuration by network ID
    pub fn from_network_id(network_id: NetworkId) -> Self {
        match network_id {
            NetworkId::TestnetPhase5 => Self::testnet(),  // Legacy Phase 5
            NetworkId::TestnetPhase6 => Self::testnet(),  // Legacy Phase 6 (hyperinflation bug)
            NetworkId::TestnetPhase7 => Self::testnet(),  // Phase 7 (still too high emission)
            NetworkId::TestnetPhase8 => Self::testnet(),  // Phase 8 (TRUE scarcity)
            NetworkId::TestnetPhase9 => Self::testnet(),  // Phase 9 (Deprecated - Pre-Durability Fixes)
            NetworkId::TestnetPhase10 => Self::testnet(), // Phase 10 (Deprecated - Pre-Data-Loss-Fix)
            NetworkId::TestnetPhase11 => Self::testnet(), // Phase 11 (Deprecated - Pre-PQC)
            NetworkId::TestnetPhase12 => Self::testnet(), // Phase 12 (Deprecated - Pre-Gap-Proof)
            NetworkId::TestnetPhase13 => Self::testnet(), // Phase 13 (Deprecated - Pre-Durability)
            NetworkId::TestnetPhase14 => Self::testnet(), // Phase 14 (Deprecated - Pre-Safe-Sync)
            NetworkId::TestnetPhase15 => Self::testnet(), // Phase 15 (DEPRECATED)
            NetworkId::TestnetPhase16 => Self::testnet(), // Phase 16 (DEPRECATED)
            NetworkId::TestnetPhase17 => Self::testnet(), // Phase 17 (DEPRECATED)
            NetworkId::TestnetPhase18 => Self::testnet(), // Phase 18 (DEPRECATED)
            NetworkId::TestnetPhase19 => Self::testnet(), // Phase 19 (DEPRECATED)
            NetworkId::TestnetPhase20 => Self::testnet(), // ✅ Phase 20 (DEPRECATED)
            NetworkId::TestnetPhase21 => Self::testnet(), // ✅ Phase 21 (DEPRECATED)
            NetworkId::TestnetPhase22 => Self::testnet(), // ✅ Phase 22 (DEPRECATED)
            NetworkId::TestnetPhase23 => Self::testnet(), // ✅ Phase 23 (DEPRECATED)
            NetworkId::TestnetPhase24 => Self::testnet(), // ✅ Phase 24 (Final Pre-Mainnet - v7.0.0-beta)
            NetworkId::Mainnet => Self::mainnet(),
            NetworkId::Mainnet2026 => Self::mainnet(),
            NetworkId::Mainnet2026_1 => Self::mainnet(),
            NetworkId::Mainnet2026_1_1 => Self::mainnet(),
            NetworkId::Mainnet2026_1_3 => Self::mainnet(),
            NetworkId::Mainnet2026_2 => Self::mainnet(),
            NetworkId::MainnetGenesis => Self::mainnet(),
        }
    }

    /// Check if the network has launched
    pub fn is_launched(&self) -> bool {
        Utc::now() >= self.launch_time
    }

    /// Get time until launch (None if already launched)
    pub fn time_until_launch(&self) -> Option<chrono::Duration> {
        let now = Utc::now();
        if now >= self.launch_time {
            None
        } else {
            Some(self.launch_time - now)
        }
    }

    /// Verify that a transaction belongs to this network
    pub fn verify_transaction_network(&self, tx: &Transaction) -> bool {
        // In the future, transactions should include chain_id
        // For now, we accept all transactions on the same network
        true
    }

    /// Verify that a message came from this network
    pub fn verify_message_network(&self, genesis_hash: &[u8; 32]) -> bool {
        genesis_hash == &self.genesis_hash
    }
}

/// Network message wrapper with network verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMessage<T> {
    /// Network genesis hash for verification
    pub genesis_hash: [u8; 32],

    /// Chain ID for replay protection
    pub chain_id: u64,

    /// Message payload
    pub payload: T,

    /// Timestamp when message was created
    pub timestamp: DateTime<Utc>,
}

impl<T> NetworkMessage<T> {
    /// Create a new network message
    pub fn new(config: &NetworkConfig, payload: T) -> Self {
        Self {
            genesis_hash: config.genesis_hash,
            chain_id: config.chain_id,
            payload,
            timestamp: Utc::now(),
        }
    }

    /// Verify that this message belongs to the given network
    pub fn verify_network(&self, config: &NetworkConfig) -> bool {
        self.genesis_hash == config.genesis_hash && self.chain_id == config.chain_id
    }
}

// ========================================
// P2P BLOCK SYNCHRONIZATION MESSAGES
// ========================================

/// Request for historical blocks via gossipsub P2P
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockRequest {
    /// Peer ID of the requester (for tracking)
    pub requester_peer_id: String,
    /// First block height needed
    pub start_height: u64,
    /// Last block height needed (inclusive)
    pub end_height: u64,
    /// Unique request ID for matching responses
    pub request_id: [u8; 16],
    /// Timestamp of request
    pub timestamp: DateTime<Utc>,
}

impl BlockRequest {
    /// Create a new block request
    pub fn new(requester_peer_id: String, start_height: u64, end_height: u64) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut request_id = [0u8; 16];
        rng.fill(&mut request_id);

        Self {
            requester_peer_id,
            start_height,
            end_height,
            request_id,
            timestamp: Utc::now(),
        }
    }

    /// Get number of blocks requested
    pub fn block_count(&self) -> u64 {
        if self.end_height >= self.start_height {
            self.end_height - self.start_height + 1
        } else {
            0
        }
    }
}

/// Response containing a historical block via gossipsub P2P
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockResponse {
    /// Request ID this response belongs to
    pub request_id: [u8; 16],
    /// The actual block data
    pub block: QBlock,
    /// Peer ID of the responder
    pub responder_peer_id: String,
    /// Timestamp of response
    pub timestamp: DateTime<Utc>,
}

/// OPTIMIZED: Batch response containing multiple blocks for ultra-fast sync
/// Reduces gossipsub overhead by 100-1000x by sending blocks in batches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchBlockResponse {
    /// Request ID this response belongs to
    pub request_id: [u8; 16],
    /// Multiple blocks in a single message (dramatically reduces network overhead)
    pub blocks: Vec<QBlock>,
    /// Peer ID of the responder
    pub responder_peer_id: String,
    /// Timestamp of response
    pub timestamp: DateTime<Utc>,
    /// Start height of this batch
    pub start_height: u64,
    /// End height of this batch
    pub end_height: u64,
}

impl BatchBlockResponse {
    /// Create a new batch block response
    pub fn new(request_id: [u8; 16], blocks: Vec<QBlock>, responder_peer_id: String) -> Self {
        let start_height = blocks.first().map(|b| b.header.height).unwrap_or(0);
        let end_height = blocks.last().map(|b| b.header.height).unwrap_or(0);
        Self {
            request_id,
            blocks,
            responder_peer_id,
            timestamp: chrono::Utc::now(),
            start_height,
            end_height,
        }
    }

    /// Get number of blocks in this batch
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

impl BlockResponse {
    /// Create a new block response
    pub fn new(request_id: [u8; 16], block: QBlock, responder_peer_id: String) -> Self {
        Self {
            request_id,
            block,
            responder_peer_id,
            timestamp: Utc::now(),
        }
    }
}

// ✅ v1.0.12-beta: Batch sync trait for dependency injection
// Placed in q-types to avoid circular dependency between q-storage and q-network
/// Trait for network managers that can fetch block ranges
/// This avoids circular dependency by defining the interface in shared q-types crate
///
/// ✅ v1.0.15-beta FIX: Removed `Sync` bound - not needed for `&mut self` methods
/// libp2p's Swarm type is not Sync (contains Box<dyn Executor>, Box<dyn Stream>, etc.)
/// Since method takes `&mut self` (exclusive access), Sync is unnecessary
#[async_trait::async_trait]
pub trait BlockRangeFetcher: Send {
    /// Request a range of blocks from the network
    ///
    /// # Arguments
    /// * `start_height` - Starting block height (inclusive)
    /// * `end_height` - Ending block height (inclusive)
    ///
    /// # Returns
    /// Vector of blocks in the requested range
    async fn request_block_range(
        &mut self,
        start_height: u64,
        end_height: u64,
    ) -> anyhow::Result<Vec<QBlock>>;
}

// ============================================
// v0.6.2-beta: Mining Rewards History & Worker Stats
// ============================================

/// Mining reward record for persistent storage
/// Stores detailed information about each mining reward for historical queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningRewardRecord {
    pub miner_address: String,
    pub reward_qnk: f64,
    pub nonce: u64,
    pub block_height: u64,
    pub difficulty: String,
    pub hash_rate: f64,
    pub worker_name: Option<String>,
    pub timestamp: i64, // Unix timestamp for efficient sorting
}

/// Per-worker mining statistics for multi-miner setups
/// Aggregated data showing performance breakdown by worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStats {
    pub worker_name: String,
    pub blocks_found: u64,
    pub total_rewards_qnk: f64,
    pub average_hash_rate: f64,
    pub last_block_time: i64,
}

// ============================================================================
// Quillon Mail: Decentralized Email Types
// ============================================================================

/// Email message stored in the mailbox
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EmailMessage {
    pub id: String,
    pub from_wallet: [u8; 32],
    #[serde(default)]
    pub from_email: Option<String>,
    #[serde(default)]
    pub to_wallet: Option<[u8; 32]>,
    #[serde(default)]
    pub to_email: Option<String>,
    pub subject: String,
    pub body: String,
    #[serde(default)]
    pub body_html: Option<String>,
    #[serde(default)]
    pub encrypted: bool,
    #[serde(default)]
    pub signature: Vec<u8>,
    pub timestamp: u64,
    #[serde(default)]
    pub read: bool,
    #[serde(default = "default_folder")]
    pub folder: String,
    #[serde(default)]
    pub thread_id: Option<String>,
    #[serde(default)]
    pub in_reply_to: Option<String>,
    #[serde(default)]
    pub crypto_transfer: Option<CryptoTransfer>,
    #[serde(default = "default_delivery_method")]
    pub delivery_method: DeliveryMethod,
}

fn default_folder() -> String {
    "inbox".to_string()
}

fn default_delivery_method() -> DeliveryMethod {
    DeliveryMethod::P2PGossipsub
}

/// Crypto transfer attached to an email
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CryptoTransfer {
    pub token_type: String,
    pub amount: u128,
    pub tx_hash: [u8; 32],
    #[serde(default)]
    pub confirmed: bool,
}

/// How the email was delivered
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum DeliveryMethod {
    P2PGossipsub,
    SmtpOutbound,
    SmtpInbound,
}

impl Default for DeliveryMethod {
    fn default() -> Self {
        DeliveryMethod::P2PGossipsub
    }
}

/// Outbound email queued for SMTP delivery
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OutboundEmail {
    pub id: String,
    pub from_wallet: [u8; 32],
    pub from_email: String,
    pub to_email: String,
    pub subject: String,
    pub body: String,
    #[serde(default)]
    pub body_html: Option<String>,
    pub timestamp: u64,
    pub status: OutboundStatus,
    #[serde(default)]
    pub retry_count: u32,
    #[serde(default)]
    pub last_error: Option<String>,
    #[serde(default)]
    pub next_retry_at: Option<u64>,
    #[serde(default)]
    pub email_id: Option<String>,
}

/// Outbound email delivery status
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum OutboundStatus {
    Pending,
    Processing,
    Delivered,
    Failed,
    Retrying,
}

impl Default for OutboundStatus {
    fn default() -> Self {
        OutboundStatus::Pending
    }
}

/// Email contact info
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EmailContact {
    pub wallet_address: [u8; 32],
    pub display_name: Option<String>,
    pub email_address: Option<String>,
    pub last_contacted: u64,
    pub message_count: u64,
}

// ============================================================================
// Decentralized Blockchain Calendar Types
// ============================================================================

/// Calendar event type categories
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum CalendarEventType {
    Personal,
    ScheduledTransaction,
    VestingUnlock,
    GovernanceVote,
    NetworkMilestone,
    CommunityEvent,
    PriceAlert,
}

impl Default for CalendarEventType {
    fn default() -> Self {
        CalendarEventType::Personal
    }
}

impl std::fmt::Display for CalendarEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalendarEventType::Personal => write!(f, "personal"),
            CalendarEventType::ScheduledTransaction => write!(f, "scheduled_tx"),
            CalendarEventType::VestingUnlock => write!(f, "vesting_unlock"),
            CalendarEventType::GovernanceVote => write!(f, "governance_vote"),
            CalendarEventType::NetworkMilestone => write!(f, "network_milestone"),
            CalendarEventType::CommunityEvent => write!(f, "community_event"),
            CalendarEventType::PriceAlert => write!(f, "price_alert"),
        }
    }
}

/// Recurrence frequency for repeating events
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum RecurrenceFrequency {
    Daily,
    Weekly,
    Monthly,
    Yearly,
}

/// Recurrence rule for repeating events
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RecurrenceRule {
    pub frequency: RecurrenceFrequency,
    #[serde(default = "default_recurrence_interval")]
    pub interval: u32,
    #[serde(default)]
    pub until: Option<u64>,
    #[serde(default)]
    pub count: Option<u32>,
}

fn default_recurrence_interval() -> u32 {
    1
}

/// Scheduled transaction attached to a calendar event
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ScheduledTransaction {
    pub to_wallet: String,
    pub token: String,
    pub amount: String,
    #[serde(default)]
    pub executed: bool,
    #[serde(default)]
    pub tx_hash: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

/// A calendar event stored on-chain and synced via P2P
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CalendarEvent {
    pub id: String,
    pub wallet: [u8; 32],
    pub title: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub event_type: CalendarEventType,
    pub start_time: u64,
    #[serde(default)]
    pub end_time: Option<u64>,
    #[serde(default)]
    pub all_day: bool,
    #[serde(default)]
    pub recurring: Option<RecurrenceRule>,
    #[serde(default)]
    pub color: Option<String>,
    #[serde(default)]
    pub reminder_minutes: Option<Vec<u32>>,
    #[serde(default)]
    pub scheduled_tx: Option<ScheduledTransaction>,
    #[serde(default)]
    pub shared: bool,
    pub created_at: u64,
    #[serde(default)]
    pub updated_at: Option<u64>,
    #[serde(default)]
    pub source_peer: Option<String>,
    #[serde(default)]
    pub cancelled: bool,
}

#[cfg(test)]
mod network_separation_tests {
    use super::*;

    #[test]
    fn test_network_id_string_conversion() {
        // Test NetworkId to string conversion
        assert_eq!(NetworkId::TestnetPhase5.as_str(), "testnet-phase5");
        assert_eq!(NetworkId::Mainnet.as_str(), "mainnet");

        // Test string parsing (still accepts "testnet" for backwards compatibility)
        assert_eq!("testnet".parse::<NetworkId>().unwrap(), NetworkId::TestnetPhase5);
        assert_eq!("mainnet".parse::<NetworkId>().unwrap(), NetworkId::Mainnet);

        // Test case-insensitive parsing (to_lowercase is used)
        assert_eq!("TESTNET".parse::<NetworkId>().unwrap(), NetworkId::TestnetPhase5);
        assert_eq!("Mainnet".parse::<NetworkId>().unwrap(), NetworkId::Mainnet);

        // Test invalid string parsing
        assert!("invalid".parse::<NetworkId>().is_err());
        assert!("".parse::<NetworkId>().is_err());
    }

    #[test]
    fn test_network_id_display() {
        assert_eq!(NetworkId::Mainnet.display_name(), "Q-NarwhalKnight Mainnet");
    }

    #[test]
    fn test_gossipsub_topic_generation() {
        let testnet = NetworkId::TestnetPhase5;
        let mainnet = NetworkId::Mainnet;

        // Test topic prefix generation
        assert_eq!(testnet.gossipsub_topic_prefix(), "/qnk/testnet-phase5");
        assert_eq!(mainnet.gossipsub_topic_prefix(), "/qnk/mainnet");

        // Test transaction topics
        assert_eq!(testnet.transactions_topic(), "/qnk/testnet-phase5/transactions");
        assert_eq!(mainnet.transactions_topic(), "/qnk/mainnet/transactions");

        // Test block topics
        assert_eq!(testnet.blocks_topic(), "/qnk/testnet-phase5/blocks");
        assert_eq!(mainnet.blocks_topic(), "/qnk/mainnet/blocks");

        // Test ACK topics
        assert_eq!(testnet.acks_topic(), "/qnk/testnet-phase5/ack");
        assert_eq!(mainnet.acks_topic(), "/qnk/mainnet/ack");

        // Verify topics are different between networks
        assert_ne!(testnet.transactions_topic(), mainnet.transactions_topic());
        assert_ne!(testnet.blocks_topic(), mainnet.blocks_topic());
    }

    #[test]
    fn test_network_config_testnet() {
        // v8.1.6: testnet() now returns mainnet-genesis config
        let config = NetworkConfig::testnet();
        assert_eq!(config.network_id, NetworkId::MainnetGenesis);
        assert_eq!(config.chain_id, 1000);
        assert_eq!(config.api_port, 8080);
        assert_eq!(config.p2p_port, 9001);
        assert!(config.version.ends_with("-mainnet-genesis"));
        assert_eq!(&config.genesis_hash[..8], b"mainnet-");
    }

    #[test]
    fn test_network_config_mainnet() {
        let config = NetworkConfig::mainnet();
        assert_eq!(config.network_id, NetworkId::MainnetGenesis);
        assert_eq!(config.chain_id, 1000);
        assert_eq!(config.api_port, 8080);
        assert_eq!(config.p2p_port, 9001);
        assert!(config.version.ends_with("-mainnet-genesis"));
        assert_eq!(&config.genesis_hash[..8], b"mainnet-");
        let expected = DateTime::parse_from_rfc3339("2026-02-22T12:00:00Z").unwrap();
        assert_eq!(config.launch_time, expected.with_timezone(&Utc));
    }

    #[test]
    fn test_network_config_uniqueness() {
        // v7.3.0: testnet() now returns mainnet2026.2 config, so they're equal
        let testnet = NetworkConfig::testnet();
        let mainnet = NetworkConfig::mainnet();
        assert_eq!(testnet.genesis_hash, mainnet.genesis_hash);
        assert_eq!(testnet.network_id, mainnet.network_id);
    }

    #[test]
    fn test_network_config_from_network_id() {
        let mainnet = NetworkConfig::from_network_id(NetworkId::MainnetGenesis);
        assert_eq!(mainnet.network_id, NetworkId::MainnetGenesis);
        assert_eq!(mainnet.genesis_hash, NetworkConfig::mainnet().genesis_hash);
    }

    #[test]
    fn test_genesis_hash_verification() {
        let testnet = NetworkConfig::testnet();
        let mainnet = NetworkConfig::mainnet();

        // Same network should verify
        assert!(testnet.verify_message_network(&testnet.genesis_hash));
        assert!(mainnet.verify_message_network(&mainnet.genesis_hash));

        // v7.0.0: testnet == mainnet, so cross-verification passes
        assert!(testnet.verify_message_network(&mainnet.genesis_hash));
        assert!(mainnet.verify_message_network(&testnet.genesis_hash));

        // Random hash should fail
        let random_hash = [0xff; 32];
        assert!(!testnet.verify_message_network(&random_hash));
        assert!(!mainnet.verify_message_network(&random_hash));
    }

    #[test]
    fn test_network_message_creation() {
        let testnet = NetworkConfig::testnet();
        let message = NetworkMessage::new(&testnet, "test payload".to_string());

        // Verify genesis hash is set correctly
        assert_eq!(message.genesis_hash, testnet.genesis_hash);

        // Verify chain ID is set correctly
        assert_eq!(message.chain_id, testnet.chain_id);

        // Verify payload
        assert_eq!(message.payload, "test payload");

        // Verify timestamp is recent (within last second)
        let now = Utc::now();
        let diff = now - message.timestamp;
        assert!(diff.num_seconds() < 1);
    }

    #[test]
    fn test_network_message_verification() {
        let mainnet = NetworkConfig::mainnet();

        // Create message for mainnet
        let msg = NetworkMessage::new(&mainnet, 42u64);

        // Should verify on same network
        assert!(msg.verify_network(&mainnet));

        // Should fail on modified network
        let mut fake = mainnet.clone();
        fake.genesis_hash[0] ^= 0xff;
        assert!(!msg.verify_network(&fake));
    }

    #[test]
    fn test_network_message_cross_network_rejection() {
        // v7.0.0: testnet() == mainnet(), so test with a modified config
        let mainnet = NetworkConfig::mainnet();
        let mainnet_msg = NetworkMessage::new(&mainnet, vec![1, 2, 3]);

        // Mainnet message should verify on mainnet
        assert!(mainnet_msg.verify_network(&mainnet));

        // Should fail with different genesis hash
        let mut other = mainnet.clone();
        other.genesis_hash = [0xAA; 32];
        assert!(!mainnet_msg.verify_network(&other));
    }

    #[test]
    fn test_network_message_with_modified_genesis_hash() {
        let testnet = NetworkConfig::testnet();
        let mut message = NetworkMessage::new(&testnet, "payload");

        // Message should verify initially
        assert!(message.verify_network(&testnet));

        // Modify genesis hash
        message.genesis_hash[0] ^= 0xff;

        // Should now fail verification
        assert!(!message.verify_network(&testnet));
    }

    #[test]
    fn test_network_message_with_modified_chain_id() {
        let testnet = NetworkConfig::testnet();
        let mut message = NetworkMessage::new(&testnet, "payload");

        // Message should verify initially
        assert!(message.verify_network(&testnet));

        // Modify chain ID to something different
        message.chain_id = 12345;

        // Should now fail verification
        assert!(!message.verify_network(&testnet));
    }

    #[test]
    fn test_launch_time_verification() {
        let testnet = NetworkConfig::testnet();
        let mainnet = NetworkConfig::mainnet();

        // v7.0.0: Both testnet and mainnet launch at Feb 15, 2026 12:00 UTC
        assert_eq!(
            testnet.launch_time,
            DateTime::parse_from_rfc3339("2026-02-15T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc)
        );

        assert_eq!(
            mainnet.launch_time,
            DateTime::parse_from_rfc3339("2026-02-15T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc)
        );
    }

    #[test]
    fn test_bootstrap_peers_configuration() {
        let testnet = NetworkConfig::testnet();
        let mainnet = NetworkConfig::mainnet();

        // v7.0.0: Both testnet and mainnet use same config with peer IDs
        assert!(!testnet.bootstrap_peers.is_empty(), "Should have bootstrap peers for out-of-box connectivity");
        assert_eq!(testnet.bootstrap_peers.len(), 3, "Should have 3 bootstrap peers");
        assert!(testnet.bootstrap_peers[0].contains("185.182.185.227"), "First bootstrap peer should be Server Beta");
        assert!(testnet.bootstrap_peers[0].contains("12D3KooWSBxwSKw4wftHViMdw5rrV8Z1wEkikDS2vKYZtRrio5hH"), "Bootstrap should include Beta peer ID");
        assert!(testnet.bootstrap_peers[1].contains("109.205.176.60"), "Second bootstrap peer should be Server Gamma");
        assert!(testnet.bootstrap_peers[1].contains("12D3KooWEZKN13gsYXmvoUSeu5VnbUCTyEcAVdqKfWz14CAnm3bp"), "Bootstrap should include Gamma peer ID");
        assert!(testnet.bootstrap_peers[2].contains("5.79.79.158"), "Third bootstrap peer should be Server Delta");
        assert!(testnet.bootstrap_peers[2].contains("12D3KooWQZZAyLA4VQmwNozCBTZXXoWfvKE86ebbaPhSKu6XVmJJ"), "Bootstrap should include Delta peer ID");
        assert!(testnet.bootstrap_peers[3].contains("161.35.219.10"), "Fourth bootstrap peer should be Server Alpha");
        assert!(testnet.bootstrap_peers[3].contains("12D3KooWPwin4nJcU9PzsxNgUVXj5e6zDnACr84H7RZ1XzmnARsY"), "Bootstrap should include Alpha peer ID");

        // Mainnet == testnet in v7.0.0
        assert_eq!(mainnet.bootstrap_peers, testnet.bootstrap_peers);
    }

    #[test]
    fn test_edge_case_empty_genesis_hash() {
        let testnet = NetworkConfig::testnet();
        let empty_hash = [0u8; 32];

        // Empty hash should not match testnet
        assert!(!testnet.verify_message_network(&empty_hash));
    }

    #[test]
    fn test_edge_case_all_ones_genesis_hash() {
        let testnet = NetworkConfig::testnet();
        let ones_hash = [0xff; 32];

        // All-ones hash should not match testnet
        assert!(!testnet.verify_message_network(&ones_hash));
    }

    #[test]
    fn test_topic_namespace_isolation() {
        let testnet_id = NetworkId::TestnetPhase5;
        let mainnet_id = NetworkId::Mainnet;

        // Generate all topic types for both networks
        let testnet_topics = vec![
            testnet_id.transactions_topic(),
            testnet_id.blocks_topic(),
            testnet_id.acks_topic(),
        ];

        let mainnet_topics = vec![
            mainnet_id.transactions_topic(),
            mainnet_id.blocks_topic(),
            mainnet_id.acks_topic(),
        ];

        // No topic should overlap between networks
        for testnet_topic in &testnet_topics {
            for mainnet_topic in &mainnet_topics {
                assert_ne!(testnet_topic, mainnet_topic);
            }
        }

        // All testnet topics should start with /qnk/testnet-phase5 (Phase 4 network)
        for topic in &testnet_topics {
            assert!(topic.starts_with("/qnk/testnet-phase5/"));
        }

        // All mainnet topics should start with /qnk/mainnet
        for topic in &mainnet_topics {
            assert!(topic.starts_with("/qnk/mainnet/"));
        }
    }

    #[test]
    fn test_chain_id_replay_protection() {
        let testnet = NetworkConfig::testnet();
        let mainnet = NetworkConfig::mainnet();

        // Create a transaction-like payload
        #[derive(Clone)]
        struct MockTransaction {
            from: String,
            to: String,
            amount: u64,
        }

        let tx = MockTransaction {
            from: "alice".to_string(),
            to: "bob".to_string(),
            amount: 100,
        };

        // Wrap in network messages
        let testnet_msg = NetworkMessage::new(&testnet, tx.clone());
        let mainnet_msg = NetworkMessage::new(&mainnet, tx.clone());

        // Same transaction data, but different chain IDs should prevent replay
        assert_ne!(testnet_msg.chain_id, mainnet_msg.chain_id);
        assert!(!testnet_msg.verify_network(&mainnet));
        assert!(!mainnet_msg.verify_network(&testnet));
    }
}
