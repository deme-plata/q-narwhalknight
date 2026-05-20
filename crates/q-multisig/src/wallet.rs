//! `MultisigWallet` — the on-chain record binding member keys to a threshold.

use ed25519_dalek::VerifyingKey;
use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::PublicKey as _;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use thiserror::Error;

/// `Address` mirrors `q_types::Address` — kept local to avoid a workspace
/// dep cycle on `q-types`. Compile-time assert these stay in sync via the
/// `address_compat` test below.
pub type Address = [u8; 32];

#[derive(Debug, Error)]
pub enum WalletError {
    #[error("threshold {threshold} exceeds member count {members}")]
    ThresholdTooHigh { threshold: u8, members: usize },

    #[error("threshold must be at least 1")]
    ThresholdZero,

    #[error("multisig requires >= 2 members; got {0}")]
    NotEnoughMembers(usize),

    #[error("multisig accepts at most 255 members; got {0}")]
    TooManyMembers(usize),

    #[error("duplicate member address in wallet")]
    DuplicateMember,

    #[error("invalid Dilithium5 public key length: {0} (expected {})", dilithium5::public_key_bytes())]
    InvalidDilithiumKeyLength(usize),
}

/// A member's hybrid public-key bundle. Both halves are required — verifier
/// checks Ed25519 AND Dilithium5 signatures pass.
#[derive(Clone, Serialize, Deserialize)]
pub struct HybridPublicKey {
    /// Ed25519 verifying key (32 bytes, classical, fast).
    #[serde(with = "ed25519_bytes")]
    pub ed25519: VerifyingKey,
    /// Dilithium5 public key (2592 bytes, FIPS 204 / NIST Level 5 PQ).
    /// Serialized as Vec to side-step `serde-big-array` const-generic issues
    /// with the 2592-byte length.
    pub dilithium5: Vec<u8>,
}

impl HybridPublicKey {
    /// Derive an address from the hybrid pubkey: `SHA3-256(ed25519 || dilithium5)`.
    /// This is the on-chain identity of a single signing member.
    pub fn member_address(&self) -> Address {
        let mut h = Sha3_256::new();
        h.update(self.ed25519.as_bytes());
        h.update(&self.dilithium5);
        h.finalize().into()
    }

    pub fn dilithium5_pubkey(&self) -> Result<dilithium5::PublicKey, WalletError> {
        dilithium5::PublicKey::from_bytes(&self.dilithium5)
            .map_err(|_| WalletError::InvalidDilithiumKeyLength(self.dilithium5.len()))
    }
}

impl std::fmt::Debug for HybridPublicKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridPublicKey")
            .field("ed25519", &hex::encode(self.ed25519.as_bytes()))
            .field("dilithium5_addr_prefix", &hex::encode(&self.dilithium5[..8]))
            .finish()
    }
}

impl PartialEq for HybridPublicKey {
    fn eq(&self, other: &Self) -> bool {
        self.ed25519.as_bytes() == other.ed25519.as_bytes() && self.dilithium5 == other.dilithium5
    }
}
impl Eq for HybridPublicKey {}

/// A single signing member of a multisig wallet.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Member {
    /// Human-readable label ("user", "claude", "treasurer", etc).
    pub label: String,
    /// The member's hybrid pubkey bundle.
    pub pubkey: HybridPublicKey,
}

/// A multisig wallet — the persisted record of a member set + default
/// threshold. The on-chain `address` is derived from the canonical
/// concatenation of sorted member addresses + threshold, so two wallets with
/// the same members in different declaration order share an address.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MultisigWallet {
    pub address: Address,
    /// Default M used when a proposal doesn't specify `required` override.
    /// Per-proposal overrides allow the "any one of us can spend OR both
    /// must agree" model the user asked for.
    pub default_threshold: u8,
    /// All co-signers, sorted by `member_address()` so address derivation is
    /// canonical.
    pub members: Vec<Member>,
    /// Unix-seconds when this wallet was registered.
    pub created_at_unix: i64,
    /// Optional human label ("us-claude-treasury", "qshare-council").
    pub label: String,
}

impl MultisigWallet {
    /// Build a new wallet. Sorts members canonically + verifies invariants.
    pub fn new(
        mut members: Vec<Member>,
        default_threshold: u8,
        label: impl Into<String>,
    ) -> Result<Self, WalletError> {
        if members.len() < 2 {
            return Err(WalletError::NotEnoughMembers(members.len()));
        }
        if members.len() > 255 {
            return Err(WalletError::TooManyMembers(members.len()));
        }
        if default_threshold == 0 {
            return Err(WalletError::ThresholdZero);
        }
        if (default_threshold as usize) > members.len() {
            return Err(WalletError::ThresholdTooHigh {
                threshold: default_threshold,
                members: members.len(),
            });
        }

        // Canonical sort by member address.
        members.sort_by_key(|m| m.pubkey.member_address());

        // Check for duplicates (post-sort: adjacent compare).
        for w in members.windows(2) {
            if w[0].pubkey.member_address() == w[1].pubkey.member_address() {
                return Err(WalletError::DuplicateMember);
            }
        }

        // Address = SHA3-256(threshold || member_addr_0 || ... || member_addr_{N-1}).
        let mut h = Sha3_256::new();
        h.update([default_threshold]);
        for m in &members {
            h.update(m.pubkey.member_address());
        }
        let address: Address = h.finalize().into();

        Ok(Self {
            address,
            default_threshold,
            members,
            created_at_unix: chrono::Utc::now().timestamp(),
            label: label.into(),
        })
    }

    /// "qnk" + hex of the 32-byte address — matches the project's wallet
    /// address convention.
    pub fn address_string(&self) -> String {
        format!("qnk{}", hex::encode(self.address))
    }

    /// Find a member by their derived per-member address.
    pub fn find_member(&self, member_addr: &Address) -> Option<&Member> {
        self.members
            .iter()
            .find(|m| &m.pubkey.member_address() == member_addr)
    }

    /// Resolve effective threshold for a proposal: use the override if
    /// supplied, otherwise the wallet's default. Clamped to `[1, members.len()]`.
    pub fn effective_threshold(&self, override_required: Option<u8>) -> u8 {
        let raw = override_required.unwrap_or(self.default_threshold);
        raw.clamp(1, self.members.len() as u8)
    }
}

/// serde helper to round-trip `VerifyingKey` as a length-32 byte array.
mod ed25519_bytes {
    use ed25519_dalek::VerifyingKey;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(key: &VerifyingKey, s: S) -> Result<S::Ok, S::Error> {
        key.as_bytes().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<VerifyingKey, D::Error> {
        let bytes: [u8; 32] = Deserialize::deserialize(d)?;
        VerifyingKey::from_bytes(&bytes).map_err(serde::de::Error::custom)
    }
}

