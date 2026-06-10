//! `MultisigProposal` — a pending action awaiting M-of-N signatures.

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use uuid::Uuid;

use crate::wallet::Address;

pub type ProposalId = Uuid;

/// What the proposal is actually asking signers to authorize. Concrete types
/// land as the surface grows; keep this small for v0.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MultisigAction {
    /// Move QUG (or a custom token) from the multisig wallet to a recipient.
    Transfer {
        token: String, // "QUG" or qnk-prefixed token address
        recipient: Address,
        amount_raw: u128, // 24-decimal raw
        memo: Option<String>,
    },
    /// Mint a new joint-control token. The first action the user described:
    /// "we create a token together."
    MintToken {
        symbol: String,
        name: String,
        decimals: u8,
        initial_supply_raw: u128,
        initial_holders: Vec<(Address, u128)>,
    },
    /// Update the wallet's default threshold without rotating members.
    SetDefaultThreshold { new_threshold: u8 },
    /// Add or remove a member (requires unanimous current-member sign-off by
    /// convention, enforced at proposal-time by setting `required_override =
    /// Some(members.len())`).
    RotateMember {
        remove_addr: Option<Address>,
        add_member_label: Option<String>,
        add_member_ed25519: Option<[u8; 32]>,
        add_member_dilithium5: Option<Vec<u8>>,
    },
}

impl MultisigAction {
    /// Compute the canonical payload hash that members sign.
    /// SHA3-256(postcard(action) || wallet_addr || proposal_id_bytes ||
    /// required_threshold || created_at_unix.to_be_bytes()).
    pub fn payload_hash(
        &self,
        wallet_addr: &Address,
        proposal_id: &ProposalId,
        required_threshold: u8,
        created_at_unix: i64,
    ) -> [u8; 32] {
        let mut h = Sha3_256::new();
        let action_bytes = postcard::to_allocvec(self).expect("postcard never fails for known type");
        h.update(&action_bytes);
        h.update(wallet_addr);
        h.update(proposal_id.as_bytes());
        h.update([required_threshold]);
        h.update(created_at_unix.to_be_bytes());
        h.finalize().into()
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProposalStatus {
    /// Waiting for more signatures.
    Pending,
    /// Threshold reached, ready to execute.
    Threshold,
    /// Submitted on-chain.
    Executed,
    /// Cancelled by quorum or expired.
    Cancelled,
}

/// One member's signature contribution on a proposal. Hybrid: BOTH halves
/// must verify against the member's pubkey for the contribution to count.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignatureContribution {
    /// Which member is signing (their per-member address).
    pub member_addr: Address,
    /// Ed25519 signature over the payload hash (64 bytes).
    #[serde(with = "ed25519_sig_bytes")]
    pub ed25519_sig: ed25519_dalek::Signature,
    /// Dilithium5 signed-message envelope (4627 bytes — includes embedded
    /// message). We use the SignedMessage form rather than a detached
    /// signature so we can pass through pqcrypto's verify API directly.
    pub dilithium5_signed_msg: Vec<u8>,
    /// Unix-seconds when this contribution was collected.
    pub at_unix: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultisigProposal {
    pub id: ProposalId,
    pub wallet_addr: Address,
    pub action: MultisigAction,
    /// Per-proposal threshold override. `None` = use wallet's default.
    /// This is THE knob the user described: "dynamically chosen — only send
    /// if everyone agrees" → `Some(members.len())`; "either of us can spend"
    /// → `Some(1)`.
    pub required_override: Option<u8>,
    /// Resolved at construction time so signers see the same value the
    /// verifier will use.
    pub required: u8,
    pub created_at_unix: i64,
    /// Sorted by `member_addr` so the same physical proposal hashes to the
    /// same bytes regardless of arrival order of partial sigs.
    pub signatures: Vec<SignatureContribution>,
    pub status: ProposalStatus,
}

impl MultisigProposal {
    pub fn new(
        wallet_addr: Address,
        action: MultisigAction,
        required_override: Option<u8>,
        wallet_default_threshold: u8,
        wallet_member_count: usize,
    ) -> Self {
        let raw = required_override.unwrap_or(wallet_default_threshold);
        let required = raw.clamp(1, wallet_member_count.min(255) as u8);
        Self {
            id: Uuid::new_v4(),
            wallet_addr,
            action,
            required_override,
            required,
            created_at_unix: chrono::Utc::now().timestamp(),
            signatures: Vec::new(),
            status: ProposalStatus::Pending,
        }
    }

    pub fn payload_hash(&self) -> [u8; 32] {
        self.action.payload_hash(
            &self.wallet_addr,
            &self.id,
            self.required,
            self.created_at_unix,
        )
    }

    /// Add a contribution (verifier checks happen in `verify::verify_proposal`).
    /// Idempotent: a member signing twice replaces their earlier contribution.
    pub fn add_signature(&mut self, contrib: SignatureContribution) {
        self.signatures.retain(|s| s.member_addr != contrib.member_addr);
        self.signatures.push(contrib);
        self.signatures.sort_by_key(|s| s.member_addr);
    }

    /// True if `signatures.len() >= required` (does NOT verify them — that's
    /// the verifier's job).
    pub fn has_threshold_count(&self) -> bool {
        self.signatures.len() >= self.required as usize
    }
}

/// serde helper for ed25519 Signature.
mod ed25519_sig_bytes {
    use ed25519_dalek::Signature;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(sig: &Signature, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(&sig.to_bytes())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Signature, D::Error> {
        let bytes: Vec<u8> = Vec::<u8>::deserialize(d)?;
        if bytes.len() != 64 {
            return Err(serde::de::Error::custom("ed25519 sig must be 64 bytes"));
        }
        let mut arr = [0u8; 64];
        arr.copy_from_slice(&bytes);
        Ok(Signature::from_bytes(&arr))
    }
}
