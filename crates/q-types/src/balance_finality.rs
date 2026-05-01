//! Balance Finality Types — Bracha Reliable Broadcast over DAG-Knight
//!
//! Implements BFT-safe balance finalization for all non-block balance updates.
//!
//! Key design decisions (DeepSeek review 2026-05-01):
//! - ECHO carries the full P2PBalanceUpdate (not just broadcast_id) for validation
//! - Validator witnesses stored as u128 bitmask, not Vec<NodeId>
//! - Round window [start_round, end_round] to tolerate bounded round-clock drift
//! - All phases carry the full update to enable cross-phase validation

use serde::{Deserialize, Serialize};
use crate::balance_update::P2PBalanceUpdate;

/// Serde helper for [u8; 64] (Ed25519 signatures).
/// Standard serde does not derive Serialize/Deserialize for arrays > 32 bytes.
mod bytes64_serde {
    use serde::{Deserializer, Serializer, de::Error};

    pub fn serialize<S: Serializer>(v: &[u8; 64], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(v)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 64], D::Error> {
        let bytes: Vec<u8> = serde::Deserialize::deserialize(d)?;
        if bytes.len() != 64 {
            return Err(D::Error::custom(format!("expected 64 bytes, got {}", bytes.len())));
        }
        let mut arr = [0u8; 64];
        arr.copy_from_slice(&bytes);
        Ok(arr)
    }
}

/// Maximum validators this bitmask supports (u128 = 128 bits)
pub const MAX_VALIDATORS: usize = 128;

/// Round window size — nodes accept Bracha messages within this many rounds
/// At 100ms gossipsub heartbeat: 20 rounds ≈ 2 seconds of drift tolerance
pub const BRACHA_ROUND_WINDOW: u64 = 20;

/// How many Bracha rounds before a stalled proposal times out and is dropped.
/// 50 rounds × 100ms heartbeat ≈ 5 seconds.
pub const BRACHA_PROPOSAL_TIMEOUT_ROUNDS: u64 = 50;

/// Maximum records to batch into one DAG anchor vertex
pub const MAX_ANCHOR_BATCH: usize = 1000;

/// Maximum wall-clock seconds before pending_anchor is flushed to an anchor-only vertex
pub const ANCHOR_FLUSH_SECS: u64 = 5;

/// Phase in the Bracha three-phase reliable broadcast protocol.
///
/// ECHO carries the full update (not just broadcast_id) so receivers can
/// independently validate that the echoed value matches the original SEND.
/// This prevents a Byzantine SEND from being echoed with a tampered amount.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrachaPhase {
    /// Phase 1: Originator broadcasts the update to all peers
    Send,
    /// Phase 2: Receiver echoes the FULL update (validated against SEND) to all peers.
    /// Quorum: 2f+1 identical echoes trigger a READY.
    Echo,
    /// Phase 3: Ready — either triggered by 2f+1 echoes or amplified from f+1 ready msgs.
    /// Quorum: 2f+1 ready msgs → DELIVER (write to RocksDB).
    Ready,
}

/// Compact bitmask representing which validators in the known set have voted.
/// Uses u128 for up to 128 validators. For 4-node bootstrap: bits 0-3.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ValidatorBitmask(pub u128);

impl ValidatorBitmask {
    pub fn set(&mut self, index: u8) {
        if (index as usize) < MAX_VALIDATORS {
            self.0 |= 1u128 << index;
        }
    }

    pub fn has(&self, index: u8) -> bool {
        if (index as usize) < MAX_VALIDATORS {
            (self.0 >> index) & 1 == 1
        } else {
            false
        }
    }

    pub fn count(&self) -> u32 {
        self.0.count_ones()
    }

    pub fn union(&self, other: ValidatorBitmask) -> ValidatorBitmask {
        ValidatorBitmask(self.0 | other.0)
    }
}

/// A Bracha-wrapped balance update message.
///
/// Travels over `/qnk/{network}/consensus/balance-rb` gossipsub topic.
/// ALL phases carry the full `P2PBalanceUpdate` so every receiver can verify
/// that the echoed/ready value exactly matches the original SEND.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrachaBalanceMsg {
    /// Round window start — this proposal is valid when current_round is in
    /// [start_round, start_round + BRACHA_ROUND_WINDOW]
    pub start_round: u64,

    /// Unique ID for this Bracha broadcast instance.
    /// Computed as: BLAKE3(wallet_addr || amount_le || dag_round || nonce)
    pub broadcast_id: [u8; 32],

    /// The balance update being agreed upon.
    /// Carried in ALL phases (Send, Echo, Ready) for cross-phase validation.
    pub update: P2PBalanceUpdate,

    /// Bracha protocol phase
    pub phase: BrachaPhase,

    /// Sender's index in the current validator set (for bitmask tracking)
    pub sender_index: u8,

    /// Sender's Ed25519 public key (32 bytes)
    pub sender_pubkey: [u8; 32],

    /// Ed25519 signature over: broadcast_id || phase_byte || sender_index || update.signing_payload()
    #[serde(with = "bytes64_serde")]
    pub signature: [u8; 64],
}

impl BrachaBalanceMsg {
    /// Compute the message that is signed.
    /// Covers: broadcast_id + phase discriminant + sender_index + update payload
    pub fn signing_message(&self) -> [u8; 32] {
        use sha3::{Digest, Sha3_256};
        let mut h = Sha3_256::new();
        h.update(self.broadcast_id);
        h.update([self.phase_byte()]);
        h.update([self.sender_index]);
        h.update(self.update.signing_payload());
        h.finalize().into()
    }

    pub fn phase_byte(&self) -> u8 {
        match self.phase {
            BrachaPhase::Send  => 0,
            BrachaPhase::Echo  => 1,
            BrachaPhase::Ready => 2,
        }
    }

    /// Verify that this message's signature is valid for the declared sender_pubkey.
    pub fn verify_signature(&self) -> bool {
        use ed25519_dalek::{VerifyingKey, Signature, Verifier};
        let Ok(vk) = VerifyingKey::from_bytes(&self.sender_pubkey) else { return false };
        let Ok(sig) = Signature::from_slice(&self.signature) else { return false };
        let msg = self.signing_message();
        vk.verify(&msg, &sig).is_ok()
    }

    /// Whether this message's `update` content exactly matches `other`'s update.
    /// Used when processing ECHO/READY to reject tampered values.
    pub fn update_matches(&self, other: &BrachaBalanceMsg) -> bool {
        self.broadcast_id == other.broadcast_id
            && self.update.wallet_address == other.update.wallet_address
            && self.update.new_balance == other.update.new_balance
            && self.update.amount == other.update.amount
            && self.update.nonce == other.update.nonce
    }

    /// Whether this message is within the valid round window given current_round.
    pub fn in_round_window(&self, current_round: u64) -> bool {
        current_round >= self.start_round
            && current_round <= self.start_round.saturating_add(BRACHA_ROUND_WINDOW)
    }

    /// Serialize to CBOR bytes for gossipsub transport
    pub fn to_cbor(&self) -> Result<Vec<u8>, serde_cbor::Error> {
        serde_cbor::to_vec(self)
    }

    pub fn from_cbor(data: &[u8]) -> Result<Self, serde_cbor::Error> {
        serde_cbor::from_slice(data)
    }
}

/// Tracks per-broadcast Bracha state on a receiving node.
#[derive(Debug)]
pub struct BrachaInstance {
    /// The original SEND message (used to validate ECHOs)
    pub send_msg: Option<BrachaBalanceMsg>,

    /// Bitmask of validators that have sent an ECHO for this broadcast_id
    pub echo_mask: ValidatorBitmask,

    /// Bitmask of validators that have sent a READY for this broadcast_id
    pub ready_mask: ValidatorBitmask,

    /// Whether we have sent our own ECHO for this broadcast_id
    pub echoed: bool,

    /// Whether we have sent our own READY for this broadcast_id
    pub ready_sent: bool,

    /// Whether this instance has been delivered (written to DB)
    pub delivered: bool,

    /// DAG round when this instance was created (for timeout tracking)
    pub created_round: u64,
}

impl BrachaInstance {
    pub fn new(created_round: u64) -> Self {
        Self {
            send_msg: None,
            echo_mask: ValidatorBitmask::default(),
            ready_mask: ValidatorBitmask::default(),
            echoed: false,
            ready_sent: false,
            delivered: false,
            created_round,
        }
    }

    pub fn is_timed_out(&self, current_round: u64) -> bool {
        current_round > self.created_round.saturating_add(BRACHA_PROPOSAL_TIMEOUT_ROUNDS)
    }
}

/// A finalized balance record — written to RocksDB after Bracha delivery (2f+1 READY).
/// Stored in the `manifest` CF under key `balance_finality_proof:{hex_wallet}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceFinalityRecord {
    /// 32-byte wallet address
    pub wallet_address: [u8; 32],

    /// The agreed-upon new balance (u128)
    pub new_balance: u128,

    /// DAG-Knight round in which delivery occurred
    pub dag_round: u64,

    /// Hash of the DAG vertex that anchored this record (filled after anchoring)
    pub dag_vertex_hash: Option<[u8; 32]>,

    /// The Bracha broadcast_id
    pub broadcast_id: [u8; 32],

    /// Block height when finalized
    pub finalized_at_height: u64,

    /// Compact bitmask of validators that sent READY (audit trail, 16 bytes max)
    pub ready_witness_mask: ValidatorBitmask,

    /// Unix timestamp (seconds)
    pub finalized_ts: u64,
}

impl BalanceFinalityRecord {
    pub fn db_key(wallet: &[u8; 32]) -> String {
        format!("balance_finality_proof:{}", hex::encode(wallet))
    }

    pub fn to_cbor(&self) -> Result<Vec<u8>, serde_cbor::Error> {
        serde_cbor::to_vec(self)
    }

    pub fn from_cbor(data: &[u8]) -> Result<Self, serde_cbor::Error> {
        serde_cbor::from_slice(data)
    }
}

/// Response for `/api/v1/sync/dag-balance-anchor`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagBalanceAnchorResponse {
    /// Finalized records that have been anchored into DAG vertices
    pub anchored: Vec<BalanceFinalityRecord>,

    /// Delivered but not yet anchored into a vertex (pending_anchor buffer).
    /// IMPORTANT: fresh nodes must apply these too — they are already finalized
    /// (2f+1 READY received) but the DAG vertex hasn't been produced yet.
    pub pending_anchor: Vec<BalanceFinalityRecord>,

    /// Latest DAG round on this node
    pub latest_dag_round: u64,

    /// Block height at time of response
    pub block_height: u64,

    /// Number of active validators known to this node
    pub validator_count: usize,
}
