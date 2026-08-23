//! Producer endorsements — decentralized enrollment of new block producers.
//!
//! # The problem this solves
//!
//! PQC-004 requires every block producer's Dilithium5 public key to be known to
//! every verifying node, or the block is rejected outright
//! (`"producer Dilithium pubkey unknown — pin Q_PRODUCER_DILITHIUM_PUBKEY_HEX"`).
//! Until now the only way to supply those keys was a STATIC allowlist: the
//! `Q_PRODUCER_DILITHIUM_PUBKEY_HEX` env var, or a pin file baked into the binary.
//!
//! That makes adding a producer an O(number of nodes) manual operation — edit
//! config on every node in the network and restart each one — which is both
//! painful and, more importantly, centralized: a new node physically cannot
//! contribute block production until a human touches every other node.
//! Measured consequence: a fresh node that starts mining produces a private fork
//! (every one of its blocks rejected by peers) rather than contributing work.
//!
//! # Why we cannot simply auto-learn the key
//!
//! Ed25519 producer keys ARE already auto-learned from blocks ("KEY-LEARN"
//! trust-on-first-use). Doing the same for Dilithium5 would remove the ONLY hard
//! producer-authentication gate in the system: any node could then self-declare
//! as a producer and have its blocks — including their coinbase rewards —
//! accepted by the whole network. That is an unlimited-minting hole. Auto-accept
//! is therefore not an option.
//!
//! # The design: an endorsement chain rooted at the existing pinned producers
//!
//! A `ProducerEndorsement` is a statement, signed by an ALREADY-AUTHORIZED
//! producer, that some new node's producer keys are legitimate. Nodes accept a
//! previously-unknown producer key if and only if it arrives with an endorsement
//! that verifies against a producer they ALREADY trust.
//!
//! This gives:
//!   * **Automatic propagation** — endorsements travel with state-sync, so one
//!     endorsement reaches the whole network without editing any node's config.
//!   * **Sybil resistance** — a node cannot endorse itself into authority; an
//!     existing authorized producer must sign for it exactly once.
//!   * **Decentralized verification** — every node checks the signature itself
//!     against a key it already trusts. No node has to trust the messenger.
//!   * **No new trust root** — the chain is rooted in the same pinned producer
//!     key(s) that the network already relies on today.
//!
//! # Security invariants (do not weaken these)
//!
//! 1. **The endorser must itself be a FULLY authorized producer** — i.e. one with
//!    a known Dilithium5 key. It must NOT be sufficient to be present in the
//!    registry via the Ed25519 trust-on-first-use auto-learn path, because that
//!    path accepts arbitrary keys off the wire. Endorsing off a TOFU-learned key
//!    would let an attacker bootstrap themselves into producer authority.
//! 2. **Self-endorsement is rejected** — the endorser and the subject must differ.
//! 3. **The signature covers every field that matters**, under a domain separator,
//!    so an endorsement cannot be replayed to authorize a different key, a
//!    different node, or on a different network.
//! 4. Endorsements are additive only. There is deliberately no revocation encoded
//!    here; removing a producer remains an explicit operator action (pin edit),
//!    so a compromised gossip path can never *remove* a legitimate producer.

use serde::{Deserialize, Serialize};

/// Domain separator — prevents an endorsement signature from being valid in any
/// other signing context, and vice versa.
pub const PRODUCER_ENDORSEMENT_DOMAIN: &[u8] = b"quillon.producer-endorsement.v1";

/// A signed statement that a new node is authorized to produce blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProducerEndorsement {
    /// Node id (== Ed25519 public key) of the producer being authorized.
    pub subject_node_id: [u8; 32],
    /// Subject's Dilithium5 public key (2,592 bytes) — the thing PQC-004 needs.
    pub subject_dilithium5: Vec<u8>,
    /// Subject's SQIsign public key (64 bytes), optional/empty if unused.
    #[serde(default)]
    pub subject_sqisign: Vec<u8>,
    /// Node id (== Ed25519 public key) of the ALREADY-AUTHORIZED endorsing producer.
    pub endorser_node_id: [u8; 32],
    /// Network this endorsement is valid on (e.g. "mainnet-genesis"). Bound into
    /// the signature so an endorsement minted on a testnet cannot be replayed
    /// onto mainnet.
    pub network_id: String,
    /// Unix seconds when issued (informational + tie-breaking; not a validity window).
    pub issued_at: u64,
    /// Ed25519 signature by `endorser_node_id` over `signing_bytes()`.
    pub signature: Vec<u8>,
}

/// Why an endorsement was rejected. Kept explicit so callers can log precisely
/// instead of collapsing every failure into "invalid".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EndorsementError {
    /// Endorser is not a known, fully-authorized producer on this node.
    EndorserNotAuthorized,
    /// Endorser is known only via Ed25519 trust-on-first-use, which is not
    /// sufficient authority to endorse (see security invariant 1).
    EndorserNotPqcAuthorized,
    /// A node tried to endorse itself into authority.
    SelfEndorsement,
    /// Endorsement is for a different network.
    NetworkMismatch { expected: String, got: String },
    /// Subject's Dilithium5 key is not a plausible Dilithium5 public key.
    BadSubjectKeyLength { got: usize },
    /// Signature was malformed or did not verify.
    BadSignature,
}

impl std::fmt::Display for EndorsementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EndorserNotAuthorized => write!(f, "endorser is not a known authorized producer"),
            Self::EndorserNotPqcAuthorized => write!(
                f,
                "endorser is known only via Ed25519 auto-learn and lacks PQC authority to endorse"
            ),
            Self::SelfEndorsement => write!(f, "self-endorsement is not permitted"),
            Self::NetworkMismatch { expected, got } => {
                write!(f, "endorsement is for network '{}', we are '{}'", got, expected)
            }
            Self::BadSubjectKeyLength { got } => {
                write!(f, "subject Dilithium5 key has wrong length {} (expected {})", got, DILITHIUM5_PUBKEY_LEN)
            }
            Self::BadSignature => write!(f, "endorsement signature invalid"),
        }
    }
}

impl std::error::Error for EndorsementError {}

/// Dilithium5 public keys are exactly this long; anything else is malformed.
pub const DILITHIUM5_PUBKEY_LEN: usize = 2592;

impl ProducerEndorsement {
    /// Canonical bytes covered by the signature.
    ///
    /// Every security-relevant field is included, length-prefixed where variable,
    /// so that no two distinct endorsements can produce the same signing payload
    /// (which would otherwise allow swapping one subject key for another).
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            PRODUCER_ENDORSEMENT_DOMAIN.len() + 32 + 32 + self.subject_dilithium5.len() + 64,
        );
        out.extend_from_slice(PRODUCER_ENDORSEMENT_DOMAIN);
        out.extend_from_slice(&self.subject_node_id);
        out.extend_from_slice(&(self.subject_dilithium5.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.subject_dilithium5);
        out.extend_from_slice(&(self.subject_sqisign.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.subject_sqisign);
        out.extend_from_slice(&self.endorser_node_id);
        let net = self.network_id.as_bytes();
        out.extend_from_slice(&(net.len() as u32).to_le_bytes());
        out.extend_from_slice(net);
        out.extend_from_slice(&self.issued_at.to_le_bytes());
        out
    }

    /// Structural checks that need no registry access.
    pub fn check_well_formed(&self, our_network_id: &str) -> Result<(), EndorsementError> {
        if self.subject_node_id == self.endorser_node_id {
            return Err(EndorsementError::SelfEndorsement);
        }
        if self.subject_dilithium5.len() != DILITHIUM5_PUBKEY_LEN {
            return Err(EndorsementError::BadSubjectKeyLength {
                got: self.subject_dilithium5.len(),
            });
        }
        if self.network_id != our_network_id {
            return Err(EndorsementError::NetworkMismatch {
                expected: our_network_id.to_string(),
                got: self.network_id.clone(),
            });
        }
        Ok(())
    }

    /// Full verification against the local view of who is already authorized.
    ///
    /// `endorser_ed25519` must be the endorser's Ed25519 public key as this node
    /// already knows it, and `endorser_has_pqc_authority` must be true only when
    /// that endorser is authorized in the strong sense (a pinned/endorsed producer
    /// with a known Dilithium5 key) — NOT merely auto-learned from a block header.
    pub fn verify(
        &self,
        our_network_id: &str,
        endorser_ed25519: Option<&[u8]>,
        endorser_has_pqc_authority: bool,
    ) -> Result<(), EndorsementError> {
        self.check_well_formed(our_network_id)?;

        let endorser_key = endorser_ed25519.ok_or(EndorsementError::EndorserNotAuthorized)?;
        if !endorser_has_pqc_authority {
            return Err(EndorsementError::EndorserNotPqcAuthorized);
        }

        let key_bytes: [u8; 32] = endorser_key
            .try_into()
            .map_err(|_| EndorsementError::EndorserNotAuthorized)?;
        let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&key_bytes)
            .map_err(|_| EndorsementError::EndorserNotAuthorized)?;

        let sig_bytes: [u8; 64] = self
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| EndorsementError::BadSignature)?;
        let signature = ed25519_dalek::Signature::from_bytes(&sig_bytes);

        use ed25519_dalek::Verifier;
        verifying_key
            .verify(&self.signing_bytes(), &signature)
            .map_err(|_| EndorsementError::BadSignature)
    }

    /// Mint an endorsement. Run on a node that is ALREADY an authorized producer.
    pub fn sign_with(
        signing_key: &ed25519_dalek::SigningKey,
        subject_node_id: [u8; 32],
        subject_dilithium5: Vec<u8>,
        subject_sqisign: Vec<u8>,
        network_id: String,
        issued_at: u64,
    ) -> Self {
        use ed25519_dalek::Signer;
        let endorser_node_id = signing_key.verifying_key().to_bytes();
        let mut e = Self {
            subject_node_id,
            subject_dilithium5,
            subject_sqisign,
            endorser_node_id,
            network_id,
            issued_at,
            signature: Vec::new(),
        };
        e.signature = signing_key.sign(&e.signing_bytes()).to_bytes().to_vec();
        e
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn endorser_key() -> SigningKey {
        SigningKey::from_bytes(&[7u8; 32])
    }

    fn make(subject: [u8; 32], net: &str) -> ProducerEndorsement {
        ProducerEndorsement::sign_with(
            &endorser_key(),
            subject,
            vec![9u8; DILITHIUM5_PUBKEY_LEN],
            vec![3u8; 64],
            net.to_string(),
            1_787_000_000,
        )
    }

    #[test]
    fn valid_endorsement_verifies() {
        let e = make([1u8; 32], "mainnet-genesis");
        let pk = endorser_key().verifying_key().to_bytes();
        assert!(e.verify("mainnet-genesis", Some(&pk), true).is_ok());
    }

    #[test]
    fn tofu_only_endorser_is_rejected() {
        // Security invariant 1: an endorser known only via Ed25519 auto-learn
        // must NOT be able to confer producer authority.
        let e = make([1u8; 32], "mainnet-genesis");
        let pk = endorser_key().verifying_key().to_bytes();
        assert_eq!(
            e.verify("mainnet-genesis", Some(&pk), false).unwrap_err(),
            EndorsementError::EndorserNotPqcAuthorized
        );
    }

    #[test]
    fn unknown_endorser_is_rejected() {
        let e = make([1u8; 32], "mainnet-genesis");
        assert_eq!(
            e.verify("mainnet-genesis", None, true).unwrap_err(),
            EndorsementError::EndorserNotAuthorized
        );
    }

    #[test]
    fn self_endorsement_is_rejected() {
        let endorser = endorser_key().verifying_key().to_bytes();
        let e = make(endorser, "mainnet-genesis");
        let pk = endorser;
        assert_eq!(
            e.verify("mainnet-genesis", Some(&pk), true).unwrap_err(),
            EndorsementError::SelfEndorsement
        );
    }

    #[test]
    fn wrong_network_is_rejected() {
        let e = make([1u8; 32], "testnet-phase8");
        let pk = endorser_key().verifying_key().to_bytes();
        assert!(matches!(
            e.verify("mainnet-genesis", Some(&pk), true).unwrap_err(),
            EndorsementError::NetworkMismatch { .. }
        ));
    }

    #[test]
    fn tampering_with_subject_key_breaks_signature() {
        // The whole point: an attacker must not be able to lift a valid signature
        // onto a different subject key.
        let mut e = make([1u8; 32], "mainnet-genesis");
        e.subject_dilithium5 = vec![0xAAu8; DILITHIUM5_PUBKEY_LEN];
        let pk = endorser_key().verifying_key().to_bytes();
        assert_eq!(
            e.verify("mainnet-genesis", Some(&pk), true).unwrap_err(),
            EndorsementError::BadSignature
        );
    }

    #[test]
    fn tampering_with_subject_id_breaks_signature() {
        let mut e = make([1u8; 32], "mainnet-genesis");
        e.subject_node_id = [2u8; 32];
        let pk = endorser_key().verifying_key().to_bytes();
        assert_eq!(
            e.verify("mainnet-genesis", Some(&pk), true).unwrap_err(),
            EndorsementError::BadSignature
        );
    }

    #[test]
    fn malformed_subject_key_length_is_rejected() {
        let mut e = make([1u8; 32], "mainnet-genesis");
        e.subject_dilithium5 = vec![9u8; 100];
        let pk = endorser_key().verifying_key().to_bytes();
        assert!(matches!(
            e.verify("mainnet-genesis", Some(&pk), true).unwrap_err(),
            EndorsementError::BadSubjectKeyLength { .. }
        ));
    }

    #[test]
    fn signature_from_a_different_key_is_rejected() {
        let e = make([1u8; 32], "mainnet-genesis");
        let impostor = SigningKey::from_bytes(&[8u8; 32]).verifying_key().to_bytes();
        assert_eq!(
            e.verify("mainnet-genesis", Some(&impostor), true).unwrap_err(),
            EndorsementError::BadSignature
        );
    }

    #[test]
    fn roundtrips_through_json() {
        let e = make([1u8; 32], "mainnet-genesis");
        let s = serde_json::to_string(&e).unwrap();
        let back: ProducerEndorsement = serde_json::from_str(&s).unwrap();
        assert_eq!(e, back);
        let pk = endorser_key().verifying_key().to_bytes();
        assert!(back.verify("mainnet-genesis", Some(&pk), true).is_ok());
    }
}
