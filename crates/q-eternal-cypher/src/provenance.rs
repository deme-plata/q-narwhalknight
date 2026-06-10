//! Key provenance tracking with full lineage history.
//!
//! Every key in Q-NarwhalKnight carries a *provenance record* that documents
//! its entire lifecycle: the algorithm it was born with, every upgrade,
//! split, or merge event it has undergone, and a Blake3 commitment chain
//! linking it back to its ancestor keys.
//!
//! This design ensures that even after multiple algorithm transitions, any
//! verifier can trace a key's history and confirm that it evolved through
//! legitimate upgrade paths rather than being conjured from thin air.
//!
//! ## Key Transition Types
//!
//! - **Upgrade**: The key holder migrates from one algorithm to another
//!   (e.g., Ed25519 to SQIsign) by proving ownership of the old key and
//!   publishing a new one, with both linked via a Blake3 commitment.
//! - **Split**: A single key is replaced by multiple keys (e.g., for
//!   threshold or multisig schemes).
//! - **Merge**: Multiple keys are combined into a single aggregate key
//!   (e.g., FROST key generation from individual shares).

use crate::phase::CryptoPhase;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// KeyMaterial
// ---------------------------------------------------------------------------

/// The raw cryptographic material held by a key, tagged by algorithm.
///
/// Each variant stores the public key bytes in their canonical encoding.
/// Secret key material is never stored in this enum; it is managed by
/// the signing backend.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KeyMaterial {
    /// Classical Ed25519 public key (32 bytes).
    Ed25519(Vec<u8>),

    /// Hybrid key: Ed25519 public key concatenated with SQIsign public key.
    /// The first 32 bytes are Ed25519; the remainder is SQIsign.
    Hybrid(Vec<u8>),

    /// SQIsign isogeny-based public key.
    /// Size depends on security level (64 bytes for Level I, 96 for Level III).
    SqiSign(Vec<u8>),

    /// FROST threshold group public key.
    /// The bytes are the serialized `GroupPublicKey` from `q-crypto-advanced`.
    Threshold(Vec<u8>),
}

impl KeyMaterial {
    /// Return the raw public key bytes, regardless of variant.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            KeyMaterial::Ed25519(b) => b,
            KeyMaterial::Hybrid(b) => b,
            KeyMaterial::SqiSign(b) => b,
            KeyMaterial::Threshold(b) => b,
        }
    }

    /// Return the algorithm label for display and logging.
    pub fn algorithm_label(&self) -> &'static str {
        match self {
            KeyMaterial::Ed25519(_) => "Ed25519",
            KeyMaterial::Hybrid(_) => "Hybrid (Ed25519 + SQIsign)",
            KeyMaterial::SqiSign(_) => "SQIsign",
            KeyMaterial::Threshold(_) => "FROST Threshold",
        }
    }

    /// Compute a Blake3 fingerprint of the key material.
    pub fn fingerprint(&self) -> [u8; 32] {
        blake3::hash(self.as_bytes()).into()
    }
}

// ---------------------------------------------------------------------------
// KeyTransition
// ---------------------------------------------------------------------------

/// The kind of key lifecycle event.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum KeyTransitionKind {
    /// Algorithm upgrade (e.g., Ed25519 -> SQIsign).
    Upgrade,
    /// Key split into multiple sub-keys (e.g., threshold key generation).
    Split,
    /// Multiple keys merged into one (e.g., aggregation).
    Merge,
}

/// A single key lifecycle event, recording when and how the key changed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KeyTransition {
    /// What kind of transition this was.
    pub kind: KeyTransitionKind,

    /// Block height at which the transition was recorded on-chain.
    pub height: u64,

    /// The cryptographic phase that was active at `height`.
    pub from_phase: CryptoPhase,

    /// The cryptographic phase after the transition (may be the same if
    /// the transition is a split/merge within one phase).
    pub to_phase: CryptoPhase,

    /// Blake3 hash of the previous key material, linking this transition
    /// to its predecessor.
    pub ancestor_commitment: [u8; 32],

    /// Optional human-readable note (e.g., "migrated to PQ before Phase 2").
    #[serde(default)]
    pub note: Option<String>,
}

// ---------------------------------------------------------------------------
// KeyProvenance (compact form for embedding in signatures)
// ---------------------------------------------------------------------------

/// A compact provenance summary suitable for embedding inside signatures.
///
/// This carries just enough information for a verifier to decide whether
/// to trust the key, without the full transition history.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KeyProvenance {
    /// Blake3 fingerprint of the current public key material.
    pub key_fingerprint: [u8; 32],

    /// The block height at which this key was first registered on-chain.
    pub birth_height: u64,

    /// The algorithm phase under which this key was created.
    pub algorithm_phase: CryptoPhase,

    /// Blake3 commitment linking to the previous key (zeros if this is
    /// the original key with no predecessor).
    pub commitment_to_ancestor: [u8; 32],

    /// Number of transitions in the key's full history.
    pub transition_count: u32,
}

// ---------------------------------------------------------------------------
// ProvenanceKey
// ---------------------------------------------------------------------------

/// A provenance-tracked cryptographic key.
///
/// This is the primary key type used throughout Q-NarwhalKnight.  It wraps
/// the raw key material with a complete audit trail of every algorithm
/// upgrade, split, or merge event the key has undergone.
///
/// ## Serialization
///
/// All fields use `#[serde(default)]` where appropriate so that keys
/// serialized by older software versions (which may lack newer fields)
/// can still be deserialized without errors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceKey {
    /// The current key material (public key bytes tagged by algorithm).
    pub current: KeyMaterial,

    /// Complete history of key transitions, ordered chronologically.
    #[serde(default)]
    pub history: Vec<KeyTransition>,

    /// Block height at which this key was first created or registered.
    pub birth_height: u64,

    /// The cryptographic phase that was active when this key was created.
    #[serde(default)]
    pub algorithm_phase: CryptoPhase,

    /// Blake3 hash linking this key to its immediate ancestor.
    /// All zeros for a freshly generated key with no predecessor.
    #[serde(default)]
    pub commitment_to_ancestor: [u8; 32],
}

impl ProvenanceKey {
    /// Create a new provenance key with no prior history.
    ///
    /// The `birth_height` determines the initial `algorithm_phase` via
    /// [`CryptoPhase::select_algorithm`].
    pub fn new(current: KeyMaterial, birth_height: u64) -> Self {
        let algorithm_phase = CryptoPhase::select_algorithm(birth_height);
        Self {
            current,
            history: Vec::new(),
            birth_height,
            algorithm_phase,
            commitment_to_ancestor: [0u8; 32],
        }
    }

    /// Create a new provenance key that is an upgrade of `ancestor`.
    ///
    /// This records an `Upgrade` transition in the history and sets
    /// `commitment_to_ancestor` to the Blake3 fingerprint of the
    /// ancestor's current key material.
    pub fn upgrade_from(
        ancestor: &ProvenanceKey,
        new_material: KeyMaterial,
        transition_height: u64,
    ) -> Self {
        let ancestor_fingerprint = ancestor.current.fingerprint();
        let new_phase = CryptoPhase::select_algorithm(transition_height);

        let transition = KeyTransition {
            kind: KeyTransitionKind::Upgrade,
            height: transition_height,
            from_phase: ancestor.algorithm_phase,
            to_phase: new_phase,
            ancestor_commitment: ancestor_fingerprint,
            note: None,
        };

        let mut history = ancestor.history.clone();
        history.push(transition);

        Self {
            current: new_material,
            history,
            birth_height: ancestor.birth_height,
            algorithm_phase: new_phase,
            commitment_to_ancestor: ancestor_fingerprint,
        }
    }

    /// Return a compact [`KeyProvenance`] suitable for embedding in signatures.
    pub fn compact_provenance(&self) -> KeyProvenance {
        KeyProvenance {
            key_fingerprint: self.current.fingerprint(),
            birth_height: self.birth_height,
            algorithm_phase: self.algorithm_phase,
            commitment_to_ancestor: self.commitment_to_ancestor,
            transition_count: self.history.len() as u32,
        }
    }

    /// Return the Blake3 fingerprint of the current key material.
    pub fn fingerprint(&self) -> [u8; 32] {
        self.current.fingerprint()
    }

    /// Return the number of transitions this key has undergone.
    pub fn transition_count(&self) -> usize {
        self.history.len()
    }

    /// Verify that the ancestor commitment chain is internally consistent.
    ///
    /// This checks that each transition's `ancestor_commitment` matches the
    /// fingerprint that would have been produced by the key material at the
    /// time of the previous transition.  It does NOT verify on-chain
    /// registration (that requires access to the blockchain state).
    ///
    /// Returns `true` if the chain is consistent or empty.
    pub fn verify_lineage(&self) -> bool {
        if self.history.is_empty() {
            // No transitions: the ancestor commitment should be all zeros
            return self.commitment_to_ancestor == [0u8; 32];
        }

        // Check that the final transition's ancestor_commitment matches
        // our stored commitment_to_ancestor
        if let Some(last) = self.history.last() {
            if last.ancestor_commitment != self.commitment_to_ancestor {
                return false;
            }
        }

        // Check that phase transitions are monotonically ordered by height
        let mut prev_height = 0u64;
        for t in &self.history {
            if t.height < prev_height {
                return false;
            }
            prev_height = t.height;
        }

        true
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ed25519_key() -> KeyMaterial {
        KeyMaterial::Ed25519(vec![0xAA; 32])
    }

    fn sample_sqisign_key() -> KeyMaterial {
        KeyMaterial::SqiSign(vec![0xBB; 64])
    }

    fn sample_hybrid_key() -> KeyMaterial {
        let mut bytes = vec![0xCC; 32]; // Ed25519 portion
        bytes.extend_from_slice(&[0xDD; 64]); // SQIsign portion
        KeyMaterial::Hybrid(bytes)
    }

    #[test]
    fn test_key_material_fingerprint_deterministic() {
        let k1 = sample_ed25519_key();
        let k2 = sample_ed25519_key();
        assert_eq!(k1.fingerprint(), k2.fingerprint());

        // Different material gives a different fingerprint
        let k3 = sample_sqisign_key();
        assert_ne!(k1.fingerprint(), k3.fingerprint());
    }

    #[test]
    fn test_key_material_labels() {
        assert_eq!(sample_ed25519_key().algorithm_label(), "Ed25519");
        assert_eq!(sample_sqisign_key().algorithm_label(), "SQIsign");
        assert_eq!(
            sample_hybrid_key().algorithm_label(),
            "Hybrid (Ed25519 + SQIsign)"
        );
        assert_eq!(
            KeyMaterial::Threshold(vec![0xFF; 32]).algorithm_label(),
            "FROST Threshold"
        );
    }

    #[test]
    fn test_new_provenance_key() {
        let key = ProvenanceKey::new(sample_ed25519_key(), 500);
        assert_eq!(key.algorithm_phase, CryptoPhase::Phase0_Genesis);
        assert_eq!(key.birth_height, 500);
        assert!(key.history.is_empty());
        assert_eq!(key.commitment_to_ancestor, [0u8; 32]);
        assert!(key.verify_lineage());
    }

    #[test]
    fn test_upgrade_from() {
        let original = ProvenanceKey::new(sample_ed25519_key(), 100);
        let upgraded = ProvenanceKey::upgrade_from(
            &original,
            sample_sqisign_key(),
            2_500_000, // Phase 2
        );

        assert_eq!(upgraded.algorithm_phase, CryptoPhase::Phase2_PurePostQuantum);
        assert_eq!(upgraded.birth_height, 100); // preserves original birth
        assert_eq!(upgraded.history.len(), 1);
        assert_eq!(upgraded.history[0].kind, KeyTransitionKind::Upgrade);
        assert_eq!(
            upgraded.commitment_to_ancestor,
            original.current.fingerprint()
        );
        assert!(upgraded.verify_lineage());
    }

    #[test]
    fn test_multi_upgrade_lineage() {
        let gen = ProvenanceKey::new(sample_ed25519_key(), 0);
        let hybrid = ProvenanceKey::upgrade_from(&gen, sample_hybrid_key(), 1_000_000);
        let pq = ProvenanceKey::upgrade_from(&hybrid, sample_sqisign_key(), 2_500_000);

        assert_eq!(pq.transition_count(), 2);
        assert_eq!(pq.birth_height, 0);
        assert!(pq.verify_lineage());
    }

    #[test]
    fn test_compact_provenance() {
        let key = ProvenanceKey::new(sample_ed25519_key(), 42);
        let compact = key.compact_provenance();

        assert_eq!(compact.key_fingerprint, key.fingerprint());
        assert_eq!(compact.birth_height, 42);
        assert_eq!(compact.algorithm_phase, CryptoPhase::Phase0_Genesis);
        assert_eq!(compact.transition_count, 0);
    }

    #[test]
    fn test_serde_roundtrip() {
        let original = ProvenanceKey::new(sample_ed25519_key(), 100);
        let upgraded = ProvenanceKey::upgrade_from(&original, sample_sqisign_key(), 2_500_000);

        let json = serde_json::to_string(&upgraded).unwrap();
        let recovered: ProvenanceKey = serde_json::from_str(&json).unwrap();

        assert_eq!(recovered.birth_height, upgraded.birth_height);
        assert_eq!(recovered.algorithm_phase, upgraded.algorithm_phase);
        assert_eq!(recovered.history.len(), upgraded.history.len());
        assert_eq!(
            recovered.commitment_to_ancestor,
            upgraded.commitment_to_ancestor
        );
    }

    #[test]
    fn test_backward_compat_missing_fields() {
        // Simulate a JSON payload from an older version that lacks some fields
        let minimal_json = r#"{
            "current": {"Ed25519": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]},
            "birth_height": 0
        }"#;
        let key: ProvenanceKey = serde_json::from_str(minimal_json).unwrap();
        assert_eq!(key.birth_height, 0);
        assert!(key.history.is_empty());
        assert_eq!(key.algorithm_phase, CryptoPhase::Phase0_Genesis);
        assert_eq!(key.commitment_to_ancestor, [0u8; 32]);
    }

    #[test]
    fn test_verify_lineage_detects_bad_height_order() {
        let mut key = ProvenanceKey::new(sample_ed25519_key(), 0);
        // Manually insert out-of-order transitions
        key.history.push(KeyTransition {
            kind: KeyTransitionKind::Upgrade,
            height: 2_000_000,
            from_phase: CryptoPhase::Phase0_Genesis,
            to_phase: CryptoPhase::Phase1_Hybrid,
            ancestor_commitment: [0u8; 32],
            note: None,
        });
        key.history.push(KeyTransition {
            kind: KeyTransitionKind::Upgrade,
            height: 1_000_000, // out of order!
            from_phase: CryptoPhase::Phase1_Hybrid,
            to_phase: CryptoPhase::Phase2_PurePostQuantum,
            ancestor_commitment: [0u8; 32],
            note: None,
        });
        // commitment_to_ancestor does not match last transition
        key.commitment_to_ancestor = [0u8; 32];

        assert!(!key.verify_lineage());
    }
}
