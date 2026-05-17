//! Height-aware eternal signatures.
//!
//! An [`EternalSignature`] bundles raw signature bytes with metadata that
//! tells verifiers *which* algorithm produced the signature, *when* it was
//! created (block height), and optionally *who* signed it (compact key
//! provenance).
//!
//! This allows a single verification entry point to dispatch to the correct
//! algorithm automatically, which is essential for replaying historical
//! blocks that may have been signed under a different cryptographic phase
//! than the one currently active.

use crate::phase::CryptoPhase;
use crate::provenance::KeyProvenance;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// SignatureAlgorithm
// ---------------------------------------------------------------------------

/// The specific algorithm used to produce a signature.
///
/// This is more granular than [`CryptoPhase`]: a single phase may accept
/// multiple algorithms (e.g., Phase 1 accepts both Ed25519 and SQIsign),
/// and this enum identifies exactly which one was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignatureAlgorithm {
    /// Classical Ed25519 (RFC 8032).
    Ed25519,

    /// Hybrid: Ed25519 signature concatenated with SQIsign signature.
    /// Both must verify independently.
    Hybrid,

    /// SQIsign Level I (NIST Level I, 128-bit post-quantum security).
    SqiSignLevelI,

    /// SQIsign Level III (NIST Level III, 192-bit post-quantum security).
    SqiSignLevelIII,

    /// FROST threshold signature (t-of-n Schnorr over Ed25519 curve).
    Threshold,
}

impl SignatureAlgorithm {
    /// Return a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            SignatureAlgorithm::Ed25519 => "Ed25519",
            SignatureAlgorithm::Hybrid => "Hybrid (Ed25519 + SQIsign)",
            SignatureAlgorithm::SqiSignLevelI => "SQIsign Level I",
            SignatureAlgorithm::SqiSignLevelIII => "SQIsign Level III",
            SignatureAlgorithm::Threshold => "FROST Threshold",
        }
    }

    /// Check whether this algorithm is valid in the given phase.
    pub fn is_valid_in_phase(&self, phase: CryptoPhase) -> bool {
        match self {
            SignatureAlgorithm::Ed25519 => phase.accepts_ed25519(),
            SignatureAlgorithm::Hybrid => phase.requires_dual_signature() || phase.accepts_ed25519(),
            SignatureAlgorithm::SqiSignLevelI | SignatureAlgorithm::SqiSignLevelIII => {
                phase.accepts_sqisign()
            }
            SignatureAlgorithm::Threshold => phase.accepts_threshold(),
        }
    }
}

impl std::fmt::Display for SignatureAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// EternalSignature
// ---------------------------------------------------------------------------

/// A self-describing signature that carries enough metadata for any verifier
/// to validate it without external context beyond the public key.
///
/// ## Fields
///
/// - `data`: The raw signature bytes in the format defined by `algorithm`.
/// - `algorithm`: Which algorithm produced this signature.
/// - `signed_at_height`: The block height at which the signature was created.
///   Verifiers use this to determine the active [`CryptoPhase`] and confirm
///   that `algorithm` was valid at that height.
/// - `key_provenance`: Optional compact provenance of the signing key.
/// - `migration_proof`: Optional proof that the signing key was legitimately
///   migrated from an earlier algorithm (used during phase transitions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EternalSignature {
    /// Raw signature bytes.
    pub data: Vec<u8>,

    /// The algorithm that produced this signature.
    pub algorithm: SignatureAlgorithm,

    /// Block height at which this signature was created.
    pub signed_at_height: u64,

    /// Compact provenance of the signing key (optional).
    #[serde(default)]
    pub key_provenance: Option<KeyProvenance>,

    /// Migration proof bytes (optional).
    ///
    /// When a key is upgraded from one algorithm to another, the holder
    /// may include a proof that they controlled the old key.  The format
    /// of this proof depends on the specific migration path.
    #[serde(default)]
    pub migration_proof: Option<Vec<u8>>,
}

impl EternalSignature {
    /// Construct a new signature with the given parameters.
    pub fn new(
        data: Vec<u8>,
        algorithm: SignatureAlgorithm,
        signed_at_height: u64,
    ) -> Self {
        Self {
            data,
            algorithm,
            signed_at_height,
            key_provenance: None,
            migration_proof: None,
        }
    }

    /// Attach key provenance metadata to this signature.
    pub fn with_provenance(mut self, provenance: KeyProvenance) -> Self {
        self.key_provenance = Some(provenance);
        self
    }

    /// Attach a migration proof to this signature.
    pub fn with_migration_proof(mut self, proof: Vec<u8>) -> Self {
        self.migration_proof = Some(proof);
        self
    }

    /// Verify this signature against a message and public key, using the
    /// algorithm recorded in the signature and the rules active at the
    /// block height where it was created.
    ///
    /// This method dispatches to the appropriate verification backend based
    /// on `self.algorithm`.  It first checks that the algorithm was valid
    /// in the phase active at `self.signed_at_height`.
    ///
    /// # Arguments
    ///
    /// * `msg` - The message that was signed.
    /// * `pubkey` - The public key bytes to verify against.
    /// * `height` - The block height to use for phase selection.  Typically
    ///   this should equal `self.signed_at_height`, but callers may pass a
    ///   different height for cross-checks.
    ///
    /// # Returns
    ///
    /// `true` if the signature is valid, `false` otherwise.  Returns `false`
    /// (rather than an error) if the algorithm is not valid in the active
    /// phase, since that represents a policy violation rather than a
    /// cryptographic failure.
    pub fn verify_at_height(&self, msg: &[u8], pubkey: &[u8], height: u64) -> bool {
        let phase = CryptoPhase::select_algorithm(height);

        // Policy check: is this algorithm valid at the given height?
        if !self.algorithm.is_valid_in_phase(phase) {
            return false;
        }

        match self.algorithm {
            SignatureAlgorithm::Ed25519 => self.verify_ed25519(msg, pubkey),
            SignatureAlgorithm::Hybrid => self.verify_hybrid(msg, pubkey),
            SignatureAlgorithm::SqiSignLevelI | SignatureAlgorithm::SqiSignLevelIII => {
                self.verify_sqisign(msg, pubkey)
            }
            SignatureAlgorithm::Threshold => self.verify_threshold(msg, pubkey),
        }
    }

    /// Ed25519 verification using `ed25519-dalek`.
    fn verify_ed25519(&self, msg: &[u8], pubkey: &[u8]) -> bool {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};

        let pk_array: [u8; 32] = match pubkey.try_into() {
            Ok(a) => a,
            Err(_) => return false,
        };

        let Ok(vk) = VerifyingKey::from_bytes(&pk_array) else {
            return false;
        };

        let Ok(sig) = Signature::from_slice(&self.data) else {
            return false;
        };

        vk.verify(msg, &sig).is_ok()
    }

    /// Hybrid verification: first 64 bytes are Ed25519 sig, remainder is
    /// SQIsign sig.  The pubkey is split as: first 32 bytes Ed25519,
    /// remainder SQIsign.  Both must verify.
    fn verify_hybrid(&self, msg: &[u8], pubkey: &[u8]) -> bool {
        if self.data.len() < 64 || pubkey.len() < 32 {
            return false;
        }

        // Ed25519 portion
        let ed_sig_bytes = &self.data[..64];
        let ed_pubkey = &pubkey[..32];

        let ed_ok = {
            use ed25519_dalek::{Signature, Verifier, VerifyingKey};
            let pk_array: [u8; 32] = match ed_pubkey.try_into() {
                Ok(a) => a,
                Err(_) => return false,
            };
            let Ok(vk) = VerifyingKey::from_bytes(&pk_array) else {
                return false;
            };
            let Ok(sig) = Signature::from_slice(ed_sig_bytes) else {
                return false;
            };
            vk.verify(msg, &sig).is_ok()
        };

        if !ed_ok {
            return false;
        }

        // SQIsign portion: delegate to q-crypto-advanced
        let pq_sig_bytes = &self.data[64..];
        let pq_pubkey = &pubkey[32..];
        self.verify_sqisign_inner(msg, pq_pubkey, pq_sig_bytes)
    }

    /// SQIsign verification via `q-crypto-advanced`.
    fn verify_sqisign(&self, msg: &[u8], pubkey: &[u8]) -> bool {
        self.verify_sqisign_inner(msg, pubkey, &self.data)
    }

    /// Inner SQIsign verification helper.
    fn verify_sqisign_inner(&self, msg: &[u8], pubkey: &[u8], sig_bytes: &[u8]) -> bool {
        use q_crypto_advanced::sqisign::{SqiSignLevel, SqiSignature, SqiSignVerifier};

        let level = match self.algorithm {
            SignatureAlgorithm::SqiSignLevelI => SqiSignLevel::Level1,
            SignatureAlgorithm::SqiSignLevelIII => SqiSignLevel::Level3,
            // For hybrid, default to Level I
            _ => SqiSignLevel::Level1,
        };

        let Ok(signature) = SqiSignature::from_bytes(sig_bytes) else {
            return false;
        };

        // Reconstruct a minimal public key for verification
        let element_size = match level {
            SqiSignLevel::Level1 => 32,
            SqiSignLevel::Level3 => 48,
            SqiSignLevel::Level5 => 64,
        };

        use q_crypto_advanced::sqisign::{Fp2Element, SupersingularCurve, SqiSignPublicKey};
        let j_invariant = if pubkey.len() >= element_size {
            Fp2Element::new(
                pubkey[..element_size / 2].to_vec(),
                pubkey[element_size / 2..element_size].to_vec(),
            )
        } else {
            return false;
        };

        let curve = SupersingularCurve::new(
            Fp2Element::zero(element_size / 2),
            Fp2Element::one(element_size / 2),
            j_invariant,
        );
        let pk = SqiSignPublicKey::new(curve, level);

        let verifier = SqiSignVerifier::new(level);
        verifier.verify(&pk, msg, &signature).unwrap_or(false)
    }

    /// FROST threshold signature verification.
    fn verify_threshold(&self, msg: &[u8], pubkey: &[u8]) -> bool {
        use q_crypto_advanced::frost::{FrostVerifier, ThresholdSignature};

        // Deserialize the group public key
        let Ok(group_pk) =
            q_crypto_advanced::frost::GroupPublicKey::deserialize(pubkey)
        else {
            return false;
        };

        let ts = ThresholdSignature {
            signature_bytes: self.data.clone(),
            signers: Vec::new(), // signers list not needed for verification
        };

        FrostVerifier::verify(&group_pk, msg, &ts).unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signature_algorithm_labels() {
        assert_eq!(SignatureAlgorithm::Ed25519.label(), "Ed25519");
        assert_eq!(SignatureAlgorithm::Hybrid.label(), "Hybrid (Ed25519 + SQIsign)");
        assert_eq!(SignatureAlgorithm::SqiSignLevelI.label(), "SQIsign Level I");
        assert_eq!(SignatureAlgorithm::SqiSignLevelIII.label(), "SQIsign Level III");
        assert_eq!(SignatureAlgorithm::Threshold.label(), "FROST Threshold");
    }

    #[test]
    fn test_algorithm_phase_validity() {
        // Ed25519 valid in Phase 0 and Phase 1, not Phase 2/3
        assert!(SignatureAlgorithm::Ed25519.is_valid_in_phase(CryptoPhase::Phase0_Genesis));
        assert!(SignatureAlgorithm::Ed25519.is_valid_in_phase(CryptoPhase::Phase1_Hybrid));
        assert!(!SignatureAlgorithm::Ed25519.is_valid_in_phase(CryptoPhase::Phase2_PurePostQuantum));
        assert!(!SignatureAlgorithm::Ed25519.is_valid_in_phase(CryptoPhase::Phase3_ThresholdGuardian));

        // SQIsign valid in Phase 1, 2, 3
        assert!(!SignatureAlgorithm::SqiSignLevelIII.is_valid_in_phase(CryptoPhase::Phase0_Genesis));
        assert!(SignatureAlgorithm::SqiSignLevelIII.is_valid_in_phase(CryptoPhase::Phase1_Hybrid));
        assert!(SignatureAlgorithm::SqiSignLevelIII.is_valid_in_phase(CryptoPhase::Phase2_PurePostQuantum));
        assert!(SignatureAlgorithm::SqiSignLevelIII.is_valid_in_phase(CryptoPhase::Phase3_ThresholdGuardian));

        // Threshold only in Phase 3
        assert!(!SignatureAlgorithm::Threshold.is_valid_in_phase(CryptoPhase::Phase0_Genesis));
        assert!(!SignatureAlgorithm::Threshold.is_valid_in_phase(CryptoPhase::Phase1_Hybrid));
        assert!(!SignatureAlgorithm::Threshold.is_valid_in_phase(CryptoPhase::Phase2_PurePostQuantum));
        assert!(SignatureAlgorithm::Threshold.is_valid_in_phase(CryptoPhase::Phase3_ThresholdGuardian));
    }

    #[test]
    fn test_new_signature() {
        let sig = EternalSignature::new(
            vec![0xAA; 64],
            SignatureAlgorithm::Ed25519,
            42,
        );
        assert_eq!(sig.data.len(), 64);
        assert_eq!(sig.algorithm, SignatureAlgorithm::Ed25519);
        assert_eq!(sig.signed_at_height, 42);
        assert!(sig.key_provenance.is_none());
        assert!(sig.migration_proof.is_none());
    }

    #[test]
    fn test_with_provenance() {
        let provenance = KeyProvenance {
            key_fingerprint: [0x11; 32],
            birth_height: 0,
            algorithm_phase: CryptoPhase::Phase0_Genesis,
            commitment_to_ancestor: [0u8; 32],
            transition_count: 0,
        };

        let sig = EternalSignature::new(vec![0xBB; 64], SignatureAlgorithm::Ed25519, 100)
            .with_provenance(provenance.clone());

        assert!(sig.key_provenance.is_some());
        assert_eq!(sig.key_provenance.unwrap().birth_height, 0);
    }

    #[test]
    fn test_with_migration_proof() {
        let sig = EternalSignature::new(vec![0xCC; 64], SignatureAlgorithm::Hybrid, 1_000_000)
            .with_migration_proof(vec![0xDD; 128]);

        assert!(sig.migration_proof.is_some());
        assert_eq!(sig.migration_proof.unwrap().len(), 128);
    }

    #[test]
    fn test_verify_ed25519_real_signature() {
        use ed25519_dalek::{Signer, SigningKey};

        let signing_key = SigningKey::generate(&mut rand_core::OsRng);
        let verifying_key = signing_key.verifying_key();

        let msg = b"eternal cypher test message";
        let dalek_sig = signing_key.sign(msg);

        let eternal_sig = EternalSignature::new(
            dalek_sig.to_bytes().to_vec(),
            SignatureAlgorithm::Ed25519,
            500, // Phase 0 height
        );

        // Should verify at a Phase 0 height
        assert!(eternal_sig.verify_at_height(msg, verifying_key.as_bytes(), 500));

        // Should fail at a Phase 2 height (Ed25519 not accepted)
        assert!(!eternal_sig.verify_at_height(msg, verifying_key.as_bytes(), 2_500_000));
    }

    #[test]
    fn test_verify_wrong_message() {
        use ed25519_dalek::{Signer, SigningKey};

        let signing_key = SigningKey::generate(&mut rand_core::OsRng);
        let verifying_key = signing_key.verifying_key();

        let msg = b"correct message";
        let dalek_sig = signing_key.sign(msg);

        let eternal_sig = EternalSignature::new(
            dalek_sig.to_bytes().to_vec(),
            SignatureAlgorithm::Ed25519,
            100,
        );

        assert!(!eternal_sig.verify_at_height(b"wrong message", verifying_key.as_bytes(), 100));
    }

    #[test]
    fn test_verify_bad_pubkey() {
        let sig = EternalSignature::new(vec![0xFF; 64], SignatureAlgorithm::Ed25519, 100);
        assert!(!sig.verify_at_height(b"test", &[0u8; 32], 100));
    }

    #[test]
    fn test_serde_roundtrip() {
        let provenance = KeyProvenance {
            key_fingerprint: [0x42; 32],
            birth_height: 10,
            algorithm_phase: CryptoPhase::Phase0_Genesis,
            commitment_to_ancestor: [0u8; 32],
            transition_count: 0,
        };

        let sig = EternalSignature::new(vec![0xAB; 64], SignatureAlgorithm::SqiSignLevelI, 1_500_000)
            .with_provenance(provenance)
            .with_migration_proof(vec![0xEF; 32]);

        let json = serde_json::to_string(&sig).unwrap();
        let recovered: EternalSignature = serde_json::from_str(&json).unwrap();

        assert_eq!(recovered.algorithm, sig.algorithm);
        assert_eq!(recovered.signed_at_height, sig.signed_at_height);
        assert_eq!(recovered.data, sig.data);
        assert!(recovered.key_provenance.is_some());
        assert!(recovered.migration_proof.is_some());
    }

    #[test]
    fn test_backward_compat_missing_optional_fields() {
        // Simulate a minimal JSON from an older version
        let json = r#"{
            "data": [1,2,3],
            "algorithm": "Ed25519",
            "signed_at_height": 0
        }"#;
        let sig: EternalSignature = serde_json::from_str(json).unwrap();
        assert!(sig.key_provenance.is_none());
        assert!(sig.migration_proof.is_none());
    }

    #[test]
    fn test_display_algorithm() {
        let alg = SignatureAlgorithm::SqiSignLevelIII;
        assert_eq!(format!("{}", alg), "SQIsign Level III");
    }
}
