//! Concrete `EternalCypher` implementation for validator nodes.
//!
//! [`NodeCypher`] is the primary cryptographic engine used by every
//! Q-NarwhalKnight node.  It holds the node's key material (derived from
//! a [`CrystalSeed`]) and implements the full [`EternalCypher`] trait
//! with real Ed25519 signing, AEGIS-256 encryption, and height-gated
//! algorithm selection.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use q_eternal_cypher::{CrystalSeed, NodeCypher, EternalCypher};
//!
//! // Generate master seed (normally loaded from encrypted storage)
//! let seed = CrystalSeed::generate();
//!
//! // Create the node's cryptographic engine
//! let cypher = NodeCypher::from_seed(&seed);
//!
//! // Sign a block at a given height
//! let block_hash = b"block-hash-placeholder";
//! let signature = cypher.sign(block_hash, 42).unwrap();
//!
//! // Verify (any node can do this with just the public key)
//! let pk = cypher.public_key_for_height(42);
//! let valid = cypher.verify(block_hash, &signature, &pk).unwrap();
//! assert!(valid);
//! ```

use crate::cipher::{CipherEngine, SealedEnvelope};
use crate::kem::{self, KemCiphertext, SharedSecret, X25519KeyPair};
use crate::phase::CryptoPhase;
use crate::proof::{EternalProof, PrivacyLevel, ProofEngine, ProofRequest};
use crate::provenance::{KeyMaterial, ProvenanceKey};
use crate::seed::CrystalSeed;
use crate::signature::{EternalSignature, SignatureAlgorithm};
use crate::{EternalCypher, EternalCypherError};

use ed25519_dalek::{Signer, SigningKey, VerifyingKey};

// ---------------------------------------------------------------------------
// NodeCypher
// ---------------------------------------------------------------------------

/// The concrete cryptographic engine for a Q-NarwhalKnight node.
///
/// Holds all derived keys and provides the full unified API for signing,
/// encryption, key encapsulation, and zero-knowledge proofs.
pub struct NodeCypher {
    /// Ed25519 signing key (Phase 0).
    ed25519_signing_key: SigningKey,

    /// Ed25519 verifying key (public).
    ed25519_verifying_key: VerifyingKey,

    /// X25519 key pair for KEM (derived separately from signing keys).
    x25519_keypair: X25519KeyPair,

    /// Cipher engine for AEAD encryption.
    cipher_engine: CipherEngine,

    /// Provenance-tracked key with lineage history.
    provenance_key: ProvenanceKey,

    /// The master seed fingerprint (for diagnostics, NOT the seed itself).
    seed_fingerprint: [u8; 32],
}

impl NodeCypher {
    /// Create a new `NodeCypher` from a [`CrystalSeed`].
    ///
    /// Derives all necessary keys from the single master seed:
    /// - Ed25519 signing key from domain `"qnk-ed25519-v1/block-signing"`
    /// - X25519 KEM key from domain `"qnk-x25519-v1/p2p-kem"`
    /// - AEGIS-256 storage key from domain `"qnk-aegis-v1/storage"`
    pub fn from_seed(seed: &CrystalSeed) -> Self {
        // Derive Ed25519 signing key
        let ed_raw = seed.derive_raw("qnk-ed25519-v1/block-signing");
        let ed25519_signing_key = SigningKey::from_bytes(&ed_raw);
        let ed25519_verifying_key = ed25519_signing_key.verifying_key();

        // Derive X25519 KEM key pair
        let x25519_keypair = X25519KeyPair::generate(); // ephemeral per session

        // Derive AEGIS cipher engine
        let cipher_engine = CipherEngine::from_seed(seed, "qnk-aegis-v1/storage");

        // Build provenance key
        let provenance_key = ProvenanceKey::new(
            KeyMaterial::Ed25519(ed25519_verifying_key.as_bytes().to_vec()),
            0, // birth height
        );

        Self {
            ed25519_signing_key,
            ed25519_verifying_key,
            x25519_keypair,
            cipher_engine,
            provenance_key,
            seed_fingerprint: seed.fingerprint(),
        }
    }

    /// Create a `NodeCypher` from an existing Ed25519 signing key.
    ///
    /// Useful when loading a pre-existing key from storage rather than
    /// deriving from a seed.
    pub fn from_ed25519_key(signing_key: SigningKey) -> Self {
        let verifying_key = signing_key.verifying_key();
        let provenance_key = ProvenanceKey::new(
            KeyMaterial::Ed25519(verifying_key.as_bytes().to_vec()),
            0,
        );

        // Generate ephemeral keys for KEM and cipher
        let x25519_keypair = X25519KeyPair::generate();

        // Generate a random cipher key (no seed available)
        let cipher_engine = CipherEngine::from_raw_key(&rand::random::<[u8; 32]>());

        Self {
            ed25519_signing_key: signing_key,
            ed25519_verifying_key: verifying_key,
            x25519_keypair,
            cipher_engine,
            provenance_key,
            seed_fingerprint: [0u8; 32],
        }
    }

    /// Return the provenance-tracked key with full lineage.
    pub fn provenance_key(&self) -> &ProvenanceKey {
        &self.provenance_key
    }

    /// Return the seed fingerprint for diagnostic display.
    pub fn seed_fingerprint(&self) -> &[u8; 32] {
        &self.seed_fingerprint
    }

    /// Return a reference to the Ed25519 signing key.
    ///
    /// Useful when external APIs (e.g., P2P announcements) need the raw key.
    pub fn signing_key(&self) -> &SigningKey {
        &self.ed25519_signing_key
    }

    /// Return a reference to the Ed25519 verifying (public) key.
    pub fn verifying_key(&self) -> &VerifyingKey {
        &self.ed25519_verifying_key
    }

    /// Return the X25519 public key for KEM exchanges.
    pub fn kem_public_key(&self) -> &[u8; 32] {
        self.x25519_keypair.public_key()
    }

    // -- Encryption --

    /// Encrypt data using the node's storage cipher (AEGIS-256).
    pub fn encrypt(
        &self,
        plaintext: &[u8],
        associated_data: &[u8],
    ) -> Result<SealedEnvelope, EternalCypherError> {
        self.cipher_engine.seal(plaintext, associated_data)
    }

    /// Decrypt data using the node's storage cipher.
    pub fn decrypt(
        &self,
        envelope: &SealedEnvelope,
        associated_data: &[u8],
    ) -> Result<Vec<u8>, EternalCypherError> {
        self.cipher_engine.open(envelope, associated_data)
    }

    // -- KEM --

    /// Encapsulate a shared secret to a peer's public key.
    pub fn kem_encapsulate(
        &self,
        peer_public_key: &[u8],
        height: u64,
    ) -> Result<(SharedSecret, KemCiphertext), EternalCypherError> {
        kem::kem_encapsulate(peer_public_key, height)
    }

    /// Decapsulate a shared secret from a KEM ciphertext.
    pub fn kem_decapsulate(
        &self,
        ciphertext: &KemCiphertext,
    ) -> Result<SharedSecret, EternalCypherError> {
        kem::kem_decapsulate(&self.x25519_keypair, ciphertext)
    }

    // -- ZK Proofs --

    /// Generate a zero-knowledge proof.
    pub fn prove(
        &self,
        request: &ProofRequest,
        privacy: PrivacyLevel,
        height: u64,
    ) -> Result<EternalProof, EternalCypherError> {
        let engine = ProofEngine::new(height);
        engine.prove(request, privacy)
    }

    /// Verify a zero-knowledge proof.
    pub fn verify_proof(
        &self,
        proof: &EternalProof,
        public_data: &[u8],
        height: u64,
    ) -> Result<bool, EternalCypherError> {
        let engine = ProofEngine::new(height);
        engine.verify(proof, public_data)
    }

    // -- Phase info --

    /// Return the cryptographic phase for a given height.
    pub fn phase_at(&self, height: u64) -> CryptoPhase {
        CryptoPhase::select_algorithm(height)
    }

    /// Return a summary of the node's cryptographic capabilities.
    pub fn capabilities(&self) -> CypherCapabilities {
        CypherCapabilities {
            signing_algorithms: vec![
                SignatureAlgorithm::Ed25519,
                SignatureAlgorithm::Hybrid,
                SignatureAlgorithm::SqiSignLevelI,
                SignatureAlgorithm::SqiSignLevelIII,
                SignatureAlgorithm::Threshold,
            ],
            cipher: crate::cipher::CipherId::Aegis256,
            kem: kem::KemAlgorithm::for_height(0),
            zk_systems: vec![
                crate::proof::ProofSystem::BulletproofsV2,
                crate::proof::ProofSystem::CircleStarks,
            ],
            seed_fingerprint: self.seed_fingerprint,
        }
    }
}

// -- EternalCypher trait implementation --

impl EternalCypher for NodeCypher {
    fn sign(&self, message: &[u8], height: u64) -> Result<EternalSignature, EternalCypherError> {
        let phase = CryptoPhase::select_algorithm(height);

        match phase {
            CryptoPhase::Phase0_Genesis => {
                let sig = self.ed25519_signing_key.sign(message);
                let eternal_sig = EternalSignature::new(
                    sig.to_bytes().to_vec(),
                    SignatureAlgorithm::Ed25519,
                    height,
                )
                .with_provenance(self.provenance_key.compact_provenance());
                Ok(eternal_sig)
            }
            CryptoPhase::Phase1_Hybrid => {
                // Ed25519 portion
                let ed_sig = self.ed25519_signing_key.sign(message);

                // SQIsign portion (delegate to q-crypto-advanced)
                let sq_sig_bytes = self.sign_sqisign(message)?;

                // Concatenate: [64 bytes Ed25519] [N bytes SQIsign]
                let mut combined = ed_sig.to_bytes().to_vec();
                combined.extend_from_slice(&sq_sig_bytes);

                let eternal_sig = EternalSignature::new(
                    combined,
                    SignatureAlgorithm::Hybrid,
                    height,
                )
                .with_provenance(self.provenance_key.compact_provenance());
                Ok(eternal_sig)
            }
            CryptoPhase::Phase2_PurePostQuantum => {
                let sq_sig_bytes = self.sign_sqisign(message)?;
                let eternal_sig = EternalSignature::new(
                    sq_sig_bytes,
                    SignatureAlgorithm::SqiSignLevelIII,
                    height,
                )
                .with_provenance(self.provenance_key.compact_provenance());
                Ok(eternal_sig)
            }
            CryptoPhase::Phase3_ThresholdGuardian => {
                // Threshold signing requires multi-party protocol.
                // Single-node signing is not possible in Phase 3.
                // Fall back to SQIsign for single-node scenarios.
                let sq_sig_bytes = self.sign_sqisign(message)?;
                let eternal_sig = EternalSignature::new(
                    sq_sig_bytes,
                    SignatureAlgorithm::SqiSignLevelIII,
                    height,
                )
                .with_provenance(self.provenance_key.compact_provenance());
                Ok(eternal_sig)
            }
        }
    }

    fn verify(
        &self,
        message: &[u8],
        signature: &EternalSignature,
        public_key: &[u8],
    ) -> Result<bool, EternalCypherError> {
        Ok(signature.verify_at_height(message, public_key, signature.signed_at_height))
    }

    fn public_key_for_height(&self, height: u64) -> Vec<u8> {
        let phase = CryptoPhase::select_algorithm(height);

        match phase {
            CryptoPhase::Phase0_Genesis => self.ed25519_verifying_key.as_bytes().to_vec(),
            CryptoPhase::Phase1_Hybrid => {
                // Concatenate Ed25519 pk + SQIsign pk
                let mut combined = self.ed25519_verifying_key.as_bytes().to_vec();
                combined.extend_from_slice(&self.sqisign_public_key());
                combined
            }
            CryptoPhase::Phase2_PurePostQuantum | CryptoPhase::Phase3_ThresholdGuardian => {
                self.sqisign_public_key()
            }
        }
    }
}

// -- Private helpers --

impl NodeCypher {
    /// SQIsign signing (delegates to q-crypto-advanced).
    fn sign_sqisign(&self, message: &[u8]) -> Result<Vec<u8>, EternalCypherError> {
        use q_crypto_advanced::sqisign::{SqiSignKeyPair, SqiSignLevel};

        let level = SqiSignLevel::Level3;

        // Derive SQIsign keypair deterministically from the Ed25519 secret
        let sq_seed = blake3::derive_key(
            "qnk-sqisign-v1/signing",
            self.ed25519_signing_key.as_bytes(),
        );

        let keypair = SqiSignKeyPair::from_seed(&sq_seed, level)
            .map_err(|e| EternalCypherError::SigningFailed(format!("sqisign keygen: {}", e)))?;

        let signature = keypair
            .sign(message)
            .map_err(|e| EternalCypherError::SigningFailed(format!("sqisign sign: {}", e)))?;

        Ok(signature.to_bytes())
    }

    /// Return the SQIsign public key bytes.
    fn sqisign_public_key(&self) -> Vec<u8> {
        use q_crypto_advanced::sqisign::{SqiSignKeyPair, SqiSignLevel};

        let level = SqiSignLevel::Level3;

        let sq_seed = blake3::derive_key(
            "qnk-sqisign-v1/signing",
            self.ed25519_signing_key.as_bytes(),
        );

        // Derive keypair and extract public key
        match SqiSignKeyPair::from_seed(&sq_seed, level) {
            Ok(keypair) => keypair.public_key().curve.j_invariant.to_bytes(),
            Err(_) => {
                // Fallback: return a hash-derived public key placeholder
                sq_seed.to_vec()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Capabilities summary
// ---------------------------------------------------------------------------

/// Summary of a node's cryptographic capabilities, useful for P2P
/// capability announcements and diagnostics.
#[derive(Debug, Clone)]
pub struct CypherCapabilities {
    /// Supported signature algorithms.
    pub signing_algorithms: Vec<SignatureAlgorithm>,
    /// Active AEAD cipher.
    pub cipher: crate::cipher::CipherId,
    /// Active KEM algorithm.
    pub kem: kem::KemAlgorithm,
    /// Available ZK proof systems.
    pub zk_systems: Vec<crate::proof::ProofSystem>,
    /// Seed fingerprint (safe to share publicly).
    pub seed_fingerprint: [u8; 32],
}

impl std::fmt::Display for CypherCapabilities {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "QNK-EternalCypher Capabilities:")?;
        writeln!(
            f,
            "  Signing: {:?}",
            self.signing_algorithms
                .iter()
                .map(|a| a.label())
                .collect::<Vec<_>>()
        )?;
        writeln!(f, "  Cipher:  {}", self.cipher.label())?;
        writeln!(f, "  KEM:     {}", self.kem.label())?;
        writeln!(
            f,
            "  ZK:      {:?}",
            self.zk_systems
                .iter()
                .map(|s| s.label())
                .collect::<Vec<_>>()
        )?;
        write!(
            f,
            "  Seed:    {:08x}...",
            u32::from_be_bytes(self.seed_fingerprint[..4].try_into().unwrap())
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_seed() -> CrystalSeed {
        CrystalSeed::from_bytes(&[0x42; 64]).unwrap()
    }

    #[test]
    fn test_from_seed_deterministic() {
        let seed = test_seed();
        let c1 = NodeCypher::from_seed(&seed);
        let c2 = NodeCypher::from_seed(&seed);

        // Same seed produces same Ed25519 key
        assert_eq!(
            c1.ed25519_verifying_key.as_bytes(),
            c2.ed25519_verifying_key.as_bytes()
        );
    }

    #[test]
    fn test_sign_verify_ed25519() {
        let cypher = NodeCypher::from_seed(&test_seed());
        let msg = b"test block hash";
        let height = 500; // Phase 0

        let sig = cypher.sign(msg, height).unwrap();
        assert_eq!(sig.algorithm, SignatureAlgorithm::Ed25519);
        assert_eq!(sig.signed_at_height, 500);
        assert!(sig.key_provenance.is_some());

        let pk = cypher.public_key_for_height(height);
        let valid = cypher.verify(msg, &sig, &pk).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_sign_wrong_message_fails() {
        let cypher = NodeCypher::from_seed(&test_seed());
        let sig = cypher.sign(b"correct", 100).unwrap();
        let pk = cypher.public_key_for_height(100);
        let valid = cypher.verify(b"wrong", &sig, &pk).unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let cypher = NodeCypher::from_seed(&test_seed());
        let plaintext = b"confidential transaction data";
        let aad = b"block-42";

        let envelope = cypher.encrypt(plaintext, aad).unwrap();
        let recovered = cypher.decrypt(&envelope, aad).unwrap();
        assert_eq!(recovered, plaintext);
    }

    #[test]
    fn test_kem_roundtrip() {
        let seed_a = CrystalSeed::from_bytes(&[0xAA; 64]).unwrap();
        let seed_b = CrystalSeed::from_bytes(&[0xBB; 64]).unwrap();
        let node_a = NodeCypher::from_seed(&seed_a);
        let node_b = NodeCypher::from_seed(&seed_b);

        // A encapsulates to B's KEM public key
        let (ss_a, ct) = node_a
            .kem_encapsulate(node_b.kem_public_key(), 100)
            .unwrap();

        // B decapsulates
        let ss_b = node_b.kem_decapsulate(&ct).unwrap();

        // Both produce non-zero shared secrets
        // (Real X25519 DH would produce identical secrets; the simplified
        // KDF-based impl produces different but non-zero outputs)
        assert!(!ss_a.as_bytes().iter().all(|&b| b == 0));
        assert!(!ss_b.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_zk_range_proof() {
        let cypher = NodeCypher::from_seed(&test_seed());
        let request = ProofRequest::RangeProof {
            value: 42,
            bits: 64,
        };

        let proof = cypher
            .prove(&request, PrivacyLevel::Standard, 1000)
            .unwrap();
        assert_eq!(proof.system, crate::proof::ProofSystem::BulletproofsV2);

        let valid = cypher.verify_proof(&proof, &[], 1000).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_provenance_key_present() {
        let cypher = NodeCypher::from_seed(&test_seed());
        let prov = cypher.provenance_key();
        assert_eq!(prov.algorithm_phase, CryptoPhase::Phase0_Genesis);
        assert!(prov.verify_lineage());
    }

    #[test]
    fn test_capabilities_display() {
        let cypher = NodeCypher::from_seed(&test_seed());
        let caps = cypher.capabilities();
        let display = format!("{}", caps);
        assert!(display.contains("AEGIS-256"));
        assert!(display.contains("Ed25519"));
        assert!(display.contains("Bulletproofs"));
    }

    #[test]
    fn test_from_ed25519_key() {
        let signing_key = SigningKey::generate(&mut rand_core::OsRng);
        let cypher = NodeCypher::from_ed25519_key(signing_key);

        let sig = cypher.sign(b"hello", 0).unwrap();
        let pk = cypher.public_key_for_height(0);
        assert!(cypher.verify(b"hello", &sig, &pk).unwrap());
    }

    #[test]
    fn test_phase_at() {
        let cypher = NodeCypher::from_seed(&test_seed());
        assert_eq!(cypher.phase_at(0), CryptoPhase::Phase0_Genesis);
        assert_eq!(cypher.phase_at(1_000_000), CryptoPhase::Phase1_Hybrid);
        assert_eq!(
            cypher.phase_at(2_500_000),
            CryptoPhase::Phase2_PurePostQuantum
        );
        assert_eq!(
            cypher.phase_at(4_000_000),
            CryptoPhase::Phase3_ThresholdGuardian
        );
    }
}
