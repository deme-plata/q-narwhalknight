//! # Q-Eternal-Cypher: Unified Cryptographic Framework
//!
//! This crate defines the **Eternal Cypher** abstraction layer for Q-NarwhalKnight,
//! providing a unified interface across all cryptographic eras of the network.
//!
//! ## Design Philosophy
//!
//! Blockchains must survive algorithm transitions without breaking historical
//! verification. The Eternal Cypher framework achieves this through:
//!
//! - **Provenance-tracked keys**: Every key records its full lineage of algorithm
//!   upgrades, splits, and merges via Blake3-committed ancestor chains.
//! - **Height-gated algorithm selection**: The active cryptographic phase is
//!   determined solely by block height, enabling deterministic replay of any
//!   historical block under the rules that were active when it was produced.
//! - **Eternal signatures**: Signatures carry metadata about the algorithm and
//!   height at which they were created, allowing verifiers to select the correct
//!   verification path automatically.
//! - **Crystal seeds**: A single 512-bit master seed from which all domain-specific
//!   keys are derived using Blake3 KDF, with optional quantum entropy mixing.
//! - **Unified cipher engine**: AEGIS-256 authenticated encryption with
//!   self-describing sealed envelopes for storage and P2P traffic.
//! - **Key encapsulation**: X25519 + future Kyber1024 hybrid KEM for P2P
//!   session key establishment.
//! - **Zero-knowledge proofs**: Unified prove/verify interface dispatching to
//!   Bulletproofs v2 (range proofs) or Circle STARKs (execution proofs).
//!
//! ## Module Overview
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`phase`] | `CryptoPhase` enum and height-gated algorithm selection |
//! | [`provenance`] | `ProvenanceKey` with full key lineage tracking |
//! | [`signature`] | `EternalSignature` with height-aware verification |
//! | [`seed`] | `CrystalSeed` for deterministic key derivation |
//! | [`cipher`] | `CipherEngine` — AEGIS-256 authenticated encryption |
//! | [`kem`] | Key encapsulation (X25519 + Kyber1024 hybrid) |
//! | [`proof`] | `ProofEngine` — Bulletproofs v2 + Circle STARKs |
//! | [`node_cypher`] | `NodeCypher` — concrete `EternalCypher` implementation |
//!
//! ## Quick Start
//!
//! ```rust
//! use q_eternal_cypher::{CrystalSeed, NodeCypher, CryptoPhase, EternalCypher};
//!
//! // Generate a master seed (normally loaded from encrypted storage)
//! let seed = CrystalSeed::generate();
//!
//! // Create the node's unified cryptographic engine
//! let cypher = NodeCypher::from_seed(&seed);
//!
//! // Sign a block (auto-selects Ed25519 for Phase 0)
//! let block_hash = b"block-hash";
//! let signature = cypher.sign(block_hash, 42).unwrap();
//!
//! // Encrypt data (AEGIS-256)
//! let envelope = cypher.encrypt(b"secret data", b"block-42").unwrap();
//! let recovered = cypher.decrypt(&envelope, b"block-42").unwrap();
//! assert_eq!(recovered, b"secret data");
//! ```

pub mod phase;
pub mod provenance;
pub mod seed;
pub mod signature;
pub mod cipher;
pub mod kem;
pub mod proof;
pub mod node_cypher;

// ---- Re-exports for convenience ----

pub use phase::CryptoPhase;
pub use provenance::{KeyMaterial, KeyProvenance, KeyTransition, KeyTransitionKind, ProvenanceKey};
pub use seed::CrystalSeed;
pub use signature::{EternalSignature, SignatureAlgorithm};
pub use cipher::{CipherEngine, CipherId, SealedEnvelope};
pub use kem::{KemAlgorithm, KemCiphertext, SharedSecret, X25519KeyPair};
pub use proof::{EternalProof, PrivacyLevel, ProofEngine, ProofRequest, ProofSystem};
pub use node_cypher::{NodeCypher, CypherCapabilities};

/// The `EternalCypher` trait defines the contract that any cryptographic backend
/// must satisfy to participate in Q-NarwhalKnight consensus.
///
/// Implementations are expected to delegate actual cryptographic operations to
/// the primitives in `q-crypto-advanced` (FROST, SQIsign, etc.) while this
/// trait provides the unified, height-aware interface.
///
/// The primary concrete implementation is [`NodeCypher`], which provides the
/// full unified API including encryption, KEM, and ZK proofs beyond what
/// this trait requires.
pub trait EternalCypher: Send + Sync {
    /// Sign a message using the algorithm appropriate for the given block height.
    ///
    /// The implementation must select the correct algorithm based on
    /// [`CryptoPhase::select_algorithm`] and attach provenance metadata to
    /// the resulting [`EternalSignature`].
    fn sign(&self, message: &[u8], height: u64) -> Result<EternalSignature, EternalCypherError>;

    /// Verify a signature, automatically selecting the verification path based
    /// on the algorithm recorded in the signature and the block height at which
    /// it was produced.
    ///
    /// Returns `true` if the signature is valid under the rules that were active
    /// at `signature.signed_at_height`.
    fn verify(
        &self,
        message: &[u8],
        signature: &EternalSignature,
        public_key: &[u8],
    ) -> Result<bool, EternalCypherError>;

    /// Return the public key bytes appropriate for the given height.
    ///
    /// During hybrid phases, this may return a concatenation of classical and
    /// post-quantum public keys.
    fn public_key_for_height(&self, height: u64) -> Vec<u8>;

    /// Return the current cryptographic phase for diagnostic or UI display.
    fn current_phase(&self, height: u64) -> CryptoPhase {
        CryptoPhase::select_algorithm(height)
    }
}

/// Errors produced by the Eternal Cypher framework.
#[derive(Debug, thiserror::Error)]
pub enum EternalCypherError {
    /// The requested algorithm is not available or not yet activated.
    #[error("algorithm not available at height {height}: {reason}")]
    AlgorithmNotAvailable { height: u64, reason: String },

    /// Signature verification failed.
    #[error("signature verification failed: {0}")]
    VerificationFailed(String),

    /// Signing operation failed.
    #[error("signing failed: {0}")]
    SigningFailed(String),

    /// Key derivation or generation failed.
    #[error("key error: {0}")]
    KeyError(String),

    /// Serialization or deserialization error.
    #[error("serialization error: {0}")]
    SerializationError(String),

    /// The seed material is invalid or corrupted.
    #[error("invalid seed: {0}")]
    InvalidSeed(String),

    /// An error propagated from `q-crypto-advanced`.
    #[error("underlying crypto error: {0}")]
    CryptoAdvanced(#[from] q_crypto_advanced::CryptoError),
}

/// Crate version, mirroring the workspace version for compatibility checks.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_is_set() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_error_display() {
        let err = EternalCypherError::AlgorithmNotAvailable {
            height: 42,
            reason: "SQIsign not yet active".into(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("42"));
        assert!(msg.contains("SQIsign"));
    }

    #[test]
    fn test_crypto_phase_reexport() {
        // Verify re-exports compile and are accessible
        let phase = CryptoPhase::select_algorithm(0);
        assert_eq!(phase, CryptoPhase::Phase0_Genesis);
    }

    // Integration test: full NodeCypher workflow
    #[test]
    fn test_full_workflow() {
        let seed = CrystalSeed::generate();
        let cypher = NodeCypher::from_seed(&seed);

        // 1. Sign and verify
        let msg = b"block payload";
        let sig = cypher.sign(msg, 100).unwrap();
        let pk = cypher.public_key_for_height(100);
        assert!(cypher.verify(msg, &sig, &pk).unwrap());

        // 2. Encrypt and decrypt
        let envelope = cypher.encrypt(b"secret", b"aad").unwrap();
        let plain = cypher.decrypt(&envelope, b"aad").unwrap();
        assert_eq!(plain, b"secret");

        // 3. Range proof
        let request = ProofRequest::RangeProof { value: 1000, bits: 64 };
        let proof = cypher.prove(&request, PrivacyLevel::Standard, 100).unwrap();
        assert!(cypher.verify_proof(&proof, &[], 100).unwrap());

        // 4. Capabilities
        let caps = cypher.capabilities();
        assert!(!caps.signing_algorithms.is_empty());
    }
}
