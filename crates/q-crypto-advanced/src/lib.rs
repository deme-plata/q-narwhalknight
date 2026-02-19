//! Q-NarwhalKnight Advanced Cryptography Module
//!
//! This crate implements cutting-edge cryptographic primitives from recent IACR ePrint papers:
//!
//! ## FROST Threshold Signatures (IACR 2025/1024)
//! Memory-optimal two-round threshold Schnorr signatures for validator committees.
//! Enables t-of-n threshold signing without trusted dealer setup.
//!
//! ## AEGIS-256 (IACR 2024/268)
//! Fast authenticated encryption that is 2-5x faster than AES-GCM on modern CPUs.
//! Uses AES-NI instructions for hardware-accelerated performance.
//!
//! ## Circle STARKs (IACR 2024/278 - Starkware)
//! 10-100x smaller proofs than traditional STARKs with same security level.
//! Uses circle groups for more efficient polynomial commitments.
//!
//! ## Lattice-Based Aggregate Signatures (IACR 2025/1056)
//! ~98% size reduction through signature aggregation using Module-LWE.
//! Supports both same-message and different-message aggregation.
//!
//! ## Genus-2 Curve VDF (IACR 2025/1050)
//! Quantum-safe verifiable delay function using hyperelliptic curves.
//! Resistant to Shor's algorithm unlike RSA or class group VDFs.
//!
//! ## SQIsign: Compact Isogeny Signatures (IACR 2025/847)
//! Smallest post-quantum signatures (204 bytes) based on supersingular isogenies.
//! Uses dimension-4 isogenies for efficient signing and verification.
//!
//! ## Improved Bulletproofs v2 (IACR 2024/313)
//! Efficient zero-knowledge range proofs with 2x faster verification.
//! Supports aggregated proofs for multiple values.
//!
//! v1.0.58-beta: Added Bulletproofs v2

pub mod frost;
pub mod aegis;
pub mod circle_stark;
pub mod lattice_aggregate;
pub mod genus2_vdf;
pub mod sqisign;
pub mod bulletproofs_v2;
pub mod timelock;
pub mod errors;

pub use frost::{
    FrostKeyGen, FrostSigner, FrostVerifier,
    ThresholdSignature, ValidatorCommittee, KeyShare,
    Identifier, SigningCommitments, SignatureShare, GroupPublicKey,
};
pub use aegis::{Aegis256, AegisKey, AegisNonce, AegisStreamEncryptor, AegisStreamDecryptor};
pub use circle_stark::{CircleStarkProver, CircleStarkVerifier, CircleProof, FIELD_MODULUS};
pub use lattice_aggregate::{
    LatticeParams, LatticeKeyPair, LatticePublicKey, LatticeSecretKey,
    LatticeSignature, AggregateSignature, SignatureAggregator,
};
pub use genus2_vdf::{
    Genus2Params, Genus2Vdf, VdfSecurityLevel, VdfOutput, VdfProof,
    JacobianPoint, FieldElement, VdfBatchVerifier,
};
pub use sqisign::{
    SqiSignLevel, SqiSignParams, SqiSignKeyPair, SqiSignPublicKey,
    SqiSignature, SqiSignVerifier, SqiSignBatchVerifier,
    SqiSignAggregator, AggregatedSqiSign,
};
pub use bulletproofs_v2::{
    Scalar, Point, RangeProof, BulletproofsProver, BulletproofsVerifier,
    AggregatedRangeProof, AggregatedProver, InnerProductProof,
    DEFAULT_RANGE_BITS,
};
pub use timelock::{TimeLockConfig, TimeLockedCiphertext, VDFChallenge};
pub use errors::CryptoError;

/// Version of the crypto module
pub const VERSION: &str = "1.0.58-beta";

/// Security levels supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityLevel {
    /// 128-bit security (standard)
    Standard128,
    /// 192-bit security (enhanced)
    Enhanced192,
    /// 256-bit security (maximum)
    Maximum256,
}

impl Default for SecurityLevel {
    fn default() -> Self {
        SecurityLevel::Standard128
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_levels() {
        assert_eq!(SecurityLevel::default(), SecurityLevel::Standard128);
    }
}
