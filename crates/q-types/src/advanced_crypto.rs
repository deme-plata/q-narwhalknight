//! Advanced Cryptographic Primitives Integration
//!
//! Re-exports cutting-edge cryptographic primitives from `q-crypto-advanced` crate.
//! These implement recent IACR ePrint papers for post-quantum and zero-knowledge cryptography.
//!
//! ## Available Primitives
//!
//! ### Phase 1 (Implemented)
//! - **FROST Threshold Signatures** (IACR 2025/1024): t-of-n threshold Schnorr signing
//! - **AEGIS-256** (IACR 2024/268): Fast authenticated encryption (2-5x faster than AES-GCM)
//! - **Circle STARKs** (IACR 2024/278): Compact zero-knowledge proofs
//!
//! ### Phase 2 (Implemented)
//! - **Lattice Aggregate Signatures** (IACR 2025/1056): ~98% signature size reduction
//! - **Genus-2 VDF** (IACR 2025/1050): Quantum-safe verifiable delay function
//! - **SQIsign** (IACR 2025/847): Smallest PQ signatures (204 bytes)
//! - **Bulletproofs v2** (IACR 2024/313): Efficient range proofs
//!
//! ## Usage
//!
//! Enable the `advanced-crypto` feature in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! q-types = { path = "...", features = ["advanced-crypto"] }
//! ```
//!
//! Then import the primitives:
//!
//! ```ignore
//! use q_types::advanced_crypto::frost::FrostKeyGen;
//! use q_types::advanced_crypto::aegis::Aegis256;
//! use q_types::advanced_crypto::sqisign::SqiSignKeyPair;
//! ```

#[cfg(feature = "advanced-crypto")]
pub use q_crypto_advanced::{
    // Phase 1: FROST Threshold Signatures
    frost::{
        FrostKeyGen, FrostSigner, FrostVerifier,
        ThresholdSignature, ValidatorCommittee, KeyShare,
        Identifier, SigningCommitments, SignatureShare, GroupPublicKey,
    },

    // Phase 1: AEGIS-256 Fast AEAD
    aegis::{
        Aegis256, AegisKey, AegisNonce,
        AegisStreamEncryptor, AegisStreamDecryptor,
    },

    // Phase 1: Circle STARKs
    circle_stark::{
        CircleStarkProver, CircleStarkVerifier, CircleProof,
        FIELD_MODULUS,
    },

    // Phase 2: Lattice Aggregate Signatures
    lattice_aggregate::{
        LatticeParams, LatticeKeyPair, LatticePublicKey, LatticeSecretKey,
        LatticeSignature, AggregateSignature, SignatureAggregator,
    },

    // Phase 2: Genus-2 VDF
    genus2_vdf::{
        Genus2Params, Genus2Vdf, VdfSecurityLevel, VdfOutput, VdfProof,
        JacobianPoint, FieldElement, VdfBatchVerifier,
    },

    // Phase 2: SQIsign Compact Signatures
    sqisign::{
        SqiSignLevel, SqiSignParams, SqiSignKeyPair, SqiSignPublicKey,
        SqiSignature, SqiSignVerifier, SqiSignBatchVerifier,
        SqiSignAggregator, AggregatedSqiSign,
    },

    // Phase 2: Bulletproofs v2
    bulletproofs_v2::{
        Scalar, Point, RangeProof, BulletproofsProver, BulletproofsVerifier,
        AggregatedRangeProof, AggregatedProver, InnerProductProof,
        DEFAULT_RANGE_BITS,
    },

    // Common types
    CryptoError, SecurityLevel, VERSION,
};

/// Re-export individual modules for direct access
#[cfg(feature = "advanced-crypto")]
pub mod frost {
    pub use q_crypto_advanced::frost::*;
}

#[cfg(feature = "advanced-crypto")]
pub mod aegis {
    pub use q_crypto_advanced::aegis::*;
}

#[cfg(feature = "advanced-crypto")]
pub mod circle_stark {
    pub use q_crypto_advanced::circle_stark::*;
}

#[cfg(feature = "advanced-crypto")]
pub mod lattice_aggregate {
    pub use q_crypto_advanced::lattice_aggregate::*;
}

#[cfg(feature = "advanced-crypto")]
pub mod genus2_vdf {
    pub use q_crypto_advanced::genus2_vdf::*;
}

#[cfg(feature = "advanced-crypto")]
pub mod sqisign {
    pub use q_crypto_advanced::sqisign::*;
}

#[cfg(feature = "advanced-crypto")]
pub mod bulletproofs_v2 {
    pub use q_crypto_advanced::bulletproofs_v2::*;
}

#[cfg(not(feature = "advanced-crypto"))]
compile_error!("The `advanced-crypto` feature is required to use this module. Enable it in Cargo.toml: q-types = { features = [\"advanced-crypto\"] }");
