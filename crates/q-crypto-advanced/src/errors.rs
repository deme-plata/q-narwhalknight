//! Error types for advanced cryptography operations

use thiserror::Error;

/// Errors that can occur during cryptographic operations
#[derive(Error, Debug)]
pub enum CryptoError {
    // FROST Errors
    #[error("FROST: Invalid threshold parameters (t={threshold}, n={total})")]
    InvalidThreshold { threshold: usize, total: usize },

    #[error("FROST: Not enough participants for signing (have={have}, need={need})")]
    InsufficientParticipants { have: usize, need: usize },

    #[error("FROST: Invalid key share")]
    InvalidKeyShare,

    #[error("FROST: Invalid signature share from participant {0}")]
    InvalidSignatureShare(u16),

    #[error("FROST: Key generation failed: {0}")]
    KeyGenFailed(String),

    #[error("FROST: Signing failed: {0}")]
    SigningFailed(String),

    #[error("FROST: Verification failed")]
    VerificationFailed,

    #[error("FROST: Participant {0} not found")]
    ParticipantNotFound(u16),

    // AEGIS Errors
    #[error("AEGIS: Invalid key length (expected 32 bytes, got {0})")]
    InvalidKeyLength(usize),

    #[error("AEGIS: Invalid nonce length (expected 32 bytes, got {0})")]
    InvalidNonceLength(usize),

    #[error("AEGIS: Encryption failed")]
    EncryptionFailed,

    #[error("AEGIS: Decryption failed - authentication tag mismatch")]
    DecryptionFailed,

    #[error("AEGIS: Ciphertext too short")]
    CiphertextTooShort,

    // Circle STARK Errors
    #[error("Circle STARK: Invalid proof format")]
    InvalidProofFormat,

    #[error("Circle STARK: Proof verification failed")]
    ProofVerificationFailed,

    #[error("Circle STARK: Witness generation failed: {0}")]
    WitnessGenerationFailed(String),

    #[error("Circle STARK: Trace generation failed: {0}")]
    TraceGenerationFailed(String),

    #[error("Circle STARK: Polynomial commitment failed")]
    CommitmentFailed,

    // Bulletproofs Errors
    #[error("Bulletproofs: Proof generation failed: {0}")]
    ProofGenerationFailed(String),

    #[error("Bulletproofs: Range proof verification failed")]
    RangeProofVerificationFailed,

    // Lattice Aggregate Signature Errors
    #[error("Lattice: Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("Lattice: Aggregation failed: {0}")]
    AggregationFailed(String),

    #[error("Lattice: Invalid public key")]
    InvalidPublicKey,

    #[error("Lattice: Invalid signature")]
    InvalidSignature,

    // General Errors
    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    #[error("Random number generation failed")]
    RngFailed,

    #[error("Internal error: {0}")]
    InternalError(String),
}

impl From<std::io::Error> for CryptoError {
    fn from(e: std::io::Error) -> Self {
        CryptoError::InternalError(e.to_string())
    }
}

impl From<serde_json::Error> for CryptoError {
    fn from(e: serde_json::Error) -> Self {
        CryptoError::SerializationError(e.to_string())
    }
}
