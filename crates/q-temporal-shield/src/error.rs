//! Error types for TemporalShield-STARK
//!
//! Comprehensive error handling for all cryptographic operations.

use thiserror::Error;

/// Main error type for TemporalShield operations
#[derive(Debug, Error)]
pub enum TemporalError {
    // Configuration errors
    #[error("Trustee count mismatch: expected {expected}, got {actual}")]
    TrusteeCountMismatch { expected: usize, actual: usize },

    #[error("Invalid threshold: k={threshold} must be <= n={total_trustees} and > 0")]
    InvalidThreshold { threshold: usize, total_trustees: usize },

    #[error("Configuration hash mismatch")]
    ConfigMismatch,

    #[error("Metadata does not match expected configuration")]
    MetadataMismatch,

    // Share-related errors
    #[error("Insufficient shares for reconstruction: have {have}, need {need}")]
    InsufficientShares { have: usize, need: usize },

    #[error("Share commitment mismatch at index {index}")]
    ShareCommitmentMismatch { index: usize },

    #[error("Invalid share index: {0}")]
    InvalidShareIndex(usize),

    #[error("Share decryption failed for trustee {trustee_id}")]
    ShareDecryptionFailed { trustee_id: String },

    // Key-related errors
    #[error("Key length mismatch: key={key_len}, ciphertext={ciphertext_len}")]
    KeyLengthMismatch { key_len: usize, ciphertext_len: usize },

    #[error("Key commitment verification failed")]
    KeyCommitmentMismatch,

    #[error("Key not found in HSM: {}", hex::encode(.0))]
    KeyNotFound([u8; 32]),

    #[error("Key generation failed: {0}")]
    KeyGenerationFailed(String),

    // Cryptographic operation errors
    #[error("Encryption failed: {0}")]
    EncryptionFailed(String),

    #[error("Decryption failed: {0}")]
    DecryptionFailed(String),

    #[error("KEM encapsulation failed: {0}")]
    KemEncapsulationFailed(String),

    #[error("KEM decapsulation failed: {0}")]
    KemDecapsulationFailed(String),

    #[error("Signature generation failed: {0}")]
    SignatureFailed(String),

    #[error("Signature verification failed")]
    SignatureVerificationFailed,

    // STARK proof errors
    #[error("STARK proof generation failed: {0}")]
    ProofGenerationFailed(String),

    #[error("STARK proof verification failed: {0}")]
    ProofVerificationFailed(String),

    #[error("Invalid proof format")]
    InvalidProofFormat,

    #[error("Trace generation failed: {0}")]
    TraceGenerationFailed(String),

    // Field arithmetic errors
    #[error("Field element overflow")]
    FieldOverflow,

    #[error("Division by zero in field arithmetic")]
    DivisionByZero,

    #[error("Invalid field element encoding")]
    InvalidFieldElement,

    #[error("Polynomial evaluation failed")]
    PolynomialEvaluationFailed,

    // Serialization errors
    #[error("Serialization failed: {0}")]
    SerializationFailed(String),

    #[error("Deserialization failed: {0}")]
    DeserializationFailed(String),

    // System errors
    #[error("Random number generation failed: {0}")]
    RandomnessFailed(String),

    #[error("System time error")]
    TimeError,

    #[error("HSM operation failed: {0}")]
    HsmError(String),

    #[error("I/O error: {0}")]
    IoError(String),
}

impl From<std::io::Error> for TemporalError {
    fn from(err: std::io::Error) -> Self {
        TemporalError::IoError(err.to_string())
    }
}

impl From<bincode::Error> for TemporalError {
    fn from(err: bincode::Error) -> Self {
        TemporalError::SerializationFailed(err.to_string())
    }
}

impl From<getrandom::Error> for TemporalError {
    fn from(err: getrandom::Error) -> Self {
        TemporalError::RandomnessFailed(err.to_string())
    }
}

/// Result type alias for TemporalShield operations
pub type TemporalResult<T> = Result<T, TemporalError>;
