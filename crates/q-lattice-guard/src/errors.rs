//! Error types for LatticeGuard

use thiserror::Error;

/// Errors that can occur in LatticeGuard operations
#[derive(Error, Debug)]
pub enum LatticeGuardError {
    /// Invalid parameter configuration
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),

    /// Polynomial degree exceeds maximum
    #[error("Polynomial degree {0} exceeds maximum {1}")]
    DegreeTooLarge(usize, usize),

    /// Proof verification failed
    #[error("Proof verification failed: {0}")]
    VerificationFailed(String),

    /// Error bound exceeded in approximate computation
    #[error("Error bound exceeded: actual {0}, maximum {1}")]
    ErrorBoundExceeded(u64, u64),

    /// Commitment verification failed
    #[error("Commitment verification failed")]
    CommitmentInvalid,

    /// RLWE encryption error
    #[error("RLWE encryption error: {0}")]
    EncryptionError(String),

    /// RLWE decryption error
    #[error("RLWE decryption error: {0}")]
    DecryptionError(String),

    /// NTT operation error
    #[error("NTT operation error: {0}")]
    NttError(String),

    /// Transcript error
    #[error("Transcript error: {0}")]
    TranscriptError(String),

    /// Circuit constraint violation
    #[error("Circuit constraint {0} violated")]
    ConstraintViolation(usize),

    /// Witness size mismatch
    #[error("Witness size mismatch: expected {0}, got {1}")]
    WitnessSizeMismatch(usize, usize),

    /// SRS insufficient for circuit
    #[error("SRS max constraints {0} insufficient for circuit with {1} constraints")]
    SrsInsufficient(usize, usize),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(String),

    /// Internal error
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl From<bincode::Error> for LatticeGuardError {
    fn from(e: bincode::Error) -> Self {
        LatticeGuardError::SerializationError(e.to_string())
    }
}
