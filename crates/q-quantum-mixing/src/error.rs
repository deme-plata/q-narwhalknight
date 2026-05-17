//! Error handling for the quantum mixing system

use thiserror::Error;

/// Result type for mixing operations
pub type Result<T> = std::result::Result<T, MixingError>;

/// Errors that can occur during mixing operations
#[derive(Error, Debug, Clone)]
pub enum MixingError {
    #[error("Cryptographic error: {0}")]
    CryptographicError(String),

    #[error("Ring signature verification failed: {0}")]
    RingSignatureError(String),

    #[error("Zero-knowledge proof generation failed: {0}")]
    ZKProofError(String),

    #[error("Stealth address generation failed: {0}")]
    StealthAddressError(String),

    #[error("Mixing pool error: {0}")]
    PoolError(String),

    #[error("Quantum entropy error: {0}")]
    EntropyError(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Compliance rejection: {0}")]
    ComplianceRejection(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("Insufficient participants: required {required}, got {actual}")]
    InsufficientParticipants { required: usize, actual: usize },

    #[error("Byzantine fault detected: {0}")]
    ByzantineFault(String),

    #[error("Timeout during operation: {0}")]
    Timeout(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Feature not implemented: {0}")]
    NotImplemented(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Security violation: {0}")]
    SecurityViolation(String),
}

impl From<bincode::Error> for MixingError {
    fn from(err: bincode::Error) -> Self {
        MixingError::SerializationError(err.to_string())
    }
}

impl From<serde_json::Error> for MixingError {
    fn from(err: serde_json::Error) -> Self {
        MixingError::SerializationError(err.to_string())
    }
}

impl From<tokio::time::error::Elapsed> for MixingError {
    fn from(err: tokio::time::error::Elapsed) -> Self {
        MixingError::Timeout(err.to_string())
    }
}

impl From<chrono::OutOfRangeError> for MixingError {
    fn from(err: chrono::OutOfRangeError) -> Self {
        MixingError::InvalidParameters(format!("Time out of range: {}", err))
    }
}