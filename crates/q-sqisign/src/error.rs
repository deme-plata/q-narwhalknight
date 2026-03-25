//! Error types for the SQIsign wrapper.

use thiserror::Error;

/// Errors that can occur during SQIsign operations.
#[derive(Debug, Error)]
pub enum SqiSignError {
    /// Key generation failed (C library returned non-zero).
    #[error("SQIsign key generation failed (status={0})")]
    KeyGenFailed(i32),

    /// Signing failed (C library returned non-zero).
    #[error("SQIsign signing failed (status={0})")]
    SigningFailed(i32),

    /// Signature verification failed (invalid signature).
    #[error("SQIsign verification failed: invalid signature")]
    VerificationFailed,

    /// Invalid public key size.
    #[error("invalid public key size: expected {expected}, got {got}")]
    InvalidPublicKeySize { expected: usize, got: usize },

    /// Invalid signature size.
    #[error("invalid signature size: expected {expected}, got {got}")]
    InvalidSignatureSize { expected: usize, got: usize },
}
