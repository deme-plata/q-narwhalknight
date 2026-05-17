//! Pool error types

use thiserror::Error;

/// Pool result type
pub type PoolResult<T> = Result<T, PoolError>;

/// Pool error types
#[derive(Debug, Error)]
pub enum PoolError {
    /// Worker errors
    #[error("Worker error: {0}")]
    Worker(#[from] WorkerError),

    /// Share errors
    #[error("Share error: {0}")]
    Share(#[from] ShareError),

    /// Job errors
    #[error("Job error: {0}")]
    Job(#[from] JobError),

    /// Stratum protocol errors
    #[error("Stratum error: {0}")]
    Stratum(#[from] StratumError),

    /// Payout errors
    #[error("Payout error: {0}")]
    Payout(#[from] PayoutError),

    /// Network I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// Database errors
    #[error("Database error: {0}")]
    Database(String),

    /// Internal errors
    #[error("Internal error: {0}")]
    Internal(String),

    /// Invalid worker ID
    #[error("Invalid worker: {0}")]
    InvalidWorker(String),

    /// Invalid job ID
    #[error("Invalid job: {0}")]
    InvalidJob(String),
}

/// Worker-related errors
#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("Worker not found: {0}")]
    NotFound(String),

    #[error("Invalid worker name: {0}")]
    InvalidName(String),

    #[error("Worker already exists: {0}")]
    AlreadyExists(String),

    #[error("Worker banned: {0}")]
    Banned(String),

    #[error("Rate limit exceeded for worker: {0}")]
    RateLimitExceeded(String),

    #[error("Invalid wallet address: {0}")]
    InvalidWallet(String),

    #[error("Authentication failed for worker: {0}")]
    AuthenticationFailed(String),
}

/// Share-related errors
#[derive(Debug, Error)]
pub enum ShareError {
    #[error("Stale share (job {0} expired)")]
    Stale(String),

    #[error("Duplicate share")]
    Duplicate,

    #[error("Low difficulty share (got {got:.6}, need {need:.6})")]
    LowDifficulty { got: f64, need: f64 },

    #[error("Invalid nonce")]
    InvalidNonce,

    #[error("Invalid hash")]
    InvalidHash,

    #[error("Malformed submission")]
    Malformed,

    #[error("Job not found: {0}")]
    JobNotFound(String),
}

/// Job-related errors
#[derive(Debug, Error)]
pub enum JobError {
    #[error("No job available")]
    NoJobAvailable,

    #[error("Job expired: {0}")]
    Expired(String),

    #[error("Failed to create job: {0}")]
    CreationFailed(String),

    #[error("Invalid block template: {0}")]
    InvalidBlockTemplate(String),
}

/// Stratum protocol errors
#[derive(Debug, Error)]
pub enum StratumError {
    #[error("Connection closed")]
    ConnectionClosed,

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Invalid message format")]
    InvalidMessage,

    #[error("Unknown method: {0}")]
    UnknownMethod(String),

    #[error("Missing required parameter: {0}")]
    MissingParameter(String),

    #[error("TLS error: {0}")]
    Tls(String),

    #[error("Max connections reached")]
    MaxConnectionsReached,
}

/// Payout-related errors
#[derive(Debug, Error)]
pub enum PayoutError {
    #[error("Insufficient balance for payout")]
    InsufficientBalance,

    #[error("Payout below minimum threshold")]
    BelowThreshold,

    #[error("Transaction failed: {0}")]
    TransactionFailed(String),

    #[error("Batch transaction failed: {0}")]
    BatchFailed(String),

    #[error("Invalid recipient address: {0}")]
    InvalidRecipient(String),
}

/// Stratum error codes (JSON-RPC compatible)
#[derive(Debug, Clone, Copy)]
pub enum StratumErrorCode {
    /// Stale share
    Stale = 21,
    /// Low difficulty
    LowDifficulty = 23,
    /// Duplicate share
    Duplicate = 22,
    /// Invalid nonce
    InvalidNonce = 24,
    /// Unauthorized
    Unauthorized = 25,
    /// Not subscribed
    NotSubscribed = 26,
    /// Unknown
    Unknown = 20,
}

impl From<&ShareError> for StratumErrorCode {
    fn from(err: &ShareError) -> Self {
        match err {
            ShareError::Stale(_) => StratumErrorCode::Stale,
            ShareError::LowDifficulty { .. } => StratumErrorCode::LowDifficulty,
            ShareError::Duplicate => StratumErrorCode::Duplicate,
            ShareError::InvalidNonce => StratumErrorCode::InvalidNonce,
            _ => StratumErrorCode::Unknown,
        }
    }
}

impl StratumErrorCode {
    /// Convert to JSON-RPC error tuple
    pub fn to_json_rpc(self, message: &str) -> (i32, String, Option<serde_json::Value>) {
        (self as i32, message.to_string(), None)
    }
}
