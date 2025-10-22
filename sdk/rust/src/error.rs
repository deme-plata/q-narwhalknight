//! Error types for the Q-NarwhalKnight PaaS SDK

use thiserror::Error;

pub type Result<T> = std::result::Result<T, PaaSError>;

#[derive(Error, Debug)]
pub enum PaaSError {
    #[error("API request failed: {0}")]
    ApiError(String),

    #[error("Authentication failed: {0}")]
    AuthError(String),

    #[error("Invalid API key")]
    InvalidApiKey,

    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    #[error("Insufficient balance: required {required} QUG, have {available} QUG")]
    InsufficientBalance { required: u64, available: u64 },

    #[error("Bitcoin error: {0}")]
    BitcoinError(String),

    #[error("Ethereum error: {0}")]
    EthereumError(String),

    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Invalid transaction: {0}")]
    InvalidTransaction(String),

    #[error("Wallet error: {0}")]
    WalletError(String),

    #[error("Timeout: operation took too long")]
    Timeout,

    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl From<bitcoin::address::ParseError> for PaaSError {
    fn from(e: bitcoin::address::ParseError) -> Self {
        PaaSError::BitcoinError(format!("Address parse error: {}", e))
    }
}
