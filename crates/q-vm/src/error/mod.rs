use std::fmt;
use thiserror::Error;

#[derive(Debug, Error, Clone)]
pub enum Error {
    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("I/O error: {0}")]
    Io(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("General error: {0}")]
    General(String),

    #[error("Security error: {0}")]
    Security(String),

    #[error("Feature not implemented: {0}")]
    NotImplemented(String),
}

// VmError is a placeholder for now
#[derive(Debug, Clone)]
pub struct VmError(pub String);

// Define a Result type
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VM Error: {}", self.0)
    }
}

impl std::error::Error for VmError {}

// Allow conversion from various error types
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

impl From<String> for Error {
    fn from(e: String) -> Self {
        Error::General(e)
    }
}

impl From<&str> for Error {
    fn from(e: &str) -> Self {
        Error::General(e.to_string())
    }
}
