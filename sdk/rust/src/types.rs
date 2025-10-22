//! Common types for the Q-NarwhalKnight PaaS SDK

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Privacy level for transaction mixing
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PrivacyLevel {
    /// Standard privacy (10 QUG/tx)
    Standard,
    /// Enhanced privacy (50 QUG/tx)
    Enhanced,
    /// Maximum privacy (200 QUG/tx)
    Maximum,
}

impl std::fmt::Display for PrivacyLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrivacyLevel::Standard => write!(f, "standard"),
            PrivacyLevel::Enhanced => write!(f, "enhanced"),
            PrivacyLevel::Maximum => write!(f, "maximum"),
        }
    }
}

/// Result from a mixing operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixResult {
    pub mix_id: String,
    pub status: MixStatus,
    pub estimated_completion_seconds: u64,
    pub quantum_entropy_applied: bool,
    pub tor_circuit_used: Option<String>,
}

/// Status of a mixing operation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MixStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

/// API key information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyInfo {
    pub key_id: String,
    pub tier: ApiTier,
    pub rate_limit: u32,
    pub expires_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
}

/// API subscription tier
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ApiTier {
    Free,
    Standard,
    Premium,
    Enterprise,
}

/// Billing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingInfo {
    pub balance_qug: u64,
    pub balance_usd: f64,
    pub total_spent_qug: u64,
    pub total_requests: u64,
}

/// API request wrapper
#[derive(Debug, Serialize)]
pub struct ApiRequest<T> {
    pub idempotency_key: String,
    #[serde(flatten)]
    pub data: T,
}

/// API response wrapper
#[derive(Debug, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}
