use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};
use chrono::Utc;
use dashmap::DashMap;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::AppState;
use q_types::ApiResponse;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Status of a payment request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PaymentRequestStatus {
    Pending,
    Paid,
    Expired,
    Cancelled,
}

/// Persisted payment request (stored in the DashMap).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentRequest {
    pub request_id: String,
    pub to_address: String,
    pub amount: f64,
    pub memo: Option<String>,
    pub currency: String,
    pub status: PaymentRequestStatus,
    pub tx_hash: Option<String>,
    pub paid_at: Option<i64>,
    pub expires_at: i64,
    pub created_at: i64,
}

/// JSON body accepted by `POST /api/v1/payment-requests`.
#[derive(Debug, Deserialize)]
pub struct CreatePaymentRequestInput {
    pub to_address: String,
    pub amount: f64,
    /// Optional human-readable memo (e.g. "Coffee + pastry").
    pub memo: Option<String>,
    /// Currency symbol. Defaults to "QUG" when absent.
    pub currency: Option<String>,
    /// Time-to-live in seconds. Defaults to 300 (5 min) when absent.
    pub expiry_secs: Option<u64>,
}

/// JSON body returned by the create endpoint.
#[derive(Debug, Serialize, Deserialize)]
pub struct CreatePaymentRequestOutput {
    pub request_id: String,
    /// URI suitable for encoding into a QR code.
    pub qr_uri: String,
    /// Same value as `qr_uri` -- kept for backwards compat / clarity.
    pub qr_data: String,
    pub expires_at: i64,
    pub status: PaymentRequestStatus,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a request ID: `pay_` followed by 12 random hex characters.
fn generate_request_id() -> String {
    let mut rng = rand::thread_rng();
    let bytes: [u8; 6] = rng.gen();
    format!("pay_{}", hex::encode(bytes))
}

/// Build the `quillon:` URI that will be encoded into a QR code.
///
/// Format: `quillon:ADDRESS?amount=AMOUNT&memo=MEMO&request_id=ID`
fn build_qr_uri(address: &str, amount: f64, memo: Option<&str>, request_id: &str) -> String {
    let mut uri = format!(
        "quillon:{}?amount={}&request_id={}",
        address, amount, request_id
    );
    if let Some(m) = memo {
        // URL-encode the memo so special characters are safe for QR scanners.
        let encoded = urlencoded(m);
        uri.push_str(&format!("&memo={}", encoded));
    }
    uri
}

/// Minimal percent-encoding for the memo parameter.
/// Encodes characters that are not unreserved per RFC 3986.
fn urlencoded(input: &str) -> String {
    let mut out = String::with_capacity(input.len() * 2);
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z'
            | b'a'..=b'z'
            | b'0'..=b'9'
            | b'-'
            | b'_'
            | b'.'
            | b'~' => out.push(byte as char),
            _ => {
                out.push('%');
                out.push_str(&format!("{:02X}", byte));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /api/v1/payment-requests
///
/// Creates a new payment request and returns a QR-ready URI.
pub async fn create_payment_request(
    State(state): State<Arc<AppState>>,
    Json(input): Json<CreatePaymentRequestInput>,
) -> Result<Json<ApiResponse<CreatePaymentRequestOutput>>, StatusCode> {
    let request_id = generate_request_id();
    let currency = input.currency.unwrap_or_else(|| "QUG".to_string());
    let expiry_secs = input.expiry_secs.unwrap_or(300);
    let now = Utc::now().timestamp();
    let expires_at = now + expiry_secs as i64;

    let qr_uri = build_qr_uri(
        &input.to_address,
        input.amount,
        input.memo.as_deref(),
        &request_id,
    );

    let payment = PaymentRequest {
        request_id: request_id.clone(),
        to_address: input.to_address,
        amount: input.amount,
        memo: input.memo,
        currency,
        status: PaymentRequestStatus::Pending,
        tx_hash: None,
        paid_at: None,
        expires_at,
        created_at: now,
    };

    state.payment_requests.insert(request_id.clone(), payment);

    info!("Payment request created: {}", request_id);

    let output = CreatePaymentRequestOutput {
        request_id,
        qr_data: qr_uri.clone(),
        qr_uri,
        expires_at,
        status: PaymentRequestStatus::Pending,
    };

    Ok(Json(ApiResponse::success(output)))
}

/// GET /api/v1/payment-requests/:id
///
/// Returns the current status of a payment request.
pub async fn get_payment_request(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ApiResponse<PaymentRequest>>, StatusCode> {
    debug!("Looking up payment request: {}", id);

    match state.payment_requests.get(&id) {
        Some(entry) => {
            let mut pr = entry.value().clone();
            // Auto-expire if past the deadline and still pending.
            if pr.status == PaymentRequestStatus::Pending
                && Utc::now().timestamp() > pr.expires_at
            {
                pr.status = PaymentRequestStatus::Expired;
                // Update in storage as well.
                drop(entry);
                state.payment_requests.insert(id, pr.clone());
            }
            Ok(Json(ApiResponse::success(pr)))
        }
        None => Ok(Json(ApiResponse::error(
            "Payment request not found".to_string(),
        ))),
    }
}

/// DELETE /api/v1/payment-requests/:id
///
/// Cancels a pending payment request. Already-paid or expired requests
/// cannot be cancelled.
pub async fn cancel_payment_request(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ApiResponse<PaymentRequest>>, StatusCode> {
    debug!("Cancelling payment request: {}", id);

    match state.payment_requests.get_mut(&id) {
        Some(mut entry) => {
            let pr = entry.value_mut();
            match pr.status {
                PaymentRequestStatus::Pending => {
                    pr.status = PaymentRequestStatus::Cancelled;
                    info!("Payment request cancelled: {}", id);
                    let snapshot = pr.clone();
                    drop(entry);
                    Ok(Json(ApiResponse::success(snapshot)))
                }
                other => {
                    let msg = format!(
                        "Cannot cancel payment request in {:?} state",
                        other
                    );
                    warn!("{}: {}", msg, id);
                    Ok(Json(ApiResponse::error(msg)))
                }
            }
        }
        None => Ok(Json(ApiResponse::error(
            "Payment request not found".to_string(),
        ))),
    }
}

// ---------------------------------------------------------------------------
// Background cleanup task
// ---------------------------------------------------------------------------

/// Spawns a tokio task that runs every 30 seconds and marks expired pending
/// requests as `Expired`, then removes requests that have been in a terminal
/// state (expired / cancelled) for more than 1 hour.
pub fn spawn_expiry_cleanup(payment_requests: Arc<DashMap<String, PaymentRequest>>) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
        loop {
            interval.tick().await;
            let now = Utc::now().timestamp();
            let one_hour_ago = now - 3600;

            let mut to_remove: Vec<String> = Vec::new();

            for mut entry in payment_requests.iter_mut() {
                let pr = entry.value_mut();
                // Transition pending -> expired.
                if pr.status == PaymentRequestStatus::Pending && now > pr.expires_at {
                    pr.status = PaymentRequestStatus::Expired;
                    debug!("Payment request expired: {}", pr.request_id);
                }
                // Collect terminal entries older than 1 hour for removal.
                if matches!(
                    pr.status,
                    PaymentRequestStatus::Expired | PaymentRequestStatus::Cancelled
                ) && pr.expires_at < one_hour_ago
                {
                    to_remove.push(pr.request_id.clone());
                }
            }

            for id in to_remove {
                payment_requests.remove(&id);
                debug!("Cleaned up stale payment request: {}", id);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_request_id() {
        let id = generate_request_id();
        assert!(id.starts_with("pay_"));
        // "pay_" (4) + 12 hex chars = 16 total
        assert_eq!(id.len(), 16);
        // All chars after prefix should be valid hex.
        assert!(id[4..].chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_build_qr_uri_without_memo() {
        let uri = build_qr_uri("0xABCDEF", 10.5, None, "pay_abc123");
        assert_eq!(
            uri,
            "quillon:0xABCDEF?amount=10.5&request_id=pay_abc123"
        );
    }

    #[test]
    fn test_build_qr_uri_with_memo() {
        let uri = build_qr_uri("0xABCDEF", 10.5, Some("Coffee + pastry"), "pay_abc123");
        assert!(uri.contains("memo=Coffee%20%2B%20pastry"));
        assert!(uri.starts_with("quillon:0xABCDEF?amount=10.5"));
        assert!(uri.contains("request_id=pay_abc123"));
    }

    #[test]
    fn test_urlencoded() {
        assert_eq!(urlencoded("hello"), "hello");
        assert_eq!(urlencoded("a b"), "a%20b");
        assert_eq!(urlencoded("a+b"), "a%2Bb");
        assert_eq!(urlencoded("100%"), "100%25");
    }

    #[test]
    fn test_payment_request_status_serde() {
        let json = serde_json::to_string(&PaymentRequestStatus::Pending).unwrap();
        assert_eq!(json, "\"pending\"");
        let json = serde_json::to_string(&PaymentRequestStatus::Paid).unwrap();
        assert_eq!(json, "\"paid\"");
    }
}
