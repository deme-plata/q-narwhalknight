// REST endpoint: GET /api/v1/turn/credentials
//
// Returns time-limited TURN credentials for the authenticated wallet.
// The q-turn server validates them via the same HMAC-SHA256(secret, username) formula.
//
// Response format matches RTCIceServer / WebRTC credential spec:
//   { "username": "1715000000:qnk...", "password": "hexhmac...",
//     "ttl": 600, "uris": ["turn:quillon.xyz:3478?transport=udp"] }

use axum::{http::StatusCode, response::Json};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use serde::Serialize;

use crate::wallet_auth::AuthenticatedWallet;

type HmacSha256 = Hmac<Sha256>;

#[derive(Serialize)]
pub struct TurnCredentials {
    pub username: String,
    pub password: String,
    pub ttl:      u64,
    pub uris:     Vec<String>,
}

/// Handler: GET /api/v1/turn/credentials
/// Requires a valid X-Wallet-Auth header (enforced by AuthenticatedWallet extractor).
pub async fn get_turn_credentials(
    auth: AuthenticatedWallet,
) -> Result<Json<TurnCredentials>, StatusCode> {
    let secret = std::env::var("Q_TURN_SECRET").unwrap_or_default();
    if secret.is_empty() {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    let ttl: u64 = std::env::var("Q_TURN_TTL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);

    let wallet_str = format!("qnk{}", hex::encode(auth.address));
    let ts = chrono::Utc::now().timestamp();
    let username = format!("{}:{}", ts, wallet_str);
    let password = derive_turn_password(&secret, &username);

    let turn_host = std::env::var("Q_TURN_HOST")
        .unwrap_or_else(|_| "quillon.xyz:3478".to_string());

    Ok(Json(TurnCredentials {
        username,
        password,
        ttl,
        uris: vec![
            format!("turn:{}?transport=udp", turn_host),
        ],
    }))
}

/// HMAC-SHA256(secret, username) — must match q-turn/src/auth.rs::derive_password()
pub fn derive_turn_password(secret: &str, username: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .expect("HMAC accepts any key size");
    mac.update(username.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}
