//! Signed game orders (v10.11.114h).
//!
//! `POST /action` and `POST /join` decide who commands a realm, so they must come
//! from the wallet they name. Until v10.11.114h both trusted the `wallet` string in
//! the body: anyone could order any realm, or claim a free faction in another
//! wallet's name.
//!
//! The proof is the node's `X-Wallet-Auth` header (q-api-server `wallet_auth.rs`),
//! Ed25519 scheme, with one field that is REQUIRED here: `body_hash`, the hex
//! SHA3-256 of the exact request body bytes. The signed message is
//!
//! ```text
//! SHA3-256( pubkey ‖ timestamp as i64 LE ‖ full request path ‖ SHA3-256(body) )
//! ```
//!
//! which is byte-for-byte what `AuthenticatedWallet` builds when `body_hash` is
//! present, so a client that already signs node requests only adds the body hash.
//! A Quillon address is `qnk` + hex(Ed25519 public key): the address IS the key.
//!
//! An accepted signature is remembered for the timestamp window and refused if it
//! arrives again. Ed25519 is deterministic, so two identical orders signed in the
//! same second carry the same signature; clients add a `nonce` field to the body
//! (the game ignores unknown fields) to make them distinct.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use axum::http::{HeaderMap, StatusCode};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use sha3::{Digest, Sha3_256};

/// Header carrying the signature, shared with every other signed node route.
pub const AUTH_HEADER: &str = "X-Wallet-Auth";

/// Accepted distance between the signer's clock and ours (same as the node).
pub const MAX_CLOCK_SKEW_SECS: i64 = 300;

#[derive(Debug, Deserialize)]
struct WalletAuthHeader {
    address: String,
    timestamp: i64,
    #[serde(default)]
    scheme: Option<String>,
    #[serde(default)]
    signature: Option<String>,
    #[serde(default)]
    body_hash: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderAuthError {
    MissingHeader,
    BadHeader(String),
    UnsupportedScheme(String),
    Expired { age_secs: i64 },
    BadAddress,
    MissingBodyHash,
    BodyHashMismatch,
    BadSignature,
    WrongWallet { signer: String, named: String },
    Replayed,
}

impl OrderAuthError {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::WrongWallet { .. } => StatusCode::FORBIDDEN,
            Self::Replayed => StatusCode::CONFLICT,
            _ => StatusCode::UNAUTHORIZED,
        }
    }
}

impl std::fmt::Display for OrderAuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingHeader => write!(
                f,
                "Game orders must be signed: send an X-Wallet-Auth header (Ed25519) with body_hash"
            ),
            Self::BadHeader(e) => write!(f, "Invalid X-Wallet-Auth header: {e}"),
            Self::UnsupportedScheme(s) => {
                write!(f, "Game orders accept only the Ed25519 scheme, got '{s}'")
            }
            Self::Expired { age_secs } => write!(
                f,
                "Signature timestamp is {age_secs} s off the server clock (limit ±{MAX_CLOCK_SKEW_SECS} s)"
            ),
            Self::BadAddress => write!(f, "X-Wallet-Auth address is not a 32-byte qnk address"),
            Self::MissingBodyHash => write!(
                f,
                "Game orders must bind the body: X-Wallet-Auth needs body_hash = hex SHA3-256 of the request body"
            ),
            Self::BodyHashMismatch => {
                write!(f, "body_hash does not match the request body that arrived")
            }
            Self::BadSignature => write!(f, "Ed25519 signature verification failed"),
            Self::WrongWallet { signer, named } => write!(
                f,
                "Order signed by {signer} but names wallet {named}; a wallet can only order its own realm"
            ),
            Self::Replayed => write!(
                f,
                "This exact signed order was already received; add a nonce field to repeat an order"
            ),
        }
    }
}

/// A signature that verified, before the replay check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedSigner {
    pub key: [u8; 32],
    pub signature: [u8; 64],
    pub timestamp: i64,
}

/// `qnk` + 64 lowercase hex: the form realms store.
pub fn canonical_wallet(key: &[u8; 32]) -> String {
    format!("qnk{}", hex::encode(key))
}

/// Parse a `qnk…` (or bare hex) address into its 32 key bytes, any hex case.
pub fn parse_wallet(s: &str) -> Option<[u8; 32]> {
    let h = s.strip_prefix("qnk").unwrap_or(s);
    hex::decode(h).ok()?.try_into().ok()
}

/// Verify `X-Wallet-Auth` over `path` and the exact `body` bytes. Pure: no replay state.
pub fn verify_signed_order(
    headers: &HeaderMap,
    path: &str,
    body: &[u8],
    now: i64,
) -> Result<VerifiedSigner, OrderAuthError> {
    let raw = headers
        .get(AUTH_HEADER)
        .ok_or(OrderAuthError::MissingHeader)?
        .to_str()
        .map_err(|_| OrderAuthError::BadHeader("not visible ASCII".into()))?;
    let auth: WalletAuthHeader =
        serde_json::from_str(raw).map_err(|e| OrderAuthError::BadHeader(e.to_string()))?;

    match auth.scheme.as_deref() {
        None | Some("Ed25519") => {}
        Some(other) => return Err(OrderAuthError::UnsupportedScheme(other.to_string())),
    }

    let age_secs = now - auth.timestamp;
    if age_secs.abs() > MAX_CLOCK_SKEW_SECS {
        return Err(OrderAuthError::Expired { age_secs });
    }

    let key = parse_wallet(&auth.address).ok_or(OrderAuthError::BadAddress)?;

    let claimed = auth.body_hash.as_deref().ok_or(OrderAuthError::MissingBodyHash)?;
    let claimed = hex::decode(claimed.strip_prefix("0x").unwrap_or(claimed))
        .map_err(|_| OrderAuthError::BadHeader("body_hash is not hex".into()))?;
    let body_hash = Sha3_256::digest(body);
    if claimed.as_slice() != body_hash.as_slice() {
        return Err(OrderAuthError::BodyHashMismatch);
    }

    let mut hasher = Sha3_256::new();
    hasher.update(key);
    hasher.update(auth.timestamp.to_le_bytes());
    hasher.update(path.as_bytes());
    hasher.update(body_hash);
    let message = hasher.finalize();

    let signature: [u8; 64] = auth
        .signature
        .as_deref()
        .and_then(|s| hex::decode(s).ok())
        .and_then(|v| v.try_into().ok())
        .ok_or(OrderAuthError::BadSignature)?;
    let verifying_key = VerifyingKey::from_bytes(&key).map_err(|_| OrderAuthError::BadAddress)?;
    verifying_key
        .verify_strict(&message, &Signature::from_bytes(&signature))
        .map_err(|_| OrderAuthError::BadSignature)?;

    Ok(VerifiedSigner { key, signature, timestamp: auth.timestamp })
}

/// Signatures accepted within the clock window, refused a second time.
#[derive(Debug, Default)]
pub struct ReplayGuard {
    seen: Mutex<HashMap<[u8; 64], i64>>,
}

impl ReplayGuard {
    pub fn admit(&self, signer: &VerifiedSigner, now: i64) -> Result<(), OrderAuthError> {
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        // A signature is valid while |now - ts| <= window, so keep it exactly that long.
        seen.retain(|_, ts| (now - *ts).abs() <= MAX_CLOCK_SKEW_SECS);
        if seen.contains_key(&signer.signature) {
            return Err(OrderAuthError::Replayed);
        }
        seen.insert(signer.signature, signer.timestamp);
        Ok(())
    }
}

fn replay_guard() -> &'static ReplayGuard {
    static GUARD: OnceLock<ReplayGuard> = OnceLock::new();
    GUARD.get_or_init(ReplayGuard::default)
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Everything an order handler needs before it touches the game: the signature
/// verifies over this body, the body parses, the wallet it names is the signer,
/// and the signature has not been used before. Returns the body and the signer's
/// canonical wallet, which handlers use instead of the body's spelling.
pub fn authorize_order<T: DeserializeOwned>(
    headers: &HeaderMap,
    path: &str,
    body: &[u8],
    named_wallet: impl FnOnce(&T) -> &str,
) -> Result<(T, String), (StatusCode, String)> {
    let reject = |e: OrderAuthError| (e.status(), e.to_string());
    let now = unix_now();
    let signer = verify_signed_order(headers, path, body, now).map_err(reject)?;

    let parsed: T = serde_json::from_slice(body)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid request body: {e}")))?;
    let named = named_wallet(&parsed);
    if parse_wallet(named) != Some(signer.key) {
        return Err(reject(OrderAuthError::WrongWallet {
            signer: canonical_wallet(&signer.key),
            named: named.to_string(),
        }));
    }

    replay_guard().admit(&signer, now).map_err(reject)?;
    Ok((parsed, canonical_wallet(&signer.key)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    use ed25519_dalek::{Signer, SigningKey};

    const PATH: &str = "/api/v1/crown-ash/action";

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn sign(sk: &SigningKey, ts: i64, path: &str, body: &[u8], hash_field: Option<String>) -> HeaderMap {
        let pk = sk.verifying_key().to_bytes();
        let body_hash = Sha3_256::digest(body);
        let mut h = Sha3_256::new();
        h.update(pk);
        h.update(ts.to_le_bytes());
        h.update(path.as_bytes());
        h.update(body_hash);
        let sig = sk.sign(&h.finalize());
        let header = serde_json::json!({
            "address": canonical_wallet(&pk),
            "timestamp": ts,
            "scheme": "Ed25519",
            "signature": hex::encode(sig.to_bytes()),
            "body_hash": hash_field.unwrap_or_else(|| hex::encode(body_hash)),
        });
        let mut headers = HeaderMap::new();
        headers.insert(AUTH_HEADER, HeaderValue::from_str(&header.to_string()).unwrap());
        headers
    }

    #[test]
    fn valid_signature_verifies() {
        let sk = key(7);
        let body = br#"{"wallet":"x","action":"Pass"}"#;
        let h = sign(&sk, 1_000, PATH, body, None);
        let v = verify_signed_order(&h, PATH, body, 1_010).unwrap();
        assert_eq!(v.key, sk.verifying_key().to_bytes());
    }

    #[test]
    fn changed_body_is_refused() {
        let sk = key(7);
        let h = sign(&sk, 1_000, PATH, br#"{"a":1}"#, None);
        assert_eq!(
            verify_signed_order(&h, PATH, br#"{"a":2}"#, 1_000),
            Err(OrderAuthError::BodyHashMismatch)
        );
    }

    #[test]
    fn body_hash_matching_new_body_but_not_signed_is_refused() {
        // Attacker swaps the body AND rewrites body_hash to match it: the signature
        // covered the old hash, so it must fail.
        let sk = key(7);
        let new_body = br#"{"a":2}"#;
        let forged_hash = hex::encode(Sha3_256::digest(new_body));
        let h = sign(&sk, 1_000, PATH, br#"{"a":1}"#, Some(forged_hash));
        assert_eq!(
            verify_signed_order(&h, PATH, new_body, 1_000),
            Err(OrderAuthError::BadSignature)
        );
    }

    #[test]
    fn other_path_is_refused() {
        let sk = key(7);
        let body = br#"{}"#;
        let h = sign(&sk, 1_000, "/api/v1/crown-ash/join", body, None);
        assert_eq!(verify_signed_order(&h, PATH, body, 1_000), Err(OrderAuthError::BadSignature));
    }

    #[test]
    fn missing_header_and_missing_body_hash_are_refused() {
        assert_eq!(
            verify_signed_order(&HeaderMap::new(), PATH, b"{}", 0),
            Err(OrderAuthError::MissingHeader)
        );
        let sk = key(7);
        let pk = sk.verifying_key().to_bytes();
        let header = serde_json::json!({
            "address": canonical_wallet(&pk), "timestamp": 0, "signature": "00",
        });
        let mut h = HeaderMap::new();
        h.insert(AUTH_HEADER, HeaderValue::from_str(&header.to_string()).unwrap());
        assert_eq!(verify_signed_order(&h, PATH, b"{}", 0), Err(OrderAuthError::MissingBodyHash));
    }

    #[test]
    fn stale_or_future_timestamp_is_refused() {
        let sk = key(7);
        let h = sign(&sk, 1_000, PATH, b"{}", None);
        assert_eq!(
            verify_signed_order(&h, PATH, b"{}", 1_301),
            Err(OrderAuthError::Expired { age_secs: 301 })
        );
        assert_eq!(
            verify_signed_order(&h, PATH, b"{}", 699),
            Err(OrderAuthError::Expired { age_secs: -301 })
        );
    }

    #[test]
    fn other_schemes_are_refused() {
        let mut h = HeaderMap::new();
        let header = serde_json::json!({"address": "qnk00", "timestamp": 0, "scheme": "AegisQL"});
        h.insert(AUTH_HEADER, HeaderValue::from_str(&header.to_string()).unwrap());
        assert_eq!(
            verify_signed_order(&h, PATH, b"{}", 0),
            Err(OrderAuthError::UnsupportedScheme("AegisQL".into()))
        );
    }

    #[test]
    fn replay_guard_refuses_second_use_and_forgets_after_window() {
        let guard = ReplayGuard::default();
        let s = VerifiedSigner { key: [1; 32], signature: [9; 64], timestamp: 1_000 };
        assert_eq!(guard.admit(&s, 1_000), Ok(()));
        assert_eq!(guard.admit(&s, 1_200), Err(OrderAuthError::Replayed));
        // After the window the signature could not verify anyway; the entry is dropped.
        assert_eq!(guard.admit(&s, 1_301), Ok(()));
    }

    /// Signed by the quillon-wallet MCP's own `signXWalletAuth` (v2.21.3, throwaway
    /// seed "crown-ash-test-vector"): pins the client and this verifier to one layout.
    #[test]
    fn mcp_signed_vector_verifies() {
        let body = br#"{"wallet":"qnk5f1f1be43c9000aded69cfb94128cbe4bda3a96299148c82e641fb305c0b434f","action":{"RaiseArmy":{"province":0}},"nonce":"0011223344556677"}"#;
        let header = r#"{"address":"qnk5f1f1be43c9000aded69cfb94128cbe4bda3a96299148c82e641fb305c0b434f","timestamp":1793404800,"scheme":"Ed25519","signature":"81a3952a687ef5e262443168ea1ff907a644593be3dec8482fc87625d03274c032fde3a153c51ae2da5f1e2a0be1cf92e41ed2cb5d0aee352dd1942ce26dd00a","public_key":"5f1f1be43c9000aded69cfb94128cbe4bda3a96299148c82e641fb305c0b434f","body_hash":"67a3e1094adcd9619ff6a536242eac30fec0cf652f22a27d4fe43f9f46570b5a"}"#;
        let mut h = HeaderMap::new();
        h.insert(AUTH_HEADER, HeaderValue::from_str(header).unwrap());
        let v = verify_signed_order(&h, PATH, body, 1_793_404_800).unwrap();
        assert_eq!(
            canonical_wallet(&v.key),
            "qnk5f1f1be43c9000aded69cfb94128cbe4bda3a96299148c82e641fb305c0b434f"
        );
    }

    #[test]
    fn wallet_parsing_is_case_insensitive_and_canonical_is_lowercase() {
        let k = [0xab; 32];
        let upper = format!("qnk{}", hex::encode(k).to_uppercase());
        assert_eq!(parse_wallet(&upper), Some(k));
        assert_eq!(parse_wallet(&hex::encode(k)), Some(k));
        assert_eq!(canonical_wallet(&k), format!("qnk{}", "ab".repeat(32)));
        assert_eq!(parse_wallet("qnk1234"), None);
    }
}
