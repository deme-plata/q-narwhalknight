// Long-term credential authentication for TURN (RFC 5389 §10.2 + RFC 5766 §4).
//
// Credential format (matches q-api-server's /api/v1/turn/credentials):
//   username = "{unix_timestamp}:{wallet_address}"
//   password = hex(HMAC-SHA256(shared_secret, username))
//
// The client then derives the HMAC-SHA1 key as per RFC 5389:
//   mi_key = MD5(username + ":" + realm + ":" + password)
//
// Nonces are generated on demand, stored with a 5-minute expiry.

use dashmap::DashMap;
use md5::Digest as _;
use ring::hmac;
use std::sync::Arc;
use std::time::{Duration, Instant};

const NONCE_TTL: Duration = Duration::from_secs(300);
const CREDENTIAL_SLACK: i64 = 300; // allow credentials up to 5 min outside TTL window

pub struct AuthState {
    nonces:  DashMap<String, Instant>,
    secret:  String,
    realm:   String,
    cred_ttl: u64,
}

impl AuthState {
    pub fn new(secret: String, realm: String, cred_ttl: u64) -> Arc<Self> {
        Arc::new(Self {
            nonces: DashMap::new(),
            secret,
            realm,
            cred_ttl,
        })
    }

    /// Generate a fresh nonce, register it, and return it as a hex string.
    pub fn generate_nonce(&self) -> String {
        let mut bytes = [0u8; 16];
        use rand::RngCore;
        rand::rng().fill_bytes(&mut bytes);
        let nonce = hex::encode(bytes);
        self.nonces.insert(nonce.clone(), Instant::now());
        nonce
    }

    /// Check that a nonce was issued by us and hasn't expired.
    fn nonce_valid(&self, nonce: &str) -> bool {
        if let Some(entry) = self.nonces.get(nonce) {
            if entry.elapsed() < NONCE_TTL {
                return true;
            }
        }
        // Evict stale nonce
        self.nonces.remove(nonce);
        false
    }

    /// Evict all expired nonces (call periodically to prevent unbounded growth).
    pub fn evict_nonces(&self) {
        self.nonces.retain(|_, issued| issued.elapsed() < NONCE_TTL);
    }

    /// Validate long-term credentials.
    ///
    /// Returns `Some(mi_key)` (the HMAC-SHA1 key for MESSAGE-INTEGRITY) on success,
    /// or `None` if any check fails.
    pub fn validate_credentials(
        &self,
        username: &str,
        realm: &str,
        nonce: &str,
    ) -> Option<Vec<u8>> {
        // Realm must match ours
        if realm != self.realm { return None; }

        // Nonce must be valid
        if !self.nonce_valid(nonce) { return None; }

        // Username format: "{timestamp}:{wallet_address}"
        let mut parts = username.splitn(2, ':');
        let ts_str = parts.next()?;
        let _wallet = parts.next()?; // validated implicitly by HMAC check

        // Timestamp freshness: within [now - cred_ttl - slack, now + slack]
        let ts: i64 = ts_str.parse().ok()?;
        let now = chrono::Utc::now().timestamp();
        if (now - ts).abs() > (self.cred_ttl as i64 + CREDENTIAL_SLACK) {
            return None;
        }

        // Derive expected password = hex(HMAC-SHA256(secret, username))
        let expected_password = derive_password(&self.secret, username);

        // Derive MI key = MD5(username ":" realm ":" password)
        let mi_key_input = format!("{}:{}:{}", username, realm, expected_password);
        let mi_key = md5_hash(mi_key_input.as_bytes());
        Some(mi_key.to_vec())
    }

    pub fn realm(&self) -> &str { &self.realm }
}

/// Generate a time-limited TURN password from the shared secret.
/// password = hex(HMAC-SHA256(secret, username))
pub fn derive_password(secret: &str, username: &str) -> String {
    let key = hmac::Key::new(hmac::HMAC_SHA256, secret.as_bytes());
    let tag = hmac::sign(&key, username.as_bytes());
    hex::encode(tag.as_ref())
}

/// Derive the MI key from long-term credential components.
/// mi_key = MD5(username ":" realm ":" password)
pub fn derive_mi_key(username: &str, realm: &str, password: &str) -> Vec<u8> {
    let input = format!("{}:{}:{}", username, realm, password);
    md5_hash(input.as_bytes()).to_vec()
}

fn md5_hash(input: &[u8]) -> [u8; 16] {
    let result = md5::Md5::digest(input);
    let mut out = [0u8; 16];
    out.copy_from_slice(&result);
    out
}

/// Build a username for a given wallet address (used by the API credential endpoint).
pub fn build_username(wallet_addr: &str) -> String {
    let ts = chrono::Utc::now().timestamp();
    format!("{}:{}", ts, wallet_addr)
}
